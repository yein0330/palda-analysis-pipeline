from __future__ import annotations

import math
import shutil
import time
from collections import Counter
from pathlib import Path
from typing import Any

from .io import iter_jsonl, read_json, write_json, write_jsonl


SENTIMENTS = ("positive", "negative", "neutral")
SPECIAL_THEMES = {"OTHER", "UNUSABLE"}


def train_service_encoder(
    release_dir: str | Path,
    output_dir: str | Path,
    *,
    model_name: str = "beomi/KcELECTRA-base",
    epochs: int = 3,
    batch_size: int = 16,
    gradient_accumulation_steps: int = 2,
    learning_rate: float = 2e-5,
    head_learning_rate: float = 1e-4,
    max_length: int = 256,
    seed: int = 42,
    device_name: str = "auto",
    silver_weight: float = 0.7,
    qna_silver_weight: float | None = None,
    primary_silver_weight: float | None = None,
    primary_balance_power: float = 0.0,
    review_balance_power: float = 0.0,
    secondary_balance_power: float = 0.0,
    review_positive_base: float = 3.0,
    secondary_positive_base: float = 3.0,
    secondary_loss_weight: float = 0.5,
    threshold_beta: float = 1.0,
    pooling: str = "cls",
    audit_test: bool = False,
    max_train_samples: int | None = None,
    max_eval_samples: int | None = None,
) -> dict[str, Any]:
    try:
        import numpy as np
        import torch
        from sklearn.metrics import f1_score
        from torch import nn
        from torch.utils.data import DataLoader, Dataset
        from transformers import AutoModel, AutoTokenizer, get_linear_schedule_with_warmup
    except ImportError as exc:
        raise RuntimeError("Install training dependencies with: pip install -e '.[train]'") from exc

    torch.manual_seed(seed)
    if qna_silver_weight is not None and (not math.isfinite(qna_silver_weight) or qna_silver_weight < 0):
        raise ValueError("qna_silver_weight must be finite and nonnegative")
    if primary_silver_weight is not None and (not math.isfinite(primary_silver_weight) or primary_silver_weight < 0):
        raise ValueError("primary_silver_weight must be finite and nonnegative")
    if not math.isfinite(secondary_loss_weight) or secondary_loss_weight < 0:
        raise ValueError("secondary_loss_weight must be finite and nonnegative")
    if threshold_beta <= 0:
        raise ValueError("threshold_beta must be positive")
    if pooling not in {"cls", "mean"}:
        raise ValueError("pooling must be 'cls' or 'mean'")
    if review_positive_base <= 0 or secondary_positive_base <= 0:
        raise ValueError("positive loss bases must be positive")
    np.random.seed(seed)
    release = Path(release_dir)
    data_splits = ("train", "validation", "test") if audit_test else ("train", "validation")
    rows = {
        split: list(iter_jsonl(release / f"{split}.jsonl"))
        for split in data_splits
    }
    if max_train_samples is not None:
        rows["train"] = rows["train"][:max_train_samples]
    if max_eval_samples is not None:
        rows["validation"] = rows["validation"][:max_eval_samples]
        if audit_test:
            rows["test"] = rows["test"][:max_eval_samples]
    if not rows["train"]:
        raise ValueError("Training split is empty")
    taxonomy_path = release / "taxonomy.json"
    taxonomy = read_json(taxonomy_path) if taxonomy_path.exists() else {}
    taxonomy_themes = {
        str(theme["id"])
        for theme in taxonomy.get("themes", [])
        if isinstance(theme, dict) and theme.get("id")
    }
    themes = sorted(taxonomy_themes or {
        theme
        for values in rows.values()
        for row in values
        for theme in (
            list(row.get("candidate_theme_ids") or [])
            + [label["theme_id"] for label in row.get("labels", [])]
        )
    })
    theme_to_id = {theme: index for index, theme in enumerate(themes)}
    sentiment_to_id = {value: index for index, value in enumerate(SENTIMENTS)}
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    primary_counts = np.zeros(len(themes), dtype=float)
    review_counts = np.zeros((len(themes), len(SENTIMENTS)), dtype=float)
    secondary_counts = np.zeros(len(themes), dtype=float)
    for row in rows["train"]:
        if row["record_type"] == "review":
            for label in row.get("labels", []):
                theme = theme_to_id.get(label.get("theme_id"))
                sentiment = sentiment_to_id.get(label.get("sentiment"))
                if theme is not None and sentiment is not None:
                    review_counts[theme, sentiment] += 1
            continue
        primary = next((label for label in row.get("labels", []) if label.get("role") == "primary"), None)
        if primary is None and row.get("labels"):
            primary = row["labels"][0]
        if primary and primary.get("theme_id") in theme_to_id:
            primary_counts[theme_to_id[primary["theme_id"]]] += 1
        for label in row.get("labels", []):
            if label is primary or label.get("role") == "primary":
                continue
            theme = theme_to_id.get(label.get("theme_id"))
            if theme is not None:
                secondary_counts[theme] += 1
    primary_class_weights = np.ones(len(themes), dtype=float)
    supported_primary = primary_counts > 0
    if primary_balance_power > 0 and supported_primary.any():
        maximum = primary_counts[supported_primary].max()
        primary_class_weights[supported_primary] = np.minimum(
            (maximum / primary_counts[supported_primary]) ** primary_balance_power, 5.0
        )
        primary_class_weights[supported_primary] /= primary_class_weights[supported_primary].mean()
    primary_class_weights_tensor = torch.tensor(
        primary_class_weights, dtype=torch.float32
    )
    review_pos_weights_tensor = torch.tensor(
        _positive_weights(np, review_counts, review_balance_power, review_positive_base), dtype=torch.float32
    )
    secondary_pos_weights_tensor = torch.tensor(
        _positive_weights(np, secondary_counts, secondary_balance_power, secondary_positive_base), dtype=torch.float32
    )

    class RowDataset(Dataset):
        def __init__(self, values: list[dict[str, Any]]):
            self.values = values

        def __len__(self) -> int:
            return len(self.values)

        def __getitem__(self, index: int) -> dict[str, Any]:
            return self.values[index]

    def collate(values: list[dict[str, Any]]) -> dict[str, Any]:
        texts = []
        candidate_mask = torch.zeros((len(values), len(themes)), dtype=torch.bool)
        review_targets = torch.zeros((len(values), len(themes), len(SENTIMENTS)))
        primary_targets = torch.full((len(values),), -100, dtype=torch.long)
        secondary_targets = torch.zeros((len(values), len(themes)))
        review_task = torch.zeros(len(values), dtype=torch.bool)
        usable_targets = torch.zeros(len(values))
        sample_weights = torch.ones(len(values))
        for row_index, row in enumerate(values):
            parts = [part.strip() for part in str(row.get("category_key") or "").split(">")]
            cat1 = parts[0] if parts else ""
            cat2 = parts[1] if len(parts) > 1 else ""
            texts.append(
                f"[TASK={row['record_type'].upper()}] [CAT1={cat1}] [CAT2={cat2}] {row['text']}"
            )
            allowed = set(row.get("candidate_theme_ids") or themes)
            for theme in allowed:
                if theme in theme_to_id:
                    candidate_mask[row_index, theme_to_id[theme]] = True
            is_review = row["record_type"] == "review"
            review_task[row_index] = is_review
            usable_targets[row_index] = float(row.get("usable", True))
            if row.get("label_tier") == "silver":
                sample_weights[row_index] = (
                    qna_silver_weight if not is_review and qna_silver_weight is not None else silver_weight
                )
            labels = row.get("labels", [])
            if is_review:
                for label in labels:
                    theme = theme_to_id.get(label["theme_id"])
                    sentiment = sentiment_to_id.get(label.get("sentiment"))
                    if theme is not None and sentiment is not None:
                        review_targets[row_index, theme, sentiment] = 1
            else:
                primary = next((label for label in labels if label.get("role") == "primary"), None)
                if primary is None and labels:
                    primary = labels[0]
                if primary and primary["theme_id"] in theme_to_id:
                    primary_targets[row_index] = theme_to_id[primary["theme_id"]]
                elif "OTHER" in theme_to_id:
                    primary_targets[row_index] = theme_to_id["OTHER"]
                for label in labels:
                    if label is primary or label.get("role") == "primary":
                        continue
                    theme = theme_to_id.get(label["theme_id"])
                    if theme is not None:
                        secondary_targets[row_index, theme] = 1
        encoded = tokenizer(
            texts, padding=True, truncation=True, max_length=max_length, return_tensors="pt"
        )
        non_special_mask = candidate_mask.clone()
        primary_sample_weights = sample_weights.clone()
        if primary_silver_weight is not None:
            for row_index, row in enumerate(values):
                if row["record_type"] == "qna" and row.get("label_tier") == "silver":
                    primary_sample_weights[row_index] = primary_silver_weight
        for theme in SPECIAL_THEMES:
            if theme in theme_to_id:
                non_special_mask[:, theme_to_id[theme]] = False
        return {
            **encoded,
            "rows": values,
            "candidate_mask": candidate_mask,
            "non_special_mask": non_special_mask,
            "review_targets": review_targets,
            "primary_targets": primary_targets,
            "secondary_targets": secondary_targets,
            "review_task": review_task,
            "usable_targets": usable_targets,
            "sample_weights": sample_weights,
            "primary_sample_weights": primary_sample_weights,
        }

    class ServiceEncoder(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.backbone = AutoModel.from_pretrained(model_name)
            hidden = int(self.backbone.config.hidden_size)
            self.dropout = nn.Dropout(float(getattr(self.backbone.config, "hidden_dropout_prob", 0.1)))
            self.review_head = nn.Linear(hidden, len(themes) * len(SENTIMENTS))
            self.primary_head = nn.Linear(hidden, len(themes))
            self.secondary_head = nn.Linear(hidden, len(themes))
            self.usable_head = nn.Linear(hidden, 1)

        def forward(self, input_ids: Any, attention_mask: Any, token_type_ids: Any = None) -> dict[str, Any]:
            kwargs = {"input_ids": input_ids, "attention_mask": attention_mask}
            if token_type_ids is not None:
                kwargs["token_type_ids"] = token_type_ids
            hidden_states = self.backbone(**kwargs).last_hidden_state
            pooled = self.dropout(_pool_sentence(hidden_states, attention_mask, pooling))
            return {
                "review": self.review_head(pooled).view(-1, len(themes), len(SENTIMENTS)),
                "primary": self.primary_head(pooled),
                "secondary": self.secondary_head(pooled),
                "usable": self.usable_head(pooled).squeeze(-1),
            }

    device = _choose_device(torch, device_name)
    primary_class_weights_tensor = primary_class_weights_tensor.to(device)
    review_pos_weights_tensor = review_pos_weights_tensor.to(device)
    secondary_pos_weights_tensor = secondary_pos_weights_tensor.to(device)
    model = ServiceEncoder().to(device)
    loader_splits = ("train", "validation", "test") if audit_test else ("train", "validation")
    loaders = {
        split: DataLoader(
            RowDataset(values), batch_size=batch_size * (2 if split != "train" else 1),
            shuffle=split == "train", collate_fn=collate, num_workers=0,
        )
        for split in loader_splits
        for values in (rows[split],)
    }
    backbone_parameters, head_parameters = [], []
    for name, parameter in model.named_parameters():
        (backbone_parameters if name.startswith("backbone.") else head_parameters).append(parameter)
    optimizer = torch.optim.AdamW([
        {"params": backbone_parameters, "lr": learning_rate},
        {"params": head_parameters, "lr": head_learning_rate},
    ], weight_decay=0.01)
    update_steps = math.ceil(len(loaders["train"]) / gradient_accumulation_steps)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, int(update_steps * epochs * 0.06), update_steps * epochs
    )
    history = []
    best_state = None
    best_score = -math.inf
    best_thresholds: dict[str, Any] = {}
    for epoch in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        losses = []
        for step, batch in enumerate(loaders["train"], 1):
            batch.pop("rows")
            batch = {key: value.to(device) if hasattr(value, "to") else value for key, value in batch.items()}
            outputs = model(
                input_ids=batch["input_ids"], attention_mask=batch["attention_mask"],
                token_type_ids=batch.get("token_type_ids"),
            )
            loss, _ = _service_loss(
                torch, nn, outputs, batch, primary_class_weights_tensor,
                review_pos_weights_tensor, secondary_pos_weights_tensor,
                secondary_loss_weight=secondary_loss_weight,
            )
            (loss / gradient_accumulation_steps).backward()
            if step % gradient_accumulation_steps == 0 or step == len(loaders["train"]):
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
            losses.append(float(loss.detach().cpu()))
        validation_output = _collect(torch, model, loaders["validation"], device, themes)
        thresholds = _calibrate(np, validation_output, themes, beta=threshold_beta)
        validation_metrics = _service_metrics(np, f1_score, validation_output, thresholds, themes)
        score = float(validation_metrics["service_macro_f1"])
        history.append({
            "epoch": epoch,
            "train_loss": sum(losses) / max(len(losses), 1),
            "validation": validation_metrics,
        })
        if score > best_score:
            best_score = score
            best_thresholds = thresholds
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)
    validation_output = _collect(torch, model, loaders["validation"], device, themes)
    validation_metrics = _service_metrics(np, f1_score, validation_output, best_thresholds, themes)
    test_output = None
    test_metrics = None
    if audit_test:
        test_output = _collect(torch, model, loaders["test"], device, themes, measure_latency=True)
        test_metrics = _service_metrics(np, f1_score, test_output, best_thresholds, themes)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    model.backbone.save_pretrained(output / "backbone")
    tokenizer.save_pretrained(output / "tokenizer")
    if (release / "taxonomy.json").exists():
        shutil.copy2(release / "taxonomy.json", output / "taxonomy.json")
    torch.save({
        "architecture": "palda-service-encoder-v2",
        "review_head": model.review_head.state_dict(),
        "primary_head": model.primary_head.state_dict(),
        "secondary_head": model.secondary_head.state_dict(),
        "usable_head": model.usable_head.state_dict(),
    }, output / "heads.pt")
    write_json(output / "label_map.json", {"themes": themes, "sentiments": list(SENTIMENTS)})
    write_json(output / "thresholds.json", best_thresholds)
    _write_service_predictions(output / "validation_predictions.jsonl", validation_output, best_thresholds, themes)
    if test_output is not None:
        _write_service_predictions(output / "test_predictions.jsonl", test_output, best_thresholds, themes)
    metrics = {
        "architecture": "palda-service-encoder-v2",
        "model_name": model_name,
        "device": str(device),
        "config": {
            "epochs": epochs, "batch_size": batch_size,
            "gradient_accumulation_steps": gradient_accumulation_steps,
            "learning_rate": learning_rate, "head_learning_rate": head_learning_rate,
            "max_length": max_length, "seed": seed, "silver_weight": silver_weight,
            "qna_silver_weight": qna_silver_weight,
            "primary_silver_weight": primary_silver_weight,
            "primary_balance_power": primary_balance_power,
            "review_balance_power": review_balance_power,
            "secondary_balance_power": secondary_balance_power,
            "review_positive_base": review_positive_base,
            "secondary_positive_base": secondary_positive_base,
            "secondary_loss_weight": secondary_loss_weight,
            "threshold_beta": threshold_beta,
            "pooling": pooling,
        },
        "history": history,
        "validation": validation_metrics,
        "test": test_metrics,
        "test_audited": audit_test,
        "evaluation_label_tiers": {
            split: dict(Counter(row.get("label_tier", "unknown") for row in rows[split]))
            for split in (("validation", "test") if audit_test else ("validation",))
        },
    }
    write_json(output / "metrics.json", metrics)
    return metrics


def _choose_device(torch: Any, name: str) -> Any:
    if name != "auto":
        return torch.device(name)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _positive_weights(np: Any, counts: Any, power: float, base: float = 3.0) -> Any:
    weights = np.full(counts.shape, base, dtype=float)
    supported = counts > 0
    if power > 0 and supported.any():
        maximum = counts[supported].max()
        weights[supported] = np.minimum(base * (maximum / counts[supported]) ** power, 20.0)
    return weights


def _masked_bce(torch: Any, logits: Any, targets: Any, mask: Any, weights: Any, pos_weight: Any) -> Any:
    loss = torch.nn.functional.binary_cross_entropy_with_logits(
        logits, targets, reduction="none",
        pos_weight=torch.as_tensor(pos_weight, device=logits.device, dtype=logits.dtype),
    )
    while weights.ndim < loss.ndim:
        weights = weights.unsqueeze(-1)
    weighted_mask = mask.float() * weights
    return (loss * weighted_mask).sum() / weighted_mask.sum().clamp_min(1)


def _service_loss(
    torch: Any, nn: Any, outputs: dict[str, Any], batch: dict[str, Any],
    primary_class_weights: Any = None,
    review_pos_weights: Any = 3.0,
    secondary_pos_weights: Any = 3.0,
    *, secondary_loss_weight: float = 0.5,
) -> tuple[Any, dict[str, float]]:
    review = batch["review_task"]
    qna = ~review
    weights = batch["sample_weights"]
    review_mask = batch["non_special_mask"].unsqueeze(-1).expand_as(batch["review_targets"]) & review[:, None, None]
    review_loss = _masked_bce(
        torch, outputs["review"], batch["review_targets"], review_mask, weights,
        review_pos_weights,
    )
    masked_primary = outputs["primary"].masked_fill(~batch["candidate_mask"], -1e4)
    valid_primary = qna & (batch["primary_targets"] >= 0)
    if valid_primary.any():
        each = nn.functional.cross_entropy(
            masked_primary[valid_primary], batch["primary_targets"][valid_primary],
            reduction="none", weight=primary_class_weights,
        )
        primary_weights = batch.get("primary_sample_weights", weights)
        primary_loss = (each * primary_weights[valid_primary]).sum() / primary_weights[valid_primary].sum().clamp_min(1)
    else:
        primary_loss = outputs["primary"].sum() * 0
    secondary_mask = batch["non_special_mask"] & qna[:, None]
    secondary_loss = _masked_bce(
        torch, outputs["secondary"], batch["secondary_targets"], secondary_mask, weights,
        secondary_pos_weights,
    )
    if qna.any():
        each = nn.functional.binary_cross_entropy_with_logits(
            outputs["usable"][qna], batch["usable_targets"][qna], reduction="none"
        )
        usable_loss = (each * weights[qna]).sum() / weights[qna].sum().clamp_min(1)
    else:
        usable_loss = outputs["usable"].sum() * 0
    total = review_loss + primary_loss + secondary_loss_weight * secondary_loss + 0.2 * usable_loss
    return total, {
        "review": float(review_loss.detach()), "primary": float(primary_loss.detach()),
        "secondary": float(secondary_loss.detach()), "usable": float(usable_loss.detach()),
    }


def _collect(torch: Any, model: Any, loader: Any, device: Any, themes: list[str], *, measure_latency: bool = False) -> dict[str, Any]:
    model.eval()
    result = {"rows": [], "review_probs": [], "review_targets": [], "primary_predictions": [],
              "primary_targets": [], "secondary_probs": [], "secondary_targets": [], "usable_probs": [],
              "usable_targets": [], "candidate_masks": []}
    started = time.perf_counter()
    with torch.no_grad():
        for batch in loader:
            source_rows = batch.pop("rows")
            moved = {key: value.to(device) if hasattr(value, "to") else value for key, value in batch.items()}
            outputs = model(
                input_ids=moved["input_ids"], attention_mask=moved["attention_mask"],
                token_type_ids=moved.get("token_type_ids"),
            )
            primary = outputs["primary"].masked_fill(~moved["candidate_mask"], -1e4).argmax(-1)
            result["rows"].extend(source_rows)
            for key, value in (
                (
                    "review_probs",
                    torch.sigmoid(outputs["review"]) * moved["non_special_mask"].unsqueeze(-1),
                ),
                ("review_targets", moved["review_targets"]),
                ("primary_predictions", primary),
                ("primary_targets", moved["primary_targets"]),
                (
                    "secondary_probs",
                    torch.sigmoid(outputs["secondary"]) * moved["non_special_mask"],
                ),
                ("secondary_targets", moved["secondary_targets"]),
                ("usable_probs", torch.sigmoid(outputs["usable"])),
                ("usable_targets", moved["usable_targets"]),
                ("candidate_masks", moved["candidate_mask"]),
            ):
                result[key].append(value.detach().cpu())
    if str(device).startswith("cuda"):
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    for key in tuple(result):
        if key != "rows":
            result[key] = torch.cat(result[key]).numpy()
    result["inference_seconds"] = elapsed if measure_latency else None
    result["samples_per_second"] = len(result["rows"]) / elapsed if measure_latency and elapsed else None
    return result


def _pool_sentence(hidden_states: Any, attention_mask: Any, pooling: str) -> Any:
    """Reduce token states to one sentence vector.

    Sentence-embedding backbones such as jhgan/ko-sroberta-multitask are trained
    with mean pooling, so their CLS position holds no pretrained sentence
    representation. `cls` stays the default because every recorded candidate was
    trained with it.
    """
    if pooling == "mean":
        mask = attention_mask.unsqueeze(-1).to(hidden_states.dtype)
        return (hidden_states * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)
    return hidden_states[:, 0]


def _calibrate(
    np: Any, output: dict[str, Any], themes: list[str], *, beta: float = 1.0
) -> dict[str, Any]:
    review_mask = np.asarray([row["record_type"] == "review" for row in output["rows"]])
    qna_mask = ~review_mask
    review_threshold = _best_global_threshold(
        np, output["review_probs"][review_mask], output["review_targets"][review_mask], beta=beta
    )
    secondary_threshold = _best_global_threshold(
        np, output["secondary_probs"][qna_mask], output["secondary_targets"][qna_mask], beta=beta
    )
    review_by_label = _per_label_thresholds(
        np, output["review_probs"][review_mask], output["review_targets"][review_mask],
        review_threshold, beta=beta,
    )
    secondary_by_theme = _per_label_thresholds(
        np, output["secondary_probs"][qna_mask], output["secondary_targets"][qna_mask],
        secondary_threshold, beta=beta,
    )
    return {
        "review": review_by_label.tolist(),
        "review_global": review_threshold,
        "secondary": secondary_by_theme.tolist(),
        "secondary_global": secondary_threshold,
        "usable": 0.5,
    }


def _per_label_thresholds(
    np: Any, probabilities: Any, targets: Any, default: float, *,
    min_support: int = 3, beta: float = 1.0,
) -> Any:
    original_shape = probabilities.shape[1:]
    flat_probabilities = probabilities.reshape((len(probabilities), -1))
    flat_targets = targets.reshape((len(targets), -1))
    thresholds = np.full(flat_probabilities.shape[1], default, dtype=float)
    for index in range(flat_probabilities.shape[1]):
        if flat_targets[:, index].sum() >= min_support:
            thresholds[index] = _best_global_threshold(
                np, flat_probabilities[:, index], flat_targets[:, index], beta=beta
            )
    return thresholds.reshape(original_shape)


def _best_global_threshold(
    np: Any, probabilities: Any, targets: Any, *, beta: float = 1.0
) -> float:
    best = (0.0, 0.5)
    for threshold in np.arange(0.2, 0.91, 0.05):
        predicted = probabilities >= threshold
        actual = targets >= 0.5
        tp = np.logical_and(predicted, actual).sum()
        fp = np.logical_and(predicted, ~actual).sum()
        fn = np.logical_and(~predicted, actual).sum()
        beta_squared = beta * beta
        score = (1 + beta_squared) * tp / max(
            (1 + beta_squared) * tp + beta_squared * fn + fp, 1
        )
        if score > best[0]:
            best = (float(score), float(threshold))
    return round(best[1], 2)


def _set_metrics(np: Any, f1_score: Any, actual: Any, predicted: Any) -> dict[str, float]:
    if actual.shape[0] == 0:
        return {
            "micro_precision": 0.0, "micro_recall": 0.0, "micro_f1": 0.0,
            "macro_f1": 0.0, "exact_match": 0.0,
        }
    tp = np.logical_and(predicted, actual).sum()
    fp = np.logical_and(predicted, ~actual).sum()
    fn = np.logical_and(~predicted, actual).sum()
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    # Count classes predicted without Gold support as false-positive classes.
    # Looking only at Gold-supported columns makes a broad over-predictor appear
    # artificially strong, especially with a large category-aware taxonomy.
    supported = np.logical_or(actual.sum(axis=0) > 0, predicted.sum(axis=0) > 0)
    macro = f1_score(actual[:, supported], predicted[:, supported], average="macro", zero_division=0) if supported.any() else 0.0
    return {
        "micro_precision": round(float(precision), 6),
        "micro_recall": round(float(recall), 6),
        "micro_f1": round(float(2 * precision * recall / max(precision + recall, 1e-12)), 6),
        "macro_f1": round(float(macro), 6),
        "exact_match": round(float(np.mean(np.all(actual == predicted, axis=1))), 6),
    }


def _service_metrics(np: Any, f1_score: Any, output: dict[str, Any], thresholds: dict[str, Any], themes: list[str]) -> dict[str, Any]:
    review_mask = np.asarray([row["record_type"] == "review" for row in output["rows"]])
    qna_mask = ~review_mask
    review_actual = output["review_targets"][review_mask] >= 0.5
    review_predicted = _exclusive_review_predictions(
        np, output["review_probs"][review_mask], np.asarray(thresholds["review"])
    )
    review_metrics = _set_metrics(
        np, f1_score,
        review_actual.reshape((len(review_actual), len(themes) * len(SENTIMENTS))),
        review_predicted.reshape((len(review_predicted), len(themes) * len(SENTIMENTS))),
    )
    secondary_metrics = _set_metrics(
        np, f1_score, output["secondary_targets"][qna_mask] >= 0.5,
        output["secondary_probs"][qna_mask] >= np.asarray(thresholds["secondary"]),
    )
    primary_actual = output["primary_targets"][qna_mask]
    primary_predicted = output["primary_predictions"][qna_mask]
    valid = primary_actual >= 0
    primary_accuracy = float(np.mean(primary_actual[valid] == primary_predicted[valid])) if valid.any() else 0.0
    # Include false-positive classes in macro F1. Restricting labels to Gold
    # classes alone can hide predictions into unsupported themes.
    labels = (
        np.union1d(primary_actual[valid], primary_predicted[valid])
        if valid.any() else []
    )
    primary_macro = f1_score(primary_actual[valid], primary_predicted[valid], labels=labels, average="macro", zero_division=0) if valid.any() else 0.0
    usable_actual = output["usable_targets"][qna_mask] >= 0.5
    usable_predicted = output["usable_probs"][qna_mask] >= thresholds["usable"]
    usable_accuracy = float(np.mean(usable_actual == usable_predicted)) if len(usable_actual) else 0.0
    components = [review_metrics["macro_f1"], float(primary_macro), secondary_metrics["macro_f1"]]
    return {
        "samples": len(output["rows"]),
        "service_macro_f1": round(sum(components) / len(components), 6),
        "review_theme_sentiment": review_metrics,
        "qna_primary": {"accuracy": round(primary_accuracy, 6), "macro_f1": round(float(primary_macro), 6)},
        "qna_secondary": secondary_metrics,
        "usable": {"accuracy": round(usable_accuracy, 6), "negative_support": int((~usable_actual).sum())},
        "inference_seconds": output.get("inference_seconds"),
        "samples_per_second": output.get("samples_per_second"),
    }


def _exclusive_review_predictions(np: Any, probabilities: Any, thresholds: Any) -> Any:
    """Select at most one eligible sentiment for each item/theme pair."""
    eligible = probabilities >= thresholds
    masked = np.where(eligible, probabilities, -1.0)
    selected = masked.argmax(axis=2)
    predicted = np.zeros_like(eligible, dtype=bool)
    row_indices, theme_indices = np.indices(selected.shape)
    has_candidate = eligible.any(axis=2)
    predicted[row_indices[has_candidate], theme_indices[has_candidate], selected[has_candidate]] = True
    return predicted


def _write_service_predictions(path: Path, output: dict[str, Any], thresholds: dict[str, Any], themes: list[str]) -> None:
    rows = []
    review_thresholds = thresholds["review"]
    secondary_thresholds = thresholds["secondary"]
    for index, source in enumerate(output["rows"]):
        value: dict[str, Any] = {"sample_id": source["sample_id"], "record_type": source["record_type"]}
        if source["record_type"] == "review":
            value["predicted_labels"] = []
            for ti, theme in enumerate(themes):
                candidates = [
                    (float(output["review_probs"][index, ti, si]), sentiment)
                    for si, sentiment in enumerate(SENTIMENTS)
                    if output["review_probs"][index, ti, si] >= review_thresholds[ti][si]
                ]
                if candidates:
                    confidence, sentiment = max(candidates)
                    value["predicted_labels"].append({
                        "theme_id": theme, "sentiment": sentiment,
                        "confidence": round(confidence, 6),
                    })
        else:
            primary = int(output["primary_predictions"][index])
            value["usable_probability"] = round(float(output["usable_probs"][index]), 6)
            value["primary_theme_id"] = themes[primary]
            value["secondary_themes"] = [
                {"theme_id": theme, "confidence": round(float(output["secondary_probs"][index, ti]), 6)}
                for ti, theme in enumerate(themes)
                if output["secondary_probs"][index, ti] >= secondary_thresholds[ti]
            ]
        rows.append(value)
    write_jsonl(path, rows)
