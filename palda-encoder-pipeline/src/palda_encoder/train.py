from __future__ import annotations

import json
import math
import time
from collections import Counter
from pathlib import Path
from typing import Any

from .io import iter_jsonl, write_json, write_jsonl


SENTIMENT_TO_ID = {"positive": 0, "negative": 1, "neutral": 2}


def train_encoder(
    release_dir: str | Path,
    output_dir: str | Path,
    *,
    model_name: str = "monologg/koelectra-base-v3-discriminator",
    epochs: int = 3,
    batch_size: int = 32,
    learning_rate: float = 2e-5,
    max_length: int = 128,
    seed: int = 42,
    device_name: str = "auto",
    sentiment_loss_weight: float = 1.0,
    max_train_samples: int | None = None,
    max_eval_samples: int | None = None,
) -> dict[str, Any]:
    try:
        import numpy as np
        import torch
        from torch import nn
        from torch.utils.data import DataLoader, Dataset
        from transformers import AutoModel, AutoTokenizer
    except ImportError as exc:
        raise RuntimeError(
            "Training dependencies are missing. Install with: pip install -e '.[train]'"
        ) from exc

    torch.manual_seed(seed)
    np.random.seed(seed)
    release = Path(release_dir)
    rows = {
        split: list(iter_jsonl(release / f"{split}.jsonl"))
        for split in ("train", "validation", "test")
    }
    if max_train_samples is not None:
        rows["train"] = rows["train"][:max_train_samples]
    if max_eval_samples is not None:
        rows["validation"] = rows["validation"][:max_eval_samples]
        rows["test"] = rows["test"][:max_eval_samples]
    if not rows["train"]:
        raise ValueError("Training split is empty")
    themes = sorted(
        {
            label["theme_id"]
            for split_rows in rows.values()
            for row in split_rows
            for label in row.get("labels", [])
        }
    )
    if not themes:
        raise ValueError("No theme labels found in dataset release")
    theme_to_id = {theme: index for index, theme in enumerate(themes)}
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    class EncoderDataset(Dataset):
        def __init__(self, values: list[dict[str, Any]]):
            self.values = values

        def __len__(self) -> int:
            return len(self.values)

        def __getitem__(self, index: int) -> dict[str, Any]:
            row = self.values[index]
            text = (
                f"[TYPE={row['record_type']}] [CATEGORY={row.get('category_key', '')}] "
                f"{row['text']}"
            )
            encoded = tokenizer(
                text,
                truncation=True,
                max_length=max_length,
                padding=False,
            )
            theme_targets = torch.zeros(len(themes), dtype=torch.float32)
            sentiment_targets = torch.full((len(themes),), -100, dtype=torch.long)
            candidate_ids = set(row.get("candidate_theme_ids") or themes)
            candidate_mask = torch.tensor(
                [1.0 if theme in candidate_ids else 0.0 for theme in themes],
                dtype=torch.float32,
            )
            for label in row.get("labels", []):
                theme_id = theme_to_id.get(label["theme_id"])
                if theme_id is None:
                    continue
                theme_targets[theme_id] = 1.0
                sentiment = label.get("sentiment")
                if row["record_type"] == "review" and sentiment in SENTIMENT_TO_ID:
                    sentiment_targets[theme_id] = SENTIMENT_TO_ID[sentiment]
            return {
                **encoded,
                "theme_targets": theme_targets,
                "sentiment_targets": sentiment_targets,
                "candidate_mask": candidate_mask,
                "sample_id": row["sample_id"],
                "record_type": row["record_type"],
            }

    def collate(batch: list[dict[str, Any]]) -> dict[str, Any]:
        features = [
            {
                key: value
                for key, value in item.items()
                if key not in {
                    "theme_targets", "sentiment_targets", "candidate_mask", "sample_id", "record_type"
                }
            }
            for item in batch
        ]
        encoded = tokenizer.pad(features, padding=True, return_tensors="pt")
        encoded["theme_targets"] = torch.stack([item["theme_targets"] for item in batch])
        encoded["sentiment_targets"] = torch.stack([item["sentiment_targets"] for item in batch])
        encoded["candidate_mask"] = torch.stack([item["candidate_mask"] for item in batch])
        encoded["sample_ids"] = [item["sample_id"] for item in batch]
        encoded["record_types"] = [item["record_type"] for item in batch]
        return encoded

    class MultiTaskEncoder(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.backbone = AutoModel.from_pretrained(model_name)
            hidden = int(self.backbone.config.hidden_size)
            self.dropout = nn.Dropout(float(getattr(self.backbone.config, "hidden_dropout_prob", 0.1)))
            self.theme_head = nn.Linear(hidden, len(themes))
            self.sentiment_head = nn.Linear(hidden, len(themes) * len(SENTIMENT_TO_ID))

        def forward(self, input_ids: Any, attention_mask: Any, token_type_ids: Any = None) -> tuple[Any, Any]:
            kwargs = {"input_ids": input_ids, "attention_mask": attention_mask}
            if token_type_ids is not None:
                kwargs["token_type_ids"] = token_type_ids
            hidden = self.backbone(**kwargs).last_hidden_state[:, 0]
            hidden = self.dropout(hidden)
            return self.theme_head(hidden), self.sentiment_head(hidden).view(
                -1, len(themes), len(SENTIMENT_TO_ID)
            )

    device = _choose_device(torch, device_name)
    model = MultiTaskEncoder().to(device)
    pos_weight = _positive_weights(torch, rows["train"], theme_to_id, device)
    theme_loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight, reduction="none")
    sentiment_loss_fn = nn.CrossEntropyLoss(ignore_index=-100)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    loaders = {
        split: DataLoader(
            EncoderDataset(values),
            batch_size=batch_size,
            shuffle=split == "train",
            collate_fn=collate,
            num_workers=0,
        )
        for split, values in rows.items()
    }

    history: list[dict[str, Any]] = []
    best_state = None
    best_score = -math.inf
    for epoch in range(1, epochs + 1):
        model.train()
        losses: list[float] = []
        for batch in loaders["train"]:
            theme_targets = batch.pop("theme_targets").to(device)
            sentiment_targets = batch.pop("sentiment_targets").to(device)
            candidate_mask = batch.pop("candidate_mask").to(device)
            batch.pop("sample_ids")
            batch.pop("record_types")
            inputs = {key: value.to(device) for key, value in batch.items()}
            optimizer.zero_grad(set_to_none=True)
            theme_logits, sentiment_logits = model(**inputs)
            theme_loss_matrix = theme_loss_fn(theme_logits, theme_targets)
            theme_loss = (theme_loss_matrix * candidate_mask).sum() / candidate_mask.sum().clamp_min(1.0)
            sentiment_mask = sentiment_targets >= 0
            if sentiment_mask.any():
                sentiment_loss = sentiment_loss_fn(
                    sentiment_logits[sentiment_mask], sentiment_targets[sentiment_mask]
                )
            else:
                sentiment_loss = theme_loss.new_zeros(())
            loss = theme_loss + sentiment_loss_weight * sentiment_loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            losses.append(float(loss.detach().cpu()))

        val_output = _predict(torch, model, loaders["validation"], device)
        thresholds = _calibrate_thresholds(np, val_output["theme_probs"], val_output["theme_targets"])
        val_metrics = _metrics(np, val_output, thresholds, rows["train"], themes)
        epoch_row = {
            "epoch": epoch,
            "train_loss": sum(losses) / max(len(losses), 1),
            "validation": val_metrics,
        }
        history.append(epoch_row)
        score = float(val_metrics.get("theme_macro_f1", 0.0))
        if score > best_score:
            best_score = score
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)
    val_output = _predict(torch, model, loaders["validation"], device)
    thresholds = _calibrate_thresholds(np, val_output["theme_probs"], val_output["theme_targets"])
    validation_metrics = _metrics(np, val_output, thresholds, rows["train"], themes)
    test_output = _predict(torch, model, loaders["test"], device, measure_latency=True)
    test_metrics = _metrics(np, test_output, thresholds, rows["train"], themes)

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    _write_predictions(out / "validation_predictions.jsonl", val_output, thresholds, themes)
    _write_predictions(out / "test_predictions.jsonl", test_output, thresholds, themes)
    model.backbone.save_pretrained(out / "backbone")
    tokenizer.save_pretrained(out / "tokenizer")
    torch.save(
        {
            "theme_head": model.theme_head.state_dict(),
            "sentiment_head": model.sentiment_head.state_dict(),
        },
        out / "heads.pt",
    )
    write_json(out / "label_map.json", {"themes": themes, "sentiments": SENTIMENT_TO_ID})
    write_json(out / "thresholds.json", {theme: thresholds[index] for index, theme in enumerate(themes)})
    validation_label_tiers = Counter(
        row.get("label_tier", "unknown") for row in rows["validation"]
    )
    test_label_tiers = Counter(row.get("label_tier", "unknown") for row in rows["test"])
    metrics = {
        "model_name": model_name,
        "device": str(device),
        "config": {
            "epochs": epochs,
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "max_length": max_length,
            "seed": seed,
            "max_train_samples": max_train_samples,
            "max_eval_samples": max_eval_samples,
        },
        "history": history,
        "validation": validation_metrics,
        "test": test_metrics,
        "evaluation_label_tiers": {
            "validation": dict(validation_label_tiers),
            "test": dict(test_label_tiers),
        },
        "warning": (
            "Test metrics are not a human-gold estimate."
            if set(test_label_tiers) != {"gold"}
            else None
        ),
    }
    write_json(out / "metrics.json", metrics)
    return metrics


def _choose_device(torch: Any, name: str) -> Any:
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _positive_weights(torch: Any, rows: list[dict[str, Any]], theme_to_id: dict[str, int], device: Any) -> Any:
    positives = [0] * len(theme_to_id)
    for row in rows:
        for label in row.get("labels", []):
            if label["theme_id"] in theme_to_id:
                positives[theme_to_id[label["theme_id"]]] += 1
    total = max(len(rows), 1)
    values = [min(max((total - count) / max(count, 1), 1.0), 20.0) for count in positives]
    return torch.tensor(values, dtype=torch.float32, device=device)


def _predict(
    torch: Any,
    model: Any,
    loader: Any,
    device: Any,
    *,
    measure_latency: bool = False,
) -> dict[str, Any]:
    model.eval()
    probabilities, theme_targets, sentiments, sentiment_targets = [], [], [], []
    record_types: list[str] = []
    sample_ids: list[str] = []
    samples = 0
    started = time.perf_counter()
    with torch.no_grad():
        for batch in loader:
            y_theme = batch.pop("theme_targets")
            y_sentiment = batch.pop("sentiment_targets")
            candidate_mask = batch.pop("candidate_mask")
            sample_ids.extend(batch.pop("sample_ids"))
            record_types.extend(batch.pop("record_types"))
            inputs = {key: value.to(device) for key, value in batch.items()}
            theme_logits, sentiment_logits = model(**inputs)
            probabilities.append((torch.sigmoid(theme_logits).cpu() * candidate_mask))
            sentiments.append(sentiment_logits.argmax(dim=-1).cpu())
            theme_targets.append(y_theme)
            sentiment_targets.append(y_sentiment)
            samples += int(y_theme.shape[0])
    if str(device).startswith("cuda"):
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    if probabilities:
        return {
            "theme_probs": torch.cat(probabilities).numpy(),
            "theme_targets": torch.cat(theme_targets).numpy(),
            "sentiment_predictions": torch.cat(sentiments).numpy(),
            "sentiment_targets": torch.cat(sentiment_targets).numpy(),
            "latency_seconds": elapsed if measure_latency else None,
            "samples_per_second": samples / elapsed if measure_latency and elapsed else None,
            "record_types": record_types,
            "sample_ids": sample_ids,
        }
    import numpy as np

    return {
        "theme_probs": np.zeros((0, 0)),
        "theme_targets": np.zeros((0, 0)),
        "sentiment_predictions": np.zeros((0, 0)),
        "sentiment_targets": np.zeros((0, 0)),
        "latency_seconds": elapsed if measure_latency else None,
        "samples_per_second": 0.0,
        "record_types": [],
        "sample_ids": [],
    }


def _calibrate_thresholds(
    np: Any, probabilities: Any, targets: Any, *, min_support: int = 3
) -> list[float]:
    """Per-theme thresholds with a global fallback, matching `service_train`.

    A theme with no Gold positives scores F1 0 at every threshold, so a plain
    argmax settles on whichever threshold the grid visits first and hands that
    theme the loosest setting available. Themes below `min_support` therefore
    take the globally calibrated threshold instead of a value fitted to noise.
    """
    if probabilities.size == 0:
        return [0.5] * (targets.shape[1] if targets.ndim == 2 else 0)
    default = _best_threshold(np, probabilities.reshape(-1), targets.reshape(-1))
    thresholds: list[float] = []
    for column in range(probabilities.shape[1]):
        if int((targets[:, column] >= 0.5).sum()) < min_support:
            thresholds.append(default)
            continue
        thresholds.append(_best_threshold(np, probabilities[:, column], targets[:, column]))
    return thresholds


def _best_threshold(np: Any, probabilities: Any, targets: Any) -> float:
    best_threshold, best_f1 = 0.5, -1.0
    for threshold in np.arange(0.2, 0.91, 0.05):
        predicted = probabilities >= threshold
        actual = targets >= 0.5
        tp = int(np.logical_and(predicted, actual).sum())
        fp = int(np.logical_and(predicted, ~actual).sum())
        fn = int(np.logical_and(~predicted, actual).sum())
        f1 = 2 * tp / max(2 * tp + fp + fn, 1)
        if f1 > best_f1:
            best_threshold, best_f1 = round(float(threshold), 2), f1
    return best_threshold


def _metrics(np: Any, output: dict[str, Any], thresholds: list[float], train_rows: list[dict[str, Any]], themes: list[str]) -> dict[str, Any]:
    probs, actual = output["theme_probs"], output["theme_targets"] >= 0.5
    if probs.size == 0:
        return {"theme_micro_f1": 0.0, "theme_macro_f1": 0.0, "samples": 0}
    predicted = probs >= np.asarray(thresholds)[None, :]
    tp = np.logical_and(predicted, actual).sum(axis=0)
    fp = np.logical_and(predicted, ~actual).sum(axis=0)
    fn = np.logical_and(~predicted, actual).sum(axis=0)
    per_label_f1 = 2 * tp / np.maximum(2 * tp + fp + fn, 1)
    per_label_precision = tp / np.maximum(tp + fp, 1)
    per_label_recall = tp / np.maximum(tp + fn, 1)
    # Macro over themes the split actually exercises: Gold-supported themes plus
    # themes the model predicted into. Averaging a hard 0 over never-seen themes
    # buries real differences under a constant, which is what `service_train`
    # avoids in `_set_metrics`.
    supported = np.logical_or(actual.sum(axis=0) > 0, predicted.sum(axis=0) > 0)
    micro_f1 = 2 * tp.sum() / max(2 * tp.sum() + fp.sum() + fn.sum(), 1)
    micro_precision = tp.sum() / max(tp.sum() + fp.sum(), 1)
    micro_recall = tp.sum() / max(tp.sum() + fn.sum(), 1)
    train_counts = Counter(
        label["theme_id"] for row in train_rows for label in row.get("labels", [])
    )
    tail_indices = [
        index for index, theme in enumerate(themes)
        if train_counts[theme] < 20 and supported[index]
    ]
    sentiment_actual = output["sentiment_targets"]
    sentiment_mask = sentiment_actual >= 0
    sentiment_correct = np.logical_and(
        output["sentiment_predictions"] == sentiment_actual, sentiment_mask
    ).sum()
    sentiment_total = sentiment_mask.sum()
    result = {
        "samples": int(actual.shape[0]),
        "theme_micro_precision": round(float(micro_precision), 6),
        "theme_micro_recall": round(float(micro_recall), 6),
        "theme_micro_f1": round(float(micro_f1), 6),
        "theme_macro_f1": (
            round(float(per_label_f1[supported].mean()), 6) if supported.any() else 0.0
        ),
        "tail_theme_macro_f1": (
            round(float(per_label_f1[tail_indices].mean()), 6) if tail_indices else None
        ),
        "sentiment_accuracy_on_gold_themes": round(
            float(sentiment_correct / max(sentiment_total, 1)), 6
        ),
        "inference_seconds": output.get("latency_seconds"),
        "samples_per_second": output.get("samples_per_second"),
        "per_theme": {
            theme: {
                "precision": round(float(per_label_precision[index]), 6),
                "recall": round(float(per_label_recall[index]), 6),
                "f1": round(float(per_label_f1[index]), 6),
                "gold_support": int(actual[:, index].sum()),
                "train_support": int(train_counts[theme]),
            }
            for index, theme in enumerate(themes)
        },
    }
    record_types = np.asarray(output.get("record_types") or [])
    if len(record_types) == len(actual):
        result["by_record_type"] = {
            kind: _theme_metrics_for_mask(np, predicted, actual, record_types == kind)
            for kind in ("review", "qna")
        }
    return result


def _theme_metrics_for_mask(np: Any, predicted: Any, actual: Any, mask: Any) -> dict[str, Any]:
    predicted = predicted[mask]
    actual = actual[mask]
    tp = np.logical_and(predicted, actual).sum()
    fp = np.logical_and(predicted, ~actual).sum()
    fn = np.logical_and(~predicted, actual).sum()
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * tp / max(2 * tp + fp + fn, 1)
    return {
        "samples": int(mask.sum()),
        "theme_micro_precision": round(float(precision), 6),
        "theme_micro_recall": round(float(recall), 6),
        "theme_micro_f1": round(float(f1), 6),
    }


def _write_predictions(
    path: Path,
    output: dict[str, Any],
    thresholds: list[float],
    themes: list[str],
) -> None:
    rows = []
    for row_index, sample_id in enumerate(output.get("sample_ids") or []):
        probabilities = output["theme_probs"][row_index]
        targets = output["theme_targets"][row_index] >= 0.5
        predicted = probabilities >= thresholds
        rows.append(
            {
                "sample_id": sample_id,
                "record_type": output["record_types"][row_index],
                "truth_theme_ids": [
                    theme for index, theme in enumerate(themes) if bool(targets[index])
                ],
                "predictions": [
                    {
                        "theme_id": theme,
                        "probability": round(float(probabilities[index]), 6),
                        "threshold": round(float(thresholds[index]), 6),
                    }
                    for index, theme in enumerate(themes)
                    if bool(predicted[index])
                ],
                "false_positive_theme_ids": [
                    theme
                    for index, theme in enumerate(themes)
                    if bool(predicted[index]) and not bool(targets[index])
                ],
                "false_negative_theme_ids": [
                    theme
                    for index, theme in enumerate(themes)
                    if not bool(predicted[index]) and bool(targets[index])
                ],
            }
        )
    write_jsonl(path, rows)
