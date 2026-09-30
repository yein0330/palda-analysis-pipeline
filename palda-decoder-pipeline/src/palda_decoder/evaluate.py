from __future__ import annotations

import time
from pathlib import Path
from typing import Any

from .io import append_jsonl, iter_jsonl, write_json, write_jsonl
from .rewards import (
    concise_reward, format_reward, grounding_reward, label_consistency_reward, reference_reward,
)


def evaluate_adapter(
    release_dir: str | Path,
    adapter_dir: str | Path,
    output_dir: str | Path,
    *,
    batch_size: int = 4,
    max_input_length: int = 2048,
    max_new_tokens: int = 512,
    max_samples: int | None = None,
    split: str = "test",
) -> dict[str, Any]:
    return evaluate_adapter_splits(
        release_dir,
        adapter_dir,
        output_dir,
        splits=(split,),
        batch_size=batch_size,
        max_input_length=max_input_length,
        max_new_tokens=max_new_tokens,
        max_samples=max_samples,
    )[split]


def evaluate_adapter_splits(
    release_dir: str | Path,
    adapter_dir: str | Path,
    output_dir: str | Path,
    *,
    splits: tuple[str, ...] = ("validation", "test"),
    batch_size: int = 4,
    max_input_length: int = 2048,
    max_new_tokens: int = 512,
    max_samples: int | None = None,
) -> dict[str, dict[str, Any]]:
    try:
        import torch
        from peft import AutoPeftModelForCausalLM
        from transformers import AutoTokenizer
    except ImportError as exc:
        raise RuntimeError("Install training packages with: pip install -e '.[train]'") from exc
    model = AutoPeftModelForCausalLM.from_pretrained(
        str(adapter_dir), torch_dtype=torch.bfloat16, device_map="auto"
    )
    tokenizer = AutoTokenizer.from_pretrained(str(adapter_dir))
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model.eval()
    results: dict[str, dict[str, Any]] = {}
    for split in splits:
        if split not in {"validation", "test"}:
            raise ValueError(f"Unsupported evaluation split: {split}")
        rows = list(iter_jsonl(Path(release_dir) / f"{split}.jsonl"))
        if max_samples is not None:
            rows = rows[:max_samples]
        split_output = Path(output_dir) / split
        if not rows:
            metrics = {"split": split, "samples": 0, "warning": f"{split} split is empty"}
            write_json(split_output / "metrics.json", metrics)
            results[split] = metrics
            continue
        results[split] = _evaluate_rows(
            torch,
            model,
            tokenizer,
            rows,
            split_output,
            split=split,
            batch_size=batch_size,
            max_input_length=max_input_length,
            max_new_tokens=max_new_tokens,
        )
    write_json(Path(output_dir) / "metrics.json", results)
    return results


def _evaluate_rows(
    torch: Any,
    model: Any,
    tokenizer: Any,
    rows: list[dict[str, Any]],
    output: Path,
    *,
    split: str,
    batch_size: int,
    max_input_length: int,
    max_new_tokens: int,
) -> dict[str, Any]:
    prediction_path = output / "predictions.jsonl"
    completed = {
        str(row["example_id"]): str(row["prediction"])
        for row in iter_jsonl(prediction_path)
    } if prediction_path.exists() else {}
    pending = [row for row in rows if str(row["example_id"]) not in completed]
    started = time.perf_counter()
    for offset in range(0, len(pending), batch_size):
        batch = pending[offset : offset + batch_size]
        prompts = [
            tokenizer.apply_chat_template(row["prompt"], tokenize=False, add_generation_prompt=True)
            for row in batch
        ]
        encoded = tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_input_length,
        ).to(model.device)
        with torch.no_grad():
            generated = model.generate(
                **encoded,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
            )
        input_width = encoded["input_ids"].shape[1]
        values = tokenizer.batch_decode(generated[:, input_width:], skip_special_tokens=True)
        checkpoint = []
        for row, prediction in zip(batch, values):
            example_id = str(row["example_id"])
            completed[example_id] = prediction
            checkpoint.append({"example_id": example_id, "prediction": prediction})
        append_jsonl(prediction_path, checkpoint)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    predictions = [completed[str(row["example_id"])] for row in rows]
    task_types = [row["task_type"] for row in rows]
    source_texts = [row.get("source_texts", []) for row in rows]
    allowed = [row.get("allowed_theme_ids", []) for row in rows]
    references = [row["reference_completion"] for row in rows]
    component_rewards = {
        "format": format_reward(predictions, task_types),
        "grounding": grounding_reward(predictions, task_types, source_texts, allowed),
        "label_consistency": label_consistency_reward(predictions, references, task_types),
        "concise": concise_reward(predictions, task_types),
        "reference": reference_reward(predictions, references),
    }
    write_jsonl(
        output / "failures.jsonl",
        (
            {
                "example_id": row["example_id"], "task_type": row["task_type"],
                "scores": {name: values[index] for name, values in component_rewards.items()},
                "prediction": predictions[index],
                "reference_completion": row["reference_completion"],
                "source_texts": row.get("source_texts", []),
            }
            for index, row in enumerate(rows)
            if any(
                component_rewards[name][index] < 1.0
                for name in ("format", "grounding", "label_consistency", "concise")
            )
        ),
    )
    metrics = {
        "split": split,
        "samples": len(rows),
        "seconds": elapsed,
        "generated_samples": len(pending),
        "reused_samples": len(rows) - len(pending),
        "samples_per_second": len(pending) / elapsed if elapsed and pending else None,
        **{
            f"{name}_reward": sum(values) / max(len(values), 1)
            for name, values in component_rewards.items()
        },
    }
    write_json(output / "metrics.json", metrics)
    return metrics
