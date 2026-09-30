#!/usr/bin/env python
"""Compare base and adapter JSON generation on held-out product records."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from qna_review_model.paths import DATA_ROOT


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen3-8B")
    parser.add_argument("--adapter", type=Path, default=Path("models/adapters/qlora-legacy/bootstrap-smoke/adapter"))
    parser.add_argument("--test-file", type=Path, default=DATA_ROOT / "archive/training/bootstrap-keyword-v1/test.jsonl")
    parser.add_argument("--output", type=Path, default=Path("models/adapters/qlora-legacy/bootstrap-smoke/generation_eval.json"))
    parser.add_argument("--samples", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--seed", type=int, default=20260826)
    return parser.parse_args()


def _parse(text: str) -> dict[str, Any] | None:
    start, end = text.find("{"), text.rfind("}")
    if start < 0 or end < start:
        return None
    try:
        value = json.loads(text[start:end + 1])
        return value if isinstance(value, dict) else None
    except json.JSONDecodeError:
        return None


def _mention_set(value: dict[str, Any] | None) -> set[tuple[str, str]]:
    if not value or not isinstance(value.get("mentions"), list):
        return set()
    return {
        (str(item.get("theme_id")), str(item.get("polarity")))
        for item in value["mentions"] if isinstance(item, dict)
    }


def _service_set(value: dict[str, Any] | None) -> set[tuple[str, str]]:
    if not value or not isinstance(value.get("service_mentions"), list):
        return set()
    return {
        (str(item.get("service_group")), str(item.get("polarity")))
        for item in value["service_mentions"] if isinstance(item, dict)
    }


def _f1(expected_sets: list[set], actual_sets: list[set]) -> tuple[float, float, float]:
    true_positive = sum(len(expected & actual) for expected, actual in zip(expected_sets, actual_sets, strict=True))
    false_positive = sum(len(actual - expected) for expected, actual in zip(expected_sets, actual_sets, strict=True))
    false_negative = sum(len(expected - actual) for expected, actual in zip(expected_sets, actual_sets, strict=True))
    precision = true_positive / (true_positive + false_positive) if true_positive + false_positive else 0.0
    recall = true_positive / (true_positive + false_negative) if true_positive + false_negative else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return round(precision, 4), round(recall, 4), round(f1, 4)


def _metrics(targets: list[dict[str, Any]], outputs: list[str]) -> dict[str, float | int]:
    parsed = [_parse(text) for text in outputs]
    product_precision, product_recall, product_f1 = _f1(
        [_mention_set(value) for value in targets], [_mention_set(value) for value in parsed]
    )
    service_precision, service_recall, service_f1 = _f1(
        [_service_set(value) for value in targets], [_service_set(value) for value in parsed]
    )
    stage_pairs = [
        (target.get("qna_stage"), prediction.get("qna_stage") if prediction else None)
        for target, prediction in zip(targets, parsed, strict=True) if target.get("qna_stage") is not None
    ]
    return {
        "samples": len(targets),
        "json_valid_rate": round(sum(value is not None for value in parsed) / len(parsed), 4),
        "exact_match_rate": round(sum(a == b for a, b in zip(targets, parsed, strict=True)) / len(parsed), 4),
        "mention_precision": product_precision, "mention_recall": product_recall, "mention_f1": product_f1,
        "service_precision": service_precision, "service_recall": service_recall, "service_f1": service_f1,
        "qna_stage_accuracy": round(sum(a == b for a, b in stage_pairs) / len(stage_pairs), 4) if stage_pairs else 0.0,
    }


def _select_rows(rows: list[dict[str, Any]], samples: int, seed: int) -> list[dict[str, Any]]:
    strata: dict[tuple[str, bool], list[dict[str, Any]]] = {
        (task, has_mentions): [] for task in ("qna", "review") for has_mentions in (False, True)
    }
    for row in rows:
        target = json.loads(row["messages"][-1]["content"])
        task = "qna" if row["record_id"].startswith("qna_") else "review"
        strata[(task, bool(target.get("mentions") or target.get("service_mentions")))].append(row)
    for key, values in strata.items():
        values.sort(key=lambda row: hashlib.sha256(f"{seed}:{row['record_id']}".encode()).digest())
    selected: list[dict[str, Any]] = []
    offset = 0
    while len(selected) < samples:
        added = False
        for key in sorted(strata):
            if offset < len(strata[key]):
                selected.append(strata[key][offset])
                added = True
                if len(selected) == samples:
                    break
        if not added:
            break
        offset += 1
    return selected


def _generate(
    model: PeftModel, tokenizer: AutoTokenizer, prompts: list[str], batch_size: int,
    base: bool, max_new_tokens: int,
) -> list[str]:
    outputs: list[str] = []
    context = model.disable_adapter() if base else nullcontext()
    with context, torch.inference_mode():
        for start in range(0, len(prompts), batch_size):
            batch = tokenizer(
                prompts[start:start + batch_size], padding=True, truncation=True,
                max_length=1024, return_tensors="pt",
            ).to(model.device)
            generated = model.generate(
                **batch, max_new_tokens=max_new_tokens, do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
            )
            prompt_length = batch["input_ids"].shape[1]
            outputs.extend(tokenizer.batch_decode(generated[:, prompt_length:], skip_special_tokens=True))
    return outputs


def main() -> None:
    args = arguments()
    os.environ.setdefault("NCCL_P2P_DISABLE", "1")
    os.environ.setdefault("NCCL_IB_DISABLE", "1")
    rows = [json.loads(line) for line in args.test_file.read_text(encoding="utf-8").splitlines() if line]
    rows = _select_rows(rows, args.samples, args.seed)
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    prompts = [tokenizer.apply_chat_template(
        row["messages"][:2], tokenize=False, add_generation_prompt=True, enable_thinking=False,
    ) for row in rows]
    targets = [json.loads(row["messages"][-1]["content"]) for row in rows]
    quantization = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    base_model = AutoModelForCausalLM.from_pretrained(
        args.model, local_files_only=True, quantization_config=quantization,
        device_map={"": 0}, dtype=torch.bfloat16,
    )
    model = PeftModel.from_pretrained(base_model, args.adapter)
    model.eval()
    base_outputs = _generate(model, tokenizer, prompts, args.batch_size, base=True, max_new_tokens=args.max_new_tokens)
    adapter_outputs = _generate(model, tokenizer, prompts, args.batch_size, base=False, max_new_tokens=args.max_new_tokens)
    report = {
        "evaluation_source": str(args.test_file),
        "base": _metrics(targets, base_outputs), "adapter": _metrics(targets, adapter_outputs),
        "examples": [
            {"record_id": row["record_id"], "target": target, "base": base, "adapter": adapter}
            for row, target, base, adapter in zip(rows[:5], targets[:5], base_outputs[:5], adapter_outputs[:5], strict=True)
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
