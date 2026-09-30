from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

from .io import iter_jsonl, write_json, write_jsonl


SYSTEM_PROMPT = (
    "당신은 PALDA의 한국어 커머스 분석 모델입니다. 입력 근거만 사용하고, "
    "인용문은 원문에서 그대로 가져오며, 지정된 JSON 형식으로만 답하세요."
)


def create_training_release(
    selected_paths: Iterable[str | Path],
    output_dir: str | Path,
    *,
    seed: int = 42,
    train_ratio: float = 0.8,
    validation_ratio: float = 0.1,
) -> dict[str, Any]:
    if train_ratio <= 0 or validation_ratio < 0 or train_ratio + validation_ratio >= 1:
        raise ValueError("Split ratios must leave a positive test split")
    examples: list[dict[str, Any]] = []
    seen: set[str] = set()
    for path in selected_paths:
        for row in iter_jsonl(path):
            if row.get("status") != "accepted" or row["task_id"] in seen:
                continue
            seen.add(row["task_id"])
            task = row["task"]
            completion = row["completion"]
            prompt = task_prompt(task)
            source_texts = [item["text"] for item in task.get("source_items", [])]
            allowed_theme_ids = sorted(
                {item["theme_id"] for item in task.get("summary_items", [])}
            )
            examples.append(
                {
                    "example_id": row["task_id"],
                    "analysis_id": task["analysis_id"],
                    "task_type": task["task_type"],
                    "prompt": [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": prompt}],
                    "completion": [{"role": "assistant", "content": json.dumps(completion, ensure_ascii=False, sort_keys=True)}],
                    "source_texts": source_texts,
                    "allowed_theme_ids": allowed_theme_ids,
                    "reference_completion": json.dumps(completion, ensure_ascii=False, sort_keys=True),
                    "quality_score": row.get("quality_score"),
                }
            )
    splits = {"train": [], "validation": [], "test": []}
    for example in sorted(examples, key=lambda row: row["example_id"]):
        value = _hash_fraction(str(example["analysis_id"]), seed)
        split = "train" if value < train_ratio else "validation" if value < train_ratio + validation_ratio else "test"
        splits[split].append(example)
    out = Path(output_dir)
    for split, rows in splits.items():
        write_jsonl(out / f"{split}.jsonl", rows)
    # GRPO only needs prompts and reward context; reference completions remain reward anchors.
    write_jsonl(out / "grpo_train.jsonl", splits["train"])
    manifest = {
        "counts": {name: len(rows) for name, rows in splits.items()},
        "task_types": dict(Counter(row["task_type"] for row in examples)),
        "examples": len(examples),
        "seed": seed,
        "split_unit": "analysis_id",
    }
    write_json(out / "manifest.json", manifest)
    return manifest


def task_prompt(task: dict[str, Any]) -> str:
    if task["task_type"] == "selling_point":
        return (
            "다음 테마별 분석만 사용해 셀러가 상품 상세페이지에서 활용할 수 있는 핵심 소구점 1~3개를 만드세요. "
            "과장하거나 새로운 사실을 추가하지 마세요.\n"
            f"INPUT={json.dumps(task, ensure_ascii=False, sort_keys=True)}\n"
            'OUTPUT={"selling_points":[{"title":"...","message":"...","evidence_theme_ids":["..."]}]}'
        )
    return (
        "다음 동일 테마 문장 묶음의 반복 패턴을 요약하세요. quote는 입력 원문의 정확한 부분 문자열이어야 합니다.\n"
        f"INPUT={json.dumps(task, ensure_ascii=False, sort_keys=True)}\n"
        f'OUTPUT={{"theme_id":"{task["theme_id"]}","sentiment":{json.dumps(task.get("sentiment"), ensure_ascii=False)},"quote":"...","summary":"..."}}'
    )


def _hash_fraction(value: str, seed: int) -> float:
    raw = hashlib.sha256(f"{seed}:{value}".encode()).digest()
    return int.from_bytes(raw[:8], "big") / 2**64

