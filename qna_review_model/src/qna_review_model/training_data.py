"""Create explicitly weak, record-level bootstrap targets for QLoRA plumbing tests."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any

from .pilot import AnnotationSample
from .schemas import QNARecord, ReviewRecord, ThemeDefinition


SYSTEM_PROMPT = (
    "당신은 PALDA의 카테고리 인지형 고객의견 분석기입니다. "
    "주어진 허용 theme에서 텍스트에 직접 나타난 theme만 고르고 JSON 객체만 출력하세요. "
    "추측하거나 원문에 없는 근거를 만들지 마세요."
)


def _mentions(record: ReviewRecord | QNARecord, themes: list[ThemeDefinition]) -> list[dict[str, str]]:
    text = record.text if isinstance(record, ReviewRecord) else record.question
    lowered = text.casefold()
    polarity = "concern"
    if isinstance(record, ReviewRecord):
        polarity = "positive" if record.score is not None and record.score >= 4 else (
            "negative" if record.score is not None and record.score <= 2 else "neutral"
        )
    return [
        {"theme_id": theme.theme_id, "polarity": polarity}
        for theme in themes if any(keyword.casefold() in lowered for keyword in theme.keywords)
    ]


def _row(sample: AnnotationSample, record: ReviewRecord | QNARecord) -> dict[str, Any]:
    task = "REVIEW" if isinstance(record, ReviewRecord) else "QNA"
    text = record.text if isinstance(record, ReviewRecord) else record.question
    allowed = [{"theme_id": theme.theme_id, "label": theme.label} for theme in sample.category_themes]
    target = {"record_id": record.record_id, "mentions": _mentions(record, sample.category_themes)}
    user = (
        f"TASK: {task}\n"
        f"CATEGORY: {sample.product.category_path or sample.product.cat1}\n"
        f"ALLOWED_THEMES: {json.dumps(allowed, ensure_ascii=False, separators=(',', ':'))}\n"
        f"RECORD_ID: {record.record_id}\n"
        f"TEXT: {text}"
    )
    return {
        "sample_id": sample.sample_id, "record_id": record.record_id, "split": sample.split,
        "source_label": "bootstrap_keyword_v1",
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user},
            {"role": "assistant", "content": json.dumps(target, ensure_ascii=False, separators=(",", ":"))},
        ],
    }


def build_bootstrap_dataset(pilot_dir: Path, output_dir: Path) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    counts: Counter[str] = Counter()
    positive_targets: Counter[str] = Counter()
    product_ids: dict[str, set[str]] = {}
    for split in ("train", "validation", "test"):
        product_ids[split] = set()
        with (pilot_dir / f"{split}.jsonl").open(encoding="utf-8") as source, (
            output_dir / f"{split}.jsonl"
        ).open("w", encoding="utf-8") as target_file:
            for line in source:
                sample = AnnotationSample.model_validate_json(line)
                product_ids[split].add(sample.product.product_id)
                for record in [*sample.reviews, *sample.qna]:
                    row = _row(sample, record)
                    target_file.write(json.dumps(row, ensure_ascii=False) + "\n")
                    counts[split] += 1
                    parsed = json.loads(row["messages"][-1]["content"])
                    positive_targets[split] += bool(parsed["mentions"])
    overlap = any(product_ids[a] & product_ids[b] for a, b in (
        ("train", "validation"), ("train", "test"), ("validation", "test")
    ))
    if overlap:
        raise ValueError("product leakage detected between bootstrap splits")
    manifest = {
        "schema_version": "palda_sft_manifest_v1",
        "source_label": "bootstrap_keyword_v1",
        "warning": "Weak keyword labels for pipeline smoke tests only; not Gold and not a quality baseline.",
        "records": dict(counts), "records_with_mentions": dict(positive_targets),
        "product_split_leakage": False,
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return {"output_dir": str(output_dir), **manifest}
