#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

from palda_encoder.io import iter_jsonl


SPECIAL_THEMES = {"OTHER", "UNUSABLE"}


def _labels(path: Path) -> tuple[Counter[tuple[str, str, str | None]], dict[tuple[str, str, str | None], set[str]]]:
    counts: Counter[tuple[str, str, str | None]] = Counter()
    categories: dict[tuple[str, str, str | None], set[str]] = defaultdict(set)
    for row in iter_jsonl(path):
        record_type = row["record_type"]
        labels = row.get("labels", [])
        if record_type == "review":
            values = [("review", label["theme_id"], label.get("sentiment")) for label in labels]
        else:
            primary = next((label for label in labels if label.get("role") == "primary"), labels[0] if labels else None)
            values = [("qna", primary["theme_id"], None)] if primary else []
        for key in values:
            counts[key] += 1
            categories[key].add(row["category_key"])
    return counts, categories


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a Gold-validation-guided coverage generation plan")
    parser.add_argument("--release-dir", type=Path, required=True)
    parser.add_argument("--taxonomy", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-total", type=int, default=1500)
    parser.add_argument("--max-per-cell", type=int, default=3)
    args = parser.parse_args()
    train, _ = _labels(args.release_dir / "train.jsonl")
    validation, validation_categories = _labels(args.release_dir / "validation.jsonl")
    taxonomy = json.loads(args.taxonomy.read_text(encoding="utf-8"))
    category_themes = taxonomy.get("category_themes") or {}

    deficits = []
    for key, gold_support in validation.items():
        record_type, theme, sentiment = key
        if theme in SPECIAL_THEMES:
            continue
        multiplier = 3 if record_type == "qna" or sentiment != "positive" else 2
        floor = 30 if sentiment == "negative" else 20
        target = max(floor, gold_support * multiplier)
        deficit = min(max(target - train[key], 0), 150)
        if deficit:
            deficits.append((train[key] / target, -gold_support, key, deficit))
    deficits.sort()

    rows = []
    planned = 0
    for _, _, key, deficit in deficits:
        if planned >= args.max_total:
            break
        record_type, theme, sentiment = key
        eligible = sorted(
            category for category, themes in category_themes.items()
            if theme in themes
        )
        preferred = sorted(validation_categories[key])
        categories = preferred + [category for category in eligible if category not in validation_categories[key]]
        if not categories:
            continue
        remaining = min(deficit, args.max_total - planned)
        for category in categories:
            if remaining <= 0:
                break
            count = min(args.max_per_cell, remaining)
            rows.append({
                "record_type": record_type,
                "category_key": category,
                "category_path": category,
                "theme_id": theme,
                "secondary_theme_ids": [],
                "sentiment": sentiment,
                "usable": True,
                "difficulty": "hard",
                "constraints": (
                    "목표 테마와 감정이 명확하되 다른 테마는 추가하지 말고, 실제 쇼핑몰에서 "
                    "볼 법한 구어체·오탈자·구체적 사용 맥락을 다양하게 포함할 것"
                    if record_type == "review" else
                    "구매 전 질문 하나만 자연스럽게 작성하고 목표 테마가 primary가 되게 할 것"
                ),
                "count": count,
            })
            planned += count
            remaining -= count
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )
    print(json.dumps({"cells": len(rows), "samples": planned, "targets": len(deficits)}))


if __name__ == "__main__":
    main()
