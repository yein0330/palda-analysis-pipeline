#!/usr/bin/env python3
"""Turn OOF Gold errors into label-preserving contrastive transformations."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--examples", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-rows", type=int, default=400)
    args = parser.parse_args()
    examples = [
        json.loads(line)
        for line in args.examples.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    plan: list[dict[str, Any]] = []
    for row in examples:
        labels = row.get("gold_labels") or []
        if not labels:
            continue
        if row["record_type"] == "review":
            # The generator claim currently supports one Review target. Keeping only
            # single-label anchors avoids silently dropping a true secondary label.
            if len(labels) != 1 or not labels[0].get("sentiment"):
                continue
            target = labels[0]
            confusable = sorted({
                issue.split(":", 2)[1]
                for issue in row["issues"]
                if issue.startswith("review_fp:")
            } - {target["theme_id"]})
            item = {
                "record_type": "review",
                "category_key": row["category_key"],
                "category_path": row["category_path"],
                "theme_id": target["theme_id"],
                "sentiment": target["sentiment"],
                "strategy": "source_transform",
                "source_text": row["text"],
                "confusable_theme_ids": confusable,
                "forbidden_theme_ids": confusable,
                "difficulty": "hard",
                "constraints": "원문의 단일 정답 테마와 감정만 유지하고 혼동 테마 의미는 추가하지 말 것",
                "count": 1,
            }
        else:
            primary = next((label for label in labels if label.get("role") == "primary"), labels[0])
            secondary = [
                label["theme_id"] for label in labels
                if label is not primary and label.get("role") == "secondary"
            ]
            confusable = sorted({
                issue.split("->", 1)[1]
                for issue in row["issues"]
                if issue.startswith("qna_primary:") and "->" in issue
            } - {primary["theme_id"], *secondary})
            item = {
                "record_type": "qna",
                "category_key": row["category_key"],
                "category_path": row["category_path"],
                "theme_id": primary["theme_id"],
                "secondary_theme_ids": secondary,
                "strategy": "source_transform",
                "source_text": row["text"],
                "confusable_theme_ids": confusable,
                "forbidden_theme_ids": confusable,
                "difficulty": "hard",
                "constraints": "원문의 primary와 secondary 의도 순서를 유지하고 혼동 의도는 추가하지 말 것",
                "count": 1,
            }
        plan.append(item)
        if len(plan) >= args.max_rows:
            break
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in plan),
        encoding="utf-8",
    )
    print(json.dumps({
        "planned": len(plan),
        "review": sum(row["record_type"] == "review" for row in plan),
        "qna": sum(row["record_type"] == "qna" for row in plan),
    }))


if __name__ == "__main__":
    main()
