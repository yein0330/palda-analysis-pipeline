#!/usr/bin/env python3
"""Summarize out-of-fold Gold errors without exposing validation/test rows."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fold-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--examples", type=Path, required=True)
    parser.add_argument("--limit-per-kind", type=int, default=200)
    args = parser.parse_args()

    qna_confusions: Counter[tuple[str, str]] = Counter()
    review_false_positive: Counter[tuple[str, str]] = Counter()
    review_false_negative: Counter[tuple[str, str]] = Counter()
    secondary_false_positive: Counter[str] = Counter()
    secondary_false_negative: Counter[str] = Counter()
    example_buckets: dict[str, list[dict[str, Any]]] = {}
    sample_ids: set[str] = set()

    for fold_dir in sorted(args.fold_root.glob("fold-*")):
        gold = {row["sample_id"]: row for row in read_jsonl(fold_dir / "release/validation.jsonl")}
        predicted = {
            row["sample_id"]: row
            for row in read_jsonl(fold_dir / "model/validation_predictions.jsonl")
        }
        if set(gold) != set(predicted):
            raise RuntimeError(f"prediction coverage mismatch in {fold_dir.name}")
        for sample_id, row in gold.items():
            if sample_id in sample_ids:
                raise RuntimeError(f"duplicate OOF prediction: {sample_id}")
            sample_ids.add(sample_id)
            prediction = predicted[sample_id]
            issues: list[str] = []
            if row["record_type"] == "review":
                actual = {(x["theme_id"], x["sentiment"]) for x in row.get("labels", [])}
                guessed = {
                    (x["theme_id"], x["sentiment"])
                    for x in prediction.get("predicted_labels", [])
                }
                for label in guessed - actual:
                    review_false_positive[label] += 1
                    issues.append(f"review_fp:{label[0]}:{label[1]}")
                for label in actual - guessed:
                    review_false_negative[label] += 1
                    issues.append(f"review_fn:{label[0]}:{label[1]}")
            else:
                labels = row.get("labels", [])
                primary = next((x for x in labels if x.get("role") == "primary"), labels[0] if labels else None)
                actual_primary = primary["theme_id"] if primary else "OTHER"
                guessed_primary = prediction["primary_theme_id"]
                if actual_primary != guessed_primary:
                    qna_confusions[(actual_primary, guessed_primary)] += 1
                    issues.append(f"qna_primary:{actual_primary}->{guessed_primary}")
                actual_secondary = {
                    x["theme_id"] for x in labels if x.get("role") == "secondary"
                }
                guessed_secondary = {
                    x["theme_id"] for x in prediction.get("secondary_themes", [])
                }
                for theme in guessed_secondary - actual_secondary:
                    secondary_false_positive[theme] += 1
                    issues.append(f"qna_secondary_fp:{theme}")
                for theme in actual_secondary - guessed_secondary:
                    secondary_false_negative[theme] += 1
                    issues.append(f"qna_secondary_fn:{theme}")
            if issues:
                kind = "review" if row["record_type"] == "review" else "qna"
                example_buckets.setdefault(kind, []).append({
                    "sample_id": sample_id,
                    "product_id": row["product_id"],
                    "record_type": row["record_type"],
                    "category_key": row["category_key"],
                    "category_path": row["category_path"],
                    "text": row["text"],
                    "gold_labels": row.get("labels", []),
                    "issues": issues,
                })

    def top(counter: Counter[Any], limit: int = 30) -> list[dict[str, Any]]:
        return [{"label": list(key) if isinstance(key, tuple) else key, "count": count}
                for key, count in counter.most_common(limit)]

    report = {
        "schemaVersion": 1,
        "oofGoldRows": len(sample_ids),
        "qnaPrimaryConfusions": top(qna_confusions),
        "qnaSecondaryFalsePositive": top(secondary_false_positive),
        "qnaSecondaryFalseNegative": top(secondary_false_negative),
        "reviewFalsePositive": top(review_false_positive),
        "reviewFalseNegative": top(review_false_negative),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    selected = []
    for kind in ("review", "qna"):
        values = sorted(example_buckets.get(kind, []), key=lambda row: (-len(row["issues"]), row["sample_id"]))
        selected.extend(values[: args.limit_per_kind])
    args.examples.write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in selected),
        encoding="utf-8",
    )
    print(json.dumps({**report, "exampleRows": len(selected)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
