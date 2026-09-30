#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from palda_encoder.io import iter_jsonl, write_json


def _counts(path: Path) -> dict[str, Counter[str]]:
    result = {
        "review": Counter(),
        "qna_primary": Counter(),
        "qna_secondary": Counter(),
    }
    for row in iter_jsonl(path):
        if row["record_type"] == "review":
            for label in row.get("labels", []):
                result["review"][f"{label['theme_id']}::{label.get('sentiment')}"] += 1
            continue
        labels = row.get("labels", [])
        primary = next((label for label in labels if label.get("role") == "primary"), labels[0] if labels else None)
        if primary:
            result["qna_primary"][primary["theme_id"]] += 1
        for label in labels:
            if label is not primary and label.get("role") != "primary":
                result["qna_secondary"][label["theme_id"]] += 1
    return result


def audit(release_dir: Path, low_support: int) -> dict[str, object]:
    train = _counts(release_dir / "train.jsonl")
    validation = _counts(release_dir / "validation.jsonl")
    report: dict[str, object] = {"release": str(release_dir), "tasks": {}}
    for task in train:
        gold_labels = set(validation[task])
        missing = sorted(label for label in gold_labels if train[task][label] == 0)
        low = sorted(
            (
                {"label": label, "train": train[task][label], "validation": validation[task][label]}
                for label in gold_labels
                if 0 < train[task][label] < low_support
            ),
            key=lambda value: (value["train"], -value["validation"], value["label"]),
        )
        report["tasks"][task] = {
            "train_total": sum(train[task].values()),
            "validation_total": sum(validation[task].values()),
            "train_labels": len(train[task]),
            "validation_labels": len(validation[task]),
            "missing_validation_labels": missing,
            "low_support_validation_labels": low,
            "train_top": train[task].most_common(20),
            "validation_top": validation[task].most_common(20),
        }
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit train coverage against Gold validation labels")
    parser.add_argument("--release-dir", type=Path, required=True)
    parser.add_argument("--low-support", type=int, default=10)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = audit(args.release_dir, args.low_support)
    if args.output:
        write_json(args.output, report)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
