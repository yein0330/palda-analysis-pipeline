#!/usr/bin/env python3
"""Convert the service review package into a diverse, isolated labeling pilot."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def select_diverse(
    rows: list[dict[str, Any]], record_type: str, limit: int
) -> list[dict[str, Any]]:
    pools: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if row["sourceType"] == record_type:
            pools[row["hardNegative"]["issue"]].append(row)
    for values in pools.values():
        values.sort(
            key=lambda row: (
                row["category"]["level2Id"],
                -float(row["hardNegative"]["score"]),
                row["itemId"],
            )
        )
    selected: list[dict[str, Any]] = []
    active = sorted(pools)
    while active and len(selected) < limit:
        remaining = []
        for issue in active:
            if len(selected) >= limit:
                break
            selected.append(pools[issue].pop(0))
            if pools[issue]:
                remaining.append(issue)
        active = remaining
    return selected


def sample_row(row: dict[str, Any]) -> dict[str, Any]:
    category = row["category"]
    return {
        "sample_id": row["itemId"],
        "product_id": row["productGroupId"],
        "record_type": row["sourceType"],
        "text": row["text"],
        "category_path": category["path"],
        "category_key": f"{category['level1Name']} > {category['level2Name']}",
        "source_kind": "real",
        "source_split": "train",
        "usable": bool(row.get("usable", True)),
        "metadata": {
            "created_at": row.get("createdAt"),
            "hard_negative_issue": row["hardNegative"]["issue"],
            "hard_negative_score": row["hardNegative"]["score"],
            "pool_status": row["hardNegative"]["poolStatus"],
        },
    }


def blocked_sample_ids(directories: list[Path]) -> set[str]:
    result: set[str] = set()
    for directory in directories:
        for split in ("train", "validation", "test", "simulation"):
            path = directory / f"{split}.jsonl"
            if path.exists():
                result.update(row["sample_id"] for row in read_jsonl(path))
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--blocked-dir", type=Path, action="append", default=[])
    parser.add_argument("--review-limit", type=int, default=240)
    parser.add_argument("--qna-limit", type=int, default=160)
    args = parser.parse_args()

    source = read_jsonl(args.source)
    selected = [
        *select_diverse(source, "review", args.review_limit),
        *select_diverse(source, "qna", args.qna_limit),
    ]
    rows = [sample_row(row) for row in selected]
    ids = [row["sample_id"] for row in rows]
    if len(ids) != len(set(ids)):
        raise ValueError("Hard-negative pilot contains duplicate sample ids")
    overlap = set(ids) & blocked_sample_ids(args.blocked_dir)
    if overlap:
        raise ValueError(f"Hard-negative pilot overlaps protected splits: {len(overlap)}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    payload = "".join(
        json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows
    )
    args.output.write_text(payload, encoding="utf-8")
    issues = Counter(row["metadata"]["hard_negative_issue"] for row in rows)
    report = {
        "schema": "palda-hard-negative-labeling-pilot-v1",
        "rows": len(rows),
        "review": sum(row["record_type"] == "review" for row in rows),
        "qna": sum(row["record_type"] == "qna" for row in rows),
        "products": len({row["product_id"] for row in rows}),
        "category_keys": len({row["category_key"] for row in rows}),
        "issues": dict(sorted(issues.items())),
        "protected_overlap": 0,
        "sha256": hashlib.sha256(payload.encode()).hexdigest(),
    }
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
