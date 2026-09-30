#!/usr/bin/env python
"""Merge Decoder draft shards and audit them against Oracle product IDs."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--cleanup-shards", action="store_true")
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def choose(existing: dict[str, Any] | None, candidate: dict[str, Any]) -> dict[str, Any]:
    if existing is None:
        return candidate
    existing_valid = existing.get("validationStatus") == "valid"
    candidate_valid = candidate.get("validationStatus") == "valid"
    if candidate_valid and not existing_valid:
        return candidate
    return existing


def main() -> None:
    args = arguments()
    base = args.dataset_dir
    train_sources = [
        base / "train.drafts.jsonl",
        *sorted(base.glob("train.shard-*.drafts.jsonl")),
    ]
    selected: dict[str, dict[str, Any]] = {}
    for path in train_sources:
        for row in read_jsonl(path):
            product_id = str(row["productGroupId"])
            selected[product_id] = choose(selected.get(product_id), row)

    oracle_train = read_jsonl(base / "train.jsonl")
    train_ids = [str(row["productGroupId"]) for row in oracle_train]
    missing = sorted(set(train_ids) - set(selected))
    extra = sorted(set(selected) - set(train_ids))
    if missing or extra:
        raise ValueError(
            f"Train draft/oracle mismatch: missing={len(missing)}, extra={len(extra)}"
        )
    train_rows = [selected[product_id] for product_id in train_ids]
    (base / "train.drafts.jsonl").write_text(
        "".join(
            json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n"
            for row in train_rows
        ),
        encoding="utf-8",
    )

    split_stats: dict[str, Any] = {}
    all_product_ids: dict[str, set[str]] = {}
    for split in ("train", "validation", "test"):
        oracle_rows = read_jsonl(base / f"{split}.jsonl")
        draft_rows = read_jsonl(base / f"{split}.drafts.jsonl")
        oracle_ids = {str(row["productGroupId"]) for row in oracle_rows}
        draft_ids = [str(row["productGroupId"]) for row in draft_rows]
        if len(draft_ids) != len(set(draft_ids)):
            raise ValueError(f"duplicate productGroupId in {split} drafts")
        if set(draft_ids) != oracle_ids:
            raise ValueError(f"{split} drafts do not match Oracle inputs")
        all_product_ids[split] = set(draft_ids)
        statuses = Counter(row.get("validationStatus") for row in draft_rows)
        split_stats[split] = {
            "products": len(draft_rows),
            "validationStatus": dict(statuses),
            "normalizationWarnings": sum(
                bool(row.get("normalizationWarnings")) for row in draft_rows
            ),
        }

    leakage = any(all_product_ids[a] & all_product_ids[b] for a, b in (
        ("train", "validation"),
        ("train", "test"),
        ("validation", "test"),
    ))
    if leakage:
        raise ValueError("product leakage detected between Decoder draft splits")

    manifest_path = base / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["draftGeneration"] = {
        "status": "draft_pending_human_review",
        "model": "Qwen/Qwen3-8B",
        "outputSchemaVersion": "report-output-v1",
        "splits": split_stats,
        "productSplitLeakage": False,
        "trainingEligibleNow": False,
        "trainingGate": (
            "Human approval is required; two structurally invalid drafts remain "
            "explicitly marked needs_review."
        ),
    }
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    removed = []
    if args.cleanup_shards:
        for path in sorted(base.glob("train.shard-*.drafts.jsonl")):
            path.unlink()
            removed.append(path.name)
    print(json.dumps({
        "datasetDir": str(base),
        "splits": split_stats,
        "productSplitLeakage": False,
        "removedTemporaryShards": removed,
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
