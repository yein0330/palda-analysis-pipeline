#!/usr/bin/env python3
"""Add trusted train-only consensus rows without changing protected evaluation splits."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from collections import Counter
from pathlib import Path
from typing import Any


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def text_key(text: str) -> str:
    compact = "".join(text.split()).lower()
    return hashlib.sha256(compact.encode()).hexdigest()


def file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-release-dir", type=Path, required=True)
    parser.add_argument("--samples", type=Path, required=True)
    parser.add_argument("--consensus", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--min-models", type=int, default=3)
    parser.add_argument("--min-agreement", type=float, default=2 / 3)
    args = parser.parse_args()

    base = args.base_release_dir
    train = read_jsonl(base / "train.jsonl")
    protected = {
        split: read_jsonl(base / f"{split}.jsonl")
        for split in ("validation", "test", "simulation")
        if (base / f"{split}.jsonl").exists()
    }
    taxonomy = json.loads((base / "taxonomy.json").read_text(encoding="utf-8"))
    category_themes = taxonomy.get("category_themes", {})
    samples = {row["sample_id"]: row for row in read_jsonl(args.samples)}
    decisions = {row["sample_id"]: row for row in read_jsonl(args.consensus)}

    protected_products = {
        str(row.get("product_id") or row["sample_id"])
        for rows in protected.values()
        for row in rows
    }
    protected_texts = {
        text_key(str(row["text"])) for rows in protected.values() for row in rows
    }
    seen_texts = {text_key(str(row["text"])) for row in train}
    accepted: list[dict[str, Any]] = []
    skipped = Counter()
    for sample_id in sorted(samples):
        sample = samples[sample_id]
        decision = decisions.get(sample_id)
        if not decision or decision.get("status") != "accepted":
            skipped["not_accepted"] += 1
            continue
        if len(set(decision.get("models") or [])) < args.min_models:
            skipped["too_few_models"] += 1
            continue
        if float(decision.get("agreement_score", 0)) + 1e-9 < args.min_agreement:
            skipped["low_agreement"] += 1
            continue
        product_id = str(sample.get("product_id") or sample_id)
        if product_id in protected_products:
            raise ValueError(f"Candidate product overlaps a protected split: {product_id}")
        key = text_key(str(sample["text"]))
        if key in protected_texts:
            skipped["protected_text_duplicate"] += 1
            continue
        if key in seen_texts:
            skipped["train_text_duplicate"] += 1
            continue
        seen_texts.add(key)
        labels = list(decision.get("labels") or [])
        accepted.append(
            {
                **sample,
                "source_split": "train",
                "labels": labels,
                "candidate_theme_ids": category_themes.get(sample.get("category_key"), []),
                "label_tier": "silver",
                "agreement_score": decision.get("agreement_score", 0.0),
                "usable": bool(sample.get("usable", True))
                and not any(label.get("theme_id") == "UNUSABLE" for label in labels),
            }
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    files = {"train": [*train, *accepted], **protected}
    for split, rows in files.items():
        write_jsonl(args.output_dir / f"{split}.jsonl", rows)
    shutil.copy2(base / "taxonomy.json", args.output_dir / "taxonomy.json")
    report = {
        "schema": "palda-consensus-augmented-release-v1",
        "base_release": str(base.resolve()),
        "counts": {split: len(rows) for split, rows in files.items()},
        "added_silver": len(accepted),
        "skipped": dict(sorted(skipped.items())),
        "acceptance": {
            "min_models": args.min_models,
            "min_agreement": args.min_agreement,
        },
        "leakage": {"protected_product": 0, "protected_exact_text": 0},
        "files": {
            split: {
                "path": f"{split}.jsonl",
                "sha256": file_hash(args.output_dir / f"{split}.jsonl"),
            }
            for split in files
        },
        "taxonomy": {
            "path": "taxonomy.json",
            "sha256": file_hash(args.output_dir / "taxonomy.json"),
        },
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
