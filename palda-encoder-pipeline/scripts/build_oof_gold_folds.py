#!/usr/bin/env python3
"""Build product/text-isolated folds for out-of-fold Gold error mining."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import unicodedata
from collections import defaultdict
from pathlib import Path
from typing import Any


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def normalized_text(value: object) -> str:
    return " ".join(unicodedata.normalize("NFKC", str(value)).casefold().split())


class UnionFind:
    def __init__(self, size: int) -> None:
        self.parent = list(range(size))

    def find(self, value: int) -> int:
        while self.parent[value] != value:
            self.parent[value] = self.parent[self.parent[value]]
            value = self.parent[value]
        return value

    def union(self, left: int, right: int) -> None:
        left_root, right_root = self.find(left), self.find(right)
        if left_root != right_root:
            self.parent[right_root] = left_root


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--release-dir", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--folds", type=int, default=4)
    args = parser.parse_args()
    if args.folds < 2:
        raise ValueError("folds must be at least 2")

    rows = read_jsonl(args.release_dir / "train.jsonl")
    union = UnionFind(len(rows))
    first_by_product: dict[str, int] = {}
    first_by_text: dict[str, int] = {}
    for index, row in enumerate(rows):
        product = str(row["product_id"])
        text = normalized_text(row["text"])
        if product in first_by_product:
            union.union(index, first_by_product[product])
        else:
            first_by_product[product] = index
        if text in first_by_text:
            union.union(index, first_by_text[text])
        else:
            first_by_text[text] = index

    components: dict[int, list[int]] = defaultdict(list)
    for index in range(len(rows)):
        components[union.find(index)].append(index)
    ordered = sorted(
        components.values(),
        key=lambda values: (
            -sum(rows[index].get("label_tier") == "gold" for index in values),
            hashlib.sha256(str(rows[values[0]]["sample_id"]).encode()).hexdigest(),
        ),
    )
    fold_gold_counts = [0] * args.folds
    fold_assignments: dict[int, int] = {}
    for values in ordered:
        gold_count = sum(rows[index].get("label_tier") == "gold" for index in values)
        if gold_count:
            fold = min(range(args.folds), key=lambda value: (fold_gold_counts[value], value))
            fold_gold_counts[fold] += gold_count
        else:
            digest = hashlib.sha256(str(rows[values[0]]["sample_id"]).encode()).digest()
            fold = int.from_bytes(digest[:4], "big") % args.folds
        for index in values:
            fold_assignments[index] = fold

    args.output_root.mkdir(parents=True, exist_ok=True)
    manifest: dict[str, Any] = {"schemaVersion": 1, "folds": args.folds, "items": []}
    for fold in range(args.folds):
        release = args.output_root / f"fold-{fold}" / "release"
        release.mkdir(parents=True, exist_ok=True)
        train = [row for index, row in enumerate(rows) if fold_assignments[index] != fold]
        validation = [
            row for index, row in enumerate(rows)
            if fold_assignments[index] == fold and row.get("label_tier") == "gold"
        ]
        write_jsonl(release / "train.jsonl", train)
        write_jsonl(release / "validation.jsonl", validation)
        shutil.copyfile(args.release_dir / "taxonomy.json", release / "taxonomy.json")
        train_products = {str(row["product_id"]) for row in train}
        validation_products = {str(row["product_id"]) for row in validation}
        train_texts = {normalized_text(row["text"]) for row in train}
        validation_texts = {normalized_text(row["text"]) for row in validation}
        item = {
            "fold": fold,
            "trainRows": len(train),
            "validationGoldRows": len(validation),
            "validationProducts": len(validation_products),
            "productOverlap": len(train_products & validation_products),
            "normalizedTextOverlap": len(train_texts & validation_texts),
        }
        if item["productOverlap"] or item["normalizedTextOverlap"]:
            raise RuntimeError(f"fold isolation failed: {item}")
        manifest["items"].append(item)
    manifest["validationGoldRows"] = sum(item["validationGoldRows"] for item in manifest["items"])
    (args.output_root / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
