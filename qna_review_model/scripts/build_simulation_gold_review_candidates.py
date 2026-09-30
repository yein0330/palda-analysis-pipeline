#!/usr/bin/env python
"""Select a diverse single-reviewer Gold eval queue from simulator Validation/Test."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

from qna_review_model.encoder_data import EncoderTrainingRecord


def stable(seed: int, *values: object) -> str:
    return hashlib.sha256("\n".join(map(str, (seed, *values))).encode()).hexdigest()


def assignments(path: Path) -> dict[str, str]:
    result = {}
    for split in ("validation", "test"):
        with (path / f"{split}.jsonl").open(encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    row = json.loads(line); result[row["original_item_id"]] = split
    return result


def quota(total: int, names: list[str]) -> dict[str, int]:
    base, remainder = divmod(total, len(names))
    return {name: base + (index < remainder) for index, name in enumerate(sorted(names))}


def choose(rows: list[EncoderTrainingRecord], count: int, seed: int) -> list[EncoderTrainingRecord]:
    cat1s = sorted({row.category.level1Name for row in rows})
    source_targets = {"review": round(count * 0.65)}
    source_targets["qna"] = count - source_targets["review"]
    selected = []; selected_ids = set(); selected_products = set()
    for source in ("review", "qna"):
        cat_targets = quota(source_targets[source], cat1s)
        for cat1, target in cat_targets.items():
            candidates = [row for row in rows if row.sourceType == source and row.category.level1Name == cat1]
            candidates.sort(key=lambda row: stable(seed, source, cat1, row.productGroupId, row.itemId))
            # One item per product first, then fill any sparse stratum with remaining items.
            first = []; seen = set()
            for row in candidates:
                if row.productGroupId not in seen:
                    seen.add(row.productGroupId); first.append(row)
            ordered = [*first, *(row for row in candidates if row.itemId not in {x.itemId for x in first})]
            for row in ordered:
                if row.itemId in selected_ids or row.productGroupId in selected_products:
                    continue
                selected.append(row); selected_ids.add(row.itemId); selected_products.add(row.productGroupId)
                if sum(x.sourceType == source and x.category.level1Name == cat1 for x in selected) >= target:
                    break
    if len(selected) < count:
        leftovers = sorted((row for row in rows if row.itemId not in selected_ids),
                           key=lambda row: (row.productGroupId in selected_products, stable(seed, "fill", row.itemId)))
        selected.extend(leftovers[:count - len(selected)])
    return selected[:count]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--encoder-source", type=Path, required=True)
    parser.add_argument("--simulation-split-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--per-split", type=int, default=500)
    parser.add_argument("--seed", type=int, default=20260901)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite: {args.output}")
    split_by_id = assignments(args.simulation_split_dir)
    pools = {"validation": [], "test": []}; gold_products = defaultdict(set)
    records = [EncoderTrainingRecord.model_validate_json(line) for line in args.encoder_source.read_text(encoding="utf-8").splitlines() if line.strip()]
    for row in records:
        split = split_by_id.get(row.itemId)
        if split is None:
            continue
        updated = row.model_copy(update={"annotation": row.annotation.model_copy(update={"sourceSplit": split})})
        if updated.annotation.grade == "gold":
            gold_products[split].add(updated.productGroupId)
        else:
            pools[split].append(updated)
    chosen = []
    for split in ("validation", "test"):
        fresh = [row for row in pools[split] if row.productGroupId not in gold_products[split]]
        source = fresh if len(fresh) >= args.per_split else pools[split]
        chosen.extend(choose(source, args.per_split, args.seed + (split == "test")))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        for row in chosen:
            handle.write(row.model_dump_json() + "\n")
    manifest = {
        "schemaVersion": "palda-final-gold-review-candidates-v1",
        "source": str(args.encoder_source), "splitVersion": "product-dup-balanced-70-15-15-v2",
        "records": len(chosen), "products": len({row.productGroupId for row in chosen}),
        "split": dict(Counter(row.annotation.sourceSplit for row in chosen)),
        "recordType": dict(Counter(row.sourceType for row in chosen)),
        "cat1": dict(Counter(row.category.level1Name for row in chosen)),
        "gradeBeforeReview": dict(Counter(row.annotation.grade for row in chosen)),
        "existingGoldProductsExcluded": {key: len(value) for key, value in gold_products.items()},
        "seed": args.seed,
    }
    args.output.with_suffix(".manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
