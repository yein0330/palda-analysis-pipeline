#!/usr/bin/env python
"""Export Encoder training/eval files from an immutable simulator split."""

from __future__ import annotations

import argparse
import json
import shutil
from collections import Counter
from pathlib import Path

from qna_review_model.encoder_data import EncoderTrainingRecord
from scripts.build_encoder_experiment_v1 import clean_train_record, normalized_text, write_jsonl


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--encoder-source", type=Path, required=True)
    parser.add_argument("--simulation-split-dir", type=Path, required=True)
    parser.add_argument("--label-map", type=Path, required=True)
    parser.add_argument("--taxonomy", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def simulation_assignment(path: Path) -> tuple[dict[str, str], dict[str, str]]:
    item_split = {}
    item_duplicate = {}
    for split in ("train", "validation", "test"):
        with (path / f"{split}.jsonl").open(encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                row = json.loads(line)
                item_id = row["original_item_id"]
                if item_id in item_split:
                    raise ValueError(f"duplicate simulation item assignment: {item_id}")
                item_split[item_id] = split
                item_duplicate[item_id] = row["duplicate_group_id"]
    return item_split, item_duplicate


def main() -> None:
    args = arguments()
    if args.output_dir.exists():
        raise FileExistsError(f"refusing to overwrite output: {args.output_dir}")
    assignment, duplicate_group = simulation_assignment(args.simulation_split_dir)
    records = [
        EncoderTrainingRecord.model_validate_json(line)
        for line in args.encoder_source.read_text(encoding="utf-8").splitlines() if line.strip()
    ]
    if set(assignment) != {row.itemId for row in records}:
        raise ValueError("simulation split IDs do not match Encoder source IDs")
    repartitioned = {name: [] for name in ("train", "validation", "test")}
    for row in records:
        split = assignment[row.itemId]
        annotation = row.annotation.model_copy(update={"sourceSplit": split})
        repartitioned[split].append(row.model_copy(update={"annotation": annotation}))

    train = []
    seen_text = set()
    audit = Counter()
    for row in repartitioned["train"]:
        cleaned, reasons = clean_train_record(row)
        if cleaned is None:
            audit.update(reasons)
            continue
        key = normalized_text(cleaned.text)
        if key in seen_text:
            audit["duplicate_text"] += 1
            continue
        seen_text.add(key)
        train.append(cleaned)
        audit.update(reasons)
    validation = [row for row in repartitioned["validation"] if row.annotation.grade == "gold"]
    test = [row for row in repartitioned["test"] if row.annotation.grade == "gold"]

    products = {
        "train": {row.productGroupId for row in train},
        "validation": {row.productGroupId for row in validation},
        "test": {row.productGroupId for row in test},
    }
    duplicate_sets = {
        split: {duplicate_group[row.itemId] for row in values}
        for split, values in {"train": train, "validation": validation, "test": test}.items()
    }
    leakage = {}
    for left, right in (("train", "validation"), ("train", "test"), ("validation", "test")):
        leakage[f"{left}_{right}"] = {
            "products": len(products[left] & products[right]),
            "duplicateGroups": len(duplicate_sets[left] & duplicate_sets[right]),
        }
    if any(value for pair in leakage.values() for value in pair.values()):
        raise ValueError(f"training split leakage: {leakage}")
    args.output_dir.mkdir(parents=True)
    write_jsonl(args.output_dir / "train.jsonl", train)
    write_jsonl(args.output_dir / "validation.jsonl", validation)
    write_jsonl(args.output_dir / "test.jsonl", test)
    shutil.copyfile(args.label_map, args.output_dir / "label_map.json")
    shutil.copyfile(args.taxonomy, args.output_dir / "theme_taxonomy.json")
    simulation_manifest = json.loads((args.simulation_split_dir / "dataset_manifest.json").read_text(encoding="utf-8"))
    manifest = {
        "schemaVersion": "encoder-simulation-split-dataset-v1",
        "datasetVersion": simulation_manifest["dataset_version"],
        "splitVersion": simulation_manifest["split_version"],
        "taxonomyVersion": simulation_manifest["taxonomy_version"],
        "source": str(args.encoder_source),
        "policy": "Simulator product+duplicate split; clean Train; Gold-only Validation/Test",
        "records": {"train": len(train), "validation": len(validation), "test": len(test)},
        "trainGrade": dict(Counter(row.annotation.grade for row in train)),
        "recordType": {
            split: dict(Counter(row.sourceType for row in values))
            for split, values in {"train": train, "validation": validation, "test": test}.items()
        },
        "products": {split: len(value) for split, value in products.items()},
        "auditCounts": dict(audit),
        "leakage": leakage,
        "warnings": [
            "Validation/Test are Gold-only subsets of the immutable simulator split.",
            "Do not tune thresholds or prompts on Test."
        ],
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
