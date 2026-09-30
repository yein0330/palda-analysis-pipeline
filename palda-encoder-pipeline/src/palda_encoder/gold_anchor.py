from __future__ import annotations

import hashlib
from collections import Counter
from pathlib import Path
from typing import Any

from .io import iter_jsonl, read_json, sha256_file, stable_hash, write_json, write_jsonl


def create_gold_anchor_releases(
    benchmark_dir: str | Path,
    base_silver_release_dir: str | Path,
    output_root: str | Path,
    *,
    selective_release_dir: str | Path | None = None,
    train_ratio_of_simulation: float = 0.5,
    seed: int = 20260909,
) -> dict[str, Any]:
    if not 0 < train_ratio_of_simulation < 1:
        raise ValueError("train_ratio_of_simulation must be between zero and one")
    benchmark = Path(benchmark_dir)
    base = Path(base_silver_release_dir)
    simulation = list(iter_jsonl(benchmark / "simulation.jsonl"))
    validation = list(iter_jsonl(benchmark / "validation.jsonl"))
    test = list(iter_jsonl(benchmark / "test.jsonl"))
    gold_train, simulation_holdout = _split_products(
        simulation, train_ratio=train_ratio_of_simulation, seed=seed
    )
    for row in gold_train:
        row["source_split"] = "train"
        row["label_tier"] = "gold"
    for row in simulation_holdout:
        row["source_split"] = "simulation"
    silver = list(iter_jsonl(base / "train.jsonl"))
    variants = {
        "gold_only": list(gold_train),
        "gold_silver": [*silver, *gold_train],
    }
    if selective_release_dir:
        selective = list(iter_jsonl(Path(selective_release_dir) / "train.jsonl"))
        variants["gold_silver_selective"] = [*selective, *gold_train]
    taxonomy = read_json(base / "taxonomy.json")
    output = Path(output_root)
    report: dict[str, Any] = {
        "schema": "palda-gold-anchor-experiment-v1",
        "seed": seed,
        "simulation_policy": "product-isolated half to Gold train; remainder sealed for simulation",
        "gold_anchor": {"rows": len(gold_train), "products": len(_products(gold_train))},
        "simulation_holdout": {
            "rows": len(simulation_holdout), "products": len(_products(simulation_holdout))
        },
        "variants": {},
    }
    for name, train in variants.items():
        destination = output / name
        destination.mkdir(parents=True, exist_ok=True)
        _assert_no_leakage(train, validation, test, simulation_holdout)
        files = {"train": train, "validation": validation, "test": test,
                 "simulation": simulation_holdout}
        for split, rows in files.items():
            write_jsonl(destination / f"{split}.jsonl", rows)
        write_json(destination / "taxonomy.json", taxonomy)
        manifest = {
            "schema": "palda-gold-anchor-release-v1",
            "strategy": name,
            "counts": {split: len(rows) for split, rows in files.items()},
            "products": {split: len(_products(rows)) for split, rows in files.items()},
            "label_tiers": dict(Counter(row.get("label_tier", "unknown") for row in train)),
            "leakage": {"product": 0, "exact_text": 0},
            "files": {split: {"path": f"{split}.jsonl",
                               "sha256": sha256_file(destination / f"{split}.jsonl")}
                      for split in files},
            "taxonomy": {"path": "taxonomy.json",
                         "sha256": sha256_file(destination / "taxonomy.json")},
        }
        write_json(destination / "manifest.json", manifest)
        report["variants"][name] = manifest
    write_json(output / "manifest.json", report)
    return report


def _split_products(
    rows: list[dict[str, Any]], *, train_ratio: float, seed: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    counts = Counter(str(row.get("product_id") or row["sample_id"]) for row in rows)
    ordered = sorted(counts, key=lambda value: stable_hash(value, seed))
    target = round(len(rows) * train_ratio)
    selected: set[str] = set()
    selected_rows = 0
    for product in ordered:
        size = counts[product]
        if abs((selected_rows + size) - target) <= abs(selected_rows - target):
            selected.add(product)
            selected_rows += size
    train, holdout = [], []
    for raw in rows:
        row = dict(raw)
        product = str(row.get("product_id") or row["sample_id"])
        (train if product in selected else holdout).append(row)
    return train, holdout


def _products(rows: list[dict[str, Any]]) -> set[str]:
    return {str(row.get("product_id") or row["sample_id"]) for row in rows}


def _text_keys(rows: list[dict[str, Any]]) -> set[str]:
    return {hashlib.sha256("".join(str(row["text"]).split()).lower().encode()).hexdigest()
            for row in rows}


def _assert_no_leakage(
    train: list[dict[str, Any]], validation: list[dict[str, Any]], test: list[dict[str, Any]],
    simulation: list[dict[str, Any]],
) -> None:
    splits = {"train": train, "validation": validation, "test": test, "simulation": simulation}
    names = list(splits)
    for index, left in enumerate(names):
        for right in names[index + 1:]:
            product_overlap = _products(splits[left]) & _products(splits[right])
            text_overlap = _text_keys(splits[left]) & _text_keys(splits[right])
            if product_overlap or text_overlap:
                raise AssertionError(
                    f"release leakage {left}/{right}: products={len(product_overlap)}, texts={len(text_overlap)}"
                )
