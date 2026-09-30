from __future__ import annotations

import hashlib
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from .io import iter_jsonl, read_json, sha256_file, stable_hash, write_json, write_jsonl


def create_release(
    samples_path: str | Path,
    consensus_path: str | Path,
    output_dir: str | Path,
    *,
    taxonomy_path: str | Path | None = None,
    seed: int = 42,
    train_ratio: float = 0.8,
    validation_ratio: float = 0.1,
    preserve_source_split: bool = False,
) -> dict[str, Any]:
    if train_ratio <= 0 or validation_ratio < 0 or train_ratio + validation_ratio >= 1:
        raise ValueError("Split ratios must leave a positive test split")
    samples = {row["sample_id"]: row for row in iter_jsonl(samples_path)}
    consensus = {
        row["sample_id"]: row
        for row in iter_jsonl(consensus_path)
        if row.get("status") == "accepted"
    }
    taxonomy = read_json(taxonomy_path) if taxonomy_path else None

    seen_text: set[str] = set()
    splits: dict[str, list[dict[str, Any]]] = defaultdict(list)
    duplicates = 0
    for sample_id in sorted(consensus):
        sample = samples.get(sample_id)
        if not sample:
            continue
        text_key = hashlib.sha256("".join(str(sample["text"]).split()).lower().encode("utf-8")).hexdigest()
        if text_key in seen_text:
            duplicates += 1
            continue
        seen_text.add(text_key)
        split = _split(sample, seed, train_ratio, validation_ratio, preserve_source_split)
        consensus_labels = consensus[sample_id].get("labels", [])
        released = {
                **sample,
                "usable": bool(sample.get("usable", True)) and not any(
                    label.get("theme_id") == "UNUSABLE" for label in consensus_labels
                ),
                "labels": consensus_labels,
                "label_tier": consensus[sample_id].get("label_tier", "silver"),
                "agreement_score": consensus[sample_id].get("agreement_score", 0.0),
        }
        if taxonomy:
            released["candidate_theme_ids"] = taxonomy.get("category_themes", {}).get(
                sample.get("category_key", ""),
                [item["id"] for item in taxonomy.get("themes", [])],
            )
        splits[split].append(released)

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}
    for split in ("train", "validation", "test"):
        paths[split] = out / f"{split}.jsonl"
        write_jsonl(paths[split], splits.get(split, []))

    label_counts = Counter(
        label["theme_id"]
        for rows in splits.values()
        for row in rows
        for label in row.get("labels", [])
    )
    manifest = {
        "seed": seed,
        "split_strategy": "source_split" if preserve_source_split else "product_hash",
        "counts": {split: len(splits.get(split, [])) for split in ("train", "validation", "test")},
        "record_types": dict(
            Counter(row["record_type"] for rows in splits.values() for row in rows)
        ),
        "source_kinds": dict(
            Counter(row.get("source_kind", "unknown") for rows in splits.values() for row in rows)
        ),
        "label_tiers": dict(
            Counter(row["label_tier"] for rows in splits.values() for row in rows)
        ),
        "theme_counts": dict(sorted(label_counts.items())),
        "deduplicated_texts": duplicates,
        "files": {
            split: {"path": path.name, "sha256": sha256_file(path)} for split, path in paths.items()
        },
    }
    write_json(out / "manifest.json", manifest)
    return manifest


def create_isolated_release(
    samples_path: str | Path,
    consensus_path: str | Path,
    benchmark_dir: str | Path,
    output_dir: str | Path,
    *,
    taxonomy_path: str | Path,
) -> dict[str, Any]:
    """Build Silver-train + human-Gold eval while preserving simulation isolation."""
    samples = {row["sample_id"]: row for row in iter_jsonl(samples_path)}
    accepted = {
        row["sample_id"]: row
        for row in iter_jsonl(consensus_path)
        if row.get("status") == "accepted"
    }
    taxonomy = read_json(taxonomy_path)
    benchmark = Path(benchmark_dir)
    benchmark_manifest = read_json(benchmark / "manifest.json")
    blocked_products = {
        line.strip()
        for line in (benchmark / benchmark_manifest["blocked_products"]["path"])
        .read_text(encoding="utf-8")
        .splitlines()
        if line.strip()
    }
    evaluation = {
        split: list(iter_jsonl(benchmark / f"{split}.jsonl"))
        for split in ("simulation", "validation", "test")
    }
    blocked_texts = {
        hashlib.sha256("".join(str(row["text"]).split()).lower().encode()).hexdigest()
        for rows in evaluation.values()
        for row in rows
    }
    train: list[dict[str, Any]] = []
    seen_train_texts: set[str] = set()
    excluded = Counter()
    for sample_id in sorted(accepted):
        sample = samples.get(sample_id)
        if not sample:
            excluded["missing_sample"] += 1
            continue
        product_id = str(sample.get("product_id") or sample_id)
        aliases = {product_id, product_id.removeprefix("catalog:")}
        if aliases & blocked_products:
            excluded["benchmark_product"] += 1
            continue
        text_key = hashlib.sha256(
            "".join(str(sample["text"]).split()).lower().encode()
        ).hexdigest()
        if text_key in blocked_texts:
            excluded["benchmark_text"] += 1
            continue
        if text_key in seen_train_texts:
            excluded["duplicate_train_text"] += 1
            continue
        seen_train_texts.add(text_key)
        consensus_labels = accepted[sample_id].get("labels", [])
        train.append({
            **sample,
            "usable": bool(sample.get("usable", True)) and not any(
                label.get("theme_id") == "UNUSABLE" for label in consensus_labels
            ),
            "source_split": "train",
            "labels": consensus_labels,
            "candidate_theme_ids": taxonomy.get("category_themes", {}).get(
                sample.get("category_key", ""),
                [item["id"] for item in taxonomy.get("themes", [])],
            ),
            "label_tier": "silver",
            "agreement_score": accepted[sample_id].get("agreement_score", 0.0),
        })

    splits = {"train": train, "validation": evaluation["validation"], "test": evaluation["test"]}
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    paths = {split: out / f"{split}.jsonl" for split in splits}
    for split, rows in splits.items():
        write_jsonl(paths[split], rows)
    taxonomy_output = out / "taxonomy.json"
    write_json(taxonomy_output, taxonomy)
    products = {
        split: {str(row.get("product_id") or row["sample_id"]) for row in rows}
        for split, rows in splits.items()
    }
    leakage = {
        "train_validation_products": len(products["train"] & products["validation"]),
        "train_test_products": len(products["train"] & products["test"]),
        "validation_test_products": len(products["validation"] & products["test"]),
        "train_benchmark_texts": len(seen_train_texts & blocked_texts),
    }
    if any(leakage.values()):
        raise AssertionError(f"release leakage detected: {leakage}")
    manifest = {
        "schema": "palda-isolated-encoder-release-v1",
        "split_policy": "LLM-consensus Silver train; human Gold validation/test; simulation excluded",
        "counts": {split: len(rows) for split, rows in splits.items()},
        "products": {split: len(values) for split, values in products.items()},
        "label_tiers": {"train": "silver", "validation": "gold", "test": "gold"},
        "excluded_train": dict(sorted(excluded.items())),
        "leakage": leakage,
        "simulation_benchmark": {
            "manifest": str((benchmark / "manifest.json").resolve()),
            "count": len(evaluation["simulation"]),
            "sha256": benchmark_manifest["files"]["simulation"]["sha256"],
            "included_in_release": False,
        },
        "files": {
            split: {"path": path.name, "sha256": sha256_file(path)}
            for split, path in paths.items()
        },
        "taxonomy": {"path": taxonomy_output.name, "sha256": sha256_file(taxonomy_output)},
    }
    write_json(out / "manifest.json", manifest)
    return manifest


def _split(
    sample: dict[str, Any],
    seed: int,
    train_ratio: float,
    validation_ratio: float,
    preserve_source_split: bool,
) -> str:
    # Synthetic examples may improve training coverage, but allowing them into
    # validation/test would change the benchmark every iteration and inflate scores.
    if sample.get("source_kind") == "synthetic":
        return "train"
    source_split = str(sample.get("source_split") or "").lower()
    if preserve_source_split and source_split in {"train", "validation", "test"}:
        return source_split
    value = stable_hash(str(sample.get("product_id") or sample["sample_id"]), seed)
    if value < train_ratio:
        return "train"
    if value < train_ratio + validation_ratio:
        return "validation"
    return "test"
