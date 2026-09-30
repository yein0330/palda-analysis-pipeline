#!/usr/bin/env python3
"""Convert PALDA's existing encoder-v1 split into this pipeline's release schema."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any, Iterable


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--include-needs-review", action="store_true")
    return parser.parse_args()


def read_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL at {path}:{line_no}") from exc


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def labels_for(row: dict[str, Any]) -> list[dict[str, Any]]:
    source_type = row["sourceType"]
    labels = row.get("labels") or {}
    if source_type == "review":
        return [
            {
                "theme_id": value["themeId"],
                "sentiment": value.get("sentiment"),
                "evidence": None,
            }
            for value in labels.get("reviewLabels") or []
            if value.get("themeId")
        ]
    values = []
    primary = labels.get("qnaPrimaryTheme")
    if isinstance(primary, dict) and primary.get("themeId"):
        values.append(primary["themeId"])
    for secondary in labels.get("qnaSecondaryThemes") or []:
        theme_id = secondary.get("themeId") if isinstance(secondary, dict) else secondary
        if theme_id and theme_id not in values:
            values.append(theme_id)
    return [{"theme_id": value, "sentiment": None, "evidence": None} for value in values]


def taxonomy_index(path: Path) -> tuple[dict[tuple[str, str], list[str]], dict[str, Any]]:
    source = json.loads(path.read_text(encoding="utf-8"))
    index: dict[tuple[str, str], list[str]] = {}
    category_themes: dict[str, list[str]] = {}
    for category in source.get("categories", []):
        allowed = sorted(
            item["canonicalThemeId"]
            for item in category.get("allowedThemes", [])
            if item.get("canonicalThemeId") and item.get("annotationReady", True)
        )
        index[(str(category.get("level1Id")), str(category.get("level2Id")))] = allowed
        category_themes[str(category.get("path") or "")] = allowed
    converted = {
        "version": source.get("taxonomyVersion", "unknown"),
        "themes": [
            {
                "id": item["canonicalThemeId"],
                "review_label": item.get("reviewLabel"),
                "qna_label": item.get("qnaLabel"),
            }
            for item in source.get("canonicalThemes", [])
            if item.get("canonicalThemeId")
        ],
        "category_themes": category_themes,
        "source_schema_version": source.get("schemaVersion"),
    }
    return index, converted


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    args = parse_args()
    source = args.source_dir.resolve()
    output = args.output_dir.resolve()
    taxonomy_by_ids, taxonomy = taxonomy_index(source / "theme_taxonomy.json")
    output.mkdir(parents=True, exist_ok=True)
    split_rows: dict[str, list[dict[str, Any]]] = {}
    product_sets: dict[str, set[str]] = {}
    skipped = Counter()
    theme_counts = Counter()
    tier_counts = Counter()

    for split in ("train", "validation", "test"):
        converted: list[dict[str, Any]] = []
        products: set[str] = set()
        for row in read_jsonl(source / f"{split}.jsonl"):
            annotation = row.get("annotation") or {}
            if not row.get("usable", True):
                skipped["unusable"] += 1
                continue
            if annotation.get("validationStatus") != "valid" and not args.include_needs_review:
                skipped["needs_review"] += 1
                continue
            category = row.get("category") or {}
            category_key = " > ".join(
                value for value in (category.get("level1Name"), category.get("level2Name")) if value
            )
            labels = labels_for(row)
            if not labels:
                skipped["no_labels"] += 1
                continue
            product_id = str(row.get("productGroupId") or row["itemId"])
            candidate_ids = taxonomy_by_ids.get(
                (str(category.get("level1Id")), str(category.get("level2Id"))),
                sorted({label["theme_id"] for label in labels}),
            )
            released = {
                "sample_id": str(row["itemId"]),
                "product_id": product_id,
                "record_type": str(row["sourceType"]),
                "text": str(row["text"]),
                "category_path": str(category.get("path") or category_key),
                "category_key": category_key,
                "source_kind": "real",
                "source_split": split,
                "usable": True,
                "metadata": {
                    "created_at": row.get("createdAt"),
                    "category_level1_id": category.get("level1Id"),
                    "category_level2_id": category.get("level2Id"),
                    "annotation_source": annotation.get("source"),
                    "sample_weight": annotation.get("sampleWeight", 1.0),
                },
                "labels": labels,
                "candidate_theme_ids": candidate_ids,
                "label_tier": str(annotation.get("grade") or "silver"),
                "agreement_score": 1.0 if annotation.get("grade") == "gold" else 0.75,
            }
            converted.append(released)
            products.add(product_id)
            tier_counts[released["label_tier"]] += 1
            theme_counts.update(label["theme_id"] for label in labels)
        split_rows[split] = converted
        product_sets[split] = products
        write_jsonl(output / f"{split}.jsonl", converted)

    overlaps = {
        "train_validation": sorted(product_sets["train"] & product_sets["validation"]),
        "train_test": sorted(product_sets["train"] & product_sets["test"]),
        "validation_test": sorted(product_sets["validation"] & product_sets["test"]),
    }
    if any(overlaps.values()):
        raise ValueError({name: len(values) for name, values in overlaps.items()})
    (output / "taxonomy.json").write_text(
        json.dumps(taxonomy, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    manifest = {
        "schema": "palda-encoder-pipeline-release-v1",
        "source": str(source),
        "source_manifest": str(source / "manifest.json"),
        "counts": {split: len(rows) for split, rows in split_rows.items()},
        "products": {split: len(values) for split, values in product_sets.items()},
        "label_tiers": dict(sorted(tier_counts.items())),
        "theme_counts": dict(sorted(theme_counts.items())),
        "skipped": dict(sorted(skipped.items())),
        "split_strategy": "preserved_existing_product_duplicate_split",
        "leakage": {name: len(values) for name, values in overlaps.items()},
        "files": {
            split: {"path": f"{split}.jsonl", "sha256": sha256(output / f"{split}.jsonl")}
            for split in split_rows
        },
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
