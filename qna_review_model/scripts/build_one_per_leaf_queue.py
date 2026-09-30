#!/usr/bin/env python
"""Build one non-overlapping annotation record for every collected leaf category."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

from qna_review_model.annotation import AllowedTheme, AnnotationQueueRow
from qna_review_model.crawler import iter_pages, normalize_page
from qna_review_model.schemas import ProductContext
from qna_review_model.themes import ThemeDictionary


@dataclass(frozen=True)
class ProductLeaf:
    product_id: str
    product_name: str
    category_cid: str
    category_path: str
    brand: str | None
    price: int | None


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--coverage-csv", type=Path, required=True)
    parser.add_argument("--themes", type=Path, required=True)
    parser.add_argument("--exclude-queue", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260828)
    parser.add_argument("--qna-ratio", type=float, default=0.35)
    return parser.parse_args()


def stable_fraction(seed: int, value: str) -> float:
    digest = hashlib.sha256(f"{seed}:{value}".encode()).digest()
    return int.from_bytes(digest[:8], "big") / 2**64


def split_for(seed: int, product_id: str) -> str:
    value = stable_fraction(seed, product_id)
    return "train" if value < 0.8 else "validation" if value < 0.9 else "test"


def product_aliases(row: dict) -> set[str]:
    raw = row.get("product") or {}
    aliases = {str(row["key"])}
    for field in ("productId", "nvMid", "catalogNvMid", "channelProductNo"):
        if raw.get(field) is not None:
            aliases.add(str(raw[field]))
            aliases.add(f"catalog:{raw[field]}")
    return aliases


def main() -> None:
    args = arguments()
    if not 0 <= args.qna_ratio <= 1:
        raise ValueError("qna-ratio must be between 0 and 1")

    with args.coverage_csv.open(encoding="utf-8-sig", newline="") as handle:
        coverage = {
            row["category_cid"]: row for row in csv.DictReader(handle)
            if int(row["collected_record_count"]) > 0
        }
    desired_kind = {}
    for cid, row in coverage.items():
        reviews = int(row["collected_review_count"])
        qna = int(row["collected_qna_count"])
        desired_kind[cid] = (
            "qna" if qna and (not reviews or stable_fraction(args.seed, cid) < args.qna_ratio) else "review"
        )

    excluded_ids: set[str] = set()
    known_product_splits: dict[str, str] = {}
    with args.exclude_queue.open(encoding="utf-8") as handle:
        for line in handle:
            row = AnnotationQueueRow.model_validate_json(line)
            excluded_ids.add(row.annotation_id)
            previous = known_product_splits.setdefault(row.product_id, row.split)
            if previous != row.split:
                raise ValueError(f"existing queue has product split leakage: {row.product_id}")

    aliases: dict[str, list[ProductLeaf]] = {}
    with (args.run_dir / "inventory.jsonl").open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            raw = row.get("product") or {}
            metadata = raw.get("rawMetadata") or {}
            leaves = []
            sources = row.get("sources") or ([row["source"]] if row.get("source") else [])
            for source in sources:
                cid = str(source.get("cid") or raw.get("categoryCid") or "").strip()
                path = str(source.get("fullPath") or "").strip()
                if cid not in coverage or not path:
                    continue
                leaves.append(ProductLeaf(
                    product_id=str(row["key"]),
                    product_name=str(raw.get("name") or row["key"]),
                    category_cid=cid,
                    category_path=path,
                    brand=str(metadata.get("brand") or "").strip() or None,
                    price=raw.get("price") if isinstance(raw.get("price"), int) else None,
                ))
            if not leaves:
                continue
            for alias in product_aliases(row):
                current = aliases.setdefault(alias, [])
                existing = {leaf.category_cid for leaf in current}
                current.extend(leaf for leaf in leaves if leaf.category_cid not in existing)

    dictionary = ThemeDictionary.from_path(args.themes)
    preferred: dict[str, AnnotationQueueRow] = {}
    fallback: dict[str, AnnotationQueueRow] = {}
    used_ids: set[str] = set(excluded_ids)
    pages = 0

    def make_queue(source_record, leaf: ProductLeaf) -> AnnotationQueueRow:
        levels = [part.strip() for part in leaf.category_path.split(">") if part.strip()]
        context = ProductContext(
            product_id=leaf.product_id, name=leaf.product_name,
            category_cid=leaf.category_cid, category_path=leaf.category_path,
            category_levels=levels, brand=leaf.brand, price=leaf.price,
        )
        split = known_product_splits.get(leaf.product_id) or split_for(args.seed, leaf.product_id)
        text = source_record.text if source_record.record_type == "review" else source_record.question
        return AnnotationQueueRow(
            annotation_id=source_record.record_id, split=split,
            product_id=leaf.product_id, product_name=leaf.product_name,
            category_path=leaf.category_path, record_type=source_record.record_type,
            text=text, score=source_record.score if source_record.record_type == "review" else None,
            allowed_themes=[
                AllowedTheme(theme_id=theme.theme_id, label=theme.label)
                for theme in dictionary.resolve(context)
            ],
        )

    for page in iter_pages(args.run_dir):
        pages += 1
        product_key = str(page.get("productKey") or "")
        leaves = aliases.get(product_key, [])
        if not leaves:
            continue
        records = [record for record in normalize_page(page) if record.record_id not in used_ids]
        if not records:
            continue
        for leaf in leaves:
            cid = leaf.category_cid
            is_preferred = page.get("kind") == desired_kind[cid]
            target = preferred if is_preferred else fallback
            if cid in preferred or cid in target:
                continue
            record = next((value for value in records if value.record_id not in used_ids), None)
            if record is None:
                break
            target[cid] = make_queue(record, leaf)
            used_ids.add(record.record_id)
        if pages % 100_000 == 0:
            print(json.dumps({
                "pages": pages, "preferred": len(preferred), "fallback": len(fallback),
                "target": len(coverage),
            }), flush=True)
        if len(preferred) == len(coverage):
            break

    selected = {cid: preferred.get(cid) or fallback.get(cid) for cid in coverage}
    selected = {cid: row for cid, row in selected.items() if row is not None}
    rows = sorted(selected.values(), key=lambda row: row.category_path)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    queue_path = args.output_dir / "queue.jsonl"
    with queue_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(row.model_dump_json() + "\n")

    category_ids = set(selected)
    missing = [coverage[cid] for cid in coverage.keys() - category_ids]
    missing_path = args.output_dir / "categories-without-selected-record.csv"
    with missing_path.open("w", encoding="utf-8-sig", newline="") as handle:
        fields = ["category_cid", "category_path", "coverage_status", "collected_record_count"]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows({field: row[field] for field in fields} for row in missing)

    counts = {"train": 0, "validation": 0, "test": 0, "review": 0, "qna": 0}
    for row in rows:
        counts[row.split] += 1
        counts[row.record_type] += 1
    manifest = {
        "schema_version": "palda_one_per_leaf_queue_v1",
        "source_run": str(args.run_dir),
        "excluded_queue": str(args.exclude_queue),
        "excluded_record_count": len(excluded_ids),
        "taxonomy_category_count": 5090,
        "categories_with_collected_records": len(coverage),
        "selected_category_count": len(rows),
        "unselected_despite_coverage": len(missing),
        "counts": counts,
        "qna_target_ratio": args.qna_ratio,
        "seed": args.seed,
        "pages_scanned": pages,
        "product_split_reused_when_present": True,
        "outputs": {"queue": str(queue_path), "missing": str(missing_path)},
        "annotation_status": "unlabeled_queue",
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
