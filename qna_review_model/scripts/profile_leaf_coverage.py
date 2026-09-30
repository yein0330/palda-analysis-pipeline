#!/usr/bin/env python
"""Profile record coverage for every leaf category attempted by the crawler."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--taxonomy", type=Path)
    parser.add_argument("--target-category-count", type=int, default=5090)
    parser.add_argument("--records-per-category", type=int, default=20)
    return parser.parse_args()


def summary_product_stats(run_dir: Path) -> dict[tuple[str, str, str], dict[str, Any]]:
    payload = json.loads((run_dir / "content-summary.json").read_text(encoding="utf-8"))
    result: dict[tuple[str, str, str], dict[str, Any]] = {}
    for category in payload.get("categories", []):
        leaf = str(category.get("categoryName") or "")
        for product in category.get("products", []):
            key = (leaf, str(product.get("productName") or ""), str(product.get("productType") or ""))
            result[key] = product
    return result


def main() -> None:
    args = arguments()
    product_stats = summary_product_stats(args.run_dir)
    coverage: dict[str, dict[str, Any]] = defaultdict(
        lambda: {"product_ids": set(), "reviews": 0, "qna": 0, "review_products": 0, "qna_products": 0}
    )

    with (args.run_dir / "inventory.jsonl").open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            raw = row.get("product") or {}
            sources = row.get("sources") or ([row["source"]] if row.get("source") else [])
            product_type = "catalog" if raw.get("productGroup") == "catalog" else "single"
            product_name = str(raw.get("name") or "")
            for source in sources:
                path = str(source.get("fullPath") or "").strip()
                cid = str(source.get("cid") or raw.get("categoryCid") or "").strip()
                if not path or not cid:
                    continue
                leaf = path.split(">")[-1].strip()
                stats = product_stats.get((leaf, product_name, product_type), {})
                bucket = coverage[cid]
                product_id = str(row.get("key") or "")
                if product_id in bucket["product_ids"]:
                    continue
                bucket["product_ids"].add(product_id)
                reviews = int(stats.get("collectedReviewCount") or 0)
                qna = int(stats.get("collectedQnaCount") or 0)
                bucket["reviews"] += reviews
                bucket["qna"] += qna
                bucket["review_products"] += reviews > 0
                bucket["qna_products"] += qna > 0

    ranking_rows: dict[str, dict[str, Any]] = {}
    with (args.run_dir / "rankings.jsonl").open(encoding="utf-8") as handle:
        for line in handle:
            ranking = json.loads(line)
            category = ranking.get("category") or {}
            cid = str(category.get("cid") or "").strip()
            if cid:
                ranking_rows[cid] = ranking

    if args.taxonomy:
        with args.taxonomy.open(encoding="utf-8-sig", newline="") as handle:
            taxonomy_rows = list(csv.DictReader(handle))
    else:
        taxonomy_rows = [
            {
                "cid": cid,
                "fullPath": str((ranking.get("category") or {}).get("fullPath") or ""),
                "nodeName": str((ranking.get("category") or {}).get("nodeName") or ""),
            }
            for cid, ranking in ranking_rows.items()
        ]

    rows: list[dict[str, Any]] = []
    for taxonomy in sorted(taxonomy_rows, key=lambda row: row["fullPath"]):
        cid = str(taxonomy["cid"])
        path = str(taxonomy["fullPath"])
        ranking = ranking_rows.get(cid)
        values = coverage[cid]
        reviews, qna = int(values["reviews"]), int(values["qna"])
        available = reviews + qna
        rows.append({
            "category_cid": cid,
            "category_path": path,
            "cat1": path.split(">")[0].strip(),
            "leaf_name": path.split(">")[-1].strip(),
            "crawl_attempted": ranking is not None,
            "ranking_status": str((ranking or {}).get("status") or "not_attempted"),
            "ranking_product_count": int((ranking or {}).get("productCount") or 0),
            "inventory_product_count": len(values["product_ids"]),
            "review_product_count": int(values["review_products"]),
            "qna_product_count": int(values["qna_products"]),
            "collected_review_count": reviews,
            "collected_qna_count": qna,
            "collected_record_count": available,
            "sample_capacity": min(available, args.records_per_category),
            "coverage_status": "review_and_qna" if reviews and qna else "review_only" if reviews else "qna_only" if qna else "no_records",
        })

    args.output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.output_dir / "leaf-category-coverage.csv"
    with csv_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    missing_path = args.output_dir / "missing-from-crawl.csv"
    with missing_path.open("w", encoding="utf-8-sig", newline="") as handle:
        fields = ["category_cid", "category_path", "cat1", "leaf_name"]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows({field: row[field] for field in fields} for row in rows if not row["crawl_attempted"])
    no_records_path = args.output_dir / "categories-without-records.csv"
    with no_records_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(row for row in rows if row["collected_record_count"] == 0)

    statuses = Counter(row["coverage_status"] for row in rows)
    report = {
        "schema_version": "palda_leaf_category_coverage_v1",
        "target_category_count": args.target_category_count,
        "taxonomy_category_count": len(rows),
        "crawler_attempted_category_count": sum(row["crawl_attempted"] for row in rows),
        "missing_from_crawl_count": sum(not row["crawl_attempted"] for row in rows),
        "inventory_category_count": sum(row["inventory_product_count"] > 0 for row in rows),
        "category_with_any_records_count": sum(row["collected_record_count"] > 0 for row in rows),
        "coverage_status": dict(statuses),
        "records_per_category_target": args.records_per_category,
        "adaptive_dataset_record_count": sum(row["sample_capacity"] for row in rows),
        "full_quota_category_count": sum(row["sample_capacity"] == args.records_per_category for row in rows),
        "cat1_category_counts": dict(sorted(Counter(row["cat1"] for row in rows).items())),
        "missing_from_crawl_by_cat1": dict(sorted(Counter(
            row["cat1"] for row in rows if not row["crawl_attempted"]
        ).items())),
        "outputs": {
            "csv": str(csv_path),
            "missing_from_crawl_csv": str(missing_path),
            "categories_without_records_csv": str(no_records_path),
        },
    }
    json_path = args.output_dir / "leaf-category-coverage.json"
    report["outputs"]["json"] = str(json_path)
    json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
