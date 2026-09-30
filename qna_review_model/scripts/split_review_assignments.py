#!/usr/bin/env python
"""Split a draft review set into product-safe, category-balanced assignments."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--drafts", type=Path, required=True)
    parser.add_argument("--service-drafts", type=Path, required=True)
    parser.add_argument("--review-csv", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--reviewers", type=int, default=3)
    parser.add_argument("--seed", type=int, default=20260828)
    return parser.parse_args()


def stable(seed: int, *values: object) -> str:
    return hashlib.sha256(": ".join(map(str, (seed, *values))).encode()).hexdigest()


def assign_products(drafts: list[dict[str, Any]], reviewers: int, seed: int) -> dict[str, int]:
    products: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in drafts:
        products[row["input"]["product_id"]].append(row)
    by_cat1: dict[str, list[str]] = defaultdict(list)
    for product_id, rows in products.items():
        by_cat1[rows[0]["input"]["category_path"].split(">")[0].strip()].append(product_id)
    total_records = [0] * reviewers
    root_products: dict[str, list[int]] = defaultdict(lambda: [0] * reviewers)
    assignment: dict[str, int] = {}
    for cat1 in sorted(by_cat1):
        for product_id in sorted(by_cat1[cat1], key=lambda value: stable(seed, cat1, value)):
            candidates = sorted(
                range(reviewers),
                key=lambda index: (root_products[cat1][index], total_records[index], stable(seed, product_id, index)),
            )
            reviewer = candidates[0]
            assignment[product_id] = reviewer
            root_products[cat1][reviewer] += 1
            total_records[reviewer] += len(products[product_id])
    return assignment


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")


def main() -> None:
    args = arguments()
    drafts = [json.loads(line) for line in args.drafts.read_text(encoding="utf-8").splitlines()]
    services = {json.loads(line)["annotation_id"]: json.loads(line) for line in args.service_drafts.read_text(encoding="utf-8").splitlines()}
    with args.review_csv.open(encoding="utf-8-sig", newline="") as handle:
        csv_rows = {row["annotation_id"]: row for row in csv.DictReader(handle)}
    product_assignment = assign_products(drafts, args.reviewers, args.seed)
    assigned: list[list[dict[str, Any]]] = [[] for _ in range(args.reviewers)]
    for row in drafts:
        assigned[product_assignment[row["input"]["product_id"]]].append(row)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest: dict[str, Any] = {
        "schema_version": "palda_review_assignments_v1",
        "seed": args.seed,
        "reviewer_count": args.reviewers,
        "total_records": len(drafts),
        "assignments": [],
    }
    for index, rows in enumerate(assigned, 1):
        reviewer_id = f"annotator-{index}"
        directory = args.output_dir / reviewer_id
        directory.mkdir(parents=True, exist_ok=True)
        ids = [row["input"]["annotation_id"] for row in rows]
        id_set = set(ids)
        write_jsonl(directory / "drafts.jsonl", rows)
        write_jsonl(directory / "service-drafts.jsonl", [services[value] for value in ids])
        fieldnames = list(next(iter(csv_rows.values())))
        with (directory / "review.csv").open("w", encoding="utf-8-sig", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(csv_rows[value] for value in ids)
        cat1 = Counter(row["input"]["category_path"].split(">")[0].strip() for row in rows)
        leaves = {row["input"]["category_path"] for row in rows}
        record_types = Counter(row["input"]["record_type"] for row in rows)
        splits = Counter(row["input"]["split"] for row in rows)
        assignment_row = {
            "reviewer_id": reviewer_id,
            "records": len(rows),
            "products": len({row["input"]["product_id"] for row in rows}),
            "leaf_categories": len(leaves),
            "cat1": dict(sorted(cat1.items())),
            "record_types": dict(record_types),
            "splits": dict(splits),
            "annotation_ids": ids,
        }
        (directory / "assignment.json").write_text(
            json.dumps(assignment_row, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        manifest["assignments"].append({key: value for key, value in assignment_row.items() if key != "annotation_ids"})
        assert len(id_set) == len(ids)
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
