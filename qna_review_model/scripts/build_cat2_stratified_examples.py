#!/usr/bin/env python
"""Collect Cat2 examples stratified by review score and Q&A from the full raw crawl."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

from qna_review_model.crawler import iter_pages, normalize_page


TARGETS = {"positive": 5, "neutral": 5, "negative": 5, "qna": 10}


def cat2_path(path: str) -> str | None:
    parts = [part.strip() for part in path.split(">") if part.strip()]
    return " > ".join(parts[:2]) if len(parts) >= 2 else None


def bucket(record) -> str:
    if record.record_type == "qna": return "qna"
    if record.score is None: return "unknown"
    if record.score <= 2: return "negative"
    if record.score < 4: return "neutral"
    return "positive"


def aliases(row: dict) -> set[str]:
    raw = row.get("product") or {}; values = {str(row["key"])}
    for field in ("productId", "nvMid", "catalogNvMid", "channelProductNo"):
        if raw.get(field) is not None:
            values.add(str(raw[field])); values.add(f"catalog:{raw[field]}")
    return values


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--coverage-csv", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    target_paths = set()
    with args.coverage_csv.open(encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            if int(row["collected_record_count"]) > 0:
                path = cat2_path(row["category_path"])
                if path: target_paths.add(path)

    product_cat2 = defaultdict(list)
    with (args.run_dir / "inventory.jsonl").open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line); raw = row.get("product") or {}
            sources = row.get("sources") or ([row["source"]] if row.get("source") else [])
            entries = []
            for source in sources:
                full_path = str(source.get("fullPath") or "").strip(); path = cat2_path(full_path)
                if path in target_paths and not any(value["cat2_path"] == path for value in entries):
                    entries.append({
                        "cat2_path": path, "category_path": full_path,
                        "product_id": str(row["key"]), "product_name": str(raw.get("name") or row["key"]),
                    })
            for alias in aliases(row):
                known = {value["cat2_path"] for value in product_cat2[alias]}
                product_cat2[alias].extend(value for value in entries if value["cat2_path"] not in known)

    selected = {path: defaultdict(list) for path in target_paths}
    per_product = Counter(); used_ids = {path: set() for path in target_paths}
    pages = 0
    for page in iter_pages(args.run_dir):
        pages += 1
        entries = product_cat2.get(str(page.get("productKey") or ""), [])
        if not entries: continue
        records = list(normalize_page(page))
        for entry in entries:
            path = entry["cat2_path"]
            for record in records:
                name = bucket(record)
                if name not in TARGETS or len(selected[path][name]) >= TARGETS[name]: continue
                if record.record_id in used_ids[path]: continue
                key = (path, name, entry["product_id"])
                if per_product[key] >= 2: continue
                text = record.text if record.record_type == "review" else record.question
                selected[path][name].append({
                    "record_id": record.record_id, "record_type": record.record_type,
                    "text": text, "score": record.score if record.record_type == "review" else None,
                    "sentiment_bucket": name, "category_path": entry["category_path"],
                    "product_id": entry["product_id"], "product_name": entry["product_name"],
                    "example_source": "full_raw_stratified",
                })
                used_ids[path].add(record.record_id); per_product[key] += 1
        if pages % 100_000 == 0:
            complete = sum(all(len(selected[p][b]) >= n for b, n in TARGETS.items()) for p in target_paths)
            print(json.dumps({"pages": pages, "complete_cat2": complete, "target_cat2": len(target_paths)}), flush=True)
        if all(all(len(selected[p][b]) >= n for b, n in TARGETS.items()) for p in target_paths): break

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        for path in sorted(target_paths):
            records = [record for name in TARGETS for record in selected[path][name]]
            handle.write(json.dumps({
                "schema_version": "palda_cat2_stratified_examples_v1",
                "cat2_path": path, "records": records,
                "counts": {name: len(selected[path][name]) for name in TARGETS},
            }, ensure_ascii=False) + "\n")
    totals = Counter()
    for values in selected.values():
        for name in TARGETS: totals[name] += len(values[name])
    report = {
        "schema_version": "palda_cat2_stratified_examples_manifest_v1",
        "cat2": len(target_paths), "pages_scanned": pages, "counts": dict(totals),
        "complete_cat2": sum(all(len(selected[p][b]) >= n for b, n in TARGETS.items()) for p in target_paths),
        "output": str(args.output),
    }
    args.output.with_suffix(".manifest.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
