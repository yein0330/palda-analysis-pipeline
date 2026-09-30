#!/usr/bin/env python
"""Enrich Cat2 theme review rows with leaf/product/type/polarity-diverse examples."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

from qna_review_model.annotation import AnnotationQueueRow


def cat2_path(path: str) -> str | None:
    parts = [part.strip() for part in path.split(">") if part.strip()]
    return " > ".join(parts[:2]) if len(parts) >= 2 else None


def bucket(record_type: str, score: float | None) -> str:
    if record_type == "qna": return "qna"
    if score is None: return "unknown"
    if score <= 2: return "negative"
    if score < 4: return "neutral"
    return "positive"


def stable(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def diverse_pick(records: list[dict], limit: int) -> list[dict]:
    groups = defaultdict(list)
    for record in records:
        groups[record["sentiment_bucket"]].append(record)
    for values in groups.values():
        values.sort(key=lambda value: stable(value["record_id"]))
    order = ["qna", "negative", "neutral", "positive", "unknown"]
    picked, used = [], set()
    product_counts, leaf_counts = Counter(), Counter()
    while len(picked) < limit:
        added = False
        for name in order:
            values = groups[name]
            while values:
                record = values.pop(0)
                if record["record_id"] in used: continue
                product = record.get("product_id")
                leaf = record.get("category_path")
                if product and product_counts[product] >= 2: continue
                if leaf and leaf_counts[leaf] >= 2: continue
                picked.append(record); used.add(record["record_id"])
                if product: product_counts[product] += 1
                if leaf: leaf_counts[leaf] += 1
                added = True; break
            if len(picked) >= limit: break
        if not added: break
    return picked


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--review", type=Path, required=True)
    parser.add_argument("--leaf-queue", type=Path, required=True)
    parser.add_argument("--stratified-samples", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-examples", type=int, default=60)
    args = parser.parse_args()

    stratified = {}
    if args.stratified_samples:
        stratified = {
            row["cat2_path"]: row["records"]
            for row in map(json.loads, args.stratified_samples.read_text(encoding="utf-8").splitlines())
        }

    leaf_rows = defaultdict(list)
    for line in args.leaf_queue.read_text(encoding="utf-8").splitlines():
        row = AnnotationQueueRow.model_validate_json(line)
        path = cat2_path(row.category_path)
        if not path: continue
        leaf_rows[path].append({
            "record_id": row.annotation_id, "record_type": row.record_type,
            "text": row.text, "score": row.score,
            "sentiment_bucket": bucket(row.record_type, row.score),
            "category_path": row.category_path, "product_id": row.product_id,
            "product_name": row.product_name, "example_source": "leaf_coverage",
        })

    output_rows = []
    for line in args.review.read_text(encoding="utf-8").splitlines():
        row = json.loads(line); path = row["cat2_path"]
        raw_records = [{
            **record, "sentiment_bucket": bucket(record["record_type"], record.get("score")),
            "category_path": None, "product_id": None, "product_name": None,
            "example_source": "full_raw_cat2_sample",
        } for record in row.get("sample_records", [])]
        selected = list(stratified.get(path, []))
        selected_ids = {record["record_id"] for record in selected}
        leaf_limit = min(25, args.max_examples - len(selected))
        selected.extend(diverse_pick(
            [record for record in leaf_rows[path] if record["record_id"] not in selected_ids],
            leaf_limit,
        ))
        selected_ids = {record["record_id"] for record in selected}
        selected.extend(diverse_pick(
            [record for record in raw_records if record["record_id"] not in selected_ids],
            args.max_examples - len(selected),
        ))
        if len(selected) < args.max_examples:
            selected_ids = {record["record_id"] for record in selected}
            selected.extend(diverse_pick(
                [*leaf_rows[path], *raw_records], args.max_examples - len(selected)
            ))
            deduplicated = {}
            for record in selected:
                deduplicated.setdefault(record["record_id"], record)
            selected = list(deduplicated.values())[:args.max_examples]

        for theme in row["accepted_themes"]:
            keywords = [value for value in theme["keywords"] if len(value) >= 2]
            matched = []
            used_leaf = set()
            ranked = sorted(selected, key=lambda record: (
                -sum(keyword in record["text"] for keyword in keywords),
                stable(record["record_id"]),
            ))
            for record in ranked:
                if not any(keyword in record["text"] for keyword in keywords): continue
                leaf = record.get("category_path")
                if leaf and leaf in used_leaf and len(matched) < 3: continue
                matched.append(record)
                if leaf: used_leaf.add(leaf)
                if len(matched) == 6: break
            theme["matched_examples"] = matched

        counts = Counter(record["sentiment_bucket"] for record in selected)
        row["diverse_examples"] = selected
        row["diversity"] = {
            "examples": len(selected),
            "leaf_categories": len({record["category_path"] for record in selected if record.get("category_path")}),
            "products": len({record["product_id"] for record in selected if record.get("product_id")}),
            "record_buckets": dict(counts),
        }
        output_rows.append(row)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        for row in output_rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    report = {
        "schema_version": "palda_cat2_theme_diverse_examples_v1",
        "cat2": len(output_rows),
        "examples": sum(row["diversity"]["examples"] for row in output_rows),
        "leaf_categories": len({
            record["category_path"] for row in output_rows for record in row["diverse_examples"]
            if record.get("category_path")
        }),
        "theme_matched_examples": sum(
            len(theme["matched_examples"]) for row in output_rows for theme in row["accepted_themes"]
        ),
        "output": str(args.output),
    }
    args.output.with_suffix(".report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
