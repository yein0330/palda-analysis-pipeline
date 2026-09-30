#!/usr/bin/env python
"""Sample full-crawl Review/Q&A texts for Cat2 theme discovery."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

from qna_review_model.crawler import iter_pages, normalize_page


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--coverage-csv", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--records-per-cat2", type=int, default=30)
    parser.add_argument("--qna-per-cat2", type=int, default=10)
    return parser.parse_args()


def cat2_path(full_path: str) -> str | None:
    parts = [part.strip() for part in full_path.split(">") if part.strip()]
    return " > ".join(parts[:2]) if len(parts) >= 2 else None


def aliases(row: dict) -> set[str]:
    raw = row.get("product") or {}
    values = {str(row["key"])}
    for field in ("productId", "nvMid", "catalogNvMid", "channelProductNo"):
        if raw.get(field) is not None:
            values.add(str(raw[field])); values.add(f"catalog:{raw[field]}")
    return values


def main() -> None:
    args = arguments()
    aggregate = defaultdict(lambda: {"review": 0, "qna": 0})
    with args.coverage_csv.open(encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            path = cat2_path(row["category_path"])
            if path:
                aggregate[path]["review"] += int(row["collected_review_count"])
                aggregate[path]["qna"] += int(row["collected_qna_count"])
    aggregate = {path: values for path, values in aggregate.items() if sum(values.values()) > 0}

    targets = {}
    for path, values in aggregate.items():
        qna = min(args.qna_per_cat2, values["qna"], args.records_per_cat2)
        review = min(args.records_per_cat2 - qna, values["review"])
        if review + qna < args.records_per_cat2:
            qna = min(args.records_per_cat2 - review, values["qna"])
        targets[path] = {"review": review, "qna": qna}

    product_cat2: dict[str, list[str]] = defaultdict(list)
    with (args.run_dir / "inventory.jsonl").open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line); raw = row.get("product") or {}
            sources = row.get("sources") or ([row["source"]] if row.get("source") else [])
            paths = []
            for source in sources:
                path = cat2_path(str(source.get("fullPath") or ""))
                if path in targets and path not in paths:
                    paths.append(path)
            for alias in aliases(row):
                for path in paths:
                    if path not in product_cat2[alias]:
                        product_cat2[alias].append(path)

    samples = {path: {"review": [], "qna": []} for path in targets}
    seen = {path: set() for path in targets}
    pages = 0
    for page in iter_pages(args.run_dir):
        pages += 1
        kind = page.get("kind")
        if kind not in {"review", "qna"}:
            continue
        paths = product_cat2.get(str(page.get("productKey") or ""), [])
        paths = [path for path in paths if len(samples[path][kind]) < targets[path][kind]]
        if not paths:
            continue
        records = list(normalize_page(page))
        for path in paths:
            for record in records:
                record_id = record.record_id
                if record_id in seen[path]:
                    continue
                text = record.text if record.record_type == "review" else record.question
                samples[path][kind].append({
                    "record_id": record_id, "record_type": record.record_type,
                    "text": text, "score": record.score if record.record_type == "review" else None,
                })
                seen[path].add(record_id)
                if len(samples[path][kind]) >= targets[path][kind]:
                    break
        if pages % 100_000 == 0:
            complete = sum(all(len(samples[p][k]) >= targets[p][k] for k in ("review", "qna")) for p in targets)
            print(json.dumps({"pages": pages, "complete_cat2": complete, "target_cat2": len(targets)}), flush=True)
        if all(all(len(samples[p][k]) >= targets[p][k] for k in ("review", "qna")) for p in targets):
            break

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        for path in sorted(samples):
            values = samples[path]
            handle.write(json.dumps({
                "schema_version": "palda_cat2_theme_sample_v1",
                "cat1": path.split(" > ")[0], "cat2_path": path,
                "source": "full_raw_crawl", "records": values["review"] + values["qna"],
                "counts": {kind: len(values[kind]) for kind in ("review", "qna")},
            }, ensure_ascii=False) + "\n")
    report = {
        "schema_version": "palda_cat2_theme_sample_manifest_v1",
        "source": str(args.run_dir), "cat2_count": len(samples), "pages_scanned": pages,
        "records": sum(len(v["review"]) + len(v["qna"]) for v in samples.values()),
        "complete_cat2": sum(len(v["review"]) + len(v["qna"]) == args.records_per_cat2 for v in samples.values()),
        "output": str(args.output),
    }
    args.output.with_suffix(".manifest.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
