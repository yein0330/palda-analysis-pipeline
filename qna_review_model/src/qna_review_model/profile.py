"""Phase-1 crawler data profiling without materializing raw responses."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any

from .crawler import iter_pages, load_products, validate_completed_run


def profile_run(run_dir: Path, max_pages: int | None = None) -> dict[str, Any]:
    progress = validate_completed_run(run_dir)
    summary = json.loads((run_dir / "content-summary.json").read_text(encoding="utf-8"))
    manifest = json.loads((run_dir / "content-manifest.json").read_text(encoding="utf-8"))
    products = load_products(run_dir)
    page_counts: Counter[str] = Counter()
    status_counts: Counter[str] = Counter()
    source_records: Counter[str] = Counter()
    product_ids: dict[str, set[str]] = {"review": set(), "qna": set()}
    for page in iter_pages(run_dir, "all", max_pages):
        kind = str(page.get("kind"))
        page_counts[kind] += 1
        status_counts[str(page.get("status"))] += 1
        source_records[kind] += int(page.get("sourceRecordCount") or 0)
        if kind in product_ids:
            product_ids[kind].add(str(page.get("productKey")))
    canonical = {context.product_id: context for context in products.values()}
    categories = Counter(context.cat1 for context in canonical.values())
    review_products, qna_products = product_ids["review"], product_ids["qna"]
    collection = summary["collection"]
    full_scan = max_pages is None
    return {
        "run": {
            "status": progress["status"],
            "queue_drained": progress["completion"]["queueDrained"],
            "summary": collection,
        },
        "inventory": {
            "products": len(canonical),
            "cat1_distribution": dict(categories.most_common()),
            "with_brand": sum(bool(p.brand) for p in canonical.values()),
            "with_price": sum(p.price is not None for p in canonical.values()),
            "with_category_path": sum(bool(p.category_path) for p in canonical.values()),
        },
        "pages": {
            "sampled": sum(page_counts.values()),
            "limit": max_pages,
            "by_kind": dict(page_counts),
            "by_status": dict(status_counts),
            "source_record_counts": dict(source_records),
        },
        "cross_stage": {
            "review_products": len(review_products),
            "qna_products": len(qna_products),
            "both": len(review_products & qna_products),
            "review_only": len(review_products - qna_products),
            "qna_only": len(qna_products - review_products),
        },
        "consistency": {
            "full_scan": full_scan,
            "manifest_page_records": sum(int(chunk.get("record_count") or 0) for chunk in manifest["chunks"]),
            "observed_pages_minus_manifest": sum(page_counts.values()) - sum(
                int(chunk.get("record_count") or 0) for chunk in manifest["chunks"]
            ) if full_scan else None,
            "review_source_records_minus_summary": source_records["review"] - collection["reviews"]["collectedCount"] if full_scan else None,
            "qna_source_records_minus_summary": source_records["qna"] - collection["qna"]["collectedCount"] if full_scan else None,
        },
    }


def write_profile(run_dir: Path, output: Path, max_pages: int | None = None) -> dict[str, Any]:
    report = profile_run(run_dir, max_pages)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return report
