"""Build a bounded, reproducible product-level annotation pilot."""

from __future__ import annotations

import hashlib
import heapq
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Literal

from pydantic import Field

from .crawler import iter_pages, load_products, normalize_page, validate_completed_run
from .schemas import ProductAnalysis, ProductContext, QNARecord, ReviewRecord, StrictModel, ThemeDefinition
from .themes import ThemeDictionary


Split = Literal["train", "validation", "test"]


class AnnotationSample(StrictModel):
    """One service-shaped unit for annotation and later supervised training."""

    schema_version: Literal["palda_annotation_v1"] = "palda_annotation_v1"
    sample_id: str
    split: Split
    annotation_status: Literal["unlabeled", "silver", "gold"] = "unlabeled"
    product: ProductContext
    category_themes: list[ThemeDefinition]
    reviews: list[ReviewRecord] = Field(min_length=1)
    qna: list[QNARecord] = Field(min_length=1)
    target: ProductAnalysis | None = None


def _hash_int(seed: int, *values: Any) -> int:
    raw = "\x1f".join([str(seed), *(str(value) for value in values)])
    return int.from_bytes(hashlib.sha256(raw.encode()).digest()[:8], "big")


def _product_type(raw: dict[str, Any]) -> str:
    return "catalog" if raw.get("productGroup") == "catalog" else "single"


def _summary_stats(run_dir: Path) -> dict[tuple[str, str, str], dict[str, Any]]:
    summary = json.loads((run_dir / "content-summary.json").read_text(encoding="utf-8"))
    result: dict[tuple[str, str, str], dict[str, Any]] = {}
    for category in summary.get("categories", []):
        leaf = category.get("categoryName")
        for product in category.get("products", []):
            key = (str(leaf), str(product.get("productName")), str(product.get("productType")))
            result[key] = product
    return result


def select_candidates(
    run_dir: Path,
    category: str,
    size: int,
    min_reviews: int,
    min_qna: int,
    seed: int,
) -> list[ProductContext]:
    """Select eligible products, rotating over leaves to retain category breadth."""
    stats = _summary_stats(run_dir)
    aliases = load_products(run_dir)
    contexts = {context.product_id: context for context in aliases.values()}
    eligible: dict[str, list[ProductContext]] = defaultdict(list)

    # Read inventory again because summary joins on leaf, display name and product type.
    with (run_dir / "inventory.jsonl").open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            raw = row.get("product") or {}
            sources = row.get("sources") or ([row["source"]] if row.get("source") else [])
            source = sources[0] if sources else {}
            levels = [part.strip() for part in str(source.get("fullPath") or "").split(">") if part.strip()]
            if not levels or levels[0] != category:
                continue
            item = stats.get((levels[-1], str(raw.get("name")), _product_type(raw)))
            if not item:
                continue
            if int(item.get("collectedReviewCount") or 0) < min_reviews:
                continue
            if int(item.get("collectedQnaCount") or 0) < min_qna:
                continue
            context = contexts.get(str(row["key"]))
            if context is not None:
                eligible[levels[-1]].append(context)

    for leaf, values in eligible.items():
        values.sort(key=lambda context: _hash_int(seed, leaf, context.product_id))
    leaf_order = sorted(eligible, key=lambda leaf: _hash_int(seed, leaf))
    selected: list[ProductContext] = []
    offset = 0
    while len(selected) < size:
        added = False
        for leaf in leaf_order:
            if offset < len(eligible[leaf]):
                selected.append(eligible[leaf][offset])
                added = True
                if len(selected) == size:
                    break
        if not added:
            break
        offset += 1
    if len(selected) < size:
        raise ValueError(
            f"only {len(selected)} eligible products for {category!r}; requested {size} "
            f"with min_reviews={min_reviews}, min_qna={min_qna}"
        )
    return selected


def select_multicategory_candidates(
    run_dir: Path, categories: list[str], size_per_category: int,
    min_reviews: int, min_qna: int, seed: int,
) -> dict[str, list[ProductContext]]:
    """Select an equal number of distinct products per Cat1 in one inventory pass."""
    stats = _summary_stats(run_dir)
    aliases = load_products(run_dir)
    contexts = {context.product_id: context for context in aliases.values()}
    eligible: dict[str, dict[str, dict[str, ProductContext]]] = {
        category: defaultdict(dict) for category in categories
    }
    with (run_dir / "inventory.jsonl").open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line); raw = row.get("product") or {}
            sources = row.get("sources") or ([row["source"]] if row.get("source") else [])
            source = sources[0] if sources else {}
            levels = [part.strip() for part in str(source.get("fullPath") or "").split(">") if part.strip()]
            if not levels or levels[0] not in eligible:
                continue
            item = stats.get((levels[-1], str(raw.get("name")), _product_type(raw)))
            if not item or int(item.get("collectedReviewCount") or 0) < min_reviews:
                continue
            if int(item.get("collectedQnaCount") or 0) < min_qna:
                continue
            context = contexts.get(str(row["key"]))
            if context is not None:
                eligible[levels[0]][levels[-1]][context.product_id] = context
    selected: dict[str, list[ProductContext]] = {}
    globally_used: set[str] = set()
    for category in categories:
        by_leaf = eligible[category]
        leaf_order = sorted(by_leaf, key=lambda leaf: _hash_int(seed, category, leaf))
        values = {
            leaf: sorted(rows.values(), key=lambda context: _hash_int(seed, category, leaf, context.product_id))
            for leaf, rows in by_leaf.items()
        }
        chosen: list[ProductContext] = []
        offset = 0
        while len(chosen) < size_per_category:
            added = False
            for leaf in leaf_order:
                while offset < len(values[leaf]) and values[leaf][offset].product_id in globally_used:
                    values[leaf].pop(offset)
                if offset < len(values[leaf]):
                    context = values[leaf][offset]
                    chosen.append(context); globally_used.add(context.product_id); added = True
                    if len(chosen) == size_per_category:
                        break
            if not added:
                break
            offset += 1
        if len(chosen) < size_per_category:
            raise ValueError(f"only {len(chosen)} distinct eligible products for {category!r}")
        selected[category] = chosen
    return selected


class _Reservoir:
    """Keep the records with the smallest stable hash using bounded memory."""

    def __init__(self, capacity: int, seed: int) -> None:
        self.capacity = capacity
        self.seed = seed
        self._heap: list[tuple[int, str, str, ReviewRecord | QNARecord]] = []

    def add(self, record: ReviewRecord | QNARecord) -> None:
        priority = _hash_int(self.seed, record.record_id)
        item = (-priority, record.record_id, record.model_dump_json(), record)
        if len(self._heap) < self.capacity:
            heapq.heappush(self._heap, item)
        elif item > self._heap[0]:
            heapq.heapreplace(self._heap, item)

    def records(self) -> list[ReviewRecord | QNARecord]:
        return [item[3] for item in sorted(self._heap, key=lambda item: (-item[0], item[1], item[2]))]


def _review_bucket(record: ReviewRecord) -> str:
    if record.score is None:
        return "unknown"
    if record.score <= 2:
        return "negative"
    if record.score < 4:
        return "neutral"
    return "positive"


def _temporal_spread(records: list[ReviewRecord | QNARecord], count: int) -> list[ReviewRecord | QNARecord]:
    def key(record: ReviewRecord | QNARecord) -> tuple[str, str]:
        return (record.created_at.isoformat() if record.created_at else "", record.record_id)

    if len(records) <= count:
        return sorted(records, key=key, reverse=True)
    ordered = sorted(records, key=key)
    if count == 1:
        return [ordered[-1]]
    indexes = {round(index * (len(ordered) - 1) / (count - 1)) for index in range(count)}
    return [ordered[index] for index in sorted(indexes, reverse=True)]


def _select_reviews(buckets: dict[str, _Reservoir], limit: int) -> list[ReviewRecord]:
    weights = {"positive": 0.50, "negative": 0.25, "neutral": 0.15, "unknown": 0.10}
    chosen: list[ReviewRecord] = []
    leftovers: list[ReviewRecord] = []
    for bucket, weight in weights.items():
        records = [record for record in buckets[bucket].records() if isinstance(record, ReviewRecord)]
        take = min(len(records), round(limit * weight))
        selected = _temporal_spread(records, take)
        selected_ids = {record.record_id for record in selected}
        chosen.extend(selected)
        leftovers.extend(record for record in records if record.record_id not in selected_ids)
    if len(chosen) < limit:
        chosen.extend(_temporal_spread(leftovers, limit - len(chosen)))
    return sorted(
        chosen[:limit],
        key=lambda record: (record.created_at.isoformat() if record.created_at else "", record.record_id),
        reverse=True,
    )


def _assign_splits(product_ids: list[str], seed: int) -> dict[str, Split]:
    ordered = sorted(product_ids, key=lambda product_id: _hash_int(seed, "split", product_id))
    total = len(ordered)
    train_end = round(total * 0.8)
    validation_end = train_end + round(total * 0.1)
    return {
        product_id: "train" if index < train_end else "validation" if index < validation_end else "test"
        for index, product_id in enumerate(ordered)
    }


def build_pilot(
    run_dir: Path,
    themes_path: Path,
    output_dir: Path,
    category: str = "가구/인테리어",
    size: int = 200,
    max_reviews: int = 50,
    max_qna: int = 30,
    min_reviews: int = 10,
    min_qna: int = 3,
    seed: int = 20260826,
) -> dict[str, Any]:
    validate_completed_run(run_dir)
    candidates = select_candidates(run_dir, category, size, min_reviews, min_qna, seed)
    candidate_ids = {context.product_id for context in candidates}
    aliases = load_products(run_dir)
    alias_to_id = {alias: context.product_id for alias, context in aliases.items() if context.product_id in candidate_ids}
    review_pools = {
        product_id: {bucket: _Reservoir(max_reviews, seed) for bucket in ("positive", "negative", "neutral", "unknown")}
        for product_id in candidate_ids
    }
    qna_pools = {product_id: _Reservoir(max_qna, seed) for product_id in candidate_ids}

    for page in iter_pages(run_dir):
        product_id = alias_to_id.get(str(page.get("productKey") or ""))
        if product_id is None:
            continue
        for record in normalize_page(page):
            if isinstance(record, ReviewRecord):
                review_pools[product_id][_review_bucket(record)].add(record)
            else:
                qna_pools[product_id].add(record)

    dictionary = ThemeDictionary.from_path(themes_path)
    splits = _assign_splits(sorted(candidate_ids), seed)
    samples: list[AnnotationSample] = []
    missing: list[str] = []
    for context in candidates:
        reviews = _select_reviews(review_pools[context.product_id], max_reviews)
        qna = [record for record in qna_pools[context.product_id].records() if isinstance(record, QNARecord)]
        qna = _temporal_spread(qna, max_qna)
        if len(reviews) < min_reviews or len(qna) < min_qna:
            missing.append(context.product_id)
            continue
        samples.append(AnnotationSample(
            sample_id=f"product_{context.product_id}", split=splits[context.product_id], product=context,
            category_themes=dictionary.resolve(context), reviews=reviews, qna=qna,
        ))
    if missing:
        raise ValueError(f"summary/content mismatch for {len(missing)} selected products: {missing[:5]}")

    output_dir.mkdir(parents=True, exist_ok=True)
    counts: dict[str, int] = {}
    for split in ("train", "validation", "test"):
        path = output_dir / f"{split}.jsonl"
        rows = [sample for sample in samples if sample.split == split]
        with path.open("w", encoding="utf-8") as handle:
            for sample in rows:
                handle.write(sample.model_dump_json() + "\n")
        counts[split] = len(rows)
    manifest = {
        "schema_version": "palda_pilot_manifest_v1", "category": category, "seed": seed,
        "selection": {"products": size, "min_reviews": min_reviews, "min_qna": min_qna},
        "caps": {"reviews_per_product": max_reviews, "qna_per_product": max_qna},
        "splits": counts,
        "records": {"reviews": sum(len(sample.reviews) for sample in samples), "qna": sum(len(sample.qna) for sample in samples)},
        "annotation_status": "unlabeled",
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (output_dir / "annotation_sample.schema.json").write_text(
        json.dumps(AnnotationSample.model_json_schema(), ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return {"output_dir": str(output_dir), **manifest}


def build_multicategory_pilot(
    run_dir: Path, themes_path: Path, output_dir: Path, categories: list[str],
    size_per_category: int = 100, max_reviews: int = 50, max_qna: int = 30,
    min_reviews: int = 10, min_qna: int = 3, seed: int = 20260827,
) -> dict[str, Any]:
    validate_completed_run(run_dir)
    selected = select_multicategory_candidates(
        run_dir, categories, size_per_category, min_reviews, min_qna, seed,
    )
    category_by_product = {
        context.product_id: category for category, contexts in selected.items() for context in contexts
    }
    candidates = [context for contexts in selected.values() for context in contexts]
    candidate_ids = set(category_by_product)
    aliases = load_products(run_dir)
    alias_to_id = {
        alias: context.product_id for alias, context in aliases.items() if context.product_id in candidate_ids
    }
    review_pools = {
        product_id: {bucket: _Reservoir(max_reviews, seed) for bucket in ("positive", "negative", "neutral", "unknown")}
        for product_id in candidate_ids
    }
    qna_pools = {product_id: _Reservoir(max_qna, seed) for product_id in candidate_ids}
    for page in iter_pages(run_dir):
        product_id = alias_to_id.get(str(page.get("productKey") or ""))
        if product_id is None:
            continue
        for record in normalize_page(page):
            if isinstance(record, ReviewRecord):
                review_pools[product_id][_review_bucket(record)].add(record)
            else:
                qna_pools[product_id].add(record)
    dictionary = ThemeDictionary.from_path(themes_path)
    splits: dict[str, Split] = {}
    for category, contexts in selected.items():
        splits.update(_assign_splits([context.product_id for context in contexts], seed))
    samples: list[AnnotationSample] = []
    missing: list[str] = []
    for context in candidates:
        reviews = _select_reviews(review_pools[context.product_id], max_reviews)
        qna = [record for record in qna_pools[context.product_id].records() if isinstance(record, QNARecord)]
        qna = _temporal_spread(qna, max_qna)
        if len(reviews) < min_reviews or len(qna) < min_qna:
            missing.append(context.product_id); continue
        samples.append(AnnotationSample(
            sample_id=f"product_{context.product_id}", split=splits[context.product_id], product=context,
            category_themes=dictionary.resolve(context), reviews=reviews, qna=qna,
        ))
    if missing:
        raise ValueError(f"summary/content mismatch for {len(missing)} selected products: {missing[:5]}")
    output_dir.mkdir(parents=True, exist_ok=True)
    counts: dict[str, int] = {}
    category_counts: dict[str, dict[str, int]] = {}
    for split in ("train", "validation", "test"):
        rows = [sample for sample in samples if sample.split == split]
        with (output_dir / f"{split}.jsonl").open("w", encoding="utf-8") as handle:
            for sample in rows:
                handle.write(sample.model_dump_json() + "\n")
        counts[split] = len(rows)
    for category in categories:
        category_rows = [sample for sample in samples if category_by_product[sample.product.product_id] == category]
        category_counts[category] = {
            split: sum(sample.split == split for sample in category_rows)
            for split in ("train", "validation", "test")
        }
    manifest = {
        "schema_version": "palda_multicategory_pilot_v1", "categories": categories,
        "seed": seed, "selection": {"products_per_category": size_per_category, "min_reviews": min_reviews, "min_qna": min_qna},
        "caps": {"reviews_per_product": max_reviews, "qna_per_product": max_qna},
        "splits": counts, "category_splits": category_counts,
        "records": {"reviews": sum(len(sample.reviews) for sample in samples), "qna": sum(len(sample.qna) for sample in samples)},
        "annotation_status": "unlabeled", "product_split_leakage": False,
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (output_dir / "annotation_sample.schema.json").write_text(
        json.dumps(AnnotationSample.model_json_schema(), ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return {"output_dir": str(output_dir), **manifest}
