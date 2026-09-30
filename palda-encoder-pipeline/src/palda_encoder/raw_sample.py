from __future__ import annotations

import hashlib
import heapq
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterator

from .io import sha256_file, write_json, write_jsonl
from .schema import make_category_key


def build_balanced_raw_sample(
    run_dir: str | Path,
    output_path: str | Path,
    *,
    blocked_products_path: str | Path | None = None,
    target_records: int = 15_000,
    products_per_leaf: int = 3,
    reviews_per_product: int = 3,
    qna_per_product: int = 2,
    seed: int = 20260904,
) -> dict[str, Any]:
    """Sample raw crawl texts broadly across leaf categories with bounded memory."""
    run = Path(run_dir)
    _validate_run(run)
    blocked = _load_blocked_products(blocked_products_path)
    summary = _summary_index(run)
    candidates: dict[str, list[dict[str, Any]]] = defaultdict(list)
    blocked_inventory_products = 0
    with (run / "inventory.jsonl").open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            raw = row.get("product") or {}
            sources = row.get("sources") or ([row["source"]] if row.get("source") else [])
            if not sources:
                continue
            source = sources[0]
            path = str(source.get("fullPath") or "").strip()
            parts = [value.strip() for value in path.split(">") if value.strip()]
            if len(parts) < 2:
                continue
            canonical = str(row["key"])
            aliases = _aliases(row)
            if blocked & aliases:
                blocked_inventory_products += 1
                continue
            product_type = "catalog" if raw.get("productGroup") == "catalog" else "single"
            stats = summary.get((parts[-1], str(raw.get("name") or ""), product_type))
            if not stats:
                continue
            review_count = int(stats.get("collectedReviewCount") or 0)
            qna_count = int(stats.get("collectedQnaCount") or 0)
            if review_count + qna_count == 0:
                continue
            candidates[path].append({
                "product_id": canonical,
                "aliases": aliases,
                "category_path": path,
                "category_key": make_category_key(path),
                "product_name": str(raw.get("name") or canonical),
                "review_count": review_count,
                "qna_count": qna_count,
            })

    selected: dict[str, dict[str, Any]] = {}
    for leaf, values in candidates.items():
        # Prefer products containing both modalities, then keep deterministic variety.
        values.sort(key=lambda value: (
            -(value["review_count"] > 0 and value["qna_count"] > 0),
            _hash_int(seed, leaf, value["product_id"]),
        ))
        for value in values[:products_per_leaf]:
            selected[value["product_id"]] = value
    alias_to_product = {
        alias: product_id
        for product_id, value in selected.items()
        for alias in value["aliases"]
    }
    reservoirs = {
        (product_id, kind): _Reservoir(
            reviews_per_product if kind == "review" else qna_per_product, seed
        )
        for product_id in selected
        for kind in ("review", "qna")
    }
    pages = Counter()
    for page in _iter_pages(run):
        product_id = alias_to_product.get(str(page.get("productKey") or ""))
        if product_id is None:
            continue
        pages[str(page.get("kind"))] += 1
        context = selected[product_id]
        for record in _normalize_page(page, product_id, context):
            reservoirs[(product_id, record["record_type"])].add(record)

    by_leaf: dict[str, dict[str, list[dict[str, Any]]]] = defaultdict(
        lambda: {"review": [], "qna": []}
    )
    for (product_id, kind), reservoir in reservoirs.items():
        leaf = selected[product_id]["category_path"]
        by_leaf[leaf][kind].extend(reservoir.records())
    for leaf in by_leaf:
        for kind in ("review", "qna"):
            by_leaf[leaf][kind].sort(key=lambda row: _hash_int(seed, row["sample_id"]))

    chosen: list[dict[str, Any]] = []
    seen_text: set[str] = set()
    leaf_order = sorted(by_leaf, key=lambda value: _hash_int(seed, "leaf", value))
    offsets = {leaf: {"review": 0, "qna": 0} for leaf in leaf_order}
    # Each round gives every leaf and modality an equal chance before taking more.
    while len(chosen) < target_records:
        added = False
        for kind in ("review", "qna"):
            for leaf in leaf_order:
                values = by_leaf[leaf][kind]
                index = offsets[leaf][kind]
                while index < len(values):
                    row = values[index]
                    index += 1
                    key = _text_key(row["text"])
                    if key in seen_text:
                        continue
                    seen_text.add(key)
                    chosen.append(row)
                    added = True
                    break
                offsets[leaf][kind] = index
                if len(chosen) >= target_records:
                    break
            if len(chosen) >= target_records:
                break
        if not added:
            break
    if len(chosen) < target_records:
        raise ValueError(
            f"only {len(chosen)} unique records were available; requested {target_records}"
        )
    chosen.sort(key=lambda row: row["sample_id"])
    output = Path(output_path)
    write_jsonl(output, chosen)
    report = {
        "schema": "palda-balanced-raw-sample-v1",
        "source_run": str(run.resolve()),
        "seed": seed,
        "target_records": target_records,
        "counts": dict(Counter(row["record_type"] for row in chosen)),
        "products": len({row["product_id"] for row in chosen}),
        "cat2_categories": len({row["category_key"] for row in chosen}),
        "leaf_categories": len({row["category_path"] for row in chosen}),
        "available_leaf_categories": len(candidates),
        "selected_products": len(selected),
        "blocked_inventory_products": blocked_inventory_products,
        "matched_pages": dict(pages),
        "caps": {
            "products_per_leaf": products_per_leaf,
            "reviews_per_product": reviews_per_product,
            "qna_per_product": qna_per_product,
        },
        "file": {"path": output.name, "sha256": sha256_file(output)},
    }
    write_json(output.with_suffix(".manifest.json"), report)
    return report


class _Reservoir:
    def __init__(self, capacity: int, seed: int) -> None:
        self.capacity = capacity
        self.seed = seed
        self.heap: list[tuple[int, str, dict[str, Any]]] = []

    def add(self, row: dict[str, Any]) -> None:
        priority = _hash_int(self.seed, row["sample_id"])
        item = (-priority, row["sample_id"], row)
        if len(self.heap) < self.capacity:
            heapq.heappush(self.heap, item)
        elif item > self.heap[0]:
            heapq.heapreplace(self.heap, item)

    def records(self) -> list[dict[str, Any]]:
        return [value[2] for value in sorted(self.heap, reverse=True)]


def _validate_run(run: Path) -> None:
    progress = json.loads((run / "progress.json").read_text(encoding="utf-8"))
    if progress.get("status") != "complete":
        raise ValueError("crawler run is not complete")
    if (progress.get("completion") or {}).get("queueDrained") is not True:
        raise ValueError("crawler queue is not drained")


def _load_blocked_products(path: str | Path | None) -> set[str]:
    if path is None:
        return set()
    values = {line.strip() for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()}
    expanded = set(values)
    for value in values:
        expanded.add(value.removeprefix("catalog:"))
        expanded.add(f"catalog:{value.removeprefix('catalog:')}")
    return expanded


def _summary_index(run: Path) -> dict[tuple[str, str, str], dict[str, Any]]:
    summary = json.loads((run / "content-summary.json").read_text(encoding="utf-8"))
    result = {}
    for category in summary.get("categories", []):
        leaf = str(category.get("categoryName") or "")
        for product in category.get("products", []):
            result[(leaf, str(product.get("productName") or ""), str(product.get("productType") or ""))] = product
    return result


def _aliases(row: dict[str, Any]) -> set[str]:
    raw = row.get("product") or {}
    values = {str(row["key"])}
    for field in ("productId", "nvMid", "catalogNvMid", "channelProductNo"):
        if raw.get(field) is not None:
            value = str(raw[field])
            values.update((value, f"catalog:{value}"))
    return values


def _iter_pages(run: Path) -> Iterator[dict[str, Any]]:
    manifest = json.loads((run / "content-manifest.json").read_text(encoding="utf-8"))
    for chunk in manifest.get("chunks", []):
        if chunk.get("status") != "sealed":
            continue
        path = run / chunk["relative_path"]
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    yield json.loads(line)


def _normalize_page(
    page: dict[str, Any], product_id: str, context: dict[str, Any]
) -> Iterator[dict[str, Any]]:
    if page.get("status") not in (200, 204):
        return
    response = page.get("response") or {}
    if page.get("kind") == "review":
        if isinstance(response.get("contents"), list):
            values, source = response["contents"], "naver_brand"
        else:
            values, source = (response.get("data") or {}).get("reviews") or [], "naver_catalog"
        for index, item in enumerate(values):
            text = item.get("reviewContent") if source == "naver_brand" else item.get("content")
            text = str(text or "").strip()
            if not text:
                continue
            raw_id = item.get("id") if source == "naver_brand" else item.get("mallReviewId") or item.get("id")
            sample_id = (
                f"review_{source}_{raw_id}"
                if raw_id is not None
                else _stable_id("review", source, product_id, page.get("page"), index, text)
            )
            yield _sample(sample_id, product_id, "review", text, context, source)
    elif page.get("kind") == "qna":
        for index, item in enumerate(response.get("qnaList") or []):
            if item.get("isReply") or item.get("isSecret"):
                continue
            text = str(item.get("text") or "").strip()
            if not text:
                continue
            raw_id = item.get("id")
            sample_id = (
                f"qna_naver_brand_{raw_id}"
                if raw_id is not None
                else _stable_id("qna", product_id, page.get("page"), index, text)
            )
            yield _sample(sample_id, product_id, "qna", text, context, "naver_brand")


def _sample(
    sample_id: str,
    product_id: str,
    record_type: str,
    text: str,
    context: dict[str, Any],
    source: str,
) -> dict[str, Any]:
    return {
        "sample_id": sample_id,
        "product_id": product_id,
        "record_type": record_type,
        "text": text,
        "category_path": context["category_path"],
        "category_key": context["category_key"],
        "source_kind": "real",
        "source_split": "train",
        "usable": True,
        "metadata": {"source": source, "product_name": context["product_name"]},
    }


def _stable_id(prefix: str, *values: Any) -> str:
    raw = "\x1f".join(str(value or "") for value in values)
    return f"{prefix}_{hashlib.sha256(raw.encode()).hexdigest()[:20]}"


def _hash_int(seed: int, *values: Any) -> int:
    raw = "\x1f".join([str(seed), *(str(value) for value in values)])
    return int.from_bytes(hashlib.sha256(raw.encode()).digest()[:8], "big")


def _text_key(text: str) -> str:
    return hashlib.sha256("".join(text.split()).lower().encode()).hexdigest()
