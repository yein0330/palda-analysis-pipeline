"""Streaming reader and normalizer for completed Naver crawler runs."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import Any, Literal

from .schemas import ProductContext, QNARecord, ReviewRecord


def validate_completed_run(run_dir: Path) -> dict[str, Any]:
    progress = json.loads((run_dir / "progress.json").read_text(encoding="utf-8"))
    completion = progress.get("completion", {})
    jobs = progress.get("consumer", {}).get("jobs", {})
    if progress.get("status") != "complete":
        raise ValueError(f"crawler run is not complete: {progress.get('status')!r}")
    if completion.get("queueDrained") is not True:
        raise ValueError("crawler queue is not drained")
    if jobs.get("failed", 0):
        raise ValueError(f"crawler run contains {jobs['failed']} failed jobs")
    return progress


def manifest_chunks(run_dir: Path, kind: Literal["review", "qna", "all"] = "all") -> list[Path]:
    manifest = json.loads((run_dir / "content-manifest.json").read_text(encoding="utf-8"))
    paths: list[Path] = []
    for chunk in manifest["chunks"]:
        if chunk.get("status") != "sealed":
            continue
        chunk_kind = chunk.get("kind")
        if kind != "all" and chunk_kind not in {kind, "mixed"}:
            continue
        path = run_dir / chunk["relative_path"]
        if not path.is_file():
            raise FileNotFoundError(f"manifest chunk is missing: {path}")
        paths.append(path)
    return paths


def iter_pages(
    run_dir: Path,
    kind: Literal["review", "qna", "all"] = "all",
    max_pages: int | None = None,
) -> Iterator[dict[str, Any]]:
    seen = 0
    for path in manifest_chunks(run_dir, kind):
        with path.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, 1):
                if not line.strip():
                    continue
                try:
                    page = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"invalid JSONL: {path}:{line_number}: {exc}") from exc
                if kind != "all" and page.get("kind") != kind:
                    continue
                yield page
                seen += 1
                if max_pages is not None and seen >= max_pages:
                    return


def _stable_id(prefix: str, *values: Any) -> str:
    raw = "\x1f".join(str(value or "") for value in values)
    return f"{prefix}_{hashlib.sha256(raw.encode()).hexdigest()[:20]}"


def _nonblank(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _review_items(page: dict[str, Any]) -> tuple[Iterable[dict[str, Any]], str]:
    response = page.get("response") or {}
    if isinstance(response.get("contents"), list):
        return response["contents"], "naver_brand"
    data = response.get("data") or {}
    if isinstance(data, dict) and isinstance(data.get("reviews"), list):
        return data["reviews"], "naver_catalog"
    return [], "naver_brand"


def normalize_page(page: dict[str, Any]) -> Iterator[ReviewRecord | QNARecord]:
    product_id = str(page.get("productKey") or "").strip()
    if not product_id or page.get("status") not in {200, 204}:
        return
    if page.get("kind") == "review":
        items, source = _review_items(page)
        for index, item in enumerate(items):
            if source == "naver_brand":
                text = _nonblank(item.get("reviewContent"))
                raw_id = item.get("id")
                score = item.get("reviewScore")
                option = _nonblank(item.get("productOptionContent"))
                created_at = item.get("createDate")
            else:
                text = _nonblank(item.get("content"))
                raw_id = item.get("mallReviewId") or item.get("id")
                score = item.get("starScore")
                option = _nonblank(item.get("buyOption"))
                created_at = item.get("registerDate") or item.get("modifyDate")
            if not text:
                continue
            record_id = f"review_{source}_{raw_id}" if raw_id else _stable_id(
                "review", source, product_id, page.get("page"), index, text
            )
            yield ReviewRecord(
                record_id=record_id, product_id=product_id, text=text, score=score,
                option=option, created_at=created_at, source=source,
            )
    elif page.get("kind") == "qna":
        response = page.get("response") or {}
        for index, item in enumerate(response.get("qnaList") or []):
            if item.get("isReply") or item.get("isSecret"):
                continue
            question = _nonblank(item.get("text"))
            if not question:
                continue
            raw_id = item.get("id")
            record_id = f"qna_naver_brand_{raw_id}" if raw_id else _stable_id(
                "qna", product_id, page.get("page"), index, question
            )
            yield QNARecord(
                record_id=record_id, product_id=product_id, question=question,
                created_at=item.get("regDate"), has_seller_answer=item.get("hasSellerAnswer"),
            )


def iter_records(
    run_dir: Path,
    kind: Literal["review", "qna", "all"] = "all",
    max_pages: int | None = None,
    max_records: int | None = None,
) -> Iterator[ReviewRecord | QNARecord]:
    count = 0
    for page in iter_pages(run_dir, kind, max_pages):
        for record in normalize_page(page):
            yield record
            count += 1
            if max_records is not None and count >= max_records:
                return


def load_products(run_dir: Path) -> dict[str, ProductContext]:
    """Index product contexts by canonical key and all crawler aliases."""
    products: dict[str, ProductContext] = {}
    with (run_dir / "inventory.jsonl").open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid inventory JSONL at line {line_number}") from exc
            raw = row.get("product") or {}
            sources = row.get("sources") or ([row["source"]] if row.get("source") else [])
            source = sources[0] if sources else {}
            metadata = raw.get("rawMetadata") or {}
            path = _nonblank(source.get("fullPath"))
            levels = [part.strip() for part in path.split(">") if part.strip()] if path else []
            context = ProductContext(
                product_id=str(row["key"]), name=_nonblank(raw.get("name")) or str(row["key"]),
                category_cid=_nonblank(source.get("cid") or raw.get("categoryCid")),
                category_path=path, category_levels=levels,
                brand=_nonblank(metadata.get("brand")),
                price=raw.get("price") if isinstance(raw.get("price"), int) else None,
                options=[x.strip() for x in str(metadata.get("characterValue") or "").split("|") if x.strip()],
            )
            aliases = {str(row["key"])}
            for field in ("productId", "nvMid", "catalogNvMid", "channelProductNo"):
                if raw.get(field) is not None:
                    aliases.add(str(raw[field])); aliases.add(f"catalog:{raw[field]}")
            for alias in aliases:
                products.setdefault(alias, context)
    return products


def write_normalized(
    run_dir: Path, output: Path, kind: Literal["review", "qna", "all"] = "all",
    max_pages: int | None = None, max_records: int | None = None,
) -> dict[str, Any]:
    validate_completed_run(run_dir)
    output.parent.mkdir(parents=True, exist_ok=True)
    counts = {"review": 0, "qna": 0}
    with output.open("w", encoding="utf-8") as handle:
        for record in iter_records(run_dir, kind, max_pages, max_records):
            handle.write(record.model_dump_json() + "\n")
            counts[record.record_type] += 1
    return {"output": str(output), **counts, "total": sum(counts.values())}
