"""Build product-level Decoder inputs from human-reviewed Encoder records."""

from __future__ import annotations

import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable


REPORT_INPUT_SCHEMA_VERSION = "report-input-v1"


def load_theme_labels(taxonomy_path: Path) -> dict[str, dict[str, str]]:
    """Return canonical theme ID -> Review/Q&A display labels."""
    payload = json.loads(taxonomy_path.read_text(encoding="utf-8"))
    return {
        row["canonicalThemeId"]: {
            "review": row["reviewLabel"],
            "qna": row["qnaLabel"],
        }
        for row in payload["canonicalThemes"]
    }


def _normal_text(value: str) -> str:
    return re.sub(r"\s+", " ", value).strip()


def _evidence(rows: Iterable[dict[str, Any]], limit: int) -> list[dict[str, str]]:
    """Select stable, non-duplicate evidence while preferring informative text."""
    candidates: list[tuple[int, str, str]] = []
    seen: set[str] = set()
    for row in rows:
        text = _normal_text(str(row.get("text") or ""))
        key = text.casefold()
        if not text or key in seen:
            continue
        seen.add(key)
        candidates.append((-len(text), str(row["itemId"]), text))
    candidates.sort()
    return [
        {"id": item_id, "text": text}
        for _, item_id, text in candidates[:limit]
    ]


def aggregate_product(
    rows: list[dict[str, Any]],
    theme_labels: dict[str, dict[str, str]],
    *,
    evidence_limit: int = 3,
) -> dict[str, Any]:
    """Aggregate one product's Gold item labels into report-input-v1."""
    if not rows:
        raise ValueError("cannot aggregate an empty product")
    product_ids = {str(row["productGroupId"]) for row in rows}
    if len(product_ids) != 1:
        raise ValueError("all rows must belong to one productGroupId")

    usable = [row for row in rows if row.get("usable", True)]
    reviews = [row for row in usable if row.get("sourceType") == "review"]
    qna = [row for row in usable if row.get("sourceType") == "qna"]
    review_groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    qna_groups: dict[str, list[dict[str, Any]]] = defaultdict(list)

    for row in reviews:
        for label in row.get("labels", {}).get("reviewLabels", []):
            theme_id = str(label["themeId"])
            sentiment = str(label["sentiment"])
            if theme_id not in {"OTHER", "UNUSABLE"}:
                review_groups[(theme_id, sentiment)].append(row)

    for row in qna:
        primary = row.get("labels", {}).get("qnaPrimaryTheme")
        theme_id = (
            primary.get("themeId") if isinstance(primary, dict) else primary
        )
        if theme_id and theme_id not in {"OTHER", "UNUSABLE"}:
            qna_groups[str(theme_id)].append(row)

    first = rows[0]
    category = first.get("category", {})
    review_output = []
    for (theme_id, sentiment), group_rows in review_groups.items():
        labels = theme_labels.get(theme_id, {})
        review_output.append({
            "themeId": theme_id,
            "displayTheme": labels.get("review", theme_id),
            "sentiment": sentiment,
            "mentionCount": len(group_rows),
            "evidence": _evidence(group_rows, evidence_limit),
        })
    review_output.sort(
        key=lambda row: (-row["mentionCount"], row["themeId"], row["sentiment"])
    )

    qna_output = []
    denominator = len(qna)
    for theme_id, group_rows in qna_groups.items():
        labels = theme_labels.get(theme_id, {})
        qna_output.append({
            "themeId": theme_id,
            "displayTheme": labels.get("qna", theme_id),
            "primaryCount": len(group_rows),
            "ratio": round(len(group_rows) / denominator, 6) if denominator else 0.0,
            "evidence": _evidence(group_rows, evidence_limit),
        })
    qna_output.sort(key=lambda row: (-row["primaryCount"], row["themeId"]))

    return {
        "schemaVersion": REPORT_INPUT_SCHEMA_VERSION,
        "taxonomyVersion": first.get("taxonomyVersion", "theme-v1.0.0"),
        "category": {"path": category.get("path", "")},
        "sample": {
            "reviewSampleSize": len(reviews),
            "reviewMaxItems": 5000,
            "qnaSampleSize": len(qna),
            "qnaMaxItems": 5000,
        },
        "reviewGroups": review_output,
        "qnaGroups": qna_output,
    }


def build_oracle_decoder_inputs(
    input_dir: Path,
    output_dir: Path,
    taxonomy_path: Path,
    *,
    evidence_limit: int = 3,
) -> dict[str, Any]:
    """Build one Oracle Decoder dataset from Gold Encoder rows."""
    output_dir.mkdir(parents=True, exist_ok=True)
    theme_labels = load_theme_labels(taxonomy_path)
    counts: dict[str, int] = {}
    record_counts: dict[str, int] = {}
    product_sets: dict[str, set[str]] = {}

    for split in ("train", "validation", "test"):
        grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
        source_path = input_dir / f"{split}.jsonl"
        with source_path.open(encoding="utf-8") as source:
            for line in source:
                row = json.loads(line)
                if row.get("annotation", {}).get("grade") != "gold":
                    continue
                grouped[str(row["productGroupId"])].append(row)

        product_sets[split] = set(grouped)
        record_counts[split] = sum(len(rows) for rows in grouped.values())
        counts[split] = len(grouped)
        with (output_dir / f"{split}.jsonl").open("w", encoding="utf-8") as target:
            for product_id in sorted(grouped):
                rows = grouped[product_id]
                wrapper = {
                    "productGroupId": product_id,
                    "split": split,
                    "datasetGrade": "gold-oracle",
                    "sourceRecordCount": len(rows),
                    "reportInput": aggregate_product(
                        rows, theme_labels, evidence_limit=evidence_limit
                    ),
                }
                target.write(
                    json.dumps(wrapper, ensure_ascii=False, separators=(",", ":")) + "\n"
                )

    leakage = any(product_sets[a] & product_sets[b] for a, b in (
        ("train", "validation"), ("train", "test"), ("validation", "test")
    ))
    if leakage:
        raise ValueError("product leakage detected between Decoder splits")

    manifest = {
        "schemaVersion": "palda-decoder-oracle-input-manifest-v1",
        "datasetVersion": "palda-decoder-oracle-v1",
        "datasetRole": "decoder-input",
        "datasetGrade": "gold-oracle",
        "inputSchemaVersion": REPORT_INPUT_SCHEMA_VERSION,
        "taxonomyVersion": "theme-v1.0.0",
        "source": str(input_dir),
        "policy": "Human-reviewed Gold Encoder labels aggregated by product; no generated report targets yet.",
        "products": counts,
        "sourceGoldRecords": record_counts,
        "evidenceLimitPerGroup": evidence_limit,
        "productSplitLeakage": False,
        "nextStep": "Generate and human-review report-output-v1 targets without changing these inputs.",
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return manifest
