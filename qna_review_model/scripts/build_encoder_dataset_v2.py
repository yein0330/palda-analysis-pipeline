#!/usr/bin/env python
"""Convert PALDA v1 draft annotations into the v2 Encoder Silver dataset."""

from __future__ import annotations

import argparse
import csv
import html
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

from qna_review_model.encoder_data import (
    EncoderAnnotationMetadata,
    EncoderCategory,
    EncoderQnaTheme,
    EncoderReviewLabel,
    EncoderTargets,
    EncoderTrainingRecord,
    canonical_theme_id,
    category_theme_id,
    stable_id,
)


SERVICE_THEME_MAP = {
    "delivery_speed": ("DELIVERY", "배송", "배송 문의"),
    "delivery_reliability": ("DELIVERY", "배송", "배송 문의"),
    "delivery_staff": ("DELIVERY", "배송", "배송 문의"),
    "customer_support": ("CUSTOMER_SUPPORT", "고객지원", "고객지원 문의"),
    "returns_refunds": ("RETURN_EXCHANGE", "교환·반품", "교환·반품 문의"),
    "packaging": ("PACKAGING", "포장", "포장 문의"),
    "installation": ("INSTALLATION_SERVICE", "설치 서비스", "설치 문의"),
    "after_sales": ("AFTER_SALES", "A/S", "A/S 문의"),
}

RETURN_PRIMARY_PATTERN = re.compile(
    r"(?:교환|반품|환불).{0,16}(?:접수|신청|수거|택배|처리|완료|취소|부탁|가능|되나|되나요|어떻게|언제|안\s*됐|안됐|불가)"
    r"|(?:접수|신청|수거|처리).{0,16}(?:교환|반품|환불)"
)


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--drafts", type=Path, nargs="+", required=True)
    parser.add_argument("--service-drafts", type=Path, nargs="*", default=[])
    parser.add_argument("--theme-config", type=Path, required=True)
    parser.add_argument("--category-csv", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--taxonomy-version", default="theme-v1.0.0")
    return parser.parse_args()


def read_jsonl(paths: Iterable[Path]) -> Iterable[dict[str, Any]]:
    for path in paths:
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    yield json.loads(line)


def parse_raw_object(raw: object) -> dict[str, Any] | None:
    text = str(raw or "").strip()
    start, end = text.find("{"), text.rfind("}")
    if start < 0 or end < start:
        return None
    try:
        value = json.loads(text[start:end + 1])
    except json.JSONDecodeError:
        return None
    return value if isinstance(value, dict) else None


def cat2_path(category_path: str) -> str:
    levels = [part.strip() for part in category_path.split(">") if part.strip()]
    return " > ".join(levels[:2]) if len(levels) >= 2 else (levels[0] if levels else "기타")


def clean_text(value: object) -> str:
    text = html.unescape(str(value or ""))
    text = re.sub(r"<br\s*/?>", "\n", text, flags=re.IGNORECASE)
    return text.strip()


def load_leaf_ids(path: Path) -> dict[str, str]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return {row["fullPath"].strip(): row["cid"].strip() for row in csv.DictReader(handle)}


def unique_mentions(row: dict[str, Any]) -> list[dict[str, str]]:
    """Recover mixed Review sentiment from raw output while filtering invalid labels."""
    source = row["input"]
    allowed = {theme["theme_id"] for theme in source.get("allowed_themes", [])}
    raw = parse_raw_object(row.get("raw_output"))
    candidates = raw.get("mentions", []) if raw else []
    if not isinstance(candidates, list) or not candidates:
        candidates = (row.get("draft") or {}).get("mentions", [])
    found: list[dict[str, str]] = []
    seen: set[tuple[str, str]] = set()
    for mention in candidates:
        if not isinstance(mention, dict):
            continue
        theme_id = str(mention.get("theme_id") or "").strip()
        polarity = str(mention.get("polarity") or "").strip()
        if theme_id not in allowed:
            continue
        if source["record_type"] == "review" and polarity not in {"positive", "negative", "neutral"}:
            continue
        key = (theme_id, polarity)
        if key not in seen:
            seen.add(key)
            found.append({"theme_id": theme_id, "polarity": polarity})
    return found


def service_mentions(row: dict[str, Any] | None) -> list[dict[str, str]]:
    if not row or not row.get("draft"):
        return []
    found = []
    for mention in row["draft"].get("mentions", []):
        service_type = str(mention.get("service_type") or "")
        if service_type in SERVICE_THEME_MAP:
            found.append({
                "theme_id": SERVICE_THEME_MAP[service_type][0],
                "polarity": str(mention.get("polarity") or "neutral"),
            })
    return found


def theme_catalog(config: dict[str, Any], observed_cat2: set[str], taxonomy_version: str) -> dict[str, Any]:
    categories = []
    canonical_labels: dict[str, dict[str, str]] = {}
    for path in sorted(observed_cat2):
        levels = [part.strip() for part in path.split(">")]
        cat1 = levels[0]
        inherited = [
            *((theme, "global", True) for theme in config.get("global", [])),
            *((theme, "cat1", True) for theme in config.get("cat1", {}).get(cat1, [])),
            *((theme, "cat2", False) for theme in config.get("cat2", {}).get(path, [])),
        ]
        themes = []
        seen: set[str] = set()
        for theme, level, annotation_ready in inherited:
            canonical = canonical_theme_id(theme["theme_id"])
            ctid = category_theme_id(path, canonical)
            if ctid in seen:
                continue
            seen.add(ctid)
            label = str(theme["label"])
            canonical_labels.setdefault(canonical, {"reviewLabel": label, "qnaLabel": f"{label} 문의"})
            themes.append({
                "categoryThemeId": ctid,
                "canonicalThemeId": canonical,
                "reviewLabel": label,
                "qnaLabel": f"{label} 문의",
                "sourceLevel": level,
                "keywords": theme.get("keywords", []),
                "annotationReady": annotation_ready,
            })
        for service_id, review_label, qna_label in SERVICE_THEME_MAP.values():
            ctid = category_theme_id(path, service_id)
            if ctid not in seen:
                seen.add(ctid)
                themes.append({
                    "categoryThemeId": ctid,
                    "canonicalThemeId": service_id,
                    "reviewLabel": review_label,
                    "qnaLabel": qna_label,
                    "sourceLevel": "global",
                    "keywords": [],
                    "annotationReady": True,
                })
            canonical_labels[service_id] = {"reviewLabel": review_label, "qnaLabel": qna_label}
        for special, review_label, qna_label in (
            ("OTHER", "기타", "기타 문의"), ("UNUSABLE", "분석 불가", "분석 불가")
        ):
            themes.append({
                "categoryThemeId": category_theme_id(path, special),
                "canonicalThemeId": special,
                "reviewLabel": review_label,
                "qnaLabel": qna_label,
                "sourceLevel": "global",
                "keywords": [],
                "annotationReady": True,
            })
            canonical_labels[special] = {"reviewLabel": review_label, "qnaLabel": qna_label}
        categories.append({
            "level1Id": stable_id("cat1", cat1),
            "level1Name": cat1,
            "level2Id": stable_id("cat2", path),
            "level2Name": levels[1] if len(levels) > 1 else cat1,
            "path": path,
            "allowedThemes": themes,
        })
    return {
        "schemaVersion": "theme-taxonomy-v1",
        "taxonomyVersion": taxonomy_version,
        "status": "draft_pending_human_review",
        "canonicalThemes": [
            {"canonicalThemeId": key, **value} for key, value in sorted(canonical_labels.items())
        ],
        "categories": categories,
    }


def main() -> None:
    args = arguments()
    drafts = list(read_jsonl(args.drafts))
    by_id: dict[str, dict[str, Any]] = {}
    duplicate_ids = 0
    for row in drafts:
        annotation_id = row["input"]["annotation_id"]
        if annotation_id in by_id:
            duplicate_ids += 1
            continue
        by_id[annotation_id] = row
    services = {row["annotation_id"]: row for row in read_jsonl(args.service_drafts)}
    config = json.loads(args.theme_config.read_text(encoding="utf-8"))
    leaf_ids = load_leaf_ids(args.category_csv)
    observed_cat2 = {cat2_path(row["input"]["category_path"]) for row in by_id.values()}
    taxonomy = theme_catalog(config, observed_cat2, args.taxonomy_version)
    taxonomy_by_cat2 = {row["path"]: row for row in taxonomy["categories"]}

    records: list[EncoderTrainingRecord] = []
    excluded_rows: list[dict[str, Any]] = []
    excluded = Counter()
    stats = Counter()
    theme_counts = Counter()
    mixed_review_records = 0
    for annotation_id, row in by_id.items():
        source = row["input"]
        draft = row.get("draft")
        if draft is None:
            excluded["unparseable_product_draft"] += 1
            excluded_rows.append({
                "annotationId": annotation_id,
                "reason": "unparseable_product_draft",
                "input": source,
                "validationErrors": row.get("validation_errors", []),
                "rawOutput": row.get("raw_output", ""),
            })
            continue
        path = cat2_path(source["category_path"])
        category_row = taxonomy_by_cat2[path]
        allowed = {item["canonicalThemeId"]: item for item in category_row["allowedThemes"]}
        product_mentions = unique_mentions(row)
        service = service_mentions(services.get(annotation_id))
        confidence = str(draft.get("confidence") or "medium")
        weight = {"high": 1.0, "medium": 0.75, "low": 0.5}.get(confidence, 0.75)
        if row.get("validation_status") != "valid":
            weight *= 0.7
        labels: EncoderTargets
        usable = True
        if source["record_type"] == "review":
            review_labels: list[EncoderReviewLabel] = []
            seen_pairs: set[tuple[str, str]] = set()
            sentiments_by_theme: dict[str, set[str]] = defaultdict(set)
            combined = [*product_mentions, *service]
            for mention in combined:
                canonical = canonical_theme_id(mention["theme_id"])
                sentiment = mention["polarity"]
                if sentiment not in {"positive", "negative", "neutral"} or canonical not in allowed:
                    continue
                pair = (canonical, sentiment)
                if pair in seen_pairs:
                    continue
                seen_pairs.add(pair)
                sentiments_by_theme[canonical].add(sentiment)
                review_labels.append(EncoderReviewLabel(
                    themeId=canonical,
                    categoryThemeId=allowed[canonical]["categoryThemeId"],
                    sentiment=sentiment,
                ))
                theme_counts[f"review/{sentiment}/{canonical}"] += 1
            mixed_review_records += any(len(values) > 1 for values in sentiments_by_theme.values())
            labels = EncoderTargets(reviewLabels=review_labels)
        else:
            qna_stage = draft.get("qna_stage")
            usable = qna_stage != "irrelevant"
            ordered: list[str] = []
            for mention in [*product_mentions, *service]:
                canonical = canonical_theme_id(mention["theme_id"])
                if canonical in allowed and canonical not in ordered:
                    ordered.append(canonical)
            if "RETURN_EXCHANGE" in ordered and RETURN_PRIMARY_PATTERN.search(clean_text(source["text"])):
                ordered.remove("RETURN_EXCHANGE")
                ordered.insert(0, "RETURN_EXCHANGE")
            if not usable:
                primary_id, secondary_ids = "UNUSABLE", []
            elif ordered:
                primary_id, secondary_ids = ordered[0], ordered[1:]
            else:
                primary_id, secondary_ids = "OTHER", []
            primary = EncoderQnaTheme(
                themeId=primary_id,
                categoryThemeId=allowed[primary_id]["categoryThemeId"],
            )
            secondaries = [EncoderQnaTheme(
                themeId=value,
                categoryThemeId=allowed[value]["categoryThemeId"],
            ) for value in secondary_ids]
            labels = EncoderTargets(qnaPrimaryTheme=primary, qnaSecondaryThemes=secondaries)
            theme_counts[f"qna/primary/{primary_id}"] += 1
            for value in secondary_ids:
                theme_counts[f"qna/secondary/{value}"] += 1
        category = EncoderCategory(
            level1Id=category_row["level1Id"],
            level1Name=category_row["level1Name"],
            level2Id=category_row["level2Id"],
            level2Name=category_row["level2Name"],
            path=source["category_path"],
        )
        record = EncoderTrainingRecord(
            taxonomyVersion=args.taxonomy_version,
            itemId=annotation_id,
            productGroupId=source["product_id"],
            sourceType=source["record_type"],
            category=category,
            text=clean_text(source["text"]),
            usable=usable,
            labels=labels,
            annotation=EncoderAnnotationMetadata(
                grade="silver",
                source="qwen3-8b-draft-plus-service-v1",
                sourceSplit=source["split"],
                modelConfidence=confidence,
                validationStatus=row["validation_status"],
                reviewerStatus=row.get("reviewer_status", "pending"),
                sampleWeight=round(weight, 3),
            ),
        )
        records.append(record)
        stats[source["record_type"]] += 1
        stats[f"source_split/{source['split']}"] += 1
        stats[f"validation/{row['validation_status']}"] += 1

    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "silver-train.jsonl").open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(record.model_dump_json() + "\n")
    with (args.output_dir / "excluded-for-human-review.jsonl").open("w", encoding="utf-8") as handle:
        for row in excluded_rows:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
    (args.output_dir / "theme_taxonomy.json").write_text(
        json.dumps(taxonomy, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    reviewer_columns = [
        "annotation_id", "product_id", "source_split", "category_path", "record_type", "text", "score",
        "usable", "review_labels_json", "qna_primary_theme_id", "qna_secondary_theme_ids_json",
        "model_confidence", "validation_status", "reviewer_status", "reviewer_notes",
    ]
    with (args.output_dir / "review.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=reviewer_columns)
        writer.writeheader()
        source_by_id = {row["input"]["annotation_id"]: row["input"] for row in by_id.values()}
        for record in records:
            source = source_by_id[record.itemId]
            writer.writerow({
                "annotation_id": record.itemId,
                "product_id": record.productGroupId,
                "source_split": record.annotation.sourceSplit,
                "category_path": record.category.path,
                "record_type": record.sourceType,
                "text": record.text,
                "score": source.get("score"),
                "usable": str(record.usable).lower(),
                "review_labels_json": json.dumps(
                    [item.model_dump() for item in record.labels.reviewLabels], ensure_ascii=False, separators=(",", ":")
                ),
                "qna_primary_theme_id": record.labels.qnaPrimaryTheme.themeId if record.labels.qnaPrimaryTheme else "",
                "qna_secondary_theme_ids_json": json.dumps(
                    [item.themeId for item in record.labels.qnaSecondaryThemes], ensure_ascii=False, separators=(",", ":")
                ),
                "model_confidence": record.annotation.modelConfidence,
                "validation_status": record.annotation.validationStatus,
                "reviewer_status": record.annotation.reviewerStatus,
                "reviewer_notes": "",
            })
    manifest = {
        "schemaVersion": "encoder-dataset-manifest-v1",
        "taxonomyVersion": args.taxonomy_version,
        "grade": "silver",
        "trainingPolicy": "Silver train candidate only. Validation and test must be built from human-reviewed Gold.",
        "sourceFiles": [str(path) for path in args.drafts],
        "serviceSourceFiles": [str(path) for path in args.service_drafts],
        "records": len(records),
        "counts": dict(sorted(stats.items())),
        "excluded": dict(excluded),
        "duplicateItemIds": duplicate_ids,
        "products": len({record.productGroupId for record in records}),
        "cat2": len(observed_cat2),
        "mixedSentimentReviewRecordsRecovered": mixed_review_records,
        "themeTargetCounts": dict(theme_counts.most_common()),
        "leafCategoryIdsAvailable": sum(source_by_id[r.itemId]["category_path"] in leaf_ids for r in records),
        "warnings": [
            "Cat2 candidate themes remain annotationReady=false until human review.",
            "Current records are model-generated Silver labels, not Gold evaluation data.",
            "sourceSplit is preserved as provenance only; silver-train.jsonl is not a Gold validation/test split.",
        ],
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
