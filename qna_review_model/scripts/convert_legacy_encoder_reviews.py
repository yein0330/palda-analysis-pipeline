#!/usr/bin/env python
"""Convert a reviewed PALDA v1 CSV into the Encoder review CSV schema.

The legacy UI allowed free-form Korean labels.  Conversion therefore resolves
labels against each record's Cat2 taxonomy and falls back to the existing
Encoder target when a reviewed value cannot be mapped safely.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any

from qna_review_model.encoder_data import EncoderTrainingRecord, canonical_theme_id
from qna_review_model.encoder_review import load_review_rows


KOREAN_THEME_CANDIDATES = {
    "가격": ["PRICE"], "가격/가성비": ["PRICE"], "가성비": ["PRICE"],
    "배송": ["DELIVERY"], "배송비": ["DELIVERY", "PRICE"],
    "포장": ["PACKAGING"],
    "서비스": ["CUSTOMER_SUPPORT"], "고객 서비스": ["CUSTOMER_SUPPORT"],
    "고객 응대": ["CUSTOMER_SUPPORT"], "고객 관리": ["CUSTOMER_SUPPORT"],
    "고객상담": ["CUSTOMER_SUPPORT"], "고객센터": ["CUSTOMER_SUPPORT"],
    "반품": ["RETURN_EXCHANGE"], "교환": ["RETURN_EXCHANGE"],
    "환불": ["RETURN_EXCHANGE"],
    "설치": ["INSTALLATION", "INSTALLATION_SERVICE"], "AS": ["AFTER_SALES"],
    "기능/효과": ["FUNCTIONALITY", "EFFECT", "PERFORMANCE"], "효과": ["EFFECT"],
    "품질": ["QUALITY"], "사용성": ["USABILITY"], "편리함": ["CONVENIENCE"],
    "디자인": ["DESIGN"], "안전성": ["SAFETY"], "내구성": ["DURABILITY"],
    "사이즈": ["SIZE", "FIT", "DIMENSIONS"], "크기": ["SIZE", "DIMENSIONS"],
    "총길이": ["DIMENSIONS", "SIZE"], "용량": ["CAPACITY", "QUANTITY"],
    "위생": ["HYGIENE", "CLEANLINESS"], "소음": ["NOISE"],
    "위치": ["LOCATION"], "예약": ["RESERVATION"],
    "재고": ["OTHER"], "재구매": ["OTHER"], "재구매 의향": ["OTHER"],
    "재주문 의사": ["OTHER"], "의사소통": ["CUSTOMER_SUPPORT"],
    "운영": ["CUSTOMER_SUPPORT"], "여행": ["RENTAL_EXPERIENCE", "OTHER"],
    "기준": ["OTHER"], "기타": ["OTHER"],
}

SERVICE_CANDIDATES = {
    "delivery": ["DELIVERY"], "배송": ["DELIVERY"], "포장": ["PACKAGING"],
    "support": ["CUSTOMER_SUPPORT"], "고객 서비스": ["CUSTOMER_SUPPORT"],
    "고객 응대": ["CUSTOMER_SUPPORT"], "고객 상담": ["CUSTOMER_SUPPORT"],
    "고객상담": ["CUSTOMER_SUPPORT"], "고객센터": ["CUSTOMER_SUPPORT"],
    "installation_as": ["INSTALLATION_SERVICE", "AFTER_SALES"],
    "서비스": ["CUSTOMER_SUPPORT"], "교환 요청": ["RETURN_EXCHANGE"],
    "취소 처리": ["RETURN_EXCHANGE"],
}

SENTIMENT = {
    "positive": "positive", "긍정": "positive",
    "negative": "negative", "부정": "negative",
    "neutral": "neutral", "중립": "neutral",
}

OUTPUT_COLUMNS = [
    "reviewer_id", "annotation_id", "product_id", "source_split", "category_path",
    "record_type", "text", "usable", "review_labels_json", "qna_primary_theme_id",
    "qna_secondary_theme_ids_json", "model_confidence", "validation_status",
    "reviewer_status", "reviewer_notes",
]


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--taxonomy", type=Path, required=True)
    parser.add_argument("--legacy-csv", type=Path, required=True)
    parser.add_argument("--newer-reviewed-csv", type=Path, nargs="*", default=[])
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--audit-json", type=Path, required=True)
    return parser.parse_args()


def choose_theme(raw: str, allowed: dict[str, dict[str, Any]]) -> str | None:
    value = raw.strip()
    if not value:
        return None
    labels = {}
    for theme_id, theme in allowed.items():
        labels[str(theme["reviewLabel"]).strip().lower()] = theme_id
        labels[str(theme["qnaLabel"]).removesuffix(" 문의").strip().lower()] = theme_id
    direct = canonical_theme_id(value)
    candidates = [direct, *KOREAN_THEME_CANDIDATES.get(value, [])]
    if value.lower() in labels:
        candidates.insert(0, labels[value.lower()])
    for candidate in candidates:
        if candidate in allowed:
            return candidate
    return None


def service_theme(raw: str, allowed: dict[str, dict[str, Any]]) -> str | None:
    for candidate in SERVICE_CANDIDATES.get(raw.strip(), []):
        if candidate in allowed:
            return candidate
    return None


def base_targets(record: EncoderTrainingRecord) -> tuple[list[dict[str, str]], str, list[str]]:
    if record.sourceType == "review":
        labels = [
            {"themeId": item.themeId, "sentiment": item.sentiment}
            for item in record.labels.reviewLabels
        ]
        return labels, "", []
    primary = record.labels.qnaPrimaryTheme
    return [], primary.themeId if primary else "OTHER", [
        item.themeId for item in record.labels.qnaSecondaryThemes
    ]


def main() -> None:
    args = arguments()
    base = {
        row.itemId: row for row in (
            EncoderTrainingRecord.model_validate_json(line)
            for line in args.dataset.read_text(encoding="utf-8").splitlines() if line.strip()
        )
    }
    taxonomy = json.loads(args.taxonomy.read_text(encoding="utf-8"))
    allowed_by_cat = {
        row["level2Id"]: {theme["canonicalThemeId"]: theme for theme in row["allowedThemes"]}
        for row in taxonomy["categories"]
    }
    newer, newer_errors = load_review_rows(args.newer_reviewed_csv)
    newer_done = {
        item_id for item_id, row in newer.items()
        if str(row.get("reviewer_status") or "").strip().lower() in {"approved", "corrected"}
    }
    stats = Counter()
    unmapped = Counter()
    rows = []
    with args.legacy_csv.open(encoding="utf-8-sig", newline="") as handle:
        for legacy in csv.DictReader(handle):
            item_id = str(legacy.get("annotation_id") or "").strip()
            record = base.get(item_id)
            if record is None:
                stats["unknown_annotation_id"] += 1
                continue
            if item_id in newer_done:
                stats["skipped_newer_review_overlap"] += 1
                continue
            if str(legacy.get("reviewer_status") or "").strip().lower() not in {"approved", "corrected"}:
                stats["skipped_not_completed"] += 1
                continue
            allowed = allowed_by_cat[record.category.level2Id]
            themes = []
            for index in range(1, 4):
                raw = str(legacy.get(f"theme_{index}") or "").strip()
                if not raw:
                    continue
                mapped = choose_theme(raw, allowed)
                if mapped is None:
                    unmapped[raw] += 1
                    continue
                themes.append((mapped, str(legacy.get(f"polarity_{index}") or "").strip()))
            for index in range(1, 4):
                raw = str(legacy.get(f"service_group_{index}") or "").strip()
                if not raw:
                    continue
                mapped = service_theme(raw, allowed)
                if mapped is None:
                    unmapped[f"service:{raw}"] += 1
                    continue
                themes.append((mapped, str(legacy.get(f"service_polarity_{index}") or "").strip()))

            review_labels, primary, secondary = base_targets(record)
            used_fallback = False
            usable = record.usable
            if record.sourceType == "review":
                converted = []
                seen = set()
                for theme_id, polarity in themes:
                    sentiment = SENTIMENT.get(polarity)
                    if sentiment is None or theme_id in {"OTHER", "UNUSABLE"}:
                        continue
                    pair = (theme_id, sentiment)
                    if pair not in seen:
                        seen.add(pair)
                        converted.append({"themeId": theme_id, "sentiment": sentiment})
                if converted:
                    review_labels = converted
                else:
                    used_fallback = True
            else:
                stage = str(legacy.get("qna_stage") or "").strip()
                usable = stage != "irrelevant"
                ordered = []
                for theme_id, _ in themes:
                    if theme_id not in ordered and theme_id != "UNUSABLE":
                        ordered.append(theme_id)
                if not usable:
                    primary, secondary = "UNUSABLE", []
                elif ordered:
                    primary, secondary = ordered[0], ordered[1:]
                else:
                    used_fallback = True
            if used_fallback:
                stats["records_using_base_target_fallback"] += 1
            output = {
                "reviewer_id": str(legacy.get("reviewer_id") or "annotator-1"),
                "annotation_id": item_id,
                "product_id": record.productGroupId,
                "source_split": record.annotation.sourceSplit,
                "category_path": record.category.path,
                "record_type": record.sourceType,
                "text": record.text,
                "usable": str(usable).lower(),
                "review_labels_json": json.dumps(review_labels, ensure_ascii=False, separators=(",", ":")),
                "qna_primary_theme_id": primary,
                "qna_secondary_theme_ids_json": json.dumps(secondary, ensure_ascii=False, separators=(",", ":")),
                "model_confidence": record.annotation.modelConfidence,
                "validation_status": record.annotation.validationStatus,
                "reviewer_status": str(legacy.get("reviewer_status") or "approved").strip().lower(),
                "reviewer_notes": str(legacy.get("reviewer_notes") or ""),
            }
            rows.append(output)
            stats[f"converted_{record.annotation.sourceSplit}"] += 1
            stats[f"converted_{record.sourceType}"] += 1

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.output_csv.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=OUTPUT_COLUMNS)
        writer.writeheader(); writer.writerows(rows)
    audit = {
        "schemaVersion": "legacy-encoder-review-conversion-audit-v1",
        "source": str(args.legacy_csv),
        "output": str(args.output_csv),
        "newerReviewedCsv": [str(path) for path in args.newer_reviewed_csv],
        "records": len(rows),
        "stats": dict(stats),
        "unmappedMentions": dict(unmapped.most_common()),
        "newerReviewLoadErrors": newer_errors,
        "policy": "newer Encoder review wins; category-taxonomy mapping; unmappable-only records retain base targets",
    }
    args.audit_json.write_text(json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(audit, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
