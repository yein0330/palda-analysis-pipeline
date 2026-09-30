#!/usr/bin/env python
"""Build strict Gold-style SFT splits from the reviewed product/service CSV."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

from qna_review_model.annotation import AnnotationQueueRow, normalize_draft_payload, validate_draft
from scripts.build_review_html import clean_text, simplify_service_mentions


SYSTEM = (
    "당신은 PALDA의 카테고리 인지형 고객의견 분석기입니다. 허용된 상품 Theme와 "
    "서비스 그룹(delivery, support, installation_as)만 사용하세요. 원문에 직접 나타난 "
    "항목만 고르고 evidence_quote는 원문의 정확한 연속 부분문자열로 작성하세요. "
    "JSON 객체 하나만 출력하세요."
)


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--queue", type=Path, required=True)
    parser.add_argument("--review-csv", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--approved-csv", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = arguments()
    queue = {
        row.annotation_id: row
        for row in map(AnnotationQueueRow.model_validate_json, args.queue.read_text(encoding="utf-8").splitlines())
    }
    with args.review_csv.open(encoding="utf-8-sig", newline="") as source:
        reader = csv.DictReader(source)
        fieldnames = reader.fieldnames or []
        csv_rows = list(reader)
    outputs: dict[str, list[dict]] = {split: [] for split in ("train", "validation", "test")}
    errors: list[dict[str, str]] = []
    service_groups: Counter[str] = Counter()
    approved_rows = []
    for csv_row in csv_rows:
        annotation_id = str(csv_row.get("annotation_id") or "")
        original = queue.get(annotation_id)
        if original is None:
            errors.append({"annotation_id": annotation_id, "error": "unknown annotation_id"})
            continue
        source_row = original.model_copy(update={"text": clean_text(csv_row.get("text"))})
        mentions = []
        for index in range(1, 4):
            theme_id = str(csv_row.get(f"theme_{index}") or "").strip()
            if theme_id:
                mentions.append({
                    "theme_id": theme_id,
                    "polarity": str(csv_row.get(f"polarity_{index}") or "").strip(),
                    "evidence_quote": clean_text(csv_row.get(f"evidence_{index}")),
                })
        product_payload = {
            "record_id": annotation_id,
            "qna_stage": str(csv_row.get("qna_stage") or "").strip() or None,
            "mentions": mentions,
            "new_theme_candidates": [
                value.strip() for value in str(csv_row.get("new_theme_candidates") or "").split("|") if value.strip()
            ],
            "confidence": str(csv_row.get("confidence") or "medium").strip(),
        }
        product, product_errors = validate_draft(
            source_row, normalize_draft_payload(source_row, product_payload)
        )
        detailed_services = []
        for index in range(1, 4):
            service_name = str(
                csv_row.get(f"service_group_{index}") or csv_row.get(f"service_type_{index}") or ""
            ).strip()
            if service_name:
                detailed_services.append({
                    "service_group": service_name,
                    "polarity": str(csv_row.get(f"service_polarity_{index}") or "neutral").strip(),
                    "evidence_quote": clean_text(csv_row.get(f"service_evidence_{index}")),
                })
        services = simplify_service_mentions(detailed_services)
        service_errors = [
            f"service evidence is not exact: {item['evidence_quote']!r}"
            for item in services if item["evidence_quote"] not in source_row.text
        ]
        row_errors = [*product_errors, *service_errors]
        approved = dict(csv_row)
        approved["reviewer_status"] = "approved" if product is not None and not row_errors else "pending"
        approved_rows.append(approved)
        if product is None or row_errors:
            errors.append({"annotation_id": annotation_id, "error": " | ".join(row_errors)})
            continue
        allowed = [theme.model_dump() for theme in source_row.allowed_themes]
        user = (
            f"TASK: {source_row.record_type.upper()}\nCATEGORY: {source_row.category_path}\n"
            f"ALLOWED_THEMES: {json.dumps(allowed, ensure_ascii=False, separators=(',', ':'))}\n"
            "SERVICE_GROUPS: delivery(배송), support(고객지원), installation_as(설치·AS)\n"
            f"RECORD_ID: {annotation_id}\nTEXT: {source_row.text}"
        )
        target = {
            "record_id": annotation_id,
            "qna_stage": product.qna_stage,
            "mentions": [mention.model_dump() for mention in product.mentions],
            "service_mentions": services,
        }
        for item in services:
            service_groups[item["service_group"]] += 1
        outputs[source_row.split].append({
            "record_id": annotation_id, "product_id": source_row.product_id,
            "split": source_row.split, "source_label": "reviewed_auto_approved_v1",
            "messages": [
                {"role": "system", "content": SYSTEM},
                {"role": "user", "content": user},
                {"role": "assistant", "content": json.dumps(target, ensure_ascii=False, separators=(",", ":"))},
            ],
        })
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for split, rows in outputs.items():
        with (args.output_dir / f"{split}.jsonl").open("w", encoding="utf-8") as target:
            for row in rows:
                target.write(json.dumps(row, ensure_ascii=False) + "\n")
    args.approved_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.approved_csv.open("w", encoding="utf-8-sig", newline="") as target:
        writer = csv.DictWriter(target, fieldnames=fieldnames)
        writer.writeheader(); writer.writerows(approved_rows)
    with (args.output_dir / "errors.csv").open("w", encoding="utf-8-sig", newline="") as target:
        writer = csv.DictWriter(target, fieldnames=["annotation_id", "error"])
        writer.writeheader(); writer.writerows(errors)
    product_ids = {split: {row["product_id"] for row in rows} for split, rows in outputs.items()}
    leakage = any(product_ids[a] & product_ids[b] for a, b in (
        ("train", "validation"), ("train", "test"), ("validation", "test")
    ))
    manifest = {
        "schema_version": "palda_reviewed_sft_v1", "source": str(args.review_csv),
        "approval_policy": "Auto-approved only after product schema, allowed-theme, polarity, and exact-evidence validation.",
        "records": {split: len(rows) for split, rows in outputs.items()},
        "excluded": len(errors), "service_groups": dict(service_groups),
        "product_split_leakage": leakage,
    }
    if leakage:
        raise ValueError("product leakage detected between SFT splits")
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({"output_dir": str(args.output_dir), "approved_csv": str(args.approved_csv), **manifest}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
