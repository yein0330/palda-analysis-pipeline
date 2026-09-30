"""Reusable reviewed-CSV parsing for Encoder datasets."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

from .encoder_data import (
    EncoderAnnotationMetadata,
    EncoderQnaTheme,
    EncoderReviewLabel,
    EncoderTargets,
    EncoderTrainingRecord,
)


def parse_bool(value: object) -> bool:
    normalized = str(value or "").strip().lower()
    if normalized in {"true", "1", "yes", "y"}:
        return True
    if normalized in {"false", "0", "no", "n"}:
        return False
    raise ValueError(f"invalid boolean: {value!r}")


def load_review_rows(paths: list[Path]) -> tuple[dict[str, dict[str, str]], list[dict[str, str]]]:
    rows: dict[str, dict[str, str]] = {}
    errors = []
    completed = {"approved", "corrected"}
    for path in paths:
        with path.open(encoding="utf-8-sig", newline="") as handle:
            for row in csv.DictReader(handle):
                item_id = str(row.get("annotation_id") or "").strip()
                if item_id in rows:
                    old_done = str(rows[item_id].get("reviewer_status") or "").strip().lower() in completed
                    new_done = str(row.get("reviewer_status") or "").strip().lower() in completed
                    if old_done != new_done:
                        if new_done:
                            rows[item_id] = row
                        continue
                    errors.append({"annotation_id": item_id, "error": f"duplicate reviewed row in {path}"})
                    continue
                rows[item_id] = row
    return rows, errors


def reviewed_record(
    base: EncoderTrainingRecord,
    row: dict[str, str],
    allowed: dict[str, dict[str, Any]],
) -> EncoderTrainingRecord:
    if str(row.get("product_id") or "") != base.productGroupId:
        raise ValueError("product_id changed")
    if str(row.get("category_path") or "") != base.category.path:
        raise ValueError("category_path changed")
    if str(row.get("record_type") or "") != base.sourceType:
        raise ValueError("record_type changed")
    if str(row.get("text") or "") != base.text:
        raise ValueError("text changed")
    usable = parse_bool(row.get("usable"))
    if base.sourceType == "review":
        raw_labels = json.loads(str(row.get("review_labels_json") or "[]"))
        if not isinstance(raw_labels, list):
            raise ValueError("review_labels_json must be an array")
        review_labels = []
        seen = set()
        for value in raw_labels:
            theme_id = str(value.get("themeId") or "")
            sentiment = str(value.get("sentiment") or "")
            if theme_id not in allowed or theme_id in {"OTHER", "UNUSABLE"}:
                raise ValueError(f"Review theme is not allowed: {theme_id}")
            pair = (theme_id, sentiment)
            if pair in seen:
                continue
            seen.add(pair)
            review_labels.append(EncoderReviewLabel(
                themeId=theme_id,
                categoryThemeId=allowed[theme_id]["categoryThemeId"],
                sentiment=sentiment,
            ))
        labels = EncoderTargets(reviewLabels=review_labels)
    else:
        primary_id = str(row.get("qna_primary_theme_id") or "").strip()
        if not usable:
            primary_id = "UNUSABLE"
        elif not primary_id or primary_id == "UNUSABLE":
            primary_id = "OTHER"
        if primary_id not in allowed:
            raise ValueError(f"Q&A primary theme is not allowed: {primary_id}")
        raw_secondary = json.loads(str(row.get("qna_secondary_theme_ids_json") or "[]"))
        if not isinstance(raw_secondary, list):
            raise ValueError("qna_secondary_theme_ids_json must be an array")
        secondary = []
        seen = {primary_id}
        if usable:
            for value in raw_secondary:
                theme_id = str(value)
                if theme_id not in allowed or theme_id in {"OTHER", "UNUSABLE"}:
                    raise ValueError(f"Q&A secondary theme is not allowed: {theme_id}")
                if theme_id in seen:
                    continue
                seen.add(theme_id)
                secondary.append(EncoderQnaTheme(
                    themeId=theme_id,
                    categoryThemeId=allowed[theme_id]["categoryThemeId"],
                ))
        labels = EncoderTargets(
            qnaPrimaryTheme=EncoderQnaTheme(
                themeId=primary_id,
                categoryThemeId=allowed[primary_id]["categoryThemeId"],
            ),
            qnaSecondaryThemes=secondary,
        )
    reviewer_status = str(row.get("reviewer_status") or "").strip().lower()
    return base.model_copy(update={
        "usable": usable,
        "labels": labels,
        "annotation": EncoderAnnotationMetadata(
            grade="gold",
            source="human-review-v1",
            sourceSplit=base.annotation.sourceSplit,
            modelConfidence=base.annotation.modelConfidence,
            validationStatus=base.annotation.validationStatus,
            reviewerStatus=reviewer_status,
            sampleWeight=1.0,
        ),
    })
