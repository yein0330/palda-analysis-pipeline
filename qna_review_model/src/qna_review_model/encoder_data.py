"""Schemas and deterministic helpers for PALDA Encoder training data v1."""

from __future__ import annotations

import hashlib
from typing import Literal

from pydantic import Field, model_validator

from .schemas import StrictModel


Sentiment = Literal["positive", "negative", "neutral"]


class EncoderCategory(StrictModel):
    level1Id: str
    level1Name: str
    level2Id: str
    level2Name: str
    path: str


class EncoderReviewLabel(StrictModel):
    themeId: str
    categoryThemeId: str
    sentiment: Sentiment


class EncoderQnaTheme(StrictModel):
    themeId: str
    categoryThemeId: str


class EncoderTargets(StrictModel):
    reviewLabels: list[EncoderReviewLabel] = Field(default_factory=list)
    qnaPrimaryTheme: EncoderQnaTheme | None = None
    qnaSecondaryThemes: list[EncoderQnaTheme] = Field(default_factory=list)


class EncoderAnnotationMetadata(StrictModel):
    grade: Literal["gold", "silver"]
    source: str
    sourceSplit: Literal["train", "validation", "test"]
    modelConfidence: Literal["low", "medium", "high"]
    validationStatus: Literal["valid", "needs_review"]
    reviewerStatus: Literal["pending", "approved", "corrected", "rejected"]
    sampleWeight: float = Field(gt=0, le=1)


class EncoderTrainingRecord(StrictModel):
    schemaVersion: Literal["encoder-train-v1"] = "encoder-train-v1"
    taxonomyVersion: str
    itemId: str
    productGroupId: str
    sourceType: Literal["review", "qna"]
    category: EncoderCategory
    text: str = Field(min_length=1)
    createdAt: str | None = None
    usable: bool
    labels: EncoderTargets
    annotation: EncoderAnnotationMetadata

    @model_validator(mode="after")
    def task_consistency(self) -> "EncoderTrainingRecord":
        if self.sourceType == "review":
            if self.labels.qnaPrimaryTheme is not None or self.labels.qnaSecondaryThemes:
                raise ValueError("Review records cannot contain Q&A targets")
        else:
            if self.labels.reviewLabels:
                raise ValueError("Q&A records cannot contain Review targets")
            if self.usable and self.labels.qnaPrimaryTheme is None:
                raise ValueError("Usable Q&A requires qnaPrimaryTheme")
        return self


def stable_id(prefix: str, value: str, length: int = 16) -> str:
    """Return a reproducible, path-safe identifier without leaking path text."""
    digest = hashlib.sha256(value.encode("utf-8")).hexdigest()[:length]
    return f"{prefix}_{digest}"


CANONICAL_ALIASES = {
    "value": "PRICE",
    "price": "PRICE",
    "cost": "PRICE",
    "refund": "RETURN_EXCHANGE",
    "returns_refunds": "RETURN_EXCHANGE",
    "service": "CUSTOMER_SUPPORT",
    "cs": "CUSTOMER_SUPPORT",
    "delivery_speed": "DELIVERY",
    "delivery_reliability": "DELIVERY",
    "delivery_staff": "DELIVERY",
    "customer_support": "CUSTOMER_SUPPORT",
    "packaging": "PACKAGING",
    "installation_service": "INSTALLATION_SERVICE",
    "after_sales": "AFTER_SALES",
}


def canonical_theme_id(theme_id: str) -> str:
    normalized = theme_id.strip().lower()
    return CANONICAL_ALIASES.get(normalized, normalized.upper())


def category_theme_id(cat2_path: str, canonical_id: str) -> str:
    return stable_id("ct", f"{cat2_path}\n{canonical_id}")
