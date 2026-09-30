"""Canonical schemas for category-aware pre/post-purchase analysis."""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ProductContext(StrictModel):
    product_id: str
    name: str
    category_cid: str | None = None
    category_path: str | None = None
    category_levels: list[str] = Field(default_factory=list)
    brand: str | None = None
    price: int | None = Field(default=None, ge=0)
    options: list[str] = Field(default_factory=list)

    @property
    def cat1(self) -> str:
        if self.category_levels:
            return self.category_levels[0]
        if self.category_path:
            return self.category_path.split(">")[0].strip()
        return "기타"


class ReviewRecord(StrictModel):
    record_type: Literal["review"] = "review"
    record_id: str
    product_id: str
    text: str = Field(min_length=1)
    score: float | None = Field(default=None, ge=0, le=5)
    option: str | None = None
    created_at: datetime | None = None
    source: Literal["naver_brand", "naver_catalog"]


class QNARecord(StrictModel):
    record_type: Literal["qna"] = "qna"
    record_id: str
    product_id: str
    question: str = Field(min_length=1)
    created_at: datetime | None = None
    has_seller_answer: bool | None = None
    source: Literal["naver_brand"] = "naver_brand"


class ThemeDefinition(StrictModel):
    theme_id: str
    label: str
    keywords: list[str] = Field(default_factory=list)
    source_level: Literal["global", "cat1", "cat2", "leaf"] = "cat1"


class ThemeEvidence(StrictModel):
    record_id: str
    quote: str
    stage: Literal["pre_purchase", "post_purchase"]
    polarity: Literal["positive", "negative", "neutral", "concern"]


class ThemeSignal(StrictModel):
    theme_id: str
    theme: str
    qna_mentions: int = Field(ge=0)
    qna_share: float = Field(ge=0, le=1)
    review_mentions: int = Field(ge=0)
    positive_mentions: int = Field(ge=0)
    negative_mentions: int = Field(ge=0)
    neutral_mentions: int = Field(ge=0)
    satisfaction: float | None = Field(default=None, ge=0, le=1)
    evidence: list[ThemeEvidence] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_review_counts(self) -> "ThemeSignal":
        if self.positive_mentions + self.negative_mentions + self.neutral_mentions != self.review_mentions:
            raise ValueError("review polarity counts must equal review_mentions")
        return self


class Quadrant(str, Enum):
    CORE_SELLING_POINT = "core_selling_point"
    HIDDEN_SELLING_POINT = "hidden_selling_point"
    PURCHASE_BARRIER = "purchase_barrier"
    POTENTIAL_PAIN_POINT = "potential_pain_point"
    INSUFFICIENT = "insufficient_evidence"


class CrossStageInsight(StrictModel):
    theme_id: str
    theme: str
    classification: Quadrant
    qna_percentile: float | None = Field(default=None, ge=0, le=1)
    satisfaction_percentile: float | None = Field(default=None, ge=0, le=1)
    insight: str
    evidence: list[ThemeEvidence] = Field(default_factory=list)
    confidence: Literal["low", "medium", "high"]


class SellerAction(StrictModel):
    theme_id: str
    action: str
    rationale: str
    evidence_ids: list[str] = Field(min_length=1)


class ProductAnalysis(StrictModel):
    methodology: Literal["category_aware_cross_stage_v1"] = "category_aware_cross_stage_v1"
    product: ProductContext
    category_themes: list[ThemeDefinition]
    review_count: int = Field(ge=0)
    qna_count: int = Field(ge=0)
    signals: list[ThemeSignal]
    insights: list[CrossStageInsight] = Field(default_factory=list)
    recommended_actions: list[SellerAction] = Field(default_factory=list)
