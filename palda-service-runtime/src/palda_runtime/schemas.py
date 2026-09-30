from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


class Item(BaseModel):
    model_config = ConfigDict(extra="forbid")
    id: str = Field(min_length=1, max_length=200)
    text: str = Field(min_length=1, max_length=20000)
    createdAt: str | None = None


class Sample(BaseModel):
    model_config = ConfigDict(extra="forbid")
    collectionOrder: Literal["newest"] = "newest"
    maxItems: int = Field(default=5000, ge=1, le=5000)
    items: list[Item] = Field(max_length=5000)

    @field_validator("items")
    @classmethod
    def unique_ids(cls, items: list[Item]) -> list[Item]:
        ids = [item.id for item in items]
        if len(ids) != len(set(ids)):
            raise ValueError("item IDs must be unique within a sample")
        return items


class Product(BaseModel):
    model_config = ConfigDict(extra="forbid")
    productId: str = Field(min_length=1, max_length=200)
    categoryId: str | None = None
    categoryPath: str = Field(min_length=1, max_length=1000)


class AnalysisRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    product: Product
    reviewSample: Sample
    qnaSample: Sample

    @field_validator("qnaSample")
    @classmethod
    def ids_do_not_cross_sources(cls, qna: Sample, info: object) -> Sample:
        data = getattr(info, "data", {})
        review = data.get("reviewSample")
        if review and {item.id for item in review.items} & {item.id for item in qna.items}:
            raise ValueError("item IDs must be unique across Review and Q&A")
        return qna


class JobCreated(BaseModel):
    analysisId: str
    status: Literal["queued"]
    progress: int = 0


class JobStatus(BaseModel):
    analysisId: str
    status: Literal["queued", "classifying", "aggregating", "generating", "validating", "completed", "partial", "failed"]
    progress: int = Field(ge=0, le=100)
    stages: dict[str, str] | None = None
    result: dict | None = None
    error: str | None = None


class ResultSampleStats(BaseModel):
    model_config = ConfigDict(extra="forbid")
    collectionOrder: Literal["newest"]
    sampleSize: int = Field(ge=0, le=5000)
    usableSize: int = Field(ge=0, le=5000)
    maxItems: int = Field(ge=1, le=5000)


class ReviewThemeResult(BaseModel):
    model_config = ConfigDict(extra="forbid")
    themeId: str
    displayTheme: str | None = None
    theme: str
    mentionCount: int = Field(ge=1)
    sentiment: Literal["positive", "negative"]
    quoteEvidenceId: str | None = None
    quote: str = Field(min_length=1)
    summary: str = Field(min_length=1)
    evidenceIds: list[str] = Field(min_length=1)
    fallbackUsed: bool | None = None


class QnaThemeResult(BaseModel):
    model_config = ConfigDict(extra="forbid")
    themeId: str
    displayTheme: str | None = None
    theme: str
    primaryCount: int = Field(ge=1)
    ratio: float = Field(ge=0, le=1)
    questionEvidenceId: str | None = None
    question: str = Field(min_length=1)
    summary: str = Field(min_length=1)
    evidenceIds: list[str] = Field(min_length=1)
    fallbackUsed: bool | None = None


class SellingPointResult(BaseModel):
    model_config = ConfigDict(extra="forbid")
    type: Literal["core", "hidden", "barrier", "pain"]
    text: str = Field(min_length=1)
    reviewThemes: list[str]
    qnaThemes: list[str]
    evidenceIds: list[str] = Field(min_length=1)
    confidence: str


class ResultVersions(BaseModel):
    model_config = ConfigDict(extra="allow")
    encoderModel: str
    decoderModel: str
    taxonomy: str
    calculation: str


class ResultSample(BaseModel):
    model_config = ConfigDict(extra="forbid")
    review: ResultSampleStats
    qna: ResultSampleStats


class ReviewAnalysisResult(BaseModel):
    model_config = ConfigDict(extra="forbid")
    positiveThemes: list[ReviewThemeResult]
    negativeThemes: list[ReviewThemeResult]


class QnaAnalysisResult(BaseModel):
    model_config = ConfigDict(extra="forbid")
    themes: list[QnaThemeResult]


class ReportOutput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    analysisId: str = Field(min_length=1)
    sample: ResultSample
    reviewAnalysis: ReviewAnalysisResult
    qnaAnalysis: QnaAnalysisResult
    sellingPoints: list[SellingPointResult] = Field(max_length=3)
    versions: ResultVersions
