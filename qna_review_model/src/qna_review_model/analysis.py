"""Deterministic baseline for category-aware cross-stage analysis."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable
from statistics import fmean

from .schemas import (
    CrossStageInsight, ProductAnalysis, ProductContext, QNARecord, Quadrant,
    ReviewRecord, SellerAction, ThemeDefinition, ThemeEvidence, ThemeSignal,
)


def _matches(text: str, theme: ThemeDefinition) -> bool:
    lowered = text.casefold()
    return any(keyword.casefold() in lowered for keyword in theme.keywords)


def _quote(text: str, limit: int = 240) -> str:
    compact = " ".join(text.split())
    return compact if len(compact) <= limit else compact[: limit - 1] + "…"


def analyze_product(
    product: ProductContext,
    themes: list[ThemeDefinition],
    records: Iterable[ReviewRecord | QNARecord],
    max_evidence_per_theme: int = 6,
) -> ProductAnalysis:
    reviews: list[ReviewRecord] = []
    qnas: list[QNARecord] = []
    for record in records:
        (reviews if isinstance(record, ReviewRecord) else qnas).append(record)
    signals: list[ThemeSignal] = []
    for theme in themes:
        qna_hits = [record for record in qnas if _matches(record.question, theme)]
        review_hits = [record for record in reviews if _matches(record.text, theme)]
        positive = sum(record.score is not None and record.score >= 4 for record in review_hits)
        negative = sum(record.score is not None and record.score <= 2 for record in review_hits)
        neutral = len(review_hits) - positive - negative
        evidence = [
            ThemeEvidence(record_id=r.record_id, quote=_quote(r.question), stage="pre_purchase", polarity="concern")
            for r in qna_hits
        ] + [
            ThemeEvidence(
                record_id=r.record_id,
                quote=_quote(r.text),
                stage="post_purchase",
                polarity="positive" if r.score is not None and r.score >= 4 else (
                    "negative" if r.score is not None and r.score <= 2 else "neutral"
                ),
            )
            for r in review_hits
        ]
        signals.append(ThemeSignal(
            theme_id=theme.theme_id, theme=theme.label,
            qna_mentions=len(qna_hits), qna_share=round(len(qna_hits) / len(qnas), 6) if qnas else 0,
            review_mentions=len(review_hits), positive_mentions=positive,
            negative_mentions=negative, neutral_mentions=neutral,
            satisfaction=round(positive / len(review_hits), 6) if review_hits else None,
            evidence=evidence[:max_evidence_per_theme],
        ))
    return ProductAnalysis(
        product=product, category_themes=themes, review_count=len(reviews),
        qna_count=len(qnas), signals=signals,
    )


def _percentile(value: float, population: list[float]) -> float | None:
    if len(population) < 2:
        return None
    below = sum(x < value for x in population)
    equal = sum(x == value for x in population)
    return round((below + 0.5 * equal) / len(population), 6)


def _classification(qna_pct: float | None, satisfaction_pct: float | None) -> Quadrant:
    if qna_pct is None or satisfaction_pct is None:
        return Quadrant.INSUFFICIENT
    qna_high, qna_low = qna_pct >= 0.75, qna_pct <= 0.25
    sat_high, sat_low = satisfaction_pct >= 0.75, satisfaction_pct <= 0.25
    if qna_high and sat_high:
        return Quadrant.CORE_SELLING_POINT
    if qna_low and sat_high:
        return Quadrant.HIDDEN_SELLING_POINT
    if qna_high and sat_low:
        return Quadrant.PURCHASE_BARRIER
    if qna_low and sat_low:
        return Quadrant.POTENTIAL_PAIN_POINT
    return Quadrant.INSUFFICIENT


_INSIGHT_TEXT = {
    Quadrant.CORE_SELLING_POINT: "구매 전 관심이 높고 구매 후 만족도도 높은 핵심 소구점입니다.",
    Quadrant.HIDDEN_SELLING_POINT: "구매 전 문의는 적지만 구매 후 만족이 확인되는 숨은 소구점입니다.",
    Quadrant.PURCHASE_BARRIER: "구매 전 우려가 높고 구매 후 불만도 확인되는 구매장벽입니다.",
    Quadrant.POTENTIAL_PAIN_POINT: "구매 전에는 잘 드러나지 않지만 구매 후 불만이 발생하는 잠재 문제입니다.",
    Quadrant.INSUFFICIENT: "카테고리 내 상대평가 또는 양 단계의 근거가 충분하지 않습니다.",
}


def add_category_relative_insights(analyses: list[ProductAnalysis]) -> list[ProductAnalysis]:
    populations: dict[tuple[str, str], dict[str, list[float]]] = defaultdict(
        lambda: {"qna": [], "satisfaction": []}
    )
    for analysis in analyses:
        for signal in analysis.signals:
            key = (analysis.product.cat1, signal.theme_id)
            populations[key]["qna"].append(signal.qna_share)
            if signal.satisfaction is not None:
                populations[key]["satisfaction"].append(signal.satisfaction)
    for analysis in analyses:
        insights: list[CrossStageInsight] = []
        actions: list[SellerAction] = []
        for signal in analysis.signals:
            population = populations[(analysis.product.cat1, signal.theme_id)]
            qna_pct = _percentile(signal.qna_share, population["qna"])
            sat_pct = _percentile(signal.satisfaction, population["satisfaction"]) if signal.satisfaction is not None else None
            classification = _classification(qna_pct, sat_pct)
            evidence = signal.evidence
            confidence = "high" if len(evidence) >= 5 else "medium" if len(evidence) >= 2 else "low"
            insights.append(CrossStageInsight(
                theme_id=signal.theme_id, theme=signal.theme, classification=classification,
                qna_percentile=qna_pct, satisfaction_percentile=sat_pct,
                insight=_INSIGHT_TEXT[classification], evidence=evidence, confidence=confidence,
            ))
            if classification in {Quadrant.PURCHASE_BARRIER, Quadrant.POTENTIAL_PAIN_POINT} and evidence:
                actions.append(SellerAction(
                    theme_id=signal.theme_id,
                    action=f"상세페이지와 FAQ의 '{signal.theme}' 정보를 보강하고 반복 불만을 제품 개선 항목으로 검토합니다.",
                    rationale=_INSIGHT_TEXT[classification],
                    evidence_ids=[item.record_id for item in evidence],
                ))
        analysis.insights = insights
        analysis.recommended_actions = actions
    return analyses


def category_signal_summary(analyses: list[ProductAnalysis]) -> dict[str, dict[str, float]]:
    values: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for analysis in analyses:
        for signal in analysis.signals:
            values[analysis.product.cat1][signal.theme_id].append(signal.qna_share)
    return {
        category: {theme: round(fmean(shares), 6) for theme, shares in themes.items()}
        for category, themes in values.items()
    }
