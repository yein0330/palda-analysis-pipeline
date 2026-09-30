import pytest

from palda_runtime.aggregation import (
    _fallback_selling_points, _materialize_selling_points, _selling_point_type, build_report_input,
    materialize_grounded_report,
)
from palda_runtime.schemas import AnalysisRequest
from palda_runtime.models import _exact_quote_id, _fallback_summary, _parse_json_or_empty


@pytest.mark.parametrize("kind,sentiment,evidence", [
    ("core", "positive", "r-positive"), ("hidden", "positive", "r-positive"),
    ("barrier", "negative", "r-negative"), ("pain", "negative", "r-negative"),
])
def test_selling_points_attach_matching_sentiment_and_qna_evidence(kind, sentiment, evidence):
    report_input = {
        "reviewGroups": [
            {"themeId": "QUALITY", "sentiment": value, "mentionCount": count,
             "evidence": [{"id": f"r-{value}"}]}
            for value, count in [("positive", 7), ("negative", 2)]
        ],
        "qnaGroups": [{"themeId": "QUALITY", "evidence": [{"id": "q1"}]}],
    }
    result = _materialize_selling_points([
        {"type": kind, "text": "품질 관련 후보", "evidence_theme_ids": ["QUALITY"]}
    ], report_input)
    assert result[0]["reviewThemes"] == [f"{sentiment}/QUALITY"]
    assert result[0]["qnaThemes"] == ["QUALITY"]
    assert result[0]["evidenceIds"] == [evidence, "q1"]


def test_primary_qna_ratios_and_backend_quotes_are_grounded() -> None:
    request = {
        "analysisId": "a1", "taxonomyVersion": "theme-v1.0.0",
        "product": {"productId": "p1", "categoryPath": "디지털/가전 > 음향가전"},
        "reviewSample": {"maxItems": 5000, "items": [
            {"id": "r1", "text": "연결은 편하지만 가격은 비싸요."},
        ]},
        "qnaSample": {"maxItems": 5000, "items": [
            {"id": "q1", "text": "금요일 주문하면 언제 오나요?"},
            {"id": "q2", "text": "착용 후 아프면 교환되나요?"},
        ]},
    }
    predictions = [
        {"id": "r1", "usable": True, "labels": [
            {"themeId": "CONVENIENCE", "sentiment": "positive", "confidence": .9},
            {"themeId": "PRICE", "sentiment": "negative", "confidence": .8},
        ]},
        {"id": "q1", "usable": True,
         "primaryTheme": {"themeId": "DELIVERY", "confidence": .9},
         "secondaryThemes": [{"themeId": "RETURN_EXCHANGE", "confidence": .7}]},
        {"id": "q2", "usable": True,
         "primaryTheme": {"themeId": "RETURN_EXCHANGE", "confidence": .8}},
    ]
    report_input = build_report_input(request, predictions)
    assert sum(group["ratio"] for group in report_input["qnaGroups"]) == 1.0
    assert {group["primaryCount"] for group in report_input["qnaGroups"]} == {1}
    decoder = {
        "reviewAnalysis": {"positiveThemes": [{
            "themeId": "CONVENIENCE", "quoteEvidenceId": "r1",
            "summary": "연결 편의성이 장점입니다."
        }], "negativeThemes": []},
        "qnaAnalysis": {"themes": [{
            "themeId": "DELIVERY", "questionEvidenceId": "q1",
            "summary": "배송일 문의입니다.", "ratio": 0.99, "primaryCount": 999,
        }]},
        "sellingPoints": [{
            "title": "편의성", "message": "연결 편의성을 소구점 후보로 제안합니다.",
            "evidence_theme_ids": ["CONVENIENCE"],
        }],
    }
    report = materialize_grounded_report(report_input, decoder)
    assert report["reviewAnalysis"]["positiveThemes"][0]["quote"] == request["reviewSample"]["items"][0]["text"]
    assert report["qnaAnalysis"]["themes"][0]["question"] == request["qnaSample"]["items"][0]["text"]
    assert report["qnaAnalysis"]["themes"][0]["ratio"] == 0.5
    assert report["qnaAnalysis"]["themes"][0]["primaryCount"] == 1
    assert report["sellingPoints"][0]["evidenceIds"] == ["r1"]
    assert report["sellingPoints"][0]["text"] == "연결 편의성을 소구점 후보로 제안합니다."
    assert report["sample"]["qna"]["sampleSize"] == 2
    assert report["sample"]["qna"]["collectionOrder"] == "newest"


def test_report_input_ranks_and_caps_display_groups() -> None:
    request = {
        "analysisId": "a1", "taxonomyVersion": "theme-v1.0.0",
        "product": {"productId": "p1", "categoryPath": "가전 > 이어폰"},
        "reviewSample": {"maxItems": 5000, "items": [
            {"id": f"r{i}", "text": f"리뷰 {i}"} for i in range(4)
        ]},
        "qnaSample": {"maxItems": 5000, "items": [
            {"id": f"q{i}", "text": f"질문 {i}"} for i in range(5)
        ]},
    }
    predictions = [
        {"id": f"r{i}", "usable": True, "labels": [
            {"themeId": f"R{i}", "sentiment": "positive", "confidence": 0.9},
            {"themeId": "NEUTRAL", "sentiment": "neutral", "confidence": 0.99},
        ]} for i in range(4)
    ] + [
        {"id": f"q{i}", "usable": True,
         "primaryTheme": {"themeId": f"Q{i}", "confidence": 0.9}}
        for i in range(5)
    ]

    report = build_report_input(
        request, predictions, review_group_limit_per_sentiment=2, qna_group_limit=3
    )

    assert len(report["reviewGroups"]) == 2
    assert all(group["sentiment"] != "neutral" for group in report["reviewGroups"])
    assert len(report["qnaGroups"]) == 3
    assert sum(group["ratio"] for group in report["qnaGroups"]) == pytest.approx(0.6)


def test_api_request_rejects_cross_source_duplicate_ids() -> None:
    request = {
        "product": {"productId": "p1", "categoryPath": "가전 > 이어폰"},
        "reviewSample": {"items": [{"id": "same", "text": "좋아요"}]},
        "qnaSample": {"items": [{"id": "same", "text": "언제 오나요?"}]},
    }
    try:
        AnalysisRequest.model_validate(request)
    except ValueError as exc:
        assert "unique across Review and Q&A" in str(exc)
    else:
        raise AssertionError("cross-source duplicate must be rejected")


def test_decoder_typo_falls_back_to_backend_ranked_evidence() -> None:
    evidence = [{"id": "q1", "text": "설정은 어떻케 하나요?"}]
    assert _exact_quote_id("설정은 어떻게 하나요?", evidence) == "q1"


def test_decoder_cannot_reassign_evidence_to_another_theme() -> None:
    report_input = {
        "analysisId": "a1", "sample": {
            "reviewSampleSize": 1, "reviewUsableSize": 1, "reviewMaxItems": 5000,
            "qnaSampleSize": 0, "qnaUsableSize": 0, "qnaMaxItems": 5000,
        }, "_evidenceLookup": {"r1": "가격이 좋아요"},
        "reviewGroups": [{
            "themeId": "PRICE", "displayTheme": "가격", "sentiment": "positive",
            "mentionCount": 1, "evidence": [{"id": "r1", "text": "가격이 좋아요"}],
        }],
        "qnaGroups": [],
    }
    report = materialize_grounded_report(report_input, {
        "reviewAnalysis": {"positiveThemes": [{
            "themeId": "QUALITY", "quoteEvidenceId": "r1", "summary": "품질이 좋습니다."
        }], "negativeThemes": []},
        "qnaAnalysis": {"themes": []}, "sellingPoints": [],
    })
    assert report["reviewAnalysis"]["positiveThemes"] == []


def test_legacy_selling_point_titles_map_to_service_quadrants() -> None:
    assert _selling_point_type({"title": "핵심 소구점"}) == "core"
    assert _selling_point_type({"title": "숨은 강점"}) == "hidden"
    assert _selling_point_type({"title": "구매 장벽"}) == "barrier"
    assert _selling_point_type({"title": "잠재 Pain Point"}) == "pain"


def test_invalid_decoder_json_has_grounded_fallback() -> None:
    assert _parse_json_or_empty("not-json") == {}
    assert _fallback_summary({"displayTheme": "배송 문의"}, "qna") == "배송 문의가 확인됩니다."


def test_selling_point_fallback_uses_cross_stage_quadrants() -> None:
    report = {
        "reviewGroups": [{
            "themeId": "QUALITY", "displayTheme": "품질", "sentiment": "negative", "mentionCount": 3,
        }],
        "qnaGroups": [{"themeId": "QUALITY", "primaryCount": 2}],
    }
    point = _fallback_selling_points(report)[0]
    assert point["type"] == "barrier"
    assert point["evidence_theme_ids"] == ["QUALITY"]
