from palda_runtime.models import _sanitize_verification, _strongest_eligible_sentiment


CATALOG = [
    {"themeId": "COLOR", "reviewName": "색상", "qnaName": "색상 문의"},
    {"themeId": "SIZE", "reviewName": "사이즈", "qnaName": "사이즈 문의"},
    {"themeId": "OTHER", "reviewName": "기타", "qnaName": "기타 문의"},
    {"themeId": "UNUSABLE", "reviewName": "분류 불가", "qnaName": "분류 불가"},
]


def test_strongest_eligible_sentiment_is_exclusive() -> None:
    assert _strongest_eligible_sentiment(
        [0.8, 0.7, 0.1], [0.5, 0.5, 0.5]
    ) == (0, 0.8)
    assert _strongest_eligible_sentiment(
        [0.4, 0.7, 0.1], [0.5, 0.8, 0.5]
    ) is None


def test_review_verification_filters_invalid_and_duplicate_labels() -> None:
    original = {"id": "r1", "usable": True, "labels": []}
    result = _sanitize_verification(
        "review", {"usable": True, "labels": [
            {"themeId": "COLOR", "sentiment": "positive"},
            {"themeId": "COLOR", "sentiment": "negative"},
            {"themeId": "UNKNOWN", "sentiment": "positive"},
        ]}, original, CATALOG,
    )
    assert result["labels"] == [{
        "themeId": "COLOR", "theme": "색상", "sentiment": "positive", "confidence": 1.0,
    }]


def test_qna_verification_has_one_primary_and_distinct_secondaries() -> None:
    original = {"id": "q1", "usable": True, "primaryTheme": {"themeId": "OTHER"}}
    result = _sanitize_verification(
        "qna", {"usable": True, "primaryThemeId": "SIZE",
                "secondaryThemeIds": ["SIZE", "COLOR", "UNKNOWN"]},
        original, CATALOG,
    )
    assert result["primaryTheme"]["themeId"] == "SIZE"
    assert [item["themeId"] for item in result["secondaryThemes"]] == ["COLOR"]


def test_malformed_verification_falls_back_to_encoder_prediction() -> None:
    original = {"id": "r1", "usable": True, "labels": [{"themeId": "COLOR"}]}
    assert _sanitize_verification("review", {}, original, CATALOG) is original
