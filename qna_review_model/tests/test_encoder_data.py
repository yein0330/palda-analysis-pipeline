import pytest

from qna_review_model.encoder_data import (
    EncoderTrainingRecord,
    canonical_theme_id,
    category_theme_id,
)
from scripts.build_encoder_dataset_v2 import RETURN_PRIMARY_PATTERN, clean_text


def base_record(source_type="review"):
    return {
        "taxonomyVersion": "theme-v1.0.0",
        "itemId": "r1",
        "productGroupId": "p1",
        "sourceType": source_type,
        "category": {
            "level1Id": "cat1_a", "level1Name": "가구/인테리어",
            "level2Id": "cat2_b", "level2Name": "거실가구",
            "path": "가구/인테리어 > 거실가구 > 소파",
        },
        "text": "가격은 좋지만 배송은 늦어요",
        "usable": True,
        "labels": {"reviewLabels": []},
        "annotation": {
            "grade": "silver", "source": "test", "sourceSplit": "train",
            "modelConfidence": "high", "validationStatus": "valid",
            "reviewerStatus": "pending", "sampleWeight": 1.0,
        },
    }


def test_canonical_alias_and_stable_category_theme_id():
    assert canonical_theme_id("value") == "PRICE"
    assert canonical_theme_id("returns_refunds") == "RETURN_EXCHANGE"
    assert category_theme_id("가구/인테리어 > 거실가구", "PRICE") == category_theme_id(
        "가구/인테리어 > 거실가구", "PRICE"
    )


def test_review_accepts_same_theme_with_two_sentiments():
    payload = base_record()
    payload["labels"]["reviewLabels"] = [
        {"themeId": "PRICE", "categoryThemeId": "ct_x", "sentiment": "positive"},
        {"themeId": "PRICE", "categoryThemeId": "ct_x", "sentiment": "negative"},
    ]
    record = EncoderTrainingRecord.model_validate(payload)
    assert len(record.labels.reviewLabels) == 2


def test_usable_qna_requires_primary_theme():
    payload = base_record("qna")
    with pytest.raises(ValueError, match="requires qnaPrimaryTheme"):
        EncoderTrainingRecord.model_validate(payload)


def test_text_cleanup_and_explicit_return_request_detection():
    assert clean_text("첫 줄<br>둘째 줄&amp;") == "첫 줄\n둘째 줄&"
    assert RETURN_PRIMARY_PATTERN.search("반품택배 신청 부탁드립니다")
    assert not RETURN_PRIMARY_PATTERN.search("고장이라 교환이나 반품을 해야 하는 상황이면")
