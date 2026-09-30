from scripts.import_encoder_reviews_v2 import parse_bool, reviewed_record
from qna_review_model.encoder_data import EncoderTrainingRecord
from qna_review_model.encoder_review import load_review_rows


def sample():
    return EncoderTrainingRecord.model_validate({
        "taxonomyVersion": "theme-v1.0.0", "itemId": "q1", "productGroupId": "p1", "sourceType": "qna",
        "category": {"level1Id": "c1", "level1Name": "식품", "level2Id": "c2", "level2Name": "음료", "path": "식품 > 음료"},
        "text": "교환되나요?", "usable": True,
        "labels": {"qnaPrimaryTheme": {"themeId": "RETURN_EXCHANGE", "categoryThemeId": "ct1"}},
        "annotation": {"grade": "silver", "source": "test", "sourceSplit": "test", "modelConfidence": "high", "validationStatus": "valid", "reviewerStatus": "pending", "sampleWeight": 1.0},
    })


def test_parse_bool_and_unusable_qna_normalization():
    assert parse_bool("true") is True
    allowed = {
        "RETURN_EXCHANGE": {"categoryThemeId": "ct1"},
        "OTHER": {"categoryThemeId": "ct2"},
        "UNUSABLE": {"categoryThemeId": "ct3"},
    }
    row = {
        "product_id": "p1", "category_path": "식품 > 음료", "record_type": "qna", "text": "교환되나요?",
        "usable": "false", "qna_primary_theme_id": "RETURN_EXCHANGE", "qna_secondary_theme_ids_json": "[]",
        "reviewer_status": "corrected",
    }
    result = reviewed_record(sample(), row, allowed)
    assert result.usable is False
    assert result.labels.qnaPrimaryTheme.themeId == "UNUSABLE"
    assert result.annotation.grade == "gold"


def test_completed_review_wins_over_duplicate_pending_assignment(tmp_path):
    completed = tmp_path / "completed.csv"
    pending = tmp_path / "pending.csv"
    completed.write_text("annotation_id,reviewer_status\nq1,approved\n", encoding="utf-8")
    pending.write_text("annotation_id,reviewer_status\nq1,pending\n", encoding="utf-8")

    rows, errors = load_review_rows([completed, pending])

    assert rows["q1"]["reviewer_status"] == "approved"
    assert errors == []
