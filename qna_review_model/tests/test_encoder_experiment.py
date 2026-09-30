from qna_review_model.encoder_data import EncoderTrainingRecord
from scripts.build_encoder_experiment_v1 import clean_train_record, diverse_pick


def record(item_id="r1", source_type="review", text="가격이 저렴해요", split="train"):
    labels = {
        "reviewLabels": [{"themeId": "PRICE", "categoryThemeId": "ct1", "sentiment": "positive"}]
    }
    if source_type == "qna":
        labels = {"qnaPrimaryTheme": {"themeId": "PRICE", "categoryThemeId": "ct1"}}
    return EncoderTrainingRecord.model_validate({
        "taxonomyVersion": "theme-v1.0.0", "itemId": item_id, "productGroupId": f"p-{item_id}",
        "sourceType": source_type,
        "category": {"level1Id": "c1", "level1Name": "식품", "level2Id": "c2", "level2Name": "음료", "path": "식품 > 음료"},
        "text": text, "usable": True, "labels": labels,
        "annotation": {"grade": "silver", "source": "test", "sourceSplit": split, "modelConfidence": "high", "validationStatus": "valid", "reviewerStatus": "pending", "sampleWeight": 1.0},
    })


def test_clean_train_excludes_unsupported_only_price_label():
    cleaned, reasons = clean_train_record(record(text="디자인이 예뻐요"))
    assert cleaned is None
    assert "unsupported_review_theme:PRICE" in reasons


def test_clean_train_keeps_grounded_price_label():
    cleaned, reasons = clean_train_record(record())
    assert cleaned is not None
    assert reasons == []


def test_approved_gold_survives_lexical_filter_without_relabeling():
    row = record(text="부담 없는 금액대라 재구매해요")
    row = row.model_copy(update={"text": "부담 없어서 재구매해요", "annotation": row.annotation.model_copy(
        update={"grade": "gold", "reviewerStatus": "approved"})})
    cleaned, reasons = clean_train_record(row)
    assert cleaned == row
    assert reasons == ["gold_lexical_audit_only:PRICE"]


def test_corrected_gold_qna_survives_lexical_filter():
    row = record(source_type="qna", text="두 개면 얼마인가요?")
    row = row.model_copy(update={"annotation": row.annotation.model_copy(
        update={"grade": "gold", "reviewerStatus": "corrected"})})
    cleaned, reasons = clean_train_record(row)
    assert cleaned == row
    assert reasons == ["gold_lexical_audit_only:PRICE"]


def test_silver_price_recognizes_korean_inflection():
    for text in ("싼 맛에 쓰기 좋네요", "싼 이유임"):
        cleaned, reasons = clean_train_record(record(text=text))
        assert cleaned is not None
        assert reasons == []


def test_rejected_and_unreviewed_gold_cannot_enter_training():
    for grade, status, reason in [("gold", "pending", "gold_not_reviewed"),
                                  ("gold", "rejected", "reviewer_rejected"),
                                  ("silver", "rejected", "reviewer_rejected")]:
        row = record()
        row = row.model_copy(update={"annotation": row.annotation.model_copy(
            update={"grade": grade, "reviewerStatus": status})})
        cleaned, reasons = clean_train_record(row)
        assert cleaned is None
        assert reasons == [reason]


def test_gold_still_requires_qna_primary_for_training():
    row = record(source_type="qna")
    row = row.model_copy(update={
        "usable": False,
        "labels": row.labels.model_copy(update={"qnaPrimaryTheme": None}),
        "annotation": row.annotation.model_copy(update={"grade": "gold", "reviewerStatus": "approved"}),
    })
    cleaned, reasons = clean_train_record(row)
    assert cleaned is None
    assert reasons == ["missing_qna_primary"]


def test_clean_train_excludes_punctuation_only_text():
    cleaned, reasons = clean_train_record(record(text="........"))
    assert cleaned is None
    assert reasons == ["no_lexical_content"]


def test_diverse_pick_respects_requested_size():
    rows = [record(item_id=f"r{i}", split="validation", source_type="review" if i % 2 else "qna") for i in range(20)]
    assert len(diverse_pick(rows, 10, seed=7)) == 10


def test_diverse_pick_forces_unusable_qna_into_eval():
    rows = [record(item_id=f"r{i}", split="validation", source_type="qna") for i in range(20)]
    special = rows[-1].model_copy(update={
        "usable": False,
        "labels": rows[-1].labels.model_copy(update={
            "qnaPrimaryTheme": rows[-1].labels.qnaPrimaryTheme.model_copy(update={"themeId": "UNUSABLE"})
        }),
    })
    rows[-1] = special
    picked = diverse_pick(rows, 5, seed=7)
    assert any(row.labels.qnaPrimaryTheme.themeId == "UNUSABLE" for row in picked)
