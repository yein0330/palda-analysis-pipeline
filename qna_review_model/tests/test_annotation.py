from qna_review_model.annotation import AnnotationQueueRow, DraftAnnotationRow, import_review_csv, normalize_draft_payload, validate_draft


def queue(record_type="review"):
    return AnnotationQueueRow(
        annotation_id="r1", split="train", product_id="p1", product_name="상품",
        category_path="가구/인테리어 > 거실가구", record_type=record_type,
        text="배송은 빠르지만 조립은 어려워요", score=3 if record_type == "review" else None,
        allowed_themes=[{"theme_id": "delivery", "label": "배송"}, {"theme_id": "assembly", "label": "조립/설치"}],
    )


def test_valid_multi_theme_review_draft():
    draft, errors = validate_draft(queue(), {
        "record_id": "r1", "qna_stage": None, "confidence": "high", "new_theme_candidates": [],
        "mentions": [
            {"theme_id": "delivery", "polarity": "positive", "evidence_quote": "배송은 빠르지만"},
            {"theme_id": "assembly", "polarity": "negative", "evidence_quote": "조립은 어려워요"},
        ],
    })
    assert draft is not None
    assert errors == []


def test_rejects_hallucinated_evidence_and_theme():
    _, errors = validate_draft(queue(), {
        "record_id": "r1", "qna_stage": None, "confidence": "low", "new_theme_candidates": [],
        "mentions": [{"theme_id": "price", "polarity": "negative", "evidence_quote": "너무 비싸요"}],
    })
    assert len(errors) == 2


def test_normalizes_source_known_fields_and_duplicate_qna_themes():
    source = queue("qna")
    payload = normalize_draft_payload(source, {
        "record_id": "r1", "qna_stage": "pre_purchase", "confidence": "medium", "new_theme_candidates": [],
        "mentions": [
            {"theme_id": "delivery", "polarity": "neutral", "evidence_quote": "배송은 빠르지만"},
            {"theme_id": "delivery", "polarity": "negative", "evidence_quote": "배송은 빠르지만"},
        ],
    })
    draft, errors = validate_draft(source, payload)
    assert errors == []
    assert draft is not None and draft.mentions[0].polarity == "concern"


def test_import_reviewed_csv_creates_only_approved_gold(tmp_path):
    source = queue()
    draft, errors = validate_draft(source, {
        "record_id": "r1", "qna_stage": None, "confidence": "high", "new_theme_candidates": [],
        "mentions": [{"theme_id": "delivery", "polarity": "positive", "evidence_quote": "배송은 빠르지만"}],
    })
    assert draft is not None and not errors
    drafts_path = tmp_path / "drafts.jsonl"
    drafts_path.write_text(DraftAnnotationRow(
        input=source, draft=draft, validation_status="valid", model="test", raw_output="{}",
    ).model_dump_json() + "\n")
    csv_path = tmp_path / "review.csv"
    csv_path.write_text(
        "annotation_id,reviewer_status,reviewer_notes\n"
        "r1,approved,checked\n", encoding="utf-8-sig",
    )
    result = import_review_csv(drafts_path, csv_path, tmp_path / "gold.jsonl", tmp_path / "errors.csv")
    assert result["gold"] == 1
    assert '"reviewer_status":"approved"' in (tmp_path / "gold.jsonl").read_text()
