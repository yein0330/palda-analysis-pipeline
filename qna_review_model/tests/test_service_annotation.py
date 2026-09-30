from qna_review_model.annotation import AnnotationQueueRow
from qna_review_model.service_annotation import normalize_service_payload, validate_service_draft


def source():
    return AnnotationQueueRow(
        annotation_id="r1", split="train", product_id="p1", product_name="상품", category_path="가구/인테리어",
        record_type="review", text="배송 기사님이 친절하고 설치도 꼼꼼했어요", score=5, allowed_themes=[],
    )


def test_valid_multi_service_draft():
    draft, errors = validate_service_draft(source(), {
        "record_id": "r1", "service_relevant": True, "confidence": "high",
        "mentions": [
            {"service_type": "delivery_staff", "polarity": "positive", "evidence_quote": "배송 기사님이 친절하고"},
            {"service_type": "installation", "polarity": "positive", "evidence_quote": "설치도 꼼꼼했어요"},
        ],
    })
    assert draft is not None and errors == []


def test_normalization_deduplicates_service_type():
    payload = normalize_service_payload(source(), {
        "record_id": "r1", "service_relevant": True, "confidence": "medium",
        "mentions": [
            {"service_type": "installation", "polarity": "positive", "evidence_quote": "설치도 꼼꼼했어요"},
            {"service_type": "installation", "polarity": "neutral", "evidence_quote": "설치도 꼼꼼했어요"},
        ],
    })
    assert [item["service_type"] for item in payload["mentions"]].count("installation") == 1


def test_normalization_removes_product_property_false_positive():
    row = source().model_copy(update={"text": "너무 얇고 쉽게 녹아요. 반품비가 9500원이에요"})
    payload = normalize_service_payload(row, {
        "record_id": "r1", "service_relevant": True, "confidence": "high",
        "mentions": [
            {"service_type": "packaging", "polarity": "negative", "evidence_quote": "너무 얇고 쉽게 녹아요"},
            {"service_type": "returns_refunds", "polarity": "negative", "evidence_quote": "반품비가 9500원이에요"},
        ],
    })
    assert [item["service_type"] for item in payload["mentions"]] == ["returns_refunds"]


def test_normalization_recovers_unanswered_customer_support():
    row = source().model_copy(update={"text": "문의글 남겼고 문자 보내라고 했으나 연락도 없고...."})
    payload = normalize_service_payload(row, {
        "record_id": "r1", "service_relevant": False, "confidence": "high", "mentions": [],
    })
    assert payload["service_relevant"] is True
    assert payload["mentions"][0]["service_type"] == "customer_support"
    assert payload["mentions"][0]["evidence_quote"] in row.text


def test_normalization_does_not_treat_self_installation_as_service():
    row = source().model_copy(update={"text": "타공 구멍이 안 맞아서 도어락 설치가 힘들었어요"})
    payload = normalize_service_payload(row, {
        "record_id": "r1", "service_relevant": True, "confidence": "medium",
        "mentions": [{"service_type": "installation", "polarity": "negative", "evidence_quote": row.text}],
    })
    assert payload["mentions"] == []
    assert payload["service_relevant"] is False


def test_normalization_does_not_treat_remodeling_repair_as_after_sales():
    row = source().model_copy(update={"text": "올리모델링 수리중이라 신혼집으로 주문했습니다"})
    payload = normalize_service_payload(row, {
        "record_id": "r1", "service_relevant": True, "confidence": "medium",
        "mentions": [{"service_type": "after_sales", "polarity": "negative", "evidence_quote": row.text}],
    })
    assert payload["mentions"] == []
