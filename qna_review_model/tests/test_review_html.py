from scripts.build_review_html import simplify_service_mentions


def test_service_mentions_collapse_to_three_groups():
    result = simplify_service_mentions([
        {"service_type": "delivery_speed", "polarity": "positive", "evidence_quote": "배송이 빨라요"},
        {"service_type": "delivery_staff", "polarity": "negative", "evidence_quote": "기사님은 불친절해요"},
        {"service_type": "returns_refunds", "polarity": "negative", "evidence_quote": "환불이 늦어요"},
    ])
    assert result == [
        {"service_group": "delivery", "polarity": "mixed", "evidence_quote": "배송이 빨라요"},
        {"service_group": "support", "polarity": "negative", "evidence_quote": "환불이 늦어요"},
    ]
