from datetime import datetime

from qna_review_model.schemas import QNARecord, ReviewRecord, ThemeDefinition
from qna_review_model.training_data import _mentions


THEMES = [ThemeDefinition(theme_id="assembly", label="조립", keywords=["조립"])]


def test_bootstrap_review_label_uses_direct_theme_and_score_polarity():
    record = ReviewRecord(
        record_id="r1", product_id="p1", text="조립이 너무 어려워요", score=1,
        created_at=datetime(2026, 1, 1), source="naver_catalog",
    )
    assert _mentions(record, THEMES) == [{"theme_id": "assembly", "polarity": "negative"}]


def test_bootstrap_qna_label_is_concern():
    record = QNARecord(record_id="q1", product_id="p1", question="조립 서비스가 있나요?")
    assert _mentions(record, THEMES) == [{"theme_id": "assembly", "polarity": "concern"}]
