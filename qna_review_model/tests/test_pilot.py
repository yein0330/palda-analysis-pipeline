from datetime import datetime

from qna_review_model.pilot import _Reservoir, _assign_splits, _select_reviews
from qna_review_model.schemas import ReviewRecord


def review(index: int, score: int) -> ReviewRecord:
    return ReviewRecord(
        record_id=f"r{index}", product_id="p1", text=f"review {index}", score=score,
        created_at=datetime(2026, 1, (index % 28) + 1), source="naver_catalog",
    )


def test_reservoir_is_bounded_and_order_independent():
    first = _Reservoir(10, 7)
    second = _Reservoir(10, 7)
    records = [review(index, 5) for index in range(100)]
    for item in records:
        first.add(item)
    for item in reversed(records):
        second.add(item)
    assert len(first.records()) == 10
    assert [x.record_id for x in first.records()] == [x.record_id for x in second.records()]


def test_review_selection_uses_score_buckets_and_cap():
    pools = {name: _Reservoir(50, 3) for name in ("positive", "negative", "neutral", "unknown")}
    for index in range(40):
        pools["positive"].add(review(index, 5))
    for index in range(40, 70):
        pools["negative"].add(review(index, 1))
    for index in range(70, 90):
        pools["neutral"].add(review(index, 3))
    selected = _select_reviews(pools, 20)
    assert len(selected) == 20
    assert {item.score for item in selected} == {1, 3, 5}


def test_product_level_split_is_exact_and_disjoint():
    result = _assign_splits([f"p{x}" for x in range(200)], 11)
    assert list(result.values()).count("train") == 160
    assert list(result.values()).count("validation") == 20
    assert list(result.values()).count("test") == 20
    assert len(result) == 200
