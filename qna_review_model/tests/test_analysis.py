from qna_review_model.analysis import add_category_relative_insights, analyze_product
from qna_review_model.schemas import ProductContext, QNARecord, Quadrant, ReviewRecord, ThemeDefinition
from qna_review_model.themes import ThemeDictionary


THEMES = [ThemeDefinition(theme_id="comfort", label="착용감", keywords=["편안", "불편"])]


def product(product_id="P1"):
    return ProductContext(
        product_id=product_id, name="신발", category_path="패션잡화 > 신발",
        category_levels=["패션잡화", "신발"],
    )


def test_theme_dictionary_inherits_and_overrides():
    dictionary = ThemeDictionary({
        "global": [{"theme_id": "quality", "label": "품질", "keywords": ["품질"]}],
        "cat1": {"패션잡화": [{"theme_id": "quality", "label": "마감", "keywords": ["마감"]}]},
        "cat2": {}, "leaf": {},
    })
    themes = dictionary.resolve(product())
    assert len(themes) == 1
    assert themes[0].label == "마감"
    assert themes[0].source_level == "cat1"


def test_product_signal_is_evidence_grounded():
    records = [
        QNARecord(record_id="q1", product_id="P1", question="오래 신어도 편안한가요?"),
        ReviewRecord(record_id="r1", product_id="P1", text="아주 편안합니다", score=5, source="naver_catalog"),
        ReviewRecord(record_id="r2", product_id="P1", text="조금 불편합니다", score=1, source="naver_catalog"),
    ]
    result = analyze_product(product(), THEMES, records)
    signal = result.signals[0]
    assert signal.qna_mentions == 1
    assert signal.review_mentions == 2
    assert signal.satisfaction == 0.5
    assert {e.record_id for e in signal.evidence} == {"q1", "r1", "r2"}


def test_category_relative_quadrants():
    analyses = []
    values = [(1, 5), (0, 5), (1, 1), (0, 1)]
    for index, (qna_hit, score) in enumerate(values):
        records = [ReviewRecord(
            record_id=f"r{index}", product_id=f"P{index}", text="편안" if score == 5 else "불편",
            score=score, source="naver_catalog",
        )]
        if qna_hit:
            records.append(QNARecord(record_id=f"q{index}", product_id=f"P{index}", question="편안한가요?"))
        else:
            records.append(QNARecord(record_id=f"q{index}", product_id=f"P{index}", question="배송되나요?"))
        analyses.append(analyze_product(product(f"P{index}"), THEMES, records))
    add_category_relative_insights(analyses)
    assert [a.insights[0].classification for a in analyses] == [
        Quadrant.CORE_SELLING_POINT, Quadrant.HIDDEN_SELLING_POINT,
        Quadrant.PURCHASE_BARRIER, Quadrant.POTENTIAL_PAIN_POINT,
    ]
    assert analyses[2].recommended_actions[0].evidence_ids
