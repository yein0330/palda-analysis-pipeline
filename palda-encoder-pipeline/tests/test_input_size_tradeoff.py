import importlib.util
from pathlib import Path

SPEC = importlib.util.spec_from_file_location("tradeoff", Path(__file__).parents[1] / "scripts/input_size_tradeoff.py")
m = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(m)


def product():
    return {"id": "p", "category": "A > B", "goldAnchor": [], "items": [
        {"id": str(i), "kind": "review" if i % 2 else "qna", "text": f"real text {i}"}
        for i in range(20)]}


def ids(request):
    return {r["id"] for key in ("reviewSample", "qnaSample") for r in request[key]["items"]}


def test_nested_reproducible_no_replication():
    p = product()
    small, large = m.sample_request(p, 5, 42), m.sample_request(p, 15, 42)
    assert ids(small) < ids(large)
    assert len(ids(large)) == 15
    assert small == m.sample_request(p, 5, 42)
    assert m.sample_request(p, 21, 42) is None
    assert small != m.sample_request(p, 5, 43)


def empty_result():
    return {"reviewAnalysis": {"positiveThemes": [], "negativeThemes": []},
            "qnaAnalysis": {"themes": []}, "sellingPoints": []}


def test_empty_quality_is_unknown_not_perfect():
    p = product()
    q = m.quality(m.sample_request(p, 5, 42), [], empty_result(), p)
    assert q["quote_exact_match"] is None
    assert q["gold_review_micro_f1"] is None
    assert q["summary_semantic_faithfulness"] is None
    assert q["gold_anchor_theme_coverage"] is None


def test_sparse_gold_and_exact_quote_are_separate():
    p = product()
    p["items"][1]["gold"] = {"tier": "gold", "usable": True,
                            "labels": [{"theme_id": "QUALITY", "sentiment": "positive"}]}
    p["goldAnchor"] = [{"kind": "review", **p["items"][1]["gold"]}]
    result = empty_result()
    result["reviewAnalysis"]["positiveThemes"] = [{"themeId": "QUALITY", "sentiment": "positive",
         "quoteEvidenceId": "1", "quote": "invented", "evidenceIds": ["1"], "fallbackUsed": False}]
    pred = [{"id": "1", "usable": True, "labels": [{"themeId": "WRONG", "sentiment": "positive"}]}]
    q = m.quality(m.sample_request(p, 20, 42), pred, result, p)
    assert q["gold_review_micro_f1"] == 0
    assert q["gold_review_rows"] == 1
    assert q["quote_exact_match"] == 0
    assert q["evidence_id_validity"] == 1
    assert q["gold_anchor_theme_coverage"] == 1
    assert q["selling_point_usefulness"] is None


def test_themes_separate_source_and_sentiment():
    result = empty_result()
    result["reviewAnalysis"]["positiveThemes"] = [{"themeId": "DELIVERY", "sentiment": "positive"}]
    result["reviewAnalysis"]["negativeThemes"] = [{"themeId": "DELIVERY", "sentiment": "negative"}]
    result["qnaAnalysis"]["themes"] = [{"themeId": "DELIVERY"}]
    assert len(m.theme_set(result)) == 3
