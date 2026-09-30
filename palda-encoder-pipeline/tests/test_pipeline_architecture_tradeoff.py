import importlib.util
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from pipeline_architecture_tradeoff import prompt, route, valid

CAT = [{"themeId": "PRICE", "reviewName": "가격", "qnaName": "가격 문의"}]


def test_independent_prompt_has_no_candidate():
    p = prompt("review", "싸요", "가전", CAT)
    assert '"candidate"' not in p
    assert '"candidate"' in prompt("review", "싸요", "가전", CAT, {"id": "a"})


def test_route_is_capped_deterministic_and_includes_no_labels():
    rows = [{"id": str(i), "usable": True, "usableConfidence": .9, "labels": []} for i in range(10)]
    chosen, eligible = route(rows)
    assert chosen == {"0", "1"} and eligible == 10
    assert route(list(reversed(rows))) == (chosen, eligible)


def test_validation_rejects_invalid_and_duplicate_labels():
    good = {"usable": True, "labels": [{"themeId": "PRICE", "sentiment": "positive"}]}
    assert valid(good, "review", CAT)
    assert not valid({}, "review", CAT)
    assert not valid({**good, "usable": "true"}, "review", CAT)
    assert not valid({**good, "labels": good["labels"] * 2}, "review", CAT)
    assert not valid({"usable": True, "primaryThemeId": "BAD", "secondaryThemeIds": []}, "qna", CAT)
