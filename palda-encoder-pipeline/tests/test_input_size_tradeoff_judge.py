import importlib.util
from pathlib import Path
import sys

import pytest

SCRIPTS = Path(__file__).parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))
SPEC = importlib.util.spec_from_file_location("judge", SCRIPTS / "input_size_tradeoff_judge.py")
m = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(m)


def test_parse_requires_explicit_bounded_ratings():
    good = '{"summary_faithfulness": 4, "theme_alignment": 3, "selling_groundedness": null, "selling_usefulness": null, "reasons":"소구점 없음"}'
    assert m.parse_rating(good)["selling_usefulness"] is None
    with pytest.raises(ValueError):
        m.parse_rating(good.replace('"theme_alignment": 3', '"theme_alignment": 6'))
    with pytest.raises(ValueError):
        m.parse_rating(good.replace('"theme_alignment": 3,', ''))
    with pytest.raises(ValueError):
        m.parse_rating(good.replace('"theme_alignment": 3', '"theme_alignment": true'))


def test_blinds_runtime_and_condition_metadata():
    c = {"caseId": "secret_condition", "category": "A > B", "size": 1000, "seconds": 50,
         "evidenceBudget": 8, "decoderInput": {"reviewGroups": [], "qnaGroups": []},
         "result": {"reviewAnalysis": {"positiveThemes": [], "negativeThemes": []},
                    "qnaAnalysis": {"themes": []}, "sellingPoints": []}}
    p = m.payload(c)
    assert "seconds" not in p and "size" not in p and "caseId" not in p
