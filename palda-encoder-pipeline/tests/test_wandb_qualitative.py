import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from wandb_qualitative import redact, page, build, alias, encoder_snapshot, ROOT
import pytest


def test_sensitive_patterns_redacted_recursively():
    value = {"nested": ["010-1234-5678 person@example.com https://example.com/private", "wandb_v1_example_secret_key_123456789"]}
    output = str(redact(value))
    assert "1234-5678" not in output and "person@" not in output
    assert "https://" not in output and "example_secret" not in output
    assert "가림" in output


def test_quantities_preserved():
    assert redact("허리 63cm, 가격 15000원, 사이즈 95") == "허리 63cm, 가격 15000원, 사이즈 95"


def test_real_validation_views_and_quote_grounding():
    data, traces = build()
    assert len(data["01_inputs"]["data"]) == 707
    assert len(data["02_encoder_comparison"]["data"]) == 8484
    assert len(traces) == 6 and all(t["stage"] == "completed" for t in traces)
    for t in traces:
        sources = {i["id"]: i["text"] for k in ("reviewSample", "qnaSample") for i in t["input"][k]["items"]}
        final = t["finalOutput"]
        assert len(t["encoderOutput"]) == len(sources)
        for key in ("positiveThemes", "negativeThemes"):
            for group in final["reviewAnalysis"][key]:
                assert group["quote"] in sources[group["quoteEvidenceId"]]
        for group in final["qnaAnalysis"]["themes"]:
            assert group["question"] in sources[group["questionEvidenceId"]]
        for point in final["sellingPoints"]:
            assert set(point["evidenceIds"]) <= sources.keys()


def test_html_escapes_source_text():
    _, traces = build()
    traces[0]["input"]["reviewSample"]["items"][0]["text"] = '<script>alert("x")</script>'
    result = page(traces)
    assert '<script>' not in result
    assert '&lt;script&gt;' in result
    assert alias("original-source-identifier").startswith("v_")


def test_encoder_snapshot_scope():
    path = ROOT / "artifacts/label-quality-20260910/encoder-relabel384-seed42/validation_predictions.jsonl"
    table = encoder_snapshot(path)
    assert len(table["data"]) == 707
    with pytest.raises(ValueError):
        encoder_snapshot(path.with_name("test_predictions.jsonl"))
