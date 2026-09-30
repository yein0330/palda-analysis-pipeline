import importlib


def test_consensus_requires_valid_votes_and_matching_secondary_roles(monkeypatch):
    from pathlib import Path
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "scripts"))
    summarize = importlib.import_module("compare_label_quality").summarize
    original = [{"theme_id": "DELIVERY", "role": "primary"}]

    def row(sample, model, labels, valid=True):
        return {"sample_id": sample, "model_id": model, "arm": "rubric",
                "labels": labels, "original_labels": original, "valid": valid,
                "latency_seconds": 1.0}

    rows = [row("agreed", model, original) for model in ("a", "b")]
    rows += [row("invalid", "a", original), row("invalid", "b", original, False)]
    rows += [row("secondary", "a", original), row("secondary", "b", original + [
        {"theme_id": "PRICE", "role": "secondary"}])]
    result = summarize(rows)
    ensemble = result["full_label_unanimity"]["rubric"]
    assert ensemble["accepted"] == 1
    assert ensemble["coverage"] == 1 / 3
    assert ensemble["primaryAgreementAccepted"] == 1.0
    assert result["individual"]["b/rubric"]["primaryAgreementAll"] == 2 / 3


def test_role_repair_is_limited_to_an_unambiguous_singleton(monkeypatch):
    from pathlib import Path
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "scripts"))
    normalize = importlib.import_module("normalize_label_votes").normalize
    label = {"theme_id": "PRICE", "role": None, "evidence": "가격"}
    raw = {"labels": [label], "valid": False, "errors": ["requires_one_primary"]}
    repaired = normalize(raw)
    assert repaired["valid"] and repaired["labels"][0]["role"] == "primary"
    assert raw["labels"][0]["role"] is None
    assert not normalize({**raw, "labels": [label, label]})["valid"]
    assert not normalize({**raw, "errors": ["requires_one_primary", "requires_exact_evidence"]})["valid"]


def test_repair_needs_two_valid_votes_and_never_changes_gold(monkeypatch):
    from pathlib import Path
    import pytest
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "scripts"))
    decision = importlib.import_module("build_rejudged_release").decision
    old = [{"theme_id": "OTHER", "role": "primary"}]
    new = [{"theme_id": "DELIVERY", "role": "primary", "evidence": "배송"}]
    row = {"label_tier": "silver", "labels": old}
    votes = [{"model_id": m, "valid": True, "labels": new} for m in ("a", "b")]
    assert decision(row, votes, {"a", "b"}, "relabel")[0]["labels"] == new
    assert decision(row, votes[:1], {"a", "b"}, "relabel")[0] == row
    assert decision(row, votes[:1], {"a", "b"}, "filter")[0] == row
    assert decision(row, votes, {"a", "b"}, "filter")[0] is None
    assert row["labels"] == old
    with pytest.raises(ValueError, match="Only Silver"):
        decision({**row, "label_tier": "gold"}, votes, {"a", "b"}, "relabel")
