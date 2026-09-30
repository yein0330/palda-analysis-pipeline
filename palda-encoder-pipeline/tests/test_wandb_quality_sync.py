import json
from pathlib import Path
import sys
from types import SimpleNamespace
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from wandb_quality_sync import flatten, payload, snapshots, verify_destination


def test_metrics_allowlist():
    result = flatten({"service_macro_f1": .25, "secret": "not-allowed",
                      "qna_primary": {"accuracy": .5, "text": "private"},
                      "qna_secondary": {"macro_f1": float("nan")}})
    assert result == {"validation/service_macro_f1": .25, "validation/qna_primary/accuracy": .5}


def test_payload_excludes_text_test_keys_and_errors(tmp_path):
    (tmp_path / "metrics.json").write_text(json.dumps({
        "config": {"seed": 42, "api_key": "private-key", "text": "private-text"},
        "test": {"service_macro_f1": .99},
        "validation": {"service_macro_f1": .2},
        "history": [{"epoch": 1, "train_loss": 1.2, "validation": {"service_macro_f1": .1}}]}))
    (tmp_path / "live.json").write_text(json.dumps({"stage": "completed", "error": "private-key"}))
    result = payload(tmp_path, 42, "test-method", .25)
    serialized = json.dumps(result)
    assert "private-key" not in serialized and "private-text" not in serialized and ".99" not in serialized
    assert result["summary"]["selected_validation/service_macro_f1"] == .2
    assert result["history"][0]["train/epoch_loss"] == 1.2


def test_actual_twelve_runs_no_raw_predictions():
    result = snapshots([])
    assert len(result) == 12
    assert all(len(v["history"]) == 3 for v in result.values())
    assert "predicted_labels" not in json.dumps(result)


@pytest.mark.parametrize("access,members,expected_ok", [
    ("PRIVATE", ["yein0330"], True),
    ("PUBLIC", ["yein0330"], False),
    ("PRIVATE", ["yein0330", "other"], False),
    (None, ["yein0330"], False),
])
def test_destination_fails_closed(monkeypatch, access, members, expected_ok):
    monkeypatch.setitem(sys.modules, "wandb_gql", SimpleNamespace(gql=lambda x: x))
    team = SimpleNamespace(load=lambda: None, members=[SimpleNamespace(username=m) for m in members])
    api = SimpleNamespace(viewer=SimpleNamespace(username="yein0330"),
        client=SimpleNamespace(execute=lambda *a: {"project": {"access": access, "public": None, "readOnly": False}}),
        team=lambda _: team)
    if expected_ok:
        assert verify_destination(api, "team", "project", "yein0330")["access"] == "PRIVATE"
    else:
        with pytest.raises(RuntimeError):
            verify_destination(api, "team", "project", "yein0330")
