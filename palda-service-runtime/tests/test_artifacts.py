from __future__ import annotations

import pytest

from palda_runtime.artifacts import _revision, _validate


def test_revision_requires_immutable_commit(monkeypatch) -> None:
    monkeypatch.setenv("PALDA_ENCODER_REVISION", "main")
    with pytest.raises(RuntimeError, match="immutable"):
        _revision("PALDA_ENCODER_REVISION")
    monkeypatch.setenv("PALDA_ENCODER_REVISION", "a" * 40)
    assert _revision("PALDA_ENCODER_REVISION") == "a" * 40


def test_encoder_artifact_contract(tmp_path) -> None:
    with pytest.raises(RuntimeError, match="incomplete"):
        _validate(tmp_path, ("heads.pt", "taxonomy.json"), "encoder")


def test_decoder_artifact_requires_pinned_base_revision(tmp_path) -> None:
    (tmp_path / "adapter_model.safetensors").write_bytes(b"adapter")
    (tmp_path / "adapter_config.json").write_text(
        '{"base_model_name_or_path":"Qwen/Qwen2.5-7B-Instruct","revision":null}',
        encoding="utf-8",
    )
    with pytest.raises(RuntimeError, match="pin its base model"):
        _validate(tmp_path, ("adapter_config.json",), "decoder")
    (tmp_path / "adapter_config.json").write_text(
        '{"base_model_name_or_path":"Qwen/Qwen2.5-7B-Instruct","revision":"' + "a" * 40 + '"}',
        encoding="utf-8",
    )
    _validate(tmp_path, ("adapter_config.json",), "decoder")
