from __future__ import annotations

from palda_runtime import models
from palda_runtime.runtime import analyzer_from_env


class _Encoder:
    def __init__(self, *_args, **kwargs):
        self.kwargs = kwargs


class _Decoder:
    def __init__(self, *_args, **kwargs):
        self.kwargs = kwargs


def test_analyzer_uses_configured_encoder_training_length(monkeypatch) -> None:
    monkeypatch.setenv("PALDA_ENCODER_DIR", "/models/encoder")
    monkeypatch.setenv("PALDA_DECODER_DIR", "/models/decoder")
    monkeypatch.setenv("PALDA_ENCODER_MAX_LENGTH", "384")
    monkeypatch.setattr(models, "ServiceEncoderModel", _Encoder)
    monkeypatch.setattr(models, "GroundedDecoderModel", _Decoder)

    analyzer = analyzer_from_env()

    assert analyzer.encoder.kwargs["max_length"] == 384


def test_analyzer_defaults_to_promoted_encoder_length(monkeypatch) -> None:
    monkeypatch.setenv("PALDA_ENCODER_DIR", "/models/encoder")
    monkeypatch.setenv("PALDA_DECODER_DIR", "/models/decoder")
    monkeypatch.delenv("PALDA_ENCODER_MAX_LENGTH", raising=False)
    monkeypatch.setattr(models, "ServiceEncoderModel", _Encoder)
    monkeypatch.setattr(models, "GroundedDecoderModel", _Decoder)

    analyzer = analyzer_from_env()

    assert analyzer.encoder.kwargs["max_length"] == 256
