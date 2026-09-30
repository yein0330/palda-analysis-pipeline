from __future__ import annotations

from typing import Any, Callable, Protocol
import os

from .aggregation import build_report_input, materialize_grounded_report
from .schemas import ReportOutput


class Encoder(Protocol):
    async def classify(self, request: dict[str, Any]) -> list[dict[str, Any]]: ...


class Decoder(Protocol):
    async def generate(self, report_input: dict[str, Any]) -> dict[str, Any]: ...


class PipelineAnalyzer:
    def __init__(
        self,
        encoder: Encoder,
        decoder: Decoder,
        *,
        encoder_version: str,
        decoder_version: str,
        taxonomy_version: str,
        review_group_limit_per_sentiment: int = 3,
        qna_group_limit: int = 4,
        verify_with_decoder: bool = False,
    ) -> None:
        self.encoder = encoder
        self.decoder = decoder
        self.versions = {
            "encoderModel": encoder_version,
            "decoderModel": decoder_version,
            "taxonomy": taxonomy_version,
            "calculation": "review-qna-calc-v1.0.1",
        }
        self.review_group_limit_per_sentiment = review_group_limit_per_sentiment
        self.qna_group_limit = qna_group_limit
        self.verify_with_decoder = verify_with_decoder
        if verify_with_decoder:
            self.versions["calculation"] = "review-qna-calc-v1.1.1-decoder-verified"

    async def analyze(
        self,
        request: dict[str, Any],
        on_stage: Callable[[str, int], None] | None = None,
    ) -> dict[str, Any]:
        update = on_stage or (lambda _stage, _progress: None)
        update("classifying", 10)
        predictions = await self.encoder.classify(request)
        if self.verify_with_decoder:
            catalog_method = getattr(self.encoder, "allowed_theme_catalog", None)
            verifier = getattr(self.decoder, "verify_predictions", None)
            if not callable(catalog_method) or not callable(verifier):
                raise RuntimeError("decoder verification requires compatible encoder and decoder models")
            update("classifying", 45)
            predictions = await verifier(
                request, predictions, catalog_method(request["product"]["categoryPath"])
            )
        update("aggregating", 65)
        report_input = build_report_input(
            request, predictions,
            review_group_limit_per_sentiment=self.review_group_limit_per_sentiment,
            qna_group_limit=self.qna_group_limit,
        )
        decoder_input = {key: value for key, value in report_input.items() if not key.startswith("_")}
        update("generating", 75)
        generated = await self.decoder.generate(decoder_input)
        update("validating", 95)
        result = materialize_grounded_report(report_input, generated)
        result["versions"] = self.versions
        return ReportOutput.model_validate(result).model_dump()


def analyzer_from_env() -> PipelineAnalyzer:
    """Build the production analyzer from explicit, versioned artifact paths."""
    from .models import GroundedDecoderModel, ServiceEncoderModel

    encoder_dir = _required_env("PALDA_ENCODER_DIR")
    decoder_dir = _required_env("PALDA_DECODER_DIR")
    taxonomy_path = os.environ.get(
        "PALDA_TAXONOMY_PATH", os.path.join(encoder_dir, "taxonomy.json")
    )
    encoder = ServiceEncoderModel(
        encoder_dir,
        taxonomy_path,
        device=os.environ.get("PALDA_ENCODER_DEVICE", "auto"),
        batch_size=int(os.environ.get("PALDA_ENCODER_BATCH_SIZE", "64")),
        max_length=int(os.environ.get("PALDA_ENCODER_MAX_LENGTH", "256")),
    )
    decoder = GroundedDecoderModel(
        decoder_dir,
        device_map=os.environ.get("PALDA_DECODER_DEVICE_MAP", "auto"),
        load_in_4bit=os.environ.get("PALDA_DECODER_4BIT", "1") == "1",
        batch_size=int(os.environ.get("PALDA_DECODER_BATCH_SIZE", "10")),
    )
    return PipelineAnalyzer(
        encoder,
        decoder,
        encoder_version=os.environ.get("PALDA_ENCODER_VERSION", os.path.basename(encoder_dir)),
        decoder_version=os.environ.get("PALDA_DECODER_VERSION", os.path.basename(decoder_dir)),
        taxonomy_version=os.environ.get("PALDA_TAXONOMY_VERSION", "theme-v1.0.0"),
        review_group_limit_per_sentiment=int(os.environ.get("PALDA_REVIEW_GROUP_LIMIT", "3")),
        qna_group_limit=int(os.environ.get("PALDA_QNA_GROUP_LIMIT", "4")),
        verify_with_decoder=os.environ.get("PALDA_DECODER_VERIFY", "0") == "1",
    )


def _required_env(name: str) -> str:
    value = os.environ.get(name)
    if not value:
        raise RuntimeError(f"Required environment variable is missing: {name}")
    return value
