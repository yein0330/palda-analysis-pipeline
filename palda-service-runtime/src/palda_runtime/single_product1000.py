"""Single-product 1000-item repeated-text load measurement, not quality evaluation."""
import asyncio
import json
import os
from pathlib import Path

from .benchmark import benchmark
from .schemas import AnalysisRequest, ReportOutput


def main():
    root = Path(__file__).resolve().parents[2]
    out = root / "artifacts/single-product-1000-20260910"
    out.mkdir(parents=True, exist_ok=False)
    request = json.loads((root / "artifacts/max-load-synthetic-request.json").read_text())
    request["product"]["productId"] = "latency-single-product-1000-repeated"
    for field in ("reviewSample", "qnaSample"):
        request[field]["items"] = request[field]["items"][:500]
    AnalysisRequest.model_validate(request)
    (out / "request.json").write_text(json.dumps(request, ensure_ascii=False, indent=2))
    manifest = {
        "products": 1, "reviews": 500, "qna": 500, "totalItems": 1000,
        "uniqueReviewTexts": len({r["text"] for r in request["reviewSample"]["items"]}),
        "uniqueQnaTexts": len({r["text"] for r in request["qnaSample"]["items"]}),
        "measurement": "In-process real encoder+decoder E2E; excludes HTTP, queue, download",
        "qualityBenchmark": False, "repeatedTextLoadTest": True,
        "testRead": False, "sealedSimulationRead": False,
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    encoder = root.parent / "palda-encoder-pipeline/artifacts/gold-anchor-backbone-pilot/ko-sroberta-champion-candidate-test-audit"
    adapter = root / "artifacts/existing-model-1000-20260910-dense/decoder-adapter"
    os.environ.update(PALDA_ENCODER_DIR=str(encoder), PALDA_TAXONOMY_PATH=str(encoder / "taxonomy.json"),
        PALDA_ENCODER_DEVICE="cuda:0", PALDA_ENCODER_BATCH_SIZE="64", PALDA_ENCODER_MAX_LENGTH="256",
        PALDA_ENCODER_VERSION="ko-sroberta-gold-silver-s20260909", PALDA_DECODER_DIR=str(adapter),
        PALDA_DECODER_VERSION="decoder-imported-v2-checkpoint-221", PALDA_DECODER_4BIT="1",
        PALDA_DECODER_DEVICE_MAP="auto", PALDA_DECODER_BATCH_SIZE="10", PALDA_DECODER_VERIFY="0",
        PALDA_REVIEW_GROUP_LIMIT="3", PALDA_QNA_GROUP_LIMIT="4")
    (out / "status.json").write_text(json.dumps({"stage": "running", **manifest}, indent=2))
    try:
        result = asyncio.run(benchmark(request, runs=3))
        ReportOutput.model_validate(result["lastResult"])
        (out / "benchmark.json").write_text(json.dumps(result, ensure_ascii=False, indent=2))
        (out / "status.json").write_text(json.dumps({"stage": "completed", "timing": result["timing"]}, indent=2))
        print(json.dumps({"manifest": manifest, "timing": result["timing"]}, indent=2), flush=True)
    except Exception as exc:
        (out / "status.json").write_text(json.dumps({"stage": "failed", "error": repr(exc)}))
        raise


if __name__ == "__main__":
    main()
