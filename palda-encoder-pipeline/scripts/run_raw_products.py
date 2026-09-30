#!/usr/bin/env python3
"""Run the current production path on whole raw products.

Builds one service request per product from the raw crawl records - every
review, and every Q&A question that is neither a reply nor secret - and runs
the integration encoder plus decoder through the runtime `PipelineAnalyzer`,
unchanged. With `--diverse-evidence` it also runs the cross-group evidence
prototype on the same predictions so the two responses can be compared.

Input: `raw_products.json` written from the raw crawl for Validation products.
"""
from __future__ import annotations

import argparse
import asyncio
import importlib.util
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
RUNTIME = ROOT.parent / "palda-service-runtime"
ENCODER = ROOT / "artifacts/gold-anchor-backbone-pilot/ko-sroberta-champion-candidate-test-audit"
DECODER_ADAPTER = RUNTIME / "artifacts/existing-model-1000-20260910/decoder-adapter"


def build_request(product: dict) -> dict:
    meta = (product.get("inventory") or {}).get("product") or {}
    source = (product.get("inventory") or {}).get("source") or {}
    sources = (product.get("inventory") or {}).get("sources") or []
    path = source.get("fullPath") or (sources[0].get("fullPath") if sources else "")
    reviews = [{"id": f"r_{r['id']}", "text": r["reviewContent"].strip()}
               for r in product["reviews"] if (r.get("reviewContent") or "").strip()]
    questions = [{"id": f"q_{q['id']}", "text": q["text"].strip()}
                 for q in product["qna"]
                 if not q.get("isReply") and not q.get("isSecret") and (q.get("text") or "").strip()]
    return {
        "analysisId": f"raw-{product['productKey']}",
        "taxonomyVersion": "theme-v1.0.0",
        "product": {"productId": product["productKey"], "categoryPath": path, "name": meta.get("name")},
        "reviewSample": {"collectionOrder": "newest", "maxItems": 5000, "items": reviews},
        "qnaSample": {"collectionOrder": "newest", "maxItems": 5000, "items": questions},
    }


async def main(args) -> None:
    sys.path.insert(0, str(RUNTIME / "src"))
    import palda_runtime.runtime as runtime
    from palda_runtime.models import GroundedDecoderModel, ServiceEncoderModel

    encoder = ServiceEncoderModel(ENCODER, ENCODER / "taxonomy.json", device=args.device)
    decoder = GroundedDecoderModel(DECODER_ADAPTER, device_map=args.device, load_in_4bit=True, batch_size=10)
    original_build = runtime.build_report_input
    spec = importlib.util.spec_from_file_location(
        "prototype_evidence_diversity", ROOT / "scripts/prototype_evidence_diversity.py")
    prototype = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(prototype)

    class Replay:
        def __init__(self, predictions):
            self.predictions = predictions

        async def classify(self, request):
            return self.predictions

    results = []
    for product in json.loads(Path(args.input).read_text()):
        request = build_request(product)
        started = time.perf_counter()
        predictions = await encoder.classify(request)
        encoder_seconds = time.perf_counter() - started

        runtime.build_report_input = original_build
        started = time.perf_counter()
        current = await runtime.PipelineAnalyzer(
            Replay(predictions), decoder, encoder_version=ENCODER.name,
            decoder_version="decoder-imported-v2-checkpoint-221", taxonomy_version="theme-v1.0.0",
        ).analyze(request)
        row = {"productKey": product["productKey"], "name": request["product"]["name"],
               "category": request["product"]["categoryPath"],
               "reviewCount": len(request["reviewSample"]["items"]),
               "qnaCount": len(request["qnaSample"]["items"]),
               "encoderSeconds": round(encoder_seconds, 2),
               "currentSeconds": round(time.perf_counter() - started, 2),
               "request": request, "encoderPredictions": predictions, "currentResponse": current}

        if args.diverse_evidence:
            runtime.build_report_input = lambda req, preds, **kw: prototype.diversify(
                original_build(req, preds, evidence_limit=10_000, **kw))
            row["diverseEvidenceResponse"] = await runtime.PipelineAnalyzer(
                Replay(predictions), decoder, encoder_version=ENCODER.name,
                decoder_version="decoder-imported-v2-checkpoint-221", taxonomy_version="theme-v1.0.0",
            ).analyze(request)
            runtime.build_report_input = original_build
        results.append(row)
        print(f"{product['productKey']}: reviews {row['reviewCount']} qna {row['qnaCount']} "
              f"encoder {row['encoderSeconds']}s total {row['currentSeconds']}s", flush=True)

    out = Path(args.output)
    out.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(out)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default=str(ROOT / "artifacts/quality-examples-20260929/raw_products.json"))
    parser.add_argument("--output", default=str(ROOT / "artifacts/quality-examples-20260929/raw_products_output.json"))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--diverse-evidence", action="store_true")
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    asyncio.run(main(parser.parse_args()))
