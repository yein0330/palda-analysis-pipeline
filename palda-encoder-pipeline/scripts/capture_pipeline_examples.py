#!/usr/bin/env python3
"""Capture actual runtime stages on three deterministic, diverse Validation products."""
import asyncio
from collections import defaultdict
import hashlib
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
RUNTIME = ROOT.parent / "palda-service-runtime"
OUT = ROOT / "artifacts/pipeline-examples-20260911"
VALIDATION = ROOT / "data/releases/label-quality-20260910/relabel384/validation.jsonl"


def write(path, value):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2))
    temporary.replace(path)


def selected_products():
    groups = defaultdict(list)
    for line in VALIDATION.read_text().splitlines():
        row = json.loads(line)
        groups[(row["product_id"], row["category_key"])].append(row)
    selected, categories = [], set()
    for key, rows in sorted(groups.items(), key=lambda pair: (-len(pair[1]), pair[0])):
        cat1 = key[1].split(" > ")[0]
        if cat1 in categories or {r["record_type"] for r in rows} != {"review", "qna"}:
            continue
        # Cap per type, do not duplicate input to manufacture a large workload.
        sample = [r for kind in ("review", "qna")
                  for r in sorted((r for r in rows if r["record_type"] == kind), key=lambda r: r["sample_id"])[:10]]
        selected.append((key, sample))
        categories.add(cat1)
        if len(selected) == 3:
            break
    assert len(selected) == 3
    return selected


async def main():
    sys.path.insert(0, str(RUNTIME / "src"))
    from palda_runtime.models import GroundedDecoderModel, ServiceEncoderModel
    from palda_runtime.runtime import PipelineAnalyzer
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required")
    OUT.mkdir(parents=True, exist_ok=True)
    if (OUT / "status.json").exists():
        prior = json.loads((OUT / "status.json").read_text())
        if prior.get("stage") != "failed" or list(OUT.glob("*-product-*.json")):
            raise RuntimeError("Refusing to overwrite previous capture")
        (OUT / "status.json").rename(OUT / f"failed-start-{int(time.time())}.json")
    status = {"stage": "loading", "completed": 0, "planned": 6, "split": "validation",
              "purpose": "qualitative demonstration, not independent quality or latency benchmark",
              "startedAt": time.time()}
    write(OUT / "status.json", status)
    versions = {
        "integration": ROOT / "artifacts/gold-anchor-backbone-pilot/ko-sroberta-champion-candidate-test-audit",
        "research": ROOT / "artifacts/label-quality-20260910/encoder-relabel384-seed42",
    }
    try:
        decoder = GroundedDecoderModel(
            RUNTIME / "artifacts/existing-model-1000-20260910/decoder-adapter",
            device_map="cuda:1", load_in_4bit=True, batch_size=10)
        actual_complete = decoder._complete_many
        for version, artifact in versions.items():
            encoder = ServiceEncoderModel(artifact, artifact / "taxonomy.json", device="cuda:0")
            for index, (key, rows) in enumerate(selected_products(), 1):
                trace = {"caseId": f"product-{index}", "variant": version,
                         "encoder": artifact.name, "decoder": "Qwen2.5-7B-LoRA-checkpoint-221",
                         "split": "validation", "productionApproved": False,
                         "humanLabelsUsedAsModelInput": False, "stage": "input_ready",
                         "gold": {}, "decoderCompletions": [], "stageEvents": []}
                request = {"analysisId": f"qualitative-{version}-{index}", "taxonomyVersion": "theme-v1.0.0",
                           "product": {"productId": f"product-{index}", "categoryPath": key[1]},
                           "reviewSample": {"collectionOrder": "newest", "maxItems": 5000, "items": []},
                           "qnaSample": {"collectionOrder": "newest", "maxItems": 5000, "items": []}}
                for row in rows:
                    sid = "v_" + hashlib.sha256(row["sample_id"].encode()).hexdigest()[:12]
                    request["reviewSample" if row["record_type"] == "review" else "qnaSample"]["items"].append(
                        {"id": sid, "text": row["text"]})
                    trace["gold"][sid] = row["labels"]
                trace["input"] = request
                path = OUT / f"{version}-product-{index}.json"

                def save():
                    trace["updatedAt"] = time.time()
                    write(path, trace)

                class CaptureEncoder:
                    async def classify(self, value):
                        trace["encoderOutput"] = await encoder.classify(value)
                        trace["stage"] = "encoder_completed"
                        save()
                        return trace["encoderOutput"]

                class CaptureDecoder:
                    async def generate(self, value):
                        trace["decoderInput"] = value
                        trace["stage"] = "decoder_generating"
                        save()
                        result = await decoder.generate(value)
                        trace["decoderOutput"] = result
                        trace["stage"] = "decoder_completed"
                        save()
                        return result

                def capture_completions(prompts, **kwargs):
                    result = actual_complete(prompts, **kwargs)
                    trace["decoderCompletions"].append({"batch": len(trace["decoderCompletions"]) + 1,
                                                       "outputs": result})
                    save()
                    return result

                decoder._complete_many = capture_completions
                analyzer = PipelineAnalyzer(CaptureEncoder(), CaptureDecoder(),
                    encoder_version=artifact.name, decoder_version="decoder-imported-v2-checkpoint-221",
                    taxonomy_version="theme-v1.0.0", verify_with_decoder=False)
                started = time.perf_counter()

                def stage(name, progress):
                    trace["stageEvents"].append({"stage": name, "progress": progress,
                                                 "elapsedSeconds": round(time.perf_counter() - started, 3)})
                    save()

                save()
                trace["finalOutput"] = await analyzer.analyze(request, on_stage=stage)
                trace.update(stage="completed", elapsedSeconds=round(time.perf_counter() - started, 3))
                save()
                status.update(stage="running", completed=status["completed"] + 1)
                write(OUT / "status.json", status)
                print(f"Completed {version} product-{index}, {trace['elapsedSeconds']}s", flush=True)
            del encoder
            torch.cuda.empty_cache()
        status.update(stage="completed", finishedAt=time.time())
        write(OUT / "status.json", status)
    except Exception as exc:
        status.update(stage="failed", error=type(exc).__name__)
        write(OUT / "status.json", status)
        raise


if __name__ == "__main__":
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    asyncio.run(main())
