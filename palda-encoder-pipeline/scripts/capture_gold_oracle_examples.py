#!/usr/bin/env python3
"""Run the real decoder path with human Gold labels in place of encoder predictions.

This is the oracle arm for the encoder-bottleneck question. Everything downstream
of classification - aggregation, group limits, the decoder adapter, the output
schema - is the production runtime, unchanged. Only the classifier is replaced,
by a lookup of the human Gold labels the 2026-09-11 capture already recorded as
reference. If selling points improve materially here, the bottleneck is upstream
of the decoder.

Products, inputs and the decoder checkpoint match
`scripts/capture_pipeline_examples.py` so the traces line up case by case.

Gold labels are model input in this arm and nowhere else. They are read from the
fixed Validation split; Test and the sealed simulation are not touched. Nothing
is promoted, uploaded or written outside `artifacts/`.
"""
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
OUT = ROOT / "artifacts/gold-oracle-20260923"
VALIDATION = ROOT / "data/releases/label-quality-20260910/relabel384/validation.jsonl"
ENCODER_ARTIFACT = ROOT / "artifacts/gold-anchor-backbone-pilot/ko-sroberta-champion-candidate-test-audit"
DECODER_ADAPTER = RUNTIME / "artifacts/existing-model-1000-20260910/decoder-adapter"
SPECIAL_THEMES = {"UNUSABLE"}


def write(path, value):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2))
    temporary.replace(path)


def selected_products():
    """The same three products the 2026-09-11 capture used."""
    groups = defaultdict(list)
    for line in VALIDATION.read_text().splitlines():
        row = json.loads(line)
        groups[(row["product_id"], row["category_key"])].append(row)
    selected, categories = [], set()
    for key, rows in sorted(groups.items(), key=lambda pair: (-len(pair[1]), pair[0])):
        cat1 = key[1].split(" > ")[0]
        if cat1 in categories or {r["record_type"] for r in rows} != {"review", "qna"}:
            continue
        sample = [r for kind in ("review", "qna")
                  for r in sorted((r for r in rows if r["record_type"] == kind), key=lambda r: r["sample_id"])[:10]]
        selected.append((key, sample))
        categories.add(cat1)
        if len(selected) == 3:
            break
    assert len(selected) == 3
    return selected


class GoldOracleEncoder:
    """Returns human Gold labels in the runtime's classifier output shape.

    Confidences are 1.0 because a human label carries no model probability. The
    runtime only uses confidence to break ties when ranking groups, so a constant
    leaves group ordering to mention counts, which is the intended behaviour here.
    """

    def __init__(self, gold_by_id, theme_names, catalog):
        self.gold_by_id = gold_by_id
        self.theme_names = theme_names
        self._catalog = catalog

    def allowed_theme_catalog(self, category_path):
        return self._catalog

    def _name(self, theme, kind):
        return self.theme_names.get(theme, {}).get(kind, theme)

    async def classify(self, request):
        predictions = []
        for kind, key in (("review", "reviewSample"), ("qna", "qnaSample")):
            for item in request[key]["items"]:
                row = self.gold_by_id[str(item["id"])]
                labels = [
                    label for label in row.get("labels", [])
                    if label.get("theme_id") not in SPECIAL_THEMES
                ]
                result = {
                    "id": str(item["id"]),
                    "usable": bool(row.get("usable", True)),
                    "usableConfidence": 1.0,
                }
                if kind == "review":
                    result["labels"] = [
                        {
                            "themeId": label["theme_id"],
                            "theme": self._name(label["theme_id"], "review"),
                            "sentiment": label["sentiment"],
                            "confidence": 1.0,
                        }
                        for label in labels
                        if label.get("sentiment") in ("positive", "negative")
                    ]
                else:
                    primary = next((label for label in labels if label.get("role") == "primary"), None)
                    if primary is None and labels:
                        primary = labels[0]
                    theme = primary["theme_id"] if primary else "OTHER"
                    result["primaryTheme"] = {
                        "themeId": theme,
                        "theme": self._name(theme, "qna"),
                        "confidence": 1.0,
                    }
                    result["secondaryThemes"] = [
                        {
                            "themeId": label["theme_id"],
                            "theme": self._name(label["theme_id"], "qna"),
                            "confidence": 1.0,
                        }
                        for label in labels
                        if label is not primary and label.get("role") != "primary"
                    ]
                predictions.append(result)
        return predictions


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

    taxonomy = json.loads((ENCODER_ARTIFACT / "taxonomy.json").read_text())
    theme_names = {
        str(item["id"]): {
            "review": str(item.get("review_label") or item["id"]),
            "qna": str(item.get("qna_label") or item["id"]),
        }
        for item in taxonomy.get("themes", [])
    }

    status = {"stage": "loading", "completed": 0, "planned": 3, "split": "validation",
              "arm": "gold_oracle",
              "purpose": "decoder ceiling with perfect classification; not a quality or latency benchmark",
              "humanLabelsUsedAsModelInput": True, "startedAt": time.time()}
    write(OUT / "status.json", status)
    try:
        # Loaded only for allowed_theme_catalog, so the decoder sees exactly the
        # Cat2-scoped taxonomy the integration arm showed it.
        reference_encoder = ServiceEncoderModel(
            ENCODER_ARTIFACT, ENCODER_ARTIFACT / "taxonomy.json", device="cuda:0")
        decoder = GroundedDecoderModel(
            DECODER_ADAPTER, device_map="cuda:1", load_in_4bit=True, batch_size=10)
        actual_complete = decoder._complete_many
        for index, (key, rows) in enumerate(selected_products(), 1):
            trace = {"caseId": f"product-{index}", "variant": "gold_oracle",
                     "encoder": "human-gold-oracle", "decoder": "Qwen2.5-7B-LoRA-checkpoint-221",
                     "split": "validation", "productionApproved": False,
                     "humanLabelsUsedAsModelInput": True, "stage": "input_ready",
                     "gold": {}, "decoderCompletions": [], "stageEvents": []}
            request = {"analysisId": f"gold-oracle-{index}", "taxonomyVersion": "theme-v1.0.0",
                       "product": {"productId": f"product-{index}", "categoryPath": key[1]},
                       "reviewSample": {"collectionOrder": "newest", "maxItems": 5000, "items": []},
                       "qnaSample": {"collectionOrder": "newest", "maxItems": 5000, "items": []}}
            gold_by_id = {}
            for row in rows:
                sid = "v_" + hashlib.sha256(row["sample_id"].encode()).hexdigest()[:12]
                request["reviewSample" if row["record_type"] == "review" else "qnaSample"]["items"].append(
                    {"id": sid, "text": row["text"]})
                trace["gold"][sid] = row["labels"]
                gold_by_id[sid] = row
            trace["input"] = request
            path = OUT / f"gold-oracle-product-{index}.json"

            def save():
                trace["updatedAt"] = time.time()
                write(path, trace)

            oracle = GoldOracleEncoder(
                gold_by_id, theme_names,
                reference_encoder.allowed_theme_catalog(key[1]),
            )

            class CaptureEncoder:
                async def classify(self, value):
                    trace["encoderOutput"] = await oracle.classify(value)
                    trace["stage"] = "encoder_completed"
                    save()
                    return trace["encoderOutput"]

                def allowed_theme_catalog(self, category_path):
                    return oracle.allowed_theme_catalog(category_path)

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
                trace["decoderCompletions"].append(
                    {"batch": len(trace["decoderCompletions"]) + 1, "outputs": result})
                save()
                return result

            decoder._complete_many = capture_completions
            analyzer = PipelineAnalyzer(
                CaptureEncoder(), CaptureDecoder(),
                encoder_version="human-gold-oracle",
                decoder_version="decoder-imported-v2-checkpoint-221",
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
            print(f"Completed gold-oracle product-{index}, {trace['elapsedSeconds']}s", flush=True)
        status.update(stage="completed", finishedAt=time.time())
        write(OUT / "status.json", status)
    except Exception as exc:
        status.update(stage="failed", error=f"{type(exc).__name__}: {exc}"[:500])
        write(OUT / "status.json", status)
        raise


if __name__ == "__main__":
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    asyncio.run(main())
