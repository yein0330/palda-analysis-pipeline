#!/usr/bin/env python3
"""Run integration, research and Gold-oracle arms on a larger Validation product set.

Extends the three-product comparison of `capture_gold_oracle_examples.py`. The
selection keeps the original rule (products with both Review and Q&A, largest
first, up to 10 records per type) but allows up to `--per-cat1` products per
Cat1, because Validation has only 10 Cat1 values. All arms see identical
requests and the same decoder checkpoint; only the classifier differs.

Gold labels are model input only in the gold_oracle arm. Validation only; Test
and the sealed simulation are not read.
"""
import argparse
import asyncio
from collections import Counter, defaultdict
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
RUNTIME = ROOT.parent / "palda-service-runtime"
VALIDATION = ROOT / "data/releases/label-quality-20260910/relabel384/validation.jsonl"
DECODER_ADAPTER = RUNTIME / "artifacts/existing-model-1000-20260910/decoder-adapter"
ENCODERS = {
    "integration": ROOT / "artifacts/gold-anchor-backbone-pilot/ko-sroberta-champion-candidate-test-audit",
    "research": ROOT / "artifacts/label-quality-20260910/encoder-relabel384-seed42",
}


def _load_oracle():
    spec = importlib.util.spec_from_file_location(
        "capture_gold_oracle_examples", ROOT / "scripts/capture_gold_oracle_examples.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.GoldOracleEncoder


def write(path, value):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2))
    temporary.replace(path)


def selected_products(limit, per_cat1):
    groups = defaultdict(list)
    for line in VALIDATION.read_text().splitlines():
        row = json.loads(line)
        groups[(row["product_id"], row["category_key"])].append(row)
    selected, used = [], Counter()
    for key, rows in sorted(groups.items(), key=lambda pair: (-len(pair[1]), pair[0])):
        cat1 = key[1].split(" > ")[0]
        if used[cat1] >= per_cat1 or {r["record_type"] for r in rows} != {"review", "qna"}:
            continue
        sample = [r for kind in ("review", "qna")
                  for r in sorted((r for r in rows if r["record_type"] == kind), key=lambda r: r["sample_id"])[:10]]
        selected.append((key, sample))
        used[cat1] += 1
        if len(selected) == limit:
            break
    return selected


async def main(args):
    sys.path.insert(0, str(RUNTIME / "src"))
    from palda_runtime.models import GroundedDecoderModel, ServiceEncoderModel
    from palda_runtime.runtime import PipelineAnalyzer
    import torch
    GoldOracleEncoder = _load_oracle()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    products = selected_products(args.products, args.per_cat1)
    status = {"stage": "loading", "completed": 0, "planned": len(products) * 3, "split": "validation",
              "products": len(products), "per_cat1": args.per_cat1,
              "cat1_counts": dict(Counter(key[1].split(" > ")[0] for key, _ in products)),
              "startedAt": time.time()}
    write(out / "status.json", status)

    taxonomy = json.loads((ENCODERS["integration"] / "taxonomy.json").read_text())
    theme_names = {
        str(item["id"]): {"review": str(item.get("review_label") or item["id"]),
                          "qna": str(item.get("qna_label") or item["id"])}
        for item in taxonomy.get("themes", [])
    }
    decoder = GroundedDecoderModel(DECODER_ADAPTER, device_map=args.decoder_device,
                                   load_in_4bit=True, batch_size=10)
    actual_complete = decoder._complete_many

    for arm in ("integration", "research", "gold_oracle"):
        artifact = ENCODERS.get(arm, ENCODERS["integration"])
        encoder = ServiceEncoderModel(artifact, artifact / "taxonomy.json", device=args.encoder_device)
        for index, (key, rows) in enumerate(products, 1):
            path = out / f"{arm}-product-{index:02d}.json"
            if path.exists() and json.loads(path.read_text()).get("stage") == "completed":
                continue
            trace = {"caseId": f"product-{index:02d}", "variant": arm,
                     "encoder": "human-gold-oracle" if arm == "gold_oracle" else artifact.name,
                     "decoder": "Qwen2.5-7B-LoRA-checkpoint-221", "split": "validation",
                     "productionApproved": False, "humanLabelsUsedAsModelInput": arm == "gold_oracle",
                     "stage": "input_ready", "gold": {}, "decoderCompletions": [], "stageEvents": []}
            request = {"analysisId": f"expansion-{arm}-{index:02d}", "taxonomyVersion": "theme-v1.0.0",
                       "product": {"productId": f"product-{index:02d}", "categoryPath": key[1]},
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

            def save():
                trace["updatedAt"] = time.time()
                write(path, trace)

            classifier = (GoldOracleEncoder(gold_by_id, theme_names, encoder.allowed_theme_catalog(key[1]))
                          if arm == "gold_oracle" else encoder)

            class CaptureEncoder:
                async def classify(self, value):
                    trace["encoderOutput"] = await classifier.classify(value)
                    save()
                    return trace["encoderOutput"]

                def allowed_theme_catalog(self, category_path):
                    return encoder.allowed_theme_catalog(category_path)

            class CaptureDecoder:
                async def generate(self, value):
                    trace["decoderInput"] = value
                    result = await decoder.generate(value)
                    trace["decoderOutput"] = result
                    save()
                    return result

            def capture_completions(prompts, **kwargs):
                result = actual_complete(prompts, **kwargs)
                trace["decoderCompletions"].append({"batch": len(trace["decoderCompletions"]) + 1, "outputs": result})
                return result

            decoder._complete_many = capture_completions
            analyzer = PipelineAnalyzer(CaptureEncoder(), CaptureDecoder(),
                                        encoder_version=trace["encoder"],
                                        decoder_version="decoder-imported-v2-checkpoint-221",
                                        taxonomy_version="theme-v1.0.0", verify_with_decoder=False)
            started = time.perf_counter()
            trace["finalOutput"] = await analyzer.analyze(request)
            trace.update(stage="completed", elapsedSeconds=round(time.perf_counter() - started, 3))
            save()
            status.update(stage="running", completed=status["completed"] + 1)
            write(out / "status.json", status)
            print(f"{arm} product-{index:02d} {trace['elapsedSeconds']}s", flush=True)
        del encoder
        torch.cuda.empty_cache()
    status.update(stage="completed", finishedAt=time.time())
    write(out / "status.json", status)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default=str(ROOT / "artifacts/oracle-expansion-20260928"))
    parser.add_argument("--products", type=int, default=20)
    parser.add_argument("--per-cat1", type=int, default=2)
    parser.add_argument("--encoder-device", default="cuda:0")
    parser.add_argument("--decoder-device", default="cuda:0")
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    asyncio.run(main(parser.parse_args()))
