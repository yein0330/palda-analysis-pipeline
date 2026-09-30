#!/usr/bin/env python3
"""Compare the decoder LoRA against its own base model on identical inputs.

Replays saved encoder predictions through the unchanged runtime path twice:
once with the production adapter (checkpoint 221) and once with the adapter
disabled, i.e. base Qwen2.5-7B-Instruct with the same prompts and greedy
decoding. Any difference is the adapter's effect.

Mechanical measures only (format/fallback, quote grounding, generic phrasing,
selling points copied from a group summary, type/content mismatch, prompt
truncation). None rates persuasiveness.
"""
from __future__ import annotations

import argparse
import asyncio
import difflib
import json
import os
import re
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
RUNTIME = ROOT.parent / "palda-service-runtime"
DECODER_ADAPTER = RUNTIME / "artifacts/existing-model-1000-20260910/decoder-adapter"
GENERIC = re.compile(r"(긍정적인?|부정적인?) (피드백|평가|반응)|만족(도가 높|스럽|하는 것으로)|피드백이 (있|많)")
NEGATIVE = re.compile(r"불만|불편|부정적|문제가|문제점|부족|떨어지|떨어진|싫어|아쉽|아쉬|실망|개선이 필요|개선 필요")


def load_cases(args) -> list[dict]:
    cases = []
    for path in sorted(Path(args.capture_dir).glob("integration-product-*.json")):
        trace = json.loads(path.read_text())
        cases.append({"name": trace["caseId"], "request": trace["input"], "predictions": trace["encoderOutput"]})
    for row in json.loads(Path(args.raw_output).read_text()):
        cases.append({"name": f"raw-{row['productKey']}", "request": row["request"],
                      "predictions": row["encoderPredictions"]})
    return cases


def measure(response: dict, request: dict) -> dict:
    texts = {i["id"]: i["text"] for k in ("reviewSample", "qnaSample") for i in request[k]["items"]}
    groups = response["reviewAnalysis"]["positiveThemes"] + response["reviewAnalysis"]["negativeThemes"]
    qna = response["qnaAnalysis"]["themes"]
    summaries = [g["summary"] for g in groups + qna]
    points = response["sellingPoints"]
    return {
        "groups": len(groups) + len(qna),
        "fallback_summaries": sum(bool(g.get("fallbackUsed")) for g in groups + qna),
        "grounded_quotes": sum(g["quote"] in texts.get(g["quoteEvidenceId"], "") for g in groups)
        + sum(g["question"] in texts.get(g["questionEvidenceId"], "") for g in qna),
        "generic_summaries": sum(bool(GENERIC.search(s)) for s in summaries),
        "selling_points": len(points),
        "fallback_selling_points": sum("후보로 활용할 수 있습니다" in p["text"] or "개선이 필요합니다." == p["text"][-10:] and "확인되어" in p["text"] for p in points),
        "generic_selling_points": sum(bool(GENERIC.search(p["text"])) for p in points),
        "selling_points_copying_summary": sum(
            max((difflib.SequenceMatcher(None, p["text"], s).ratio() for s in summaries), default=0) >= 0.8
            for p in points),
        "type_content_mismatch": sum(p["type"] in ("core", "hidden") and bool(NEGATIVE.search(p["text"])) for p in points),
        "non_korean_texts": sum(not re.search(r"[가-힣]", t) for t in summaries + [p["text"] for p in points]),
    }


async def main(args) -> None:
    sys.path.insert(0, str(RUNTIME / "src"))
    import palda_runtime.runtime as runtime
    from palda_runtime.models import GroundedDecoderModel

    decoder = GroundedDecoderModel(DECODER_ADAPTER, device_map=args.device, load_in_4bit=True, batch_size=10)
    original = decoder._complete_many
    tokenizer = decoder.tokenizer
    truncated = {"lora": 0, "base": 0, "prompts": 0}

    class Replay:
        def __init__(self, predictions):
            self.predictions = predictions

        async def classify(self, request):
            return self.predictions

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    totals = {"lora": {}, "base": {}}
    for case in load_cases(args):
        row = {"case": case["name"], "request": case["request"]}
        for arm in ("lora", "base"):
            def complete(prompts, *, max_new_tokens=None, disable_adapter=False, _arm=arm):
                for prompt in prompts:
                    text = tokenizer.apply_chat_template(
                        [{"role": "system", "content": decoder.SYSTEM}, {"role": "user", "content": prompt}],
                        tokenize=False, add_generation_prompt=True)
                    truncated["prompts"] += _arm == "lora"
                    truncated[_arm] += len(tokenizer(text)["input_ids"]) > decoder.max_input_length
                return original(prompts, max_new_tokens=max_new_tokens,
                                disable_adapter=disable_adapter or _arm == "base")
            decoder._complete_many = complete
            response = await runtime.PipelineAnalyzer(
                Replay(case["predictions"]), decoder, encoder_version="replayed",
                decoder_version=f"checkpoint-221-{arm}", taxonomy_version="theme-v1.0.0",
            ).analyze(case["request"])
            row[f"{arm}Response"] = response
            row[f"{arm}Metrics"] = measure(response, case["request"])
            for key, value in row[f"{arm}Metrics"].items():
                totals[arm][key] = totals[arm].get(key, 0) + value
        (out / f"{case['name']}.json").write_text(json.dumps(row, ensure_ascii=False, indent=2))
        print(case["name"], "lora", row["loraMetrics"], "| base", row["baseMetrics"], flush=True)
    summary = {"cases": len(load_cases(args)), "totals": totals, "prompts_over_max_input_length": truncated,
               "max_input_length": decoder.max_input_length}
    (out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2))
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture-dir", default=str(ROOT / "artifacts/oracle-expansion-20260928"))
    parser.add_argument("--raw-output", default=str(ROOT / "artifacts/quality-examples-20260929/raw_products_output.json"))
    parser.add_argument("--output-dir", default=str(ROOT / "artifacts/decoder-adapter-vs-base-20260929"))
    parser.add_argument("--device", default="cuda:0")
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    asyncio.run(main(parser.parse_args()))
