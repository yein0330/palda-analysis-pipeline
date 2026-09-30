#!/usr/bin/env python3
"""Prototype cross-group evidence diversity and measure it end to end.

The runtime's `_select_evidence` removes duplicate text only within one theme
group. A review the encoder tags with several themes is the most confident
item in each of them, so it becomes the first evidence, and usually the quote,
of every such group, and the report shows the same sentence under different
themes.

This prototype keeps grouping, ranking and group limits exactly as the runtime
computes them and changes only which items fill each review group's evidence:
groups with the fewest candidates choose first, and each takes items no other
group has used yet, falling back to a shared item only when it has nothing else.
It replays saved encoder predictions (no encoder rerun) and runs the unchanged
decoder, so any difference comes from evidence selection alone.

The runtime package is not modified; `build_report_input` is wrapped in-process.
Validation data only.
"""
from __future__ import annotations

import argparse
import asyncio
from collections import Counter
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
RUNTIME = ROOT.parent / "palda-service-runtime"
DECODER_ADAPTER = RUNTIME / "artifacts/existing-model-1000-20260910/decoder-adapter"
EVIDENCE_LIMIT = 3


def diversify(report_input: dict) -> dict:
    groups = report_input.get("reviewGroups", [])
    order = sorted(range(len(groups)), key=lambda i: (len(groups[i]["evidence"]), i))
    used: set[str] = set()
    for index in order:
        candidates = groups[index]["evidence"]
        fresh = [item for item in candidates if item["id"] not in used]
        chosen = fresh[:EVIDENCE_LIMIT] or candidates[:1]
        groups[index]["evidence"] = chosen
        used.update(item["id"] for item in chosen)
    for group in report_input.get("qnaGroups", []):
        group["evidence"] = group["evidence"][:EVIDENCE_LIMIT]
    return report_input


def shared_quote_rate(result: dict) -> tuple[int, int]:
    groups = result["reviewAnalysis"]["positiveThemes"] + result["reviewAnalysis"]["negativeThemes"]
    counts = Counter(group.get("quote") for group in groups)
    return sum(c for q, c in counts.items() if q and c > 1), len(groups)


def load_cases(args) -> list[dict]:
    cases = []
    for path in sorted(Path(args.capture_dir).glob("integration-product-*.json")):
        trace = json.loads(path.read_text())
        cases.append({"name": trace["caseId"], "request": trace["input"],
                      "predictions": trace["encoderOutput"], "baseline": trace["finalOutput"]})
    for product in args.size_products:
        for size in (10, 100, 1000):
            trace = json.loads((Path(args.size_dir) / f"cases/{product}-n{size}-e3-r0.json").read_text())
            cases.append({"name": f"{product}-n{size}", "request": trace["request"],
                          "predictions": trace["predictions"], "baseline": trace["result"]})
    return cases


async def main(args) -> None:
    sys.path.insert(0, str(RUNTIME / "src"))
    import palda_runtime.runtime as runtime
    from palda_runtime.models import GroundedDecoderModel

    original = runtime.build_report_input

    def build_diverse(request, predictions, **kwargs):
        # Ask for every candidate, then choose evidence across groups.
        return diversify(original(request, predictions, evidence_limit=10_000, **kwargs))

    runtime.build_report_input = build_diverse
    decoder = GroundedDecoderModel(DECODER_ADAPTER, device_map=args.device, load_in_4bit=True, batch_size=10)

    class Replay:
        def __init__(self, predictions):
            self.predictions = predictions

        async def classify(self, request):
            return self.predictions

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    for case in load_cases(args):
        path = out / f"{case['name']}.json"
        if path.exists():
            rows.append(json.loads(path.read_text()))
            continue
        analyzer = runtime.PipelineAnalyzer(
            Replay(case["predictions"]), decoder, encoder_version="replayed-predictions",
            decoder_version="decoder-imported-v2-checkpoint-221", taxonomy_version="theme-v1.0.0")
        started = time.perf_counter()
        result = await analyzer.analyze(case["request"])
        before, groups_before = shared_quote_rate(case["baseline"])
        after, groups_after = shared_quote_rate(result)
        row = {"case": case["name"], "seconds": round(time.perf_counter() - started, 1),
               "sharedQuotesBefore": before, "groupsBefore": groups_before,
               "sharedQuotesAfter": after, "groupsAfter": groups_after,
               "request": case["request"], "baselineResponse": case["baseline"],
               "diverseEvidenceResponse": result}
        path.write_text(json.dumps(row, ensure_ascii=False, indent=2))
        rows.append(row)
        print(f"{case['name']}: shared quotes {before}/{groups_before} -> {after}/{groups_after}", flush=True)

    summary = {
        "cases": len(rows),
        "sharedQuotesBefore": sum(r["sharedQuotesBefore"] for r in rows),
        "sharedQuotesAfter": sum(r["sharedQuotesAfter"] for r in rows),
        "groupsBefore": sum(r["groupsBefore"] for r in rows),
        "groupsAfter": sum(r["groupsAfter"] for r in rows),
    }
    (out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2))
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture-dir", default=str(ROOT / "artifacts/oracle-expansion-20260928"))
    parser.add_argument("--size-dir", default=str(ROOT / "artifacts/input-size-tradeoff-20260921"))
    parser.add_argument("--size-products", nargs="*", default=["p_327c50dbce72393d", "p_12b197eb0ef62b3a"])
    parser.add_argument("--output-dir", default=str(ROOT / "artifacts/evidence-diversity-20260929"))
    parser.add_argument("--device", default="cuda:0")
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    asyncio.run(main(parser.parse_args()))
