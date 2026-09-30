from __future__ import annotations

import argparse
import asyncio
import json
import math
import os
import statistics
import time
from pathlib import Path
from typing import Any

from .runtime import analyzer_from_env


def _percentile(values: list[float], percentile: float) -> float:
    ordered = sorted(values)
    index = max(0, math.ceil(percentile * len(ordered)) - 1)
    return ordered[index]


async def benchmark(request: dict[str, Any], runs: int) -> dict[str, Any]:
    request = dict(request)
    request.setdefault("analysisId", "benchmark")
    request.setdefault("taxonomyVersion", os.environ.get("PALDA_TAXONOMY_VERSION", "theme-v1.0.0"))
    load_started = time.perf_counter()
    analyzer = analyzer_from_env()
    model_load_seconds = time.perf_counter() - load_started

    durations: list[float] = []
    stage_durations: list[dict[str, float]] = []
    result: dict[str, Any] | None = None
    for _ in range(runs):
        started = time.perf_counter()
        stage_started: dict[str, float] = {}

        def on_stage(stage: str, _progress: int) -> None:
            stage_started.setdefault(stage, time.perf_counter())

        result = await analyzer.analyze(request, on_stage=on_stage)
        finished = time.perf_counter()
        durations.append(finished - started)
        boundaries = [
            ("classification", "classifying", "aggregating"),
            ("aggregation", "aggregating", "generating"),
            ("reportGeneration", "generating", "validating"),
        ]
        measured = {
            name: round(stage_started[end] - stage_started[begin], 3)
            for name, begin, end in boundaries
        }
        measured["validation"] = round(finished - stage_started["validating"], 3)
        stage_durations.append(measured)

    item_count = len(request["reviewSample"]["items"]) + len(request["qnaSample"]["items"])
    return {
        "input": {
            "reviewItems": len(request["reviewSample"]["items"]),
            "qnaItems": len(request["qnaSample"]["items"]),
            "totalItems": item_count,
        },
        "timing": {
            "runs": runs,
            "modelLoadSeconds": round(model_load_seconds, 3),
            "analysisSeconds": [round(value, 3) for value in durations],
            "meanSeconds": round(statistics.fmean(durations), 3),
            "p50Seconds": round(statistics.median(durations), 3),
            "p95Seconds": round(_percentile(durations, 0.95), 3),
            "meanItemsPerSecond": round(item_count / statistics.fmean(durations), 3),
            "stageSeconds": stage_durations,
            "stageMeanSeconds": {
                stage: round(statistics.fmean(run[stage] for run in stage_durations), 3)
                for stage in stage_durations[0]
            },
        },
        "versions": (result or {}).get("versions", {}),
        "lastResult": result,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark repeated warm PALDA model inference.")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.runs < 1:
        parser.error("--runs must be at least 1")

    request = json.loads(args.input.read_text(encoding="utf-8"))
    payload = asyncio.run(benchmark(request, args.runs))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    rendered = json.dumps(payload, ensure_ascii=False, indent=2)
    args.output.write_text(rendered + "\n", encoding="utf-8")
    print(json.dumps({"input": payload["input"], "timing": payload["timing"]}, indent=2))


if __name__ == "__main__":
    main()
