"""Measure the real POST/polling path against an already-ready API."""
from __future__ import annotations

import argparse
import asyncio
import json
import statistics
import time
from pathlib import Path

import httpx

from .schemas import ReportOutput


async def measure(client: httpx.AsyncClient, request: dict, *, timeout: float = 600) -> dict:
    started = time.perf_counter()
    async with asyncio.timeout(timeout):
        response = await client.post("/api/product-analyses", json=request)
        if response.status_code != 202:
            raise RuntimeError(f"Job submission returned HTTP {response.status_code}")
        accepted = time.perf_counter() - started
        analysis_id = response.json()["analysisId"]
        polls = 0
        while True:
            response = await client.get(f"/api/product-analyses/{analysis_id}")
            response.raise_for_status()
            polls += 1
            job = response.json()
            if job["status"] == "failed":
                raise RuntimeError(f"Analysis {analysis_id} failed")
            if job["status"] == "completed":
                result = ReportOutput.model_validate(job["result"]).model_dump()
                return {
                    "acceptSeconds": round(accepted, 4),
                    "completionObservedSeconds": round(time.perf_counter() - started, 4),
                    "pollCount": polls,
                    "result": result,
                }
            await asyncio.sleep(max(0.1, float(response.headers.get("Retry-After", "2"))))


async def run(args: argparse.Namespace) -> None:
    request = json.loads(args.input.read_text(encoding="utf-8"))
    measurements = []
    async with httpx.AsyncClient(base_url=args.base_url, timeout=30) as client:
        (await client.get("/ready")).raise_for_status()
        for index in range(args.runs):
            measured = await measure(client, request, timeout=args.timeout)
            measurements.append(measured)
            # Persist each completed run so an interruption does not discard results.
            durations = [row["completionObservedSeconds"] for row in measurements]
            report = {
                "measurement": "HTTP POST to observed completed polling response; includes polling delay",
                "workload": "repeated fixed integration request, not representative production traffic",
                "runs": measurements,
                "p50Seconds": statistics.median(durations),
                "minSeconds": min(durations),
                "maxSeconds": max(durations),
                "productionP95Established": False,
            }
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            print(json.dumps({"run": index + 1, **{k: v for k, v in measured.items() if k != "result"}}), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:8080")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--timeout", type=float, default=600)
    args = parser.parse_args()
    if args.runs < 1 or args.timeout <= 0:
        parser.error("runs and timeout must be positive")
    asyncio.run(run(args))


if __name__ == "__main__":
    main()
