#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json

from palda_encoder.benchmark import build_human_benchmark


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the isolated PALDA human benchmark")
    parser.add_argument("--source", required=True)
    parser.add_argument("--taxonomy", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--seed", type=int, default=20260904)
    parser.add_argument("--simulation-ratio", type=float, default=0.5)
    parser.add_argument("--validation-ratio", type=float, default=0.1)
    args = parser.parse_args()
    report = build_human_benchmark(
        args.source,
        args.taxonomy,
        args.output_dir,
        seed=args.seed,
        simulation_ratio=args.simulation_ratio,
        validation_ratio=args.validation_ratio,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
