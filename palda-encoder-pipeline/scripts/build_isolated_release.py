#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json

from palda_encoder.release import create_isolated_release


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a simulation-isolated encoder release")
    parser.add_argument("--samples", required=True)
    parser.add_argument("--consensus", required=True)
    parser.add_argument("--benchmark-dir", required=True)
    parser.add_argument("--taxonomy", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    report = create_isolated_release(
        args.samples,
        args.consensus,
        args.benchmark_dir,
        args.output_dir,
        taxonomy_path=args.taxonomy,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
