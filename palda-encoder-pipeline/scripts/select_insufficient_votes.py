#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Select samples with too few distinct valid model votes")
    parser.add_argument("--samples", type=Path, required=True)
    parser.add_argument("--label-runs", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--minimum", type=int, default=2)
    parser.add_argument(
        "--missing-model",
        help="Select rows without a valid vote from this model instead of using --minimum.",
    )
    args = parser.parse_args()
    votes: dict[str, set[str]] = defaultdict(set)
    with args.label_runs.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                row = json.loads(line)
                if row.get("valid"):
                    votes[str(row["sample_id"])].add(str(row["model_id"]))
    selected = []
    with args.samples.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                row = json.loads(line)
                model_votes = votes[str(row["sample_id"])]
                if (
                    args.missing_model not in model_votes
                    if args.missing_model else len(model_votes) < args.minimum
                ):
                    selected.append(row)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in selected),
        encoding="utf-8",
    )
    print(json.dumps({
        "selected": len(selected), "minimum": args.minimum,
        "missing_model": args.missing_model,
    }))


if __name__ == "__main__":
    main()
