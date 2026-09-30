#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path


def read_rows(path: Path) -> dict[str, dict]:
    with path.open(encoding="utf-8") as handle:
        return {
            row["sample_id"]: row
            for line in handle
            if line.strip()
            for row in [json.loads(line)]
        }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Promote only high-agreement newly accepted consensus rows."
    )
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--min-models", type=int, default=3)
    parser.add_argument("--min-agreement", type=float, default=1.0)
    args = parser.parse_args()
    base = read_rows(args.base)
    candidate = read_rows(args.candidate)
    promoted = 0
    rows = []
    for sample_id in sorted(base):
        row = base[sample_id]
        proposed = candidate.get(sample_id)
        if (
            row.get("status") != "accepted"
            and proposed
            and proposed.get("status") == "accepted"
            and len(set(proposed.get("models") or [])) >= args.min_models
            and float(proposed.get("agreement_score", 0)) >= args.min_agreement
        ):
            row = proposed
            promoted += 1
        rows.append(row)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )
    print(json.dumps({"rows": len(rows), "promoted": promoted}))


if __name__ == "__main__":
    main()
