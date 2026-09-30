#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Recover source samples by consensus status")
    parser.add_argument("--samples", type=Path, required=True)
    parser.add_argument("--consensus", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--status", action="append", choices=("accepted", "review_needed", "insufficient_votes"),
        required=True,
    )
    args = parser.parse_args()
    selected_ids = set()
    with args.consensus.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                row = json.loads(line)
                if row.get("status") in args.status:
                    selected_ids.add(str(row["sample_id"]))
    selected = []
    with args.samples.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                row = json.loads(line)
                if str(row["sample_id"]) in selected_ids:
                    selected.append(row)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in selected),
        encoding="utf-8",
    )
    print(json.dumps({"selected": len(selected), "status": args.status}, ensure_ascii=False))


if __name__ == "__main__":
    main()
