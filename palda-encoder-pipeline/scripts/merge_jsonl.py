#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Merge JSONL files with deterministic key deduplication")
    parser.add_argument("--input", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--key", action="append",
        help="Deduplication key. Repeat for a composite key (default: sample_id).",
    )
    args = parser.parse_args()
    keys = args.key or ["sample_id"]
    rows = {}
    for path in args.input:
        with path.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, 1):
                if not line.strip():
                    continue
                row = json.loads(line)
                missing = [key for key in keys if key not in row]
                if missing:
                    raise KeyError(f"{path}:{line_number} is missing {', '.join(missing)}")
                composite = tuple(str(row[key]) for key in keys)
                rows[composite] = row
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        "".join(json.dumps(rows[key], ensure_ascii=False, sort_keys=True) + "\n" for key in sorted(rows)),
        encoding="utf-8",
    )
    print(json.dumps({"rows": len(rows), "output": str(args.output)}))


if __name__ == "__main__":
    main()
