#!/usr/bin/env python3
"""Repair only the unambiguous missing role of a singleton Q&A label."""
import argparse
from copy import deepcopy
from pathlib import Path

from palda_encoder.io import iter_jsonl, write_json, write_jsonl
from compare_label_quality import summarize


def normalize(row):
    result = deepcopy(row)
    labels = result["labels"]
    result["raw_valid"] = row["valid"]
    result["role_repaired"] = False
    if (len(labels) == 1 and labels[0].get("role") is None
            and set(row.get("errors", [])) == {"requires_one_primary"}):
        labels[0]["role"] = "primary"
        result.update(valid=True, errors=[], role_repaired=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rows = [normalize(r) for r in iter_jsonl(args.input)]
    write_jsonl(args.output, rows)
    summary = summarize(rows)
    summary["singletonRoleRepairs"] = sum(r["role_repaired"] for r in rows)
    write_json(args.output.with_suffix(".summary.json"), summary)
    import json
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
