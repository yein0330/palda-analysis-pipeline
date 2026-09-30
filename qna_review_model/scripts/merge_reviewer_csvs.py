#!/usr/bin/env python
"""Validate and merge CSV exports returned by assigned reviewers."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--assignments-dir", type=Path, required=True)
    parser.add_argument("--reviewed-csv", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = arguments()
    expected: dict[str, set[str]] = {}
    for path in sorted(args.assignments_dir.glob("annotator-*/assignment.json")):
        row = json.loads(path.read_text(encoding="utf-8"))
        expected[row["reviewer_id"]] = set(row["annotation_ids"])
    merged: dict[str, dict[str, str]] = {}
    reports = {}
    fieldnames: list[str] | None = None
    for path in args.reviewed_csv:
        with path.open(encoding="utf-8-sig", newline="") as handle:
            rows = list(csv.DictReader(handle))
        if not rows:
            raise ValueError(f"empty reviewed CSV: {path}")
        reviewer_ids = {row.get("reviewer_id", "") for row in rows}
        if len(reviewer_ids) != 1:
            raise ValueError(f"CSV must contain exactly one reviewer_id: {path}")
        reviewer_id = reviewer_ids.pop()
        if reviewer_id not in expected:
            raise ValueError(f"unknown reviewer_id {reviewer_id!r}: {path}")
        ids = {row["annotation_id"] for row in rows}
        unexpected = ids - expected[reviewer_id]
        if unexpected:
            raise ValueError(f"{reviewer_id} contains {len(unexpected)} unassigned IDs")
        duplicates = ids & set(merged)
        if duplicates:
            raise ValueError(f"duplicate IDs across reviewer CSVs: {len(duplicates)}")
        merged.update((row["annotation_id"], row) for row in rows)
        statuses = Counter(row.get("reviewer_status") or "pending" for row in rows)
        reports[reviewer_id] = {
            "file": str(path), "rows": len(rows), "assigned": len(expected[reviewer_id]),
            "missing_rows": len(expected[reviewer_id] - ids), "statuses": dict(statuses),
        }
        fieldnames = list(rows[0])
    if fieldnames is None:
        raise ValueError("no rows to merge")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader(); writer.writerows(merged[key] for key in sorted(merged))
    all_expected = set().union(*expected.values())
    statuses = Counter(row.get("reviewer_status") or "pending" for row in merged.values())
    report = {
        "schema_version": "palda_reviewer_merge_report_v1",
        "expected_records": len(all_expected), "merged_records": len(merged),
        "missing_records": len(all_expected - set(merged)), "statuses": dict(statuses),
        "reviewed_records": sum(value for key, value in statuses.items() if key != "pending"),
        "reviewers": reports, "output": str(args.output),
    }
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
