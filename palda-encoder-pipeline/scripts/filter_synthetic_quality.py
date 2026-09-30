#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path

from palda_encoder.io import iter_jsonl, write_json, write_jsonl


def _compact(value: str) -> str:
    return re.sub(r"[^0-9a-z가-힣]+", "", value.lower())


def reasons(row: dict[str, object]) -> list[str]:
    text = str(row.get("text") or "").strip()
    compact = _compact(text)
    target = dict(row.get("metadata") or {}).get("generation_target") or {}
    theme_ids = [str(target.get("theme_id") or "")] + [
        str(value) for value in target.get("secondary_theme_ids", [])
    ]
    result = []
    if len(text) < 8:
        result.append("too_short")
    if len(text) > 500:
        result.append("too_long")
    if len(re.findall(r"[가-힣]", text)) < 3:
        result.append("insufficient_korean")
    if any(marker in text.lower() for marker in ("theme_id", "required_theme", "테마 id", "테마:", "테마는")):
        result.append("meta_instruction_leak")
    for theme_id in theme_ids:
        normalized_theme = _compact(theme_id)
        if len(normalized_theme) >= 5 and normalized_theme in compact:
            result.append(f"label_shortcut:{theme_id}")
    return sorted(set(result))


def main() -> None:
    parser = argparse.ArgumentParser(description="Filter generated samples with label or prompt leakage")
    parser.add_argument("--samples", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    accepted = []
    rejected = []
    seen = set()
    for row in iter_jsonl(args.samples):
        row_reasons = reasons(row)
        identity = _compact(str(row.get("text") or ""))
        if identity in seen:
            row_reasons.append("duplicate_text")
        if row_reasons:
            rejected.append({"sample_id": row.get("sample_id"), "reasons": sorted(set(row_reasons))})
            continue
        seen.add(identity)
        accepted.append(row)
    write_jsonl(args.output, accepted)
    counts = Counter(reason for row in rejected for reason in row["reasons"])
    report = {
        "input": len(accepted) + len(rejected),
        "accepted": len(accepted),
        "rejected": len(rejected),
        "reason_counts": dict(sorted(counts.items())),
        "rejections": rejected,
    }
    write_json(args.report, report)
    print(json.dumps({key: value for key, value in report.items() if key != "rejections"}, ensure_ascii=False))


if __name__ == "__main__":
    main()
