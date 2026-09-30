#!/usr/bin/env python3
"""Measure how much of what the decoder actually reads carries the right theme.

Raw encoder micro precision is not what reaches the decoder. The runtime keeps
only the top review groups per sentiment and the top Q&A groups by count, then
shows each group a bounded evidence list. This script scores those evidence
items against the human Gold labels recorded in the same capture trace, so the
number describes the decoder's real input rather than the classifier's full
output.

Reads capture traces written by `capture_pipeline_examples.py` or
`capture_gold_oracle_examples.py`. Gold labels are used for scoring only.
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any


def _gold_sets(labels: list[dict[str, Any]]) -> tuple[set[tuple[str, str]], set[str], str | None]:
    exact = {
        (str(label["theme_id"]), str(label["sentiment"]))
        for label in labels
        if label.get("sentiment") in ("positive", "negative")
    }
    themes = {str(label["theme_id"]) for label in labels}
    primary = next(
        (str(label["theme_id"]) for label in labels if label.get("role") == "primary"),
        str(labels[0]["theme_id"]) if labels else None,
    )
    return exact, themes, primary


def score_trace(trace: dict[str, Any]) -> dict[str, Any]:
    gold = {key: _gold_sets(value) for key, value in trace.get("gold", {}).items()}
    decoder_input = trace.get("decoderInput") or {}
    review_hits = review_theme_hits = review_total = 0
    qna_hits = qna_theme_hits = qna_total = 0
    groups: list[dict[str, Any]] = []

    for group in decoder_input.get("reviewGroups", []):
        theme, sentiment = str(group["themeId"]), str(group["sentiment"])
        evidence = group.get("evidence", [])
        exact = sum(1 for item in evidence if (theme, sentiment) in gold.get(str(item["id"]), (set(), set(), None))[0])
        theme_only = sum(1 for item in evidence if theme in gold.get(str(item["id"]), (set(), set(), None))[1])
        review_hits += exact
        review_theme_hits += theme_only
        review_total += len(evidence)
        groups.append({
            "kind": "review", "themeId": theme, "sentiment": sentiment,
            "mentionCount": group.get("mentionCount"), "evidence": len(evidence),
            "exact_matches": exact, "theme_matches": theme_only,
        })

    for group in decoder_input.get("qnaGroups", []):
        theme = str(group["themeId"])
        evidence = group.get("evidence", [])
        primary = sum(1 for item in evidence if gold.get(str(item["id"]), (set(), set(), None))[2] == theme)
        anywhere = sum(1 for item in evidence if theme in gold.get(str(item["id"]), (set(), set(), None))[1])
        qna_hits += primary
        qna_theme_hits += anywhere
        qna_total += len(evidence)
        groups.append({
            "kind": "qna", "themeId": theme, "sentiment": None,
            "mentionCount": group.get("primaryCount"), "evidence": len(evidence),
            "exact_matches": primary, "theme_matches": anywhere,
        })

    return {
        "caseId": trace.get("caseId"),
        "variant": trace.get("variant"),
        "encoder": trace.get("encoder"),
        "review_group_precision_exact": review_hits / review_total if review_total else None,
        "review_group_precision_theme_only": review_theme_hits / review_total if review_total else None,
        "review_evidence_items": review_total,
        "qna_group_precision_primary": qna_hits / qna_total if qna_total else None,
        "qna_group_precision_theme_anywhere": qna_theme_hits / qna_total if qna_total else None,
        "qna_evidence_items": qna_total,
        "selling_points": len((trace.get("decoderOutput") or {}).get("sellingPoints") or []),
        "groups": groups,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("traces", nargs="+", help="Capture trace JSON files or directories")
    parser.add_argument("--output")
    args = parser.parse_args()

    paths: list[Path] = []
    for value in args.traces:
        path = Path(value)
        paths.extend(sorted(path.glob("*-product-*.json")) if path.is_dir() else [path])

    results = []
    for path in paths:
        trace = json.loads(path.read_text(encoding="utf-8"))
        if trace.get("stage") != "completed":
            continue
        results.append(score_trace(trace) | {"path": str(path)})

    by_variant: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for result in results:
        by_variant[str(result["variant"])].append(result)

    summary = {}
    for variant, rows in sorted(by_variant.items()):
        def ratio(hit_key: str, total_key: str) -> float | None:
            total = sum(row[total_key] for row in rows)
            if not total:
                return None
            return sum(row[hit_key] * row[total_key] for row in rows if row[hit_key] is not None) / total
        summary[variant] = {
            "cases": len(rows),
            "review_group_precision_exact": ratio("review_group_precision_exact", "review_evidence_items"),
            "review_group_precision_theme_only": ratio("review_group_precision_theme_only", "review_evidence_items"),
            "review_evidence_items": sum(row["review_evidence_items"] for row in rows),
            "qna_group_precision_primary": ratio("qna_group_precision_primary", "qna_evidence_items"),
            "qna_group_precision_theme_anywhere": ratio("qna_group_precision_theme_anywhere", "qna_evidence_items"),
            "qna_evidence_items": sum(row["qna_evidence_items"] for row in rows),
            "selling_points": sum(row["selling_points"] for row in rows),
        }

    output = {"summary": summary, "cases": results}
    print(json.dumps(output, ensure_ascii=False, indent=2))
    if args.output:
        Path(args.output).write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
