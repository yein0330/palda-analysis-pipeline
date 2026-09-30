#!/usr/bin/env python3
"""Compare final selling points across capture arms, product by product.

Reads `<arm>-product-<n>.json` traces from one capture directory and writes a
JSON summary plus a Markdown side-by-side table. Every measure is mechanical;
none of them rates persuasiveness, which still needs a human or a judge.

* points: final selling points after runtime materialisation;
* overlap_pairs: pairs of points in one product whose evidence themes intersect;
* type_content_mismatch: core/hidden points whose text contains complaint
  phrasing (the defect `validate_completion` now rejects in the decoder repo);
* evidence precision: share of evidence items the decoder read that carry the
  group's Gold label (see decoder_input_group_precision.py).
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import re
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NEGATIVE = re.compile(r"불만|불편|부정적|문제가|문제점|부족|떨어지|떨어진|싫어|아쉽|아쉬|실망|개선이 필요|개선 필요")


def _group_precision():
    spec = importlib.util.spec_from_file_location(
        "decoder_input_group_precision", ROOT / "scripts/decoder_input_group_precision.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.score_trace


def _final_points(trace: dict) -> list[dict]:
    def find(value):
        if isinstance(value, dict):
            for key, item in value.items():
                if key == "sellingPoints":
                    return item
                found = find(item)
                if found is not None:
                    return found
        return None
    return find(trace.get("finalOutput")) or []


def _raw_points(trace: dict) -> list[dict]:
    return (trace.get("decoderOutput") or {}).get("sellingPoints") or []


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("capture_dir")
    parser.add_argument("--arms", nargs="+", default=["integration", "research", "gold_oracle"])
    args = parser.parse_args()
    score_trace = _group_precision()
    root = Path(args.capture_dir)

    by_case: dict[str, dict[str, dict]] = defaultdict(dict)
    for arm in args.arms:
        for path in sorted(root.glob(f"{arm}-product-*.json")):
            trace = json.loads(path.read_text())
            if trace.get("stage") == "completed":
                by_case[trace["caseId"]][arm] = trace
    cases = sorted(case for case, arms in by_case.items() if all(arm in arms for arm in args.arms))

    summary = {}
    for arm in args.arms:
        points = overlaps = mismatch = themes = raw_points = 0
        ev = {"review_hit": 0, "review_total": 0, "qna_hit": 0, "qna_total": 0}
        for case in cases:
            trace = by_case[case][arm]
            final = _final_points(trace)
            raw = _raw_points(trace)
            points += len(final)
            raw_points += len(raw)
            sets = [set(p.get("evidence_theme_ids") or p.get("evidenceThemeIds") or []) for p in raw]
            overlaps += sum(1 for a in range(len(sets)) for b in range(a + 1, len(sets)) if sets[a] & sets[b])
            themes += len(set().union(*sets)) if sets else 0
            mismatch += sum(1 for p in final if p.get("type") in ("core", "hidden") and NEGATIVE.search(p.get("text", "")))
            score = score_trace(trace)
            for kind in ("review", "qna"):
                total = score[f"{kind}_evidence_items"]
                precision = score["review_group_precision_exact" if kind == "review" else "qna_group_precision_primary"]
                ev[f"{kind}_total"] += total
                ev[f"{kind}_hit"] += round((precision or 0) * total)
        summary[arm] = {
            "cases": len(cases),
            "final_points": points,
            "raw_points": raw_points,
            "points_per_product": round(points / max(len(cases), 1), 3),
            "overlap_pairs": overlaps,
            "overlap_pairs_per_product": round(overlaps / max(len(cases), 1), 3),
            "themes_covered": themes,
            "type_content_mismatch": mismatch,
            "review_evidence_precision": round(ev["review_hit"] / max(ev["review_total"], 1), 6),
            "qna_evidence_precision": round(ev["qna_hit"] / max(ev["qna_total"], 1), 6),
        }

    lines = ["| 상품 | 카테고리 | " + " | ".join(args.arms) + " |",
             "|---|---|" + "---|" * len(args.arms)]
    for case in cases:
        category = by_case[case][args.arms[0]]["input"]["product"]["categoryPath"]
        cells = []
        for arm in args.arms:
            cells.append("<br>".join(
                f"[{p.get('type')}] {p.get('text', '').replace('|', '/')}" for p in _final_points(by_case[case][arm])
            ) or "(없음)")
        lines.append(f"| {case} | {category} | " + " | ".join(cells) + " |")
    (root / "selling_point_comparison.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (root / "selling_point_comparison.json").write_text(
        json.dumps({"summary": summary, "cases": cases}, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
