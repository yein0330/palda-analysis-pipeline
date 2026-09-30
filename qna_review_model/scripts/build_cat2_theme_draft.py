#!/usr/bin/env python
"""Merge raw Cat2 model outputs into a conservative draft theme dictionary."""

from __future__ import annotations

import argparse
import glob
import json
import re
from collections import Counter
from pathlib import Path


SERVICE_PATTERN = re.compile(
    r"delivery|shipping|packag|customer.?support|return|refund|exchange|courier|after.?sales|"
    r"배송|택배|포장|상담|고객.?지원|교환|반품|환불|배송기사|설치기사|\ba/?s\b",
    re.IGNORECASE,
)

GROUNDED_FALLBACKS = {
    "스포츠/레저 > 배구": [
        {
            "theme_id": "impact_softness", "label": "충격 완화감",
            "keywords": ["말랑", "푹신", "아프지", "질감"],
            "evidence_quote": "공이 푹신푹신해서 피구하기 너무 좋아요~맞아도 안 아프고요ㅎㅎㅎ",
        },
        {
            "theme_id": "shape_retention", "label": "형태 유지력",
            "keywords": ["흐물흐물", "겉피", "내용물", "늘어난"],
            "evidence_quote": "속 안에 내용물이 준건지 아니면 밖에 겉피가 늘어난건지 흐물흐물거려요",
        },
    ],
}

CURATED_NORMALIZATION = {
    ("디지털/가전 > 주변기기", "ergonomics"): {"label": "인체공학"},
    ("스포츠/레저 > 야구", "signed_ball"): {
        "theme_id": "collectibility", "label": "수집·사인 용도",
    },
    ("스포츠/레저 > 야구", "case_inclusion"): {
        "theme_id": "storage_case_inclusion", "label": "보관 케이스 구성",
    },
}


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", type=Path, required=True)
    parser.add_argument("--candidates", nargs="+", required=True)
    parser.add_argument("--base-themes", type=Path, required=True)
    parser.add_argument("--output-themes", type=Path, required=True)
    parser.add_argument("--output-review", type=Path, required=True)
    return parser.parse_args()


def extract(raw: str) -> dict | None:
    start, end = raw.find("{"), raw.rfind("}")
    if start < 0 or end < start:
        return None
    try:
        value = json.loads(raw[start:end + 1])
    except json.JSONDecodeError:
        return None
    return value if isinstance(value, dict) else None


def main() -> None:
    args = arguments()
    samples = {json.loads(line)["cat2_path"]: json.loads(line) for line in args.samples.read_text(encoding="utf-8").splitlines()}
    rows = []
    for pattern in args.candidates:
        for path in glob.glob(pattern):
            rows.extend(json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines())
    by_path = {row["cat2_path"]: row for row in rows}
    base = json.loads(args.base_themes.read_text(encoding="utf-8"))
    cat2: dict[str, list[dict]] = {}
    review_rows = []
    stats = Counter()
    for path in sorted(samples):
        sample = samples[path]; cat1 = sample["cat1"]
        row = by_path.get(path, {})
        payload = extract(str(row.get("raw_output") or ""))
        existing_rows = [*base.get("global", []), *base.get("cat1", {}).get(cat1, [])]
        existing_ids = {theme["theme_id"] for theme in existing_rows}
        existing_labels = {str(theme["label"]).replace("/", "").replace("·", "") for theme in existing_rows}
        source_texts = [str(record["text"]) for record in sample["records"]]
        accepted = []
        rejected = []
        seen = set()
        raw_themes = payload.get("themes", []) if payload else []
        for theme in raw_themes[:6]:
            if not isinstance(theme, dict):
                continue
            theme_id = str(theme.get("theme_id") or "").strip()
            label = str(theme.get("label") or "").strip()
            normalized = CURATED_NORMALIZATION.get((path, theme_id), {})
            theme_id = str(normalized.get("theme_id", theme_id))
            label = str(normalized.get("label", label))
            keywords = list(dict.fromkeys(
                str(value).strip() for value in (theme.get("keywords") or []) if str(value).strip()
            ))[:8]
            evidence = str(theme.get("evidence_quote") or "").strip()
            reasons = []
            if not re.fullmatch(r"[a-z][a-z0-9_]{1,39}", theme_id): reasons.append("invalid_id")
            if not label or len(keywords) < 2: reasons.append("incomplete")
            if theme_id in existing_ids: reasons.append("existing_id")
            if theme_id in seen: reasons.append("duplicate_id")
            compact_label = label.replace("/", "").replace("·", "")
            if compact_label in existing_labels: reasons.append("existing_label")
            if SERVICE_PATTERN.search(" ".join([theme_id, label, *keywords])): reasons.append("service_scope")
            evidence_exact = bool(evidence and any(evidence in text for text in source_texts))
            if not evidence_exact: stats["evidence_warning"] += 1
            candidate = {
                "theme_id": theme_id, "label": label, "keywords": keywords,
                "evidence_quote": evidence, "evidence_exact": evidence_exact,
            }
            if reasons:
                candidate["rejection_reasons"] = reasons; rejected.append(candidate)
                stats["rejected"] += 1
                continue
            seen.add(theme_id); accepted.append(candidate); stats["accepted"] += 1
        if not accepted and path in GROUNDED_FALLBACKS:
            for theme in GROUNDED_FALLBACKS[path]:
                accepted.append({**theme, "evidence_exact": any(
                    theme["evidence_quote"] in text for text in source_texts
                )})
                stats["accepted"] += 1
            stats["manual_grounded_fallback"] += 1
        cat2[path] = [
            {"theme_id": theme["theme_id"], "label": theme["label"], "keywords": theme["keywords"]}
            for theme in accepted
        ]
        review_rows.append({
            "schema_version": "palda_cat2_theme_review_v1",
            "cat1": cat1, "cat2_path": path,
            "inherited_themes": existing_rows, "accepted_themes": accepted,
            "rejected_themes": rejected, "sample_counts": sample["counts"],
            "sample_records": sample["records"],
            "review_status": "pending",
        })
        stats["cat2"] += 1
        stats["cat2_with_themes"] += bool(accepted)
    draft = dict(base)
    draft["version"] = 2
    draft["cat2"] = cat2
    draft["generation"] = {
        "status": "draft_pending_human_review", "source": "full_raw_crawl",
        "samples_per_cat2": 30, "cat2_count": len(cat2),
        "warning": "Cat2 themes are model-generated candidates and are not canonical until reviewed.",
    }
    args.output_themes.parent.mkdir(parents=True, exist_ok=True)
    args.output_themes.write_text(json.dumps(draft, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    with args.output_review.open("w", encoding="utf-8") as handle:
        for row in review_rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    report = {
        "schema_version": "palda_cat2_theme_draft_report_v1", **stats,
        "theme_count_distribution": dict(Counter(len(value) for value in cat2.values())),
        "output_themes": str(args.output_themes), "output_review": str(args.output_review),
    }
    args.output_review.with_suffix(".report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
