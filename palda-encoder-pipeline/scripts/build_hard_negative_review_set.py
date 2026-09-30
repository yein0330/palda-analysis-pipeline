#!/usr/bin/env python
"""Extract an isolated, model-ranked hard-negative human review set."""

from __future__ import annotations

import argparse
import asyncio
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from palda_runtime.models import ServiceEncoderModel


REVIEW_FP_THEMES = {
    "PRICE", "USABILITY", "QUALITY", "MATERIAL", "DURABILITY", "COLOR",
    "COMFORT", "ASSEMBLY",
}
QNA_FP_THEMES = {"OTHER", "UNUSABLE", "CUSTOMER_SUPPORT", "FUNCTIONALITY", "RESERVATION"}


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw", type=Path, required=True)
    parser.add_argument("--adjudicated", type=Path, required=True)
    parser.add_argument("--release-dir", type=Path, required=True)
    parser.add_argument("--benchmark-dir", type=Path, required=True)
    parser.add_argument("--artifact-dir", type=Path, required=True)
    parser.add_argument("--review-taxonomy", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--review-limit", type=int, default=900)
    parser.add_argument("--qna-limit", type=int, default=600)
    parser.add_argument("--device", default="cuda:0")
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def excluded_ids(release: Path, benchmark: Path) -> set[str]:
    result: set[str] = set()
    for split in ("train", "validation", "test"):
        result.update(row["sample_id"] for row in read_jsonl(release / f"{split}.jsonl"))
    for split in ("simulation", "validation", "test"):
        result.update(row["sample_id"] for row in read_jsonl(benchmark / f"{split}.jsonl"))
    return result


async def predict(rows: list[dict[str, Any]], model: ServiceEncoderModel) -> dict[str, dict[str, Any]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[row["category_path"]].append(row)
    result = {}
    for category_path, values in sorted(groups.items()):
        request = {
            "product": {"productId": "hard-negative-pool", "categoryPath": category_path},
            "reviewSample": {"items": [
                {"id": row["sample_id"], "text": row["text"]}
                for row in values if row["record_type"] == "review"
            ]},
            "qnaSample": {"items": [
                {"id": row["sample_id"], "text": row["text"]}
                for row in values if row["record_type"] == "qna"
            ]},
        }
        for item in await model.classify(request):
            result[item["id"]] = item
    return result


def risk(row: dict[str, Any], prediction: dict[str, Any]) -> tuple[str, float]:
    if row["record_type"] == "review":
        labels = prediction.get("labels") or []
        targeted = [item for item in labels if item["themeId"] in REVIEW_FP_THEMES]
        if targeted:
            strongest = max(targeted, key=lambda item: item["confidence"])
            issue = f"review_fp_{strongest['themeId'].lower()}"
        else:
            issue = "review_high_cardinality"
        return issue, len(labels) + sum(float(item["confidence"]) for item in targeted)
    primary = prediction.get("primaryTheme") or {}
    primary_id = str(primary.get("themeId") or "OTHER")
    secondary = prediction.get("secondaryThemes") or []
    if primary_id in QNA_FP_THEMES:
        issue = f"qna_primary_{primary_id.lower()}"
    elif secondary:
        issue = "qna_secondary_overprediction"
    else:
        issue = "qna_low_confidence"
    confidence = float(primary.get("confidence", 0))
    return issue, len(secondary) + (1 - confidence)


def diversified(rows: list[dict[str, Any]], limit: int) -> list[dict[str, Any]]:
    buckets: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        buckets[(row["hard_negative_issue"], row["category_key"])].append(row)
    for values in buckets.values():
        values.sort(key=lambda row: (-row["hard_negative_score"], row["sample_id"]))
    keys = sorted(buckets)
    selected = []
    while keys and len(selected) < limit:
        remaining = []
        for key in keys:
            values = buckets[key]
            if values and len(selected) < limit:
                selected.append(values.pop(0))
            if values:
                remaining.append(key)
        keys = remaining
    return selected


def ui_record(
    row: dict[str, Any], prediction: dict[str, Any], category: dict[str, Any]
) -> dict[str, Any]:
    allowed = {item["canonicalThemeId"]: item for item in category["allowedThemes"]}
    if row["record_type"] == "review":
        review_labels = [{
            "themeId": item["themeId"],
            "categoryThemeId": allowed[item["themeId"]]["categoryThemeId"],
            "sentiment": item["sentiment"],
        } for item in prediction.get("labels", []) if item["themeId"] in allowed]
        primary = None; secondary = []
        confidence = max((float(item["confidence"]) for item in prediction.get("labels", [])), default=0.0)
    else:
        primary_raw = prediction.get("primaryTheme") or {"themeId": "OTHER", "confidence": 0.0}
        primary_id = primary_raw["themeId"] if primary_raw["themeId"] in allowed else "OTHER"
        primary = {"themeId": primary_id, "categoryThemeId": allowed[primary_id]["categoryThemeId"]}
        secondary = [{
            "themeId": item["themeId"],
            "categoryThemeId": allowed[item["themeId"]]["categoryThemeId"],
        } for item in prediction.get("secondaryThemes", []) if item["themeId"] in allowed and item["themeId"] != primary_id]
        review_labels = []; confidence = float(primary_raw.get("confidence", 0))
    return {
        "schemaVersion": "encoder-train-v1", "taxonomyVersion": "theme-v1.0.0",
        "itemId": row["sample_id"], "productGroupId": str(row["product_id"]),
        "sourceType": row["record_type"],
        "category": {
            "level1Id": category["level1Id"], "level1Name": category["level1Name"],
            "level2Id": category["level2Id"], "level2Name": category["level2Name"],
            "path": row["category_path"],
        },
        "text": row["text"], "createdAt": (row.get("metadata") or {}).get("created_at"),
        "usable": bool(prediction.get("usable", True)),
        "labels": {"reviewLabels": review_labels, "qnaPrimaryTheme": primary, "qnaSecondaryThemes": secondary},
        "annotation": {
            "grade": "silver", "source": "encoder-hard-negative-v1", "sourceSplit": "train",
            "modelConfidence": f"{confidence:.4f}", "validationStatus": "needs_review",
            "reviewerStatus": "pending", "sampleWeight": 1.0,
        },
        "hardNegative": {
            "issue": row["hard_negative_issue"], "score": round(row["hard_negative_score"], 6),
            "poolStatus": row["pool_status"],
        },
    }


def main() -> None:
    args = arguments(); args.output_dir.mkdir(parents=True, exist_ok=True)
    raw = {row["sample_id"]: row for row in read_jsonl(args.raw)}
    adjudicated = {row["sample_id"]: row for row in read_jsonl(args.adjudicated)}
    excluded = excluded_ids(args.release_dir, args.benchmark_dir)
    candidates = []
    for sample_id, decision in adjudicated.items():
        if decision["status"] == "accepted" or sample_id in excluded or sample_id not in raw:
            continue
        candidates.append({**raw[sample_id], "pool_status": decision["status"]})
    model = ServiceEncoderModel(args.artifact_dir, args.artifact_dir / "taxonomy.json", device=args.device)
    predictions = asyncio.run(predict(candidates, model))
    ranked = []
    for row in candidates:
        issue, score = risk(row, predictions[row["sample_id"]])
        ranked.append({**row, "hard_negative_issue": issue, "hard_negative_score": score})
    selected = [
        *diversified([row for row in ranked if row["record_type"] == "review"], args.review_limit),
        *diversified([row for row in ranked if row["record_type"] == "qna"], args.qna_limit),
    ]
    taxonomy = json.loads(args.review_taxonomy.read_text(encoding="utf-8"))
    by_path = {row["path"]: row for row in taxonomy["categories"]}
    records = [
        ui_record(row, predictions[row["sample_id"]], by_path[row["category_key"]])
        for row in selected
    ]
    dataset = args.output_dir / "hard-negative-review-set.jsonl"
    dataset.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in records), encoding="utf-8")
    manifest = {
        "schemaVersion": "palda-hard-negative-review-set-v1",
        "records": len(records), "review": sum(row["sourceType"] == "review" for row in records),
        "qna": sum(row["sourceType"] == "qna" for row in records),
        "products": len({row["productGroupId"] for row in records}),
        "cat1": len({row["category"]["level1Id"] for row in records}),
        "cat2": len({row["category"]["level2Id"] for row in records}),
        "issues": dict(Counter(row["hardNegative"]["issue"] for row in records)),
        "isolation": {
            "excludedReleaseAndBenchmarkIds": len(excluded),
            "selectedOverlap": sum(row["itemId"] in excluded for row in records),
        },
        "dataset": str(dataset),
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
