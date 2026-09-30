#!/usr/bin/env python3
"""Convert structurally valid report-output-v1 drafts into decoder SFT examples."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any, Iterable


SYSTEM_PROMPT = (
    "당신은 PALDA의 한국어 커머스 분석 모델입니다. 입력 근거만 사용하고, "
    "인용문은 원문에서 그대로 가져오며, 지정된 JSON 형식으로만 답하세요."
)
POINT_TITLES = {
    "core": "핵심 소구점",
    "hidden": "숨은 강점",
    "barrier": "구매 장벽",
    "seller_action": "판매자 개선 제안",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def read_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL at {path}:{line_no}") from exc


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def stable_id(prefix: str, *parts: Any) -> str:
    raw = ":".join(str(part) for part in parts)
    return f"{prefix}_{hashlib.sha256(raw.encode()).hexdigest()[:20]}"


def category_key(path: str) -> str:
    return " > ".join(part.strip() for part in path.split(">") if part.strip())


def task_prompt(task: dict[str, Any]) -> str:
    if task["task_type"] == "selling_point":
        return (
            "다음 테마별 분석만 사용해 셀러가 활용할 수 있는 핵심 소구점 1~3개를 만드세요. "
            "과장하거나 새로운 사실을 추가하지 마세요.\n"
            f"INPUT={json.dumps(task, ensure_ascii=False, sort_keys=True)}\n"
            'OUTPUT={"selling_points":[{"title":"...","message":"...","evidence_theme_ids":["..."]}]}'
        )
    return (
        "다음 동일 테마 문장 묶음의 반복 패턴을 요약하세요. quote는 입력 원문의 정확한 부분 문자열이어야 합니다.\n"
        f"INPUT={json.dumps(task, ensure_ascii=False, sort_keys=True)}\n"
        f'OUTPUT={{"theme_id":"{task["theme_id"]}","sentiment":{json.dumps(task.get("sentiment"), ensure_ascii=False)},"quote":"...","summary":"..."}}'
    )


def make_example(task: dict[str, Any], completion: dict[str, Any], split: str) -> dict[str, Any]:
    source_texts = [item["text"] for item in task.get("source_items", [])]
    allowed = sorted({item["theme_id"] for item in task.get("summary_items", [])})
    encoded_completion = json.dumps(completion, ensure_ascii=False, sort_keys=True)
    return {
        "example_id": task["task_id"],
        "analysis_id": task["analysis_id"],
        "task_type": task["task_type"],
        "prompt": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": task_prompt(task)},
        ],
        "completion": [{"role": "assistant", "content": encoded_completion}],
        "source_texts": source_texts,
        "allowed_theme_ids": allowed,
        "reference_completion": encoded_completion,
        "quality_score": 1.0,
        "label_tier": "silver",
        "source_split": split,
        "source_schema": "report-output-v1",
    }


def evidence_items(group: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        {
            "sample_id": str(item["id"]),
            "text": str(item["text"]),
            "evidence": str(item["text"]),
            "agreement_score": 1.0,
        }
        for item in group.get("evidence", [])
        if item.get("id") and item.get("text")
    ]


def convert_product(row: dict[str, Any], split: str) -> list[dict[str, Any]]:
    report_input = row["reportInput"]
    report_output = row["reportOutputDraft"]
    product_id = str(row["productGroupId"])
    category = category_key(str((report_input.get("category") or {}).get("path") or ""))
    review_lookup = {
        (str(group["themeId"]), str(group["sentiment"])): group
        for group in report_input.get("reviewGroups", [])
    }
    qna_lookup = {
        str(group["themeId"]): group for group in report_input.get("qnaGroups", [])
    }
    review_total = int((report_input.get("sample") or {}).get("reviewSampleSize", 0))
    qna_total = int((report_input.get("sample") or {}).get("qnaSampleSize", 0))
    selected_summaries: list[tuple[dict[str, Any], dict[str, Any]]] = []
    review_output = report_output.get("reviewAnalysis") or {}
    for field, sentiment in (("positiveThemes", "positive"), ("negativeThemes", "negative")):
        for completion in review_output.get(field, []):
            theme_id = str(completion.get("themeId") or "")
            group = review_lookup.get((theme_id, sentiment))
            if not group:
                continue
            evidence = evidence_items(group)
            by_id = {item["sample_id"]: item["text"] for item in evidence}
            quote = by_id.get(str(completion.get("quoteEvidenceId")))
            if not quote:
                continue
            task = {
                "task_id": stable_id("sum", product_id, "review", theme_id, sentiment),
                "task_type": "review_summary",
                "analysis_id": product_id,
                "category_key": category,
                "theme_id": theme_id,
                "theme_name": completion.get("theme") or group.get("displayTheme") or theme_id,
                "sentiment": sentiment,
                "group_count": int(group.get("mentionCount", len(evidence))),
                "total_count": review_total,
                "percentage": round(int(group.get("mentionCount", 0)) / max(review_total, 1) * 100, 1),
                "source_items": evidence,
            }
            target = {
                "theme_id": theme_id,
                "sentiment": sentiment,
                "quote": quote,
                "summary": str(completion.get("summary") or ""),
            }
            selected_summaries.append((task, target))

    for completion in (report_output.get("qnaAnalysis") or {}).get("themes", []):
        theme_id = str(completion.get("themeId") or "")
        group = qna_lookup.get(theme_id)
        if not group:
            continue
        evidence = evidence_items(group)
        by_id = {item["sample_id"]: item["text"] for item in evidence}
        quote = by_id.get(str(completion.get("questionEvidenceId")))
        if not quote:
            continue
        task = {
            "task_id": stable_id("sum", product_id, "qna", theme_id, None),
            "task_type": "qna_summary",
            "analysis_id": product_id,
            "category_key": category,
            "theme_id": theme_id,
            "theme_name": completion.get("theme") or group.get("displayTheme") or theme_id,
            "sentiment": None,
            "group_count": int(group.get("primaryCount", len(evidence))),
            "total_count": qna_total,
            "percentage": round(float(group.get("ratio", 0.0)) * 100, 1),
            "source_items": evidence,
        }
        target = {
            "theme_id": theme_id,
            "sentiment": None,
            "quote": quote,
            "summary": str(completion.get("summary") or ""),
        }
        selected_summaries.append((task, target))

    examples = [make_example(task, target, split) for task, target in selected_summaries]
    summary_items = [
        {
            "task_type": task["task_type"],
            "theme_id": task["theme_id"],
            "sentiment": task.get("sentiment"),
            "percentage": task.get("percentage"),
            "quote": target["quote"],
            "summary": target["summary"],
        }
        for task, target in selected_summaries
    ]
    points = []
    allowed = {item["theme_id"] for item in summary_items}
    for point in report_output.get("sellingPoints", []):
        themes = {
            str(value).split("/", 1)[-1] for value in point.get("reviewThemes", [])
        } | {str(value) for value in point.get("qnaThemes", [])}
        themes &= allowed
        if not themes or not str(point.get("text") or "").strip():
            continue
        points.append(
            {
                "title": POINT_TITLES.get(str(point.get("type")), "상품 인사이트"),
                "message": str(point["text"]),
                "evidence_theme_ids": sorted(themes),
            }
        )
    if summary_items and points:
        task = {
            "task_id": stable_id("ins", product_id, category),
            "task_type": "selling_point",
            "analysis_id": product_id,
            "category_key": category,
            "summary_items": summary_items,
        }
        examples.append(make_example(task, {"selling_points": points[:3]}, split))
    return examples


def main() -> None:
    args = parse_args()
    source = args.source_dir.resolve()
    output = args.output_dir.resolve()
    split_rows: dict[str, list[dict[str, Any]]] = {}
    skipped = Counter()
    products: dict[str, set[str]] = {}
    for split in ("train", "validation", "test"):
        examples: list[dict[str, Any]] = []
        accepted_products: set[str] = set()
        for row in read_jsonl(source / f"{split}.drafts.jsonl"):
            if row.get("validationStatus") != "valid":
                skipped[f"{split}_needs_review"] += 1
                continue
            converted = convert_product(row, split)
            if not converted:
                skipped[f"{split}_no_examples"] += 1
                continue
            examples.extend(converted)
            accepted_products.add(str(row["productGroupId"]))
        split_rows[split] = sorted(examples, key=lambda item: item["example_id"])
        products[split] = accepted_products
        write_jsonl(output / f"{split}.jsonl", split_rows[split])
    write_jsonl(output / "grpo_train.jsonl", split_rows["train"])
    overlaps = {
        "train_validation": products["train"] & products["validation"],
        "train_test": products["train"] & products["test"],
        "validation_test": products["validation"] & products["test"],
    }
    if any(overlaps.values()):
        raise ValueError({name: len(values) for name, values in overlaps.items()})
    all_rows = [row for rows in split_rows.values() for row in rows]
    manifest = {
        "schema": "palda-decoder-pipeline-release-v1",
        "source": str(source),
        "counts": {split: len(rows) for split, rows in split_rows.items()},
        "products": {split: len(values) for split, values in products.items()},
        "task_types": dict(sorted(Counter(row["task_type"] for row in all_rows).items())),
        "label_tiers": {"silver": len(all_rows)},
        "skipped": dict(sorted(skipped.items())),
        "split_unit": "preserved_product_group_id",
        "leakage": {name: len(values) for name, values in overlaps.items()},
    }
    output.mkdir(parents=True, exist_ok=True)
    (output / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
