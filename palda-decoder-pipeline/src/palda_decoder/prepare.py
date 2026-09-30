from __future__ import annotations

import re
from collections import Counter, defaultdict
from pathlib import Path
from statistics import median
from typing import Any, Iterable

from .io import iter_jsonl, stable_id, write_json, write_jsonl


def prepare_summary_tasks(
    input_paths: Iterable[str | Path],
    output_path: str | Path,
    *,
    max_records_per_type: int = 5000,
    representatives_per_group: int = 10,
    min_group_size: int = 1,
    include_neutral: bool = False,
) -> dict[str, Any]:
    if max_records_per_type < 1 or representatives_per_group < 1 or min_group_size < 1:
        raise ValueError("Preparation limits must be positive")
    rows = [row for path in input_paths for row in iter_jsonl(path)]
    kept: list[dict[str, Any]] = []
    truncated: Counter[str] = Counter()
    by_scope_type: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        record_type = str(row.get("record_type") or "")
        if record_type not in {"review", "qna"}:
            continue
        scope_id = str(row.get("analysis_id") or row.get("product_id") or "default")
        by_scope_type[(scope_id, record_type)].append(row)
    for (_, record_type), scoped_rows in sorted(by_scope_type.items()):
        values = sorted(scoped_rows, key=lambda row: str(row.get("sample_id", "")))
        kept.extend(values[:max_records_per_type])
        truncated[record_type] += max(len(values) - max_records_per_type, 0)

    totals: Counter[tuple[str, str, str]] = Counter()
    grouped: dict[tuple[str, str, str, str, str | None], list[dict[str, Any]]] = defaultdict(list)
    for row in kept:
        scope_id = str(row.get("analysis_id") or row.get("product_id") or "default")
        category = str(row.get("category_key") or row.get("category_path") or "")
        record_type = str(row["record_type"])
        totals[(scope_id, category, record_type)] += 1
        for label in row.get("labels", []):
            theme_id = str(label.get("theme_id") or "").strip()
            if not theme_id:
                continue
            sentiment = label.get("sentiment") if record_type == "review" else None
            if record_type == "review" and sentiment not in {"positive", "negative", "neutral"}:
                continue
            if record_type == "review" and sentiment == "neutral" and not include_neutral:
                continue
            grouped[(scope_id, category, record_type, theme_id, sentiment)].append(
                {
                    "sample_id": str(row.get("sample_id") or ""),
                    "text": str(row.get("text") or "").strip(),
                    "evidence": label.get("evidence"),
                    "theme_name": label.get("theme_name") or theme_id,
                    "agreement_score": float(row.get("agreement_score", 0.0)),
                }
            )

    tasks: list[dict[str, Any]] = []
    for key, items in sorted(grouped.items(), key=lambda item: tuple(str(value) for value in item[0])):
        scope_id, category, record_type, theme_id, sentiment = key
        unique = _deduplicate(items)
        if len(unique) < min_group_size:
            continue
        selected = _select_representatives(unique, representatives_per_group)
        total = totals[(scope_id, category, record_type)]
        task_id = stable_id("sum", scope_id, category, record_type, theme_id, sentiment)
        tasks.append(
            {
                "task_id": task_id,
                "task_type": "review_summary" if record_type == "review" else "qna_summary",
                "analysis_id": scope_id,
                "category_key": category,
                "theme_id": theme_id,
                "theme_name": selected[0].get("theme_name") or theme_id,
                "sentiment": sentiment,
                "group_count": len(unique),
                "total_count": total,
                "percentage": round(len(unique) / max(total, 1) * 100, 1),
                "source_items": selected,
            }
        )
    write_jsonl(output_path, tasks)
    report = {
        "input_records": len(rows),
        "kept_records": len(kept),
        "truncated": dict(truncated),
        "tasks": len(tasks),
        "review_tasks": sum(task["task_type"] == "review_summary" for task in tasks),
        "qna_tasks": sum(task["task_type"] == "qna_summary" for task in tasks),
    }
    write_json(Path(output_path).with_suffix(".report.json"), report)
    return report


def prepare_insight_tasks(
    selected_summaries_path: str | Path,
    output_path: str | Path,
) -> dict[str, int]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in iter_jsonl(selected_summaries_path):
        if row.get("status") != "accepted":
            continue
        task = row["task"]
        grouped[(task["analysis_id"], task["category_key"])].append(
            {
                "task_type": task["task_type"],
                "theme_id": task["theme_id"],
                "sentiment": task.get("sentiment"),
                "percentage": task.get("percentage"),
                "quote": row["completion"]["quote"],
                "summary": row["completion"]["summary"],
            }
        )
    tasks = [
        {
            "task_id": stable_id("ins", analysis_id, category),
            "task_type": "selling_point",
            "analysis_id": analysis_id,
            "category_key": category,
            "summary_items": sorted(items, key=lambda item: (item["task_type"], item["theme_id"], str(item["sentiment"]))),
        }
        for (analysis_id, category), items in sorted(grouped.items())
    ]
    write_jsonl(output_path, tasks)
    report = {"tasks": len(tasks), "source_summaries": sum(len(task["summary_items"]) for task in tasks)}
    write_json(Path(output_path).with_suffix(".report.json"), report)
    return report


def _deduplicate(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for item in items:
        key = re.sub(r"\s+", "", item["text"]).lower()
        if key and (key not in result or item["agreement_score"] > result[key]["agreement_score"]):
            result[key] = item
    return list(result.values())


def _select_representatives(items: list[dict[str, Any]], limit: int) -> list[dict[str, Any]]:
    middle = median(len(item["text"]) for item in items)
    ranked = sorted(
        items,
        key=lambda item: (
            -float(item["agreement_score"]),
            0 if item.get("evidence") and item["evidence"] in item["text"] else 1,
            abs(len(item["text"]) - middle),
            item["sample_id"],
        ),
    )
    return ranked[:limit]
