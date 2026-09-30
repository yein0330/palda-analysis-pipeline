from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any

from .io import iter_jsonl, write_json, write_jsonl


def export_service_payloads(
    selected_summaries_path: str | Path,
    selected_insights_path: str | Path,
    output_path: str | Path,
) -> dict[str, int]:
    payloads: dict[tuple[str, str], dict[str, Any]] = {}
    qna_totals: dict[tuple[str, str], int] = defaultdict(int)
    for row in iter_jsonl(selected_summaries_path):
        if row.get("status") != "accepted":
            continue
        task, completion = row["task"], row["completion"]
        key = (task["analysis_id"], task["category_key"])
        payload = payloads.setdefault(key, _empty_payload(*key))
        item = {
            "theme_id": task["theme_id"],
            "theme_name": task.get("theme_name") or task["theme_id"],
            "count": task["group_count"],
            "percentage": task["percentage"],
            "quote": completion["quote"],
            "summary": completion["summary"],
        }
        if task["task_type"] == "review_summary":
            sentiment = str(task.get("sentiment"))
            if sentiment in payload["reviews"]:
                payload["reviews"][sentiment].append(item)
        else:
            payload["qna"]["themes"].append(item)
            qna_totals[key] = max(qna_totals[key], int(task.get("total_count", 0)))

    for row in iter_jsonl(selected_insights_path):
        if row.get("status") != "accepted":
            continue
        task = row["task"]
        key = (task["analysis_id"], task["category_key"])
        payload = payloads.setdefault(key, _empty_payload(*key))
        payload["selling_points"] = row["completion"].get("selling_points", [])

    output = []
    for key, payload in sorted(payloads.items()):
        payload["qna"]["total_count"] = qna_totals[key]
        for sentiment in ("positive", "negative"):
            payload["reviews"][sentiment].sort(key=lambda item: (-item["count"], item["theme_id"]))
        payload["qna"]["themes"].sort(key=lambda item: (-item["percentage"], item["theme_id"]))
        output.append(payload)
    write_jsonl(output_path, output)
    report = {"analyses": len(output)}
    write_json(Path(output_path).with_suffix(".report.json"), report)
    return report


def _empty_payload(analysis_id: str, category_key: str) -> dict[str, Any]:
    return {
        "analysis_id": analysis_id,
        "category_key": category_key,
        "reviews": {"positive": [], "negative": []},
        "qna": {"total_count": 0, "themes": []},
        "selling_points": [],
    }

