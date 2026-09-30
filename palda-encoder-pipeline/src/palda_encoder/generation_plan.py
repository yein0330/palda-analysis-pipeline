from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any

from .io import iter_jsonl, write_json, write_jsonl


def plan_tail_generation(
    samples_path: str | Path,
    consensus_path: str | Path,
    output_path: str | Path,
    *,
    target_per_combination: int = 20,
    max_per_combination: int = 10,
    max_total: int = 1000,
    include_neutral: bool = False,
) -> dict[str, Any]:
    if target_per_combination < 1 or max_per_combination < 1 or max_total < 1:
        raise ValueError("Generation limits must be positive")
    samples = {row["sample_id"]: row for row in iter_jsonl(samples_path)}
    counts: Counter[tuple[str, str, str, str | None]] = Counter()
    observed: set[tuple[str, str, str]] = set()
    for consensus in iter_jsonl(consensus_path):
        if consensus.get("status") != "accepted":
            continue
        sample = samples.get(consensus["sample_id"])
        if not sample:
            continue
        record_type = str(sample["record_type"])
        category_key = str(sample.get("category_key") or sample.get("category_path") or "")
        for label in consensus.get("labels", []):
            theme_id = str(label["theme_id"])
            observed.add((record_type, category_key, theme_id))
            sentiment = label.get("sentiment") if record_type == "review" else None
            counts[(record_type, category_key, theme_id, sentiment)] += 1

    candidates: list[tuple[int, dict[str, Any]]] = []
    sentiments = ["positive", "negative"] + (["neutral"] if include_neutral else [])
    for record_type, category_key, theme_id in sorted(observed):
        target_sentiments: list[str | None] = sentiments if record_type == "review" else [None]
        for sentiment in target_sentiments:
            current = counts[(record_type, category_key, theme_id, sentiment)]
            deficit = max(target_per_combination - current, 0)
            if not deficit:
                continue
            candidates.append(
                (
                    current,
                    {
                        "record_type": record_type,
                        "category_key": category_key,
                        "category_path": category_key,
                        "theme_id": theme_id,
                        "sentiment": sentiment,
                        "difficulty": "hard" if current else "normal",
                        "constraints": "한 문장에 불필요한 테마를 추가하지 말 것",
                        "count": min(deficit, max_per_combination),
                        "current_count": current,
                        "target_count": target_per_combination,
                    },
                )
            )

    # Rarest cells are filled first. The cap prevents an uncontrolled synthetic-data explosion.
    candidates.sort(key=lambda item: (item[0], item[1]["record_type"], item[1]["category_key"], item[1]["theme_id"], str(item[1]["sentiment"])))
    plan: list[dict[str, Any]] = []
    remaining = max_total
    for _, row in candidates:
        if remaining <= 0:
            break
        row = dict(row)
        row["count"] = min(int(row["count"]), remaining)
        remaining -= int(row["count"])
        plan.append(row)
    write_jsonl(output_path, plan)
    report = {
        "observed_combinations": len(observed),
        "planned_cells": len(plan),
        "planned_samples": sum(int(row["count"]) for row in plan),
        "max_total": max_total,
        "target_per_combination": target_per_combination,
    }
    write_json(Path(output_path).with_suffix(".report.json"), report)
    return report
