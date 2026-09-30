from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any

from .io import iter_jsonl, write_json, write_jsonl


def select_candidates(
    tasks_path: str | Path,
    candidates_path: str | Path,
    output_path: str | Path,
    *,
    judgments_path: str | Path | None = None,
    min_judges: int = 1,
    min_score: float = 0.65,
) -> dict[str, int]:
    tasks = {row["task_id"]: row for row in iter_jsonl(tasks_path)}
    candidates: dict[tuple[str, str], dict[str, Any]] = {
        (row["task_id"], row["model_id"]): row
        for row in iter_jsonl(candidates_path)
        if row.get("valid")
    }
    judgments: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    if judgments_path and Path(judgments_path).exists():
        for row in iter_jsonl(judgments_path):
            if row.get("valid"):
                judgments[(row["task_id"], row["candidate_model_id"])].append(row)

    output: list[dict[str, Any]] = []
    for task_id, task in sorted(tasks.items()):
        ranked: list[tuple[float, str, dict[str, Any], int, bool]] = []
        for (candidate_task_id, model_id), candidate in candidates.items():
            if candidate_task_id != task_id:
                continue
            votes = judgments.get((task_id, model_id), [])
            if votes:
                # A hallucination flag disqualifies the candidate. Dropping the
                # flagged vote instead would raise the average of the remaining
                # judges, rewarding the candidate for being caught.
                hallucinated = any(vote["scores"].get("hallucination") for vote in votes)
                score = 0.0 if hallucinated else sum(
                    sum(float(vote["scores"][name]) for name in ("grounding", "coverage", "specificity", "style")) / 4
                    for vote in votes
                ) / len(votes)
                judge_count = len(votes)
            else:
                score, judge_count, hallucinated = 1.0, 0, False
            ranked.append((score, model_id, candidate, judge_count, hallucinated))
        ranked.sort(key=lambda item: (-item[0], item[1]))
        if not ranked:
            output.append({"task_id": task_id, "task": task, "status": "rejected", "reasons": ["no_valid_candidate"]})
            continue
        score, model_id, candidate, judge_count, hallucinated = ranked[0]
        reasons = []
        if judgments_path and judge_count < min_judges:
            reasons.append("too_few_judges")
        if hallucinated:
            reasons.append("hallucination_flagged")
        if score < min_score:
            reasons.append("quality_score_below_minimum")
        output.append(
            {
                "task_id": task_id,
                "task": task,
                "status": "accepted" if not reasons else "rejected",
                "completion": candidate["completion"],
                "generator_model_id": model_id,
                "quality_score": round(score, 6),
                "judge_count": judge_count,
                "reasons": reasons,
            }
        )
    write_jsonl(output_path, output)
    report = {
        "tasks": len(output),
        "accepted": sum(row["status"] == "accepted" for row in output),
        "rejected": sum(row["status"] != "accepted" for row in output),
    }
    write_json(Path(output_path).with_suffix(".report.json"), report)
    return report

