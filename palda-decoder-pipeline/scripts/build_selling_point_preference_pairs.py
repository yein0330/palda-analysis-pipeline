#!/usr/bin/env python3
"""Build (chosen, rejected) selling-point pairs for DPO from judged candidates.

Two judgment formats are accepted:

* absolute — the current `palda-decoder judge` output, one row per
  (task, candidate, judge) with grounding/coverage/specificity/style plus a
  hallucination flag. A pair is emitted only when the mean score gap clears
  `--min-margin`, because small gaps between two near-identical outputs teach
  the model a preference the judges did not actually express.
* pairwise — `{"task_id", "winner_model_id", "loser_model_id", "judge_model_id"}`,
  the format the comparison-based evaluation proposal produces. Pairs are
  emitted on judge majority, which needs no margin threshold at all.

A hallucination-flagged candidate is never the chosen side, whatever it scored.

Training is out of scope: this writes the dataset and a report, nothing else.
"""
from __future__ import annotations

import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

SCORES = ("grounding", "coverage", "specificity", "style")


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def _prompt_messages(task: dict[str, Any], system_prompt: str, instruction: str) -> list[dict[str, str]]:
    """Mirror the release prompt so DPO sees the same input as SFT."""
    return [
        {"role": "system", "content": system_prompt},
        {
            "role": "user",
            "content": f"{instruction}\nINPUT={json.dumps(task, ensure_ascii=False)}",
        },
    ]


def _absolute_preferences(
    judgments: list[dict[str, Any]], *, min_judges: int, min_margin: float
) -> dict[str, dict[str, Any]]:
    """Mean score and hallucination status per (task, candidate)."""
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in judgments:
        if row.get("valid"):
            grouped[(row["task_id"], row["candidate_model_id"])].append(row)
    summary: dict[tuple[str, str], dict[str, Any]] = {}
    for key, votes in grouped.items():
        summary[key] = {
            "score": sum(sum(float(vote["scores"][name]) for name in SCORES) / 4 for vote in votes) / len(votes),
            "hallucinated": any(vote["scores"].get("hallucination") for vote in votes),
            "judges": len(votes),
        }
    by_task: dict[str, dict[str, Any]] = defaultdict(dict)
    for (task_id, model_id), value in summary.items():
        by_task[task_id][model_id] = value
    return {
        task_id: {"mode": "absolute", "candidates": values, "min_judges": min_judges, "min_margin": min_margin}
        for task_id, values in by_task.items()
    }


def _pairwise_preferences(judgments: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    wins: dict[str, Counter] = defaultdict(Counter)
    seen: dict[str, set[str]] = defaultdict(set)
    for row in judgments:
        if not row.get("valid", True):
            continue
        winner, loser = row.get("winner_model_id"), row.get("loser_model_id")
        if not winner or not loser or winner == loser:
            continue
        wins[row["task_id"]][winner] += 1
        seen[row["task_id"]].update({winner, loser})
    return {
        task_id: {"mode": "pairwise", "wins": counter, "models": seen[task_id]}
        for task_id, counter in wins.items()
    }


def _choose(task_id: str, preference: dict[str, Any], available: set[str]) -> tuple[str | None, str | None, str]:
    if preference["mode"] == "pairwise":
        models = sorted(preference["models"] & available)
        if len(models) < 2:
            return None, None, "fewer_than_two_judged_candidates"
        ranked = sorted(models, key=lambda model: (-preference["wins"][model], model))
        best, worst = ranked[0], ranked[-1]
        if preference["wins"][best] == preference["wins"][worst]:
            return None, None, "judges_tied"
        return best, worst, "ok"

    candidates = {model: value for model, value in preference["candidates"].items() if model in available}
    if len(candidates) < 2:
        return None, None, "fewer_than_two_judged_candidates"
    if any(value["judges"] < preference["min_judges"] for value in candidates.values()):
        return None, None, "too_few_judges"
    clean = {model: value for model, value in candidates.items() if not value["hallucinated"]}
    if not clean:
        return None, None, "all_candidates_hallucinated"
    best = max(clean, key=lambda model: (clean[model]["score"], model))
    worst = min(candidates, key=lambda model: (candidates[model]["score"], model))
    if best == worst:
        return None, None, "single_distinct_candidate"
    # A flagged candidate is a valid rejected side even when it scored well.
    if candidates[worst]["hallucinated"]:
        return best, worst, "ok"
    if candidates[best]["score"] - candidates[worst]["score"] < preference["min_margin"]:
        return None, None, "margin_below_minimum"
    return best, worst, "ok"


def build_pairs(
    tasks_path: Path,
    candidates_path: Path,
    judgments_path: Path,
    output_path: Path,
    *,
    mode: str = "absolute",
    min_judges: int = 2,
    min_margin: float = 0.1,
    system_prompt: str,
    instruction: str,
    seed: int = 20260923,
) -> dict[str, Any]:
    tasks = {row["task_id"]: row for row in _read_jsonl(tasks_path) if row.get("task_type") == "selling_point"}
    candidates: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in _read_jsonl(candidates_path):
        if row.get("valid") and row.get("task_type") == "selling_point" and row["task_id"] in tasks:
            candidates[row["task_id"]][row["model_id"]] = row
    judgments = _read_jsonl(judgments_path)
    preferences = (
        _pairwise_preferences(judgments)
        if mode == "pairwise"
        else _absolute_preferences(judgments, min_judges=min_judges, min_margin=min_margin)
    )

    random.Random(seed).random()  # fixed for reproducibility; order below is deterministic
    pairs: list[dict[str, Any]] = []
    skipped: Counter = Counter()
    for task_id in sorted(tasks):
        available = set(candidates.get(task_id, {}))
        if len(available) < 2:
            skipped["fewer_than_two_valid_candidates"] += 1
            continue
        preference = preferences.get(task_id)
        if preference is None:
            skipped["no_judgments"] += 1
            continue
        chosen_model, rejected_model, reason = _choose(task_id, preference, available)
        if chosen_model is None or rejected_model is None:
            skipped[reason] += 1
            continue
        task = tasks[task_id]
        pairs.append(
            {
                "task_id": task_id,
                "analysis_id": task.get("analysis_id"),
                "category_key": task.get("category_key"),
                "prompt": _prompt_messages(task, system_prompt, instruction),
                "chosen": json.dumps(candidates[task_id][chosen_model]["completion"], ensure_ascii=False),
                "rejected": json.dumps(candidates[task_id][rejected_model]["completion"], ensure_ascii=False),
                "chosen_model_id": chosen_model,
                "rejected_model_id": rejected_model,
                "preference_mode": mode,
            }
        )
    _write_jsonl(output_path, pairs)
    report = {
        "mode": mode,
        "tasks": len(tasks),
        "pairs": len(pairs),
        "skipped": dict(skipped),
        "min_judges": min_judges,
        "min_margin": min_margin,
        "chosen_model_counts": dict(Counter(pair["chosen_model_id"] for pair in pairs)),
        "rejected_model_counts": dict(Counter(pair["rejected_model_id"] for pair in pairs)),
        "seed": seed,
    }
    output_path.with_suffix(".report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tasks", required=True)
    parser.add_argument("--candidates", required=True)
    parser.add_argument("--judgments", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--mode", choices=("absolute", "pairwise"), default="absolute")
    parser.add_argument("--min-judges", type=int, default=2)
    parser.add_argument(
        "--min-margin", type=float, default=0.1,
        help="Absolute mode only: smallest mean score gap that counts as a preference.",
    )
    parser.add_argument("--seed", type=int, default=20260923)
    parser.add_argument(
        "--system-prompt",
        default="당신은 PALDA의 한국어 커머스 분석 모델입니다. 입력 근거만 사용하고, "
                "인용문은 원문에서 그대로 가져오며, 지정된 JSON 형식으로만 답하세요.",
    )
    parser.add_argument(
        "--instruction",
        default="다음 테마별 분석만 사용해 셀러가 활용할 수 있는 핵심 소구점 1~3개를 만드세요. "
                "과장하거나 새로운 사실을 추가하지 마세요.",
    )
    args = parser.parse_args()
    report = build_pairs(
        Path(args.tasks), Path(args.candidates), Path(args.judgments), Path(args.output),
        mode=args.mode, min_judges=args.min_judges, min_margin=args.min_margin,
        system_prompt=args.system_prompt, instruction=args.instruction, seed=args.seed,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
