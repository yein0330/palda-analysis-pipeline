#!/usr/bin/env python3
"""Sample 50 products for a blinded head-to-head human review of selling points.

Each sheet row shows one product's theme analysis and two selling-point outputs
as A and B. Side assignment is randomised per row from `--seed`, and the key
mapping A/B back to the producing model is written to a separate file so the
reviewer's copy carries no hint.

Products, not tasks, are the sampling unit, and the sample is spread over
category_key proportionally so one large category cannot fill the sheet.
"""
from __future__ import annotations

import argparse
import csv
import json
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def _stratified(groups: dict[str, list[str]], size: int, rng: random.Random) -> list[str]:
    """Largest-remainder allocation over categories, then a random draw inside each."""
    total = sum(len(values) for values in groups.values())
    if total <= size:
        return sorted(task for values in groups.values() for task in values)
    exact = {key: len(values) / total * size for key, values in groups.items()}
    quota = {key: int(value) for key, value in exact.items()}
    remaining = size - sum(quota.values())
    for key in sorted(groups, key=lambda item: (-(exact[item] - quota[item]), item))[:remaining]:
        quota[key] += 1
    selected: list[str] = []
    for key in sorted(groups):
        pool = sorted(groups[key])
        rng.shuffle(pool)
        selected.extend(pool[: quota[key]])
    return sorted(selected)


def _format_points(completion: dict[str, Any]) -> str:
    points = completion.get("selling_points") or []
    return "\n".join(
        f"{index}. {str(point.get('title') or '').strip()} — {str(point.get('message') or '').strip()} "
        f"[근거: {', '.join(point.get('evidence_theme_ids') or []) or '없음'}]"
        for index, point in enumerate(points, 1)
        if isinstance(point, dict)
    )


def _format_analysis(task: dict[str, Any]) -> str:
    return "\n".join(
        f"- {item.get('theme_id')} / {item.get('task_type')} / "
        f"{item.get('sentiment') or 'n/a'} / 비중 {item.get('percentage')}%: {item.get('summary')}"
        for item in task.get("summary_items", [])
    )


def build_sheet(
    tasks_path: Path,
    candidates_path: Path,
    output_dir: Path,
    *,
    size: int = 50,
    seed: int = 20260923,
    arm_a: str | None = None,
    arm_b: str | None = None,
) -> dict[str, Any]:
    tasks = {row["task_id"]: row for row in _read_jsonl(tasks_path) if row.get("task_type") == "selling_point"}
    by_task: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in _read_jsonl(candidates_path):
        if row.get("valid") and row.get("task_type") == "selling_point" and row["task_id"] in tasks:
            by_task[row["task_id"]][row["model_id"]] = row

    comparable: dict[str, tuple[str, str]] = {}
    for task_id, models in by_task.items():
        if arm_a and arm_b:
            if arm_a in models and arm_b in models:
                comparable[task_id] = (arm_a, arm_b)
            continue
        names = sorted(models)
        if len(names) >= 2:
            comparable[task_id] = (names[0], names[1])
    if not comparable:
        raise SystemExit("No task has two comparable selling-point candidates.")

    groups: dict[str, list[str]] = defaultdict(list)
    for task_id in comparable:
        groups[str(tasks[task_id].get("category_key") or "unknown")].append(task_id)
    rng = random.Random(seed)
    selected = _stratified(groups, size, rng)

    sheet: list[dict[str, Any]] = []
    key: list[dict[str, Any]] = []
    for task_id in selected:
        left, right = comparable[task_id]
        if rng.random() < 0.5:
            left, right = right, left
        task = tasks[task_id]
        sheet.append(
            {
                "row_id": f"{task_id}",
                "category_key": task.get("category_key"),
                "analysis_id": task.get("analysis_id"),
                "theme_analysis": _format_analysis(task),
                "option_A": _format_points(by_task[task_id][left]["completion"]),
                "option_B": _format_points(by_task[task_id][right]["completion"]),
                "better": "",
                "why": "",
            }
        )
        key.append({"row_id": task_id, "option_A_model_id": left, "option_B_model_id": right})

    output_dir.mkdir(parents=True, exist_ok=True)
    sheet_csv = output_dir / "human_eval_sheet.csv"
    with sheet_csv.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(sheet[0].keys()))
        writer.writeheader()
        writer.writerows(sheet)
    _write_jsonl(output_dir / "human_eval_sheet.jsonl", sheet)
    _write_jsonl(output_dir / "human_eval_key.jsonl", key)

    report = {
        "requested": size,
        "sampled": len(sheet),
        "comparable_tasks": len(comparable),
        "categories": len(groups),
        "per_category": dict(Counter(row["category_key"] for row in sheet)),
        "arms": {
            " vs ".join(pair): count
            for pair, count in Counter(
                tuple(sorted((item["option_A_model_id"], item["option_B_model_id"]))) for item in key
            ).items()
        },
        "option_A_model_counts": dict(Counter(item["option_A_model_id"] for item in key)),
        "seed": seed,
        "sheet_csv": str(sheet_csv),
        "note": "'better' 열에 A 또는 B를, 'why' 열에 한 문장 이유를 적는다. 키 파일은 채점 전까지 열지 않는다.",
    }
    (output_dir / "human_eval.report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tasks", required=True)
    parser.add_argument("--candidates", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--size", type=int, default=50)
    parser.add_argument("--seed", type=int, default=20260923)
    parser.add_argument("--arm-a", help="Model id for one side; defaults to the two lowest ids present.")
    parser.add_argument("--arm-b")
    args = parser.parse_args()
    report = build_sheet(
        Path(args.tasks), Path(args.candidates), Path(args.output_dir),
        size=args.size, seed=args.seed, arm_a=args.arm_a, arm_b=args.arm_b,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
