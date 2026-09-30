from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

from .io import iter_jsonl, write_json
from .rewards import concise_reward, format_reward, grounding_reward


def audit_release(release_dir: str | Path, output_path: str | Path | None = None) -> dict[str, Any]:
    root = Path(release_dir)
    rows = {
        split: list(iter_jsonl(root / f"{split}.jsonl"))
        for split in ("train", "validation", "test")
    }
    analyses = {
        split: {str(row["analysis_id"]) for row in values}
        for split, values in rows.items()
    }
    example_ids = {
        split: {str(row["example_id"]) for row in values}
        for split, values in rows.items()
    }
    leakage = {}
    for left, right in (("train", "validation"), ("train", "test"), ("validation", "test")):
        leakage[f"analysis_{left}_{right}"] = len(analyses[left] & analyses[right])
        leakage[f"example_{left}_{right}"] = len(example_ids[left] & example_ids[right])

    quality = {}
    for split, values in rows.items():
        completions = [row["reference_completion"] for row in values]
        task_types = [row["task_type"] for row in values]
        format_values = format_reward(completions, task_types)
        grounding_values = grounding_reward(
            completions,
            task_types,
            [row.get("source_texts", []) for row in values],
            [row.get("allowed_theme_ids", []) for row in values],
        )
        concise_values = concise_reward(completions, task_types)
        quality[split] = {
            "examples": len(values),
            "format_pass_rate": _mean(format_values),
            "grounding_pass_rate": _mean(grounding_values),
            "concise_pass_rate": _mean(concise_values),
            "format_failures": sum(value < 1 for value in format_values),
            "grounding_failures": sum(value < 1 for value in grounding_values),
            "concise_failures": sum(value < 1 for value in concise_values),
        }
    report = {
        "schema": "palda-decoder-release-audit-v1",
        "release_dir": str(root.resolve()),
        "counts": {split: len(values) for split, values in rows.items()},
        "analyses": {split: len(values) for split, values in analyses.items()},
        "leakage": leakage,
        "quality": quality,
        "files": {
            split: {
                "path": f"{split}.jsonl",
                "sha256": _sha256(root / f"{split}.jsonl"),
            }
            for split in rows
        },
        "hard_gate_passed": not any(leakage.values())
        and all(
            result["format_failures"] == 0 and result["grounding_failures"] == 0
            for result in quality.values()
        ),
    }
    if output_path:
        write_json(output_path, report)
    return report


def _mean(values: list[float]) -> float:
    return round(sum(values) / max(len(values), 1), 6)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
