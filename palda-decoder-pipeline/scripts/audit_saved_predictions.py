#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from palda_decoder.io import iter_jsonl, write_json, write_jsonl
from palda_decoder.rewards import (
    concise_reward, format_reward, grounding_reward, label_consistency_reward, reference_reward,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Score checkpointed decoder predictions without loading a model")
    parser.add_argument("--release-dir", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--split", choices=("validation", "test"), required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    source = {row["example_id"]: row for row in iter_jsonl(args.release_dir / f"{args.split}.jsonl")}
    predictions = list(iter_jsonl(args.predictions))
    rows = [source[row["example_id"]] for row in predictions]
    values = [row["prediction"] for row in predictions]
    task_types = [row["task_type"] for row in rows]
    references = [row["reference_completion"] for row in rows]
    components = {
        "format": format_reward(values, task_types),
        "grounding": grounding_reward(
            values, task_types,
            [row.get("source_texts", []) for row in rows],
            [row.get("allowed_theme_ids", []) for row in rows],
        ),
        "label_consistency": label_consistency_reward(values, references, task_types),
        "concise": concise_reward(values, task_types),
        "reference": reference_reward(values, references),
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    metrics = {
        "split": args.split,
        "samples": len(rows),
        **{f"{name}_reward": sum(scores) / max(len(scores), 1) for name, scores in components.items()},
    }
    write_json(args.output_dir / "metrics.json", metrics)
    write_jsonl(
        args.output_dir / "failures.jsonl",
        (
            {
                "example_id": row["example_id"], "task_type": row["task_type"],
                "scores": {name: scores[index] for name, scores in components.items()},
                "prediction": values[index], "reference_completion": references[index],
                "source_texts": row.get("source_texts", []),
            }
            for index, row in enumerate(rows)
            if any(
                components[name][index] < 1.0
                for name in ("format", "grounding", "label_consistency", "concise")
            )
        ),
    )
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
