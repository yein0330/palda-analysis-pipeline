from __future__ import annotations

import hashlib
from collections import Counter
from pathlib import Path
from typing import Any

from .io import iter_jsonl, read_json, sha256_file, write_json, write_jsonl


def create_strategy_releases(
    base_release_dir: str | Path,
    strategy_root: str | Path,
    output_root: str | Path,
    *, strategies: list[str] | None = None,
    include_combined: bool = True,
    include_selective: bool = True,
) -> dict[str, Any]:
    base = Path(base_release_dir)
    root = Path(strategy_root)
    output = Path(output_root)
    taxonomy = read_json(base / "taxonomy.json")
    allowed = taxonomy.get("category_themes", {})
    base_train = list(iter_jsonl(base / "train.jsonl"))
    validation = list(iter_jsonl(base / "validation.jsonl"))
    test = list(iter_jsonl(base / "test.jsonl"))
    selected = strategies or sorted(
        item.name for item in root.iterdir()
        if item.is_dir() and (item / "consensus.jsonl").exists()
    )
    additions = {strategy: _accepted_rows(root / strategy, allowed) for strategy in selected}
    variants = dict(additions)
    if include_combined:
        variants["combined"] = [row for strategy in selected for row in additions[strategy]]
    if include_selective and {"attributed", "contrastive"} <= set(additions):
        # Pilot policy: retain only the strategies that improved Review metrics,
        # and do not perturb the Q&A primary head with single-intent synthetic rows.
        variants["selective_review"] = [
            row for strategy in ("attributed", "contrastive")
            for row in additions[strategy] if row.get("record_type") == "review"
        ]
    report: dict[str, Any] = {"schema": "palda-synthetic-strategy-release-v1", "variants": {}}
    for name, extra in variants.items():
        destination = output / name
        destination.mkdir(parents=True, exist_ok=True)
        train, duplicates = _merge_by_text(base_train, extra)
        files = {"train": train, "validation": validation, "test": test}
        for split, rows in files.items():
            write_jsonl(destination / f"{split}.jsonl", rows)
        write_json(destination / "taxonomy.json", taxonomy)
        manifest = {
            "schema": "palda-synthetic-strategy-release-v1",
            "strategy": name,
            "base_release": str(base.resolve()),
            "counts": {split: len(rows) for split, rows in files.items()},
            "added_silver": len(train) - len(base_train),
            "duplicate_additions": duplicates,
            "label_tiers": {"train": "silver", "validation": "gold", "test": "gold"},
            "files": {
                split: {"path": f"{split}.jsonl", "sha256": sha256_file(destination / f"{split}.jsonl")}
                for split in files
            },
            "taxonomy": {"path": "taxonomy.json", "sha256": sha256_file(destination / "taxonomy.json")},
        }
        write_json(destination / "manifest.json", manifest)
        report["variants"][name] = manifest
    write_json(output / "manifest.json", report)
    return report


def _accepted_rows(root: Path, allowed: dict[str, list[str]]) -> list[dict[str, Any]]:
    samples = {row["sample_id"]: row for row in iter_jsonl(root / "samples.filtered.jsonl")}
    rows = []
    for consensus in iter_jsonl(root / "consensus.jsonl"):
        if consensus.get("status") != "accepted" or consensus["sample_id"] not in samples:
            continue
        sample = samples[consensus["sample_id"]]
        labels = list(consensus.get("labels") or [])
        rows.append({
            **sample,
            "source_split": "train",
            "usable": bool(sample.get("usable", True)) and not any(
                label.get("theme_id") == "UNUSABLE" for label in labels
            ),
            "labels": labels,
            "candidate_theme_ids": allowed.get(str(sample.get("category_key") or ""), []),
            "label_tier": "silver",
            "agreement_score": consensus.get("agreement_score", 0.0),
        })
    return rows


def _merge_by_text(
    base: list[dict[str, Any]], additions: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], int]:
    result = list(base)
    seen = {_text_key(str(row.get("text") or "")) for row in base}
    duplicates = 0
    for row in additions:
        key = _text_key(str(row.get("text") or ""))
        if not key or key in seen:
            duplicates += 1
            continue
        seen.add(key)
        result.append(row)
    return result, duplicates


def _text_key(text: str) -> str:
    return hashlib.sha256("".join(text.split()).lower().encode()).hexdigest() if text else ""
