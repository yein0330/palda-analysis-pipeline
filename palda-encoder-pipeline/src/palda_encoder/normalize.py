from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

from .io import write_json, write_jsonl
from .schema import Label, Sample, make_category_key, utc_now, validate_labels


TRUE_VALUES = {"1", "true", "yes", "y"}


def normalize_annotator_csvs(
    input_paths: Iterable[str | Path],
    output_dir: str | Path,
    *,
    taxonomy_version: str = "taxonomy-v1",
    legacy_model_id: str = "legacy_auto",
) -> dict[str, Any]:
    samples: dict[str, dict[str, Any]] = {}
    label_runs: list[dict[str, Any]] = []
    themes: set[str] = set()
    category_themes: dict[str, set[str]] = defaultdict(set)
    duplicates = 0
    invalid_samples = 0

    for input_path in sorted(map(Path, input_paths)):
        with input_path.open(encoding="utf-8-sig", newline="") as handle:
            for row in csv.DictReader(handle):
                sample_id = (row.get("annotation_id") or "").strip()
                category_path = (row.get("category_path") or "").strip()
                sample = Sample(
                    sample_id=sample_id,
                    product_id=(row.get("product_id") or sample_id).strip(),
                    record_type=(row.get("record_type") or "").strip().lower(),
                    text=(row.get("text") or "").strip(),
                    category_path=category_path,
                    category_key=make_category_key(category_path),
                    source_kind="real",
                    source_split=(row.get("source_split") or "").strip() or None,
                    usable=(row.get("usable") or "").strip().lower() in TRUE_VALUES,
                    metadata={"source_file": input_path.name},
                )
                sample_errors = sample.validate()
                if sample_errors:
                    invalid_samples += 1
                    continue
                previous = samples.get(sample_id)
                if previous:
                    if previous["text"] != sample.text:
                        raise ValueError(f"Conflicting text for sample_id={sample_id}")
                    duplicates += 1
                else:
                    samples[sample_id] = sample.to_dict()

                labels = _legacy_labels(row, sample.record_type)
                for label in labels:
                    themes.add(label.theme_id)
                    category_themes[sample.category_key].add(label.theme_id)
                errors = validate_labels(labels, record_type=sample.record_type, text=sample.text)
                label_runs.append(
                    {
                        "sample_id": sample_id,
                        "annotator_id": row.get("reviewer_id") or legacy_model_id,
                        "model_id": legacy_model_id,
                        "model_version": "unknown",
                        "prompt_version": "legacy-import",
                        "taxonomy_version": taxonomy_version,
                        "labels": [label.to_dict() for label in labels],
                        "self_reported_confidence": row.get("model_confidence") or None,
                        "valid": not errors and sample.usable,
                        "validation_errors": errors,
                        "source": "legacy_csv",
                        "created_at": utc_now(),
                    }
                )

    out = Path(output_dir)
    ordered_samples = [samples[key] for key in sorted(samples)]
    write_jsonl(out / "samples.jsonl", ordered_samples)
    write_jsonl(out / "label_runs.legacy.jsonl", label_runs)
    taxonomy = {
        "version": taxonomy_version,
        "themes": [{"id": theme} for theme in sorted(themes)],
        "category_themes": {
            category: sorted(values) for category, values in sorted(category_themes.items())
        },
    }
    write_json(out / "taxonomy.derived.json", taxonomy)
    report = {
        "samples": len(ordered_samples),
        "label_runs": len(label_runs),
        "themes": len(themes),
        "categories": len(category_themes),
        "duplicate_sample_ids": duplicates,
        "invalid_samples_skipped": invalid_samples,
    }
    write_json(out / "normalize_report.json", report)
    return report


def _legacy_labels(row: dict[str, str], record_type: str) -> list[Label]:
    labels: list[Label] = []
    if record_type == "review":
        raw = row.get("review_labels_json") or "[]"
        for value in _json_list(raw):
            if isinstance(value, dict):
                labels.append(Label.from_dict(value))
    elif record_type == "qna":
        primary = (row.get("qna_primary_theme_id") or "").strip()
        if primary:
            labels.append(Label(theme_id=primary))
        for value in _json_list(row.get("qna_secondary_theme_ids_json") or "[]"):
            theme_id = str(value.get("themeId") if isinstance(value, dict) else value).strip()
            if theme_id and theme_id not in {item.theme_id for item in labels}:
                labels.append(Label(theme_id=theme_id))
    return labels


def _json_list(raw: str) -> list[Any]:
    try:
        value = json.loads(raw)
    except json.JSONDecodeError:
        return []
    return value if isinstance(value, list) else []
