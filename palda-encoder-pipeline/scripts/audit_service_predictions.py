#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from sklearn.metrics import f1_score

from palda_encoder.io import iter_jsonl, write_json


def _set_scores(actual: list[set[str]], predicted: list[set[str]]) -> dict[str, float]:
    labels = sorted(set().union(*actual, *predicted))
    if not labels:
        return {"micro_f1": 0.0, "macro_f1": 0.0, "exact_match": 1.0}
    gold = [[label in values for label in labels] for values in actual]
    guess = [[label in values for label in labels] for values in predicted]
    return {
        "micro_f1": round(float(f1_score(gold, guess, average="micro", zero_division=0)), 6),
        "macro_f1": round(float(f1_score(gold, guess, average="macro", zero_division=0)), 6),
        "exact_match": round(sum(a == p for a, p in zip(actual, predicted)) / max(len(actual), 1), 6),
    }


def audit(gold_path: Path, prediction_path: Path) -> dict[str, object]:
    gold = {row["sample_id"]: row for row in iter_jsonl(gold_path)}
    predictions = {row["sample_id"]: row for row in iter_jsonl(prediction_path)}
    missing = sorted(set(gold) - set(predictions))
    extra = sorted(set(predictions) - set(gold))
    if missing or extra:
        raise ValueError(f"Prediction coverage mismatch: missing={len(missing)}, extra={len(extra)}")

    reviews = [sample_id for sample_id, row in gold.items() if row["record_type"] == "review"]
    qnas = [sample_id for sample_id, row in gold.items() if row["record_type"] == "qna"]
    review_actual = [
        {f"{label['theme_id']}::{label['sentiment']}" for label in gold[sample_id].get("labels", [])}
        for sample_id in reviews
    ]
    review_predicted = [
        {f"{label['theme_id']}::{label['sentiment']}" for label in predictions[sample_id].get("predicted_labels", [])}
        for sample_id in reviews
    ]
    secondary_actual = [
        {label["theme_id"] for label in gold[sample_id].get("labels", []) if label.get("role") == "secondary"}
        for sample_id in qnas
    ]
    secondary_predicted = [
        {label["theme_id"] for label in predictions[sample_id].get("secondary_themes", [])}
        for sample_id in qnas
    ]
    primary_actual = []
    primary_predicted = []
    for sample_id in qnas:
        labels = gold[sample_id].get("labels", [])
        primary = next((label for label in labels if label.get("role") == "primary"), labels[0] if labels else None)
        primary_actual.append(primary["theme_id"] if primary else "OTHER")
        primary_predicted.append(predictions[sample_id]["primary_theme_id"])
    primary_labels = sorted(set(primary_actual) | set(primary_predicted))
    primary_macro = f1_score(
        primary_actual, primary_predicted, labels=primary_labels, average="macro", zero_division=0
    )
    primary_accuracy = sum(a == p for a, p in zip(primary_actual, primary_predicted)) / max(len(qnas), 1)
    usable_actual = [bool(gold[sample_id].get("usable", True)) for sample_id in qnas]
    usable_predicted = [predictions[sample_id].get("usable_probability", 1.0) >= 0.5 for sample_id in qnas]

    review_scores = _set_scores(review_actual, review_predicted)
    secondary_scores = _set_scores(secondary_actual, secondary_predicted)
    return {
        "samples": len(gold),
        "review_samples": len(reviews),
        "qna_samples": len(qnas),
        "service_macro_f1": round(
            (review_scores["macro_f1"] + float(primary_macro) + secondary_scores["macro_f1"]) / 3,
            6,
        ),
        "review_theme_sentiment": review_scores,
        "qna_primary": {
            "accuracy": round(primary_accuracy, 6),
            "macro_f1": round(float(primary_macro), 6),
        },
        "qna_secondary": secondary_scores,
        "usable": {
            "accuracy": round(
                sum(a == p for a, p in zip(usable_actual, usable_predicted)) / max(len(qnas), 1), 6
            ),
            "negative_support": sum(not value for value in usable_actual),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit saved service-encoder predictions offline")
    parser.add_argument("--gold", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    metrics = audit(args.gold, args.predictions)
    if args.output:
        write_json(args.output, metrics)
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
