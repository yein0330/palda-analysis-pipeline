"""Recompute service_macro_f1 from saved validation predictions and compare two
candidates with a paired bootstrap over products.

The metric is rebuilt from `validation_predictions.jsonl` plus the release's
`validation.jsonl`, so no retraining is needed to re-score archived runs. The
recomputation is checked against each run's recorded `metrics.json` first; a
mismatch aborts rather than silently reporting a different metric.

Products, not rows, are the resampling unit: rows from one product share text
and annotator, so resampling rows would understate the variance.
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.metrics import f1_score

SENTIMENTS = ("positive", "negative", "neutral")


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _themes(release: Path) -> list[str]:
    taxonomy = json.loads((release / "taxonomy.json").read_text(encoding="utf-8"))
    return sorted(str(theme["id"]) for theme in taxonomy.get("themes", []) if theme.get("id"))


def build_arrays(release: Path, predictions_path: Path) -> dict[str, Any]:
    """Target and prediction matrices in the layout `_service_metrics` expects."""
    themes = _themes(release)
    theme_to_id = {theme: index for index, theme in enumerate(themes)}
    sentiment_to_id = {value: index for index, value in enumerate(SENTIMENTS)}
    gold = {row["sample_id"]: row for row in _read_jsonl(release / "validation.jsonl")}
    predictions = _read_jsonl(predictions_path)

    review_actual, review_predicted, review_products = [], [], []
    primary_actual, primary_predicted = [], []
    secondary_actual, secondary_predicted, qna_products = [], [], []

    for prediction in predictions:
        row = gold[prediction["sample_id"]]
        product = str(row.get("product_id") or row.get("metadata", {}).get("product_id") or prediction["sample_id"])
        labels = row.get("labels", [])
        if prediction["record_type"] == "review":
            actual = np.zeros((len(themes), len(SENTIMENTS)), dtype=bool)
            for label in labels:
                theme, sentiment = theme_to_id.get(label["theme_id"]), sentiment_to_id.get(label.get("sentiment"))
                if theme is not None and sentiment is not None:
                    actual[theme, sentiment] = True
            predicted = np.zeros((len(themes), len(SENTIMENTS)), dtype=bool)
            for label in prediction.get("predicted_labels", []):
                theme, sentiment = theme_to_id.get(label["theme_id"]), sentiment_to_id.get(label.get("sentiment"))
                if theme is not None and sentiment is not None:
                    predicted[theme, sentiment] = True
            review_actual.append(actual.reshape(-1))
            review_predicted.append(predicted.reshape(-1))
            review_products.append(product)
            continue

        primary = next((label for label in labels if label.get("role") == "primary"), None)
        if primary is None and labels:
            primary = labels[0]
        if primary and primary["theme_id"] in theme_to_id:
            primary_actual.append(theme_to_id[primary["theme_id"]])
        elif "OTHER" in theme_to_id:
            primary_actual.append(theme_to_id["OTHER"])
        else:
            primary_actual.append(-1)
        primary_predicted.append(theme_to_id.get(prediction.get("primary_theme_id"), -1))

        actual_secondary = np.zeros(len(themes), dtype=bool)
        for label in labels:
            if label is primary or label.get("role") == "primary":
                continue
            theme = theme_to_id.get(label["theme_id"])
            if theme is not None:
                actual_secondary[theme] = True
        predicted_secondary = np.zeros(len(themes), dtype=bool)
        for item in prediction.get("secondary_themes", []):
            theme = theme_to_id.get(item["theme_id"])
            if theme is not None:
                predicted_secondary[theme] = True
        secondary_actual.append(actual_secondary)
        secondary_predicted.append(predicted_secondary)
        qna_products.append(product)

    return {
        "review_actual": np.asarray(review_actual, dtype=bool).reshape(len(review_actual), -1),
        "review_predicted": np.asarray(review_predicted, dtype=bool).reshape(len(review_predicted), -1),
        "review_products": np.asarray(review_products),
        "primary_actual": np.asarray(primary_actual, dtype=int),
        "primary_predicted": np.asarray(primary_predicted, dtype=int),
        "secondary_actual": np.asarray(secondary_actual, dtype=bool).reshape(len(secondary_actual), -1),
        "secondary_predicted": np.asarray(secondary_predicted, dtype=bool).reshape(len(secondary_predicted), -1),
        "qna_products": np.asarray(qna_products),
    }


def _set_macro(actual: np.ndarray, predicted: np.ndarray) -> float:
    if actual.shape[0] == 0:
        return 0.0
    supported = np.logical_or(actual.sum(axis=0) > 0, predicted.sum(axis=0) > 0)
    if not supported.any():
        return 0.0
    return float(f1_score(actual[:, supported], predicted[:, supported], average="macro", zero_division=0))


def service_macro_f1(arrays: dict[str, Any], review_index: np.ndarray, qna_index: np.ndarray) -> float:
    review = _set_macro(arrays["review_actual"][review_index], arrays["review_predicted"][review_index])
    secondary = _set_macro(arrays["secondary_actual"][qna_index], arrays["secondary_predicted"][qna_index])
    actual, predicted = arrays["primary_actual"][qna_index], arrays["primary_predicted"][qna_index]
    valid = actual >= 0
    if valid.any():
        labels = np.union1d(actual[valid], predicted[valid])
        primary = float(f1_score(actual[valid], predicted[valid], labels=labels, average="macro", zero_division=0))
    else:
        primary = 0.0
    return (review + primary + secondary) / 3


def _product_index(arrays: dict[str, Any]) -> tuple[list[str], dict[str, np.ndarray], dict[str, np.ndarray]]:
    review_by, qna_by = defaultdict(list), defaultdict(list)
    for index, product in enumerate(arrays["review_products"]):
        review_by[product].append(index)
    for index, product in enumerate(arrays["qna_products"]):
        qna_by[product].append(index)
    products = sorted(set(review_by) | set(qna_by))
    return (
        products,
        {product: np.asarray(review_by.get(product, []), dtype=int) for product in products},
        {product: np.asarray(qna_by.get(product, []), dtype=int) for product in products},
    )


def paired_bootstrap(
    baselines: list[dict[str, Any]],
    candidates: list[dict[str, Any]],
    *,
    iterations: int = 2000,
    seed: int = 20260923,
    resample_seeds: bool = False,
) -> dict[str, Any]:
    """Bootstrap the seed-averaged delta over products.

    Every seed pair is scored on the same product resample within a draw, then
    averaged, and the percentile interval is taken over those averaged deltas.
    Averaging separately computed per-seed intervals instead would not be a
    valid interval for the mean.

    `resample_seeds` additionally draws the seed pairs with replacement inside
    each iteration, so the interval also covers "would another seed draw have
    changed the decision". It is a strictly wider, more conservative reading.
    """
    pairs = min(len(baselines), len(candidates))
    products, review_by, qna_by = _product_index(baselines[0])
    rng = np.random.default_rng(seed)
    empty = np.array([], dtype=int)
    pooled = np.empty(iterations, dtype=float)
    per_seed = np.empty((iterations, pairs), dtype=float)
    for step in range(iterations):
        drawn = rng.choice(len(products), size=len(products), replace=True)
        review_index = np.concatenate([review_by[products[i]] for i in drawn] or [empty])
        qna_index = np.concatenate([qna_by[products[i]] for i in drawn] or [empty])
        for index in range(pairs):
            per_seed[step, index] = (
                service_macro_f1(candidates[index], review_index, qna_index)
                - service_macro_f1(baselines[index], review_index, qna_index)
            )
        chosen = rng.choice(pairs, size=pairs, replace=True) if resample_seeds else np.arange(pairs)
        pooled[step] = float(np.mean(per_seed[step, chosen]))
    low, high = np.percentile(pooled, [2.5, 97.5])
    return {
        "delta": float(np.mean(pooled)),
        "ci_low": float(low),
        "ci_high": float(high),
        "ci_width": float(high - low),
        "p_candidate_better": float(np.mean(pooled > 0)),
        "per_seed": [
            {
                "delta": float(np.mean(per_seed[:, index])),
                "ci_low": float(np.percentile(per_seed[:, index], 2.5)),
                "ci_high": float(np.percentile(per_seed[:, index], 97.5)),
                "p_candidate_better": float(np.mean(per_seed[:, index] > 0)),
            }
            for index in range(pairs)
        ],
    }


def _full_index(arrays: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    return (
        np.arange(arrays["review_actual"].shape[0]),
        np.arange(arrays["secondary_actual"].shape[0]),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--release-dir", required=True)
    parser.add_argument("--baseline", required=True, nargs="+", help="Baseline run directories, one per seed")
    parser.add_argument("--candidate", required=True, nargs="+", help="Candidate run directories, one per seed")
    parser.add_argument("--iterations", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=20260923)
    parser.add_argument("--output")
    parser.add_argument(
        "--resample-seeds", action="store_true",
        help="Also resample the seed pairs, so the interval covers seed choice as well.",
    )
    parser.add_argument(
        "--tolerance", type=float, default=1e-4,
        help="Allowed gap between the recomputed metric and the run's recorded metrics.json.",
    )
    args = parser.parse_args()
    release = Path(args.release_dir)

    def load(directory: str) -> tuple[dict[str, Any], float, float]:
        path = Path(directory)
        arrays = build_arrays(release, path / "validation_predictions.jsonl")
        recomputed = service_macro_f1(arrays, *_full_index(arrays))
        recorded = float(json.loads((path / "metrics.json").read_text(encoding="utf-8"))["validation"]["service_macro_f1"])
        if abs(recomputed - recorded) > args.tolerance:
            raise SystemExit(
                f"{path}: recomputed {recomputed:.6f} does not match recorded {recorded:.6f}; "
                "predictions and metrics are out of sync."
            )
        return arrays, recomputed, recorded

    baselines = [load(directory) for directory in args.baseline]
    candidates = [load(directory) for directory in args.candidate]
    baseline_scores = [score for _, score, _ in baselines]
    candidate_scores = [score for _, score, _ in candidates]

    bootstrap = paired_bootstrap(
        [arrays for arrays, _, _ in baselines],
        [arrays for arrays, _, _ in candidates],
        iterations=args.iterations, seed=args.seed, resample_seeds=args.resample_seeds,
    )
    for index, entry in enumerate(bootstrap["per_seed"]):
        entry["baseline"] = args.baseline[index]
        entry["candidate"] = args.candidate[index]
    result = {
        "release_dir": str(release),
        "baseline_seeds": len(baselines),
        "candidate_seeds": len(candidates),
        "baseline_mean": float(np.mean(baseline_scores)),
        "candidate_mean": float(np.mean(candidate_scores)),
        "mean_delta": float(np.mean(candidate_scores) - np.mean(baseline_scores)),
        "baseline_scores": baseline_scores,
        "candidate_scores": candidate_scores,
        "seed_spread_baseline": float(np.max(baseline_scores) - np.min(baseline_scores)),
        "seed_spread_candidate": float(np.max(candidate_scores) - np.min(candidate_scores)),
        "resample_seeds": args.resample_seeds,
        "paired_bootstrap": bootstrap,
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if args.output:
        Path(args.output).write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
