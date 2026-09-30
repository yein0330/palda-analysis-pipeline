from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

from .io import iter_jsonl, read_json, write_json, write_jsonl
from .schema import Label, Sample, validate_labels


def build_consensus(
    samples_path: str | Path,
    label_run_paths: Iterable[str | Path],
    taxonomy_path: str | Path,
    output_path: str | Path,
    hard_pool_path: str | Path,
    *,
    reliability_path: str | Path | None = None,
    min_models: int = 2,
    theme_threshold: float = 2 / 3,
    sentiment_threshold: float = 2 / 3,
) -> dict[str, int]:
    samples = {row["sample_id"]: Sample.from_dict(row) for row in iter_jsonl(samples_path)}
    taxonomy = read_json(taxonomy_path)
    reliability = read_json(reliability_path) if reliability_path else {}
    latest: dict[tuple[str, str], dict[str, Any]] = {}
    for path in label_run_paths:
        for run in iter_jsonl(path):
            if run.get("valid"):
                latest[(str(run["sample_id"]), str(run["model_id"]))] = run
    by_sample: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for (sample_id, _), run in latest.items():
        if sample_id in samples:
            by_sample[sample_id].append(run)

    outputs: list[dict[str, Any]] = []
    hard_pool: list[dict[str, Any]] = []
    for sample_id in sorted(samples):
        sample = samples[sample_id]
        runs = sorted(by_sample.get(sample_id, []), key=lambda item: item["model_id"])
        # A generator's intended label is a provenance claim, not an independent
        # annotation. Synthetic rows are accepted only when the two other LLMs
        # agree, preventing generator + one judge from forming a false consensus.
        if sample.source_kind == "synthetic":
            runs = [run for run in runs if run.get("source") != "generator_claim"]
        consensus = _consensus_one(
            sample,
            runs,
            taxonomy,
            reliability,
            min_models=min_models,
            theme_threshold=theme_threshold,
            sentiment_threshold=sentiment_threshold,
        )
        outputs.append(consensus)
        if consensus["status"] != "accepted":
            hard_pool.append(
                {
                    **sample.to_dict(),
                    "consensus_status": consensus["status"],
                    "consensus_reasons": consensus["reasons"],
                    "model_outputs": [
                        {"model_id": run["model_id"], "labels": run.get("labels", [])}
                        for run in runs
                    ],
                }
            )
    write_jsonl(output_path, outputs)
    write_jsonl(hard_pool_path, hard_pool)
    report = {
        "samples": len(outputs),
        "accepted": sum(row["status"] == "accepted" for row in outputs),
        "review_needed": sum(row["status"] == "review_needed" for row in outputs),
        "insufficient_votes": sum(row["status"] == "insufficient_votes" for row in outputs),
        "hard_pool": len(hard_pool),
    }
    write_json(Path(output_path).with_suffix(".report.json"), report)
    return report


def _consensus_one(
    sample: Sample,
    runs: list[dict[str, Any]],
    taxonomy: dict[str, Any],
    reliability: dict[str, Any],
    *,
    min_models: int,
    theme_threshold: float,
    sentiment_threshold: float,
) -> dict[str, Any]:
    if len(runs) < min_models:
        return _result(sample.sample_id, "insufficient_votes", [], 0.0, runs, ["too_few_models"])

    allowed = set(taxonomy.get("category_themes", {}).get(sample.category_key) or [])
    if not allowed:
        allowed = {item["id"] for item in taxonomy.get("themes", [])}
    positives: dict[str, list[tuple[dict[str, Any], Label, float]]] = defaultdict(list)
    for run in runs:
        for raw_label in run.get("labels", []):
            label = Label.from_dict(raw_label)
            if label.theme_id in allowed:
                positives[label.theme_id].append(
                    (
                        run,
                        label,
                        _weight(run["model_id"], label.theme_id, sample.category_key, reliability),
                    )
                )

    accepted: list[Label] = []
    theme_scores: list[float] = []
    reasons: list[str] = []
    all_empty = all(not run.get("labels") for run in runs)
    for theme_id in sorted(allowed):
        votes = positives.get(theme_id, [])
        total_theme_weight = sum(
            _weight(run["model_id"], theme_id, sample.category_key, reliability) for run in runs
        )
        voter_weights = {run["model_id"]: weight for run, _, weight in votes}
        score = sum(voter_weights.values()) / max(total_theme_weight, 1e-9)
        if score + 1e-9 < theme_threshold:
            continue
        theme_scores.append(score)
        evidence = _best_evidence(votes)
        if sample.record_type == "qna":
            accepted.append(Label(theme_id=theme_id, evidence=evidence))
            continue
        sentiment_weights: dict[str, float] = defaultdict(float)
        for _, label, weight in votes:
            if label.sentiment:
                sentiment_weights[label.sentiment] += weight
        if not sentiment_weights or total_theme_weight <= 0:
            reasons.append(f"missing_sentiment:{theme_id}")
            continue
        accepted_sentiments = [
            sentiment
            for sentiment, weight in sorted(sentiment_weights.items())
            if weight / total_theme_weight + 1e-9 >= sentiment_threshold
        ]
        if not accepted_sentiments:
            reasons.append(f"sentiment_disagreement:{theme_id}")
            continue
        accepted.extend(
            Label(theme_id=theme_id, sentiment=sentiment, evidence=evidence)
            for sentiment in accepted_sentiments
        )

    if sample.record_type == "qna":
        if not accepted and all_empty and "OTHER" in allowed:
            accepted = [Label(theme_id="OTHER", role="primary")]
        elif accepted:
            primary = _qna_primary_theme(accepted, runs, sample.category_key, reliability)
            accepted = [
                Label(
                    theme_id=label.theme_id,
                    evidence=label.evidence,
                    role="primary" if label.theme_id == primary else "secondary",
                )
                for label in accepted
            ]

    errors = validate_labels(
        accepted,
        record_type=sample.record_type,
        text=sample.text,
        allowed_themes=allowed,
    )
    reasons.extend(errors)
    if reasons:
        status = "review_needed"
    elif accepted or all_empty:
        status = "accepted"
    else:
        status = "review_needed"
        reasons.append("no_theme_reached_threshold")
    agreement = sum(theme_scores) / len(theme_scores) if theme_scores else (1.0 if all_empty else 0.0)
    return _result(sample.sample_id, status, accepted, agreement, runs, reasons)


def _weight(
    model_id: str,
    theme_id: str | None,
    category_path: str,
    reliability: dict[str, Any],
) -> float:
    default = reliability.get("default", {})
    weight = float(default.get(model_id, 1.0))
    if theme_id:
        weight = float(reliability.get("by_theme", {}).get(theme_id, {}).get(model_id, weight))
        weight = float(
            reliability.get("by_category_theme", {})
            .get(category_path, {})
            .get(theme_id, {})
            .get(model_id, weight)
        )
    return max(weight, 0.0)


def _best_evidence(votes: list[tuple[dict[str, Any], Label, float]]) -> str | None:
    candidates = [(weight, label.evidence) for _, label, weight in votes if label.evidence]
    return max(candidates, default=(0.0, None), key=lambda item: (item[0], len(item[1] or "")))[1]


def _qna_primary_theme(
    accepted: list[Label],
    runs: list[dict[str, Any]],
    category_key: str,
    reliability: dict[str, Any],
) -> str:
    allowed = {label.theme_id for label in accepted}
    scores: dict[str, float] = defaultdict(float)
    for run in runs:
        parsed = [Label.from_dict(value) for value in run.get("labels", [])]
        explicit = next(
            (
                label.theme_id
                for label in parsed
                if label.role == "primary" and label.theme_id in allowed
            ),
            None,
        )
        fallback = next((label.theme_id for label in parsed if label.theme_id in allowed), None)
        theme_id = explicit or fallback
        if theme_id:
            scores[theme_id] += _weight(run["model_id"], theme_id, category_key, reliability)
    return max(sorted(allowed), key=lambda theme_id: scores.get(theme_id, 0.0))


def _result(
    sample_id: str,
    status: str,
    labels: list[Label],
    agreement: float,
    runs: list[dict[str, Any]],
    reasons: list[str],
) -> dict[str, Any]:
    return {
        "sample_id": sample_id,
        "status": status,
        "label_tier": "silver",
        "labels": [label.to_dict() for label in labels],
        "agreement_score": round(agreement, 6),
        "models": [run["model_id"] for run in runs],
        "reasons": reasons,
    }
