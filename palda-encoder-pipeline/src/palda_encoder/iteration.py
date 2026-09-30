from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any, Callable

from .io import read_json, write_json


PipelineRunner = Callable[..., dict[str, Any]]


def run_iterations(
    config_path: str | Path,
    *,
    resume: bool = False,
    pipeline_runner: PipelineRunner | None = None,
) -> dict[str, Any]:
    """Run bounded data-generation/training rounds with champion selection.

    Each candidate starts from the configured base encoder. Synthetic examples
    accumulate across rounds, while release.py keeps them out of validation/test.
    The controller only updates a champion pointer; it never deploys a model.
    """
    if pipeline_runner is None:
        # Avoid a cli -> iteration -> cli import cycle at module import time.
        from .cli import run_pipeline

        pipeline_runner = run_pipeline

    config_file = Path(config_path).resolve()
    base_config = read_json(config_file)
    base = config_file.parent
    settings = dict(base_config.get("iteration") or {})
    _validate_settings(base_config, settings)

    root = _resolve(base, settings.get("root_dir", "../artifacts/iterations"))
    root.mkdir(parents=True, exist_ok=True)
    state_path = root / "iteration_state.json"
    champion_path = root / "champion.json"

    if state_path.exists():
        if not resume:
            raise FileExistsError(
                f"Iteration state already exists at {state_path}. Use --resume or choose another root_dir."
            )
        state = read_json(state_path)
        if state.get("config_path") != str(config_file):
            raise ValueError("Existing iteration state belongs to a different config file")
        if state.get("status") == "completed":
            return state
        state["status"] = "running"
        state.pop("error", None)
        state.pop("failed_round", None)
    else:
        state = {
            "status": "running",
            "config_path": str(config_file),
            "next_round": 0,
            "rounds": [],
            "champion": None,
            "stale_rounds": 0,
            "warnings": [],
        }

    max_rounds = int(settings.get("max_rounds", 5))
    patience = int(settings.get("patience", 2))
    min_improvement = float(settings.get("min_improvement", 0.002))
    target_macro_f1 = _optional_float(settings.get("target_macro_f1"))
    target_tail_f1 = _optional_float(settings.get("target_tail_f1"))
    max_synthetic_fraction = float(settings.get("max_synthetic_fraction", 0.4))

    for round_index in range(int(state.get("next_round", 0)), max_rounds):
        round_dir = root / f"round-{round_index:03d}"
        round_config = _round_config(base_config, config_file, root, round_dir, round_index, settings)
        round_config_path = round_dir / "pipeline.json"
        round_dir.mkdir(parents=True, exist_ok=True)
        _reuse_previous_labels(root, round_dir, round_index)
        write_json(round_config_path, round_config)

        try:
            summary = pipeline_runner(round_config_path, skip_label=False, skip_train=False)
            decision = _assess_candidate(
                summary,
                state.get("champion"),
                round_index=round_index,
                artifact_dir=round_dir / "artifact",
                min_improvement=min_improvement,
                max_synthetic_fraction=max_synthetic_fraction,
                settings=settings,
            )
            round_record = {
                "round": round_index,
                "status": "completed",
                "config": str(round_config_path),
                "artifact_dir": str(round_dir / "artifact"),
                "metrics": decision["metrics"],
                "dataset": decision["dataset"],
                "eligible": decision["eligible"],
                "promoted": decision["promoted"],
                "reasons": decision["reasons"],
            }
            state["rounds"].append(round_record)
            if decision["promoted"]:
                state["champion"] = decision["candidate"]
                state["stale_rounds"] = 0
                write_json(champion_path, decision["candidate"])
            else:
                state["stale_rounds"] = int(state.get("stale_rounds", 0)) + 1

            state["next_round"] = round_index + 1
            stop_reason = _stop_reason(
                decision,
                stale_rounds=int(state["stale_rounds"]),
                patience=patience,
                target_macro_f1=target_macro_f1,
                target_tail_f1=target_tail_f1,
                max_synthetic_fraction=max_synthetic_fraction,
            )
            if decision.get("warning") and decision["warning"] not in state["warnings"]:
                state["warnings"].append(decision["warning"])
            if stop_reason:
                state["status"] = "completed"
                state["stop_reason"] = stop_reason
            write_json(state_path, state)
            if stop_reason:
                break
        except BaseException as exc:
            state["status"] = "failed"
            state["failed_round"] = round_index
            state["error"] = f"{type(exc).__name__}: {exc}"[:1000]
            write_json(state_path, state)
            raise

    if state.get("status") == "running":
        state["status"] = "completed"
        state["stop_reason"] = "max_rounds"
        write_json(state_path, state)
    return state


def _round_config(
    base_config: dict[str, Any],
    config_file: Path,
    root: Path,
    round_dir: Path,
    round_index: int,
    settings: dict[str, Any],
) -> dict[str, Any]:
    config = json.loads(json.dumps(base_config))
    config.pop("iteration", None)
    config["input_csv_globs"] = [
        str(_resolve(config_file.parent, value)) for value in config["input_csv_globs"]
    ]
    for key in ("taxonomy_path", "models_path", "reliability_path", "generation_plan_path"):
        if config.get(key):
            config[key] = str(_resolve(config_file.parent, config[key]))
    config["work_dir"] = str(round_dir / "work")
    config["release_dir"] = str(round_dir / "release")
    config["artifact_dir"] = str(round_dir / "artifact")
    # Keep the product split fixed across rounds so candidate metrics are
    # directly comparable. Only stochastic generation and training change.
    base_seed = int(base_config.get("seed", 42))
    config["seed"] = base_seed
    config["split_seed"] = int(base_config.get("split_seed", base_seed))
    config["generation_seed"] = int(base_config.get("generation_seed", base_seed)) + round_index
    train = dict(config.get("train") or {})
    train["seed"] = int(train.get("seed", base_seed)) + round_index
    config["train"] = train

    # Iterative mode always derives a fresh tail plan from accumulated data.
    # A fixed manual plan would repeat the same cells on every round.
    config["generation_plan_path"] = None
    generation = dict(config.get("auto_generation") or {})
    generation["enabled"] = round_index > 0 or bool(settings.get("generate_in_baseline", False))
    base_target = int(generation.get("target_per_combination", 20))
    increment = int(settings.get("target_increment_per_round", 10))
    generation["target_per_combination"] = base_target + max(round_index - 1, 0) * increment
    generation["max_total"] = int(settings.get("max_generated_per_round", generation.get("max_total", 1000)))
    config["auto_generation"] = generation

    if round_index > 0:
        previous_work = root / f"round-{round_index - 1:03d}" / "work"
        previous_samples = previous_work / "samples.merged.jsonl"
        if not previous_samples.exists():
            previous_samples = previous_work / "samples.jsonl"
        config["bootstrap_samples_paths"] = [str(previous_samples)]
    return config


def _reuse_previous_labels(root: Path, round_dir: Path, round_index: int) -> None:
    if round_index <= 0:
        return
    source = root / f"round-{round_index - 1:03d}" / "work" / "label_runs.llm.jsonl"
    target = round_dir / "work" / "label_runs.llm.jsonl"
    if source.exists() and not target.exists():
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)


def _assess_candidate(
    summary: dict[str, Any],
    champion: dict[str, Any] | None,
    *,
    round_index: int,
    artifact_dir: Path,
    min_improvement: float,
    max_synthetic_fraction: float,
    settings: dict[str, Any],
) -> dict[str, Any]:
    train = summary.get("train")
    if not isinstance(train, dict):
        raise ValueError("Iterative mode requires train.enabled=true and training metrics")
    # Model selection must use validation only. Repeatedly selecting on the test
    # split leaks evaluation information into the training loop.
    metrics = dict(train.get("validation") or {})
    if not metrics:
        raise ValueError("Iterative mode requires validation metrics for champion selection")
    test_metrics = dict(train.get("test") or {})
    score = float(metrics.get("theme_macro_f1", 0.0))
    tail = _optional_float(metrics.get("tail_theme_macro_f1"))
    sentiment = _optional_float(metrics.get("sentiment_accuracy_on_gold_themes"))
    speed = _optional_float(metrics.get("samples_per_second"))
    release = dict(summary.get("release") or {})
    source_kinds = release.get("source_kinds") or {}
    total = sum(int(value) for value in source_kinds.values())
    synthetic_fraction = int(source_kinds.get("synthetic", 0)) / max(total, 1)
    evaluation_tiers = train.get("evaluation_label_tiers") or {}
    label_tiers = evaluation_tiers.get("validation", {})
    gold_only = bool(label_tiers) and set(label_tiers) == {"gold"}

    reasons: list[str] = []
    if synthetic_fraction > max_synthetic_fraction:
        reasons.append("synthetic_fraction_exceeded")
    min_speed = _optional_float(settings.get("min_samples_per_second"))
    if min_speed is not None and (speed is None or speed < min_speed):
        reasons.append("throughput_below_minimum")
    if settings.get("require_gold_evaluation", False) and not gold_only:
        reasons.append("gold_evaluation_required")

    previous_score = float(champion["metrics"]["theme_macro_f1"]) if champion else None
    improvement = None if previous_score is None else score - previous_score
    if improvement is not None and improvement < min_improvement:
        reasons.append("macro_f1_improvement_below_minimum")
    if champion:
        regression = float(settings.get("max_secondary_metric_regression", 0.01))
        old_metrics = champion["metrics"]
        for name, value in (("tail_theme_macro_f1", tail), ("sentiment_accuracy_on_gold_themes", sentiment)):
            old_value = _optional_float(old_metrics.get(name))
            if old_value is not None and value is not None and value < old_value - regression:
                reasons.append(f"{name}_regressed")

    eligible = not reasons
    candidate = {
        "round": round_index,
        "artifact_dir": str(artifact_dir),
        "metrics": metrics,
        "test_audit_metrics": test_metrics,
        "evaluation_label_tiers": label_tiers,
        "synthetic_fraction": round(synthetic_fraction, 6),
    }
    warning = None if gold_only else (
        "Champion selection used non-Gold evaluation labels; keep automatic deployment disabled."
    )
    return {
        "candidate": candidate,
        "metrics": metrics,
        "test_audit_metrics": test_metrics,
        "dataset": {
            "counts": release.get("counts", {}),
            "source_kinds": source_kinds,
            "synthetic_fraction": round(synthetic_fraction, 6),
        },
        "eligible": eligible,
        "promoted": eligible,
        "reasons": reasons,
        "improvement": improvement,
        "warning": warning,
    }


def _stop_reason(
    decision: dict[str, Any],
    *,
    stale_rounds: int,
    patience: int,
    target_macro_f1: float | None,
    target_tail_f1: float | None,
    max_synthetic_fraction: float,
) -> str | None:
    metrics = decision["metrics"]
    if decision["dataset"]["synthetic_fraction"] > max_synthetic_fraction:
        return "synthetic_fraction_limit"
    macro_reached = target_macro_f1 is not None and float(metrics.get("theme_macro_f1", 0.0)) >= target_macro_f1
    tail_value = _optional_float(metrics.get("tail_theme_macro_f1"))
    tail_reached = target_tail_f1 is None or (tail_value is not None and tail_value >= target_tail_f1)
    if decision["promoted"] and macro_reached and tail_reached:
        return "quality_target_reached"
    if stale_rounds >= patience:
        return "no_meaningful_improvement"
    return None


def _validate_settings(config: dict[str, Any], settings: dict[str, Any]) -> None:
    if not config.get("train", {}).get("enabled", True):
        raise ValueError("iteration requires train.enabled=true")
    if int(settings.get("max_rounds", 5)) < 1:
        raise ValueError("iteration.max_rounds must be positive")
    if int(settings.get("patience", 2)) < 1:
        raise ValueError("iteration.patience must be positive")
    fraction = float(settings.get("max_synthetic_fraction", 0.4))
    if not 0 <= fraction <= 1:
        raise ValueError("iteration.max_synthetic_fraction must be between 0 and 1")


def _optional_float(value: Any) -> float | None:
    return None if value is None else float(value)


def _resolve(base: Path, value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else (base / path).resolve()
