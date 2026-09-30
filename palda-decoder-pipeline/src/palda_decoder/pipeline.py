from __future__ import annotations

from pathlib import Path
from typing import Any

from .evaluate import evaluate_adapter_splits
from .export import export_service_payloads
from .io import read_json, write_json
from .llm import generate_candidates, judge_candidates
from .prepare import prepare_insight_tasks, prepare_summary_tasks
from .release import create_training_release
from .select import select_candidates
from .train import train_grpo, train_sft


def run_pipeline(
    config_path: str | Path,
    *,
    skip_generation: bool = False,
    skip_training: bool = False,
) -> dict[str, Any]:
    config_file = Path(config_path).resolve()
    config = read_json(config_file)
    base = config_file.parent
    work = _resolve(base, config.get("work_dir", "../data/work"))
    release_dir = _resolve(base, config.get("release_dir", "../data/releases/decoder-v0.1"))
    artifact_dir = _resolve(base, config.get("artifact_dir", "../artifacts/decoder-v0.1"))
    models = _resolve_optional(base, config.get("models_path"))
    inputs = [_resolve(base, path) for path in config["input_jsonl_paths"]]
    work.mkdir(parents=True, exist_ok=True)
    summary: dict[str, Any] = {}

    summary_tasks = work / "summary_tasks.jsonl"
    summary["prepare_summaries"] = prepare_summary_tasks(
        inputs,
        summary_tasks,
        max_records_per_type=int(config.get("max_records_per_type", 5000)),
        representatives_per_group=int(config.get("representatives_per_group", 10)),
        min_group_size=int(config.get("min_group_size", 1)),
        include_neutral=bool(config.get("include_neutral", False)),
    )
    summary_candidates = work / "summary_candidates.jsonl"
    summary_judgments = work / "summary_judgments.jsonl"
    if not skip_generation:
        if not models:
            raise ValueError("models_path is required unless --skip-generation is used")
        summary["generate_summaries"] = generate_candidates(
            summary_tasks,
            models,
            summary_candidates,
            max_workers=int(config.get("max_workers", 4)),
            retries=int(config.get("retries", 2)),
        )
        if config.get("use_llm_judges", True):
            summary["judge_summaries"] = judge_candidates(
                summary_tasks,
                summary_candidates,
                models,
                summary_judgments,
                max_workers=int(config.get("max_workers", 4)),
                retries=int(config.get("retries", 2)),
            )
    selected_summaries = work / "selected_summaries.jsonl"
    summary["select_summaries"] = select_candidates(
        summary_tasks,
        summary_candidates,
        selected_summaries,
        judgments_path=summary_judgments if config.get("use_llm_judges", True) else None,
        min_judges=int(config.get("min_judges", 1)),
        min_score=float(config.get("min_quality_score", 0.65)),
    )

    insight_tasks = work / "insight_tasks.jsonl"
    summary["prepare_insights"] = prepare_insight_tasks(selected_summaries, insight_tasks)
    insight_candidates = work / "insight_candidates.jsonl"
    insight_judgments = work / "insight_judgments.jsonl"
    if not skip_generation:
        summary["generate_insights"] = generate_candidates(
            insight_tasks,
            models,
            insight_candidates,
            max_workers=int(config.get("max_workers", 4)),
            retries=int(config.get("retries", 2)),
        )
        if config.get("use_llm_judges", True):
            summary["judge_insights"] = judge_candidates(
                insight_tasks,
                insight_candidates,
                models,
                insight_judgments,
                max_workers=int(config.get("max_workers", 4)),
                retries=int(config.get("retries", 2)),
            )
    selected_insights = work / "selected_insights.jsonl"
    summary["select_insights"] = select_candidates(
        insight_tasks,
        insight_candidates,
        selected_insights,
        judgments_path=insight_judgments if config.get("use_llm_judges", True) else None,
        min_judges=int(config.get("min_judges", 1)),
        min_score=float(config.get("min_quality_score", 0.65)),
    )
    summary["export_service"] = export_service_payloads(
        selected_summaries,
        selected_insights,
        work / "service_payloads.jsonl",
    )
    summary["release"] = create_training_release(
        [selected_summaries, selected_insights],
        release_dir,
        seed=int(config.get("seed", 42)),
        train_ratio=float(config.get("train_ratio", 0.8)),
        validation_ratio=float(config.get("validation_ratio", 0.1)),
    )

    train_config = dict(config.get("train") or {})
    if not skip_training and train_config.get("sft_enabled", True):
        sft_dir = artifact_dir / "sft_adapter"
        summary["sft"] = train_sft(
            release_dir,
            sft_dir,
            model_name=str(train_config.get("model_name", "Qwen/Qwen2.5-7B-Instruct")),
            epochs=float(train_config.get("epochs", 1)),
            batch_size=int(train_config.get("batch_size", 1)),
            gradient_accumulation_steps=int(train_config.get("gradient_accumulation_steps", 16)),
            learning_rate=float(train_config.get("learning_rate", 1e-4)),
            max_length=int(train_config.get("max_length", 2048)),
            lora_r=int(train_config.get("lora_r", 16)),
            lora_alpha=int(train_config.get("lora_alpha", 32)),
            use_4bit=bool(train_config.get("use_4bit", True)),
            seed=int(train_config.get("seed", config.get("seed", 42))),
            max_steps=train_config.get("max_steps"),
            max_train_samples=train_config.get("max_train_samples"),
            max_eval_samples=train_config.get("max_eval_samples"),
        )
        final_adapter = sft_dir
        if train_config.get("grpo_enabled", False):
            grpo = dict(train_config.get("grpo") or {})
            summary["grpo"] = train_grpo(
                release_dir,
                sft_dir,
                artifact_dir / "grpo_adapter",
                max_steps=int(grpo.get("max_steps", 200)),
                batch_size=int(grpo.get("batch_size", 1)),
                gradient_accumulation_steps=int(grpo.get("gradient_accumulation_steps", 8)),
                learning_rate=float(grpo.get("learning_rate", 5e-6)),
                num_generations=int(grpo.get("num_generations", 4)),
                max_completion_length=int(grpo.get("max_completion_length", 512)),
            )
            final_adapter = artifact_dir / "grpo_adapter"
        evaluation = dict(train_config.get("evaluation") or {})
        if evaluation.get("enabled", True):
            evaluations = evaluate_adapter_splits(
                release_dir,
                final_adapter,
                artifact_dir / "evaluation",
                batch_size=int(evaluation.get("batch_size", 4)),
                max_input_length=int(evaluation.get("max_input_length", 2048)),
                max_new_tokens=int(evaluation.get("max_new_tokens", 512)),
                max_samples=evaluation.get("max_samples"),
            )
            # Validation selects a decoder candidate; test remains an audit and
            # must not drive iterative model or reward decisions.
            summary["validation_evaluation"] = evaluations["validation"]
            summary["test_audit"] = evaluations["test"]
    write_json(work / "run_summary.json", summary)
    return summary


def _resolve(base: Path, value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else (base / path).resolve()


def _resolve_optional(base: Path, value: str | Path | None) -> Path | None:
    return _resolve(base, value) if value else None
