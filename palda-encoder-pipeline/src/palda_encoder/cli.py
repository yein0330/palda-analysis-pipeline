from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path
from typing import Any

from .consensus import build_consensus
from .generation_plan import plan_tail_generation
from .gold_anchor import create_gold_anchor_releases
from .io import iter_jsonl, read_json, write_json, write_jsonl
from .iteration import run_iterations
from .llm import generate_synthetic, label_samples
from .normalize import normalize_annotator_csvs
from .release import create_release
from .service_train import train_service_encoder
from .strategy_plan import build_strategy_plans
from .strategy_release import create_strategy_releases
from .train import train_encoder


def main(argv: list[str] | None = None) -> None:
    parser = _parser()
    args = parser.parse_args(argv)
    if args.command == "normalize":
        paths = _expand(args.input)
        result = normalize_annotator_csvs(paths, args.output_dir)
    elif args.command == "generate":
        result = generate_synthetic(
            args.plan,
            args.taxonomy,
            args.models,
            args.samples_output,
            args.claims_output,
            seed=args.seed,
        )
    elif args.command == "plan-strategies":
        result = build_strategy_plans(
            args.release_dir, args.output_dir, strategies=args.strategy or None,
            max_per_strategy=args.max_per_strategy, seed=args.seed,
        )
    elif args.command == "release-gold-anchor":
        result = create_gold_anchor_releases(
            args.benchmark_dir, args.base_silver_release_dir, args.output_root,
            selective_release_dir=args.selective_release_dir,
            train_ratio_of_simulation=args.train_ratio_of_simulation, seed=args.seed,
        )
    elif args.command == "release-strategies":
        result = create_strategy_releases(
            args.base_release_dir, args.strategy_root, args.output_root,
            strategies=args.strategy or None, include_combined=not args.no_combined,
            include_selective=not args.no_selective,
        )
    elif args.command == "plan-generation":
        result = plan_tail_generation(
            args.samples,
            args.consensus,
            args.output,
            target_per_combination=args.target_per_combination,
            max_per_combination=args.max_per_combination,
            max_total=args.max_total,
            include_neutral=args.include_neutral,
        )
    elif args.command == "label":
        result = label_samples(
            args.samples,
            args.taxonomy,
            args.models,
            args.output,
            max_workers=args.max_workers,
            retries=args.retries,
            repair_attempts=args.repair_attempts,
            limit=args.limit,
        )
    elif args.command == "consensus":
        result = build_consensus(
            args.samples,
            args.label_runs,
            args.taxonomy,
            args.output,
            args.hard_pool,
            reliability_path=args.reliability,
            min_models=args.min_models,
            theme_threshold=args.theme_threshold,
            sentiment_threshold=args.sentiment_threshold,
        )
    elif args.command == "release":
        result = create_release(
            args.samples,
            args.consensus,
            args.output_dir,
            taxonomy_path=args.taxonomy,
            seed=args.seed,
            train_ratio=args.train_ratio,
            validation_ratio=args.validation_ratio,
            preserve_source_split=args.preserve_source_split,
        )
    elif args.command == "train":
        result = train_encoder(
            args.release_dir,
            args.output_dir,
            model_name=args.model_name,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            max_length=args.max_length,
            seed=args.seed,
            device_name=args.device,
            max_train_samples=args.max_train_samples,
            max_eval_samples=args.max_eval_samples,
        )
    elif args.command == "train-service":
        result = train_service_encoder(
            args.release_dir,
            args.output_dir,
            model_name=args.model_name,
            epochs=args.epochs,
            batch_size=args.batch_size,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            learning_rate=args.learning_rate,
            head_learning_rate=args.head_learning_rate,
            max_length=args.max_length,
            seed=args.seed,
            device_name=args.device,
            silver_weight=args.silver_weight,
            qna_silver_weight=args.qna_silver_weight,
            primary_silver_weight=args.primary_silver_weight,
            primary_balance_power=args.primary_balance_power,
            review_balance_power=args.review_balance_power,
            secondary_balance_power=args.secondary_balance_power,
            review_positive_base=args.review_positive_base,
            secondary_positive_base=args.secondary_positive_base,
            secondary_loss_weight=args.secondary_loss_weight,
            threshold_beta=args.threshold_beta,
            pooling=args.pooling,
            audit_test=args.audit_test,
            max_train_samples=args.max_train_samples,
            max_eval_samples=args.max_eval_samples,
        )
    elif args.command == "run":
        result = run_pipeline(args.config, skip_label=args.skip_label, skip_train=args.skip_train)
    elif args.command == "iterate":
        result = run_iterations(args.config, resume=args.resume)
    else:
        parser.error("A command is required")
        return
    print(json.dumps(result, ensure_ascii=False, indent=2))


def run_pipeline(
    config_path: str | Path,
    *,
    skip_label: bool = False,
    skip_train: bool = False,
) -> dict[str, Any]:
    config_file = Path(config_path).resolve()
    config = read_json(config_file)
    base = config_file.parent
    work = _resolve(base, config.get("work_dir", "../data/work"))
    release_dir = _resolve(base, config.get("release_dir", "../data/releases/dataset-v0.1"))
    artifact_dir = _resolve(base, config.get("artifact_dir", "../artifacts/encoder-v0.1"))
    work.mkdir(parents=True, exist_ok=True)
    inputs = _expand([str(_resolve(base, value)) for value in config["input_csv_globs"]])
    summary: dict[str, Any] = {}
    summary["normalize"] = normalize_annotator_csvs(
        inputs,
        work,
        taxonomy_version=config.get("taxonomy_version", "taxonomy-v1"),
    )
    taxonomy = _resolve_optional(base, config.get("taxonomy_path")) or work / "taxonomy.derived.json"
    samples_path = work / "samples.jsonl"
    bootstrap_samples = [
        _resolve(base, value) for value in config.get("bootstrap_samples_paths", [])
    ]
    if bootstrap_samples:
        merged_inputs = work / "samples.bootstrap.merged.jsonl"
        _merge_unique_jsonl([samples_path, *bootstrap_samples], merged_inputs, "sample_id")
        samples_path = merged_inputs
    models_path = _resolve_optional(base, config.get("models_path"))
    llm_runs = work / "label_runs.llm.jsonl"
    if not skip_label:
        if not models_path:
            raise ValueError("models_path is required unless --skip-label is used")
        summary["label"] = label_samples(
            samples_path,
            taxonomy,
            models_path,
            llm_runs,
            max_workers=int(config.get("max_workers", 4)),
            retries=int(config.get("retries", 2)),
            limit=config.get("label_limit"),
        )
    label_run_paths: list[Path] = []
    if llm_runs.exists():
        label_run_paths.append(llm_runs)
    if config.get("include_legacy_labels", False):
        label_run_paths.append(work / "label_runs.legacy.jsonl")
    if not label_run_paths:
        raise ValueError("No label run files are available for consensus")

    generation_plan = _resolve_optional(base, config.get("generation_plan_path"))
    auto_generation = config.get("auto_generation", {})
    if generation_plan or auto_generation.get("enabled", False):
        initial_consensus = work / "consensus.initial.jsonl"
        summary["initial_consensus"] = build_consensus(
            samples_path,
            label_run_paths,
            taxonomy,
            initial_consensus,
            work / "hard_pool.initial.jsonl",
            reliability_path=_resolve_optional(base, config.get("reliability_path")),
            min_models=int(config.get("min_models", 2)),
            theme_threshold=float(config.get("theme_threshold", 2 / 3)),
            sentiment_threshold=float(config.get("sentiment_threshold", 2 / 3)),
        )
        if not generation_plan:
            generation_plan = work / "generation_plan.auto.jsonl"
            summary["plan_generation"] = plan_tail_generation(
                samples_path,
                initial_consensus,
                generation_plan,
                target_per_combination=int(auto_generation.get("target_per_combination", 20)),
                max_per_combination=int(auto_generation.get("max_per_combination", 10)),
                max_total=int(auto_generation.get("max_total", 1000)),
                include_neutral=bool(auto_generation.get("include_neutral", False)),
            )
        if not models_path:
            raise ValueError("models_path is required for synthetic generation")
        synthetic_samples = work / "samples.synthetic.jsonl"
        synthetic_claims = work / "label_runs.generator_claims.jsonl"
        summary["generate"] = generate_synthetic(
            generation_plan,
            taxonomy,
            models_path,
            synthetic_samples,
            synthetic_claims,
            seed=int(config.get("generation_seed", config.get("seed", 42))),
        )
        merged_samples = work / "samples.merged.jsonl"
        _merge_unique_jsonl([samples_path, synthetic_samples], merged_samples, "sample_id")
        samples_path = merged_samples
        if not skip_label:
            summary["label_after_generation"] = label_samples(
                samples_path,
                taxonomy,
                models_path,
                llm_runs,
                max_workers=int(config.get("max_workers", 4)),
                retries=int(config.get("retries", 2)),
            )
        label_run_paths.insert(0, synthetic_claims)

    consensus_path = work / "consensus.jsonl"
    summary["consensus"] = build_consensus(
        samples_path,
        label_run_paths,
        taxonomy,
        consensus_path,
        work / "hard_pool.jsonl",
        reliability_path=_resolve_optional(base, config.get("reliability_path")),
        min_models=int(config.get("min_models", 2)),
        theme_threshold=float(config.get("theme_threshold", 2 / 3)),
        sentiment_threshold=float(config.get("sentiment_threshold", 2 / 3)),
    )
    summary["release"] = create_release(
        samples_path,
        consensus_path,
        release_dir,
        taxonomy_path=taxonomy,
        seed=int(config.get("split_seed", config.get("seed", 42))),
        train_ratio=float(config.get("train_ratio", 0.8)),
        validation_ratio=float(config.get("validation_ratio", 0.1)),
        preserve_source_split=bool(config.get("preserve_source_split", False)),
    )
    train_config = config.get("train", {})
    if not skip_train and train_config.get("enabled", True):
        summary["train"] = train_encoder(
            release_dir,
            artifact_dir,
            model_name=str(
                train_config.get("model_name", "monologg/koelectra-base-v3-discriminator")
            ),
            epochs=int(train_config.get("epochs", 3)),
            batch_size=int(train_config.get("batch_size", 32)),
            learning_rate=float(train_config.get("learning_rate", 2e-5)),
            max_length=int(train_config.get("max_length", 128)),
            seed=int(train_config.get("seed", config.get("seed", 42))),
            device_name=str(train_config.get("device", "auto")),
            sentiment_loss_weight=float(train_config.get("sentiment_loss_weight", 1.0)),
            max_train_samples=train_config.get("max_train_samples"),
            max_eval_samples=train_config.get("max_eval_samples"),
        )
    write_json(work / "run_summary.json", summary)
    return summary


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="palda-encoder")
    sub = parser.add_subparsers(dest="command")

    cmd = sub.add_parser("normalize", help="Normalize legacy annotator CSV files")
    cmd.add_argument("--input", action="append", required=True, help="CSV path or glob; repeatable")
    cmd.add_argument("--output-dir", required=True)

    cmd = sub.add_parser("generate", help="Generate targeted synthetic samples")
    cmd.add_argument("--plan", required=True)
    cmd.add_argument("--taxonomy", required=True)
    cmd.add_argument("--models", required=True)
    cmd.add_argument("--samples-output", required=True)
    cmd.add_argument("--claims-output", required=True)
    cmd.add_argument("--seed", type=int, default=42)

    cmd = sub.add_parser("plan-strategies", help="Build comparable synthetic strategy plans")
    cmd.add_argument("--release-dir", required=True)
    cmd.add_argument("--output-dir", required=True)
    cmd.add_argument("--strategy", action="append")
    cmd.add_argument("--max-per-strategy", type=int, default=120)
    cmd.add_argument("--seed", type=int, default=42)

    cmd = sub.add_parser("release-gold-anchor", help="Build product-isolated Gold anchor releases")
    cmd.add_argument("--benchmark-dir", required=True)
    cmd.add_argument("--base-silver-release-dir", required=True)
    cmd.add_argument("--selective-release-dir")
    cmd.add_argument("--output-root", required=True)
    cmd.add_argument("--train-ratio-of-simulation", type=float, default=0.5)
    cmd.add_argument("--seed", type=int, default=20260909)

    cmd = sub.add_parser("release-strategies", help="Build Gold-isolated strategy releases")
    cmd.add_argument("--base-release-dir", required=True)
    cmd.add_argument("--strategy-root", required=True)
    cmd.add_argument("--output-root", required=True)
    cmd.add_argument("--strategy", action="append")
    cmd.add_argument("--no-combined", action="store_true")
    cmd.add_argument("--no-selective", action="store_true")

    cmd = sub.add_parser("plan-generation", help="Plan synthetic data for tail combinations")
    cmd.add_argument("--samples", required=True)
    cmd.add_argument("--consensus", required=True)
    cmd.add_argument("--output", required=True)
    cmd.add_argument("--target-per-combination", type=int, default=20)
    cmd.add_argument("--max-per-combination", type=int, default=10)
    cmd.add_argument("--max-total", type=int, default=1000)
    cmd.add_argument("--include-neutral", action="store_true")

    cmd = sub.add_parser("label", help="Run independent LLM labelers")
    cmd.add_argument("--samples", required=True)
    cmd.add_argument("--taxonomy", required=True)
    cmd.add_argument("--models", required=True)
    cmd.add_argument("--output", required=True)
    cmd.add_argument("--max-workers", type=int, default=4)
    cmd.add_argument("--retries", type=int, default=2)
    cmd.add_argument(
        "--repair-attempts", type=int,
        help="Response repair attempts; defaults to --retries for compatibility.",
    )
    cmd.add_argument("--limit", type=int)

    cmd = sub.add_parser("consensus", help="Build weighted multi-LLM consensus labels")
    cmd.add_argument("--samples", required=True)
    cmd.add_argument("--label-runs", action="append", required=True)
    cmd.add_argument("--taxonomy", required=True)
    cmd.add_argument("--output", required=True)
    cmd.add_argument("--hard-pool", required=True)
    cmd.add_argument("--reliability")
    cmd.add_argument("--min-models", type=int, default=2)
    cmd.add_argument("--theme-threshold", type=float, default=2 / 3)
    cmd.add_argument("--sentiment-threshold", type=float, default=2 / 3)

    cmd = sub.add_parser("release", help="Create leakage-resistant train/validation/test files")
    cmd.add_argument("--samples", required=True)
    cmd.add_argument("--consensus", required=True)
    cmd.add_argument("--output-dir", required=True)
    cmd.add_argument("--taxonomy")
    cmd.add_argument("--seed", type=int, default=42)
    cmd.add_argument("--train-ratio", type=float, default=0.8)
    cmd.add_argument("--validation-ratio", type=float, default=0.1)
    cmd.add_argument("--preserve-source-split", action="store_true")

    cmd = sub.add_parser("train", help="Train and evaluate the KoELECTRA multi-task encoder")
    cmd.add_argument("--release-dir", required=True)
    cmd.add_argument("--output-dir", required=True)
    cmd.add_argument("--model-name", default="monologg/koelectra-base-v3-discriminator")
    cmd.add_argument("--epochs", type=int, default=3)
    cmd.add_argument("--batch-size", type=int, default=32)
    cmd.add_argument("--learning-rate", type=float, default=2e-5)
    cmd.add_argument("--max-length", type=int, default=128)
    cmd.add_argument("--seed", type=int, default=42)
    cmd.add_argument("--device", default="auto")
    cmd.add_argument("--max-train-samples", type=int)
    cmd.add_argument("--max-eval-samples", type=int)

    cmd = sub.add_parser("train-service", help="Train the service-contract four-head encoder")
    cmd.add_argument("--release-dir", required=True)
    cmd.add_argument("--output-dir", required=True)
    cmd.add_argument("--model-name", default="beomi/KcELECTRA-base")
    cmd.add_argument("--epochs", type=int, default=3)
    cmd.add_argument("--batch-size", type=int, default=16)
    cmd.add_argument("--gradient-accumulation-steps", type=int, default=2)
    cmd.add_argument("--learning-rate", type=float, default=2e-5)
    cmd.add_argument("--head-learning-rate", type=float, default=1e-4)
    cmd.add_argument("--max-length", type=int, default=256)
    cmd.add_argument("--seed", type=int, default=42)
    cmd.add_argument("--device", default="auto")
    cmd.add_argument("--silver-weight", type=float, default=0.7)
    cmd.add_argument("--qna-silver-weight", type=float, default=None)
    cmd.add_argument("--primary-silver-weight", type=float, default=None)
    cmd.add_argument("--primary-balance-power", type=float, default=0.0)
    cmd.add_argument("--review-balance-power", type=float, default=0.0)
    cmd.add_argument("--secondary-balance-power", type=float, default=0.0)
    cmd.add_argument("--review-positive-base", type=float, default=3.0)
    cmd.add_argument("--secondary-positive-base", type=float, default=3.0)
    cmd.add_argument("--secondary-loss-weight", type=float, default=0.5)
    cmd.add_argument(
        "--threshold-beta", type=float, default=1.0,
        help="F-beta used for Review/secondary threshold calibration; values below 1 favor precision.",
    )
    cmd.add_argument(
        "--pooling", choices=("cls", "mean"), default="cls",
        help="Sentence pooling over the backbone output. Sentence-embedding "
             "backbones are pretrained with mean pooling.",
    )
    cmd.add_argument(
        "--audit-test", action="store_true",
        help="Evaluate the sealed test split. Use only once for a frozen champion.",
    )
    cmd.add_argument("--max-train-samples", type=int)
    cmd.add_argument("--max-eval-samples", type=int)

    cmd = sub.add_parser("run", help="Run the configured end-to-end encoder pipeline")
    cmd.add_argument("--config", required=True)
    cmd.add_argument("--skip-label", action="store_true")
    cmd.add_argument("--skip-train", action="store_true")

    cmd = sub.add_parser("iterate", help="Run bounded generate/train/evaluate iterations")
    cmd.add_argument("--config", required=True)
    cmd.add_argument("--resume", action="store_true")
    return parser


def _expand(patterns: list[str]) -> list[Path]:
    paths: list[Path] = []
    for pattern in patterns:
        matches = [Path(item) for item in glob.glob(pattern)]
        paths.extend(matches or [Path(pattern)])
    missing = [str(path) for path in paths if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Input files not found: {missing}")
    return sorted(set(path.resolve() for path in paths))


def _resolve(base: Path, value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else (base / path).resolve()


def _resolve_optional(base: Path, value: str | Path | None) -> Path | None:
    return _resolve(base, value) if value else None


def _merge_unique_jsonl(paths: list[Path], output: Path, key: str) -> None:
    merged: dict[str, dict[str, Any]] = {}
    for path in paths:
        for row in iter_jsonl(path):
            merged[str(row[key])] = row
    write_jsonl(output, (merged[item] for item in sorted(merged)))


if __name__ == "__main__":
    main()
