from __future__ import annotations

import argparse
import json

from .audit import audit_release
from .evaluate import evaluate_adapter_splits
from .export import export_service_payloads
from .llm import generate_candidates, judge_candidates
from .pipeline import run_pipeline
from .prepare import prepare_insight_tasks, prepare_summary_tasks
from .release import create_training_release
from .select import select_candidates
from .train import train_grpo, train_sft


def main(argv: list[str] | None = None) -> None:
    parser = _parser()
    args = parser.parse_args(argv)
    if args.command == "prepare":
        result = prepare_summary_tasks(
            args.input,
            args.output,
            max_records_per_type=args.max_records_per_type,
            representatives_per_group=args.representatives_per_group,
            min_group_size=args.min_group_size,
            include_neutral=args.include_neutral,
        )
    elif args.command == "prepare-insights":
        result = prepare_insight_tasks(args.summaries, args.output)
    elif args.command == "generate":
        result = generate_candidates(
            args.tasks, args.models, args.output, max_workers=args.max_workers, limit=args.limit
        )
    elif args.command == "judge":
        result = judge_candidates(args.tasks, args.candidates, args.models, args.output, max_workers=args.max_workers)
    elif args.command == "select":
        result = select_candidates(
            args.tasks,
            args.candidates,
            args.output,
            judgments_path=args.judgments,
            min_judges=args.min_judges,
            min_score=args.min_score,
        )
    elif args.command == "release":
        result = create_training_release(args.selected, args.output_dir, seed=args.seed)
    elif args.command == "export":
        result = export_service_payloads(args.summaries, args.insights, args.output)
    elif args.command == "audit-release":
        result = audit_release(args.release_dir, args.output)
    elif args.command == "evaluate-adapter":
        result = evaluate_adapter_splits(
            args.release_dir,
            args.adapter_dir,
            args.output_dir,
            batch_size=args.batch_size,
            max_input_length=args.max_input_length,
            max_new_tokens=args.max_new_tokens,
            max_samples=args.max_samples,
            splits=tuple(args.split or ("validation", "test")),
        )
    elif args.command == "train-sft":
        result = train_sft(
            args.release_dir,
            args.output_dir,
            model_name=args.model_name,
            epochs=args.epochs,
            batch_size=args.batch_size,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            learning_rate=args.learning_rate,
            max_length=args.max_length,
            lora_r=args.lora_r,
            lora_alpha=args.lora_alpha,
            use_4bit=not args.no_4bit,
            seed=args.seed,
            max_steps=args.max_steps,
            max_train_samples=args.max_train_samples,
            max_eval_samples=args.max_eval_samples,
        )
    elif args.command == "train-grpo":
        result = train_grpo(
            args.release_dir,
            args.sft_adapter_dir,
            args.output_dir,
            max_steps=args.max_steps,
            batch_size=args.batch_size,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            learning_rate=args.learning_rate,
            num_generations=args.num_generations,
            max_completion_length=args.max_completion_length,
        )
    elif args.command == "run":
        result = run_pipeline(args.config, skip_generation=args.skip_generation, skip_training=args.skip_training)
    else:
        parser.error("A command is required")
        return
    print(json.dumps(result, ensure_ascii=False, indent=2))


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="palda-decoder")
    sub = parser.add_subparsers(dest="command")
    command = sub.add_parser("prepare")
    command.add_argument("--input", action="append", required=True)
    command.add_argument("--output", required=True)
    command.add_argument("--max-records-per-type", type=int, default=5000)
    command.add_argument("--representatives-per-group", type=int, default=10)
    command.add_argument("--min-group-size", type=int, default=1)
    command.add_argument("--include-neutral", action="store_true")
    command = sub.add_parser("prepare-insights")
    command.add_argument("--summaries", required=True)
    command.add_argument("--output", required=True)
    command = sub.add_parser("generate")
    command.add_argument("--tasks", required=True)
    command.add_argument("--models", required=True)
    command.add_argument("--output", required=True)
    command.add_argument("--max-workers", type=int, default=4)
    command.add_argument("--limit", type=int)
    command = sub.add_parser("judge")
    command.add_argument("--tasks", required=True)
    command.add_argument("--candidates", required=True)
    command.add_argument("--models", required=True)
    command.add_argument("--output", required=True)
    command.add_argument("--max-workers", type=int, default=4)
    command = sub.add_parser("select")
    command.add_argument("--tasks", required=True)
    command.add_argument("--candidates", required=True)
    command.add_argument("--judgments")
    command.add_argument("--output", required=True)
    command.add_argument("--min-judges", type=int, default=1)
    command.add_argument("--min-score", type=float, default=0.65)
    command = sub.add_parser("release")
    command.add_argument("--selected", action="append", required=True)
    command.add_argument("--output-dir", required=True)
    command.add_argument("--seed", type=int, default=42)
    command = sub.add_parser("export")
    command.add_argument("--summaries", required=True)
    command.add_argument("--insights", required=True)
    command.add_argument("--output", required=True)
    command = sub.add_parser("audit-release")
    command.add_argument("--release-dir", required=True)
    command.add_argument("--output")
    command = sub.add_parser("evaluate-adapter")
    command.add_argument("--release-dir", required=True)
    command.add_argument("--adapter-dir", required=True)
    command.add_argument("--output-dir", required=True)
    command.add_argument("--batch-size", type=int, default=4)
    command.add_argument("--max-input-length", type=int, default=2048)
    command.add_argument("--max-new-tokens", type=int, default=512)
    command.add_argument("--max-samples", type=int)
    command.add_argument(
        "--split", action="append", choices=("validation", "test"),
        help="Split to evaluate; repeat for both. Defaults to validation and test.",
    )
    command = sub.add_parser("train-sft")
    command.add_argument("--release-dir", required=True)
    command.add_argument("--output-dir", required=True)
    command.add_argument("--model-name", default="Qwen/Qwen2.5-7B-Instruct")
    command.add_argument("--epochs", type=float, default=1.0)
    command.add_argument("--batch-size", type=int, default=1)
    command.add_argument("--gradient-accumulation-steps", type=int, default=16)
    command.add_argument("--learning-rate", type=float, default=1e-4)
    command.add_argument("--max-length", type=int, default=2048)
    command.add_argument("--lora-r", type=int, default=16)
    command.add_argument("--lora-alpha", type=int, default=32)
    command.add_argument("--no-4bit", action="store_true")
    command.add_argument("--seed", type=int, default=42)
    command.add_argument("--max-steps", type=int)
    command.add_argument("--max-train-samples", type=int)
    command.add_argument("--max-eval-samples", type=int)
    command = sub.add_parser("train-grpo")
    command.add_argument("--release-dir", required=True)
    command.add_argument("--sft-adapter-dir", required=True)
    command.add_argument("--output-dir", required=True)
    command.add_argument("--max-steps", type=int, default=100)
    command.add_argument("--batch-size", type=int, default=1)
    command.add_argument("--gradient-accumulation-steps", type=int, default=8)
    command.add_argument("--learning-rate", type=float, default=5e-6)
    command.add_argument("--num-generations", type=int, default=4)
    command.add_argument("--max-completion-length", type=int, default=512)
    command = sub.add_parser("run")
    command.add_argument("--config", required=True)
    command.add_argument("--skip-generation", action="store_true")
    command.add_argument("--skip-training", action="store_true")
    return parser


if __name__ == "__main__":
    main()
