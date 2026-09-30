#!/usr/bin/env python
"""Small, reproducible QLoRA entry point for PALDA record annotation."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import torch
from datasets import load_dataset
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, set_seed
from trl import SFTConfig, SFTTrainer

from qna_review_model.paths import DATA_ROOT


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen3-8B")
    parser.add_argument("--data-dir", type=Path, default=DATA_ROOT / "archive/training/bootstrap-keyword-v1")
    parser.add_argument("--output-dir", type=Path, default=Path("models/adapters/qlora-legacy/bootstrap-smoke"))
    parser.add_argument("--max-steps", type=int, default=5)
    parser.add_argument("--max-train-samples", type=int, default=512)
    parser.add_argument("--max-eval-samples", type=int, default=64)
    parser.add_argument("--max-length", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=20260826)
    return parser.parse_args()


def main() -> None:
    args = arguments()
    # Accelerate requires these guards on consumer RTX 40-series multi-GPU hosts,
    # even when a run is intentionally pinned to one visible device.
    os.environ.setdefault("NCCL_P2P_DISABLE", "1")
    os.environ.setdefault("NCCL_IB_DISABLE", "1")
    set_seed(args.seed)
    dataset = load_dataset("json", data_files={
        "train": str(args.data_dir / "train.jsonl"),
        "validation": str(args.data_dir / "validation.jsonl"),
    })
    train = dataset["train"].shuffle(seed=args.seed).select(range(min(args.max_train_samples, len(dataset["train"]))))
    validation = dataset["validation"].shuffle(seed=args.seed).select(range(min(args.max_eval_samples, len(dataset["validation"]))))
    quantization = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    model = AutoModelForCausalLM.from_pretrained(
        args.model, local_files_only=True, quantization_config=quantization, device_map={"": 0},
        dtype=torch.bfloat16,
    )
    model.config.use_cache = False
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    tokenizer.pad_token = tokenizer.eos_token
    peft_config = LoraConfig(
        r=16, lora_alpha=32, lora_dropout=0.05, bias="none", task_type="CAUSAL_LM",
        target_modules="all-linear",
    )
    config = SFTConfig(
        output_dir=str(args.output_dir), max_steps=args.max_steps,
        per_device_train_batch_size=1, per_device_eval_batch_size=1,
        gradient_accumulation_steps=8, learning_rate=2e-4, warmup_ratio=0.03,
        lr_scheduler_type="cosine", optim="paged_adamw_8bit", bf16=True, tf32=True,
        gradient_checkpointing=True, gradient_checkpointing_kwargs={"use_reentrant": False},
        max_length=args.max_length, completion_only_loss=True,
        eval_strategy="steps", eval_steps=args.max_steps, save_strategy="steps", save_steps=args.max_steps,
        logging_steps=1, report_to="none", seed=args.seed, data_seed=args.seed,
    )
    trainer = SFTTrainer(
        model=model, args=config, train_dataset=train, eval_dataset=validation,
        processing_class=tokenizer, peft_config=peft_config,
    )
    result = trainer.train()
    evaluation = trainer.evaluate()
    trainer.save_model(str(args.output_dir / "adapter"))
    metrics = {key: float(value) if isinstance(value, (int, float)) else value for key, value in result.metrics.items()}
    metrics.update({key: float(value) if isinstance(value, (int, float)) else value for key, value in evaluation.items()})
    metrics.update({"model": args.model, "train_samples": len(train), "eval_samples": len(validation)})
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "run_summary.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
