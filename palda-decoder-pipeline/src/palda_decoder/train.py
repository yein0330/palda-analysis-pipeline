from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from .io import read_json, write_json
from .rewards import (
    concise_reward,
    format_reward,
    grounding_reward,
    label_consistency_reward,
    reference_reward,
)


def train_sft(
    release_dir: str | Path,
    output_dir: str | Path,
    *,
    model_name: str = "Qwen/Qwen2.5-7B-Instruct",
    epochs: float = 1.0,
    batch_size: int = 1,
    gradient_accumulation_steps: int = 16,
    learning_rate: float = 1e-4,
    max_length: int = 2048,
    lora_r: int = 16,
    lora_alpha: int = 32,
    use_4bit: bool = True,
    seed: int = 42,
    max_steps: int | None = None,
    max_train_samples: int | None = None,
    max_eval_samples: int | None = None,
) -> dict[str, Any]:
    try:
        import torch
        from datasets import load_dataset
        from peft import LoraConfig
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
        from trl import SFTConfig, SFTTrainer
    except ImportError as exc:
        raise RuntimeError("Install training packages with: pip install -e '.[train]'") from exc
    cuda_available = bool(torch.cuda.is_available())
    bf16_available = cuda_available and bool(torch.cuda.is_bf16_supported())
    if cuda_available and any(
        "RTX 40" in torch.cuda.get_device_name(index)
        for index in range(torch.cuda.device_count())
    ):
        # Consumer Ada GPUs do not support the NCCL P2P/IB path expected by
        # Accelerate. These flags are harmless for the single-GPU QLoRA run.
        os.environ.setdefault("NCCL_P2P_DISABLE", "1")
        os.environ.setdefault("NCCL_IB_DISABLE", "1")
    if use_4bit and not cuda_available:
        raise RuntimeError("4-bit QLoRA requires a working CUDA device; use --no-4bit for CPU smoke tests")
    release = Path(release_dir)
    data_files = {"train": str(release / "train.jsonl")}
    if (release / "validation.jsonl").stat().st_size:
        data_files["validation"] = str(release / "validation.jsonl")
    dataset = load_dataset("json", data_files=data_files)
    if not len(dataset["train"]):
        raise ValueError("SFT training split is empty")
    if max_train_samples is not None:
        dataset["train"] = dataset["train"].shuffle(seed=seed).select(
            range(min(max_train_samples, len(dataset["train"])))
        )
    validation = dataset.get("validation")
    if validation is not None and max_eval_samples is not None:
        validation = validation.shuffle(seed=seed).select(
            range(min(max_eval_samples, len(validation)))
        )
    args = SFTConfig(
        output_dir=str(output_dir),
        num_train_epochs=epochs,
        max_steps=max_steps if max_steps is not None else -1,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        learning_rate=learning_rate,
        max_length=max_length,
        eval_strategy="epoch" if validation is not None and len(validation) else "no",
        save_strategy="epoch",
        logging_steps=10,
        bf16=bf16_available,
        fp16=cuda_available and not bf16_available,
        gradient_checkpointing=True,
        completion_only_loss=True,
        seed=seed,
        data_seed=seed,
        report_to="none",
    )
    peft_config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    )
    quantization = (
        BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16 if bf16_available else torch.float16,
        )
        if use_4bit
        else None
    )
    model_kwargs: dict[str, Any] = {
        "torch_dtype": torch.bfloat16 if bf16_available else torch.float32,
    }
    if quantization is not None:
        model_kwargs["quantization_config"] = quantization
        model_kwargs["device_map"] = {"": 0}
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    overflowing = {
        "train": _prompt_overflow_count(dataset["train"], tokenizer, max_length),
        "validation": _prompt_overflow_count(validation, tokenizer, max_length)
        if validation is not None
        else 0,
    }
    if any(overflowing.values()):
        raise ValueError(
            f"Prompt exceeds max_length before completion tokens: {overflowing}. "
            "Increase max_length or reduce evidence in the affected tasks."
        )
    model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)
    model.config.use_cache = False
    trainer = SFTTrainer(
        model=model,
        args=args,
        train_dataset=dataset["train"],
        eval_dataset=validation if validation is not None and len(validation) else None,
        processing_class=tokenizer,
        peft_config=peft_config,
    )
    result = trainer.train()
    trainer.save_model(str(output_dir))
    metrics = dict(result.metrics)
    metrics.update(
        {
            "model_name": model_name,
            "train_samples": len(dataset["train"]),
            "validation_samples": len(validation) if validation is not None else 0,
            "max_steps": max_steps,
            "seed": seed,
        }
    )
    write_json(Path(output_dir) / "train_metrics.json", metrics)
    return metrics


def _prompt_overflow_count(dataset: Any, tokenizer: Any, max_length: int) -> int:
    return sum(
        len(
            tokenizer.apply_chat_template(
                row["prompt"], tokenize=True, add_generation_prompt=True
            )
        )
        >= max_length
        for row in dataset
    )


def train_grpo(
    release_dir: str | Path,
    sft_adapter_dir: str | Path,
    output_dir: str | Path,
    *,
    max_steps: int = 200,
    batch_size: int = 1,
    gradient_accumulation_steps: int = 8,
    learning_rate: float = 5e-6,
    num_generations: int = 4,
    max_completion_length: int = 512,
) -> dict[str, Any]:
    try:
        import torch
        from datasets import load_dataset
        from peft import AutoPeftModelForCausalLM
        from transformers import BitsAndBytesConfig
        from trl import GRPOConfig, GRPOTrainer
    except ImportError as exc:
        raise RuntimeError("Install training packages with: pip install -e '.[train]'") from exc
    if not torch.cuda.is_available():
        raise RuntimeError("GRPO QLoRA requires a working CUDA device")
    if any("RTX 40" in torch.cuda.get_device_name(index) for index in range(torch.cuda.device_count())):
        os.environ.setdefault("NCCL_P2P_DISABLE", "1")
        os.environ.setdefault("NCCL_IB_DISABLE", "1")
    bf16_available = bool(torch.cuda.is_bf16_supported())
    dataset = load_dataset("json", data_files=str(Path(release_dir) / "grpo_train.jsonl"), split="train")
    quantization = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16 if bf16_available else torch.float16,
    )
    model = AutoPeftModelForCausalLM.from_pretrained(
        str(sft_adapter_dir),
        is_trainable=True,
        quantization_config=quantization,
        device_map={"": 0},
        torch_dtype=torch.bfloat16 if bf16_available else torch.float16,
    )
    args = GRPOConfig(
        output_dir=str(output_dir),
        max_steps=max_steps,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        learning_rate=learning_rate,
        num_generations=num_generations,
        max_completion_length=max_completion_length,
        bf16=bf16_available,
        fp16=not bf16_available,
        gradient_checkpointing=True,
        logging_steps=5,
        save_steps=50,
        report_to="none",
        reward_weights=[1.0, 1.5, 1.5, 0.5, 0.25],
    )
    trainer = GRPOTrainer(
        model=model,
        args=args,
        train_dataset=dataset,
        reward_funcs=[
            format_reward,
            grounding_reward,
            label_consistency_reward,
            concise_reward,
            reference_reward,
        ],
    )
    result = trainer.train()
    trainer.save_model(str(output_dir))
    metrics = dict(result.metrics)
    write_json(Path(output_dir) / "train_metrics.json", metrics)
    return metrics
