#!/usr/bin/env python
"""Generate resumable, schema-checked draft labels with a local instruction model."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from qna_review_model.annotation import AnnotationQueueRow, DraftAnnotationRow, normalize_draft_payload, validate_draft


SYSTEM = """당신은 한국 이커머스 Review/Q&A 전문 데이터 라벨러입니다.
반드시 JSON 객체 하나만 출력합니다. 원문에 명시된 내용만 라벨링하고 추측하지 않습니다.
한 원문에 여러 theme가 있으면 모두 분리합니다. evidence_quote는 원문의 정확한 연속 부분문자열이어야 합니다.
Review는 각 theme에 positive/negative/neutral 중 하나를 지정합니다.
Q&A는 구매 전 질문이면 qna_stage=pre_purchase와 polarity=concern을 사용합니다.
이미 주문·수령한 뒤 배송, 파손, 사용법, 교환을 묻는 Q&A는 qna_stage=post_purchase_support입니다.
내용 없는 문의 안내는 qna_stage=irrelevant, mentions=[]입니다.
허용 Theme로 표현하기 어려운 명시적 속성만 new_theme_candidates에 짧은 한국어 명사로 기록합니다."""


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--queue", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", default="Qwen/Qwen3-8B")
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-new-tokens", type=int, default=320)
    parser.add_argument("--completed-files", type=Path, nargs="*", default=[])
    parser.add_argument("--remaining-index", type=int, default=0)
    parser.add_argument("--remaining-shards", type=int, default=1)
    return parser.parse_args()


def _prompt(row: AnnotationQueueRow) -> str:
    themes = [{"theme_id": theme.theme_id, "label": theme.label} for theme in row.allowed_themes]
    schema = {
        "record_id": row.annotation_id,
        "qna_stage": None if row.record_type == "review" else "pre_purchase | post_purchase_support | irrelevant",
        "mentions": [{"theme_id": "allowed id", "polarity": "positive | negative | neutral | concern", "evidence_quote": "exact quote"}],
        "new_theme_candidates": [], "confidence": "low | medium | high",
    }
    score_note = "별점은 참고하지 말고 원문 표현으로 감성을 판단하세요."
    return (
        f"RECORD_TYPE: {row.record_type}\nCATEGORY: {row.category_path}\n"
        f"ALLOWED_THEMES: {json.dumps(themes, ensure_ascii=False)}\n"
        f"RECORD_ID: {row.annotation_id}\nTEXT: {row.text}\n{score_note}\n"
        f"중요: {'Review이므로 qna_stage는 반드시 JSON null입니다.' if row.record_type == 'review' else 'Q&A이므로 qna_stage를 반드시 세 값 중 하나로 판단합니다.'}\n"
        "mentions는 가장 명시적인 Theme를 최대 3개까지만 기록하세요.\n"
        f"OUTPUT_SCHEMA: {json.dumps(schema, ensure_ascii=False)}"
    )


def _extract(text: str) -> dict[str, Any] | None:
    start, end = text.find("{"), text.rfind("}")
    if start < 0 or end < start:
        return None
    try:
        value = json.loads(text[start:end + 1])
        return value if isinstance(value, dict) else None
    except json.JSONDecodeError:
        return None


def main() -> None:
    args = arguments()
    os.environ.setdefault("NCCL_P2P_DISABLE", "1"); os.environ.setdefault("NCCL_IB_DISABLE", "1")
    rows = [
        AnnotationQueueRow.model_validate_json(line) for index, line in enumerate(args.queue.read_text(encoding="utf-8").splitlines())
        if index % args.num_shards == args.shard_index
    ]
    if args.limit is not None:
        rows = rows[:args.limit]
    completed: set[str] = set()
    if args.output.exists():
        with args.output.open(encoding="utf-8") as existing:
            for line in existing:
                completed.add(DraftAnnotationRow.model_validate_json(line).input.annotation_id)
    for path in args.completed_files:
        if path.exists():
            with path.open(encoding="utf-8") as existing:
                for line in existing:
                    completed.add(DraftAnnotationRow.model_validate_json(line).input.annotation_id)
    rows = [row for row in rows if row.annotation_id not in completed]
    rows = [row for index, row in enumerate(rows) if index % args.remaining_shards == args.remaining_index]
    print(json.dumps({"shard": args.shard_index, "remaining": len(rows), "already_completed": len(completed)}), flush=True)
    if not rows:
        return
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    tokenizer.pad_token = tokenizer.eos_token; tokenizer.padding_side = "left"
    quantization = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    model = AutoModelForCausalLM.from_pretrained(
        args.model, local_files_only=True, quantization_config=quantization,
        device_map={"": 0}, dtype=torch.bfloat16,
    )
    model.eval(); args.output.parent.mkdir(parents=True, exist_ok=True)
    valid = needs_review = 0
    with args.output.open("a", encoding="utf-8") as output, torch.inference_mode():
        for start in range(0, len(rows), args.batch_size):
            batch_rows = rows[start:start + args.batch_size]
            prompts = [tokenizer.apply_chat_template(
                [{"role": "system", "content": SYSTEM}, {"role": "user", "content": _prompt(row)}],
                tokenize=False, add_generation_prompt=True, enable_thinking=False,
            ) for row in batch_rows]
            inputs = tokenizer(prompts, padding=True, truncation=True, max_length=1536, return_tensors="pt").to(model.device)
            generated = model.generate(
                **inputs, max_new_tokens=args.max_new_tokens, do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
            )
            raw_outputs = tokenizer.batch_decode(generated[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True)
            for row, raw in zip(batch_rows, raw_outputs, strict=True):
                payload = _extract(raw)
                if payload is None:
                    draft, errors = None, ["output is not a JSON object"]
                else:
                    payload = normalize_draft_payload(row, payload)
                    draft, errors = validate_draft(row, payload)
                status = "valid" if draft is not None and not errors else "needs_review"
                valid += status == "valid"; needs_review += status == "needs_review"
                result = DraftAnnotationRow(
                    input=row, draft=draft, validation_status=status, validation_errors=errors,
                    model=args.model, raw_output=raw,
                )
                output.write(result.model_dump_json() + "\n"); output.flush()
            done = min(start + len(batch_rows), len(rows))
            print(json.dumps({"shard": args.shard_index, "done": done, "total": len(rows), "valid": valid, "needs_review": needs_review}), flush=True)


if __name__ == "__main__":
    main()
