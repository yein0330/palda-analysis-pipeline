#!/usr/bin/env python
"""Generate service-only draft labels with a local Qwen instruction model."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from qna_review_model.annotation import AnnotationQueueRow
from qna_review_model.service_annotation import ServiceDraftRow, normalize_service_payload, validate_service_draft


SYSTEM = """당신은 한국 이커머스 판매·배송 서비스 전문 데이터 라벨러입니다.
상품 자체의 품질과 판매자·물류·설치 서비스를 엄격히 분리하고 JSON 객체 하나만 출력합니다.
서비스 유형은 delivery_speed, delivery_reliability, delivery_staff, installation,
customer_support, returns_refunds, packaging, after_sales 중에서만 고릅니다.
배송 속도·지연은 delivery_speed, 일정 미준수·오배송·누락은 delivery_reliability,
배송기사 태도는 delivery_staff, 기사 방문 설치 품질은 installation,
문의 답변·상담 태도는 customer_support, 교환·반품·환불은 returns_refunds,
포장 상태는 packaging, 수리·보증·AS는 after_sales입니다.
판매자, 물류, 기사, 상담, 포장, 반품/AS 과정이 원문에 명시된 경우만 서비스입니다.
조립이 쉽다, 설치가 어렵다, 제품이 튼튼하다, 두께·냄새·색상·재고·주문제작 문의처럼
상품 속성이나 구매조건만 말한 문장은 서비스가 아닙니다. "설치"라는 단어도 사용자가
직접 설치한 이야기면 서비스가 아니며, 기사·업체의 방문 시공을 평가할 때만 installation입니다.
제품 색상·규격이 기대와 다른 것은 서비스가 아니지만 다른 상품 배송·수량 누락은 delivery_reliability입니다.
packaging은 박스, 포장, 완충재, 파손 방지 포장을 직접 언급할 때만 사용합니다.
returns_refunds는 실제 교환·반품·환불 절차나 비용을 직접 언급할 때만 사용합니다.
evidence_quote는 원문의 정확한 연속 부분문자열이어야 하며 최대 3개까지만 기록합니다.
서비스 언급이 전혀 없으면 service_relevant=false, mentions=[]로 출력합니다."""

SYSTEM += """

판단 예시:
TEXT: 항상 기대 이상입니다
OUTPUT: {"record_id":"example_1","service_relevant":false,"mentions":[],"confidence":"high"}
TEXT: 두께감이 있고 붙이기 편해요
OUTPUT: {"record_id":"example_2","service_relevant":false,"mentions":[],"confidence":"high"}
TEXT: 타공 구멍이 안 맞아 도어락 설치가 힘들었어요
OUTPUT: {"record_id":"example_3","service_relevant":false,"mentions":[],"confidence":"high"}
TEXT: 언제 재입고되나요?
OUTPUT: {"record_id":"example_4","service_relevant":false,"mentions":[],"confidence":"high"}
TEXT: 배송이 빨라요
OUTPUT: {"record_id":"example_5","service_relevant":true,"mentions":[{"service_type":"delivery_speed","polarity":"positive","evidence_quote":"배송이 빨라요"}],"confidence":"high"}
TEXT: 기사님이 친절하고 설치도 꼼꼼했습니다
OUTPUT: {"record_id":"example_6","service_relevant":true,"mentions":[{"service_type":"delivery_staff","polarity":"positive","evidence_quote":"기사님이 친절하고"},{"service_type":"installation","polarity":"positive","evidence_quote":"설치도 꼼꼼했습니다"}],"confidence":"high"}
"""


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--queue", type=Path, required=True); parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", default="Qwen/Qwen3-8B")
    parser.add_argument("--shard-index", type=int, default=0); parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--limit", type=int); parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-new-tokens", type=int, default=240)
    return parser.parse_args()


def prompt(row: AnnotationQueueRow) -> str:
    return (
        f"RECORD_TYPE: {row.record_type}\nCATEGORY: {row.category_path}\nRECORD_ID: {row.annotation_id}\n"
        f"TEXT: {row.text}\n"
        "먼저 서비스 관련 표현이 명시됐는지 판단하고 없으면 빈 배열을 출력하세요.\n"
        f"출력 JSON에는 record_id={row.annotation_id}, service_relevant(boolean), mentions(array), confidence 문자열을 반드시 포함하세요."
    )


def extract(text: str) -> dict | None:
    start, end = text.find("{"), text.rfind("}")
    if start < 0 or end < start:
        return None
    try:
        value = json.loads(text[start:end + 1]); return value if isinstance(value, dict) else None
    except json.JSONDecodeError:
        return None


def main() -> None:
    args = arguments(); os.environ.setdefault("NCCL_P2P_DISABLE", "1"); os.environ.setdefault("NCCL_IB_DISABLE", "1")
    rows = [AnnotationQueueRow.model_validate_json(line) for i, line in enumerate(args.queue.read_text(encoding="utf-8").splitlines()) if i % args.num_shards == args.shard_index]
    if args.limit is not None: rows = rows[:args.limit]
    completed = set()
    if args.output.exists():
        completed = {ServiceDraftRow.model_validate_json(line).annotation_id for line in args.output.read_text(encoding="utf-8").splitlines()}
    rows = [row for row in rows if row.annotation_id not in completed]
    print(json.dumps({"shard": args.shard_index, "remaining": len(rows), "completed": len(completed)}), flush=True)
    if not rows: return
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True); tokenizer.pad_token = tokenizer.eos_token; tokenizer.padding_side = "left"
    quant = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=torch.bfloat16)
    model = AutoModelForCausalLM.from_pretrained(args.model, local_files_only=True, quantization_config=quant, device_map={"": 0}, dtype=torch.bfloat16); model.eval()
    args.output.parent.mkdir(parents=True, exist_ok=True); valid = needs_review = 0
    with args.output.open("a", encoding="utf-8") as target, torch.inference_mode():
        for start in range(0, len(rows), args.batch_size):
            batch_rows = rows[start:start + args.batch_size]
            prompts = [tokenizer.apply_chat_template([{"role": "system", "content": SYSTEM}, {"role": "user", "content": prompt(row)}], tokenize=False, add_generation_prompt=True, enable_thinking=False) for row in batch_rows]
            inputs = tokenizer(prompts, padding=True, truncation=True, max_length=1280, return_tensors="pt").to(model.device)
            generated = model.generate(**inputs, max_new_tokens=args.max_new_tokens, do_sample=False, pad_token_id=tokenizer.pad_token_id)
            outputs = tokenizer.batch_decode(generated[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True)
            for row, raw in zip(batch_rows, outputs, strict=True):
                payload = extract(raw)
                if payload is None: draft, errors = None, ["output is not a JSON object"]
                else: draft, errors = validate_service_draft(row, normalize_service_payload(row, payload))
                status = "valid" if draft is not None and not errors else "needs_review"
                valid += status == "valid"; needs_review += status == "needs_review"
                target.write(ServiceDraftRow(annotation_id=row.annotation_id, draft=draft, validation_status=status, validation_errors=errors, model=args.model, raw_output=raw).model_dump_json() + "\n"); target.flush()
            print(json.dumps({"shard": args.shard_index, "done": min(start + len(batch_rows), len(rows)), "total": len(rows), "valid": valid, "needs_review": needs_review}), flush=True)


if __name__ == "__main__":
    main()
