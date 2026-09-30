#!/usr/bin/env python
"""Generate grounded Cat2-specific theme candidates with a local model."""

from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


SYSTEM = """당신은 한국 이커머스 VOC taxonomy 설계자입니다.
주어진 Cat2 상품군의 Review/Q&A 원문에서 실제로 반복되는 상품 속성 Theme를 설계합니다.
기존 공통/Cat1 Theme와 의미가 겹치는 Theme는 만들지 않습니다.
배송 속도, 포장, 상담, 교환·반품, 설치기사, AS 같은 판매·물류 서비스는 별도 체계이므로 제외합니다.
Theme ID는 짧은 영문 snake_case, label은 짧은 한국어 명사로 작성합니다.
각 Theme는 원문에서 찾을 수 있는 한국어 keywords와 정확한 원문 부분문자열 evidence_quote 하나를 포함해야 합니다.
과도하게 세부적인 상품명·브랜드·일회성 이슈는 Theme로 만들지 않습니다.
반드시 JSON 객체 하나만 출력합니다."""


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", type=Path, required=True)
    parser.add_argument("--themes", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", default="Qwen/Qwen3-8B")
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-new-tokens", type=int, default=720)
    return parser.parse_args()


def prompt(row: dict, hierarchy: dict) -> str:
    existing = [*hierarchy.get("global", []), *hierarchy.get("cat1", {}).get(row["cat1"], [])]
    records = []
    for index, record in enumerate(row["records"], 1):
        text = str(record["text"]).replace("\n", " ").strip()[:420]
        records.append(f"[{index:02d}|{record['record_type']}] {text}")
    schema = {
        "cat2_path": row["cat2_path"],
        "themes": [{
            "theme_id": "english_snake_case", "label": "한국어 명사",
            "keywords": ["한국어 키워드"], "evidence_quote": "아래 원문의 정확한 부분문자열",
        }],
    }
    return (
        f"CAT2: {row['cat2_path']}\n"
        f"EXISTING_THEMES: {json.dumps(existing, ensure_ascii=False)}\n"
        "TASK: 기존 Theme로 충분히 표현되지 않는 Cat2 전용 상품 속성 Theme를 3~6개 제안하세요. "
        "원문 근거가 약하면 개수를 억지로 채우지 말고 1~2개만 제안해도 됩니다.\n"
        f"OUTPUT_SCHEMA: {json.dumps(schema, ensure_ascii=False)}\n"
        "SAMPLED_RAW_RECORDS:\n" + "\n".join(records)
    )


def extract(raw: str) -> dict | None:
    start, end = raw.find("{"), raw.rfind("}")
    if start < 0 or end < start:
        return None
    try:
        value = json.loads(raw[start:end + 1])
    except json.JSONDecodeError:
        return None
    return value if isinstance(value, dict) else None


def validate(row: dict, payload: dict | None, hierarchy: dict) -> tuple[list[dict], list[str]]:
    if payload is None:
        return [], ["output is not a JSON object"]
    errors = []
    if payload.get("cat2_path") != row["cat2_path"]:
        errors.append("cat2_path mismatch")
    raw_themes = payload.get("themes")
    if not isinstance(raw_themes, list) or not raw_themes:
        return [], [*errors, "themes must be a non-empty list"]
    existing = {
        theme["theme_id"] for theme in
        [*hierarchy.get("global", []), *hierarchy.get("cat1", {}).get(row["cat1"], [])]
    }
    source_texts = [str(record["text"]) for record in row["records"]]
    themes = []
    seen = set()
    for index, raw_theme in enumerate(raw_themes[:6], 1):
        if not isinstance(raw_theme, dict):
            errors.append(f"theme {index} is not an object"); continue
        theme_id = str(raw_theme.get("theme_id") or "").strip()
        label = str(raw_theme.get("label") or "").strip()
        keywords = raw_theme.get("keywords")
        evidence = str(raw_theme.get("evidence_quote") or "").strip()
        if not re.fullmatch(r"[a-z][a-z0-9_]{1,39}", theme_id):
            errors.append(f"invalid theme_id: {theme_id!r}"); continue
        if theme_id in existing:
            errors.append(f"theme overlaps existing ID: {theme_id}"); continue
        if theme_id in seen:
            errors.append(f"duplicate theme_id: {theme_id}"); continue
        if not label:
            errors.append(f"missing label: {theme_id}"); continue
        if not isinstance(keywords, list) or len([x for x in keywords if str(x).strip()]) < 2:
            errors.append(f"at least two keywords required: {theme_id}"); continue
        if not evidence or not any(evidence in text for text in source_texts):
            errors.append(f"evidence is not exact: {theme_id}"); continue
        seen.add(theme_id)
        themes.append({
            "theme_id": theme_id, "label": label,
            "keywords": list(dict.fromkeys(str(x).strip() for x in keywords if str(x).strip()))[:8],
            "evidence_quote": evidence,
        })
    return themes, errors


def main() -> None:
    args = arguments()
    os.environ.setdefault("NCCL_P2P_DISABLE", "1"); os.environ.setdefault("NCCL_IB_DISABLE", "1")
    hierarchy = json.loads(args.themes.read_text(encoding="utf-8"))
    rows = [
        json.loads(line) for index, line in enumerate(args.samples.read_text(encoding="utf-8").splitlines())
        if index % args.num_shards == args.shard_index
    ]
    completed = set()
    if args.output.exists():
        completed = {json.loads(line)["cat2_path"] for line in args.output.read_text(encoding="utf-8").splitlines()}
    rows = [row for row in rows if row["cat2_path"] not in completed]
    print(json.dumps({"shard": args.shard_index, "remaining": len(rows), "completed": len(completed)}), flush=True)
    if not rows:
        return
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    tokenizer.pad_token = tokenizer.eos_token; tokenizer.padding_side = "left"
    quant = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    model = AutoModelForCausalLM.from_pretrained(
        args.model, local_files_only=True, quantization_config=quant,
        device_map={"": 0}, dtype=torch.bfloat16,
    ); model.eval()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("a", encoding="utf-8") as target, torch.inference_mode():
        valid_count = needs_review = 0
        for start in range(0, len(rows), args.batch_size):
            batch = rows[start:start + args.batch_size]
            prompts = [tokenizer.apply_chat_template(
                [{"role": "system", "content": SYSTEM}, {"role": "user", "content": prompt(row, hierarchy)}],
                tokenize=False, add_generation_prompt=True, enable_thinking=False,
            ) for row in batch]
            inputs = tokenizer(prompts, padding=True, truncation=True, max_length=7168, return_tensors="pt").to(model.device)
            generated = model.generate(
                **inputs, max_new_tokens=args.max_new_tokens, do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
            )
            outputs = tokenizer.batch_decode(generated[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True)
            for row, raw in zip(batch, outputs, strict=True):
                themes, errors = validate(row, extract(raw), hierarchy)
                status = "valid" if themes and not errors else "needs_review"
                valid_count += status == "valid"; needs_review += status == "needs_review"
                target.write(json.dumps({
                    "schema_version": "palda_cat2_theme_candidate_v1",
                    "cat1": row["cat1"], "cat2_path": row["cat2_path"],
                    "themes": themes, "validation_status": status,
                    "validation_errors": errors, "model": args.model, "raw_output": raw,
                }, ensure_ascii=False) + "\n"); target.flush()
            print(json.dumps({
                "shard": args.shard_index, "done": min(start + len(batch), len(rows)),
                "total": len(rows), "valid": valid_count, "needs_review": needs_review,
            }), flush=True)


if __name__ == "__main__":
    main()
