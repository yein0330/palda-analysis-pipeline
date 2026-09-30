#!/usr/bin/env python
"""Generate resumable report-output-v1 drafts from Oracle report inputs."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from qna_review_model.decoder_draft import normalize_output


SYSTEM = """당신은 PALDA의 근거 기반 한국 이커머스 상품 분석기입니다.
반드시 JSON 객체 하나만 출력합니다. 입력에 없는 숫자, Theme, Evidence ID, 사실을 만들지 않습니다.
summary와 selling point는 입력 Evidence가 직접 뒷받침하는 내용만 간결한 존댓말로 씁니다.
인용문을 새로 작성하지 않고 quoteEvidenceId 또는 questionEvidenceId만 선택합니다.
긍정과 부정을 과장하지 않으며 근거가 부족한 배열은 비워 둡니다."""


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", default="Qwen/Qwen3-8B")
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--max-new-tokens", type=int, default=1200)
    return parser.parse_args()


def output_schema() -> dict[str, Any]:
    return {
        "reviewAnalysis": {
            "positiveThemes": [{
                "themeId": "입력 themeId",
                "theme": "입력 displayTheme",
                "quoteEvidenceId": "해당 그룹 Evidence ID",
                "summary": "근거 기반 한 문장",
                "evidenceIds": ["해당 그룹 Evidence ID"],
            }],
            "negativeThemes": [{
                "themeId": "입력 themeId",
                "theme": "입력 displayTheme",
                "quoteEvidenceId": "해당 그룹 Evidence ID",
                "summary": "근거 기반 한 문장",
                "evidenceIds": ["해당 그룹 Evidence ID"],
            }],
        },
        "qnaAnalysis": {
            "themes": [{
                "themeId": "입력 themeId",
                "theme": "입력 displayTheme",
                "ratio": "입력 ratio 숫자를 그대로 복사",
                "questionEvidenceId": "해당 그룹 Evidence ID",
                "summary": "근거 기반 한 문장",
                "evidenceIds": ["해당 그룹 Evidence ID"],
            }]
        },
        "sellingPoints": [{
            "type": "core | hidden | barrier | seller_action",
            "text": "근거 기반 한 문장",
            "reviewThemes": ["positive/THEME_ID 또는 negative/THEME_ID"],
            "qnaThemes": ["THEME_ID"],
            "evidenceIds": ["입력 Evidence ID"],
            "confidence": "high | medium | low",
        }],
    }


def prompt(report_input: dict[str, Any]) -> str:
    return (
        "다음 report-input-v1을 report-output-v1으로 변환하세요.\n"
        "- mentionCount, primaryCount, ratio는 요약문에서 임의로 바꾸거나 새로 계산하지 마세요.\n"
        "- positiveThemes에는 sentiment=positive만, negativeThemes에는 sentiment=negative만 넣으세요.\n"
        "- neutral Review 그룹은 긍정/부정 배열에 넣지 마세요.\n"
        "- 각 항목의 ID는 반드시 같은 입력 그룹의 Evidence 중에서 선택하세요.\n"
        "- Q&A ratio는 입력 값을 숫자 그대로 복사하세요.\n"
        "- Review 긍정과 부정은 각각 중요도가 높은 최대 3개만 작성하세요.\n"
        "- Q&A 테마는 빈도가 높은 최대 4개만 작성하세요.\n"
        "- 테마마다 Evidence가 약하거나 내용이 불명확하면 해당 결과를 생략하세요.\n"
        "- sellingPoints는 최대 3개이며 직접 근거가 있는 경우만 작성하세요.\n"
        f"OUTPUT_SCHEMA: {json.dumps(output_schema(), ensure_ascii=False)}\n"
        f"INPUT: {json.dumps(report_input, ensure_ascii=False)}"
    )


def extract_json(text: str) -> dict[str, Any] | None:
    start, end = text.find("{"), text.rfind("}")
    if start < 0 or end < start:
        return None
    try:
        value = json.loads(text[start:end + 1])
    except json.JSONDecodeError:
        return None
    return value if isinstance(value, dict) else None


def validate_output(
    report_input: dict[str, Any], output: dict[str, Any] | None
) -> list[str]:
    if output is None:
        return ["output is not a JSON object"]
    errors: list[str] = []
    review_lookup = {
        (group["themeId"], group["sentiment"]): group
        for group in report_input["reviewGroups"]
    }
    qna_lookup = {group["themeId"]: group for group in report_input["qnaGroups"]}
    all_evidence = {
        evidence["id"]
        for group in [*report_input["reviewGroups"], *report_input["qnaGroups"]]
        for evidence in group["evidence"]
    }
    analysis = output.get("reviewAnalysis")
    qna_analysis = output.get("qnaAnalysis")
    selling_points = output.get("sellingPoints")
    if not isinstance(analysis, dict):
        errors.append("reviewAnalysis must be an object")
        analysis = {}
    if not isinstance(qna_analysis, dict):
        errors.append("qnaAnalysis must be an object")
        qna_analysis = {}
    if not isinstance(selling_points, list):
        errors.append("sellingPoints must be an array")
        selling_points = []

    for field, sentiment in (
        ("positiveThemes", "positive"), ("negativeThemes", "negative")
    ):
        items = analysis.get(field)
        if not isinstance(items, list):
            errors.append(f"reviewAnalysis.{field} must be an array")
            continue
        for index, item in enumerate(items):
            prefix = f"{field}[{index}]"
            if not isinstance(item, dict):
                errors.append(f"{prefix} must be an object")
                continue
            group = review_lookup.get((item.get("themeId"), sentiment))
            if group is None:
                errors.append(f"{prefix} theme/sentiment is not in input")
                continue
            valid_ids = {evidence["id"] for evidence in group["evidence"]}
            ids = item.get("evidenceIds")
            if not isinstance(ids, list) or not ids or not set(ids) <= valid_ids:
                errors.append(f"{prefix} has invalid evidenceIds")
            if item.get("quoteEvidenceId") not in valid_ids:
                errors.append(f"{prefix} has invalid quoteEvidenceId")
            if item.get("theme") != group["displayTheme"]:
                errors.append(f"{prefix} display theme changed")
            if not str(item.get("summary") or "").strip():
                errors.append(f"{prefix} summary is empty")

    qna_items = qna_analysis.get("themes")
    if not isinstance(qna_items, list):
        errors.append("qnaAnalysis.themes must be an array")
    else:
        for index, item in enumerate(qna_items):
            prefix = f"qnaThemes[{index}]"
            if not isinstance(item, dict):
                errors.append(f"{prefix} must be an object")
                continue
            group = qna_lookup.get(item.get("themeId"))
            if group is None:
                errors.append(f"{prefix} theme is not in input")
                continue
            valid_ids = {evidence["id"] for evidence in group["evidence"]}
            ids = item.get("evidenceIds")
            if not isinstance(ids, list) or not ids or not set(ids) <= valid_ids:
                errors.append(f"{prefix} has invalid evidenceIds")
            if item.get("questionEvidenceId") not in valid_ids:
                errors.append(f"{prefix} has invalid questionEvidenceId")
            if item.get("theme") != group["displayTheme"]:
                errors.append(f"{prefix} display theme changed")
            if item.get("ratio") != group["ratio"]:
                errors.append(f"{prefix} ratio changed")
            if not str(item.get("summary") or "").strip():
                errors.append(f"{prefix} summary is empty")

    if len(selling_points) > 3:
        errors.append("sellingPoints exceeds 3")
    valid_review_refs = {
        f"{sentiment}/{theme_id}"
        for theme_id, sentiment in review_lookup
    }
    for index, item in enumerate(selling_points):
        prefix = f"sellingPoints[{index}]"
        if not isinstance(item, dict):
            errors.append(f"{prefix} must be an object")
            continue
        if item.get("type") not in {"core", "hidden", "barrier", "seller_action"}:
            errors.append(f"{prefix} has invalid type")
        if item.get("confidence") not in {"high", "medium", "low"}:
            errors.append(f"{prefix} has invalid confidence")
        if not set(item.get("reviewThemes") or []) <= valid_review_refs:
            errors.append(f"{prefix} has invalid reviewThemes")
        if not set(item.get("qnaThemes") or []) <= set(qna_lookup):
            errors.append(f"{prefix} has invalid qnaThemes")
        ids = item.get("evidenceIds")
        if not isinstance(ids, list) or not ids or not set(ids) <= all_evidence:
            errors.append(f"{prefix} has invalid evidenceIds")
        if not str(item.get("text") or "").strip():
            errors.append(f"{prefix} text is empty")
    return errors


def main() -> None:
    args = arguments()
    os.environ.setdefault("NCCL_P2P_DISABLE", "1")
    os.environ.setdefault("NCCL_IB_DISABLE", "1")
    rows = [
        json.loads(line)
        for index, line in enumerate(args.input.read_text(encoding="utf-8").splitlines())
        if index % args.num_shards == args.shard_index
    ]
    completed: set[str] = set()
    valid_existing: list[str] = []
    if args.output.exists():
        for line in args.output.read_text(encoding="utf-8").splitlines():
            previous = json.loads(line)
            if previous.get("validationStatus") == "valid":
                completed.add(str(previous["productGroupId"]))
                valid_existing.append(line)
        args.output.write_text(
            "\n".join(valid_existing) + ("\n" if valid_existing else ""),
            encoding="utf-8",
        )
    rows = [row for row in rows if str(row["productGroupId"]) not in completed]
    if args.limit is not None:
        rows = rows[:args.limit]
    print(json.dumps({"remaining": len(rows), "completed": len(completed)}), flush=True)
    if not rows:
        return

    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    quantization = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        local_files_only=True,
        quantization_config=quantization,
        device_map={"": args.device},
        dtype=torch.bfloat16,
    )
    model.eval()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("a", encoding="utf-8") as target, torch.inference_mode():
        for start in range(0, len(rows), args.batch_size):
            batch = rows[start:start + args.batch_size]
            chats = [
                tokenizer.apply_chat_template(
                    [
                        {"role": "system", "content": SYSTEM},
                        {"role": "user", "content": prompt(row["reportInput"])},
                    ],
                    tokenize=False,
                    add_generation_prompt=True,
                    enable_thinking=False,
                )
                for row in batch
            ]
            inputs = tokenizer(
                chats,
                padding=True,
                truncation=True,
                max_length=7168,
                return_tensors="pt",
            ).to(model.device)
            generated = model.generate(
                **inputs,
                max_new_tokens=args.max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
            )
            texts = tokenizer.batch_decode(
                generated[:, inputs["input_ids"].shape[1]:],
                skip_special_tokens=True,
            )
            for row, raw in zip(batch, texts, strict=True):
                parsed = extract_json(raw)
                warnings = validate_output(row["reportInput"], parsed)
                draft = normalize_output(row["reportInput"], parsed)
                errors = validate_output(row["reportInput"], draft)
                result = {
                    "productGroupId": row["productGroupId"],
                    "split": row["split"],
                    "datasetGrade": "draft",
                    "validationStatus": "valid" if not errors else "needs_review",
                    "validationErrors": errors,
                    "normalizationWarnings": warnings,
                    "reportInput": row["reportInput"],
                    "reportOutputDraft": draft,
                    "model": args.model,
                    "rawOutput": raw if draft is None else None,
                }
                target.write(
                    json.dumps(result, ensure_ascii=False, separators=(",", ":")) + "\n"
                )
                target.flush()
            print(json.dumps({
                "done": min(start + len(batch), len(rows)),
                "total": len(rows),
            }), flush=True)


if __name__ == "__main__":
    main()

