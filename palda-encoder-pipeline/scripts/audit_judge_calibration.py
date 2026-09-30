#!/usr/bin/env python3
"""Train-only, product-isolated prompt comparison; never reads Test or Validation."""
from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
from collections import Counter
from dataclasses import replace
from pathlib import Path

from palda_encoder.io import iter_jsonl, append_jsonl, write_json
from palda_encoder.llm import load_endpoints, _chat, _label_prompt, _parse_labels
from palda_encoder.schema import Sample, validate_labels


def primary(labels):
    found = next((x for x in labels if x.get("role") == "primary"), None)
    return (found or (labels[0] if labels else {})).get("theme_id", "OTHER")


def compact(text):
    return "".join(text.lower().split())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--release-dir", type=Path, required=True)
    parser.add_argument("--models", type=Path, default=Path("configs/models.local.json"))
    parser.add_argument("--model-id", default="qwen")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=24)
    args = parser.parse_args()
    train = list(iter_jsonl(args.release_dir / "train.jsonl"))
    gold = [r for r in train if r.get("label_tier") == "gold" and r["record_type"] == "qna"]
    # Reserve whole products before selecting a balanced diagnostic subset.
    heldout = [r for r in gold if int(hashlib.sha256(r["product_id"].encode()).hexdigest(), 16) % 4 == 0]
    blocked_products = {r["product_id"] for r in heldout}
    blocked_texts = {compact(r["text"]) for r in heldout}
    reference = [r for r in gold if r["product_id"] not in blocked_products and compact(r["text"]) not in blocked_texts]
    counts = Counter()
    selected = []
    remaining = sorted(heldout, key=lambda r: hashlib.sha256(r["sample_id"].encode()).hexdigest())
    while remaining and len(selected) < args.limit:
        row = min(remaining, key=lambda r: counts[primary(r["labels"])])
        remaining.remove(row)
        selected.append(row)
        counts[primary(row["labels"])] += 1
    taxonomy = json.loads((args.release_dir / "taxonomy.json").read_text())
    endpoint = replace(next(x for x in load_endpoints(args.models) if x.id == args.model_id), timeout_s=60, max_tokens=600)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "trainSha256": hashlib.sha256((args.release_dir / "train.jsonl").read_bytes()).hexdigest(),
        "taxonomySha256": hashlib.sha256((args.release_dir / "taxonomy.json").read_bytes()).hexdigest(),
        "model": endpoint.model, "purpose": "train-only judge calibration diagnostic",
        "selectedIds": [r["sample_id"] for r in selected],
        "referenceRows": len(reference), "primaryCounts": dict(counts),
        "productOverlap": 0, "normalizedTextOverlap": 0,
    }
    manifest_path = args.output_dir / "manifest.json"
    if manifest_path.exists() and json.loads(manifest_path.read_text()) != manifest:
        raise ValueError("Output belongs to a different experiment")
    write_json(manifest_path, manifest)
    output = args.output_dir / "predictions.jsonl"
    existing = list(iter_jsonl(output)) if output.exists() else []
    done = {(r["sample_id"], r["arm"]) for r in existing}

    def label(row, arm):
        allowed = sorted(taxonomy["category_themes"].get(row["category_key"]) or [t["id"] for t in taxonomy["themes"]])
        prompt = _label_prompt(Sample.from_dict(row), allowed)
        if arm != "baseline":
            catalog = {t["id"]: t.get("qna_label", t["id"]) for t in taxonomy["themes"] if t["id"] in allowed}
            prompt += "\n테마 표시명: " + json.dumps(catalog, ensure_ascii=False)
            prompt += "\n직접 묻는 내용만 분류하세요. 교환·반품 가능 여부는 RETURN_EXCHANGE, 상담 대응 자체는 CUSTOMER_SUPPORT입니다. 사용 방법은 USABILITY입니다. 부가 테마는 별도 문의 의도가 문구로 확인될 때만 추가하세요."
        if arm == "fewshot":
            tokens = set(row["text"].split())
            ranked = sorted(reference, key=lambda r: (
                -len(tokens & set(r["text"].split())), r["sample_id"]
            ))
            examples = [{"text": r["text"], "labels": r["labels"]} for r in ranked[:3]]
            prompt += "\n별도 상품의 사람 라벨 예시(출력 형식과 판단 기준 참고): " + json.dumps(examples, ensure_ascii=False)
        result = {"sample_id": row["sample_id"], "arm": arm, "gold": row["labels"]}
        try:
            labels = _parse_labels(_chat(endpoint, prompt, retries=0), record_type="qna")
            errors = validate_labels(labels, record_type="qna", text=row["text"], allowed_themes=set(allowed))
            if sum(x.role == "primary" for x in labels) != 1:
                errors.append("requires_exactly_one_primary")
            if any(not x.evidence or x.evidence not in row["text"] for x in labels):
                errors.append("requires_exact_evidence")
            result.update(labels=[x.to_dict() for x in labels], valid=not errors, errors=errors)
        except Exception as exc:
            result.update(labels=[], valid=False, errors=[type(exc).__name__])
        return result

    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(label, row, arm) for row in selected
                   for arm in ("baseline", "definitions", "fewshot") if (row["sample_id"], arm) not in done]
        for future in concurrent.futures.as_completed(futures):
            value = future.result()
            append_jsonl(output, [value])
            print(json.dumps({"sample_id": value["sample_id"], "arm": value["arm"], "valid": value["valid"]}), flush=True)
    summary = {}
    all_rows = list(iter_jsonl(output))
    paired_ids = set.intersection(*[
        {r["sample_id"] for r in all_rows if r["arm"] == arm and r["valid"]}
        for arm in ("baseline", "definitions", "fewshot")
    ])
    for arm in ("baseline", "definitions", "fewshot"):
        rows = [r for r in all_rows if r["arm"] == arm]
        def secondary(labels):
            return {x["theme_id"] for x in labels if x.get("role") == "secondary"}
        summary[arm] = {
            "rows": len(rows), "valid": sum(r["valid"] for r in rows),
            "primaryAgreementAll": sum(r["valid"] and primary(r["labels"]) == primary(r["gold"]) for r in rows) / max(1, len(rows)),
            "pairedValidRows": len(paired_ids),
            "primaryAgreementPairedValid": sum(primary(r["labels"]) == primary(r["gold"]) for r in rows if r["sample_id"] in paired_ids) / max(1, len(paired_ids)),
            "secondaryExactAll": sum(r["valid"] and secondary(r["labels"]) == secondary(r["gold"]) for r in rows) / max(1, len(rows)),
            "confusions": dict(Counter(primary(r["gold"]) + "->" + primary(r["labels"]) for r in rows if r["valid"] and primary(r["gold"]) != primary(r["labels"]))),
        }
    write_json(args.output_dir / "summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
