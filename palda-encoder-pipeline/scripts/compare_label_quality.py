#!/usr/bin/env python3
"""Compare annotation methods on Gold Train or rejudge an isolated Silver subset."""
from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import time
from collections import Counter
from dataclasses import replace
from pathlib import Path

from sklearn.feature_extraction.text import TfidfVectorizer

from palda_encoder.io import iter_jsonl, append_jsonl, write_json
from palda_encoder.llm import load_endpoints, _chat, _label_prompt, _parse_labels
from palda_encoder.schema import Sample, validate_labels
from audit_judge_calibration import compact, primary


def fingerprint(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def identity(labels):
    return tuple(sorted((x["theme_id"], x.get("role")) for x in labels))


def summarize(rows):
    groups = {}
    for key in sorted({(r["model_id"], r["arm"]) for r in rows}):
        subset = [r for r in rows if (r["model_id"], r["arm"]) == key]
        valid = [r for r in subset if r["valid"]]
        correct = sum(primary(r["labels"]) == primary(r["original_labels"]) for r in valid)
        exact = sum(identity(r["labels"]) == identity(r["original_labels"]) for r in valid)
        groups["/".join(key)] = {
            "rows": len(subset), "valid": len(valid), "primaryMatches": correct,
            "primaryAgreementAll": correct / max(1, len(subset)),
            "primaryAgreementValid": correct / max(1, len(valid)),
            "fullLabelAgreementAll": exact / max(1, len(subset)),
            "meanLatencySeconds": sum(r["latency_seconds"] for r in subset) / max(1, len(subset)),
        }
    models = {r["model_id"] for r in rows}
    ensembles = {}
    for arm in sorted({r["arm"] for r in rows}):
        accepted = []
        ids = {r["sample_id"] for r in rows if r["arm"] == arm}
        for sample_id in ids:
            votes = [r for r in rows if r["sample_id"] == sample_id and r["arm"] == arm]
            if len(votes) == len(models) and all(r["valid"] for r in votes) and len({identity(r["labels"]) for r in votes}) == 1:
                accepted.append(votes[0])
        ensembles[arm] = {
            "modelsRequired": len(models), "rows": len(ids), "accepted": len(accepted),
            "coverage": len(accepted) / max(1, len(ids)),
            "primaryAgreementAccepted": sum(primary(r["labels"]) == primary(r["original_labels"]) for r in accepted) / max(1, len(accepted)),
            "fullAgreementAccepted": sum(identity(r["labels"]) == identity(r["original_labels"]) for r in accepted) / max(1, len(accepted)),
        }
    return {"individual": groups, "full_label_unanimity": ensembles}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--release-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--models", type=Path, default=Path("configs/models.local.json"))
    parser.add_argument("--model-ids", nargs="+", default=["qwen", "llama"])
    parser.add_argument("--arms", nargs="+", choices=["baseline", "rubric", "retrieved", "schema"], default=["baseline", "rubric", "retrieved"])
    parser.add_argument("--limit", type=int, default=32)
    parser.add_argument("--mode", choices=["gold", "silver"], default="gold")
    parser.add_argument("--exclude-manifest", type=Path)
    parser.add_argument("--reuse-run", type=Path)
    parser.add_argument("--rubric", type=Path, default=Path("configs/annotation_rubric_v2.txt"))
    args = parser.parse_args()
    train = list(iter_jsonl(args.release_dir / "train.jsonl"))
    gold = [r for r in train if r["label_tier"] == "gold" and r["record_type"] == "qna"]
    heldout = [r for r in gold if int(hashlib.sha256(r["product_id"].encode()).hexdigest(), 16) % 4 == 0]
    pool = heldout if args.mode == "gold" else [r for r in train if r["label_tier"] == "silver" and r["record_type"] == "qna"]
    excluded = set(json.loads(args.exclude_manifest.read_text())["selectedIds"]) if args.exclude_manifest else set()
    remaining = sorted([r for r in pool if r["sample_id"] not in excluded], key=lambda r: hashlib.sha256(r["sample_id"].encode()).hexdigest())
    selected, counts = [], Counter()
    while remaining and len(selected) < args.limit:
        row = min(remaining, key=lambda r: counts[primary(r["labels"])])
        remaining.remove(row)
        selected.append(row)
        counts[primary(row["labels"])] += 1
    blocked_products = {r["product_id"] for r in heldout + selected}
    blocked_texts = {compact(r["text"]) for r in heldout + selected}
    reference = [r for r in gold if r["product_id"] not in blocked_products and compact(r["text"]) not in blocked_texts]
    assert not ({r["product_id"] for r in reference} & {r["product_id"] for r in selected})
    assert not ({compact(r["text"]) for r in reference} & {compact(r["text"]) for r in selected})
    vectorizer = TfidfVectorizer(analyzer="char", ngram_range=(2, 4), max_features=30000)
    reference_vectors = vectorizer.fit_transform([r["text"] for r in reference])
    rubric = args.rubric.read_text()
    taxonomy = json.loads((args.release_dir / "taxonomy.json").read_text())
    endpoints = [replace(x, timeout_s=90, max_tokens=700) for x in load_endpoints(args.models) if x.id in args.model_ids]
    if {e.id for e in endpoints} != set(args.model_ids):
        raise ValueError("Requested endpoint missing")
    manifest = {
        "version": "label-quality-v2", "mode": args.mode, "arms": args.arms,
        "sourceTrainHash": fingerprint(args.release_dir / "train.jsonl"),
        "taxonomyHash": fingerprint(args.release_dir / "taxonomy.json"),
        "rubricHash": fingerprint(args.rubric), "scriptHash": fingerprint(Path(__file__)),
        "models": {e.id: {"model": e.model, "systemPromptHash": hashlib.sha256(e.system_prompt.encode()).hexdigest()} for e in endpoints},
        "selectedIds": [r["sample_id"] for r in selected], "referenceIds": [r["sample_id"] for r in reference],
        "productOverlap": 0, "normalizedTextOverlap": 0,
        "protectedEvaluationRead": False,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = args.output_dir / "manifest.json"
    if manifest_path.exists() and json.loads(manifest_path.read_text()) != manifest:
        raise ValueError("Existing output has different configuration")
    write_json(manifest_path, manifest)
    output = args.output_dir / "predictions.jsonl"
    if args.reuse_run and not output.exists():
        prior = json.loads((args.reuse_run / "manifest.json").read_text())
        for key in ("mode", "arms", "sourceTrainHash", "taxonomyHash", "rubricHash", "models", "referenceIds"):
            if prior[key] != manifest[key]:
                raise ValueError(f"Cannot reuse votes with changed {key}")
        if not set(prior["selectedIds"]) <= set(manifest["selectedIds"]):
            raise ValueError("Reused rows are not a subset of current plan")
        append_jsonl(output, list(iter_jsonl(args.reuse_run / "predictions.jsonl")))
        write_json(args.output_dir / "reused-votes.json", {
            "source": str(args.reuse_run),
            "predictionsHash": fingerprint(args.reuse_run / "predictions.jsonl"),
        })
    previous = list(iter_jsonl(output)) if output.exists() else []
    done = {(r["sample_id"], r["model_id"], r["arm"]) for r in previous}

    def label(row, endpoint, arm):
        allowed = sorted(taxonomy["category_themes"].get(row["category_key"]) or [x["id"] for x in taxonomy["themes"]])
        prompt = _label_prompt(Sample.from_dict(row), allowed)
        example_ids = []
        if arm != "baseline":
            catalog = {x["id"]: x.get("qna_label", x["id"]) for x in taxonomy["themes"] if x["id"] in allowed}
            prompt += "\n" + rubric + "\n테마 표시명: " + json.dumps(catalog, ensure_ascii=False)
        if arm in {"retrieved", "schema"}:
            scores = (reference_vectors @ vectorizer.transform([row["text"]]).T).toarray().ravel()
            ranked = sorted(range(len(reference)), key=lambda i: (-scores[i], reference[i]["sample_id"]))
            examples, seen_labels = [], set()
            for index in ranked:
                ref = reference[index]
                if primary(ref["labels"]) not in allowed or primary(ref["labels"]) in seen_labels:
                    continue
                examples.append({"text": ref["text"], "labels": ref["labels"]})
                example_ids.append(ref["sample_id"])
                seen_labels.add(primary(ref["labels"]))
                if len(examples) == 3:
                    break
            prompt += "\n다른 상품의 사람 라벨 예시. 현재 문장과 다르면 그대로 복사하지 않는다: " + json.dumps(examples, ensure_ascii=False)
        started = time.perf_counter()
        result = {"sample_id": row["sample_id"], "model_id": endpoint.id, "arm": arm, "original_labels": row["labels"], "example_ids": example_ids}
        try:
            effective_endpoint = endpoint
            if arm == "schema":
                effective_endpoint = replace(endpoint, system_prompt=(
                    '한국어 쇼핑 Q&A를 주어진 분류 지침과 허용 테마에 따라 분류한다. '
                    '출력은 JSON 객체 하나뿐이다: {"labels":[{"theme_id":"허용 ID",'
                    '"sentiment":null,"role":"primary","evidence":"원문 연속 구절"}]}. '
                    'primary는 정확히 한 개, 추가 문의 의도에만 role="secondary"를 사용한다. '
                    '모든 라벨에 role과 정확한 evidence를 반드시 넣는다. '
                    'OTHER도 해석 가능한 주 테마이며, UNUSABLE은 읽을 수 없거나 의미 없는 내용에만 사용한다.'
                ))
            labels = _parse_labels(_chat(effective_endpoint, prompt, retries=1), record_type="qna")
            errors = validate_labels(labels, record_type="qna", text=row["text"], allowed_themes=set(allowed))
            if sum(x.role == "primary" for x in labels) != 1:
                errors.append("requires_one_primary")
            if any(not x.evidence or x.evidence not in row["text"] for x in labels):
                errors.append("requires_exact_evidence")
            result.update(labels=[x.to_dict() for x in labels], valid=not errors, errors=errors)
        except Exception as exc:
            result.update(labels=[], valid=False, errors=[type(exc).__name__])
        result["latency_seconds"] = round(time.perf_counter() - started, 3)
        return result

    pools = {e.id: concurrent.futures.ThreadPoolExecutor(max_workers=min(2, e.workers or 2)) for e in endpoints}
    try:
        futures = [pools[e.id].submit(label, r, e, arm) for r in selected for e in endpoints for arm in args.arms if (r["sample_id"], e.id, arm) not in done]
        for index, future in enumerate(concurrent.futures.as_completed(futures), 1):
            row = future.result()
            append_jsonl(output, [row])
            if index % 16 == 0:
                print(json.dumps({"completed": index, "scheduled": len(futures)}), flush=True)
    finally:
        for pool in pools.values():
            pool.shutdown(wait=True)
    summary = summarize(list(iter_jsonl(output)))
    write_json(args.output_dir / "summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
