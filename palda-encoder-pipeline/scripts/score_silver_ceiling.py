#!/usr/bin/env python3
"""Score consensus LLM labels on human Gold Validation with service_macro_f1.

This is the ceiling of the current training recipe: an encoder that reproduced
the Silver labelling function perfectly would score this on Validation. The
metric is the same `service_macro_f1` that `service_train._service_metrics`
reports, rebuilt by `paired_bootstrap_service.py` from prediction rows.

Two readings are reported, because consensus discards rows it cannot agree on:

* full coverage - rows without accepted consensus count as empty predictions
  (a Q&A row with no primary is scored as OTHER, the consensus default);
* accepted subset - only rows the consensus accepted, plus the coverage rate.

Human Gold labels are only read for scoring; nothing is written back to them.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def _bootstrap_module():
    spec = importlib.util.spec_from_file_location(
        "paired_bootstrap_service", ROOT / "scripts/paired_bootstrap_service.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _to_prediction(sample_id: str, record_type: str, labels: list[dict]) -> dict:
    if record_type == "review":
        return {"sample_id": sample_id, "record_type": "review", "predicted_labels": [
            {"theme_id": label["theme_id"], "sentiment": label.get("sentiment")}
            for label in labels if label.get("sentiment")
        ]}
    primary = next((label for label in labels if label.get("role") == "primary"), None)
    return {
        "sample_id": sample_id, "record_type": "qna",
        "primary_theme_id": primary["theme_id"] if primary else "OTHER",
        "secondary_themes": [
            {"theme_id": label["theme_id"]} for label in labels
            if label is not primary and label.get("role") != "primary"
        ],
    }


def _components(module, arrays) -> dict:
    from sklearn.metrics import f1_score
    review_index = np.arange(arrays["review_actual"].shape[0])
    qna_index = np.arange(arrays["secondary_actual"].shape[0])
    review = module._set_macro(arrays["review_actual"], arrays["review_predicted"])
    secondary = module._set_macro(arrays["secondary_actual"], arrays["secondary_predicted"])
    actual, predicted = arrays["primary_actual"], arrays["primary_predicted"]
    valid = actual >= 0
    labels = np.union1d(actual[valid], predicted[valid])
    primary = float(f1_score(actual[valid], predicted[valid], labels=labels, average="macro", zero_division=0))
    tp = np.logical_and(arrays["review_actual"], arrays["review_predicted"]).sum()
    fp = np.logical_and(~arrays["review_actual"], arrays["review_predicted"]).sum()
    fn = np.logical_and(arrays["review_actual"], ~arrays["review_predicted"]).sum()
    return {
        "service_macro_f1": round(module.service_macro_f1(arrays, review_index, qna_index), 6),
        "review_macro_f1": round(review, 6),
        "qna_primary_macro_f1": round(primary, 6),
        "qna_primary_accuracy": round(float(np.mean(actual[valid] == predicted[valid])), 6),
        "qna_secondary_macro_f1": round(secondary, 6),
        "review_micro_precision": round(float(tp / max(tp + fp, 1)), 6),
        "review_micro_recall": round(float(tp / max(tp + fn, 1)), 6),
        "rows": int(len(review_index) + len(qna_index)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--release-dir", required=True)
    parser.add_argument("--consensus", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    module = _bootstrap_module()
    release = Path(args.release_dir)
    gold = {json.loads(line)["sample_id"]: json.loads(line)
            for line in (release / "validation.jsonl").read_text().splitlines() if line.strip()}
    consensus = {row["sample_id"]: row for row in
                 (json.loads(line) for line in Path(args.consensus).read_text().splitlines() if line.strip())}

    status = Counter(consensus.get(sid, {}).get("status", "missing") for sid in gold)
    full, accepted = [], []
    for sample_id, row in gold.items():
        entry = consensus.get(sample_id, {})
        ok = entry.get("status") == "accepted"
        labels = entry.get("labels", []) if ok else []
        prediction = _to_prediction(sample_id, row["record_type"], labels)
        full.append(prediction)
        if ok:
            accepted.append(prediction)

    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    result = {"status_counts": dict(status), "coverage": round(status["accepted"] / len(gold), 6)}
    for name, predictions in (("full_coverage", full), ("accepted_subset", accepted)):
        path = out / f"predictions_{name}.jsonl"
        path.write_text("".join(json.dumps(p, ensure_ascii=False) + "\n" for p in predictions))
        result[name] = _components(module, module.build_arrays(release, path))
    (out / "silver_ceiling.json").write_text(json.dumps(result, ensure_ascii=False, indent=2))
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
