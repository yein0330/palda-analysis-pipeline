"""Audit frozen Train/Validation and prepare a small, explicitly non-blind LaaJ queue.

No Test access, model calls, training, or source-data edits. Existing outputs are refused.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path


def read_rows(path):
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def label_set(row):
    labels = row["labels"]
    if row["sourceType"] == "review":
        return {f"{x['themeId']}::{x['sentiment']}" for x in labels["reviewLabels"]}
    primary = labels.get("qnaPrimaryTheme")
    return ({"primary:" + primary["themeId"]} if primary else set()) | {
        "secondary:" + x["themeId"] for x in labels["qnaSecondaryThemes"]}


def score_sets(truths, predictions):
    tp, fp, fn, support = Counter(), Counter(), Counter(), Counter()
    for truth, pred in zip(truths, predictions, strict=True):
        tp.update(truth & pred); fp.update(pred - truth); fn.update(truth - pred); support.update(truth)
    labels = sorted(set(tp) | set(fp) | set(fn))
    per_label = []
    for label in labels:
        denominator = 2 * tp[label] + fp[label] + fn[label]
        per_label.append({"label": label, "support": support[label], "tp": tp[label],
                          "fp": fp[label], "fn": fn[label],
                          "f1": round(2 * tp[label] / denominator, 6) if denominator else 0})
    denominator = 2 * sum(tp.values()) + sum(fp.values()) + sum(fn.values())
    return {"records": len(truths),
            "microF1": round(2 * sum(tp.values()) / denominator, 6) if denominator else 0,
            "macroF1": round(sum(x["f1"] for x in per_label) / len(per_label), 6) if per_label else 0,
            "perLabel": per_label}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--source", type=Path, help="Optional pre-cleaning source; compare retained Train IDs only")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    train_path, val_path = args.data_dir / "train.jsonl", args.data_dir / "validation.jsonl"
    train, validation = read_rows(train_path), read_rows(val_path)
    predictions = read_rows(args.predictions)
    val_by_id = {r["itemId"]: r for r in validation}
    pred_by_id = {p["itemId"]: p for p in predictions}
    if len(val_by_id) != len(validation) or len(pred_by_id) != len(predictions) or set(val_by_id) != set(pred_by_id):
        raise ValueError("Validation and prediction IDs must be unique and match exactly")
    config_path = args.checkpoint / "training_config.json"
    config = json.loads(config_path.read_text())
    label_map = json.loads((args.data_dir / "label_map.json").read_text())
    taxonomy = json.loads((args.data_dir / "theme_taxonomy.json").read_text())
    categories = {c["level2Id"]: c for c in taxonomy["categories"]}
    mask_errors = []
    for split, rows in (("train", train), ("validation", validation)):
        for r in rows:
            labels = r["labels"]["reviewLabels"] + r["labels"]["qnaSecondaryThemes"]
            if r["labels"].get("qnaPrimaryTheme"):
                labels = labels + [r["labels"]["qnaPrimaryTheme"]]
            allowed = label_map["categoryMasks"].get(r["category"]["level2Id"], [])
            for label in labels:
                index = label_map["categoryThemeToIndex"].get(label["categoryThemeId"])
                if index is None or index not in allowed:
                    mask_errors.append({"split": split, "itemId": r["itemId"], "label": label})
    review_true, review_pred, secondary_true, secondary_pred = [], [], [], []
    buckets = defaultdict(list)
    confusions = Counter()
    for r in validation:
        p = pred_by_id[r["itemId"]]
        if r["sourceType"] != p["sourceType"]:
            raise ValueError("Prediction task mismatch")
        if r["sourceType"] == "review":
            truth = label_set(r)
            if truth != set(p["reviewTruth"]):
                raise ValueError("Stale review prediction labels")
            predicted = {label for label, score in p["reviewCandidates"] if score >= config["reviewThreshold"]}
            review_true.append(truth); review_pred.append(predicted)
        else:
            truth = label_set(r)
            stored = {"primary:" + p["primaryTruth"]} | {"secondary:" + x for x in p["secondaryTruth"]}
            if truth != stored:
                raise ValueError("Stale Q&A prediction labels")
            sec = {label for label, score in p["secondaryCandidates"] if score >= config["secondaryThreshold"]}
            predicted = {"primary:" + p["primaryPrediction"]} | {"secondary:" + x for x in sec}
            secondary_true.append(set(p["secondaryTruth"])); secondary_pred.append(sec)
            if p["primaryTruth"] != p["primaryPrediction"]:
                confusions[(p["primaryTruth"], p["primaryPrediction"])] += 1
        bucket = r["sourceType"] + ("_agreement" if truth == predicted else "_disagreement")
        buckets[bucket].append({"itemId": r["itemId"], "sourceType": r["sourceType"],
            "productGroupId": r["productGroupId"], "category": r["category"], "text": r["text"],
            "reference": sorted(truth), "prediction": sorted(predicted),
            "bucket": bucket, "annotation": r["annotation"],
            "allowedThemes": categories[r["category"]["level2Id"]]["allowedThemes"]})
    # Equal error/control samples per task: diagnostic, deliberately not representative accuracy.
    queue = []
    for bucket in sorted(buckets):
        ranked = sorted(buckets[bucket], key=lambda r: hashlib.sha256(
            ("palda-laaj-pilot-v1:" + r["itemId"]).encode()).hexdigest())
        queue.extend(ranked[:3])
    retained = {r["itemId"]: r for r in train}
    changed = []
    if args.source:
        with args.source.open(encoding="utf-8") as handle:
            for line in handle:
                original = json.loads(line)
                current = retained.get(original["itemId"])
                if current is None:
                    continue
                removed = label_set(original) - label_set(current)
                if removed:
                    changed.append({"itemId": current["itemId"], "grade": original["annotation"]["grade"],
                        "text": original["text"], "removedLabels": sorted(removed)})
    hashed = [train_path, val_path, config_path, args.predictions,
              args.data_dir / "label_map.json", args.data_dir / "theme_taxonomy.json"]
    if args.source:
        hashed.append(args.source)
    report = {"schemaVersion": "palda-validation-audit-v1",
        "sourceHashes": {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in hashed},
        "records": {"train": len(train), "validation": len(validation)},
        "maskErrors": mask_errors,
        "review": score_sets(review_true, review_pred),
        "qnaSecondary": score_sets(secondary_true, secondary_pred),
        "qnaPrimaryConfusions": [{"truth": t, "prediction": p, "count": n}
                                 for (t, p), n in confusions.most_common()],
        "retainedTrainRowsWithRemovedLabelsByGrade": dict(Counter(r["grade"] for r in changed)),
        "trainGradeWeights": dict(Counter(f"{r['annotation']['grade']}:{r['annotation']['sampleWeight']}" for r in train)),
        "queueBuckets": dict(Counter(r["bucket"] for r in queue)),
        "limitations": ["Test files not accessed", "Queue is non-blind diagnostic sampling, not an accuracy estimate",
                         "Source comparison covers retained Train IDs, not rows dropped entirely",
                         "No labels, thresholds, models or input datasets changed by this audit"]}
    args.output_dir.mkdir(parents=True)
    (args.output_dir / "audit.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    for name, rows in (("judge_queue.jsonl", queue), ("train_label_changes.jsonl", changed)):
        (args.output_dir / name).write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
    print(json.dumps({"output": str(args.output_dir), "maskErrors": len(mask_errors),
                     "queue": len(queue), "changedTrainRows": report["retainedTrainRowsWithRemovedLabelsByGrade"]}))


if __name__ == "__main__":
    main()
