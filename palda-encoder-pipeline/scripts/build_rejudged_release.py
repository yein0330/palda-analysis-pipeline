#!/usr/bin/env python3
"""Train-only Silver repair/filter ablation with immutable protected splits."""
import argparse
import hashlib
import json
import shutil
from collections import Counter
from pathlib import Path

from palda_encoder.io import iter_jsonl, write_jsonl, write_json
from compare_label_quality import identity


def decision(row, votes, models, mode):
    if row.get("label_tier") != "silver":
        raise ValueError("Only Silver labels may be reconsidered")
    by_model = {v["model_id"]: v for v in votes}
    if len(by_model) != len(votes):
        raise ValueError("Duplicate judge votes")
    complete = set(by_model) == set(models) and all(v["valid"] for v in votes)
    agreed = (complete
              and len({identity(v["labels"]) for v in votes}) == 1)
    if mode == "filter":
        if not complete:
            return row, "unchanged_invalid_votes"
        keep = agreed and identity(votes[0]["labels"]) == identity(row["labels"])
        return (row if keep else None), ("confirmed" if keep else "removed_disagreement")
    if not agreed:
        return row, "unchanged_disagreement"
    labels = votes[0]["labels"]
    changed = identity(labels) != identity(row["labels"])
    result = dict(row)
    if changed:
        result.update(labels=labels, usable=not any(l["theme_id"] == "UNUSABLE" for l in labels))
        result["metadata"] = {**row.get("metadata", {}), "label_repair": {
            "version": "label-quality-v2", "judges": sorted(models),
            "original_labels": row["labels"],
        }}
    return result, "relabelled" if changed else "confirmed"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--votes", type=Path, required=True)
    parser.add_argument("--judge-manifest", type=Path, required=True)
    parser.add_argument("--arm", required=True)
    parser.add_argument("--mode", choices=["filter", "relabel"], required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("Use a new output directory")
    manifest = json.loads(args.judge_manifest.read_text())
    train_path = args.base / "train.jsonl"
    if hashlib.sha256(train_path.read_bytes()).hexdigest() != manifest["sourceTrainHash"]:
        raise ValueError("Judge data does not match base training split")
    models = set(manifest["models"])
    if len(models) < 2 or manifest["mode"] != "silver":
        raise ValueError("Require independent judges on Silver rows")
    selected = set(manifest["selectedIds"])
    votes = [v for v in iter_jsonl(args.votes) if v["arm"] == args.arm]
    grouped = {}
    for vote in votes:
        if vote["sample_id"] not in selected:
            raise ValueError("Unplanned vote")
        grouped.setdefault(vote["sample_id"], []).append(vote)
    train = list(iter_jsonl(train_path))
    if not selected <= {r["sample_id"] for r in train}:
        raise ValueError("Selected sample outside training")
    output, counts = [], Counter()
    for row in train:
        if row["sample_id"] in selected:
            result, status = decision(row, grouped.get(row["sample_id"], []), models, args.mode)
            counts[status] += 1
        else:
            result = row
        if result is not None:
            output.append(result)
    assert [r for r in output if r["label_tier"] == "gold"] == [r for r in train if r["label_tier"] == "gold"]
    args.output.mkdir(parents=True)
    write_jsonl(args.output / "train.jsonl", output)
    hashes = {}
    for name in ("taxonomy.json", "validation.jsonl", "test.jsonl", "simulation.jsonl"):
        shutil.copy2(args.base / name, args.output / name)
        original = hashlib.sha256((args.base / name).read_bytes()).hexdigest()
        assert hashlib.sha256((args.output / name).read_bytes()).hexdigest() == original
        hashes[name] = original
    report = {"mode": args.mode, "arm": args.arm, "changes": dict(counts),
              "inputRows": len(train), "outputRows": len(output),
              "sourceTrainHash": manifest["sourceTrainHash"],
              "votesHash": hashlib.sha256(args.votes.read_bytes()).hexdigest(),
              "protectedHashes": hashes, "goldUnchanged": True,
              "trainIdsAndTexts": "subset of original train; no new examples"}
    write_json(args.output / "manifest.json", report)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
