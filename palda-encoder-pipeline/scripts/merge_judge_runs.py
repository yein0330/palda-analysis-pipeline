#!/usr/bin/env python3
"""Merge separately scheduled independent judges on the exact same planned rows."""
import argparse
import hashlib
from pathlib import Path

from palda_encoder.io import read_json, iter_jsonl, write_json, write_jsonl
from compare_label_quality import summarize
from normalize_label_votes import normalize


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", type=Path, nargs="+", required=True)
    parser.add_argument("--model-ids", nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    manifests = [read_json(p / "manifest.json") for p in args.inputs]
    for key in ("mode", "sourceTrainHash", "taxonomyHash", "rubricHash", "selectedIds", "referenceIds", "arms"):
        if any(m[key] != manifests[0][key] for m in manifests):
            raise ValueError(f"Incompatible plans: {key}")
    rows = [normalize(r) for p in args.inputs for r in iter_jsonl(p / "predictions.jsonl") if r["model_id"] in args.model_ids]
    keys = {(r["sample_id"], r["model_id"], r["arm"]) for r in rows}
    expected = {(sid, model, arm) for sid in manifests[0]["selectedIds"] for model in args.model_ids for arm in manifests[0]["arms"]}
    if len(keys) != len(rows) or keys != expected:
        raise ValueError("Missing or duplicate votes")
    manifest = dict(manifests[0])
    manifest["models"] = {k: v for m in manifests for k, v in m["models"].items() if k in args.model_ids}
    manifest["sourceManifests"] = {str(p): hashlib.sha256((p / "manifest.json").read_bytes()).hexdigest() for p in args.inputs}
    if args.output.exists():
        raise ValueError("Use a new output directory")
    args.output.mkdir(parents=True)
    write_json(args.output / "manifest.json", manifest)
    write_jsonl(args.output / "predictions.jsonl", rows)
    summary = summarize(rows)
    write_json(args.output / "summary.json", summary)
    import json
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
