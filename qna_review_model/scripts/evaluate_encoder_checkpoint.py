#!/usr/bin/env python
"""Evaluate a saved PALDA Encoder checkpoint without retraining."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from scripts.train_encoder_transformer import (
    PaldaCollator,
    PaldaDataset,
    PaldaEncoder,
    collect_predictions,
    evaluate_predictions,
    read_jsonl,
    runtime_label_map,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--split", choices=("validation", "test"), default="test")
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(f"refusing to overwrite output: {args.output_dir}")
    config = json.loads((args.checkpoint / "training_config.json").read_text(encoding="utf-8"))
    stored_map = json.loads((args.data_dir / "label_map.json").read_text(encoding="utf-8"))
    label_map = runtime_label_map(stored_map, config["labelSpace"])
    if len(label_map["indexToCanonicalTheme"]) != config["themeCount"]:
        raise ValueError("checkpoint and label map theme counts differ")
    if label_map["indexToCanonicalTheme"] != config["indexToCanonicalTheme"]:
        raise ValueError("checkpoint and label map theme order differs")
    model_path = config["model"]
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    collator = PaldaCollator(tokenizer, label_map, int(config["args"]["max_length"]))
    test = read_jsonl(args.data_dir / f"{args.split}.jsonl")
    loader = DataLoader(PaldaDataset(test), batch_size=args.batch_size, shuffle=False, collate_fn=collator)
    device = torch.device(args.device)
    model = PaldaEncoder(model_path, config["themeCount"], pooling=config["args"].get("pooling", "cls"),
                         review_pooling=config["args"].get("review_pooling", "shared"),
                         qna_pooling=config["args"].get("qna_pooling", "shared")).to(device)
    state = torch.load(args.checkpoint / "model_state.pt", map_location=device, weights_only=True)
    model.load_state_dict(state)
    predictions = collect_predictions(model, loader, device, label_map)
    metrics = evaluate_predictions(predictions, config["reviewThreshold"], config["secondaryThreshold"])
    args.output_dir.mkdir(parents=True)
    with (args.output_dir / f"{args.split}-predictions.jsonl").open("w", encoding="utf-8") as handle:
        for row in predictions:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
    report = {
        "schemaVersion": "palda-encoder-checkpoint-evaluation-v1",
        "checkpoint": str(args.checkpoint), "dataDir": str(args.data_dir),
        "device": str(device), "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "records": len(test), "thresholds": {"review": config["reviewThreshold"], "secondary": config["secondaryThreshold"]},
        "metrics": metrics,
        "split": args.split,
        "warning": "Use dataset provenance for annotation quality; validation is development-only. Do not tune on Test.",
    }
    (args.output_dir / "metrics.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
