#!/usr/bin/env python3
"""Opt-in research loss ablations. Runtime and default trainer are unchanged."""
import argparse
import concurrent.futures
import hashlib
import json
import os
from pathlib import Path
import subprocess

from palda_encoder import service_train as training
from run_label_quality_round2 import compare

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "artifacts/research-loss-20260910"
RELEASE = ROOT / "data/releases/label-quality-20260910/relabel384"
BASE = ROOT / "artifacts/label-quality-20260910"
SEEDS = (20260909, 42, 1337)
ARMS = ("secondary_asl", "silver_negative_discount", "silver_primary_gce")
ORIGINAL_LOSS = training._service_loss


def secondary_elements(torch, logits, targets, pos_weight, mode, silver):
    """Keep positive weighting and reduction; alter only the specified terms."""
    if mode == "secondary_asl":
        p = logits.sigmoid()
        negative_probability = (1 - p + .05).clamp(max=1)
        # ASL gamma_pos=1, gamma_neg=4, detached focal factors.
        positive_focus = (1 - p).detach()
        negative_focus = (1 - negative_probability).pow(4).detach()
        return (-targets * pos_weight * positive_focus * p.clamp_min(1e-8).log()
                - (1 - targets) * negative_focus * negative_probability.clamp_min(1e-8).log())
    loss = torch.nn.functional.binary_cross_entropy_with_logits(
        logits, targets, reduction="none", pos_weight=pos_weight)
    if mode == "silver_negative_discount":
        discount = torch.where(silver[:, None] & (targets == 0), .25, 1.)
        loss = loss * discount
    return loss


def experimental_loss(mode, torch, nn, outputs, batch, primary_class_weights=None,
                      review_pos_weights=3., secondary_pos_weights=3., *, secondary_loss_weight=.5):
    total, details = ORIGINAL_LOSS(torch, nn, outputs, batch, primary_class_weights,
                                  review_pos_weights, secondary_pos_weights,
                                  secondary_loss_weight=secondary_loss_weight)
    if mode == "baseline":
        return total, details
    # This entrypoint fixes Gold weight=1 and Silver weight=.3, without overrides.
    silver = batch["sample_weights"] < 1
    qna = ~batch["review_task"]
    if mode in ("secondary_asl", "silver_negative_discount"):
        logits = outputs["secondary"]
        targets = batch["secondary_targets"]
        mask = batch["non_special_mask"] & qna[:, None]
        weights = batch["sample_weights"]
        old = training._masked_bce(torch, logits, targets, mask, weights, secondary_pos_weights)
        pw = torch.as_tensor(secondary_pos_weights, device=logits.device, dtype=logits.dtype)
        elements = secondary_elements(torch, logits, targets, pw, mode, silver)
        weighted_mask = mask.float() * weights[:, None]
        new = (elements * weighted_mask).sum() / weighted_mask.sum().clamp_min(1)
        total = total + secondary_loss_weight * (new - old)
        details["secondary"] = float(new.detach())
    elif mode == "silver_primary_gce":
        valid = qna & (batch["primary_targets"] >= 0)
        if valid.any():
            logits = outputs["primary"].masked_fill(~batch["candidate_mask"], -1e4)[valid]
            labels = batch["primary_targets"][valid]
            ce = nn.functional.cross_entropy(logits, labels, reduction="none", weight=primary_class_weights)
            pt = logits.softmax(-1).gather(1, labels[:, None]).squeeze(1)
            gce = (1 - pt.pow(.7)) / .7
            if primary_class_weights is not None:
                gce = gce * primary_class_weights[labels]
            each = torch.where(silver[valid], gce, ce)
            weights = batch.get("primary_sample_weights", batch["sample_weights"])[valid]
            old = (ce * weights).sum() / weights.sum().clamp_min(1)
            new = (each * weights).sum() / weights.sum().clamp_min(1)
            total = total + new - old
            details["primary"] = float(new.detach())
    else:
        raise ValueError(mode)
    return total, details


def write(path, value):
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(value, indent=2) + "\n")
    tmp.replace(path)


def single(arm, seed, gpu):
    output = OUT / f"{arm}-seed{seed}"
    if output.exists():
        raise ValueError(f"Refusing overwrite: {output}")
    output.mkdir(parents=True)
    write(output / "research_recipe.json", {
        "arm": arm, "seed": seed, "silverWeight": .3,
        "sourceHash": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "trainerHash": hashlib.sha256(Path(training.__file__).read_bytes()).hexdigest(),
        "trainHash": hashlib.sha256((RELEASE / "train.jsonl").read_bytes()).hexdigest(),
        "validationHash": hashlib.sha256((RELEASE / "validation.jsonl").read_bytes()).hexdigest(),
        "testRead": False, "simulationRead": False,
        "adaptation": "Research-inspired ablation, not a full published benchmark reproduction"})
    training._service_loss = lambda *a, **kw: experimental_loss(arm, *a, **kw)
    try:
        training.train_service_encoder(RELEASE, output, model_name="jhgan/ko-sroberta-multitask",
            epochs=3, seed=seed, device_name=f"cuda:{gpu}", silver_weight=.3,
            primary_balance_power=.5, review_balance_power=.5, secondary_balance_power=1.)
    finally:
        training._service_loss = ORIGINAL_LOSS


def suite():
    OUT.mkdir(parents=True, exist_ok=True)
    before = [json.loads((BASE / f"encoder-relabel384-seed{s}/metrics.json").read_text())["validation"] for s in SEEDS]
    results = {}
    for arm in ARMS:
        write(OUT / "status.json", {"stage": "training", "arm": arm, "completedArms": list(results)})
        def launch(pair):
            gpu, seed = pair
            with (OUT / f"{arm}-seed{seed}.log").open("a") as log:
                subprocess.run([str(ROOT / ".venv/bin/python"), str(Path(__file__).resolve()),
                    "--arm", arm, "--seed", str(seed), "--gpu", str(gpu)], cwd=ROOT,
                    env={**os.environ, "OMP_NUM_THREADS": "4", "HF_HUB_OFFLINE": "1"},
                    stdout=log, stderr=subprocess.STDOUT, check=True)
        with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
            list(executor.map(launch, enumerate(SEEDS)))
        metrics = [json.loads((OUT / f"{arm}-seed{s}/metrics.json").read_text()) for s in SEEDS]
        assert all(m["test_audited"] is False and m["test"] is None for m in metrics)
        after = [m["validation"] for m in metrics]
        results[arm] = {**compare(before, after), "validation": after, "seeds": SEEDS}
        write(OUT / "comparison.json", results)
    write(OUT / "status.json", {"stage": "completed", "completedArms": list(results),
        "testAudited": False, "simulationAudited": False, "runtimeChanged": False,
        "promotion": "manual result review and latency guard required"})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", action="store_true")
    parser.add_argument("--arm", choices=(*ARMS, "baseline"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--gpu", type=int, default=0)
    args = parser.parse_args()
    os.chdir(ROOT)
    if args.suite:
        try:
            suite()
        except Exception as exc:
            write(OUT / "status.json", {"stage": "failed", "error": repr(exc)})
            raise
    elif args.arm:
        single(args.arm, args.seed, args.gpu)
    else:
        parser.error("Use --suite or --arm")
