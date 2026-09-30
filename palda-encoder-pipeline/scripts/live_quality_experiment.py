#!/usr/bin/env python3
"""Bounded negative-label discount sweep; opt-in, no runtime/default changes."""
import argparse
import concurrent.futures
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import time

import research_loss_experiments as research
from run_label_quality_round2 import compare

ROOT = research.ROOT
OUT = ROOT / "artifacts/live-quality-20260911"
SEEDS = research.SEEDS
RELEASE = research.RELEASE


def read(path):
    return json.loads(path.read_text())


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    temporary.replace(path)


def run_name(discount, seed):
    return f"negative-{discount:g}-seed{seed}"


def discounted_elements(torch, logits, targets, pos_weight, silver, discount):
    loss = torch.nn.functional.binary_cross_entropy_with_logits(
        logits, targets, reduction="none", pos_weight=pos_weight)
    return loss * torch.where(silver[:, None] & (targets == 0), discount, 1.)


def train(discount, seed, gpu):
    import torch
    output = OUT / run_name(discount, seed)
    output.mkdir(parents=True, exist_ok=False)
    steps = math.ceil(sum(1 for _ in (RELEASE / "train.jsonl").open()) / 16)
    state = {"stage": "loading", "seed": seed, "discount": discount,
             "step": 0, "totalSteps": steps * 3, "stepsPerEpoch": steps,
             "startedAt": time.time(), "history": []}

    def publish():
        state["updatedAt"] = time.time()
        write(output / "live.json", state)

    publish()
    write(output / "recipe.json", {
        "discount": discount, "seed": seed, "epochs": 3, "silverWeight": .3,
        "trainHash": hashlib.sha256((RELEASE / "train.jsonl").read_bytes()).hexdigest(),
        "validationHash": hashlib.sha256((RELEASE / "validation.jsonl").read_bytes()).hexdigest(),
        "scriptHash": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "researchLossHash": hashlib.sha256(Path(research.__file__).read_bytes()).hexdigest(),
        "trainerHash": hashlib.sha256(Path(research.training.__file__).read_bytes()).hexdigest(),
        "testRead": False, "simulationRead": False, "productionApproved": False})
    old_elements = research.secondary_elements
    old_metrics = research.training._service_metrics
    old_loss = research.training._service_loss

    def elements(torch, logits, targets, pos_weight, mode, silver):
        if mode != "silver_negative_discount":
            return old_elements(torch, logits, targets, pos_weight, mode, silver)
        return discounted_elements(torch, logits, targets, pos_weight, silver, discount)

    def loss(*args, **kwargs):
        value, details = research.experimental_loss("silver_negative_discount", *args, **kwargs)
        state["step"] += 1
        if state["step"] % 25 == 0 or state["step"] == 1:
            state.update(stage="training", loss=float(value.detach()),
                         epoch=min(3, (state["step"] - 1) // steps + 1))
            publish()
        return value, details

    def metrics(np, f1, collected, thresholds, themes):
        result = old_metrics(np, f1, collected, thresholds, themes)
        epoch = len(state["history"]) + 1
        kind = "epoch" if epoch <= 3 else "selected_checkpoint"
        temporary = output / "live_predictions.tmp"
        research.training._write_service_predictions(temporary, collected, thresholds, themes)
        temporary.replace(output / "live_predictions.jsonl")
        if kind == "epoch":
            shutil.copyfile(output / "live_predictions.jsonl", output / f"epoch-{epoch}_predictions.jsonl")
        state["predictionStage"] = f"epoch {epoch}" if kind == "epoch" else kind
        state["validation"] = result
        if kind == "epoch":
            state["history"].append({"epoch": epoch, "validation": result})
        state["stage"] = "validation_complete" if kind == "epoch" else "saving"
        publish()
        return result

    research.secondary_elements = elements
    research.training._service_loss = loss
    research.training._service_metrics = metrics
    try:
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA unavailable; refusing CPU fallback")
        research.training.train_service_encoder(
            RELEASE, output, model_name="jhgan/ko-sroberta-multitask", epochs=3,
            seed=seed, device_name=f"cuda:{gpu}", silver_weight=.3,
            primary_balance_power=.5, review_balance_power=.5, secondary_balance_power=1.,
            audit_test=False)
        state.update(stage="completed", finishedAt=time.time())
        publish()
    except Exception as exc:
        state.update(stage="failed", error=repr(exc))
        publish()
        raise
    finally:
        research.secondary_elements = old_elements
        research.training._service_loss = old_loss
        research.training._service_metrics = old_metrics


def suite():
    OUT.mkdir(parents=True, exist_ok=True)
    if (OUT / "status.json").exists():
        raise RuntimeError("Existing suite: refusing overwrite or duplicate launch")
    status = {"stage": "starting", "startedAt": time.time(), "plannedRuns": 6,
              "completedRuns": 0, "testRead": False, "simulationRead": False,
              "runtimeChanged": False, "productionApproved": False}
    write(OUT / "status.json", status)
    try:
        if shutil.disk_usage(OUT).free < 8 * 1024**3:
            raise RuntimeError("At least 8 GiB free disk required")
        baseline = [read(research.BASE / f"encoder-relabel384-seed{s}/metrics.json")["validation"] for s in SEEDS]
        previous = [read(research.OUT / f"silver_negative_discount-seed{s}/metrics.json")["validation"] for s in SEEDS]
        comparisons = {}
        for discount in (.1, .5):
            status.update(stage="training", discount=discount, updatedAt=time.time())
            write(OUT / "status.json", status)

            def launch(pair):
                gpu, seed = pair
                with (OUT / f"{run_name(discount, seed)}.log").open("x") as log:
                    subprocess.run([str(ROOT / ".venv/bin/python"), str(Path(__file__).resolve()),
                                    "--discount", str(discount), "--seed", str(seed), "--gpu", str(gpu)],
                                   cwd=ROOT, env={**os.environ, "OMP_NUM_THREADS": "4", "HF_HUB_OFFLINE": "1",
                                                  "PYTHONUNBUFFERED": "1"},
                                   stdout=log, stderr=subprocess.STDOUT, check=True)

            with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
                list(pool.map(launch, enumerate(SEEDS)))
            metrics = [read(OUT / run_name(discount, s) / "metrics.json") for s in SEEDS]
            assert all(m["test_audited"] is False and m["test"] is None for m in metrics)
            after = [m["validation"] for m in metrics]
            comparisons[str(discount)] = {
                "versusRepairedBaseline": compare(baseline, after),
                "versusPreviousDiscount025": compare(previous, after),
                "validation": after, "seeds": SEEDS}
            write(OUT / "comparison.json", comparisons)
            status["completedRuns"] += 3
        status.update(stage="completed", finishedAt=time.time(),
                      promotion="none; component quality and isolated latency review required")
        write(OUT / "status.json", status)
    except Exception as exc:
        status.update(stage="failed", error=repr(exc), updatedAt=time.time())
        write(OUT / "status.json", status)
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", action="store_true")
    parser.add_argument("--discount", type=float, choices=(.1, .5))
    parser.add_argument("--seed", type=int, choices=SEEDS, default=42)
    parser.add_argument("--gpu", type=int, default=0)
    args = parser.parse_args()
    if args.suite:
        suite()
    elif args.discount is not None:
        train(args.discount, args.seed, args.gpu)
    else:
        parser.error("Specify --suite or --discount")
