#!/usr/bin/env python3
"""Finish the predeclared 768-row experiment; never deploy or evaluate Test."""
import concurrent.futures
import json
import os
from pathlib import Path
import subprocess
import time

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "artifacts/label-quality-20260910-round2"
OLD = ROOT / "artifacts/label-quality-20260910"
BASE = ROOT / "data/releases/gold-anchor-experiment-v1/gold_silver"
RELEASE = ROOT / "data/releases/label-quality-20260910/relabel768"
PYTHON = ROOT / ".venv/bin/python"
SEEDS = (20260909, 42, 1337)
COMPONENTS = ("review_theme_sentiment", "qna_primary", "qna_secondary")


def read(path):
    return json.loads(path.read_text())


def write(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")
    temporary.replace(path)


def status(stage, **extra):
    value = {"stage": stage, "updatedUnix": time.time(), **extra}
    write(OUT / "status.json", value)
    print(json.dumps(value), flush=True)


def run(name, args):
    environment = {**os.environ, "OMP_NUM_THREADS": "4", "HF_HUB_OFFLINE": "1"}
    with (OUT / f"{name}.log").open("a") as log:
        subprocess.run([str(a) for a in args], cwd=ROOT, env=environment,
                       stdout=log, stderr=subprocess.STDOUT, check=True)


def compare(before, after):
    deltas = [a["service_macro_f1"] - b["service_macro_f1"]
              for b, a in zip(before, after)]
    components = {k: sum(a[k]["macro_f1"] - b[k]["macro_f1"]
                        for b, a in zip(before, after)) / len(before)
                  for k in COMPONENTS}
    improvement = sum(deltas) / len(deltas)
    return {"baselineMean": sum(b["service_macro_f1"] for b in before) / len(before),
            "candidateMean": sum(a["service_macro_f1"] for a in after) / len(after),
            "pairedDeltas": deltas, "meanImprovement": improvement,
            "componentMeanDeltas": components,
            "methodPassed": improvement >= .002 and sum(x > 0 for x in deltas) >= 2
                            and min(components.values()) >= -.01}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    status("waiting_for_judgments")
    deadline = time.monotonic() + 3600
    judges = [OUT / f"silver-{model}768" for model in ("qwen", "kanana")]
    while not all((p / "summary.json").exists() for p in judges):
        if time.monotonic() > deadline:
            raise TimeoutError("Judgment completion not confirmed within one hour")
        time.sleep(10)
    for path in judges:
        rows = [json.loads(line) for line in (path / "predictions.jsonl").read_text().splitlines()]
        if len(rows) != 768 or len({r["sample_id"] for r in rows}) != 768:
            raise ValueError("Incomplete judgments")
    status("building_train_release")
    committee = OUT / "silver-committee768"
    if not committee.exists():
        run("merge", [PYTHON, "scripts/merge_judge_runs.py", "--inputs", *judges,
                      "--model-ids", "qwen", "kanana", "--output", committee])
    if not RELEASE.exists():
        run("release", [PYTHON, "scripts/build_rejudged_release.py", "--base", BASE,
                        "--votes", committee / "predictions.jsonl", "--judge-manifest",
                        committee / "manifest.json", "--arm", "retrieved", "--mode",
                        "relabel", "--output", RELEASE])
    # Do not resume partially created releases or silently lose earlier repairs.
    manifest = read(RELEASE / "manifest.json")
    prior_rows = {r["sample_id"]: r for r in map(json.loads,
        (ROOT / "data/releases/label-quality-20260910/relabel384/train.jsonl").read_text().splitlines())}
    new_rows = {r["sample_id"]: r for r in map(json.loads,
        (RELEASE / "train.jsonl").read_text().splitlines())}
    assert new_rows.keys() == prior_rows.keys()
    for sid, row in prior_rows.items():
        if row.get("metadata", {}).get("label_repair"):
            assert new_rows[sid] == row, "Earlier repair unexpectedly changed"
    run("stop-judges", [PYTHON, "scripts/manage_llm_servers.py", "stop", "--model-id",
                       "qwen", "--model-id", "kanana", "--runtime-dir",
                       ".runtime/label-quality-20260910", "--wait-seconds", "2"])
    status("training_three_seeds", releaseChanges=manifest["changes"])

    def train(pair):
        gpu, seed = pair
        destination = OUT / f"encoder-relabel768-seed{seed}"
        if destination.exists():
            raise ValueError(f"Refusing to overwrite training output: {destination}")
        run(f"train-seed{seed}", [ROOT / ".venv/bin/palda-encoder", "train-service",
            "--release-dir", RELEASE, "--output-dir", destination,
            "--model-name", "jhgan/ko-sroberta-multitask", "--epochs", "3",
            "--seed", seed, "--device", f"cuda:{gpu}", "--silver-weight", "0.3",
            "--primary-balance-power", "0.5", "--review-balance-power", "0.5",
            "--secondary-balance-power", "1"])

    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
        list(executor.map(train, enumerate(SEEDS)))
    before = [read(OLD / f"encoder-relabel384-seed{s}/metrics.json")["validation"] for s in SEEDS]
    metrics = [read(OUT / f"encoder-relabel768-seed{s}/metrics.json") for s in SEEDS]
    assert all(m["test_audited"] is False and m["test"] is None for m in metrics)
    after = [m["validation"] for m in metrics]
    result = compare(before, after)
    result.update(seeds=SEEDS, baselineValidation=before, candidateValidation=after,
                  releaseChanges=manifest["changes"], testAudited=False,
                  simulationAudited=False, activeRuntimeChanged=False,
                  productionApproved=False)
    champion = before[SEEDS.index(42)]
    eligible = [i for i, v in enumerate(after)
                if v["service_macro_f1"] >= champion["service_macro_f1"] + .002
                and all(v[k]["macro_f1"] >= champion[k]["macro_f1"] - .01 for k in COMPONENTS)]
    result["checkpointQualityPassed"] = result["methodPassed"] and bool(eligible)
    # Method-level improvement does not automatically replace the best checkpoint.
    if result["checkpointQualityPassed"]:
        best = max(eligible, key=lambda i: after[i]["service_macro_f1"])
        candidate = OUT / f"encoder-relabel768-seed{SEEDS[best]}"
        status("benchmarking_candidate")
        run("latency", [PYTHON, "scripts/benchmark_encoder_pair.py", "--runtime-src",
            ROOT.parent / "palda-service-runtime/src", "--validation", BASE / "validation.jsonl",
            "--base", OLD / "encoder-relabel384-seed42", "--candidate", candidate,
            "--output", OUT / "encoder-runtime-latency.json"])
        result["qualityEligibleArtifact"] = str(candidate)
        result["latencyReport"] = str(OUT / "encoder-runtime-latency.json")
        result["promotionStatus"] = "latency_review_required_no_automatic_promotion"
    else:
        result["promotionStatus"] = "retain_relabel384_seed42"
    write(OUT / "comparison.json", result)
    status("completed", comparison=str(OUT / "comparison.json"),
           methodPassed=result["methodPassed"], promotionStatus=result["promotionStatus"])


if __name__ == "__main__":
    os.chdir(ROOT)
    try:
        main()
    except Exception as exc:
        status("failed", error=f"{type(exc).__name__}: {exc}")
        raise
