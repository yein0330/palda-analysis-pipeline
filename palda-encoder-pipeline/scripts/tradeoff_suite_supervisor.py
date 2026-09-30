#!/usr/bin/env python3
"""Durable, bounded, serial research jobs; no training, publishing or service changes."""
import argparse
import fcntl
import os
from pathlib import Path
import subprocess
import sys
import time

from input_size_tradeoff import ROOT, DEFAULT_OUT, read, write
from pipeline_architecture_tradeoff import OUT, METHODS


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--launch", action="store_true")
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    if args.launch:
        with (OUT / "supervisor.log").open("a") as log:
            child = subprocess.Popen([sys.executable, __file__], cwd=ROOT,
                                     stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
                                     start_new_session=True)
        print(f"Supervisor PID {child.pid}; status: {OUT / 'suite-status.json'}", flush=True)
        return
    lock = (OUT / "suite.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    python = str(ROOT / ".venv/bin/python")
    env = dict(os.environ, HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", OMP_NUM_THREADS="4", MKL_NUM_THREADS="4")
    jobs = [("resume-input-size", "1", [python, "scripts/input_size_tradeoff.py", "run", "--repeats", "2"], DEFAULT_OUT)]
    jobs += [("architecture-"+m, "1", [python, "scripts/pipeline_architecture_tradeoff.py", m], OUT / ("architecture-"+m)) for m in METHODS]
    folders = [DEFAULT_OUT] + [OUT / ("architecture-"+m) for m in METHODS]
    jobs += [("judge-"+folder.name, "3", [python, "scripts/input_size_tradeoff_judge.py", "--output", str(folder)], folder / "judge") for folder in folders]
    # Sync only metrics; destination privacy/identity checks remain fail-closed.
    jobs += [("sync-"+folder.name, "", [str(ROOT / ".venv-tracking/bin/python"), "scripts/input_size_tradeoff_monitor.py", "--output", str(folder)], None) for folder in folders]
    results = []
    for name, gpu, command, target in jobs:
        if target and (target / "status.json").exists() and read(target / "status.json").get("stage") == "completed":
            results.append({"job": name, "status": "already_completed"})
            continue
        if name.startswith("judge-") and read(target.parent / "status.json").get("stage") != "completed":
            results.append({"job": name, "status": "skipped_incomplete_inference"})
            continue
        if name.startswith("sync-") and not (Path(command[-1]) / "status.json").exists():
            continue
        if gpu:
            used = int(subprocess.check_output(["nvidia-smi", "-i", gpu,
                "--query-gpu=memory.used", "--format=csv,noheader,nounits"], text=True).strip())
            if used > 512:
                write(OUT / "suite-status.json", {"stage": "blocked_gpu_occupied", "job": name,
                    "gpu": gpu, "usedMiB": used, "results": results})
                return
        if name == "resume-input-size":
            cases = [read(p) for p in (DEFAULT_OUT / "cases").glob("*.json")]
            if any(c.get("status") != "completed" for c in cases):
                raise RuntimeError("Existing failed size case needs explicit review before resuming")
            for filename in ("load.json", "status.json"):
                source = DEFAULT_OUT / filename
                backup = DEFAULT_OUT / (filename + ".before-resume-20260922")
                if source.exists() and not backup.exists():
                    write(backup, read(source))
        write(OUT / "suite-status.json", {"stage": "running", "job": name, "pid": os.getpid(), "results": results})
        print(f"Starting {name}", flush=True)
        with (OUT / (name + ".log")).open("a") as log:
            process = subprocess.Popen(command, cwd=ROOT, env=dict(env, CUDA_VISIBLE_DEVICES=gpu), stdout=log, stderr=subprocess.STDOUT)
            try:
                code = process.wait(timeout=7200)
            except subprocess.TimeoutExpired:
                process.terminate()
                try:
                    process.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    process.kill(); process.wait()
                code = 124
        results.append({"job": name, "exitCode": code, "finishedAt": time.time()})
        if code and target:
            target.mkdir(parents=True, exist_ok=True)
            write(target / "status.json", {"stage": "failed", "exitCode": code, "log": str(OUT / (name + ".log"))})
    write(OUT / "suite-status.json", {"stage": "finished_with_errors" if any(r.get("exitCode", 0) for r in results) else "completed", "results": results})


if __name__ == "__main__":
    main()
