#!/usr/bin/env python3
"""Metrics-only W&B sidecar. Dry run by default; no training or dataset access."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import re
import time

from quality_dashboard import MODELS, ROOT, read

METRICS = {
    "service_macro_f1": None,
    "review_theme_sentiment": ("macro_f1", "micro_f1", "micro_precision", "micro_recall", "exact_match"),
    "qna_primary": ("macro_f1", "accuracy"),
    "qna_secondary": ("macro_f1", "micro_f1", "micro_precision", "micro_recall", "exact_match"),
}
CONFIG_KEYS = ("seed", "epochs", "batch_size", "max_length", "silver_weight", "learning_rate",
               "head_learning_rate", "secondary_loss_weight", "gradient_accumulation_steps")


def number(value):
    return isinstance(value, (float, int)) and not isinstance(value, bool) and math.isfinite(value)


def flatten(values, prefix="validation"):
    result = {}
    for key, children in METRICS.items():
        if children is None:
            if number(values.get(key)):
                result[f"{prefix}/{key}"] = values[key]
        else:
            for child in children:
                value = values.get(key, {}).get(child)
                if number(value):
                    result[f"{prefix}/{key}/{child}"] = value
    return result


def payload(path, seed, title, discount):
    metrics = read(path / "metrics.json", {})
    live = read(path / "live.json", {})
    if not metrics and not live:
        return None
    config = {k: v for k, v in metrics.get("config", {}).items()
              if k in CONFIG_KEYS and number(v)}
    config.update(seed=seed, method=title, negative_discount=discount,
                  evaluation_split="validation", production_approved=False)
    history = []
    for row in metrics.get("history", live.get("history", [])):
        epoch = row.get("epoch")
        if not isinstance(epoch, int) or epoch < 1:
            continue
        event = {"epoch": epoch, **flatten(row.get("validation", {}))}
        if number(row.get("train_loss")):
            event["train/epoch_loss"] = row["train_loss"]
        history.append(event)
    stage = live.get("stage", "completed" if metrics else "pending")
    if stage not in {"pending", "loading", "training", "validation_complete", "saving", "completed", "failed"}:
        stage = "unknown"
    summary = {"experiment/stage": stage,
               "evaluation/independent_test": False,
               "upload/contains_source_text": False,
               "upload/contains_weights": False}
    if metrics:
        summary.update(flatten(metrics.get("validation", {}), "selected_validation"))
    for key in ("step", "totalSteps", "loss"):
        if number(live.get(key)):
            summary[f"progress/{key}"] = live[key]
    return {"config": config, "history": history, "summary": summary}


def snapshots(extra_roots):
    sources = dict(MODELS)
    for root in extra_roots:
        root = root.resolve()
        if not root.is_relative_to(ROOT / "artifacts"):
            raise ValueError("Watch roots must be inside encoder artifacts")
        for file in sorted(set(root.glob("*/metrics.json")) | set(root.glob("*/live.json"))):
            folder = file.parent
            live = read(folder / "live.json", {})
            metrics = read(folder / "metrics.json", {})
            seed = metrics.get("config", {}).get("seed", live.get("seed", 0))
            discount = live.get("discount", 1.)
            sources[f"{root.name}-{folder.name}"] = (folder, seed, "additional-experiment", discount)
    return {key: data for key, (path, seed, title, discount) in sources.items()
            if (data := payload(path, seed, title, discount)) is not None}


def settings(wandb):
    return wandb.Settings(console="off", disable_code=True, disable_git=True,
                          save_code=False, disable_job_creation=True, x_disable_meta=True,
                          x_disable_stats=True, x_disable_machine_info=True,
                          init_timeout=30)


def verify_destination(api, entity, project_name, expected_user):
    """Read-only fail-closed check, never change visibility or membership."""
    from wandb_gql import gql
    if api.viewer.username != expected_user:
        raise RuntimeError("Logged-in user mismatch; refusing upload")
    response = api.client.execute(gql(
        "query($name:String!,$entity:String!){project(name:$name,entityName:$entity){name access public readOnly}}"
    ), {"name": project_name, "entity": entity})
    project = response.get("project")
    if not project or project.get("access") not in {"PRIVATE", "RESTRICTED"} or project.get("public") is True:
        raise RuntimeError("Project is missing or not confirmed private; refusing upload")
    if project.get("readOnly") is not False:
        raise RuntimeError("Project is not confirmed writable")
    team = api.team(entity)
    team.load()
    if {m.username for m in team.members} != {expected_user}:
        raise RuntimeError("Destination must have only the expected user as a team member")
    return {"username": expected_user, "entity": entity, "project": project_name,
            "access": project["access"], "teamMembers": [expected_user]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("dry-run", "offline", "online"), default="dry-run")
    parser.add_argument("--entity")
    parser.add_argument("--project")
    parser.add_argument("--expected-user")
    parser.add_argument("--confirm-private", action="store_true",
                        help="Only after checking project visibility and all members in W&B UI")
    parser.add_argument("--watch", action="store_true")
    parser.add_argument("--include-examples", action="store_true",
                        help="Explicit opt-in: upload redacted fixed Validation source/prediction tables")
    parser.add_argument("--watch-root", type=Path, action="append", default=[])
    args = parser.parse_args()
    target = ROOT / ".runtime/wandb-quality"
    target.mkdir(parents=True, exist_ok=True)
    if args.mode == "dry-run":
        data = snapshots(args.watch_root)
        (target / "upload-preview.json").write_text(json.dumps(data, ensure_ascii=False, indent=2))
        print(f"Prepared {len(data)} runs locally; no upload. Preview: {target / 'upload-preview.json'}")
        return

    import wandb
    if args.mode == "online":
        if not all((args.entity, args.project, args.expected_user, args.confirm_private)):
            parser.error("Online requires explicit entity/project/expected-user and verified --confirm-private")
        if not all(re.fullmatch(r"[A-Za-z0-9_.-]+", v) for v in (args.entity, args.project, args.expected_user)):
            parser.error("Use account/project slugs, not URLs or API keys")
        api = wandb.Api(timeout=20)
        verified = verify_destination(api, args.entity, args.project, args.expected_user)
        (target / "destination-check.json").write_text(json.dumps(verified, indent=2))
        print(f"Explicit destination: https://wandb.ai/{args.entity}/{args.project}")

    active, completed = {}, set()
    try:
        while True:
            if args.mode == "online":
                verify_destination(api, args.entity, args.project, args.expected_user)
            for key, data in snapshots(args.watch_root).items():
                if key in completed:
                    continue
                if key not in active:
                    run_id = hashlib.sha256(f"palda-quality-v1/{key}".encode()).hexdigest()[:16]
                    run = wandb.init(entity=args.entity if args.mode == "online" else "local-preview",
                                     project=args.project if args.mode == "online" else "palda-quality-preview",
                                     id=run_id, name=key, group="encoder-label-quality",
                                     config=data["config"], mode=args.mode, dir=str(target),
                                     resume="allow" if args.mode == "online" else None,
                                     reinit="create_new", settings=settings(wandb))
                    run.define_metric("epoch")
                    run.define_metric("validation/*", step_metric="epoch")
                    run.define_metric("train/*", step_metric="epoch")
                    active[key] = (run, int(run.summary.get("tracking/last_epoch", run.summary.get("epoch", 0))))
                run, last_epoch = active[key]
                for event in data["history"]:
                    if event["epoch"] > last_epoch:
                        run.log(event)
                        last_epoch = event["epoch"]
                        run.summary["tracking/last_epoch"] = last_epoch
                if args.include_examples:
                    from wandb_qualitative import encoder_snapshot
                    path = MODELS[key][0] if key in MODELS else None
                    if path is None:
                        for root in args.watch_root:
                            prefix = root.resolve().name + "-"
                            if key.startswith(prefix):
                                possible = root.resolve() / key[len(prefix):]
                                if possible.is_dir() and possible.resolve().is_relative_to(ROOT / "artifacts"):
                                    path = possible
                                    break
                    if path is not None:
                        candidates = sorted(path.glob("epoch-*_predictions.jsonl"))
                        final = path / "validation_predictions.jsonl"
                        if final.exists():
                            candidates.append(final)
                        for snapshot in candidates:
                            digest = hashlib.sha256(snapshot.read_bytes()).hexdigest()
                            summary_key = f"examples_hash/{snapshot.stem}"
                            if run.summary.get(summary_key) != digest:
                                table = encoder_snapshot(snapshot)
                                run.log({f"examples/{snapshot.stem}": wandb.Table(**table)})
                                run.summary[summary_key] = digest
                                run.summary["upload/contains_source_text"] = True
                previously_uploaded_text = bool(run.summary.get("upload/contains_source_text", False))
                run.summary.update(data["summary"])
                if args.include_examples or previously_uploaded_text:
                    run.summary["upload/contains_source_text"] = True
                active[key] = (run, last_epoch)
                if data["summary"]["experiment/stage"] in {"completed", "failed"}:
                    run.finish(exit_code=1 if data["summary"]["experiment/stage"] == "failed" else 0)
                    del active[key]
                    completed.add(key)
            if not args.watch:
                break
            time.sleep(5)
    finally:
        for run, _ in active.values():
            run.summary["sync/watcher_stopped"] = True
            run.finish()


if __name__ == "__main__":
    main()
