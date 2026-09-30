#!/usr/bin/env python3
"""Manage local OpenAI-compatible Transformers LLM servers."""

from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any


def load_models(path: Path, selected: set[str]) -> list[dict[str, Any]]:
    value = json.loads(path.read_text(encoding="utf-8"))
    models = value.get("models", value) if isinstance(value, dict) else value
    if not isinstance(models, list) or not models:
        raise ValueError("Model config must contain a non-empty 'models' list")
    result = [item for item in models if not selected or str(item["id"]) in selected]
    missing = selected - {str(item["id"]) for item in result}
    if missing:
        raise ValueError(f"Unknown model IDs: {', '.join(sorted(missing))}")
    return result


def endpoint(model: dict[str, Any], suffix: str) -> str:
    base = str(model.get("base_url") or model.get("url") or "").rstrip("/")
    if not base:
        raise ValueError(f"Missing base_url for {model.get('id')}")
    if not base.endswith("/v1"):
        base += "/v1"
    return base + suffix


def request_json(url: str, payload: dict[str, Any] | None = None, timeout: float = 5) -> Any:
    data = json.dumps(payload).encode() if payload is not None else None
    headers = {"Content-Type": "application/json"} if data else {}
    request = urllib.request.Request(url, data=data, headers=headers, method="POST" if data else "GET")
    with urllib.request.urlopen(request, timeout=timeout) as response:
        body = response.read().decode()
        if "text/event-stream" not in response.headers.get("Content-Type", ""):
            return json.loads(body)
        chunks: list[str] = []
        for line in body.splitlines():
            if not line.startswith("data:") or line[5:].strip() in {"", "[DONE]"}:
                continue
            value = json.loads(line[5:].strip())
            content = value.get("choices", [{}])[0].get("delta", {}).get("content")
            if content:
                chunks.append(str(content))
        return {"choices": [{"message": {"content": "".join(chunks)}}]}


def is_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
        return True
    except (ProcessLookupError, PermissionError):
        return False


def health(model: dict[str, Any], timeout: float = 2) -> tuple[bool, str]:
    try:
        url = endpoint(model, "").removesuffix("/v1") + "/health"
        with urllib.request.urlopen(url, timeout=timeout) as response:
            return response.status == 200, f"HTTP {response.status}"
    except Exception as exc:
        return False, f"{type(exc).__name__}: {exc}"


def start(models: list[dict[str, Any]], runtime: Path, python: str, wait_s: int) -> int:
    runtime.mkdir(parents=True, exist_ok=True)
    failed = 0
    for model in models:
        model_id = str(model["id"])
        ready, detail = health(model)
        if ready:
            print(json.dumps({"id": model_id, "status": "already_running", "detail": detail}))
            continue
        parsed = urllib.parse.urlparse(str(model["base_url"]))
        if parsed.hostname not in {"127.0.0.1", "localhost"}:
            print(json.dumps({"id": model_id, "status": "skipped", "error": "start only manages localhost endpoints"}))
            failed += 1
            continue
        port = parsed.port or 8000
        pid_path = runtime / f"{model_id}.pid"
        if pid_path.exists():
            old_pid = int(pid_path.read_text().strip())
            if is_alive(old_pid):
                print(json.dumps({"id": model_id, "status": "process_alive_but_unhealthy", "pid": old_pid}))
                failed += 1
                continue
            pid_path.unlink()
        command = [
            python, "-m", "transformers.commands.transformers_cli", "serve",
            "--host", parsed.hostname or "127.0.0.1", "--port", str(port),
            "--dtype", str(model.get("dtype", "bfloat16")),
            "--model-timeout", str(model.get("model_timeout_s", 86400)),
            "--force-model", str(model["model"]),
        ]
        if model.get("trust_remote_code", False):
            command += ["--trust-remote-code", "true"]
        environment = os.environ.copy()
        environment["CUDA_VISIBLE_DEVICES"] = str(model.get("gpu", 0))
        environment.setdefault("HF_HUB_OFFLINE", "1")
        log_path = runtime / f"{model_id}.log"
        log_file = log_path.open("ab", buffering=0)
        process = subprocess.Popen(command, stdin=subprocess.DEVNULL, stdout=log_file,
                                   stderr=subprocess.STDOUT, env=environment, start_new_session=True)
        log_file.close()
        pid_path.write_text(str(process.pid), encoding="utf-8")
        deadline = time.monotonic() + wait_s
        while time.monotonic() < deadline and process.poll() is None:
            ready, detail = health(model)
            if ready:
                break
            time.sleep(1)
        if ready:
            print(json.dumps({"id": model_id, "status": "started", "pid": process.pid, "gpu": model.get("gpu", 0)}))
        else:
            state = "exited" if process.poll() is not None else "starting"
            print(json.dumps({"id": model_id, "status": state, "pid": process.pid, "log": str(log_path), "detail": detail}))
            failed += int(state == "exited")
    return failed


def status(models: list[dict[str, Any]], runtime: Path) -> int:
    failed = 0
    for model in models:
        model_id = str(model["id"])
        pid_path = runtime / f"{model_id}.pid"
        pid = int(pid_path.read_text().strip()) if pid_path.exists() else None
        ready, detail = health(model)
        state = "ready" if ready else "process_only" if pid and is_alive(pid) else "down"
        failed += int(not ready)
        print(json.dumps({"id": model_id, "status": state, "pid": pid, "detail": detail}))
    return failed


def smoke(models: list[dict[str, Any]], timeout_s: int) -> int:
    failed = 0
    for model in models:
        model_id = str(model["id"])
        payload = {
            "model": str(model["model"]),
            "messages": [{"role": "user", "content": ("/no_think\n" if "qwen3" in str(model["model"]).lower() else "") + "JSON만 반환하세요: {\"status\":\"ok\"}"}],
            "temperature": 0, "max_tokens": 128,
        }
        started = time.perf_counter()
        try:
            value = request_json(endpoint(model, "/chat/completions"), payload, timeout_s)
            content = str(value["choices"][0]["message"]["content"])
            print(json.dumps({"id": model_id, "status": "ok", "latency_s": round(time.perf_counter() - started, 2), "content": content}, ensure_ascii=False))
        except Exception as exc:
            failed += 1
            print(json.dumps({"id": model_id, "status": "failed", "error": f"{type(exc).__name__}: {exc}"}, ensure_ascii=False))
    return failed


def stop(models: list[dict[str, Any]], runtime: Path, wait_s: int) -> int:
    failed = 0
    for model in models:
        model_id = str(model["id"])
        pid_path = runtime / f"{model_id}.pid"
        if not pid_path.exists():
            print(json.dumps({"id": model_id, "status": "not_managed"}))
            continue
        pid = int(pid_path.read_text().strip())
        if is_alive(pid):
            os.killpg(pid, signal.SIGTERM)
            deadline = time.monotonic() + wait_s
            while is_alive(pid) and time.monotonic() < deadline:
                time.sleep(0.5)
            if is_alive(pid):
                failed += 1
                print(json.dumps({"id": model_id, "status": "still_running", "pid": pid}))
                continue
        pid_path.unlink(missing_ok=True)
        print(json.dumps({"id": model_id, "status": "stopped", "pid": pid}))
    return failed


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("start", "status", "smoke", "stop"))
    parser.add_argument("--config", type=Path, default=Path("configs/models.local.json"))
    parser.add_argument("--runtime-dir", type=Path, default=Path(".runtime/llm"))
    parser.add_argument("--model-id", action="append", default=[])
    parser.add_argument("--python", default=os.environ.get("PALDA_LLM_PYTHON", sys.executable))
    parser.add_argument("--wait-seconds", type=int, default=30)
    parser.add_argument("--timeout-seconds", type=int, default=600)
    args = parser.parse_args()
    models = load_models(args.config, set(args.model_id))
    if args.action == "start":
        code = start(models, args.runtime_dir, args.python, args.wait_seconds)
    elif args.action == "status":
        code = status(models, args.runtime_dir)
    elif args.action == "smoke":
        code = smoke(models, args.timeout_seconds)
    else:
        code = stop(models, args.runtime_dir, args.wait_seconds)
    raise SystemExit(1 if code else 0)


if __name__ == "__main__":
    main()
