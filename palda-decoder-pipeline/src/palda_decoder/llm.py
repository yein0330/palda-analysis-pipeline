from __future__ import annotations

import concurrent.futures
import hashlib
import json
import os
import re
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .io import append_jsonl, iter_jsonl, read_json


@dataclass(frozen=True)
class Endpoint:
    id: str
    model: str
    url: str
    roles: tuple[str, ...]
    api_key_env: str | None = None
    timeout_s: float = 120.0
    max_tokens: int = 700
    temperature: float = 0.2
    workers: int | None = None

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "Endpoint":
        base = str(value.get("base_url") or value.get("url") or "").rstrip("/")
        if not base:
            raise ValueError(f"Missing base_url for {value.get('id')}")
        if base.endswith("/v1"):
            base += "/chat/completions"
        elif not base.endswith("/chat/completions"):
            base += "/v1/chat/completions"
        roles = value.get("roles") or ["generator", "judge"]
        return cls(
            id=str(value["id"]),
            model=str(value["model"]),
            url=base,
            roles=tuple(str(role) for role in roles),
            api_key_env=value.get("api_key_env"),
            timeout_s=float(value.get("timeout_s", 120)),
            max_tokens=int(value.get("max_tokens", 700)),
            temperature=float(value.get("temperature", 0.2)),
            workers=(int(value["workers"]) if value.get("workers") is not None else None),
        )


def load_endpoints(path: str | Path, role: str) -> list[Endpoint]:
    value = read_json(path)
    items = value.get("models", value) if isinstance(value, dict) else value
    endpoints = [Endpoint.from_dict(item) for item in items]
    selected = [endpoint for endpoint in endpoints if role in endpoint.roles]
    if not selected:
        raise ValueError(f"No model has the '{role}' role")
    return selected


def generate_candidates(
    tasks_path: str | Path,
    models_path: str | Path,
    output_path: str | Path,
    *,
    max_workers: int = 4,
    retries: int = 2,
    limit: int | None = None,
) -> dict[str, int]:
    tasks = list(iter_jsonl(tasks_path))
    if limit is not None:
        tasks = tasks[:limit]
    endpoints = load_endpoints(models_path, "generator")
    target = Path(output_path)
    existing = set()
    if target.exists():
        existing = {(row["task_id"], row["model_id"]) for row in iter_jsonl(target) if row.get("valid")}
    jobs = [
        (task, endpoint)
        for task in tasks
        for endpoint in endpoints
        if (task["task_id"], endpoint.id) not in existing
    ]
    rows: list[dict[str, Any]] = []
    written = valid = failed = 0
    workers_per_model = max(1, max_workers // len(endpoints))
    pools = {
        endpoint.id: concurrent.futures.ThreadPoolExecutor(
            max_workers=endpoint.workers or workers_per_model
        )
        for endpoint in endpoints
    }
    try:
        futures = [
            pools[endpoint.id].submit(_generate_one, task, endpoint, retries)
            for task, endpoint in jobs
        ]
        for future in concurrent.futures.as_completed(futures):
            row = future.result()
            rows.append(row)
            valid += bool(row["valid"])
            failed += not bool(row["valid"])
            if len(rows) >= 100:
                written += append_jsonl(target, rows)
                rows.clear()
    finally:
        for pool in pools.values():
            pool.shutdown(wait=True)
    if rows:
        written += append_jsonl(target, rows)
    return {
        "scheduled": len(jobs),
        "written": written,
        "valid": valid,
        "failed": failed,
    }


def judge_candidates(
    tasks_path: str | Path,
    candidates_path: str | Path,
    models_path: str | Path,
    output_path: str | Path,
    *,
    max_workers: int = 4,
    retries: int = 2,
) -> dict[str, int]:
    tasks = {row["task_id"]: row for row in iter_jsonl(tasks_path)}
    candidates = [row for row in iter_jsonl(candidates_path) if row.get("valid")]
    endpoints = load_endpoints(models_path, "judge")
    target = Path(output_path)
    existing = set()
    if target.exists():
        existing = {
            (row["task_id"], row["candidate_model_id"], row["judge_model_id"])
            for row in iter_jsonl(target)
            if row.get("valid")
        }
    jobs = [
        (tasks[candidate["task_id"]], candidate, endpoint)
        for candidate in candidates
        for endpoint in endpoints
        if endpoint.id != candidate["model_id"]
        and (candidate["task_id"], candidate["model_id"], endpoint.id) not in existing
    ]
    rows: list[dict[str, Any]] = []
    written = valid = failed = 0
    workers_per_model = max(1, max_workers // len(endpoints))
    pools = {
        endpoint.id: concurrent.futures.ThreadPoolExecutor(
            max_workers=endpoint.workers or workers_per_model
        )
        for endpoint in endpoints
    }
    try:
        futures = [
            pools[endpoint.id].submit(_judge_one, task, candidate, endpoint, retries)
            for task, candidate, endpoint in jobs
        ]
        for future in concurrent.futures.as_completed(futures):
            row = future.result()
            rows.append(row)
            valid += bool(row["valid"])
            failed += not bool(row["valid"])
            if len(rows) >= 100:
                written += append_jsonl(target, rows)
                rows.clear()
    finally:
        for pool in pools.values():
            pool.shutdown(wait=True)
    if rows:
        written += append_jsonl(target, rows)
    return {
        "scheduled": len(jobs),
        "written": written,
        "valid": valid,
        "failed": failed,
    }


def _generate_one(task: dict[str, Any], endpoint: Endpoint, retries: int) -> dict[str, Any]:
    started = time.perf_counter()
    raw = ""
    try:
        base_prompt = _generation_prompt(task)
        prompt = base_prompt
        completion: dict[str, Any] = {}
        errors: list[str] = []
        for repair_attempt in range(retries + 1):
            try:
                raw = _chat(endpoint, prompt, retries, temperature=endpoint.temperature)
                completion = _extract_json(raw)
                errors = validate_completion(task, completion)
            except (ValueError, KeyError, json.JSONDecodeError) as exc:
                completion = {}
                errors = [f"parse_error:{type(exc).__name__}"]
            if not errors or repair_attempt >= retries:
                break
            prompt = _repair_generation_prompt(base_prompt, raw, errors)
        valid = not errors
        error = None
    except Exception as exc:
        completion, errors, valid = {}, ["request_or_parse_failure"], False
        error = f"{type(exc).__name__}: {exc}"[:500]
    return {
        "task_id": task["task_id"],
        "task_type": task["task_type"],
        "model_id": endpoint.id,
        "model_version": endpoint.model,
        "completion": completion,
        "valid": valid,
        "validation_errors": errors,
        "error": error,
        "latency_ms": round((time.perf_counter() - started) * 1000, 3),
        "raw_response_sha256": hashlib.sha256(raw.encode()).hexdigest() if raw else None,
    }


def _judge_one(
    task: dict[str, Any], candidate: dict[str, Any], endpoint: Endpoint, retries: int
) -> dict[str, Any]:
    raw = ""
    try:
        prompt = (
            "다음 PALDA 출력 후보를 입력 근거와 실제로 비교해 평가하세요. 예시 값을 복사하거나 "
            "근거 없이 전 항목에 같은 점수를 주지 마세요. grounding은 모든 주장이 입력에 근거하면 1, "
            "무관하면 0입니다. coverage는 핵심 내용 포함 정도, specificity는 구체성, style은 자연스러운 "
            "한국어와 간결성을 뜻합니다. 앞의 네 값은 0~1 사이 JSON number이고, 입력에 없는 사실을 "
            "추가했을 때만 hallucination을 true로 두세요. 설명 없이 grounding, coverage, specificity, "
            "style, hallucination 다섯 key를 가진 JSON object만 반환하세요.\n"
            f"TASK={json.dumps(task, ensure_ascii=False)}\n"
            f"CANDIDATE={json.dumps(candidate['completion'], ensure_ascii=False)}\n"
        )
        raw = _chat(endpoint, prompt, retries, temperature=0.0)
        scores = _extract_json(raw)
        required = ("grounding", "coverage", "specificity", "style")
        if any(not 0 <= float(scores[name]) <= 1 for name in required):
            raise ValueError("Judge scores must be between zero and one")
        scores = {name: float(scores[name]) for name in required} | {
            "hallucination": bool(scores.get("hallucination", False))
        }
        valid, error = True, None
    except Exception as exc:
        scores, valid = {}, False
        error = f"{type(exc).__name__}: {exc}"[:500]
    return {
        "task_id": task["task_id"],
        "candidate_model_id": candidate["model_id"],
        "judge_model_id": endpoint.id,
        "scores": scores,
        "valid": valid,
        "error": error,
        "raw_response_sha256": hashlib.sha256(raw.encode()).hexdigest() if raw else None,
    }


# The runtime reads a selling point's type from its title and derives the
# evidence sentiment from that type, so a complaint titled as a strength is
# shown next to positive reviews. These phrases mark complaint content.
_NEGATIVE_CONTENT = re.compile(
    r"불만|불편|부정적|문제가|문제점|부족|떨어지|떨어진|싫어|아쉽|아쉬|실망|개선이 필요|개선 필요"
)


def selling_point_type(point: dict[str, Any]) -> str:
    """Mirror palda_runtime.aggregation._selling_point_type."""
    explicit = str(point.get("type") or "").lower()
    if explicit in {"core", "hidden", "barrier", "pain"}:
        return explicit
    title = str(point.get("title") or "").lower().replace(" ", "")
    if "숨은" in title:
        return "hidden"
    if "장벽" in title:
        return "barrier"
    if "pain" in title or "페인" in title or "잠재" in title:
        return "pain"
    return "core"


def validate_completion(task: dict[str, Any], completion: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    if task["task_type"] in {"review_summary", "qna_summary"}:
        quote = str(completion.get("quote") or "").strip()
        summary = str(completion.get("summary") or "").strip()
        texts = [str(item["text"]) for item in task.get("source_items", [])]
        if not quote or not any(quote in text for text in texts):
            errors.append("quote_not_grounded")
        if not 15 <= len(summary) <= 180:
            errors.append("summary_length")
        if completion.get("theme_id") != task.get("theme_id"):
            errors.append("theme_mismatch")
        expected_sentiment = task.get("sentiment")
        if completion.get("sentiment") != expected_sentiment:
            errors.append("sentiment_mismatch")
    elif task["task_type"] == "selling_point":
        points = completion.get("selling_points")
        if not isinstance(points, list) or not 1 <= len(points) <= 3:
            errors.append("selling_points_count")
        else:
            allowed = {item["theme_id"] for item in task.get("summary_items", [])}
            for point in points:
                if not str(point.get("title") or "").strip() or not str(point.get("message") or "").strip():
                    errors.append("selling_point_missing_text")
                evidence = set(point.get("evidence_theme_ids") or [])
                if not evidence or not evidence <= allowed:
                    errors.append("selling_point_ungrounded")
                if selling_point_type(point) in {"core", "hidden"} and _NEGATIVE_CONTENT.search(
                    str(point.get("message") or "")
                ):
                    errors.append("selling_point_type_content_mismatch")
    else:
        errors.append("unknown_task_type")
    return sorted(set(errors))


def _generation_prompt(task: dict[str, Any]) -> str:
    if task["task_type"] == "selling_point":
        return (
            "PALDA 셀러용 소구점을 작성하세요. 제공된 요약에서만 근거를 사용하고 과장하지 마세요. "
            "서로 다른 핵심 장점 1~3개를 JSON으로 반환하세요.\n"
            f"INPUT={json.dumps(task, ensure_ascii=False)}\n"
            'OUTPUT={"selling_points":[{"title":"...","message":"...","evidence_theme_ids":["..."]}]}'
        )
    sentiment_rule = (
        f"sentiment는 반드시 {task.get('sentiment')}입니다."
        if task["task_type"] == "review_summary"
        else "sentiment는 null입니다. percentage를 설명에 억지로 반복하지 마세요."
    )
    return (
        "한국어 쇼핑 리뷰/Q&A 묶음을 PALDA 화면용으로 요약하세요. quote는 입력 text의 정확한 부분 문자열이어야 합니다. "
        "summary는 반복되는 패턴을 한두 문장으로 구체적으로 설명하고 입력에 없는 사실을 만들지 마세요. "
        f"{sentiment_rule}\nINPUT={json.dumps(task, ensure_ascii=False)}\n"
        f'OUTPUT={{"theme_id":"{task["theme_id"]}","sentiment":{json.dumps(task.get("sentiment"), ensure_ascii=False)},"quote":"...","summary":"..."}}'
    )


def _repair_generation_prompt(base_prompt: str, previous: str, errors: list[str]) -> str:
    return (
        base_prompt
        + "\n이전 출력이 검증에 실패했습니다. quote는 source_items의 text에서 띄어쓰기와 줄바꿈까지 "
        "완전히 동일한 연속 부분 문자열로 복사하고, summary는 반드시 한국어로 작성하세요. JSON만 반환하세요."
        + f"\nvalidation_errors={json.dumps(errors, ensure_ascii=False)}"
        + f"\nprevious_output={json.dumps(previous, ensure_ascii=False)}"
    )


def _chat(endpoint: Endpoint, prompt: str, retries: int, *, temperature: float) -> str:
    if "qwen3" in endpoint.model.lower():
        prompt = "/no_think\n" + prompt
    payload = {
        "model": endpoint.model,
        "messages": [
            {"role": "system", "content": "You produce strict JSON grounded only in the supplied PALDA data."},
            {"role": "user", "content": prompt},
        ],
        "temperature": temperature,
        "max_tokens": endpoint.max_tokens,
    }
    headers = {"Content-Type": "application/json"}
    if endpoint.api_key_env:
        secret = os.environ.get(endpoint.api_key_env)
        if not secret:
            raise ValueError(f"Environment variable {endpoint.api_key_env} is not set")
        headers["Authorization"] = f"Bearer {secret}"
    request = urllib.request.Request(endpoint.url, json.dumps(payload).encode(), headers, method="POST")
    for attempt in range(retries + 1):
        try:
            with urllib.request.urlopen(request, timeout=endpoint.timeout_s) as response:
                return _decode_chat_body(
                    response.read().decode(), response.headers.get("Content-Type", "")
                )
        except (urllib.error.URLError, TimeoutError, KeyError, json.JSONDecodeError):
            if attempt >= retries:
                raise
            time.sleep(min(2**attempt, 8))
    raise RuntimeError("unreachable")


def _decode_chat_body(body: str, content_type: str = "") -> str:
    """Read both standard OpenAI JSON and Transformers Serve SSE responses."""
    if "text/event-stream" not in content_type and not body.lstrip().startswith("data:"):
        value = json.loads(body)
        return str(value["choices"][0]["message"]["content"])
    chunks: list[str] = []
    for line in body.splitlines():
        if not line.startswith("data:"):
            continue
        data = line[5:].strip()
        if not data or data == "[DONE]":
            continue
        value = json.loads(data)
        choice = value.get("choices", [{}])[0]
        content = choice.get("delta", {}).get("content")
        if content is None:
            content = choice.get("message", {}).get("content")
        if content:
            chunks.append(str(content))
    if not chunks:
        raise ValueError("Chat stream did not contain assistant content")
    return "".join(chunks)


def _extract_json(content: str) -> dict[str, Any]:
    text = content.strip()
    if text.startswith("```"):
        lines = text.splitlines()[1:]
        if lines and lines[-1].strip() == "```":
            lines.pop()
        text = "\n".join(lines).strip()
    try:
        value = json.loads(text)
    except json.JSONDecodeError:
        start, end = text.find("{"), text.rfind("}")
        if start < 0 or end <= start:
            raise
        value = json.loads(text[start : end + 1])
    if not isinstance(value, dict):
        raise ValueError("Expected a JSON object")
    return value
