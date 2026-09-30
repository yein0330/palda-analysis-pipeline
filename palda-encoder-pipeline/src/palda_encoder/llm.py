from __future__ import annotations

import concurrent.futures
import hashlib
import json
import os
import random
import time
import urllib.error
import urllib.request
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from .io import append_jsonl, iter_jsonl, read_json, write_jsonl
from .schema import Label, Sample, utc_now, validate_labels


DEFAULT_SYSTEM_PROMPT = """You label Korean commerce reviews and Q&A.
Use only the allowed theme IDs. A text may have multiple themes.
For reviews, sentiment must be positive, negative, or neutral for every selected theme.
For Q&A, sentiment must be null.
For a private, unavailable, empty, spam, or meaningless Q&A, use UNUSABLE as the primary theme.
Evidence must be an exact substring of the input text.
Return JSON only: {"labels":[{"theme_id":"...","sentiment":"... or null","evidence":"..."}]}.
If no theme applies, return {"labels":[]}.
"""


@dataclass(frozen=True)
class ModelEndpoint:
    id: str
    model: str
    url: str
    api_key_env: str | None = None
    weight: float = 1.0
    timeout_s: float = 120.0
    max_tokens: int = 800
    temperature: float = 0.0
    workers: int | None = None
    system_prompt: str = DEFAULT_SYSTEM_PROMPT

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "ModelEndpoint":
        base_url = str(value.get("base_url") or value.get("url") or "").rstrip("/")
        if not base_url:
            raise ValueError(f"Missing base_url for model {value.get('id')}")
        if base_url.endswith("/v1"):
            base_url += "/chat/completions"
        elif not base_url.endswith("/chat/completions"):
            base_url += "/v1/chat/completions"
        return cls(
            id=str(value["id"]),
            model=str(value["model"]),
            url=base_url,
            api_key_env=value.get("api_key_env"),
            weight=float(value.get("weight", 1.0)),
            timeout_s=float(value.get("timeout_s", 120.0)),
            max_tokens=int(value.get("max_tokens", 800)),
            temperature=float(value.get("temperature", 0.0)),
            workers=(int(value["workers"]) if value.get("workers") is not None else None),
            system_prompt=str(value.get("system_prompt") or DEFAULT_SYSTEM_PROMPT),
        )


def load_endpoints(path: str | Path) -> list[ModelEndpoint]:
    value = read_json(path)
    items = value.get("models", value) if isinstance(value, dict) else value
    if not isinstance(items, list) or not items:
        raise ValueError("Model config must contain a non-empty 'models' list")
    endpoints = [ModelEndpoint.from_dict(item) for item in items]
    if len({item.id for item in endpoints}) != len(endpoints):
        raise ValueError("Model IDs must be unique")
    return endpoints


def label_samples(
    samples_path: str | Path,
    taxonomy_path: str | Path,
    models_path: str | Path,
    output_path: str | Path,
    *,
    max_workers: int = 4,
    retries: int = 2,
    repair_attempts: int | None = None,
    limit: int | None = None,
) -> dict[str, int]:
    # Unusable examples are required to train the usability head. They still
    # carry non-empty placeholder/private/spam text and must be labeled rather
    # than silently discarded here.
    samples = [Sample.from_dict(row) for row in iter_jsonl(samples_path)]
    if limit is not None:
        samples = samples[:limit]
    taxonomy = read_json(taxonomy_path)
    endpoints = load_endpoints(models_path)
    existing: set[tuple[str, str]] = set()
    target = Path(output_path)
    if target.exists():
        existing = {
            (row["sample_id"], row["model_id"])
            for row in iter_jsonl(target)
            if row.get("valid")
        }

    resolved_repairs = retries if repair_attempts is None else max(repair_attempts, 0)
    jobs = [
        (sample, endpoint)
        for sample in samples
        for endpoint in endpoints
        if (sample.sample_id, endpoint.id) not in existing
        and sample.metadata.get("generator_model_id") != endpoint.id
    ]
    buffer: list[dict[str, Any]] = []
    written = valid = failed = 0
    # A single shared queue lets a slower endpoint occupy every worker while
    # faster GPUs sit idle. Give each model its own pool so all GPUs keep making
    # progress independently.
    workers_per_model = max(1, max_workers // len(endpoints))
    pools = {
        endpoint.id: concurrent.futures.ThreadPoolExecutor(
            max_workers=endpoint.workers or workers_per_model
        )
        for endpoint in endpoints
    }
    try:
        futures = [
            pools[endpoint.id].submit(
                _label_one, sample, endpoint, taxonomy, retries, resolved_repairs
            )
            for sample, endpoint in jobs
        ]
        for future in concurrent.futures.as_completed(futures):
            row = future.result()
            buffer.append(row)
            valid += bool(row["valid"])
            failed += not bool(row["valid"])
            # Persist progress throughout long multi-GPU labeling runs. JSONL row
            # order is not semantically meaningful; consensus sorts by ID later.
            if len(buffer) >= 20:
                written += append_jsonl(target, buffer)
                buffer.clear()
    except BaseException:
        # A cancelled long run must not discard the completed rows still below
        # the regular 20-row checkpoint boundary. Do not wait for every queued
        # request: resume mode will safely schedule only missing valid keys.
        if buffer:
            append_jsonl(target, buffer)
            buffer.clear()
        for future in futures:
            future.cancel()
        for pool in pools.values():
            pool.shutdown(wait=False, cancel_futures=True)
        raise
    else:
        for pool in pools.values():
            pool.shutdown(wait=True)
    if buffer:
        written += append_jsonl(target, buffer)
    return {
        "scheduled": len(jobs),
        "written": written,
        "valid": valid,
        "failed": failed,
    }


def generate_synthetic(
    generation_plan_path: str | Path,
    taxonomy_path: str | Path,
    models_path: str | Path,
    samples_output_path: str | Path,
    claims_output_path: str | Path,
    *,
    seed: int = 42,
    retries: int = 2,
) -> dict[str, int]:
    taxonomy = read_json(taxonomy_path)
    endpoints = load_endpoints(models_path)
    themes = {item["id"] for item in taxonomy.get("themes", [])}
    rng = random.Random(seed)
    sample_output = Path(samples_output_path)
    claim_output = Path(claims_output_path)
    completed: Counter[str] = Counter()
    if sample_output.exists():
        for row in iter_jsonl(sample_output):
            target = (row.get("metadata") or {}).get("generation_target")
            if target:
                completed[_target_key(target)] += 1
    counter = 0
    reused = 0
    jobs: list[tuple[int, dict[str, Any], ModelEndpoint, int]] = []

    for target in iter_jsonl(generation_plan_path):
        theme_id = str(target["theme_id"])
        if theme_id not in themes:
            raise ValueError(f"Unknown theme in generation plan: {theme_id}")
        count = int(target.get("count", 1))
        target_key = _target_key(target)
        already_done = min(completed[target_key], count)
        reused += already_done
        for item_index in range(count):
            endpoint = endpoints[counter % len(endpoints)]
            counter += 1
            generation_seed = rng.randint(0, 2**31 - 1)
            if item_index < already_done:
                continue
            jobs.append((counter, target, endpoint, generation_seed))

    generated_samples: list[dict[str, Any]] = []
    claims: list[dict[str, Any]] = []
    generated = failed = 0
    pools = {
        endpoint.id: concurrent.futures.ThreadPoolExecutor(max_workers=endpoint.workers or 1)
        for endpoint in endpoints
    }
    try:
        futures = {
            pools[endpoint.id].submit(
                _generate_one, target, endpoint, retries, generation_seed, taxonomy
            ):
            (item_counter, target, endpoint)
            for item_counter, target, endpoint, generation_seed in jobs
        }
        for future in concurrent.futures.as_completed(futures):
            item_counter, target, endpoint = futures[future]
            try:
                text = future.result()
            except Exception:
                failed += 1
                continue
            sample, claim = _make_generated_rows(
                target, endpoint, text, seed, item_counter, themes, taxonomy
            )
            generated_samples.append(sample)
            claims.append(claim)
            generated += 1
            if len(generated_samples) >= 20:
                append_jsonl(sample_output, generated_samples)
                append_jsonl(claim_output, claims)
                generated_samples.clear()
                claims.clear()
    except BaseException:
        if generated_samples:
            append_jsonl(sample_output, generated_samples)
            append_jsonl(claim_output, claims)
            generated_samples.clear()
            claims.clear()
        for future in futures:
            future.cancel()
        for pool in pools.values():
            pool.shutdown(wait=False, cancel_futures=True)
        raise
    else:
        for pool in pools.values():
            pool.shutdown(wait=True)

    if generated_samples:
        append_jsonl(sample_output, generated_samples)
        append_jsonl(claim_output, claims)
    return {"planned": counter, "generated": generated, "failed": failed, "reused": reused}


def _make_generated_rows(
    target: dict[str, Any], endpoint: ModelEndpoint, text: str, seed: int,
    counter: int, themes: set[str], taxonomy: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    theme_id = str(target["theme_id"])
    sample_id = "syn_" + hashlib.sha256(
        f"{seed}:{counter}:{endpoint.id}:{text}".encode("utf-8")
    ).hexdigest()[:20]
    record_type = str(target.get("record_type") or "review")
    sentiment = target.get("sentiment") if record_type == "review" else None
    secondary_theme_ids = [
        str(value) for value in target.get("secondary_theme_ids", [])
        if str(value) in themes and str(value) != theme_id
    ]
    sample = Sample(
        sample_id=sample_id,
        product_id=sample_id,
        record_type=record_type,
        text=text,
        category_path=str(target.get("category_path") or ""),
        category_key=str(target.get("category_key") or target.get("category_path") or ""),
        source_kind="synthetic",
        usable=bool(target.get("usable", theme_id != "UNUSABLE")),
        metadata={
            "generator_model_id": endpoint.id,
            "generation_target": target,
            "generation_seed": seed,
        },
    ).to_dict()
    claim = {
        "sample_id": sample_id,
        "annotator_id": endpoint.id,
        "model_id": endpoint.id,
        "model_version": endpoint.model,
        "prompt_version": f"synthetic-generator-v2-{target.get('strategy', 'plain')}",
        "taxonomy_version": taxonomy.get("version", "unknown"),
        "labels": (
            [Label(
                theme_id=theme_id,
                sentiment=sentiment,
                role="primary" if record_type == "qna" else None,
            ).to_dict()]
            + [Label(theme_id=value, role="secondary").to_dict() for value in secondary_theme_ids]
        ),
        "valid": True,
        "validation_errors": [],
        "source": "generator_claim",
        "created_at": utc_now(),
    }
    return sample, claim


def _target_key(target: dict[str, Any]) -> str:
    return json.dumps(target, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _label_one(
    sample: Sample,
    endpoint: ModelEndpoint,
    taxonomy: dict[str, Any],
    request_retries: int,
    repair_attempts: int,
) -> dict[str, Any]:
    allowed = set(taxonomy.get("category_themes", {}).get(sample.category_key) or [])
    if not allowed:
        allowed = {item["id"] for item in taxonomy.get("themes", [])}
    prompt = _label_prompt(sample, sorted(allowed))
    started = time.perf_counter()
    content = ""
    try:
        errors: list[str] = []
        for repair_attempt in range(repair_attempts + 1):
            content = _chat(endpoint, prompt, retries=request_retries)
            try:
                labels = _parse_labels(content, record_type=sample.record_type)
                compact_text = "".join(sample.text.split()).lower()
                labels = [
                    Label(
                        theme_id=label.theme_id,
                        sentiment=label.sentiment,
                        role=label.role,
                        evidence=(
                            label.evidence
                            if not label.evidence
                            or "".join(label.evidence.split()).lower() in compact_text
                            else None
                        ),
                    )
                    for label in labels
                ]
                errors = validate_labels(
                    labels,
                    record_type=sample.record_type,
                    text=sample.text,
                    allowed_themes=allowed,
                )
            except (ValueError, KeyError, json.JSONDecodeError) as exc:
                labels = []
                errors = [f"parse_error:{type(exc).__name__}"]
            if not errors or repair_attempt >= repair_attempts:
                break
            prompt = _repair_label_prompt(sample, sorted(allowed), content, errors)
        valid = not errors
        error_message = None
    except Exception as exc:  # Persist failures so the pipeline can retry them later.
        labels = []
        errors = ["request_or_parse_failure"]
        valid = False
        error_message = f"{type(exc).__name__}: {exc}"[:500]
    return {
        "sample_id": sample.sample_id,
        "annotator_id": endpoint.id,
        "model_id": endpoint.id,
        "model_version": endpoint.model,
        "prompt_version": "encoder-label-v1",
        "taxonomy_version": taxonomy.get("version", "unknown"),
        "labels": [label.to_dict() for label in labels],
        "valid": valid,
        "validation_errors": errors,
        "error": error_message,
        "source": "llm_labeler",
        "latency_ms": round((time.perf_counter() - started) * 1000, 3),
        "raw_response_sha256": hashlib.sha256(content.encode("utf-8")).hexdigest() if content else None,
        "created_at": utc_now(),
    }


def _generate_one(
    target: dict[str, Any], endpoint: ModelEndpoint, retries: int, generation_seed: int,
    taxonomy: dict[str, Any] | None = None,
) -> str:
    record_type = str(target.get("record_type") or "review")
    sentiment = target.get("sentiment")
    strategy = str(target.get("strategy") or "plain")
    allowed = set((taxonomy or {}).get("category_themes", {}).get(
        str(target.get("category_key") or target.get("category_path") or ""), []
    ))
    theme_rows = {
        str(item["id"]): item for item in (taxonomy or {}).get("themes", [])
    }
    relevant_ids = [str(target["theme_id"]), *map(str, target.get("secondary_theme_ids", [])),
                    *map(str, target.get("confusable_theme_ids", []))]
    theme_context = []
    for theme_id in dict.fromkeys(relevant_ids):
        if allowed and theme_id not in allowed:
            continue
        row = theme_rows.get(theme_id, {})
        theme_context.append({
            "theme_id": theme_id,
            "name": row.get("review_label" if record_type == "review" else "qna_label", theme_id),
            "definition": (target.get("theme_definitions") or {}).get(theme_id),
        })
    strategy_rules = {
        "plain": "목표 라벨이 분명한 현실적인 문장 한 건을 작성하세요.",
        "attributed": (
            "아래 attributes를 모두 반영해 문체와 상황을 다양화하세요. 목표 라벨의 단어를 그대로 "
            "반복하지 말고 실제 구매자가 쓴 것처럼 작성하세요."
        ),
        "contrastive": (
            "목표 테마와 confusable themes의 의미 경계를 시험하는 어려운 문장을 작성하세요. "
            "목표 의미는 명시하되 forbidden themes의 의미는 포함하지 마세요. 단순히 테마 이름을 쓰지 마세요."
        ),
        "source_transform": (
            "source_text의 의미와 라벨은 그대로 유지하면서 표현, 어순, 길이 또는 오타 스타일을 바꾸세요. "
            "새로운 상품 사실이나 새 테마를 추가하지 마세요. 원문을 그대로 복사하지 마세요."
        ),
        "compositional": (
            "primary와 secondary 목표가 각각 명시적으로 드러나는 복합 문장을 작성하세요. "
            "Review라면 테마별 감정이 섞이지 않게 하고, Q&A라면 핵심 질문 의도가 primary가 되게 하세요."
        ),
    }
    prompt = (
        "한국 온라인 쇼핑몰의 encoder 학습용 텍스트를 정확히 한 건 생성하세요.\n"
        f"strategy: {strategy}\n"
        f"strategy_rule: {strategy_rules.get(strategy, strategy_rules['plain'])}\n"
        f"record_type: {record_type}\n"
        f"category: {target.get('category_path', '')}\n"
        f"target_theme: {target['theme_id']}\n"
        f"secondary_themes: {json.dumps(target.get('secondary_theme_ids', []), ensure_ascii=False)}\n"
        f"confusable_themes: {json.dumps(target.get('confusable_theme_ids', []), ensure_ascii=False)}\n"
        f"forbidden_themes: {json.dumps(target.get('forbidden_theme_ids', []), ensure_ascii=False)}\n"
        f"theme_context: {json.dumps(theme_context, ensure_ascii=False)}\n"
        f"target_sentiment: {sentiment if record_type == 'review' else 'null'}\n"
        f"attributes: {json.dumps(target.get('attributes', {}), ensure_ascii=False)}\n"
        f"source_text: {json.dumps(target.get('source_text'), ensure_ascii=False)}\n"
        f"difficulty: {target.get('difficulty', 'normal')}\n"
        f"constraints: {target.get('constraints', '')}\n"
        f"generation_seed: {generation_seed}\n"
        "금지: 라벨 ID 노출, 메타 설명, 정답 해설, JSON 밖의 텍스트.\n"
        "Return JSON only: {\"text\":\"...\"}"
    )
    last_error: Exception | None = None
    for _ in range(retries + 1):
        # The outer loop already owns generation retries. Retrying again inside
        # _chat multiplies the endpoint timeout and can pin a whole batch.
        content = _chat(endpoint, prompt, retries=0, temperature=max(endpoint.temperature, 0.7))
        try:
            parsed = _extract_json(content)
            text = str(parsed.get("text") or "").strip() if isinstance(parsed, dict) else ""
            if text and text != str(target.get("source_text") or "").strip():
                return text
            last_error = ValueError("Generator returned empty or unchanged text")
        except (ValueError, KeyError, json.JSONDecodeError) as exc:
            last_error = exc
        prompt += "\n이전 응답은 무효입니다. 완전한 JSON 객체 한 개만 다시 반환하세요."
    raise last_error or ValueError("Generator returned invalid text")


def _label_prompt(sample: Sample, allowed_themes: list[str]) -> str:
    if sample.record_type == "qna":
        output_rule = (
            'Return exactly one JSON object: {"labels":[{"theme_id":"ALLOWED_ID",'
            '"sentiment":null,"role":"primary|secondary","evidence":"exact Korean substring"}]}. '
            "For an analyzable Q&A, mark exactly one most important theme as primary and any "
            "additional inquiry themes as secondary. Use OTHER as primary when no specific theme applies."
            " Use UNUSABLE as primary for private, unavailable, spam, empty, or meaningless text."
        )
    else:
        output_rule = (
            'Return exactly one JSON object: {"labels":[{"theme_id":"ALLOWED_ID",'
            '"sentiment":"positive|negative|neutral","role":null,'
            '"evidence":"exact Korean substring"}]}. '
            "The same theme may appear twice only when the text explicitly expresses both positive "
            "and negative evidence for that theme."
        )
    return (
        f"record_type: {sample.record_type}\n"
        f"category_key: {sample.category_key}\n"
        f"full_category_path: {sample.category_path}\n"
        f"allowed_theme_ids: {json.dumps(allowed_themes, ensure_ascii=False)}\n"
        f"text: {json.dumps(sample.text, ensure_ascii=False)}\n"
        + output_rule
    )


def _repair_label_prompt(
    sample: Sample, allowed_themes: list[str], previous: str, errors: list[str]
) -> str:
    return (
        _label_prompt(sample, allowed_themes)
        + "\nThe previous answer was invalid. Correct every listed error and return JSON only."
        + f"\nvalidation_errors: {json.dumps(errors, ensure_ascii=False)}"
        + f"\nprevious_answer: {json.dumps(previous, ensure_ascii=False)}"
    )


def _parse_labels(content: str, *, record_type: str | None = None) -> list[Label]:
    parsed = _extract_json(content)
    raw_labels = parsed.get("labels", [])
    if not isinstance(raw_labels, list):
        raise ValueError("'labels' must be a list")
    normalized: list[dict[str, Any]] = []
    seen: set[tuple[str, str | None]] = set()
    for value in raw_labels:
        if not isinstance(value, dict):
            continue
        item = dict(value)
        theme_id = str(item.get("theme_id") or item.get("themeId") or "").strip()
        if isinstance(item.get("sentiment"), str):
            item["sentiment"] = item["sentiment"].lower()
        # Q&A has no sentiment target. Some otherwise-correct instruction-tuned
        # models emit a sentiment despite the prompt, so canonicalize it here.
        if record_type == "qna":
            item["sentiment"] = None
        identity = (theme_id, item.get("sentiment") if record_type == "review" else None)
        if identity in seen:
            continue
        seen.add(identity)
        normalized.append(item)
    return [Label.from_dict(value) for value in normalized]


def _chat(
    endpoint: ModelEndpoint,
    user_prompt: str,
    *,
    retries: int,
    temperature: float | None = None,
) -> str:
    if "qwen3" in endpoint.model.lower():
        user_prompt = "/no_think\n" + user_prompt
    payload = {
        "model": endpoint.model,
        "messages": [
            {"role": "system", "content": endpoint.system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": endpoint.temperature if temperature is None else temperature,
        "max_tokens": endpoint.max_tokens,
    }
    headers = {"Content-Type": "application/json"}
    if endpoint.api_key_env:
        api_key = os.environ.get(endpoint.api_key_env)
        if not api_key:
            raise ValueError(f"Environment variable {endpoint.api_key_env} is not set")
        headers["Authorization"] = f"Bearer {api_key}"
    request = urllib.request.Request(
        endpoint.url,
        data=json.dumps(payload).encode("utf-8"),
        headers=headers,
        method="POST",
    )
    for attempt in range(retries + 1):
        try:
            with urllib.request.urlopen(request, timeout=endpoint.timeout_s) as response:
                return _decode_chat_body(
                    response.read().decode("utf-8"),
                    response.headers.get("Content-Type", ""),
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
    stripped = content.strip()
    if stripped.startswith("```"):
        lines = stripped.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        stripped = "\n".join(lines).strip()
    try:
        value = json.loads(stripped)
    except json.JSONDecodeError:
        start, end = stripped.find("{"), stripped.rfind("}")
        if start < 0 or end <= start:
            raise
        value = json.loads(stripped[start : end + 1])
    if isinstance(value, list):
        value = {"labels": value}
    if not isinstance(value, dict):
        raise ValueError("LLM response must be a JSON object or labels array")
    return value
