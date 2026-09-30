from __future__ import annotations

import json
from typing import Any


def format_reward(completions: list[Any], task_type: list[str], **_: Any) -> list[float]:
    rewards = []
    for completion, kind in zip(completions, task_type):
        value = _parse(completion)
        if not value:
            rewards.append(0.0)
        elif kind == "selling_point":
            points = value.get("selling_points")
            rewards.append(1.0 if isinstance(points, list) and 1 <= len(points) <= 3 else 0.0)
        else:
            rewards.append(1.0 if all(key in value for key in ("theme_id", "sentiment", "quote", "summary")) else 0.0)
    return rewards


def grounding_reward(
    completions: list[Any],
    task_type: list[str],
    source_texts: list[list[str]],
    allowed_theme_ids: list[list[str]],
    **_: Any,
) -> list[float]:
    rewards = []
    for completion, kind, texts, allowed in zip(completions, task_type, source_texts, allowed_theme_ids):
        value = _parse(completion)
        if not value:
            rewards.append(0.0)
            continue
        if kind == "selling_point":
            points = value.get("selling_points") or []
            valid = bool(points) and all(
                set(point.get("evidence_theme_ids") or []) <= set(allowed)
                and bool(point.get("evidence_theme_ids"))
                for point in points
                if isinstance(point, dict)
            )
        else:
            quote = str(value.get("quote") or "")
            valid = bool(quote) and any(quote in text for text in texts)
        rewards.append(1.0 if valid else 0.0)
    return rewards


def concise_reward(completions: list[Any], task_type: list[str], **_: Any) -> list[float]:
    rewards = []
    for completion, kind in zip(completions, task_type):
        value = _parse(completion)
        if not value:
            rewards.append(0.0)
        elif kind == "selling_point":
            messages = [str(point.get("message") or "") for point in value.get("selling_points", []) if isinstance(point, dict)]
            rewards.append(1.0 if messages and all(10 <= len(message) <= 100 for message in messages) else 0.0)
        else:
            length = len(str(value.get("summary") or ""))
            rewards.append(1.0 if 15 <= length <= 180 else 0.0)
    return rewards


def reference_reward(
    completions: list[Any], reference_completion: list[str], **_: Any
) -> list[float]:
    return [
        _bigram_f1(_content_text(_parse(completion)), _content_text(_parse(reference)))
        for completion, reference in zip(completions, reference_completion)
    ]


def label_consistency_reward(
    completions: list[Any], reference_completion: list[str], task_type: list[str], **_: Any
) -> list[float]:
    """Require summary outputs to preserve the backend-owned theme and sentiment."""
    rewards = []
    for completion, reference, kind in zip(completions, reference_completion, task_type):
        if kind == "selling_point":
            rewards.append(1.0)
            continue
        value, expected = _parse(completion), _parse(reference)
        rewards.append(1.0 if value and (
            value.get("theme_id") == expected.get("theme_id")
            and value.get("sentiment") == expected.get("sentiment")
        ) else 0.0)
    return rewards


def _parse(completion: Any) -> dict[str, Any]:
    if isinstance(completion, list) and completion:
        completion = completion[-1].get("content", "")
    text = str(completion).strip()
    try:
        value = json.loads(text)
    except json.JSONDecodeError:
        start, end = text.find("{"), text.rfind("}")
        if start < 0 or end <= start:
            return {}
        try:
            value = json.loads(text[start : end + 1])
        except json.JSONDecodeError:
            return {}
    return value if isinstance(value, dict) else {}


def _content_text(value: dict[str, Any]) -> str:
    if "summary" in value:
        return str(value.get("summary") or "")
    return " ".join(
        f"{point.get('title', '')} {point.get('message', '')}"
        for point in value.get("selling_points", [])
        if isinstance(point, dict)
    )


def _bigram_f1(left: str, right: str) -> float:
    left_set = {left[index : index + 2] for index in range(max(len(left) - 1, 0))}
    right_set = {right[index : index + 2] for index in range(max(len(right) - 1, 0))}
    if not left_set or not right_set:
        return 0.0
    overlap = len(left_set & right_set)
    return 2 * overlap / (len(left_set) + len(right_set))
