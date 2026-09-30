from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any


SENTIMENTS = {"positive", "negative", "neutral"}
RECORD_TYPES = {"review", "qna"}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass(frozen=True)
class Label:
    theme_id: str
    sentiment: str | None = None
    evidence: str | None = None
    role: str | None = None

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "Label":
        theme_id = str(value.get("theme_id") or value.get("themeId") or "").strip()
        sentiment = value.get("sentiment")
        if sentiment is not None:
            sentiment = str(sentiment).strip().lower()
        evidence = value.get("evidence")
        if evidence is not None:
            evidence = str(evidence).strip() or None
        role = value.get("role") or value.get("qna_role")
        if role is not None:
            role = str(role).strip().lower() or None
        return cls(theme_id=theme_id, sentiment=sentiment, evidence=evidence, role=role)

    def to_dict(self) -> dict[str, Any]:
        return {
            "theme_id": self.theme_id,
            "sentiment": self.sentiment,
            "evidence": self.evidence,
            "role": self.role,
        }


@dataclass
class Sample:
    sample_id: str
    product_id: str
    record_type: str
    text: str
    category_path: str
    category_key: str = ""
    source_kind: str = "real"
    source_split: str | None = None
    usable: bool = True
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "Sample":
        category_path = str(value.get("category_path") or "").strip()
        return cls(
            sample_id=str(value["sample_id"]),
            product_id=str(value.get("product_id") or value["sample_id"]),
            record_type=str(value["record_type"]).lower(),
            text=str(value["text"]).strip(),
            category_path=category_path,
            category_key=str(value.get("category_key") or make_category_key(category_path)).strip(),
            source_kind=str(value.get("source_kind") or "real"),
            source_split=value.get("source_split"),
            usable=bool(value.get("usable", True)),
            metadata=dict(value.get("metadata") or {}),
        )

    def validate(self) -> list[str]:
        errors: list[str] = []
        if not self.sample_id:
            errors.append("missing_sample_id")
        if self.record_type not in RECORD_TYPES:
            errors.append("invalid_record_type")
        if not self.text:
            errors.append("empty_text")
        return errors

    def to_dict(self) -> dict[str, Any]:
        return {
            "sample_id": self.sample_id,
            "product_id": self.product_id,
            "record_type": self.record_type,
            "text": self.text,
            "category_path": self.category_path,
            "category_key": self.category_key or make_category_key(self.category_path),
            "source_kind": self.source_kind,
            "source_split": self.source_split,
            "usable": self.usable,
            "metadata": self.metadata,
        }


def validate_labels(
    labels: list[Label],
    *,
    record_type: str,
    text: str,
    allowed_themes: set[str] | None = None,
) -> list[str]:
    errors: list[str] = []
    seen: set[tuple[str, str | None]] = set()
    primary_count = 0
    for label in labels:
        if not label.theme_id:
            errors.append("missing_theme_id")
            continue
        identity = (
            label.theme_id,
            label.sentiment if record_type == "review" else None,
        )
        if identity in seen:
            errors.append(f"duplicate_label:{label.theme_id}:{label.sentiment}")
        seen.add(identity)
        if allowed_themes is not None and label.theme_id not in allowed_themes:
            errors.append(f"theme_not_allowed:{label.theme_id}")
        if record_type == "review" and label.sentiment not in SENTIMENTS:
            errors.append(f"invalid_sentiment:{label.theme_id}")
        if record_type == "review" and label.role is not None:
            errors.append(f"review_role_must_be_null:{label.theme_id}")
        if record_type == "qna" and label.sentiment is not None:
            errors.append(f"qna_sentiment_must_be_null:{label.theme_id}")
        if record_type == "qna" and label.role not in {None, "primary", "secondary"}:
            errors.append(f"invalid_qna_role:{label.theme_id}")
        if record_type == "qna" and label.role == "primary":
            primary_count += 1
        if label.evidence and _compact(label.evidence) not in _compact(text):
            errors.append(f"evidence_not_in_text:{label.theme_id}")
    if record_type == "qna" and primary_count > 1:
        errors.append("multiple_qna_primary_labels")
    return errors


def _compact(text: str) -> str:
    return "".join(text.split()).lower()


def make_category_key(category_path: str, levels: int = 2) -> str:
    parts = [part.strip() for part in category_path.split(">") if part.strip()]
    return " > ".join(parts[:levels])
