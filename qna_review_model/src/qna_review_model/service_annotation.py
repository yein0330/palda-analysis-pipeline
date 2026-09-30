"""Separate seller/logistics service evaluation from product theme annotation."""

from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path
from typing import Literal

from pydantic import Field, model_validator

from .annotation import AnnotationQueueRow
from .schemas import StrictModel


ServiceType = Literal[
    "delivery_speed", "delivery_reliability", "delivery_staff", "installation",
    "customer_support", "returns_refunds", "packaging", "after_sales",
]


class ServiceMention(StrictModel):
    service_type: ServiceType
    polarity: Literal["positive", "negative", "neutral", "concern"]
    evidence_quote: str = Field(min_length=1)


class ServiceDraft(StrictModel):
    record_id: str
    service_relevant: bool
    mentions: list[ServiceMention] = Field(default_factory=list, max_length=3)
    confidence: Literal["low", "medium", "high"]

    @model_validator(mode="after")
    def consistent(self) -> "ServiceDraft":
        if self.service_relevant != bool(self.mentions):
            raise ValueError("service_relevant must equal bool(mentions)")
        types = [mention.service_type for mention in self.mentions]
        if len(types) != len(set(types)):
            raise ValueError("duplicate service_type")
        return self


class ServiceDraftRow(StrictModel):
    schema_version: Literal["palda_service_draft_v1"] = "palda_service_draft_v1"
    annotation_id: str
    draft: ServiceDraft | None = None
    validation_status: Literal["valid", "needs_review"]
    validation_errors: list[str] = Field(default_factory=list)
    model: str
    raw_output: str


SERVICE_CUES = {
    "delivery_speed": re.compile(r"(?:배송|도착|출고).{0,12}(?:빠|빨|늦|지연|오래|당일|신속)|(?:빠|빨|늦|지연|오래).{0,12}(?:배송|도착|왔|와서)"),
    "delivery_reliability": re.compile(r"(?:누락|오배송|엉뚱|잘못\s*(?:왔|보내|배송)|안\s*(?:왔|보내)|덜\s*(?:왔|보내)|빠져|수량|따로\s*(?:왔|배송)|배송.{0,18}(?:약속|일정|시간)|(?:주문|시킨).{0,18}(?:다른|아닌|엉뚱)|(?:담아|보내)주.{0,8}(?:않|안))"),
    "delivery_staff": re.compile(r"(?:배송|배달)?\s*기사|배송하시는\s*분|배달하시는\s*분"),
    "installation": re.compile(r"(?:설치|시공).{0,12}(?:해\s*주|해주|기사|업체)|(?:기사|업체).{0,12}(?:설치|시공)|(?:설치|시공)기사"),
    "customer_support": re.compile(r"문의|답변|상담|응대|고객센터|연락|전화"),
    "returns_refunds": re.compile(r"교환|반품|환불"),
    "packaging": re.compile(r"포장|박스|완충|뽁뽁이|포장재"),
    "after_sales": re.compile(r"(?:^|\s)A/?S(?:\s|$)|애프터서비스|(?:무상|유상)\s*수리|수리.{0,8}(?:해\s*주|받|신청|처리|가능)|보증(?:기간|수리|서비스)?"),
}

NEGATIVE_CUES = re.compile(r"안\s|않|없|못|늦|지연|누락|오배송|엉뚱|잘못|불친절|불가|파손|비싸|아쉽|문제|이상")
POSITIVE_CUES = re.compile(r"빠르|빨|친절|꼼꼼|잘해|잘 해|만족|감사|신속|좋")


def _service_sentences(text: str) -> list[str]:
    return [part.strip() for part in re.split(r"(?<=[.!?。！？])\s+|[\r\n]+", text) if part.strip()]


def _polarity(text: str, record_type: str) -> str:
    if NEGATIVE_CUES.search(text):
        return "concern" if record_type == "qna" else "negative"
    if POSITIVE_CUES.search(text):
        return "positive"
    return "concern" if record_type == "qna" else "neutral"


def _rule_mentions(queue: AnnotationQueueRow) -> list[dict]:
    """Recover only explicit, high-precision service mentions missed by the model."""
    recovered: list[dict] = []
    for sentence in _service_sentences(queue.text):
        candidates: list[str] = []
        if SERVICE_CUES["delivery_speed"].search(sentence):
            candidates.append("delivery_speed")
        if SERVICE_CUES["delivery_reliability"].search(sentence):
            candidates.append("delivery_reliability")
        if SERVICE_CUES["delivery_staff"].search(sentence) and re.search(r"친절|불친절|태도|응대|연락|약속", sentence):
            candidates.append("delivery_staff")
        if SERVICE_CUES["installation"].search(sentence):
            candidates.append("installation")
        if SERVICE_CUES["customer_support"].search(sentence) and re.search(r"없|안\s|않|못|늦|친절|불친절|답변|응대|상담", sentence):
            candidates.append("customer_support")
        if SERVICE_CUES["returns_refunds"].search(sentence):
            candidates.append("returns_refunds")
        if SERVICE_CUES["packaging"].search(sentence) and re.search(r"좋|꼼꼼|안전|튼튼|엉망|허술|파손|뜯|찢|찌그러|젖", sentence):
            candidates.append("packaging")
        if SERVICE_CUES["after_sales"].search(sentence) and re.search(r"받|신청|처리|기사|센터|안\s|못|없|친절|불친절", sentence):
            candidates.append("after_sales")
        for service_type in candidates:
            if any(item["service_type"] == service_type for item in recovered):
                continue
            recovered.append({
                "service_type": service_type,
                "polarity": _polarity(sentence, queue.record_type),
                "evidence_quote": sentence,
            })
    return recovered


def normalize_service_payload(queue: AnnotationQueueRow, payload: dict) -> dict:
    normalized = dict(payload); raw_mentions = normalized.get("mentions")
    if not isinstance(raw_mentions, list):
        return normalized
    unique: dict[str, dict] = {}
    for raw in raw_mentions:
        if not isinstance(raw, dict):
            continue
        mention = dict(raw); service_type = str(mention.get("service_type") or "")
        evidence = str(mention.get("evidence_quote") or "")
        cue = SERVICE_CUES.get(service_type)
        if cue is None or not cue.search(evidence):
            continue
        if queue.record_type == "qna" and mention.get("polarity") not in {"positive", "negative", "neutral", "concern"}:
            mention["polarity"] = "concern"
        unique.setdefault(service_type, mention)
    for mention in _rule_mentions(queue):
        unique.setdefault(mention["service_type"], mention)
    normalized["mentions"] = list(unique.values())[:3]
    normalized["service_relevant"] = bool(normalized["mentions"])
    return normalized


def validate_service_draft(queue: AnnotationQueueRow, payload: dict) -> tuple[ServiceDraft | None, list[str]]:
    try:
        draft = ServiceDraft.model_validate(payload)
    except Exception as exc:
        return None, [str(exc)]
    errors = []
    if draft.record_id != queue.annotation_id:
        errors.append("record_id does not match input")
    for mention in draft.mentions:
        if mention.evidence_quote not in queue.text:
            errors.append(f"evidence_quote is not an exact substring: {mention.evidence_quote!r}")
    return draft, errors


def merge_service_shards(queue_path: Path, shards: list[Path], output: Path) -> dict:
    by_id: dict[str, ServiceDraftRow] = {}
    for shard in shards:
        for line in shard.read_text(encoding="utf-8").splitlines():
            row = ServiceDraftRow.model_validate_json(line); by_id[row.annotation_id] = row
    valid = needs_review = missing = 0
    output.parent.mkdir(parents=True, exist_ok=True)
    with queue_path.open(encoding="utf-8") as queue, output.open("w", encoding="utf-8") as target:
        for line in queue:
            source = AnnotationQueueRow.model_validate_json(line); row = by_id.get(source.annotation_id)
            if row is None:
                missing += 1; continue
            target.write(row.model_dump_json() + "\n")
            valid += row.validation_status == "valid"; needs_review += row.validation_status == "needs_review"
    return {"output": str(output), "total": valid + needs_review, "valid": valid, "needs_review": needs_review, "missing": missing}


def write_service_report(drafts_path: Path, output: Path) -> dict:
    status: Counter[str] = Counter(); types: Counter[str] = Counter(); polarities: Counter[str] = Counter()
    relevant = parsed = 0
    for line in drafts_path.read_text(encoding="utf-8").splitlines():
        row = ServiceDraftRow.model_validate_json(line); status[row.validation_status] += 1
        if row.draft is None:
            continue
        parsed += 1; relevant += row.draft.service_relevant
        for mention in row.draft.mentions:
            types[mention.service_type] += 1; polarities[mention.polarity] += 1
    report = {
        "schema_version": "palda_service_report_v1", "total": sum(status.values()), "parsed": parsed,
        "service_relevant": relevant, "validation_status": dict(status),
        "service_types": dict(types.most_common()), "polarities": dict(polarities),
        "warning": "Model-generated service draft pending human review.",
    }
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return {"output": str(output), **report}
