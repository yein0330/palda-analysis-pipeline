"""Human-reviewable annotation queue, draft schema, and validation helpers."""

from __future__ import annotations

import csv
import hashlib
import json
import shutil
from pathlib import Path
from typing import Any, Literal

from pydantic import Field, model_validator

from .pilot import AnnotationSample
from .schemas import StrictModel


class AllowedTheme(StrictModel):
    theme_id: str
    label: str


class AnnotationQueueRow(StrictModel):
    schema_version: Literal["palda_annotation_queue_v1"] = "palda_annotation_queue_v1"
    annotation_id: str
    split: Literal["train", "validation", "test"]
    product_id: str
    product_name: str
    category_path: str
    record_type: Literal["review", "qna"]
    text: str = Field(min_length=1)
    score: float | None = None
    allowed_themes: list[AllowedTheme]


class DraftMention(StrictModel):
    theme_id: str
    polarity: Literal["positive", "negative", "neutral", "concern"]
    evidence_quote: str = Field(min_length=1)


class DraftLabel(StrictModel):
    record_id: str
    qna_stage: Literal["pre_purchase", "post_purchase_support", "irrelevant"] | None = None
    mentions: list[DraftMention] = Field(default_factory=list)
    new_theme_candidates: list[str] = Field(default_factory=list)
    confidence: Literal["low", "medium", "high"]

    @model_validator(mode="after")
    def unique_themes(self) -> "DraftLabel":
        ids = [mention.theme_id for mention in self.mentions]
        if len(ids) != len(set(ids)):
            raise ValueError("duplicate theme_id in mentions")
        return self


class DraftAnnotationRow(StrictModel):
    schema_version: Literal["palda_draft_annotation_v1"] = "palda_draft_annotation_v1"
    input: AnnotationQueueRow
    draft: DraftLabel | None = None
    validation_status: Literal["valid", "needs_review"]
    validation_errors: list[str] = Field(default_factory=list)
    model: str
    raw_output: str
    reviewer_status: Literal["pending", "approved", "corrected", "rejected"] = "pending"
    reviewer_notes: str = ""


class GoldAnnotationRow(StrictModel):
    schema_version: Literal["palda_gold_annotation_v1"] = "palda_gold_annotation_v1"
    input: AnnotationQueueRow
    target: DraftLabel
    reviewer_status: Literal["approved", "corrected"]
    reviewer_notes: str = ""


def _hash(seed: int, value: str) -> bytes:
    return hashlib.sha256(f"{seed}:{value}".encode()).digest()


def _score_bucket(score: float | None) -> str:
    if score is None:
        return "unknown"
    if score <= 2:
        return "negative"
    if score < 4:
        return "neutral"
    return "positive"


def _pick_reviews(records: list[Any], count: int, seed: int) -> list[Any]:
    buckets = {name: [] for name in ("positive", "negative", "neutral", "unknown")}
    for record in records:
        buckets[_score_bucket(record.score)].append(record)
    for values in buckets.values():
        values.sort(key=lambda record: _hash(seed, record.record_id))
    quotas = {"positive": round(count * 0.5), "negative": round(count * 0.3), "neutral": round(count * 0.1)}
    quotas["unknown"] = max(0, count - sum(quotas.values()))
    chosen, leftovers = [], []
    for bucket, values in buckets.items():
        take = min(quotas[bucket], len(values))
        chosen.extend(values[:take]); leftovers.extend(values[take:])
    leftovers.sort(key=lambda record: _hash(seed, record.record_id))
    return (chosen + leftovers[:count - len(chosen)])[:count]


def build_annotation_queue(
    pilot_dir: Path,
    output: Path,
    records_per_product: int = 15,
    qna_per_product: int = 5,
    seed: int = 20260826,
) -> dict[str, Any]:
    output.parent.mkdir(parents=True, exist_ok=True)
    counts = {"products": 0, "review": 0, "qna": 0, "total": 0}
    split_counts = {"train": 0, "validation": 0, "test": 0}
    with output.open("w", encoding="utf-8") as target:
        for split in ("train", "validation", "test"):
            with (pilot_dir / f"{split}.jsonl").open(encoding="utf-8") as source:
                for line in source:
                    sample = AnnotationSample.model_validate_json(line)
                    qnas = sorted(sample.qna, key=lambda record: _hash(seed, record.record_id))[:qna_per_product]
                    reviews = _pick_reviews(sample.reviews, records_per_product - len(qnas), seed)
                    themes = [AllowedTheme(theme_id=x.theme_id, label=x.label) for x in sample.category_themes]
                    for record in [*reviews, *qnas]:
                        text = record.text if record.record_type == "review" else record.question
                        row = AnnotationQueueRow(
                            annotation_id=record.record_id, split=sample.split,
                            product_id=sample.product.product_id, product_name=sample.product.name,
                            category_path=sample.product.category_path or sample.product.cat1,
                            record_type=record.record_type, text=text,
                            score=record.score if record.record_type == "review" else None,
                            allowed_themes=themes,
                        )
                        target.write(row.model_dump_json() + "\n")
                        counts[record.record_type] += 1; counts["total"] += 1; split_counts[split] += 1
                    counts["products"] += 1
    manifest = {
        "schema_version": "palda_annotation_queue_manifest_v1", "source": str(pilot_dir),
        "records_per_product": records_per_product, "qna_target_per_product": qna_per_product,
        "seed": seed, "counts": counts, "splits": split_counts,
    }
    manifest_path = output.with_suffix(".manifest.json")
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return {"output": str(output), "manifest": str(manifest_path), **manifest}


def validate_draft(queue: AnnotationQueueRow, payload: dict[str, Any]) -> tuple[DraftLabel | None, list[str]]:
    errors: list[str] = []
    try:
        draft = DraftLabel.model_validate(payload)
    except Exception as exc:
        return None, [str(exc)]
    if draft.record_id != queue.annotation_id:
        errors.append("record_id does not match input")
    if queue.record_type == "qna" and draft.qna_stage is None:
        errors.append("qna_stage is required for Q&A")
    if queue.record_type == "review" and draft.qna_stage is not None:
        errors.append("qna_stage must be null for Review")
    allowed = {theme.theme_id for theme in queue.allowed_themes}
    for mention in draft.mentions:
        if mention.theme_id not in allowed:
            errors.append(f"theme_id is not allowed: {mention.theme_id}")
        if mention.evidence_quote not in queue.text:
            errors.append(f"evidence_quote is not an exact substring: {mention.evidence_quote!r}")
        if queue.record_type == "review" and mention.polarity == "concern":
            errors.append("Review mention cannot use concern polarity")
        if queue.record_type == "qna" and draft.qna_stage == "pre_purchase" and mention.polarity != "concern":
            errors.append("pre-purchase Q&A mention must use concern polarity")
    if draft.qna_stage == "irrelevant" and draft.mentions:
        errors.append("irrelevant Q&A must have no mentions")
    return draft, errors


def normalize_draft_payload(queue: AnnotationQueueRow, payload: dict[str, Any]) -> dict[str, Any]:
    """Repair fields that are deterministic from source metadata, without inventing labels."""
    normalized = dict(payload)
    if queue.record_type == "review":
        normalized["qna_stage"] = None
    mentions = normalized.get("mentions")
    if not isinstance(mentions, list):
        return normalized
    deduplicated: dict[str, dict[str, Any]] = {}
    for raw in mentions:
        if not isinstance(raw, dict):
            continue
        mention = dict(raw)
        theme_id = str(mention.get("theme_id") or "")
        if queue.record_type == "qna" and normalized.get("qna_stage") == "pre_purchase":
            mention["polarity"] = "concern"
        # One ThemeMention per record is the canonical schema. Preserve the
        # first exact evidence span when a model repeats the same theme.
        deduplicated.setdefault(theme_id, mention)
    normalized["mentions"] = list(deduplicated.values())
    return normalized


def merge_draft_shards(queue_path: Path, shards: list[Path], output: Path) -> dict[str, Any]:
    by_id: dict[str, DraftAnnotationRow] = {}
    for shard in shards:
        with shard.open(encoding="utf-8") as handle:
            for line in handle:
                row = DraftAnnotationRow.model_validate_json(line)
                by_id[row.input.annotation_id] = row
    output.parent.mkdir(parents=True, exist_ok=True)
    missing = valid = needs_review = 0
    with output.open("w", encoding="utf-8") as handle, queue_path.open(encoding="utf-8") as queue:
        for line in queue:
            source = AnnotationQueueRow.model_validate_json(line)
            row = by_id.get(source.annotation_id)
            if row is None:
                missing += 1
                continue
            handle.write(row.model_dump_json() + "\n")
            if row.validation_status == "valid": valid += 1
            else: needs_review += 1
    return {"output": str(output), "total": valid + needs_review, "valid": valid, "needs_review": needs_review, "missing": missing}


def export_review_csv(
    drafts_path: Path,
    output: Path,
    validation_status: Literal["valid", "needs_review"] | None = None,
    limit: int | None = None,
) -> dict[str, Any]:
    columns = [
        "annotation_id", "split", "product_id", "product_name", "category_path", "record_type", "score", "text",
        "qna_stage", "theme_1", "polarity_1", "evidence_1", "theme_2", "polarity_2", "evidence_2",
        "theme_3", "polarity_3", "evidence_3", "new_theme_candidates", "confidence",
        "validation_status", "validation_errors", "reviewer_status", "reviewer_notes",
    ]
    output.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with drafts_path.open(encoding="utf-8") as source, output.open("w", encoding="utf-8-sig", newline="") as target:
        writer = csv.DictWriter(target, fieldnames=columns); writer.writeheader()
        for line in source:
            row = DraftAnnotationRow.model_validate_json(line); draft = row.draft
            if validation_status is not None and row.validation_status != validation_status:
                continue
            if limit is not None and count >= limit:
                break
            values: dict[str, Any] = {
                "annotation_id": row.input.annotation_id, "split": row.input.split,
                "product_id": row.input.product_id, "product_name": row.input.product_name,
                "category_path": row.input.category_path, "record_type": row.input.record_type,
                "score": row.input.score, "text": row.input.text,
                "qna_stage": draft.qna_stage if draft else "", "confidence": draft.confidence if draft else "",
                "new_theme_candidates": " | ".join(draft.new_theme_candidates) if draft else "",
                "validation_status": row.validation_status,
                "validation_errors": " | ".join(row.validation_errors),
                "reviewer_status": row.reviewer_status, "reviewer_notes": row.reviewer_notes,
            }
            for index, mention in enumerate((draft.mentions if draft else [])[:3], 1):
                values[f"theme_{index}"] = mention.theme_id
                values[f"polarity_{index}"] = mention.polarity
                values[f"evidence_{index}"] = mention.evidence_quote
            writer.writerow(values); count += 1
    return {"output": str(output), "rows": count}


def write_draft_report(drafts_path: Path, output: Path) -> dict[str, Any]:
    from collections import Counter

    status: Counter[str] = Counter(); record_types: Counter[str] = Counter()
    stages: Counter[str] = Counter(); themes: Counter[str] = Counter(); polarities: Counter[str] = Counter()
    confidence: Counter[str] = Counter(); mention_counts: Counter[str] = Counter(); errors: Counter[str] = Counter()
    splits: Counter[str] = Counter()
    total = 0
    with drafts_path.open(encoding="utf-8") as source:
        for line in source:
            row = DraftAnnotationRow.model_validate_json(line); total += 1
            status[row.validation_status] += 1; record_types[row.input.record_type] += 1; splits[row.input.split] += 1
            if row.draft is not None:
                stages[str(row.draft.qna_stage)] += row.input.record_type == "qna"
                confidence[row.draft.confidence] += 1; mention_counts[str(len(row.draft.mentions))] += 1
                for mention in row.draft.mentions:
                    themes[mention.theme_id] += 1; polarities[mention.polarity] += 1
            else:
                mention_counts["unparsed"] += 1
            for error in row.validation_errors:
                if "exact substring" in error: errors["evidence_not_exact"] += 1
                elif "JSON object" in error: errors["invalid_json"] += 1
                elif "theme_id is not allowed" in error: errors["theme_not_allowed"] += 1
                else: errors["other_schema_error"] += 1
    report = {
        "schema_version": "palda_draft_report_v1", "annotation_status": "draft_pending_human_review",
        "total": total, "validation_status": dict(status), "splits": dict(splits),
        "record_types": dict(record_types), "qna_stages": dict(stages),
        "mention_counts_per_record": dict(mention_counts), "theme_mentions": dict(themes.most_common()),
        "polarities": dict(polarities), "confidence": dict(confidence), "validation_errors": dict(errors),
        "warning": "valid means schema/evidence checks passed; it does not mean human-approved Gold.",
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return {"output": str(output), **report}


def import_review_csv(drafts_path: Path, review_csv: Path, output: Path, errors_output: Path) -> dict[str, Any]:
    drafts = {
        row.input.annotation_id: row
        for row in (DraftAnnotationRow.model_validate_json(line) for line in drafts_path.read_text(encoding="utf-8").splitlines())
    }
    approved = corrected = rejected = pending = invalid = 0
    errors: list[dict[str, str]] = []
    output.parent.mkdir(parents=True, exist_ok=True)
    with review_csv.open(encoding="utf-8-sig", newline="") as source, output.open("w", encoding="utf-8") as target:
        for csv_row in csv.DictReader(source):
            annotation_id = str(csv_row.get("annotation_id") or "")
            source_row = drafts.get(annotation_id)
            if source_row is None:
                invalid += 1; errors.append({"annotation_id": annotation_id, "error": "unknown annotation_id"}); continue
            reviewer_status = str(csv_row.get("reviewer_status") or "pending").strip().lower()
            if reviewer_status == "pending": pending += 1; continue
            if reviewer_status == "rejected": rejected += 1; continue
            if reviewer_status not in {"approved", "corrected"}:
                invalid += 1; errors.append({"annotation_id": annotation_id, "error": f"invalid reviewer_status: {reviewer_status}"}); continue
            if reviewer_status == "approved":
                target_label = source_row.draft
                validation_errors = [] if target_label is not None else ["approved row has no parseable draft"]
            else:
                mentions = []
                for index in range(1, 4):
                    theme_id = str(csv_row.get(f"theme_{index}") or "").strip()
                    if not theme_id:
                        continue
                    mentions.append({
                        "theme_id": theme_id,
                        "polarity": str(csv_row.get(f"polarity_{index}") or "").strip(),
                        "evidence_quote": str(csv_row.get(f"evidence_{index}") or ""),
                    })
                payload = {
                    "record_id": annotation_id,
                    "qna_stage": str(csv_row.get("qna_stage") or "").strip() or None,
                    "mentions": mentions,
                    "new_theme_candidates": [x.strip() for x in str(csv_row.get("new_theme_candidates") or "").split("|") if x.strip()],
                    "confidence": str(csv_row.get("confidence") or "medium").strip(),
                }
                target_label, validation_errors = validate_draft(source_row.input, normalize_draft_payload(source_row.input, payload))
            if target_label is None or validation_errors:
                invalid += 1
                errors.append({"annotation_id": annotation_id, "error": " | ".join(validation_errors)})
                continue
            gold = GoldAnnotationRow(
                input=source_row.input, target=target_label, reviewer_status=reviewer_status,
                reviewer_notes=str(csv_row.get("reviewer_notes") or ""),
            )
            target.write(gold.model_dump_json() + "\n")
            if reviewer_status == "approved": approved += 1
            else: corrected += 1
    errors_output.parent.mkdir(parents=True, exist_ok=True)
    with errors_output.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["annotation_id", "error"]); writer.writeheader(); writer.writerows(errors)
    return {
        "output": str(output), "errors_output": str(errors_output), "approved": approved,
        "corrected": corrected, "rejected": rejected, "pending": pending, "invalid": invalid,
        "gold": approved + corrected,
    }


def bulk_set_review_status(
    review_csv: Path,
    ids_csv: Path,
    reviewer_status: Literal["approved", "rejected"],
) -> dict[str, Any]:
    with ids_csv.open(encoding="utf-8-sig", newline="") as source:
        ids = {str(row.get("annotation_id") or "") for row in csv.DictReader(source)}
    with review_csv.open(encoding="utf-8-sig", newline="") as source:
        reader = csv.DictReader(source); fieldnames = reader.fieldnames; rows = list(reader)
    if not fieldnames or "reviewer_status" not in fieldnames:
        raise ValueError("review CSV is missing reviewer_status")
    backup = review_csv.with_name(f"{review_csv.stem}.before-bulk-status{review_csv.suffix}")
    shutil.copyfile(review_csv, backup)
    changed = 0
    for row in rows:
        if str(row.get("annotation_id") or "") in ids:
            row["reviewer_status"] = reviewer_status; changed += 1
    temporary = review_csv.with_suffix(review_csv.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8-sig", newline="") as target:
        writer = csv.DictWriter(target, fieldnames=fieldnames); writer.writeheader(); writer.writerows(rows)
    temporary.replace(review_csv)
    return {
        "review_csv": str(review_csv), "backup": str(backup), "requested_ids": len(ids),
        "changed": changed, "reviewer_status": reviewer_status,
    }
