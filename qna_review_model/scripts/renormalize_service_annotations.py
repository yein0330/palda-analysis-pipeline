#!/usr/bin/env python
"""Reapply the current deterministic service normalization to saved model outputs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from qna_review_model.annotation import AnnotationQueueRow
from qna_review_model.service_annotation import ServiceDraftRow, normalize_service_payload, validate_service_draft


def extract(raw: str) -> dict | None:
    start, end = raw.find("{"), raw.rfind("}")
    if start < 0 or end < start:
        return None
    try:
        value = json.loads(raw[start:end + 1])
    except json.JSONDecodeError:
        return None
    return value if isinstance(value, dict) else None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--queue", type=Path, required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    queue = {
        row.annotation_id: row
        for row in map(AnnotationQueueRow.model_validate_json, args.queue.read_text(encoding="utf-8").splitlines())
    }
    valid = needs_review = 0
    with args.output.open("w", encoding="utf-8") as target:
        for line in args.input.read_text(encoding="utf-8").splitlines():
            old = ServiceDraftRow.model_validate_json(line)
            payload = extract(old.raw_output)
            if payload is None:
                draft, errors = old.draft, old.validation_errors
            else:
                draft, errors = validate_service_draft(queue[old.annotation_id], normalize_service_payload(queue[old.annotation_id], payload))
            status = "valid" if draft is not None and not errors else "needs_review"
            valid += status == "valid"
            needs_review += status == "needs_review"
            target.write(ServiceDraftRow(
                annotation_id=old.annotation_id, draft=draft, validation_status=status,
                validation_errors=errors, model=old.model, raw_output=old.raw_output,
            ).model_dump_json() + "\n")
    print(json.dumps({"output": str(args.output), "valid": valid, "needs_review": needs_review}))


if __name__ == "__main__":
    main()
