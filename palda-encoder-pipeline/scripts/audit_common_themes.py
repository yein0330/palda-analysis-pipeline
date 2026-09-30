#!/usr/bin/env python3
"""Validate/reproduce a Codex-authored exploratory taxonomy audit, not a gold set.

Reads existing validation only. No model calls, training, Test reads or API writes.
Offsets use Python/Unicode code points (end exclusive), NOT JavaScript UTF-16.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
# Private audit row identifiers are not part of the public source release.
# Reconstruct this set from your authorized annotation configuration when auditing.
REVIEW_UNITS: set[tuple[str, int]] = set()
SENTIMENTS = {"긍정", "부정", "혼합", "중립", "해당 없음", "판단 불가"}
ACTS = {"사용 경험", "평가", "정보 질문", "정보 제공", "요청·불만", "기타"}


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def span(text: str, quote: str) -> dict:
    start = text.find(quote)
    if not quote or start < 0:
        raise ValueError(f"Non-verbatim evidence: {quote!r}")
    if text.find(quote, start + 1) >= 0:
        raise ValueError(f"Ambiguous repeated span: {quote!r}")
    return {"start": start, "end": start + len(quote), "text": quote}


def selected_rows(rows: list[dict], seed: str) -> list[dict]:
    groups = defaultdict(list)
    for row in rows:
        groups[(row["category_key"].split(" > ")[0], row["record_type"])].append(row)
    selected = []
    for _, group in sorted(groups.items()):
        # Practical preview exclusion, NOT comprehensive PII detection.
        safe = [r for r in group if len(r["text"]) <= 650 and not re.search(
            r"\d{2,4}[- ]\d{3,4}[- ]\d{4}|@|주소|연락처|계좌", r["text"]
        )]
        selected.extend(sorted(safe, key=lambda r: sha((seed + r["sample_id"]).encode()))[:2])
    return selected


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "artifacts/common-theme-audit-20260928")
    args = parser.parse_args()
    source = ROOT / "data/releases/label-quality-20260910/relabel384/validation.jsonl"
    annotations = ROOT / "configs/common_theme_audit_20260928.json"
    spec = json.loads(annotations.read_text())
    raw = source.read_bytes()
    rows = [json.loads(line) for line in raw.splitlines() if line.strip()]
    selected = selected_rows(rows, spec["selection_seed"])
    assert len(selected) == 40
    assert {r["sample_id"] for r in selected} == set(spec["samples"])
    records, units = [], []
    for row in selected:
        source_id, text = row["sample_id"], row["text"]
        record = {
            "source_id": source_id, "product_id": row["product_id"],
            "category_path": row["category_path"], "record_type": row["record_type"],
            "source_split": "validation", "audit_use": "exploratory_dev_not_gold",
            "source_text": text, "source_text_sha256": sha(text.encode()),
            "author_role": "unknown", "source_position": "question" if row["record_type"] == "qna" else "review",
            "linked_answer": None, "answer_availability": "not_preserved_in_normalized_source" if row["record_type"] == "qna" else "not_applicable",
            "applicability": {f"T{i}": "unknown" for i in range(1, 11)},
            "units": [],
        }
        # Do not infer a verified buyer or seller role from record type alone.
        for index, annotation in enumerate(spec["samples"][source_id]):
            assert annotation["act"] in ACTS
            labels = []
            for theme, subtopic, sentiment in annotation["labels"]:
                assert theme in record["applicability"] and sentiment in SENTIMENTS
                if theme == "T10":
                    assert annotation.get("other_reason")
                labels.append({"theme_id": theme, "subtopic": subtopic, "sentiment": sentiment})
            needs_review = (source_id, index) in REVIEW_UNITS
            assert not needs_review or annotation.get("note")
            unit = {
                "unit_id": f"{source_id}:u{index + 1}", "source_id": source_id,
                "evidence": span(text, annotation["quote"]),
                "context": {"source_id": source_id, "start": 0, "end": len(text)},
                "labels": labels, "speech_act": annotation["act"],
                "secondary_speech_acts": annotation.get("secondary_acts", []),
                "modality": annotation["modality"], "claim_status": "author_statement_not_verified_fact",
                "referent": annotation.get("referent", "current_product_or_transaction"),
                "experiencer": annotation.get("experiencer", "not_explicitly_resolved"),
                "comparison_target": annotation.get("comparison_target"),
                "conditions": [span(text, c) for c in annotation.get("conditions", [])],
                "context_conditions": [span(text, c) for c in annotation.get("context_conditions", [])],
                "candidate_theme_ids": annotation.get("candidates", []),
                "review_required": needs_review,
                "review_reason": annotation.get("note", "") if needs_review else "",
                "decision_rationale": annotation.get("note", ""),
                "other_reason": annotation.get("other_reason"),
                "exclude_from_current_product_claims": annotation.get("exclude_from_current_product_claims", False),
            }
            assert text[unit["evidence"]["start"]:unit["evidence"]["end"]] == unit["evidence"]["text"]
            record["units"].append(unit)
            units.append(unit)
        mentioned = {l["theme_id"] for u in record["units"] for l in u["labels"]}
        record["mention_state_in_this_record"] = {t: "mentioned" if t in mentioned else "not_observed" for t in record["applicability"]}
        record["coverage_note"] = "Meaning-unit audit; greetings/repeated endorsements may be omitted. Source context retained; not token-exhaustive segmentation gold."
        records.append(record)

    theme_units = Counter(l["theme_id"] for u in units for l in u["labels"])
    other_units = [u for u in units if u["other_reason"]]
    summary = {
        "schema": spec["schema"], "author": spec["annotation_author"],
        "source_path": str(source), "source_sha256": sha(raw),
        "annotations_sha256": sha(annotations.read_bytes()),
        "script_sha256": sha(Path(__file__).read_bytes()),
        "selection": "2 reviews and 2 questions per cat1, SHA256 order after <=650-character and limited PII-pattern filter; not representative population sampling",
        "selection_seed": spec["selection_seed"], "source_rows": len(rows),
        "records": len(records), "units": len(units),
        "products": len({r['product_id'] for r in records}),
        "cat1_count": len({r['category_path'].split(' > ')[0] for r in records}),
        "cat2_count": len({r['category_path'].split(' > ')[0] + ' > ' + r['category_path'].split(' > ')[1] for r in records}),
        "record_types": dict(Counter(r['record_type'] for r in records)),
        "theme_unit_counts_multilabel": dict(sorted(theme_units.items(), key=lambda p: int(p[0][1:]))),
        "sentiment_label_counts": dict(Counter(l['sentiment'] for u in units for l in u['labels'])),
        "review_required_units": sum(u['review_required'] for u in units),
        "other_units": len(other_units), "other_unit_fraction": len(other_units) / len(units),
        "other_records": len({u['source_id'] for u in other_units}),
        "other_reason_counts": dict(Counter(u['other_reason'] for u in other_units)),
        "other_subtopics": dict(Counter(l['subtopic'] for u in other_units for l in u['labels'] if l['theme_id'] == 'T10')),
        "verbatim_spans_passed": len(units), "answers_annotated": 0,
        "author_role_verified": 0, "activation_accuracy_measured": False,
        "model_accuracy_measured": False, "latency_measured": False,
        "test_opened": False, "near_duplicate_audit_done": False,
        "raw_qna_probe": {"chunk": "content-chunks/kind=mixed/writer=legacy-import/part-000001.jsonl", "selection": "first 100 Q&A pages only; separate schema probe, not annotated sample", "pages": 100, "question_entries": 1974, "hasSellerAnswer_true": 1974, "reply_entries": 0, "nested_item_fields": []},
    }
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "audit.json").write_text(json.dumps({"summary": summary, "records": records}, ensure_ascii=False, indent=2) + "\n")
    lines = ["# 공통 테마 실제 표본 주석 — 탐색용", "", "Codex 직접 주석. 사람 정답·모델 성능 평가가 아니며 외부 공개 전 개인정보 재검토 필요.", "", f"원문 {len(records)}건 / 의미 단위 {len(units)}개. offset은 Unicode code point, 끝 위치 제외.", ""]
    for record in records:
        lines += [f"## {record['source_id']}", "", f"{record['category_path']} · {record['record_type']} · 상품 {record['product_id']}", "", "원문:", ""]
        lines.extend("> " + line for line in record['source_text'].splitlines())
        lines += ["", "| 근거 구절 | 테마·세부주제·감성 | 발화/양태 | 검토·판단 이유 |", "|---|---|---|---|"]
        for u in record['units']:
            cells = [u['evidence']['text'], "; ".join(f"{l['theme_id']} {l['subtopic']} ({l['sentiment']})" for l in u['labels']), f"{u['speech_act']} / {u['modality']}", ("검토 필요: " if u['review_required'] else "") + u['decision_rationale']]
            lines.append("| " + " | ".join(c.replace("|", "\\|").replace("\n", "<br>") for c in cells) + " |")
        lines.append("")
    (args.output / "samples.md").write_text("\n".join(lines) + "\n")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
