"""Conservative mechanical normalization for Decoder report drafts."""

from __future__ import annotations

from typing import Any


def normalize_output(
    report_input: dict[str, Any], output: dict[str, Any] | None
) -> dict[str, Any] | None:
    """Repair only references and copied fields; never invent or edit summary text."""
    if output is None:
        return None
    review_lookup = {
        (group["themeId"], group["sentiment"]): group
        for group in report_input["reviewGroups"]
    }
    qna_lookup = {group["themeId"]: group for group in report_input["qnaGroups"]}
    all_evidence = {
        evidence["id"]
        for group in [*report_input["reviewGroups"], *report_input["qnaGroups"]]
        for evidence in group["evidence"]
    }

    normalized_review: dict[str, list[dict[str, Any]]] = {
        "positiveThemes": [],
        "negativeThemes": [],
    }
    source_analysis = output.get("reviewAnalysis")
    if not isinstance(source_analysis, dict):
        source_analysis = {}
    for field, sentiment in (
        ("positiveThemes", "positive"),
        ("negativeThemes", "negative"),
    ):
        items = source_analysis.get(field)
        if not isinstance(items, list):
            continue
        for item in items[:3]:
            if not isinstance(item, dict):
                continue
            group = review_lookup.get((item.get("themeId"), sentiment))
            if group is None or not str(item.get("summary") or "").strip():
                continue
            valid_ids = [evidence["id"] for evidence in group["evidence"]]
            selected = [
                value
                for value in item.get("evidenceIds", [])
                if value in valid_ids
            ]
            quote_id = item.get("quoteEvidenceId")
            if quote_id not in valid_ids:
                quote_id = selected[0] if selected else valid_ids[0]
            if quote_id not in selected:
                selected.insert(0, quote_id)
            normalized_review[field].append({
                **item,
                "theme": group["displayTheme"],
                "quoteEvidenceId": quote_id,
                "evidenceIds": selected,
            })

    normalized_qna = []
    source_qna = output.get("qnaAnalysis")
    qna_items = source_qna.get("themes") if isinstance(source_qna, dict) else []
    if isinstance(qna_items, list):
        for item in qna_items[:4]:
            if not isinstance(item, dict):
                continue
            group = qna_lookup.get(item.get("themeId"))
            if group is None or not str(item.get("summary") or "").strip():
                continue
            valid_ids = [evidence["id"] for evidence in group["evidence"]]
            selected = [
                value
                for value in item.get("evidenceIds", [])
                if value in valid_ids
            ]
            question_id = item.get("questionEvidenceId")
            if question_id not in valid_ids:
                question_id = selected[0] if selected else valid_ids[0]
            if question_id not in selected:
                selected.insert(0, question_id)
            normalized_qna.append({
                **item,
                "theme": group["displayTheme"],
                "ratio": group["ratio"],
                "questionEvidenceId": question_id,
                "evidenceIds": selected,
            })

    valid_review_refs = {
        f"{sentiment}/{theme_id}"
        for theme_id, sentiment in review_lookup
    }
    normalized_selling = []
    source_selling = output.get("sellingPoints")
    if isinstance(source_selling, list):
        for item in source_selling[:3]:
            if not isinstance(item, dict) or not str(item.get("text") or "").strip():
                continue
            evidence_ids = [
                value
                for value in item.get("evidenceIds", [])
                if value in all_evidence
            ]
            if not evidence_ids:
                continue
            item_type = item.get("type")
            confidence = item.get("confidence")
            normalized_selling.append({
                **item,
                "type": item_type
                if item_type in {"core", "hidden", "barrier", "seller_action"}
                else "hidden",
                "reviewThemes": [
                    value
                    for value in item.get("reviewThemes", [])
                    if value in valid_review_refs
                ],
                "qnaThemes": [
                    value
                    for value in item.get("qnaThemes", [])
                    if value in qna_lookup
                ],
                "evidenceIds": evidence_ids,
                "confidence": confidence
                if confidence in {"high", "medium", "low"}
                else "low",
            })

    return {
        "reviewAnalysis": normalized_review,
        "qnaAnalysis": {"themes": normalized_qna},
        "sellingPoints": normalized_selling,
    }
