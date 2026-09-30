from __future__ import annotations

from collections import defaultdict
from typing import Any


def build_report_input(
    request: dict[str, Any],
    predictions: list[dict[str, Any]],
    *,
    evidence_limit: int = 3,
    review_group_limit_per_sentiment: int = 3,
    qna_group_limit: int = 4,
) -> dict[str, Any]:
    """Build report-input-v1; all counts and ratios are backend-owned."""
    review_items = list((request.get("reviewSample") or {}).get("items") or [])
    qna_items = list((request.get("qnaSample") or {}).get("items") or [])
    item_lookup = {str(item["id"]): item for item in review_items + qna_items}
    prediction_lookup = {str(row["id"]): row for row in predictions}
    review_groups: dict[tuple[str, str, str], list[tuple[float, dict[str, Any]]]] = defaultdict(list)
    qna_groups: dict[tuple[str, str], list[tuple[float, dict[str, Any]]]] = defaultdict(list)
    usable_review = usable_qna = 0

    for item in review_items:
        prediction = prediction_lookup.get(str(item["id"]))
        if not prediction or not prediction.get("usable", True):
            continue
        usable_review += 1
        for label in prediction.get("labels", []):
            sentiment = str(label.get("sentiment") or "")
            theme_id = str(label.get("themeId") or "")
            # The public report contract has positive/negative sections only.
            if not theme_id or sentiment not in {"positive", "negative"}:
                continue
            theme_name = str(label.get("theme") or theme_id)
            review_groups[(theme_id, theme_name, sentiment)].append(
                (float(label.get("confidence", 0)), item)
            )

    for item in qna_items:
        prediction = prediction_lookup.get(str(item["id"]))
        if not prediction or not prediction.get("usable", True):
            continue
        primary = prediction.get("primaryTheme") or {}
        theme_id = str(primary.get("themeId") or "OTHER")
        usable_qna += 1
        theme_name = str(primary.get("theme") or f"{theme_id} 문의")
        qna_groups[(theme_id, theme_name)].append((float(primary.get("confidence", 0)), item))

    review_candidates = []
    for (theme_id, theme_name, sentiment), values in review_groups.items():
        review_candidates.append({
            "themeId": theme_id,
            "displayTheme": theme_name,
            "sentiment": sentiment,
            "mentionCount": len(values),
            "evidence": _select_evidence(values, evidence_limit),
            "_rankConfidence": sum(score for score, _ in values) / len(values),
        })
    report_review_groups = []
    for sentiment in ("positive", "negative"):
        ranked = sorted(
            (group for group in review_candidates if group["sentiment"] == sentiment),
            key=lambda group: (-group["mentionCount"], -group["_rankConfidence"], group["themeId"]),
        )[:review_group_limit_per_sentiment]
        for group in ranked:
            group.pop("_rankConfidence")
        report_review_groups.extend(ranked)
    qna_candidates = []
    for (theme_id, theme_name), values in qna_groups.items():
        qna_candidates.append({
            "themeId": theme_id,
            "displayTheme": theme_name,
            "primaryCount": len(values),
            "ratio": round(len(values) / max(usable_qna, 1), 6),
            "evidence": _select_evidence(values, evidence_limit),
            "_rankConfidence": sum(score for score, _ in values) / len(values),
        })
    report_qna_groups = sorted(
        qna_candidates,
        key=lambda group: (-group["primaryCount"], -group["_rankConfidence"], group["themeId"]),
    )[:qna_group_limit]
    for group in report_qna_groups:
        group.pop("_rankConfidence")
    ratio_sum = sum(group["ratio"] for group in report_qna_groups)
    if usable_qna and len(report_qna_groups) == len(qna_groups) and abs(ratio_sum - 1.0) > 1e-5:
        largest = max(report_qna_groups, key=lambda group: group["primaryCount"])
        largest["ratio"] = round(largest["ratio"] + 1.0 - ratio_sum, 6)

    product = request["product"]
    return {
        "schemaVersion": "report-input-v1",
        "taxonomyVersion": request["taxonomyVersion"],
        "analysisId": request["analysisId"],
        "productId": product["productId"],
        "category": {"path": product["categoryPath"]},
        "sample": {
            "reviewSampleSize": len(review_items),
            "reviewUsableSize": usable_review,
            "reviewMaxItems": int((request.get("reviewSample") or {}).get("maxItems", 5000)),
            "qnaSampleSize": len(qna_items),
            "qnaUsableSize": usable_qna,
            "qnaMaxItems": int((request.get("qnaSample") or {}).get("maxItems", 5000)),
        },
        "reviewGroups": report_review_groups,
        "qnaGroups": report_qna_groups,
        "_evidenceLookup": {
            item_id: str(item["text"]) for item_id, item in item_lookup.items()
        },
    }


def materialize_grounded_report(
    report_input: dict[str, Any], decoder_output: dict[str, Any]
) -> dict[str, Any]:
    """Replace decoder-selected evidence IDs with backend-owned exact text."""
    evidence = report_input.get("_evidenceLookup") or {}
    review_groups = {
        (str(group["themeId"]), str(group["sentiment"])): group
        for group in report_input.get("reviewGroups", [])
    }
    qna_groups = {
        str(group["themeId"]): group for group in report_input.get("qnaGroups", [])
    }
    result = dict(decoder_output)
    review = dict(result.get("reviewAnalysis") or {})
    for field in ("positiveThemes", "negativeThemes"):
        values = []
        for item in review.get(field, []):
            expected_sentiment = "positive" if field == "positiveThemes" else "negative"
            group = review_groups.get((str(item.get("themeId") or ""), expected_sentiment))
            evidence_id = str(item.get("quoteEvidenceId") or "")
            allowed_evidence = {str(value["id"]) for value in (group or {}).get("evidence", [])}
            if not group or evidence_id not in allowed_evidence or evidence_id not in evidence:
                continue
            values.append({
                **item,
                "theme": group["displayTheme"],
                "mentionCount": group["mentionCount"],
                "sentiment": expected_sentiment,
                "evidenceIds": [evidence_id],
                "quote": evidence[evidence_id],
            })
        review[field] = values
    result["reviewAnalysis"] = review
    qna = dict(result.get("qnaAnalysis") or {})
    qna_values = []
    for item in qna.get("themes", []):
        group = qna_groups.get(str(item.get("themeId") or ""))
        evidence_id = str(item.get("questionEvidenceId") or "")
        allowed_evidence = {str(value["id"]) for value in (group or {}).get("evidence", [])}
        if not group or evidence_id not in allowed_evidence or evidence_id not in evidence:
            continue
        qna_values.append({
            **item,
            "theme": group["displayTheme"],
            "primaryCount": group["primaryCount"],
            "ratio": group["ratio"],
            "evidenceIds": [evidence_id],
            "question": evidence[evidence_id],
        })
    qna["themes"] = qna_values
    result["qnaAnalysis"] = qna
    result["sellingPoints"] = _materialize_selling_points(
        result.get("sellingPoints", result.pop("selling_points", [])), report_input
    )
    result["analysisId"] = report_input["analysisId"]
    source_sample = report_input["sample"]
    result["sample"] = {
        "review": {
            "collectionOrder": "newest",
            "sampleSize": source_sample["reviewSampleSize"],
            "usableSize": source_sample["reviewUsableSize"],
            "maxItems": source_sample["reviewMaxItems"],
        },
        "qna": {
            "collectionOrder": "newest",
            "sampleSize": source_sample["qnaSampleSize"],
            "usableSize": source_sample["qnaUsableSize"],
            "maxItems": source_sample["qnaMaxItems"],
        },
    }
    return result


def _materialize_selling_points(
    points: list[dict[str, Any]], report_input: dict[str, Any], *, allow_fallback: bool = True,
) -> list[dict[str, Any]]:
    review_by_theme: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for group in report_input.get("reviewGroups", []):
        review_by_theme[str(group["themeId"])].append(group)
    qna_by_theme = {group["themeId"]: group for group in report_input.get("qnaGroups", [])}
    result = []
    for point in points[:3]:
        themes = [str(value) for value in point.get("evidence_theme_ids", [])]
        point_type = _selling_point_type(point)
        sentiment = "positive" if point_type in {"core", "hidden"} else "negative"
        review_themes = []
        qna_themes = []
        evidence_ids = []
        for theme in themes:
            for group in review_by_theme.get(theme, []):
                if group["sentiment"] != sentiment or not group.get("evidence"):
                    continue
                review_themes.append(f"{sentiment}/{theme}")
                evidence_ids.append(str(group["evidence"][0]["id"]))
            qna_group = qna_by_theme.get(theme)
            if qna_group and qna_group.get("evidence"):
                qna_themes.append(theme)
                evidence_ids.append(str(qna_group["evidence"][0]["id"]))
        text = str(point.get("text") or point.get("message") or "").strip()
        if text and evidence_ids:
            result.append({
                "type": point_type,
                "text": text,
                "reviewThemes": list(dict.fromkeys(review_themes)),
                "qnaThemes": list(dict.fromkeys(qna_themes)),
                "evidenceIds": list(dict.fromkeys(evidence_ids)),
                "confidence": str(point.get("confidence") or "candidate"),
            })
    if not result and allow_fallback:
        return _materialize_selling_points(
            _fallback_selling_points(report_input), report_input, allow_fallback=False
        )
    return result


def _fallback_selling_points(report: dict[str, Any]) -> list[dict[str, Any]]:
    qna = {str(group["themeId"]): group for group in report.get("qnaGroups", [])}
    candidates = []
    for group in report.get("reviewGroups", []):
        theme_id = str(group["themeId"])
        sentiment = str(group.get("sentiment") or "")
        if sentiment not in {"positive", "negative"}:
            continue
        has_qna = theme_id in qna
        point_type = (
            "core" if sentiment == "positive" and has_qna else
            "hidden" if sentiment == "positive" else
            "barrier" if has_qna else "pain"
        )
        theme = str(group.get("displayTheme") or theme_id)
        messages = {
            "core": f"{theme} 만족과 구매 전 관심이 함께 확인되어 핵심 소구점 후보로 활용할 수 있습니다.",
            "hidden": f"{theme} 만족이 확인되어 숨은 강점 후보로 활용할 수 있습니다.",
            "barrier": f"{theme} 문의와 불만이 함께 확인되어 구매 장벽 개선이 필요합니다.",
            "pain": f"{theme} 불만이 확인되어 잠재 문제점 개선이 필요합니다.",
        }
        importance = int(group.get("mentionCount", 0)) + int((qna.get(theme_id) or {}).get("primaryCount", 0))
        candidates.append((importance, {
            "type": point_type, "message": messages[point_type],
            "evidence_theme_ids": [theme_id], "fallbackUsed": True,
        }))
    return [point for _, point in sorted(candidates, key=lambda value: -value[0])[:3]]


def _selling_point_type(point: dict[str, Any]) -> str:
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


def _select_evidence(
    values: list[tuple[float, dict[str, Any]]], limit: int
) -> list[dict[str, str]]:
    selected = []
    seen = set()
    for _, item in sorted(values, key=lambda value: (-value[0], str(value[1]["id"]))):
        normalized = "".join(str(item["text"]).split()).lower()
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        selected.append({"id": str(item["id"]), "text": str(item["text"])})
        if len(selected) >= limit:
            break
    return selected
