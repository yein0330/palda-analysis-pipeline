#!/usr/bin/env python3
"""Export shareable end-to-end quality examples from an oracle-expansion capture.

For every product the file holds the masked input records with their human Gold
labels, what the current encoder predicted for each record, and the final
service output (review/Q&A summaries and selling points) for the current
production-path arm and, for contrast, the Gold-oracle arm. Detected issues are
listed per product so readers can find the failure without reading everything.

Text is masked with the same patterns as `wandb_qualitative.py` before it is
written. Pattern masking does not guarantee anonymisation: names and free-form
addresses can remain. Product and record IDs are already aliases/hashes.
Validation split only; Test and the sealed simulation are not read.
"""
from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
MASKS = [
    (re.compile(r"(?:hf_|github_pat_|wandb_v1_)[A-Za-z0-9_-]{12,}"), "[키 가림]"),
    (re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}"), "[이메일 가림]"),
    (re.compile(r"(?<!\d)(?:\+82[ -]?)?0?1[016789][ .-]?\d{3,4}[ .-]?\d{4}(?!\d)"), "[전화번호 가림]"),
    (re.compile(r"(?<!\d)\d{6}[ -][1-4]\d{6}(?!\d)"), "[식별번호 가림]"),
    (re.compile(r"https?://[^\s<>]+"), "[URL 가림]"),
    (re.compile(r"[가-힣]+(?:로|길)\s*\d+(?:-\d+)?(?:\s+\d+(?:동|호))*"), "[주소 가림]"),
]
NEGATIVE = re.compile(r"불만|불편|부정적|문제가|문제점|부족|떨어지|떨어진|싫어|아쉽|아쉬|실망|개선이 필요|개선 필요")
GENERIC = re.compile(r"(긍정적인?|부정적인?) (피드백|평가|반응)|만족(도가 높|스럽|하는 것으로)|피드백이 (있|많)")


def mask(value: Any) -> Any:
    if isinstance(value, str):
        for pattern, replacement in MASKS:
            value = pattern.sub(replacement, value)
        return value
    if isinstance(value, list):
        return [mask(item) for item in value]
    if isinstance(value, dict):
        # IDs are hashes; masking them would break the links between records,
        # groups and selling points (a 12-hex hash can look like a phone number).
        return {key: item if _is_id_key(key) else mask(item) for key, item in value.items()}
    return value


def _is_id_key(key: str) -> bool:
    return key == "id" or key.endswith(("Id", "Ids", "_id", "_ids"))


def _gold_view(labels: list[dict]) -> list[dict]:
    return [{k: label.get(k) for k in ("theme_id", "sentiment", "role") if label.get(k) is not None}
            for label in labels]


def _prediction_view(prediction: dict) -> dict:
    if "labels" in prediction:
        return {"labels": [{"themeId": x["themeId"], "sentiment": x["sentiment"],
                            "confidence": round(float(x["confidence"]), 3)} for x in prediction["labels"]]}
    return {
        "primaryTheme": prediction["primaryTheme"]["themeId"],
        "primaryConfidence": round(float(prediction["primaryTheme"]["confidence"]), 3),
        "secondaryThemes": [x["themeId"] for x in prediction.get("secondaryThemes", [])],
        "usable": prediction.get("usable"),
    }


def _correct(prediction: dict, gold: list[dict]) -> bool:
    if "labels" in prediction:
        predicted = {(x["themeId"], x["sentiment"]) for x in prediction["labels"]}
        expected = {(x["theme_id"], x["sentiment"]) for x in gold if x.get("sentiment") in ("positive", "negative")}
        return predicted == expected
    primary = next((x["theme_id"] for x in gold if x.get("role") == "primary"), None)
    return prediction["primaryTheme"]["themeId"] == primary


def _issues(trace: dict) -> list[dict]:
    output = trace["finalOutput"]
    gold = trace["gold"]
    issues = []
    for point in output.get("sellingPoints", []):
        if point.get("type") in ("core", "hidden") and NEGATIVE.search(point.get("text", "")):
            issues.append({"kind": "selling_point_type_content_mismatch",
                           "detail": f"[{point['type']}] {point['text']}",
                           "why": "불만 내용이 강점 유형으로 분류되어 긍정 리뷰가 근거로 붙음"})
        if GENERIC.search(point.get("text", "")):
            issues.append({"kind": "generic_selling_point", "detail": point["text"],
                           "why": "무엇이 좋은지만 말하고 구매자의 어떤 고민을 해결하는지 말하지 않음"})
    raw = (trace.get("decoderOutput") or {}).get("sellingPoints") or []
    sets = [set(p.get("evidence_theme_ids") or []) for p in raw]
    for a in range(len(sets)):
        for b in range(a + 1, len(sets)):
            if sets[a] & sets[b]:
                issues.append({"kind": "overlapping_selling_points",
                               "detail": f"소구점 {a + 1}과 {b + 1}의 근거 테마 {sorted(sets[a] & sets[b])} 중복",
                               "why": "서로 다른 구매 결정 요인을 다루지 못함"})
    quotes: dict[str, list[str]] = {}
    for group in output["reviewAnalysis"]["positiveThemes"] + output["reviewAnalysis"]["negativeThemes"]:
        quotes.setdefault(group.get("quote", ""), []).append(f"{group['sentiment']}/{group['themeId']}")
    for quote, groups in quotes.items():
        if quote and len(groups) > 1:
            issues.append({"kind": "same_quote_for_multiple_themes", "detail": f"{groups}: {quote}",
                           "why": "하나의 리뷰가 여러 테마의 대표 인용문으로 반복됨"})
    for group in (trace.get("decoderInput") or {}).get("reviewGroups", []):
        wrong = [item["id"] for item in group.get("evidence", [])
                 if (group["themeId"], group["sentiment"]) not in
                 {(x["theme_id"], x.get("sentiment")) for x in gold.get(item["id"], [])}]
        if wrong:
            issues.append({"kind": "misclassified_evidence",
                           "detail": f"{group['sentiment']}/{group['themeId']} 근거 {len(wrong)}/{len(group['evidence'])}건이 사람 정답과 다름",
                           "evidenceIds": wrong,
                           "why": "인코더 오분류로 다른 테마의 리뷰가 이 테마 요약·소구점의 근거가 됨"})
    return issues


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture-dir", default=str(ROOT / "artifacts/oracle-expansion-20260928"))
    parser.add_argument("--output", default=str(ROOT / "artifacts/quality-examples-20260929/quality_examples.json"))
    args = parser.parse_args()
    capture = Path(args.capture_dir)
    comparison = json.loads((capture / "selling_point_comparison.json").read_text())

    products = []
    for path in sorted(capture.glob("integration-product-*.json")):
        current = json.loads(path.read_text())
        oracle = json.loads((capture / path.name.replace("integration", "gold_oracle")).read_text())
        predictions = {p["id"]: p for p in current["encoderOutput"]}
        records = []
        for kind, key in (("review", "reviewSample"), ("qna", "qnaSample")):
            for item in current["input"][key]["items"]:
                gold = current["gold"][item["id"]]
                records.append({
                    "id": item["id"], "type": kind, "text": item["text"],
                    "humanGold": _gold_view(gold),
                    "currentEncoder": _prediction_view(predictions[item["id"]]),
                    "encoderMatchesGold": _correct(predictions[item["id"]], gold),
                })
        products.append(mask({
            "product": current["caseId"],
            "category": current["input"]["product"]["categoryPath"],
            "records": records,
            "encoderAccuracyOnThisProduct": round(
                sum(r["encoderMatchesGold"] for r in records) / max(len(records), 1), 3),
            "currentModelOutput": {k: current["finalOutput"][k]
                                   for k in ("reviewAnalysis", "qnaAnalysis", "sellingPoints")},
            "currentModelIssues": _issues(current),
            "goldOracleOutput": {
                "note": "분류기만 사람 정답으로 바꾸고 디코더는 동일. 인코더가 완벽할 때의 출력.",
                "reviewAnalysis": oracle["finalOutput"]["reviewAnalysis"],
                "qnaAnalysis": oracle["finalOutput"]["qnaAnalysis"],
                "sellingPoints": oracle["finalOutput"]["sellingPoints"],
            },
            "goldOracleIssues": [i for i in _issues(oracle) if i["kind"] != "misclassified_evidence"],
        }))

    issue_counts: dict[str, dict[str, int]] = {"currentModel": {}, "goldOracle": {}}
    for product in products:
        for arm, key in (("currentModel", "currentModelIssues"), ("goldOracle", "goldOracleIssues")):
            for issue in product[key]:
                issue_counts[arm][issue["kind"]] = issue_counts[arm].get(issue["kind"], 0) + 1

    document = {
        "meta": {
            "title": "PALDA 인코더→디코더 실제 출력 품질 예시",
            "generatedAt": time.strftime("%Y-%m-%d"),
            "split": "human Gold validation (Test·봉인 시뮬레이션 미사용)",
            "products": len(products),
            "selection": "리뷰·Q&A가 모두 있는 validation 상품을 표본 수 내림차순으로, Cat1당 최대 2개. 상품당 리뷰·Q&A 각 최대 10건. 실제 상품의 전체 리뷰 분포를 대표하지 않음.",
            "currentModel": {"encoder": "ko-sroberta-champion-candidate-test-audit (seed 20260909)",
                             "decoder": "Qwen2.5-7B-Instruct + LoRA checkpoint 221, 4-bit, greedy"},
            "goldOracle": "같은 입력·같은 디코더, 분류만 사람 정답 라벨로 대체",
            "masking": "이메일·전화번호·식별번호·URL·키·도로명 주소 패턴을 가림. 패턴 기반이라 이름이나 자유형 주소가 남을 수 있음.",
            "issueDetection": "이슈는 자동 규칙(어휘 매칭·근거 비교)으로 표시한 것이며 사람 평가가 아님.",
            "encoderMatchesGold": "리뷰는 긍정·부정 (테마, 감성) 집합 일치(중립은 런타임이 쓰지 않아 제외), Q&A는 주 테마 일치.",
            "productionApproved": False,
        },
        "summary": {
            "metricsByArm": comparison["summary"],
            "issueCounts": issue_counts,
            "headline": [
                "현재 모델이 디코더에 넘기는 리뷰 근거의 약 47%, Q&A 근거의 약 51%만 사람 정답과 테마가 맞음.",
                "분류를 사람 정답으로 바꾸면 소구점 중복(상품당 0.95→0.35쌍)과 유형-내용 불일치(7→1건)는 줄지만, '긍정적인 피드백이 있습니다' 류 상투 문장은 그대로임.",
            ],
        },
        "products": products,
    }
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(document, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"output": str(out), "bytes": out.stat().st_size, "products": len(products),
                      "issueCounts": issue_counts}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
