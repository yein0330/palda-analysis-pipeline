#!/usr/bin/env python3
"""User-approved, redacted Validation input/output tables and staged E2E viewer."""
import argparse
import hashlib
import html
import json
from pathlib import Path
import re
import time

from quality_dashboard import MODELS, ROOT, samples, read, lines
from wandb_quality_sync import settings, verify_destination

TRACE = ROOT / "artifacts/pipeline-examples-20260911"
OUT = ROOT / ".runtime/wandb-qualitative"
RUN_ID = "palda-io-20260911"
ENTITY = "yein0330-soganag-university"
PROJECT = "PALDA"
PATTERNS = [
    (re.compile(r"(?:hf_|github_pat_|wandb_v1_)[A-Za-z0-9_-]{12,}"), "[키 가림]"),
    (re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}"), "[이메일 가림]"),
    (re.compile(r"(?<!\d)(?:\+82[ -]?)?0?1[016789][ .-]?\d{3,4}[ .-]?\d{4}(?!\d)"), "[전화번호 가림]"),
    (re.compile(r"(?<!\d)\d{6}[ -][1-4]\d{6}(?!\d)"), "[식별번호 가림]"),
    (re.compile(r"https?://[^\s<>]+"), "[URL 가림]"),
    (re.compile(r"[가-힣]+(?:로|길)\s*\d+(?:-\d+)?(?:\s+\d+(?:동|호))*"), "[주소 가림]"),
]


def redact(value):
    if isinstance(value, str):
        for pattern, replacement in PATTERNS:
            value = pattern.sub(replacement, value)
        return value
    if isinstance(value, list):
        return [redact(v) for v in value]
    if isinstance(value, dict):
        return {k: redact(v) for k, v in value.items()}
    return value


def compact(value):
    return json.dumps(value, ensure_ascii=False)


def alias(value):
    return "v_" + hashlib.sha256(value.encode()).hexdigest()[:12]


def encoder_snapshot(path):
    """Pair a saved prediction snapshot with only the fixed Validation inputs."""
    path = path.resolve()
    if not path.is_relative_to(ROOT / "artifacts") or not re.fullmatch(
            r"(?:epoch-\d+_predictions|validation_predictions|live_predictions)\.jsonl", path.name):
        raise ValueError("Not an allowlisted encoder prediction snapshot")
    gold = {r["id"]: r for r in samples("baseline-42")["rows"]}
    predictions = lines(path)
    if len(predictions) != len(gold) or {p["sample_id"] for p in predictions} != set(gold):
        raise ValueError("Snapshot must match the fixed 707 Validation IDs exactly")
    rows = []
    for prediction in predictions:
        r = gold[prediction["sample_id"]]
        clean_prediction = {k: v for k,v in prediction.items() if k in (
            "predicted_labels", "primary_theme_id", "secondary_themes", "usable_probability")}
        rows.append([alias(r["id"]), r["type"], r["category"], r["text"],
                     "\n".join(r["goldDisplay"] or []), compact(clean_prediction), path.name])
    return {"columns": ["sample_id", "종류", "카테고리", "입력 원문", "사람 정답", "인코더 출력·신뢰도", "예측 시점"],
            "data": redact(rows)}


def esc(value):
    return html.escape(str(value))


def table_html(headers, rows):
    return '<div class="scroll"><table><tr>' + ''.join(f'<th>{esc(h)}</th>' for h in headers) + '</tr>' + ''.join(
        '<tr>' + ''.join(f'<td>{esc(v)}</td>' for v in row) + '</tr>' for row in rows) + '</table></div>'


def page(traces):
    parts = ['''<!doctype html><html lang="ko"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>PALDA 입력부터 소구점까지</title><style>body{font:16px/1.65 system-ui,sans-serif;color:#25314a;background:#f3f5fa;padding:20px;margin:auto;max-width:1150px}article,section{background:white;padding:22px;margin:16px 0;border-radius:12px}h2{color:#315dbc}h3{margin-bottom:6px}blockquote{background:#f0f4fa;margin:10px 0;padding:12px;border-left:4px solid #91aad8;white-space:pre-wrap}table{border-collapse:collapse;width:100%;font-size:14px}th,td{border:1px solid #dce3ee;padding:9px;text-align:left;vertical-align:top;white-space:pre-wrap;min-width:80px}pre{white-space:pre-wrap;overflow-wrap:anywhere;background:#f5f7fb;padding:14px}.scroll{overflow:auto}summary{cursor:pointer;font-weight:bold}small{color:#66758a}.warning{background:#fff0d0;padding:15px}.tag{color:#315dbc;font-weight:bold}</style>
<h1>PALDA · 입력 → 분류 → 요약 → 소구점</h1>
<p class="warning">실제 실행 결과입니다. Validation에서 고른 3개 상품을 2개 모델로 비교합니다. 독립 Test·봉인 시뮬레이션·운영 품질 인증이 아닙니다. 사람 정답은 모델에 입력하지 않았습니다. 입력은 검증용 부분 표본이며 최신순 수집을 재현한 데이터가 아닙니다.</p>
<p>기존 연동 모델과 연구 후보를 구분해서 보세요. 인코더가 분류한 테마 중 상위 그룹만 디코더로 전달됩니다. 아래 소구점은 검증된 제품 효능이 아니라 생성 후보입니다. 개인정보 패턴은 표시·업로드 단계에서 가렸으며 자동 탐지가 완전한 익명화를 보장하지는 않습니다.</p>''']
    if not traces:
        parts.append('<section>파이프라인 실행을 기다리는 중입니다. 입력과 분류 표는 W&B의 별도 Tables에서 먼저 볼 수 있습니다.</section>')
    for trace in traces:
        request = trace["input"]
        predictions = {p["id"]: p for p in trace.get("encoderOutput", [])}
        parts.append(f'<article><h2>{esc(trace["caseId"])} · {"기존 연동 모델" if trace["variant"] == "integration" else "연구 후보 모델"}</h2><p>{esc(request["product"]["categoryPath"])}</p><p class="tag">단계: {esc(trace["stage"])} · 실행 시간: {esc(trace.get("elapsedSeconds", "진행 중"))}초</p><small>Encoder: {esc(trace["encoder"])} / Decoder: {esc(trace["decoder"])}</small>')
        input_rows = []
        for kind, sample_key in (("리뷰", "reviewSample"), ("Q&A", "qnaSample")):
            for item in request[sample_key]["items"]:
                p = predictions.get(item["id"], {})
                labels = p.get("labels") if kind == "리뷰" else {"주 문의": p.get("primaryTheme"), "보조 문의": p.get("secondaryThemes")}
                input_rows.append([item["id"], kind, item["text"], compact(trace["gold"].get(item["id"], [])),
                                   compact(labels), p.get("usable", "대기")])
        parts.append('<h3>1. 입력 · 사람 정답 · 실제 인코더 출력</h3>' + table_html(
            ["근거 ID", "종류", "입력 원문", "사람 정답(참고용)", "인코더 분류·신뢰도", "분석 사용 여부"], input_rows))
        parts.append('<details><summary>2. 디코더에 전달된 테마별 근거 보기</summary><pre>' + esc(json.dumps(trace.get("decoderInput", {}), ensure_ascii=False, indent=2)) + '</pre></details>')
        result = trace.get("finalOutput", {})
        parts.append('<h3>3. 최종 리뷰 요약·인용문</h3>')
        for key, label in (("positiveThemes", "긍정"), ("negativeThemes", "부정")):
            for group in result.get("reviewAnalysis", {}).get(key, []):
                parts.append(f'<section><b>{label} · {esc(group["theme"])} · {group["mentionCount"]}건</b><p>{esc(group["summary"])}</p><blockquote>{esc(group["quote"])}</blockquote><small>근거: {esc(group["evidenceIds"])} · 요약 대체문구: {esc(group.get("fallbackUsed"))}</small></section>')
        parts.append('<h3>4. 최종 Q&A 요약·대표 질문</h3>')
        for group in result.get("qnaAnalysis", {}).get("themes", []):
            parts.append(f'<section><b>{esc(group["theme"])} · {group["primaryCount"]}건 · {group["ratio"]*100:.1f}%</b><p>{esc(group["summary"])}</p><blockquote>{esc(group["question"])}</blockquote><small>근거: {esc(group["evidenceIds"])} · 요약 대체문구: {esc(group.get("fallbackUsed"))}</small></section>')
        parts.append('<h3>5. 최종 소구점 후보</h3>')
        for item in result.get("sellingPoints", []):
            parts.append(f'<section><b>{esc(item["type"])}</b><p>{esc(item["text"])}</p><small>리뷰 테마: {esc(item.get("reviewThemes"))} · Q&A 테마: {esc(item.get("qnaThemes"))} · 근거: {esc(item.get("evidenceIds"))}</small></section>')
        if not result:
            parts.append('<p>최종 결과 생성 중입니다. 아직 없는 출력은 표시하지 않습니다.</p>')
        parts.append('<details><summary>6. 디코더 원출력 / 후처리 전 출력 / 최종 JSON 비교</summary><h4>LLM 응답 (파싱된 JSON)</h4><pre>' + esc(json.dumps(trace.get("decoderCompletions", []), ensure_ascii=False, indent=2)) + '</pre><h4>디코더 반환값</h4><pre>' + esc(json.dumps(trace.get("decoderOutput", {}), ensure_ascii=False, indent=2)) + '</pre><h4>최종 응답</h4><pre>' + esc(json.dumps(result, ensure_ascii=False, indent=2)) + '</pre></details></article>')
    return '\n'.join(parts) + '</html>'


def build():
    base = samples("baseline-42")["rows"]
    inputs = [[alias(r["id"]), r["type"], r["category"], r["text"], "\n".join(r["goldDisplay"] or [])] for r in base]
    comparisons = []
    outcome_names = {"fixed": "기존 오답→정답", "regressed": "기존 정답→오답", "correct": "계속 정답", "wrong": "계속 오답", "pending": "대기"}
    for model in MODELS:
        for r in samples(model)["rows"]:
            comparisons.append([model, alias(r["id"]), r["type"], r["category"], r["text"],
                                "\n".join(r["goldDisplay"] or []), "\n".join(r["beforeDisplay"] or []),
                                "\n".join(r["afterDisplay"] or []), outcome_names[r["outcome"]]])
    traces = [read(p) for p in sorted(TRACE.glob("*-product-*.json"))]
    traces = [t for t in traces if t and t.get("split") == "validation"]
    groups, selling, stage_rows = [], [], []
    for t in traces:
        final = t.get("finalOutput", {})
        for kind in ("positiveThemes", "negativeThemes"):
            for g in final.get("reviewAnalysis", {}).get(kind, []):
                groups.append([t["caseId"], t["variant"], "review", g["theme"], g["sentiment"], g["mentionCount"],
                               g["summary"], g["quote"], compact(g["evidenceIds"]), g.get("fallbackUsed")])
        for g in final.get("qnaAnalysis", {}).get("themes", []):
            groups.append([t["caseId"], t["variant"], "qna", g["theme"], "primary", g["primaryCount"],
                           g["summary"], g["question"], compact(g["evidenceIds"]), g.get("fallbackUsed")])
        for item in final.get("sellingPoints", []):
            selling.append([t["caseId"], t["variant"], item["type"], item["text"], compact(item.get("reviewThemes", [])),
                            compact(item.get("qnaThemes", [])), compact(item.get("evidenceIds", [])), "후보 / 의미 품질 미승인"])
        stage_rows.append([t["caseId"], t["variant"], t["encoder"], t["stage"],
                           compact(t["input"]), compact(t.get("encoderOutput", [])),
                           compact(t.get("decoderInput", {})), compact(t.get("decoderCompletions", [])),
                           compact(t.get("decoderOutput", {})), compact(final)])
    data = {
        "01_inputs": {"columns": ["sample_id", "종류", "카테고리", "입력 원문", "사람 정답"], "data": inputs},
        "02_encoder_comparison": {"columns": ["모델", "sample_id", "종류", "카테고리", "입력 원문", "사람 정답", "동일 seed 기준 예측", "선택 모델 예측", "완전일치 비교"], "data": comparisons},
        "03_pipeline_stages": {"columns": ["상품", "모델 구분", "인코더", "진행 단계", "입력 JSON", "인코더 출력", "디코더 입력 근거", "LLM 원응답 JSON", "디코더 반환값", "최종 응답 JSON"], "data": stage_rows},
        "04_summaries_quotes": {"columns": ["상품", "모델 구분", "종류", "테마", "감성/역할", "건수", "요약", "인용문/질문", "근거 ID", "대체 요약 여부"], "data": groups},
        "05_selling_points": {"columns": ["상품", "모델 구분", "소구점 유형", "소구점 문장", "리뷰 근거 테마", "Q&A 근거 테마", "근거 ID", "품질 상태"], "data": selling},
    }
    return redact(data), redact(traces)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--online", action="store_true")
    parser.add_argument("--watch", action="store_true", help="Watch until the current capture suite terminates")
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    run = None
    previous = {}
    try:
        while True:
            data, traces = build()
            rendered = page(traces)
            (OUT / "review.html").write_text(rendered)
            (OUT / "tables.json").write_text(json.dumps(data, ensure_ascii=False))
            if args.online:
                import wandb
                api = wandb.Api(timeout=20)
                verify_destination(api, ENTITY, PROJECT, "yein0330")
                if run is None:
                    run = wandb.init(entity=ENTITY, project=PROJECT, id=RUN_ID,
                        name="입력-인코더-디코더-소구점 확인", group="qualitative-preview", resume="allow",
                        dir=str(OUT), settings=settings(wandb),
                        config={"source": "validation", "independent_test": False, "production_approved": False,
                                "contains_redacted_source_text": True, "contains_weights": False})
                event = {}
                for key, table in data.items():
                    fingerprint = hashlib.sha256(compact(table).encode()).hexdigest()
                    if fingerprint != previous.get(key):
                        event[key] = wandb.Table(columns=table["columns"], data=table["data"])
                        previous[key] = fingerprint
                fingerprint = hashlib.sha256(rendered.encode()).hexdigest()
                if fingerprint != previous.get("html"):
                    event["00_readable_pipeline"] = wandb.Html(rendered, inject=False)
                    previous["html"] = fingerprint
                if event:
                    run.log(event)
                run.summary.update({"validation_inputs": len(data["01_inputs"]["data"]),
                                    "encoder_comparisons": len(data["02_encoder_comparison"]["data"]),
                                    "pipeline_cases": len(traces),
                                    "completed_pipeline_cases": sum(t["stage"] == "completed" for t in traces)})
            status = read(TRACE / "status.json", {})
            print({"tables": {k: len(v["data"]) for k,v in data.items()}, "stage": status.get("stage")}, flush=True)
            if not args.watch or status.get("stage") in {"completed", "failed"}:
                break
            time.sleep(10)
    finally:
        if run is not None:
            run.finish()


if __name__ == "__main__":
    main()
