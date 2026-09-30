"""Training-text-only E2E inspection, not a held-out quality benchmark."""
import asyncio
from collections import defaultdict
import hashlib
import html
import json
import os
from pathlib import Path
import statistics
import time

from .runtime import analyzer_from_env
from .schemas import AnalysisRequest, ReportOutput

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "artifacts/existing-model-1000-20260910-dense"
PALDA = ROOT.parent


def save(name, value):
    path = OUT / name
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    tmp.replace(path)


def render(records):
    esc = lambda x: html.escape(str(x))
    sections = []
    for row in records:
        request = row["request"]
        result = row.get("result", {})
        body = [f'<h2>{esc(request["product"]["categoryPath"])}</h2>',
                f'<p>상품 {esc(request["product"]["productId"])} · {row["seconds"]:.2f}초</p>']
        if row.get("error"):
            body.append(f'<p>실패: {esc(row["error"])}</p>')
        for label, key in (("긍정 테마", "positiveThemes"), ("부정 테마", "negativeThemes")):
            body.append(f'<h3>{label}</h3>')
            for item in result.get("reviewAnalysis", {}).get(key, []):
                body.append(f'<h4>{esc(item["theme"])} · {item["mentionCount"]}건</h4><p>{esc(item["summary"])}</p><blockquote>{esc(item["quote"])}</blockquote>')
        body.append('<h3>Q&A</h3>')
        for item in result.get("qnaAnalysis", {}).get("themes", []):
            body.append(f'<h4>{esc(item["theme"])} · {item["ratio"]*100:.1f}%</h4><p>{esc(item["summary"])}</p><blockquote>{esc(item["question"])}</blockquote>')
        body.append('<h3>소구점</h3>')
        for item in result.get("sellingPoints", []):
            body.append(f'<p>[{esc(item["type"])}] {esc(item["text"])}</p>')
        body.append('<details><summary>입력 원문과 전체 결과 JSON 보기</summary><pre>' + esc(json.dumps(row, ensure_ascii=False, indent=2)) + '</pre></details>')
        sections.append('<section>' + ''.join(body) + '</section>')
    page = '<!doctype html><html lang="ko"><meta charset="utf-8"><title>PALDA 1,000건 점검</title><style>body{max-width:960px;margin:30px auto;font:16px/1.7 sans-serif;padding:16px;background:#f5f6f8}section{background:white;padding:24px;margin:24px 0;border-radius:12px}blockquote{border-left:4px solid #6d83b0;padding-left:16px}pre{white-space:pre-wrap;overflow-wrap:anywhere}</style><h1>기존 모델 1,000건 출력 점검</h1><p>학습에 사용된 실제 문장입니다. 독립적인 정확도 평가가 아니며 상품별로 분리하여 분석합니다. 실행 중에는 새로고침하면 완료된 상품이 추가됩니다.</p>' + ''.join(sections) + '</html>'
    tmp = OUT / "results.html.tmp"
    tmp.write_text(page)
    tmp.replace(OUT / "results.html")


async def main():
    if OUT.exists():
        raise ValueError(f"Refusing to overwrite existing run: {OUT}")
    OUT.mkdir(parents=True)
    source = PALDA / "palda-encoder-pipeline/data/releases/label-quality-20260910/relabel384/train.jsonl"
    groups = defaultdict(list)
    for line in source.open():
        r = json.loads(line)
        groups[(r["category_key"], r["product_id"])].append(r)
    categories = defaultdict(list)
    for key in sorted(groups, key=lambda k: hashlib.sha256(str(k).encode()).hexdigest()):
        if len(groups[key]) < 10:
            continue
        categories[key[0]].append(key)
    requests, seen, count = [], set(), 0
    while count < 1000 and any(categories.values()):
        for category in sorted(categories):
            if not categories[category] or count == 1000:
                continue
            key = categories[category].pop(0)
            selected = []
            for row in groups[key]:
                normalized = ''.join(row["text"].split()).casefold()
                if normalized in seen or count == 1000:
                    continue
                selected.append(row)
                seen.add(normalized)
                count += 1
            if not selected:
                continue
            req = {"product": {"productId": key[1], "categoryPath": key[0]}}
            for kind, field in (("review", "reviewSample"), ("qna", "qnaSample")):
                req[field] = {"collectionOrder": "newest", "maxItems": 5000,
                    "items": [{"id": r["sample_id"], "text": r["text"]} for r in selected if r["record_type"] == kind]}
            AnalysisRequest.model_validate(req)
            requests.append(req)
    assert count == 1000
    manifest = {"items": count, "products": len(requests),
        "categories": len({r["product"]["categoryPath"] for r in requests}),
        "reviewItems": sum(len(r["reviewSample"]["items"]) for r in requests),
        "qnaItems": sum(len(r["qnaSample"]["items"]) for r in requests),
        "source": str(source), "sourceSha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "heldOutEvaluation": False, "testRead": False, "sealedSimulationRead": False,
        "selection": "products with >=10 training rows; category round-robin; global whitespace/casefold exact dedup; no text repetition",
        "orderCaveat": "training subset, not verified chronological newest sampling"}
    save("manifest.json", manifest)
    save("requests.json", requests)
    # Pin the cached base in a run-local adapter view; original checkpoint is untouched.
    original = PALDA / "palda-decoder-pipeline/artifacts/decoder-imported-v2-qwen2.5-7b/checkpoint-221"
    adapter = OUT / "decoder-adapter"
    adapter.mkdir()
    for p in original.iterdir():
        if p.name == "adapter_config.json":
            continue
        if p.name in {"optimizer.pt", "rng_state.pth", "scheduler.pt", "training_args.bin", "trainer_state.json"}:
            continue
        (adapter / p.name).symlink_to(p)
    cfg = json.loads((original / "adapter_config.json").read_text())
    cfg["revision"] = "a09a35458c702b33eeacc393d103063234e8bc28"
    save("decoder-adapter/adapter_config.json", cfg)
    encoder = PALDA / "palda-encoder-pipeline/artifacts/gold-anchor-backbone-pilot/ko-sroberta-champion-candidate-test-audit"
    os.environ.update(PALDA_ENCODER_DIR=str(encoder), PALDA_TAXONOMY_PATH=str(encoder / "taxonomy.json"),
        PALDA_ENCODER_DEVICE="cuda:0", PALDA_ENCODER_BATCH_SIZE="64", PALDA_ENCODER_MAX_LENGTH="256",
        PALDA_ENCODER_VERSION="ko-sroberta-gold-silver-s20260909", PALDA_DECODER_DIR=str(adapter),
        PALDA_DECODER_VERSION="decoder-imported-v2-checkpoint-221", PALDA_DECODER_4BIT="1",
        PALDA_DECODER_DEVICE_MAP="auto", PALDA_DECODER_BATCH_SIZE="10", PALDA_DECODER_VERIFY="0")
    save("status.json", {"stage": "loading_models", **manifest})
    started = time.perf_counter()
    analyzer = analyzer_from_env()
    load_seconds = time.perf_counter() - started
    records = []
    render(records)
    for i, request in enumerate(requests):
        start = time.perf_counter()
        item = {"request": request}
        try:
            payload = {**request, "analysisId": f"inspection-{i:04}", "taxonomyVersion": "theme-v1.0.0"}
            item["result"] = await analyzer.analyze(payload)
            ReportOutput.model_validate(item["result"])
            lookup = {r["id"]: r["text"] for f in ("reviewSample", "qnaSample") for r in request[f]["items"]}
            review = item["result"]["reviewAnalysis"]
            assert all(x["quote"] == lookup[x["quoteEvidenceId"]] for f in ("positiveThemes", "negativeThemes") for x in review[f])
            assert all(x["question"] == lookup[x["questionEvidenceId"]] for x in item["result"]["qnaAnalysis"]["themes"])
            item["schemaAndQuoteChecksPassed"] = True
        except Exception as exc:
            item["error"] = f"{type(exc).__name__}: {exc}"
        item["seconds"] = round(time.perf_counter() - start, 3)
        records.append(item)
        save(f"product-{i:04}.json", item)
        render(records)
        status = {"stage": "running" if i + 1 < len(requests) else "completed",
            "completedProducts": len(records), "totalProducts": len(requests),
            "completedItems": sum(len(r["request"][f]["items"]) for r in records for f in ("reviewSample", "qnaSample")),
            "failedProducts": sum("error" in r for r in records), "modelLoadSeconds": round(load_seconds, 3),
            "totalAnalysisSeconds": round(sum(r["seconds"] for r in records), 3),
            "medianProductSeconds": statistics.median(r["seconds"] for r in records)}
        save("status.json", status)
        print(json.dumps(status), flush=True)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except Exception as exc:
        if OUT.exists():
            save("failure.json", {"error": f"{type(exc).__name__}: {exc}"})
        raise
