#!/usr/bin/env python3
"""Paired input-size/evidence-budget inference pilot. Never trains or opens sealed Test.

Raw expansion is restricted to existing Validation products. Automatic report
consistency/grounding checks are proxies, NOT semantic quality ground truth.
"""
import argparse
import asyncio
from collections import Counter, defaultdict
import csv
import hashlib
import heapq
import html
import json
import os
from pathlib import Path
import random
import re
import statistics
import subprocess
import sys
import threading
import time

ROOT = Path(__file__).resolve().parents[1]
RUNTIME = ROOT.parent / "palda-service-runtime"
RELEASE = ROOT / "data/releases/label-quality-20260910/relabel384"
RAW = ROOT.parent / "qna_review_data/raw/crawling/2026-08-14-full-5090-w12-sqlite"
DEFAULT_OUT = ROOT / "artifacts/input-size-tradeoff-20260921"


def digest(value):
    return hashlib.sha256(value.encode()).hexdigest()


def text_key(value):
    return digest("".join(value.split()).casefold())


def file_hash(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def lines(path):
    with Path(path).open() as stream:
        for line in stream:
            if line.strip():
                yield json.loads(line)


def write(path, value):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    tmp.replace(path)


def sample_request(product, size, seed):
    """Nested, deterministic real-row prefixes; no replication or forced latest order."""
    ordered = sorted(product["items"], key=lambda r: (digest(f"{seed}/{r['id']}"), r["id"]))
    if size > len(ordered):
        return None
    chosen = ordered[:size]
    return {
        "analysisId": "analysis_" + digest(f"{product['id']}/{size}/{seed}")[:32],
        "taxonomyVersion": "theme-v1.0.0",
        "product": {"productId": product["id"], "categoryPath": product["category"]},
        **{key: {"collectionOrder": "newest", "maxItems": 5000,
                 "items": [{"id": r["id"], "text": r["text"]} for r in chosen if r["kind"] == kind]}
           for key, kind in (("reviewSample", "review"), ("qnaSample", "qna"))},
    }


def prepare(args):
    sys.path.insert(0, str(ROOT / "src"))
    from palda_encoder.raw_sample import _aliases, _normalize_page, _summary_index, _validate_run
    out = args.output
    out.mkdir(parents=True, exist_ok=True)
    if (out / "dataset.json").exists():
        raise RuntimeError("Dataset already exists; use its fixed snapshot or a NEW output directory")
    validation = list(lines(RELEASE / "validation.jsonl"))
    training = list(lines(RELEASE / "train.jsonl"))
    val = defaultdict(list)
    for row in validation:
        val[row["product_id"]].append(row)
    train_ids = {r["product_id"].removeprefix("catalog:") for r in training}
    if any(pid.removeprefix("catalog:") in train_ids for pid in val):
        raise RuntimeError("Existing Validation/Train product overlap; stop")
    train_texts = {text_key(r["text"]) for r in training}
    _validate_run(RAW)
    summary = _summary_index(RAW)
    contexts = {}
    for inv in lines(RAW / "inventory.jsonl"):
        aliases = _aliases(inv)
        matches = aliases & set(val)
        if len(matches) != 1 or any(a.removeprefix("catalog:") in train_ids for a in aliases):
            continue
        pid = next(iter(matches))
        source = (inv.get("sources") or [inv.get("source") or {}])[0]
        raw_product = inv.get("product") or {}
        path = source.get("fullPath", val[pid][0]["category_key"])
        stats = summary.get((path.split(">")[-1].strip(), raw_product.get("name", ""),
                             "catalog" if raw_product.get("productGroup") == "catalog" else "single"), {})
        contexts[pid] = {"category_path": path, "category_key": val[pid][0]["category_key"],
                         "product_name": raw_product.get("name", ""), "aliases": sorted(aliases),
                         "review_count": stats.get("collectedReviewCount", 0),
                         "qna_count": stats.get("collectedQnaCount", 0)}
    # Prefer dense products in distinct top-level categories. Selection never uses predictions.
    ranked = sorted(contexts, key=lambda p: (-min(contexts[p]["qna_count"], 1000),
                                           -contexts[p]["review_count"], p))
    candidates, cats = [], set()
    for pid in ranked:
        cat = contexts[pid]["category_key"].split(" > ")[0]
        if cat not in cats:
            candidates.append(pid)
            cats.add(cat)
        if len(candidates) >= args.products * 2:
            break
    if not candidates:
        raise RuntimeError("No eligible Validation raw products")
    aliases = {}
    for pid in candidates:
        for alias in contexts[pid]["aliases"]:
            if alias in aliases and aliases[alias] != pid:
                raise RuntimeError("Ambiguous raw product alias")
            aliases[alias] = pid
    manifest = read(RAW / "content-manifest.json")
    chunks = [RAW / c["relative_path"] for c in manifest["chunks"] if c["status"] == "sealed"]
    pattern = r'"productKey"\s*:\s*"(?:' + '|'.join(re.escape(a) for a in sorted(aliases)) + ')"'
    write(out / "status.json", {"stage": "extracting_validation_products", "candidateProducts": len(candidates)})
    print(f"Scanning {len(chunks)} manifest-listed chunks for {len(candidates)} Validation products", flush=True)
    process = subprocess.Popen(["rg", "--text", "--no-filename", "--no-heading", "--no-line-number",
                                "-e", pattern, "--", *map(str, chunks)], stdout=subprocess.PIPE, text=True)
    heaps, seen, audit = defaultdict(list), set(), Counter()
    for line in process.stdout:
        page = json.loads(line)
        pid = aliases.get(str(page.get("productKey")))
        if pid is None:
            continue
        audit["matched_pages"] += 1
        for row in _normalize_page(page, pid, contexts[pid]):
            key = text_key(row["text"])
            if key in train_texts:
                audit["train_exact_text_excluded"] += 1
                continue
            identity = (pid, row["record_type"], key)
            if identity in seen:
                audit["exact_duplicates_removed"] += 1
                continue
            seen.add(identity)
            item_id = "e_" + digest('/'.join(identity))[:24]
            item = {"id": item_id, "kind": row["record_type"], "text": row["text"], "textKey": key}
            priority = int(digest(f"{args.seed}/{item_id}"), 16)
            heap = heaps[pid]
            entry = (-priority, item_id, item)
            if len(heap) < 5000:
                heapq.heappush(heap, entry)
            elif entry > heap[0]:
                heapq.heapreplace(heap, entry)
    if process.wait() not in (0, 1):
        raise RuntimeError("Raw scan failed")
    products = []
    across_products = set()
    for pid in candidates:
        gold = {(r["record_type"], text_key(r["text"])): r for r in val[pid]}
        items = []
        for _, _, item in sorted(heaps[pid], reverse=True):
            if item["textKey"] in across_products:
                audit["cross_product_exact_duplicates_removed"] += 1
                continue
            g = gold.get((item["kind"], item["textKey"]))
            if g:
                item["gold"] = {"labels": g["labels"], "usable": g["usable"], "tier": g["label_tier"]}
            items.append(item)
        if len(items) < max(args.sizes):
            continue
        products.append({"id": "p_" + digest(pid)[:16], "category": contexts[pid]["category_key"],
                         "items": items,
                         "goldAnchor": [{"kind": r["record_type"], "labels": r["labels"],
                                         "usable": r["usable"], "tier": r["label_tier"]} for r in val[pid]]})
        across_products.update(r["textKey"] for r in items)
        if len(products) >= args.products:
            break
    if not products:
        raise RuntimeError("No product has the requested natural input size; reduce maximum size")
    dataset = {"version": "validation-product-expansion-v1", "products": products,
               "sampling": "seeded nested real-row subsets; NOT chronological newest; no replicated rows",
               "split": "validation-exploratory", "sealedTestRead": False, "trainingPerformed": False,
               "nearDuplicateAudit": "not performed; exact whitespace/casefold dedup only",
               "goldCaveat": "Human labels cover at most 15 rows/product, not the expanded corpus",
               "sourceHashes": {"validation": file_hash(RELEASE / "validation.jsonl"),
                                "train_exclusion": file_hash(RELEASE / "train.jsonl"),
                                "raw_manifest": file_hash(RAW / "content-manifest.json")}, "audit": dict(audit)}
    write(out / "dataset.json", dataset)
    write(out / "status.json", {"stage": "prepared", "products": len(products),
                               "rows": sum(len(p["items"]) for p in products)})
    print(json.dumps({"products": [{"id": p["id"], "category": p["category"],
                      "counts": dict(Counter(i["kind"] for i in p["items"])),
                      "matchedGoldRows": sum("gold" in i for i in p["items"])} for p in products],
                      "audit": dict(audit)}, ensure_ascii=False), flush=True)


def theme_set(result):
    return {f"review/{g['sentiment']}/{g['themeId']}"
            for field in ("positiveThemes", "negativeThemes")
            for g in result.get("reviewAnalysis", {}).get(field, [])} | {
                "qna/" + g["themeId"] for g in result.get("qnaAnalysis", {}).get("themes", [])}


def quality(request, predictions, result, product):
    """Separate mechanical validity, sparse anchor accuracy, and report coverage."""
    source = {r["id"]: r["text"] for k in ("reviewSample", "qnaSample") for r in request[k]["items"]}
    groups = [g for k in ("positiveThemes", "negativeThemes") for g in result["reviewAnalysis"][k]] + result["qnaAnalysis"]["themes"]
    quotes_ok = sum(source.get(g.get("quoteEvidenceId", g.get("questionEvidenceId"))) == g.get("quote", g.get("question")) for g in groups)
    ids = [i for g in groups + result["sellingPoints"] for i in g.get("evidenceIds", [])]
    gold_rows = {r["id"]: r for r in product["items"] if r.get("gold", {}).get("tier") == "gold"}
    tp = fp = fn = nreview = nqna = qcorrect = 0
    for pred in predictions:
        row = gold_rows.get(pred["id"])
        if row is None:
            continue
        gold = row["gold"]
        if row["kind"] == "review":
            nreview += 1
            expected = {(g["theme_id"], g["sentiment"]) for g in gold["labels"] if gold["usable"] and g.get("sentiment")}
            actual = {(g["themeId"], g["sentiment"]) for g in pred.get("labels", [])} if pred["usable"] else set()
            tp += len(expected & actual); fp += len(actual - expected); fn += len(expected - actual)
        else:
            expected = next((g["theme_id"] for g in gold["labels"] if g.get("role") == "primary"), None)
            if gold["usable"] and expected:
                nqna += 1
                qcorrect += bool(pred["usable"] and pred.get("primaryTheme", {}).get("themeId") == expected)
    anchor = set()
    for row in product["goldAnchor"]:
        if not row["usable"] or row["tier"] != "gold":
            continue
        for label in row["labels"]:
            if row["kind"] == "review" and label.get("sentiment") in ("positive", "negative"):
                anchor.add(f"review/{label['sentiment']}/{label['theme_id']}")
            elif row["kind"] == "qna" and label.get("role") == "primary":
                anchor.add("qna/" + label["theme_id"])
    return {"quote_exact_match": quotes_ok / len(groups) if groups else None,
            "evidence_id_validity": sum(i in source for i in ids) / len(ids) if ids else None,
            "fallback_summary_rate": sum(g.get("fallbackUsed") is True for g in groups) / len(groups) if groups else None,
            "gold_review_rows": nreview, "gold_review_tp": tp, "gold_review_fp": fp, "gold_review_fn": fn,
            "gold_review_micro_f1": 2 * tp / (2 * tp + fp + fn) if nreview and 2 * tp + fp + fn else None,
            "gold_qna_rows": nqna, "gold_qna_correct": qcorrect,
            "gold_qna_accuracy": qcorrect / nqna if nqna else None,
            "gold_anchor_theme_coverage": len(anchor & theme_set(result)) / len(anchor) if anchor else None,
            "gold_anchor_theme_count": len(anchor), "report_group_count": len(groups),
            "selling_point_count": len(result["sellingPoints"]),
            "summary_semantic_faithfulness": None, "selling_point_usefulness": None}


class MemorySampler:
    """Sample process RSS every 50ms; GPU allocator peaks measured separately."""
    def __enter__(self):
        import psutil
        self.process = psutil.Process()
        self.peak = self.process.memory_info().rss
        self.stop = threading.Event()
        def poll():
            while not self.stop.wait(.05):
                self.peak = max(self.peak, self.process.memory_info().rss)
        self.thread = threading.Thread(target=poll, daemon=True)
        self.thread.start()
        return self

    def __exit__(self, *_):
        self.peak = max(self.peak, self.process.memory_info().rss)
        self.stop.set(); self.thread.join()


def render(out):
    records = [read(p) for p in sorted((out / "cases").glob("*.json"))]
    refs = {(r["product"], r["repeat"]): r for r in records
            if r.get("status") == "completed" and r["isReference"]}
    flattened = []
    for r in records:
        if r.get("status") != "completed":
            continue
        ref = refs.get((r["product"], r["repeat"]))
        q = r["quality"]
        row = {k: r[k] for k in ("caseId", "product", "category", "size", "evidenceBudget", "repeat", "seconds", "reviewRows", "qnaRows")}
        row.update(q)
        row.update(r["memory"])
        if ref:
            a, b = theme_set(r["result"]), theme_set(ref["result"])
            row["full_sample_theme_jaccard_proxy"] = len(a & b) / len(a | b) if a | b else None
        flattened.append(row)
    if flattened:
        keys = sorted({k for row in flattened for k in row})
        with (out / "summary.csv").open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=keys); writer.writeheader(); writer.writerows(flattened)
    write(out / "summary.json", flattened)
    esc = lambda v: html.escape(str(v))
    points = []
    max_time = max((r["seconds"] for r in flattened), default=1)
    colors = {1: "#db7629", 3: "#3972d8", 8: "#139c76"}
    for r in flattened:
        y = r.get("gold_anchor_theme_coverage")
        if y is not None:
            x = 65 + r["seconds"] / max_time * 650
            points.append(f'<circle cx="{x:.1f}" cy="{300-y*250:.1f}" r="6" fill="{colors.get(r["evidenceBudget"], "#777")}"><title>{esc(r["caseId"])}: {r["seconds"]:.2f}s / {y:.3f}</title></circle>')
    doc = ['''<!doctype html><html lang="ko"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>PALDA 시간·입력량 비교</title>
<style>body{font:16px/1.6 system-ui;max-width:1200px;margin:auto;padding:24px;background:#f4f6fb;color:#223}section{background:white;padding:20px;margin:16px 0;border-radius:12px}table{border-collapse:collapse;width:100%}td,th{border:1px solid #ddd;padding:8px;text-align:left}pre{white-space:pre-wrap;overflow-wrap:anywhere}summary{cursor:pointer}blockquote{border-left:4px solid #abc;padding:10px}.warning{background:#fff0ce;padding:16px}</style>
<h1>PALDA · 입력량 / 근거 수 / 시간 비교</h1><p class="warning">실제 GPU 실행 · 탐색적 Validation 상품 확장. 최신순 표본이 아닌 고정 seed 무작위 부분집합입니다. 입력 원문은 로컬에만 저장합니다. 세로축은 일부 사람 정답 테마의 포착률이며 요약·소구점 의미 품질 점수가 아닙니다. 정답 없는 항목은 미평가입니다. 최대 입력 결과도 정답이 아닙니다.</p>''']
    doc.append(f'<section><h2>시간 × 사람 정답 테마 포착률</h2><p>파랑: 근거 3개, 주황: 1개, 초록: 8개. 점에 마우스를 올리면 조건을 확인할 수 있습니다.</p><svg viewBox="0 0 800 350" role="img" aria-label="추론시간과 일부 정답 테마 포착률 산점도"><path d="M65 40V300H730" stroke="#777" fill="none"/><text x="5" y="50">100%</text><text x="15" y="300">0%</text><text x="65" y="330">0초</text><text x="650" y="330">{max_time:.1f}초</text>{"".join(points)}</svg></section>')
    headers = ["상품", "입력 건수", "근거/테마", "반복", "시간(s)", "정답 테마 포착률", "VRAM allocated(GiB)", "RSS RAM(GiB)"]
    doc.append('<section><h2>현재까지 완료된 실험</h2><table><tr>' + ''.join(f'<th>{h}</th>' for h in headers) + '</tr>')
    for r in flattened:
        vals = [r["product"], r["size"], r["evidenceBudget"], r["repeat"], round(r["seconds"], 2), r.get("gold_anchor_theme_coverage"), round(r["gpu_allocated_gib"], 2), round(r["rss_peak_gib"], 2)]
        doc.append('<tr>' + ''.join(f'<td>{esc(v)}</td>' for v in vals) + '</tr>')
    doc.append('</table></section>')
    for r in records:
        doc.append(f'<section><details><summary>{esc(r["caseId"])} · {esc(r["category"])} · {esc(r.get("status"))}</summary>')
        if r.get("status") == "completed":
            for key in ("positiveThemes", "negativeThemes"):
                for g in r["result"]["reviewAnalysis"][key]:
                    doc.append(f'<h3>{esc(g["theme"])} · {esc(g["sentiment"])}</h3><p>{esc(g["summary"])}</p><blockquote>{esc(g["quote"])}</blockquote>')
            for g in r["result"]["qnaAnalysis"]["themes"]:
                doc.append(f'<h3>Q&A · {esc(g["theme"])}</h3><p>{esc(g["summary"])}</p><blockquote>{esc(g["question"])}</blockquote>')
            doc.append('<h3>소구점 후보</h3>' + ''.join(f'<p><b>{esc(g["type"])}</b> {esc(g["text"])}</p>' for g in r["result"]["sellingPoints"]))
        for key, label in (("request", "입력 원문"), ("predictions", "인코더 출력"), ("decoderInput", "디코더 근거"), ("decoderRaw", "생성 원응답"), ("quality", "평가 지표")):
            doc.append(f'<details><summary>{label}</summary><pre>{esc(json.dumps(r.get(key), ensure_ascii=False, indent=2))}</pre></details>')
        doc.append('</details></section>')
    tmp = out / "review.tmp.html"
    tmp.write_text('\n'.join(doc) + '</html>'); tmp.replace(out / "review.html")


async def run(args):
    sys.path.insert(0, str(RUNTIME / "src"))
    from palda_runtime.models import GroundedDecoderModel, ServiceEncoderModel
    from palda_runtime.aggregation import build_report_input, materialize_grounded_report
    from palda_runtime.schemas import ReportOutput
    import torch
    out = args.output
    dataset = read(out / "dataset.json")
    products = dataset["products"][:args.products]
    encoder_dir = ROOT / "artifacts/gold-anchor-backbone-pilot/ko-sroberta-champion-candidate-test-audit"
    decoder_dir = RUNTIME / "artifacts/existing-model-1000-20260910/decoder-adapter"
    published = read(RUNTIME / "artifacts/integration-release-20260910/published.json")
    for kind, directory in (("encoder", encoder_dir), ("decoder", decoder_dir)):
        for name, expected in published[kind]["files"].items():
            if name == "metrics.json":
                continue
            if file_hash(directory / name) != expected:
                raise RuntimeError(f"Pinned {kind} artifact checksum mismatch: {name}")
    config = {"sizes": args.sizes, "evidenceBudgets": args.evidence_budgets, "repeats": args.repeats,
              "seed": args.seed, "products": [p["id"] for p in products], "datasetSha256": file_hash(out / "dataset.json"),
              "encoder": str(encoder_dir), "decoder": str(decoder_dir), "decoderBatch": 10,
              "decoderMaxInputTokens": 2048, "encoderMaxTokens": 256,
              "sourceHashes": {f: file_hash(RUNTIME / "src/palda_runtime" / f) for f in ("models.py", "aggregation.py", "schemas.py")},
              "torch": torch.__version__, "physicalGpu": os.environ.get("CUDA_VISIBLE_DEVICES"),
              "artifactChecksumsVerified": True, "runnerSha256": file_hash(__file__),
              "split": dataset["split"], "httpQueueIncluded": False,
              "qualityScope": "sparse human anchor coverage + mechanical/proxy checks; semantic quality NOT evaluated"}
    if (out / "config.json").exists() and read(out / "config.json") != config:
        raise RuntimeError("Cannot resume with changed configuration; use a new output directory")
    write(out / "config.json", config)
    (out / "cases").mkdir(exist_ok=True)
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("Expose exactly one free GPU through CUDA_VISIBLE_DEVICES")
    torch.manual_seed(args.seed)
    started = time.perf_counter()
    write(out / "status.json", {"stage": "loading", "completed": 0})
    with MemorySampler() as memory:
        encoder = ServiceEncoderModel(encoder_dir, encoder_dir / "taxonomy.json", device="cuda:0", batch_size=64)
        decoder = GroundedDecoderModel(decoder_dir, device_map="cuda:0", load_in_4bit=True, batch_size=10)
        torch.cuda.synchronize()
    write(out / "load.json", {"seconds": time.perf_counter() - started, "rss_peak_gib": memory.peak / 2**30,
                              "gpu_allocated_gib": torch.cuda.max_memory_allocated() / 2**30,
                              "gpu_reserved_gib": torch.cuda.max_memory_reserved() / 2**30,
                              "gpu": torch.cuda.get_device_name(0)})
    versions = {"encoderModel": "213eb63610eeac4bb24b5d0272095b2110553a05",
                "decoderModel": "3b97b7aefad2769f67742beec183056f4d06804b",
                "taxonomy": "theme-v1.0.0", "calculation": "review-qna-calc-v1.0.1"}
    actual_complete = decoder._complete_many
    actual_generate = decoder.model.generate
    generated_batches, token_stats = [], []
    def count_generate(*a, **kw):
        result = actual_generate(*a, **kw)
        width = kw["input_ids"].shape[1]
        continuation = result[:, width:]
        lengths = []
        eos = decoder.tokenizer.eos_token_id
        for row in continuation.tolist():
            lengths.append(row.index(eos) + 1 if eos in row else len(row))
        token_stats.append({"input_tokens": int(kw["attention_mask"].sum()),
                            "generated_tokens": sum(lengths), "sequences": len(lengths)})
        return result
    decoder.model.generate = count_generate
    def capture(prompts, **kw):
        result = actual_complete(prompts, **kw)
        generated_batches.append(result)
        return result
    decoder._complete_many = capture
    # Warm the entire model pipeline once. Excluded from timed observations.
    warm = sample_request(products[0], min(args.sizes), args.seed)
    p = await encoder.classify(warm)
    ri = build_report_input(warm, p)
    await decoder.generate({k: v for k, v in ri.items() if not k.startswith('_')})
    tasks = []
    skipped = []
    for rep in range(args.repeats):
        for product in products:
            for size in args.sizes:
                if size > len(product["items"]):
                    skipped.append({"product": product["id"], "size": size, "reason": "insufficient real rows"})
                    continue
                budgets = args.evidence_budgets if size == max(args.sizes) else [3]
                tasks.extend((product, size, b, rep) for b in budgets)
    random.Random(args.seed).shuffle(tasks)
    write(out / "skipped.json", skipped)
    for index, (product, size, budget, rep) in enumerate(tasks):
        case_id = f"{product['id']}-n{size}-e{budget}-r{rep}"
        target = out / "cases" / (case_id + ".json")
        if target.exists():
            continue
        write(out / "status.json", {"stage": "running", "current": case_id,
                                   "completed": len(list((out / "cases").glob('*.json'))), "planned": len(tasks)})
        request = sample_request(product, size, args.seed)
        trace = {"caseId": case_id, "product": product["id"], "category": product["category"],
                 "size": size, "evidenceBudget": budget, "repeat": rep, "request": request,
                 "isReference": size == max(args.sizes) and budget == 3,
                 "reviewRows": len(request["reviewSample"]["items"]), "qnaRows": len(request["qnaSample"]["items"])}
        generated_batches.clear(); token_stats.clear()
        torch.cuda.reset_peak_memory_stats(); torch.cuda.synchronize()
        begin = time.perf_counter()
        try:
            with MemorySampler() as memory:
                predictions = await encoder.classify(request)
                torch.cuda.synchronize(); classified = time.perf_counter()
                report = build_report_input(request, predictions, evidence_limit=budget)
                decoder_input = {k: v for k, v in report.items() if not k.startswith('_')}
                aggregated = time.perf_counter()
                raw = await decoder.generate(decoder_input)
                torch.cuda.synchronize(); generated = time.perf_counter()
                result = materialize_grounded_report(report, raw)
                result["versions"] = versions
                result = ReportOutput.model_validate(result).model_dump()
                finish = time.perf_counter()
            trace.update(status="completed", seconds=finish-begin, predictions=predictions,
                         decoderInput=decoder_input, decoderRaw=list(generated_batches), result=result,
                         stages={"encoder": classified-begin, "aggregation": aggregated-classified,
                                 "decoder": generated-aggregated, "validation": finish-generated},
                         memory={"rss_peak_gib": memory.peak / 2**30,
                                 "gpu_allocated_gib": torch.cuda.max_memory_allocated() / 2**30,
                                 "gpu_reserved_gib": torch.cuda.max_memory_reserved() / 2**30},
                         tokens={k: sum(b[k] for b in token_stats) for k in ("input_tokens", "generated_tokens", "sequences")},
                         quality=quality(request, predictions, result, product))
        except Exception as exc:
            trace.update(status="failed", seconds=time.perf_counter()-begin, errorType=type(exc).__name__)
            write(target, trace); render(out)
            write(out / "status.json", {"stage": "failed", "current": case_id, "errorType": type(exc).__name__})
            raise
        write(target, trace); render(out)
        print(json.dumps({"case": case_id, "seconds": round(trace["seconds"], 2),
                          "anchorCoverage": trace["quality"]["gold_anchor_theme_coverage"],
                          "progress": f"{index+1}/{len(tasks)}"}), flush=True)
    render(out)
    write(out / "status.json", {"stage": "completed", "completed": len(tasks), "planned": len(tasks),
                               "finishedAt": time.time(), "semanticQualityEvaluated": False})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("prepare", "run", "render"))
    parser.add_argument("--output", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--sizes", type=int, nargs="+", default=[10, 50, 100, 300, 500, 1000])
    parser.add_argument("--evidence-budgets", type=int, nargs="+", default=[1, 3, 8])
    parser.add_argument("--products", type=int, default=3)
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if not args.sizes or min(args.sizes) < 1 or max(args.sizes) > 5000 or args.products < 1 or args.repeats < 1:
        parser.error("Positive sizes <=5000, products and repeats required")
    if 3 not in args.evidence_budgets or min(args.evidence_budgets) < 1:
        parser.error("Evidence budgets must include baseline 3 and be positive")
    args.sizes = sorted(set(args.sizes))
    args.evidence_budgets = sorted(set(args.evidence_budgets))
    if args.mode == "prepare":
        prepare(args)
    elif args.mode == "render":
        render(args.output)
    else:
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
        asyncio.run(run(args))


if __name__ == "__main__":
    main()
