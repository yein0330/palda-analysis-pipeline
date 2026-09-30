#!/usr/bin/env python3
"""Read-only loopback dashboard. Only allowlisted Validation outputs are served."""
import argparse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import time
from urllib.parse import parse_qs, urlparse

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "artifacts/live-quality-20260911"
BASE = ROOT / "artifacts/label-quality-20260910"
PREVIOUS = ROOT / "artifacts/research-loss-20260910"
SEEDS = (20260909, 42, 1337)
MODELS = {}
for seed in SEEDS:
    MODELS[f"baseline-{seed}"] = (BASE / f"encoder-relabel384-seed{seed}", seed, "보정 라벨 기준", 1.)
    MODELS[f"discount025-{seed}"] = (PREVIOUS / f"silver_negative_discount-seed{seed}", seed, "기존 실험 · 강도 0.25", .25)
    for discount in (.1, .5):
        MODELS[f"negative-{discount:g}-seed{seed}"] = (
            OUT / f"negative-{discount:g}-seed{seed}", seed, f"새 실험 · 강도 {discount:g}", discount)


def read(path, default=None):
    try:
        return json.loads(path.read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        return default


def lines(path):
    try:
        return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    except (FileNotFoundError, json.JSONDecodeError):
        return []


def label_keys(row, predicted=False):
    if not predicted:
        return sorted(f"{v['theme_id']} / {v.get('sentiment') or v.get('role') or 'primary'}"
                      for v in row.get("labels", []))
    if row.get("record_type") == "review":
        return sorted(f"{v['theme_id']} / {v['sentiment']}" for v in row.get("predicted_labels", []))
    return sorted([f"{row['primary_theme_id']} / primary"] +
                  [f"{v['theme_id']} / secondary" for v in row.get("secondary_themes", [])])


def state():
    runs = []
    for key, (path, seed, title, discount) in MODELS.items():
        metrics = read(path / "metrics.json", {})
        live = read(path / "live.json", {})
        runs.append({"id": key, "title": title, "seed": seed, "discount": discount,
                     "stage": live.get("stage", "completed" if metrics else "pending"),
                     "validation": metrics.get("validation", live.get("validation")),
                     "history": metrics.get("history", live.get("history", [])),
                     "predictionStage": "선택된 체크포인트" if metrics else live.get("predictionStage", "아직 없음"),
                     "step": live.get("step", 0), "totalSteps": live.get("totalSteps", 0),
                     "updatedAt": live.get("updatedAt"), "loss": live.get("loss"),
                     "error": live.get("error")})
    return {"runs": runs, "status": read(OUT / "status.json", {"stage": "not_started"}),
            "comparison": read(OUT / "comparison.json", {}), "serverTime": time.time()}


def samples(model_id):
    path, seed, _, _ = MODELS[model_id]
    taxonomy = read(ROOT / "data/releases/label-quality-20260910/relabel384/taxonomy.json", {})
    names = {t["id"]: t for t in taxonomy.get("themes", [])}
    roles = {"primary": "주 문의", "secondary": "보조 문의", "positive": "긍정",
             "negative": "부정", "neutral": "중립"}

    def display(keys, record_type):
        if keys is None:
            return None
        result = []
        for key in keys:
            theme, role = key.split(" / ", 1)
            title = names.get(theme, {}).get("review_label" if record_type == "review" else "qna_label", theme)
            result.append(f"{title} · {roles.get(role, role)}")
        return result
    pred_path = path / "validation_predictions.jsonl"
    if not pred_path.exists():
        pred_path = path / "live_predictions.jsonl"
    predicted = {r["sample_id"]: r for r in lines(pred_path)}
    baseline = {r["sample_id"]: r for r in lines(BASE / f"encoder-relabel384-seed{seed}/validation_predictions.jsonl")}
    values = []
    for source in lines(ROOT / "data/releases/label-quality-20260910/relabel384/validation.jsonl"):
        sid = source["sample_id"]
        gold = label_keys(source)
        before = label_keys(baseline[sid], True) if sid in baseline else None
        after = label_keys(predicted[sid], True) if sid in predicted else None
        outcome = "pending" if after is None else (
            "fixed" if after == gold and before != gold else
            "regressed" if after != gold and before == gold else
            "correct" if after == gold else "wrong")
        values.append({"id": sid, "type": source["record_type"], "text": source["text"],
                       "category": source["category_path"], "gold": gold,
                       "before": before, "after": after, "outcome": outcome,
                       "goldDisplay": display(gold, source["record_type"]),
                       "beforeDisplay": display(before, source["record_type"]),
                       "afterDisplay": display(after, source["record_type"])})
    return {"rows": values, "modelId": model_id, "split": "validation"}


class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        parsed = urlparse(self.path)
        if parsed.path == "/":
            body = (Path(__file__).parent / "quality_dashboard.html").read_bytes()
            mime = "text/html; charset=utf-8"
        elif parsed.path == "/api/state":
            body = json.dumps(state(), ensure_ascii=False).encode()
            mime = "application/json; charset=utf-8"
        elif parsed.path == "/api/samples":
            key = parse_qs(parsed.query).get("model", [""])[0]
            if key not in MODELS:
                self.send_error(400, "Unknown model")
                return
            body = json.dumps(samples(key), ensure_ascii=False).encode()
            mime = "application/json; charset=utf-8"
        else:
            self.send_error(404)
            return
        self.send_response(200)
        self.send_header("Content-Type", mime)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Content-Security-Policy", "default-src 'self'; script-src 'self' 'unsafe-inline'; style-src 'self' 'unsafe-inline'; connect-src 'self'")
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *_):
        pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=18765)
    args = parser.parse_args()
    print(f"Validation dashboard: http://127.0.0.1:{args.port}", flush=True)
    ThreadingHTTPServer(("127.0.0.1", args.port), Handler).serve_forever()
