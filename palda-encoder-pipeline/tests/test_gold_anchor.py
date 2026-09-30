from __future__ import annotations

import json

from palda_encoder.gold_anchor import create_gold_anchor_releases


def _write_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")


def _row(sample_id, product_id, text, tier="gold"):
    return {
        "sample_id": sample_id, "product_id": product_id, "text": text,
        "record_type": "review", "label_tier": tier, "source_split": "simulation",
    }


def test_gold_anchor_release_preserves_product_isolation(tmp_path):
    benchmark = tmp_path / "benchmark"
    base = tmp_path / "base"
    output = tmp_path / "output"
    _write_jsonl(benchmark / "simulation.jsonl", [
        _row("s1", "p1", "좋아요"), _row("s2", "p1", "편해요"),
        _row("s3", "p2", "비싸요"), _row("s4", "p2", "작아요"),
    ])
    _write_jsonl(benchmark / "validation.jsonl", [_row("v1", "p3", "검증")])
    _write_jsonl(benchmark / "test.jsonl", [_row("t1", "p4", "시험")])
    _write_jsonl(base / "train.jsonl", [_row("b1", "p5", "실버", "silver")])
    (base / "taxonomy.json").write_text("{}", encoding="utf-8")

    report = create_gold_anchor_releases(benchmark, base, output, seed=7)

    assert report["gold_anchor"]["rows"] == 2
    assert report["simulation_holdout"]["rows"] == 2
    manifest = json.loads((output / "gold_silver" / "manifest.json").read_text())
    assert manifest["counts"] == {"train": 3, "validation": 1, "test": 1, "simulation": 2}
    assert manifest["label_tiers"] == {"silver": 1, "gold": 2}
    train_products = {json.loads(line)["product_id"] for line in (output / "gold_silver" / "train.jsonl").read_text().splitlines()}
    holdout_products = {json.loads(line)["product_id"] for line in (output / "gold_silver" / "simulation.jsonl").read_text().splitlines()}
    assert train_products.isdisjoint(holdout_products)
