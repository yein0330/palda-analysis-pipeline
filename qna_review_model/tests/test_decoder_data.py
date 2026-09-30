import json

import pytest

from qna_review_model.decoder_data import aggregate_product, build_oracle_decoder_inputs


THEMES = {
    "USABILITY": {"review": "사용성", "qna": "사용성 문의"},
    "RETURN_EXCHANGE": {"review": "교환·반품", "qna": "교환·반품 문의"},
}


def row(item_id, source_type, text, labels, product_id="p1", grade="gold"):
    return {
        "taxonomyVersion": "theme-v1.0.0",
        "itemId": item_id,
        "productGroupId": product_id,
        "sourceType": source_type,
        "category": {"path": "생활/건강 > 테스트"},
        "text": text,
        "usable": True,
        "labels": labels,
        "annotation": {"grade": grade},
    }


def test_aggregate_product_preserves_counts_ratios_and_evidence_ids():
    rows = [
        row("r1", "review", "쓰기 편해요", {"reviewLabels": [
            {"themeId": "USABILITY", "sentiment": "positive"}
        ]}),
        row("r2", "review", "정말 쓰기 편하고 좋아요", {"reviewLabels": [
            {"themeId": "USABILITY", "sentiment": "positive"}
        ]}),
        row("q1", "qna", "교환되나요?", {
            "reviewLabels": [], "qnaPrimaryTheme": "RETURN_EXCHANGE"
        }),
        row("q2", "qna", "사용법이 궁금해요", {
            "reviewLabels": [], "qnaPrimaryTheme": "USABILITY"
        }),
    ]
    result = aggregate_product(rows, THEMES, evidence_limit=1)
    assert result["sample"] == {
        "reviewSampleSize": 2, "reviewMaxItems": 5000,
        "qnaSampleSize": 2, "qnaMaxItems": 5000,
    }
    assert result["reviewGroups"][0]["mentionCount"] == 2
    assert result["reviewGroups"][0]["evidence"] == [
        {"id": "r2", "text": "정말 쓰기 편하고 좋아요"}
    ]
    assert [group["ratio"] for group in result["qnaGroups"]] == [0.5, 0.5]


def test_aggregate_product_rejects_mixed_products():
    rows = [
        row("r1", "review", "좋아요", {"reviewLabels": []}, product_id="p1"),
        row("r2", "review", "좋아요", {"reviewLabels": []}, product_id="p2"),
    ]
    with pytest.raises(ValueError, match="one productGroupId"):
        aggregate_product(rows, THEMES)


def test_builder_uses_gold_only_and_keeps_product_splits(tmp_path):
    input_dir = tmp_path / "input"
    output_dir = tmp_path / "output"
    input_dir.mkdir()
    taxonomy = tmp_path / "taxonomy.json"
    taxonomy.write_text(json.dumps({"canonicalThemes": [
        {
            "canonicalThemeId": "USABILITY",
            "reviewLabel": "사용성",
            "qnaLabel": "사용성 문의",
        }
    ]}), encoding="utf-8")
    for split, product_id in (
        ("train", "p1"), ("validation", "p2"), ("test", "p3")
    ):
        records = [
            row(f"{split}-gold", "review", "편해요", {"reviewLabels": [
                {"themeId": "USABILITY", "sentiment": "positive"}
            ]}, product_id=product_id),
            row(
                f"{split}-silver", "review", "보류", {"reviewLabels": []},
                product_id=f"{product_id}-silver", grade="silver",
            ),
        ]
        (input_dir / f"{split}.jsonl").write_text(
            "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records),
            encoding="utf-8",
        )
    manifest = build_oracle_decoder_inputs(input_dir, output_dir, taxonomy)
    assert manifest["products"] == {"train": 1, "validation": 1, "test": 1}
    assert manifest["sourceGoldRecords"] == {
        "train": 1, "validation": 1, "test": 1
    }
    train_rows = [
        json.loads(line)
        for line in (output_dir / "train.jsonl").read_text(
            encoding="utf-8"
        ).splitlines()
    ]
