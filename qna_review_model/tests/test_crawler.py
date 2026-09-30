import json

import pytest

from qna_review_model.crawler import (
    iter_pages, load_products, normalize_page, validate_completed_run,
)
from qna_review_model.schemas import QNARecord, ReviewRecord


def brand_review_page():
    return {
        "productKey": "store:1", "kind": "review", "page": 1, "status": 200,
        "response": {"contents": [{
            "id": 10, "reviewContent": " 편하고 튼튼해요 ", "reviewScore": 5,
            "productOptionContent": "검정", "createDate": "2026-08-01T00:00:00Z",
            "writerId": "masked***", "productOrderNo": "must-not-leak",
        }]},
    }


def test_normalizes_brand_review_without_private_fields():
    record = list(normalize_page(brand_review_page()))[0]
    assert isinstance(record, ReviewRecord)
    assert record.text == "편하고 튼튼해요"
    assert record.score == 5
    assert record.source == "naver_brand"
    assert "writer" not in record.model_dump_json()
    assert "must-not-leak" not in record.model_dump_json()


def test_normalizes_catalog_review():
    page = {
        "productKey": "catalog:2", "kind": "review", "page": 1, "status": 200,
        "response": {"data": {"reviews": [{
            "mallReviewId": "m1", "content": "배터리가 오래갑니다", "starScore": 4,
            "buyOption": "black", "registerDate": "2026-08-01T00:00:00Z",
        }]}},
    }
    record = list(normalize_page(page))[0]
    assert record.record_id == "review_naver_catalog_m1"
    assert record.option == "black"


def test_normalizes_public_questions_only():
    page = {
        "productKey": "catalog:2", "kind": "qna", "page": 1, "status": 200,
        "response": {"qnaList": [
            {"id": 1, "text": "호환되나요?", "hasSellerAnswer": True, "isReply": False, "isSecret": False},
            {"id": 2, "text": "답변", "isReply": True, "isSecret": False},
            {"id": 3, "text": "비밀", "isReply": False, "isSecret": True},
        ]},
    }
    records = list(normalize_page(page))
    assert len(records) == 1
    assert isinstance(records[0], QNARecord)
    assert records[0].question == "호환되나요?"


def _run(tmp_path):
    chunk = tmp_path / "content-chunks" / "part.jsonl"
    chunk.parent.mkdir()
    chunk.write_text(json.dumps(brand_review_page(), ensure_ascii=False) + "\n", encoding="utf-8")
    (tmp_path / "content-manifest.json").write_text(json.dumps({"chunks": [{
        "status": "sealed", "kind": "mixed", "relative_path": "content-chunks/part.jsonl"
    }]}))
    (tmp_path / "progress.json").write_text(json.dumps({
        "status": "complete", "completion": {"queueDrained": True},
        "consumer": {"jobs": {"failed": 0}},
    }))
    (tmp_path / "inventory.jsonl").write_text(json.dumps({
        "key": "store:1", "product": {
            "name": "의자", "price": 10000, "productId": "1",
            "rawMetadata": {"brand": "브랜드", "characterValue": "검정|대형"},
        },
        "sources": [{"cid": "500", "fullPath": "가구/인테리어 > 의자 > 사무용의자"}],
    }, ensure_ascii=False) + "\n", encoding="utf-8")
    return tmp_path


def test_completed_run_manifest_and_product_aliases(tmp_path):
    run = _run(tmp_path)
    assert validate_completed_run(run)["status"] == "complete"
    assert len(list(iter_pages(run))) == 1
    products = load_products(run)
    assert products["1"].cat1 == "가구/인테리어"
    assert products["store:1"].brand == "브랜드"


def test_incomplete_run_is_rejected(tmp_path):
    run = _run(tmp_path)
    progress = json.loads((run / "progress.json").read_text())
    progress["completion"]["queueDrained"] = False
    (run / "progress.json").write_text(json.dumps(progress))
    with pytest.raises(ValueError, match="not drained"):
        validate_completed_run(run)
