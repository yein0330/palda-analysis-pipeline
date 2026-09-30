from __future__ import annotations

import argparse
import asyncio
import json
import os
import time
import uuid
from pathlib import Path
from typing import Any

from .runtime import analyzer_from_env


def default_request() -> dict[str, Any]:
    return {
        "analysisId": f"smoke-{uuid.uuid4().hex[:12]}",
        "taxonomyVersion": os.environ.get("PALDA_TAXONOMY_VERSION", "theme-v1.0.0"),
        "product": {
            "productId": "palda-runtime-smoke",
            "categoryPath": "가구/인테리어 > 침구세트",
        },
        "reviewSample": {
            "maxItems": 5000,
            "items": [
                {"id": "r1", "text": "촉감이 부드럽고 색감도 사진과 같아서 만족해요."},
                {"id": "r2", "text": "박스가 찌그러져 왔고 배송도 예상보다 늦었어요."},
                {"id": "r3", "text": "세탁 후에도 모양이 잘 유지돼서 쓰기 편합니다."},
            ],
        },
        "qnaSample": {
            "maxItems": 5000,
            "items": [
                {"id": "q1", "text": "퀸 사이즈 침대에도 맞나요?"},
                {"id": "q2", "text": "오늘 주문하면 언제 출고되나요?"},
                {"id": "q3", "text": "세탁한 상품도 교환할 수 있나요?"},
            ],
        },
    }


async def run(request: dict[str, Any]) -> dict[str, Any]:
    started = time.perf_counter()
    analyzer = analyzer_from_env()
    loaded = time.perf_counter()
    result = await analyzer.analyze(request)
    finished = time.perf_counter()
    return {
        "timing": {
            "modelLoadSeconds": round(loaded - started, 3),
            "analysisSeconds": round(finished - loaded, 3),
            "totalSeconds": round(finished - started, 3),
        },
        "result": result,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Load the configured encoder and decoder and run one real end-to-end analysis."
    )
    parser.add_argument("--input", type=Path, help="Optional service request JSON")
    parser.add_argument("--output", type=Path, help="Optional result JSON path")
    args = parser.parse_args()
    request = (
        json.loads(args.input.read_text(encoding="utf-8"))
        if args.input else default_request()
    )
    payload = asyncio.run(run(request))
    rendered = json.dumps(payload, ensure_ascii=False, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)


if __name__ == "__main__":
    main()
