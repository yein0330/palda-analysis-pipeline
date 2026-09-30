from __future__ import annotations

import hashlib
import random
from collections import Counter
from pathlib import Path
from typing import Any

from .io import iter_jsonl, read_json, write_json, write_jsonl


THEME_DEFINITIONS = {
    "QUALITY": "제작 완성도, 마감, 결함과 전반적 제품 품질. 단순한 '좋아요'는 포함하지 않음",
    "DURABILITY": "시간이나 반복 사용 뒤에도 성능과 형태가 유지되는 정도",
    "FIT": "몸 또는 대상에 맞는 정도, 조임, 지지력과 착용 핏",
    "COMFORT": "사용·착용 중 편안함, 압박감, 통증 또는 불편함",
    "USABILITY": "착용·조작·설치·사용 절차가 쉽거나 어려운 정도",
    "MATERIAL": "소재의 종류, 촉감, 통기성 등 재질 자체에 대한 평가",
    "COLOR": "실물 색상, 사진 대비 색 차이와 색감",
    "SIZE": "치수, 크기, 사이즈 선택 또는 실제 크기에 대한 평가·문의",
    "DELIVERY": "출고·도착 시점과 배송 속도에 대한 내용",
    "PACKAGING": "박스, 완충재, 포장 상태에 대한 내용",
    "RETURN_EXCHANGE": "반품·교환 가능 여부, 조건과 절차에 대한 문의",
    "PRICE": "가격, 할인, 쿠폰과 가성비에 대한 평가",
}

CONFUSABLES = {
    "QUALITY": ["DURABILITY", "MATERIAL", "USABILITY"],
    "DURABILITY": ["QUALITY", "MATERIAL"],
    "FIT": ["COMFORT", "SIZE", "ADJUSTABILITY"],
    "COMFORT": ["FIT", "MATERIAL", "USABILITY"],
    "USABILITY": ["COMFORT", "ADJUSTABILITY", "FIT"],
    "MATERIAL": ["QUALITY", "COMFORT", "DURABILITY"],
    "COLOR": ["DESIGN", "QUALITY"],
    "SIZE": ["FIT", "ADJUSTABILITY"],
    "DELIVERY": ["PACKAGING", "CUSTOMER_SUPPORT"],
    "PACKAGING": ["DELIVERY", "QUALITY"],
    "RETURN_EXCHANGE": ["SIZE", "AFTER_SALES"],
    "PRICE": ["QUALITY"],
}

ATTRIBUTE_GRID = [
    {"length": "very_short", "style": "구어체", "noise": "오타 한 개", "stance": "단일 장단점"},
    {"length": "short", "style": "후기체", "noise": "띄어쓰기 생략", "stance": "담담함"},
    {"length": "medium", "style": "비교 후기", "noise": "없음", "stance": "장단점 혼합"},
    {"length": "short", "style": "질문체", "noise": "축약어", "stance": "구매 전 고민"},
    {"length": "medium", "style": "모바일 구어체", "noise": "문장부호 생략", "stance": "구체적 사용 상황"},
]


def build_strategy_plans(
    release_dir: str | Path,
    output_dir: str | Path,
    *,
    strategies: list[str] | None = None,
    max_per_strategy: int = 120,
    seed: int = 42,
) -> dict[str, Any]:
    if max_per_strategy < 1:
        raise ValueError("max_per_strategy must be positive")
    release = Path(release_dir)
    taxonomy = read_json(release / "taxonomy.json")
    allowed_by_category = {
        str(key): set(map(str, value))
        for key, value in taxonomy.get("category_themes", {}).items()
    }
    candidates = []
    for row in iter_jsonl(release / "train.jsonl"):
        labels = list(row.get("labels") or [])
        for label in labels:
            theme_id = str(label.get("theme_id") or "")
            if not theme_id or theme_id in {"OTHER", "UNUSABLE"}:
                continue
            candidates.append((row, label))
    if not candidates:
        raise ValueError("release train split has no labeled candidates")

    selected_strategies = strategies or ["plain", "attributed", "contrastive", "source_transform"]
    unknown = set(selected_strategies) - {"plain", "attributed", "contrastive", "source_transform"}
    if unknown:
        raise ValueError(f"unknown strategies: {sorted(unknown)}")
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    manifest: dict[str, Any] = {
        "schema": "palda-synthetic-strategy-plan-v1",
        "release": str(release.resolve()),
        "seed": seed,
        "max_per_strategy": max_per_strategy,
        "strategies": {},
    }
    for strategy_index, strategy in enumerate(selected_strategies):
        eligible = [
            item for item in candidates
            if strategy != "contrastive" or str(item[1].get("theme_id")) in CONFUSABLES
        ]
        rng = random.Random(seed + strategy_index * 1009)
        rng.shuffle(eligible)
        # Greedily prefer cells used less often to preserve Cat2/theme/sentiment diversity.
        usage: Counter[tuple[str, str, str, str | None]] = Counter()
        rows = []
        pool = list(eligible)
        while pool and len(rows) < max_per_strategy:
            window = pool[: min(200, len(pool))]
            best = min(window, key=lambda item: usage[_cell(item)])
            pool.remove(best)
            sample, label = best
            target = _target(sample, label, strategy, len(rows))
            category_key = str(target["category_key"])
            allowed = allowed_by_category.get(category_key, set())
            target["confusable_theme_ids"] = [
                value for value in target.get("confusable_theme_ids", []) if value in allowed
            ]
            target["forbidden_theme_ids"] = list(target["confusable_theme_ids"])
            rows.append(target)
            usage[_cell(best)] += 1
        path = out / f"{strategy}.jsonl"
        write_jsonl(path, rows)
        manifest["strategies"][strategy] = {
            "path": path.name,
            "samples": len(rows),
            "record_types": dict(sorted(Counter(row["record_type"] for row in rows).items())),
            "themes": dict(sorted(Counter(row["theme_id"] for row in rows).items())),
            "categories": len({row["category_key"] for row in rows}),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }
    write_json(out / "manifest.json", manifest)
    return manifest


def _cell(item: tuple[dict[str, Any], dict[str, Any]]) -> tuple[str, str, str, str | None]:
    sample, label = item
    return (str(sample["record_type"]), str(sample.get("category_key") or ""),
            str(label.get("theme_id") or ""), label.get("sentiment"))


def _target(
    sample: dict[str, Any], label: dict[str, Any], strategy: str, index: int,
) -> dict[str, Any]:
    record_type = str(sample["record_type"])
    theme_id = str(label["theme_id"])
    row: dict[str, Any] = {
        "record_type": record_type,
        "category_key": str(sample.get("category_key") or sample.get("category_path") or ""),
        "category_path": str(sample.get("category_path") or sample.get("category_key") or ""),
        "theme_id": theme_id,
        "sentiment": label.get("sentiment") if record_type == "review" else None,
        "difficulty": "hard" if strategy == "contrastive" else "normal",
        "strategy": strategy,
        "theme_definitions": {
            key: THEME_DEFINITIONS[key]
            for key in [theme_id, *CONFUSABLES.get(theme_id, [])]
            if key in THEME_DEFINITIONS
        },
        "confusable_theme_ids": list(CONFUSABLES.get(theme_id, [])),
        "constraints": "목표 의미가 원문에 명시되어야 하며 관련 없는 테마를 추가하지 말 것",
        "count": 1,
        "parent_sample_id": str(sample["sample_id"]),
    }
    if strategy == "attributed":
        row["attributes"] = ATTRIBUTE_GRID[index % len(ATTRIBUTE_GRID)]
    if strategy == "source_transform":
        row["source_text"] = str(sample["text"])
        row["attributes"] = ATTRIBUTE_GRID[index % len(ATTRIBUTE_GRID)]
    return row
