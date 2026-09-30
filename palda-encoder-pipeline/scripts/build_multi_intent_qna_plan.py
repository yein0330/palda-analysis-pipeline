#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Build Cat2-balanced two-intent Q&A generation cells")
    parser.add_argument("--taxonomy", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--pairs-per-category", type=int, default=3)
    parser.add_argument("--count-per-pair", type=int, default=1)
    parser.add_argument("--seed", type=int, default=20260904)
    args = parser.parse_args()
    taxonomy = json.loads(args.taxonomy.read_text(encoding="utf-8"))
    rows = []
    for category_key, allowed in sorted((taxonomy.get("category_themes") or {}).items()):
        themes = [value for value in allowed if value not in {"OTHER", "UNUSABLE"}]
        if len(themes) < 2:
            continue
        category_seed = int.from_bytes(
            hashlib.sha256(f"{args.seed}:{category_key}".encode()).digest()[:8], "big"
        )
        random.Random(category_seed).shuffle(themes)
        for index in range(min(args.pairs_per_category, len(themes))):
            primary = themes[index]
            secondary = themes[(index + 1) % len(themes)]
            rows.append({
                "record_type": "qna",
                "category_key": category_key,
                "category_path": category_key,
                "theme_id": primary,
                "secondary_theme_ids": [secondary],
                "sentiment": None,
                "usable": True,
                "difficulty": "hard",
                "constraints": (
                    "하나의 자연스러운 구매 전 질문에 두 문의를 모두 명시하되 "
                    "첫 번째 문의가 primary, 두 번째 문의가 secondary가 되게 할 것"
                ),
                "count": args.count_per_pair,
            })
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )
    print(json.dumps({"cells": len(rows), "samples": len(rows) * args.count_per_pair}))


if __name__ == "__main__":
    main()
