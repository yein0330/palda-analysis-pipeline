#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Build one UNUSABLE Q&A generation cell per Cat2")
    parser.add_argument("--taxonomy", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--count-per-category", type=int, default=1)
    args = parser.parse_args()
    taxonomy = json.loads(args.taxonomy.read_text(encoding="utf-8"))
    rows = []
    for category_key, themes in sorted((taxonomy.get("category_themes") or {}).items()):
        if "UNUSABLE" not in themes:
            continue
        rows.append({
            "record_type": "qna",
            "category_key": category_key,
            "category_path": category_key,
            "theme_id": "UNUSABLE",
            "sentiment": None,
            "usable": False,
            "difficulty": "normal",
            "constraints": (
                "비공개 문의, 삭제된 문의, 광고, 의미 없는 반복 문자 중 하나를 짧게 표현하고 "
                "개인정보는 포함하지 말 것"
            ),
            "count": args.count_per_category,
        })
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )
    print(json.dumps({"categories": len(rows), "samples": len(rows) * args.count_per_category}))


if __name__ == "__main__":
    main()
