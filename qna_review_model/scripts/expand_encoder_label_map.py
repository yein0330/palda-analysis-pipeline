#!/usr/bin/env python
"""Expand category masks for known canonical themes without changing head size."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--label-map", type=Path, required=True)
    parser.add_argument("--taxonomy", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    original = json.loads(args.label_map.read_text(encoding="utf-8"))
    taxonomy = json.loads(args.taxonomy.read_text(encoding="utf-8"))
    canonical = {row["canonicalThemeId"] for row in original["categoryThemeMetadata"].values()}
    themes = []
    metadata = {}
    masks = {}
    for category in taxonomy["categories"]:
        category_ids = []
        for theme in category["allowedThemes"]:
            if theme["canonicalThemeId"] not in canonical:
                continue
            theme_id = theme["categoryThemeId"]
            themes.append(theme_id); category_ids.append(theme_id)
            metadata[theme_id] = {
                "canonicalThemeId": theme["canonicalThemeId"],
                "level2Id": category["level2Id"], "level2Name": category["level2Name"],
                "reviewLabel": theme["reviewLabel"], "qnaLabel": theme["qnaLabel"],
            }
        masks[category["level2Id"]] = category_ids
    ordered = sorted(set(themes)); index = {value: position for position, value in enumerate(ordered)}
    result = {
        **original,
        "categoryThemeToIndex": index,
        "indexToCategoryTheme": ordered,
        "categoryThemeMetadata": metadata,
        "categoryMasks": {key: [index[value] for value in values] for key, values in masks.items()},
        "maskPolicy": "all taxonomy entries whose canonical theme exists in the fixed model head",
    }
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"canonicalThemes": len(canonical), "categoryThemes": len(ordered),
                      "categories": len(masks), "output": str(args.output)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
