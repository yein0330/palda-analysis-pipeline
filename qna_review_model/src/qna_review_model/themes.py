"""Hierarchical Global -> Cat1 -> Cat2 -> leaf theme resolution."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .schemas import ProductContext, ThemeDefinition


class ThemeDictionary:
    def __init__(self, payload: dict[str, Any]):
        self.payload = payload

    @classmethod
    def from_path(cls, path: Path) -> "ThemeDictionary":
        return cls(json.loads(path.read_text(encoding="utf-8")))

    def resolve(self, product: ProductContext) -> list[ThemeDefinition]:
        merged: dict[str, ThemeDefinition] = {}

        def add(rows: list[dict[str, Any]], level: str) -> None:
            for row in rows:
                merged[row["theme_id"]] = ThemeDefinition(**row, source_level=level)

        add(self.payload.get("global", []), "global")
        levels = product.category_levels
        if levels:
            add(self.payload.get("cat1", {}).get(levels[0], []), "cat1")
        if len(levels) >= 2:
            add(self.payload.get("cat2", {}).get(" > ".join(levels[:2]), []), "cat2")
        if product.category_path:
            add(self.payload.get("leaf", {}).get(product.category_path, []), "leaf")
        return list(merged.values())

    def validate(self) -> dict[str, int]:
        counts = {"global": len(self.payload.get("global", [])), "cat1": 0, "cat2": 0, "leaf": 0}
        for level in ("cat1", "cat2", "leaf"):
            for rows in self.payload.get(level, {}).values():
                counts[level] += len(rows)
                for row in rows:
                    ThemeDefinition(**row, source_level=level)
        return counts
