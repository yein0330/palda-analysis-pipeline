from __future__ import annotations

import hashlib
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from .io import iter_jsonl, read_json, sha256_file, stable_hash, write_json, write_jsonl
from .schema import make_category_key


def build_human_benchmark(
    source_path: str | Path,
    taxonomy_path: str | Path,
    output_dir: str | Path,
    *,
    seed: int = 20260904,
    simulation_ratio: float = 0.5,
    validation_ratio: float = 0.1,
) -> dict[str, Any]:
    """Create immutable human-only evaluation splits.

    Products connected by an exact normalized-text duplicate are treated as one
    component, so neither a product nor a duplicate text can cross a boundary.
    Gold rows that are not marked valid are deliberately excluded.
    """
    if not 0 < simulation_ratio < 1:
        raise ValueError("simulation_ratio must be between zero and one")
    if not 0 < validation_ratio < 1 - simulation_ratio:
        raise ValueError("validation_ratio must leave a positive test split")

    taxonomy = read_json(taxonomy_path)
    taxonomy_by_ids, category_themes = _taxonomy_indexes(taxonomy)
    rows: list[dict[str, Any]] = []
    excluded = Counter()
    for raw in iter_jsonl(source_path):
        annotation = raw.get("annotation") or {}
        if str(annotation.get("grade") or "").lower() != "gold":
            excluded["not_gold"] += 1
            continue
        if annotation.get("validationStatus") != "valid":
            excluded["not_valid"] += 1
            continue
        usable = bool(raw.get("usable", True))
        labels = (
            _labels_for(raw)
            if usable
            else [{
                "theme_id": "UNUSABLE", "sentiment": None,
                "evidence": None, "role": "primary",
            }]
        )
        if not labels:
            excluded["no_labels"] += 1
            continue
        category = raw.get("category") or {}
        category_path = str(category.get("path") or "")
        category_key = " > ".join(
            value
            for value in (category.get("level1Name"), category.get("level2Name"))
            if value
        ) or make_category_key(category_path)
        product_id = str(raw.get("productGroupId") or raw["itemId"])
        candidates = taxonomy_by_ids.get(
            (str(category.get("level1Id")), str(category.get("level2Id")))
        ) or category_themes.get(category_key) or sorted({x["theme_id"] for x in labels})
        rows.append(
            {
                "sample_id": str(raw["itemId"]),
                "product_id": product_id,
                "record_type": str(raw["sourceType"]),
                "text": str(raw["text"]),
                "category_path": category_path,
                "category_key": category_key,
                "source_kind": "real",
                "source_split": None,
                "usable": usable,
                "metadata": {
                    "created_at": raw.get("createdAt"),
                    "category_level1_id": category.get("level1Id"),
                    "category_level2_id": category.get("level2Id"),
                    "annotation_source": annotation.get("source"),
                    "reviewer_status": annotation.get("reviewerStatus"),
                    "benchmark_seed": seed,
                },
                "labels": labels,
                "candidate_theme_ids": candidates,
                "label_tier": "gold",
                "agreement_score": 1.0,
            }
        )

    components = _product_components(rows)
    rows_by_product = Counter(row["product_id"] for row in rows)
    split_by_product = _assign_components(
        components, rows_by_product, seed, simulation_ratio, validation_ratio
    )
    splits: dict[str, list[dict[str, Any]]] = defaultdict(list)
    seen_text: set[str] = set()
    duplicates = 0
    for row in sorted(rows, key=lambda value: value["sample_id"]):
        text_key = _text_key(row["text"])
        if text_key in seen_text:
            duplicates += 1
            continue
        seen_text.add(text_key)
        split = split_by_product[row["product_id"]]
        row["source_split"] = split
        splits[split].append(row)

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    paths = {name: out / f"{name}.jsonl" for name in ("simulation", "validation", "test")}
    for name, path in paths.items():
        write_jsonl(path, splits.get(name, []))
    blocked_products = out / "blocked_product_ids.txt"
    blocked_products.write_text(
        "".join(f"{value}\n" for value in sorted(split_by_product)), encoding="utf-8"
    )

    products = {
        name: {row["product_id"] for row in splits.get(name, [])} for name in paths
    }
    text_sets = {
        name: {_text_key(row["text"]) for row in splits.get(name, [])} for name in paths
    }
    leakage = {
        "product_simulation_validation": len(products["simulation"] & products["validation"]),
        "product_simulation_test": len(products["simulation"] & products["test"]),
        "product_validation_test": len(products["validation"] & products["test"]),
        "text_simulation_validation": len(text_sets["simulation"] & text_sets["validation"]),
        "text_simulation_test": len(text_sets["simulation"] & text_sets["test"]),
        "text_validation_test": len(text_sets["validation"] & text_sets["test"]),
    }
    if any(leakage.values()):
        raise AssertionError(f"benchmark leakage detected: {leakage}")
    manifest = {
        "schema": "palda-human-benchmark-service-v2",
        "source": str(Path(source_path).resolve()),
        "seed": seed,
        "requested_ratios": {
            "simulation": simulation_ratio,
            "validation": validation_ratio,
            "test": 1 - simulation_ratio - validation_ratio,
        },
        "split_strategy": "seeded_product_and_exact_duplicate_components_balanced_by_rows",
        "eligibility": "human Gold, validationStatus=valid; unusable rows map to UNUSABLE",
        "counts": {name: len(splits.get(name, [])) for name in paths},
        "products": {name: len(products[name]) for name in paths},
        "record_types": {
            name: dict(Counter(row["record_type"] for row in splits.get(name, [])))
            for name in paths
        },
        "deduplicated_texts": duplicates,
        "excluded": dict(sorted(excluded.items())),
        "leakage": leakage,
        "files": {
            name: {"path": path.name, "sha256": sha256_file(path)}
            for name, path in paths.items()
        },
        "blocked_products": {
            "path": blocked_products.name,
            "count": len(split_by_product),
            "sha256": sha256_file(blocked_products),
        },
    }
    write_json(out / "manifest.json", manifest)
    return manifest


def _taxonomy_indexes(
    taxonomy: dict[str, Any],
) -> tuple[dict[tuple[str, str], list[str]], dict[str, list[str]]]:
    by_ids: dict[tuple[str, str], list[str]] = {}
    by_names: dict[str, list[str]] = {}
    for category in taxonomy.get("categories", []):
        allowed = sorted(
            item["canonicalThemeId"]
            for item in category.get("allowedThemes", [])
            if item.get("canonicalThemeId") and item.get("annotationReady", True)
        )
        by_ids[(str(category.get("level1Id")), str(category.get("level2Id")))] = allowed
        by_names[str(category.get("path") or "")] = allowed
    by_names.update(taxonomy.get("category_themes") or {})
    return by_ids, by_names


def _labels_for(row: dict[str, Any]) -> list[dict[str, Any]]:
    labels = row.get("labels") or {}
    if row.get("sourceType") == "review":
        return [
            {
                "theme_id": value["themeId"],
                "sentiment": value.get("sentiment"),
                "evidence": None,
                "role": None,
            }
            for value in labels.get("reviewLabels") or []
            if value.get("themeId")
        ]
    output: list[dict[str, Any]] = []
    primary = labels.get("qnaPrimaryTheme")
    if isinstance(primary, dict) and primary.get("themeId"):
        output.append({
            "theme_id": primary["themeId"], "sentiment": None,
            "evidence": None, "role": "primary",
        })
    for value in labels.get("qnaSecondaryThemes") or []:
        theme_id = value.get("themeId") if isinstance(value, dict) else value
        if theme_id and all(item["theme_id"] != theme_id for item in output):
            output.append({
                "theme_id": theme_id, "sentiment": None,
                "evidence": None, "role": "secondary",
            })
    return output


def _text_key(text: str) -> str:
    return hashlib.sha256("".join(str(text).split()).lower().encode("utf-8")).hexdigest()


def _product_components(rows: list[dict[str, Any]]) -> list[set[str]]:
    parent: dict[str, str] = {}

    def find(value: str) -> str:
        parent.setdefault(value, value)
        while parent[value] != value:
            parent[value] = parent[parent[value]]
            value = parent[value]
        return value

    def union(left: str, right: str) -> None:
        left_root, right_root = find(left), find(right)
        if left_root != right_root:
            parent[max(left_root, right_root)] = min(left_root, right_root)

    first_product_by_text: dict[str, str] = {}
    for row in rows:
        product_id = row["product_id"]
        find(product_id)
        previous = first_product_by_text.setdefault(_text_key(row["text"]), product_id)
        union(previous, product_id)
    groups: dict[str, set[str]] = defaultdict(set)
    for product_id in parent:
        groups[find(product_id)].add(product_id)
    return list(groups.values())


def _assign_components(
    components: list[set[str]],
    rows_by_product: Counter[str],
    seed: int,
    simulation_ratio: float,
    validation_ratio: float,
) -> dict[str, str]:
    ordered = sorted(
        components,
        key=lambda values: stable_hash("\x1f".join(sorted(values)), seed),
    )
    total = sum(rows_by_product.values())
    simulation_target = total * simulation_ratio
    validation_target = total * validation_ratio
    assigned_rows = Counter()
    result: dict[str, str] = {}
    for component in ordered:
        weight = sum(rows_by_product[value] for value in component)
        if assigned_rows["simulation"] < simulation_target:
            split = "simulation"
        elif assigned_rows["validation"] < validation_target:
            split = "validation"
        else:
            split = "test"
        assigned_rows[split] += weight
        for product_id in component:
            result[product_id] = split
    return result
