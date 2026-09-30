#!/usr/bin/env python
"""Build a clean Encoder baseline dataset and balanced provisional eval sets."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

from qna_review_model.encoder_data import EncoderTrainingRecord
from qna_review_model.encoder_review import load_review_rows, reviewed_record


LEXICAL_THEMES = {
    "PRICE": re.compile(r"가격|가성비|할인|쿠폰|적립|저렴|싸(?:다|게|요|서)?|싼|비싸|금액|비용|정가|특가|\d[\d,]*\s*원|값"),
    "DELIVERY": re.compile(r"배송|배달|택배|출고|도착|송장|받았|받아|받음|잘\s*받|왔(?:어요|습니다|네요|는데|고)?|와서|오는\s*데|오기\s*까지"),
    "PACKAGING": re.compile(r"포장|박스|완충|뽁뽁|포장재|비닐|찌그러|눌려서\s*도착"),
    "RETURN_EXCHANGE": re.compile(r"교환|반품|환불|회수|수거"),
    "CUSTOMER_SUPPORT": re.compile(r"문의|답변|상담|응대|고객\s*센터|연락|전화|판매자"),
    "INSTALLATION_SERVICE": re.compile(r"설치\s*기사|시공\s*기사|방문\s*설치|설치해\s*주|시공해\s*주"),
    "AFTER_SALES": re.compile(r"(?:^|\W)A/?S(?:\W|$)|에이에스|수리|보증"),
}


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--taxonomy", type=Path, required=True)
    parser.add_argument("--reviewed-csv", type=Path, nargs="*", default=[])
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--eval-per-split", type=int, default=500)
    parser.add_argument("--seed", type=int, default=20260831)
    return parser.parse_args()


def stable(seed: int, *values: object) -> str:
    value = "\n".join(map(str, (seed, *values)))
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def normalized_text(text: str) -> str:
    return re.sub(r"[^0-9a-z가-힣]+", "", text.lower())


def unsupported(theme_id: str, text: str) -> bool:
    pattern = LEXICAL_THEMES.get(theme_id)
    return bool(pattern and not pattern.search(text))


def clean_train_record(record: EncoderTrainingRecord) -> tuple[EncoderTrainingRecord | None, list[str]]:
    reasons = []
    if record.annotation.sourceSplit != "train":
        return None, ["not_source_train"]
    if not normalized_text(record.text):
        return None, ["no_lexical_content"]
    if record.annotation.reviewerStatus == "rejected":
        return None, ["reviewer_rejected"]
    if record.sourceType == "qna" and record.labels.qnaPrimaryTheme is None:
        return None, ["missing_qna_primary"]
    if record.annotation.grade == "gold":
        if record.annotation.reviewerStatus not in {"approved", "corrected"}:
            return None, ["gold_not_reviewed"]
        # Keyword absence is not evidence that an approved semantic label is wrong.
        # Preserve Gold targets; route lexical disagreements to an audit instead.
        labels = list(record.labels.reviewLabels) + list(record.labels.qnaSecondaryThemes)
        if record.labels.qnaPrimaryTheme is not None:
            labels.append(record.labels.qnaPrimaryTheme)
        flagged = sorted({label.themeId for label in labels if unsupported(label.themeId, record.text)})
        return record, [f"gold_lexical_audit_only:{theme}" for theme in flagged]
    if record.annotation.grade == "silver" and (
        record.annotation.validationStatus != "valid" or record.annotation.modelConfidence != "high"
    ):
        return None, ["silver_not_valid_high"]
    if record.sourceType == "review":
        kept = []
        removed = []
        for label in record.labels.reviewLabels:
            if unsupported(label.themeId, record.text):
                removed.append(label.themeId)
            else:
                kept.append(label)
        if removed:
            reasons.extend(f"unsupported_review_theme:{value}" for value in sorted(set(removed)))
        if removed and not kept:
            return None, [*reasons, "all_positive_targets_removed"]
        labels = record.labels.model_copy(update={"reviewLabels": kept})
        return record.model_copy(update={"labels": labels}), reasons
    primary = record.labels.qnaPrimaryTheme
    if primary is None:
        return None, ["missing_qna_primary"]
    if primary.themeId not in {"OTHER", "UNUSABLE"} and unsupported(primary.themeId, record.text):
        return None, [f"unsupported_qna_primary:{primary.themeId}"]
    kept_secondary = []
    for label in record.labels.qnaSecondaryThemes:
        if unsupported(label.themeId, record.text):
            reasons.append(f"unsupported_qna_secondary:{label.themeId}")
        else:
            kept_secondary.append(label)
    labels = record.labels.model_copy(update={"qnaSecondaryThemes": kept_secondary})
    return record.model_copy(update={"labels": labels}), reasons


def quota(total: int, names: list[str]) -> dict[str, int]:
    if not names:
        return {}
    base, remainder = divmod(total, len(names))
    return {name: base + (index < remainder) for index, name in enumerate(sorted(names))}


def diverse_pick(rows: list[EncoderTrainingRecord], count: int, seed: int) -> list[EncoderTrainingRecord]:
    """Balance Cat1 and task, then round-robin products within each bucket."""
    if len(rows) <= count:
        return list(rows)
    cat1_names = sorted({row.category.level1Name for row in rows})
    type_total = {"review": round(count * 0.65)}
    type_total["qna"] = count - type_total["review"]
    selected: list[EncoderTrainingRecord] = []
    selected_ids: set[str] = set()
    for source_type in ("review", "qna"):
        available_cat1 = [
            name for name in cat1_names
            if any(row.sourceType == source_type and row.category.level1Name == name for row in rows)
        ]
        targets = quota(type_total[source_type], available_cat1)
        for cat1, target in targets.items():
            candidates = [
                row for row in rows
                if row.sourceType == source_type and row.category.level1Name == cat1
            ]
            by_product: dict[str, list[EncoderTrainingRecord]] = defaultdict(list)
            for row in candidates:
                by_product[row.productGroupId].append(row)
            for values in by_product.values():
                values.sort(key=lambda row: stable(seed, row.category.level2Id, row.itemId))
            product_ids = sorted(by_product, key=lambda value: stable(seed, cat1, source_type, value))
            bucket: list[EncoderTrainingRecord] = []
            depth = 0
            while len(bucket) < target:
                added = False
                for product_id in product_ids:
                    values = by_product[product_id]
                    if depth < len(values):
                        bucket.append(values[depth]); added = True
                        if len(bucket) == target:
                            break
                if not added:
                    break
                depth += 1
            for row in bucket:
                if row.itemId not in selected_ids:
                    selected.append(row); selected_ids.add(row.itemId)
    if len(selected) < count:
        leftovers = sorted(
            (row for row in rows if row.itemId not in selected_ids),
            key=lambda row: stable(seed, "leftover", row.itemId),
        )
        selected.extend(leftovers[:count - len(selected)])
    selected = selected[:count]
    selected_ids = {row.itemId for row in selected}
    special_qna = [
        row for row in rows
        if row.sourceType == "qna"
        and row.labels.qnaPrimaryTheme is not None
        and row.labels.qnaPrimaryTheme.themeId in {"OTHER", "UNUSABLE"}
        and row.itemId not in selected_ids
    ]
    for special in sorted(special_qna, key=lambda row: stable(seed, "special", row.itemId)):
        replacement = next((
            index for index in range(len(selected) - 1, -1, -1)
            if selected[index].sourceType == "qna"
            and selected[index].category.level1Name == special.category.level1Name
            and selected[index].labels.qnaPrimaryTheme is not None
            and selected[index].labels.qnaPrimaryTheme.themeId not in {"OTHER", "UNUSABLE"}
        ), None)
        if replacement is None:
            replacement = next((
                index for index in range(len(selected) - 1, -1, -1)
                if selected[index].sourceType == "qna"
                and selected[index].labels.qnaPrimaryTheme is not None
                and selected[index].labels.qnaPrimaryTheme.themeId not in {"OTHER", "UNUSABLE"}
            ), None)
        if replacement is not None:
            selected_ids.discard(selected[replacement].itemId)
            selected[replacement] = special
            selected_ids.add(special.itemId)
    return selected


def gold_first_pick(rows: list[EncoderTrainingRecord], count: int, seed: int) -> list[EncoderTrainingRecord]:
    """Use reviewed Gold evaluation rows first, filling any shortage with Silver."""
    gold = [row for row in rows if row.annotation.grade == "gold"]
    if len(gold) >= count:
        return diverse_pick(gold, count, seed)
    chosen = list(gold)
    chosen_ids = {row.itemId for row in chosen}
    silver = [row for row in rows if row.itemId not in chosen_ids]
    chosen.extend(diverse_pick(silver, count - len(chosen), seed))
    return chosen


def write_jsonl(path: Path, rows: Iterable[EncoderTrainingRecord]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(row.model_dump_json() + "\n")


def main() -> None:
    args = arguments()
    records = [
        EncoderTrainingRecord.model_validate_json(line)
        for line in args.dataset.read_text(encoding="utf-8").splitlines() if line.strip()
    ]
    taxonomy = json.loads(args.taxonomy.read_text(encoding="utf-8"))
    allowed_by_category = {
        row["level2Id"]: {theme["canonicalThemeId"]: theme for theme in row["allowedThemes"]}
        for row in taxonomy["categories"]
    }
    reviewed_rows, reviewed_load_errors = load_review_rows(args.reviewed_csv)
    overlay_errors = list(reviewed_load_errors)
    approved_overlay = 0
    overlaid: list[EncoderTrainingRecord] = []
    for record in records:
        reviewed = reviewed_rows.get(record.itemId)
        status = str((reviewed or {}).get("reviewer_status") or "pending").lower()
        if reviewed and status in {"approved", "corrected"}:
            try:
                record = reviewed_record(record, reviewed, allowed_by_category[record.category.level2Id])
                approved_overlay += 1
            except Exception as exc:
                overlay_errors.append({"annotation_id": record.itemId, "error": str(exc)})
        overlaid.append(record)

    clean_train = []
    audit_rows = []
    duplicate_count = 0
    seen_text: set[str] = set()
    audit_counts = Counter()
    for record in overlaid:
        cleaned, reasons = clean_train_record(record)
        if cleaned is None:
            if record.annotation.sourceSplit == "train":
                for reason in reasons:
                    audit_counts[reason] += 1
                audit_rows.append({"itemId": record.itemId, "action": "excluded", "reasons": reasons})
            continue
        key = normalized_text(cleaned.text)
        if key and key in seen_text:
            duplicate_count += 1
            audit_counts["duplicate_text"] += 1
            audit_rows.append({"itemId": record.itemId, "action": "excluded", "reasons": ["duplicate_text"]})
            continue
        if key:
            seen_text.add(key)
        if reasons:
            for reason in reasons:
                audit_counts[reason] += 1
            action = "audit_only" if record.annotation.grade == "gold" else "labels_removed"
            audit_rows.append({"itemId": record.itemId, "action": action, "reasons": reasons})
        clean_train.append(cleaned)

    provisional = {}
    for split in ("validation", "test"):
        pool = [record for record in overlaid if record.annotation.sourceSplit == split]
        provisional[split] = gold_first_pick(pool, min(args.eval_per_split, len(pool)), args.seed)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(args.output_dir / "train.jsonl", clean_train)
    write_jsonl(args.output_dir / "validation.jsonl", provisional["validation"])
    write_jsonl(args.output_dir / "test.jsonl", provisional["test"])
    write_jsonl(args.output_dir / "eval-review-candidates.jsonl", [*provisional["validation"], *provisional["test"]])
    (args.output_dir / "theme_taxonomy.json").write_text(
        json.dumps(taxonomy, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    ready_themes = sorted({
        theme["categoryThemeId"]
        for category in taxonomy["categories"]
        for theme in category["allowedThemes"]
        if theme["annotationReady"]
    })
    theme_index = {value: index for index, value in enumerate(ready_themes)}
    metadata = {}
    category_masks = {}
    for category in taxonomy["categories"]:
        indices = []
        for theme in category["allowedThemes"]:
            category_theme_id = theme["categoryThemeId"]
            if not theme["annotationReady"] or category_theme_id not in theme_index:
                continue
            indices.append(theme_index[category_theme_id])
            metadata[category_theme_id] = {
                "canonicalThemeId": theme["canonicalThemeId"],
                "level2Id": category["level2Id"],
                "level2Name": category["level2Name"],
                "reviewLabel": theme["reviewLabel"],
                "qnaLabel": theme["qnaLabel"],
            }
        category_masks[category["level2Id"]] = indices
    label_map = {
        "schemaVersion": "encoder-label-map-v1",
        "taxonomyVersion": taxonomy["taxonomyVersion"],
        "sentiments": ["positive", "negative", "neutral"],
        "categoryThemeToIndex": theme_index,
        "indexToCategoryTheme": ready_themes,
        "categoryThemeMetadata": metadata,
        "categoryMasks": category_masks,
        "reviewTargetEncoding": "flat_index = categoryThemeIndex * 3 + sentimentIndex",
        "qnaPrimaryTargetEncoding": "categoryThemeIndex with category mask",
        "qnaSecondaryTargetEncoding": "multi-hot categoryThemeIndex with category mask",
    }
    (args.output_dir / "label_map.json").write_text(
        json.dumps(label_map, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    with (args.output_dir / "train-audit.jsonl").open("w", encoding="utf-8") as handle:
        for row in audit_rows:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
    with (args.output_dir / "overlay-errors.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["annotation_id", "error"])
        writer.writeheader(); writer.writerows(overlay_errors)

    split_products = {
        "train": {row.productGroupId for row in clean_train},
        "validation": {row.productGroupId for row in provisional["validation"]},
        "test": {row.productGroupId for row in provisional["test"]},
    }
    leakage = {
        "train_validation": len(split_products["train"] & split_products["validation"]),
        "train_test": len(split_products["train"] & split_products["test"]),
        "validation_test": len(split_products["validation"] & split_products["test"]),
    }
    if any(leakage.values()):
        raise ValueError(f"product leakage detected: {leakage}")
    manifest = {
        "schemaVersion": "encoder-experiment-dataset-v1",
        "taxonomyVersion": taxonomy["taxonomyVersion"],
        "sourceDataset": str(args.dataset),
        "reviewedCsv": [str(path) for path in args.reviewed_csv],
        "seed": args.seed,
        "trainPolicy": "sourceSplit=train; Silver requires valid+high; approved human overlay allowed; lexical audit; exact-text dedupe",
        "evaluationPolicy": "Human-reviewed Gold first; fill any shortage with provisional Silver.",
        "records": {
            "train": len(clean_train),
            "validation": len(provisional["validation"]),
            "test": len(provisional["test"]),
        },
        "trainGrade": dict(Counter(row.annotation.grade for row in clean_train)),
        "trainRecordType": dict(Counter(row.sourceType for row in clean_train)),
        "evalRecordType": {
            split: dict(Counter(row.sourceType for row in values)) for split, values in provisional.items()
        },
        "evalGrade": {
            split: dict(Counter(row.annotation.grade for row in rows))
            for split, rows in provisional.items()
        },
        "evalCat1": {
            split: dict(Counter(row.category.level1Name for row in values)) for split, values in provisional.items()
        },
        "approvedOverlay": approved_overlay,
        "overlayErrors": len(overlay_errors),
        "duplicateTextsRemoved": duplicate_count,
        "auditCounts": dict(audit_counts.most_common()),
        "products": {split: len(values) for split, values in split_products.items()},
        "productSplitLeakage": leakage,
        "warnings": ([
            "Evaluation still contains provisional Silver labels; report Gold-only metrics for final model quality."
        ] if any(row.annotation.grade == "silver" for rows in provisional.values() for row in rows) else []),
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
