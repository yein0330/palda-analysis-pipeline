"""Build isolated Train-only ablations; byte-copy Validation; never access Test."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil


def build(data_dir, output_dir, mode, source=None):
    if output_dir.exists():
        raise FileExistsError(output_dir)
    rows = [json.loads(x) for x in (data_dir / "train.jsonl").open() if x.strip()]
    changes = []
    if mode in ("price-inflection", "gold-preserve", "gold-review-preserve"):
        if source is None:
            raise ValueError(f"{mode} requires pre-cleaning source")
        ids = {r["itemId"] for r in rows}
        originals = {}
        with source.open() as handle:
            for line in handle:
                r = json.loads(line)
                if r["itemId"] in ids:
                    if r["itemId"] in originals:
                        raise ValueError("Duplicate source Train ID")
                    originals[r["itemId"]] = r
        label_map = json.loads((data_dir / "label_map.json").read_text())
        for row in rows:
            if mode in ("gold-preserve", "gold-review-preserve"):
                if mode == "gold-review-preserve" and row["sourceType"] != "review":
                    continue
                original = originals.get(row["itemId"])
                if (row["annotation"]["grade"] != "gold" or not original
                    or original["annotation"]["grade"] != "gold"
                    or original["annotation"].get("reviewerStatus") not in ("approved", "corrected")):
                    continue
                if (original["text"] != row["text"] or original["category"] != row["category"]
                    or original["sourceType"] != row["sourceType"]):
                    raise ValueError("Gold source content mismatch")
                keys = ("reviewLabels",) if row["sourceType"] == "review" else ("qnaSecondaryThemes",)
                # Restore only removed labels. Primary labels, examples and annotations stay fixed.
                for key in keys:
                    existing = {(x["categoryThemeId"], x.get("sentiment")) for x in row["labels"][key]}
                    for label in original["labels"][key]:
                        identity = (label["categoryThemeId"], label.get("sentiment"))
                        index = label_map["categoryThemeToIndex"].get(label["categoryThemeId"])
                        if index is None or index not in label_map["categoryMasks"][row["category"]["level2Id"]]:
                            raise ValueError("Gold source label is not allowed for category")
                        if identity not in existing:
                            row["labels"][key].append(label)
                            existing.add(identity)
                            changes.append({"itemId": row["itemId"], "field": key, "restored": label,
                                            "reason": "preserve_existing_approved_gold_despite_lexical_filter"})
                continue
            if row["sourceType"] != "review" or "싼" not in row["text"]:
                continue
            original = originals.get(row["itemId"])
            if not original or original["text"] != row["text"]:
                continue
            existing = {(x["categoryThemeId"], x["sentiment"]) for x in row["labels"]["reviewLabels"]}
            for label in original["labels"]["reviewLabels"]:
                index = label_map["categoryThemeToIndex"].get(label["categoryThemeId"])
                if (label["themeId"] == "PRICE" and (label["categoryThemeId"], label["sentiment"]) not in existing
                    and index in label_map["categoryMasks"][row["category"]["level2Id"]]):
                    row["labels"]["reviewLabels"].append(label)
                    existing.add((label["categoryThemeId"], label["sentiment"]))
                    changes.append({"itemId": row["itemId"], "restored": label, "reason": "original_price_label_with_explicit_싼"})
    elif mode == "silver-weight-03":
        for row in rows:
            if row["annotation"]["grade"] == "silver":
                old = row["annotation"]["sampleWeight"]
                row["annotation"]["sampleWeight"] = 0.3
                changes.append({"itemId": row["itemId"], "oldWeight": old, "newWeight": 0.3})
    else:
        raise ValueError(mode)
    output_dir.mkdir(parents=True)
    for name in ("validation.jsonl", "label_map.json", "theme_taxonomy.json"):
        shutil.copyfile(data_dir / name, output_dir / name)
    with (output_dir / "train.jsonl").open("w") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    with (output_dir / "changes.jsonl").open("w") as handle:
        for row in changes:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    manifest = {"mode": mode, "parent": str(data_dir), "trainRows": len(rows), "changes": len(changes),
        "source": str(source) if source else None,
        "sourceSha256": hashlib.sha256(source.read_bytes()).hexdigest() if source else None,
        "parentHashes": {name: hashlib.sha256((data_dir / name).read_bytes()).hexdigest()
                         for name in ("train.jsonl", "validation.jsonl", "label_map.json", "theme_taxonomy.json")},
        "testPolicy": "Not read or copied; train with --skip-test",
        "warning": "Isolated preprocessing/weight ablation; source provenance retained, not a fresh Gold annotation. Restoration does not guarantee semantic correctness."}
    (output_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    return manifest


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-dir", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--mode", choices=["price-inflection", "silver-weight-03", "gold-preserve", "gold-review-preserve"], required=True)
    p.add_argument("--source", type=Path)
    a = p.parse_args()
    print(json.dumps(build(a.data_dir, a.output_dir, a.mode, a.source), ensure_ascii=False, indent=2))
