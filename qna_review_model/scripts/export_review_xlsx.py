#!/usr/bin/env python
"""Create a compact, styled Excel workbook for human annotation review."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from openpyxl import Workbook
from openpyxl.formatting.rule import FormulaRule
from openpyxl.styles import Alignment, Font, PatternFill
from openpyxl.worksheet.datavalidation import DataValidation


HEADERS = [
    "reviewer_status", "검증상태", "유형", "카테고리", "상품명", "원문",
    "Q&A 단계", "모델 초안 (Theme / 감성 / 근거)", "오류", "검수 메모",
    "별점", "split", "annotation_id",
]


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--review-csv", type=Path, required=True)
    parser.add_argument("--themes", type=Path, default=Path("config/category_themes.json"))
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def theme_labels(path: Path) -> dict[str, str]:
    payload = json.loads(path.read_text(encoding="utf-8")); result: dict[str, str] = {}
    groups = [payload.get("global", [])]
    for level in ("cat1", "cat2", "leaf"):
        groups.extend(payload.get(level, {}).values())
    for rows in groups:
        for row in rows:
            result[str(row["theme_id"])] = str(row["label"])
    return result


def summary(row: dict[str, str], labels: dict[str, str]) -> str:
    parts = []
    for index in range(1, 4):
        theme_id = (row.get(f"theme_{index}") or "").strip()
        if not theme_id:
            continue
        label = labels.get(theme_id, theme_id)
        polarity = row.get(f"polarity_{index}") or ""
        evidence = row.get(f"evidence_{index}") or ""
        parts.append(f"{index}. {label} ({theme_id}) · {polarity}\n   근거: {evidence}")
    if row.get("new_theme_candidates"):
        parts.append(f"신규 Theme 후보: {row['new_theme_candidates']}")
    return "\n".join(parts) if parts else "Theme 없음"


def add_sheet(workbook: Workbook, title: str, rows: list[dict[str, str]], labels: dict[str, str]) -> None:
    sheet = workbook.create_sheet(title)
    sheet.append(HEADERS)
    header_fill = PatternFill("solid", fgColor="1F4E78")
    for cell in sheet[1]:
        cell.fill = header_fill; cell.font = Font(color="FFFFFF", bold=True)
        cell.alignment = Alignment(horizontal="center", vertical="center")
    for row in rows:
        sheet.append([
            row.get("reviewer_status"), row.get("validation_status"), row.get("record_type"),
            row.get("category_path"), row.get("product_name"), row.get("text"), row.get("qna_stage"),
            summary(row, labels), row.get("validation_errors"), row.get("reviewer_notes"),
            row.get("score"), row.get("split"), row.get("annotation_id"),
        ])
    sheet.freeze_panes = "A2"; sheet.auto_filter.ref = sheet.dimensions
    sheet.sheet_view.showGridLines = False
    widths = {"A": 18, "B": 14, "C": 11, "D": 35, "E": 32, "F": 75, "G": 22, "H": 70, "I": 45, "J": 30, "K": 9, "L": 12, "M": 28}
    for column, width in widths.items(): sheet.column_dimensions[column].width = width
    for row_index in range(2, sheet.max_row + 1):
        sheet.row_dimensions[row_index].height = 105
        for cell in sheet[row_index]:
            cell.alignment = Alignment(wrap_text=True, vertical="top")
    status_validation = DataValidation(type="list", formula1='"pending,approved,corrected,rejected"', allow_blank=False)
    stage_validation = DataValidation(type="list", formula1='"pre_purchase,post_purchase_support,irrelevant"', allow_blank=True)
    sheet.add_data_validation(status_validation); sheet.add_data_validation(stage_validation)
    if sheet.max_row >= 2:
        status_validation.add(f"A2:A{sheet.max_row}"); stage_validation.add(f"G2:G{sheet.max_row}")
        sheet.conditional_formatting.add(f"A2:A{sheet.max_row}", FormulaRule(formula=["$A2=\"approved\""], fill=PatternFill("solid", fgColor="C6EFCE")))
        sheet.conditional_formatting.add(f"A2:A{sheet.max_row}", FormulaRule(formula=["$A2=\"pending\""], fill=PatternFill("solid", fgColor="FFF2CC")))
        sheet.conditional_formatting.add(f"B2:B{sheet.max_row}", FormulaRule(formula=["$B2=\"needs_review\""], fill=PatternFill("solid", fgColor="F4CCCC")))
    sheet.auto_filter.ref = f"A1:M{sheet.max_row}"


def main() -> None:
    args = arguments(); labels = theme_labels(args.themes)
    with args.review_csv.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    pending = [row for row in rows if row.get("reviewer_status") == "pending"]
    approved = [row for row in rows if row.get("reviewer_status") != "pending"]
    workbook = Workbook(); workbook.remove(workbook.active)
    add_sheet(workbook, f"미검수_{len(pending)}", pending, labels)
    add_sheet(workbook, f"검수완료_{len(approved)}", approved, labels)
    args.output.parent.mkdir(parents=True, exist_ok=True); workbook.save(args.output)
    print(json.dumps({"output": str(args.output), "pending": len(pending), "reviewed": len(approved)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
