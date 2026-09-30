from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

from .analysis import add_category_relative_insights, analyze_product
from .crawler import iter_records, load_products, validate_completed_run, write_normalized
from .profile import write_profile
from .pilot import AnnotationSample, build_multicategory_pilot, build_pilot
from .training_data import build_bootstrap_dataset
from .annotation import (
    AnnotationQueueRow, DraftAnnotationRow, GoldAnnotationRow, bulk_set_review_status,
    build_annotation_queue, export_review_csv, import_review_csv, merge_draft_shards,
    write_draft_report,
)
from .service_annotation import ServiceDraftRow, merge_service_shards, write_service_report
from .schemas import (
    CrossStageInsight, ProductAnalysis, ProductContext, QNARecord, ReviewRecord,
    SellerAction, ThemeDefinition, ThemeEvidence, ThemeSignal,
)
from .themes import ThemeDictionary
from .paths import DATA_ROOT

DEFAULT_RUN = DATA_ROOT / "raw/crawling/2026-08-14-full-5090-w12-sqlite"
DEFAULT_THEMES = Path("config/category_themes.json")


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(description="PALDA category-aware Q&A–Review pipeline")
    commands = root.add_subparsers(dest="command", required=True)
    validate = commands.add_parser("validate-run", help="validate crawler completion and files")
    validate.add_argument("--run-dir", type=Path, default=DEFAULT_RUN)
    profile = commands.add_parser("profile", help="create the Phase-1 data profile")
    profile.add_argument("--run-dir", type=Path, default=DEFAULT_RUN)
    profile.add_argument("--output", type=Path, default=Path("models/reports/data_profile.json"))
    profile.add_argument("--max-pages", type=int, help="sample pages; omit for the complete run")
    normalize = commands.add_parser("normalize", help="flatten crawler pages into safe canonical records")
    normalize.add_argument("--run-dir", type=Path, default=DEFAULT_RUN)
    normalize.add_argument("--kind", choices=["review", "qna", "all"], default="all")
    normalize.add_argument("--output", type=Path, default=DATA_ROOT / "interim/normalized/records.jsonl")
    normalize.add_argument("--max-pages", type=int)
    normalize.add_argument("--max-records", type=int)
    themes = commands.add_parser("themes", help="validate the hierarchical theme dictionary")
    themes.add_argument("--themes", type=Path, default=DEFAULT_THEMES)
    analyze = commands.add_parser("analyze", help="run the deterministic category-aware baseline")
    analyze.add_argument("--run-dir", type=Path, default=DEFAULT_RUN)
    analyze.add_argument("--themes", type=Path, default=DEFAULT_THEMES)
    analyze.add_argument("--output", type=Path, default=DATA_ROOT / "reports/analysis/product_analysis.jsonl")
    analyze.add_argument("--max-pages", type=int)
    analyze.add_argument("--max-records", type=int, default=100_000)
    analyze.add_argument("--require-both", action="store_true")
    schemas = commands.add_parser("schemas", help="export current JSON Schemas")
    schemas.add_argument("--output-dir", type=Path, default=Path("schemas"))
    pilot = commands.add_parser("pilot", help="build a product-level annotation pilot")
    pilot.add_argument("--run-dir", type=Path, default=DEFAULT_RUN)
    pilot.add_argument("--themes", type=Path, default=DEFAULT_THEMES)
    pilot.add_argument("--output-dir", type=Path, default=DATA_ROOT / "interim/pilots/home-interior-200")
    pilot.add_argument("--category", default="가구/인테리어")
    pilot.add_argument("--size", type=int, default=200)
    pilot.add_argument("--max-reviews", type=int, default=50)
    pilot.add_argument("--max-qna", type=int, default=30)
    pilot.add_argument("--min-reviews", type=int, default=10)
    pilot.add_argument("--min-qna", type=int, default=3)
    pilot.add_argument("--seed", type=int, default=20260826)
    multi_pilot = commands.add_parser("multi-pilot", help="build a balanced multi-Cat1 annotation pilot")
    multi_pilot.add_argument("--run-dir", type=Path, default=DEFAULT_RUN)
    multi_pilot.add_argument("--themes", type=Path, default=DEFAULT_THEMES)
    multi_pilot.add_argument("--output-dir", type=Path, default=DATA_ROOT / "interim/pilots/multicategory-1000")
    multi_pilot.add_argument("--categories", nargs="+", required=True)
    multi_pilot.add_argument("--size-per-category", type=int, default=100)
    multi_pilot.add_argument("--max-reviews", type=int, default=50)
    multi_pilot.add_argument("--max-qna", type=int, default=30)
    multi_pilot.add_argument("--min-reviews", type=int, default=10)
    multi_pilot.add_argument("--min-qna", type=int, default=3)
    multi_pilot.add_argument("--seed", type=int, default=20260827)
    bootstrap = commands.add_parser("bootstrap-sft", help="create weak record-level targets for a QLoRA smoke test")
    bootstrap.add_argument("--pilot-dir", type=Path, default=DATA_ROOT / "interim/pilots/home-interior-200")
    bootstrap.add_argument("--output-dir", type=Path, default=DATA_ROOT / "archive/training/bootstrap-keyword-v1")
    queue = commands.add_parser("annotation-queue", help="build a human-review Gold annotation queue")
    queue.add_argument("--pilot-dir", type=Path, default=DATA_ROOT / "interim/pilots/home-interior-200")
    queue.add_argument("--output", type=Path, default=DATA_ROOT / "interim/annotations/home-interior-draft-3000/queue.jsonl")
    queue.add_argument("--records-per-product", type=int, default=15)
    queue.add_argument("--qna-per-product", type=int, default=5)
    queue.add_argument("--seed", type=int, default=20260826)
    merge = commands.add_parser("merge-drafts", help="merge model-generated annotation shards")
    merge.add_argument("--queue", type=Path, required=True)
    merge.add_argument("--shards", type=Path, nargs="+", required=True)
    merge.add_argument("--output", type=Path, required=True)
    review_csv = commands.add_parser("review-csv", help="export draft annotations for spreadsheet review")
    review_csv.add_argument("--drafts", type=Path, required=True)
    review_csv.add_argument("--output", type=Path, required=True)
    review_csv.add_argument("--status", choices=["valid", "needs_review"])
    review_csv.add_argument("--limit", type=int)
    draft_report = commands.add_parser("draft-report", help="profile model-generated draft annotations")
    draft_report.add_argument("--drafts", type=Path, required=True)
    draft_report.add_argument("--output", type=Path, required=True)
    import_review = commands.add_parser("import-review-csv", help="validate reviewed CSV and create human-approved Gold JSONL")
    import_review.add_argument("--drafts", type=Path, required=True)
    import_review.add_argument("--review-csv", type=Path, required=True)
    import_review.add_argument("--output", type=Path, required=True)
    import_review.add_argument("--errors-output", type=Path, required=True)
    bulk_status = commands.add_parser("bulk-review-status", help="set reviewer status for IDs listed in another review CSV")
    bulk_status.add_argument("--review-csv", type=Path, required=True)
    bulk_status.add_argument("--ids-csv", type=Path, required=True)
    bulk_status.add_argument("--status", choices=["approved", "rejected"], required=True)
    merge_service = commands.add_parser("merge-service-drafts", help="merge service annotation shards")
    merge_service.add_argument("--queue", type=Path, required=True)
    merge_service.add_argument("--shards", type=Path, nargs="+", required=True)
    merge_service.add_argument("--output", type=Path, required=True)
    service_report = commands.add_parser("service-report", help="profile service annotation drafts")
    service_report.add_argument("--drafts", type=Path, required=True)
    service_report.add_argument("--output", type=Path, required=True)
    return root


def _analyze(args: argparse.Namespace) -> dict[str, int | str]:
    validate_completed_run(args.run_dir)
    products = load_products(args.run_dir)
    dictionary = ThemeDictionary.from_path(args.themes)
    grouped: dict[str, list[ReviewRecord | QNARecord]] = defaultdict(list)
    contexts: dict[str, ProductContext] = {}
    for record in iter_records(args.run_dir, "all", args.max_pages, args.max_records):
        context = products.get(record.product_id)
        if context is None:
            continue
        grouped[context.product_id].append(record)
        contexts[context.product_id] = context
    analyses: list[ProductAnalysis] = []
    for product_id, records in grouped.items():
        if args.require_both and {record.record_type for record in records} != {"review", "qna"}:
            continue
        context = contexts[product_id]
        analyses.append(analyze_product(context, dictionary.resolve(context), records))
    add_category_relative_insights(analyses)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        for analysis in analyses:
            handle.write(analysis.model_dump_json() + "\n")
    return {"products": len(analyses), "output": str(args.output)}


def main() -> None:
    args = parser().parse_args()
    if args.command == "validate-run":
        progress = validate_completed_run(args.run_dir)
        result = {"status": progress["status"], "queue_drained": progress["completion"]["queueDrained"], "jobs": progress["consumer"]["jobs"]}
    elif args.command == "profile":
        result = write_profile(args.run_dir, args.output, args.max_pages)
    elif args.command == "normalize":
        result = write_normalized(args.run_dir, args.output, args.kind, args.max_pages, args.max_records)
    elif args.command == "themes":
        result = ThemeDictionary.from_path(args.themes).validate()
    elif args.command == "analyze":
        result = _analyze(args)
    elif args.command == "pilot":
        result = build_pilot(
            args.run_dir, args.themes, args.output_dir, args.category, args.size,
            args.max_reviews, args.max_qna, args.min_reviews, args.min_qna, args.seed,
        )
    elif args.command == "multi-pilot":
        result = build_multicategory_pilot(
            args.run_dir, args.themes, args.output_dir, args.categories, args.size_per_category,
            args.max_reviews, args.max_qna, args.min_reviews, args.min_qna, args.seed,
        )
    elif args.command == "bootstrap-sft":
        result = build_bootstrap_dataset(args.pilot_dir, args.output_dir)
    elif args.command == "annotation-queue":
        result = build_annotation_queue(args.pilot_dir, args.output, args.records_per_product, args.qna_per_product, args.seed)
    elif args.command == "merge-drafts":
        result = merge_draft_shards(args.queue, args.shards, args.output)
    elif args.command == "review-csv":
        result = export_review_csv(args.drafts, args.output, args.status, args.limit)
    elif args.command == "draft-report":
        result = write_draft_report(args.drafts, args.output)
    elif args.command == "import-review-csv":
        result = import_review_csv(args.drafts, args.review_csv, args.output, args.errors_output)
    elif args.command == "bulk-review-status":
        result = bulk_set_review_status(args.review_csv, args.ids_csv, args.status)
    elif args.command == "merge-service-drafts":
        result = merge_service_shards(args.queue, args.shards, args.output)
    elif args.command == "service-report":
        result = write_service_report(args.drafts, args.output)
    else:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        models = {
            "product_context": ProductContext, "review_record": ReviewRecord,
            "qna_record": QNARecord, "theme_definition": ThemeDefinition,
            "theme_evidence": ThemeEvidence, "theme_signal": ThemeSignal,
            "cross_stage_insight": CrossStageInsight, "seller_action": SellerAction,
            "product_analysis": ProductAnalysis, "annotation_sample": AnnotationSample,
            "annotation_queue_row": AnnotationQueueRow,
            "draft_annotation_row": DraftAnnotationRow,
            "gold_annotation_row": GoldAnnotationRow,
            "service_draft_row": ServiceDraftRow,
        }
        for name, model in models.items():
            (args.output_dir / f"{name}.schema.json").write_text(
                json.dumps(model.model_json_schema(), ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
            )
        result = {"schemas": len(models), "output_dir": str(args.output_dir)}
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
