#!/usr/bin/env python3
"""Private, metrics-only W&B view of the input-size pilot; no source text upload."""
import argparse
import json
from pathlib import Path
import time

from input_size_tradeoff import DEFAULT_OUT, read
from wandb_quality_sync import settings, verify_destination


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--watch", action="store_true")
    args = parser.parse_args()
    import wandb
    entity, project = "yein0330-soganag-university", "PALDA"
    api = wandb.Api(timeout=20)
    verify_destination(api, entity, project, "yein0330")
    folder = args.output / "wandb"
    folder.mkdir(exist_ok=True)
    method_file = args.output / "method.json"
    method = read(method_file)["method"] if method_file.exists() else None
    run_id = args.output.name + "-20260922" if method else args.output.name
    run = wandb.init(entity=entity, project=project, id=run_id,
                     name=(f"분류 방식 비교 · {method}" if method else "입력 크기 × 근거 수 · 시간/품질 탐색"),
                     group="architecture-tradeoff-20260922" if method else "input-size-tradeoff",
                     resume="allow", dir=str(folder), settings=settings(wandb),
                     config={"split": "validation-product-expansion", "contains_source_text": False,
                             "quality_caveat": "anchor coverage and consistency are not semantic accuracy",
                             "independent_test": False, "production_approved": False,
                             **({"classification_method": method,
                                 "encoder_stage_is_total_classification": True} if method else {})})
    write_path = args.output / "wandb-link.json"
    write_path.write_text(json.dumps({"url": run.url, "sourceTextUploaded": False}))
    seen = set()
    try:
        while True:
            records = [read(p) for p in sorted((args.output / "cases").glob("*.json"))]
            completed = [r for r in records if r.get("status") == "completed"]
            for r in completed:
                if r["caseId"] in seen:
                    continue
                event = {"input/rows": r["size"], "input/evidence_per_theme": r["evidenceBudget"],
                         "latency/seconds": r["seconds"], "progress/completed": len(seen) + 1,
                         **{f"memory/{k}": v for k, v in r["memory"].items()},
                         **{f"stage/{k}_seconds": v for k, v in r["stages"].items()},
                         **{f"tokens/{k}": v for k, v in r["tokens"].items()},
                         **{f"quality/{k}": v for k, v in r["quality"].items() if isinstance(v, (int, float))}}
                run.log(event)
                seen.add(r["caseId"])
            if completed:
                rows = [[r["caseId"], r["category"], r["size"], r["evidenceBudget"], r["repeat"],
                         r["seconds"], r["quality"]["gold_anchor_theme_coverage"],
                         r["memory"]["gpu_allocated_gib"], r["memory"]["rss_peak_gib"]] for r in completed]
                table = wandb.Table(columns=["case", "category", "input_rows", "evidence_per_theme", "repeat", "seconds",
                                           "anchor_coverage", "gpu_allocated_gib", "rss_peak_gib"], data=rows)
                run.log({"comparison": table,
                         "tradeoff/time_vs_anchor_coverage": wandb.plot.scatter(table, "seconds", "anchor_coverage",
                                           title="시간 × 일부 사람 정답 테마 포착률 (의미 품질 아님)"),
                         "tradeoff/input_vs_time": wandb.plot.scatter(table, "input_rows", "seconds",
                                           title="상품 입력 건수 × 추론 시간")})
            status = read(args.output / "status.json")
            judged = {p.stem: read(p) for p in (args.output / "judge").glob("p_*.json")}
            ratings = []
            dimensions = ["summary_faithfulness", "theme_alignment", "selling_groundedness", "selling_usefulness"]
            for r in completed:
                j = judged.get(r["caseId"], {})
                if j.get("status") == "completed":
                    ratings.append([r["caseId"], r["category"], r["size"], r["evidenceBudget"], r["seconds"],
                                    *[j["ratings"][k] for k in dimensions]])
            if ratings:
                table = wandb.Table(columns=["case", "category", "input_rows", "evidence_per_theme", "seconds", *dimensions], data=ratings)
                charts = {"llm_judge/ratings": table}
                for dimension in dimensions:
                    charts[f"llm_judge/time_vs_{dimension}"] = wandb.plot.scatter(table, "seconds", dimension,
                        title=f"시간 × Kanana 자동 평가 {dimension} (사람 정답 아님)")
                run.log(charts)
            run.summary.update({"stage": status["stage"], "completed": len(completed),
                                "planned": status.get("planned"), "human_report_quality_evaluated": False,
                                "llm_judge_rated_cases": len(ratings)})
            print(json.dumps({"completed": len(completed), "stage": status["stage"], "url": run.url}), flush=True)
            if not args.watch or status["stage"] in ("completed", "failed"):
                break
            time.sleep(30)
    finally:
        run.finish()


if __name__ == "__main__":
    main()
