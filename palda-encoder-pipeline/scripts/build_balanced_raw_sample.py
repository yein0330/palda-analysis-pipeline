#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json

from palda_encoder.raw_sample import build_balanced_raw_sample


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a category-balanced raw crawl sample")
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--blocked-products")
    parser.add_argument("--target-records", type=int, default=15_000)
    parser.add_argument("--products-per-leaf", type=int, default=3)
    parser.add_argument("--reviews-per-product", type=int, default=3)
    parser.add_argument("--qna-per-product", type=int, default=2)
    parser.add_argument("--seed", type=int, default=20260904)
    args = parser.parse_args()
    report = build_balanced_raw_sample(
        args.run_dir,
        args.output,
        blocked_products_path=args.blocked_products,
        target_records=args.target_records,
        products_per_leaf=args.products_per_leaf,
        reviews_per_product=args.reviews_per_product,
        qna_per_product=args.qna_per_product,
        seed=args.seed,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
