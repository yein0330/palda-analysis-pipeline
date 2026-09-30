#!/usr/bin/env python
"""Build product-level Oracle inputs for Decoder report generation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from qna_review_model.decoder_data import build_oracle_decoder_inputs


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--taxonomy", type=Path, required=True)
    parser.add_argument("--evidence-limit", type=int, default=3)
    return parser.parse_args()


def main() -> None:
    args = arguments()
    manifest = build_oracle_decoder_inputs(
        args.input_dir,
        args.output_dir,
        args.taxonomy,
        evidence_limit=args.evidence_limit,
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
