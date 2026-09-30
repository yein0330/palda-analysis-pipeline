"""Shared project and external data paths."""

from __future__ import annotations

import os
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = Path(
    os.environ.get("PALDA_DATA_ROOT", PROJECT_ROOT.parent / "qna_review_data")
).expanduser().resolve()

