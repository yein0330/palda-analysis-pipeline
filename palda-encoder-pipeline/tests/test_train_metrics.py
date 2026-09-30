from __future__ import annotations

import numpy as np

from palda_encoder.train import _calibrate_thresholds, _metrics


def test_unsupported_theme_takes_the_global_threshold_not_the_loosest_grid_point() -> None:
    # Theme 0 has Gold positives; theme 1 has none, so every threshold scores
    # F1 0 for it and nothing is learned from its own column.
    probabilities = np.asarray([[0.9, 0.8], [0.8, 0.7], [0.3, 0.6], [0.2, 0.55]])
    targets = np.asarray([[1.0, 0.0], [1.0, 0.0], [0.0, 0.0], [0.0, 0.0]])

    thresholds = _calibrate_thresholds(np, probabilities, targets)

    assert thresholds[1] >= 0.2
    assert thresholds[1] == _calibrate_thresholds(np, probabilities, targets)[1]
    # The old argmax-from-0.1 fallback handed the empty theme the loosest
    # threshold in the grid, maximising its false positives.
    assert thresholds[1] > 0.1


def test_macro_f1_ignores_themes_with_no_gold_support_and_no_predictions() -> None:
    themes = ["A", "B", "C"]
    train_rows = [{"labels": [{"theme_id": "A"}, {"theme_id": "B"}]}]
    # Theme A is predicted perfectly, B is missed entirely, C is never seen.
    output = {
        "theme_probs": np.asarray([[0.9, 0.1, 0.1]]),
        "theme_targets": np.asarray([[1.0, 1.0, 0.0]]),
        "sentiment_targets": np.asarray([-1]),
        "sentiment_predictions": np.asarray([0]),
    }

    metrics = _metrics(np, output, [0.5, 0.5, 0.5], train_rows, themes)

    # Mean over A (1.0) and B (0.0) only. Including the never-seen theme C would
    # report 0.333333 and shrink every real difference by a third.
    assert metrics["theme_macro_f1"] == 0.5
