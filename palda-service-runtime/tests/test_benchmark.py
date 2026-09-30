from palda_runtime.benchmark import _percentile


def test_nearest_rank_percentile() -> None:
    assert _percentile([3.0, 1.0, 2.0], 0.50) == 2.0
    assert _percentile([3.0, 1.0, 2.0], 0.95) == 3.0
