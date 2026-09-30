"""Keep private artifact integration checks explicit in the public snapshot."""
import os
import pytest

PRIVATE_ARTIFACT_TESTS = {
    "test_dashboard_allowlist_and_actual_validation",
    "test_actual_twelve_runs_no_raw_predictions",
    "test_real_validation_views_and_quote_grounding",
    "test_html_escapes_source_text",
    "test_encoder_snapshot_scope",
}


def pytest_collection_modifyitems(items):
    if os.environ.get("PALDA_RUN_PRIVATE_ARTIFACT_TESTS") == "1":
        return
    for item in items:
        if item.originalname in PRIVATE_ARTIFACT_TESTS:
            item.add_marker(pytest.mark.skip(reason="Requires private release/artifact data; opt in with PALDA_RUN_PRIVATE_ARTIFACT_TESTS=1"))
