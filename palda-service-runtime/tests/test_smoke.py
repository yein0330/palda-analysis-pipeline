import pytest
from pydantic import ValidationError

from palda_runtime.schemas import ReportOutput
from palda_runtime.smoke import default_request


def test_default_smoke_request_covers_review_and_qna() -> None:
    request = default_request()
    assert request["analysisId"].startswith("smoke-")
    assert request["taxonomyVersion"]
    assert request["reviewSample"]["items"]
    assert request["qnaSample"]["items"]


def test_report_output_contract_is_strict() -> None:
    payload = {
        "analysisId": "a1",
        "sample": {
            "review": {"collectionOrder": "newest", "sampleSize": 1, "usableSize": 1, "maxItems": 5000},
            "qna": {"collectionOrder": "newest", "sampleSize": 0, "usableSize": 0, "maxItems": 5000},
        },
        "reviewAnalysis": {"positiveThemes": [], "negativeThemes": []},
        "qnaAnalysis": {"themes": []},
        "sellingPoints": [],
        "versions": {"encoderModel": "e", "decoderModel": "d", "taxonomy": "t", "calculation": "c"},
    }
    assert ReportOutput.model_validate(payload).analysisId == "a1"
    with pytest.raises(ValidationError):
        ReportOutput.model_validate({**payload, "unexpected": True})
