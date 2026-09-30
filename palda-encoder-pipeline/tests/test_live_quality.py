"""Safety checks for bounded opt-in sweep and read-only Validation viewer."""
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from live_quality_experiment import discounted_elements
from quality_dashboard import label_keys, MODELS, samples, state


@pytest.mark.parametrize("discount", [.1, .25, .5, 1.])
def test_only_silver_negative_terms_discounted(discount):
    logits = torch.zeros(2, 2, requires_grad=True)
    targets = torch.tensor([[0., 1.], [0., 1.]])
    pos = torch.tensor(3.)
    base = torch.nn.functional.binary_cross_entropy_with_logits(logits, targets, pos_weight=pos, reduction="none")
    result = discounted_elements(torch, logits, targets, pos, torch.tensor([False, True]), discount)
    torch.testing.assert_close(result[0], base[0])
    torch.testing.assert_close(result[1, 1], base[1, 1])
    torch.testing.assert_close(result[1, 0], base[1, 0] * discount)
    result.sum().backward()
    assert torch.isfinite(logits.grad).all()


def test_dashboard_roles_and_order():
    gold = {"labels": [{"theme_id": "SIZE", "role": "secondary"},
                       {"theme_id": "RETURN_EXCHANGE", "role": "primary"}]}
    predicted = {"record_type": "qna", "primary_theme_id": "RETURN_EXCHANGE",
                 "secondary_themes": [{"theme_id": "SIZE"}]}
    assert label_keys(gold) == label_keys(predicted, True)


def test_dashboard_allowlist_and_actual_validation():
    assert len(MODELS) == 12
    with pytest.raises(KeyError):
        samples("../../test")
    output = samples("baseline-42")
    assert output["split"] == "validation"
    assert len(output["rows"]) == 707
    assert all(r["outcome"] in ("correct", "wrong") for r in output["rows"])
    assert len(state()["runs"]) == 12
