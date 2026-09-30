"""Guard opt-in loss adaptations before launching full training."""
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from research_loss_experiments import experimental_loss, secondary_elements


def batch_and_outputs():
    batch = {
        "review_task": torch.tensor([False, False, True]),
        "sample_weights": torch.tensor([1., .3, 1.]),
        "non_special_mask": torch.ones((3, 2), dtype=torch.bool),
        "candidate_mask": torch.ones((3, 2), dtype=torch.bool),
        "review_targets": torch.zeros((3, 2, 3)),
        "primary_targets": torch.tensor([0, 1, -100]),
        "secondary_targets": torch.tensor([[0., 1.], [1., 0.], [0., 0.]]),
        "usable_targets": torch.ones(3),
    }
    outputs = {"review": torch.zeros((3, 2, 3), requires_grad=True),
               "primary": torch.zeros((3, 2), requires_grad=True),
               "secondary": torch.zeros((3, 2), requires_grad=True),
               "usable": torch.zeros(3, requires_grad=True)}
    return batch, outputs


@pytest.mark.parametrize("arm,changed", [("secondary_asl", "secondary"),
    ("silver_negative_discount", "secondary"), ("silver_primary_gce", "primary")])
def test_only_requested_head_gradients_change(arm, changed):
    gradients = []
    for mode in ("baseline", arm):
        batch, outputs = batch_and_outputs()
        loss, _ = experimental_loss(mode, torch, torch.nn, outputs, batch)
        assert torch.isfinite(loss)
        loss.backward()
        gradients.append({k: v.grad.clone() for k, v in outputs.items()})
    for key in gradients[0]:
        if key != changed:
            torch.testing.assert_close(gradients[0][key], gradients[1][key])
    assert not torch.allclose(gradients[0][changed], gradients[1][changed])
    if arm != "secondary_asl":
        torch.testing.assert_close(gradients[0][changed][0], gradients[1][changed][0])


def test_discount_preserves_positives_and_gold():
    x = torch.zeros((2, 2))
    y = torch.tensor([[0., 1.], [0., 1.]])
    pos = torch.tensor(3.)
    original = torch.nn.functional.binary_cross_entropy_with_logits(x, y, pos_weight=pos, reduction="none")
    result = secondary_elements(torch, x, y, pos, "silver_negative_discount", torch.tensor([False, True]))
    torch.testing.assert_close(result[0], original[0])
    torch.testing.assert_close(result[1, 1], original[1, 1])
    torch.testing.assert_close(result[1, 0], original[1, 0] * .25)


def test_asl_extremes_finite_and_masked_labels_no_gradient():
    batch, outputs = batch_and_outputs()
    batch["non_special_mask"][:, 1] = False
    outputs["secondary"] = torch.tensor([[-100., 100.]] * 3, requires_grad=True)
    loss, _ = experimental_loss("secondary_asl", torch, torch.nn, outputs, batch)
    loss.backward()
    assert torch.isfinite(loss)
    assert torch.isfinite(outputs["secondary"].grad).all()
    assert (outputs["secondary"].grad[:, 1] == 0).all()
