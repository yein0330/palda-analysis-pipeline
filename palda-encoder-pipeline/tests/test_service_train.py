from __future__ import annotations

import json

import numpy as np
from sklearn.metrics import f1_score

from palda_encoder.service_train import (
    _best_global_threshold, _exclusive_review_predictions, _pool_sentence,
    _positive_weights, _set_metrics, _write_service_predictions,
)


def test_macro_f1_penalizes_predictions_in_unsupported_classes() -> None:
    actual = np.asarray([[True, False]])
    predicted = np.asarray([[True, True]])

    metrics = _set_metrics(np, f1_score, actual, predicted)

    assert metrics["micro_f1"] == 0.666667
    assert metrics["macro_f1"] == 0.5


def test_positive_weights_emphasize_rare_multilabel_targets() -> None:
    weights = _positive_weights(np, np.asarray([100.0, 4.0, 0.0]), 0.5)
    assert weights[0] == 3.0
    assert weights[1] > weights[0]
    assert weights[1] <= 20.0
    assert weights[2] == 3.0
    unweighted = _positive_weights(np, np.asarray([100.0, 4.0]), 0.0, base=1.0)
    assert unweighted.tolist() == [1.0, 1.0]


def test_lower_threshold_beta_favors_precision() -> None:
    probabilities = np.asarray([0.9, 0.4, 0.6, 0.3])
    targets = np.asarray([1.0, 1.0, 0.0, 0.0])
    f1_threshold = _best_global_threshold(np, probabilities, targets, beta=1.0)
    precision_threshold = _best_global_threshold(np, probabilities, targets, beta=0.5)
    assert precision_threshold > f1_threshold


def test_service_predictions_use_serialized_threshold_lists(tmp_path) -> None:
    output = {
        "rows": [{"sample_id": "r1", "record_type": "review"}],
        "review_probs": np.asarray([[[0.8, 0.1, 0.1]]]),
        "primary_predictions": np.asarray([0]),
        "secondary_probs": np.asarray([[0.0]]),
        "usable_probs": np.asarray([1.0]),
    }
    path = tmp_path / "predictions.jsonl"

    _write_service_predictions(
        path,
        output,
        {"review": [[0.5, 0.5, 0.5]], "secondary": [0.5]},
        ["QUALITY"],
    )

    row = json.loads(path.read_text(encoding="utf-8"))
    assert row["predicted_labels"] == [
        {"confidence": 0.8, "sentiment": "positive", "theme_id": "QUALITY"}
    ]


def test_review_prediction_keeps_only_strongest_eligible_sentiment() -> None:
    probabilities = np.asarray([[[0.8, 0.7, 0.1]]])
    thresholds = np.asarray([[0.5, 0.5, 0.5]])

    predicted = _exclusive_review_predictions(np, probabilities, thresholds)

    assert predicted.tolist() == [[[True, False, False]]]


def test_secondary_loss_weight_scales_only_secondary_gradient() -> None:
    import torch
    from palda_encoder.service_train import _service_loss

    batch = {
        "review_task": torch.tensor([False]),
        "sample_weights": torch.ones(1),
        "non_special_mask": torch.ones((1, 2), dtype=torch.bool),
        "candidate_mask": torch.ones((1, 2), dtype=torch.bool),
        "review_targets": torch.zeros((1, 2, 3)),
        "primary_targets": torch.tensor([0]),
        "secondary_targets": torch.tensor([[0., 1.]]),
        "usable_targets": torch.ones(1),
    }
    gradients = []
    for weight in (0.5, 2.0):
        outputs = {
            "review": torch.zeros((1, 2, 3), requires_grad=True),
            "primary": torch.zeros((1, 2), requires_grad=True),
            "secondary": torch.zeros((1, 2), requires_grad=True),
            "usable": torch.zeros(1, requires_grad=True),
        }
        loss, _ = _service_loss(torch, torch.nn, outputs, batch, secondary_loss_weight=weight)
        loss.backward()
        gradients.append({key: value.grad.clone() for key, value in outputs.items()})
    torch.testing.assert_close(gradients[1]["secondary"], gradients[0]["secondary"] * 4)
    for key in ("review", "primary", "usable"):
        torch.testing.assert_close(gradients[0][key], gradients[1][key])


def test_primary_sample_weight_does_not_reweight_secondary_or_usability() -> None:
    import torch
    from palda_encoder.service_train import _service_loss

    batch = {
        "review_task": torch.tensor([False, False]),
        "sample_weights": torch.tensor([1., .3]),
        "non_special_mask": torch.ones((2, 2), dtype=torch.bool),
        "candidate_mask": torch.ones((2, 2), dtype=torch.bool),
        "review_targets": torch.zeros((2, 2, 3)),
        "primary_targets": torch.tensor([0, 1]),
        "secondary_targets": torch.tensor([[0., 1.], [1., 0.]]),
        "usable_targets": torch.ones(2),
    }
    gradients = []
    for weight in (.3, .1):
        batch["primary_sample_weights"] = torch.tensor([1., weight])
        outputs = {
            "review": torch.zeros((2, 2, 3), requires_grad=True),
            "primary": torch.zeros((2, 2), requires_grad=True),
            "secondary": torch.zeros((2, 2), requires_grad=True),
            "usable": torch.zeros(2, requires_grad=True),
        }
        loss, _ = _service_loss(torch, torch.nn, outputs, batch)
        loss.backward()
        gradients.append({key: value.grad.clone() for key, value in outputs.items()})
        ratio = outputs["primary"].grad[1].abs().sum() / outputs["primary"].grad[0].abs().sum()
        torch.testing.assert_close(ratio, torch.tensor(weight))
    for key in ("review", "secondary", "usable"):
        torch.testing.assert_close(gradients[0][key], gradients[1][key])


def test_mean_pooling_ignores_padding_and_cls_pooling_does_not() -> None:
    import torch

    hidden_states = torch.tensor([[[1.0, 1.0], [3.0, 3.0], [99.0, 99.0]]])
    attention_mask = torch.tensor([[1, 1, 0]])

    cls_pooled = _pool_sentence(hidden_states, attention_mask, "cls")
    mean_pooled = _pool_sentence(hidden_states, attention_mask, "mean")

    assert cls_pooled.tolist() == [[1.0, 1.0]]
    # Padding at position 2 must not enter the average.
    assert mean_pooled.tolist() == [[2.0, 2.0]]


def test_train_service_encoder_rejects_unknown_pooling() -> None:
    import pytest

    from palda_encoder.service_train import train_service_encoder

    with pytest.raises(ValueError, match="pooling"):
        train_service_encoder("unused", "unused", pooling="max")
