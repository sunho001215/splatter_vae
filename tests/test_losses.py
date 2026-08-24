from __future__ import annotations

import torch

from models.gaussian.motion import construct_chronological_gaussian_sequence
from models.training.losses import (
    compute_optical_flow_loss,
    confidence_weighted_metric_depth_l1,
    cross_view_info_nce,
    masked_rgb_reconstruction_losses,
    scale_invariant_log_depth_loss,
)


def test_rgb_reconstruction_ignores_padding() -> None:
    target = torch.zeros(1, 3, 8, 8)
    predicted = target.clone()
    predicted[..., :2, :] = 1.0
    validity = torch.ones(1, 1, 8, 8, dtype=torch.bool)
    validity[..., :2, :] = False
    l1, _dssim, _metrics = masked_rgb_reconstruction_losses(predicted, target, validity)
    assert l1.item() == 0.0


def test_metric_and_scale_invariant_depth_losses() -> None:
    target = torch.linspace(0.5, 2.0, 64).view(1, 1, 8, 8)
    predicted = target * 1.7
    confidence = torch.ones_like(target)
    validity = torch.ones_like(target, dtype=torch.bool)
    metric, metrics = confidence_weighted_metric_depth_l1(
        predicted, target, confidence, validity
    )
    si = scale_invariant_log_depth_loss(predicted, target, confidence, validity)
    assert metric.item() > 0.0
    assert metrics["depth_valid_fraction"].item() == 1.0
    assert si.item() < 0.002


def test_flow_loss_uses_validity_without_semantics() -> None:
    predicted = torch.zeros(1, 3, 2, 2, 8, 8, requires_grad=True)
    teacher = predicted.detach().clone()
    auxiliary = torch.ones(1, 3, 2, 1, 8, 8)
    loss, metrics = compute_optical_flow_loss(
        predicted, teacher, auxiliary, auxiliary.bool(), auxiliary.bool()
    )
    loss.backward()
    assert loss.item() == 0.0
    assert predicted.grad is not None
    assert metrics["flow_valid_fraction"].item() == 1.0


def test_cross_view_infonce_has_only_pair_positive() -> None:
    features = torch.randn(4, 2, 32, requires_grad=True)
    loss, metrics = cross_view_info_nce(features, temperature=0.1)
    loss.backward()
    assert torch.isfinite(loss)
    assert features.grad is not None and features.grad.isfinite().all()
    assert set(metrics) == {"positive_cosine_similarity", "negative_cosine_similarity"}


def test_gaussian_dynamics_are_anchored_at_current_frame() -> None:
    current = torch.tensor([[[1.0, 2.0, 3.0]]])
    delta01 = torch.tensor([[[0.1, 0.2, 0.3]]])
    delta12 = torch.tensor([[[0.4, 0.5, 0.6]]])
    sequence = construct_chronological_gaussian_sequence(
        {"xyz": current, "delta_xyz_01": delta01, "delta_xyz_12": delta12},
        "current",
    )
    torch.testing.assert_close(sequence[2]["xyz"], current)
    torch.testing.assert_close(sequence[1]["xyz"], current - delta12)
    torch.testing.assert_close(sequence[0]["xyz"], current - delta12 - delta01)
