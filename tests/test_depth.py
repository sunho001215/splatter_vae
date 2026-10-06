"""Depth alignment must not let geometry receive gradients through its target."""

from __future__ import annotations

import pytest
import torch

from s4d.losses.depth import abs_rel, align_teacher, depth_gradient_loss, depth_l1


@pytest.mark.parametrize("jitter", [0.7, 0.85, 1.0, 1.4])
def test_scale_alignment_is_invariant_to_per_image_teacher_scale(jitter):
    rendered = torch.linspace(0.25, 2.5, 2 * 16 * 16).reshape(2, 1, 16, 16)
    rendered[1] += 0.6
    teacher = rendered * torch.tensor([jitter, 1.3]).reshape(2, 1, 1, 1)
    teacher[:, :, 0, 0] = 0
    alpha = torch.ones_like(rendered)
    aligned, stats = align_teacher(rendered, teacher, alpha, mode="scale")
    valid = teacher > 0
    torch.testing.assert_close(aligned[valid], rendered[valid], atol=5e-7, rtol=1e-6)
    torch.testing.assert_close(stats["scale"], torch.tensor([1 / jitter, 1 / 1.3]), atol=5e-7, rtol=1e-6)
    assert torch.equal(aligned[~valid], torch.zeros_like(aligned[~valid]))
    torch.testing.assert_close(stats["shift"], torch.zeros(2), atol=0, rtol=0)


@pytest.mark.parametrize("jitter,expected", [(0.1, 2.0), (10.0, 0.5)])
def test_scale_alignment_clamps_estimate_to_required_bounds(jitter, expected):
    rendered = torch.ones(1, 1, 8, 8)
    _, stats = align_teacher(rendered, rendered * jitter, rendered, mode="scale")
    torch.testing.assert_close(stats["scale"], torch.tensor([expected]), atol=0, rtol=0)


@pytest.mark.parametrize("mode", ["none", "scale", "scale_shift"])
def test_empty_valid_support_is_identity_finite_and_keeps_invalid_zero(mode):
    rendered = torch.full((2, 1, 16, 16), 2.0)
    teacher = torch.zeros_like(rendered)
    aligned, stats = align_teacher(rendered, teacher, torch.zeros_like(rendered), mode=mode)
    assert torch.isfinite(aligned).all()
    assert torch.equal(aligned, teacher)
    assert torch.equal(stats["scale"], torch.ones(2))
    assert torch.equal(stats["shift"], torch.zeros(2))


@pytest.mark.parametrize("mode", ["scale", "scale_shift"])
def test_alignment_ignores_insufficient_support(mode):
    rendered = torch.full((1, 1, 4, 4), 2.0)
    teacher = torch.ones_like(rendered)
    alpha = torch.ones_like(rendered)
    alpha[..., 0, 0] = 0
    aligned, stats = align_teacher(rendered, teacher, alpha, mode=mode)
    torch.testing.assert_close(aligned, teacher, atol=0, rtol=0)
    torch.testing.assert_close(stats["scale"], torch.ones(1), atol=0, rtol=0)


def test_scale_estimation_uses_only_visible_depth_support():
    rendered = torch.full((1, 1, 16, 16), 2.0)
    teacher = torch.ones_like(rendered)
    teacher[..., :8, :] = 30.0
    alpha = torch.ones_like(rendered)
    alpha[..., :8, :] = 0.5  # Strictly greater than 0.5 is required.
    aligned, stats = align_teacher(rendered, teacher, alpha, mode="scale")
    torch.testing.assert_close(stats["scale"], torch.tensor([2.0]), atol=0, rtol=0)
    torch.testing.assert_close(aligned[..., 8:, :], rendered[..., 8:, :], atol=0, rtol=0)


def test_scale_shift_recovers_exact_affine_teacher_mismatch():
    teacher = torch.linspace(0.4, 3.0, 256).reshape(1, 1, 16, 16)
    rendered = teacher * 1.4 + 0.2
    aligned, stats = align_teacher(rendered, teacher, torch.ones_like(teacher), mode="scale_shift")
    torch.testing.assert_close(aligned, rendered, atol=2e-6, rtol=1e-6)
    torch.testing.assert_close(stats["scale"], torch.tensor([1.4]), atol=2e-6, rtol=1e-6)
    torch.testing.assert_close(stats["shift"], torch.tensor([0.2]), atol=2e-6, rtol=1e-6)


def test_alignment_target_cannot_backpropagate_into_rendered_geometry():
    rendered = torch.full((1, 1, 16, 16), 2.0, requires_grad=True)
    teacher = torch.ones_like(rendered, requires_grad=True)
    aligned, _ = align_teacher(rendered, teacher, torch.ones_like(rendered), mode="scale")
    aligned.sum().backward()
    assert rendered.grad is None
    assert teacher.grad is not None and torch.isfinite(teacher.grad).all()


def test_unknown_alignment_mode_rejected():
    depth = torch.ones(1, 1, 16, 16)
    with pytest.raises(ValueError, match="alignment"):
        align_teacher(depth, depth, depth, mode="affine-not-a-mode")


def test_perfect_depth_losses_zero_with_finite_zero_gradients():
    target = torch.linspace(0.3, 2.0, 256).reshape(1, 1, 16, 16)
    rendered = target.clone().requires_grad_()
    valid = torch.ones_like(target, dtype=torch.bool)
    weights = torch.ones_like(target)
    loss = (depth_l1(rendered, target, valid, weights) + depth_gradient_loss(rendered, target, valid)).sum()
    assert loss.item() == 0.0
    assert abs_rel(rendered, target, valid).item() == 0.0
    loss.backward()
    assert rendered.grad is not None
    assert torch.isfinite(rendered.grad).all()
    assert torch.count_nonzero(rendered.grad).item() == 0


def test_empty_depth_supervision_has_finite_zero_loss_and_gradients():
    rendered = torch.zeros(2, 1, 16, 16, requires_grad=True)
    target = torch.zeros_like(rendered)
    valid = torch.zeros_like(rendered, dtype=torch.bool)
    loss = (
        depth_l1(rendered, target, valid, torch.ones_like(rendered)) + depth_gradient_loss(rendered, target, valid)
    ).sum()
    assert loss.item() == 0.0
    loss.backward()
    assert rendered.grad is not None and torch.isfinite(rendered.grad).all()
