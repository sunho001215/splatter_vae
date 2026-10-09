"""Motion supervision includes every target and detaches source Gaussian geometry."""

from __future__ import annotations

import torch

from s4d.losses.motion import expected_displacement, motion_loss
from s4d.model import render
from s4d.model.gaussians import DYNAMIC_GROUP, GaussianSet


def test_alpha_normalisation_recovers_world_displacement():
    target = torch.tensor([0.03, -0.01, 0.02]).reshape(1, 1, 1, 3, 1, 1).expand(2, 3, 2, -1, 4, 4)
    coverage = torch.full((2, 3, 2, 1, 4, 4), 0.25)
    prediction = expected_displacement(target * coverage, coverage)
    torch.testing.assert_close(prediction, target, atol=0, rtol=0)


def test_zero_coverage_with_zero_features_is_finite_zero():
    features = torch.zeros(1, 3, 1, 3, 2, 2)
    coverage = torch.zeros(1, 3, 1, 1, 2, 2)
    assert torch.equal(expected_displacement(features, coverage), features)


def test_perfect_motion_has_zero_loss_zero_metrics_and_finite_gradients():
    target = torch.zeros(1, 3, 1, 3, 4, 4)
    target[:, 0, :, 0] = 0.03
    target[:, 1, :, 1] = 0.04
    target[:, 2] = target[:, 0] + target[:, 1]
    prediction = target.clone().requires_grad_()
    coverage = torch.ones(1, 3, 1, 1, 4, 4)
    loss, metrics = motion_loss(prediction, coverage, target, torch.ones_like(coverage))
    assert loss.item() == 0.0
    for name, value in metrics.items():
        if "static" in name and "epe" in name:
            assert torch.isnan(value), "No static support is not a perfect static-motion measurement"
        elif name.startswith("relepe_bin") and float(metrics[f"motion_{name.split('_')[1]}_count_{name[-2:]}"]) == 0:
            assert torch.isnan(value), name  # every target here moves >= 3 cm: the two lower bins are empty
        elif "loss" in name or "epe" in name:
            assert value.item() == 0.0, name
    loss.backward()
    assert prediction.grad is not None and torch.isfinite(prediction.grad).all()
    assert torch.count_nonzero(prediction.grad).item() == 0


def test_pair_weights_are_04_04_02():
    target = torch.zeros(1, 3, 1, 3, 2, 2)
    prediction = target.clone().requires_grad_()
    with torch.no_grad():
        prediction[:, 0, :, 0] = 0.02
        prediction[:, 1, :, 0] = 0.04
        prediction[:, 2, :, 0] = 0.08
    coverage = torch.ones(1, 3, 1, 1, 2, 2)
    loss, metrics = motion_loss(prediction, coverage, target, coverage)
    expected = 0.4 * (0.02 - 0.005) + 0.4 * (0.04 - 0.005) + 0.2 * (0.08 - 0.005)
    torch.testing.assert_close(loss, torch.tensor(expected), atol=1e-7, rtol=1e-6)
    for pair, epe in zip(("01", "12", "02"), (0.02, 0.04, 0.08)):
        torch.testing.assert_close(metrics[f"epe_{pair}"], torch.tensor(epe), atol=1e-7, rtol=1e-6)


def test_missing_predicted_coverage_cannot_hide_valid_motion_targets():
    target = torch.zeros(1, 3, 1, 3, 2, 2)
    target[:, :, :, 0] = 0.03
    prediction = torch.zeros_like(target, requires_grad=True)
    coverage = torch.zeros(1, 3, 1, 1, 2, 2)
    weight = torch.ones_like(coverage)
    loss, metrics = motion_loss(prediction, coverage, target, weight)
    assert loss.item() > 0.0, "Valid target pixels must not disappear when the model renders no coverage."
    for pair in ("01", "12", "02"):
        torch.testing.assert_close(metrics[f"epe_moving_{pair}"], torch.tensor(0.03), atol=1e-7, rtol=1e-6)
        torch.testing.assert_close(metrics[f"rel_epe_{pair}"], torch.tensor(1.0), atol=1e-7, rtol=1e-6)


def _gaussians():
    B, N = 1, 4
    return GaussianSet(
        torch.rand(B, N, 3, requires_grad=True),
        torch.full((B, N, 3), 0.02, requires_grad=True),
        torch.tensor([1.0, 0.0, 0.0, 0.0]).expand(B, N, 4).clone().requires_grad_(),
        torch.full((B, N), 0.8, requires_grad=True),
        torch.full((B, N, 3), 0.5, requires_grad=True),
        torch.full((B, N, 3), 0.01, requires_grad=True),
        torch.full((B, N, 3), 0.02, requires_grad=True),
        torch.full((N,), DYNAMIC_GROUP, dtype=torch.long),
    )


def _synthetic_rasterizer(**kwargs):
    """Differentiable stand-in, never importing or launching gsplat CUDA kernels."""
    means = kwargs["means"]
    B, S = means.shape[:2]
    V = kwargs["viewmats"].shape[2]
    H, W = kwargs["height"], kwargs["width"]
    geometry = (
        means.mean((-1, -2))
        + kwargs["quats"].mean((-1, -2))
        + kwargs["scales"].mean((-1, -2))
        + kwargs["opacities"].mean(-1)
    )
    colors = kwargs["colors"]
    channels = colors.mean(-2) if colors is not None else means[..., 2].mean(-1, keepdim=True)
    channels = channels + geometry[..., None]
    output = channels[:, :, None, None, None].expand(B, S, V, H, W, channels.shape[-1])
    alpha = kwargs["opacities"].mean(-1)[:, :, None, None, None, None].expand(B, S, V, H, W, 1)
    return output, alpha, {}


def test_detached_motion_render_only_backpropagates_into_displacements(monkeypatch):
    gs = _gaussians()
    detached = gs.detach_geometry()
    states = torch.stack((gs.xyz.detach(), (gs.xyz + gs.delta01).detach()), dim=1)
    features = torch.stack(
        (torch.cat((gs.delta01, gs.delta01 + gs.delta12), -1), torch.cat((gs.delta12, torch.zeros_like(gs.delta12)), -1)),
        dim=1,
    )
    monkeypatch.setattr(render, "_rasterize", _synthetic_rasterizer)
    cameras = torch.eye(4).reshape(1, 1, 4, 4)
    K = torch.eye(3).reshape(1, 1, 3, 3)
    result = render.render_features(detached, states, features, cameras, K, 2, 2, 0.05, 3.0)
    result["features"].sum().backward()
    for attr in ("xyz", "scales", "quats", "opacity", "rgb"):
        assert getattr(gs, attr).grad is None, attr
    for attr in ("delta01", "delta12"):
        grad = getattr(gs, attr).grad
        assert grad is not None and torch.isfinite(grad).all() and grad.abs().sum() > 0, attr


def test_hard_depth_fixes_opacity_and_detaches_shapes(monkeypatch):
    gs = _gaussians()
    observed = {}

    def rasterize(**kwargs):
        observed.update(kwargs)
        return _synthetic_rasterizer(**kwargs)

    monkeypatch.setattr(render, "_rasterize", rasterize)
    result = render.render_hard_depth(
        gs, gs.xyz_sequence(), torch.eye(4).reshape(1, 1, 4, 4), torch.eye(3).reshape(1, 1, 3, 3), 2, 2, 0.05, 3.0
    )
    assert observed["render_mode"] == "ED"
    assert not observed["quats"].requires_grad
    assert not observed["scales"].requires_grad
    assert not observed["opacities"].requires_grad
    torch.testing.assert_close(observed["opacities"], torch.full_like(observed["opacities"], 0.95), atol=0, rtol=0)
    result["depth"].sum().backward()
    assert gs.xyz.grad is not None and torch.isfinite(gs.xyz.grad).all()
    for attr in ("scales", "quats", "opacity", "rgb"):
        assert getattr(gs, attr).grad is None, attr


def test_absent_moving_region_is_unavailable_not_a_passing_metric():
    target = torch.zeros(1, 3, 1, 3, 2, 2)
    coverage = torch.ones(1, 3, 1, 1, 2, 2)
    loss, metrics = motion_loss(target, coverage, target, coverage)
    assert loss == 0
    for pair in ("01", "12", "02"):
        assert torch.isnan(metrics[f"rel_epe_{pair}"])
        assert torch.isnan(metrics[f"epe_moving_{pair}"])
        assert metrics[f"motion_moving_count_{pair}"] == 0
        assert metrics[f"epe_static_{pair}"] == 0
