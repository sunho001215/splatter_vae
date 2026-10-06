"""Actual FP32 CUDA rasterization with the existing gsplat native extension."""

from __future__ import annotations

import importlib
import importlib.util
from pathlib import Path

import pytest
import torch

from s4d.losses.motion import expected_displacement
from s4d.model.gaussians import DYNAMIC_GROUP, GaussianSet
from s4d.model.render import render_features, render_hard_depth, render_rgbd


@pytest.fixture(scope="module", autouse=True)
def prebuilt_gsplat_only():
    spec = importlib.util.find_spec("gsplat")
    assert spec is not None and spec.submodule_search_locations, "gsplat is not installed"
    directory = Path(next(iter(spec.submodule_search_locations)))
    assert list(directory.glob("csrc*.so")), "A prebuilt extension is required; never attempt a source build."
    # Load the native extension before gsplat's package initializer. Its normal
    # importer catches ABI errors and silently attempts a forbidden JIT build.
    extension_path = next(directory.glob("csrc*.so"))
    extension_spec = importlib.util.spec_from_file_location("gsplat.csrc", extension_path)
    compiled = importlib.util.module_from_spec(extension_spec)
    extension_spec.loader.exec_module(compiled)
    import sys

    sys.modules["gsplat.csrc"] = compiled
    assert compiled.__file__.endswith(".so")
    backend = importlib.import_module("gsplat.cuda._backend")
    assert backend._C is compiled
    return str(compiled.__file__)


def _gaussians(count=1):
    xyz = torch.tensor([[-0.03, 0.02, 2.0], [0.08, -0.04, 2.15]], device="cuda")[:count]
    return GaussianSet(
        xyz[None].clone().requires_grad_(),
        torch.full((1, count, 3), 0.12, device="cuda", requires_grad=True),
        torch.tensor([1.0, 0.0, 0.0, 0.0], device="cuda").expand(1, count, 4).clone().requires_grad_(),
        torch.full((1, count), 0.999, device="cuda", requires_grad=True),
        torch.tensor([0.2, 0.6, 0.9], device="cuda").expand(1, count, 3).clone().requires_grad_(),
        torch.tensor([0.03, -0.01, 0.02], device="cuda").expand(1, count, 3).clone().requires_grad_(),
        torch.tensor([-0.01, 0.02, 0.01], device="cuda").expand(1, count, 3).clone().requires_grad_(),
        torch.full((count,), DYNAMIC_GROUP, device="cuda", dtype=torch.long),
    )


def _cameras():
    return (
        torch.eye(4, device="cuda").reshape(1, 1, 4, 4),
        torch.tensor([[45.0, 0.0, 16.0], [0.0, 45.0, 16.0], [0.0, 0.0, 1.0]], device="cuda").reshape(1, 1, 3, 3),
    )


def test_single_opaque_gaussian_rgb_expected_depth_equals_camera_z():
    gs = _gaussians()
    w2c, K = _cameras()
    result = render_rgbd(gs, gs.xyz[:, None], w2c, K, 32, 32, 0.05, 5.0)
    assert result["rgb"].shape == (1, 1, 1, 3, 32, 32)
    assert result["depth"].shape == result["alpha"].shape == (1, 1, 1, 1, 32, 32)
    assert result["alpha"].max() > 0.9
    valid = result["alpha"] > 0.05
    torch.testing.assert_close(result["depth"][valid], torch.full_like(result["depth"][valid], 2.0), atol=1e-5, rtol=1e-5)
    color = result["rgb"] / result["alpha"].clamp_min(1e-8)
    expanded_valid = valid.expand_as(color)
    expected = gs.rgb[:, 0, :][:, None, None, :, None, None].expand_as(color)
    torch.testing.assert_close(color[expanded_valid], expected[expanded_valid], atol=1e-5, rtol=1e-5)
    assert all(value.dtype == torch.float32 and torch.isfinite(value).all() for value in result.values())


def test_constant_world_displacement_survives_feature_splat_normalisation():
    gs = _gaussians(count=2)
    w2c, K = _cameras()
    result = render_features(gs.detach_geometry(), gs.xyz.detach()[:, None], gs.delta01[:, None], w2c, K, 32, 32, 0.05, 5.0)
    # The same normalization is used for B,P,V channels in motion supervision.
    expected = expected_displacement(result["features"], result["alpha"])
    valid = (result["alpha"] > 0.05).expand_as(expected)
    vector = torch.tensor([0.03, -0.01, 0.02], device="cuda").reshape(1, 1, 1, 3, 1, 1).expand_as(expected)
    assert valid.any()
    torch.testing.assert_close(expected[valid], vector[valid], atol=1e-6, rtol=1e-5)


def test_cuda_feature_backward_only_updates_displacement_tensors():
    gs = _gaussians(count=2)
    w2c, K = _cameras()
    states = torch.stack((gs.xyz.detach(), (gs.xyz + gs.delta01).detach()), dim=1)
    features = torch.stack(
        (torch.cat((gs.delta01, gs.delta01 + gs.delta12), -1), torch.cat((gs.delta12, torch.zeros_like(gs.delta12)), -1)),
        dim=1,
    )
    result = render_features(gs.detach_geometry(), states, features, w2c, K, 32, 32, 0.05, 5.0)
    result["features"].square().sum().backward()
    for attr in ("xyz", "scales", "quats", "opacity", "rgb"):
        assert getattr(gs, attr).grad is None, attr
    for attr in ("delta01", "delta12"):
        grad = getattr(gs, attr).grad
        assert grad is not None and torch.isfinite(grad).all() and grad.abs().sum() > 0, attr


def test_cuda_hard_depth_backward_only_updates_centers():
    gs = _gaussians(count=2)
    w2c, K = _cameras()
    result = render_hard_depth(gs, gs.xyz[:, None], w2c, K, 32, 32, 0.05, 5.0)
    valid = result["alpha"].detach() > 0.05
    assert valid.any()
    result["depth"][valid].mean().backward()
    assert gs.xyz.grad is not None and torch.isfinite(gs.xyz.grad).all() and gs.xyz.grad.abs().sum() > 0
    for attr in ("scales", "quats", "opacity", "rgb", "delta01", "delta12"):
        assert getattr(gs, attr).grad is None, attr
