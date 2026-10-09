"""Occlusion / free-space loss and Gaussian-usage diagnostics (directive item 1)."""

from __future__ import annotations

import pytest
import torch

from s4d.losses.occlusion import occlusion_loss, sample_depth, usage_diagnostics
from s4d.model.gaussians import DYNAMIC_GROUP, GaussianSet
from s4d.model.render import render_blending_weights, render_rgbd

H = W = 32
NEAR, FAR, M = 0.05, 3.0, 0.02


def _cams(V=1, device="cpu"):
    K = torch.tensor([[45.0, 0.0, W / 2], [0.0, 45.0, H / 2], [0.0, 0.0, 1.0]], device=device)
    return torch.eye(4, device=device).expand(1, V, 4, 4).clone(), K.expand(1, V, 3, 3).clone()


def _loss(points, planes):
    """Centres (N,3) at one state; one camera per plane depth, all at the identity pose."""
    w2c, K = _cams(len(planes))
    depth = torch.stack([torch.full((1, H, W), d) for d in planes])[None, None]  # (1,1,V,1,H,W)
    xyz = torch.tensor(points, dtype=torch.float32)[None, None].requires_grad_()
    return occlusion_loss(xyz, depth, w2c, K, NEAR, FAR, M), xyz


def test_centres_on_or_near_the_surface_cost_nothing():
    loss, _ = _loss([[0.0, 0.0, 2.0], [0.0, 0.0, 2.015], [0.0, 0.0, 1.985]], [2.0])
    assert float(loss) == 0.0


def test_hidden_centres_are_pulled_forward_and_floaters_pushed_back():
    loss, xyz = _loss([[0.0, 0.0, 2.1]], [2.0])
    assert float(loss) == pytest.approx(0.08, abs=1e-6)
    loss.sum().backward()
    assert float(xyz.grad[0, 0, 0, 2]) == pytest.approx(1.0)  # descent moves it towards the camera
    loss, xyz = _loss([[0.0, 0.0, 1.5]], [2.0])
    assert float(loss) == pytest.approx(0.48, abs=1e-6)
    loss.sum().backward()
    assert float(xyz.grad[0, 0, 0, 2]) == pytest.approx(-1.0)


def test_behind_is_the_minimum_over_cameras_and_front_averages_free_space_cameras():
    # behind the surface of camera 1 by 10 cm but in free space of camera 2 by 10 cm: only the front term counts
    loss, _ = _loss([[0.0, 0.0, 2.1]], [2.0, 2.2])
    assert float(loss) == pytest.approx(0.08, abs=1e-6)
    # in free space of two cameras (by 0.5 and 0.3 m) and on the surface of a third: front = mean over the two
    loss, _ = _loss([[0.0, 0.0, 1.5]], [2.0, 1.8, 1.5])
    assert float(loss) == pytest.approx((0.48 + 0.28) / 2, abs=1e-6)
    # averaged over Gaussians
    loss, _ = _loss([[0.0, 0.0, 2.1], [0.0, 0.0, 2.0]], [2.0])
    assert float(loss) == pytest.approx(0.04, abs=1e-6)


def test_invalid_depth_outside_image_and_behind_camera_are_not_counted():
    for planes in ([0.0], [FAR], [FAR + 1.0]):
        assert float(_loss([[0.0, 0.0, 1.0]], planes)[0]) == 0.0
    assert float(_loss([[5.0, 0.0, 1.0]], [2.0])[0]) == 0.0  # projects far outside the image
    assert float(_loss([[0.0, 0.0, -1.0]], [2.0])[0]) == 0.0  # behind the camera
    z, D, valid = sample_depth(
        torch.tensor([[[[0.0, 0.0, 1.0], [5.0, 0.0, 1.0]]]]), torch.full((1, 1, 1, 1, H, W), 2.0), *_cams(), NEAR, FAR
    )
    assert valid.tolist() == [[[[True, False]]]] and float(D[0, 0, 0, 0]) == 2.0 and float(z[0, 0, 0, 0]) == 1.0


def test_depth_is_read_at_the_nearest_pixel_of_each_state():
    depth = torch.full((1, 2, 1, 1, H, W), 2.0)
    depth[0, 1, 0, 0, :, W // 2 :] = 1.0  # at state 1 the right half is nearer
    w2c, K = _cams()
    xyz = torch.tensor([[[0.01, 0.0, 2.0]], [[0.01, 0.0, 2.0]]])[None]  # u = 16 + 45 * 0.005 = 16.2 -> pixel 16
    z, D, valid = sample_depth(xyz, depth, w2c, K, NEAR, FAR)
    assert D[0, :, 0, 0].tolist() == [2.0, 1.0] and valid.all()
    loss = occlusion_loss(xyz, depth, w2c, K, NEAR, FAR, M)
    assert loss.shape == (2,) and float(loss[0]) == 0.0 and float(loss[1]) == pytest.approx(0.98, abs=1e-6)


# ------------------------------------------------------------------------------------------ CUDA diagnostics
def _gaussians(xyz, opacity, scale=0.02):
    N = len(xyz)
    return GaussianSet(
        torch.tensor(xyz, device="cuda")[None],
        torch.full((1, N, 3), scale, device="cuda"),
        torch.tensor([1.0, 0.0, 0.0, 0.0], device="cuda").expand(1, N, 4).clone(),
        torch.tensor(opacity, device="cuda")[None],
        torch.full((1, N, 3), 0.5, device="cuda"),
        torch.zeros(1, N, 3, device="cuda"),
        torch.zeros(1, N, 3, device="cuda"),
        torch.full((N,), DYNAMIC_GROUP, device="cuda", dtype=torch.long),
    )


def test_blending_weights_of_a_single_gaussian_sum_its_rendered_alpha():
    gs = _gaussians([[0.0, 0.0, 2.0]], [0.8], scale=0.05)
    w2c, K = _cams(2, "cuda")
    w2c[0, 1, 0, 3] = 10.0  # the second camera looks away from it
    weights = render_blending_weights(gs, gs.xyz, w2c, K, H, W, NEAR, FAR)
    alpha = render_rgbd(gs, gs.xyz[:, None], w2c, K, H, W, NEAR, FAR)["alpha"]
    assert weights.shape == (1, 2, 1)
    torch.testing.assert_close(weights[0, 0, 0], alpha[0, 0, 0].sum(), rtol=1e-4, atol=1e-4)
    assert float(weights[0, 1, 0]) == 0.0


def test_usage_diagnostics_count_used_hidden_and_floating_gaussians():
    # surface at 2 m; three opaque wide occluders at 1.99-2.0 m stop compositing before a small Gaussian at 2.5 m
    xyz = [[0.0, 0.0, 1.99], [0.0, 0.0, 1.995], [0.0, 0.0, 2.0], [0.0, 0.0, 2.5], [0.05, 0.05, 1.5], [0.05, -0.05, 1.5]]
    opacity = [0.999, 0.999, 0.999, 0.999, 0.999, 0.1]
    gs = _gaussians(xyz, opacity, scale=0.02)
    gs.scales[0, :3] = 1.0
    w2c, K = _cams(1, "cuda")
    depth0 = torch.full((1, 1, 1, H, W), 2.0, device="cuda")
    metrics, used = usage_diagnostics(gs, depth0, w2c, K, NEAR, FAR, M)
    assert used[0].tolist() == [True, True, True, False, True, True]
    assert float(metrics["utilisation"][0]) == pytest.approx(5 / 6)
    assert float(metrics["hidden_fraction"][0]) == pytest.approx(1 / 5)  # of the five opaque Gaussians
    assert float(metrics["floater_fraction"][0]) == pytest.approx(1 / 5)  # the transparent floater is not counted
