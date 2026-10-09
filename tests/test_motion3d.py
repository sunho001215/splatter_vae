"""Gaussian-space 3D motion loss (M3D) and the binned motion metrics (directive item 2)."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from s4d.config import load_config
from s4d.diag.local_log import RunLogger
from s4d.geometry import lift_depth
from s4d.losses.motion import motion_loss
from s4d.losses.motion3d import motion3d_loss, sample_track_points
from s4d.model.gaussians import DYNAMIC_GROUP, GaussianSet
from s4d.train import evaluate as E

REPO = Path(__file__).resolve().parents[1]
H = W = 8


def _scene():
    """One camera at the identity looking at a plane 1 m away; pixel (0, 0) moves 2 cm along x in every pair."""
    K = torch.tensor([[8.0, 0.0, 4.0], [0.0, 8.0, 4.0], [0.0, 0.0, 1.0]])[None, None]
    c2w = torch.eye(4)[None, None]
    depth = torch.ones(1, 3, 1, 1, H, W)
    target = torch.zeros(1, 3, 1, 3, H, W)
    target[0, :, 0, 0, 0, 0] = 0.02
    weight = torch.zeros(1, 3, 1, 1, H, W)
    weight[..., 0, 0] = 1.0  # only the moving pixel and one static pixel are valid
    weight[..., 0, 1] = 1.0
    return K, c2w, depth, target, weight


def _gaussians(xyz, delta, groups):
    N = len(xyz)
    xyz = torch.tensor(xyz)[None].requires_grad_()
    delta = torch.tensor(delta)[None].requires_grad_()
    gs = GaussianSet(
        xyz,
        torch.full((1, N, 3), 0.01),
        torch.tensor([1.0, 0.0, 0.0, 0.0]).expand(1, N, 4),
        torch.ones(1, N),
        torch.ones(1, N, 3),
        delta,
        delta * 0.5,
        torch.tensor(groups),
    )
    return gs, xyz, delta


def _pixel_points(K, c2w, depth):
    return lift_depth(depth[0, 0, 0, 0], K[0, 0], c2w[0, 0])  # (H,W,3)


def test_track_points_favour_moving_pixels_and_fall_back_to_uniform():
    weight = torch.zeros(3, 16)
    weight[0, :8] = 1.0
    weight[1, :8] = 1.0
    target = torch.zeros(3, 16)
    target[0, 3] = 0.02  # row 0 has one moving pixel; row 1 none; row 2 no valid pixel
    idx, ok = sample_track_points(weight, target, num_points=8, generator=torch.Generator().manual_seed(0))
    assert idx.shape == (3, 8) and ok.tolist() == [True, True, False]
    assert idx[0, :4].tolist() == [3, 3, 3, 3] and (idx[0, 4:] < 8).all()
    assert (idx[1] < 8).all()


def test_matching_displacements_cost_nothing_and_errors_reach_only_nearby_dynamic_deltas():
    K, c2w, depth, target, weight = _scene()
    pts = _pixel_points(K, c2w, depth)
    p_move, p_static = pts[0, 0].tolist(), pts[0, 1].tolist()
    far = [p_move[0] + 0.5, p_move[1], p_move[2]]
    xyz = [p_move, p_static, far, p_move]
    groups = [DYNAMIC_GROUP, DYNAMIC_GROUP, DYNAMIC_GROUP, 0]  # the last one is a scene Gaussian
    exact = [[0.02, 0.0, 0.0], [0.0, 0.0, 0.0], [0.3, 0.0, 0.0], [0.3, 0.0, 0.0]]
    gs, _, _ = _gaussians(xyz, exact, groups)
    # pair 1->2 uses delta12 = delta / 2; set the pair targets accordingly so that every pair matches exactly
    target[0, 1, 0, 0, 0, 0] = 0.01
    target[0, 2, 0, 0, 0, 0] = 0.03
    loss, metrics = motion3d_loss(gs, gs.xyz_sequence(), gs.group == DYNAMIC_GROUP, depth, K, c2w, target, weight)
    # pair 1->2 sources at t1, where the moving Gaussian has moved 2 cm away from the t0 point it was placed on
    assert float(metrics["m3d_disp_01"]) == pytest.approx(0.0, abs=1e-7)
    assert float(metrics["m3d_attr_01"]) == pytest.approx(0.0, abs=1e-7)
    assert float(metrics["m3d_disp_02"]) == pytest.approx(0.0, abs=1e-7)
    wrong = [[0.0, 0.0, 0.0], [0.01, 0.0, 0.0], [0.3, 0.0, 0.0], [0.3, 0.0, 0.0]]
    gs, xyz_t, delta = _gaussians(xyz, wrong, groups)
    loss, _ = motion3d_loss(gs, gs.xyz_sequence(), gs.group == DYNAMIC_GROUP, depth, K, c2w, target, weight)
    loss.backward()
    assert float(loss) > 0
    assert float(delta.grad[0, 0, 0]) < 0  # the moving Gaussian is pushed towards +2 cm
    assert float(delta.grad[0, 1, 0]) > 0  # the static point pulls its Gaussian back to zero motion
    assert float(delta.grad[0, 2].abs().sum()) == 0.0  # 50 cm away: outside the radius
    assert float(delta.grad[0, 3].abs().sum()) == 0.0  # scene Gaussians are never supervised


def test_attraction_pulls_the_nearest_dynamic_centre_to_moving_points():
    K, c2w, depth, target, weight = _scene()
    p_move = _pixel_points(K, c2w, depth)[0, 0].tolist()
    off = [p_move[0], p_move[1] + 0.05, p_move[2]]  # 5 cm away: beyond the radius, so only attraction acts
    gs, xyz, delta = _gaussians([off], [[0.0, 0.0, 0.0]], [DYNAMIC_GROUP])
    weight[..., 0, 1] = 0.0
    loss, metrics = motion3d_loss(gs, gs.xyz_sequence(), gs.group == DYNAMIC_GROUP, depth, K, c2w, target, weight)
    assert float(metrics["m3d_attr_01"]) == pytest.approx(0.05 - 0.005, abs=1e-5)  # Huber, delta 1 cm
    assert float(metrics["m3d_disp_01"]) == 0.0 and float(metrics["m3d_covered_01"]) == 0.0
    loss.backward()
    assert float(xyz.grad[0, 0, 1]) > 0  # descent moves it back towards the point (-y)


def test_binned_relative_epe_and_epe_in_millimetres():
    target = torch.zeros(1, 3, 1, 3, 1, 4)
    target[..., 0, 0, :] = torch.tensor([0.007, 0.02, 0.05, 0.0])  # one pixel per bin and a static pixel
    pred = target * 0.5
    weight, cov = torch.ones(1, 3, 1, 1, 1, 4), torch.ones(1, 3, 1, 1, 1, 4)
    _, metrics = motion_loss(pred, cov, target, weight)
    for n in (1, 2, 3):
        assert float(metrics[f"relepe_bin{n}_02"]) == pytest.approx(0.5)
        assert float(metrics[f"motion_bin{n}_count_02"]) == 1.0
    assert float(metrics["epe_moving_mm_02"]) == pytest.approx(1000 * (0.0035 + 0.01 + 0.025) / 3)


def test_evaluator_weights_binned_relative_epe_by_bin_counts(batch, tmp_path, monkeypatch):
    values = iter([(1.0, 1.0), (3.0, 3.0)])  # (relative EPE, pixels in bin 2) per batch

    def fake_forward(m, b, cfg_, step, *, source, mask_ratio, return_renders=False):
        rel, count = next(values)
        B, T, V = b["images"].shape[:3]
        return {
            "gs": None,
            "slots": torch.randn(B, V, 2, 8),
            "losses": {"total": torch.tensor(1.0)},
            "metrics": {"relepe_bin2_02": torch.tensor(rel), "motion_bin2_count_02": torch.tensor(count)},
            "source": torch.zeros(B, dtype=torch.long),
            "rgb": b["images"].float() / 255.0,
        }

    monkeypatch.setattr(E, "forward_losses", fake_forward)
    monkeypatch.setattr(E.Evaluator, "panels", lambda *a, **k: None)
    cfg = load_config([REPO / "configs/metaworld/base.yaml"])
    cfg["eval"]["heldout_sets"] = False
    logger = RunLogger(tmp_path / "run")
    evaluator = E.Evaluator(cfg, {2: [batch, batch]}, None, logger, torch.device("cpu"), 2)
    summary = evaluator(torch.nn.Linear(1, 1), 1)
    assert summary["metric/relepe_bin2_02@s2"] == pytest.approx((1.0 * 1.0 + 3.0 * 3.0) / 4.0)
    logger.close()
