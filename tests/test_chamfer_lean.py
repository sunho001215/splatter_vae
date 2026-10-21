"""Variant chamfer_lean: bidirectional 3D Chamfer supervision and the leaner geometry losses (2026-10-21)."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

from s4d.config import load_config
from s4d.losses import chamfer as chamfer_mod
from s4d.losses import rgb
from s4d.losses.chamfer import chamfer_loss, nearest_neighbours, target_clouds
from s4d.train.loop import build_model, forward_losses

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))


def _brute(src, dst):
    d2 = ((src[:, :, None] - dst[:, None]) ** 2).sum(-1)
    return d2.min(-1)


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_nearest_neighbours_match_brute_force_on_every_backend(device):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.fail("the native suite runs with a guarded GPU; CUDA must be available")
    g = torch.Generator().manual_seed(0)
    src = torch.randn(3, 517, 3, generator=g).to(device)
    dst = torch.randn(3, 263, 3, generator=g).to(device)
    ref_d, ref_i = _brute(src.cpu().double(), dst.cpu().double())
    backends = [False] + ([True] if device == "cuda" and chamfer_mod.HAVE_TRITON else [])
    if device == "cuda":
        assert chamfer_mod.HAVE_TRITON, "the fused kernel must be available on the CUDA training path"
    for fused in backends:
        d, i = nearest_neighbours(src, dst, fused=fused)
        assert torch.equal(i.cpu(), ref_i), fused
        torch.testing.assert_close(d.cpu().double(), ref_d, atol=1e-5, rtol=1e-5)


def test_chamfer_value_and_gradient_equal_the_dense_min_formulation():
    g = torch.Generator().manual_seed(1)
    centres = torch.randn(2, 3, 40, 3, generator=g, dtype=torch.float32).requires_grad_(True)
    targets = torch.randn(2, 3, 25, 3, generator=g)
    valid = torch.ones(2, 3, dtype=torch.bool)
    out = chamfer_loss(centres, targets, valid)
    loss = out["loss"].sum()
    loss.backward()
    grad = centres.grad.clone()
    c2 = centres.detach().clone().requires_grad_(True)
    d2 = ((c2[:, :, :, None] - targets[:, :, None]) ** 2).sum(-1)  # (B,S,N,M), dense reference
    ref = (0.5 * d2.min(-1).values.mean(-1) + 0.5 * d2.min(-2).values.mean(-1)).mean(0)  # (S,)
    torch.testing.assert_close(out["loss"], ref.detach())
    ref.sum().backward()
    torch.testing.assert_close(grad, c2.grad)
    torch.testing.assert_close(out["pred_to_target"], d2.min(-1).values.mean(-1).mean(0).detach())
    torch.testing.assert_close(out["target_to_pred"], d2.min(-2).values.mean(-1).mean(0).detach())


def test_invalid_samples_do_not_enter_the_mean():
    centres = torch.zeros(2, 1, 4, 3)
    targets = torch.zeros(2, 1, 3, 3)
    targets[1] = 10.0  # would dominate if it were counted
    out = chamfer_loss(centres, targets, torch.tensor([[True], [False]]))
    assert float(out["loss"][0]) == 0.0


def _cloud_inputs():
    B, T, V, H, W = 1, 2, 2, 4, 4
    depth = torch.full((B, T, V, H, W), 1.0)
    depth[0, 0, 0, 0, :2] = 0.0  # no depth
    depth[0, 0, 1, 1, :] = 3.5  # beyond far (3.0)
    K = torch.tensor([[2.0, 0.0, 2.0], [0.0, 2.0, 2.0], [0.0, 0.0, 1.0]]).expand(B, V, 3, 3)
    c2w = torch.eye(4).expand(B, V, 4, 4).clone()
    c2w[:, 1, 0, 3] = 1.0
    return depth, K, c2w


def test_target_clouds_apply_the_validity_rule_voxel_means_and_a_fixed_budget():
    depth, K, c2w = _cloud_inputs()
    pts, valid, stats = target_clouds(depth, K, c2w, far=3.0, valid_far=True, voxel=0.01, budget=50)
    assert pts.shape == (1, 2, 50, 3) and valid.all()
    assert all(float(p[2]) < 3.0 for p in pts[0, 0])  # nothing at or beyond the far plane
    n_excluded_zero, n_excluded_far = 2, 4
    total = depth.numel()
    torch.testing.assert_close(stats["chamfer_excluded_no_depth"], torch.tensor(n_excluded_zero / total))
    torch.testing.assert_close(stats["chamfer_excluded_beyond_far"], torch.tensor(n_excluded_far / total))
    # the cameras overlap: points seen by both fall into one voxel. t0: 14 + 12 valid pixels, 6 shared positions -> 20
    # voxels; t1: 16 + 16, 8 shared -> 24; the budget of 50 cycles them
    assert len(torch.unique(pts[0, 0], dim=0)) == 20 and len(torch.unique(pts[0, 1], dim=0)) == 24
    # a voxel holding every point (cameras moved into the positive octant, 100 m voxels) yields their mean
    from s4d.geometry import lift_depth

    shifted = c2w.clone()
    shifted[:, :, :2, 3] += 5.0
    coarse, _, _ = target_clouds(depth[:, 1:], K, shifted, far=3.0, valid_far=True, voxel=100.0, budget=3)
    lifted = lift_depth(depth[0, 1], K[0], shifted[0]).reshape(-1, 3)
    torch.testing.assert_close(coarse[0, 0], lifted.mean(0).expand(3, 3))
    # without depth_valid_far the far pixels count (the reference rule)
    _, _, stats_all = target_clouds(depth, K, c2w, far=3.0, valid_far=False, voxel=0.01, budget=8)
    assert float(stats_all["chamfer_excluded_beyond_far"]) == 0.0


def test_target_clouds_sample_without_replacement_when_enough_voxels_exist():
    depth, K, c2w = _cloud_inputs()
    pts, _, _ = target_clouds(depth, K, c2w, far=3.0, valid_far=True, voxel=0.01, budget=20)
    assert len(torch.unique(pts[0, 1], dim=0)) == 20


def test_an_empty_cloud_is_marked_invalid():
    depth, K, c2w = _cloud_inputs()
    depth[0, 0] = 0.0
    _, valid, _ = target_clouds(depth, K, c2w, far=3.0, valid_far=True, voxel=0.01, budget=8)
    assert valid.tolist() == [[False, True]]


def _cfg(variant: bool, **loss):
    files = [REPO / "configs/metaworld/base.yaml"]
    if variant:
        files.append(REPO / "configs/metaworld/variants/chamfer_lean.yaml")
    cfg = load_config(files)
    cfg["train"]["bf16"] = False
    cfg["loss"] = {**cfg["loss"], **loss}
    cfg["model"]["encoder"].update(image_height=16, image_width=16, patch_size=4, width=16, depth=1, heads=4,
                                   num_slots=2, slot_dim=8, drop_path=0)
    cfg["model"]["decoder"].update(dim=16, depth=1, heads=4)
    for group in ("scene", "dynamic"):
        cfg["model"]["decoder"][group].update(parents=2, children=2)
    return cfg


def _fake(monkeypatch, calls):
    from s4d.model import render
    from tests.test_review_options import _fake_rasterizer

    monkeypatch.setattr(render, "_rasterize", _fake_rasterizer(calls))
    monkeypatch.setattr(rgb, "_ssim_map", lambda x, y: 1 - (x - y).square())


def test_variant_config_is_the_reference_plus_the_registered_changes():
    ref = load_config([REPO / "configs/metaworld/base.yaml"])
    var = load_config([REPO / "configs/metaworld/base.yaml", REPO / "configs/metaworld/variants/chamfer_lean.yaml"])
    changed = {k for k in set(ref["loss"]) | set(var["loss"]) if ref["loss"].get(k) != var["loss"].get(k)}
    assert changed == {"depth_valid_far", "motion_space", "depth_hard", "depth_grad", "visibility", "occlusion",
                       "chamfer", "chamfer_points", "chamfer_voxel_m", "motion_feature_render"}
    assert var["loss"]["chamfer"] > 0 and var["loss"]["depth_hard"] == var["loss"]["depth_grad"] == 0
    assert var["loss"]["visibility"] == var["loss"]["occlusion"] == 0
    for key in ("model", "data", "render", "train"):
        assert var[key] == ref[key], key
    assert var["aug"] == {**ref.get("aug", {}), "synth_views": 2, "synth_render": True}


@pytest.mark.parametrize("step", [10, 49])
def test_disabled_terms_and_the_hard_depth_pass_are_skipped(batch, monkeypatch, step):
    from s4d.train import loop

    calls = []
    _fake(monkeypatch, calls)
    for name in ("depth_gradient_loss", "visibility_loss", "occlusion_loss", "render_hard_depth"):
        monkeypatch.setattr(loop, name, lambda *a, _n=name, **k: pytest.fail(f"{_n} computed although disabled"))
    cfg = _cfg(True)
    model = build_model(cfg).train()
    out = forward_losses(model, batch, cfg, step, source=torch.tensor([0, 1]))
    assert torch.isfinite(out["total"])
    modes = [m for m, _, _ in calls]
    assert "ED" not in modes  # no hard-depth render
    log_step = (step + 1) % cfg["train"]["log_every"] == 0
    feature_renders = sum(1 for m, _, grad in calls if m == "RGB" and not grad)  # the no-grad diagnostic feature render
    assert ("epe_moving_01" in out["metrics"]) == log_step
    assert all(f"chamfer_t{t}" in out["losses"] for t in range(3))
    assert not any(k.startswith(("depth_grad", "visibility", "depth_hard")) for k in out["losses"])
    if log_step:
        assert "chamfer_voxels_per_cloud" in out["metrics"] and "dyn_alpha_share_moving" in out["metrics"]


def test_log_step_metrics_do_not_depend_on_the_feature_render_mode(batch, monkeypatch):
    _fake(monkeypatch, [])
    outs = []
    for mode in ("every_step", "log_steps"):
        cfg = _cfg(True, motion_feature_render=mode)
        torch.manual_seed(0)
        model = build_model(cfg).train()
        torch.manual_seed(1)
        outs.append(forward_losses(model, batch, cfg, 49, source=torch.tensor([0, 1])))
    a, b = outs
    torch.testing.assert_close(a["total"], b["total"])
    for k in set(a["metrics"]) | set(b["metrics"]):
        torch.testing.assert_close(a["metrics"][k], b["metrics"][k], equal_nan=True, msg=k)


def test_reference_configuration_still_computes_every_reference_term(batch, monkeypatch):
    calls = []
    _fake(monkeypatch, calls)
    cfg = _cfg(False)
    model = build_model(cfg).train()
    out = forward_losses(model, batch, cfg, 10, source=torch.tensor([0, 1]))
    assert "ED" in [m for m, _, _ in calls]  # hard-depth pass
    assert all(f"depth_grad_t{t}" in out["losses"] for t in range(3)) and "visibility" in out["losses"]
    assert not any(k.startswith("chamfer") for k in out["losses"])
    assert "epe_moving_01" in out["metrics"]  # every-step feature render in the reference


def test_chamfer_uses_only_the_training_cameras_and_reaches_encoder_decoder_and_motion(heldout_batch, monkeypatch):
    from s4d.train import loop

    _fake(monkeypatch, [])
    seen = []
    original = loop.target_clouds

    def recording(depth, K, c2w, *a, **k):
        seen.append((depth, K, c2w))
        return original(depth, K, c2w, *a, **k)

    monkeypatch.setattr(loop, "target_clouds", recording)
    cfg = _cfg(True, rgb=0.0, coverage=0.0, depth_l1=0.0, motion=0.0, invariance=0.0, consistency=0.0)
    cfg["aug"]["synth_render"] = False
    model = build_model(cfg).train()
    out = forward_losses(model, heldout_batch, cfg, 30000, source=torch.tensor([0, 1]))
    depth, K, c2w = seen[0]
    assert depth.shape == heldout_batch["depth"][:, :, :, 0].shape and torch.equal(depth, heldout_batch["depth"][:, :, :, 0])
    assert K is heldout_batch["K"] and c2w is heldout_batch["c2w"]  # never eval_* (held-out) fields
    out["total"].backward()
    enc = sum(float(p.grad.abs().sum()) for p in model.encoder.parameters() if p.grad is not None)
    xyz = sum(float(p.grad.abs().sum()) for n, p in model.decoder.named_parameters() if p.grad is not None and "xyz_head" in n)
    motion = sum(float(p.grad.abs().sum()) for n, p in model.decoder.named_parameters()
                 if p.grad is not None and "motion" in n)
    assert enc > 0 and xyz > 0 and motion > 0  # later timesteps move the dynamic centres through the motion head
