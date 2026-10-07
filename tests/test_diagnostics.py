"""Exercise diagnostics without substituting synthetic renders for native acceptance."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from s4d.config import get, load_config
from s4d.diag import panels as P
from s4d.diag import wandb_log as W
from s4d.diag.local_log import RunLogger
from s4d.diag.pointclouds import cloud_panel, fused_gt_points, gaussian_points
from s4d.diag.probes import cross_view_retrieval, fit_and_score, probe_targets
from s4d.diag.tracks import sample_track_pixels, track_panel, trajectories
from s4d.losses import rgb
from s4d.losses.depth import depth_gradient_loss
from s4d.losses.regularizers import visibility_loss
from s4d.model.decoder import DecoderConfig, GaussianDecoder, GroupConfig
from s4d.model.encoder import Encoder, EncoderConfig
from s4d.train import evaluate as E
from s4d.train.loop import Model, temporal_ramp

REPO = Path(__file__).resolve().parents[1]


def _model():
    encoder = Encoder(
        EncoderConfig(
            image_height=16, image_width=16, patch_size=4, width=16, depth=1, heads=4, num_slots=2, slot_dim=8, drop_path=0
        )
    )
    group = GroupConfig(parents=2, children=2, offset_scale=0.2, child_radius=0.05, scale_min=0.001, scale_max=0.08)
    decoder = GaussianDecoder(DecoderConfig(slot_dim=8, dim=16, depth=2, heads=4, scene=group, dynamic=group))
    return Model(encoder, decoder)


def test_rgb_coverage_and_dssim_convention_with_controlled_ssim_map(monkeypatch):
    pred = torch.zeros(2, 3, 8, 8, requires_grad=True)
    target = torch.ones_like(pred)
    weights = rgb.pixel_weights(torch.rand(2, 1, 8, 8), 1)
    torch.testing.assert_close(weights.mean((1, 2, 3)), torch.ones(2))
    monkeypatch.setattr(rgb, "_ssim_map", lambda x, y: x * 0 + 0.25)
    loss = rgb.rgb_loss(pred, target, weights)
    torch.testing.assert_close(loss, torch.full((2,), 1 + 0.2 * (1 - 0.25) / 2))
    loss.sum().backward()
    assert torch.isfinite(pred.grad).all() and pred.grad.abs().sum() > 0
    valid = torch.ones_like(weights, dtype=torch.bool)
    assert rgb.coverage_loss(torch.ones_like(weights), valid, weights).sum() == 0
    values = rgb.masked_ssim(pred, target, valid)
    torch.testing.assert_close(values, torch.full((2,), 0.25))
    assert torch.isnan(rgb.masked_ssim(pred, target, ~valid)).all()
    assert torch.isnan(rgb.masked_psnr(pred, target, ~valid)).all()
    assert rgb.rgb_loss(target, target, weights, ssim_weight=0).sum() == 0


def test_visibility_has_zero_inside_and_finite_gradient_outside(batch):
    points = torch.tensor([[[[0.0, 0.0, 1.0], [0.0, 0.0, 2.0]]]], requires_grad=True)
    cams = batch["w2c"][:1, :1]
    intrinsics = batch["K"][:1, :1]
    assert visibility_loss(points, cams, intrinsics, 16, 16, 0.05, 3).sum() == 0
    bad = torch.tensor([[[[2.0, 0.0, 0.01], [0.0, 0.0, 4.0]]]], requires_grad=True)
    loss = visibility_loss(bad, cams, intrinsics, 16, 16, 0.05, 3).sum()
    assert loss > 0
    loss.backward()
    assert torch.isfinite(bad.grad).all() and bad.grad.abs().sum() > 0


def test_depth_gradient_respects_dynamic_weights_and_perfect_prediction():
    target = torch.ones(1, 1, 16, 16)
    pred = torch.exp(torch.linspace(0, 1, 16)).reshape(1, 1, 1, 16).expand_as(target).clone().requires_grad_()
    valid = torch.ones_like(target, dtype=torch.bool)
    baseline = depth_gradient_loss(pred, target, valid)
    weighted = depth_gradient_loss(pred, target, valid, weights=torch.full_like(target, 2))
    torch.testing.assert_close(weighted, 2 * baseline)
    weighted.sum().backward()
    assert torch.isfinite(pred.grad).all()
    assert depth_gradient_loss(target, target, valid).sum() == 0


def test_track_vectors_keep_point_axis_and_project_same_tracks_into_other_camera(batch):
    pixels = torch.tensor([[3, 4], [7, 8], [10, 9], [12, 12]])
    batch["motion3d"][:, 0, :, 0] = 0.1
    batch["motion3d"][:, 2, :, 0] = 0.2
    out = {"pred_disp": batch["motion3d"].clone()}
    gt, predicted, uv = trajectories(batch, out, 0, 0, pixels)
    assert gt.shape == (4, 3, 3) and uv.shape == (2, 4, 3, 2)
    np.testing.assert_array_equal(gt, predicted)
    np.testing.assert_allclose(gt[:, 1, 0] - gt[:, 0, 0], 0.1, atol=1e-7)
    assert track_panel(batch, out, 0, 0, pixels, target_v=1).shape == (16, 52, 3)
    score = batch["motion_score"][0, 0, 0].clone()
    score[:, :8] = 1
    selected = sample_track_pixels(score, batch["motion_weight"][0, 0, 0], n_moving=8, n_static=4)
    assert selected.shape == (12, 2)


def test_image_panels_cloud_mirrors_and_logger_roundtrip(batch, tmp_path):
    model = _model()
    gs = model.decoder(torch.randn(1, 2, 8))
    clouds = gaussian_points(gs, 0)
    fused = fused_gt_points(batch, 0)
    assert fused.shape[1] == 6
    image = cloud_panel(clouds["by_group"], clouds["vectors"], size=32)
    assert image.shape == (32, 100, 3)
    assert cloud_panel(np.zeros((0, 6)), size=32).shape == image.shape
    for function, args in (
        (P.colorize_depth, (batch["depth"][0, 0, 0], 0.05, 3)),
        (P.colorize_error, (torch.zeros(16, 16), 0.1)),
        (P.colorize_gray, (torch.zeros(16, 16),)),
    ):
        assert function(*args).shape == (16, 16, 3)
    flow = P.flow_color(np.zeros((16, 16, 2)), 10)
    assert flow.dtype == np.uint8
    canvas = np.zeros((16, 16, 3), dtype=np.uint8)
    P.draw_points(canvas, np.array([[-1e6, -1e6]]), (255, 0, 0), radius=1)
    assert not canvas.any()
    P.draw_line(canvas, np.array([-1e10, 8]), np.array([1e10, 8]), (255, 0, 0))
    assert canvas[:, :, 0].sum() > 0
    logger = RunLogger(tmp_path)
    logger.text("Synthetic diagnostic exercise, no rendered model acceptance")
    logger.scalars(0, {"test/value": 1})
    logger.save_png(tmp_path / "cloud.png", image)
    logger.save_gif(tmp_path / "clip.gif", np.stack([image, image]), fps=2)
    logger.save_ply(tmp_path / "fused.ply", fused)
    logger.close()
    assert json.loads((tmp_path / "metrics.jsonl").read_text())["test/value"] == 1
    assert (tmp_path / "fused.ply").read_text().startswith("ply\n")


def test_probes_use_physical_time_and_retrieval_uses_distinct_states():
    states = torch.eye(8)
    view_states = states[:, None].expand(-1, 3, -1)
    retrieval = cross_view_retrieval(view_states, 2)
    assert retrieval["retrieval_top1_train"] == retrieval["retrieval_top1_heldout"] == 1
    probe = torch.zeros(8, 3, 11)
    probe[:, 2, :3] = 0.6
    targets = probe_targets(probe, torch.full((8,), 0.3))
    torch.testing.assert_close(targets["hand_vel"], torch.ones(8, 3))
    x = torch.randn(100, 4)
    y = x @ torch.randn(4, 3)
    result = fit_and_score(x, {"position": y}, {"same": (x, {"position": y})})
    assert result["r2_position_same"] > 0.999
    assert len(E._select_retrieval_windows(["one"] * 100, list(range(100)))) == 1
    assert get(load_config([REPO / "configs/droid/pretrain.yaml"]), "model.encoder.num_slots") == 16


def _synthetic_render(gs, xyz, w2c, K, height, width, near, far):
    # Only a unit-test fake. It never imports gsplat or claims rasterization correctness.
    b, times = xyz.shape[:2]
    views = w2c.shape[1]
    shape = (b, times, views, 1, height, width)
    scalar = gs.rgb.mean(-2)[:, None, None, :, None, None]
    return {
        "rgb": scalar.expand(b, times, views, 3, height, width),
        "depth": torch.ones(shape),
        "alpha": torch.full(shape, 0.8),
    }


def test_all_evaluator_panels_with_explicit_synthetic_renderer(batch, tmp_path, monkeypatch):
    model = _model()
    cfg = load_config([REPO / "configs/metaworld/base.yaml"])
    cfg["wandb"]["enabled"] = False
    cfg["eval"]["probe_every"] = 0
    gs = model.decoder(torch.randn(2, 2, 8))
    rendered = _synthetic_render(gs, gs.xyz_sequence(), batch["w2c"], batch["K"], 16, 16, 0.05, 3)
    out = {
        **rendered,
        "pred_disp": batch["motion3d"].clone(),
        "source": torch.zeros(2, dtype=torch.long),
        "pairs": ((0, 1, 0), (1, 2, 1), (0, 2, 0)),
    }
    logger = RunLogger(tmp_path)
    evaluator = E.Evaluator(cfg, {}, None, logger, torch.device("cpu"), 2)
    monkeypatch.setattr(E, "render_rgbd", _synthetic_render)
    monkeypatch.setattr(E, "encode_states", lambda m, x: torch.zeros(x.shape[0], x.shape[2], 2, 8))
    W.init_wandb({}, "synthetic", "splatter4d-metaworld", enabled=False, run_dir=tmp_path)
    summary = {}
    evaluator.panels(model, (batch, out, gs, None, torch.ones(2, 3, 2)), 0, summary, tag="s2")
    logger.close()
    output = tmp_path / "eval/step_0000000/s2"
    for name in (
        "recon_cam0.png",
        "motion.png",
        "same_tracks_source0_target1.png",
        "cross_source.png",
        "pointcloud_gt_fused.png",
        "sequence.gif",
        "orbit.gif",
        "samples.json",
    ):
        assert (output / name).exists(), name
    assert summary["metric/cross_source_std"] == 0


def test_temporal_ramp_boundary_and_encoder_state_keeps_all_slots(batch):
    assert temporal_ramp(0, 20000) == 0
    assert temporal_ramp(10000, 20000) == 0.5
    assert temporal_ramp(20000, 20000) == 1
    assert temporal_ramp(40000, 20000) == 1
    slots = E.encode_states(_model().eval(), batch["images"])
    assert slots.shape == (2, 2, 2, 8)
    assert slots.flatten(2).shape == (2, 2, 16)


def test_complete_shared_loss_backward_with_explicit_synthetic_rasterizer(batch, monkeypatch):
    from s4d.model import render
    from s4d.train.loop import forward_losses

    calls = []

    def fake(**kw):
        # Deliberately synthetic, differentiable shape/gradient regression, not CUDA correctness.
        means, opacity, colors = kw["means"], kw["opacities"], kw["colors"]
        B, T = means.shape[:2]
        V, H, W = kw["viewmats"].shape[2], kw["height"], kw["width"]
        alpha = opacity.mean(-1)
        depth = means[..., 2].mean(-1).abs().add(0.5)[..., None]
        if colors is None:
            values = depth
        else:
            values = colors.mean(-2) * alpha[..., None]
            if kw["render_mode"] == "RGB+ED":
                values = torch.cat((values, depth), -1)
        output = values[:, :, None, None, None].expand(B, T, V, H, W, values.shape[-1])
        covered = alpha[:, :, None, None, None, None].expand(B, T, V, H, W, 1)
        calls.append((kw["render_mode"], kw))
        return output, covered, {}

    monkeypatch.setattr(render, "_rasterize", fake)
    monkeypatch.setattr(rgb, "_ssim_map", lambda x, y: 1 - (x - y).square())
    batch["images"] = torch.randint(0, 256, batch["images"].shape, dtype=torch.uint8)
    batch["motion_score"][:, 2] = 1  # verifies dynamic share includes t2, not only t0
    batch["motion3d"][:, :, :, 0] = 0.04
    cfg = load_config([REPO / "configs/metaworld/base.yaml"])
    cfg["train"]["bf16"] = False
    cfg["loss"]["depth_align"] = "scale_shift"
    model = _model().train()
    source = torch.tensor([0, 1])
    out = forward_losses(model, batch, cfg, 20000, source=source, mask_ratio=0, return_renders=True)
    assert torch.equal(out["source"], source)
    assert out["metrics"]["ramp"] == 1
    assert out["dyn_share_map"].shape == batch["motion_score"].shape
    assert torch.isfinite(out["metrics"]["dyn_alpha_share_moving"])
    torch.testing.assert_close(out["metrics"]["dyn_alpha_share_moving"], out["dyn_share_map"][:, 2].mean())
    assert all(torch.isfinite(value) for value in out["losses"].values())
    out["total"].backward()
    for module in (model.encoder, model.decoder):
        grads = [p.grad for p in module.parameters() if p.grad is not None]
        assert grads and all(torch.isfinite(g).all() for g in grads)
        assert sum(g.abs().sum() for g in grads) > 0
    assert [mode for mode, _ in calls] == ["RGB+ED", "ED", "RGB"]
    feature_call = calls[-1][1]
    assert not feature_call["means"].requires_grad
    assert not feature_call["scales"].requires_grad
    assert not feature_call["quats"].requires_grad
    assert not feature_call["opacities"].requires_grad
    assert feature_call["colors"].shape[-1] == 7 and feature_call["colors"].shape[1] == 3

    # The T=1 ablation removes every temporal loss contribution without unused parameters.
    calls.clear()
    cfg["model"]["single_frame"] = True
    model.encoder.cfg.num_frames = 1
    # Temporal embedding remains allocated, but forward strictly sees only one RGB frame.
    out = forward_losses(model, batch, cfg, 20000, source=source, mask_ratio=0)
    assert out["metrics"]["ramp"] == 0
    assert torch.count_nonzero(out["gs"].delta01) == torch.count_nonzero(out["gs"].delta12) == 0
    out["total"].backward()
    assert torch.isfinite(out["total"])


def test_deletion_entrypoint_is_disabled_and_never_touches_data(monkeypatch):
    import scripts.delete_stage0_cache as deletion

    monkeypatch.setattr("sys.argv", ["delete_stage0_cache.py", "--execute"])
    assert deletion.main() == 2


def test_table_thumbnails_have_local_paths(batch, tmp_path, monkeypatch):
    # Reuse the complete synthetic media exercise and verify each mirrored table thumbnail.
    test_all_evaluator_panels_with_explicit_synthetic_renderer(batch, tmp_path, monkeypatch)
    output = tmp_path / "eval/step_0000000/s2"
    rows = json.loads((output / "samples.json").read_text())
    assert all((output / row["render_t0"]).exists() for row in rows)
