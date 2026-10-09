"""Training options screened in the mid-campaign review (items 2 and 4); all are off by default."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from s4d.config import load_config
from s4d.data.metaworld.cameras import LOOKAT
from s4d.losses import rgb
from s4d.losses.motion import motion_loss
from s4d.model.decoder import AdaLNZeroBlock, DecoderConfig, GaussianDecoder, GroupConfig, fourier_features
from s4d.model.encoder import Encoder, EncoderConfig
from s4d.train import augment as A
from s4d.train.loop import Model, build_model, build_optimizer, forward_losses

REPO = Path(__file__).resolve().parents[1]
GROUP = GroupConfig(parents=2, children=2, offset_scale=0.2, child_radius=0.05, scale_min=0.001, scale_max=0.08)


def _model(**decoder):
    encoder = Encoder(
        EncoderConfig(image_height=16, image_width=16, patch_size=4, width=16, depth=1, heads=4, num_slots=2, slot_dim=8, drop_path=0)
    )
    cfg = DecoderConfig(slot_dim=8, dim=16, depth=2, heads=4, scene=GROUP, dynamic=GROUP, num_slots=2, **decoder)
    return Model(encoder, GaussianDecoder(cfg))


# ------------------------------------------------------------------------------------------ item 4e
def test_moving_normalisation_weights_moving_pixels_and_adds_a_static_term():
    pred = torch.zeros(1, 3, 1, 3, 1, 4)
    target = torch.zeros(1, 3, 1, 3, 1, 4)
    target[:, :, :, 0, 0, 0] = 0.03  # one moving pixel (3 cm) per pair; three static pixels
    pred[:, :, :, 0, 0, 3] = 0.001  # 1 mm error on one static pixel
    weight, cov = torch.ones(1, 3, 1, 1, 1, 4), torch.ones(1, 3, 1, 1, 1, 4)
    loss, metrics = motion_loss(pred, cov, target, weight, huber_delta=0.01, normalization="moving")
    huber = lambda e: 0.5 * e**2 / 0.01 if e < 0.01 else e - 0.005  # noqa: E731
    moving = huber(0.03)  # w = 2 on the only moving pixel; sum w = 2 > floor 0.04
    static = huber(0.001) / 3
    per_pair = moving + 0.1 * static
    assert float(loss) == pytest.approx(per_pair * (0.4 + 0.4 + 0.2), rel=1e-5)
    valid, _ = motion_loss(pred, cov, target, weight, huber_delta=0.01)
    assert float(valid) == pytest.approx((huber(0.03) + huber(0.001)) / 4, rel=1e-5)
    assert metrics["rel_epe_02"] == pytest.approx(1.0)
    with pytest.raises(ValueError):
        motion_loss(pred, cov, target, weight, normalization="other")


# ------------------------------------------------------------------------------------------ item 2a / 2b helpers
def test_crop_is_identity_when_off_and_shared_across_frames_when_on():
    x = torch.rand(4, 3, 3, 16, 16)
    same, _ = A.random_resized_crop(x, None, prob=0.0)
    assert torch.equal(same, x)
    full, _ = A.random_resized_crop(x, None, prob=1.0, scale=(1.0, 1.0), ratio=(1.0, 1.0))
    torch.testing.assert_close(full, x, atol=1e-5, rtol=0)
    frames = x[:, :1].expand(4, 3, 3, 16, 16).contiguous()
    score = torch.rand(4, 3, 1, 16, 16)
    cropped, cropped_score = A.random_resized_crop(frames, score, prob=1.0, scale=(0.8, 0.8))
    assert not torch.allclose(cropped, frames)
    for t in (1, 2):
        torch.testing.assert_close(cropped[:, t], cropped[:, 0])
    assert cropped_score.shape == score.shape


def test_jittered_cameras_look_at_the_workspace_within_the_trajectory_ranges():
    K = torch.tensor([[110.0, 0, 64], [0, 110.0, 64], [0, 0, 1]]).expand(3, 3, 3)
    cams = A.jitter_cameras(3, 2, K, 128, 128, seed=0)
    assert cams["w2c"].shape == (3, 2, 4, 4) and cams["K"].shape == (3, 2, 3, 3)
    centres = cams["c2w"][..., :3, 3]
    distance = (centres - torch.tensor(LOOKAT)).norm(dim=-1)
    assert bool(((distance > 0.95 - 1e-4) & (distance < 1.05 + 0.13)).all())
    forward = cams["c2w"][..., :3, 2]  # OpenCV +z looks at LOOKAT
    to_lookat = torch.nn.functional.normalize(torch.tensor(LOOKAT) - centres, dim=-1)
    assert bool(((forward * to_lookat).sum(-1) > 0.9999).all())
    assert torch.equal(A.jitter_cameras(3, 2, K, 128, 128, seed=0)["w2c"], cams["w2c"])


def test_synthesised_own_views_reproduce_the_images_and_holes_are_masked(batch):
    batch["images"] = torch.randint(0, 256, batch["images"].shape, dtype=torch.uint8)
    for key in ("images", "depth"):  # one camera: re-splatting its own lifted pixels must be exact
        batch[key] = batch[key][:, :, :1]
    for key in ("K", "w2c", "c2w"):
        batch[key] = batch[key][:, :1]
    own = {"K": batch["K"], "w2c": batch["w2c"], "c2w": batch["c2w"]}
    syn = A.synthesize_views(batch, own, near=0.05, far=3.0)
    assert syn["images"].shape == batch["images"].shape[:3] + (3, 16, 16)
    assert bool(syn["covered"].all())
    torch.testing.assert_close(syn["images"], batch["images"].float() / 255.0)
    torch.testing.assert_close(syn["depth"], batch["depth"])
    covered = torch.ones(1, 3, 1, 16, 16, dtype=torch.bool)
    covered[:, 1, :, :4, :4] = False  # patch 0 has a hole in one frame only
    for _ in range(20):
        visible = A.coverage_visible(covered, 4, keep=8)
        assert int(visible.sum()) == 8 and not bool(visible[0, 0])
    assert int(A.coverage_visible(torch.zeros(1, 3, 1, 16, 16, dtype=torch.bool), 4, keep=8).sum()) == 8


def test_encoder_visible_override_matches_an_explicit_mask():
    encoder = Encoder(EncoderConfig(image_height=16, image_width=16, patch_size=4, width=16, depth=1, heads=4, slot_dim=8, drop_path=0))
    encoder.eval()
    x = torch.rand(2, 3, 3, 16, 16)
    visible = torch.zeros(2, 16, dtype=torch.bool)
    visible[:, :6] = True
    out = encoder.forward_visible(x, visible)
    assert torch.equal(out["visible"], visible) and out["patch_tokens"].shape[1] == 3 * 6
    assert torch.equal(encoder(x)["slots"], encoder.forward_visible(x, torch.ones(2, 16, dtype=torch.bool))["slots"])


# ------------------------------------------------------------------------------------------ item 4a-4c
def test_decoder_lr_multiplier_scales_only_decoder_groups():
    cfg = load_config([REPO / "configs/metaworld/base.yaml"])
    model = _model()
    opt, sched = build_optimizer(model, cfg)
    assert len(opt.param_groups) == 2 and {g["initial_lr"] for g in opt.param_groups} == {5e-4}
    cfg["train"]["decoder_lr_mult"] = 3.0
    opt, sched = build_optimizer(model, cfg)
    decoder = {id(p) for p in model.decoder.parameters()}
    assert len(opt.param_groups) == 4
    for group in opt.param_groups:
        in_decoder = {id(p) in decoder for p in group["params"]}
        assert len(in_decoder) == 1
        assert group["initial_lr"] == pytest.approx(1.5e-3 if in_decoder.pop() else 5e-4)
    assert sum(len(g["params"]) for g in opt.param_groups) == len(list(model.parameters()))


def test_adaln_zero_blocks_start_as_identity_and_fourier_features_are_periodic():
    block = AdaLNZeroBlock(16, 4, 4.0)
    x, kv, cond = torch.randn(2, 5, 16), torch.randn(2, 3, 16), torch.randn(2, 16)
    assert torch.equal(block(x, kv, cond), x)
    f = fourier_features(torch.tensor([[0.25, -0.5, 1.0]]), 3)
    assert f.shape == (1, 18)
    torch.testing.assert_close(fourier_features(torch.tensor([[2.25, 1.5, 3.0]]), 3), f, atol=1e-5, rtol=0)
    for options in ({"conditioning": "adaln_zero", "anchor_fourier": 6}, {"state_concat": True}):
        gs = _model(**options).decoder(torch.randn(3, 2, 8))
        assert gs.xyz.shape == (3, 2 * 2 * 2, 3) and torch.isfinite(gs.xyz).all()
    concat = _model(state_concat=True).decoder.state_cat
    assert torch.equal(concat.weight[:, :16], torch.eye(16)) and concat.weight.shape == (16, 16 + 2 * 8)
    with pytest.raises(ValueError, match="conditioning"):
        _model(conditioning="other")
    built = build_model({"model": {"encoder": {"num_slots": 4, "image_height": 16, "image_width": 16, "patch_size": 4,
                                               "width": 16, "depth": 1, "heads": 4, "slot_dim": 8},
                                   "decoder": {"dim": 16, "depth": 1, "heads": 4, "state_concat": True,
                                               "scene": GROUP.__dict__, "dynamic": GROUP.__dict__}}})
    assert built.decoder.cfg.num_slots == 4 and built.decoder.state_cat.in_features == 16 + 4 * 8


# ------------------------------------------------------------------------------------------ forward_losses
def _fake_rasterizer(calls):
    def fake(**kw):
        # synthetic differentiable stand-in (shapes and gradients only), as in test_diagnostics; any batch dims, colours
        # shared by the cameras [..., N, D] or per camera [..., C, N, D]
        means, opacity, colors = kw["means"], kw["opacities"], kw["colors"]
        lead = means.shape[:-2]
        V, H, W = kw["viewmats"].shape[-3], kw["height"], kw["width"]
        alpha = opacity.mean(-1)[..., None, None].expand(*lead, V, 1)
        depth = means[..., 2].mean(-1).abs().add(0.5)[..., None, None].expand(*lead, V, 1)
        if colors is None:
            values = depth
        else:
            per_camera = colors.dim() == means.dim() + 1
            values = (colors.mean(-2) if per_camera else colors.mean(-2)[..., None, :]) * alpha
            if kw["render_mode"] == "RGB+ED":
                values = torch.cat((values, depth), -1)
        calls.append((kw["render_mode"], V, torch.is_grad_enabled()))
        output = values[..., None, None, :].expand(*lead, V, H, W, values.shape[-1])
        return output, alpha[..., None, None, :].expand(*lead, V, H, W, 1), {}

    return fake


def test_forward_losses_with_every_review_option(batch, monkeypatch):
    from s4d.model import render

    calls = []
    monkeypatch.setattr(render, "_rasterize", _fake_rasterizer(calls))
    monkeypatch.setattr(rgb, "_ssim_map", lambda x, y: 1 - (x - y).square())
    batch["images"] = torch.randint(0, 256, batch["images"].shape, dtype=torch.uint8)
    batch["motion3d"][:, :, :, 0] = 0.04
    cfg = load_config([REPO / "configs/metaworld/base.yaml"])
    cfg["train"]["bf16"] = False
    cfg["aug"] = {"crop": {"prob": 1.0}, "synth_views": 2, "synth_render": True}
    cfg["loss"].update(self_render=0.5, motion_norm="moving", depth_hard_boost={"factor": 3.0, "fraction": 0.2})
    model = _model(conditioning="adaln_zero", anchor_fourier=4, state_concat=True).train()
    out = forward_losses(model, batch, cfg, 100, source=torch.tensor([0, 1]))
    assert {"synth_render", "self_render"} <= set(out["losses"]) and torch.isfinite(out["total"])
    out["total"].backward()
    for module in (model.encoder, model.decoder):
        grads = [p.grad for p in module.parameters() if p.grad is not None]
        assert grads and all(torch.isfinite(g).all() for g in grads)
    rgb_calls = [c for c in calls if c[0] == "RGB+ED"]
    assert [c[1] for c in rgb_calls] == [2, 2, 2]  # training cameras, synthetic views, self-render cameras
    assert rgb_calls[1][2] and not rgb_calls[2][2]  # self-render renders without a decoder gradient path
    # evaluation never augments: the same call in eval mode renders only the training cameras
    calls.clear()
    model.eval()
    with torch.no_grad():
        forward_losses(model, batch, cfg, 100, source=torch.tensor([0, 1]), mask_ratio=0.0)
    assert [c[1] for c in calls if c[0] == "RGB+ED"] == [2]


def test_depth_hard_boost_applies_only_in_the_first_fraction(batch, monkeypatch):
    from s4d.model import render

    monkeypatch.setattr(render, "_rasterize", _fake_rasterizer([]))
    monkeypatch.setattr(rgb, "_ssim_map", lambda x, y: 1 - (x - y).square())
    cfg = load_config([REPO / "configs/metaworld/base.yaml"])
    cfg["train"].update(bf16=False, steps=1000)
    cfg["loss"]["ramp_steps"] = 0
    model = _model().eval()
    torch.manual_seed(0)
    base = forward_losses(model, batch, cfg, 100, source=torch.tensor([0, 1]), mask_ratio=0.0)
    cfg["loss"]["depth_hard_boost"] = {"factor": 3.0, "fraction": 0.2}
    boosted = [forward_losses(model, batch, cfg, s, source=torch.tensor([0, 1]), mask_ratio=0.0) for s in (100, 300)]
    extra = 2 * 0.5 * base["losses"]["depth_hard_t0"]
    torch.testing.assert_close(boosted[0]["losses"]["render"], base["losses"]["render"] + extra)
    torch.testing.assert_close(boosted[1]["losses"]["render"], base["losses"]["render"])
    assert np.isfinite(float(extra))


# ------------------------------------------------------------------------------------------ directive item 1
def test_hard_pass_is_skipped_at_weight_zero_and_occlusion_adds_its_term(batch, monkeypatch):
    from s4d.model import render

    calls = []
    monkeypatch.setattr(render, "_rasterize", _fake_rasterizer(calls))
    monkeypatch.setattr(rgb, "_ssim_map", lambda x, y: 1 - (x - y).square())
    cfg = load_config([REPO / "configs/metaworld/base.yaml"])
    cfg["train"]["bf16"] = False
    cfg["loss"]["ramp_steps"] = 0
    model = _model().eval()
    base = forward_losses(model, batch, cfg, 100, source=torch.tensor([0, 1]), mask_ratio=0.0)
    assert "ED" in [c[0] for c in calls] and "depth_hard_t0" in base["losses"]
    calls.clear()
    cfg["loss"]["depth_hard"] = 0.0
    no_hard = forward_losses(model, batch, cfg, 100, source=torch.tensor([0, 1]), mask_ratio=0.0)
    assert "ED" not in [c[0] for c in calls] and "depth_hard_t0" not in no_hard["losses"]
    torch.testing.assert_close(no_hard["losses"]["render"], base["losses"]["render"] - 0.5 * base["losses"]["depth_hard_t0"])
    cfg["loss"].update(occlusion=2.0, occlusion_margin=0.02)
    occ = forward_losses(model, batch, cfg, 100, source=torch.tensor([0, 1]), mask_ratio=0.0)
    torch.testing.assert_close(occ["losses"]["render"], no_hard["losses"]["render"] + 2.0 * occ["losses"]["occlusion_t0"])
    # evaluation reports the usage diagnostics and the utilisation mask
    assert {"utilisation", "hidden_fraction", "floater_fraction"} <= set(occ["metrics"])
    assert occ["used"].shape == occ["gs"].opacity.shape and occ["used"].dtype == torch.bool


def test_depth_beyond_the_far_plane_is_invalid_only_with_the_fix(batch, monkeypatch):
    from s4d.model import render

    monkeypatch.setattr(render, "_rasterize", _fake_rasterizer([]))
    monkeypatch.setattr(rgb, "_ssim_map", lambda x, y: 1 - (x - y).square())
    cfg = load_config([REPO / "configs/metaworld/base.yaml"])
    cfg["train"]["bf16"] = False
    cfg["loss"]["ramp_steps"] = 0
    model = _model().eval()
    far = float(cfg["render"]["far"])
    run = lambda b: forward_losses(model, b, cfg, 100, source=torch.tensor([0, 1]), mask_ratio=0.0)  # noqa: E731
    beyond = {k: (v.clone() if torch.is_tensor(v) else v) for k, v in batch.items()}
    beyond["depth"][..., :4, :] = far + 1.0  # the top rows are background beyond the far plane
    other = {k: (v.clone() if torch.is_tensor(v) else v) for k, v in beyond.items()}
    other["depth"][..., :4, :] = far + 2.0
    a, b = run(beyond), run(other)
    assert float(a["losses"]["depth_l1_t0"]) != pytest.approx(float(b["losses"]["depth_l1_t0"]))
    cfg["loss"]["depth_valid_far"] = True
    a, b = run(beyond), run(other)
    for key in ("depth_l1_t0", "depth_grad_t0", "depth_hard_t0", "coverage_t0", "motion"):
        assert float(a["losses"][key]) == pytest.approx(float(b["losses"][key])), key
    for key in ("motion_valid_weight_01", "motion_valid_weight_02"):
        assert float(a["metrics"][key]) < float(run(batch)["metrics"][key]), key
