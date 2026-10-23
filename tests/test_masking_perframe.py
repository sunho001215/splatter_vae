"""Masking ablation and per-frame pilots (2026-10-23): random temporal-token masking, per-frame encoding with spatial
reconstruction, the pair-conditioned temporal branch, and the stacked RL policy input."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

from s4d.config import load_config
from s4d.losses import rgb
from s4d.losses.motion3d import motion3d_loss
from s4d.model.decoder import DecoderConfig, GaussianDecoder, GroupConfig
from s4d.model.encoder import Encoder, EncoderConfig, sample_random_tokens, sample_tube_mask, visible_count
from s4d.model.gaussians import DYNAMIC_GROUP
from s4d.train.augment import coverage_visible, coverage_visible_tokens
from s4d.train.loop import build_model, forward_losses

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))
VARIANTS = REPO / "configs/metaworld/variants"


def _small_encoder(**kw) -> Encoder:
    cfg = dict(image_height=16, image_width=16, patch_size=4, width=16, depth=1, heads=4, num_slots=2, slot_dim=8,
               drop_path=0.0)
    cfg.update(kw)
    return Encoder(EncoderConfig(**cfg))


def test_visible_counts_equal_the_sincro_and_mae_truncation():
    for tokens, ratio, keep in ((192, 0.75, 48), (64, 0.75, 16), (64, 0.9, 6), (64, 0.5, 32), (16, 0.75, 4)):
        assert visible_count(tokens, ratio) == keep == int(tokens * (1 - ratio) + 1e-9)


def test_random_tokens_keep_exactly_the_budget_without_a_per_frame_quota():
    g = torch.Generator().manual_seed(0)
    vis = sample_random_tokens(4000, 192, 48, "cpu", generator=g)
    assert (vis.sum(1) == 48).all()
    per_frame = vis.view(4000, 3, 64).sum(-1)
    assert (per_frame != 16).any(1).float().mean() > 0.9  # no fixed 16-per-frame quota
    assert not (vis.view(4000, 3, 64)[:, 0] == vis.view(4000, 3, 64)[:, 1]).all(1).any()  # never one shared mask
    freq = vis.float().mean(0)
    assert float((freq - 0.25).abs().max()) < 0.04  # uniform over all 192 temporal tokens


def test_random_mode_masks_temporal_tokens_and_ignores_motion_scores():
    enc = _small_encoder(num_frames=3, mask_mode="random", mask_ratio=0.75).train()
    N = enc.cfg.num_patches  # 16
    x = torch.rand(5, 3, 3, 16, 16)
    score = torch.rand(5, 3, 1, 16, 16)
    torch.manual_seed(3)
    a = enc(x, score)
    torch.manual_seed(3)
    b = enc(x, None)
    assert a["visible"].shape == (5, 3 * N) and (a["visible"].sum(1) == visible_count(3 * N, 0.75)).all()
    assert torch.equal(a["visible"], b["visible"])  # no motion prioritisation
    torch.testing.assert_close(a["slots"], b["slots"])
    assert a["patch_tokens"].shape[1] == visible_count(3 * N, 0.75)


def test_token_masks_gather_the_selected_frame_tokens():
    torch.manual_seed(0)
    enc = _small_encoder(num_frames=3, mask_mode="random").eval()
    N = enc.cfg.num_patches
    x = torch.rand(2, 3, 3, 16, 16)
    vis = torch.zeros(2, 3 * N, dtype=torch.bool)
    vis[:, :N] = True  # only frame 0's tokens
    y = x.clone()
    y[:, 1:] = torch.rand_like(y[:, 1:])  # frames 1 and 2 changed
    a, b = enc.forward_visible(x, vis), enc.forward_visible(y, vis)
    torch.testing.assert_close(a["slots"], b["slots"])  # hidden frames do not reach the state
    vis2 = torch.zeros(2, 3 * N, dtype=torch.bool)
    vis2[:, 2 * N:] = True  # only frame 2's tokens
    assert not torch.allclose(enc.forward_visible(x, vis2)["slots"], enc.forward_visible(y, vis2)["slots"])
    spatial = torch.zeros(2, N, dtype=torch.bool)
    spatial[:, :4] = True
    tokens = spatial.repeat(1, 3)  # the same spatial patches in every frame as per-token mask
    torch.testing.assert_close(enc.forward_visible(x, spatial)["slots"], enc.forward_visible(x, tokens)["slots"])
    with pytest.raises(ValueError):
        enc.forward_visible(x, torch.ones(2, N + 1, dtype=torch.bool))


def test_tube_mode_is_unchanged():
    enc = _small_encoder(num_frames=3, mask_ratio=0.5).train()
    x, score = torch.rand(3, 3, 3, 16, 16), torch.rand(3, 3, 1, 16, 16)
    torch.manual_seed(5)
    out = enc(x, score)
    torch.manual_seed(5)
    ref = sample_tube_mask(enc.patch_scores(score), 0.5, enc.cfg.motion_patch_threshold)
    assert torch.equal(out["visible"], ref) and out["visible"].shape == (3, enc.cfg.num_patches)
    with pytest.raises(ValueError):
        _small_encoder(mask_mode="tubes")


def test_coverage_tokens_prefer_each_frames_own_covered_patches():
    covered = torch.zeros(3, 3, 1, 16, 16)
    covered[:, 0, :, :8] = 1.0  # frame 0: top half covered (8 patches)
    covered[:, 1] = 1.0  # frame 1 fully covered (16 patches)
    vis = coverage_visible_tokens(covered, 4, 12)
    assert vis.shape == (3, 48) and (vis.sum(1) == 12).all()
    frac = torch.nn.functional.avg_pool2d(covered.flatten(0, 1), 4).view(3, 48)
    assert (frac[vis] >= 0.9).all()  # enough covered tokens: no holes
    many = coverage_visible_tokens(covered, 4, 30)  # only 24 covered tokens: the exception adds 6 holes
    assert (many.sum(1) == 30).all() and ((frac >= 0.9) & ~many).sum() == 0
    shared = coverage_visible(covered, 4, 4)  # the tube form needs coverage in every frame: none here
    assert shared.shape == (3, 16)


def _cfg(*variants, **loss):
    files = [REPO / "configs/metaworld/base.yaml", VARIANTS / "chamfer_lean.yaml"] + [VARIANTS / v for v in variants]
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


def test_variant_configs_change_only_their_registered_keys():
    ref = load_config([REPO / "configs/metaworld/base.yaml", VARIANTS / "chamfer_lean.yaml"])
    rm = load_config([REPO / "configs/metaworld/base.yaml", VARIANTS / "chamfer_lean.yaml", VARIANTS / "chamfer_lean_randmask.yaml"])
    assert {k for k in ref if ref[k] != rm.get(k)} == {"model"}
    assert {k for k in ref["model"]["encoder"] | rm["model"]["encoder"]
            if ref["model"]["encoder"].get(k) != rm["model"]["encoder"].get(k)} == {"mask_mode", "mask_ratio"}
    assert rm["model"]["encoder"]["mask_mode"] == "random" and rm["model"]["encoder"]["mask_ratio"] == 0.75
    assert rm["model"]["decoder"] == ref["model"]["decoder"] and rm["loss"] == ref["loss"] and rm["aug"] == ref["aug"]
    for name, temporal in (("perframe_spatial.yaml", False), ("perframe_temporal.yaml", True)):
        pf = load_config([REPO / "configs/metaworld/base.yaml", VARIANTS / "chamfer_lean.yaml", VARIANTS / name])
        assert {k for k in ref if ref[k] != pf.get(k)} == {"model"}, name
        assert pf["model"]["encoder"]["num_frames"] == 1 and pf["model"]["encoder"]["mask_ratio"] == 0.75
        assert pf["model"]["perframe"]["temporal"] is temporal and pf["model"]["decoder"]["temporal_branch"] is temporal
        assert pf["data"] == ref["data"] and pf["loss"] == ref["loss"] and pf["aug"] == ref["aug"]
        assert {k: v for k, v in pf["model"]["decoder"].items() if k not in ("motion_head", "temporal_branch",
                "temporal_hidden", "temporal_embed")} == ref["model"]["decoder"]


def test_randmask_variant_trains_with_temporal_token_masks_on_real_and_synthetic_views(batch, monkeypatch):
    _fake(monkeypatch, [])
    cfg = _cfg("chamfer_lean_randmask.yaml")
    model = build_model(cfg).train()
    seen = []
    original = model.encoder.forward_visible

    def recording(images, visible):
        seen.append(visible)
        return original(images, visible)

    monkeypatch.setattr(model.encoder, "forward_visible", recording)
    out = forward_losses(model, batch, cfg, 100, source=torch.tensor([0, 1]))
    assert torch.isfinite(out["total"])
    N = model.encoder.cfg.num_patches
    assert seen and seen[0].shape[1] == 3 * N and (seen[0].sum(1) == visible_count(3 * N, 0.75)).all()


def test_perframe_models_have_static_decoders_and_the_registered_branch():
    for name, temporal in (("perframe_spatial.yaml", False), ("perframe_temporal.yaml", True)):
        model = build_model(_cfg(name))
        assert model.frame_mode == "stack" and model.encoder.cfg.num_frames == 1
        assert all(h.motion_head is None for h in model.decoder.groups)
        assert (model.decoder.temporal is not None) is temporal
    bad = _cfg("perframe_spatial.yaml")
    bad["model"]["perframe"]["temporal"] = True
    with pytest.raises(ValueError):
        build_model(bad)
    ref = _cfg()
    ref["model"]["decoder"]["motion_head"] = False
    with pytest.raises(ValueError):
        build_model(ref)


def test_decoder_features_and_zero_initialised_branch_give_persistence():
    cfg = DecoderConfig(slot_dim=8, num_slots=2, dim=16, depth=1, heads=4, motion_head=False, temporal_branch=True,
                        scene=GroupConfig(2, 2, 0.15, 0.06, 0.001, 0.08), dynamic=GroupConfig(3, 2, 0.4, 0.04, 0.0005, 0.03))
    dec = GaussianDecoder(cfg)
    gs, feats = dec(torch.randn(4, 2, 8), return_features=True)
    assert feats.shape == (4, 6, 16) and float(gs.delta01.abs().sum() + gs.delta12.abs().sum()) == 0.0
    d = dec.temporal(feats, torch.randn(4, 2, 8), torch.randn(4, 2, 8))
    assert d.shape == (4, 6, 3) and float(d.abs().sum()) == 0.0
    with pytest.raises(ValueError):
        GaussianDecoder(DecoderConfig(slot_dim=8, num_slots=2, dim=16, depth=1, heads=4, temporal_branch=True))


def test_pair_sources_reproduce_the_reference_m3d_on_one_set():
    g = torch.Generator().manual_seed(0)
    B, V, H, W, N = 2, 2, 8, 8, 12
    from s4d.model.gaussians import GaussianSet

    xyz = torch.randn(B, N, 3, generator=g) * 0.05 + torch.tensor([0.0, 0.0, 1.0])
    d01, d12 = torch.randn(B, N, 3, generator=g) * 0.01, torch.randn(B, N, 3, generator=g) * 0.01
    group = torch.tensor([0] * 4 + [DYNAMIC_GROUP] * 8)
    gs = GaussianSet(xyz, torch.ones(B, N, 3) * 0.01, torch.tensor([1.0, 0, 0, 0]).expand(B, N, 4), torch.ones(B, N),
                     torch.ones(B, N, 3), d01, d12, group)
    K = torch.tensor([[8.0, 0, 4], [0, 8.0, 4], [0, 0, 1]]).expand(B, V, 3, 3)
    c2w = torch.eye(4).expand(B, V, 4, 4)
    depth = torch.ones(B, 3, V, 1, H, W)
    target = torch.randn(B, 3, V, 3, H, W, generator=g) * 0.01
    weight = torch.ones(B, 3, V, 1, H, W)
    xyz_seq = gs.xyz_sequence()
    a = motion3d_loss(gs, xyz_seq, group == DYNAMIC_GROUP, depth, K, c2w, target, weight,
                      generator=torch.Generator().manual_seed(1))
    b = motion3d_loss(None, None, group == DYNAMIC_GROUP, depth, K, c2w, target, weight,
                      generator=torch.Generator().manual_seed(1), pair_sources=[xyz_seq[:, 0], xyz_seq[:, 1], xyz_seq[:, 0]],
                      pair_displacements=[d01, d12, d01 + d12])
    torch.testing.assert_close(a[0], b[0])
    for k in a[1]:
        torch.testing.assert_close(a[1][k], b[1][k], msg=k)


@pytest.mark.parametrize("variant", ["perframe_spatial.yaml", "perframe_temporal.yaml"])
def test_perframe_forward_encodes_every_image_alone_and_reaches_the_encoder(batch, monkeypatch, variant):
    _fake(monkeypatch, [])
    batch["images"] = torch.randint(0, 256, batch["images"].shape, dtype=torch.uint8)
    batch["motion3d"][:, :, :, 0] = 0.04
    cfg = _cfg(variant)
    model = build_model(cfg).train()
    rows = []
    original = model.encoder._encode

    def recording(images, motion_score, mask_ratio, visible):
        rows.append((images.shape, None if visible is None else int(visible.sum(1)[0])))
        return original(images, motion_score, mask_ratio, visible)

    monkeypatch.setattr(model.encoder, "_encode", recording)
    out = forward_losses(model, batch, cfg, 30000, source=torch.tensor([0, 1]))
    B, T, V = batch["images"].shape[:3]
    assert rows[0][0][:2] == (B * V * T, 1)  # every real image alone
    assert out["slots"].shape[:2] == (B, V) and out["slots"].shape[2] == T * model.encoder.cfg.num_slots
    assert out["gs"].xyz.shape[0] == B * T
    temporal = variant == "perframe_temporal.yaml"
    N = model.encoder.cfg.num_patches
    # identical image exposure in both pilots: the real images (75 % masks) and the synthetic views, nothing else
    assert len(rows) == 2 and rows[1][0][:2] == (B * 2 * T, 1) and rows[1][1] == visible_count(N, 0.75)
    if temporal:
        assert all(f"psnr_cross_{a}{b}" in out["metrics"] for a, b in ("01", "12", "02"))
        assert out["losses"]["motion"] >= 0 and "cross_render" in out["losses"]
    else:
        assert "cross_render" not in out["losses"] and float(out["losses"]["motion"]) == 0.0
    out["total"].backward()
    assert sum(float(p.grad.abs().sum()) for p in model.encoder.parameters() if p.grad is not None) > 0
    if temporal:
        branch = sum(float(p.grad.abs().sum()) for p in model.decoder.temporal.parameters() if p.grad is not None)
        assert branch > 0


def test_temporal_branch_reuses_each_frames_spatial_encoding(batch, monkeypatch):
    """75 % source / 75 % target: the branch conditions on the source camera's encodings of frames a and b, i.e. the same
    masked encoder outputs the spatial reconstructions use (no separate target encoding)."""
    _fake(monkeypatch, [])
    batch["images"] = torch.randint(0, 256, batch["images"].shape, dtype=torch.uint8)
    cfg = _cfg("perframe_temporal.yaml")
    model = build_model(cfg).train()
    encoded = []
    original_encode = model.encoder._encode

    def recording_encode(images, motion_score, mask_ratio, visible):
        out = original_encode(images, motion_score, mask_ratio, visible)
        encoded.append(out["slots"].detach().clone())
        return out

    monkeypatch.setattr(model.encoder, "_encode", recording_encode)
    captured = []
    original = model.decoder.temporal.forward

    def recording(feats, s, t):
        captured.append((s.detach().clone(), t.detach().clone()))
        return original(feats, s, t)

    monkeypatch.setattr(model.decoder.temporal, "forward", recording)
    source = torch.tensor([0, 1])
    forward_losses(model, batch, cfg, 30000, source=source)
    B, T, V = batch["images"].shape[:3]
    real = encoded[0].view(B, V, T, *encoded[0].shape[1:])  # rows (b, v, t): the spatial encodings
    own = real[torch.arange(B), source]  # (B,T,K,Ds), source camera
    assert len(captured) == 3
    for (a, b, _), (s, t) in zip(((0, 1, 0), (1, 2, 1), (0, 2, 0)), captured):
        torch.testing.assert_close(s, own[:, a])
        torch.testing.assert_close(t, own[:, b])


def test_rl_frame_modes_keep_history_and_refuse_mismatches():
    from s4d.rl.encoders import splatter4d_frame_mode

    assert splatter4d_frame_mode("auto", 3, None) == "joint"
    assert splatter4d_frame_mode("auto", 1, None) == "latest"  # every historical T=1 run
    assert splatter4d_frame_mode("stack", 1, None) == "stack"  # historical T=1 exports, stacked
    assert splatter4d_frame_mode("auto", 1, "stack") == "stack"
    for args in (("stack", 3, None), ("latest", 3, None), ("joint", 1, None), ("latest", 1, "stack"), ("x", 1, None)):
        with pytest.raises(ValueError):
            splatter4d_frame_mode(*args)


def test_stacked_policy_state_is_the_chronological_concatenation(tmp_path):
    from dataclasses import asdict

    from s4d.rl.encoders import Splatter4DStateEncoder

    torch.manual_seed(0)
    enc = _small_encoder(num_frames=1)
    path = tmp_path / "encoder.pt"
    torch.save({"format": "splatter4d-encoder-v1", "encoder_config": asdict(enc.cfg), "state_dict": enc.state_dict()}, path)
    frames = torch.rand(2, 3, 3, 16, 16)
    stack = Splatter4DStateEncoder({"vision": {"export_path": str(path), "frame_features": "stack"}})
    latest = Splatter4DStateEncoder({"vision": {"export_path": str(path)}})
    s, l = stack(frames), latest(frames)
    D = enc.state_dim
    assert s.shape == (2, 3 * D) and l.shape == (2, D)
    for t in range(3):
        torch.testing.assert_close(s[:, t * D:(t + 1) * D], latest(frames[:, : t + 1]), atol=2e-2, rtol=2e-2)
    torch.testing.assert_close(s[:, 2 * D:], l)
    changed = frames.clone()
    changed[:, 0] = torch.rand_like(changed[:, 0])
    s2 = stack(changed)
    torch.testing.assert_close(s2[:, D:], s[:, D:])  # only the oldest block changes
    assert not torch.allclose(s2[:, :D], s[:, :D])
    torch.testing.assert_close(latest(changed), l)  # the historical T=1 input ignores older frames
