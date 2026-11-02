"""75 % hybrid masking (mask_mode "hybrid", user directive 2026-11-03): 4 motion-aware tube positions shared by the three
frames + 12 random positions per frame, disjoint across frames; exactly 48 of 192 temporal tokens, 16 per frame; the
same scheme on the synthetic point-cloud views with the existing coverage eligibility and exception; the variant
changes only the masking of chlrm."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch
import torch.nn.functional as F

from s4d.config import load_config
from s4d.model.encoder import (
    HYBRID_TUBE_PATCHES,
    Encoder,
    EncoderConfig,
    hybrid_counts,
    sample_hybrid_mask,
    sample_random_tokens,
    sample_tube_mask,
    visible_count,
)
from s4d.train.augment import (
    PATCH_COVERAGE,
    coverage_visible_tokens,
    hybrid_synthetic_visible,
    synthesize_views,
    synthesize_views_with_motion,
)
from s4d.train.loop import build_model, forward_losses

REPO = Path(__file__).resolve().parents[1]
VARIANTS = REPO / "configs/metaworld/variants"
T, N = 3, 64
THR = 0.05


def _parts(vis: torch.Tensor):
    v = vis.view(-1, T, N)
    tube = v.all(1)
    rand = v & ~tube[:, None]
    return v, tube, rand


def test_counts_match_the_design():
    assert HYBRID_TUBE_PATCHES == 4
    assert hybrid_counts(3, 64, 0.75) == (16, 12)
    assert visible_count(192, 0.75) == 48
    with pytest.raises(ValueError):
        hybrid_counts(3, 64, 0.8)  # 38 visible tokens do not split equally over three frames


def test_real_views_keep_48_tokens_16_per_frame_4_shared_and_disjoint_random_sets():
    g = torch.Generator().manual_seed(0)
    scores = torch.rand(500, N, generator=g)
    vis, info = sample_hybrid_mask(scores, T, 0.75, THR, generator=g)
    assert vis.shape == (500, T * N) and vis.dtype == torch.bool
    v, tube, rand = _parts(vis)
    assert (vis.sum(1) == 48).all() and (v.sum(2) == 16).all()
    assert (tube.sum(1) == 4).all() and torch.equal(tube, info["tube"])
    assert (rand.sum(2) == 12).all()
    assert (rand.sum(1) <= 1).all()  # every non-tube position is in at most one frame's random set
    assert (v.any(1).sum(1) == 40).all() and (info["positions_seen"] == 40).all()
    assert (info["hole_tokens"] == 0).all() and (info["random_overlap"] == 0).all()


def test_tube_takes_the_highest_scoring_reliable_dynamic_patches():
    scores = torch.zeros(2, N)
    scores[0, [3, 10, 20, 30, 40, 50]] = torch.tensor([0.2, 0.9, 0.6, 0.04, 0.7, 0.3])  # 0.04 is below the threshold
    scores[1, [7, 9]] = torch.tensor([0.5, 0.06])  # only two reliable dynamic patches: random fill for the other two
    vis, info = sample_hybrid_mask(scores, T, 0.75, THR, generator=torch.Generator().manual_seed(1))
    _, tube, _ = _parts(vis)
    assert set(tube[0].nonzero().flatten().tolist()) == {10, 40, 20, 50}
    assert {7, 9} <= set(tube[1].nonzero().flatten().tolist()) and int(tube[1].sum()) == 4
    assert info["tube_dynamic"].tolist() == [4, 2] and info["dynamic_available"].tolist() == [5, 2]


def test_tube_ranking_respects_small_score_differences_and_breaks_exact_ties_at_random():
    scores = torch.zeros(4000, N)
    scores[:, :5] = torch.tensor([0.9, 0.8, 0.7, 0.5001, 0.5])
    vis, _ = sample_hybrid_mask(scores, T, 0.75, THR, generator=torch.Generator().manual_seed(3))
    assert bool(_parts(vis)[1][:, :4].all())  # 0.5001 beats 0.5 in every row
    tied = torch.zeros(4000, N)
    tied[:, :8] = 1.0  # eight clamped (equal) scores for four tube slots: uniform among them
    vis, _ = sample_hybrid_mask(tied, T, 0.75, THR, generator=torch.Generator().manual_seed(4))
    tube = _parts(vis)[1]
    assert bool((tube[:, :8].sum(1) == 4).all())
    assert float((tube[:, :8].float().mean(0) - 0.5).abs().max()) < 0.03


def test_without_motion_the_tube_is_uniform_and_static_patches_keep_coverage():
    g = torch.Generator().manual_seed(2)
    rows = 20000
    scores = torch.zeros(rows, N)
    scores[:, 0] = 1.0  # one fixed dynamic patch: always in the tube
    vis, info = sample_hybrid_mask(scores, T, 0.75, THR, generator=g)
    v, tube, rand = _parts(vis)
    assert bool(tube[:, 0].all()) and (info["tube_dynamic"] == 1).all()
    # the other three tube positions and all random sets are uniform over the 63 static positions
    p_tube = tube[:, 1:].float().mean(0)
    assert float((p_tube - 3 / 63).abs().max()) < 0.01
    p_seen = v[:, :, 1:].any(1).float().mean(0)  # a static position is seen in a sample with probability 39/63
    assert float((p_seen - 39 / 63).abs().max()) < 0.02
    p_frame = rand[:, :, 1:].float().mean((0,))  # per frame and position: 12 of the 60 non-tube positions
    assert float((p_frame - (60 / 63) * 12 / 60).abs().max()) < 0.02


def test_encoder_hybrid_mode_gathers_48_tokens_in_training_and_none_in_eval():
    enc = Encoder(EncoderConfig(image_height=16, image_width=16, patch_size=2, width=16, depth=1, heads=4, num_slots=1,
                                slot_dim=8, drop_path=0.0, mask_mode="hybrid", mask_ratio=0.75)).train()
    assert enc.cfg.num_patches == N
    x = torch.rand(4, 3, 3, 16, 16)
    score = torch.zeros(4, 3, 1, 16, 16)
    score[:, 1, :, 0:2, 0:2] = 1.0  # patch 0 moves in frame 1 only: the max over time still selects it
    out = enc(x, score)
    assert out["visible"].shape == (4, T * N) and (out["visible"].sum(1) == 48).all()
    assert out["patch_tokens"].shape[1] == 48
    assert bool(out["visible"].view(4, T, N)[:, :, 0].all())
    torch.manual_seed(0)
    a = enc(x, score)["visible"]
    torch.manual_seed(0)
    b = sample_hybrid_mask(enc.patch_scores(score), T, 0.75, enc.cfg.motion_patch_threshold)[0]
    assert torch.equal(a, b)
    ev = enc.eval()(x, score)
    assert ev["visible"].all() and ev["patch_tokens"].shape[1] == T * N
    with pytest.raises(ValueError):
        Encoder(EncoderConfig(image_height=16, image_width=16, patch_size=2, width=16, depth=1, heads=4,
                              mask_mode="hybrid", mask_ratio=0.8))


def test_tube_and_random_modes_are_unchanged():
    enc = Encoder(EncoderConfig(image_height=16, image_width=16, patch_size=4, width=16, depth=1, heads=4,
                                num_slots=2, slot_dim=8, drop_path=0.0, mask_ratio=0.5)).train()
    x, score = torch.rand(3, 3, 3, 16, 16), torch.rand(3, 3, 1, 16, 16)
    torch.manual_seed(5)
    out = enc(x, score)
    after_enc = torch.get_rng_state()
    torch.manual_seed(5)
    assert torch.equal(out["visible"], sample_tube_mask(enc.patch_scores(score), 0.5, enc.cfg.motion_patch_threshold))
    assert torch.equal(after_enc, torch.get_rng_state())  # the tube mode consumes exactly its sampler's draws
    rnd = Encoder(EncoderConfig(image_height=16, image_width=16, patch_size=4, width=16, depth=1, heads=4,
                                num_slots=2, slot_dim=8, drop_path=0.0, mask_mode="random", mask_ratio=0.75)).train()
    torch.manual_seed(6)
    a = rnd(x, score)["visible"]
    after_enc = torch.get_rng_state()
    torch.manual_seed(6)
    assert torch.equal(a, sample_random_tokens(3, 48, 12, "cpu"))
    assert torch.equal(after_enc, torch.get_rng_state())
    assert EncoderConfig().mask_mode == "tube"


def _covered(rows: int, frame0_patches: list[int]) -> torch.Tensor:
    """(rows,3,1,16,16) coverage with 2x2-pixel patches: frames 1-2 fully covered, frame 0 only ``frame0_patches``."""
    cov = torch.ones(rows, T, 1, 16, 16)
    cov[:, 0] = 0.0
    for p in frame0_patches:
        r, c = divmod(p, 8)
        cov[:, 0, :, 2 * r:2 * r + 2, 2 * c:2 * c + 2] = 1.0
    return cov


def test_synthetic_views_respect_eligibility_and_use_the_transported_motion():
    cov = _covered(200, list(range(32)))  # frame 0: top half covered
    score = torch.zeros(200, T, 1, 16, 16)
    score[:, :, :, 0:2, 2:4] = 0.8  # patch 1 (covered everywhere) moves
    score[:, :, :, 14:16, 14:16] = 0.9  # patch 63 moves but is a hole in frame 0: never a tube position
    vis, info = hybrid_synthetic_visible(cov, score, 2, 0.75, THR)
    v, tube, rand = _parts(vis)
    frac = F.avg_pool2d(cov.flatten(0, 1), 2).view(200, T, N)
    assert (vis.sum(1) == 48).all() and (v.sum(2) == 16).all() and (tube.sum(1) == 4).all()
    assert bool(tube[:, 1].all()) and not bool(tube[:, 63].any())
    assert bool((frac[v] >= PATCH_COVERAGE).all()) and (info["hole_tokens"] == 0).all()  # enough eligible tokens
    assert (rand.sum(1) <= 1).all()


def test_synthetic_coverage_exception_keeps_the_budget_and_counts_holes():
    cov = _covered(50, list(range(10)))  # frame 0: only 10 eligible positions for 16 tokens
    vis, info = hybrid_synthetic_visible(cov, torch.zeros(50, T, 1, 16, 16), 2, 0.75, THR)
    v, tube, _ = _parts(vis)
    frac = F.avg_pool2d(cov.flatten(0, 1), 2).view(50, T, N)
    assert (v.sum(2) == 16).all() and (vis.sum(1) == 48).all()
    assert bool((tube.nonzero()[:, 1] < 10).all())  # the tube stays on positions eligible in every frame
    assert (info["hole_tokens"] == 6).all()  # 16 - 10 holes, all in frame 0
    assert bool((frac[:, 1:][v[:, 1:]] >= PATCH_COVERAGE).all())
    few = _covered(20, [0, 1])  # only two positions eligible in every frame: two tube positions are hole tokens
    vis2, info2 = hybrid_synthetic_visible(few, torch.zeros(20, T, 1, 16, 16), 2, 0.75, THR)
    v2, tube2, _ = _parts(vis2)
    assert (v2.sum(2) == 16).all() and bool(tube2[:, :2].all()) and (tube2.sum(1) == 4).all()
    assert (info2["hole_tokens"] == 14).all()  # frame 0 has 2 eligible tokens for 16
    # the random mode's existing exception is untouched
    assert (coverage_visible_tokens(cov, 2, 48).sum(1) == 48).all()


def _single_camera(batch: dict) -> dict:
    out = dict(batch)
    for k in ("images", "depth", "motion_score"):
        out[k] = batch[k][:, :, :1]
    for k in ("K", "w2c", "c2w"):
        out[k] = batch[k][:, :1]
    return out


def test_motion_transport_splats_with_the_rgb_geometry(batch):
    one = _single_camera(batch)
    one["motion_score"] = torch.rand_like(one["motion_score"])
    cams = {"K": one["K"], "w2c": one["w2c"]}  # the synthetic camera = the training camera itself
    ref = synthesize_views(one, cams, 0.05, 3.0)
    out = synthesize_views_with_motion(one, cams, 0.05, 3.0)
    for k in ("images", "covered", "depth"):
        assert torch.equal(out[k], ref[k]), k
    cov = out["covered"].bool()
    assert cov.float().mean() > 0.9
    torch.testing.assert_close(out["motion_score"][cov], one["motion_score"][cov])
    assert (out["motion_score"][~cov] == 0).all()


def _cfg(variant: str):
    from tests.test_masking_perframe import _cfg as small

    cfg = small(variant)
    cfg["model"]["encoder"]["patch_size"] = 2  # 16x16 images -> 64 patches, 192 temporal tokens (as 128 / 16)
    return cfg


def test_hybrid_variant_differs_from_chlrm_only_in_the_mask_mode():
    base = [REPO / "configs/metaworld/base.yaml", REPO / "configs/metaworld/tasks/hammer.yaml", VARIANTS / "chamfer_lean.yaml"]
    rm = load_config(base + [VARIANTS / "chamfer_lean_randmask.yaml"])
    hy = load_config(base + [VARIANTS / "chamfer_lean_hybrid75.yaml"])

    def flat(d, p=""):
        o = {}
        for k, v in d.items():
            o.update(flat(v, f"{p}{k}.") if isinstance(v, dict) else {f"{p}{k}": v})
        return o
    a, b = flat(rm), flat(hy)
    assert {k for k in a.keys() | b.keys() if a.get(k) != b.get(k)} == {"model.encoder.mask_mode"}
    enc = hy["model"]["encoder"]
    assert (enc["mask_mode"], enc["mask_ratio"], enc["width"], enc["depth"], enc["heads"], enc["num_slots"],
            enc["slot_dim"]) == ("hybrid", 0.75, 128, 8, 8, 1, 256)
    assert "stem" not in enc or enc["stem"] in (None, "patchify")
    loss = hy["loss"]
    assert (loss["chamfer"], loss["depth_hard"], loss["depth_grad"], loss["visibility"], loss["occlusion"]) == (5.0, 0.0, 0.0, 0.0, 0.0)
    assert hy["aug"] == {"synth_views": 2, "synth_render": True}


def test_hybrid_variant_trains_on_real_and_synthetic_views(batch, monkeypatch):
    from tests.test_masking_perframe import _fake

    _fake(monkeypatch, [])
    cfg = _cfg("chamfer_lean_hybrid75.yaml")
    model = build_model(cfg).train()
    assert model.encoder.cfg.num_patches == N
    seen = []
    original = model.encoder.forward_visible

    def recording(images, visible):
        seen.append(visible)
        return original(images, visible)

    monkeypatch.setattr(model.encoder, "forward_visible", recording)
    out = forward_losses(model, batch, cfg, 100, source=torch.tensor([0, 1]))
    assert torch.isfinite(out["total"])
    assert len(seen) == 1  # the synthetic views (the real views are masked inside the encoder)
    syn = seen[0].view(-1, T, N)
    assert syn.shape[0] == batch["images"].shape[0] * cfg["aug"]["synth_views"] and (syn.sum(2) == 16).all()
    m = out["metrics"]
    assert float(m["mask_hybrid_per_frame_min"]) == float(m["mask_hybrid_per_frame_max"]) == 16.0
    assert float(m["mask_hybrid_tube_positions"]) == 4.0 and float(m["mask_hybrid_positions_seen"]) == 40.0
    for k in ("synth_token_eligible_fraction", "synth_coverage_exception_rows", "synth_kept_ineligible_fraction",
              "mask_hybrid_synth_tube_fill_rows", "mask_hybrid_synth_positions_seen"):
        assert k in m and torch.isfinite(m[k]), k


def test_hybrid_is_refused_where_it_is_not_defined():
    cfg = _cfg("chamfer_lean_hybrid75.yaml")
    cfg["model"]["perframe"] = {"temporal": False}
    cfg["model"]["encoder"]["num_frames"] = 1
    cfg["model"]["decoder"]["motion_head"] = False
    with pytest.raises(ValueError, match="hybrid"):
        build_model(cfg)


def test_most_constrained_frame_first_avoids_feasible_overlap():
    """Codex review case: frame 0 allows every position, frames 1 and 2 only 12 non-tube positions each; a disjoint
    allocation exists, and the most-constrained-first order finds it."""
    scores = torch.zeros(300, N)
    scores[:, :4] = 1.0  # tube 0..3
    e = torch.zeros(300, T, N, dtype=torch.bool)
    e[:, 0] = True
    e[:, 1, :16] = True
    e[:, 2, :4] = True
    e[:, 2, 16:28] = True
    vis, info = sample_hybrid_mask(scores, T, 0.75, THR, eligible=e, generator=torch.Generator().manual_seed(0))
    v, tube, rand = _parts(vis)
    assert bool(tube[:, :4].all()) and (v.sum(2) == 16).all()
    assert (info["random_overlap"] == 0).all() and (info["positions_seen"] == 40).all() and (info["hole_tokens"] == 0).all()
    assert bool(rand[:, 1, 4:16].all()) and bool(rand[:, 2, 16:28].all())


def test_real_view_frames_are_exchangeable():
    g = torch.Generator().manual_seed(9)
    vis, _ = sample_hybrid_mask(torch.zeros(30000, N), T, 0.75, THR, generator=g)
    v = vis.view(-1, T, N).float()
    per_frame = v.mean(0)  # every (frame, position) cell has the same marginal 16/64
    assert float((per_frame - 0.25).abs().max()) < 0.02


def test_coverage_boundary_matches_the_existing_rule():
    cov = torch.ones(4, T, 1, 80, 80)  # 10 x 10-pixel patches -> 64 positions
    cov[:, 0, :, 0:10, 0:1] = 0.0  # patch 0 of frame 0: 90 % covered (eligible, as >= 0.9)
    cov[:, 0, :, 0:10, 10:12] = 0.0  # patch 1 of frame 0: 80 % covered (ineligible)
    vis, info = hybrid_synthetic_visible(cov, torch.zeros(4, T, 1, 80, 80), 10, 0.75, THR)
    assert bool(info["eligible"][:, 0, 0].all()) and not bool(info["eligible"][:, 0, 1].any())
    frac = F.avg_pool2d(cov.flatten(0, 1), 10).view(4, T * N)
    assert torch.equal(info["eligible"].flatten(1), frac >= PATCH_COVERAGE)  # the random mode's rule, token by token


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_shared_splat_takes_rgb_and_value_from_one_winner(device):
    from s4d.diag.heldout import splat_batched, splat_batched_shared

    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("no CUDA")
    # 4096 points on one pixel at exactly equal depth, each with its own colour and value
    n = 4096
    pts = torch.zeros(1, n, 3, device=device)
    pts[..., 2] = 1.0
    ids = torch.arange(n, device=device, dtype=torch.float32)
    col = (ids / n).view(1, n, 1).expand(1, n, 3).contiguous()
    val = (ids / n).view(1, n)
    ok = torch.ones(1, n, dtype=torch.bool, device=device)
    K = torch.eye(3, device=device).view(1, 1, 3, 3)
    w2c = torch.eye(4, device=device).view(1, 1, 4, 4)
    outs = [splat_batched_shared(pts, col, val, ok, K, w2c, 1, 1, 0.05) for _ in range(5)]
    for rgb, cov, dep, v in outs:
        assert bool(cov.all()) and float(dep.flatten()[0]) == 1.0
        assert float(rgb.flatten()[0]) == float(v.flatten()[0]) == (n - 1) / n  # one winner: the highest point index
    # without ties the shared splat equals splat_batched exactly
    g = torch.Generator().manual_seed(0)
    pts = (torch.rand(2, 3000, 3, generator=g) * torch.tensor([2.0, 2.0, 1.0]) + torch.tensor([-1.0, -1.0, 0.5])).to(device)
    col = torch.rand(2, 3000, 3, generator=g).to(device)
    ok = torch.rand(2, 3000, generator=g).to(device) > 0.1
    K = torch.tensor([[8.0, 0, 8.0], [0, 8.0, 8.0], [0, 0, 1.0]], device=device).view(1, 1, 3, 3).expand(2, 2, 3, 3)
    w2c = torch.eye(4, device=device).view(1, 1, 4, 4).repeat(2, 2, 1, 1)
    w2c[:, 1, 0, 3] = 0.1
    a = splat_batched(pts, col, ok, K, w2c, 16, 16, 0.05)
    b = splat_batched_shared(pts, col, col[..., 0], ok, K, w2c, 16, 16, 0.05)
    for x, y in zip(a, b[:3]):
        assert torch.equal(x, y)
    assert torch.equal(b[3], b[0][:, :, :1])  # the value channel is the winner's own (here its red channel)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="no CUDA")
def test_hybrid_encoder_backward_under_cuda_bf16():
    enc = Encoder(EncoderConfig(image_height=16, image_width=16, patch_size=2, width=16, depth=1, heads=4, num_slots=1,
                                slot_dim=8, drop_path=0.0, mask_mode="hybrid", mask_ratio=0.75)).cuda().train()
    x = torch.rand(6, 3, 3, 16, 16, device="cuda")
    score = torch.rand(6, 3, 1, 16, 16, device="cuda")
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        out = enc(x, score)
    out["slots"].float().square().mean().backward()
    assert (out["visible"].view(6, T, N).sum(2) == 16).all()
    grads = [p.grad for p in enc.parameters() if p.grad is not None]
    assert grads and all(torch.isfinite(g).all() for g in grads)


@pytest.mark.parametrize("change", ["single_frame", "crop", "shift", "selfrender_source", "legacy_self_render"])
def test_every_hybrid_guard(batch, monkeypatch, change):
    from tests.test_masking_perframe import _fake

    _fake(monkeypatch, [])
    cfg = _cfg("chamfer_lean_hybrid75.yaml")
    if change == "single_frame":
        cfg["model"]["single_frame"] = True
    elif change == "crop":
        cfg["aug"]["crop"] = {"prob": 1.0}
    elif change == "shift":
        cfg["aug"]["shift"] = {"prob": 1.0, "pad": 2}
    elif change == "selfrender_source":
        cfg["aug"].update(synth_source="gaussian_self_render", synth_render=False)
    else:
        cfg["loss"]["self_render"] = 0.1
    model = build_model(cfg).train()
    with pytest.raises(ValueError, match="hybrid"):
        forward_losses(model, batch, cfg, 100, source=torch.tensor([0, 1]))


def test_other_modes_log_no_hybrid_metrics(batch, monkeypatch):
    from tests.test_masking_perframe import _fake

    _fake(monkeypatch, [])
    cfg = _cfg("chamfer_lean_randmask.yaml")
    out = forward_losses(build_model(cfg).train(), batch, cfg, 100, source=torch.tensor([0, 1]))
    assert not [k for k in out["metrics"] if k.startswith("mask_hybrid")]
