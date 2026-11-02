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
    torch.manual_seed(5)
    assert torch.equal(out["visible"], sample_tube_mask(enc.patch_scores(score), 0.5, enc.cfg.motion_patch_threshold))
    rnd = Encoder(EncoderConfig(image_height=16, image_width=16, patch_size=4, width=16, depth=1, heads=4,
                                num_slots=2, slot_dim=8, drop_path=0.0, mask_mode="random", mask_ratio=0.75)).train()
    torch.manual_seed(6)
    a = rnd(x, score)["visible"]
    torch.manual_seed(6)
    assert torch.equal(a, sample_random_tokens(3, 48, 12, "cpu"))
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
