"""Variant selfrender_replace_v2 (user requirement 4 of 2026-10-29 with the amendment of the same day): decoded-Gaussian
self-rendered views replace csw4's point-cloud synthetic-view bundle; contribution 0 up to 20k, linear to 1 at 40k;
cameras uniform over the frozen training-supported viewpoint domain. Every new key defaults to the reference."""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pytest
import torch
import yaml

from s4d.config import load_config
from s4d.losses.invariance import multi_positive_info_nce, slot_consistency
from s4d.model import encoder as encoder_mod
from s4d.model import render
from s4d.model.encoder import Encoder, EncoderConfig, visible_count
from s4d.train import augment, loop
from s4d.train.loop import build_model, forward_losses, selfrender_representation, selfrender_weight, synthetic_settings
from tests import _enc_golden as golden
from tests import _selfrender_golden as sr_golden
from tests.test_review_options import _fake_rasterizer

REPO = Path(__file__).resolve().parents[1]
VARIANTS = REPO / "configs/metaworld/variants"
CSW4_FILES = [REPO / "configs/metaworld/base.yaml", REPO / "configs/metaworld/tasks/hammer.yaml"] + [
    VARIANTS / v for v in sr_golden.CSW4]
SR = "selfrender_replace_v2.yaml"
DOMAIN = REPO / "configs/metaworld/selfrender_domain_v1.yaml"
SOURCE = torch.tensor([0, 1])


def _flat(d, prefix=()):
    out = {}
    for k, v in d.items():
        out.update(_flat(v, prefix + (k,)) if isinstance(v, dict) else {prefix + (k,): v})
    return out


@pytest.fixture
def patched():
    saved = golden._patched()
    try:
        yield
    finally:
        golden._restore(saved)


@pytest.fixture
def batch(tmp_path):
    return sr_golden.fixture_batch(tmp_path)


def _variant_cfg():
    return sr_golden.small_csw4_cfg(SR)


def _real_only_cfg():
    cfg = sr_golden.small_csw4_cfg()
    cfg["aug"].update(synth_views=0, synth_render=False)
    return cfg


def _step(cfg, batch, step, seed=1):
    torch.manual_seed(0)
    model = build_model(cfg).train()
    torch.manual_seed(seed)
    out = forward_losses(model, batch, cfg, step, source=SOURCE)
    return model, out


# ---------------------------------------------------------------- option off: the reference bit for bit
@pytest.mark.parametrize("explicit", [False, True])
def test_option_off_reproduces_csw4_bit_for_bit(tmp_path, explicit):
    """Against tests/data/selfrender_off_golden.json, written by the unchanged sources (795bec3): three csw4 training
    steps (point-cloud synthetic views, synthetic render targets, Chamfer, cs + w4 encoder, 75 % random masking): every
    loss, metric, gradient norm, the parameters after each step and the random-number state. Elsewhere (another CPU
    may round BLAS sums differently) the platform-independent parts must match."""
    expected = json.loads(sr_golden.GOLDEN.read_text())
    got = sr_golden.fingerprints(tmp_path, {"aug.synth_source": "pointcloud"} if explicit else None)
    if got["environment"] == expected["environment"]:
        assert got == expected
    else:
        for key in ("optimizer_groups", "rng_after_steps"):
            assert got[key] == expected[key], key
        assert [sorted(s["losses"]) for s in got["steps"]] == [sorted(s["losses"]) for s in expected["steps"]]
        assert [sorted(s["metrics"]) for s in got["steps"]] == [sorted(s["metrics"]) for s in expected["steps"]]


def test_variant_config_changes_only_the_registered_keys():
    ref = load_config(CSW4_FILES)
    cfg = load_config(CSW4_FILES + [VARIANTS / SR])
    base, new = _flat(ref), _flat(cfg)
    diff = {k: new.get(k) for k in set(new) | set(base) if new.get(k) != base.get(k)}
    assert diff == {
        ("aug", "synth_source"): "gaussian_self_render",
        ("aug", "synth_render"): False,
        ("aug", "synth_ramp_steps"): [20000, 40000],
        ("aug", "synth_camera_domain"): "configs/metaworld/selfrender_domain_v1.yaml",
    }
    assert ref["aug"]["synth_render"] is True and "synth_source" not in ref["aug"]
    assert cfg["aug"]["synth_views"] == 2 and "synth_range_scale" not in cfg["aug"]
    enc = cfg["model"]["encoder"]
    assert (enc["mask_mode"], enc["mask_ratio"], enc["stem"], enc["width"], enc["depth"]) == ("random", 0.75, "patch_cnn",
                                                                                              256, 4)
    assert cfg["loss"]["chamfer"] == 5.0 and "self_render" not in cfg["loss"]
    assert (cfg["loss"]["invariance"], cfg["loss"]["consistency"], cfg["loss"]["temperature"]) == (1.0, 0.5, 0.1)


# ---------------------------------------------------------------- schedule and zero-weight exclusion
def test_ramp_values():
    ramp = (20000, 40000)
    assert [selfrender_weight(s, ramp) for s in (0, 20000, 30000, 40000, 200000)] == [0.0, 0.0, 0.5, 1.0, 1.0]
    assert 0.0 < selfrender_weight(20001, ramp) < selfrender_weight(39999, ramp) < 1.0
    assert selfrender_weight(0, None) == 1.0
    cfg = load_config(CSW4_FILES + [VARIANTS / SR])
    assert synthetic_settings(cfg["aug"], future=False, hardneg=False)["ramp"] == ramp


@pytest.mark.parametrize("step", [0, 20000])
def test_zero_weight_step_equals_real_only_csw4(batch, patched, monkeypatch, step):
    """At weight 0 nothing is sampled, rendered or encoded: losses, gradients, rasterizer calls and the random-number
    state equal csw4 without any synthetic view (InfoNCE over the 6 real views only, including its normalisation)."""
    calls = {}
    for name in ("real", "sr"):
        calls[name] = []
        render._rasterize = _fake_rasterizer(calls[name])
        if name == "sr":
            for fn in ("domain_cameras", "jitter_cameras", "synthesize_views", "self_render_positives"):
                monkeypatch.setattr(loop, fn, lambda *a, _fn=fn, **k: (_ for _ in ()).throw(AssertionError(_fn)))
        model, out = _step(_real_only_cfg() if name == "real" else _variant_cfg(), batch, step)
        out["total"].backward()
        calls[name + "_grads"] = [p.grad for p in model.parameters()]
        calls[name + "_out"], calls[name + "_rng"] = out, torch.get_rng_state()
    real, sr = calls["real_out"], calls["sr_out"]
    assert set(sr["losses"]) == set(real["losses"]) | {"infonce_real", "consistency_real"}
    for key, value in real["losses"].items():
        assert torch.equal(value, sr["losses"][key]), key
    assert torch.equal(sr["losses"]["infonce_real"], real["losses"]["infonce"])
    assert all((a is None and b is None) or torch.equal(a, b) for a, b in zip(calls["real_grads"], calls["sr_grads"]))
    assert calls["real"] == calls["sr"] and torch.equal(calls["real_rng"], calls["sr_rng"])
    assert float(sr["metrics"]["selfrender_weight"]) == 0.0 and "synth_alpha_mean" not in sr["metrics"]
    assert "synth_render" not in sr["losses"]


def test_weighted_objective_is_real_plus_weighted_contribution():
    torch.manual_seed(0)
    slots = torch.randn(4, 6, 1, 8, requires_grad=True)
    syn = torch.randn(4, 2, 1, 8, requires_grad=True)
    real_nce, _ = multi_positive_info_nce(slots.mean(2), 0.1, distributed=False)
    real_cons = slot_consistency(slots)
    joint = torch.cat((slots, syn), 1)
    joint_nce, _ = multi_positive_info_nce(joint.mean(2), 0.1, distributed=False)
    joint_cons = slot_consistency(joint)
    # weight 0: the generated encodings are absent (no logit, positive, negative or normalisation term)
    nce, cons, _, extra = selfrender_representation(slots, syn, 0.0, 0.1, False)
    assert torch.equal(nce, real_nce) and torch.equal(cons, real_cons) and "infonce_joint" not in extra
    assert torch.autograd.grad(nce + cons, syn, allow_unused=True)[0] is None
    nce, cons, _, _ = selfrender_representation(slots, None, 0.0, 0.1, False)
    assert torch.equal(nce, real_nce) and torch.equal(cons, real_cons)
    # weight 1: exactly the joint objective (csw4's form with the point-cloud encodings replaced)
    nce, cons, metrics, extra = selfrender_representation(slots, syn, 1.0, 0.1, False)
    assert torch.equal(nce, joint_nce) and torch.equal(cons, joint_cons)
    assert torch.equal(extra["infonce_real"], real_nce.detach()) and "selfrender_positive_cosine" in metrics
    # in between: the real-view term with weight 1, the self-render contribution scaled
    for w in (0.25, 0.5):
        nce, cons, _, _ = selfrender_representation(slots, syn, w, 0.1, False)
        assert torch.allclose(nce, real_nce + w * (joint_nce - real_nce), atol=1e-6)
        assert torch.allclose(cons, real_cons + w * (joint_cons - real_cons), atol=1e-6)
        (g_syn,) = torch.autograd.grad(nce, syn, retain_graph=True)
        (g_joint,) = torch.autograd.grad(joint_nce, syn, retain_graph=True)
        assert torch.allclose(g_syn, w * g_joint, atol=1e-6)  # the generated encodings' gradient scales with w


# ---------------------------------------------------------------- the self-render path
def _recording_rasterizer(log):
    fake = _fake_rasterizer([])

    def record(**kw):
        log.append({"mode": kw["render_mode"], "V": kw["viewmats"].shape[-3], "grad": torch.is_grad_enabled(),
                    "viewmats": kw["viewmats"].detach().clone()})
        return fake(**kw)

    return record


@pytest.mark.parametrize("step", [30000, 50000])
def test_point_cloud_branch_is_never_used(batch, patched, monkeypatch, step):
    for target, name in ((loop, "synthesize_views"), (augment, "fused_training_clouds"), (augment, "splat_batched"),
                         (loop, "jitter_cameras")):
        monkeypatch.setattr(target, name, lambda *a, _n=name, **k: (_ for _ in ()).throw(AssertionError(_n)))
    _, out = _step(_variant_cfg(), batch, step)
    assert torch.isfinite(out["total"]) and "synth_render" not in out["losses"]
    assert {"infonce_joint", "consistency_joint"} <= set(out["losses"])
    assert float(out["metrics"]["selfrender_weight"]) == selfrender_weight(step, (20000, 40000))


def test_renders_are_detached_fixed_over_time_inside_the_domain_and_reencoded(batch, patched, monkeypatch):
    log, enc_inputs, poses = [], [], []
    render._rasterize = _recording_rasterizer(log)
    orig_visible = Encoder.forward_visible

    def forward_visible(self, images, visible):
        enc_inputs.append((images, visible))
        return orig_visible(self, images, visible)

    orig_cams = loop.domain_cameras

    def cams(*a, **k):
        res = orig_cams(*a, **k)
        poses.append(res["poses"])
        return res

    monkeypatch.setattr(Encoder, "forward_visible", forward_visible)
    monkeypatch.setattr(loop, "domain_cameras", cams)
    cfg = _variant_cfg()
    _, out = _step(cfg, batch, 50000)
    rgbd = [c for c in log if c["mode"] == "RGB+ED"]
    assert [(c["V"], c["grad"]) for c in rgbd] == [(2, False), (2, True)]  # the self-render, then training cameras
    vm = rgbd[0]["viewmats"]  # (B,S,n,4,4): one camera per view for every reconstruction time
    assert vm.shape[1] == 3 and torch.equal(vm[:, 0], vm[:, 1]) and torch.equal(vm[:, 0], vm[:, 2])
    (images, visible), = enc_inputs
    B, n = batch["images"].shape[0], 2
    assert not images.requires_grad and images.grad_fn is None and images.shape == (B * n, 3, 3, 32, 32)
    N = 4  # 32 x 32 images, 16 px patches
    assert visible.shape == (B * n, 3 * N) and (visible.sum(1) == visible_count(3 * N, 0.75)).all()
    domain = augment.load_viewpoint_domain(DOMAIN)
    assert _inside(poses[0].reshape(-1, 3).numpy(), domain).all()
    assert torch.equal(torch.as_tensor(augment.orbit_rig([tuple(p) for p in poses[0].reshape(-1, 3).tolist()], 32, 32)
                                       ["w2c"]).view(B, n, 4, 4), vm[:, 0].float())
    for key in ("synth_alpha_mean", "synth_token_eligible_fraction", "synth_coverage_exception_rows",
                "synth_kept_ineligible_fraction"):
        assert torch.isfinite(out["metrics"][key]), key


def test_decoder_gets_no_gradient_from_the_representation_term(batch, patched, monkeypatch):
    """With the representation term isolated (InfoNCE + consistency incl. the self-rendered positives) the decoder's
    gradient is exactly zero while the encoder's is not; the self-rendered encodings alone reach only the encoder; and
    the decoder gradient of the full loss is bit-identical with the representation weights set to zero."""
    captured = {}
    orig = loop.selfrender_representation

    def capture(slots, syn_slots, weight, temperature, distributed):
        res = orig(slots, syn_slots, weight, temperature, distributed)
        captured.update(slots=slots, syn=syn_slots, nce=res[0], cons=res[1])
        return res

    monkeypatch.setattr(loop, "selfrender_representation", capture)
    cfg = _variant_cfg()
    model, out = _step(cfg, batch, 50000)
    rep = 1.0 * captured["nce"] + 0.5 * captured["cons"]
    dec, enc = list(model.decoder.parameters()), list(model.encoder.parameters())
    assert all(g is None for g in torch.autograd.grad(rep, dec, retain_graph=True, allow_unused=True))
    assert sum(float(g.abs().sum()) for g in torch.autograd.grad(rep, enc, retain_graph=True, allow_unused=True)
               if g is not None) > 0
    assert torch.autograd.grad(rep, captured["syn"], retain_graph=True)[0].abs().sum() > 0
    syn_sum = captured["syn"].sum()
    assert all(g is None for g in torch.autograd.grad(syn_sum, dec, retain_graph=True, allow_unused=True))
    assert sum(float(g.abs().sum()) for g in torch.autograd.grad(syn_sum, enc, retain_graph=True, allow_unused=True)
               if g is not None) > 0
    out["total"].backward()
    dec_full = [p.grad.clone() for p in model.decoder.parameters()]
    enc_full = [None if p.grad is None else p.grad.clone() for p in model.encoder.parameters()]
    cfg0 = _variant_cfg()
    cfg0["loss"].update(invariance=0.0, consistency=0.0)
    model0, out0 = _step(cfg0, batch, 50000)
    out0["total"].backward()
    assert all(torch.equal(a, p.grad) for a, p in zip(dec_full, model0.decoder.parameters()))
    assert any(a is not None and not torch.equal(a, p.grad) for a, p in zip(enc_full, model0.encoder.parameters()))


def test_real_view_masking_is_unchanged(batch, patched, monkeypatch):
    """The real views keep the 75 % random temporal-token masking (one draw over the B x V rows), as in csw4."""
    seen = []
    original = encoder_mod.sample_random_tokens

    def recording(rows, tokens, keep, device, generator=None):
        seen.append((rows, tokens, keep))
        return original(rows, tokens, keep, device, generator)

    monkeypatch.setattr(encoder_mod, "sample_random_tokens", recording)
    for cfg in (sr_golden.small_csw4_cfg(), _variant_cfg()):
        seen.clear()
        _step(cfg, batch, 50000)
        B, V = batch["images"].shape[0], batch["images"].shape[2]
        assert seen == [(B * V, 12, visible_count(12, 0.75))]


def _stub_render(alpha):
    def fake(gs, xyz_seq, w2c, K, H, W, near, far):
        B, S, n = xyz_seq.shape[0], xyz_seq.shape[1], w2c.shape[1]
        return {"rgb": torch.rand(B, S, n, 3, H, W), "depth": torch.ones(B, S, n, 1, H, W), "alpha": alpha}

    return fake


@pytest.mark.parametrize("eligible_patches,exception", [(3, 0.0), (1, 1.0)])
def test_alpha_coverage_rule_and_insufficient_coverage(monkeypatch, eligible_patches, exception):
    """Tokens are kept among patches whose mean detached alpha is >= 0.9 in their own frame; a row with fewer eligible
    tokens than the 75 % visible count is filled with ineligible tokens (the existing coverage exception)."""
    enc = Encoder(EncoderConfig(image_height=32, image_width=32, patch_size=16, width=16, depth=1, heads=2, slot_dim=8,
                                mask_mode="random", mask_ratio=0.75)).train()
    B, n, T = 2, 2, 3
    alpha = torch.full((B, T, n, 1, 32, 32), 0.5)
    for p in range(eligible_patches):  # the first patches (16 x 16 blocks, row-major) of frame 0 fully covered
        r, c = divmod(p, 2)
        alpha[:, 0, :, :, 16 * r:16 * (r + 1), 16 * c:16 * (c + 1)] = 1.0
    monkeypatch.setattr(loop, "render_rgbd", _stub_render(alpha))
    seen = []
    orig = Encoder.forward_visible
    monkeypatch.setattr(Encoder, "forward_visible", lambda self, x, v: seen.append(v) or orig(self, x, v))
    cams = {"w2c": torch.eye(4).expand(B, n, 4, 4), "K": torch.eye(3).expand(B, n, 3, 3)}
    xyz = torch.zeros(B, T, 4, 3)
    slots, stats = loop.self_render_positives(enc, None, xyz, cams, 3, None, False, 0.05, 3.0)
    assert slots.shape == (B, n, 1, 8)
    (visible,) = seen
    keep = visible_count(12, 0.75)
    eligible = torch.zeros(12, dtype=torch.bool)
    eligible[:eligible_patches] = True  # frame-major: frame 0's patches first
    assert (visible.sum(1) == keep).all()
    if eligible_patches >= keep:
        assert not (visible & ~eligible).any()
    else:
        assert (visible[:, eligible]).all()
    assert float(stats["synth_coverage_exception_rows"]) == exception
    assert math.isclose(float(stats["synth_token_eligible_fraction"]), eligible_patches / 12, rel_tol=1e-6)
    assert math.isclose(float(stats["synth_kept_ineligible_fraction"]), max(0, keep - eligible_patches) / keep,
                        rel_tol=1e-6)


# ---------------------------------------------------------------- viewpoint domain and sampler
def _inside(poses: np.ndarray, domain: dict) -> np.ndarray:
    d_az = (poses[:, None, 0] - domain["cells"][None, :, 0] + 180.0) % 360.0 - 180.0
    d_el = poses[:, None, 1] - domain["cells"][None, :, 1]
    tol = domain["half"] + 1e-9
    return ((np.abs(d_az) <= tol) & (np.abs(d_el) <= tol)).any(1) & np.isclose(poses[:, 2], domain["distance"])


def test_sampler_stays_inside_the_frozen_domain():
    domain = augment.load_viewpoint_domain(DOMAIN)
    torch.manual_seed(0)
    K = torch.tensor([[154.5, 0.0, 64.0], [0.0, 154.5, 64.0], [0.0, 0.0, 1.0]]).expand(500, 3, 3)
    cams = augment.domain_cameras(500, 2, K, 128, 128, domain)
    poses = cams["poses"].reshape(-1, 3).numpy()
    assert cams["w2c"].shape == (500, 2, 4, 4) and cams["K"].shape == (500, 2, 3, 3)
    assert _inside(poses, domain).all()
    assert poses[:, 1].min() >= domain["cells"][:, 1].min() - 2.5 and poses[:, 1].max() <= domain["cells"][:, 1].max() + 2.5
    rig = augment.orbit_rig([tuple(p) for p in poses[:4]], 128, 128)
    assert np.allclose(rig["w2c"], cams["w2c"].reshape(-1, 4, 4)[:4].numpy(), atol=1e-5)
    c2w = cams["c2w"].reshape(-1, 4, 4)
    assert torch.allclose(c2w[:, :3, 3].sub(torch.tensor([0.0, 0.6, 0.0])).norm(dim=-1), torch.ones(1000), atol=1e-4)
    assert math.isclose(float(domain["probs"].sum()), 1.0, rel_tol=1e-9)
    # many draws reach many cells (uniform on the sphere over the domain, not a few poses)
    cells = {tuple(c) for c in domain["cells"][np.argmin(np.abs(poses[:, None, :2] - domain["cells"][None]).sum(-1), 1)]}
    assert len(cells) > 0.5 * len(domain["cells"])


def test_frozen_domain_file_is_consistent():
    doc = yaml.safe_load(DOMAIN.read_text())
    crit = doc["criteria"]
    cells = np.asarray(doc["cells"], dtype=float)
    assert doc["version"] == "selfrender_domain_v1" and len(cells) == doc["summary"]["domain_cells"] == 314
    for j, key in enumerate(("coverage_min", "support2_min", "ws_visible_min", "ws_support2_min")):
        assert (cells[:, 2 + j] >= crit[key] - 1e-3).all(), key
    centres = {(int(a), int(e)) for a, e in cells[:, :2]}
    assert all((int(a), int(e)) in centres for a, e in doc["rig"]["train_poses_az_el"])
    assert doc["rig"]["lookat"] == [0.0, 0.6, 0.0] and doc["rig"]["distance"] == 1.0
    assert doc["analysis"]["cameras"] == "training cameras 0-5 only"


# ---------------------------------------------------------------- RGB-history-only encoder and invalid combinations
def test_encoder_sees_rgb_history_only(batch, patched, monkeypatch):
    seen = []
    orig_encode = Encoder._encode

    def encode(self, images, motion_score, mask_ratio, visible):
        seen.append(tuple(images.shape))
        return orig_encode(self, images, motion_score, mask_ratio, visible)

    monkeypatch.setattr(Encoder, "_encode", encode)
    assert "actions" not in batch
    _step(_variant_cfg(), batch, 50000)
    B, V = batch["images"].shape[0], batch["images"].shape[2]
    assert seen == [(B * V, 3, 3, 32, 32), (B * 2, 3, 3, 32, 32)]
    ref, sr = load_config(CSW4_FILES), load_config(CSW4_FILES + [VARIANTS / SR])
    assert ref["model"] == sr["model"] and get_action_dim(sr) == 0


def get_action_dim(cfg):
    return int(cfg["model"]["decoder"].get("action_dim", 0))


def test_invalid_combinations_are_rejected():
    ok = {"synth_source": "gaussian_self_render"}
    with pytest.raises(ValueError, match="never reconstruction targets"):
        synthetic_settings({**ok, "synth_render": True}, future=False, hardneg=False)
    with pytest.raises(ValueError, match="reconstruction objective"):
        synthetic_settings(ok, future=True, hardneg=False)
    with pytest.raises(ValueError, match="hardneg"):
        synthetic_settings(ok, future=False, hardneg=True)
    with pytest.raises(ValueError, match="gaussian_self_render only"):
        synthetic_settings({"synth_ramp_steps": [0, 1]}, future=False, hardneg=False)
    with pytest.raises(ValueError, match="synth_source"):
        synthetic_settings({"synth_source": "mesh"}, future=False, hardneg=False)
    with pytest.raises(ValueError, match="start <= full"):
        synthetic_settings({**ok, "synth_ramp_steps": [5, 1]}, future=False, hardneg=False)
    cfg = load_config(CSW4_FILES + [VARIANTS / SR])
    cfg["model"]["perframe"] = {"temporal": False}
    with pytest.raises(ValueError, match="per-frame"):
        forward_losses(None, {}, cfg, 0)
    assert synthetic_settings({}, future=True, hardneg=True) == {"source": "pointcloud"}  # the reference reads nothing


def test_legacy_self_render_loss_resolves_alongside_the_replacement_source_and_is_guarded_in_training():
    import pytest
    from s4d.config import load_config
    from s4d.train import loop

    files = ["configs/metaworld/base.yaml", "configs/metaworld/tasks/hammer.yaml",
             "configs/metaworld/variants/chamfer_lean.yaml", "configs/metaworld/variants/chamfer_lean_randmask.yaml",
             "configs/metaworld/variants/enc_cs.yaml", "configs/metaworld/variants/enc_w4.yaml",
             "configs/metaworld/variants/selfrender_replace_v2.yaml"]
    cfg = load_config([loop.REPO / f if hasattr(loop, "REPO") else f for f in files], ["loss.self_render=0.5"])
    assert cfg["loss"]["self_render"] == 0.5 and cfg["aug"]["synth_source"] == "gaussian_self_render"
