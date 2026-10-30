"""Variant selfrender_replace_v3 (user request 2026-10-31): selfrender_replace_v2 plus a radial distance jitter of up to
+-5 % on the self-render cameras, like the point-cloud synthetic views' radius_frac offset. The option off (key absent
or 0) reproduces v2 bit for bit; on, only the camera distance changes."""

from __future__ import annotations

import json

import numpy as np
import pytest
import torch

from s4d.config import load_config
from s4d.data.metaworld.cameras import HELDOUT_RANGES, LOOKAT
from s4d.train import augment, loop
from s4d.train.loop import synthetic_settings
from tests import _selfrender_golden as sr_golden
from tests.test_selfrender_replace import CSW4_FILES, DOMAIN, SR, VARIANTS, _flat, _inside, _step, batch, patched  # noqa: F401

V3 = "selfrender_replace_v3.yaml"
K = torch.tensor([[154.5, 0.0, 64.0], [0.0, 154.5, 64.0], [0.0, 0.0, 1.0]])


@pytest.mark.parametrize("explicit", [False, True])
def test_option_off_reproduces_v2_bit_for_bit(tmp_path, explicit):
    """Against tests/data/selfrender_v2_golden.json, written by the unchanged sources (a5d68e3): three v2 training steps
    at 50000-50002 (self-render weight 1: domain cameras drawn, rendered and re-encoded)."""
    expected = json.loads(sr_golden.GOLDEN_V2.read_text())
    got = sr_golden.fingerprints(tmp_path, {"aug.synth_radius_jitter": 0.0} if explicit else None, **sr_golden.V2_ARGS)
    assert float.fromhex(expected["steps"][0]["metrics"]["selfrender_weight"]) == 1.0
    if got["environment"] == expected["environment"]:
        assert got == expected
    else:
        for key in ("optimizer_groups", "rng_after_steps"):
            assert got[key] == expected[key], key
        assert [sorted(s["losses"]) for s in got["steps"]] == [sorted(s["losses"]) for s in expected["steps"]]
        assert [sorted(s["metrics"]) for s in got["steps"]] == [sorted(s["metrics"]) for s in expected["steps"]]


def test_variant_config_changes_only_the_radius_jitter():
    v2 = load_config(CSW4_FILES + [VARIANTS / SR])
    v3 = load_config(CSW4_FILES + [VARIANTS / SR, VARIANTS / V3])
    base, new = _flat(v2), _flat(v3)
    assert {k: new.get(k) for k in set(new) | set(base) if new.get(k) != base.get(k)} == {
        ("aug", "synth_radius_jitter"): 0.05}
    assert HELDOUT_RANGES["radius_frac"] == 0.05  # the point-cloud views' radial range it mirrors
    assert synthetic_settings(v3["aug"], future=False, hardneg=False)["radius_jitter"] == 0.05
    assert synthetic_settings(v2["aug"], future=False, hardneg=False)["radius_jitter"] == 0.0


def test_jitter_changes_only_the_distance():
    domain = augment.load_viewpoint_domain(DOMAIN)
    Kb = K.expand(1000, 3, 3)
    torch.manual_seed(3)
    ref = augment.domain_cameras(1000, 2, Kb, 128, 128, domain)
    rng_ref = torch.get_rng_state()
    torch.manual_seed(3)
    zero = augment.domain_cameras(1000, 2, Kb, 128, 128, domain, 0.0)
    assert torch.equal(torch.get_rng_state(), rng_ref)  # j = 0 draws nothing more
    assert all(torch.equal(ref[k], zero[k]) for k in ref)
    torch.manual_seed(3)
    cams = augment.domain_cameras(1000, 2, Kb, 128, 128, domain, 0.05)
    poses, ref_poses = cams["poses"].reshape(-1, 3).numpy(), ref["poses"].reshape(-1, 3).numpy()
    assert np.array_equal(poses[:, :2], ref_poses[:, :2])  # same directions as v2's draw
    dist = poses[:, 2]
    assert dist.min() >= 0.95 - 1e-12 and dist.max() <= 1.05 + 1e-12
    assert dist.min() < 0.955 and dist.max() > 1.045  # the full range is reached
    assert abs(dist.mean() - 1.0) < 0.004 and abs(np.abs(dist - 1.0).mean() - 0.025) < 0.003  # uniform in [-5 %, 5 %]
    angles = poses.copy()
    angles[:, 2] = domain["distance"]
    assert _inside(angles, domain).all()  # in angle still inside the frozen domain
    c2w = cams["c2w"].reshape(-1, 4, 4)
    centre = c2w[:, :3, 3].double()
    lookat = torch.tensor(LOOKAT, dtype=torch.float64)
    assert torch.allclose((centre - lookat).norm(dim=-1), torch.as_tensor(dist), atol=1e-5)
    forward = (lookat - centre) / (lookat - centre).norm(dim=-1, keepdim=True)
    assert torch.allclose(c2w[:, :3, 2].double(), forward, atol=1e-5)  # OpenCV +z still points at the look-at
    rig = augment.orbit_rig([tuple(p) for p in poses[:4]], 128, 128)
    assert np.allclose(rig["w2c"], cams["w2c"].reshape(-1, 4, 4)[:4].numpy(), atol=1e-5)


def test_training_step_uses_jittered_cameras(batch, patched, monkeypatch):
    poses = []
    orig = loop.domain_cameras

    def cams(*a, **k):
        res = orig(*a, **k)
        poses.append(res["poses"])
        return res

    monkeypatch.setattr(loop, "domain_cameras", cams)
    cfg = sr_golden.small_csw4_cfg(SR, V3)
    _, out = _step(cfg, batch, 10000)  # warm-up: nothing sampled, no distance metric
    assert poses == [] and "synth_camera_distance_dev" not in out["metrics"]
    _, out = _step(cfg, batch, 50000)
    (p,) = poses
    dist = p[..., 2].reshape(-1)
    assert ((dist >= 0.95 - 1e-12) & (dist <= 1.05 + 1e-12)).all() and not torch.all(dist == 1.0)
    dev = float(out["metrics"]["synth_camera_distance_dev"])
    assert np.isclose(dev, float((dist - 1.0).abs().mean()), atol=1e-6) and 0.0 < dev <= 0.05
    assert torch.isfinite(out["total"])
    v2_cfg = sr_golden.small_csw4_cfg(SR)
    _, out = _step(v2_cfg, batch, 50000)
    assert "synth_camera_distance_dev" not in out["metrics"] and torch.all(poses[-1][..., 2] == 1.0)


def test_invalid_radius_jitter_is_rejected():
    ok = {"synth_source": "gaussian_self_render", "synth_camera_domain": str(DOMAIN)}
    with pytest.raises(ValueError, match="gaussian_self_render only"):
        synthetic_settings({"synth_radius_jitter": 0.05}, future=False, hardneg=False)
    with pytest.raises(ValueError, match="needs aug.synth_camera_domain"):
        synthetic_settings({"synth_source": "gaussian_self_render", "synth_radius_jitter": 0.05}, future=False,
                           hardneg=False)
    for bad in (-0.01, 1.0):
        with pytest.raises(ValueError, match=r"\[0, 1\)"):
            synthetic_settings({**ok, "synth_radius_jitter": bad}, future=False, hardneg=False)
    assert synthetic_settings({**ok, "synth_radius_jitter": 0.0}, future=False, hardneg=False)["radius_jitter"] == 0.0
