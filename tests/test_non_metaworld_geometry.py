"""Meta-World geometry is never inherited silently by another data regime (DROID real-data findings, 2026-11-02):
synthetic views come from the sample's own cameras (aug.synth_camera_rig; the default jitter is the Meta-World rig, poses
in the Meta-World world frame), and the decoder anchors need the regime's own workspace statistics (the default anchors
are the Meta-World workspace)."""

from __future__ import annotations

import math

import numpy as np
import pytest
import torch

from s4d.data.metaworld.cameras import HELDOUT_RANGES
from s4d.train import augment as A
from s4d.train import loop


def _look(C, L, roll_deg=0.0):
    """OpenCV c2w at C looking at L, world +z up, then rolled about the optical axis."""
    C, L = np.asarray(C, float), np.asarray(L, float)
    z = (L - C) / np.linalg.norm(L - C)
    x = np.cross(z, [0.0, 0.0, 1.0])
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    R = np.stack((x, y, z), 1)
    r = math.radians(roll_deg)
    R = R @ np.array([[math.cos(r), -math.sin(r), 0], [math.sin(r), math.cos(r), 0], [0, 0, 1]])
    T = np.eye(4)
    T[:3, :3], T[:3, 3] = R, C
    return T


def _droid_like_batch(B=2, H=36, W=64):
    """Two DROID-like external cameras (robot-base frame) looking at a workspace point, constant depth to it."""
    L = np.array([0.5, 0.0, 0.1])
    poses = [_look([0.3, -0.7, 0.6], L, roll_deg=4.0), _look([0.2, 0.75, 0.55], L, roll_deg=-3.0)]
    c2w = torch.as_tensor(np.stack(poses), dtype=torch.float32).expand(B, 2, 4, 4).clone()
    K = torch.tensor([[50.0, 0.0, W / 2 + 0.3], [0.0, 50.0, H / 2 - 0.2], [0.0, 0.0, 1.0]]).expand(B, 2, 3, 3).clone()
    K[:, 1, 0, 0] = 55.0  # distinct K per camera: the synthetic camera must take its base camera's K
    dist = torch.as_tensor([np.linalg.norm(p[:3, 3] - L) for p in poses], dtype=torch.float32)
    depth = dist.view(1, 1, 2, 1, 1, 1).expand(B, 3, 2, 1, H, W).clone()
    return {"K": K, "c2w": c2w, "w2c": torch.linalg.inv(c2w), "depth": depth}, L, dist


def test_default_rig_is_the_reference_jitter_bit_for_bit():
    K = torch.tensor([[110.0, 0, 64], [0, 110.0, 64], [0, 0, 1]]).expand(3, 6, 3, 3).clone()
    batch = {"K": K, "c2w": torch.eye(4).expand(3, 6, 4, 4), "depth": torch.ones(3, 3, 6, 1, 128, 128)}
    assert A.synth_camera_rig({}) == "metaworld" and A.synth_camera_rig({"aug": {"synth_views": 2}}) == "metaworld"
    torch.manual_seed(7)
    ref = A.jitter_cameras(3, 2, K[:, 0], 128, 128, 1.0)
    ref_rng = torch.get_rng_state()
    torch.manual_seed(7)
    got = loop.synthetic_cameras("metaworld", batch, 2, 128, 128, 1.0, 0.05, 3.0)
    assert torch.equal(torch.get_rng_state(), ref_rng)
    for key in ("K", "w2c", "c2w"):
        assert torch.equal(got[key], ref[key]), key


def test_training_camera_rig_looks_at_the_scene_within_the_ranges():
    batch, L, dist = _droid_like_batch()
    cams = A.jitter_training_cameras(batch["K"], batch["c2w"], batch["depth"], 3, 1.0, seed=3, near=0.05, far=3.0)
    assert cams["w2c"].shape == (2, 3, 4, 4) and cams["K"].shape == (2, 3, 3, 3)
    c2w = cams["c2w"].double()
    R = c2w[..., :3, :3]
    assert torch.allclose(R @ R.transpose(-1, -2), torch.eye(3, dtype=R.dtype).expand_as(R), atol=1e-6)
    assert torch.allclose(torch.linalg.inv(c2w), cams["w2c"].double(), atol=1e-5)
    for b in range(2):
        for j in range(3):
            v = int(cams["base"][b, j])
            assert torch.equal(cams["K"][b, j], batch["K"][b, v])
            # the look-at point projects onto the principal point
            p = cams["w2c"][b, j].double() @ torch.tensor([*L, 1.0], dtype=torch.float64)
            uv = (cams["K"][b, j].double() @ p[:3])[:2] / p[2]
            assert torch.allclose(uv, cams["K"][b, j, :2, 2].double(), atol=1e-3)
            # distance and viewing-direction change within the Meta-World trajectory ranges
            d = float((c2w[b, j, :3, 3] - torch.as_tensor(L)).norm())
            r = HELDOUT_RANGES
            lo = float(dist[v]) * (1 - r["radius_frac"]) - 1e-4
            hi = math.hypot(float(dist[v]) * (1 + r["radius_frac"]), r["lateral_m"]) + 1e-4
            assert lo <= d <= hi
            base_dir = batch["c2w"][b, v, :3, 3].double() - torch.as_tensor(L)
            new_dir = c2w[b, j, :3, 3] - torch.as_tensor(L)
            angle = math.degrees(math.acos(float(torch.nn.functional.cosine_similarity(base_dir, new_dir, dim=0))))
            assert angle <= math.hypot(r["azimuth_deg"], r["elevation_deg"]) + math.degrees(
                math.atan(r["lateral_m"] / lo)) + 1e-3
            # roll is kept: the image x axis stays close to the base camera's
            x_new, x_base = c2w[b, j, :3, 0], batch["c2w"][b, v, :3, 0].double()
            assert float(x_new @ x_base) > math.cos(math.radians(20))
    again = A.jitter_training_cameras(batch["K"], batch["c2w"], batch["depth"], 3, 1.0, seed=3, near=0.05, far=3.0)
    assert torch.equal(again["w2c"], cams["w2c"])


def test_training_camera_rig_scale_zero_reproduces_the_base_camera_and_has_a_depth_fallback():
    batch, _, _ = _droid_like_batch()
    cams = A.jitter_training_cameras(batch["K"], batch["c2w"], batch["depth"], 2, 0.0, seed=1)
    for b in range(2):
        for j in range(2):
            assert torch.allclose(cams["c2w"][b, j], batch["c2w"][b, int(cams["base"][b, j])], atol=1e-5)
    batch["depth"].zero_()  # no valid depth: look-at at the default distance, still finite and rigid
    cams = A.jitter_training_cameras(batch["K"], batch["c2w"], batch["depth"], 2, 1.0, seed=1)
    assert torch.isfinite(cams["w2c"]).all()


def test_the_loop_dispatches_the_training_camera_rig():
    batch, _, _ = _droid_like_batch()
    torch.manual_seed(0)
    got = loop.synthetic_cameras("training_cameras", batch, 2, 36, 64, 1.0, 0.05, 3.0)
    torch.manual_seed(0)
    seed = int(torch.randint(0, 2**31 - 1, (1,)).item())
    ref = A.jitter_training_cameras(batch["K"], batch["c2w"], batch["depth"], 2, 1.0, seed=seed, near=0.05, far=3.0)
    assert torch.equal(got["w2c"], ref["w2c"]) and torch.equal(got["K"], ref["K"])


@pytest.mark.parametrize("regime", ["droid_largescale", "droid"])
def test_non_metaworld_regimes_refuse_the_metaworld_rig(regime):
    cfg = {"data": {"regime": regime}, "aug": {"synth_views": 2}, "loss": {}}
    with pytest.raises(ValueError, match="synth_camera_rig training_cameras"):
        A.synth_camera_rig(cfg)
    with pytest.raises(ValueError, match="synth_camera_rig training_cameras"):
        A.synth_camera_rig({"data": {"regime": regime}, "loss": {"self_render": 0.1}})
    cfg["aug"]["synth_camera_rig"] = "training_cameras"
    assert A.synth_camera_rig(cfg) == "training_cameras"
    cfg["aug"]["synth_camera_domain"] = "configs/metaworld/selfrender_domain_v1.yaml"
    with pytest.raises(ValueError, match="Meta-World viewpoint domain"):
        A.synth_camera_rig(cfg)
    # no synthetic camera at all: nothing to refuse (the legacy DROID-base configurations)
    assert A.synth_camera_rig({"data": {"regime": regime}, "loss": {}}) == "metaworld"
    with pytest.raises(ValueError, match="must be one of"):
        A.synth_camera_rig({"aug": {"synth_camera_rig": "orbit"}})


def test_non_metaworld_regimes_need_their_own_anchor_statistics():
    assert loop.anchor_statistics_path({"model": {"anchor_stats": None}}) is None  # Meta-World: defaults allowed
    for regime in ("droid_largescale", "droid"):
        with pytest.raises(ValueError, match="needs model.anchor_stats"):
            loop.anchor_statistics_path({"data": {"regime": regime}, "model": {"anchor_stats": None}})
    cfg = {"data": {"regime": "droid_largescale"}, "model": {"anchor_stats": "s.json"}}
    assert loop.anchor_statistics_path(cfg) == "s.json"
