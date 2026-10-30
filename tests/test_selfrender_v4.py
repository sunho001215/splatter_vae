"""Variant selfrender_replace_v4 (user instruction 2026-10-31): selfrender_replace_v2 + the +-5 % radial camera jitter
(v3) + the training-supported viewpoint domain with the coverage and ws_support2 minimums relaxed by exactly 0.95
(configs/metaworld/selfrender_domain_v1_relaxed5.yaml, scripts/selfrender_domain.py --relax)."""

from __future__ import annotations

import math

import numpy as np
import torch
import yaml

from s4d.config import load_config
from s4d.train import augment, loop
from s4d.train.loop import synthetic_settings
from tests import _selfrender_golden as sr_golden
from tests.test_selfrender_replace import (CSW4_FILES, DOMAIN, REPO, SR, VARIANTS, _flat, _inside, _step, batch,  # noqa: F401
                                           patched)

V4 = "selfrender_replace_v4.yaml"
RELAXED = REPO / "configs/metaworld/selfrender_domain_v1_relaxed5.yaml"
CRITERIA = ("coverage_min", "support2_min", "ws_visible_min", "ws_support2_min")
FACTORS = {"coverage_min": 0.95, "support2_min": 1.0, "ws_visible_min": 1.0, "ws_support2_min": 0.95}


def test_variant_config_changes_only_jitter_and_domain():
    v2 = load_config(CSW4_FILES + [VARIANTS / SR])
    v4 = load_config(CSW4_FILES + [VARIANTS / SR, VARIANTS / V4])
    base, new = _flat(v2), _flat(v4)
    assert {k: new.get(k) for k in set(new) | set(base) if new.get(k) != base.get(k)} == {
        ("aug", "synth_radius_jitter"): 0.05,
        ("aug", "synth_camera_domain"): "configs/metaworld/selfrender_domain_v1_relaxed5.yaml"}
    settings = synthetic_settings(v4["aug"], future=False, hardneg=False)
    assert settings == {"source": "gaussian_self_render", "ramp": (20000, 40000), "radius_jitter": 0.05,
                        "domain": "configs/metaworld/selfrender_domain_v1_relaxed5.yaml"}
    assert v4["aug"]["synth_views"] == 2 and v4["model"] == v2["model"] and v4["loss"] == v2["loss"]
    assert v4["train"] == v2["train"] and v4["data"] == v2["data"]


def test_relaxed_domain_file():
    v1, doc = yaml.safe_load(DOMAIN.read_text()), yaml.safe_load(RELAXED.read_text())
    assert doc["version"] == "selfrender_domain_v1_relaxed5" and doc["cell_deg"] == v1["cell_deg"] == 5.0
    assert doc["rig"] == v1["rig"] and doc["analysis"] == v1["analysis"]
    crit = doc["criteria"]
    assert crit["relax_factors"] == {"coverage": 0.95, "ws_support2": 0.95}
    for key in CRITERIA:  # the training-pose minimums are v1's; only coverage and ws_support2 are scaled, exactly
        assert crit["training_pose_minimums"][key] == v1["criteria"][key]
        assert crit[key] == round(crit["exact_minimums"][key], 5)
        assert abs(crit["exact_minimums"][key] / FACTORS[key] - v1["criteria"][key]) <= 5e-6
    assert (crit["coverage_min"], crit["ws_support2_min"]) == (0.79322, 0.63361)
    assert (crit["support2_min"], crit["ws_visible_min"]) == (v1["criteria"]["support2_min"], v1["criteria"]["ws_visible_min"])
    cells = np.asarray(doc["cells"], dtype=float)
    assert len(cells) == doc["summary"]["domain_cells"] == doc["summary"]["supported_cells_all_components"]
    for j, key in enumerate(CRITERIA):
        assert (cells[:, 2 + j] >= crit[key] - 1e-3).all(), key
    centres = {(int(a), int(e)) for a, e in cells[:, :2]}
    v1_centres = {(int(a), int(e)) for a, e, *_ in v1["cells"]}
    assert v1_centres < centres  # a strict superset of the v1 domain
    assert all((int(a), int(e)) in centres for a, e in doc["rig"]["train_poses_az_el"])
    # one 4-connected component (azimuth wraps) containing the training poses
    seen, stack = set(), [tuple(map(int, doc["rig"]["train_poses_az_el"][0]))]
    while stack:
        a, e = stack.pop()
        if (a, e) in seen or (a, e) not in centres:
            continue
        seen.add((a, e))
        stack += [(((a + 5 + 180) % 360) - 180, e), (((a - 5 + 180) % 360) - 180, e), (a, e + 5), (a, e - 5)]
    assert seen == centres
    area = sum(math.radians(5.0) * (math.sin(math.radians(e + 2.5)) - math.sin(math.radians(e - 2.5))) for _, e in centres)
    assert math.isclose(area, doc["summary"]["domain_solid_angle_sr"], abs_tol=1e-4)
    assert doc["summary"]["elevation_range"] == [int(cells[:, 1].min()), int(cells[:, 1].max())]


def test_sampler_over_the_relaxed_domain_with_jitter():
    domain = augment.load_viewpoint_domain(RELAXED)
    v1 = augment.load_viewpoint_domain(DOMAIN)
    assert len(domain["cells"]) > len(v1["cells"]) and domain["distance"] == 1.0
    torch.manual_seed(0)
    K = torch.tensor([[154.5, 0.0, 64.0], [0.0, 154.5, 64.0], [0.0, 0.0, 1.0]]).expand(2000, 3, 3)
    cams = augment.domain_cameras(2000, 2, K, 128, 128, domain, 0.05)
    poses = cams["poses"].reshape(-1, 3).numpy()
    angles = poses.copy()
    angles[:, 2] = 1.0
    assert _inside(angles, domain).all()
    assert (~_inside(angles, v1)).mean() > 0.2  # the added cells are sampled (by solid angle)
    assert poses[:, 2].min() >= 0.95 - 1e-12 and poses[:, 2].max() <= 1.05 + 1e-12
    assert abs(np.abs(poses[:, 2] - 1.0).mean() - 0.025) < 0.003


def test_training_step_draws_from_the_relaxed_domain_with_jitter(batch, patched, monkeypatch):
    poses, domains = [], []
    orig = loop.domain_cameras

    def cams(B, n, K, H, W, domain, radius_jitter=0.0):
        domains.append((domain["version"], radius_jitter))
        res = orig(B, n, K, H, W, domain, radius_jitter)
        poses.append(res["poses"])
        return res

    monkeypatch.setattr(loop, "domain_cameras", cams)
    cfg = sr_golden.small_csw4_cfg(SR, V4)
    _, out = _step(cfg, batch, 50000)
    assert domains == [("selfrender_domain_v1_relaxed5", 0.05)]
    p = poses[0].reshape(-1, 3).numpy()
    assert ((p[:, 2] >= 0.95 - 1e-12) & (p[:, 2] <= 1.05 + 1e-12)).all() and not np.all(p[:, 2] == 1.0)
    assert 0.0 < float(out["metrics"]["synth_camera_distance_dev"]) <= 0.05 and torch.isfinite(out["total"])
    _, out = _step(cfg, batch, 20000)  # warm-up end: weight 0, nothing sampled
    assert len(poses) == 1 and float(out["metrics"]["selfrender_weight"]) == 0.0

