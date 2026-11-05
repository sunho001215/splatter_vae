"""render.far 4.0 versions of the DROID pretraining profiles (user directive 2026-11-05).

configs/droid/{csw4,vits}_largescale_far4.yaml equal their far-3.0 profiles in every key except render.far; the far-3.0
files and every Meta-World configuration keep far 3.0. No other key carries the depth range: the depth-validity masks,
Chamfer targets, track weights, synthetic views and every render read render.far, which forward_losses on fixture DROID
windows shows below (docs/droid_largescale_validation/far4_effective_config.md)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))

import droid_largescale_fixtures as F  # noqa: E402
from test_droid_csw4_profile import flatten, metaworld_csw4  # noqa: E402

from s4d.config import get, load_config  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
DROID = REPO / "configs/droid"
VERSIONS = {"csw4": (DROID / "csw4_largescale.yaml", DROID / "csw4_largescale_far4.yaml"),
            "vits": (DROID / "vits_largescale.yaml", DROID / "vits_largescale_far4.yaml")}


def _differ(a: dict, b: dict) -> dict:
    return {k: (a.get(k, "<absent>"), b.get(k, "<absent>")) for k in set(a) | set(b)
            if a.get(k, "<absent>") != b.get(k, "<absent>")}


@pytest.mark.parametrize("name", VERSIONS)
def test_far4_version_differs_from_its_far3_profile_only_in_render_far(name):
    base, far4 = (flatten(load_config([p])) for p in VERSIONS[name])
    assert _differ(base, far4) == {"render.far": (3.0, 4.0)}
    assert far4["render.near"] == 0.05 and far4["loss.depth_valid_far"] is True  # validity follows render.far
    assert far4["data.regime"] == "droid_largescale" and far4["aug.synth_camera_rig"] == "training_cameras"


def test_vits_far4_relates_to_csw4_far4_exactly_as_vits_to_csw4():
    (csw4, csw4_far4), (vits, vits_far4) = ([flatten(load_config([p])) for p in VERSIONS[n]] for n in ("csw4", "vits"))
    assert _differ(csw4_far4, vits_far4) == _differ(csw4, vits) and "render.far" not in _differ(csw4, vits)


@pytest.mark.parametrize("name", VERSIONS)
def test_far4_versions_pass_the_geometry_guards(name):
    from s4d.train.augment import synth_camera_rig
    from s4d.train.loop import anchor_statistics_path

    cfg = load_config([VERSIONS[name][1]])
    assert synth_camera_rig(cfg) == "training_cameras"
    with pytest.raises(ValueError, match="needs model.anchor_stats"):
        anchor_statistics_path(cfg)


def test_metaworld_configurations_keep_far_3():
    """The far change is DROID-only: no Meta-World file sets another far plane, and every resolved Meta-World csw4
    composition (base + task + csw4 variants) keeps 3.0, as do the far-3.0 DROID profiles."""
    setting = {}
    for path in sorted((REPO / "configs/metaworld").rglob("*.yaml")):
        far = get(yaml.safe_load(path.read_text()) or {}, "render.far")
        if far is not None:
            setting[str(path.relative_to(REPO))] = far
    assert setting == {"configs/metaworld/base.yaml": 3.0}
    for task in sorted(p.stem for p in (REPO / "configs/metaworld/tasks").glob("*.yaml")):
        assert metaworld_csw4(task)["render"]["far"] == 3.0, task
    for name, (base, _) in VERSIONS.items():
        assert load_config([base])["render"]["far"] == 3.0, name


# ------------------------------------------------------------------------------------------------ consumers
@pytest.fixture(scope="module")
def env(tmp_path_factory):
    return F.build_all(tmp_path_factory.mktemp("droid_far4"))


def _window_batch(env):
    from s4d.data.contract import collate
    from s4d.data.droid.largescale.dataset import DroidLargeScaleDataset

    ds = DroidLargeScaleDataset(env["manifest"], **F.dataset_kwargs(env, max_tracks=16))
    batch = collate([ds[0], ds[3]])
    # a band of measured depth at 3.5 m (inside the far4 range only) and one at 4.5 m (outside both) in every view
    batch["depth"][..., 0:3, :] = 3.5
    batch["depth"][..., 3:6, :] = 4.5
    # track 0 at 3.5 m and track 1 at 4.5 m on each camera's optical axis (camera z = that depth), both supervised
    c2w = batch["c2w"].float()
    for n, z in ((0, 3.5), (1, 4.5)):
        batch["track_src"][:, :, :, n] = (c2w[:, None, :, :3, 3] + z * c2w[:, None, :, :3, 2]).to(batch["track_src"].dtype)
        batch["track_weight"][:, :, :, n] = 1
    return batch


@pytest.mark.parametrize("name", VERSIONS)
@pytest.mark.parametrize("far4", [False, True])
def test_every_range_consumer_reads_render_far(env, monkeypatch, name, far4):
    """One training step of forward_losses with the profile's loss/aug settings (a tiny model, the fixture windows and a
    shape-only rasterizer): every renderer call, the depth-validity mask, the Chamfer targets, the track weights, the
    synthetic cameras and their fused GT cloud, and the usage diagnostics take render.far of the profile."""
    from test_review_options import _fake_rasterizer

    from s4d.diag import heldout
    from s4d.losses import rgb
    from s4d.model import render
    from s4d.model.decoder import DecoderConfig, GaussianDecoder, GroupConfig
    from s4d.model.encoder import Encoder, EncoderConfig
    from s4d.train import augment, loop

    cfg = load_config([VERSIONS[name][int(far4)]], ["model.anchor_stats=null", "train.bf16=false"])
    far = cfg["render"]["far"]
    assert far == (4.0 if far4 else 3.0) and cfg["aug"]["synth_views"] == 2 and cfg["loss"]["chamfer"] > 0
    seen = {"raster": [], "valid": [], "clouds": [], "synth": [], "fused": [], "jitter": [], "tracks": [], "usage": []}
    fake = _fake_rasterizer([])

    def raster(**kw):
        seen["raster"].append((kw["near_plane"], kw["far_plane"]))
        return fake(**kw)

    def spy(module, attr, key, pick):
        original = getattr(module, attr)

        def wrapped(*args, **kwargs):
            seen[key].append(pick(*args, **kwargs))
            return original(*args, **kwargs)

        monkeypatch.setattr(module, attr, wrapped)

    monkeypatch.setattr(render, "_rasterize", raster)
    monkeypatch.setattr(rgb, "_ssim_map", lambda x, y: 1 - (x - y).square())
    spy(loop, "depth_l1", "valid", lambda pred, target, valid, *a, **k: valid.clone())
    spy(loop, "target_clouds", "clouds", lambda depth, K, c2w, far, valid_far, *a, **k: (far, valid_far))
    spy(loop, "synthesize_views", "synth", lambda batch, cams, near, far: (near, far))
    spy(augment, "fused_training_clouds", "fused", lambda batch, far: far)
    spy(loop, "jitter_training_cameras", "jitter", lambda *a, near, far, **k: (near, far))
    spy(loop, "motion3d_track_loss", "tracks", lambda gs, xyz, dyn, src, disp, weight: weight.clone())
    spy(loop, "usage_diagnostics", "usage", lambda gs, depth0, w2c, K, near, far, margin: (near, far))

    batch = _window_batch(env)
    B, T, V, _, H, W = batch["images"].shape
    group = GroupConfig(parents=2, children=2, offset_scale=0.2, child_radius=0.05, scale_min=0.001, scale_max=0.08)
    torch.manual_seed(0)
    enc = Encoder(EncoderConfig(image_height=H, image_width=W, patch_size=4, width=16, depth=1, heads=4, num_slots=1,
                                slot_dim=8, drop_path=0, mask_ratio=0.75, mask_mode="random"))
    model = loop.Model(enc, GaussianDecoder(DecoderConfig(slot_dim=8, dim=16, depth=1, heads=4, scene=group,
                                                          dynamic=group, num_slots=1))).train()
    out = loop.forward_losses(model, batch, cfg, 0, source=torch.zeros(B, dtype=torch.long))
    assert torch.isfinite(out["total"])

    near = cfg["render"]["near"]
    # renders: training views, synthetic-view targets, motion features; all with the profile's planes
    assert len(seen["raster"]) >= 3 and set(seen["raster"]) == {(near, far)}
    assert seen["clouds"] == [(far, True)] and seen["synth"] == [(near, far)] and seen["fused"] == [far]
    assert seen["jitter"] == [(near, far)] and seen["usage"] == [(near, far)]
    # depth validity (depth L1 / coverage): the 3.5 m band only inside far 4.0, the 4.5 m band never
    valid = seen["valid"][0].view(B, T, V, 1, H, W)
    assert bool(valid[..., 0:3, :].all()) is far4 and not valid[..., 3:6, :].any()
    rest = batch["depth"][..., 6:, :]
    assert valid[..., 6:, :].equal((rest > 0) & (rest < far))
    # Chamfer targets: the excluded fraction is exactly the measured depth at or beyond far
    excluded = float(out["metrics"]["chamfer_excluded_beyond_far"])
    assert excluded == pytest.approx(float((batch["depth"] >= far).float().mean()))
    # track weights: the track at camera z 3.5 m is supervised only with far 4.0, the one at 4.5 m never
    weight = seen["tracks"][0]
    assert bool((weight[..., 0] == 1).all()) is far4 and not weight[..., 1].any()
    # the synthetic views' fused GT cloud keeps the 3.5 m pixels only with far 4.0 (same rule, inclusive)
    _, _, fused_valid = heldout.fused_training_clouds(batch, far)
    band = torch.zeros_like(batch["depth"][:, :, :, 0], dtype=torch.bool)
    band[..., 0:3, :] = True
    assert bool(fused_valid.view_as(band)[band].all()) is far4


# ------------------------------------------------------------------------------------------------ measurement tool
def _tool():
    import importlib.util

    spec = importlib.util.spec_from_file_location("droid_ls_depth_range", REPO / "scripts/droid_ls_depth_range.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_depth_range_tool_on_the_fixture_release(env, tmp_path):
    """scripts/droid_ls_depth_range.py (the far-plane measurement): counts every stored pixel, follows the dataset's
    window rule, and its cross-camera check is exact up to pixel sampling under the true calibration (the fixture's
    DROID-3D) and clearly worse under the perturbed one (PointWorld's)."""
    tool = _tool()
    row = json.loads(Path(env["manifest"]).read_text().splitlines()[0])
    part = tool.episode_stats(row, env["tmp"] / "pw", stride=3, cross_every=1)
    frames = F.RAW_STEPS - 1
    assert part["hist_all"]["ext"].sum() == 2 * frames * F.H * F.W and part["hist_all"]["wrist"].sum() == frames * F.H * F.W
    mult = tool.window_multiplicity(row, 3)
    assert part["hist_windows"].sum() == 2 * mult.sum() * F.H * F.W and mult.sum() % 3 == 0
    summary = tool.summarize(part, 1)
    ext = summary["depth_ext_all_frames"]
    assert ext["max_valid_m"] < 4.0 and ext["fraction_valid_in_band_m"]["(3,4]"] >= 0.0  # fixture depth stops at 4 m
    cross = summary["cross_camera_relative_error_by_source_depth_bin"]
    true_p50 = [v["p50"] for v in cross["droid3d"]["selected"].values() if v["n"] > 1000]
    perturbed_p50 = [v["p50"] for v in cross["pointworld"]["selected"].values() if v["n"] > 1000]
    assert true_p50 and max(true_p50) < 0.005 and np.median(perturbed_p50) > 2 * np.median(true_p50)
    assert summary["tracks"]["visible_depth_valid_points"] > 0 and summary["tracks"]["points_z_above_4.01m"] == 0
