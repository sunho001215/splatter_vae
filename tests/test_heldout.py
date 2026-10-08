"""Near-view held-out sets (review item 1): camera geometry, reader/dataset, oracle splatting, Chamfer, evaluator."""

from __future__ import annotations

import json
import math
from pathlib import Path

import h5py
import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader

from s4d.config import load_config
from s4d.data.contract import collate
from s4d.data.metaworld import cameras as C
from s4d.data.metaworld.dataset import MetaworldWindowDataset
from s4d.data.metaworld.heldout import HeldoutSets
from s4d.diag import heldout as D
from s4d.diag.local_log import RunLogger
from s4d.model.decoder import DecoderConfig, GaussianDecoder, GroupConfig
from s4d.model.encoder import Encoder, EncoderConfig
from s4d.rl.env import _camera_at, _forward, orbit_camera
from s4d.train import evaluate as E
from s4d.train.loop import Model

REPO = Path(__file__).resolve().parents[1]


# ------------------------------------------------------------------------------------------ camera sets
def test_lateral_only_offset_is_the_rl_lateral_trajectory_pose():
    theta, phi = C.TRAIN_CAMERAS[1]
    camera = orbit_camera(theta, phi)
    forward = _forward(camera)
    right = np.cross(forward, [0.0, 0.0, 1.0])
    right /= np.linalg.norm(right)
    for offset in (0.12, -0.05):
        expected = _camera_at(np.asarray(C.LOOKAT) - camera.distance * forward + offset * right)
        got = C.perturbed_orbit(theta, phi, 0.0, 0.0, 0.0, offset)
        assert got == pytest.approx((expected.azimuth, expected.elevation, expected.distance), abs=1e-9)


def test_zero_offsets_reproduce_the_training_camera_and_its_extrinsics():
    rig = C.camera_rig(32, 32)
    for i, (theta, phi) in enumerate(C.TRAIN_CAMERAS):
        az, el, d = C.perturbed_orbit(theta, phi, 0.0, 0.0, 0.0, 0.0)
        assert math.isclose(math.cos(math.radians(az - phi)), 1.0, abs_tol=1e-12)
        assert el == pytest.approx(-theta) and d == pytest.approx(C.RADIUS)
        np.testing.assert_allclose(C.orbit_rig([(az, el, d)], 32, 32)["c2w"][0], rig["c2w"][i], atol=1e-5)


def test_heldout_sets_are_fixed_and_respect_their_ranges():
    for name, (scale, _) in C.HELDOUT_SETS.items():
        cams = C.heldout_camera_set(name)
        assert cams == C.heldout_camera_set(name)
        assert len(cams) == len(C.TRAIN_CAMERAS) * C.HELDOUT_PER_TRAIN_CAMERA
        assert sorted({c["base"] for c in cams}) == list(range(len(C.TRAIN_CAMERAS)))
        for c in cams:
            for key, limit in C.HELDOUT_RANGES.items():
                assert abs(c[key]) <= scale * limit
            assert abs(c["distance"] - C.RADIUS) <= 0.05 * scale + 0.12 * scale  # radius change plus lateral shift
    near = C.heldout_camera_set("near")
    assert max(abs(c["azimuth_deg"]) for c in near) <= 5.0
    assert C.heldout_camera_set("near") != C.heldout_camera_set("traj")


# ------------------------------------------------------------------------------------------ fixtures
H = W = 16
T_LEN = 12


def _plane_cameras(offsets_x):
    """Cameras at x = offset looking along +z (OpenCV), fx = fy = 20, so a plane at z = 1 shifts 2 px per 0.1 m."""
    K = np.tile(np.array([[20.0, 0.0, W / 2], [0.0, 20.0, H / 2], [0.0, 0.0, 1.0]], dtype=np.float32), (len(offsets_x), 1, 1))
    c2w = np.tile(np.eye(4, dtype=np.float32), (len(offsets_x), 1, 1))
    c2w[:, 0, 3] = offsets_x
    return K, c2w, np.linalg.inv(c2w).astype(np.float32)


def _gradient_rgb(n_cams):
    rgb = np.zeros((T_LEN, n_cams, H, W, 3), dtype=np.uint8)
    rgb[..., 0] = (np.arange(W, dtype=np.uint8) * 12)[None, None, None, :]
    rgb[..., 1] = (np.arange(T_LEN, dtype=np.uint8) * 10)[:, None, None, None]
    return rgb


def write_main(path: Path, episodes=("ep000", "ep001")) -> Path:
    K, c2w, w2c = _plane_cameras([0.0, 0.1, -0.1])  # train0, train1, eval0
    with h5py.File(path, "w") as f:
        f.attrs.update(
            {"task": path.stem, "depth_unit_m": 1e-4, "dt_seconds": 0.05, "background_body": 65535,
             "body_names": json.dumps(["world", "plane"])}
        )
        cams = f.create_group("cameras")
        for key, value in {"K": K, "c2w": c2w, "w2c": w2c, "is_train": np.array([True, True, False])}.items():
            cams.create_dataset(key, data=value)
        for ep in episodes:
            g = f.create_group("episodes").create_group(ep) if "episodes" not in f else f["episodes"].create_group(ep)
            g.attrs["length"] = T_LEN
            obs = np.zeros((T_LEN, 39), dtype=np.float32)
            obs[:, 0] = np.arange(T_LEN) * 0.01
            for key, value in {
                "rgb": _gradient_rgb(3),
                "depth": np.full((T_LEN, 3, H, W), 10000, dtype=np.uint16),
                "body_id": np.ones((T_LEN, 2, H, W), dtype=np.uint16),
                "xpos": np.zeros((T_LEN, 2, 3), dtype=np.float32),
                "xquat": np.tile(np.array([1.0, 0, 0, 0], dtype=np.float32), (T_LEN, 2, 1)),
                "obs": obs,
            }.items():
                g.create_dataset(key, data=value)
    return path


def write_heldout(path: Path, episodes=("ep000",)) -> Path:
    K, c2w, w2c = _plane_cameras([0.1, 0.05, -0.1, 0.2])
    with h5py.File(path, "w") as f:
        f.attrs.update({"complete": True, "depth_unit_m": 1e-4, "sets": json.dumps(["near", "traj"])})
        cams = f.create_group("cameras")
        cams.create_dataset("names", data=np.asarray(["near0", "near1", "traj0", "traj1"], dtype=h5py.string_dtype()))
        cams.create_dataset("set", data=np.asarray(["near", "near", "traj", "traj"], dtype=h5py.string_dtype()))
        for key, value in {"K": K, "c2w": c2w, "w2c": w2c}.items():
            cams.create_dataset(key, data=value)
        group = f.create_group("episodes")
        for ep in episodes:
            g = group.create_group(ep)
            g.attrs["length"] = T_LEN
            g.create_dataset("rgb", data=_gradient_rgb(4))
            g.create_dataset("depth", data=np.full((T_LEN, 4, H, W), 10000, dtype=np.uint16))
    return path


def test_reader_and_dataset_attach_sets_only_for_contained_episodes(tmp_path):
    main = write_main(tmp_path / "plane.hdf5")
    held = write_heldout(tmp_path / "held.hdf5")
    reader = HeldoutSets(held)
    assert "ep000" in reader and "ep001" not in reader
    assert {k: v.tolist() for k, v in reader.columns.items()} == {"near": [0, 1], "traj": [2, 3]}
    ds = MetaworldWindowDataset(main, ["ep000", "ep001"], strides=(2,), with_eval=True, heldout=held)
    inside = ds[0]
    assert inside["meta"]["episode"] == "ep000"
    assert inside["near_images"].shape == (3, 2, 3, H, W) and inside["near_images"].dtype == torch.uint8
    assert inside["traj_depth"].shape == (3, 2, 1, H, W)
    assert torch.allclose(inside["traj_depth"], torch.ones(3, 2, 1, H, W))
    assert inside["near_K"].shape == (2, 3, 3) and inside["traj_c2w"].shape == (2, 4, 4)
    t_idx = inside["meta"]["t_indices"]
    assert int(inside["near_images"][1, 0, 1, 0, 0]) == t_idx[1] * 10  # time-coded green channel
    outside = ds[[s[0] for s in ds.samples].index("ep001")]
    assert "near_images" not in outside and "eval_images" in outside
    assert "near_images" not in MetaworldWindowDataset(main, ["ep000"], strides=(2,), with_eval=False, heldout=held)[0]


# ------------------------------------------------------------------------------------------ oracle + Chamfer
def test_splat_reprojects_a_plane_with_the_expected_shift_and_zbuffer():
    K, c2w, w2c = (torch.from_numpy(x) for x in _plane_cameras([0.0, 0.1]))
    depth = torch.ones(H, W)
    colors = torch.zeros(H, W, 3)
    colors[..., 0] = torch.arange(W, dtype=torch.float32)[None, :]
    points = D.lift_valid(depth[None], K[:1], c2w[:1], far=3.0)
    rgb, covered = D.splat(points, colors.reshape(-1, 3), K[1:], w2c[1:], H, W, near=0.05)
    # pixel centre u + 0.5 of camera 0 lands at u + 0.5 - 2 in camera 1 (0.1 m baseline, fx 20, depth 1 m)
    assert covered[0, 0, :, : W - 2].all() and not covered[0, 0, :, W - 2 :].any()
    assert torch.equal(rgb[0, 0, :, : W - 2], colors[..., 0][:, 2:])
    # z-buffer: a nearer point on the same pixel wins regardless of order
    near_pt = torch.tensor([[0.0, 0.0, 0.5]])
    far_pt = torch.tensor([[0.0, 0.0, 1.0]])
    for order in ((near_pt, far_pt), (far_pt, near_pt)):
        pts = torch.cat(order)
        cols = torch.tensor([[1.0, 0, 0], [0, 1.0, 0]]) if order[0] is near_pt else torch.tensor([[0, 1.0, 0], [1.0, 0, 0]])
        img, cov = D.splat(pts, cols, K[:1], w2c[:1], H, W, near=0.05)
        assert cov[0, 0, H // 2, W // 2] and img[0, 0, H // 2, W // 2] == 1.0


def test_chamfer_statistics_voxels_and_empty_clouds():
    grid = torch.stack(torch.meshgrid(torch.arange(5.0), torch.arange(5.0), indexing="ij"), -1).reshape(-1, 2) * 0.1
    cloud = torch.cat((grid, torch.zeros(len(grid), 1)), 1)
    shifted = cloud + torch.tensor([0.01, 0.0, 0.0])
    stats = D.chamfer_stats(shifted, cloud)
    for key in ("p2g_mean", "p2g_p50", "p2g_p90", "g2p_mean", "g2p_p90"):
        assert stats[key] == pytest.approx(0.01, abs=1e-6)
    floater = torch.cat((cloud, torch.tensor([[0.2, 0.2, 0.5]])))
    tail = D.chamfer_stats(floater, cloud)
    assert tail["p2g_p50"] == pytest.approx(0.0, abs=1e-6) and tail["p2g_mean"] > 0.01
    assert math.isnan(D.chamfer_stats(cloud[:0], cloud)["p2g_mean"])
    pts = torch.tensor([[0.001, 0.001, 0.001], [0.002, 0.002, 0.002], [0.02, 0.0, 0.0]])
    down = D.voxel_downsample(pts, 0.005)
    assert len(down) == 2 and torch.allclose(down[down[:, 0].argmin()], torch.tensor([0.0015, 0.0015, 0.0015]))


def test_spread_windows_cover_every_episode_round_robin():
    episodes = [f"ep{e}" for e in range(10) for _ in range(50)]
    t0s = [t for _ in range(10) for t in range(50)]
    chosen = E._spread_windows(episodes, t0s)
    assert len(chosen) == 64 == len(set(chosen))
    assert {episodes[i] for i in chosen} == {f"ep{e}" for e in range(10)}
    first = sorted(t0s[i] for i in chosen if episodes[i] == "ep0")
    assert first[0] == 0 and first[-1] == 49


# ------------------------------------------------------------------------------------------ evaluator
def _model():
    encoder = Encoder(
        EncoderConfig(image_height=H, image_width=W, patch_size=4, width=16, depth=1, heads=4, num_slots=2, slot_dim=8, drop_path=0)
    )
    group = GroupConfig(parents=2, children=2, offset_scale=0.2, child_radius=0.05, scale_min=0.001, scale_max=0.08)
    return Model(encoder, GaussianDecoder(DecoderConfig(slot_dim=8, dim=16, depth=1, heads=4, scene=group, dynamic=group)))


def _synthetic_render(gs, xyz, w2c, K, height, width, near, far):
    b, times = xyz.shape[:2]
    shape = (b, times, w2c.shape[1], 1, height, width)
    return {
        "rgb": torch.full((b, times, w2c.shape[1], 3, height, width), 0.5),
        "depth": torch.ones(shape),
        "alpha": torch.full(shape, 0.9),
    }


def test_evaluator_reports_sets_oracle_chamfer_retrieval_and_probes(tmp_path, monkeypatch):
    main = write_main(tmp_path / "plane.hdf5")
    held = write_heldout(tmp_path / "held.hdf5", episodes=("ep000", "ep001"))
    held_one = write_heldout(tmp_path / "held_one.hdf5", episodes=("ep000",))
    model = _model()
    cfg = load_config([REPO / "configs/metaworld/base.yaml"])
    cfg["wandb"]["enabled"] = False
    cfg["data"].update(root=str(tmp_path), task="plane")
    cfg["eval"]["heldout_sets"] = True
    val = MetaworldWindowDataset(main, ["ep000", "ep001"], strides=(2,), with_eval=True, heldout=held)
    probe = MetaworldWindowDataset(main, ["ep000", "ep001"], strides=(2,))
    loaders = {2: DataLoader(val, batch_size=4, collate_fn=collate)}
    probes = {2: DataLoader(probe, batch_size=4, collate_fn=collate)}

    def fake_forward(m, batch, cfg_, step, *, source, mask_ratio, return_renders=False):
        B, T, V = batch["images"].shape[:3]
        gs = m.decoder(torch.randn(B, 2, 8))
        rendered = _synthetic_render(gs, gs.xyz_sequence(), batch["w2c"], batch["K"], H, W, 0.05, 3.0)
        return {
            "gs": gs,
            "slots": torch.randn(B, V, 2, 8),
            "losses": {"total": torch.tensor(1.0)},
            "metrics": {"psnr": torch.tensor(20.0)},
            "source": torch.zeros(B, dtype=torch.long),
            **rendered,
            "pred_disp": batch["motion3d"].clone(),
            "pairs": ((0, 1, 0), (1, 2, 1), (0, 2, 0)),
        }

    monkeypatch.setattr(E, "forward_losses", fake_forward)
    monkeypatch.setattr(E, "render_rgbd", _synthetic_render)
    monkeypatch.setattr(E, "encode_states", lambda m, x: torch.randn(x.shape[0], x.shape[2], 2, 8))
    monkeypatch.setattr(E.Evaluator, "panels", lambda *a, **k: None)
    logger = RunLogger(tmp_path / "run")
    summary = E.Evaluator(cfg, loaders, probes, logger, torch.device("cpu"), 2)(model, 0, full=True)
    for name in ("near", "traj", "heldout"):
        assert f"metric/psnr_{name}@s2" in summary and f"metric/psnr_{name}_covered@s2" in summary
        assert 0.0 < summary[f"metric/oracle_coverage_{name}@s2"] <= 1.0
        assert math.isfinite(summary[f"metric/oracle_psnr_covered_{name}@s2"])
        assert f"metric/retrieval_top1_{name}_val@s2" in summary
    assert summary["metric/oracle_psnr_covered_near@s2"] > 30  # the planar scene reprojects exactly
    for kind in ("centers", "centers_dyn", "motion", "motion_dyn", "render_traj"):
        for stat in ("p2g_mean", "p2g_p50", "p2g_p90", "g2p_mean", "g2p_p50", "g2p_p90"):
            assert f"metric/cd_{kind}_{stat}@s2" in summary
    assert math.isfinite(summary["metric/cd_centers_p2g_p90@s2"])
    assert summary["metric/cd_windows@s2"] == len(loaders[2])
    for target in ("hand_pos", "hand_vel", "obj_pos"):
        for name in ("traincams", "heldout", "near", "traj"):
            assert f"metric/r2_{target}_val_{name}@s2" in summary
    assert summary["metric/retrieval_val_windows@s2"] > 0
    # episodes without near/trajectory views (the one-episode gate) keep the extrapolation set and skip the rest
    gate = MetaworldWindowDataset(main, ["ep001"], strides=(2,), with_eval=True, heldout=held_one)
    gate_summary = E.Evaluator(cfg, {2: DataLoader(gate, batch_size=4, collate_fn=collate)}, None, logger, torch.device("cpu"), 2)(
        model, 0, full=True
    )
    assert "metric/psnr_heldout@s2" in gate_summary and "metric/psnr_near@s2" not in gate_summary
    assert "metric/retrieval_top1_near_val@s2" not in gate_summary and "metric/cd_centers_p2g_mean@s2" in gate_summary
    logger.close()


def test_batched_splat_and_oracle_equal_the_per_cloud_versions(batch):
    torch.manual_seed(0)
    G, N, V = 3, 200, 2
    points = torch.rand(G, N, 3) * torch.tensor([0.4, 0.4, 0.5]) + torch.tensor([-0.2, -0.2, 0.8])
    colors, valid = torch.rand(G, N, 3), torch.rand(G, N) > 0.2
    K = torch.tensor([[20.0, 0, 8], [0, 20.0, 8], [0, 0, 1]]).expand(G, V, 3, 3)
    c2w = torch.eye(4).repeat(G, V, 1, 1)
    c2w[:, 1, 0, 3] = 0.05
    w2c = torch.linalg.inv(c2w)
    rgb, cov, dep = D.splat_batched(points, colors, valid, K, w2c, 16, 16, 0.05, chunk=2)
    for g in range(G):
        r, c, d = D.splat_depth(points[g][valid[g]], colors[g][valid[g]], K[g], w2c[g], 16, 16, 0.05)
        assert torch.equal(c, cov[g]) and torch.equal(d, dep[g])
        same_depth_pixels = c.expand_as(r)
        assert torch.equal(r[same_depth_pixels], rgb[g][same_depth_pixels]) or torch.allclose(r, rgb[g])
    batch["images"] = torch.randint(0, 256, batch["images"].shape, dtype=torch.uint8)
    batch.update({"near_K": batch["K"].clone(), "near_w2c": batch["w2c"].clone()})
    rgb, covered = D.oracle_batch(batch, "near", 0.05, 3.0)
    B, T, Vt = batch["images"].shape[:3]
    i = 0
    for b in range(B):
        for t in range(T):
            r, c = D.oracle_views(batch, b, t, "near", 0.05, 3.0)
            assert torch.equal(c, covered[i : i + Vt])
            i += Vt
