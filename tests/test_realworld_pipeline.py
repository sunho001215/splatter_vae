"""End-to-end synthetic episode: ingest -> rectify -> rig depth -> calibration -> world depth -> tracks -> lifting -> export.

All model outputs here are MOCK (CPU block matching, oracle tracker, oracle landmarks) and are flagged as such;
this validates conventions, bookkeeping, idempotency and the export contract, not model integration."""

import json
import sys

import h5py
import numpy as np
import pytest

from s4d.data.contract import validate_batch
from s4d.realworld import io as rio
from s4d.realworld.adapters import get_adapter
from s4d.realworld.calibration import CalibrationConfig
from s4d.realworld.contract import EpisodeContractError, validate_episode
from s4d.realworld.dataset import RealworldCacheDataset
from s4d.realworld.depth_stages import MapAnythingBackend, load_view_depth, mapanything_depth, mapanything_rig_depth
from s4d.realworld.export import export
from s4d.realworld.geometry import project, transform_points
from s4d.realworld.pipeline import (
    TrackerBackend,
    calibrate,
    ingest,
    lift3d,
    load_meta,
    stereo_rectify,
    track2d,
)
from s4d.realworld.robot import URDFModel, landmarks
from s4d.realworld.runners import LocalRunner
from s4d.realworld.tracking import OCCLUDED, VALID
from s4d.realworld.workers import WorkerSpec

LANDMARKS = ["panda_link2", "panda_link3", "panda_link4", "panda_link5", "panda_link7", "panda_link8",
             "gripper_base", "left_finger", "right_finger"]


def mock_mapanything(root):
    return MapAnythingBackend(WorkerSpec("mock", sys.executable, timeout_s=120), LocalRunner(root / "worker"),
                              {"max_disparity": 28}, allow_mock=True)


class OracleTracker(TrackerBackend):
    identity = {"name": "synthetic oracle tracker", "is_mock": True}

    def __init__(self, adapter):
        self.adapter = adapter

    def track(self, camera_id, frames, frame_indices, query):
        H, W = frames.shape[1:3]
        rc = np.stack(np.meshgrid(np.arange(H), np.arange(W), indexing="ij"), -1).reshape(-1, 2)
        P, vis = self.adapter.gt_points(camera_id, frame_indices[query], rc, frame_indices[query:])
        rig = camera_id.split("/")[0]
        flows, visconf = [], []
        for Pt, v in zip(P, vis):
            uv, _ = project(transform_points(self.adapter.T_cam_base(rig, "left"), Pt), self.adapter.K())
            uv = uv - 0.5 - rc[:, ::-1]  # AllTracker: offsets from the integer-centre grid
            flows.append(uv.T.reshape(2, H, W))
            visconf.append(np.stack([v.astype(float), np.ones(len(v))]).reshape(2, H, W))
        return np.stack(flows).astype(np.float32), np.stack(visconf).astype(np.float32)


@pytest.fixture(scope="module")
def run(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("rw")
    adapter = get_adapter("synthetic", {"asset_dir": str(tmp / "assets"), "frames": 10})
    root = tmp / "out"
    ep = adapter.list_episodes()[0]
    rng = np.random.default_rng(0)
    recs = {"ingest": ingest(adapter, ep, root)}
    meta = load_meta(root, ep)
    model = URDFModel.from_file(meta.robot.urdf_path)

    from s4d.realworld.pipeline import RobotGeometry, robot_state_at

    geom = RobotGeometry(meta)

    def landmark_source(cam, i, links):
        q, g, _, _ = robot_state_at(meta, meta.stream_timestamps_ns[cam][[i]])
        L = landmarks(model, geom.q(q[0], g[0]), links)
        uv, _ = project(transform_points(adapter.T_cam_base(cam.split("/")[0], "left"), L), adapter.K())
        return uv + rng.normal(0, 0.5, uv.shape), np.ones(len(L))

    landmark_source.identity = {"name": "synthetic oracle landmarks", "is_mock": True}
    # tiny 64x96 synthetic images: the uncertainty gates are relaxed here (defaults are tested in calibration tests)
    cfg = CalibrationConfig(depth_term=False, min_visible_fraction=0.05, min_motion_extent_m=0.03, max_rot_std_deg=2.0,
                            max_trans_std_m=0.03)
    recs["stereo_rectify"] = stereo_rectify(root, ep)
    recs["mapanything_rig_depth"] = mapanything_rig_depth(root, ep, mock_mapanything(root))
    recs["calibrate"] = calibrate(root, ep, landmark_source, landmark_links=LANDMARKS, cfg=cfg)
    recs["mapanything_depth"] = mapanything_depth(root, ep, mock_mapanything(root),
                                                 input_depth_stage="mapanything_rig_depth")
    recs["track2d"] = track2d(root, ep, OracleTracker(adapter), window=10, step=8, grid_stride=4)
    recs["lift3d"] = lift3d(root, ep)
    recs["export"] = export(root, ep, tmp / "cache", resolutions={"scratch": (144, 256)}, allow_infrared_input=True,
                            allow_mock=True, validation_tail_frames=4)
    return {"adapter": adapter, "root": root, "ep": ep, "recs": recs, "tmp": tmp, "landmark_source": landmark_source,
            "cfg": cfg}


def test_ingest_preserves_modality_and_reports_missing_stereo(run):
    meta = load_meta(run["root"], run["ep"])
    assert all(s.modality == "infrared" and s.channels == 1 for s in meta.streams)
    assert any("rig2" in w and "missing" in w for w in run["recs"]["ingest"]["summary"]["warnings"])
    assert run["recs"]["mapanything_rig_depth"]["summary"]["rig2/left"]["skipped"].startswith("stereo geometry missing")
    # a stream claiming 3-channel infrared is rejected by the contract
    import dataclasses

    bad = dataclasses.replace(meta, streams=(dataclasses.replace(meta.streams[0], channels=3),) + meta.streams[1:])
    with pytest.raises(EpisodeContractError):
        validate_episode(bad).raise_if_errors()


def test_mapanything_rig_and_world_depth_match_ground_truth(run):
    a = run["adapter"]
    for stage in ("mapanything_rig_depth", "mapanything_depth"):
        for ref in ("left", "right"):
            depth, valid, K = load_view_depth(run["root"], run["ep"], f"rig0/{ref}", [3], stage)
            _, gt, _, _ = a.render("rig0", ref, 3)
            m = valid[0] & (gt > 0)
            rel = np.abs(depth[0][m] - gt[m]) / gt[m]
            assert m.mean() > 0.6 and np.median(rel) < 0.03, (ref, m.mean(), np.median(rel))


def test_calibration_accepts_metric_rigs(run):
    summ = run["recs"]["calibrate"]["summary"]
    assert summ["fk_vs_recorded_ee"]["max_trans_m"] < 1e-9  # URDF FK agrees with the recorded flange pose
    for cam in ("rig0/left", "rig1/left"):
        assert summ[cam]["status"] == "accepted", summ[cam]
    doc = json.loads((rio.stage_dir(run["root"], run["ep"], "calibrate") / "rig0__left.json").read_text())
    T = np.asarray(doc["T_cam_base"])
    assert np.abs(T - run["adapter"].T_cam_base("rig0", "left")).max() < 0.03
    assert list((rio.stage_dir(run["root"], run["ep"], "calibrate") / "overlays").glob("rig0__left_*.png"))


def test_lifted_tracks_match_ground_truth_and_flag_occlusion(run):
    d = rio.stage_dir(run["root"], run["ep"], "lift3d")
    rec = run["recs"]["lift3d"]["summary"]
    assert "rig2/left" in rec and "skipped" in rec["rig2/left"]
    with np.load(d / rec["rig0/left"]["chunks"][0]) as z:
        xyz, status, frames = z["xyz_base"], z["status"], z["frame_indices"]
        assert str(z["camera_id"]) == "rig0/left"
    H, W = run["adapter"].hw
    from s4d.realworld.tracking import query_grid

    rc = query_grid((H, W), 4, 2)
    P, vis = run["adapter"].gt_points("rig0/left", int(frames[0]), rc, list(frames))
    ok = status == VALID
    assert ok.mean() > 0.4
    err = np.linalg.norm(xyz[ok] - P[ok], axis=-1)
    assert np.median(err) < 0.02, np.median(err)
    assert (status[~vis] != VALID).all() and (status[~vis] == OCCLUDED).any()
    assert np.isnan(xyz[~ok]).all()


def test_export_matches_training_contract(run):
    cache = run["tmp"] / "cache"
    manifest = json.loads((cache / "manifest.json").read_text())
    assert manifest["MOCK_MODEL_OUTPUTS"] is True
    assert manifest["cameras"] == ["rig0/left", "rig1/left"] and "rig2/left" in manifest["excluded_cameras"]
    assert "NOT rgb" in manifest["image_modality"]["rig0/left"]
    with pytest.warns(UserWarning):
        ds = RealworldCacheDataset(cache)
    item = ds[0]
    assert item["images"].shape == (3, 2, 3, 144, 256) and item["track_src"].shape[:2] == (3, 2)
    batch = {k: v if k == "meta" else v.unsqueeze(0) for k, v in item.items()}
    validate_batch(batch)
    val = [w for w in manifest["windows"] if w["split"] == "validation"]
    train = [w for w in manifest["windows"] if w["split"] == "train"]
    assert val and train
    # no frame shared between splits; validation windows lie in the last 4 of 10 frames
    assert max(max(w["canonical_indices"]) for w in train) < min(min(w["canonical_indices"]) for w in val)
    assert min(min(w["canonical_indices"]) for w in val) >= 6
    with pytest.warns(UserWarning):
        assert len(RealworldCacheDataset(cache, split="validation")) == len(val)
    closure = item["probe_state"][:, 6].numpy()
    assert np.all((closure >= 0) & (closure <= 1))


def test_export_refuses_mock_and_infrared_without_opt_in(run):
    with pytest.raises(RuntimeError, match="MOCK"):
        export(run["root"], run["ep"], run["tmp"] / "cache2")
    with pytest.raises(RuntimeError, match="infrared"):
        export(run["root"], run["ep"], run["tmp"] / "cache3", allow_mock=True)


def test_stages_are_idempotent(run):
    r = run
    assert ingest(r["adapter"], r["ep"], r["root"])["skipped"]
    assert stereo_rectify(r["root"], r["ep"])["skipped"]
    assert mapanything_rig_depth(r["root"], r["ep"], mock_mapanything(r["root"]))["skipped"]
    assert mapanything_depth(r["root"], r["ep"], mock_mapanything(r["root"]),
                             input_depth_stage="mapanything_rig_depth")["skipped"]
    assert calibrate(r["root"], r["ep"], r["landmark_source"], landmark_links=LANDMARKS, cfg=r["cfg"])["skipped"]
    assert lift3d(r["root"], r["ep"])["skipped"]
    # a changed config rebuilds (and the old complete stage stays until the swap)
    assert not lift3d(r["root"], r["ep"], vis_threshold=0.6)["skipped"]
    assert rio.read_record(rio.stage_dir(r["root"], r["ep"], "lift3d"))["config"]["vis_threshold"] == 0.6
    # a forced upstream rebuild (identical config) still invalidates every downstream stage via its build id
    assert not stereo_rectify(r["root"], r["ep"], force=True)["skipped"]
    assert not mapanything_rig_depth(r["root"], r["ep"], mock_mapanything(r["root"]))["skipped"]
    assert not calibrate(r["root"], r["ep"], r["landmark_source"], landmark_links=LANDMARKS, cfg=r["cfg"])["skipped"]
    assert not track2d(r["root"], r["ep"], OracleTracker(r["adapter"]), window=10, step=8, grid_stride=4)["skipped"]


def test_unrectified_rig_is_rectified_before_matching(tmp_path):
    """Right camera rotated by ~1 deg: the stage rectifies (Bouguet), matches on the rectified pair and returns depth on
    the rectified-left grid, which equals a direct render of the rectified-left virtual camera."""
    from s4d.realworld.geometry import make_T
    from s4d.realworld.pipeline import load_meta, rig_rectification
    from s4d.realworld.synthetic import render_box_scene

    a = get_adapter("synthetic", {"asset_dir": str(tmp_path / "a"), "frames": 2,
                                  "right_rotation_rpy_deg": (0.4, -0.8, 0.6)})
    root = tmp_path / "out"
    ingest(a, "synthetic_ep000", root)
    rec = stereo_rectify(root, "synthetic_ep000")
    mapanything_rig_depth(root, "synthetic_ep000", mock_mapanything(root))
    assert rec["summary"]["rig0"]["rectified_here"] is True
    rect, K_rect, _, hw = rig_rectification(load_meta(root, "synthetic_ep000"), "rig0")
    depth, _, K = load_view_depth(root, "synthetic_ep000", "rig0/left", [1], "mapanything_rig_depth")
    assert np.allclose(K, K_rect)
    T_rect_base = make_T(rect.R1, np.zeros(3)) @ a.T_cam_base("rig0", "left")
    _, gt, _, _ = render_box_scene(T_rect_base, K_rect, hw, a.time(1))
    m = (depth[0] > 0) & (gt > 0)
    border = np.zeros_like(m)
    border[6:-6, 30:-6] = True
    m &= border
    assert m.mean() > 0.4
    assert np.median(np.abs(depth[0][m] - gt[m]) / gt[m]) < 0.03


class ResizedOracleTracker(TrackerBackend):
    """Oracle on whatever grid it is given: maps its query pixel centres to the full grid, then back (MOCK)."""

    identity = {"name": "synthetic oracle tracker (any resolution)", "is_mock": True}

    def __init__(self, adapter):
        self.adapter = adapter

    def track(self, camera_id, frames, frame_indices, query):
        from s4d.realworld.geometry import resize_affine

        H, W = frames.shape[1:3]
        aff = resize_affine(self.adapter.hw, (H, W))
        rc = np.stack(np.meshgrid(np.arange(H), np.arange(W), indexing="ij"), -1).reshape(-1, 2)
        uv_full = aff.inverse().apply_uv(rc[:, ::-1] + 0.5)
        uv = self.adapter.gt_uv(camera_id, frame_indices[query], uv_full, frame_indices[query:])
        flows = np.stack([(aff.apply_uv(u) - 0.5 - rc[:, ::-1]).T.reshape(2, H, W) for u in uv])
        return flows.astype(np.float32), np.ones((len(uv), 2, H, W), np.float32)


def test_tracking_at_reduced_resolution_maps_back_to_the_depth_grid(tmp_path):
    from s4d.realworld.pipeline import load_tracks2d

    a = get_adapter("synthetic", {"asset_dir": str(tmp_path / "a"), "frames": 4})
    root = tmp_path / "out"
    ingest(a, "synthetic_ep000", root)
    stereo_rectify(root, "synthetic_ep000")
    rec = track2d(root, "synthetic_ep000", ResizedOracleTracker(a), window=4, step=4, grid_stride=4, tracker_hw=(48, 72))
    name = rec["summary"]["rig0/left"]["chunks"][0]
    tr = load_tracks2d(rio.stage_dir(root, "synthetic_ep000", "track2d") / name, "rig0/left")
    assert tuple(tr.image_hw) == a.hw and "tracker_affine_full_to_tracker" in tr.source
    # queries are tracker pixel centres (2x coarser): on the full grid they sit at 2 * (rc + .5)
    from s4d.realworld.tracking import query_grid

    q = query_grid((48, 72), 4, 2)
    expected = 2 * (q[:, ::-1] + 0.5)
    hit = np.isfinite(tr.uv[0]).all(-1)  # rays that miss the scene have no surface point
    assert hit.mean() > 0.8 and np.allclose(tr.uv[0][hit], expected[hit], atol=1e-4)
    gt = a.gt_uv("rig0/left", 0, expected[hit], list(tr.frame_indices))
    assert np.abs(tr.uv[:, hit] - gt).max() < 1e-3


def test_rectification_records_only_for_rectified_inputs_and_rejects_missing_grid(run, tmp_path):
    from s4d.realworld.pipeline import calibration_image, default_depth_stage, grid_frames

    root, ep = run["root"], run["ep"]
    meta = load_meta(root, ep)
    assert default_depth_stage(meta, "calibrate") == "mapanything_rig_depth"
    assert default_depth_stage(meta, "lift") == "mapanything_depth"
    g = meta.stereo[0]
    with h5py.File(rio.stage_dir(root, ep, "stereo_rectify") / "rig0.h5", "r") as f:
        assert list(f) == []  # no copied frames for inputs already rectified
        np.testing.assert_array_equal(f.attrs["K_left"], meta.stream(g.left_stream).intrinsics.K())
        np.testing.assert_array_equal(f.attrs["T_rr_rl"], g.T())
    a = get_adapter("synthetic", {"asset_dir": str(tmp_path / "a"), "frames": 2,
                                  "right_rotation_rpy_deg": (0.4, -0.8, 0.6)})
    other = tmp_path / "out"
    ingest(a, "synthetic_ep000", other)
    raw_meta = load_meta(other, "synthetic_ep000")
    with pytest.raises(RuntimeError, match="run stereo_rectify first"):
        grid_frames(other, "synthetic_ep000", raw_meta, raw_meta.stereo[0], [0])
    with pytest.raises(RuntimeError, match="run stereo_rectify first"):
        calibration_image(other, "synthetic_ep000", "rig0", "rig0/left", 0)

    from dataclasses import replace

    mixed = replace(raw_meta, streams=(replace(raw_meta.streams[0], rectified=True),) + raw_meta.streams[1:])
    with pytest.raises(RuntimeError, match="run stereo_rectify first"):
        grid_frames(other, "synthetic_ep000", mixed, mixed.stereo[0], [0])


def test_stereo_world_views_use_rectified_right_pose_and_optional_right_outputs(tmp_path):
    from s4d.realworld.geometry import make_T
    from s4d.realworld.pipeline import calibrate_provided, grid_frames, rig_rectification

    a = get_adapter("synthetic", {"asset_dir": str(tmp_path / "a"), "frames": 2,
                                  "right_rotation_rpy_deg": (0.4, -0.8, 0.6)})
    root, ep = tmp_path / "out", "synthetic_ep000"
    ingest(a, ep, root)
    stereo_rectify(root, ep)
    meta = load_meta(root, ep)
    backend = mock_mapanything(root)
    mapanything_rig_depth(root, ep, backend)  # succeeds before there is any world calibration
    with pytest.raises(RuntimeError, match="accepted calibration"):
        mapanything_depth(root, ep, backend)
    provided = {}
    for g in meta.stereo:
        if g.status != "metric":
            continue
        rect, _, _, _ = rig_rectification(meta, g.rig_id)
        T = make_T(rect.R1, np.zeros(3)) @ a.T_cam_base(g.rig_id, "left")
        provided[g.left_stream] = {"T_cam_base": T, "source": {"kind": "synthetic oracle", "is_mock": True}}
    calibrate_provided(root, ep, provided)
    calls = []
    worker = backend.runner.worker

    def capture(spec, task, params, arrays, **kwargs):
        calls.append(arrays)
        return worker(spec, task, params, arrays, **kwargs)

    backend.runner.worker = capture
    rec = mapanything_depth(root, ep, backend, input_depth_stage="mapanything_rig_depth")
    streams = rec["summary"]["views"]
    assert streams == ["rig0/left", "rig0/right", "rig1/left", "rig1/right"]
    for g in meta.stereo[:2]:
        v = streams.index(g.left_stream)
        with h5py.File(rio.stage_dir(root, ep, "stereo_rectify") / f"{g.rig_id}.h5", "r") as f:
            T = f.attrs["T_rr_rl"]
            np.testing.assert_allclose(T[:3, :3], np.eye(3), atol=1e-10)
            np.testing.assert_allclose(T[:3, 3], [-g.baseline_m, 0, 0], atol=1e-10)
            np.testing.assert_array_equal(calls[0]["K"][v + 1], f.attrs["K_right"])
        left_pose = provided[g.left_stream]["T_cam_base"]
        np.testing.assert_allclose(calls[0]["cam2world"][v], np.linalg.inv(left_pose))
        np.testing.assert_allclose(calls[0]["cam2world"][v + 1], np.linalg.inv(T @ left_pose))
        expected = grid_frames(root, ep, meta, g, [0], "right")[0]
        np.testing.assert_array_equal(calls[0]["images"][0, v + 1], np.repeat(expected[..., None], 3, -1))
    left_only = mapanything_depth(root, ep, backend, input_depth_stage="mapanything_rig_depth", include_right_eyes=False)
    assert left_only["summary"]["views"] == ["rig0/left", "rig1/left"]
    assert not (rio.stage_dir(root, ep, "mapanything_depth") / "rig0__right.h5").exists()


def test_stereo_landmarks_without_depth_still_use_the_rectified_grid(tmp_path, monkeypatch):
    from dataclasses import replace

    from s4d.realworld.depth_stages import view_K
    from s4d.realworld.pipeline import rig_rectification

    a = get_adapter("synthetic", {"asset_dir": str(tmp_path / "a"), "frames": 2,
                                  "right_rotation_rpy_deg": (0.4, -0.8, 0.6)})
    read = a.read_meta

    def distorted(ep):
        meta = read(ep)
        return replace(meta, streams=tuple(replace(s, intrinsics=replace(s.intrinsics,
                         distortion_model="opencv_plumb_bob", distortion=(0.1, -0.01, 0.001, 0.001, 0)))
                         for s in meta.streams))

    monkeypatch.setattr(a, "read_meta", distorted)
    root, ep = tmp_path / "out", "synthetic_ep000"
    ingest(a, ep, root)
    stereo_rectify(root, ep)
    uv = np.array([[61.3, 47.6]])
    runner = LocalRunner(tmp_path / "worker")
    captured = []

    def capture(task, params, arrays, **kwargs):
        captured.append(arrays)
        raise RuntimeError("captured calibration inputs")

    monkeypatch.setattr(runner, "kernel", capture)
    with pytest.raises(RuntimeError, match="captured calibration inputs"):
        calibrate(root, ep, lambda *args: (uv, np.ones(1)), landmark_links=["panda_link2"],
                  cfg=CalibrationConfig(depth_term=False), overlays=0, runner=runner)
    meta = load_meta(root, ep)
    _, K, _, _ = rig_rectification(meta, "rig0")
    np.testing.assert_array_equal(captured[0]["K"], K)
    np.testing.assert_array_equal(view_K(meta, "rig0/left", root=root, episode_id=ep), K)
    np.testing.assert_array_equal(captured[0]["landmarks_uv"], np.repeat(uv[None], 2, axis=0))
    assert "depth" not in captured[0]


def test_stereo_depth_stages_require_pose_conditioning(run):
    backend = mock_mapanything(run["root"])
    backend.params["use_pose"] = False
    for stage in (mapanything_rig_depth, mapanything_depth):
        with pytest.raises(ValueError, match="use_pose=True"):
            stage(run["root"], run["ep"], backend)
