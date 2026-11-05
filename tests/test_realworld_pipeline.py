"""End-to-end synthetic RGB-D episode: ingest -> sensor_depth -> calibration -> TrackEverything -> export -> loader.

All model outputs here are MOCK (oracle landmarks, oracle TrackEverything predictions through the mock worker) and are
flagged as such; this validates conventions, bookkeeping, idempotency and the export contract, not model integration.
The registered sensor depth stands in for the IR stereo depth (same layout) in the tracking and export stages."""

import json

import numpy as np
import pytest

from s4d.data.contract import validate_batch
from s4d.realworld import io as rio
from s4d.realworld.calibration import CalibrationConfig
from s4d.realworld.contract import EpisodeContractError, validate_episode
from s4d.realworld.dataset import RealworldCacheDataset
from s4d.realworld.depth_stages import load_view_depth, sensor_depth
from s4d.realworld.export import export
from s4d.realworld.geometry import project, transform_points
from s4d.realworld.pipeline import RobotGeometry, calibrate, camera_views, ingest, load_meta, robot_state_at
from s4d.realworld.robot import URDFModel, landmarks
from s4d.realworld.runners import LocalRunner
from s4d.realworld.trackeverything_stage import track_everything
from s4d.realworld.tracking import OCCLUDED, VALID
from tests.test_realworld_trackeverything import MOCK_SPEC, ColourEpisode, SyntheticRunner

LANDMARKS = ["panda_link2", "panda_link3", "panda_link4", "panda_link5", "panda_link7", "panda_link8",
             "gripper_base", "left_finger", "right_finger"]
# tiny synthetic images: the uncertainty gates are relaxed here (defaults are tested in the calibration tests)
CALIBRATION = CalibrationConfig(depth_term=False, min_visible_fraction=0.05, min_motion_extent_m=0.03,
                                max_rot_std_deg=2.0, max_trans_std_m=0.03)
TE = dict(depth_stage="sensor_depth", window=10, step=8, grid_stride=4, model_hw=(64, 96))


def synthetic_rgbd_episode(tmp):
    """ingest -> sensor_depth -> calibrate (noisy oracle landmarks) -> track_everything (oracle worker) under
    ``tmp/out``; three RGB-D rigs (rig2's geometry unverified), 10 frames, no recording pause."""
    adapter = ColourEpisode(tmp / "assets", frames=10, pause=10, image_hw=(96, 144), focal=150.0)
    root, ep = tmp / "out", adapter.list_episodes()[0]
    rng = np.random.default_rng(0)
    recs = {"ingest": ingest(adapter, ep, root)}
    meta = load_meta(root, ep)
    model, geom = URDFModel.from_file(meta.robot.urdf_path), RobotGeometry(meta)

    def landmark_source(cam, i, links):
        q, g, _, _ = robot_state_at(meta, meta.stream_timestamps_ns[cam][[i]])
        L = landmarks(model, geom.q(q[0], g[0]), links)
        uv, _ = project(transform_points(adapter.T_cam_base(cam.split("/")[0], "left"), L), adapter.K())
        return uv + rng.normal(0, 0.5, uv.shape), np.ones(len(L))

    landmark_source.identity = {"name": "synthetic oracle landmarks", "is_mock": True}
    kernels = LocalRunner(tmp / "kernels")
    recs["sensor_depth"] = sensor_depth(root, ep, kernels)
    recs["calibrate"] = calibrate(root, ep, landmark_source, landmark_links=LANDMARKS, cfg=CALIBRATION)
    runner = SyntheticRunner(tmp / "worker", adapter, check_pose=False)
    recs.update(track_everything(root, ep, MOCK_SPEC, runner, **TE))
    return {"adapter": adapter, "root": root, "ep": ep, "recs": recs, "tmp": tmp, "landmark_source": landmark_source,
            "kernels": kernels, "runner": runner}


@pytest.fixture(scope="module")
def run(tmp_path_factory):
    out = synthetic_rgbd_episode(tmp_path_factory.mktemp("rw"))
    out["recs"]["export"] = export(out["root"], out["ep"], out["tmp"] / "cache", resolutions={"scratch": (144, 256)},
                                   depth_stage="sensor_depth", allow_mock=True, validation_tail_frames=4)
    return out


def test_ingest_preserves_modality_and_views(run):
    meta = load_meta(run["root"], run["ep"])
    assert {s.modality for s in meta.streams} == {"rgb", "depth"} and not meta.stereo
    assert [v.stream_id for v in camera_views(meta)] == ["rig0/left", "rig1/left", "rig2/left"]
    assert meta.rig_rgbd("rig2").status == "unverified"
    # a colour stream claiming 1 channel is rejected by the contract
    import dataclasses

    bad = dataclasses.replace(meta, streams=(dataclasses.replace(meta.streams[0], channels=1),) + meta.streams[1:])
    with pytest.raises(EpisodeContractError):
        validate_episode(bad).raise_if_errors()


def test_stereo_camera_views_are_refused(tmp_path):
    from s4d.realworld.adapters import get_adapter
    from s4d.realworld.pipeline import default_depth_stage

    adapter = get_adapter("synthetic", {"asset_dir": str(tmp_path / "assets"), "frames": 4})
    ep = adapter.list_episodes()[0]
    ingest(adapter, ep, tmp_path / "out")  # the contract (and ingest) still represent stereo rigs
    meta = load_meta(tmp_path / "out", ep)
    for call in (lambda: camera_views(meta), lambda: default_depth_stage(meta, "lift")):
        with pytest.raises(NotImplementedError, match="stereo camera views"):
            call()


def test_sensor_depth_matches_ground_truth(run):
    a = run["adapter"]
    depth, valid, K = load_view_depth(run["root"], run["ep"], "rig0/left", [3], "sensor_depth")
    _, gt, _, _ = a.render("rig0", "left", 3)
    m = valid[0] & (gt > 0)
    assert m.mean() > 0.8 and np.median(np.abs(depth[0][m] - gt[m])) < 1e-3
    np.testing.assert_allclose(K, a.K())
    assert "skipped" in run["recs"]["sensor_depth"]["summary"]["rig2/left"]  # unverified RGB-D geometry


def test_calibration_accepts_metric_rigs(run):
    summ = run["recs"]["calibrate"]["summary"]
    assert summ["fk_vs_recorded_ee"]["max_trans_m"] < 1e-9  # URDF FK agrees with the recorded flange pose
    for cam in ("rig0/left", "rig1/left"):
        assert summ[cam]["status"] == "accepted", summ[cam]
    doc = json.loads((rio.stage_dir(run["root"], run["ep"], "calibrate") / "rig0__left.json").read_text())
    T = np.asarray(doc["T_cam_base"])
    assert np.abs(T - run["adapter"].T_cam_base("rig0", "left")).max() < 0.03
    assert doc["diagnostics"]["depth_stage"] == "sensor_depth" and doc["diagnostics"]["depth_available"]
    assert list((rio.stage_dir(run["root"], run["ep"], "calibrate") / "overlays").glob("rig0__left_*.png"))


def test_tracks_match_ground_truth_and_flag_occlusion(run):
    from s4d.realworld.tracking import query_grid

    d = rio.stage_dir(run["root"], run["ep"], "lift3d")
    rec = run["recs"]["lift3d"]["summary"]
    assert "skipped" in rec["rig2/left"]
    with np.load(d / rec["rig0/left"]["chunks"][0]) as z:
        xyz, status, frames = z["xyz_base"], z["status"], z["frame_indices"]
        assert str(z["camera_id"]) == "rig0/left"
    rc = query_grid(run["adapter"].hw, 4, 2)
    P, vis = run["adapter"].gt_points("rig0/left", int(frames[0]), rc, list(frames))
    ok = status == VALID
    assert ok.mean() > 0.4
    np.testing.assert_allclose(xyz[ok], P[ok], atol=1e-4)  # oracle predictions restored to the base frame
    assert (status[~vis] != VALID).all() and (status[~vis] == OCCLUDED).any()
    assert np.isnan(xyz[~ok]).all()


def test_export_matches_training_contract(run):
    cache = run["tmp"] / "cache"
    manifest = json.loads((cache / "manifest.json").read_text())
    assert manifest["MOCK_MODEL_OUTPUTS"] is True
    assert manifest["cameras"] == ["rig0/left", "rig1/left"] and "rig2/left" in manifest["excluded_cameras"]
    assert manifest["image_modality"]["rig0/left"] == "rgb" and manifest["depth_stage"] == "sensor_depth"
    assert manifest["motion_target"].startswith("per-camera single-view TrackEverything 3D tracks")
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


def test_export_refuses_mock_without_opt_in(run):
    with pytest.raises(RuntimeError, match="MOCK"):
        export(run["root"], run["ep"], run["tmp"] / "cache2", depth_stage="sensor_depth")


def test_stages_are_idempotent(run):
    r = run
    root, ep = r["root"], r["ep"]

    def cal(**kw):
        return calibrate(root, ep, r["landmark_source"], landmark_links=LANDMARKS, cfg=CALIBRATION, **kw)

    def te():
        return track_everything(root, ep, MOCK_SPEC, r["runner"], **TE)

    assert ingest(r["adapter"], ep, root)["skipped"]
    assert sensor_depth(root, ep, r["kernels"])["skipped"]
    assert cal()["skipped"]
    assert all(rec["skipped"] for rec in te().values())
    # a changed config rebuilds (and the old complete stage stays until the swap)
    assert not cal(every=2)["skipped"]
    assert rio.read_record(rio.stage_dir(root, ep, "calibrate"))["config"]["every"] == 2
    # a forced upstream rebuild (identical config) still invalidates every downstream stage via its build id
    assert not sensor_depth(root, ep, r["kernels"], force=True)["skipped"]
    assert not cal(every=2)["skipped"]
    assert not any(rec["skipped"] for rec in te().values())
