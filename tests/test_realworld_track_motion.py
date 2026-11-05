"""Dynamic-track filter (stage ``track_motion``), the moving-only export and the per-dataset baseline scope."""

import json
from pathlib import Path

import numpy as np
import pytest
import yaml

from s4d.realworld.track_motion import (
    INVALID,
    MOVING,
    STATIC,
    UNCERTAIN,
    MotionFilterConfig,
    decide,
    reference_thresholds,
    segment_stats,
    segment_tracks,
)
from s4d.realworld.tracking import NO_DEPTH, OCCLUDED, VALID

ROOT = Path(__file__).resolve().parents[1]
CFG = MotionFilterConfig()
THR = {"px_move": 2.0, "px_static": 0.5,
       "z_bins": [{"z_lo": lo, "z_hi": hi, "z_move_m": 0.015, "z_static_m": 0.006}
                  for lo, hi in zip(CFG.depth_bins_m[:-1], CFG.depth_bins_m[1:])]}


def _track(uv, z=1.0, f=600.0):
    """(T,2) pixels at constant depth -> (uv (T,1,2), xyz_cam (T,1,3)) on a 640x480 pinhole."""
    uv = np.asarray(uv, float)[:, None, :]
    x = (uv[..., 0] - 320) * z / f
    y = (uv[..., 1] - 240) * z / f
    return uv, np.stack([x, y, np.full_like(x, z)], -1)


def _classify(uv, xyz, status=None, f=600.0):
    T = len(uv)
    status = np.full((T, 1), VALID, np.uint8) if status is None else status
    seg, _ = segment_tracks(status, xyz, np.arange(T) / 15.0, CFG, 1 / 15.0)
    return decide(segment_stats(uv, xyz, seg, f, CFG.min_segment_samples)[0], THR, CFG)


def test_segments_break_at_identity_loss_long_depth_holes_pauses_and_jumps():
    T = 9
    status = np.full((T, 4), VALID, np.uint8)
    status[3, 0] = OCCLUDED  # 2D identity lost -> break
    status[3:5, 1] = NO_DEPTH  # 2-frame depth hole, 2D still tracked -> bridged
    status[2:5, 2] = NO_DEPTH  # 3-frame hole -> break
    xyz = np.zeros((T, 4, 3))
    xyz[..., 2] = 1.0
    xyz[6:, 3, 0] = 0.5  # 0.5 m in one 1/15 s step: implausible -> break
    seg, counts = segment_tracks(status, xyz, np.arange(T) / 15.0, CFG, 1 / 15.0)
    assert list(seg[:, 0]) == [0, 0, 0, -1, 1, 1, 1, 1, 1]
    assert list(seg[:, 1]) == [0, 0, 0, -1, -1, 0, 0, 0, 0]
    assert list(seg[:, 2]) == [0, 0, -1, -1, -1, 1, 1, 1, 1]
    assert list(seg[:, 3]) == [0] * 6 + [1] * 3
    assert counts["identity"] == 1 and counts["depth_hole"] == 1 and counts["speed"] == 1
    times = np.arange(T) / 15.0
    times[5:] += 2.0  # recording pause between frames 4 and 5 breaks every track
    seg, counts = segment_tracks(np.full((T, 1), VALID, np.uint8), np.ones((T, 1, 3)), times, CFG, 1 / 15.0)
    assert list(seg[:, 0]) == [0] * 5 + [1] * 4 and counts["time_gap"] == 1


def test_static_noise_and_single_spikes_are_not_motion():
    rng = np.random.default_rng(0)
    uv = np.array([[300.0, 200.0]] * 16) + rng.normal(0, 0.08, (16, 2))
    assert _classify(*_track(uv))[0] == STATIC
    uv[7] += 25.0  # one noisy jump: the median of 3 removes it
    assert _classify(*_track(uv))[0] == STATIC


def test_slow_motion_is_kept_with_its_stationary_interval():
    uv = np.array([[300.0, 200.0]] * 16)
    uv[:8, 0] += 0.6 * np.arange(8)  # 0.6 px/frame for 8 frames (~7 mm at 1 m), then stationary
    uv[8:, 0] = uv[7, 0]
    cls, best = _classify(*_track(uv))
    assert cls == MOVING and best["by"] == "px" and best["samples"] == 16  # whole trajectory, stops included


def test_pixel_motion_too_small_in_metres_is_uncertain_not_moving():
    uv = np.array([[300.0, 200.0]] * 16)
    uv[:, 0] += np.linspace(0, 3.0, 16)  # 3 px at 0.5 m with f=600: 2.5 mm < 5 mm
    assert _classify(*_track(uv, z=0.5))[0] == UNCERTAIN


def test_depth_axis_motion_needs_its_own_threshold_and_support():
    uv = np.array([[320.0, 240.0]] * 16)
    _, xyz = _track(uv)
    xyz[8:, 0, 2] += 0.03  # 3 cm towards the camera axis, persistent: moving by depth
    cls, best = _classify(uv[:, None, :], xyz)
    assert cls == MOVING and best["by"] == "z"
    _, xyz = _track(uv)
    xyz[15, 0, 2] += 0.03  # last sample only: median-of-3 never sees it
    assert _classify(uv[:, None, :], xyz)[0] == STATIC


def test_too_few_valid_samples_are_invalid():
    uv, xyz = _track(np.array([[300.0, 200.0]] * 16))
    status = np.full((16, 1), OCCLUDED, np.uint8)
    status[::4] = VALID  # isolated samples: no segment of 5
    assert _classify(uv, xyz, status)[0] == INVALID


def test_reference_thresholds_follow_the_observed_quantiles_and_refuse_thin_references():
    rng = np.random.default_rng(1)
    px = np.abs(rng.normal(0, 0.3, 5000))
    z = np.abs(rng.normal(0, 0.004, 5000))
    depth = rng.uniform(0.6, 1.0, 5000)
    thr = reference_thresholds(px, z, depth, CFG)
    assert thr["px_move"] == pytest.approx(max(CFG.min_move_px, np.quantile(px, 0.995)))
    assert thr["px_static"] == pytest.approx(np.quantile(px, 0.9))
    near = [b for b in thr["z_bins"] if b["z_lo"] == 0.6][0]
    assert near["z_move_m"] == pytest.approx(max(CFG.min_move_m, np.quantile(z[depth < 0.8], 0.995)))
    far = [b for b in thr["z_bins"] if b["z_lo"] == 2.0][0]
    assert far.get("uncalibrated") and far["z_move_m"] is None and far["z_static_m"] is None  # never borrowed
    with pytest.raises(ValueError, match="reference"):
        reference_thresholds(px[:10], z[:10], depth[:10], CFG)


# ------------------------------------------------------------------ synthetic end-to-end (mock models, flagged)


@pytest.fixture(scope="module")
def filtered(tmp_path_factory):
    from tests.test_realworld_pipeline import LANDMARKS, OracleTracker, mock_mapanything

    from s4d.realworld.adapters import get_adapter
    from s4d.realworld.calibration import CalibrationConfig
    from s4d.realworld.depth_stages import mapanything_depth, mapanything_rig_depth
    from s4d.realworld.export import export
    from s4d.realworld.geometry import project, transform_points
    from s4d.realworld.pipeline import (
        RobotGeometry,
        calibrate,
        ingest,
        lift3d,
        load_meta,
        robot_state_at,
        stereo_rectify,
        track2d,
    )
    from s4d.realworld.robot import URDFModel, landmarks
    from s4d.realworld.track_motion import track_motion

    tmp = tmp_path_factory.mktemp("tm")
    adapter = get_adapter("synthetic", {"asset_dir": str(tmp / "assets"), "frames": 10})
    root, ep = tmp / "out", adapter.list_episodes()[0]
    ingest(adapter, ep, root)
    meta = load_meta(root, ep)
    model, geom = URDFModel.from_file(meta.robot.urdf_path), RobotGeometry(meta)
    rng = np.random.default_rng(0)

    def landmark_source(cam, i, links):
        q, g, _, _ = robot_state_at(meta, meta.stream_timestamps_ns[cam][[i]])
        uv, _ = project(transform_points(adapter.T_cam_base(cam.split("/")[0], "left"),
                                         landmarks(model, geom.q(q[0], g[0]), links)), adapter.K())
        return uv + rng.normal(0, 0.5, uv.shape), np.ones(len(links))

    landmark_source.identity = {"name": "synthetic oracle landmarks", "is_mock": True}
    stereo_rectify(root, ep)
    mapanything_rig_depth(root, ep, mock_mapanything(root))
    calibrate(root, ep, landmark_source, landmark_links=LANDMARKS,
              cfg=CalibrationConfig(depth_term=False, min_visible_fraction=0.05, min_motion_extent_m=0.03,
                                    max_rot_std_deg=2.0, max_trans_std_m=0.03))
    mapanything_depth(root, ep, mock_mapanything(root), input_depth_stage="mapanything_rig_depth")
    track2d(root, ep, OracleTracker(adapter), window=10, step=8, grid_stride=4)
    lift3d(root, ep)
    cfg = MotionFilterConfig(min_bin_samples=10, min_segment_samples=4, reference_robot_distance_m=0.05)
    rec = track_motion(root, ep, cfg=cfg)
    kw = dict(resolutions={"scratch": (144, 256)}, allow_infrared_input=True, allow_mock=True)
    export(root, ep, tmp / "all", **kw)
    export(root, ep, tmp / "moving", track_source="track_motion", **kw)
    return {"root": root, "ep": ep, "tmp": tmp, "rec": rec, "cfg": cfg}


def test_track_motion_stage_records_counts_thresholds_and_is_idempotent(filtered):
    from s4d.realworld.track_motion import track_motion

    s = filtered["rec"]["summary"]
    c = s["counts"]
    assert c["candidate"] == c["invalid"] + c["static"] + c["moving"] + c["uncertain"] > 0
    assert c["moving"] > 0 and c["static"] > 0  # the robot moves, the box scene does not
    for cam in s["cameras"].values():
        if "counts" in cam:
            assert cam["thresholds"]["px_move"] >= filtered["cfg"].min_move_px
    assert "inference is unchanged" in s["note"]
    assert track_motion(filtered["root"], filtered["ep"], cfg=filtered["cfg"])["skipped"]


def test_moving_only_export_keeps_the_contract_and_never_pairs_across_segments(filtered):
    import h5py

    from s4d.data.contract import validate_batch
    from s4d.data.droid.pointworld import PAIR_TIMES
    from s4d.realworld import io as rio
    from s4d.realworld.dataset import RealworldCacheDataset
    from s4d.realworld.track_motion import load_motion

    root, ep, tmp = filtered["root"], filtered["ep"], filtered["tmp"]
    m_all = json.loads((tmp / "all" / "manifest.json").read_text())
    m_mov = json.loads((tmp / "moving" / "manifest.json").read_text())
    assert m_all["track_source"] == "lift3d" and m_mov["track_source"] == "track_motion"
    assert m_mov["track_motion"]["moving"] == filtered["rec"]["summary"]["counts"]["moving"]
    assert m_mov["windows"] == m_all["windows"]
    with h5py.File(tmp / "all" / "scratch.h5", "r") as fa, h5py.File(tmp / "moving" / "scratch.h5", "r") as fm:
        for wi, w in enumerate(m_mov["windows"]):
            ga, gm = fa[f"windows/{wi:05d}"], fm[f"windows/{wi:05d}"]
            assert set(ga) == set(gm)
            np.testing.assert_array_equal(ga["images"][:], gm["images"][:])  # static geometry unchanged
            np.testing.assert_array_equal(ga["depth"][:], gm["depth"][:])
            for v, cam in enumerate(m_mov["cameras"]):
                info = w["cameras"][cam]
                mo = load_motion(root, ep, info["chunk"])
                keep = np.flatnonzero(mo["class"] == 2)
                assert gm["track_count"][v] == len(keep)
                pos = [list(mo["frame_indices"]).index(f) for f in info["frames"]]
                seg = mo["segment"][pos][:, keep]
                weight = gm["track_weight"][:, v, :len(keep)]
                for pair, (a, b) in enumerate(PAIR_TIMES):
                    on = weight[pair] > 0
                    assert (seg[a, on] == seg[b, on]).all() and (seg[a, on] >= 0).all()
                assert not gm["track_weight"][:, v, len(keep):].any()  # padding unsupervised
    with pytest.warns(UserWarning):
        ds = RealworldCacheDataset(tmp / "moving")
    item = ds[0]
    validate_batch({k: v if k == "meta" else v.unsqueeze(0) for k, v in item.items()})
    assert rio.read_record(tmp / "moving")["config"]["track_source"] == "track_motion"
    assert "track_source" not in rio.read_record(tmp / "all")["config"]  # default export fingerprint unchanged


# ------------------------------------------------------------------ baseline scope per dataset config


def _cfg(name):
    return yaml.safe_load((ROOT / f"configs/realworld/{name}.yaml").read_text())


def test_droid_config_runs_no_baseline_stages_and_refuses_them():
    from s4d.realworld.baseline_stages import required_baseline_stages
    from s4d.realworld.cli import check_stage_scope, default_stages

    cfg = _cfg("droid_raw_validation")
    assert cfg["baselines"] == [] and required_baseline_stages(cfg) == ()
    stages = default_stages(cfg)
    assert "sam3_masks" not in stages and "dino_snerl" not in stages and "track_motion" not in stages
    assert not {"sam3_masks", "dino_snerl"} & set(cfg["stages"]) and not {"sam3", "dino_snerl"} & set(cfg["workers"])
    check_stage_scope(cfg, stages)
    for extra in ("sam3_masks", "dino_snerl"):
        with pytest.raises(ValueError, match="not needed"):
            check_stage_scope(cfg, stages + [extra])


def test_realworld_config_generates_exactly_the_declared_baselines_inputs():
    from s4d.realworld.baseline_stages import BASELINE_AUX_STAGES, required_baseline_stages
    from s4d.realworld.cli import check_stage_scope, default_stages

    cfg = _cfg("mug_hanging_rgbd")
    assert set(cfg["baselines"]) <= set(BASELINE_AUX_STAGES)
    assert required_baseline_stages(cfg) == ("sam3_masks", "dino_snerl")
    stages = default_stages(cfg)
    assert stages.index("track_motion") < stages.index("export") and {"sam3_masks", "dino_snerl"} <= set(stages)
    check_stage_scope(cfg, stages)
    no_snerl = {**cfg, "baselines": ["mvp3d"]}  # calibration still needs the SAM 3 robot masks, never DINO
    assert required_baseline_stages(no_snerl) == ("sam3_masks",)
    with pytest.raises(ValueError, match="not needed"):
        check_stage_scope(no_snerl, ["dino_snerl"])
    with pytest.raises(ValueError, match="unknown baselines"):
        required_baseline_stages({**cfg, "baselines": ["pointworld"]})


# ------------------------------------------------------------------ viewer trails (display data only)


def _camera_tracks():
    from s4d.realworld.viewer_data import CameraTracks

    frames = np.arange(16)
    xyz = np.zeros((16, 2, 3), np.float32)
    xyz[:, :, 0] = np.arange(16)[:, None] * 0.01  # both tracks move along +x
    seg = np.zeros((16, 2), np.int16)
    seg[6, 0] = -1  # track 0: invalid sample at frame 6 (hole within the segment)
    seg[9:, 0] = 1  # ... and a segment break at frame 9
    chunk = {"name": "c", "query": 0, "frames": frames, "xyz": xyz, "seg": seg}
    later = {"name": "d", "query": 8, "frames": frames + 8, "xyz": xyz + 1.0, "seg": np.zeros((16, 2), np.int16)}
    window = {int(f): 0 for f in frames}
    window.update({int(f): 1 for f in range(16, 24)})
    return CameraTracks([chunk, later], window)


def test_trails_join_only_consecutive_measured_samples_of_one_segment():
    from s4d.realworld.viewer_data import trail_segments

    ct = _camera_tracks()
    segs, age, pts, page, ids = trail_segments(ct, 12, length=8)
    # frames 4..12: track 1 has 8 links; track 0 has 4-5, 7-8 (no line across the invalid frame 6), then 9-10..11-12
    assert (ids == 1).sum() == 8 and (ids == 0).sum() == 2 + 3
    assert not any((s[0, 0] < 0.0605 and s[1, 0] > 0.0595) for s in segs[ids == 0])  # nothing touches frame 6
    t0 = segs[ids == 0]
    starts = np.round(t0[:, 0, 0] / 0.01).astype(int).tolist()
    assert 8 not in starts  # never across the segment break between frames 8 and 9
    assert len(pts) == 9 + 8 and page.min() == 0 and page.max() == 8  # measured samples returned separately
    assert age.min() == 0 and age.max() == 7


def test_trails_start_over_after_a_recording_pause_and_never_join_windows():
    from s4d.realworld.viewer_data import trail_segments

    ct = _camera_tracks()
    segs, *_ , ids = trail_segments(ct, 12, length=8, break_frames={11})
    assert len(segs) == 2 * 1  # only frames 11..12 remain after the pause
    segs, *_ = trail_segments(ct, 18, length=8)  # frame 18 belongs to the later window only
    assert (segs[..., 0] >= 1.0).all()  # positions of the later window; nothing from the first one
    assert len(trail_segments(ct, 99)[0]) == 0


def test_display_cloud_stride_must_nest():
    from s4d.realworld.viewer_data import cloud

    with pytest.raises(ValueError, match="multiple"):
        cloud(None, 0, 0, "sensor_depth", stride=5, roi_stride=3)



def test_shipped_configs_construct_the_filter_config():
    import dataclasses

    from s4d.realworld.track_motion import MotionFilterConfig

    for name in ("mug_hanging_rgbd", "mug_hanging_rgbd_rawdepth"):
        mc = dict(_cfg(name)["stages"]["track_motion"])
        cfg = MotionFilterConfig(**mc)  # unknown keys would raise TypeError (the pipeline builds it the same way)
        assert set(mc) <= {f.name for f in dataclasses.fields(MotionFilterConfig)} and cfg.min_support >= 2


def test_sparse_depth_bins_never_borrow_a_less_noisy_threshold():
    """Codex geo1 #3: near bin 1 mm noise, sparse far bin 20 mm noise -> the far bin must not inherit ~5 mm."""
    rng = np.random.default_rng(2)
    px = np.abs(rng.normal(0, 0.2, 1100))
    z = np.concatenate([np.full(1000, 0.001), np.full(100, 0.020)])
    depth = np.concatenate([rng.uniform(0.6, 0.8, 1000), rng.uniform(2.1, 2.9, 100)])
    thr = reference_thresholds(px, z, depth, CFG)
    uv = np.array([[320.0, 240.0]] * 16)
    _, xyz = _track(uv, z=2.5)
    xyz[6:, 0, 2] += 0.02  # a persistent 20 mm depth excursion of a static far point (noise there)
    seg, _ = segment_tracks(np.full((16, 1), VALID, np.uint8), xyz, np.arange(16) / 15.0, CFG, 1 / 15.0)
    cls, _ = decide(segment_stats(uv[:, None, :], xyz, seg, 600.0, CFG.min_segment_samples)[0], thr, CFG)
    assert cls == UNCERTAIN  # neither moving (no calibrated depth threshold) nor claimed static


def test_depth_axis_support_is_measured_from_the_depth_extremes():
    """Codex geo1 #4: constant pixels, depth 1.000 -> 1.010 -> 0.990 m sustained: 20 mm extent, support from the
    extremes (not from the first sample, from which every displacement is only 10 mm)."""
    uv = np.array([[320.0, 240.0]] * 15)
    _, xyz = _track(uv)
    xyz[5:10, 0, 2] = 1.010
    xyz[10:, 0, 2] = 0.990
    cls, best = _classify(uv[:, None, :], xyz)
    assert cls == MOVING and best["by"] == "z"


def test_motion_score_ignores_displacement_across_segment_breaks():
    from s4d.data.droid.pointworld import splat_score
    from s4d.realworld.export import segment_motion, splat_score_given_motion

    pts = np.zeros((3, 2, 3), np.float32)
    pts[:, :, 2] = 1.0
    pts[1:, 0, 0] = 0.06  # point 0 jumps 6 cm between t0 and t1 ...
    seg = np.array([[0, 0], [1, 0], [1, 0]], np.int16)  # ... across a segment break
    np.testing.assert_allclose(segment_motion(pts, seg), [0.0, 0.0])
    seg_same = np.zeros((3, 2), np.int16)
    np.testing.assert_allclose(segment_motion(pts, seg_same), [0.06, 0.0], atol=1e-6)
    K = np.array([[100.0, 0, 32], [0, 100.0, 32], [0, 0, 1]], np.float32)
    w2c = np.eye(4, dtype=np.float32)
    depth = np.ones((3, 64, 64), np.float32)
    vis = np.ones((3, 2), bool)
    np.testing.assert_array_equal(splat_score_given_motion(pts, vis, segment_motion(pts, seg_same), K, w2c, depth),
                                  splat_score(pts, vis, K, w2c, depth))  # same rules as the shared helper
    assert splat_score_given_motion(pts, vis, segment_motion(pts, seg), K, w2c, depth).max() == 0.0


def test_calibration_evidence_frames_include_initialisation_frames():
    from s4d.realworld.calibration import evidence_frames

    fit = [{"t_index": i} for i in range(0, 40, 2) if i % 10 != 8]
    held = [{"t_index": i} for i in range(8, 40, 10)]
    doc = {"diagnostics": {"per_frame_fit": fit, "per_frame_heldout": held, "init_method": "rgbd_robot_registration",
                           "config": {"init_frames": 3}, "frames": 20, "visible_frames": 20, "rgbd_init": {}}}
    frames, how = evidence_frames(doc)
    # linspace over the 20 calibration frames picks frames 0, 20, 38; 38 is a held-out frame, now not independent
    assert how == "reconstructed" and {0, 20, 38} <= frames and 28 not in frames
    doc["diagnostics"]["rgbd_init"]["init_t_indices"] = [28]
    frames, how = evidence_frames(doc)
    assert how == "recorded" and 28 in frames and 38 not in frames
