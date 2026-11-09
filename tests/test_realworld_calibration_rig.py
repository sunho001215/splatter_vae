"""Empirical calibration uncertainty, camera-fixedness test and the shared (rig) calibration mode (study 2026-11-06).

Synthetic fixtures only: they validate the bookkeeping, the gates and the image-registration test, not real accuracy."""

import json

import numpy as np
import pytest
from scipy.ndimage import gaussian_filter, shift
from scipy.spatial.transform import Rotation
from test_realworld_calibration import HW, T_TRUE, K, make_episode

from s4d.realworld import io as rio
from s4d.realworld.calib.core import CalibrationConfig, calibrate_camera, near_best_spread, pose_distance
from s4d.realworld.calib.rig import (
    FORMAT,
    FixednessConfig,
    build_rig_calibration,
    episode_motion,
    image_motion,
    load_rig,
    publish_rig,
    replicate_spread,
    static_image,
    validate_episode,
)


def rotated(T, deg, axis=(0, 1, 0), trans=(0, 0, 0)):
    D = np.eye(4)
    D[:3, :3] = Rotation.from_rotvec(np.radians(deg) * np.asarray(axis, float)).as_matrix()
    D[:3, 3] = trans
    return D @ T


def test_pose_distance_and_near_best_spread():
    T2 = rotated(T_TRUE, 1.0, trans=(0.01, 0, 0))
    rot, centre = pose_distance(T_TRUE, T2)
    assert rot == pytest.approx(1.0, abs=1e-9)
    assert centre > 0.0
    s = near_best_spread([(0.0050, T_TRUE), (0.0052, T2), (0.0300, rotated(T_TRUE, 9.0)), (float("inf"), T_TRUE)],
                         5e-4)
    assert s["near_best_starts"] == 2 and s["starts"] == 4
    assert s["max_pair_rot_deg"] == pytest.approx(rot) and s["max_pair_centre_m"] == pytest.approx(centre)
    assert near_best_spread([(float("inf"), T_TRUE)], 5e-4)["near_best_starts"] == 0
    rep = replicate_spread(T_TRUE, [T2, T_TRUE])
    assert rep["solves"] == 3 and rep["max_pair_rot_deg"] == pytest.approx(rot)


def test_multistart_reports_empirical_spread_and_gates_on_it():
    from s4d.realworld.calib import core as cal

    frames = make_episode(T_TRUE, outlier_rate=0.0)
    prior = rotated(T_TRUE, 1.5, axis=(1, 0, 0), trans=(0.02, -0.01, 0.02))
    orig = cal._visual_init
    try:
        cal._visual_init = lambda frames, K, cfg: (None, None, None, None)
        res = calibrate_camera("rig0/left", frames, K, CalibrationConfig(image_hw=HW, refine_starts=5), T_init=prior)
        gated = calibrate_camera("rig0/left", frames, K,
                                 CalibrationConfig(image_hw=HW, refine_starts=5, near_best_tol_m=1.0,
                                                   max_near_best_rot_deg=1e-9, max_near_best_centre_m=1e-9),
                                 T_init=prior)
    finally:
        cal._visual_init = orig
    es = res.diagnostics["empirical_spread"]
    assert es["starts"] == 5 and es["near_best_starts"] >= 1
    starts = res.diagnostics["multistart"]["starts"]
    chosen = next(s for s in starts if s["start"] == res.diagnostics["multistart"]["chosen"])
    assert chosen["to_chosen_rot_deg"] == pytest.approx(0.0, abs=1e-6) and len(chosen["T_cam_base"]) == 4
    assert not any("ambiguous" in r for r in res.reasons)  # report-only by default
    # every start within the (huge) tolerance and a zero threshold -> rejected as ambiguous
    assert gated.status == "rejected" and any(r.startswith("ambiguous extrinsic") for r in gated.reasons)


def _texture(seed=0, hw=(240, 320)):
    """Textured static scene (a median image of real frames is nearly noise-free and textured at the pixel scale)."""
    tex = gaussian_filter(np.random.default_rng(seed).normal(size=hw), 1.0)
    return (tex / tex.std() * 40 + 128).astype(np.float32)


def test_image_motion_detects_subpixel_camera_motion_and_accepts_a_fixed_camera():
    ref = _texture()
    dyn = np.zeros(ref.shape, bool)
    dyn[100:160, 120:220] = True  # moving region (robot): excluded from the registration
    Kc = np.array([[300.0, 0, 160], [0, 300.0, 120], [0, 0, 1]])
    cfg = FixednessConfig()
    noisy = ref + np.random.default_rng(1).normal(0, 0.3, ref.shape)
    still = image_motion(ref, dyn, noisy.astype(np.float32), dyn, Kc, cfg)
    assert still["fixed"], still
    assert still["shift_px"] < 0.05 and abs(still["rot_deg"]) < 0.01
    moved = shift(ref, (-0.4, 0.6), order=3, mode="nearest")  # the scene moves +0.6 px in x, -0.4 px in y
    m = image_motion(ref, dyn, moved.astype(np.float32), dyn, Kc, cfg)
    assert not m["fixed"] and "image shift" in m["reason"]
    assert m["shift_xy_px"] == pytest.approx([0.6, -0.4], abs=0.12)
    # a moving region alone (robot elsewhere in this episode) does not count as camera motion
    other = ref.copy()
    other[100:160, 120:220] = _texture(5)[100:160, 120:220]
    assert image_motion(ref, dyn, other, dyn, Kc, cfg)["fixed"]
    # no static texture -> not verified (never silently accepted)
    flat = np.full(ref.shape, 128, np.float32)
    assert not image_motion(flat, dyn, flat, dyn, Kc, cfg)["fixed"]
    # smooth texture shifted by exactly 2 px (review finding: phase correlation reported 0.14 px there)
    smooth = gaussian_filter(np.random.default_rng(3).normal(size=ref.shape), 5.0)
    smooth = (smooth / smooth.std() * 40 + 128).astype(np.float32)
    m2 = image_motion(smooth, dyn, np.roll(smooth, 2, axis=1), dyn, Kc, cfg)
    assert not m2["fixed"] and m2.get("shift_xy_px", [0])[0] == pytest.approx(2.0, abs=0.1)
    # motion beyond the search radius is never read as "no motion"
    far = image_motion(ref, dyn, np.roll(ref, 15, axis=1), dyn, Kc, cfg)
    assert not far["fixed"]


def test_episode_motion_catches_a_camera_moved_for_part_of_the_episode():
    ref = _texture()
    dyn = np.zeros((40, *ref.shape), bool)
    Kc = np.array([[300.0, 0, 160], [0, 300.0, 120], [0, 0, 1]])
    frames = np.repeat(ref[None, :, :, None], 40, 0).repeat(3, -1)
    frames[24:] = np.roll(frames[24:], 4, axis=2)  # 16 of 40 frames shifted 4 px: the overall median is unchanged
    m = episode_motion(ref, dyn[0], frames, dyn, Kc, FixednessConfig())
    assert not m["fixed"] and "segment 3" in m["reason"]
    assert episode_motion(ref, dyn[0], np.repeat(ref[None, :, :, None], 40, 0).repeat(3, -1), dyn, Kc,
                          FixednessConfig())["fixed"]


def test_static_image_is_the_median_and_dilates_the_dynamic_mask():
    rgb = np.stack([np.full((20, 30, 3), v, np.uint8) for v in (10, 50, 30)])
    dyn = np.zeros((3, 20, 30), bool)
    dyn[1, 10, 15] = True
    grey, d = static_image(rgb, dyn, FixednessConfig(dynamic_dilate_px=2))
    assert np.all(grey == 30) and d[10, 15] and d[12, 15] and not d[13, 15]


def _rgbd_frames():
    frames = make_episode(T_TRUE, n=12, outlier_rate=0.0, depth_noise=0.002)
    for f in frames:
        f.landmarks_base, f.landmarks_uv, f.landmarks_conf = np.empty((0, 3)), np.empty((0, 2)), np.empty(0)
        f.robot_mask = (f.depth > 0) & (f.depth < 2.9)
    return frames


def test_validate_episode_accepts_the_true_shared_pose_and_rejects_moved_or_wrong_or_held():
    frames = _rgbd_frames()
    cfg = CalibrationConfig(evidence="rgbd", image_hw=HW, max_heldout_depth_median_m=0.012,
                            min_explained_fraction=0.7)
    rig_cam = {"T_cam_base": T_TRUE.tolist(), "status": "validated", "provenance": {"rig": "test"},
               "empirical_spread": {"max_pair_rot_deg": 0.1, "max_pair_centre_m": 0.002}}
    ok = validate_episode("rig/color", frames, K, cfg, rig_cam, {"fixed": True})
    assert ok.status == "accepted", ok.reasons
    assert ok.init_method == "rig_shared" and np.array_equal(ok.T_cam_base, T_TRUE)
    assert ok.diagnostics["std_rot_deg"] == 0.1 and ok.diagnostics["heldout_depth_abs_median_m"] < 0.012
    moved = validate_episode("rig/color", frames, K, cfg, rig_cam, {"fixed": False, "reason": "image shift 2 px"})
    assert moved.status == "rejected" and any("not verified fixed" in r for r in moved.reasons)
    wrong = {**rig_cam, "T_cam_base": rotated(T_TRUE, 4.0, trans=(0.05, 0, 0)).tolist()}
    bad = validate_episode("rig/color", frames, K, cfg, wrong, {"fixed": True})
    assert bad.status == "rejected" and any("robot-depth residual" in r or "explained" in r for r in bad.reasons)
    held = validate_episode("rig/color", frames, K, cfg, {**rig_cam, "status": "held"}, {"fixed": True})
    assert held.status == "rejected" and any("not validated" in r for r in held.reasons)
    # frames that entered the rig solve are fit frames, never held-out evidence
    pooled = [f.t_index for f in frames[:8]]
    part = validate_episode("rig/color", frames, K, cfg, rig_cam, {"fixed": True}, pooled_frames=pooled)
    assert part.diagnostics["episode_in_rig_pool"] and len(part.diagnostics["per_frame_fit"]) == 8
    assert {r["t_index"] for r in part.diagnostics["per_frame_heldout"]}.isdisjoint(pooled)
    nopose = validate_episode("rig/color", frames, K, cfg, {"status": "held", "T_cam_base": None}, {"fixed": True})
    assert nopose.status == "rejected" and nopose.T_cam_base is None and "no pose" in nopose.reasons[0]
    allp = validate_episode("rig/color", frames, K, cfg, rig_cam, {"fixed": True},
                            pooled_frames=[f.t_index for f in frames])
    assert allp.status == "rejected" and allp.diagnostics["heldout_depth_abs_median_m"] is None


def test_rig_publication_is_coherent_and_builder_refuses_unvalidatable_inputs(tmp_path):
    cam = "rig0/left"
    ref = {f"{cam.replace('/', '__')}__grey": np.zeros((4, 4), np.float32),
           f"{cam.replace('/', '__')}__dynamic": np.zeros((4, 4), bool)}
    path = tmp_path / "rig.json"
    doc = publish_rig(path, {"format": FORMAT, "name": "x", "cameras": {cam: {"status": "held"}}}, ref)
    got, arrays, fp = load_rig(path)
    assert got == doc and fp["rig_reference"] == doc["reference_sha256"] and set(arrays) == {cam}
    with pytest.raises(ValueError):  # non-finite content is refused before either published file is replaced
        publish_rig(path, {"format": FORMAT, "name": "x", "cameras": {cam: {"spread": float("inf")}}}, ref)
    assert load_rig(path)[0] == doc
    np.savez_compressed(tmp_path / "other.npz", **{k: v + 1 for k, v in ref.items()})
    (tmp_path / "other.npz").replace(path.with_suffix(".reference.npz"))  # reference replaced behind the document
    with pytest.raises(ValueError, match="do not match"):
        load_rig(path)
    with pytest.raises(ValueError, match="unique"):
        build_rig_calibration([(tmp_path, "a"), (tmp_path / "b", "a")], tmp_path / "r.json", name="r",
                              cfg=CalibrationConfig())
    from s4d.realworld.calib.rig import RigConfig

    with pytest.raises(ValueError, match="held-out"):
        build_rig_calibration([(tmp_path, "a"), (tmp_path, "b")], tmp_path / "r.json", name="r",
                              cfg=CalibrationConfig(), rig_cfg=RigConfig(holdout_every=1))


def _write_rig(path, cams, T_of, ref, status="validated"):
    doc = {"format": FORMAT, "name": "test-rig", "pool_frames": {},
           "cameras": {c: {"T_cam_base": T_of(c).tolist(), "status": status, "provenance": {"rig": "test-rig"},
                           "empirical_spread": {"max_pair_rot_deg": 0.1, "max_pair_centre_m": 0.002}} for c in cams}}
    return publish_rig(path, doc, {f"{c.replace('/', '__')}__{k}": v for c in cams for k, v in ref[c].items()})


def test_calibrate_stage_rig_mode_verifies_fixedness_and_never_solves(tmp_path):
    """Stage plumbing on the synthetic RGB-D episode: the rig pose is used only for unmoved cameras, the rig files enter
    the fingerprint, and a moved camera is rejected (its geometry consumers then skip it)."""
    import h5py
    from test_realworld_pipeline import CALIBRATION, synthetic_rgbd_episode

    from s4d.realworld.pipeline import accepted_calibration, calibrate, camera_views, load_meta, read_frames

    r = synthetic_rgbd_episode(tmp_path)
    root, ep, adapter = r["root"], r["ep"], r["adapter"]
    meta = load_meta(root, ep)
    cams = [v.stream_id for v in camera_views(meta)]

    def build_masks(tmp):
        for cam in cams:
            n = len(meta.stream_timestamps_ns[cam])
            hw = (meta.stream(cam).intrinsics.height, meta.stream(cam).intrinsics.width)
            sem = np.zeros((n, *hw), np.uint8)
            sem[:, 30:60, 50:90] = 1
            with h5py.File(tmp / f"{rio.safe_name(cam)}.h5", "w") as f:
                f.create_dataset("semantic", data=sem)
                f.attrs["class_names"] = json.dumps(["background", "robot", "object", "fixture"])
        return {}

    rio.run_stage(rio.stage_dir(root, ep, "masks_test"), stage="masks_test", version="1", config={}, inputs={},
                  build=build_masks)
    fix = {"patch": 24, "stride": 12, "min_inlier_patches": 4, "max_dynamic_fraction": 0.5, "min_gradient": 1.0,
           "search_px": 4, "segments": 2}
    fcfg = FixednessConfig(**fix)

    def reference(dx=0):
        ref = {}
        for cam in cams:
            n = len(meta.stream_timestamps_ns[cam])
            pick = np.linspace(0, n - 1, min(fcfg.frames, n)).round().astype(int)
            with h5py.File(rio.stage_dir(root, ep, "masks_test") / f"{rio.safe_name(cam)}.h5") as f:
                sem = f["semantic"][()]
            grey, dyn = static_image(read_frames(root, ep, cam, pick), sem == 1, fcfg)
            ref[cam] = {"grey": np.roll(grey, dx, axis=1) if dx else grey, "dynamic": dyn}
        return ref

    def T_of(cam):
        return adapter.T_cam_base(cam.split("/")[0], "left")

    rig = tmp_path / "rig" / "rig.json"
    rig.parent.mkdir()
    _write_rig(rig, cams, T_of, reference())
    doc, ref, fp = load_rig(rig)
    assert set(ref) == set(cams) and doc["name"] == "test-rig"
    cfg = CalibrationConfig(**{**CALIBRATION.__dict__, "evidence": "rgbd", "max_heldout_depth_median_m": 10.0,
                               "min_explained_fraction": 0.0, "min_visible_fraction": 0.0})
    rec = calibrate(root, ep, None, landmark_links=[], cfg=cfg, robot_mask_stage="masks_test",
                    rig_calibration=str(rig), fixedness=fix, force=True)
    assert rec["inputs"]["rig_calibration"] == fp["rig_calibration"]
    assert rec["inputs"]["rig_reference"] == doc["reference_sha256"]
    assert rec["config"]["rig_calibration"]["name"] == "test-rig"
    accepted = []
    for cam in cams:
        out = json.loads((rio.stage_dir(root, ep, "calibrate") / f"{rio.safe_name(cam)}.json").read_text())
        assert out["init_method"] == "rig_shared" and out["diagnostics"]["fixedness"]["fixed"], out["reasons"]
        assert np.array_equal(np.asarray(out["T_cam_base"]), T_of(cam))
        assert not any("not verified fixed" in x for x in out["reasons"])
        # relaxed evidence gates (tiny synthetic views): accepted iff this episode's robot evidence exists for the view
        if out["status"] == "accepted":
            accepted.append(cam)
            assert np.array_equal(accepted_calibration(root, ep, cam), T_of(cam))
        else:
            assert all("visibility" in x or "robot-depth" in x or "explained" in x for x in out["reasons"])
            assert accepted_calibration(root, ep, cam) is None
        assert "multistart" not in out["diagnostics"]  # no per-episode solve in rig mode
    assert accepted
    assert calibrate(root, ep, None, landmark_links=[], cfg=cfg, robot_mask_stage="masks_test",
                     rig_calibration=str(rig), fixedness=fix)["skipped"]
    # the reference no longer matches (camera moved by 2 px): new fingerprint, every camera rejected
    _write_rig(rig, cams, T_of, reference(dx=2))
    rec2 = calibrate(root, ep, None, landmark_links=[], cfg=cfg, robot_mask_stage="masks_test",
                     rig_calibration=str(rig), fixedness=fix)
    assert not rec2["skipped"]
    for cam in cams:
        out = json.loads((rio.stage_dir(root, ep, "calibrate") / f"{rio.safe_name(cam)}.json").read_text())
        assert out["status"] == "rejected" and any("not verified fixed" in x for x in out["reasons"])
        assert accepted_calibration(root, ep, cam) is None
    with pytest.raises(ValueError, match="robot_mask_stage"):
        calibrate(root, ep, None, landmark_links=[], cfg=cfg, rig_calibration=str(rig))
