"""Board-path shared calibration on synthetic data: ChArUco corner geometry, planar PnP, multi-camera bundle adjustment,
joint robot registration of the rigid camera set, and the validation gates (no OpenCV needed: detections are
synthesised; ``detect_charuco`` / ``detect_board`` need OpenCV with aruco and are tested in the worker-env suite
tests_worker_env/test_realworld_calibration_board_cv2.py, run in the s4d-rwprep image)."""

import numpy as np
import pytest
from scipy.spatial.transform import Rotation
from test_realworld_calibration import HW, K, make_episode

from s4d.realworld.calib.board import BoardRigConfig, BoardSpec, build_board_rig, planar_pnp, solve_board
from s4d.realworld.calib.core import CalibrationConfig
from s4d.realworld.geometry import invert_T, project, rotation_angle_deg, transform_points
from s4d.realworld.synthetic import look_at_T_cam_base

SPEC = BoardSpec(squares_x=8, squares_y=6, square_m=0.05, marker_m=0.037)
CAMS = {"cam_a": look_at_T_cam_base(eye=[1.3, 0.7, 0.8], target=[0.35, 0.0, 0.45]),
        "cam_b": look_at_T_cam_base(eye=[1.4, 0.0, 0.85], target=[0.35, 0.0, 0.45]),
        "cam_c": look_at_T_cam_base(eye=[1.3, -0.7, 0.8], target=[0.35, 0.0, 0.45])}
KS = {c: K for c in CAMS}


def board_poses():
    """T_base_board of four placements in the shared workspace, tilted differently (facing the cameras)."""
    out = {}
    for k, (pos, tilt) in enumerate([((0.55, -0.15, 0.30), (0, 0)), ((0.6, 0.05, 0.42), (20, 10)),
                                     ((0.5, -0.05, 0.25), (-15, 20)), ((0.65, 0.0, 0.35), (10, -25))]):
        # board z axis towards the cameras (+x of the base), x along base y
        R0 = np.array([[0, 0, 1], [-1, 0, 0], [0, -1, 0]], float)  # columns: board x, y, z in the base frame
        R = Rotation.from_euler("yz", tilt, degrees=True).as_matrix() @ R0
        T = np.eye(4)
        T[:3, :3], T[:3, 3] = R, pos
        out[f"p{k}"] = T
    return out


def detections(noise_px=0.1, seed=0, placements=None):
    rng = np.random.default_rng(seed)
    obj = SPEC.object_points()
    out = {}
    for c, T in CAMS.items():
        for p, Tb in board_poses().items():
            if placements is not None and p not in placements:
                continue
            uv, z = project(transform_points(T @ Tb, obj), K)
            ok = np.isfinite(uv).all(1) & (z > 0) & (uv[:, 0] > 2) & (uv[:, 0] < HW[1] - 2) & (uv[:, 1] > 2) & \
                (uv[:, 1] < HW[0] - 2)
            ids = np.flatnonzero(ok)
            out.setdefault(c, {})[p] = (ids, uv[ok] + rng.normal(0, noise_px, (len(ids), 2)))
    return out


def test_board_geometry_planar_pnp_and_bundle_adjustment():
    obj = SPEC.object_points()
    assert obj.shape == ((SPEC.squares_x - 1) * (SPEC.squares_y - 1), 3) and np.all(obj[:, 2] == 0)
    det = detections()
    for c, T in CAMS.items():
        assert all(len(det[c][p][0]) >= 12 for p in det[c]), {p: len(det[c][p][0]) for p in det[c]}
        p = "p1"
        Tp = planar_pnp(obj[det[c][p][0]], det[c][p][1], K)
        assert rotation_angle_deg(Tp[:3, :3], (T @ board_poses()[p])[:3, :3]) < 0.3
    sol = solve_board(det, KS, SPEC, BoardRigConfig(), "cam_b")
    for c in CAMS:
        truth = CAMS[c] @ invert_T(CAMS["cam_b"])
        assert rotation_angle_deg(sol["T_cam_ref"][c][:3, :3], truth[:3, :3]) < 0.08
        assert np.linalg.norm(sol["T_cam_ref"][c][:3, 3] - truth[:3, 3]) < 0.002
        assert sol["reproj_rms_px"][c] < 0.4


def _robot(T, seed, n=8):
    frames = make_episode(T, n=n, outlier_rate=0.0, depth_noise=0.002, seed=seed)
    for f in frames:
        f.landmarks_base, f.landmarks_uv, f.landmarks_conf = np.empty((0, 3)), np.empty((0, 2)), np.empty(0)
        f.robot_mask = (f.depth > 0) & (f.depth < 2.9)
    return frames


@pytest.fixture(scope="module")
def robot():
    fit = {c: _robot(T, 0) for c, T in CAMS.items()}
    rep = {"half0": {c: f[0::2] for c, f in fit.items()}, "half1": {c: f[1::2] for c, f in fit.items()}}
    grp = {"group0": {c: f[:4] for c, f in fit.items()}, "group1": {c: f[4:] for c, f in fit.items()}}
    held = {"held_ep": {c: _robot(T, 5, n=6) for c, T in CAMS.items()}}
    return fit, rep, grp, held


def test_board_rig_validates_the_true_rig_and_holds_on_failed_gates(robot):
    fit, rep, grp, held = robot
    cfg = CalibrationConfig(evidence="rgbd", image_hw=HW, max_heldout_depth_median_m=0.012, min_explained_fraction=0.7)
    bc = BoardRigConfig(registration_starts=3)
    D = np.eye(4)
    D[:3, :3] = Rotation.from_rotvec([0.02, -0.015, 0.01]).as_matrix()
    D[:3, 3] = [0.02, -0.01, 0.015]
    T0 = D @ CAMS["cam_b"]  # robot registration starts ~1.5 deg / 2.7 cm off
    fixed = {c: {"fixed": True} for c in CAMS}
    epm = {"held_ep": dict(fixed), "pool_ep": dict(fixed)}
    kw = dict(reference="cam_b", board_cfg=bc, replicate_frames=rep, group_frames=grp, heldout_frames=held,
              capture_motion=fixed, episode_motion=epm)
    doc = build_board_rig(detections(), KS, SPEC, fit, cfg, T0, **kw)
    for c, T in CAMS.items():
        cam = doc["cameras"][c]
        assert cam["status"] == "validated", cam["reasons"]
        Tc = np.asarray(cam["T_cam_base"])
        assert rotation_angle_deg(Tc[:3, :3], T[:3, :3]) < 0.35
        assert np.linalg.norm(invert_T(Tc)[:3, 3] - invert_T(T)[:3, 3]) < 0.008
        assert cam["empirical_spread"]["solves"] >= 5
    # the capture shows a moved camera -> that camera is held
    moved = build_board_rig(detections(), KS, SPEC, fit, cfg, T0,
                            **{**kw, "capture_motion": {**fixed, "cam_c": {"fixed": False, "reason": "shift 3 px"}}})
    assert moved["cameras"]["cam_c"]["status"] == "held" and moved["cameras"]["cam_a"]["status"] == "validated"
    # a robot-evidence episode whose camera moved, or unmeasured fixedness -> held
    bumped = build_board_rig(detections(), KS, SPEC, fit, cfg, T0,
                             **{**kw, "episode_motion": {**epm, "pool_ep": {**fixed, "cam_a": {"fixed": False}}}})
    assert bumped["cameras"]["cam_a"]["status"] == "held" and bumped["cameras"]["cam_b"]["status"] == "validated"
    unmeasured = build_board_rig(detections(), KS, SPEC, fit, cfg, T0, **{**kw, "episode_motion": None})
    assert all(v["status"] == "held" for v in unmeasured["cameras"].values())
    # empty replicate / group collections never count as evidence
    empty = build_board_rig(detections(), KS, SPEC, fit, cfg, T0, **{**kw, "replicate_frames": {}, "group_frames": {}})
    assert all(v["status"] == "held" for v in empty["cameras"].values())
    # too few placements / poor corner detections / no replicates -> held
    few_det = detections(placements={"p0", "p1"})
    for c in few_det:  # a third "placement" without usable corners does not count
        few_det[c]["p9"] = (np.zeros(0, int), np.zeros((0, 2)))
    few = build_board_rig(few_det, KS, SPEC, fit, cfg, T0, **kw)
    assert all(v["status"] == "held" and any("placements" in r for r in v["reasons"]) for v in few["cameras"].values())
    noisy = build_board_rig(detections(noise_px=3.0), KS, SPEC, fit, cfg, T0, **kw)
    assert all(any("reprojection RMS" in r for r in v["reasons"]) for v in noisy["cameras"].values())
    norep = build_board_rig(detections(), KS, SPEC, fit, cfg, T0, **{**kw, "replicate_frames": None,
                                                                         "group_frames": None})
    assert all(v["status"] == "held" for v in norep["cameras"].values())


def test_zivid_cb01_geometry_and_configuration():
    spec = BoardSpec.zivid_cb01()
    assert spec.geometry == "zivid_cb01" and spec.dictionary == "DICT_4X4_50" and spec.marker_id == 1
    obj = spec.object_points()
    assert obj.shape == (42, 3)
    assert obj[0] == pytest.approx([0.03, 0.03, 0])
    assert obj[-1] == pytest.approx([0.18, 0.21, 0])
    assert np.diff(obj.reshape(7, 6, 3), axis=1)[..., 0] == pytest.approx(np.full((7, 5), 0.03))
    assert BoardSpec().geometry == "charuco"
    with pytest.raises(ValueError, match="unknown board geometry"):
        BoardSpec(geometry="unknown")
    with pytest.raises(ValueError, match="7 x 8"):
        BoardSpec(geometry="zivid_cb01")
