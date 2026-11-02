"""Raw DROID adapter pieces: SVO frame index, ZED factory calibration reading, provided-pose calibration stage.

The conventions tested here were decided on real data (docs/realworld_droid_validation.md); these tests pin them.
"""

import json
import struct

import numpy as np
import pytest

from s4d.realworld.adapters.droid_svo import (
    FIRST_TIMESTAMP_OFFSET,
    SvoFormatError,
    index_svo,
    zed_stereo_calibration,
)

ZED2_CONF = """[LEFT_CAM_HD]
fx=532.025
fy=531.9
cx=637.95
cy=369.1295
k1=-0.0662344
k2=0.0469027
p1=0.000165288
p2=0.000136806
k3=-0.0193995

[RIGHT_CAM_HD]
fx=534.39
fy=534.375
cx=641.53
cy=367.7455
k1=-0.066068
k2=0.046036
p1=-0.000269993
p2=0.000652307
k3=-0.0187435

[LEFT_DISTO]
k1=-1.44439
k2=2.67939
k3=0.103744
k4=-1.35324
k5=2.52438
k6=0.272006
p1=0.000352861
p2=3.34753e-06

[RIGHT_DISTO]
k1=-1.46315
k2=2.46808
k3=0.197421
k4=-1.37129
k5=2.30554
k6=0.366925
p1=-0.000173296
p2=0.000803644

[STEREO]
Baseline=120.058
TY=-0.0961898
TZ=-0.141731
CV_HD=-2.91526e-05
RX_HD=0.00259768
RZ_HD=0.000764227
"""


def _svo_bytes(frames, sensor_block=b"\x07" * 37):
    """Synthetic SVO v1: header, then [ts, type, size] + Annex-B payload per frame, sensor bytes in between."""
    header = struct.pack("<iiiif", -11, 1280, 720, 22008760, 60.0)
    data = bytearray(header + b"\x00" * (FIRST_TIMESTAMP_OFFSET - len(header)))
    for k, (ts, key, payload) in enumerate(frames):
        if k:
            data += sensor_block
        body = b"\x00\x00\x00\x01" + payload
        data += struct.pack("<QII", ts, 3 if key else 0, len(body)) + body
    return bytes(data)


def test_svo_index_records_and_sensor_blocks(tmp_path):
    ts0 = 1_697_940_424_206_417_253
    frames = [(ts0, True, b"\x40\x01VPS..."), (ts0 + 66_000_000, False, b"\x02\x01P1"),
              (ts0 + 133_000_000, False, b"\x02\x01P2" * 10), (ts0 + 200_000_000, True, b"\x40\x01IDR")]
    path = tmp_path / "x.svo"
    path.write_bytes(_svo_bytes(frames))
    ix = index_svo(path)
    assert (ix.width, ix.height, ix.serial, ix.fps) == (1280, 720, 22008760, 60.0)
    assert [r.timestamp_ns for r in ix.records] == [f[0] for f in frames]
    assert [r.key for r in ix.records] == [True, False, False, True]
    data = path.read_bytes()
    for rec, (_, _, payload) in zip(ix.records, frames):
        assert data[rec.offset:rec.offset + rec.size] == b"\x00\x00\x00\x01" + payload


def test_svo_index_rejects_bad_header(tmp_path):
    path = tmp_path / "bad.svo"
    raw = bytearray(_svo_bytes([(1, True, b"\x40\x01")]))
    raw[0:4] = struct.pack("<i", -10)
    path.write_bytes(bytes(raw))
    with pytest.raises(SvoFormatError):
        index_svo(path)


def test_zed_conf_reading_convention():
    cal = zed_stereo_calibration(ZED2_CONF, "22008760")
    assert cal.left.distortion_model == "opencv_rational"  # DISTO section wins over the 5 CAM_HD coefficients
    assert cal.left.distortion == (-1.44439, 2.67939, 0.000352861, 3.34753e-06, 0.103744, -1.35324, 2.52438, 0.272006)
    T = np.asarray(cal.T_right_left)
    R, t = T[:3, :3], T[:3, 3]
    assert np.allclose(R @ R.T, np.eye(3)) and np.isclose(np.linalg.det(R), 1)
    from scipy.spatial.transform import Rotation

    assert np.allclose(Rotation.from_matrix(R).as_rotvec(), [0.00259768, -2.91526e-05, 0.000764227])
    centre = -R.T @ t  # right camera centre in the left frame = (Baseline, -TY, -TZ) mm
    assert np.allclose(centre, [0.120058, 0.0961898e-3, 0.141731e-3], atol=1e-12)
    assert np.isclose(cal.baseline_m, np.linalg.norm(t)) and t[0] < 0


def test_zed_mini_conf_is_kannala_brandt():
    conf = ZED2_CONF.split("[LEFT_DISTO]")[0].replace("p1=0.000165288\np2=0.000136806\nk3=-0.0193995",
                                                     "k3=-0.00028\nk4=-0.0027")
    conf = conf.replace("p1=-0.000269993\np2=0.000652307\nk3=-0.0187435", "k3=-0.0165918\nk4=0.00767088")
    conf += ZED2_CONF[ZED2_CONF.index("[STEREO]"):]
    cal = zed_stereo_calibration(conf, "18026681")
    assert cal.left.distortion_model == "kannala_brandt4" and len(cal.left.distortion) == 4


def test_calibrate_provided_accepts_rigid_and_rejects_missing(tmp_path):
    from s4d.realworld import io as rio
    from s4d.realworld import pipeline as P
    from s4d.realworld.adapters import get_adapter

    adapter = get_adapter("synthetic", {"asset_dir": str(tmp_path / "assets"), "frames": 2})
    ep = adapter.list_episodes()[0]
    P.ingest(adapter, ep, tmp_path)
    meta = P.load_meta(tmp_path, ep)
    cams = [g.left_stream for g in meta.stereo]
    T = np.eye(4)
    T[:3, 3] = [0.1, -0.2, 0.9]
    rec = P.calibrate_provided(tmp_path, ep, {cams[0]: {"T_cam_base": T, "source": {"kind": "test"}}})
    assert rec["summary"][cams[0]]["status"] == "accepted"
    assert np.allclose(P.accepted_calibration(tmp_path, ep, cams[0]), T)
    for cam in cams[1:]:
        assert rec["summary"][cam]["status"] == "rejected" and P.accepted_calibration(tmp_path, ep, cam) is None
    doc = json.loads((rio.stage_dir(tmp_path, ep, "calibrate") / f"{rio.safe_name(cams[0])}.json").read_text())
    assert np.allclose(doc["diagnostics"]["camera_position_base_m"], -T[:3, 3])
    bad = T.copy()
    bad[0, 0] = 2.0
    rec = P.calibrate_provided(tmp_path, ep, {cams[0]: {"T_cam_base": bad}}, force=True)
    assert rec["summary"][cams[0]]["status"] == "rejected"


def test_gripper_links_follow_gripper_subtree_for_non_panda_arm_names(tmp_path):
    """FR3-style arm link names must not be classified as gripper links (real-asset integration finding)."""
    from s4d.realworld.contract import RobotDescription
    from s4d.realworld.pipeline import RobotGeometry
    from s4d.realworld.robot import URDFModel, RobotSurface

    box = '<visual><geometry><box size="0.02 0.02 0.02"/></geometry></visual>'
    links = ["base", "fr3_link0", "fr3_link8", "world", "grip_base", "finger"]
    urdf = "<robot name='r'>" + "".join(f"<link name='{n}'>{box}</link>" for n in links)
    for parent, child, name in (("base", "fr3_link0", "j0"), ("fr3_link0", "fr3_link8", "j8"),
                                ("fr3_link8", "world", "mount"), ("world", "grip_base", "gb"),
                                ("grip_base", "finger", "f")):
        urdf += (f"<joint name='{name}' type='fixed'><parent link='{parent}'/><child link='{child}'/>"
                 "<origin xyz='0 0 0.1' rpy='0 0 0'/></joint>")
    urdf += "</robot>"
    (tmp_path / "r.urdf").write_text(urdf)
    geom = RobotGeometry.__new__(RobotGeometry)
    geom.model = URDFModel.from_file(tmp_path / "r.urdf")
    geom.surface = RobotSurface.build(geom.model, 10, seed=0)
    geom.desc = RobotDescription("franka_panda", "robotiq_2f_85", str(tmp_path / "r.urdf"), None,
                                 gripper_base_frame="world")
    assert geom.gripper_links() == ["finger", "grip_base", "world"]
    geom.desc = RobotDescription("franka_panda", "robotiq_2f_85", str(tmp_path / "r.urdf"), None)
    assert "fr3_link8" in geom.gripper_links()  # the legacy rule's misclassification, kept only without the frame


# ------------------------------------------------------------ DROID-3D joint offsets follow the calibration used

UUID = "LAB+deadbeef+2023-01-01-00h-00m-00s"
SERIAL = "FR3-295341-1000000"
DQ = [0.001, -0.002, 0.0005, -0.003, 0.02, -0.001, 0.004]


def _raw_episode(tmp_path, serial=SERIAL, T=6):
    import h5py

    from s4d.realworld.robot import panda_dh_fk

    ep = tmp_path / "raw" / UUID
    ep.mkdir(parents=True)
    (ep / f"metadata_{UUID}.json").write_text(json.dumps({"uuid": UUID, "robot_serial": serial}))
    rng = np.random.default_rng(0)
    q = np.array([0.0, -0.3, 0.0, -2.2, 0.0, 2.0, 0.8]) + 0.1 * rng.standard_normal((T, 7))
    from scipy.spatial.transform import Rotation

    cart = np.stack([np.r_[F[:3, 3], Rotation.from_matrix(F[:3, :3]).as_euler("xyz")]
                     for F in (panda_dh_fk(a)[-1] for a in q)])
    with h5py.File(ep / "trajectory.h5", "w") as f:
        f["observation/timestamp/robot_state/read_start"] = np.arange(T, dtype=np.int64) * 66
        f["observation/timestamp/robot_state/read_end"] = np.arange(T, dtype=np.int64) * 66 + 2
        f["observation/robot_state/cartesian_position"] = cart
        f["observation/robot_state/joint_positions"] = q
        f["observation/robot_state/gripper_position"] = np.zeros(T)
    return ep, q


def _x3d(tmp_path, *records):
    path = tmp_path / "extrinsics.jsonl"
    T = np.eye(4).tolist()
    path.write_text("".join(json.dumps({"episode_uuid": u, "robot_serial": s, "T0_ext1_in_world": T,
                                        "T1_ext2_in_world": T, "dq_joint_correction": dq,
                                        "dT_gw_gripper_in_world_correction": T, "_meta": {}}) + "\n"
                            for u, s, dq in records))
    return path


def _droid_adapter(tmp_path, ep, x3d, calibration_source):
    from s4d.realworld.adapters import get_adapter

    cfg = {"episodes": {UUID: str(ep)}, "calibration_dir": str(tmp_path), "cameras": [],
           "cache_dir": str(tmp_path / "cache"), "robot": {"flange_frame": "panda_link8"}}
    if x3d is not None:
        cfg["droid3d_extrinsics"] = str(x3d)
    if calibration_source is not None:
        cfg["calibration_source"] = calibration_source
    return get_adapter("droid_raw_svo", cfg)


def test_droid3d_dq_applied_only_with_the_matching_droid3d_calibration(tmp_path):
    from s4d.realworld.robot import panda_dh_fk

    ep, q = _raw_episode(tmp_path)
    x3d = _x3d(tmp_path, ("OTHER+00000000+2023-01-01-00h-00m-00s", SERIAL, [0.5] * 7), (UUID, SERIAL, DQ))
    meta = _droid_adapter(tmp_path, ep, x3d, "adapter_provided").read_meta(UUID)
    tr = meta.trajectory
    np.testing.assert_allclose(tr.joint_positions, q + np.asarray(DQ), atol=1e-12)
    np.testing.assert_allclose(tr.extra["joint_positions_raw"], q, atol=0)
    for k in range(len(q)):  # recorded EE moved like the joints: still equals FK of the stored joints
        np.testing.assert_allclose(tr.ee_pose_base[k], panda_dh_fk(tr.joint_positions[k])[-1], atol=1e-9)
    jc = json.loads(meta.provenance.notes)["joint_correction"]
    assert jc["applied"] and jc["dq"] == DQ
    assert jc["requires_calibration"] == {"kind": "droid3d_extrinsics", "path": str(x3d), "episode_uuid": UUID,
                                          "robot_serial": SERIAL}
    provider = _droid_adapter(tmp_path, ep, x3d, "adapter_provided")
    provider.cameras = ("ext1",)  # provided poses (no SVO access needed) name exactly the record the joints use
    src = provider.provided_extrinsics(UUID)["ext1/left"]["source"]
    assert all(src[k] == v for k, v in jc["requires_calibration"].items())


@pytest.mark.parametrize("case", ["missing", "other_robot", "unknown_serial", "empty_serial", "bad_dq", "nan_dq"])
def test_droid3d_dq_never_borrowed_or_invalid(tmp_path, case):
    serial = {"unknown_serial": "unknown", "empty_serial": "fr3-"}.get(case, SERIAL)
    ep, q = _raw_episode(tmp_path, serial=serial)
    rec = {"missing": ("OTHER+00000000+2023-01-01-00h-00m-00s", serial, DQ),
           "other_robot": (UUID, "FR3-295341-1999999", DQ),
           "bad_dq": (UUID, serial, DQ[:6]), "nan_dq": (UUID, serial, DQ[:6] + [float("nan")])}.get(case, (UUID, serial, DQ))
    meta = _droid_adapter(tmp_path, ep, _x3d(tmp_path, rec), "adapter_provided").read_meta(UUID)
    np.testing.assert_array_equal(meta.trajectory.joint_positions, q)
    assert "joint_positions_raw" not in meta.trajectory.extra
    jc = json.loads(meta.provenance.notes)["joint_correction"]
    assert not jc["applied"] and "raw joints" in jc["reason"]


@pytest.mark.parametrize("source", [None, "landmarks"])
def test_droid3d_dq_not_applied_when_calibration_comes_from_elsewhere(tmp_path, source):
    ep, q = _raw_episode(tmp_path)
    adapter = _droid_adapter(tmp_path, ep, _x3d(tmp_path, (UUID, SERIAL, DQ)), source)
    meta = adapter.read_meta(UUID)
    np.testing.assert_array_equal(meta.trajectory.joint_positions, q)
    assert not json.loads(meta.provenance.notes)["joint_correction"]["applied"]


def test_cli_ties_adapter_calibration_source_to_the_calibrate_stage():
    from s4d.realworld.cli import adapter_config

    cfg = {"adapter": {"config": {"episodes": {}}}, "stages": {"calibrate": {"source": "adapter_provided"}}}
    assert adapter_config(cfg)["calibration_source"] == "adapter_provided"
    assert "calibration_source" not in adapter_config({"adapter": {"config": {}}})
    cfg["adapter"]["config"]["calibration_source"] = "landmarks"
    with pytest.raises(ValueError):
        adapter_config(cfg)


def test_calibrate_provided_refuses_poses_from_another_record_than_the_joint_correction(tmp_path):
    from s4d.realworld import io as rio
    from s4d.realworld import pipeline as P
    from s4d.realworld.adapters import get_adapter

    adapter = get_adapter("synthetic", {"asset_dir": str(tmp_path / "assets"), "frames": 2})
    ep = adapter.list_episodes()[0]
    P.ingest(adapter, ep, tmp_path)
    path = rio.episode_dir(tmp_path, ep) / "normalized" / "episode.json"
    doc = json.loads(path.read_text())
    need = {"kind": "droid3d_extrinsics", "path": "/x.jsonl", "episode_uuid": ep, "robot_serial": SERIAL}
    doc["provenance"]["notes"] = json.dumps({"joint_correction": {"applied": True, "kind": "droid3d_dq_joint_correction",
                                                                  "dq": DQ, "requires_calibration": need}})
    path.write_text(json.dumps(doc))
    meta = P.load_meta(tmp_path, ep)
    cam = meta.stereo[0].left_stream
    T = np.eye(4)
    T[:3, 3] = [0.1, -0.2, 0.9]
    with pytest.raises(ValueError, match="not the record"):
        P.calibrate_provided(tmp_path, ep, {cam: {"T_cam_base": T, "source": {**need, "robot_serial": "other"}}})
    with pytest.raises(ValueError, match="not the record"):
        P.calibrate_provided(tmp_path, ep, {cam: {"T_cam_base": T, "source": {"kind": "test"}}})
    rec = P.calibrate_provided(tmp_path, ep, {cam: {"T_cam_base": T, "source": dict(need)}})
    assert rec["summary"][cam]["status"] == "accepted"
    cal = json.loads((rio.stage_dir(tmp_path, ep, "calibrate") / f"{rio.safe_name(cam)}.json").read_text())
    assert cal["diagnostics"]["robot_joint_correction_dq"] == DQ
    with pytest.raises(ValueError, match="raw joints"):
        P.calibrate(tmp_path, ep, lambda *a: None, landmark_links=[])
