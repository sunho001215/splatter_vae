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
