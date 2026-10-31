"""ZED SVO (v1, as recorded by DROID r2d2 1.3) frame index and ZED factory calibration parsing. Standard library only,
so the isolated decoder worker (s4d/realworld/workers/svo_decode_worker.py) imports the same parser.

SVO layout, read from the released DROID files (validated on AUTOLab+0d4edc83+2023-10-21-19h-07m-04s, all three
cameras): a header of little-endian ``int32 version (-11), int32 width, int32 height, int32 serial, float32 fps`` plus
calibration-free fields; then frame records ``uint64 timestamp_ns, uint32 type, uint32 size`` followed by ``size``
bytes of Annex-B HEVC (one access unit of the side-by-side left|right image, ``2*width x height``); ``type`` is 3 for
key frames (VPS/SPS/PPS + IDR) and 0 for the others. Sensor blocks (IMU, magnetometer, barometer, temperature) are
interleaved between frame records and skipped. Record timestamps equal the trajectory's
``observation/timestamp/cameras/<serial>_frame_received`` in milliseconds (host clock).

ZED factory calibration (``SN<serial>.conf`` from calib.stereolabs.com, the file the ZED SDK itself downloads):
reading conventions decided against the SDK's own rectified output (the DROID MP4 left views and the DROID-3D
SDK-rectified intrinsics), see docs/realworld_droid_validation.md:

* ``[STEREO] RX, CV, RZ`` are a Rodrigues vector (x, y, z) of the rotation ``R`` that maps left-camera coordinates
  into the right camera; the right camera centre in the left frame is ``C = (Baseline, -TY, -TZ)`` (millimetres), so
  ``T_right_left = [R | -R C]``.
* ZED 2 / 2i files with ``[LEFT_DISTO]/[RIGHT_DISTO]`` use those 8 coefficients (OpenCV rational model:
  k1 k2 p1 p2 k3 k4 k5 k6) - they reproduce the SDK rectification; the 5 ``[*_CAM_HD]`` coefficients do not.
* ZED Mini files carry 4 radial coefficients (k1..k4, Kannala-Brandt / OpenCV fisheye).
"""

from __future__ import annotations

import configparser
import math
import struct
from dataclasses import dataclass
from pathlib import Path

SVO_HEADER = struct.Struct("<iiiif")
RECORD = struct.Struct("<QII")
FIRST_TIMESTAMP_OFFSET = 0x118
KEY, DELTA = 3, 0
START_CODE = b"\x00\x00\x00\x01"


class SvoFormatError(ValueError):
    pass


@dataclass(frozen=True)
class SvoRecord:
    offset: int  # first byte of the HEVC payload
    size: int
    timestamp_ns: int
    key: bool


@dataclass(frozen=True)
class SvoIndex:
    path: str
    version: int
    width: int  # one eye
    height: int
    serial: int
    fps: float
    records: tuple[SvoRecord, ...]


def index_svo(path: str | Path, max_span_ns: int = 3_600_000_000_000) -> SvoIndex:
    """Frame records of one SVO file (no decoding). Strict: a payload must start with an HEVC start code, timestamps
    must increase, the first record must be a key frame and every record must fit in the file."""
    data = Path(path).read_bytes()
    if len(data) < FIRST_TIMESTAMP_OFFSET + RECORD.size:
        raise SvoFormatError(f"{path}: too short for an SVO file")
    version, width, height, serial, fps = SVO_HEADER.unpack_from(data, 0)
    if version != -11 or width <= 0 or height <= 0:
        raise SvoFormatError(f"{path}: unsupported SVO header (version {version}, {width}x{height})")
    ts0, typ0, size0 = RECORD.unpack_from(data, FIRST_TIMESTAMP_OFFSET)
    start0 = FIRST_TIMESTAMP_OFFSET + RECORD.size
    if typ0 != KEY or data[start0:start0 + 4] != START_CODE:
        raise SvoFormatError(f"{path}: first record is not an HEVC key frame")
    records = [SvoRecord(start0, size0, ts0, True)]
    pos = start0 + size0
    while True:
        hit = data.find(START_CODE, pos)
        if hit < 0:
            break
        ts, typ, size = RECORD.unpack_from(data, hit - RECORD.size)
        if typ in (KEY, DELTA) and records[-1].timestamp_ns < ts <= ts0 + max_span_ns and 0 < size \
                and hit + size <= len(data):
            records.append(SvoRecord(hit, size, ts, typ == KEY))
            pos = hit + size
        else:
            pos = hit + 1
    return SvoIndex(str(path), version, width, height, serial, float(fps), tuple(records))


# ---------------------------------------------------------------- ZED factory calibration


@dataclass(frozen=True)
class ZedEye:
    fx: float
    fy: float
    cx: float  # OpenCV integer-centre convention
    cy: float
    distortion_model: str  # opencv_rational / kannala_brandt4 / opencv_plumb_bob
    distortion: tuple[float, ...]


@dataclass(frozen=True)
class ZedStereoCalibration:
    serial: str
    resolution: str
    width: int
    height: int
    left: ZedEye
    right: ZedEye
    T_right_left: tuple  # 4x4 nested tuple, metres
    baseline_m: float
    reading: str  # which convention produced the values (for provenance)


RESOLUTIONS = {"HD": (1280, 720), "FHD": (1920, 1080), "2K": (2208, 1242), "VGA": (672, 376)}


def parse_zed_conf(text: str) -> configparser.ConfigParser:
    cp = configparser.ConfigParser()
    cp.optionxform = str  # keys are case-sensitive (TY, CV_HD, ...)
    cp.read_string(text)
    return cp


def _rodrigues(rx: float, ry: float, rz: float) -> list[list[float]]:
    theta = math.sqrt(rx * rx + ry * ry + rz * rz)
    if theta < 1e-15:
        return [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
    kx, ky, kz = rx / theta, ry / theta, rz / theta
    c, s, v = math.cos(theta), math.sin(theta), 1 - math.cos(theta)
    return [[c + kx * kx * v, kx * ky * v - kz * s, kx * kz * v + ky * s],
            [ky * kx * v + kz * s, c + ky * ky * v, ky * kz * v - kx * s],
            [kz * kx * v - ky * s, kz * ky * v + kx * s, c + kz * kz * v]]


def zed_stereo_calibration(conf_text: str, serial: str, resolution: str = "HD") -> ZedStereoCalibration:
    cp = parse_zed_conf(conf_text)
    width, height = RESOLUTIONS[resolution]

    def eye(side: str) -> ZedEye:
        cam = cp[f"{side}_CAM_{resolution}"]
        k = (float(cam["fx"]), float(cam["fy"]), float(cam["cx"]), float(cam["cy"]))
        if cp.has_section(f"{side}_DISTO"):
            d = cp[f"{side}_DISTO"]
            return ZedEye(*k, "opencv_rational",
                          tuple(float(d[n]) for n in ("k1", "k2", "p1", "p2", "k3", "k4", "k5", "k6")))
        if "p1" not in cam and "k4" in cam:
            return ZedEye(*k, "kannala_brandt4", tuple(float(cam[n]) for n in ("k1", "k2", "k3", "k4")))
        return ZedEye(*k, "opencv_plumb_bob", tuple(float(cam[n]) for n in ("k1", "k2", "p1", "p2", "k3")))

    st = cp["STEREO"]
    B, TY, TZ = (float(st["Baseline"]) / 1000.0, float(st["TY"]) / 1000.0, float(st["TZ"]) / 1000.0)
    R = _rodrigues(float(st[f"RX_{resolution}"]), float(st[f"CV_{resolution}"]), float(st[f"RZ_{resolution}"]))
    C = (B, -TY, -TZ)
    t = [-sum(R[i][j] * C[j] for j in range(3)) for i in range(3)]
    T = tuple(tuple(R[i]) + (t[i],) for i in range(3)) + ((0.0, 0.0, 0.0, 1.0),)
    return ZedStereoCalibration(str(serial), resolution, width, height, eye("LEFT"), eye("RIGHT"), T,
                                math.sqrt(sum(x * x for x in t)),
                                "R=Rodrigues(RX,CV,RZ) left->right; C_right_in_left=(Baseline,-TY,-TZ) mm")
