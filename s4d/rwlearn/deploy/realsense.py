"""Native RealSense depth helpers (client side). No camera startup, robot I/O or torch import.

Deploy contract for 3D-MVP depth (agent DEP, /home/ws/ws/realworld-ops/serving/DEPTH-COMPAT.md, verified with real
librealsense 2.59.1): rs.align(color) -> depth-camera z16 on the colour grid -> 0.1-6 m gate -> exact colour-camera z
-> x the accepted rig document's depth_scale["sensor_depth"] -> the training centre crop + nearest resampling, K
through the crop affine (continuous convention). Live framesets are supplied by the caller; recorded replay uses the
validated numpy rs.align emulation (``emulate_rs_align``) or, alternatively, the release ``sensor_depth`` stage kernel
(``align_recorded_depth``, already colour-camera z).
"""
from __future__ import annotations

import numpy as np

from s4d.realworld.contract import Intrinsics
from s4d.realworld.frame_preprocessing import IMAGE_SIZE
from s4d.rwlearn.deploy.contracts import NATIVE_DEPTH, SENSOR_DEPTH_RANGE_M, validate_native_correction
from s4d.rwlearn.deploy.preprocessing import preprocess_frame


def load_rig_session(path, camera_ids, K_native: dict) -> dict:
    """Load a published validated fixed-rig calibration ONCE (server or client, at session start): per camera the
    OpenCV ``c2w = inv(T_cam_base)`` (exactly the training loaders' ``np.linalg.inv(store.w2c)``; the release's
    accepted per-episode calibrate records equal the document) with ``K_native`` (the colour stream's continuous
    intrinsics), and the document's native ``depth_scale['sensor_depth']`` - the same calibration-release rule
    (``pipeline.DEPTH_SCALE_RULE``) that scaled the FoundationStereo training depth with
    ``depth_scale['foundation_stereo_depth']``. Reuses the rig reader's document/reference hash check; held cameras
    and documents without a sensor_depth scale are refused (effective status ``calib.rig.rig_camera_status``: a camera
    held only by G5a counts as validated under the document's valid fixedness exception, as in the release). Returns
    ``reset_session`` kwargs (calibration, native_depth_correction). The caller chooses the task's released document
    and checks the rig has not moved."""
    from s4d.realworld.calib.rig import load_rig, rig_camera_status

    doc, _, identity = load_rig(path)
    calibration, corrections = {}, {}
    for cam in camera_ids:
        record = doc["cameras"][f"{cam}/color"]
        status, reasons, exception = rig_camera_status(doc, f"{cam}/color")
        if status != "validated":
            raise ValueError(f"{cam}: published rig camera is {status}, not validated ({reasons})")
        if "sensor_depth" not in (record.get("depth_scale") or {}):
            raise ValueError(f"{cam}: published rig lacks a validated sensor_depth scale")
        correction = {"stage": "sensor_depth", "factor": float(record["depth_scale"]["sensor_depth"]),
                      "validated": True, "evidence": {"document": str(path), "name": doc.get("name"), **identity,
                                                      "camera_id": cam,
                                                      "fixedness_exception": exception is not None}}
        validate_native_correction(correction)
        calibration[cam] = {"K_native": np.asarray(K_native[cam], np.float64).tolist(),
                            "c2w": np.linalg.inv(np.asarray(record["T_cam_base"], np.float64)).tolist()}
        corrections[cam] = correction
    return {"calibration": calibration, "native_depth_correction": corrections}


def native_correction(root=None, episode_id=None, camera_id=None) -> dict:
    """Read once at session start. Only the accepted calibration's sensor_depth factor is eligible.

    With no accepted evidence, explicitly return identity. A source map missing sensor_depth fails through the
    existing calibration_depth_scale reader; an FS factor is NEVER borrowed.
    """
    if root is None:
        return {'stage': 'sensor_depth', 'factor': 1.0, 'validated': False,
                'evidence': 'no accepted native-sensor calibration supplied; no correction applied'}
    from s4d.realworld.pipeline import accepted_calibration_doc, calibration_depth_scale

    doc = accepted_calibration_doc(root, episode_id, camera_id)
    factor = calibration_depth_scale(root, episode_id, camera_id, 'sensor_depth')
    validated = doc is not None and 'sensor_depth' in (doc.get('depth_scale') or {})
    return {'stage': 'sensor_depth', 'factor': factor, 'validated': validated,
            'evidence': {'root': str(root), 'episode_id': episode_id, 'camera_id': camera_id,
                         'rule': 'accepted calibrate record depth_scale[sensor_depth]'} if validated else
                        'legacy/missing native calibration; no correction applied'}


def color_intrinsics(intr) -> np.ndarray:
    """librealsense integer centres -> training's continuous centres (+0.5), as the Panda H5 adapter does."""
    if any(abs(c) > 0 for c in intr.coeffs):
        raise ValueError('distorted colour stream: training requires the undistorted colour pinhole grid')
    return Intrinsics(width=intr.width, height=intr.height, fx=intr.fx, fy=intr.fy, cx=intr.ppx, cy=intr.ppy,
                      pixel_convention='integer_centers').K()


def extrinsics_matrix(ext) -> np.ndarray:
    """librealsense ``rs2_extrinsics`` (rotation COLUMN-major, as ``rs2_transform_point_to_point`` reads it) -> 4x4."""
    T = np.eye(4)
    T[:3, :3] = np.asarray(ext.rotation, np.float64).reshape(3, 3).T
    T[:3, 3] = np.asarray(ext.translation, np.float64)
    return T


def color_ray_denominator(K_color: np.ndarray, T_color_depth: np.ndarray, hw) -> np.ndarray:
    """(H, W) ``e3 . R_dc . ray_c`` of every colour pixel centre (continuous K; ``T_dc = inv(T_color_depth)``)."""
    H, W = (int(x) for x in hw)
    T_dc = np.linalg.inv(np.asarray(T_color_depth, np.float64))
    K = np.asarray(K_color, np.float64)
    v, u = np.mgrid[0:H, 0:W].astype(np.float64)
    rays = np.stack([(u + 0.5 - K[0, 2]) / K[0, 0], (v + 0.5 - K[1, 2]) / K[1, 1], np.ones_like(u)], -1)
    return rays @ T_dc[2, :3]


def depth_z_to_color_z(z_depthcam_m: np.ndarray, K_color: np.ndarray, T_color_depth: np.ndarray,
                       denominator: np.ndarray | None = None) -> np.ndarray:
    """``rs.align`` writes the DEPTH-camera z of the source pixel into the colour grid; training depth is colour-camera
    z. Exact per colour pixel centre (DEP, serving/DEPTH-COMPAT.md section 4): p_c = z_c ray_c,
    z_d = e3 . (R_dc p_c + t_dc) -> z_c = (z_d - t_dc_z) / (e3 . R_dc ray_c). 0 stays 0 (invalid)."""
    z = np.asarray(z_depthcam_m, np.float64)
    T_dc = np.linalg.inv(np.asarray(T_color_depth, np.float64))
    d = denominator if denominator is not None else color_ray_denominator(K_color, T_color_depth, z.shape)
    return np.where(z > 0, (z - T_dc[2, 3]) / d, 0.0).astype(np.float32)


def aligned_native_depth(aligned_z16: np.ndarray, depth_units_m: float, K_color: np.ndarray,
                         T_color_depth: np.ndarray, depth_range_m=SENSOR_DEPTH_RANGE_M,
                         denominator: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
    """rs.align output (z16 depth-camera z on the colour grid) -> (colour-camera z float32 metres, validity): units in
    float64 as ``register_depth``, the sensor_depth stage's range gate on the device z (as the stage gates before its
    warp), then the exact colour-z conversion (DEPTH-COMPAT.md section 5, steps 1-2)."""
    raw = np.asarray(aligned_z16)
    if raw.dtype != np.uint16:
        raise ValueError("native aligned depth must be z16")
    m = raw.astype(np.float64) * float(depth_units_m)
    lo, hi = depth_range_m
    valid = (raw > 0) & (m >= lo) & (m <= hi)
    depth = depth_z_to_color_z(np.where(valid, m, 0.0), K_color, T_color_depth, denominator)
    valid &= depth > 0
    return np.where(valid, depth, np.float32(0)), valid


def emulate_rs_align(z16: np.ndarray, K_depth: np.ndarray, K_color: np.ndarray, T_color_depth: np.ndarray,
                     color_hw, depth_units_m: float) -> np.ndarray:
    """Recorded replay of the LIVE path: numpy emulation of librealsense 2.59.1 ``rs2::align(RS2_STREAM_COLOR)`` for
    z16 (``align_z_to_other``, CPU rule) by agent DEP (serving/tools/rs_align_emul.py; validated against real
    pyrealsense2 2.59.1 on recorded frames: interior bitwise identical, differences only on border rows/columns
    0/478/479 and 0/638/639 -> output row 127 of the 128 grid). Per non-zero depth pixel the corners (x +- 0.5) are
    deprojected at its depth, moved to colour, projected; the rounded colour rectangle receives the ORIGINAL z16 with a
    z-buffer (min); rectangles touching the border are skipped. float32 arithmetic like the C code. K_* continuous
    (converted here to librealsense integer centres). Returns aligned z16 (depth-camera z), 0 = no data."""
    f32 = np.float32
    Hc, Wc = (int(x) for x in color_hw)
    Kd = np.asarray(K_depth, np.float64) - np.array([[0, 0, 0.5], [0, 0, 0.5], [0, 0, 0]])
    Kc = np.asarray(K_color, np.float64) - np.array([[0, 0, 0.5], [0, 0, 0.5], [0, 0, 0]])
    z16 = np.asarray(z16)
    if z16.dtype != np.uint16:
        raise ValueError("recorded native depth must be z16")
    ys, xs = np.nonzero(z16)
    zr = z16[ys, xs]
    depth = zr.astype(f32) * f32(depth_units_m)
    fxd, fyd, cxd, cyd = f32(Kd[0, 0]), f32(Kd[1, 1]), f32(Kd[0, 2]), f32(Kd[1, 2])
    fxc, fyc, cxc, cyc = f32(Kc[0, 0]), f32(Kc[1, 1]), f32(Kc[0, 2]), f32(Kc[1, 2])
    R = np.asarray(T_color_depth, np.float64)[:3, :3].astype(f32)
    t = np.asarray(T_color_depth, np.float64)[:3, 3].astype(f32)

    def corner(off):
        px, py = xs.astype(f32) + f32(off), ys.astype(f32) + f32(off)
        X, Y, Z = ((px - cxd) / fxd) * depth, ((py - cyd) / fyd) * depth, depth
        Xo = R[0, 0] * X + R[0, 1] * Y + R[0, 2] * Z + t[0]
        Yo = R[1, 0] * X + R[1, 1] * Y + R[1, 2] * Z + t[1]
        Zo = R[2, 0] * X + R[2, 1] * Y + R[2, 2] * Z + t[2]
        u, v = (Xo / Zo) * fxc + cxc, (Yo / Zo) * fyc + cyc
        return np.trunc(u + f32(0.5)).astype(np.int64), np.trunc(v + f32(0.5)).astype(np.int64)

    x0, y0 = corner(-0.5)
    x1, y1 = corner(0.5)
    ok = (x0 >= 0) & (y0 >= 0) & (x1 < Wc) & (y1 < Hc)
    x0, y0, x1, y1, zr = x0[ok], y0[ok], x1[ok], y1[ok], zr[ok]
    out = np.full(Hc * Wc, np.iinfo(np.uint16).max, np.int64)
    w, h = np.maximum(x1 - x0 + 1, 0), np.maximum(y1 - y0 + 1, 0)
    for dy in range(int(h.max(initial=0))):
        for dx in range(int(w.max(initial=0))):
            m = (dy < h) & (dx < w)
            np.minimum.at(out, (y0[m] + dy) * Wc + (x0[m] + dx), zr[m].astype(np.int64))
    out[out == np.iinfo(np.uint16).max] = 0
    return out.reshape(Hc, Wc).astype(np.uint16)


class RealSenseRGBD:
    def __init__(self, profile, *, correction: dict, out_size: int = IMAGE_SIZE, rs_module=None,
                 depth_range_m: tuple[float, float] = SENSOR_DEPTH_RANGE_M, T_color_depth: np.ndarray | None = None):
        """Create rs.align once; ``profile`` is an ALREADY started pipeline profile (SDK imported only here).
        ``T_color_depth`` (depth -> colour, metres) defaults to the device extrinsics of the depth and colour streams;
        ``depth_range_m``: the sensor_depth stage's range (``InputSpec.depth_range_m``); ``correction``: the session's
        native correction (``load_rig_session``)."""
        if rs_module is None:
            import pyrealsense2 as rs_module
        self.rs = rs_module
        self.align = rs_module.align(rs_module.stream.color)
        self.depth_scale_m = float(profile.get_device().first_depth_sensor().get_depth_scale())
        if not np.isfinite(self.depth_scale_m) or self.depth_scale_m <= 0:
            raise ValueError("invalid device depth scale")
        validate_native_correction(correction)
        if T_color_depth is None:
            depth_stream, color_stream = profile.get_stream(rs_module.stream.depth), profile.get_stream(
                rs_module.stream.color)
            T_color_depth = extrinsics_matrix(depth_stream.get_extrinsics_to(color_stream))
        self.T_color_depth = np.asarray(T_color_depth, np.float64)
        self.correction, self.out_size = correction, out_size
        self.depth_range_m = (float(depth_range_m[0]), float(depth_range_m[1]))
        self._denominator = None  # (K bytes, hw) -> per-pixel colour-z denominator, computed once

    def process(self, frameset) -> dict:
        """rs.align -> depth-camera z16 -> range gate -> exact colour-camera z (metres, 0 invalid) -> native correction
        -> the shared crop / resize / K update. Preserves both SDK timestamps/domains/frame numbers.

        The caller maps SDK timestamps into the common monotonic ns clock used by HistoryWindow; timestamps from
        different device clock domains MUST NOT be compared directly.
        """
        aligned = self.align.process(frameset)
        color, depth = aligned.get_color_frame(), aligned.get_depth_frame()
        if not color or not depth:
            raise ValueError("aligned colour and native depth frames are both required")
        vp = color.profile.as_video_stream_profile()
        K = color_intrinsics(vp.get_intrinsics())
        rgb = np.asanyarray(color.get_data())
        fmt = vp.format()
        if fmt == self.rs.format.bgr8:
            rgb = rgb[..., ::-1].copy()
        elif fmt != self.rs.format.rgb8:
            raise ValueError("colour stream must be rgb8 or bgr8")
        raw = np.asanyarray(depth.get_data())
        key = (K.tobytes(), raw.shape)
        if self._denominator is None or self._denominator[0] != key:
            self._denominator = (key, color_ray_denominator(K, self.T_color_depth, raw.shape))
        metres, valid = aligned_native_depth(raw, self.depth_scale_m, K, self.T_color_depth, self.depth_range_m,
                                             self._denominator[1])
        result = preprocess_frame(rgb, depth=metres, valid=valid, K_native=K, out_size=self.out_size,
                                  native_correction=self.correction)
        result["metadata"] = {"depth_source": NATIVE_DEPTH, "native_correction": self.correction,
                              "alignment": "rs.align(color) -> colour-camera z",
                              "depth_scale_m": self.depth_scale_m,
                              "color_timestamp_ms": float(color.get_timestamp()),
                              "depth_timestamp_ms": float(depth.get_timestamp()),
                              "color_timestamp_domain": str(color.get_frame_timestamp_domain()),
                              "depth_timestamp_domain": str(depth.get_frame_timestamp_domain()),
                              "color_frame_number": int(color.get_frame_number()),
                              "depth_frame_number": int(depth.get_frame_number())}
        return result


def align_recorded_depth(depth_raw: np.ndarray, K_depth: np.ndarray, K_color: np.ndarray,
                         T_color_depth: np.ndarray, color_hw, depth_scale_m: float) -> dict:
    """The release ``sensor_depth`` stage on one recorded z16 frame (``register_depth`` with the stage defaults:
    z-buffer forward warp, splat 2, 0.1-6 m, holes stay 0; colour-camera z). K matrices continuous-centre. This is the
    registration the native depth scale was fitted on; ``rs.align`` (live) is a different rasteriser (DEPTH-COMPAT.md:
    after the colour-z conversion ~20 % of pixels differ by > 1 mm, <= 0.4 mm |dz| at the 128-px representation).
    """
    from s4d.realworld.kernels import register_depth

    if np.asarray(depth_raw).dtype != np.uint16 or not np.isfinite(depth_scale_m) or depth_scale_m <= 0:
        raise ValueError('recorded native depth must be uint16 with a positive device scale')
    out, _ = register_depth({'color_hw': list(color_hw), 'depth_scale_m': depth_scale_m,
                             'min_depth_m': SENSOR_DEPTH_RANGE_M[0], 'max_depth_m': SENSOR_DEPTH_RANGE_M[1],
                             'splat': 2},
                            {'depth_raw': np.asarray(depth_raw)[None], 'K_depth': K_depth, 'K_color': K_color,
                             'T_color_depth': T_color_depth})
    return {k: v[0] for k, v in out.items()}
