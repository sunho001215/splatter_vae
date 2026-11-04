"""Compute kernels of the real-world pipeline: pure functions ``(params, arrays) -> (outputs, meta)``.

Stages read their inputs, call a kernel through a runner (``s4d.realworld.runners``), durably save the outputs and only
then acknowledge. The same kernel runs in-process (tests, ``LocalRunner``) or inside the inference container
(``s4d/realworld/workers/s4d_kernels_worker.py`` in the repository environment), so where the compute happens never
changes what it computes. Kernels only wrap the existing, tested functions of this package; they hold no I/O.
"""

from __future__ import annotations

import json

import numpy as np

KERNELS: dict = {}


def kernel(name: str):
    def register(fn):
        KERNELS[name] = fn
        return fn

    return register


def run_kernel(task: str, params: dict, arrays: dict) -> tuple[dict, dict]:
    if task not in KERNELS:
        raise KeyError(f"unknown kernel {task!r}; known: {sorted(KERNELS)}")
    return KERNELS[task](params, arrays)


@kernel("register_depth")
def register_depth(params: dict, arrays: dict) -> tuple[dict, dict]:
    """Raw sensor depth (N,Hd,Wd) uint16 -> metric camera-z on the colour grid (N,H,W) float32 + validity.

    Units: ``depth_scale_m`` per stored unit; 0 is invalid; values outside [min_depth_m, max_depth_m] are rejected
    (counted) before the z-buffer forward warp ``geometry.reproject_depth`` (holes stay invalid, never filled)."""
    from s4d.realworld.geometry import reproject_depth

    raw = arrays["depth_raw"]
    K_d, K_c = np.asarray(arrays["K_depth"], float), np.asarray(arrays["K_color"], float)
    T = np.asarray(arrays["T_color_depth"], float)
    hw = tuple(int(x) for x in params["color_hw"])
    lo, hi, scale = float(params["min_depth_m"]), float(params["max_depth_m"]), float(params["depth_scale_m"])
    out = np.zeros((len(raw),) + hw, np.float32)
    stats = []
    for k, d in enumerate(raw):
        m = d.astype(np.float64) * scale
        sensor_valid = d > 0
        keep = sensor_valid & (m >= lo) & (m <= hi)
        out[k] = reproject_depth(np.where(keep, m, 0.0), K_d, T, K_c, hw, splat=int(params.get("splat", 2)))
        stats.append({"sensor_valid_fraction": float(sensor_valid.mean()),
                      "rejected_below_min": int((sensor_valid & (m < lo)).sum()),
                      "rejected_above_max": int((sensor_valid & (m > hi)).sum()),
                      "registered_valid_fraction": float((out[k] > 0).mean())})
    return {"depth": out, "valid": out > 0}, {"frames": stats}


@kernel("calibrate_camera")
def calibrate_camera_kernel(params: dict, arrays: dict) -> tuple[dict, dict]:
    """``calibration.calibrate_camera`` on stacked frames; overlays for the requested frames."""
    from s4d.realworld.calibration import CalibrationConfig, CalibrationFrame, calibrate_camera, overlay_image

    cfg = CalibrationConfig(**{**params["cfg"], "image_hw": tuple(params["cfg"]["image_hw"])})
    n = len(arrays["t_index"])
    depth = arrays.get("depth")
    masks = arrays.get("robot_mask")
    frames = [CalibrationFrame(int(arrays["t_index"][k]), int(arrays["timestamp_ns"][k]), arrays["landmarks_base"][k],
                               arrays["landmarks_uv"][k], arrays["landmarks_conf"][k], arrays["surface_base"][k],
                               arrays["flange_base"][k], None if depth is None else depth[k],
                               None if masks is None else masks[k]) for k in range(n)]
    K = np.asarray(arrays["K"], float)
    res = calibrate_camera(params["camera_id"], frames, K, cfg,
                           T_init=None if "T_init" not in arrays else arrays["T_init"])
    overlays = []
    if res.T_cam_base is not None and "overlay_images" in arrays:
        for img, k in zip(arrays["overlay_images"], arrays["overlay_frames"]):
            overlays.append(overlay_image(img, res.T_cam_base, frames[int(k)], K, cfg.image_hw))
    out = {"overlays": np.stack(overlays) if overlays else np.zeros((0, *cfg.image_hw, 3), np.uint8)}
    return out, {"result": json.loads(json.dumps(res.to_json(), default=_json_default))}


@kernel("lift_tracks")
def lift_tracks_kernel(params: dict, arrays: dict) -> tuple[dict, dict]:
    """``tracking.lift_tracks`` for one track chunk with its frames' depth on the same grid."""
    from s4d.realworld.tracking import Tracks2D, lift_tracks, status_counts

    tr = Tracks2D(params["camera_id"], int(params["query_frame"]), arrays["frame_indices"], arrays["timestamps_ns"],
                  arrays["point_ids"], arrays["uv"], arrays["visibility"], arrays["confidence"],
                  tuple(int(x) for x in params["image_hw"]), dict(params.get("source", {})))
    t3 = lift_tracks(tr, arrays["depth"], np.asarray(arrays["K"], float), np.asarray(arrays["T_cam_base"], float),
                     vis_threshold=float(params["vis_threshold"]), conf_threshold=float(params["conf_threshold"]))
    out = {"frame_indices": t3.frame_indices, "timestamps_ns": t3.timestamps_ns, "point_ids": t3.point_ids,
           "xyz_base": t3.xyz_base, "xyz_cam": t3.xyz_cam, "status": t3.status, "visibility": t3.visibility,
           "confidence": t3.confidence}
    return out, {"status_counts": status_counts(t3), "source": json.loads(json.dumps(t3.source, default=_json_default))}


def _json_default(x):
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, (np.floating, np.integer, np.bool_)):
        return x.item()
    return str(x)
