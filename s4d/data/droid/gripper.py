"""Thirty-two approximate EE/finger surface points from verified raw RLDS state.

The release URDF mounts Robotiq to panda_link8 with +pi/2 yaw. Its opening
axis therefore follows local EE y and the approach axis follows local +z.
This deliberately avoids a full robot or mesh teacher. One constant approach
translation is fit from depth and fixed for every converted sample.
"""

from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation

from s4d.data.droid.pointworld import depth_test

STROKE_M = 0.085


def gripper_points(cartesian: np.ndarray, opening: np.ndarray, offset_m: float = 0) -> np.ndarray:
    pose = np.asarray(cartesian, dtype=np.float64)
    grip = np.asarray(opening).reshape(-1)
    if pose.ndim != 2 or pose.shape[1] != 6 or len(grip) != len(pose):
        raise ValueError("gripper state must be (T,6) Euler pose and (T,) normalized closure")
    if not np.isfinite(pose).all() or not np.isfinite(grip).all() or np.any((grip < 0) | (grip > 1)):
        raise ValueError("invalid metric EE pose or normalized gripper closure")
    local = np.zeros((len(pose), 32, 3), dtype=np.float64)
    palm = np.array([(x, y, z) for x in (-0.02, 0.02) for y in (-0.027, 0.027) for z in (0.015, 0.035)])
    local[:, :8] = palm
    half_gap = (1 - grip) * STROKE_M / 2
    j = 8
    for side in (-1, 1):
        for x in (-0.008, 0.008):
            for thick in (-0.006, 0.006):
                for z in (0.075, 0.10, 0.125):
                    local[:, j, 0] = x
                    local[:, j, 1] = side * half_gap + thick
                    local[:, j, 2] = z
                    j += 1
    local[:, :, 2] += offset_m
    rotation = Rotation.from_euler("xyz", pose[:, 3:]).as_matrix()
    return (np.einsum("tij,tnj->tni", rotation, local) + pose[:, None, :3]).astype(np.float32)


def gripper_depth_residuals(points: np.ndarray, K: np.ndarray, w2c: np.ndarray, depths: np.ndarray) -> np.ndarray:
    residuals = []
    for t in range(len(points)):
        for view in range(len(K)):
            valid, _, residual = depth_test(points[t], K[view], w2c[view], depths[t, view], surface=False)
            residuals.append(np.abs(residual[valid]))
    return np.concatenate(residuals) if residuals else np.empty(0, np.float32)


def calibrate_offset(cartesian: np.ndarray, closure: np.ndarray, K: np.ndarray, w2c: np.ndarray, depths: np.ndarray) -> dict:
    """Fit exactly one approach-axis offset, using unoccluded residuals only.

    Candidate coverage and in-front residuals are retained in the report to
    expose geometry inadequacy rather than filtering to the pass threshold.
    """
    candidates = []
    for offset in np.linspace(-0.03, 0.06, 91):
        points = gripper_points(cartesian, closure, float(offset))
        errors = gripper_depth_residuals(points, K, w2c, depths)
        valid = bool(len(errors) >= 32 and np.isfinite(errors).all())
        score = float(np.median(errors)) if valid else None
        candidates.append({"offset_m": float(offset), "median_residual_m": score, "count": len(errors), "valid": valid})
    eligible = [item for item in candidates if item["valid"]]
    if not eligible:
        raise ValueError("not enough unoccluded gripper points to calibrate one offset")
    best = min(eligible, key=lambda item: (item["median_residual_m"], abs(item["offset_m"])))
    return {
        **best,
        "approach_axis": "local EE +z",
        "opening_axis": "local EE y",
        "stroke_m": STROKE_M,
        "points_per_timestep": 32,
        "candidates": candidates,
        "selection": "in-image,positive depth,point z <= measured depth+0.02 m; no in-front residual truncation",
    }
