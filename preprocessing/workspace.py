from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np


def backproject_z_depth_to_world(
    depth: np.ndarray,
    K: np.ndarray,
    c2w: np.ndarray,
) -> np.ndarray:
    """Back-project X-Lens z-depth using OpenCV pixel-center conventions."""
    z = np.asarray(depth, dtype=np.float64)
    camera = np.asarray(K, dtype=np.float64)
    transform = np.asarray(c2w, dtype=np.float64)
    if z.ndim != 2 or camera.shape != (3, 3) or transform.shape != (4, 4):
        raise ValueError("Expected depth HxW, K 3x3, and c2w 4x4.")
    height, width = z.shape
    y, x = np.meshgrid(
        np.arange(height, dtype=np.float64) + 0.5,
        np.arange(width, dtype=np.float64) + 0.5,
        indexing="ij",
    )
    camera_points = np.stack(
        (
            (x - camera[0, 2]) / camera[0, 0] * z,
            (y - camera[1, 2]) / camera[1, 1] * z,
            z,
        ),
        axis=-1,
    )
    return camera_points @ transform[:3, :3].T + transform[:3, 3]


@dataclass(frozen=True)
class WorkspaceParameterProposal:
    global_center: tuple[float, float, float]
    anchor_initial_spread: float
    parent_displacement_scale: float
    child_radius: float
    znear: float
    zfar: float


def propose_gaussian_workspace_parameters(
    points_world: np.ndarray,
    depths: np.ndarray,
) -> WorkspaceParameterProposal:
    points = np.asarray(points_world, dtype=np.float64)
    depth = np.asarray(depths, dtype=np.float64)
    points = points[np.isfinite(points).all(axis=-1)]
    depth = depth[np.isfinite(depth) & (depth > 0.0)]
    if len(points) < 100 or len(depth) < 100:
        raise ValueError(
            "At least 100 valid geometry samples are required for workspace statistics."
        )
    low, median, high = np.percentile(points, (5.0, 50.0, 95.0), axis=0)
    extent = np.maximum(high - low, 1.0e-3)
    characteristic = float(np.max(extent))
    depth_low, depth_high = np.percentile(depth, (1.0, 99.0))
    return WorkspaceParameterProposal(
        global_center=tuple(float(value) for value in median),
        anchor_initial_spread=max(0.02, characteristic / 3.29),
        parent_displacement_scale=max(0.01, characteristic * 0.20),
        child_radius=max(0.005, characteristic * 0.035),
        znear=max(0.01, float(depth_low) * 0.5),
        zfar=max(float(depth_low) + 0.1, float(depth_high) * 1.2),
    )


def percentile_dict(
    values: np.ndarray, percentiles: Sequence[float]
) -> dict[str, object]:
    array = np.asarray(values)
    result = np.percentile(array, tuple(float(value) for value in percentiles), axis=0)
    return {
        str(float(percentile)): np.asarray(value).tolist()
        for percentile, value in zip(percentiles, result, strict=True)
    }
