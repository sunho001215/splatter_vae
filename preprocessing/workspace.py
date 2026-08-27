from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np


def estimate_scene_center_from_camera_axes(
    camera_c2w: np.ndarray,
    *,
    maximum_axis_residual_m: float = 0.35,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Estimate a robust robot workspace center from paired optical axes.

    Returns the median closest-point center, every accepted per-pair center,
    and ``(distance_along_a, distance_along_b, residual)`` diagnostics.  Only
    forward-facing, non-parallel ray pairs with a bounded closest-point
    residual contribute.
    """

    poses = np.asarray(camera_c2w, dtype=np.float64)
    if poses.ndim != 4 or poses.shape[1:] != (2, 4, 4):
        raise ValueError("Camera poses must have shape (N,2,4,4).")
    centers: list[np.ndarray] = []
    diagnostics: list[tuple[float, float, float]] = []
    for pair in poses:
        origins = pair[:, :3, 3]
        directions = pair[:, :3, 2]
        norms = np.linalg.norm(directions, axis=-1)
        if not np.isfinite(pair).all() or np.any(norms < 1.0e-8):
            continue
        directions = directions / norms[:, None]
        if abs(float(np.dot(directions[0], directions[1]))) >= 0.9999:
            continue
        system = np.stack((directions[0], -directions[1]), axis=1)
        distances = np.linalg.lstsq(
            system, origins[1] - origins[0], rcond=None
        )[0]
        point_a = origins[0] + distances[0] * directions[0]
        point_b = origins[1] + distances[1] * directions[1]
        residual = float(np.linalg.norm(point_a - point_b))
        if (
            distances[0] <= 0.0
            or distances[1] <= 0.0
            or residual > float(maximum_axis_residual_m)
        ):
            continue
        centers.append((point_a + point_b) * 0.5)
        diagnostics.append((float(distances[0]), float(distances[1]), residual))
    if not centers:
        raise ValueError("No stable forward camera-axis intersections were available.")
    accepted = np.asarray(centers, dtype=np.float64)
    return (
        np.median(accepted, axis=0),
        accepted,
        np.asarray(diagnostics, dtype=np.float64),
    )


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
