from __future__ import annotations

import math
import numpy as np
from dataclasses import dataclass
from typing import Sequence, Tuple


@dataclass(frozen=True)
class CameraPose:
    name: str
    pos: np.ndarray        # (3,)
    quat_wxyz: np.ndarray  # (4,) world_R_cam, MuJoCo quat format [w,x,y,z]
    fovy_deg: float


def _normalize(v: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    n = float(np.linalg.norm(v))
    if n < eps:
        return v * 0.0
    return v / n


def _mat_to_quat_wxyz(R: np.ndarray) -> np.ndarray:
    """
    Convert rotation matrix to quaternion [w,x,y,z].
    """
    # Robust matrix->quat (same approach as your robosuite helper)
    m = R
    t = np.trace(m)
    if t > 0.0:
        S = math.sqrt(t + 1.0) * 2.0
        w = 0.25 * S
        x = (m[2, 1] - m[1, 2]) / S
        y = (m[0, 2] - m[2, 0]) / S
        z = (m[1, 0] - m[0, 1]) / S
    elif (m[0, 0] > m[1, 1]) and (m[0, 0] > m[2, 2]):
        S = math.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2]) * 2.0
        w = (m[2, 1] - m[1, 2]) / S
        x = 0.25 * S
        y = (m[0, 1] + m[1, 0]) / S
        z = (m[0, 2] + m[2, 0]) / S
    elif m[1, 1] > m[2, 2]:
        S = math.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2]) * 2.0
        w = (m[0, 2] - m[2, 0]) / S
        x = (m[0, 1] + m[1, 0]) / S
        y = 0.25 * S
        z = (m[1, 2] + m[2, 1]) / S
    else:
        S = math.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1]) * 2.0
        w = (m[1, 0] - m[0, 1]) / S
        x = (m[0, 2] + m[2, 0]) / S
        y = (m[1, 2] + m[2, 1]) / S
        z = 0.25 * S

    q = np.array([w, x, y, z], dtype=np.float64)
    q = q / np.linalg.norm(q)
    return q


def look_at_quat_wxyz(cam_pos: Sequence[float], lookat: Sequence[float], up: Sequence[float]) -> np.ndarray:
    """
    MuJoCo camera convention:
      - camera looks along -Z in camera frame
      - +X right in image, +Y up in image

    Build world_R_cam, then convert to quat [w,x,y,z].
    """
    cam_pos = np.asarray(cam_pos, dtype=np.float64)
    lookat = np.asarray(lookat, dtype=np.float64)
    up = np.asarray(up, dtype=np.float64)

    z_cam_world = _normalize(cam_pos - lookat)               # camera +Z points "backward"
    x_cam_world = np.cross(up, z_cam_world)
    if np.linalg.norm(x_cam_world) < 1e-6:
        up2 = np.array([0.0, 1.0, 0.0], dtype=np.float64)
        x_cam_world = np.cross(up2, z_cam_world)
    x_cam_world = _normalize(x_cam_world)
    y_cam_world = _normalize(np.cross(z_cam_world, x_cam_world))

    R = np.stack([x_cam_world, y_cam_world, z_cam_world], axis=1)  # world_R_cam
    return _mat_to_quat_wxyz(R)


def spherical_camera_pose(
    *,
    name: str,
    r: float,
    theta_deg: float,
    phi_deg: float,
    lookat: Sequence[float],
    up: Sequence[float],
    fovy_deg: float,
) -> CameraPose:
    """
    Matches your robosuite convention:
      phi = azimuth around +Z, theta = elevation from XY plane.
    """
    th = math.radians(theta_deg)
    ph = math.radians(phi_deg)

    x = float(r) * math.cos(th) * math.cos(ph)
    y = float(r) * math.cos(th) * math.sin(ph)
    z = float(r) * math.sin(th)

    lookat_np = np.asarray(lookat, dtype=np.float64)
    pos = lookat_np + np.array([x, y, z], dtype=np.float64)
    quat = look_at_quat_wxyz(pos, lookat_np, up)

    return CameraPose(name=name, pos=pos, quat_wxyz=quat, fovy_deg=float(fovy_deg))


def intrinsics_from_fovy(fovy_deg: float, height: int, width: int) -> np.ndarray:
    """
    Simple pinhole intrinsics derived from vertical FOV.
    """
    fovy = math.radians(float(fovy_deg))
    fy = 0.5 * height / math.tan(0.5 * fovy)
    fovx = 2.0 * math.atan(math.tan(0.5 * fovy) * (width / height))
    fx = 0.5 * width / math.tan(0.5 * fovx)
    cx = (width - 1) / 2.0
    cy = (height - 1) / 2.0

    K = np.array([[fx, 0.0, cx],
                  [0.0, fy, cy],
                  [0.0, 0.0, 1.0]], dtype=np.float64)
    return K


def extrinsics_world_T_cam(cam_pos: np.ndarray, quat_wxyz: np.ndarray) -> np.ndarray:
    """
    Return 4x4 transform mapping camera-frame coords -> world coords.
    We store it as world_T_cam to match your robosuite dataset naming.
    """
    w, x, y, z = quat_wxyz.tolist()
    # quaternion to rotation matrix
    R = np.array([
        [1 - 2*(y*y + z*z), 2*(x*y - z*w),     2*(x*z + y*w)],
        [2*(x*y + z*w),     1 - 2*(x*x + z*z), 2*(y*z - x*w)],
        [2*(x*z - y*w),     2*(y*z + x*w),     1 - 2*(x*x + y*y)],
    ], dtype=np.float64)

    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = R
    T[:3, 3] = cam_pos.astype(np.float64)
    return T