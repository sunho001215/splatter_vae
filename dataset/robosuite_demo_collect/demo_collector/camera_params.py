from __future__ import annotations

import numpy as np
from typing import Tuple


def _camera_name_to_id(sim, camera_name: str) -> int:
    """
    Compatible with both mujoco-python bindings and older mujoco-py style APIs.
    """
    # mujoco python usually has: sim.model.camera_name2id(name)
    if hasattr(sim.model, "camera_name2id"):
        return int(sim.model.camera_name2id(camera_name))
    # fallback: some APIs use cam_name2id
    if hasattr(sim.model, "cam_name2id"):
        return int(sim.model.cam_name2id(camera_name))
    raise AttributeError("Cannot find camera_name2id on sim.model")


def get_camera_intrinsics(sim, camera_name: str, height: int, width: int) -> np.ndarray:
    """
    Returns K (3x3) using MuJoCo camera fovy + image size.

    MuJoCo camera has vertical field-of-view fovy (degrees).
    """
    cam_id = _camera_name_to_id(sim, camera_name)

    # fovy in degrees (vertical)
    if hasattr(sim.model, "cam_fovy"):
        fovy_deg = float(sim.model.cam_fovy[cam_id])
    else:
        # fallback
        fovy_deg = float(sim.model.camera_fovy[cam_id])

    fovy = np.deg2rad(fovy_deg)
    fy = 0.5 * height / np.tan(0.5 * fovy)
    fx = fy * (width / float(height))

    cx = (width - 1) * 0.5
    cy = (height - 1) * 0.5

    K = np.array([
        [fx, 0.0, cx],
        [0.0, fy, cy],
        [0.0, 0.0, 1.0],
    ], dtype=np.float64)
    return K


def get_camera_extrinsics_world_T_cam(sim, camera_name: str) -> np.ndarray:
    """
    Returns a 4x4 transform world_T_cam derived from MuJoCo sim.data:
      - cam_xpos (3,) position
      - cam_xmat (9,) rotation matrix flattened row-major

    Note: "extrinsics" can be defined either as world_T_cam or cam_T_world.
    We store world_T_cam (easy to invert later).
    """
    cam_id = _camera_name_to_id(sim, camera_name)

    # Position in world coordinates
    xpos = np.array(sim.data.cam_xpos[cam_id], dtype=np.float64)  # (3,)

    # Rotation matrix in world coordinates
    xmat = np.array(sim.data.cam_xmat[cam_id], dtype=np.float64).reshape(3, 3)

    world_T_cam = np.eye(4, dtype=np.float64)
    world_T_cam[:3, :3] = xmat
    world_T_cam[:3, 3] = xpos
    return world_T_cam
