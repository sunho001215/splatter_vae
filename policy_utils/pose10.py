"""
Utilities for building the 10D representation:
  10D = [pos(3), rot6d(6), gripper(1)]

- rot6d is the "first two columns of rotation matrix" representation.
- Also provides a "delta pose" mode:
    dpos = pos_target - pos_ref
    dR   = R_target @ R_ref.T
    drot6d = rot6d(dR)
  which tends to behave more like an action (relative command) than absolute pose.

This file is independent of LeRobot and only uses NumPy.
"""

from __future__ import annotations
import numpy as np


def quat_xyzw_to_rotmat(q_xyzw: np.ndarray) -> np.ndarray:
    """
    Convert quaternion (x, y, z, w) to a 3x3 rotation matrix.

    Notes:
    - Robosuite commonly uses quaternions in (x,y,z,w).
    - We normalize defensively to avoid numerical issues.
    """
    q = np.asarray(q_xyzw, dtype=np.float64).reshape(4)
    n = float(np.linalg.norm(q))
    if n < 1e-12:
        return np.eye(3, dtype=np.float64)
    q = q / n
    x, y, z, w = q

    xx, yy, zz = x * x, y * y, z * z
    xy, xz, yz = x * y, x * z, y * z
    wx, wy, wz = w * x, w * y, w * z

    R = np.array(
        [
            [1.0 - 2.0 * (yy + zz), 2.0 * (xy - wz),       2.0 * (xz + wy)],
            [2.0 * (xy + wz),       1.0 - 2.0 * (xx + zz), 2.0 * (yz - wx)],
            [2.0 * (xz - wy),       2.0 * (yz + wx),       1.0 - 2.0 * (xx + yy)],
        ],
        dtype=np.float64,
    )
    return R


def rotmat_to_rot6d(R: np.ndarray) -> np.ndarray:
    """
    6D rotation representation: concatenate the first two columns of R (3 + 3).
    """
    R = np.asarray(R, dtype=np.float64).reshape(3, 3)
    return np.concatenate([R[:, 0], R[:, 1]], axis=0).astype(np.float64)  # (6,)


def pose10_from_obs(eef_pos: np.ndarray, eef_quat_xyzw: np.ndarray, gripper: float) -> np.ndarray:
    """
    Absolute pose 10D = [pos(3), rot6d(6), gripper(1)].
    """
    pos = np.asarray(eef_pos, dtype=np.float64).reshape(3)
    R = quat_xyzw_to_rotmat(eef_quat_xyzw)
    r6 = rotmat_to_rot6d(R)
    out = np.concatenate([pos, r6, np.array([float(gripper)], dtype=np.float64)], axis=0)
    assert out.shape == (10,)
    return out


def delta_pose10(ref_pos: np.ndarray, ref_quat_xyzw: np.ndarray,
                tgt_pos: np.ndarray, tgt_quat_xyzw: np.ndarray,
                tgt_gripper: float) -> np.ndarray:
    """
    Relative pose 10D (useful as an "action"-like target):
      dpos  = tgt_pos - ref_pos
      dR    = R_tgt @ R_ref^T
      drot6 = rot6d(dR)
      grip  = tgt_gripper (kept as absolute gripper command/value)

    Returns: (10,)
    """
    ref_pos = np.asarray(ref_pos, dtype=np.float64).reshape(3)
    tgt_pos = np.asarray(tgt_pos, dtype=np.float64).reshape(3)
    dpos = tgt_pos - ref_pos

    R_ref = quat_xyzw_to_rotmat(ref_quat_xyzw)
    R_tgt = quat_xyzw_to_rotmat(tgt_quat_xyzw)
    dR = R_tgt @ R_ref.T
    drot6 = rotmat_to_rot6d(dR)

    out = np.concatenate([dpos, drot6, np.array([float(tgt_gripper)], dtype=np.float64)], axis=0)
    assert out.shape == (10,)
    return out
