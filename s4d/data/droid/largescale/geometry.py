"""Camera/pose conventions for joining DROID, PointWorld-DROID and DROID-3D geometry.

Conventions (all float64 internally, metres, radians, OpenCV camera axes x right / y down / z forward):

* ``T_a_b`` maps points from frame ``b`` into frame ``a``. ``c2w`` = ``T_world_cam``, ``w2c`` = ``T_cam_world``.
  world = robot base (DROID ``cartesian_position`` frame).
* DROID 6-vectors ``[x, y, z, rx, ry, rz]`` are ``T_base_X`` with ``scipy Rotation.from_euler("xyz", ...)``
  (extrinsic x-y-z, i.e. ``R = Rz @ Ry @ Rx``), exactly as DROID's ``droid/misc/transformations.py``
  (``euler_to_rmat`` = ``R.from_euler("xyz", ...)``) and as verified for ``cartesian_position`` against PointWorld's
  ``gripper_pose`` (docs/POINTWORLD_SCHEMA.md).
* PointWorld ``extrinsic`` / ``optimized_extrinsics`` are 4x4 ``w2c`` (verified by projection residual in the sample).
* PointWorld ``intrinsic`` is for 320x180 with integer pixel centres; the shared batch contract uses continuous
  centres (pixel ``(u, v)`` covers ``[u, u+1)``), so the principal point gains +0.5 before any rescale.
"""

from __future__ import annotations

import numpy as np

# ----------------------------------------------------------------------------------------------------- rotations


def euler_xyz_to_matrix(angles) -> np.ndarray:
    """``scipy.spatial.transform.Rotation.from_euler("xyz", angles).as_matrix()`` for (..., 3) radians."""
    a = np.asarray(angles, dtype=np.float64)
    x, y, z = a[..., 0], a[..., 1], a[..., 2]
    cx, sx, cy, sy, cz, sz = np.cos(x), np.sin(x), np.cos(y), np.sin(y), np.cos(z), np.sin(z)
    R = np.empty(a.shape[:-1] + (3, 3))
    R[..., 0, 0] = cy * cz
    R[..., 0, 1] = sx * sy * cz - cx * sz
    R[..., 0, 2] = cx * sy * cz + sx * sz
    R[..., 1, 0] = cy * sz
    R[..., 1, 1] = sx * sy * sz + cx * cz
    R[..., 1, 2] = cx * sy * sz - sx * cz
    R[..., 2, 0] = -sy
    R[..., 2, 1] = sx * cy
    R[..., 2, 2] = cx * cy
    return R


def quat_xyzw_to_matrix(q) -> np.ndarray:
    q = np.asarray(q, dtype=np.float64)
    q = q / np.linalg.norm(q, axis=-1, keepdims=True)
    x, y, z, w = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    R = np.empty(q.shape[:-1] + (3, 3))
    R[..., 0, 0] = 1 - 2 * (y * y + z * z)
    R[..., 0, 1] = 2 * (x * y - z * w)
    R[..., 0, 2] = 2 * (x * z + y * w)
    R[..., 1, 0] = 2 * (x * y + z * w)
    R[..., 1, 1] = 1 - 2 * (x * x + z * z)
    R[..., 1, 2] = 2 * (y * z - x * w)
    R[..., 2, 0] = 2 * (x * z - y * w)
    R[..., 2, 1] = 2 * (y * z + x * w)
    R[..., 2, 2] = 1 - 2 * (x * x + y * y)
    return R


def pose6_to_matrix(pose) -> np.ndarray:
    """DROID ``[x,y,z,rx,ry,rz]`` (..., 6) -> homogeneous ``T_base_X`` (..., 4, 4)."""
    p = np.asarray(pose, dtype=np.float64)
    if p.shape[-1] != 6 or not np.isfinite(p).all():
        raise ValueError("a DROID pose must be a finite [x, y, z, rx, ry, rz] vector")
    T = np.zeros(p.shape[:-1] + (4, 4))
    T[..., :3, :3] = euler_xyz_to_matrix(p[..., 3:])
    T[..., :3, 3] = p[..., :3]
    T[..., 3, 3] = 1
    return T


def pose7_to_matrix(pose) -> np.ndarray:
    """``[x,y,z,qx,qy,qz,qw]`` (PointWorld ``gripper_pose``) -> homogeneous matrix."""
    p = np.asarray(pose, dtype=np.float64)
    T = np.zeros(p.shape[:-1] + (4, 4))
    T[..., :3, :3] = quat_xyzw_to_matrix(p[..., 3:])
    T[..., :3, 3] = p[..., :3]
    T[..., 3, 3] = 1
    return T


def invert(T) -> np.ndarray:
    """Exact rigid inverse of (..., 4, 4) transforms."""
    T = np.asarray(T, dtype=np.float64)
    out = np.zeros_like(T)
    Rt = np.swapaxes(T[..., :3, :3], -1, -2)
    out[..., :3, :3] = Rt
    out[..., :3, 3] = -np.einsum("...ij,...j->...i", Rt, T[..., :3, 3])
    out[..., 3, 3] = 1
    return out


def check_rigid(T, name: str, *, atol: float = 1e-4) -> np.ndarray:
    T = np.asarray(T, dtype=np.float64)
    if T.shape[-2:] != (4, 4) or not np.isfinite(T).all():
        raise ValueError(f"{name}: expected finite (...,4,4) transforms")
    R = T[..., :3, :3]
    if not np.allclose(R @ np.swapaxes(R, -1, -2), np.eye(3), atol=atol):
        raise ValueError(f"{name}: rotation is not orthonormal")
    if not np.allclose(np.linalg.det(R), 1, atol=atol):
        raise ValueError(f"{name}: rotation is not proper")
    if not np.allclose(T[..., 3, :], [0, 0, 0, 1], atol=1e-6):
        raise ValueError(f"{name}: bottom row must be [0,0,0,1]")
    return T


def transform_points(T, points) -> np.ndarray:
    T = np.asarray(T, dtype=np.float64)
    p = np.asarray(points, dtype=np.float64)
    return p @ T[..., :3, :3].swapaxes(-1, -2) + T[..., None, :3, 3]


# ----------------------------------------------------------------------------------------------------- intrinsics


def to_continuous_centres(K, *, integer_centres: bool) -> np.ndarray:
    out = np.asarray(K, dtype=np.float64).copy()
    if integer_centres:
        out[0, 2] += 0.5
        out[1, 2] += 0.5
    return out


def scale_intrinsics(K_continuous, src_wh: tuple[int, int], dst_wh: tuple[int, int]) -> np.ndarray:
    """Rescale continuous-centre intrinsics between resolutions (independent x/y factors)."""
    out = np.asarray(K_continuous, dtype=np.float64).copy()
    out[0] *= dst_wh[0] / src_wh[0]
    out[1] *= dst_wh[1] / src_wh[1]
    out[2] = (0, 0, 1)
    return out


def rotate180_camera(K_continuous, c2w, width: int, height: int) -> tuple[np.ndarray, np.ndarray]:
    """The camera that sees an image rotated by 180 degrees (``img[::-1, ::-1]``).

    A continuous pixel ``(u, v)`` becomes ``(W - u, H - v)``. That equals keeping K with ``cx' = W - cx`` and
    ``cy' = H - cy`` and rotating the camera by 180 degrees about its optical axis: ``c2w' = c2w @ diag(-1,-1,1,1)``.
    """
    K = np.asarray(K_continuous, dtype=np.float64).copy()
    K[0, 2] = width - K[0, 2]
    K[1, 2] = height - K[1, 2]
    flip = np.diag([-1.0, -1.0, 1.0, 1.0])
    return K, np.asarray(c2w, dtype=np.float64) @ flip


def project(points_world, K, w2c) -> tuple[np.ndarray, np.ndarray]:
    """Continuous-centre pixels and camera z of world points."""
    cam = transform_points(w2c, points_world)
    z = cam[..., 2]
    uv = (cam @ np.asarray(K, dtype=np.float64).T)[..., :2] / np.maximum(z, 1e-9)[..., None]
    return uv, z


def unproject(uv, z, K, c2w) -> np.ndarray:
    uv = np.asarray(uv, dtype=np.float64)
    K = np.asarray(K, dtype=np.float64)
    x = (uv[..., 0] - K[0, 2]) / K[0, 0] * z
    y = (uv[..., 1] - K[1, 2]) / K[1, 1] * z
    return transform_points(c2w, np.stack([x, y, np.asarray(z, dtype=np.float64)], -1))


def w2c_direction_residual(points_world, depth_m, K, matrix) -> dict:
    """Median depth residual of projecting with ``matrix`` as w2c versus as c2w (decides the direction from data)."""
    out = {}
    for name, w2c in (("as_w2c", matrix), ("as_c2w", invert(matrix))):
        uv, z = project(points_world, K, w2c)
        h, w = depth_m.shape
        px = np.floor(uv).astype(np.int64)
        ok = (z > 0) & (px[..., 0] >= 0) & (px[..., 0] < w) & (px[..., 1] >= 0) & (px[..., 1] < h)
        d = np.zeros(z.shape)
        d[ok] = depth_m[px[ok, 1], px[ok, 0]]
        ok &= d > 0
        out[name] = float(np.median(np.abs(z[ok] - d[ok]))) if ok.any() else float("inf")
        out[name + "_count"] = int(ok.sum())
    return out


# ----------------------------------------------------------------------------------------------------- wrist cameras


def wrist_camera_c2w(ee_pose6, T_ee_cam) -> np.ndarray:
    """Per-frame wrist camera pose ``T_base_cam(t) = T_base_ee(t) @ T_ee_cam``.

    ``ee_pose6`` is DROID ``observation/cartesian_position`` (T,6) at the frames' own steps; ``T_ee_cam`` is the
    hand-eye transform (camera in the end-effector frame). DROID's ``change_pose_frame(offset, ee)`` composes the same
    way (``droid/misc/transformations.py``: ``R = R_ee @ R_offset``, ``t = R_ee @ t_offset + t_ee``).
    """
    T_base_ee = pose6_to_matrix(ee_pose6)
    return check_rigid(T_base_ee @ np.asarray(T_ee_cam, dtype=np.float64), "wrist c2w")


def hand_eye_from_pose(base_T_cam, ee_pose6) -> np.ndarray:
    """Recover ``T_ee_cam`` from one camera pose recorded together with the EE pose of the same step."""
    return invert(pose6_to_matrix(ee_pose6)) @ np.asarray(base_T_cam, dtype=np.float64)


# ----------------------------------------------------------------------------------------------------- calibration change


def rebase_static_camera_points(points_world_src, w2c_src, w2c_dst) -> np.ndarray:
    """Exactly re-express points lifted from one *static* camera under a different extrinsic calibration.

    PointWorld lifts each external camera's 2D tracks with that camera's own depth, K and its (static) optimized
    ``w2c``: ``X = c2w_src @ lift(uv, d, K)``. Under a corrected calibration the same observation lifts to
    ``X' = c2w_dst @ lift(uv, d, K) = c2w_dst @ w2c_src @ X``. This is exact for every time step because the camera
    does not move; it is *not* valid for points of a different camera or for a moving (wrist) camera.
    """
    T = invert(w2c_dst) @ np.asarray(w2c_src, dtype=np.float64)
    return transform_points(T, points_world_src)
