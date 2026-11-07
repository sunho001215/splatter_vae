"""NumPy Lie geometry for rig calibration, in radians and metres.

``T_cam_base`` maps base points to OpenCV camera coordinates (+z forward).
Twists are ``(omega, v)`` and act on the LEFT: ``Exp(xi) @ T``. In particular,
``dP_cam/dxi = [-skew(P_cam), I]``; the SE(3) translation is ``V @ v``, not v.
Pixels are continuous, with array pixel (i, j) centred at (j+.5, i+.5).
All Lie functions also accept leading batch dimensions.
"""

from __future__ import annotations

import numpy as np


def skew(v):
    """Cross-product matrices: ``skew(v) @ p == cross(v, p)``."""
    v = np.asarray(v, dtype=float)
    out = np.zeros(v.shape[:-1] + (3, 3))
    out[..., 0, 1], out[..., 0, 2] = -v[..., 2], v[..., 1]
    out[..., 1, 0], out[..., 1, 2] = v[..., 2], -v[..., 0]
    out[..., 2, 0], out[..., 2, 1] = -v[..., 1], v[..., 0]
    return out


def _coefficients(omega):
    theta2 = np.sum(omega * omega, axis=-1)
    small = theta2 < 1e-8
    theta = np.sqrt(np.where(small, 1.0, theta2))
    a = np.where(small, 1 - theta2 / 6 + theta2**2 / 120, np.sin(theta) / theta)
    b = np.where(small, .5 - theta2 / 24 + theta2**2 / 720, (1 - np.cos(theta)) / theta**2)
    c = np.where(small, 1 / 6 - theta2 / 120 + theta2**2 / 5040,
                 (theta - np.sin(theta)) / theta**3)
    return a[..., None, None], b[..., None, None], c[..., None, None]


def so3_exp(omega):
    """Rodrigues exponential, including its small-angle limit."""
    omega = np.asarray(omega, dtype=float)
    a, b, _ = _coefficients(omega)
    w = skew(omega)
    return np.eye(3) + a * w + b * (w @ w)


def so3_log(R):
    """Principal rotation vector (norm <= pi), stable near zero and pi."""
    R = np.asarray(R, dtype=float)
    v = np.stack((R[..., 2, 1] - R[..., 1, 2], R[..., 0, 2] - R[..., 2, 0],
                  R[..., 1, 0] - R[..., 0, 1]), axis=-1) / 2
    sin = np.linalg.norm(v, axis=-1)
    cos = np.clip((np.trace(R, axis1=-2, axis2=-1) - 1) / 2, -1, 1)
    theta = np.arctan2(sin, cos)
    out = v * np.where(sin > 1e-8, theta / np.maximum(sin, 1e-300),
                       1 + theta**2 / 6)[..., None]
    near_pi = cos < -1 + 1e-7
    if np.any(near_pi):
        # The axis is the eigenvector of the symmetric part with eigenvalue 1.
        _, axes = np.linalg.eigh((R[near_pi] + np.swapaxes(R[near_pi], -1, -2)) / 2)
        axis = axes[..., :, -1]
        sign = np.where(np.sum(axis * v[near_pi], axis=-1) < 0, -1.0, 1.0)
        out[near_pi] = axis * (theta[near_pi] * sign)[..., None]
    return out


def se3_exp(xi):
    """SE(3) exponential: ``R=Exp(omega)``, ``t=(I+b W+c W^2) v``."""
    xi = np.asarray(xi, dtype=float)
    omega, v = xi[..., :3], xi[..., 3:]
    _, b, c = _coefficients(omega)
    w = skew(omega)
    V = np.eye(3) + b * w + c * (w @ w)
    out = np.broadcast_to(np.eye(4), xi.shape[:-1] + (4, 4)).copy()
    out[..., :3, :3] = so3_exp(omega)
    out[..., :3, 3] = (V @ v[..., None])[..., 0]
    return out


def se3_log(T):
    """Inverse SE(3) exponential on the principal rotation branch."""
    T = np.asarray(T, dtype=float)
    omega = so3_log(T[..., :3, :3])
    theta2 = np.sum(omega * omega, axis=-1)
    small = theta2 < 1e-8
    theta = np.sqrt(np.where(small, 1.0, theta2))
    c = np.where(small, 1 / 12 + theta2 / 720 + theta2**2 / 30240,
                 (1 - .5 * theta / np.tan(.5 * theta)) / theta**2)
    w = skew(omega)
    Vinv = np.eye(3) - .5 * w + c[..., None, None] * (w @ w)
    v = (Vinv @ T[..., :3, 3, None])[..., 0]
    return np.concatenate((omega, v), axis=-1)


def project(Pc, K):
    """Return pixels (...,2) and analytic ``dpi/dPc`` (...,2,3).

    K uses continuous pixel coordinates (no half-pixel shift is added here).
    Nonfinite points or z <= 0 produce NaNs in both outputs.
    """
    Pc, K = np.asarray(Pc, dtype=float), np.asarray(K, dtype=float)
    z = Pc[..., 2]
    valid = np.isfinite(Pc).all(axis=-1) & (z > 0)
    z = np.where(valid, z, np.nan)
    uv = np.einsum("...ij,...j->...i", K, Pc)[..., :2] / z[..., None]
    jac = np.broadcast_to(K[..., :2, :], Pc.shape[:-1] + (2, 3)).copy() / z[..., None, None]
    jac[..., :, 2] -= uv / z[..., None]
    return uv, jac


def camera_centre(T):
    """Camera centre in the base frame, ``-R.T @ t`` (metres)."""
    T = np.asarray(T, dtype=float)
    return -(np.swapaxes(T[..., :3, :3], -1, -2) @ T[..., :3, 3, None])[..., 0]


def pose_distance(Ta, Tb):
    """Return (relative rotation angle in degrees, base-frame centre distance)."""
    Ta, Tb = np.asarray(Ta, dtype=float), np.asarray(Tb, dtype=float)
    rotation = np.swapaxes(Ta[..., :3, :3], -1, -2) @ Tb[..., :3, :3]
    return (np.degrees(np.linalg.norm(so3_log(rotation), axis=-1)),
            np.linalg.norm(camera_centre(Ta) - camera_centre(Tb), axis=-1))


def scaled_info_svd(J, W, D):
    """Return ``(s, Vt)`` for ``sqrt(W) J D``, without priors or damping.

    W is nonnegative row weights or a symmetric positive-semidefinite matrix;
    D is physical parameter scales (a vector or a matrix), with dx = D dz.
    Vt's ROWS are right singular vectors in dimensionless z coordinates.
    There is one singular value per parameter, including zero null directions.
    """
    J, W, D = (np.asarray(x, dtype=float) for x in (J, W, D))
    if W.ndim == 1:
        if np.any(W < 0):
            raise ValueError("W must be positive semidefinite")
        A = np.sqrt(W)[:, None] * J
    else:
        eigen, vectors = np.linalg.eigh(W)
        if np.any(eigen < -1e-12 * max(1.0, np.max(np.abs(eigen)))):
            raise ValueError("W must be positive semidefinite")
        A = (np.sqrt(np.maximum(eigen, 0))[:, None] * vectors.T) @ J
    A = A * D if D.ndim == 1 else A @ D
    # Avoid a potentially huge square U when there are more rows than columns.
    if A.shape[0] >= A.shape[1]:
        _, s, vt = np.linalg.svd(A, full_matrices=False)
    else:
        _, s, vt = np.linalg.svd(A, full_matrices=True)
        s = np.pad(s, (0, A.shape[1] - len(s)))
    return s, vt
