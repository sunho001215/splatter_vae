"""Pose arithmetic of the robot-side safety layer (numpy only): pose7 = [x y z qx qy qz qw] (metres, unit quaternion,
scalar last; the recorder's ``robot/flange_pose`` / ``action/flange_pose`` convention, base frame panda_link0).
Used for limits and rate limiting of commands, never to (re)interpret a policy output."""

from __future__ import annotations

import numpy as np


def as_pose7(p) -> np.ndarray:
    p = np.asarray(p, np.float64).reshape(7)
    if not np.isfinite(p).all():
        raise ValueError("pose7 has non-finite values")
    return p


def quat_angle(qa, qb) -> float:
    """Geodesic angle (rad) between two rotations given as quaternions (normalised here; sign-invariant)."""
    qa, qb = np.asarray(qa, np.float64), np.asarray(qb, np.float64)
    na, nb = float(np.linalg.norm(qa)), float(np.linalg.norm(qb))
    if not (np.isfinite(na) and np.isfinite(nb)) or na < 1e-9 or nb < 1e-9:
        raise ValueError("degenerate quaternion")
    d = abs(float(np.dot(qa / na, qb / nb)))
    return float(2.0 * np.arccos(min(1.0, d)))


def slerp(qa, qb, t: float) -> np.ndarray:
    """Unit quaternion a fraction ``t`` of the way along the short arc from qa to qb."""
    qa, qb = np.asarray(qa, np.float64), np.asarray(qb, np.float64)
    qa, qb = qa / np.linalg.norm(qa), qb / np.linalg.norm(qb)
    d = float(np.dot(qa, qb))
    if d < 0:
        qb, d = -qb, -d
    if d > 1.0 - 1e-12:
        q = qa + t * (qb - qa)
    else:
        theta = np.arccos(min(1.0, d))
        q = (np.sin((1 - t) * theta) * qa + np.sin(t * theta) * qb) / np.sin(theta)
    return q / np.linalg.norm(q)


def toward(start, goal, max_m: float, max_rad: float) -> np.ndarray:
    """``goal`` limited to ``max_m`` translation and ``max_rad`` rotation from ``start`` (independently, along the
    straight line / the short arc): the recorder's lead envelope and the executor's per-tick rate limit."""
    s, g = as_pose7(start), as_pose7(goal)
    out = g.copy()
    d = g[:3] - s[:3]
    n = float(np.linalg.norm(d))
    if n > max_m:
        out[:3] = s[:3] + d * (max_m / n)
    a = quat_angle(s[3:], g[3:])
    out[3:] = slerp(s[3:], g[3:], max_rad / a) if a > max_rad else g[3:] / np.linalg.norm(g[3:])
    return out


def quat_mul(a, b) -> np.ndarray:
    """Hamilton product of xyzw quaternions (a then b applied on the right: R(a) R(b))."""
    ax, ay, az, aw = np.asarray(a, np.float64)
    bx, by, bz, bw = np.asarray(b, np.float64)
    return np.array([aw * bx + ax * bw + ay * bz - az * by, aw * by - ax * bz + ay * bw + az * bx,
                     aw * bz + ax * by - ay * bx + az * bw, aw * bw - ax * bx - ay * by - az * bz])


def quat_to_rotvec(q) -> np.ndarray:
    q = np.asarray(q, np.float64) / np.linalg.norm(q)
    if q[3] < 0:
        q = -q
    s = float(np.linalg.norm(q[:3]))
    if s < 1e-15:
        return 2.0 * q[:3]
    return q[:3] / s * 2.0 * np.arctan2(s, q[3])


def rotvec_to_quat(r) -> np.ndarray:
    r = np.asarray(r, np.float64)
    a = float(np.linalg.norm(r))
    if a < 1e-15:
        return np.array([*(r / 2.0), 1.0]) / np.linalg.norm([*(r / 2.0), 1.0])
    return np.concatenate([np.sin(a / 2) * r / a, [np.cos(a / 2)]])


def rotvec_between(q_from, q_to) -> np.ndarray:
    """Base-frame rotation vector r with R(q_to) = exp(r) R(q_from) (the collector's mat_to_rotvec(R_goal R^T))."""
    qf = np.asarray(q_from, np.float64) / np.linalg.norm(q_from)
    return quat_to_rotvec(quat_mul(np.asarray(q_to, np.float64), np.array([-qf[0], -qf[1], -qf[2], qf[3]])))


def rotate_left(q, r) -> np.ndarray:
    """exp(r) R(q) as a unit quaternion."""
    out = quat_mul(rotvec_to_quat(r), q)
    return out / np.linalg.norm(out)
