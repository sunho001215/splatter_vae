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
