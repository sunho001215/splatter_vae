"""Depth diagnostics without ground truth: raw sensor depth versus MapAnything depth on the same views.

None of these is an accuracy measurement - there is no reference depth. They are consistency evidence:

* cross-view consistency: each view's valid depth, unprojected with its K and calibrated pose, reprojected into every
  other view of the same synchronized set and compared with that view's depth (points farther than the other view's
  surface by ``occlusion_m`` are treated as occluded and excluded; closer-than-surface disagreements count, since a
  point in front of an observed surface contradicts it);
* robot-geometry alignment: the calibrated, articulated URDF surface (recorded joints) rendered with a z-buffer
  and compared with the measured depth where the robot is the front-most surface (bounded by URDF/mount accuracy and
  biased toward the raw depth on frames the calibration fitted - report frames not used by the calibration separately);
* static-surface stability: per-pixel temporal spread on pixels away from the robot (dilated FK render) and away from
  the manipulated object (SAM 3 object mask);
* boundaries: flying-pixel fraction (a pixel between its horizontal or vertical neighbours' depths and differing from
  both by > ``rel`` of its depth) and the distance from depth discontinuities to the nearest RGB edge;
* regions: valid fraction / consistency inside the SAM 3 object (mug) and fixture (mug tree) masks.
Smoother depth is not evidence of accuracy; the report states which comparisons favour which source by construction.
"""

from __future__ import annotations

import numpy as np

from s4d.realworld.geometry import depth_to_points, project, transform_points


def reproject_compare(depth_a, valid_a, K_a, T_a, depth_b, valid_b, K_b, T_b, *, occlusion_m: float = 0.05,
                      max_points: int = 20000, rng=None, mask_a=None) -> np.ndarray:
    """Signed differences z_b(measured) - z(reprojected from a) at the pixels of b hit by a's valid points, excluding
    points hidden behind b's surface (difference < -occlusion_m means a's point is behind -> excluded)."""
    rng = rng or np.random.default_rng(0)
    pts, ok = depth_to_points(np.where(valid_a, depth_a, 0.0), K_a)
    sel = ok if mask_a is None else ok & mask_a
    P = pts[sel]
    if len(P) > max_points:
        P = P[rng.choice(len(P), max_points, replace=False)]
    if not len(P):
        return np.zeros(0, np.float32)
    world = transform_points(np.linalg.inv(T_a), P)
    uv, z = project(transform_points(T_b, world), K_b)
    H, W = depth_b.shape
    j, i = np.floor(uv[:, 0]).astype(int), np.floor(uv[:, 1]).astype(int)
    inside = np.isfinite(uv).all(1) & (z > 0) & (i >= 0) & (i < H) & (j >= 0) & (j < W)
    zb = np.zeros(len(z))
    vb = np.zeros(len(z), bool)
    zb[inside], vb[inside] = depth_b[i[inside], j[inside]], valid_b[i[inside], j[inside]]
    d = zb - z
    keep = inside & vb & (d > -occlusion_m)  # measured surface not in front of the point by more than the tolerance
    return d[keep].astype(np.float32)


def robot_zbuffer(surface_cam: np.ndarray, K: np.ndarray, hw, block: int = 2) -> np.ndarray:
    """(H,W) nearest robot-surface depth (inf where the robot does not project), at ``block``-pixel splats."""
    H, W = hw
    uv, z = project(surface_cam, K)
    ok = np.isfinite(uv).all(1) & (z > 0)
    out = np.full(hw, np.inf)
    for dy in range(block):
        for dx in range(block):
            j = np.floor(uv[ok, 0] + dx - (block - 1) / 2).astype(int)
            i = np.floor(uv[ok, 1] + dy - (block - 1) / 2).astype(int)
            m = (i >= 0) & (i < H) & (j >= 0) & (j < W)
            np.minimum.at(out, (i[m], j[m]), z[ok][m])
    return out


def robot_alignment(depth, valid, zbuf, *, front_tol_m: float = 0.05) -> np.ndarray:
    """Signed measured - rendered depth where the robot renders and the measurement is not far in front of it
    (an object between camera and robot, e.g. the mug, is an occluder, not an error)."""
    m = np.isfinite(zbuf) & valid
    d = depth - zbuf
    return d[m & (d > -front_tol_m)].astype(np.float32)


def flying_pixel_fraction(depth, valid, rel: float = 0.05) -> float:
    """Fraction of valid pixels lying strictly between their two horizontal (or vertical) neighbours' depths while
    differing from both by more than ``rel`` of their own depth (mixed-depth pixels at discontinuities)."""
    flags = np.zeros(depth.shape, bool)
    for axis in (0, 1):
        a = np.roll(depth, 1, axis)
        b = np.roll(depth, -1, axis)
        va = np.roll(valid, 1, axis)
        vb = np.roll(valid, -1, axis)
        lo, hi = np.minimum(a, b), np.maximum(a, b)
        tol = rel * depth
        between = (depth > lo + tol) & (depth < hi - tol)
        flags |= valid & va & vb & between
    flags[0, :] = flags[-1, :] = flags[:, 0] = flags[:, -1] = False
    n = valid.sum()
    return float(flags.sum() / n) if n else float("nan")


def _edges_from_gray(gray: np.ndarray, frac: float = 0.15) -> np.ndarray:
    gy, gx = np.gradient(gray.astype(np.float32))
    mag = np.hypot(gx, gy)
    return mag > frac * (np.percentile(mag, 99) + 1e-6)


def depth_edges(depth, valid, rel: float = 0.05) -> np.ndarray:
    """Depth discontinuities: neighbouring valid pixels differing by more than ``rel`` of the nearer depth."""
    e = np.zeros(depth.shape, bool)
    for axis in (0, 1):
        n = np.roll(depth, 1, axis)
        vn = np.roll(valid, 1, axis)
        jump = valid & vn & (np.abs(depth - n) > rel * np.minimum(depth, n))
        e |= jump
    e[0, :] = e[:, 0] = False
    return e


def edge_to_rgb_distance(depth, valid, rgb, *, max_px: int = 20) -> np.ndarray:
    """Distance (px) from each depth discontinuity to the nearest RGB edge (capped at ``max_px``)."""
    from scipy.ndimage import distance_transform_edt

    rgb_e = _edges_from_gray(rgb.astype(np.float32).mean(-1))
    d_e = depth_edges(depth, valid)
    if not d_e.any() or not rgb_e.any():
        return np.zeros(0, np.float32)
    dist = distance_transform_edt(~rgb_e)
    return np.minimum(dist[d_e], max_px).astype(np.float32)


def summarize(x: np.ndarray, tols=(0.01, 0.02, 0.05), unit: str = "mm", scale: float = 1000.0) -> dict:
    """Median/mean/p90 of |x| and the fractions within each tolerance (keys ``within_<tol*scale><unit>``)."""
    x = np.asarray(x, np.float64)
    if not len(x):
        return {"n": 0}
    a = np.abs(x)
    out = {"n": int(len(x)), "median_abs": float(np.median(a)), "mean_signed": float(x.mean()),
           "p90_abs": float(np.percentile(a, 90))}
    for t in tols:
        out[f"within_{int(round(t * scale))}{unit}"] = float((a <= t).mean())
    return out


# ---------------------------------------------------------------- v2: common-support comparison (2026-11-05)


def reproject_pair(points_a: np.ndarray, T_a, T_b, K_b, depth_b, valid_b, *, occlusion_m: float = 0.05):
    """Camera-a points (N,3) -> (signed z_b(measured) - z(reprojected) (N,), kept (N,), occluded (N,)). ``kept``: lands
    inside b on a valid pixel and is not hidden behind b's surface by more than ``occlusion_m``."""
    world = transform_points(np.linalg.inv(T_a), points_a)
    uv, z = project(transform_points(T_b, world), K_b)
    H, W = depth_b.shape
    ok = np.isfinite(uv).all(1) & (z > 0)
    j = np.where(ok, np.floor(np.nan_to_num(uv[:, 0])), -1).astype(int)
    i = np.where(ok, np.floor(np.nan_to_num(uv[:, 1])), -1).astype(int)
    inside = ok & (i >= 0) & (i < H) & (j >= 0) & (j < W)
    zb, vb = np.zeros(len(z)), np.zeros(len(z), bool)
    zb[inside], vb[inside] = depth_b[i[inside], j[inside]], valid_b[i[inside], j[inside]]
    d = zb - np.where(np.isfinite(z), z, 0.0)
    occluded = inside & vb & (d <= -occlusion_m)
    return d.astype(np.float32), inside & vb & ~occluded, occluded


def rgb_boundary(rgb: np.ndarray, px: int = 3) -> np.ndarray:
    """Source-independent boundary band: pixels within ``px`` of an RGB edge."""
    from scipy.ndimage import binary_dilation

    return binary_dilation(_edges_from_gray(rgb.astype(np.float32).mean(-1)), iterations=px)


def edge_shift_scan(depth, valid, rgb, *, radius: int = 3, max_px: int = 20) -> dict:
    """Mean capped distance from depth discontinuities to the nearest RGB edge when the depth map is shifted by
    (dx, dy) in [-radius, radius]^2: a registration offset shows as a minimum away from (0, 0)."""
    from scipy.ndimage import distance_transform_edt

    rgb_e = _edges_from_gray(rgb.astype(np.float32).mean(-1))
    d_e = depth_edges(depth, valid)
    if not d_e.any() or not rgb_e.any():
        return {}
    dist = np.minimum(distance_transform_edt(~rgb_e), max_px)
    ii, jj = np.nonzero(d_e)
    H, W = depth.shape
    scores = {}
    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            i2, j2 = ii + dy, jj + dx
            m = (i2 >= 0) & (i2 < H) & (j2 >= 0) & (j2 < W)
            scores[(dx, dy)] = float(dist[i2[m], j2[m]].mean()) if m.any() else float("inf")
    best = min(scores, key=scores.get)
    return {"best_shift_px": list(best), "at_best": scores[best], "at_zero": scores[(0, 0)]}
