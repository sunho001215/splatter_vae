"""Actual released PointWorld clip schema and deterministic source-grid targets."""

from __future__ import annotations

import numpy as np

PAIR_TIMES = ((0, 1), (1, 2), (0, 2))


def clip_windows(clip_key: str) -> list[tuple[int, int, int]]:
    start, end = map(int, clip_key.split(":"))
    if start < 0 or end <= start:
        raise ValueError(f"invalid half-open clip {clip_key!r}")
    return [(i, i + 3, i + 6) for i in range(start, end - 6)]


def nearest_timestamps(timestamps: np.ndarray, targets: np.ndarray) -> np.ndarray:
    stamps = np.asarray(timestamps, dtype=np.int64)
    targets = np.asarray(targets, dtype=np.int64)
    if stamps.ndim != 1 or len(stamps) == 0 or np.any(np.diff(stamps) < 0):
        raise ValueError("camera timestamps must be a nonempty sorted vector")
    right = np.searchsorted(stamps, targets).clip(0, len(stamps) - 1)
    left = (right - 1).clip(0, len(stamps) - 1)
    # Ties choose the earlier camera frame, deterministically.
    return np.where(np.abs(stamps[left] - targets) <= np.abs(stamps[right] - targets), left, right)


def resize_intrinsics(K: np.ndarray, width: int, height: int) -> np.ndarray:
    out = np.asarray(K, dtype=np.float32).copy()
    # Native integer centers become shared continuous centers before scaling.
    out[0, 2] += 0.5
    out[1, 2] += 0.5
    out[0] *= width / 320
    out[1] *= height / 180
    return out


def project(points: np.ndarray, K: np.ndarray, w2c: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    camera = np.asarray(points, dtype=np.float32) @ w2c[:3, :3].T + w2c[:3, 3]
    homogeneous = camera @ K.T
    uv = homogeneous[:, :2] / np.maximum(camera[:, 2:3], 1e-8)
    return uv, camera[:, 2]


def depth_test(
    points: np.ndarray,
    K: np.ndarray,
    w2c: np.ndarray,
    depth: np.ndarray,
    tolerance: float = 0.02,
    *,
    surface: bool = True,
    continuous: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """PW uses integer-centered image coordinates; snap with nearest pixel.

    surface=True requires a source-surface match within tolerance. The one-sided
    version removes occluded samples without suppressing in-front residuals.
    """
    uv, z = project(points, K, w2c)
    finite = np.isfinite(uv).all(-1) & np.isfinite(z)
    centers = uv - 0.5 if continuous else uv
    pixels = np.rint(np.where(np.isfinite(centers), centers, -1)).astype(np.int64)
    height, width = depth.shape
    inside = finite & (z > 0) & (pixels[:, 0] >= 0) & (pixels[:, 0] < width)
    inside &= (pixels[:, 1] >= 0) & (pixels[:, 1] < height)
    measured = np.zeros(len(points), dtype=np.float32)
    idx = np.flatnonzero(inside)
    measured[idx] = depth[pixels[idx, 1], pixels[idx, 0]]
    valid = inside & (measured > 0)
    valid &= np.abs(z - measured) <= tolerance if surface else z <= measured + tolerance
    return valid, pixels, z - measured


def nearest_z_winners(pixels: np.ndarray, z: np.ndarray, valid: np.ndarray, width: int) -> np.ndarray:
    idx = np.flatnonzero(valid)
    if not len(idx):
        return idx
    key = pixels[idx, 1] * width + pixels[idx, 0]
    order = np.lexsort((idx, z[idx], key))
    sorted_idx, sorted_key = idx[order], key[order]
    return sorted_idx[np.r_[True, sorted_key[1:] != sorted_key[:-1]]]


def sparse_targets(
    points: np.ndarray, visibility: np.ndarray, K: np.ndarray, w2c: np.ndarray, depths: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Splat positions' differences with weights one and nearest-z collisions.

    points=(3,N,3), visibility=(3,N), depths=(3,H,W). Scene and gripper points
    can be concatenated before this operation. No segmentation enters the loss.
    """
    height, width = depths.shape[-2:]
    motion = np.zeros((3, 3, height, width), np.float32)
    weight = np.zeros((3, 1, height, width), np.float32)
    score = np.zeros((3, 1, height, width), np.float32)
    maximum = np.stack([np.linalg.norm(points[b] - points[a], axis=-1) for a, b in PAIR_TIMES]).max(0)
    for t in range(3):
        valid, pix, _ = depth_test(points[t], K, w2c, depths[t], continuous=True)
        valid &= visibility[t] & np.isfinite(points).all(axis=(0, 2))
        _, z = project(points[t], K, w2c)
        winners = nearest_z_winners(pix, z, valid, width)
        score[t, 0, pix[winners, 1], pix[winners, 0]] = np.minimum(maximum[winners] / 0.03, 1)
    for pair, (source, target) in enumerate(PAIR_TIMES):
        valid, pix, _ = depth_test(points[source], K, w2c, depths[source], continuous=True)
        valid &= visibility[source] & np.isfinite(points[target]).all(-1)
        _, z = project(points[source], K, w2c)
        winners = nearest_z_winners(pix, z, valid, width)
        displacement = points[target, winners] - points[source, winners]
        motion[pair, :, pix[winners, 1], pix[winners, 0]] = displacement
        weight[pair, 0, pix[winners, 1], pix[winners, 0]] = 1
    return motion, weight, score
