"""Rectified stereo -> metric depth, with explicit reference-image conventions.

Conventions (validated in tests/test_realworld_stereo.py on synthetic rigs):

* Inputs are a *rectified* pair (same K_rect up to the principal-point x offset, epipolar lines = pixel rows).
* Left-reference disparity ``d_L(x, y) = x_left - x_right`` lives on the left image grid; this is what
  FoundationStereo returns (``scripts/run_demo.py``: ``depth = K[0,0] * baseline / disp`` on the left image).
* Right-reference disparity ``d_R(x, y) = x_left - x_right`` (same sign, positive) lives on the right image grid.
  It is obtained with **flip-and-swap**: feed ``flip(right)`` as the left input and ``flip(left)`` as the right
  input, then flip the returned disparity back. Under the horizontal flip x' = W - x the flipped right image is
  a valid *left* view of the mirrored scene, and the disparity magnitude is preserved: x'_l - x'_r = x_l - x_r.
  Naively swapping inputs without flipping would ask the network for negative disparities and is wrong.
* Metric depth ``z = fx * B / (d - doffs)`` with ``doffs = cx_left - cx_right`` (0 for Bouguet-rectified pairs
  with a shared K_rect). The doffs term is the same for both references (it is invariant under flip-and-swap).
* Validity: finite, ``d - doffs > min_disparity`` and the matched pixel inside the other image
  (left reference: x - d >= 0; right reference: x + d <= W), mirroring FoundationStereo's ``remove_invisible``.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from scipy.ndimage import uniform_filter

from s4d.realworld.geometry import PixelAffine

Matcher = Callable[[np.ndarray, np.ndarray], np.ndarray]  # (left, right) -> left-reference disparity (H,W)


@dataclass
class DepthResult:
    reference: str  # "left" or "right"
    disparity: np.ndarray  # (H,W) float32, reference-image grid, x_left - x_right
    depth: np.ndarray  # (H,W) float32 metres camera-z, 0 invalid
    valid: np.ndarray  # (H,W) bool
    K: np.ndarray  # continuous-convention intrinsics of the reference image (after any resize)
    baseline_m: float
    doffs: float


def disparity_to_depth(disparity: np.ndarray, fx: float, baseline_m: float, doffs: float = 0.0,
                       min_disparity: float = 1e-3) -> tuple[np.ndarray, np.ndarray]:
    """Metric camera-z depth and validity. Never invents scale: baseline must be the real metric baseline."""
    if not (np.isfinite(baseline_m) and baseline_m > 0):
        raise ValueError("a positive metric baseline is required; stereo geometry missing => no metric depth")
    d = np.asarray(disparity, np.float64) - doffs
    valid = np.isfinite(d) & (d > min_disparity)
    depth = np.zeros(d.shape, np.float32)
    depth[valid] = (fx * baseline_m / d[valid]).astype(np.float32)
    return depth, valid


def visible_in_other(disparity: np.ndarray, reference: str) -> np.ndarray:
    """True where the corresponding pixel lies inside the other image (continuous pixel centres)."""
    H, W = disparity.shape
    x = np.arange(W)[None, :] + 0.5
    if reference == "left":
        return (x - disparity) >= 0
    if reference == "right":
        return (x + disparity) <= W
    raise ValueError(reference)


def flip_and_swap_right_disparity(left: np.ndarray, right: np.ndarray, matcher: Matcher) -> np.ndarray:
    """Right-reference disparity from any left-reference matcher (see module docstring)."""
    d_flipped = matcher(np.ascontiguousarray(right[:, ::-1]), np.ascontiguousarray(left[:, ::-1]))
    return np.ascontiguousarray(np.asarray(d_flipped)[:, ::-1])


def depth_from_pair(left: np.ndarray, right: np.ndarray, K_left: np.ndarray, K_right: np.ndarray, baseline_m: float,
                    matcher: Matcher, reference: str = "left", min_disparity: float = 1e-3) -> DepthResult:
    """Depth on the requested reference image of a rectified pair. K's are continuous-convention, same size."""
    K_left, K_right = np.asarray(K_left, float), np.asarray(K_right, float)
    if not (np.isclose(K_left[0, 0], K_right[0, 0]) and np.isclose(K_left[1, 1], K_right[1, 1])
            and np.isclose(K_left[1, 2], K_right[1, 2])):
        raise ValueError("pair is not rectified to a shared focal length / cy")
    if left.shape[:2] != right.shape[:2]:
        raise ValueError("left/right sizes differ")
    doffs = float(K_left[0, 2] - K_right[0, 2])
    if reference == "left":
        disp = np.asarray(matcher(left, right), np.float32)
        K = K_left
    elif reference == "right":
        disp = flip_and_swap_right_disparity(left, right, matcher).astype(np.float32)
        K = K_right
    else:
        raise ValueError("reference must be left or right")
    depth, valid = disparity_to_depth(disp, K[0, 0], baseline_m, doffs, min_disparity)
    valid &= visible_in_other(disp, reference)
    depth[~valid] = 0
    return DepthResult(reference, disp, depth, valid, K, float(baseline_m), doffs)


def rescale_disparity(disparity_model: np.ndarray, affine_to_model: PixelAffine) -> np.ndarray:
    """Disparity measured on a resized/cropped copy (``affine_to_model`` maps original -> model pixels) expressed in
    model pixels -> original pixels. Only the x scale matters; crops shift both eyes equally. Flip is excluded."""
    if affine_to_model.sx <= 0:
        raise ValueError("flips are handled by flip_and_swap_right_disparity, not here")
    return np.asarray(disparity_model, np.float32) / affine_to_model.sx


# ---------------------------------------------------------------- CPU reference matcher (tests / sanity only)


def block_match(left: np.ndarray, right: np.ndarray, max_disparity: int = 32, window: int = 5) -> np.ndarray:
    """Integer SAD block matching with parabolic sub-pixel refinement, left reference.

    A transparent CPU oracle for synthetic tests of the conventions above. It is NOT FoundationStereo and its
    output must never be described as model output.
    """
    L = np.asarray(left, np.float64)
    R = np.asarray(right, np.float64)
    if L.ndim == 3:
        L, R = L.mean(-1), R.mean(-1)
    H, W = L.shape
    cost = np.full((max_disparity + 1, H, W), np.inf)
    for d in range(max_disparity + 1):
        diff = np.full((H, W), np.inf)
        diff[:, d:] = np.abs(L[:, d:] - R[:, : W - d])
        filled = np.where(np.isfinite(diff), diff, 1e3)
        cost[d] = uniform_filter(filled, size=window, mode="nearest")
        cost[d][:, :d] = np.inf
    best = np.argmin(cost, axis=0)
    disp = best.astype(np.float64)
    inner = (best > 0) & (best < max_disparity)
    ii, jj = np.nonzero(inner)
    c0 = cost[best[ii, jj] - 1, ii, jj]
    c1 = cost[best[ii, jj], ii, jj]
    c2 = cost[best[ii, jj] + 1, ii, jj]
    denom = c0 - 2 * c1 + c2
    ok = np.isfinite(denom) & (denom > 1e-9)
    disp[ii[ok], jj[ok]] += 0.5 * (c0[ok] - c2[ok]) / denom[ok]
    disp[~np.isfinite(cost.min(0))] = np.nan
    return disp.astype(np.float32)
