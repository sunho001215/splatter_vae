"""Square crop + uniform resize of DROID's native 180x320 frames (profile ``configs/droid/vits_largescale.yaml``).

Geometry (the repository's continuous pixel convention, ``s4d/geometry.py``: pixel ``(i, j)`` covers ``[i, i+1) x
[j, j+1)``, centre ``(i + 0.5, j + 0.5)``). A crop is an integer box ``(x0, y0, s)`` on the native grid; ``x0``/``y0``
may be negative and ``x0 + s`` / ``y0 + s`` may exceed the image (padding). Output pixel ``(i, j)`` of the ``S x S``
result (``S`` = ``out_size``, 224) covers native ``[x0 + i s/S, x0 + (i+1) s/S)``. A native continuous point
``(u, v)`` maps to ``((u - x0) S/s, (v - y0) S/s)``, i.e. ``K' = A K`` with ``A = [[a, 0, -a x0], [0, a, -a y0],
[0, 0, 1]]``, ``a = S/s``: ``fx, fy, skew`` scale by ``a``, ``cx' = (cx - x0) a``, ``cy' = (cy - y0) a``. Metric depth,
world coordinates and extrinsics are unchanged.

Resampling:

* RGB: the box is cut from the image padded with ``PAD_RGB`` (0, all channels), then resized with PIL bilinear
  (antialiased when shrinking), whose sample positions follow the same continuous convention.
* ``observed`` (S, S) bool: an output pixel is observed iff the padding indicator resampled through exactly the RGB path
  is zero there, i.e. no padded native pixel contributes to it (the filter footprint lies inside the image).
* Depth: nearest native pixel of each output pixel centre (``floor(x0 + (i + 0.5) s/S)``); padding and every
  non-observed pixel is 0 (invalid). No depth is ever interpolated or fabricated.
* Token padding (border-patch rule): a ``p x p`` patch is padded iff ANY of its output pixels is not observed. Padded
  patches are forced-masked tokens and never encoder input.

Crop sampling (training, ``sample_crop`` policy "motion"; one crop per camera per window, applied to every history frame
and every target of that camera):

1. Reliable tracks: a track's displacement is measured only between frames where it is visible (visibility + depth
   test at the native grid); its motion ``m`` is the largest such world-frame 3D displacement (metres). Static
   external cameras: tracks are in the robot-base frame, so a static background never moves. The moving wrist camera
   uses the external cameras' world tracks projected into its own per-frame pose, so its ego-motion never counts.
2. Moving tracks: ``motion_min_m < m <= motion_max_m`` (larger jumps are track/depth-edge failures). Fewer than
   ``min_moving_tracks`` reliable tracks or moving tracks -> broad fallback ("few_reliable_tracks" / "weak_motion").
3. Extent: weighted (``min(m / 3 cm, 1)``) ``extent_quantiles`` of the moving tracks' native pixel positions over all
   visible history frames (the larger of its width and height), plus a context margin ``max(margin_min_px,
   margin_frac * extent)`` on every side. If extent + 2 margin exceeds ``side_max`` the motion is dispersed -> broad
   fallback ("dispersed").
4. Side ``s`` uniform integer in ``[s_lo, min(side_max, s_lo + zoom_span_px)]``, ``s_lo = max(side_min, required)``,
   ``required`` = the larger side of (extent + margin) clipped to the image (margin beyond the border cannot be shown,
   so it never forces padding).
5. Position, per axis: where ``s`` fits inside the image the crop is clamped inside it (no padding on that axis) and
   ``x0`` is uniform among the integer positions that contain the extent + margin clipped to the image (never empty:
   ``s`` >= the required side); where ``s`` exceeds the image extent (only the 180-pixel height, for s > 180) the whole
   axis is visible and only the padding split varies: centred on the moving extent plus a uniform jitter of
   ``jitter_frac * s``, clipped so no observed row is lost.
6. Broad fallback: ``s`` uniform in ``[broad_side_min, side_max]``, horizontal position uniform inside the image,
   vertical padding centred plus the same jitter.
7. Eligible-token guarantee: if a crop leaves fewer than ``min_eligible_per_frame`` unpadded tokens per frame, ``s``
   shrinks in steps of 8 (never needed with the profile's values: >= 84 of 196 per frame at s = 320; tested).

Evaluation / inference ("center"): side ``min(H, W)``, centred (``x0 = (W - s) // 2``), no padding, deterministic. The
validation split uses it, and ``inference_center_crop`` is the real-world contract (RealSense frame -> 224 x 224).
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, fields

import numpy as np

PAD_RGB = 0  # padding fill of every RGB channel (uint8); padded pixels are never supervision nor encoder input
MOTION_WEIGHT_SCALE_M = 0.03  # weight min(m / 3 cm, 1), the scale of the batch contract's motion score
SHRINK_STEP = 8
POLICIES = ("motion", "center")
FALLBACK_CODES = {  # crop_fallback field values
    None: 0,  # motion-informed crop
    "few_reliable_tracks": 1,
    "weak_motion": 2,
    "dispersed": 3,
    "no_geometry": 4,
    "center": 5,  # the deterministic evaluation / inference crop
}


@dataclass(frozen=True)
class CropSettings:
    out_size: int = 224
    side_min: int = 180
    side_max: int = 320
    motion_min_m: float = 0.02
    motion_max_m: float = 0.4
    min_moving_tracks: int = 32
    extent_quantiles: tuple[float, float] = (0.05, 0.95)
    margin_frac: float = 0.2
    margin_min_px: int = 24
    zoom_span_px: int = 80
    broad_side_min: int = 256
    jitter_frac: float = 0.1
    eval_policy: str = "center"

    @classmethod
    def from_config(cls, cfg: dict | None) -> CropSettings | None:
        if not cfg:
            return None
        known = {f.name for f in fields(cls)}
        unknown = sorted(set(cfg) - known)
        if unknown:
            raise ValueError(f"unknown data.crop keys {unknown}")
        values = dict(cfg)
        if "extent_quantiles" in values:
            values["extent_quantiles"] = tuple(float(q) for q in values["extent_quantiles"])
        out = cls(**values)
        if not (0 < out.side_min <= out.side_max and out.broad_side_min <= out.side_max and out.out_size > 0):
            raise ValueError("data.crop needs 0 < side_min <= side_max, broad_side_min <= side_max")
        if out.eval_policy not in POLICIES:
            raise ValueError(f"data.crop.eval_policy must be one of {POLICIES}")
        lo, hi = out.extent_quantiles
        if not 0.0 <= lo < hi <= 1.0:
            raise ValueError("data.crop.extent_quantiles must satisfy 0 <= lo < hi <= 1")
        return out

    def as_dict(self) -> dict:
        d = asdict(self)
        d["extent_quantiles"] = list(self.extent_quantiles)
        return d


@dataclass(frozen=True)
class CropBox:
    """Integer native-pixel box: top-left corner ``(x0, y0)`` (may be negative) and side ``s``."""

    x0: int
    y0: int
    side: int

    def scale(self, out_size: int) -> float:
        return out_size / self.side

    def padding(self, height: int, width: int) -> tuple[int, int, int, int]:
        """Padded native rows/columns (top, bottom, left, right)."""
        return (max(0, -self.y0), max(0, self.y0 + self.side - height), max(0, -self.x0),
                max(0, self.x0 + self.side - width))

    def as_array(self) -> np.ndarray:
        return np.array([self.x0, self.y0, self.side], np.float32)


# ------------------------------------------------------------------------------------------------ geometry
def crop_matrix(box: CropBox, out_size: int) -> np.ndarray:
    a = box.scale(out_size)
    return np.array([[a, 0.0, -a * box.x0], [0.0, a, -a * box.y0], [0.0, 0.0, 1.0]])


def crop_intrinsics(K, box: CropBox, out_size: int) -> np.ndarray:
    """Continuous-centre intrinsics of the cropped and resized image: ``A @ K``."""
    out = crop_matrix(box, out_size) @ np.asarray(K, dtype=np.float64)
    out[2] = (0.0, 0.0, 1.0)
    return out


def map_pixels(uv, box: CropBox, out_size: int) -> np.ndarray:
    """Native continuous pixel coordinates (..., 2) -> output continuous coordinates."""
    a = box.scale(out_size)
    return (np.asarray(uv, dtype=np.float64) - np.array([box.x0, box.y0], np.float64)) * a


def _cut(array: np.ndarray, box: CropBox, fill) -> np.ndarray:
    """The s x s native window of ``array`` (H, W[, C]), padding with ``fill`` outside the image."""
    H, W = array.shape[:2]
    top, bottom, left, right = box.padding(H, W)
    pad = ((top, bottom), (left, right)) + ((0, 0),) * (array.ndim - 2)
    padded = np.pad(array, pad, constant_values=fill) if (top or bottom or left or right) else array
    y, x = box.y0 + top, box.x0 + left
    return padded[y : y + box.side, x : x + box.side]


def _resize(window: np.ndarray, out_size: int, nearest: bool = False) -> np.ndarray:
    from PIL import Image

    if window.shape[:2] == (out_size, out_size):
        return window.copy()
    image = Image.fromarray(window)
    mode = Image.Resampling.NEAREST if nearest else Image.Resampling.BILINEAR
    return np.asarray(image.resize((out_size, out_size), mode)).copy()


def crop_rgb(image: np.ndarray, box: CropBox, out_size: int) -> np.ndarray:
    """(H, W, 3) uint8 -> (S, S, 3) uint8; padding = ``PAD_RGB``."""
    return _resize(np.ascontiguousarray(_cut(image, box, PAD_RGB)), out_size)


def observed_mask(box: CropBox, height: int, width: int, out_size: int) -> np.ndarray:
    """(S, S) bool: output pixels to which no padded native pixel contributes (same resampling as ``crop_rgb``)."""
    if box.padding(height, width) == (0, 0, 0, 0):
        return np.ones((out_size, out_size), bool)
    pad = _resize(np.ascontiguousarray(_cut(np.zeros((height, width), np.float32), box, 1.0)), out_size)
    return pad <= 1e-6


def crop_depth(depth: np.ndarray, box: CropBox, out_size: int, observed: np.ndarray | None = None) -> np.ndarray:
    """(H, W) metres -> (S, S): nearest native pixel of each output pixel centre; 0 outside the image and wherever
    ``observed`` is False."""
    H, W = depth.shape
    centres = (np.arange(out_size, dtype=np.float64) + 0.5) * (box.side / out_size)
    cols = np.floor(box.x0 + centres).astype(np.int64)
    rows = np.floor(box.y0 + centres).astype(np.int64)
    ok_c, ok_r = (cols >= 0) & (cols < W), (rows >= 0) & (rows < H)
    out = np.zeros((out_size, out_size), np.float32)
    out[np.ix_(ok_r, ok_c)] = depth[np.ix_(rows[ok_r], cols[ok_c])]
    if observed is not None:
        out[~observed] = 0.0
    return out


def token_padding(observed: np.ndarray, patch: int) -> np.ndarray:
    """(S, S) observed -> (N,) bool, row-major patches: True iff any pixel of the patch is not observed."""
    S = observed.shape[0]
    if S % patch or observed.shape[1] % patch:
        raise ValueError("the crop size must be a multiple of the patch size")
    g, h = S // patch, observed.shape[1] // patch
    return ~observed.reshape(g, patch, h, patch).all(axis=(1, 3)).reshape(-1)


def in_observed(uv_out: np.ndarray, observed: np.ndarray) -> np.ndarray:
    """Output continuous coordinates (..., 2) -> bool: the pixel containing the point exists and is observed."""
    S = observed.shape[0]
    finite = np.isfinite(uv_out).all(-1)
    pix = np.floor(np.where(finite[..., None], uv_out, -1.0)).astype(np.int64)
    inside = finite & (pix[..., 0] >= 0) & (pix[..., 0] < observed.shape[1]) & (pix[..., 1] >= 0) & (pix[..., 1] < S)
    out = np.zeros(inside.shape, bool)
    out[inside] = observed[pix[inside][:, 1], pix[inside][:, 0]]
    return out


# ------------------------------------------------------------------------------------------------ sampling
def center_box(height: int, width: int) -> CropBox:
    s = min(height, width)
    return CropBox((width - s) // 2, (height - s) // 2, s)


def _weighted_quantile(x: np.ndarray, w: np.ndarray, q: float) -> float:
    order = np.argsort(x, kind="stable")
    x, w = x[order], w[order]
    cdf = np.cumsum(w) - 0.5 * w
    return float(np.interp(q * w.sum(), cdf, x))


def _place(rng, lo: float, hi: float, margin: float, side: int, extent: int, jitter: float) -> int:
    """Top-left coordinate on one axis. ``side <= extent``: clamped inside the image, uniform among the positions that
    keep ``[lo - margin, hi + margin]`` clipped to the image (``side`` is at least its length). ``side > extent``: the
    whole axis is visible; the padding split follows the centre of ``[lo, hi]`` plus a uniform jitter of
    ``jitter * side``, clipped so no observed pixel is lost."""
    if side <= extent:
        need_lo, need_hi = max(0.0, lo - margin), min(float(extent), hi + margin)
        a, b = max(0.0, need_hi - side), min(float(extent - side), need_lo)
        ia, ib = math.ceil(a - 1e-9), math.floor(b + 1e-9)
        return int(rng.integers(ia, ib + 1)) if ia <= ib else int(round(min(max(0.5 * (a + b), 0), extent - side)))
    target = 0.5 * (lo + hi) - 0.5 * side + rng.uniform(-jitter, jitter) * side
    return int(round(min(0.0, max(float(extent - side), target))))


def track_motion(points: np.ndarray, visible: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """points (T, N, 3) world positions, visible (T, N) -> (motion m (N,) metres, reliable (N,) bool): the largest
    displacement between two frames where the track is visible; unreliable tracks have no such pair."""
    T = points.shape[0]
    finite = np.isfinite(points).all(-1) & visible
    motion = np.zeros(points.shape[1])
    reliable = np.zeros(points.shape[1], bool)
    for a in range(T):
        for b in range(a + 1, T):
            ok = finite[a] & finite[b]
            d = np.linalg.norm(np.where(ok[:, None], points[b] - points[a], 0.0), axis=-1)
            motion = np.where(ok, np.maximum(motion, d), motion)
            reliable |= ok
    return motion, reliable


def sample_crop(
    rng: np.random.Generator,
    settings: CropSettings,
    height: int,
    width: int,
    *,
    policy: str = "motion",
    uv: np.ndarray | None = None,
    visible: np.ndarray | None = None,
    motion: np.ndarray | None = None,
    reliable: np.ndarray | None = None,
    patch: int = 16,
    min_eligible_per_frame: int = 0,
) -> tuple[CropBox, dict]:
    """One crop for one camera and window. ``uv`` (T, N, 2) native continuous pixels of the tracks at each history
    frame, ``visible`` (T, N), ``motion`` / ``reliable`` (N,) from ``track_motion``. Returns the box and diagnostics."""
    if policy not in POLICIES:
        raise ValueError(f"crop policy must be one of {POLICIES}")
    info = {"policy": policy, "fallback": None, "reliable_tracks": 0, "moving_tracks": 0, "extent": None,
            "required_side": None, "shrunk": 0}
    if policy == "center":
        info["fallback"] = "center"
        return center_box(height, width), info
    jitter = settings.jitter_frac
    box = None
    if uv is None or visible is None or motion is None:
        info["fallback"] = "no_geometry"
    else:
        rel = reliable if reliable is not None else np.ones(motion.shape, bool)
        rel = rel & (motion <= settings.motion_max_m)
        moving = rel & (motion > settings.motion_min_m)
        info["reliable_tracks"], info["moving_tracks"] = int(rel.sum()), int(moving.sum())
        sel = visible[:, moving] & np.isfinite(uv[:, moving]).all(-1)
        pts = uv[:, moving][sel]
        inside = (pts[:, 0] >= 0) & (pts[:, 0] < width) & (pts[:, 1] >= 0) & (pts[:, 1] < height)
        w = np.broadcast_to(np.minimum(motion[moving] / MOTION_WEIGHT_SCALE_M, 1.0), sel.shape)[sel][inside]
        pts = pts[inside]
        if info["reliable_tracks"] < settings.min_moving_tracks:
            info["fallback"] = "few_reliable_tracks"
        elif info["moving_tracks"] < settings.min_moving_tracks or len(pts) < settings.min_moving_tracks:
            info["fallback"] = "weak_motion"
        else:
            q0, q1 = settings.extent_quantiles
            x_lo, x_hi = _weighted_quantile(pts[:, 0], w, q0), _weighted_quantile(pts[:, 0], w, q1)
            y_lo, y_hi = _weighted_quantile(pts[:, 1], w, q0), _weighted_quantile(pts[:, 1], w, q1)
            ext = max(x_hi - x_lo, y_hi - y_lo)
            margin = max(float(settings.margin_min_px), settings.margin_frac * ext)
            # dispersion: the extent with its context does not fit the largest crop; the side itself only has to hold
            # the part of extent + margin that exists in the image (no side growth for margin beyond the border)
            dispersion = int(math.ceil(ext + 2.0 * margin))
            required = int(math.ceil(max(min(width, x_hi + margin) - max(0.0, x_lo - margin),
                                         min(height, y_hi + margin) - max(0.0, y_lo - margin))))
            info["extent"] = [round(x_lo, 2), round(y_lo, 2), round(x_hi, 2), round(y_hi, 2)]
            info["extent_with_margin"], info["required_side"] = dispersion, required
            if dispersion > settings.side_max:
                info["fallback"] = "dispersed"
            else:
                s_lo = max(settings.side_min, required)
                s_hi = min(settings.side_max, s_lo + settings.zoom_span_px)
                side = int(rng.integers(s_lo, s_hi + 1))
                box = CropBox(_place(rng, x_lo, x_hi, margin, side, width, jitter),
                              _place(rng, y_lo, y_hi, margin, side, height, jitter), side)
                info["margin"] = round(margin, 2)
    if box is None:  # broad fallback: large side, uniform inside the image where it fits, centred padding elsewhere
        side = int(rng.integers(settings.broad_side_min, settings.side_max + 1))
        x0 = int(rng.integers(0, width - side + 1)) if side <= width else _place(rng, 0, width, 0, side, width, jitter)
        y0 = int(rng.integers(0, height - side + 1)) if side <= height else _place(rng, 0, height, 0, side, height,
                                                                                    jitter)
        box = CropBox(x0, y0, side)
    while min_eligible_per_frame and box.side > min(height, width):
        observed = observed_mask(box, height, width, settings.out_size)
        if (~token_padding(observed, patch)).sum() >= min_eligible_per_frame:
            break
        side = max(min(height, width), box.side - SHRINK_STEP)
        cx, cy = box.x0 + 0.5 * box.side, box.y0 + 0.5 * box.side
        box = CropBox(int(round(min(max(cx - 0.5 * side, min(0, width - side)), max(0, width - side)))),
                      int(round(min(max(cy - 0.5 * side, min(0, height - side)), max(0, height - side)))), side)
        info["shrunk"] += 1
    return box, info


def inference_center_crop(image: np.ndarray, K, out_size: int = 224) -> tuple[np.ndarray, np.ndarray, CropBox]:
    """Real-world inference contract: the central square crop (side ``min(H, W)``, no padding) of an RGB frame (H, W, 3)
    uint8, resized to ``out_size`` x ``out_size`` (PIL bilinear, antialiased), and its continuous-centre intrinsics.
    No tracking, no motion-dependent crop."""
    box = center_box(*image.shape[:2])
    return crop_rgb(image, box, out_size), crop_intrinsics(K, box, out_size), box
