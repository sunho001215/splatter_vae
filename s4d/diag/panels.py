"""Image panel helpers (numpy uint8 HxWx3). All tensors arrive as CPU float tensors."""

from __future__ import annotations

import math

import numpy as np
import torch

_TURBO = np.array(
    [
        [0.13572138, 4.61539260, -42.66032258, 132.13108234, -152.94239396, 59.28637943],
        [0.09140261, 2.19418839, 4.84296658, -14.18503333, 4.27729857, 2.82956604],
        [0.10667330, 12.64194608, -60.58204836, 110.36276771, -89.90310912, 27.34824973],
    ]
)


def to_uint8(img: torch.Tensor | np.ndarray) -> np.ndarray:
    """(3,H,W) or (H,W,3) float in [0,1] -> (H,W,3) uint8."""
    x = img.detach().cpu().numpy() if torch.is_tensor(img) else np.asarray(img)
    if x.ndim == 3 and x.shape[0] == 3:
        x = x.transpose(1, 2, 0)
    return (np.nan_to_num(x, nan=0.0).clip(0, 1) * 255).round().astype(np.uint8)


def turbo(values: np.ndarray) -> np.ndarray:
    """(H,W) in [0,1] -> (H,W,3) uint8 Turbo colormap."""
    x = np.clip(values, 0, 1)[..., None]
    powers = np.concatenate([x**k for k in range(6)], axis=-1)  # (H,W,6)
    rgb = np.einsum("cp,hwp->hwc", _TURBO, powers)
    return (rgb.clip(0, 1) * 255).astype(np.uint8)


def colorize_depth(depth: torch.Tensor, near: float, far: float, valid: torch.Tensor | None = None) -> np.ndarray:
    d = depth.detach().cpu().numpy().squeeze()
    img = turbo((d - near) / max(far - near, 1e-6))
    mask = (d > 0) if valid is None else valid.detach().cpu().numpy().squeeze().astype(bool)
    img[~mask] = 0
    return img


def colorize_error(error: torch.Tensor, scale: float) -> np.ndarray:
    e = error.detach().cpu().numpy().squeeze()
    return turbo(np.clip(e / scale, 0, 1))


def colorize_gray(values: torch.Tensor) -> np.ndarray:
    v = values.detach().cpu().numpy().squeeze().clip(0, 1)
    return np.repeat((v * 255).astype(np.uint8)[..., None], 3, axis=-1)


def flow_color(flow_uv: np.ndarray, max_mag: float) -> np.ndarray:
    """(H,W,2) image-plane displacement -> HSV-style color (hue=direction, saturation=magnitude)."""
    u, v = flow_uv[..., 0], flow_uv[..., 1]
    mag = np.sqrt(u * u + v * v)
    hue = (np.arctan2(v, u) / (2 * math.pi) + 1.0) % 1.0
    sat = np.clip(mag / max(max_mag, 1e-6), 0, 1)
    h6 = hue * 6.0
    i = np.floor(h6).astype(int) % 6
    f = h6 - np.floor(h6)
    p, q, t = 1 - sat, 1 - f * sat, 1 - (1 - f) * sat
    one = np.ones_like(sat)
    table = np.stack(
        [
            np.stack((one, t, p), -1),
            np.stack((q, one, p), -1),
            np.stack((p, one, t), -1),
            np.stack((p, q, one), -1),
            np.stack((t, p, one), -1),
            np.stack((one, p, q), -1),
        ],
        0,
    )
    rgb = np.take_along_axis(table, i[None, ..., None], axis=0)[0]
    return (rgb * 255).astype(np.uint8)


def overlay_patches(
    img: np.ndarray, visible: torch.Tensor, grid: tuple[int, int], color=(255, 40, 40), alpha=0.55
) -> np.ndarray:
    """Tint masked (invisible) patches. visible (N,) bool."""
    gh, gw = grid
    H, W = img.shape[:2]
    mask = (~visible.detach().cpu().bool()).view(gh, gw).numpy()
    mask = np.kron(mask, np.ones((H // gh, W // gw), dtype=bool))
    out = img.astype(np.float32)
    out[mask] = (1 - alpha) * out[mask] + alpha * np.array(color, dtype=np.float32)
    return out.astype(np.uint8)


def grid(images: list[np.ndarray], ncol: int, pad: int = 2) -> np.ndarray:
    H, W = images[0].shape[:2]
    nrow = math.ceil(len(images) / ncol)
    canvas = np.zeros((nrow * H + (nrow - 1) * pad, ncol * W + (ncol - 1) * pad, 3), dtype=np.uint8)
    for k, im in enumerate(images):
        r, c = divmod(k, ncol)
        canvas[r * (H + pad) : r * (H + pad) + H, c * (W + pad) : c * (W + pad) + W] = im
    return canvas


def draw_points(img: np.ndarray, uv: np.ndarray, color, radius: int = 1) -> None:
    H, W = img.shape[:2]
    for u, v in uv:
        if not (np.isfinite(u) and np.isfinite(v)):
            continue
        x, y = int(round(u - 0.5)), int(round(v - 0.5))
        if x < -radius or x >= W + radius or y < -radius or y >= H + radius:
            continue
        img[max(0, y - radius) : min(H, y + radius + 1), max(0, x - radius) : min(W, x + radius + 1)] = color


def draw_line(img: np.ndarray, a: np.ndarray, b: np.ndarray, color) -> None:
    if not (np.isfinite(a).all() and np.isfinite(b).all()):
        return
    # Clip to the image first, avoiding enormous allocations for near-plane
    # or far-outside projections. Liang-Barsky segment clipping.
    H, W = img.shape[:2]
    delta = b - a
    lo, hi = 0.0, 1.0
    for coordinate, extent in ((0, W), (1, H)):
        for p, q in ((-delta[coordinate], a[coordinate]), (delta[coordinate], extent - a[coordinate])):
            if p == 0:
                if q < 0:
                    return
            elif p < 0:
                lo = max(lo, q / p)
            else:
                hi = min(hi, q / p)
    if lo > hi:
        return
    a, b = a + lo * delta, a + hi * delta
    n = int(max(2, np.abs(b - a).max() + 1))
    pts = a[None] + np.linspace(0, 1, n)[:, None] * (b - a)[None]
    draw_points(img, pts, color, radius=0)
