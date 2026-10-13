"""Image panels of 3D-MVP virtual views (uint8 grids; no plotting library)."""

from __future__ import annotations

import numpy as np
import torch

from s4d.baselines.mvp3d.renderer import VIEWS


def _upscale(tile: np.ndarray, scale: int) -> np.ndarray:
    return tile.repeat(scale, axis=0).repeat(scale, axis=1)


def rgb_tiles(rgb: torch.Tensor) -> list[np.ndarray]:
    """(5, 3, S, S) in [0, 1] -> five (S, S, 3) uint8 tiles."""
    return [(v.clamp(0, 1).permute(1, 2, 0).cpu().numpy() * 255).round().astype(np.uint8) for v in rgb]


def xyz_tiles(xyz: torch.Tensor, covered: torch.Tensor) -> list[np.ndarray]:
    """Cube coordinates (5, 3, S, S) in [-1, 1] shown as colour (x, y, z) -> (r, g, b); empty pixels black."""
    img = ((xyz + 1) / 2).clamp(0, 1) * covered[:, None]
    return [(v.permute(1, 2, 0).cpu().numpy() * 255).round().astype(np.uint8) for v in img]


def depth_tiles(depth: torch.Tensor, reference: torch.Tensor | None = None) -> list[np.ndarray]:
    """Normalised depth (5, S, S) (-1 = empty) as grey, near = bright. The grey scale spans each view's covered depth
    range of ``reference`` (default: ``depth``), so a prediction is shown on its ground truth's scale."""
    reference = depth if reference is None else reference
    tiles = []
    for d, r in zip(depth, reference):
        covered = r != -1
        lo, hi = (float(r[covered].min()), float(r[covered].max())) if covered.any() else (0.0, 1.0)
        span = max(hi - lo, 1e-6)
        v = 1.0 - (d - lo) / span
        shown = (d != -1) & (d > (lo - 1) / 2)  # a prediction nearer -1 (empty) than the nearest true depth is black
        v = (0.15 + 0.85 * v.clamp(0, 1)) * shown
        tiles.append(np.repeat((v.cpu().numpy()[..., None] * 255).round().astype(np.uint8), 3, axis=2))
    return tiles


def grid(rows: list[list[np.ndarray]], scale: int = 4, pad: int = 2) -> np.ndarray:
    """Rows of equally sized tiles -> one uint8 image with white separators."""
    h, w = rows[0][0].shape[:2]
    H, W = h * scale, w * scale
    out = np.full((len(rows) * (H + pad) + pad, len(rows[0]) * (W + pad) + pad, 3), 255, np.uint8)
    for i, row in enumerate(rows):
        for j, tile in enumerate(row):
            y, x = pad + i * (H + pad), pad + j * (W + pad)
            out[y : y + H, x : x + W] = _upscale(tile, scale)
    return out


def views_panel(images: torch.Tensor, scale: int = 4) -> np.ndarray:
    """One sample's virtual images (5, 10, S, S) -> rows RGB, depth, xyz; columns = the five views."""
    covered = images[:, 6] != -1
    return grid([rgb_tiles(images[:, 3:6]), depth_tiles(images[:, 6]), xyz_tiles(images[:, :3], covered)], scale)


def reconstruction_panel(images: torch.Tensor, masked: torch.Tensor, pred: torch.Tensor, scale: int = 4) -> np.ndarray:
    """Rows: RGB, masked RGB input, reconstructed RGB, depth, reconstructed depth (columns = views)."""
    return grid(
        [
            rgb_tiles(images[:, 3:6]),
            rgb_tiles(masked[:, 3:6]),
            rgb_tiles(pred[:, :3]),
            depth_tiles(images[:, 6]),
            depth_tiles(pred[:, 3], reference=images[:, 6]),
        ],
        scale,
    )


VIEW_LABELS = list(VIEWS)
