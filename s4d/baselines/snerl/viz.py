"""Sanity-panel helpers for SNeRL: semantic class colours, depth colouring, DINO-feature PCA, labelled tiles."""

from __future__ import annotations

import numpy as np
from PIL import Image, ImageDraw

# background, robot, object, fixture, unused 5th output of the official head
CLASS_COLORS = np.array([(35, 35, 35), (60, 130, 240), (235, 70, 60), (70, 190, 110), (250, 215, 0)], dtype=np.uint8)
_DEPTH_STOPS = np.array([(48, 18, 59), (70, 120, 240), (30, 200, 180), (160, 230, 60), (250, 190, 40), (190, 30, 10)])


def class_image(classes: np.ndarray) -> np.ndarray:
    """(H, W) integer classes -> (H, W, 3) uint8."""
    return CLASS_COLORS[np.clip(classes, 0, len(CLASS_COLORS) - 1)]


def depth_image(depth: np.ndarray, lo: float, hi: float, invalid: np.ndarray | None = None) -> np.ndarray:
    """Metric depth -> colours on a fixed [lo, hi] range (near = dark blue); ``invalid`` pixels black."""
    x = np.clip((np.asarray(depth, dtype=np.float64) - lo) / max(hi - lo, 1e-9), 0.0, 1.0) * (len(_DEPTH_STOPS) - 1)
    i = np.minimum(x.astype(np.int64), len(_DEPTH_STOPS) - 2)
    frac = (x - i)[..., None]
    out = (_DEPTH_STOPS[i] * (1 - frac) + _DEPTH_STOPS[i + 1] * frac).astype(np.uint8)
    if invalid is not None:
        out[invalid] = 0
    return out


def dino_pca(reference: np.ndarray, *others: np.ndarray) -> list[np.ndarray]:
    """Project (H, W, D) features on the top-3 principal components of ``reference``; colours are scaled by the
    reference's 1st-99th percentiles, so the other maps are directly comparable."""
    flat = reference.reshape(-1, reference.shape[-1]).astype(np.float64)
    mean = flat.mean(0)
    _, _, vt = np.linalg.svd(flat - mean, full_matrices=False)
    basis = vt[:3].T
    ref = (flat - mean) @ basis
    lo, hi = np.percentile(ref, 1, axis=0), np.percentile(ref, 99, axis=0)
    out = []
    for feats in (reference, *others):
        proj = (feats.reshape(-1, feats.shape[-1]).astype(np.float64) - mean) @ basis
        rgb = np.clip((proj - lo) / np.maximum(hi - lo, 1e-9), 0.0, 1.0)
        out.append((rgb.reshape(*feats.shape[:2], 3) * 255).astype(np.uint8))
    return out


def tile(image: np.ndarray, text: str, scale: int = 1, band: int = 14) -> np.ndarray:
    """Upscale ``image`` (nearest) and put ``text`` in a band above it."""
    img = Image.fromarray(np.asarray(image, dtype=np.uint8))
    if scale != 1:
        img = img.resize((img.width * scale, img.height * scale), Image.NEAREST)
    canvas = Image.new("RGB", (img.width, img.height + band), (255, 255, 255))
    canvas.paste(img, (0, band))
    ImageDraw.Draw(canvas).text((2, 1), text, fill=(0, 0, 0))
    return np.asarray(canvas)


def grid(rows: list[list[np.ndarray]], pad: int = 2) -> np.ndarray:
    """Rows of equally sized tiles -> one uint8 image with white padding."""
    h, w = rows[0][0].shape[:2]
    cols = max(len(r) for r in rows)
    out = np.full((len(rows) * (h + pad) + pad, cols * (w + pad) + pad, 3), 255, dtype=np.uint8)
    for r, row in enumerate(rows):
        for c, img in enumerate(row):
            out[pad + r * (h + pad): pad + r * (h + pad) + h, pad + c * (w + pad): pad + c * (w + pad) + w] = img
    return out
