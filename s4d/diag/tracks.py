"""Point-track panels: GT (green) vs predicted (red) trajectories of sampled t0 pixels, projected per camera."""

from __future__ import annotations

import numpy as np
import torch

from s4d.diag.panels import draw_line, draw_points, grid, to_uint8
from s4d.geometry import lift_depth, project

GT_COLOR = (40, 220, 60)
PRED_COLOR = (230, 50, 50)


def sample_track_pixels(
    score0: torch.Tensor, weight0: torch.Tensor, n_moving: int = 64, n_static: int = 16, seed: int = 0
) -> torch.Tensor:
    """Return (M,2) integer pixel (col,row) indices: moving (score>0.5) and static (score==0) with valid targets."""
    g = torch.Generator().manual_seed(seed)
    valid = weight0.squeeze(0) > 0
    out = []
    for mask, n in (((score0.squeeze(0) > 0.5) & valid, n_moving), ((score0.squeeze(0) == 0) & valid, n_static)):
        idx = mask.nonzero()
        if len(idx):
            sel = idx[torch.randperm(len(idx), generator=g)[:n]]
            out.append(torch.stack((sel[:, 1], sel[:, 0]), -1))
    return torch.cat(out) if out else torch.zeros(0, 2, dtype=torch.long)


def trajectories(batch: dict, out: dict, b: int, v: int, pixels: torch.Tensor) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """World trajectories (M,3,3) for GT and prediction, plus their projections (M,3,2) in camera v (static)."""
    depth0 = batch["depth"][b, 0, v, 0]
    K, c2w, w2c = batch["K"][b, v], batch["c2w"][b, v], batch["w2c"][b, v]
    xyz0 = lift_depth(depth0[None], K[None], c2w[None])[0]  # (H,W,3)
    cols, rows = pixels[:, 0], pixels[:, 1]
    X0 = xyz0[rows, cols]
    gt01 = batch["motion3d"][b, 0, v][:, rows, cols].T
    gt02 = batch["motion3d"][b, 2, v][:, rows, cols].T
    pr01 = out["pred_disp"][b, 0, v][:, rows, cols].T
    pr02 = out["pred_disp"][b, 2, v][:, rows, cols].T
    gt = torch.stack((X0, X0 + gt01, X0 + gt02), 1)
    pr = torch.stack((X0, X0 + pr01, X0 + pr02), 1)
    both = torch.cat((gt, pr), 0).reshape(1, -1, 3)
    uv, _ = project(both, K[None], w2c[None])
    uv = uv.reshape(2, len(pixels), 3, 2)
    return gt.numpy(), pr.numpy(), uv.numpy()


def track_panel(batch: dict, out: dict, b: int, v: int, pixels: torch.Tensor, target_v: int | None = None) -> np.ndarray:
    """Three columns (t0,t1,t2) of GT RGB with GT tracks (green) and predicted tracks (red) drawn so far."""
    gt, pr, uv = trajectories(batch, out, b, v, pixels)
    target_v = v if target_v is None else target_v
    if target_v != v:
        both = torch.from_numpy(np.concatenate((gt, pr), 0)).reshape(1, -1, 3)
        projected, depth = project(both, batch["K"][b, target_v][None], batch["w2c"][b, target_v][None])
        projected[depth <= 0] = float("nan")
        uv = projected.reshape(2, len(pixels), 3, 2).numpy()
    frames = []
    for t in range(3):
        img = to_uint8(batch["images"][b, t, target_v].float() / 255.0).copy()
        for m in range(len(pixels)):
            for s in range(t):
                draw_line(img, uv[0, m, s], uv[0, m, s + 1], GT_COLOR)
                draw_line(img, uv[1, m, s], uv[1, m, s + 1], PRED_COLOR)
            draw_points(img, uv[0, m, t : t + 1], GT_COLOR, radius=1)
            draw_points(img, uv[1, m, t : t + 1], PRED_COLOR, radius=0)
        frames.append(img)
    return grid(frames, ncol=3)
