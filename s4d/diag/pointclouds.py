"""3D point-cloud payloads: Gaussian centres (by RGB / by group / with motion vectors) and fused GT depth."""

from __future__ import annotations

import numpy as np
import torch
from PIL import Image, ImageDraw

from s4d.geometry import lift_depth
from s4d.model.gaussians import DYNAMIC_GROUP, GaussianSet

GROUP_COLORS = {0: (90, 160, 255), DYNAMIC_GROUP: (255, 90, 90)}


def gaussian_points(gs: GaussianSet, b: int, opacity_threshold: float = 0.05) -> dict[str, np.ndarray]:
    xyz = gs.xyz[b].detach().cpu()
    keep = gs.opacity[b].detach().cpu() > opacity_threshold
    rgb = gs.rgb[b].detach().cpu().clamp(0, 1) * 255
    group = gs.group.cpu()
    group_rgb = torch.tensor([GROUP_COLORS[int(g)] for g in group], dtype=torch.float32)
    by_rgb = torch.cat((xyz[keep], rgb[keep]), -1).numpy()
    by_group = torch.cat((xyz[keep], group_rgb[keep]), -1).numpy()
    dyn = keep & (group == DYNAMIC_GROUP)
    disp = gs.displacement(0, 2)[b].detach().cpu()
    idx = dyn.nonzero()[:, 0]
    if len(idx) > 200:
        idx = idx[torch.linspace(0, len(idx) - 1, 200).long()]
    vectors = torch.stack((xyz[idx], xyz[idx] + disp[idx]), 1).numpy()
    return {"by_rgb": by_rgb, "by_group": by_group, "vectors": vectors}


def fused_gt_points(batch: dict, b: int, t: int = 0, max_points: int = 20000) -> np.ndarray:
    depth = batch["depth"][b, t, :, 0]  # (V,H,W)
    xyz = lift_depth(depth, batch["K"][b], batch["c2w"][b])
    rgb = batch["images"][b, t].float().permute(0, 2, 3, 1)
    valid = depth > 0
    pts = torch.cat((xyz[valid], rgb[valid]), -1)
    if len(pts) > max_points:
        pts = pts[torch.linspace(0, len(pts) - 1, max_points).long()]
    return pts.numpy()


def cloud_panel(points: np.ndarray, vectors: np.ndarray | None = None, size: int = 256) -> np.ndarray:
    """Local PNG mirror with orthographic XY, XZ, YZ views and motion vectors."""
    from s4d.diag.panels import draw_line, draw_points, grid

    points = np.asarray(points)
    finite = np.isfinite(points).all(1)
    points = points[finite]
    if len(points) > 20000:
        points = points[np.linspace(0, len(points) - 1, 20000).astype(int)]
    if len(points):
        center = (points[:, :3].min(0) + points[:, :3].max(0)) * 0.5
        extent = max(np.ptp(points[:, :3], axis=0).max(), 0.01) * 1.05
    else:
        center, extent = np.zeros(3), 1.0
    frames = []
    for axes, label in (((0, 1), "XY"), ((0, 2), "XZ"), ((1, 2), "YZ")):
        image = np.full((size, size, 3), 245, dtype=np.uint8)
        coordinates = (points[:, axes] - center[list(axes)]) / extent * (size - 24) + size * 0.5
        coordinates[:, 1] = size - coordinates[:, 1]
        for xy, rgb in zip(coordinates, points[:, 3:6], strict=True):
            draw_points(image, xy[None], rgb.clip(0, 255).astype(np.uint8), radius=0)
        if vectors is not None:
            for vector in vectors:
                xy = (vector[:, axes] - center[list(axes)]) / extent * (size - 24) + size * 0.5
                xy[:, 1] = size - xy[:, 1]
                draw_line(image, xy[0], xy[1], (220, 40, 40))
        pil = Image.fromarray(image)
        ImageDraw.Draw(pil).text((8, 8), label, fill=(10, 10, 10))
        frames.append(np.asarray(pil))
    return grid(frames, ncol=3)
