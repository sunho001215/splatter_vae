"""Near-view held-out diagnostics (review items 1b/1c): point splatting oracle and Chamfer distances.

Distances are in metres. Every Chamfer statistic is reported per direction: ``p2g`` (prediction -> ground truth,
floaters) and ``g2p`` (ground truth -> prediction, missing surfaces), each as mean, median and 90th percentile.
"""

from __future__ import annotations

import torch

from s4d.geometry import lift_depth, project

GT_VOXEL_M = 0.005
CENTER_OPACITY = 0.3
MOVING_SCORE = 0.5
RENDER_ALPHA = 0.5
_CHUNK_ELEMENTS = 64 * 2**20  # distance-matrix entries per chunk (256 MB in float32)


def voxel_downsample(points: torch.Tensor, voxel: float = GT_VOXEL_M) -> torch.Tensor:
    """(N,3) -> one point (the voxel mean) per occupied voxel."""
    if len(points) == 0:
        return points
    keys = torch.floor(points / voxel).long()
    _, inverse = torch.unique(keys, dim=0, return_inverse=True)
    count = torch.zeros(int(inverse.max()) + 1, device=points.device).index_add_(0, inverse, torch.ones_like(points[:, 0]))
    total = torch.zeros(len(count), 3, device=points.device).index_add_(0, inverse, points.float())
    return total / count[:, None]


def nearest_distances(source: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """(N,3), (M,3) -> (N,) Euclidean distance from every source point to its nearest target point."""
    if len(source) == 0:
        return source.new_zeros(0)
    if len(target) == 0:
        return torch.full((len(source),), float("inf"), device=source.device)
    chunk = max(1, _CHUNK_ELEMENTS // len(target))
    return torch.cat([torch.cdist(s.float(), target.float()).amin(1) for s in source.split(chunk)])


def chamfer_stats(pred: torch.Tensor, gt: torch.Tensor) -> dict[str, float]:
    """{p2g,g2p}_{mean,p50,p90}; NaN when either cloud is empty."""
    out = {}
    for name, a, b in (("p2g", pred, gt), ("g2p", gt, pred)):
        if len(a) == 0 or len(b) == 0:
            out.update({f"{name}_{s}": float("nan") for s in ("mean", "p50", "p90")})
            continue
        d = nearest_distances(a, b)
        q = torch.quantile(d, torch.tensor([0.5, 0.9], device=d.device))
        out.update({f"{name}_mean": float(d.mean()), f"{name}_p50": float(q[0]), f"{name}_p90": float(q[1])})
    return out


def lift_valid(depth: torch.Tensor, K: torch.Tensor, c2w: torch.Tensor, far: float, mask: torch.Tensor | None = None):
    """depth (V,H,W) metres -> world points (N,3) of pixels with 0 < depth <= far (and ``mask``)."""
    xyz = lift_depth(depth, K, c2w)
    keep = (depth > 0) & (depth <= far)
    if mask is not None:
        keep = keep & mask
    return xyz[keep]


def splat(
    points: torch.Tensor, colors: torch.Tensor, K: torch.Tensor, w2c: torch.Tensor, height: int, width: int, near: float
) -> tuple[torch.Tensor, torch.Tensor]:
    """Z-buffered one-pixel splat of coloured points into V cameras.

    points (N,3), colors (N,3), K (V,3,3), w2c (V,4,4) -> rgb (V,3,H,W) and covered (V,1,H,W) bool.
    """
    V = len(K)
    uv, z = project(points[None].expand(V, -1, -1), K, w2c)  # (V,N,2), (V,N)
    u, v = torch.floor(uv[..., 0]).long(), torch.floor(uv[..., 1]).long()
    keep = (z > near) & (u >= 0) & (u < width) & (v >= 0) & (v < height)
    cam = torch.arange(V, device=points.device)[:, None].expand_as(u)
    index = (cam * height + v) * width + u
    index, depth = index[keep], z[keep]
    color = colors[None].expand(V, -1, -1)[keep]
    zbuf = torch.full((V * height * width,), float("inf"), device=points.device)
    zbuf.scatter_reduce_(0, index, depth, reduce="amin")
    front = depth <= zbuf[index]
    rgb = torch.zeros(V * height * width, 3, device=points.device)
    rgb[index[front]] = color[front].float()
    covered = torch.isfinite(zbuf)
    return rgb.view(V, height, width, 3).permute(0, 3, 1, 2), covered.view(V, 1, height, width)


def oracle_views(batch: dict, b: int, t: int, prefix: str, near: float, far: float):
    """Fuse the training cameras' GT depth+RGB at time t of sample b and splat it into the cameras of ``prefix``.

    Returns (rgb (V,3,H,W) in [0,1], covered (V,1,H,W)).
    """
    depth = batch["depth"][b, t, :, 0]
    colors = batch["images"][b, t].float().permute(0, 2, 3, 1) / 255.0  # (V,H,W,3)
    keep = (depth > 0) & (depth <= far)
    points = lift_depth(depth, batch["K"][b], batch["c2w"][b])[keep]
    H, W = depth.shape[-2:]
    return splat(points, colors[keep], batch[f"{prefix}_K"][b], batch[f"{prefix}_w2c"][b], H, W, near)


def gt_cloud_t0(batch: dict, b: int, far: float, prefixes=("eval", "near", "traj")) -> torch.Tensor:
    """Fused GT cloud at t0 from the training cameras and every held-out camera available in the batch."""
    clouds = [lift_valid(batch["depth"][b, 0, :, 0], batch["K"][b], batch["c2w"][b], far)]
    for prefix in prefixes:
        if f"{prefix}_depth" not in batch:
            continue
        c2w = batch.get(f"{prefix}_c2w")
        c2w = torch.linalg.inv(batch[f"{prefix}_w2c"][b]) if c2w is None else c2w[b]
        clouds.append(lift_valid(batch[f"{prefix}_depth"][b, 0, :, 0], batch[f"{prefix}_K"][b], c2w, far))
    return voxel_downsample(torch.cat(clouds))


def chamfer_metrics(batch: dict, gs, b: int, far: float, motion_pair: int = 2) -> dict[str, float]:
    """CD-centers (all and dynamic) and CD-motion (all and dynamic) for sample b; keys ``cd_<kind>_<dir>_<stat>``."""
    from s4d.model.gaussians import DYNAMIC_GROUP  # noqa: PLC0415

    out = {}
    opaque = gs.opacity[b] > CENTER_OPACITY
    dynamic = opaque & (gs.group == DYNAMIC_GROUP)
    centers = gs.xyz[b]
    gt = gt_cloud_t0(batch, b, far)
    out.update({f"cd_centers_{k}": v for k, v in chamfer_stats(centers[opaque], gt).items()})
    depth0 = batch["depth"][b, 0, :, 0]
    moving = batch["motion_score"][b, 0, :, 0] > MOVING_SCORE
    K, c2w = batch["K"][b], batch["c2w"][b]
    out.update(
        {
            f"cd_centers_dyn_{k}": v
            for k, v in chamfer_stats(centers[dynamic], voxel_downsample(lift_valid(depth0, K, c2w, far, moving))).items()
        }
    )
    # CD-motion: centres displaced by the predicted 0->2 motion vs GT t0 points displaced by the GT 0->2 motion
    moved = centers + gs.displacement(0, 2)[b]
    xyz0 = lift_depth(depth0, K, c2w)
    gt_disp = batch["motion3d"][b, motion_pair].permute(0, 2, 3, 1)  # (V,H,W,3) on the t0 grid
    valid = (depth0 > 0) & (depth0 <= far)
    gt_moved = (xyz0 + gt_disp)
    out.update({f"cd_motion_{k}": v for k, v in chamfer_stats(moved[opaque], voxel_downsample(gt_moved[valid])).items()})
    out.update(
        {
            f"cd_motion_dyn_{k}": v
            for k, v in chamfer_stats(moved[dynamic], voxel_downsample(gt_moved[valid & moving])).items()
        }
    )
    return out


def render_chamfer(render: dict, batch: dict, b: int, prefix: str, far: float) -> dict[str, float]:
    """CD-render for every camera of ``prefix`` at t0: rendered expected depth (alpha > 0.5) vs that camera's GT
    depth, averaged over cameras. ``render`` holds depth/alpha (B,T,V,1,H,W) for those cameras."""
    K, w2c = batch[f"{prefix}_K"][b], batch[f"{prefix}_w2c"][b]
    c2w = batch[f"{prefix}_c2w"][b] if f"{prefix}_c2w" in batch else torch.linalg.inv(w2c)
    rows = []
    for v in range(len(K)):
        pred_depth = render["depth"][b, 0, v, 0].float()
        pred = lift_valid(pred_depth, K[v], c2w[v], far, render["alpha"][b, 0, v, 0] > RENDER_ALPHA)
        gt = lift_valid(batch[f"{prefix}_depth"][b, 0, v, 0], K[v], c2w[v], far)
        rows.append(chamfer_stats(pred, gt))
    return {f"cd_render_{prefix}_{k}": float(torch.tensor([r[k] for r in rows]).nanmean()) for k in rows[0]}
