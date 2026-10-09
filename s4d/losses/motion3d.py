"""Gaussian-space 3D motion loss (M3D, directive item 2): GT track points supervise nearby dynamic Gaussians directly.

Per sample, view and pair (0->1 at t0, 1->2 at t1, 0->2 at t0), ``NUM_POINTS`` pixels with motion weight > 0 are
sampled (half among pixels whose target moves more than 5 mm when there are any, half uniformly) and lifted to the world
with the GT depth of the pair's source time. Displacement term: each point's k nearest dynamic Gaussians at the source
time within ``RADIUS_M`` get a Huber loss between their pair displacement and the point's target, weighted by a Gaussian
kernel (sigma = radius / 2, detached) and the point's motion weight. Attraction term: for moving points, the Huber
distance to the nearest dynamic centre at the source time (gradient to the centres). Both average over points.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from s4d.data.contract import PAIRS
from s4d.geometry import lift_depth
from s4d.losses.motion import MOVING_THRESHOLD_M, PAIR_WEIGHTS

NUM_POINTS = 256
NEIGHBOURS = 4
RADIUS_M = 0.03
HUBER_M = 0.01
ATTRACTION_WEIGHT = 1.0


def sample_track_points(
    weight: torch.Tensor, target: torch.Tensor, num_points: int = NUM_POINTS, generator: torch.Generator | None = None
) -> tuple[torch.Tensor, torch.Tensor]:
    """weight (R,H*W), target magnitude (R,H*W) -> pixel indices (R,num_points) and a row-valid mask (R,).

    The first half is drawn among moving pixels (|target| > 5 mm, weight > 0) where a row has any, else uniformly;
    the second half uniformly among pixels with weight > 0. Rows without such pixels are invalid.
    """
    valid = weight > 0
    moving = valid & (target > MOVING_THRESHOLD_M)
    has_valid, has_moving = valid.any(1), moving.any(1)
    uniform = torch.where(has_valid[:, None], valid.float(), torch.ones_like(weight))
    focus = torch.where(has_moving[:, None], moving.float(), uniform)
    half = num_points // 2
    first = torch.multinomial(focus, half, replacement=True, generator=generator)
    second = torch.multinomial(uniform, num_points - half, replacement=True, generator=generator)
    return torch.cat((first, second), 1), has_valid


def motion3d_loss(
    gs,
    xyz_seq: torch.Tensor,
    dynamic: torch.Tensor,
    depth: torch.Tensor,
    K: torch.Tensor,
    c2w: torch.Tensor,
    target: torch.Tensor,
    weight: torch.Tensor,
    num_points: int = NUM_POINTS,
    neighbours: int = NEIGHBOURS,
    radius: float = RADIUS_M,
    huber: float = HUBER_M,
    attraction_weight: float = ATTRACTION_WEIGHT,
    pair_weights=PAIR_WEIGHTS,
    generator: torch.Generator | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """xyz_seq (B,3,N,3) centres per time, dynamic (N,) bool, depth (B,T,V,1,H,W), cameras (B,V,…),
    target (B,P,V,3,H,W) and weight (B,P,V,1,H,W) as for the image-space loss. Returns the pair-weighted loss and
    per-pair metrics."""
    B, P, V, _, H, W = target.shape
    T = depth.shape[1]
    dyn_idx = torch.nonzero(dynamic, as_tuple=False)[:, 0]
    k = min(neighbours, len(dyn_idx))
    if k == 0:
        return xyz_seq.sum() * 0.0, {}
    sigma = radius / 2.0
    pair_disp = {(0, 1): gs.delta01, (1, 2): gs.delta12, (0, 2): gs.delta01 + gs.delta12}
    lifted = lift_depth(depth[:, :, :, 0].float(), K[:, None].expand(B, T, V, 3, 3), c2w[:, None].expand(B, T, V, 4, 4))
    total = xyz_seq.new_zeros(())
    metrics = {}
    for p, ((a, b, _), name) in enumerate(zip(PAIRS, ("01", "12", "02"))):
        tgt = target[:, p].float().permute(0, 1, 3, 4, 2).reshape(B * V, H * W, 3)
        w = weight[:, p, :, 0].float().reshape(B * V, H * W)
        idx, row_ok = sample_track_points(w, tgt.norm(dim=-1), num_points, generator)
        gather = lambda x: torch.gather(x, 1, idx[..., None].expand(-1, -1, x.shape[-1]))  # noqa: E731
        points = gather(lifted[:, a].reshape(B * V, H * W, 3)).view(B, V * num_points, 3)
        t_pts = gather(tgt).view(B, V * num_points, 3)
        w_pts = (torch.gather(w, 1, idx) * row_ok[:, None].float()).view(B, V * num_points)
        centres = xyz_seq[:, a][:, dyn_idx]  # (B,Nd,3)
        with torch.no_grad():
            dist, nn_idx = torch.cdist(points, centres.detach().float()).topk(k, dim=-1, largest=False)  # (B,M,k)
            kernel = torch.exp(-(dist**2) / (2 * sigma**2)) * (dist < radius).float()
        disp = pair_disp[(a, b)][:, dyn_idx].float()  # (B,Nd,3)
        nn_disp = torch.gather(disp, 1, nn_idx.reshape(B, -1)[..., None].expand(-1, -1, 3)).view(B, -1, k, 3)
        err = F.smooth_l1_loss(nn_disp, t_pts[:, :, None].expand_as(nn_disp), beta=huber, reduction="none").sum(-1)
        counted = (w_pts > 0).float()
        disp_term = ((kernel * err).sum(-1) * w_pts).sum() / counted.sum().clamp_min(1.0)
        moving = counted * (t_pts.norm(dim=-1) > MOVING_THRESHOLD_M).float()
        nearest = torch.gather(centres.float(), 1, nn_idx[..., 0].reshape(B, -1, 1).expand(-1, -1, 3))  # (B,M,3)
        gap = (nearest - points).norm(dim=-1)
        attr = F.smooth_l1_loss(gap, torch.zeros_like(gap), beta=huber, reduction="none")
        attr_term = (attr * moving).sum() / moving.sum().clamp_min(1.0)
        total = total + float(pair_weights[p]) * (disp_term + attraction_weight * attr_term)
        metrics[f"m3d_disp_{name}"] = disp_term.detach()
        metrics[f"m3d_attr_{name}"] = attr_term.detach()
        metrics[f"m3d_covered_{name}"] = ((kernel > 0).any(-1).float() * counted).sum().detach() / counted.sum().clamp_min(1.0)
        metrics[f"m3d_gap_moving_mm_{name}"] = (1000.0 * (gap.detach() * moving).sum() / moving.sum().clamp_min(1.0))
    return total, metrics
