from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch
import torch.nn.functional as F

try:
    from pytorch3d.loss import chamfer_distance as pytorch3d_chamfer_distance
except ImportError:  # pragma: no cover - exercised only when the dependency is missing.
    pytorch3d_chamfer_distance = None

from models.splatter import SplatterConfig
from models.splatter_train_config import TrainConfig


def sample_points(points: torch.Tensor, mask: torch.Tensor, max_points: Optional[int]) -> Tuple[torch.Tensor, torch.Tensor]:
    valid = mask.to(dtype=torch.bool) & torch.isfinite(points).all(dim=-1)
    if not bool(valid.any()):
        return points.new_zeros((1, 3)), torch.zeros((1,), device=points.device, dtype=torch.bool)
    selected = points[valid]
    if max_points is not None and int(max_points) > 0 and selected.shape[0] > int(max_points):
        perm = torch.randperm(selected.shape[0], device=points.device)[: int(max_points)]
        selected = selected[perm]
    return selected, torch.ones((selected.shape[0],), device=points.device, dtype=torch.bool)


def depths_to_world_point_cloud(
    depths: Optional[torch.Tensor],
    intrinsics: torch.Tensor,
    c2w: torch.Tensor,
    splatter_cfg: SplatterConfig,
    masks: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    if depths is None:
        device = intrinsics.device
        return torch.zeros((intrinsics.shape[0], 1, 3), device=device), torch.zeros((intrinsics.shape[0], 1), device=device, dtype=torch.bool)

    bsz, num_views, _, height, width = depths.shape
    device = depths.device
    dtype = depths.dtype
    if masks is not None:
        masks = masks.to(device=device, dtype=torch.bool)

    ys, xs = torch.meshgrid(
        torch.arange(height, device=device, dtype=dtype) + 0.5,
        torch.arange(width, device=device, dtype=dtype) + 0.5,
        indexing="ij",
    )
    xs = xs.view(1, 1, height, width)
    ys = ys.view(1, 1, height, width)

    point_lists: list[torch.Tensor] = []
    mask_lists: list[torch.Tensor] = []
    znear = float(splatter_cfg.data.znear)
    zfar = float(splatter_cfg.data.zfar)
    for batch_idx in range(bsz):
        per_view_points = []
        per_view_valid = []
        for view_idx in range(num_views):
            depth = depths[batch_idx, view_idx, 0].to(dtype=dtype)
            k = intrinsics[batch_idx, view_idx].to(device=device, dtype=dtype)
            z = depth
            valid = torch.isfinite(z) & (z >= znear) & (z <= zfar)
            if masks is not None:
                valid = valid & masks[batch_idx, view_idx, 0]
            x = (xs[0, 0] - k[0, 2]) / k[0, 0].clamp_min(1.0e-6) * z
            y = (ys[0, 0] - k[1, 2]) / k[1, 1].clamp_min(1.0e-6) * z
            pts_cam = torch.stack([x, y, z], dim=-1).reshape(-1, 3)
            rot = c2w[batch_idx, view_idx, :3, :3].to(dtype=dtype)
            trans = c2w[batch_idx, view_idx, :3, 3].to(dtype=dtype)
            pts_world = pts_cam @ rot.transpose(0, 1) + trans.view(1, 3)
            per_view_points.append(pts_world)
            per_view_valid.append(valid.reshape(-1))
        all_points = torch.cat(per_view_points, dim=0)
        all_valid = torch.cat(per_view_valid, dim=0)
        valid_points = all_points[all_valid]
        if valid_points.numel() == 0:
            valid_points = all_points.new_zeros((1, 3))
            valid_mask = torch.zeros((1,), device=device, dtype=torch.bool)
        else:
            valid_mask = torch.ones((valid_points.shape[0],), device=device, dtype=torch.bool)
        point_lists.append(valid_points)
        mask_lists.append(valid_mask)

    max_count = max(points.shape[0] for points in point_lists)
    padded_points = []
    padded_masks = []
    for points, mask in zip(point_lists, mask_lists):
        pad_len = max_count - points.shape[0]
        if pad_len > 0:
            points = torch.cat([points, points.new_zeros((pad_len, 3))], dim=0)
            mask = torch.cat([mask, torch.zeros((pad_len,), device=device, dtype=torch.bool)], dim=0)
        padded_points.append(points)
        padded_masks.append(mask)
    return torch.stack(padded_points, dim=0), torch.stack(padded_masks, dim=0)


def _masked_padded_points(
    points: torch.Tensor,
    mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    sampled_points = []
    lengths = []
    weights = []
    for batch_idx in range(points.shape[0]):
        valid = mask[batch_idx].to(dtype=torch.bool) & torch.isfinite(points[batch_idx]).all(dim=-1)
        selected = points[batch_idx][valid]
        is_valid = bool(selected.numel() > 0)
        if not is_valid:
            selected = points.new_zeros((1, points.shape[-1]))
        sampled_points.append(selected)
        lengths.append(int(selected.shape[0]))
        weights.append(1.0 if is_valid else 0.0)

    max_len = max(1, max(lengths))
    padded = []
    for selected in sampled_points:
        pad_len = max_len - selected.shape[0]
        if pad_len > 0:
            selected = torch.cat([selected, selected.new_zeros((pad_len, selected.shape[-1]))], dim=0)
        padded.append(selected)

    return (
        torch.stack(padded, dim=0),
        torch.tensor(lengths, device=points.device, dtype=torch.long),
        torch.tensor(weights, device=points.device, dtype=points.dtype),
    )


def _masked_huber_reduce(
    squared_distances: torch.Tensor,
    lengths: torch.Tensor,
    huber_delta: float,
) -> torch.Tensor:
    max_points = squared_distances.shape[1]
    valid = torch.arange(max_points, device=squared_distances.device).view(1, -1) < lengths.view(-1, 1)
    distances = torch.sqrt(squared_distances.clamp_min(0.0) + 1.0e-12)
    per_point = F.smooth_l1_loss(
        distances,
        torch.zeros_like(distances),
        beta=float(huber_delta),
        reduction="none",
    )
    per_point = torch.where(valid, per_point, torch.zeros_like(per_point))
    return per_point.sum(dim=1) / lengths.clamp_min(1).to(per_point.dtype)


def huber_chamfer_loss(
    pred_points: torch.Tensor,
    pred_mask: torch.Tensor,
    gt_points: torch.Tensor,
    gt_mask: torch.Tensor,
    huber_delta: float,
) -> torch.Tensor:
    if pytorch3d_chamfer_distance is None:
        raise ImportError(
            "PyTorch3D is required for Chamfer loss. Install pytorch3d or sync the project dependencies."
        )

    pred_padded, pred_lengths, pred_weights = _masked_padded_points(pred_points, pred_mask)
    gt_padded, gt_lengths, gt_weights = _masked_padded_points(gt_points, gt_mask)
    weights = pred_weights * gt_weights
    if not bool((weights > 0).any()):
        return pred_points.new_zeros(())

    chamfer_terms, _ = pytorch3d_chamfer_distance(
        pred_padded,
        gt_padded,
        x_lengths=pred_lengths,
        y_lengths=gt_lengths,
        weights=weights,
        batch_reduction=None,
        point_reduction=None,
        norm=2,
    )
    pred_to_gt, gt_to_pred = chamfer_terms
    per_batch = 0.5 * (
        _masked_huber_reduce(pred_to_gt, pred_lengths, huber_delta)
        + _masked_huber_reduce(gt_to_pred, gt_lengths, huber_delta)
    )
    return (per_batch * weights).sum() / weights.sum().clamp_min(1.0e-8)


def _zero_point_stats(device: torch.device) -> Dict[str, torch.Tensor]:
    zero = torch.zeros((), device=device)
    return {
        "point_chamfer_loss": zero,
        "gt_point_count_mean": zero,
    }


def compute_point_losses(
    pc: Dict[str, torch.Tensor],
    depths: Optional[torch.Tensor],
    intrinsics: torch.Tensor,
    c2w: torch.Tensor,
    splatter_cfg: SplatterConfig,
    cfg_train: TrainConfig,
    masks: Optional[torch.Tensor] = None,
) -> Dict[str, torch.Tensor]:
    device = pc["xyz"].device
    if depths is None:
        return _zero_point_stats(device)

    gt_points, gt_mask = depths_to_world_point_cloud(
        depths=depths,
        intrinsics=intrinsics,
        c2w=c2w,
        splatter_cfg=splatter_cfg,
        masks=masks,
    )
    pred_mask = pc.get("valid_mask", torch.ones(pc["xyz"].shape[:2], device=device, dtype=torch.bool))
    point_chamfer = huber_chamfer_loss(
        pred_points=pc["xyz"],
        pred_mask=pred_mask,
        gt_points=gt_points,
        gt_mask=gt_mask,
        huber_delta=float(cfg_train.chamfer_huber_delta),
    )
    return {
        "point_chamfer_loss": point_chamfer,
        "gt_point_count_mean": gt_mask.float().sum(dim=1).mean(),
    }
