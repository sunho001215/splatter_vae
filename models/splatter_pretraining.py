from __future__ import annotations

import math
import os
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

try:
    from pytorch3d.loss import chamfer_distance as pytorch3d_chamfer_distance
except ImportError:  # pragma: no cover - exercised only when the dependency is missing.
    pytorch3d_chamfer_distance = None
from torchvision.utils import make_grid

import wandb

from models.losses import (
    compute_all_camera_contrastive_losses,
    compute_latent_consistency_loss,
    compute_reconstruction_loss,
)
from models.point_voxel_gaussians import PointVoxelToGaussians
from models.splatter import SplatterConfig, render_predicted
from models.splatter_train_config import TrainConfig
from models.vae import SplatterVAE


def _normalize_lr_schedule(schedule: str) -> str:
    schedule = str(schedule or "constant").strip().lower().replace("-", "_")
    aliases = {
        "none": "constant",
        "off": "constant",
        "constant": "constant",
        "cosine": "warmup_cosine",
        "cosine_annealing": "warmup_cosine",
        "warmup_cosine": "warmup_cosine",
        "cosine_warmup": "warmup_cosine",
    }
    if schedule not in aliases:
        raise ValueError(f"Unknown lr_schedule={schedule!r}. Use one of: constant, warmup_cosine, cosine.")
    return aliases[schedule]


def _resolve_lr_total_steps(cfg_train: TrainConfig, train_dataloader: DataLoader) -> int:
    if cfg_train.lr_total_steps is not None:
        total_steps = int(cfg_train.lr_total_steps)
    elif cfg_train.max_global_steps is not None:
        total_steps = int(cfg_train.max_global_steps)
    else:
        try:
            total_steps = int(cfg_train.num_epochs) * len(train_dataloader)
        except TypeError:
            total_steps = int(cfg_train.lr_warmup_steps) + 1
    return max(1, total_steps)


def _compute_scheduled_lr(cfg_train: TrainConfig, global_step: int, total_steps: int) -> float:
    peak_lr = float(cfg_train.lr)
    schedule = _normalize_lr_schedule(cfg_train.lr_schedule)
    if schedule == "constant":
        return peak_lr

    min_lr = float(cfg_train.min_lr)
    if min_lr > peak_lr:
        raise ValueError(f"min_lr ({min_lr}) must be <= lr ({peak_lr}).")

    step = max(0, int(global_step))
    warmup_steps = max(0, int(cfg_train.lr_warmup_steps))
    if warmup_steps > 0 and step < warmup_steps:
        return peak_lr * float(step + 1) / float(warmup_steps)

    decay_steps = max(1, int(total_steps) - warmup_steps)
    progress = min(1.0, max(0.0, float(step - warmup_steps) / float(decay_steps)))
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return min_lr + (peak_lr - min_lr) * cosine


def _set_optimizer_lr(optimizer: torch.optim.Optimizer, lr: float) -> None:
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr


def _render_weight(cfg_train: TrainConfig, global_step: int) -> float:
    base = float(cfg_train.rec_weight)
    warmup = max(0, int(cfg_train.render_loss_warmup_steps))
    if warmup <= 0:
        return base
    return base * min(1.0, float(global_step + 1) / float(warmup))


def encode_all_camera_batch(
    vae: SplatterVAE,
    images: torch.Tensor,
) -> tuple[Dict[str, torch.Tensor], torch.Tensor, torch.Tensor]:
    """Encode every RGB camera view while preserving the ``(B, camera_num)`` layout."""
    if images.dim() != 5:
        raise ValueError(f"Expected images as (B,A,3,H,W), got {tuple(images.shape)}.")

    bsz, num_views, channels, height, width = images.shape
    if channels != 3:
        raise ValueError(f"SplatterVAE encoders expect RGB inputs only, got {channels} channels.")
    flat_images = images.reshape(bsz * num_views, channels, height, width).contiguous()

    z_inv, inv_vq_loss, z_dep, dep_vq_loss, _ = vae.encode(flat_images)
    z_inv = z_inv.reshape(bsz, num_views, *z_inv.shape[1:]).contiguous()
    z_dep = z_dep.reshape(bsz, num_views, *z_dep.shape[1:]).contiguous()
    return {"z_inv": z_inv, "z_dep": z_dep}, inv_vq_loss, dep_vq_loss


def _sqrt_positive_part(x: torch.Tensor) -> torch.Tensor:
    ret = torch.zeros_like(x)
    positive = x > 0
    ret[positive] = torch.sqrt(x[positive])
    return ret


def _compute_soft_image_region_penalty(
    xyz_world: torch.Tensor,
    world_view_transform: torch.Tensor,
    intrinsics: torch.Tensor,
    img_h: int,
    img_w: int,
    min_depth: float = 1.0e-3,
    penalty_cap: float = 100.0,
    source_view_indices: Optional[torch.Tensor] = None,
    gaussian_mask: Optional[torch.Tensor] = None,
) -> Dict[str, torch.Tensor]:
    """Softly penalize Gaussian centers outside rendered camera frustums."""
    device = xyz_world.device
    xyz_world_f = xyz_world.float()
    world_view_f = world_view_transform.float()
    intrinsics_f = intrinsics.float()

    bsz, num_gaussians, _ = xyz_world_f.shape
    ones = torch.ones((bsz, num_gaussians, 1), device=device, dtype=xyz_world_f.dtype)
    xyz_world_h = torch.cat([xyz_world_f, ones], dim=-1)

    xyz_cam_h = torch.einsum("bvij,bnj->bvni", world_view_f, xyz_world_h)
    xyz_cam = xyz_cam_h[..., :3]
    x = xyz_cam[..., 0]
    y = xyz_cam[..., 1]
    z = xyz_cam[..., 2]

    min_z = max(float(min_depth), 1.0e-3)
    valid_depth = torch.isfinite(z) & (z > min_z)
    z_for_projection = torch.where(valid_depth, z, torch.ones_like(z))

    fx = intrinsics_f[..., 0, 0].unsqueeze(-1)
    fy = intrinsics_f[..., 1, 1].unsqueeze(-1)
    cx = intrinsics_f[..., 0, 2].unsqueeze(-1)
    cy = intrinsics_f[..., 1, 2].unsqueeze(-1)

    u = fx * (x / z_for_projection) + cx
    v = fy * (y / z_for_projection) + cy
    finite_projection = torch.isfinite(u) & torch.isfinite(v)

    u = torch.nan_to_num(u, nan=0.0, posinf=0.0, neginf=0.0)
    v = torch.nan_to_num(v, nan=0.0, posinf=0.0, neginf=0.0)
    max_u = float(max(img_w - 1, 1))
    max_v = float(max(img_h - 1, 1))

    image_penalty = F.relu(-u) / max_u + F.relu(u - max_u) / max_u
    image_penalty = image_penalty + F.relu(-v) / max_v + F.relu(v - max_v) / max_v
    image_penalty = torch.where(valid_depth & finite_projection, image_penalty, torch.zeros_like(image_penalty))

    z_clean = torch.nan_to_num(z, nan=-min_z, posinf=min_z, neginf=-min_z)
    depth_penalty = F.relu(min_z - z_clean) / min_z
    depth_penalty = torch.where(torch.isfinite(z), depth_penalty, torch.full_like(depth_penalty, float(penalty_cap)))

    per_view_penalty = (image_penalty + depth_penalty).clamp(max=float(penalty_cap))
    outside_mask = (~valid_depth) | (~finite_projection) | (u < 0.0) | (u > max_u) | (v < 0.0) | (v > max_v)

    if gaussian_mask is None:
        valid_gaussian_mask = torch.ones_like(outside_mask, dtype=torch.bool)
    else:
        valid_gaussian_mask = gaussian_mask.to(device=device, dtype=torch.bool).view(bsz, 1, num_gaussians)
        valid_gaussian_mask = valid_gaussian_mask.expand_as(outside_mask)

    def masked_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        if not bool(mask.any()):
            return values.new_zeros(())
        return values.masked_select(mask).mean()

    stats: Dict[str, torch.Tensor] = {
        "frustum_loss": masked_mean(per_view_penalty, valid_gaussian_mask),
        "inactive_ratio_mean": masked_mean(outside_mask.float(), valid_gaussian_mask),
        "invalid_depth_ratio_mean": masked_mean((~valid_depth).float(), valid_gaussian_mask),
        "nonfinite_projection_ratio_mean": masked_mean((~finite_projection).float(), valid_gaussian_mask),
    }

    if source_view_indices is not None:
        num_views = world_view_transform.shape[1]
        src_idx = source_view_indices.to(device=device, dtype=torch.long).view(-1, 1, 1)
        src_idx = src_idx.expand(-1, 1, outside_mask.shape[-1])
        source_outside = outside_mask.gather(dim=1, index=src_idx).squeeze(1)
        source_valid = valid_gaussian_mask.gather(dim=1, index=src_idx).squeeze(1)
        stats["inactive_ratio_src"] = masked_mean(source_outside.float(), source_valid)

        if num_views > 1:
            target_view_ids = torch.arange(num_views, device=device).view(1, num_views, 1)
            non_source_mask = target_view_ids != source_view_indices.to(device=device).view(-1, 1, 1)
            non_source_mask = non_source_mask.expand_as(outside_mask)
            stats["inactive_ratio_tgt"] = masked_mean(outside_mask.float(), non_source_mask & valid_gaussian_mask)
        else:
            stats["inactive_ratio_tgt"] = stats["inactive_ratio_src"]
    else:
        stats["inactive_ratio_src"] = masked_mean(outside_mask[:, 0].float(), valid_gaussian_mask[:, 0])
        stats["inactive_ratio_tgt"] = stats["inactive_ratio_src"]
    return stats



def _gather_camera_rows(values: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
    batch_ids = torch.arange(values.shape[0], device=values.device)
    return values[batch_ids, indices]


def _gather_target_cameras(values: torch.Tensor, target_indices: torch.Tensor) -> torch.Tensor:
    trailing_shape = values.shape[2:]
    gather_index = target_indices.view(
        target_indices.shape[0],
        target_indices.shape[1],
        *([1] * len(trailing_shape)),
    ).expand(target_indices.shape[0], target_indices.shape[1], *trailing_shape)
    return torch.gather(values, dim=1, index=gather_index)


def _target_indices_excluding_source(source_indices: torch.Tensor, num_views: int) -> torch.Tensor:
    if num_views < 2:
        raise ValueError("Source-plus-target reconstruction requires at least two camera viewpoints.")
    all_views = torch.arange(num_views, device=source_indices.device).view(1, num_views)
    all_views = all_views.expand(source_indices.shape[0], num_views)
    keep_target = all_views != source_indices.view(-1, 1)
    return all_views[keep_target].view(source_indices.shape[0], num_views - 1)


def _masked_mean(values: torch.Tensor, valid_mask: Optional[torch.Tensor]) -> torch.Tensor:
    if valid_mask is None:
        return values.mean()
    mask = valid_mask.to(device=values.device, dtype=torch.bool)
    while mask.ndim < values.ndim:
        mask = mask.unsqueeze(-1)
    mask = mask.expand_as(values)
    if not bool(mask.any()):
        return values.new_zeros(())
    return values.masked_select(mask).mean()


def _sample_points(points: torch.Tensor, mask: torch.Tensor, max_points: Optional[int]) -> Tuple[torch.Tensor, torch.Tensor]:
    valid = mask.to(dtype=torch.bool) & torch.isfinite(points).all(dim=-1)
    if not bool(valid.any()):
        return points.new_zeros((1, 3)), torch.zeros((1,), device=points.device, dtype=torch.bool)
    selected = points[valid]
    if max_points is not None and int(max_points) > 0 and selected.shape[0] > int(max_points):
        perm = torch.randperm(selected.shape[0], device=points.device)[: int(max_points)]
        selected = selected[perm]
    return selected, torch.ones((selected.shape[0],), device=points.device, dtype=torch.bool)


def _depths_to_world_point_cloud(
    depths: Optional[torch.Tensor],
    intrinsics: torch.Tensor,
    c2w: torch.Tensor,
    splatter_cfg: SplatterConfig,
) -> Tuple[torch.Tensor, torch.Tensor]:
    if depths is None:
        device = intrinsics.device
        return torch.zeros((intrinsics.shape[0], 1, 3), device=device), torch.zeros((intrinsics.shape[0], 1), device=device, dtype=torch.bool)

    bsz, num_views, _, height, width = depths.shape
    device = depths.device
    dtype = depths.dtype
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


def _compute_point_losses(
    pc: Dict[str, torch.Tensor],
    depths: Optional[torch.Tensor],
    intrinsics: torch.Tensor,
    c2w: torch.Tensor,
    splatter_cfg: SplatterConfig,
    cfg_train: TrainConfig,
) -> Dict[str, torch.Tensor]:
    device = pc["xyz"].device
    if depths is None:
        return _zero_point_stats(device)

    gt_points, gt_mask = _depths_to_world_point_cloud(
        depths=depths,
        intrinsics=intrinsics,
        c2w=c2w,
        splatter_cfg=splatter_cfg,
    )
    point_chamfer = huber_chamfer_loss(
        pred_points=pc["raw_points"],
        pred_mask=pc["raw_valid_mask"],
        gt_points=gt_points,
        gt_mask=gt_mask,
        huber_delta=float(cfg_train.chamfer_huber_delta),
    )
    return {
        "point_chamfer_loss": point_chamfer,
        "gt_point_count_mean": gt_mask.float().sum(dim=1).mean(),
    }


def _gaussian_stats(pc: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    valid = pc.get("valid_mask", None)
    voxel_valid = pc.get("voxel_valid_mask", None)
    device = pc["xyz"].device
    if valid is None:
        valid = torch.ones(pc["xyz"].shape[:2], device=device, dtype=torch.bool)
    if voxel_valid is None:
        voxel_valid = torch.ones(pc["voxel_centers"].shape[:2], device=device, dtype=torch.bool)

    opacity = pc["opacity"]
    mass = pc.get("voxel_mass", opacity.new_zeros((*voxel_valid.shape, 1)))
    coverage = pc.get("point_to_voxel_coverage", opacity.new_zeros((opacity.shape[0],)))
    stats = {
        "active_voxel_count": voxel_valid.float().sum(dim=1).mean(),
        "final_gaussian_count": valid.float().sum(dim=1).mean(),
        "mean_opacity": _masked_mean(opacity, valid),
        "valid_gaussian_ratio": valid.float().mean(),
        "voxel_mass_mean": _masked_mean(mass, voxel_valid),
        "voxel_mass_max": mass.masked_fill(~voxel_valid.unsqueeze(-1), 0.0).amax(),
        "voxel_mass_min": mass.masked_fill(~voxel_valid.unsqueeze(-1), float("inf")).amin().clamp_max(1.0e6),
        "voxel_mean_confidence": _masked_mean(pc.get("voxel_mean_confidence", mass.new_zeros(mass.shape)), voxel_valid),
        "voxel_point_count_mean": _masked_mean(pc.get("voxel_point_count", mass.new_zeros(mass.shape)), voxel_valid),
        "point_to_voxel_coverage": coverage.mean(),
        "raw_point_confidence_mean": _masked_mean(pc["raw_confidence"], pc["raw_valid_mask"]),
    }
    return stats


def _render_selected_sources_to_views(
    vae: SplatterVAE,
    point_voxel_to_gaussians: PointVoxelToGaussians,
    splatter_cfg: SplatterConfig,
    z_inv_source: torch.Tensor,
    z_dep_source: torch.Tensor,
    source_indices: torch.Tensor,
    target_indices: torch.Tensor,
    intrinsics: torch.Tensor,
    c2w: torch.Tensor,
    w2c: torch.Tensor,
    bg: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
    source_intrinsics = _gather_camera_rows(intrinsics, source_indices)
    source_c2w = _gather_camera_rows(c2w, source_indices)

    proposal_map = vae.decode(z_inv_source.contiguous(), z_dep_source.contiguous())
    gaussian_pc = point_voxel_to_gaussians(
        proposal_map=proposal_map,
        z_inv=z_inv_source,
        source_cameras_view_to_world=source_c2w,
        intrinsics=source_intrinsics,
        activate_output=True,
    )

    source_w2c = _gather_camera_rows(w2c, source_indices).unsqueeze(1)
    target_w2c = _gather_target_cameras(w2c, target_indices)
    render_w2c = torch.cat((source_w2c, target_w2c), dim=1)

    target_intrinsics = _gather_target_cameras(intrinsics, target_indices)
    render_intrinsics = torch.cat((source_intrinsics.unsqueeze(1), target_intrinsics), dim=1)

    out = render_predicted(
        pc=gaussian_pc,
        world_view_transform=render_w2c,
        intrinsics=render_intrinsics,
        bg_color=bg,
        cfg=splatter_cfg,
        render_mode="RGB",
    )
    source_view_indices = torch.zeros(source_indices.shape[0], device=source_indices.device, dtype=torch.long)
    valid_mask = gaussian_pc.get("valid_mask", None)
    stats = _compute_soft_image_region_penalty(
        xyz_world=gaussian_pc["xyz"],
        world_view_transform=render_w2c,
        intrinsics=render_intrinsics,
        img_h=splatter_cfg.data.img_height,
        img_w=splatter_cfg.data.img_width,
        min_depth=splatter_cfg.data.znear,
        source_view_indices=source_view_indices,
        gaussian_mask=valid_mask,
    )
    stats.update(_gaussian_stats(gaussian_pc))

    render_indices = torch.cat((source_indices.view(-1, 1), target_indices), dim=1)
    return out["render"], render_indices, gaussian_pc, stats


def compute_reconstruction_and_renders(
    vae: SplatterVAE,
    point_voxel_to_gaussians: PointVoxelToGaussians,
    splatter_cfg: SplatterConfig,
    images_01: torch.Tensor,
    z_inv: torch.Tensor,
    z_dep: torch.Tensor,
    intrinsics: torch.Tensor,
    c2w: torch.Tensor,
    w2c: torch.Tensor,
    bg: torch.Tensor,
    cfg_train: TrainConfig,
    depths: Optional[torch.Tensor] = None,
    return_renders: bool = False,
) -> Dict[str, Any]:
    """Decode one source view, render RGB, and supervise geometry with depth point clouds."""
    bsz, num_views = z_inv.shape[:2]
    if num_views < 2:
        raise ValueError("Source-plus-target reconstruction requires at least two camera viewpoints.")
    device = z_inv.device
    batch_ids = torch.arange(bsz, device=device)

    source_indices = torch.randint(low=0, high=num_views, size=(bsz,), device=device)
    target_indices = _target_indices_excluding_source(source_indices, num_views)
    z_inv_source = z_inv[batch_ids, source_indices]
    z_dep_source = z_dep[batch_ids, source_indices]

    rendered, render_indices, gaussian_pc, stats = _render_selected_sources_to_views(
        vae=vae,
        point_voxel_to_gaussians=point_voxel_to_gaussians,
        splatter_cfg=splatter_cfg,
        z_inv_source=z_inv_source,
        z_dep_source=z_dep_source,
        source_indices=source_indices,
        target_indices=target_indices,
        intrinsics=intrinsics,
        c2w=c2w,
        w2c=w2c,
        bg=bg,
    )

    target_images = _gather_target_cameras(images_01, render_indices)
    rec_loss = compute_reconstruction_loss(
        predicted=rendered.reshape(-1, *rendered.shape[2:]),
        ground_truth=target_images.reshape(-1, *target_images.shape[2:]),
        ssim_weight=float(cfg_train.ssim_weight),
    )
    point_stats = _compute_point_losses(
        pc=gaussian_pc,
        depths=depths,
        intrinsics=intrinsics,
        c2w=c2w,
        splatter_cfg=splatter_cfg,
        cfg_train=cfg_train,
    )

    out_dict: Dict[str, Any] = {
        "rec_loss": rec_loss,
        "rec_self": rec_loss,
        **stats,
        **point_stats,
    }

    if return_renders:
        out_dict["target_images_self"] = target_images
        out_dict["rendered_self"] = rendered
        out_dict["source_indices"] = source_indices.detach().cpu()
        out_dict["target_indices"] = target_indices.detach().cpu()
        out_dict["render_indices"] = render_indices.detach().cpu()
        out_dict["gaussian_pc"] = {k: v.detach() for k, v in gaussian_pc.items() if torch.is_tensor(v)}
        out_dict["source_c2w"] = _gather_camera_rows(c2w, source_indices).detach()
        out_dict["source_intrinsics"] = _gather_camera_rows(intrinsics, source_indices).detach()
    return out_dict


def _make_wandb_named_image_panel(named_images: list[tuple[str, torch.Tensor]], max_vis: int) -> wandb.Image:
    max_vis = max(1, int(max_vis))
    rows = []
    names = []
    for name, images in named_images:
        rows.append(images[:max_vis].detach().cpu().clamp(0.0, 1.0))
        names.append(name)
    grid = make_grid(torch.cat(rows, dim=0), nrow=max_vis, padding=2)
    return wandb.Image(grid, caption=" | ".join(names))


def _look_at_c2w_opencv(eye: torch.Tensor, target: torch.Tensor, up: torch.Tensor) -> torch.Tensor:
    z_axis = F.normalize(target - eye, dim=0, eps=1.0e-6)
    up = F.normalize(up, dim=0, eps=1.0e-6)
    y_down = F.normalize(-up, dim=0, eps=1.0e-6)
    x_axis = F.normalize(torch.cross(y_down, z_axis, dim=0), dim=0, eps=1.0e-6)
    y_axis = F.normalize(torch.cross(z_axis, x_axis, dim=0), dim=0, eps=1.0e-6)
    mat = torch.eye(4, device=eye.device, dtype=eye.dtype)
    mat[:3, 0] = x_axis
    mat[:3, 1] = y_axis
    mat[:3, 2] = z_axis
    mat[:3, 3] = eye
    return mat


def _invert_4x4_torch(mat: torch.Tensor) -> torch.Tensor:
    rot = mat[:3, :3]
    trans = mat[:3, 3]
    out = torch.eye(4, device=mat.device, dtype=mat.dtype)
    out[:3, :3] = rot.transpose(0, 1)
    out[:3, 3] = -(rot.transpose(0, 1) @ trans)
    return out


def _trajectory_w2c(
    base_c2w: torch.Tensor,
    trajectory: str,
    num_frames: int,
    amplitude: float,
    focus_distance: float,
) -> torch.Tensor:
    device = base_c2w.device
    dtype = base_c2w.dtype
    eye0 = base_c2w[:3, 3]
    right = F.normalize(base_c2w[:3, 0], dim=0, eps=1.0e-6)
    up = F.normalize(-base_c2w[:3, 1], dim=0, eps=1.0e-6)
    forward = F.normalize(base_c2w[:3, 2], dim=0, eps=1.0e-6)
    target = eye0 + forward * float(focus_distance)
    frames = []
    t_values = torch.linspace(-1.0, 1.0, steps=max(2, int(num_frames)), device=device, dtype=dtype)
    for t in t_values:
        if trajectory == "lateral":
            eye = eye0 + right * (float(amplitude) * t)
        elif trajectory == "circular":
            angle = t * math.pi
            eye = eye0 + right * (float(amplitude) * torch.sin(angle)) + up * (float(amplitude) * 0.5 * torch.cos(angle))
        else:
            raise ValueError(f"Unknown validation trajectory: {trajectory}")
        frames.append(_invert_4x4_torch(_look_at_c2w_opencv(eye, target, up)))
    return torch.stack(frames, dim=0)


def _render_validation_trajectory_videos(
    pc: Dict[str, torch.Tensor],
    source_c2w: torch.Tensor,
    source_intrinsics: torch.Tensor,
    splatter_cfg: SplatterConfig,
    bg: torch.Tensor,
    cfg_train: TrainConfig,
) -> Dict[str, wandb.Video]:
    if not bool(cfg_train.val_render_trajectory_videos):
        return {}
    if pc["xyz"].device.type != "cuda":
        return {}

    first_pc = {k: v[:1] for k, v in pc.items() if torch.is_tensor(v) and v.shape[0] == pc["xyz"].shape[0]}
    videos: Dict[str, wandb.Video] = {}
    for trajectory in ("lateral", "circular"):
        w2c = _trajectory_w2c(
            base_c2w=source_c2w[0],
            trajectory=trajectory,
            num_frames=int(cfg_train.val_video_frames),
            amplitude=float(cfg_train.val_video_amplitude),
            focus_distance=float(cfg_train.val_video_focus_distance),
        ).unsqueeze(0)
        intrinsics = source_intrinsics[0].view(1, 1, 3, 3).expand(1, w2c.shape[1], 3, 3)
        render = render_predicted(
            pc=first_pc,
            world_view_transform=w2c,
            intrinsics=intrinsics,
            bg_color=bg,
            cfg=splatter_cfg,
            render_mode="RGB",
        )["render"][0]
        video = (render.detach().cpu().clamp(0.0, 1.0).numpy() * 255.0).astype("uint8")
        videos[f"val/video_{trajectory}"] = wandb.Video(video, fps=int(cfg_train.val_video_fps), format="mp4")
    return videos


def _sample_object3d_array(
    points: torch.Tensor,
    mask: torch.Tensor,
    max_points: int,
    rgb: tuple[float, float, float],
) -> Optional[torch.Tensor]:
    selected, selected_mask = _sample_points(points, mask, max(1, int(max_points)))
    if not bool(selected_mask.any()):
        return None
    colors = selected.new_tensor(rgb).view(1, 3).expand(selected.shape[0], 3)
    return torch.cat([selected, colors], dim=-1).detach().cpu().float()


def _make_wandb_pointcloud_payload(
    rec_out: Dict[str, Any],
    depths: Optional[torch.Tensor],
    intrinsics: torch.Tensor,
    c2w: torch.Tensor,
    splatter_cfg: SplatterConfig,
    cfg_train: TrainConfig,
) -> Dict[str, wandb.Object3D]:
    if not bool(cfg_train.val_log_pointclouds):
        return {}
    max_points = max(1, int(cfg_train.val_pointcloud_max_points))
    pc = rec_out.get("gaussian_pc", None)
    if pc is None or "raw_points" not in pc:
        return {}

    pred = _sample_object3d_array(
        points=pc["raw_points"][0],
        mask=pc["raw_valid_mask"][0],
        max_points=max_points,
        rgb=(255.0, 80.0, 80.0),
    )
    payload: Dict[str, wandb.Object3D] = {}
    if pred is not None:
        payload["val/pointcloud_pred_raw"] = wandb.Object3D(
            pred.numpy(),
            caption=f"Predicted raw points, sampled to <= {max_points}",
        )

    if depths is None:
        return payload
    gt_points, gt_mask = _depths_to_world_point_cloud(
        depths=depths[:1],
        intrinsics=intrinsics[:1],
        c2w=c2w[:1],
        splatter_cfg=splatter_cfg,
    )
    gt = _sample_object3d_array(
        points=gt_points[0],
        mask=gt_mask[0],
        max_points=max_points,
        rgb=(80.0, 220.0, 120.0),
    )
    if gt is not None:
        payload["val/pointcloud_gt_depth"] = wandb.Object3D(
            gt.numpy(),
            caption=f"GT depth point cloud, sampled to <= {max_points}",
        )

    view_arrays = []
    view_palette = (
        (80.0, 220.0, 120.0),
        (80.0, 160.0, 255.0),
        (255.0, 210.0, 80.0),
        (220.0, 100.0, 255.0),
        (255.0, 140.0, 80.0),
        (80.0, 240.0, 240.0),
    )
    num_views = int(depths.shape[1])
    per_view_cap = max(1, max_points // max(1, num_views))
    for view_idx in range(num_views):
        view_points, view_mask = _depths_to_world_point_cloud(
            depths=depths[:1, view_idx: view_idx + 1],
            intrinsics=intrinsics[:1, view_idx: view_idx + 1],
            c2w=c2w[:1, view_idx: view_idx + 1],
            splatter_cfg=splatter_cfg,
        )
        view_arr = _sample_object3d_array(
            points=view_points[0],
            mask=view_mask[0],
            max_points=per_view_cap,
            rgb=view_palette[view_idx % len(view_palette)],
        )
        if view_arr is not None:
            view_arrays.append(view_arr)
    if view_arrays:
        payload["val/pointcloud_gt_depth_by_view"] = wandb.Object3D(
            torch.cat(view_arrays, dim=0).numpy(),
            caption=f"GT depth point cloud colored by camera view, sampled to <= {max_points} total",
        )

    if pred is not None and gt is not None:
        overlay = torch.cat([pred, gt], dim=0)
        payload["val/pointcloud_pred_gt_overlay"] = wandb.Object3D(
            overlay.numpy(),
            caption="Predicted raw points red; GT depth points green",
        )
    return payload


@torch.no_grad()
def validate_and_log_wandb(
    vae: SplatterVAE,
    splatter_cfg: SplatterConfig,
    point_voxel_to_gaussians: PointVoxelToGaussians,
    valid_dataloader: DataLoader,
    device: torch.device,
    bg: torch.Tensor,
    cfg_train: TrainConfig,
    global_step: int,
) -> None:
    if wandb.run is None:
        return

    prev_vae_mode = vae.training
    prev_converter_mode = point_voxel_to_gaussians.training
    vae.eval()
    point_voxel_to_gaussians.eval()

    scalar_keys = [
        "val/rec_loss",
        "val/inv_contrastive_loss",
        "val/inv_consistency_loss",
        "val/dep_contrastive_loss",
        "val/dep_consistency_loss",
        "val/frustum_loss",
        "val/point_chamfer_loss",
        "val/gt_point_count_mean",
        "val/active_voxel_count",
        "val/final_gaussian_count",
        "val/mean_opacity",
        "val/valid_gaussian_ratio",
        "val/voxel_mass_mean",
        "val/voxel_mass_max",
        "val/voxel_mass_min",
        "val/voxel_mean_confidence",
        "val/voxel_point_count_mean",
        "val/point_to_voxel_coverage",
        "val/raw_point_confidence_mean",
        "val/inactive_pct_mean",
        "val/inactive_pct_src",
        "val/inactive_pct_tgt",
        "val/invalid_depth_pct_mean",
        "val/nonfinite_projection_pct_mean",
    ]
    scalar_sums = {key: 0.0 for key in scalar_keys}
    num_eval_batches = 0
    image_payload: Dict[str, Any] = {}

    for batch_idx, batch in enumerate(valid_dataloader):
        if cfg_train.val_num_batches > 0 and batch_idx >= cfg_train.val_num_batches:
            break

        images = batch["images"].to(device, non_blocking=True)
        depths = batch.get("depths", None)
        if depths is not None:
            depths = depths.to(device, non_blocking=True)
        intrinsics = batch["K"].to(device, non_blocking=True)
        c2w = batch["c2w"].to(device, non_blocking=True)
        w2c = batch["w2c"].to(device, non_blocking=True)
        images_01 = (images + 1.0) * 0.5

        latents, _inv_vq_loss, _dep_vq_loss = encode_all_camera_batch(vae=vae, images=images)
        inv_contrastive_loss, dep_contrastive_loss = compute_all_camera_contrastive_losses(
            z_inv=latents["z_inv"],
            z_dep=latents["z_dep"],
            temperature=cfg_train.temperature,
        )
        inv_consistency_loss = compute_latent_consistency_loss(latents["z_inv"], mode="state")
        dep_consistency_loss = compute_latent_consistency_loss(latents["z_dep"], mode="view")

        rec_out = compute_reconstruction_and_renders(
            vae=vae,
            point_voxel_to_gaussians=point_voxel_to_gaussians,
            splatter_cfg=splatter_cfg,
            images_01=images_01,
            z_inv=latents["z_inv"],
            z_dep=latents["z_dep"],
            intrinsics=intrinsics,
            c2w=c2w,
            w2c=w2c,
            bg=bg,
            cfg_train=cfg_train,
            depths=depths,
            return_renders=(num_eval_batches == 0),
        )

        metric_map = {
            "val/rec_loss": rec_out["rec_loss"],
            "val/inv_contrastive_loss": inv_contrastive_loss,
            "val/inv_consistency_loss": inv_consistency_loss,
            "val/dep_contrastive_loss": dep_contrastive_loss,
            "val/dep_consistency_loss": dep_consistency_loss,
            "val/frustum_loss": rec_out["frustum_loss"],
            "val/point_chamfer_loss": rec_out["point_chamfer_loss"],
            "val/gt_point_count_mean": rec_out["gt_point_count_mean"],
            "val/active_voxel_count": rec_out["active_voxel_count"],
            "val/final_gaussian_count": rec_out["final_gaussian_count"],
            "val/mean_opacity": rec_out["mean_opacity"],
            "val/valid_gaussian_ratio": rec_out["valid_gaussian_ratio"],
            "val/voxel_mass_mean": rec_out["voxel_mass_mean"],
            "val/voxel_mass_max": rec_out["voxel_mass_max"],
            "val/voxel_mass_min": rec_out["voxel_mass_min"],
            "val/voxel_mean_confidence": rec_out["voxel_mean_confidence"],
            "val/voxel_point_count_mean": rec_out["voxel_point_count_mean"],
            "val/point_to_voxel_coverage": rec_out["point_to_voxel_coverage"],
            "val/raw_point_confidence_mean": rec_out["raw_point_confidence_mean"],
            "val/inactive_pct_mean": 100.0 * rec_out["inactive_ratio_mean"],
            "val/inactive_pct_src": 100.0 * rec_out["inactive_ratio_src"],
            "val/inactive_pct_tgt": 100.0 * rec_out["inactive_ratio_tgt"],
            "val/invalid_depth_pct_mean": 100.0 * rec_out["invalid_depth_ratio_mean"],
            "val/nonfinite_projection_pct_mean": 100.0 * rec_out["nonfinite_projection_ratio_mean"],
        }
        for key, value in metric_map.items():
            scalar_sums[key] += float(value.item())

        if num_eval_batches == 0:
            num_targets_to_show = min(rec_out["rendered_self"].shape[1], 6)
            panel_items: list[tuple[str, torch.Tensor]] = []
            for view_slot in range(num_targets_to_show):
                view_name = "source" if view_slot == 0 else f"target{view_slot}"
                panel_items.append((f"gt_{view_name}", rec_out["target_images_self"][:, view_slot]))
            for view_slot in range(num_targets_to_show):
                view_name = "source" if view_slot == 0 else f"target{view_slot}"
                panel_items.append((f"render_{view_name}", rec_out["rendered_self"][:, view_slot]))
            image_payload["val/render_summary"] = _make_wandb_named_image_panel(
                panel_items,
                max_vis=cfg_train.val_max_vis,
            )
            image_payload.update(
                _make_wandb_pointcloud_payload(
                    rec_out=rec_out,
                    depths=depths,
                    intrinsics=intrinsics,
                    c2w=c2w,
                    splatter_cfg=splatter_cfg,
                    cfg_train=cfg_train,
                )
            )
            image_payload.update(
                _render_validation_trajectory_videos(
                    pc=rec_out["gaussian_pc"],
                    source_c2w=rec_out["source_c2w"],
                    source_intrinsics=rec_out["source_intrinsics"],
                    splatter_cfg=splatter_cfg,
                    bg=bg,
                    cfg_train=cfg_train,
                )
            )

        num_eval_batches += 1

    if num_eval_batches == 0:
        vae.train(prev_vae_mode)
        point_voxel_to_gaussians.train(prev_converter_mode)
        return

    log_dict: Dict[str, Any] = {key: value / float(num_eval_batches) for key, value in scalar_sums.items()}
    log_dict["global_step"] = global_step
    log_dict.update(image_payload)
    wandb.log(log_dict, step=global_step)

    vae.train(prev_vae_mode)
    point_voxel_to_gaussians.train(prev_converter_mode)


def _build_converter(vae: SplatterVAE, splatter_cfg: SplatterConfig, device: torch.device) -> PointVoxelToGaussians:
    z_inv_dim = int(vae.invariant_encoder_output_proj.out_features)
    return PointVoxelToGaussians(splatter_cfg, z_inv_dim=z_inv_dim).to(device)


def train_splatter_vae(
    vae: SplatterVAE,
    splatter_cfg: SplatterConfig,
    train_dataloader: DataLoader,
    cfg_train: TrainConfig,
    valid_dataloader: Optional[DataLoader] = None,
    resume_ckpt: Optional[str] = None,
):
    """Train SplatterVAE with point-cloud geometry supervision and RGB rendering."""
    device = torch.device(cfg_train.device)
    vae.to(device)
    point_voxel_to_gaussians = _build_converter(vae, splatter_cfg, device)
    optimizer = torch.optim.Adam(
        list(vae.parameters()) + list(point_voxel_to_gaussians.parameters()),
        lr=cfg_train.lr,
    )

    lr_total_steps = _resolve_lr_total_steps(cfg_train, train_dataloader)
    lr_schedule = _normalize_lr_schedule(cfg_train.lr_schedule)
    if lr_schedule != "constant":
        print(
            f"[LR] schedule={lr_schedule}, peak_lr={cfg_train.lr:g}, min_lr={cfg_train.min_lr:g}, "
            f"warmup_steps={cfg_train.lr_warmup_steps}, total_steps={lr_total_steps}"
        )
    if int(cfg_train.render_loss_warmup_steps) > 0:
        print(f"[Render Loss] linear warmup over {cfg_train.render_loss_warmup_steps} steps")

    bg = torch.ones(3, device=device) if splatter_cfg.data.white_background else torch.zeros(3, device=device)
    start_epoch = 0
    global_step = 0

    if resume_ckpt is not None and os.path.isfile(resume_ckpt):
        ckpt = torch.load(resume_ckpt, map_location="cpu")
        vae.load_state_dict(ckpt["vae_state_dict"])
        converter_state = ckpt.get("point_voxel_to_gaussians_state_dict", ckpt.get("splatter_to_gaussians_state_dict", None))
        if converter_state is not None:
            try:
                point_voxel_to_gaussians.load_state_dict(converter_state, strict=True)
            except RuntimeError as exc:
                print(f"[Resume] Skipping converter state because architecture changed: {exc}")
        try:
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        except ValueError as exc:
            print(f"[Resume] Skipping optimizer state because parameter groups changed: {exc}")
        start_epoch = int(ckpt["epoch"])
        global_step = int(ckpt["global_step"])

    os.makedirs(cfg_train.ckpt_dir, exist_ok=True)

    epoch = start_epoch
    while True:
        for step, batch in enumerate(train_dataloader):
            if cfg_train.max_global_steps is not None and global_step >= cfg_train.max_global_steps:
                print(f"[Stop] Reached max_global_steps={cfg_train.max_global_steps}.")
                return

            current_lr = _compute_scheduled_lr(cfg_train, global_step, lr_total_steps)
            _set_optimizer_lr(optimizer, current_lr)
            vae.train()
            point_voxel_to_gaussians.train()

            images = batch["images"].to(device, non_blocking=True)
            depths = batch.get("depths", None)
            if depths is not None:
                depths = depths.to(device, non_blocking=True)
            intrinsics = batch["K"].to(device, non_blocking=True)
            c2w = batch["c2w"].to(device, non_blocking=True)
            w2c = batch["w2c"].to(device, non_blocking=True)
            images_01 = (images + 1.0) * 0.5

            optimizer.zero_grad(set_to_none=True)
            latents, inv_vq_loss, dep_vq_loss = encode_all_camera_batch(vae=vae, images=images)
            rec_out = compute_reconstruction_and_renders(
                vae=vae,
                point_voxel_to_gaussians=point_voxel_to_gaussians,
                splatter_cfg=splatter_cfg,
                images_01=images_01,
                z_inv=latents["z_inv"],
                z_dep=latents["z_dep"],
                intrinsics=intrinsics,
                c2w=c2w,
                w2c=w2c,
                bg=bg,
                cfg_train=cfg_train,
                depths=depths,
                return_renders=False,
            )
            rec_loss = rec_out["rec_loss"]
            point_chamfer_loss = rec_out["point_chamfer_loss"]
            frustum_loss = rec_out["frustum_loss"]

            inv_contrastive_loss, dep_contrastive_loss = compute_all_camera_contrastive_losses(
                z_inv=latents["z_inv"],
                z_dep=latents["z_dep"],
                temperature=cfg_train.temperature,
            )
            inv_consistency_loss = compute_latent_consistency_loss(latents["z_inv"], mode="state")
            dep_consistency_loss = compute_latent_consistency_loss(latents["z_dep"], mode="view")

            vq_loss = inv_vq_loss + dep_vq_loss
            rec_weight_effective = _render_weight(cfg_train, global_step)
            total_loss = (
                rec_weight_effective * rec_loss
                + cfg_train.point_chamfer_weight * point_chamfer_loss
                + cfg_train.vq_weight * vq_loss
                + cfg_train.inv_contrastive_weight * inv_contrastive_loss
                + cfg_train.inv_consistency_weight * inv_consistency_loss
                + cfg_train.dep_contrastive_weight * dep_contrastive_loss
                + cfg_train.dep_consistency_weight * dep_consistency_loss
                + cfg_train.frustum_weight * frustum_loss
            )

            finite_terms = {
                "rec_loss": rec_loss,
                "point_chamfer_loss": point_chamfer_loss,
                "vq_loss": vq_loss,
                "inv_contrastive_loss": inv_contrastive_loss,
                "inv_consistency_loss": inv_consistency_loss,
                "dep_contrastive_loss": dep_contrastive_loss,
                "dep_consistency_loss": dep_consistency_loss,
                "frustum_loss": frustum_loss,
                "total_loss": total_loss,
            }
            bad_terms = [name for name, value in finite_terms.items() if not torch.isfinite(value).all()]
            if bad_terms:
                print(
                    f"[Warn] Non-finite loss at global_step={global_step} "
                    f"(bad={bad_terms}, inactive_pct={100.0 * rec_out['inactive_ratio_mean'].item():.2f}, "
                    f"nonfinite_proj_pct={100.0 * rec_out['nonfinite_projection_ratio_mean'].item():.2f}). "
                    "Skipping optimizer step."
                )
                if wandb.run is not None:
                    wandb.log({"global_step": global_step, "train/nonfinite_batch": 1.0, "train/lr": current_lr}, step=global_step)
                global_step += 1
                continue

            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(list(vae.parameters()) + list(point_voxel_to_gaussians.parameters()), max_norm=5.0)
            optimizer.step()

            if step % 250 == 0:
                print(
                    f"[Epoch {epoch + 1} | Step {step} | Global {global_step}] "
                    f"Loss={total_loss.item():.4f} lr={current_lr:.2e} "
                    f"(rgb={rec_loss.item():.4f}, rgb_w={rec_weight_effective:.4f}, "
                    f"raw_chamfer={point_chamfer_loss.item():.4f}, "
                    f"vq={vq_loss.item():.4f}, inv_con={inv_contrastive_loss.item():.4f}, "
                    f"dep_con={dep_contrastive_loss.item():.4f}, frustum={frustum_loss.item():.4f}, "
                    f"active_vox={rec_out['active_voxel_count'].item():.1f}, gauss={rec_out['final_gaussian_count'].item():.1f})"
                )
                if wandb.run is not None:
                    wandb.log(
                        {
                            "train/total_loss": total_loss.item(),
                            "train/lr": current_lr,
                            "train/rec_loss": rec_loss.item(),
                            "train/rec_weight": float(cfg_train.rec_weight),
                            "train/rec_weight_effective": rec_weight_effective,
                            "train/rec_loss_weighted": rec_weight_effective * rec_loss.item(),
                            "train/point_chamfer_loss": point_chamfer_loss.item(),
                            "train/point_chamfer_loss_weighted": (cfg_train.point_chamfer_weight * point_chamfer_loss).item(),
                            "train/gt_point_count_mean": rec_out["gt_point_count_mean"].item(),
                            "train/vq_loss": vq_loss.item(),
                            "train/inv_vq_loss": inv_vq_loss.item(),
                            "train/dep_vq_loss": dep_vq_loss.item(),
                            "train/inv_contrastive_loss": inv_contrastive_loss.item(),
                            "train/inv_consistency_loss": inv_consistency_loss.item(),
                            "train/dep_contrastive_loss": dep_contrastive_loss.item(),
                            "train/dep_consistency_loss": dep_consistency_loss.item(),
                            "train/frustum_loss": frustum_loss.item(),
                            "train/frustum_loss_weighted": (cfg_train.frustum_weight * frustum_loss).item(),
                            "train/active_voxel_count": rec_out["active_voxel_count"].item(),
                            "train/final_gaussian_count": rec_out["final_gaussian_count"].item(),
                            "train/mean_opacity": rec_out["mean_opacity"].item(),
                            "train/valid_gaussian_ratio": rec_out["valid_gaussian_ratio"].item(),
                            "train/voxel_mass_mean": rec_out["voxel_mass_mean"].item(),
                            "train/voxel_mass_max": rec_out["voxel_mass_max"].item(),
                            "train/voxel_mass_min": rec_out["voxel_mass_min"].item(),
                            "train/voxel_mean_confidence": rec_out["voxel_mean_confidence"].item(),
                            "train/voxel_point_count_mean": rec_out["voxel_point_count_mean"].item(),
                            "train/point_to_voxel_coverage": rec_out["point_to_voxel_coverage"].item(),
                            "train/raw_point_confidence_mean": rec_out["raw_point_confidence_mean"].item(),
                            "train/inactive_pct_mean": 100.0 * rec_out["inactive_ratio_mean"].item(),
                            "train/inactive_pct_src": 100.0 * rec_out["inactive_ratio_src"].item(),
                            "train/inactive_pct_tgt": 100.0 * rec_out["inactive_ratio_tgt"].item(),
                            "train/invalid_depth_pct_mean": 100.0 * rec_out["invalid_depth_ratio_mean"].item(),
                            "train/nonfinite_projection_pct_mean": 100.0 * rec_out["nonfinite_projection_ratio_mean"].item(),
                            "train/point_chamfer_weight": float(cfg_train.point_chamfer_weight),
                            "train/frustum_weight": float(cfg_train.frustum_weight),
                            "global_step": global_step,
                        },
                        step=global_step,
                    )

            if valid_dataloader is not None and cfg_train.eval_every > 0 and global_step > 0 and global_step % cfg_train.eval_every == 0:
                validate_and_log_wandb(
                    vae=vae,
                    splatter_cfg=splatter_cfg,
                    point_voxel_to_gaussians=point_voxel_to_gaussians,
                    valid_dataloader=valid_dataloader,
                    device=device,
                    bg=bg,
                    cfg_train=cfg_train,
                    global_step=global_step,
                )

            if cfg_train.save_every > 0 and global_step > 0 and global_step % cfg_train.save_every == 0:
                ckpt_path = os.path.join(cfg_train.ckpt_dir, f"step_{global_step:08d}.pth")
                ckpt = {
                    "epoch": epoch,
                    "global_step": global_step,
                    "vae_state_dict": vae.state_dict(),
                    "point_voxel_to_gaussians_state_dict": point_voxel_to_gaussians.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                }
                torch.save(ckpt, ckpt_path)
                print(f"[Checkpoint] Saved checkpoint to {ckpt_path}")

            global_step += 1

        epoch += 1
        if cfg_train.max_global_steps is None and epoch >= cfg_train.num_epochs:
            print(f"[Stop] Reached num_epochs={cfg_train.num_epochs}.")
            break
