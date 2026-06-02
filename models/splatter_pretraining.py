from __future__ import annotations

import math
import os
from typing import Any, Dict, Optional

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.utils import make_grid

import wandb

from models.losses import (
    compute_all_camera_contrastive_losses,
    compute_batched_reconstruction_losses,
    compute_latent_consistency_loss,
)
from models.splatter import SplatterConfig, VAESplatterToGaussians, render_predicted
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
        raise ValueError(
            f"Unknown lr_schedule={schedule!r}. "
            "Use one of: constant, warmup_cosine, cosine."
        )
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
    if peak_lr < 0.0:
        raise ValueError(f"lr must be non-negative, got {peak_lr}.")
    if min_lr < 0.0:
        raise ValueError(f"min_lr must be non-negative, got {min_lr}.")
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


def encode_all_camera_batch(
    vae: SplatterVAE,
    images: torch.Tensor,
) -> tuple[Dict[str, torch.Tensor], torch.Tensor, torch.Tensor]:
    """Encode every RGB camera view while preserving the ``(B, camera_num)`` layout."""
    if images.dim() != 5:
        raise ValueError(f"Expected images as (B,A,3,H,W), got {tuple(images.shape)}.")

    bsz, num_views, channels, height, width = images.shape
    if channels != 3:
        raise ValueError(
            "SplatterVAE encoders now expect RGB inputs only; depth is used only for depth regularization. "
            f"Got images with {channels} channels."
        )
    flat_images = images.reshape(bsz * num_views, channels, height, width).contiguous()

    z_inv, inv_vq_loss, z_dep, dep_vq_loss, _ = vae.encode(flat_images)
    z_inv = z_inv.reshape(bsz, num_views, *z_inv.shape[1:]).contiguous()
    z_dep = z_dep.reshape(bsz, num_views, *z_dep.shape[1:]).contiguous()

    return {"z_inv": z_inv, "z_dep": z_dep}, inv_vq_loss, dep_vq_loss


def _non_identity_randperm(num_items: int, device: torch.device) -> torch.Tensor:
    """Return a random permutation that changes order whenever possible."""
    if num_items <= 1:
        return torch.arange(num_items, device=device)

    perm = torch.randperm(num_items, device=device)
    if torch.equal(perm, torch.arange(num_items, device=device)):
        perm = torch.roll(perm, shifts=1, dims=0)
    return perm


def _sqrt_positive_part(x: torch.Tensor) -> torch.Tensor:
    """sqrt(max(0, x)) with zero subgradient at x <= 0."""
    ret = torch.zeros_like(x)
    positive = x > 0
    ret[positive] = torch.sqrt(x[positive])
    return ret


def _rotation_matrix_to_quaternion_wxyz(matrix: torch.Tensor) -> torch.Tensor:
    """Convert rotation matrices to real-first quaternions.

    ``VAESplatterToGaussians`` rotates predicted Gaussian orientations from the
    source camera frame to the world frame.  It expects quaternions in
    ``(w, x, y, z)`` order, matching ``utils.general_utils.quaternion_raw_multiply``.
    """
    if matrix.shape[-2:] != (3, 3):
        raise ValueError(f"Expected rotation matrices with shape (..., 3, 3), got {tuple(matrix.shape)}.")

    m00 = matrix[..., 0, 0]
    m01 = matrix[..., 0, 1]
    m02 = matrix[..., 0, 2]
    m10 = matrix[..., 1, 0]
    m11 = matrix[..., 1, 1]
    m12 = matrix[..., 1, 2]
    m20 = matrix[..., 2, 0]
    m21 = matrix[..., 2, 1]
    m22 = matrix[..., 2, 2]

    q_abs = _sqrt_positive_part(
        torch.stack(
            [
                1.0 + m00 + m11 + m22,
                1.0 + m00 - m11 - m22,
                1.0 - m00 + m11 - m22,
                1.0 - m00 - m11 + m22,
            ],
            dim=-1,
        )
    )

    # Four candidates, each numerically stable when its corresponding q_abs is
    # the largest component.  Candidate order is w, x, y, z.
    quat_by_rijk = torch.stack(
        [
            torch.stack([q_abs[..., 0] ** 2, m21 - m12, m02 - m20, m10 - m01], dim=-1),
            torch.stack([m21 - m12, q_abs[..., 1] ** 2, m10 + m01, m02 + m20], dim=-1),
            torch.stack([m02 - m20, m10 + m01, q_abs[..., 2] ** 2, m21 + m12], dim=-1),
            torch.stack([m10 - m01, m20 + m02, m21 + m12, q_abs[..., 3] ** 2], dim=-1),
        ],
        dim=-2,
    )
    quat_candidates = quat_by_rijk / (2.0 * q_abs[..., None].clamp(min=0.1))
    best = F.one_hot(q_abs.argmax(dim=-1), num_classes=4).to(dtype=torch.bool)
    quat = quat_candidates[best, :].reshape(*matrix.shape[:-2], 4)
    return F.normalize(quat, dim=-1, eps=1e-6)


def _compute_soft_image_region_penalty(
    xyz_world: torch.Tensor,
    world_view_transform: torch.Tensor,
    intrinsics: torch.Tensor,
    img_h: int,
    img_w: int,
    min_depth: float = 1e-3,
    penalty_cap: float = 100.0,
    source_view_indices: Optional[torch.Tensor] = None,
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

    min_z = max(float(min_depth), 1e-3)
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
    depth_penalty = torch.where(
        torch.isfinite(z),
        depth_penalty,
        torch.full_like(depth_penalty, float(penalty_cap)),
    )

    per_view_penalty = (image_penalty + depth_penalty).clamp(max=float(penalty_cap))
    outside_mask = (
        (~valid_depth)
        | (~finite_projection)
        | (u < 0.0)
        | (u > max_u)
        | (v < 0.0)
        | (v > max_v)
    )

    stats: Dict[str, torch.Tensor] = {
        "frustum_loss": per_view_penalty.mean(),
        "inactive_ratio_mean": outside_mask.float().mean(),
        "invalid_depth_ratio_mean": (~valid_depth).float().mean(),
        "nonfinite_projection_ratio_mean": (~finite_projection).float().mean(),
    }

    if source_view_indices is not None:
        num_views = world_view_transform.shape[1]
        src_idx = source_view_indices.to(device=device, dtype=torch.long).view(-1, 1, 1)
        src_idx = src_idx.expand(-1, 1, outside_mask.shape[-1])
        source_outside = outside_mask.gather(dim=1, index=src_idx).squeeze(1)
        stats["inactive_ratio_src"] = source_outside.float().mean()

        if num_views > 1:
            target_view_ids = torch.arange(num_views, device=device).view(1, num_views, 1)
            non_source_mask = target_view_ids != source_view_indices.to(device=device).view(-1, 1, 1)
            non_source_mask = non_source_mask.expand_as(outside_mask)
            stats["inactive_ratio_tgt"] = outside_mask.masked_select(non_source_mask).float().mean()
        else:
            stats["inactive_ratio_tgt"] = stats["inactive_ratio_src"]
    else:
        stats["inactive_ratio_src"] = outside_mask[:, 0].float().mean()
        stats["inactive_ratio_tgt"] = outside_mask[:, 1:].float().mean() if outside_mask.shape[1] > 1 else stats["inactive_ratio_src"]

    return stats


def _compute_effective_rank_regularization(scales: torch.Tensor) -> Dict[str, torch.Tensor]:
    """Effective-rank regularization from eRank-GS applied to Gaussian scales."""
    scale = torch.nan_to_num(scales.float(), nan=1e-6, posinf=1e2, neginf=1e-6).clamp_min(1e-6)
    flat_scale = scale.reshape(-1, 3)
    spectrum = flat_scale.pow(2)
    probs = spectrum / spectrum.sum(dim=-1, keepdim=True).clamp_min(1e-12)
    entropy = -(probs * torch.log(probs.clamp_min(1e-12))).sum(dim=-1)
    erank = torch.exp(entropy)
    ordered_scale, _ = torch.sort(flat_scale, dim=-1, descending=True)
    ratio = ordered_scale[:, 0] / ordered_scale[:, 2].clamp_min(1e-6)
    return {
        "erank_loss": torch.clamp(-torch.log(erank - 1.0 + 1e-7), min=0.0).mean(),
        "thin_loss": ordered_scale[:, 2].mean(),
        "erank_mean": erank.mean(),
        "erank_min": erank.amin(),
        "scale_ratio_mean": ratio.mean(),
        "scale_ratio_max": ratio.amax(),
    }


def _sanitize_metric_depth(depth: torch.Tensor, splatter_cfg: SplatterConfig) -> torch.Tensor:
    znear = float(splatter_cfg.data.znear)
    zfar = float(splatter_cfg.data.zfar)
    fill = 0.5 * (znear + zfar)
    return torch.nan_to_num(depth, nan=fill, posinf=zfar, neginf=znear).clamp(min=znear, max=zfar)


def _normalize_depth_patches(
    patches: torch.Tensor,
    global_std: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    mean = patches.mean(dim=1, keepdim=True)
    std = patches.std(dim=1, keepdim=True) if global_std is None else global_std
    dataset_std = patches.reshape(-1).std().clamp_min(1e-6)
    return (patches - mean) / (std + 1e-2 * dataset_std)


def _depth_patchify(depth: torch.Tensor, patch_size: int) -> torch.Tensor:
    patch_size = max(1, int(patch_size))
    patch_size = min(patch_size, int(depth.shape[-2]), int(depth.shape[-1]))
    return F.unfold(depth, kernel_size=patch_size, stride=patch_size).permute(0, 2, 1).reshape(
        -1,
        patch_size * patch_size,
    )


def _margin_l2_loss(pred: torch.Tensor, target: torch.Tensor, margin: float) -> torch.Tensor:
    diff = pred - target
    if margin > 0.0:
        mask = diff.abs() > float(margin)
        if not mask.any():
            return diff.new_zeros(())
        diff = diff[mask]
    return diff.pow(2).mean()


def _dngaussian_depth_regularization(
    rendered_depth: torch.Tensor,
    target_depth: torch.Tensor,
    splatter_cfg: SplatterConfig,
    cfg_train: TrainConfig,
) -> Dict[str, torch.Tensor]:
    """Patch-normalized DNGaussian depth loss for rasterized expected depth."""
    if rendered_depth is None or target_depth is None:
        zero_device = target_depth.device if target_depth is not None else torch.device("cpu")
        zero = torch.zeros((), device=zero_device)
        return {"depth_loss": zero, "depth_local_loss": zero, "depth_global_loss": zero}

    if rendered_depth.shape != target_depth.shape:
        raise ValueError(
            f"Expected rendered/target depth shapes to match, got "
            f"{tuple(rendered_depth.shape)} and {tuple(target_depth.shape)}."
        )

    pred = _sanitize_metric_depth(rendered_depth, splatter_cfg)
    target = _sanitize_metric_depth(target_depth.to(device=pred.device, dtype=pred.dtype), splatter_cfg)
    pred_flat = pred.reshape(-1, 1, pred.shape[-2], pred.shape[-1])
    target_flat = target.reshape(-1, 1, target.shape[-2], target.shape[-1])

    patch_size = int(cfg_train.depth_patch_size)
    margin = float(cfg_train.depth_error_tolerance)
    pred_patches = _depth_patchify(pred_flat, patch_size)
    target_patches = _depth_patchify(target_flat, patch_size)

    pred_local = _normalize_depth_patches(pred_patches)
    target_local = _normalize_depth_patches(target_patches)
    local_loss = _margin_l2_loss(pred_local, target_local, margin)

    pred_global = _normalize_depth_patches(pred_patches, global_std=pred_flat.std().detach().clamp_min(1e-6))
    target_global = _normalize_depth_patches(target_patches, global_std=target_flat.std().detach().clamp_min(1e-6))
    global_loss = _margin_l2_loss(pred_global, target_global, margin)

    depth_loss = (
        float(cfg_train.depth_local_weight) * local_loss
        + float(cfg_train.depth_global_weight) * global_loss
    )
    return {
        "depth_loss": depth_loss,
        "depth_local_loss": local_loss,
        "depth_global_loss": global_loss,
        "rendered_depth_mean": pred.mean(),
        "target_depth_mean": target.mean(),
    }


def _gather_camera_rows(values: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
    """Gather one camera row per batch item from ``values[:, camera]``."""
    batch_ids = torch.arange(values.shape[0], device=values.device)
    return values[batch_ids, indices]


def _gather_target_cameras(values: torch.Tensor, target_indices: torch.Tensor) -> torch.Tensor:
    """Gather multiple target camera rows per batch item.

    Args:
        values: ``(B, A, ...)`` camera-indexed tensor.
        target_indices: ``(B, T)`` target camera indices.
    """
    trailing_shape = values.shape[2:]
    gather_index = target_indices.view(
        target_indices.shape[0],
        target_indices.shape[1],
        *([1] * len(trailing_shape)),
    ).expand(target_indices.shape[0], target_indices.shape[1], *trailing_shape)
    return torch.gather(values, dim=1, index=gather_index)


def _target_indices_excluding_source(source_indices: torch.Tensor, num_views: int) -> torch.Tensor:
    """Return all camera indices except each row's selected source camera."""
    if num_views < 2:
        raise ValueError("Source-plus-target reconstruction requires at least two camera viewpoints.")
    all_views = torch.arange(num_views, device=source_indices.device).view(1, num_views)
    all_views = all_views.expand(source_indices.shape[0], num_views)
    keep_target = all_views != source_indices.view(-1, 1)
    return all_views[keep_target].view(source_indices.shape[0], num_views - 1)


def _random_other_camera_indices(source_indices: torch.Tensor, num_views: int) -> torch.Tensor:
    """Sample one non-source camera index per row for invariant shuffling."""
    if num_views < 2:
        raise ValueError("Invariant shuffling requires at least two camera viewpoints.")
    offset = torch.randint(
        low=1,
        high=num_views,
        size=source_indices.shape,
        device=source_indices.device,
    )
    return (source_indices + offset) % num_views


def _render_selected_sources_to_views(
    vae: SplatterVAE,
    splatter_to_gaussians: VAESplatterToGaussians,
    splatter_cfg: SplatterConfig,
    z_inv_source: torch.Tensor,
    z_dep_source: torch.Tensor,
    source_indices: torch.Tensor,
    target_indices: torch.Tensor,
    intrinsics: torch.Tensor,
    c2w: torch.Tensor,
    w2c: torch.Tensor,
    bg: torch.Tensor,
) -> tuple[torch.Tensor, Optional[torch.Tensor], torch.Tensor, Dict[str, torch.Tensor]]:
    """Decode selected source views and render source plus non-source targets.

    ``z_dep_source`` defines the camera frame of the decoded Splatter image.
    Therefore ``source_indices`` must point to the camera viewpoint that produced
    the dependent feature.
    """
    source_intrinsics = _gather_camera_rows(intrinsics, source_indices)
    source_c2w = _gather_camera_rows(c2w, source_indices)

    splatter = vae.decode(z_inv_source.contiguous(), z_dep_source.contiguous())
    source_quat = _rotation_matrix_to_quaternion_wxyz(source_c2w[:, :3, :3])
    gaussian_pc = splatter_to_gaussians(
        splatter=splatter,
        source_cameras_view_to_world=source_c2w,
        source_cv2wT_quat=source_quat,
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
        render_mode="RGB+ED",
    )

    source_view_indices = torch.zeros(source_indices.shape[0], device=source_indices.device, dtype=torch.long)
    frustum_stats = _compute_soft_image_region_penalty(
        xyz_world=gaussian_pc["xyz"],
        world_view_transform=render_w2c,
        intrinsics=render_intrinsics,
        img_h=splatter_cfg.data.img_height,
        img_w=splatter_cfg.data.img_width,
        min_depth=splatter_cfg.data.znear,
        source_view_indices=source_view_indices,
    )
    frustum_stats.update(_compute_effective_rank_regularization(gaussian_pc["scaling"]))

    render_indices = torch.cat((source_indices.view(-1, 1), target_indices), dim=1)
    return out["render"], out.get("depth"), render_indices, frustum_stats

def compute_reconstruction_and_renders(
    vae: SplatterVAE,
    splatter_to_gaussians: VAESplatterToGaussians,
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
    """Compute source-plus-target reconstruction and shuffled latent losses.

    ``z_inv`` carries state/scene information and ``z_dep`` carries viewpoint
    information. For dependent shuffles, the decoded source viewpoint follows
    ``z_dep`` while camera metadata and depth priors remain on the current batch
    row, so the depth prior has the invariant state and dependent viewpoint.
    """
    bsz, num_views = z_inv.shape[:2]
    if num_views < 2:
        raise ValueError("Source-plus-target reconstruction requires at least two camera viewpoints.")

    device = z_inv.device
    batch_ids = torch.arange(bsz, device=device)

    source_indices = torch.randint(low=0, high=num_views, size=(bsz,), device=device)
    inv_source_indices = _random_other_camera_indices(source_indices, num_views)
    batch_perm = _non_identity_randperm(bsz, device=device)

    # z_dep is camera-frame specific. When it is shuffled across batch rows, its
    # source viewpoint must move with it; the row/state used for camera metadata,
    # target images, and depth supervision remains the invariant branch row.
    dep_source_indices = source_indices[batch_perm]

    variant_names = ("self", "shuffle_inv", "shuffle_dep", "shuffle_both")
    num_variants = len(variant_names)

    variant_z_inv = torch.stack(
        [
            z_inv[batch_ids, source_indices],
            z_inv[batch_ids, inv_source_indices],
            z_inv[batch_ids, source_indices],
            z_inv[batch_ids, inv_source_indices],
        ],
        dim=0,
    ).contiguous()
    variant_z_dep = torch.stack(
        [
            z_dep[batch_ids, source_indices],
            z_dep[batch_ids, source_indices],
            z_dep[batch_perm, dep_source_indices],
            z_dep[batch_perm, dep_source_indices],
        ],
        dim=0,
    ).contiguous()
    variant_source_indices = torch.stack(
        [
            source_indices,
            source_indices,
            dep_source_indices,
            dep_source_indices,
        ],
        dim=0,
    ).contiguous()

    flat_count = num_variants * bsz
    flat_z_inv = variant_z_inv.reshape(flat_count, *z_inv.shape[2:]).contiguous()
    flat_z_dep = variant_z_dep.reshape(flat_count, *z_dep.shape[2:]).contiguous()
    flat_source_indices = variant_source_indices.reshape(flat_count).contiguous()
    flat_target_indices = _target_indices_excluding_source(flat_source_indices, num_views)

    # Expand camera tensors by invariant/state row. This deliberately does not
    # permute rows for shuffle_dep/shuffle_both; only the dependent viewpoint id
    # changes. The same rule is used for depth-supervision targets below.
    flat_intrinsics = intrinsics[None].expand(num_variants, *intrinsics.shape).reshape(
        flat_count,
        num_views,
        3,
        3,
    ).contiguous()
    flat_c2w = c2w[None].expand(num_variants, *c2w.shape).reshape(
        flat_count,
        num_views,
        4,
        4,
    ).contiguous()
    flat_w2c = w2c[None].expand(num_variants, *w2c.shape).reshape(
        flat_count,
        num_views,
        4,
        4,
    ).contiguous()
    flat_depths = None
    if depths is not None:
        flat_depths = depths[None].expand(num_variants, *depths.shape).reshape(
            flat_count,
            num_views,
            *depths.shape[2:],
        ).contiguous()

    rendered_flat, rendered_depth_flat, render_indices_flat, frustum_stats = _render_selected_sources_to_views(
        vae=vae,
        splatter_to_gaussians=splatter_to_gaussians,
        splatter_cfg=splatter_cfg,
        z_inv_source=flat_z_inv,
        z_dep_source=flat_z_dep,
        source_indices=flat_source_indices,
        target_indices=flat_target_indices,
        intrinsics=flat_intrinsics,
        c2w=flat_c2w,
        w2c=flat_w2c,
        bg=bg,
    )

    num_render_views = rendered_flat.shape[1]
    rendered = rendered_flat.reshape(
        num_variants,
        bsz,
        num_render_views,
        3,
        splatter_cfg.data.img_height,
        splatter_cfg.data.img_width,
    ).contiguous()

    flat_images = images_01[None].expand(num_variants, *images_01.shape).reshape(
        flat_count,
        num_views,
        *images_01.shape[2:],
    ).contiguous()
    flat_target_images = _gather_target_cameras(flat_images, render_indices_flat)
    target_images = flat_target_images.reshape(
        num_variants,
        bsz,
        num_render_views,
        *images_01.shape[2:],
    ).contiguous()
    render_indices = render_indices_flat.reshape(num_variants, bsz, num_render_views).contiguous()
    target_indices = flat_target_indices.reshape(num_variants, bsz, num_views - 1).contiguous()

    target_depths = None
    rendered_depth = None
    if flat_depths is not None and rendered_depth_flat is not None:
        flat_target_depths = _gather_target_cameras(flat_depths, render_indices_flat)
        target_depths = flat_target_depths.reshape(
            num_variants,
            bsz,
            num_render_views,
            *flat_depths.shape[2:],
        ).contiguous()
        rendered_depth = rendered_depth_flat.reshape(
            num_variants,
            bsz,
            num_render_views,
            1,
            splatter_cfg.data.img_height,
            splatter_cfg.data.img_width,
        ).contiguous()
        frustum_stats.update(
            _dngaussian_depth_regularization(
                rendered_depth=rendered_depth,
                target_depth=target_depths,
                splatter_cfg=splatter_cfg,
                cfg_train=cfg_train,
            )
        )
    else:
        zero = rendered.new_zeros(())
        frustum_stats.update(
            {
                "depth_loss": zero,
                "depth_local_loss": zero,
                "depth_global_loss": zero,
                "rendered_depth_mean": zero,
                "target_depth_mean": zero,
            }
        )

    loss_values = compute_batched_reconstruction_losses(
        predicted=rendered,
        ground_truth=target_images,
        ssim_weight=cfg_train.ssim_weight,
    )
    losses: Dict[str, torch.Tensor] = {
        name: loss_values[idx] for idx, name in enumerate(variant_names)
    }

    rec_loss = (
        losses["self"]
        + float(cfg_train.shuffle_inv_rec_weight) * losses["shuffle_inv"]
        + float(cfg_train.shuffle_dep_rec_weight) * losses["shuffle_dep"]
        + float(cfg_train.shuffle_both_rec_weight) * losses["shuffle_both"]
    )

    out_dict: Dict[str, Any] = {
        "rec_loss": rec_loss,
        "rec_self": losses["self"],
        "rec_shuffle_inv": losses["shuffle_inv"],
        "rec_shuffle_dep": losses["shuffle_dep"],
        "rec_shuffle_both": losses["shuffle_both"],
    }
    for stat_name, stat_value in frustum_stats.items():
        out_dict[stat_name] = stat_value

    if return_renders:
        out_dict["gt_images"] = images_01
        out_dict["target_images_self"] = target_images[0]
        out_dict["rendered_self"] = rendered[0]
        out_dict["rendered_shuffle_inv"] = rendered[1]
        out_dict["rendered_shuffle_dep"] = rendered[2]
        out_dict["rendered_shuffle_both"] = rendered[3]
        out_dict["source_indices"] = variant_source_indices.detach().cpu()
        out_dict["target_indices"] = target_indices.detach().cpu()
        out_dict["render_indices"] = render_indices.detach().cpu()
        if rendered_depth is not None and target_depths is not None:
            out_dict["rendered_depth_self"] = rendered_depth[0]
            out_dict["target_depths_self"] = target_depths[0]
        out_dict["batch_perm"] = batch_perm.detach().cpu()

    return out_dict

def _make_wandb_named_image_panel(
    named_images: list[tuple[str, torch.Tensor]],
    max_vis: int,
) -> wandb.Image:
    """Create one W&B image grid with one row per named tensor."""
    max_vis = max(1, int(max_vis))
    rows = []
    names = []
    for name, images in named_images:
        rows.append(images[:max_vis].detach().cpu().clamp(0.0, 1.0))
        names.append(name)
    grid = make_grid(torch.cat(rows, dim=0), nrow=max_vis, padding=2)
    return wandb.Image(grid, caption=" | ".join(names))


@torch.no_grad()
def validate_and_log_wandb(
    vae: SplatterVAE,
    splatter_cfg: SplatterConfig,
    splatter_to_gaussians: VAESplatterToGaussians,
    valid_dataloader: DataLoader,
    device: torch.device,
    bg: torch.Tensor,
    cfg_train: TrainConfig,
    global_step: int,
) -> None:
    """Run validation with the same all-camera losses used for training."""
    if wandb.run is None:
        return

    prev_vae_mode = vae.training
    prev_splatter_mode = splatter_to_gaussians.training
    vae.eval()
    splatter_to_gaussians.eval()

    scalar_sums = {
        "val/rec_loss": 0.0,
        "val/rec_self": 0.0,
        "val/rec_shuffle_inv": 0.0,
        "val/rec_shuffle_dep": 0.0,
        "val/rec_shuffle_both": 0.0,
        "val/inv_contrastive_loss": 0.0,
        "val/inv_consistency_loss": 0.0,
        "val/dep_contrastive_loss": 0.0,
        "val/dep_consistency_loss": 0.0,
        "val/frustum_loss": 0.0,
        "val/erank_loss": 0.0,
        "val/thin_loss": 0.0,
        "val/erank_mean": 0.0,
        "val/erank_min": 0.0,
        "val/scale_ratio_mean": 0.0,
        "val/scale_ratio_max": 0.0,
        "val/depth_loss": 0.0,
        "val/depth_local_loss": 0.0,
        "val/depth_global_loss": 0.0,
        "val/rendered_depth_mean": 0.0,
        "val/target_depth_mean": 0.0,
        "val/inactive_pct_mean": 0.0,
        "val/inactive_pct_src": 0.0,
        "val/inactive_pct_tgt": 0.0,
        "val/invalid_depth_pct_mean": 0.0,
        "val/nonfinite_projection_pct_mean": 0.0,
    }

    num_eval_batches = 0
    image_payload: Dict[str, wandb.Image] = {}

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
            splatter_to_gaussians=splatter_to_gaussians,
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

        scalar_sums["val/rec_loss"] += float(rec_out["rec_loss"].item())
        scalar_sums["val/rec_self"] += float(rec_out["rec_self"].item())
        scalar_sums["val/rec_shuffle_inv"] += float(rec_out["rec_shuffle_inv"].item())
        scalar_sums["val/rec_shuffle_dep"] += float(rec_out["rec_shuffle_dep"].item())
        scalar_sums["val/rec_shuffle_both"] += float(rec_out["rec_shuffle_both"].item())
        scalar_sums["val/inv_contrastive_loss"] += float(inv_contrastive_loss.item())
        scalar_sums["val/inv_consistency_loss"] += float(inv_consistency_loss.item())
        scalar_sums["val/dep_contrastive_loss"] += float(dep_contrastive_loss.item())
        scalar_sums["val/dep_consistency_loss"] += float(dep_consistency_loss.item())
        scalar_sums["val/frustum_loss"] += float(rec_out["frustum_loss"].item())
        scalar_sums["val/erank_loss"] += float(rec_out["erank_loss"].item())
        scalar_sums["val/thin_loss"] += float(rec_out["thin_loss"].item())
        scalar_sums["val/erank_mean"] += float(rec_out["erank_mean"].item())
        scalar_sums["val/erank_min"] += float(rec_out["erank_min"].item())
        scalar_sums["val/scale_ratio_mean"] += float(rec_out["scale_ratio_mean"].item())
        scalar_sums["val/scale_ratio_max"] += float(rec_out["scale_ratio_max"].item())
        scalar_sums["val/depth_loss"] += float(rec_out["depth_loss"].item())
        scalar_sums["val/depth_local_loss"] += float(rec_out["depth_local_loss"].item())
        scalar_sums["val/depth_global_loss"] += float(rec_out["depth_global_loss"].item())
        scalar_sums["val/rendered_depth_mean"] += float(rec_out["rendered_depth_mean"].item())
        scalar_sums["val/target_depth_mean"] += float(rec_out["target_depth_mean"].item())
        scalar_sums["val/inactive_pct_mean"] += float(100.0 * rec_out["inactive_ratio_mean"].item())
        scalar_sums["val/inactive_pct_src"] += float(100.0 * rec_out["inactive_ratio_src"].item())
        scalar_sums["val/inactive_pct_tgt"] += float(100.0 * rec_out["inactive_ratio_tgt"].item())
        scalar_sums["val/invalid_depth_pct_mean"] += float(100.0 * rec_out["invalid_depth_ratio_mean"].item())
        scalar_sums["val/nonfinite_projection_pct_mean"] += float(
            100.0 * rec_out["nonfinite_projection_ratio_mean"].item()
        )

        if num_eval_batches == 0:
            num_targets_to_show = min(rec_out["rendered_self"].shape[1], 6)
            panel_items: list[tuple[str, torch.Tensor]] = []
            for view_slot in range(num_targets_to_show):
                view_name = "source" if view_slot == 0 else f"target{view_slot}"
                panel_items.append((f"gt_{view_name}", rec_out["target_images_self"][:, view_slot]))
            for view_slot in range(num_targets_to_show):
                view_name = "source" if view_slot == 0 else f"target{view_slot}"
                panel_items.append((f"render_{view_name}", rec_out["rendered_self"][:, view_slot]))
            panel_items.extend(
                [
                    ("shuffle_inv_source", rec_out["rendered_shuffle_inv"][:, 0]),
                    ("shuffle_dep_source", rec_out["rendered_shuffle_dep"][:, 0]),
                    ("shuffle_both_source", rec_out["rendered_shuffle_both"][:, 0]),
                ]
            )
            image_payload = {
                "val/render_summary": _make_wandb_named_image_panel(panel_items, max_vis=cfg_train.val_max_vis),
            }

        num_eval_batches += 1

    if num_eval_batches == 0:
        vae.train(prev_vae_mode)
        splatter_to_gaussians.train(prev_splatter_mode)
        return

    log_dict: Dict[str, Any] = {
        key: value / float(num_eval_batches) for key, value in scalar_sums.items()
    }
    log_dict["global_step"] = global_step
    log_dict.update(image_payload)
    wandb.log(log_dict, step=global_step)

    vae.train(prev_vae_mode)
    splatter_to_gaussians.train(prev_splatter_mode)


def train_splatter_vae(
    vae: SplatterVAE,
    splatter_cfg: SplatterConfig,
    train_dataloader: DataLoader,
    cfg_train: TrainConfig,
    valid_dataloader: Optional[DataLoader] = None,
    resume_ckpt: Optional[str] = None,
):
    """Train SplatterVAE with all-camera rendering."""
    device = torch.device(cfg_train.device)
    vae.to(device)

    splatter_to_gaussians = VAESplatterToGaussians(splatter_cfg).to(device)
    optimizer = torch.optim.Adam(vae.parameters(), lr=cfg_train.lr)

    lr_total_steps = _resolve_lr_total_steps(cfg_train, train_dataloader)
    lr_schedule = _normalize_lr_schedule(cfg_train.lr_schedule)
    if lr_schedule != "constant":
        print(
            f"[LR] schedule={lr_schedule}, peak_lr={cfg_train.lr:g}, "
            f"min_lr={cfg_train.min_lr:g}, warmup_steps={cfg_train.lr_warmup_steps}, "
            f"total_steps={lr_total_steps}"
        )

    bg = torch.ones(3, device=device) if splatter_cfg.data.white_background else torch.zeros(3, device=device)
    start_epoch = 0
    global_step = 0

    if resume_ckpt is not None and os.path.isfile(resume_ckpt):
        ckpt = torch.load(resume_ckpt, map_location="cpu")
        vae.load_state_dict(ckpt["vae_state_dict"])
        splatter_to_gaussians.load_state_dict(ckpt["splatter_to_gaussians_state_dict"])
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
            splatter_to_gaussians.train()

            # Shapes:
            #   images: (B, A, 3, H, W), K/c2w/w2c: (B, A, ...)
            # ``A`` is stable across the batch and is the camera index used by
            # the view-dependent contrastive loss.
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
                splatter_to_gaussians=splatter_to_gaussians,
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
            frustum_loss = rec_out["frustum_loss"]
            erank_loss = rec_out["erank_loss"]
            thin_loss = rec_out["thin_loss"]
            depth_loss = rec_out["depth_loss"]

            inv_contrastive_loss, dep_contrastive_loss = compute_all_camera_contrastive_losses(
                z_inv=latents["z_inv"],
                z_dep=latents["z_dep"],
                temperature=cfg_train.temperature,
            )
            inv_consistency_loss = compute_latent_consistency_loss(latents["z_inv"], mode="state")
            dep_consistency_loss = compute_latent_consistency_loss(latents["z_dep"], mode="view")

            vq_loss = inv_vq_loss + dep_vq_loss
            total_loss = (
                cfg_train.rec_weight * rec_loss
                + cfg_train.vq_weight * vq_loss
                + cfg_train.inv_contrastive_weight * inv_contrastive_loss
                + cfg_train.inv_consistency_weight * inv_consistency_loss
                + cfg_train.dep_contrastive_weight * dep_contrastive_loss
                + cfg_train.dep_consistency_weight * dep_consistency_loss
                + cfg_train.frustum_weight * frustum_loss
                + cfg_train.erank_weight * erank_loss
                + cfg_train.thin_weight * thin_loss
                + cfg_train.depth_weight * depth_loss
            )

            finite_terms = {
                "rec_loss": rec_loss,
                "vq_loss": vq_loss,
                "inv_contrastive_loss": inv_contrastive_loss,
                "inv_consistency_loss": inv_consistency_loss,
                "dep_contrastive_loss": dep_contrastive_loss,
                "dep_consistency_loss": dep_consistency_loss,
                "frustum_loss": frustum_loss,
                "erank_loss": erank_loss,
                "thin_loss": thin_loss,
                "depth_loss": depth_loss,
                "total_loss": total_loss,
            }
            bad_terms = [name for name, value in finite_terms.items() if not torch.isfinite(value).all()]
            if bad_terms:
                print(
                    f"[Warn] Non-finite loss detected at global_step={global_step} "
                    f"(bad={bad_terms}, inactive_pct={100.0 * rec_out['inactive_ratio_mean'].item():.2f}, "
                    f"invalid_depth_pct={100.0 * rec_out['invalid_depth_ratio_mean'].item():.2f}, "
                    f"nonfinite_proj_pct={100.0 * rec_out['nonfinite_projection_ratio_mean'].item():.2f}). "
                    "Skipping optimizer step."
                )
                if wandb.run is not None:
                    wandb.log(
                        {
                            "global_step": global_step,
                            "train/nonfinite_batch": 1.0,
                            "train/lr": current_lr,
                        },
                        step=global_step,
                    )
                global_step += 1
                continue

            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(vae.parameters(), max_norm=5.0)
            optimizer.step()

            if step % 250 == 0:
                print(
                    f"[Epoch {epoch + 1} | Step {step} | Global {global_step}] "
                    f"Loss={total_loss.item():.4f} lr={current_lr:.2e} "
                    f"(rec={rec_loss.item():.4f}, self={rec_out['rec_self'].item():.4f}, "
                    f"shuf_inv={rec_out['rec_shuffle_inv'].item():.4f}, "
                    f"shuf_dep={rec_out['rec_shuffle_dep'].item():.4f}, "
                    f"shuf_both={rec_out['rec_shuffle_both'].item():.4f}, "
                    f"vq={vq_loss.item():.4f}, inv_con={inv_contrastive_loss.item():.4f}, "
                    f"inv_cons={inv_consistency_loss.item():.4f}, dep_con={dep_contrastive_loss.item():.4f}, "
                    f"dep_cons={dep_consistency_loss.item():.4f}, frustum={frustum_loss.item():.4f}, "
                    f"erank={erank_loss.item():.4f}, depth={depth_loss.item():.4f})"
                )
                if wandb.run is not None:
                    wandb.log(
                        {
                            "train/total_loss": total_loss.item(),
                            "train/lr": current_lr,
                            "train/rec_loss": rec_loss.item(),
                            "train/rec_self": rec_out["rec_self"].item(),
                            "train/rec_shuffle_inv": rec_out["rec_shuffle_inv"].item(),
                            "train/rec_shuffle_dep": rec_out["rec_shuffle_dep"].item(),
                            "train/rec_shuffle_both": rec_out["rec_shuffle_both"].item(),
                            "train/shuffle_inv_rec_weight": float(cfg_train.shuffle_inv_rec_weight),
                            "train/shuffle_dep_rec_weight": float(cfg_train.shuffle_dep_rec_weight),
                            "train/shuffle_both_rec_weight": float(cfg_train.shuffle_both_rec_weight),
                            "train/vq_loss": vq_loss.item(),
                            "train/inv_vq_loss": inv_vq_loss.item(),
                            "train/dep_vq_loss": dep_vq_loss.item(),
                            "train/inv_contrastive_loss": inv_contrastive_loss.item(),
                            "train/inv_contrastive_weight": float(cfg_train.inv_contrastive_weight),
                            "train/inv_consistency_loss": inv_consistency_loss.item(),
                            "train/inv_consistency_weight": float(cfg_train.inv_consistency_weight),
                            "train/dep_contrastive_loss": dep_contrastive_loss.item(),
                            "train/dep_contrastive_weight": float(cfg_train.dep_contrastive_weight),
                            "train/dep_consistency_loss": dep_consistency_loss.item(),
                            "train/dep_consistency_weight": float(cfg_train.dep_consistency_weight),
                            "train/frustum_loss": frustum_loss.item(),
                            "train/frustum_loss_weighted": (cfg_train.frustum_weight * frustum_loss).item(),
                            "train/erank_loss": erank_loss.item(),
                            "train/erank_loss_weighted": (cfg_train.erank_weight * erank_loss).item(),
                            "train/thin_loss": thin_loss.item(),
                            "train/thin_loss_weighted": (cfg_train.thin_weight * thin_loss).item(),
                            "train/erank_mean": rec_out["erank_mean"].item(),
                            "train/erank_min": rec_out["erank_min"].item(),
                            "train/scale_ratio_mean": rec_out["scale_ratio_mean"].item(),
                            "train/scale_ratio_max": rec_out["scale_ratio_max"].item(),
                            "train/depth_loss": depth_loss.item(),
                            "train/depth_loss_weighted": (cfg_train.depth_weight * depth_loss).item(),
                            "train/depth_local_loss": rec_out["depth_local_loss"].item(),
                            "train/depth_global_loss": rec_out["depth_global_loss"].item(),
                            "train/rendered_depth_mean": rec_out["rendered_depth_mean"].item(),
                            "train/target_depth_mean": rec_out["target_depth_mean"].item(),
                            "train/inactive_pct_mean": 100.0 * rec_out["inactive_ratio_mean"].item(),
                            "train/inactive_pct_src": 100.0 * rec_out["inactive_ratio_src"].item(),
                            "train/inactive_pct_tgt": 100.0 * rec_out["inactive_ratio_tgt"].item(),
                            "train/invalid_depth_pct_mean": 100.0 * rec_out["invalid_depth_ratio_mean"].item(),
                            "train/nonfinite_projection_pct_mean": (
                                100.0 * rec_out["nonfinite_projection_ratio_mean"].item()
                            ),
                            "train/frustum_weight": float(cfg_train.frustum_weight),
                            "train/erank_weight": float(cfg_train.erank_weight),
                            "train/thin_weight": float(cfg_train.thin_weight),
                            "train/depth_weight": float(cfg_train.depth_weight),
                            "train/depth_local_weight": float(cfg_train.depth_local_weight),
                            "train/depth_global_weight": float(cfg_train.depth_global_weight),
                            "global_step": global_step,
                        },
                        step=global_step,
                    )

            if (
                valid_dataloader is not None
                and cfg_train.eval_every > 0
                and global_step > 0
                and global_step % cfg_train.eval_every == 0
            ):
                validate_and_log_wandb(
                    vae=vae,
                    splatter_cfg=splatter_cfg,
                    splatter_to_gaussians=splatter_to_gaussians,
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
                    "splatter_to_gaussians_state_dict": splatter_to_gaussians.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                }
                torch.save(ckpt, ckpt_path)
                print(f"[Checkpoint] Saved checkpoint to {ckpt_path}")

            global_step += 1

        epoch += 1
        if cfg_train.max_global_steps is None and epoch >= cfg_train.num_epochs:
            print(f"[Stop] Reached num_epochs={cfg_train.num_epochs}.")
            break
