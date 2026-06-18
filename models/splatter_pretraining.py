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
    compute_latent_consistency_loss,
    compute_reconstruction_loss,
)
from models.splatter import SplatterConfig, render_predicted
from models.splatter_gaussians import DirectSplatterToGaussians
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


def _masked_std(values: torch.Tensor, valid_mask: torch.Tensor) -> torch.Tensor:
    valid = valid_mask.to(dtype=torch.bool) & torch.isfinite(values)
    if not bool(valid.any()):
        return values.new_tensor(1.0)
    selected = values[valid]
    if selected.numel() <= 1:
        return values.new_tensor(1.0)
    return selected.std(unbiased=False).clamp_min(1.0e-6)


def _patchify(input_tensor: torch.Tensor, patch_size: int) -> torch.Tensor:
    return F.unfold(input_tensor, kernel_size=int(patch_size), stride=int(patch_size)).permute(0, 2, 1).reshape(-1, int(patch_size) * int(patch_size))


def _normalize_patches(
    patches: torch.Tensor,
    valid: torch.Tensor,
    fallback_global_std: torch.Tensor,
    fixed_std: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    weights = valid.to(dtype=patches.dtype)
    count = weights.sum(dim=1, keepdim=True).clamp_min(1.0)
    mean = (patches * weights).sum(dim=1, keepdim=True) / count
    if fixed_std is None:
        var = ((patches - mean).square() * weights).sum(dim=1, keepdim=True) / count
        std = torch.sqrt(var.clamp_min(1.0e-12))
    else:
        std = fixed_std.to(device=patches.device, dtype=patches.dtype).view(1, 1).expand_as(mean)
    return (patches - mean) / (std + 1.0e-2 * fallback_global_std.to(device=patches.device, dtype=patches.dtype))


def _dngaussian_patch_loss(
    predicted: torch.Tensor,
    target: torch.Tensor,
    valid_mask: torch.Tensor,
    patch_size: int,
    margin: float,
    min_valid_ratio: float,
    global_std: bool,
) -> torch.Tensor:
    patch_size = int(patch_size)
    if predicted.shape[-2] < patch_size or predicted.shape[-1] < patch_size:
        return predicted.new_zeros(())

    pred_patches = _patchify(predicted, patch_size)
    target_patches = _patchify(target, patch_size)
    valid_patches = _patchify(valid_mask.to(dtype=predicted.dtype), patch_size) > 0.5
    keep_patch = valid_patches.float().mean(dim=1) >= float(min_valid_ratio)
    if not bool(keep_patch.any()):
        return predicted.new_zeros(())

    pred_patches = pred_patches[keep_patch]
    target_patches = target_patches[keep_patch]
    valid_patches = valid_patches[keep_patch]

    pred_global_std = _masked_std(predicted, valid_mask).detach()
    target_global_std = _masked_std(target, valid_mask).detach()
    pred_norm = _normalize_patches(
        pred_patches,
        valid_patches,
        fallback_global_std=pred_global_std,
        fixed_std=pred_global_std if global_std else None,
    )
    target_norm = _normalize_patches(
        target_patches,
        valid_patches,
        fallback_global_std=target_global_std,
        fixed_std=target_global_std if global_std else None,
    )

    diff = pred_norm - target_norm
    active = valid_patches & (diff.abs() > float(margin))
    if not bool(active.any()):
        return predicted.new_zeros(())
    return diff.masked_select(active).square().mean()


def _random_depth_patch_size(cfg_train: TrainConfig) -> int:
    patch_min = max(1, int(cfg_train.depth_patch_min))
    patch_max = max(patch_min, int(cfg_train.depth_patch_max))
    if patch_min == patch_max:
        return patch_min
    return int(torch.randint(patch_min, patch_max + 1, (1,)).item())


def _dngaussian_depth_loss(
    predicted_depth: torch.Tensor,
    target_depth: torch.Tensor,
    valid_mask: torch.Tensor,
    cfg_train: TrainConfig,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    batch_views = predicted_depth.reshape(-1, *predicted_depth.shape[2:])
    targets = target_depth.reshape(-1, *target_depth.shape[2:]).to(dtype=batch_views.dtype)
    masks = valid_mask.reshape(-1, *valid_mask.shape[2:]).to(device=batch_views.device, dtype=torch.bool)

    predicted_clean = torch.where(masks, torch.nan_to_num(batch_views, nan=0.0, posinf=0.0, neginf=0.0), torch.zeros_like(batch_views))
    target_clean = torch.where(masks, torch.nan_to_num(targets, nan=0.0, posinf=0.0, neginf=0.0), torch.zeros_like(targets))
    patch_size = _random_depth_patch_size(cfg_train)

    local = _dngaussian_patch_loss(
        predicted=predicted_clean,
        target=target_clean,
        valid_mask=masks,
        patch_size=patch_size,
        margin=float(cfg_train.depth_error_tolerance),
        min_valid_ratio=float(cfg_train.depth_valid_min_ratio),
        global_std=False,
    )
    global_loss = _dngaussian_patch_loss(
        predicted=predicted_clean,
        target=target_clean,
        valid_mask=masks,
        patch_size=patch_size,
        margin=float(cfg_train.depth_error_tolerance),
        min_valid_ratio=float(cfg_train.depth_valid_min_ratio),
        global_std=True,
    )
    total = float(cfg_train.depth_local_weight) * local + float(cfg_train.depth_global_weight) * global_loss
    return total, local, global_loss


def _zero_depth_stats(device: torch.device) -> Dict[str, torch.Tensor]:
    zero = torch.zeros((), device=device)
    return {
        "hard_depth_loss": zero,
        "hard_depth_local_loss": zero,
        "hard_depth_global_loss": zero,
        "soft_depth_loss": zero,
        "soft_depth_local_loss": zero,
        "soft_depth_global_loss": zero,
        "depth_valid_ratio": zero,
    }


def _compute_depth_regularization(
    gaussian_pc: Dict[str, torch.Tensor],
    render_w2c: torch.Tensor,
    render_intrinsics: torch.Tensor,
    target_depths: Optional[torch.Tensor],
    bg: torch.Tensor,
    splatter_cfg: SplatterConfig,
    cfg_train: TrainConfig,
    global_step: int,
    return_renders: bool,
) -> Dict[str, Any]:
    device = gaussian_pc["xyz"].device
    stats: Dict[str, Any] = _zero_depth_stats(device)
    if target_depths is None:
        return stats

    target_depths = target_depths.to(device=device, dtype=gaussian_pc["xyz"].dtype)
    valid = torch.isfinite(target_depths) & (target_depths >= float(splatter_cfg.data.znear)) & (target_depths <= float(splatter_cfg.data.zfar))
    stats["depth_valid_ratio"] = valid.float().mean()

    if int(global_step) >= int(cfg_train.hard_depth_start_step):
        hard_depth = render_predicted(
            pc=gaussian_pc,
            world_view_transform=render_w2c,
            intrinsics=render_intrinsics,
            bg_color=bg,
            cfg=splatter_cfg,
            render_mode="D",
            detach_scale_rotation=True,
            override_opacity=float(cfg_train.hard_depth_opacity),
        )["depth"]
        hard_total, hard_local, hard_global = _dngaussian_depth_loss(hard_depth, target_depths, valid, cfg_train)
        stats.update({
            "hard_depth_loss": hard_total,
            "hard_depth_local_loss": hard_local,
            "hard_depth_global_loss": hard_global,
        })
        if return_renders:
            stats["rendered_hard_depth"] = hard_depth.detach()
    elif return_renders:
        stats["rendered_hard_depth"] = torch.zeros_like(target_depths)

    if int(global_step) >= int(cfg_train.soft_depth_start_step):
        soft_depth = render_predicted(
            pc=gaussian_pc,
            world_view_transform=render_w2c,
            intrinsics=render_intrinsics,
            bg_color=bg,
            cfg=splatter_cfg,
            render_mode="D",
            detach_xyz=True,
            detach_scale_rotation=True,
        )["depth"]
        soft_total, soft_local, soft_global = _dngaussian_depth_loss(soft_depth, target_depths, valid, cfg_train)
        stats.update({
            "soft_depth_loss": soft_total,
            "soft_depth_local_loss": soft_local,
            "soft_depth_global_loss": soft_global,
        })
        if return_renders:
            stats["rendered_soft_depth"] = soft_depth.detach()
    elif return_renders:
        stats["rendered_soft_depth"] = torch.zeros_like(target_depths)

    if return_renders:
        stats["target_depths"] = target_depths.detach()
        stats["target_depth_valid"] = valid.detach()
    return stats


def _gaussian_stats(pc: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    valid = pc.get("valid_mask", None)
    device = pc["xyz"].device
    if valid is None:
        valid = torch.ones(pc["xyz"].shape[:2], device=device, dtype=torch.bool)
    opacity = pc["opacity"]
    return {
        "final_gaussian_count": valid.float().sum(dim=1).mean(),
        "mean_opacity": _masked_mean(opacity, valid),
        "valid_gaussian_ratio": valid.float().mean(),
    }


def _render_selected_sources_to_views(
    vae: SplatterVAE,
    splatter_to_gaussians: DirectSplatterToGaussians,
    splatter_cfg: SplatterConfig,
    z_inv_source: torch.Tensor,
    z_dep_source: torch.Tensor,
    source_indices: torch.Tensor,
    target_indices: torch.Tensor,
    intrinsics: torch.Tensor,
    c2w: torch.Tensor,
    w2c: torch.Tensor,
    bg: torch.Tensor,
    depths: Optional[torch.Tensor],
    cfg_train: TrainConfig,
    global_step: int,
    return_renders: bool,
) -> tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor], Dict[str, Any]]:
    source_intrinsics = _gather_camera_rows(intrinsics, source_indices)
    source_c2w = _gather_camera_rows(c2w, source_indices)

    splatter_map = vae.decode(z_inv_source.contiguous(), z_dep_source.contiguous())
    gaussian_pc = splatter_to_gaussians(
        splatter_map=splatter_map,
        source_cameras_view_to_world=source_c2w,
        intrinsics=source_intrinsics,
        activate_output=True,
    )

    source_w2c = _gather_camera_rows(w2c, source_indices).unsqueeze(1)
    target_w2c = _gather_target_cameras(w2c, target_indices)
    render_w2c = torch.cat((source_w2c, target_w2c), dim=1)

    target_intrinsics = _gather_target_cameras(intrinsics, target_indices)
    render_intrinsics = torch.cat((source_intrinsics.unsqueeze(1), target_intrinsics), dim=1)

    rgb_out = render_predicted(
        pc=gaussian_pc,
        world_view_transform=render_w2c,
        intrinsics=render_intrinsics,
        bg_color=bg,
        cfg=splatter_cfg,
        render_mode="RGB",
    )
    source_view_indices = torch.zeros(source_indices.shape[0], device=source_indices.device, dtype=torch.long)
    stats: Dict[str, Any] = _compute_soft_image_region_penalty(
        xyz_world=gaussian_pc["xyz"],
        world_view_transform=render_w2c,
        intrinsics=render_intrinsics,
        img_h=splatter_cfg.data.img_height,
        img_w=splatter_cfg.data.img_width,
        min_depth=splatter_cfg.data.znear,
        source_view_indices=source_view_indices,
        gaussian_mask=gaussian_pc.get("valid_mask", None),
    )
    stats.update(_gaussian_stats(gaussian_pc))

    render_indices = torch.cat((source_indices.view(-1, 1), target_indices), dim=1)
    target_depths = _gather_target_cameras(depths, render_indices) if depths is not None else None
    stats.update(
        _compute_depth_regularization(
            gaussian_pc=gaussian_pc,
            render_w2c=render_w2c,
            render_intrinsics=render_intrinsics,
            target_depths=target_depths,
            bg=bg,
            splatter_cfg=splatter_cfg,
            cfg_train=cfg_train,
            global_step=global_step,
            return_renders=return_renders,
        )
    )

    return rgb_out["render"], render_indices, gaussian_pc, stats


def compute_reconstruction_and_renders(
    vae: SplatterVAE,
    splatter_to_gaussians: DirectSplatterToGaussians,
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
    global_step: int = 0,
    return_renders: bool = False,
) -> Dict[str, Any]:
    """Decode one source view, render RGB, and apply DNGaussian depth regularization."""
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
        splatter_to_gaussians=splatter_to_gaussians,
        splatter_cfg=splatter_cfg,
        z_inv_source=z_inv_source,
        z_dep_source=z_dep_source,
        source_indices=source_indices,
        target_indices=target_indices,
        intrinsics=intrinsics,
        c2w=c2w,
        w2c=w2c,
        bg=bg,
        depths=depths,
        cfg_train=cfg_train,
        global_step=global_step,
        return_renders=return_renders,
    )

    target_images = _gather_target_cameras(images_01, render_indices)
    rec_loss = compute_reconstruction_loss(
        predicted=rendered.reshape(-1, *rendered.shape[2:]),
        ground_truth=target_images.reshape(-1, *target_images.shape[2:]),
        ssim_weight=float(cfg_train.ssim_weight),
    )

    out_dict: Dict[str, Any] = {
        "rec_loss": rec_loss,
        "rec_self": rec_loss,
        **stats,
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


def _apply_turbo_colormap(values: torch.Tensor) -> torch.Tensor:
    """Approximate Turbo colormap for normalized tensors in [0, 1]."""
    x = values.clamp(0.0, 1.0)
    r = 0.13572138 + x * (4.61539260 + x * (-42.66032258 + x * (132.13108234 + x * (-152.94239396 + x * 59.28637943))))
    g = 0.09140261 + x * (2.19418839 + x * (4.84296658 + x * (-14.18503333 + x * (4.27729857 + x * 2.82956604))))
    b = 0.10667330 + x * (12.64194608 + x * (-60.58204836 + x * (110.36276771 + x * (-89.90310912 + x * 27.34824973))))
    return torch.cat([r, g, b], dim=-3).clamp(0.0, 1.0)


def _depth_to_vis(depth: torch.Tensor, valid: Optional[torch.Tensor], splatter_cfg: SplatterConfig) -> torch.Tensor:
    znear = float(splatter_cfg.data.znear)
    zfar = float(splatter_cfg.data.zfar)
    normalized = (depth - znear) / max(zfar - znear, 1.0e-6)
    # Near depths are warm/yellow, far depths are blue/purple.
    vis = _apply_turbo_colormap(1.0 - normalized.clamp(0.0, 1.0))
    if valid is not None:
        mask = valid.to(device=depth.device, dtype=torch.bool).expand_as(vis)
        vis = torch.where(mask, vis, torch.zeros_like(vis))
    return vis


def _make_depth_panel(rec_out: Dict[str, Any], splatter_cfg: SplatterConfig, max_vis: int) -> Optional[wandb.Image]:
    target_depths = rec_out.get("target_depths", None)
    if target_depths is None:
        return None
    valid = rec_out.get("target_depth_valid", None)
    hard = rec_out.get("rendered_hard_depth", torch.zeros_like(target_depths))
    soft = rec_out.get("rendered_soft_depth", torch.zeros_like(target_depths))
    num_views = min(int(target_depths.shape[1]), 6)
    items: list[tuple[str, torch.Tensor]] = []
    for view_slot in range(num_views):
        view_name = "source" if view_slot == 0 else f"target{view_slot}"
        view_valid = valid[:, view_slot] if valid is not None else None
        items.append((f"gt_depth_{view_name}", _depth_to_vis(target_depths[:, view_slot], view_valid, splatter_cfg)))
    for view_slot in range(num_views):
        view_name = "source" if view_slot == 0 else f"target{view_slot}"
        view_valid = valid[:, view_slot] if valid is not None else None
        items.append((f"hard_depth_{view_name}", _depth_to_vis(hard[:, view_slot], view_valid, splatter_cfg)))
    for view_slot in range(num_views):
        view_name = "source" if view_slot == 0 else f"target{view_slot}"
        view_valid = valid[:, view_slot] if valid is not None else None
        items.append((f"soft_depth_{view_name}", _depth_to_vis(soft[:, view_slot], view_valid, splatter_cfg)))
    return _make_wandb_named_image_panel(items, max_vis=max_vis)


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


@torch.no_grad()
def validate_and_log_wandb(
    vae: SplatterVAE,
    splatter_cfg: SplatterConfig,
    splatter_to_gaussians: DirectSplatterToGaussians,
    valid_dataloader: DataLoader,
    device: torch.device,
    bg: torch.Tensor,
    cfg_train: TrainConfig,
    global_step: int,
) -> None:
    if wandb.run is None:
        return

    prev_vae_mode = vae.training
    prev_converter_mode = splatter_to_gaussians.training
    vae.eval()
    splatter_to_gaussians.eval()

    scalar_keys = [
        "val/rec_loss",
        "val/hard_depth_loss",
        "val/hard_depth_local_loss",
        "val/hard_depth_global_loss",
        "val/soft_depth_loss",
        "val/soft_depth_local_loss",
        "val/soft_depth_global_loss",
        "val/depth_valid_ratio",
        "val/inv_contrastive_loss",
        "val/inv_consistency_loss",
        "val/dep_contrastive_loss",
        "val/dep_consistency_loss",
        "val/frustum_loss",
        "val/final_gaussian_count",
        "val/mean_opacity",
        "val/valid_gaussian_ratio",
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
            global_step=global_step,
            return_renders=(num_eval_batches == 0),
        )

        metric_map = {
            "val/rec_loss": rec_out["rec_loss"],
            "val/hard_depth_loss": rec_out["hard_depth_loss"],
            "val/hard_depth_local_loss": rec_out["hard_depth_local_loss"],
            "val/hard_depth_global_loss": rec_out["hard_depth_global_loss"],
            "val/soft_depth_loss": rec_out["soft_depth_loss"],
            "val/soft_depth_local_loss": rec_out["soft_depth_local_loss"],
            "val/soft_depth_global_loss": rec_out["soft_depth_global_loss"],
            "val/depth_valid_ratio": rec_out["depth_valid_ratio"],
            "val/inv_contrastive_loss": inv_contrastive_loss,
            "val/inv_consistency_loss": inv_consistency_loss,
            "val/dep_contrastive_loss": dep_contrastive_loss,
            "val/dep_consistency_loss": dep_consistency_loss,
            "val/frustum_loss": rec_out["frustum_loss"],
            "val/final_gaussian_count": rec_out["final_gaussian_count"],
            "val/mean_opacity": rec_out["mean_opacity"],
            "val/valid_gaussian_ratio": rec_out["valid_gaussian_ratio"],
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
            image_payload["val/render_summary"] = _make_wandb_named_image_panel(panel_items, max_vis=cfg_train.val_max_vis)
            depth_panel = _make_depth_panel(rec_out, splatter_cfg=splatter_cfg, max_vis=cfg_train.val_max_vis)
            if depth_panel is not None:
                image_payload["val/depth_summary"] = depth_panel
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
        splatter_to_gaussians.train(prev_converter_mode)
        return

    log_dict: Dict[str, Any] = {key: value / float(num_eval_batches) for key, value in scalar_sums.items()}
    log_dict["global_step"] = global_step
    log_dict.update(image_payload)
    wandb.log(log_dict, step=global_step)

    vae.train(prev_vae_mode)
    splatter_to_gaussians.train(prev_converter_mode)


def _build_converter(splatter_cfg: SplatterConfig, device: torch.device) -> DirectSplatterToGaussians:
    return DirectSplatterToGaussians(splatter_cfg).to(device)


def train_splatter_vae(
    vae: SplatterVAE,
    splatter_cfg: SplatterConfig,
    train_dataloader: DataLoader,
    cfg_train: TrainConfig,
    valid_dataloader: Optional[DataLoader] = None,
    resume_ckpt: Optional[str] = None,
):
    """Train SplatterVAE with direct Gaussian rendering and DNGaussian depth regularization."""
    device = torch.device(cfg_train.device)
    vae.to(device)
    splatter_to_gaussians = _build_converter(splatter_cfg, device)
    optimizer = torch.optim.Adam(list(vae.parameters()), lr=cfg_train.lr)

    lr_total_steps = _resolve_lr_total_steps(cfg_train, train_dataloader)
    lr_schedule = _normalize_lr_schedule(cfg_train.lr_schedule)
    if lr_schedule != "constant":
        print(
            f"[LR] schedule={lr_schedule}, peak_lr={cfg_train.lr:g}, min_lr={cfg_train.min_lr:g}, "
            f"warmup_steps={cfg_train.lr_warmup_steps}, total_steps={lr_total_steps}"
        )
    print(
        "[DNGaussian Depth] "
        f"hard_start={cfg_train.hard_depth_start_step}, soft_start={cfg_train.soft_depth_start_step}, "
        f"local_w={cfg_train.depth_local_weight:g}, global_w={cfg_train.depth_global_weight:g}, "
        f"margin={cfg_train.depth_error_tolerance:g}"
    )

    bg = torch.ones(3, device=device) if splatter_cfg.data.white_background else torch.zeros(3, device=device)
    start_epoch = 0
    global_step = 0

    if resume_ckpt is not None and os.path.isfile(resume_ckpt):
        ckpt = torch.load(resume_ckpt, map_location="cpu")
        vae.load_state_dict(ckpt["vae_state_dict"])
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
                global_step=global_step,
                return_renders=False,
            )
            rec_loss = rec_out["rec_loss"]
            hard_depth_loss = rec_out["hard_depth_loss"]
            soft_depth_loss = rec_out["soft_depth_loss"]
            frustum_loss = rec_out["frustum_loss"]

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
                + cfg_train.hard_depth_weight * hard_depth_loss
                + cfg_train.soft_depth_weight * soft_depth_loss
                + cfg_train.vq_weight * vq_loss
                + cfg_train.inv_contrastive_weight * inv_contrastive_loss
                + cfg_train.inv_consistency_weight * inv_consistency_loss
                + cfg_train.dep_contrastive_weight * dep_contrastive_loss
                + cfg_train.dep_consistency_weight * dep_consistency_loss
                + cfg_train.frustum_weight * frustum_loss
            )

            finite_terms = {
                "rec_loss": rec_loss,
                "hard_depth_loss": hard_depth_loss,
                "soft_depth_loss": soft_depth_loss,
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
            torch.nn.utils.clip_grad_norm_(list(vae.parameters()), max_norm=5.0)
            optimizer.step()

            if step % 250 == 0:
                print(
                    f"[Epoch {epoch + 1} | Step {step} | Global {global_step}] "
                    f"Loss={total_loss.item():.4f} lr={current_lr:.2e} "
                    f"(rgb={rec_loss.item():.4f}, rgb_w={cfg_train.rec_weight:.4f}, "
                    f"hard_d={hard_depth_loss.item():.4f}, soft_d={soft_depth_loss.item():.4f}, "
                    f"vq={vq_loss.item():.4f}, inv_con={inv_contrastive_loss.item():.4f}, "
                    f"dep_con={dep_contrastive_loss.item():.4f}, frustum={frustum_loss.item():.4f}, "
                    f"gauss={rec_out['final_gaussian_count'].item():.1f})"
                )
                if wandb.run is not None:
                    wandb.log(
                        {
                            "train/total_loss": total_loss.item(),
                            "train/lr": current_lr,
                            "train/rec_loss": rec_loss.item(),
                            "train/rec_weight": float(cfg_train.rec_weight),
                            "train/rec_loss_weighted": cfg_train.rec_weight * rec_loss.item(),
                            "train/hard_depth_loss": hard_depth_loss.item(),
                            "train/hard_depth_loss_weighted": cfg_train.hard_depth_weight * hard_depth_loss.item(),
                            "train/hard_depth_local_loss": rec_out["hard_depth_local_loss"].item(),
                            "train/hard_depth_global_loss": rec_out["hard_depth_global_loss"].item(),
                            "train/soft_depth_loss": soft_depth_loss.item(),
                            "train/soft_depth_loss_weighted": cfg_train.soft_depth_weight * soft_depth_loss.item(),
                            "train/soft_depth_local_loss": rec_out["soft_depth_local_loss"].item(),
                            "train/soft_depth_global_loss": rec_out["soft_depth_global_loss"].item(),
                            "train/depth_valid_ratio": rec_out["depth_valid_ratio"].item(),
                            "train/vq_loss": vq_loss.item(),
                            "train/inv_vq_loss": inv_vq_loss.item(),
                            "train/dep_vq_loss": dep_vq_loss.item(),
                            "train/inv_contrastive_loss": inv_contrastive_loss.item(),
                            "train/inv_consistency_loss": inv_consistency_loss.item(),
                            "train/dep_contrastive_loss": dep_contrastive_loss.item(),
                            "train/dep_consistency_loss": dep_consistency_loss.item(),
                            "train/frustum_loss": frustum_loss.item(),
                            "train/frustum_loss_weighted": cfg_train.frustum_weight * frustum_loss.item(),
                            "train/final_gaussian_count": rec_out["final_gaussian_count"].item(),
                            "train/mean_opacity": rec_out["mean_opacity"].item(),
                            "train/valid_gaussian_ratio": rec_out["valid_gaussian_ratio"].item(),
                            "train/inactive_pct_mean": 100.0 * rec_out["inactive_ratio_mean"].item(),
                            "train/inactive_pct_src": 100.0 * rec_out["inactive_ratio_src"].item(),
                            "train/inactive_pct_tgt": 100.0 * rec_out["inactive_ratio_tgt"].item(),
                            "train/invalid_depth_pct_mean": 100.0 * rec_out["invalid_depth_ratio_mean"].item(),
                            "train/nonfinite_projection_pct_mean": 100.0 * rec_out["nonfinite_projection_ratio_mean"].item(),
                            "train/hard_depth_weight": float(cfg_train.hard_depth_weight),
                            "train/soft_depth_weight": float(cfg_train.soft_depth_weight),
                            "train/frustum_weight": float(cfg_train.frustum_weight),
                            "global_step": global_step,
                        },
                        step=global_step,
                    )

            if valid_dataloader is not None and cfg_train.eval_every > 0 and global_step > 0 and global_step % cfg_train.eval_every == 0:
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
                    "optimizer_state_dict": optimizer.state_dict(),
                }
                torch.save(ckpt, ckpt_path)
                print(f"[Checkpoint] Saved checkpoint to {ckpt_path}")

            global_step += 1

        epoch += 1
        if cfg_train.max_global_steps is None and epoch >= cfg_train.num_epochs:
            print(f"[Stop] Reached num_epochs={cfg_train.num_epochs}.")
            break
