from __future__ import annotations

from typing import Any, Dict, Optional

import torch

from models.losses import compute_reconstruction_loss
from models.splatter import SplatterConfig, render_predicted
from models.splatter_gaussians import DirectSplatterToGaussians
from models.splatter_point_losses import compute_point_losses
from models.splatter_train_config import TrainConfig
from models.vae import SplatterVAE
from utils.camera_tensor_utils import (
    gather_camera_rows,
    gather_target_cameras,
    mask_or_ones,
    target_indices_excluding_source,
)


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


def apply_source_mask_to_gaussians(
    pc: Dict[str, torch.Tensor],
    source_masks: Optional[torch.Tensor],
) -> Dict[str, torch.Tensor]:
    if source_masks is None:
        return pc
    mask = source_masks.to(device=pc["xyz"].device, dtype=torch.bool)
    if mask.dim() != 4 or mask.shape[1] != 1:
        raise ValueError(f"Expected source masks as (B,1,H,W), got {tuple(mask.shape)}.")
    flat_mask = mask.flatten(start_dim=2).squeeze(1)
    num_gaussians = pc["xyz"].shape[1]
    if flat_mask.shape[1] != num_gaussians:
        if num_gaussians % flat_mask.shape[1] != 0:
            raise ValueError(
                f"Cannot align {flat_mask.shape[1]} source mask pixels with {num_gaussians} Gaussians."
            )
        repeat = num_gaussians // flat_mask.shape[1]
        flat_mask = flat_mask.repeat_interleave(repeat, dim=1)
    pc = dict(pc)
    valid = pc.get("valid_mask", torch.ones_like(flat_mask, dtype=torch.bool)) & flat_mask
    pc["valid_mask"] = valid.contiguous()
    pc["opacity"] = pc["opacity"] * flat_mask.unsqueeze(-1).to(dtype=pc["opacity"].dtype)
    return pc


def render_selected_sources_to_views(
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
    masks: Optional[torch.Tensor] = None,
) -> tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
    source_intrinsics = gather_camera_rows(intrinsics, source_indices)
    source_c2w = gather_camera_rows(c2w, source_indices)

    splatter_map = vae.decode(z_inv_source.contiguous(), z_dep_source.contiguous())
    gaussian_pc = splatter_to_gaussians(
        splatter_map=splatter_map,
        source_cameras_view_to_world=source_c2w,
        intrinsics=source_intrinsics,
        activate_output=True,
    )
    source_masks = gather_camera_rows(masks, source_indices) if masks is not None else None
    gaussian_pc = apply_source_mask_to_gaussians(gaussian_pc, source_masks)

    source_w2c = gather_camera_rows(w2c, source_indices).unsqueeze(1)
    target_w2c = gather_target_cameras(w2c, target_indices)
    render_w2c = torch.cat((source_w2c, target_w2c), dim=1)

    target_intrinsics = gather_target_cameras(intrinsics, target_indices)
    render_intrinsics = torch.cat((source_intrinsics.unsqueeze(1), target_intrinsics), dim=1)

    out = render_predicted(
        pc=gaussian_pc,
        world_view_transform=render_w2c,
        intrinsics=render_intrinsics,
        bg_color=bg,
        cfg=splatter_cfg,
        render_mode="RGB",
    )
    valid_mask = gaussian_pc.get(
        "valid_mask",
        torch.ones(gaussian_pc["xyz"].shape[:2], device=gaussian_pc["xyz"].device, dtype=torch.bool),
    ).to(dtype=torch.bool)
    opacity = gaussian_pc["opacity"]
    opacity_mask = valid_mask.unsqueeze(-1).expand_as(opacity)
    if bool(opacity_mask.any()):
        mean_opacity = opacity.masked_select(opacity_mask).mean()
    else:
        mean_opacity = opacity.new_zeros(())
    stats = {"mean_opacity": mean_opacity}

    render_indices = torch.cat((source_indices.view(-1, 1), target_indices), dim=1)
    return out["render"], render_indices, gaussian_pc, stats


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
    masks: Optional[torch.Tensor] = None,
    return_renders: bool = False,
) -> Dict[str, Any]:
    """Decode one source view, render RGB, and supervise Gaussian centers with masked depth point clouds."""
    bsz, num_views = z_inv.shape[:2]
    if num_views < 2:
        raise ValueError("Source-plus-target reconstruction requires at least two camera viewpoints.")
    device = z_inv.device
    batch_ids = torch.arange(bsz, device=device)

    source_indices = torch.randint(low=0, high=num_views, size=(bsz,), device=device)
    target_indices = target_indices_excluding_source(source_indices, num_views)
    z_inv_source = z_inv[batch_ids, source_indices]
    z_dep_source = z_dep[batch_ids, source_indices]

    rendered, render_indices, gaussian_pc, stats = render_selected_sources_to_views(
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
        masks=masks,
    )

    all_masks = mask_or_ones(images_01, masks)
    target_images = gather_target_cameras(images_01, render_indices)
    target_masks = gather_target_cameras(all_masks, render_indices)
    target_images = target_images * target_masks

    background_weight = max(0.0, float(cfg_train.rec_background_weight))
    rec_pixel_weights = target_masks + (1.0 - target_masks) * background_weight
    rec_loss = compute_reconstruction_loss(
        predicted=rendered.reshape(-1, *rendered.shape[2:]),
        ground_truth=target_images.reshape(-1, *target_images.shape[2:]),
        ssim_weight=float(cfg_train.ssim_weight),
        pixel_weights=rec_pixel_weights.reshape(-1, *rec_pixel_weights.shape[2:]),
    )
    point_stats = compute_point_losses(
        pc=gaussian_pc,
        depths=depths,
        intrinsics=intrinsics,
        c2w=c2w,
        splatter_cfg=splatter_cfg,
        cfg_train=cfg_train,
        masks=masks,
    )

    out_dict: Dict[str, Any] = {
        "rec_loss": rec_loss,
        "rec_self": rec_loss,
        "mask_pixel_ratio": target_masks.mean(),
        **stats,
        **point_stats,
    }

    if return_renders:
        out_dict["target_images_self"] = target_images
        out_dict["target_masks_self"] = target_masks
        out_dict["rendered_self"] = rendered
        out_dict["source_indices"] = source_indices.detach().cpu()
        out_dict["target_indices"] = target_indices.detach().cpu()
        out_dict["render_indices"] = render_indices.detach().cpu()
        out_dict["gaussian_pc"] = {k: v.detach() for k, v in gaussian_pc.items() if torch.is_tensor(v)}
        out_dict["source_c2w"] = gather_camera_rows(c2w, source_indices).detach()
        out_dict["source_intrinsics"] = gather_camera_rows(intrinsics, source_indices).detach()
    return out_dict
