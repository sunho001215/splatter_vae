from __future__ import annotations

from typing import Any, Dict, Optional

import torch
import torch.nn.functional as F

from models.losses import compute_reconstruction_loss
from models.splatter import SplatterConfig, render_predicted
from models.gaussians import DirectSplatterToGaussians
from models.point_losses import compute_temporal_point_losses
from models.train_config import TrainConfig
from models.vae import SplatterVAE
from utils.camera_tensor_utils import gather_camera_rows


def encode_per_view_sequence_batch(
    vae: SplatterVAE,
    images: torch.Tensor,
) -> tuple[Dict[str, torch.Tensor], torch.Tensor, torch.Tensor]:
    """Encode every sample/view sequence as an independent contrastive item.

    Returns ``s_inv_by_view`` and ``z_dep_by_view`` with shape ``(B,A,D)`` so
    the caller can define positives by batch/sample id or by camera id.
    """
    if images.dim() == 6:
        bsz, timesteps, num_views, channels, height, width = images.shape
        view_sequences = images.permute(0, 2, 1, 3, 4, 5).reshape(
            bsz * num_views,
            timesteps,
            1,
            channels,
            height,
            width,
        )
    elif images.dim() == 5:
        bsz, num_views, channels, height, width = images.shape
        view_sequences = images.reshape(bsz * num_views, 1, channels, height, width)
    else:
        raise ValueError(f"Expected images as (B,T,A,3,H,W) or legacy (B,A,3,H,W), got {tuple(images.shape)}.")

    latents, inv_loss, dep_loss = vae.encode_sequence(view_sequences)
    s_inv_by_view = latents["s_inv"].view(bsz, num_views, -1).contiguous()
    z_dep_by_view = latents["z_dep_all"][:, 0].view(bsz, num_views, -1).contiguous()
    out = {
        "s_inv_by_view": s_inv_by_view,
        "z_dep_by_view": z_dep_by_view,
        "inv_mask_by_view": latents["inv_mask"].view(bsz, num_views, *latents["inv_mask"].shape[1:]).contiguous(),
        "dep_mask_by_view": latents["dep_mask"].view(bsz, num_views, *latents["dep_mask"].shape[1:]).contiguous(),
    }
    return out, inv_loss, dep_loss


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


def _ensure_temporal_images(images_01: torch.Tensor) -> torch.Tensor:
    if images_01.dim() == 5:
        return images_01[:, None].contiguous()
    if images_01.dim() != 6:
        raise ValueError(f"Expected images_01 as (B,T,A,3,H,W), got {tuple(images_01.shape)}.")
    return images_01


def _ensure_temporal_optional(value: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
    if value is None:
        return None
    if value.dim() == 5:
        return value[:, None].contiguous()
    if value.dim() != 6:
        raise ValueError(f"Expected optional temporal tensor as (B,T,A,C,H,W), got {tuple(value.shape)}.")
    return value


def _expand_camera_time(camera: torch.Tensor, timesteps: int) -> torch.Tensor:
    if camera.dim() == 4:
        return camera[:, None].expand(-1, timesteps, -1, -1, -1).contiguous()
    if camera.dim() == 5:
        if camera.shape[1] != timesteps:
            raise ValueError(f"Camera tensor has T={camera.shape[1]}, expected {timesteps}.")
        return camera
    raise ValueError(f"Expected camera tensor as (B,A,...) or (B,T,A,...), got {tuple(camera.shape)}.")


def _gather_source_time0(values: torch.Tensor, source_indices: torch.Tensor) -> torch.Tensor:
    if values.dim() >= 5:
        values = values[:, 0]
    return gather_camera_rows(values, source_indices)


def _gather_source_sequence(values: torch.Tensor, source_indices: torch.Tensor) -> torch.Tensor:
    if values.dim() == 5:
        return gather_camera_rows(values, source_indices)
    if values.dim() != 6:
        raise ValueError(f"Expected temporal values as (B,T,A,C,H,W), got {tuple(values.shape)}.")
    batch_ids = torch.arange(values.shape[0], device=values.device)
    return values[batch_ids, :, source_indices].contiguous()


def _delta_map_to_xyz(delta_map: torch.Tensor, gaussians_per_pixel: int) -> torch.Tensor:
    batch, channels, height, width = delta_map.shape
    expected = 3 * int(gaussians_per_pixel)
    if channels != expected:
        raise ValueError(f"Expected {expected} delta channels, got {channels}.")
    delta = delta_map.view(batch, gaussians_per_pixel, 3, height, width)
    delta = delta.permute(0, 3, 4, 1, 2).reshape(batch, height * width * gaussians_per_pixel, 3)
    return delta.contiguous()


def apply_xyz_delta(
    pc: Dict[str, torch.Tensor],
    delta_map: torch.Tensor,
    gaussians_per_pixel: int,
) -> Dict[str, torch.Tensor]:
    delta_xyz = _delta_map_to_xyz(delta_map, gaussians_per_pixel)
    if delta_xyz.shape != pc["xyz"].shape:
        raise ValueError(f"Delta xyz shape {tuple(delta_xyz.shape)} does not match pc xyz {tuple(pc['xyz'].shape)}.")
    out = dict(pc)
    out["xyz"] = (pc["xyz"] + delta_xyz).contiguous()
    out["delta_xyz"] = delta_xyz.contiguous()
    return out


def _render_sequence(
    pc_sequence: list[Dict[str, torch.Tensor]],
    w2c: torch.Tensor,
    intrinsics: torch.Tensor,
    bg: torch.Tensor,
    splatter_cfg: SplatterConfig,
    occupancy_opacity: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    renders = []
    occupancy_alphas = []
    timesteps = w2c.shape[1]
    for time_idx in range(timesteps):
        pc = pc_sequence[min(time_idx, len(pc_sequence) - 1)]
        out = render_predicted(
            pc=pc,
            world_view_transform=w2c[:, time_idx],
            intrinsics=intrinsics[:, time_idx],
            bg_color=bg,
            cfg=splatter_cfg,
            occupancy_opacity=occupancy_opacity,
            render_mode="RGB",
        )
        renders.append(out["render"])
        occupancy_alpha = out.get("occupancy_alpha", None)
        if occupancy_alpha is None:
            raise RuntimeError("render_predicted did not return occupancy_alpha while occupancy_opacity was set.")
        occupancy_alphas.append(occupancy_alpha)
    return torch.stack(renders, dim=1).contiguous(), torch.stack(occupancy_alphas, dim=1).contiguous()


def _mask_or_ones_sequence(images_01: torch.Tensor, masks: Optional[torch.Tensor]) -> torch.Tensor:
    if masks is None:
        return torch.ones((*images_01.shape[:3], 1, *images_01.shape[-2:]), device=images_01.device, dtype=images_01.dtype)
    return masks.to(device=images_01.device, dtype=images_01.dtype).clamp(0.0, 1.0)


def _expand_mask_for_image_loss(mask: torch.Tensor, dilation_pixels: int) -> torch.Tensor:
    radius = max(0, int(dilation_pixels))
    if radius == 0:
        return mask
    if mask.shape[-3] != 1:
        raise ValueError(f"Expected single-channel masks, got {tuple(mask.shape)}.")
    flat = mask.reshape(-1, 1, *mask.shape[-2:])
    expanded = F.max_pool2d(flat, kernel_size=2 * radius + 1, stride=1, padding=radius)
    return expanded.reshape_as(mask).clamp(0.0, 1.0)


def _mean_valid_opacity(pc: Dict[str, torch.Tensor]) -> torch.Tensor:
    valid_mask = pc.get(
        "valid_mask",
        torch.ones(pc["xyz"].shape[:2], device=pc["xyz"].device, dtype=torch.bool),
    ).to(dtype=torch.bool)
    opacity = pc["opacity"]
    opacity_mask = valid_mask.unsqueeze(-1).expand_as(opacity)
    if bool(opacity_mask.any()):
        return opacity.masked_select(opacity_mask).mean()
    return opacity.new_zeros(())


def compute_reconstruction_and_renders(
    vae: SplatterVAE,
    splatter_to_gaussians: DirectSplatterToGaussians,
    splatter_cfg: SplatterConfig,
    images_01: torch.Tensor,
    intrinsics: torch.Tensor,
    c2w: torch.Tensor,
    w2c: torch.Tensor,
    bg: torch.Tensor,
    cfg_train: TrainConfig,
    depths: Optional[torch.Tensor] = None,
    masks: Optional[torch.Tensor] = None,
    return_renders: bool = False,
) -> Dict[str, Any]:
    """Encode one sampled source-view sequence, decode it, and render all views."""
    images_01 = _ensure_temporal_images(images_01)
    depths = _ensure_temporal_optional(depths)
    masks = _ensure_temporal_optional(masks)
    bsz, timesteps, num_views = images_01.shape[:3]
    if num_views < 1:
        raise ValueError("Temporal reconstruction requires at least one camera viewpoint.")

    intrinsics_t = _expand_camera_time(intrinsics, timesteps)
    c2w_t = _expand_camera_time(c2w, timesteps)
    w2c_t = _expand_camera_time(w2c, timesteps)

    device = images_01.device
    source_indices = torch.randint(low=0, high=num_views, size=(bsz,), device=device)

    source_images_01 = _gather_source_sequence(images_01, source_indices)
    source_images = source_images_01.mul(2.0).sub(1.0).unsqueeze(2).contiguous()
    source_latents, _inv_embed_loss, _dep_embed_loss = vae.encode_sequence(source_images)
    s_inv = source_latents["s_inv"]
    z_dep_source = source_latents["z_dep_all"][:, 0]

    decoded = vae.decode_sequence(s_inv=s_inv, z_dep_source=z_dep_source)
    source_intrinsics = _gather_source_time0(intrinsics_t, source_indices)
    source_c2w = _gather_source_time0(c2w_t, source_indices)
    pc0 = splatter_to_gaussians(
        splatter_map=decoded["base_map"],
        source_cameras_view_to_world=source_c2w,
        intrinsics=source_intrinsics,
        activate_output=True,
    )
    source_masks = _gather_source_time0(masks, source_indices) if masks is not None else None
    pc0 = apply_source_mask_to_gaussians(pc0, source_masks)
    pc1 = apply_xyz_delta(pc0, decoded["delta01_map"], vae.gaussians_per_pixel)
    pc2 = apply_xyz_delta(pc1, decoded["delta12_map"], vae.gaussians_per_pixel)
    pc_sequence = [pc0, pc1, pc2]

    rendered, occupancy_alpha = _render_sequence(
        pc_sequence=pc_sequence,
        w2c=w2c_t,
        intrinsics=intrinsics_t,
        bg=bg,
        splatter_cfg=splatter_cfg,
        occupancy_opacity=float(cfg_train.occupancy_fixed_opacity),
    )
    original_masks = _mask_or_ones_sequence(images_01, masks)
    expanded_masks = _expand_mask_for_image_loss(
        original_masks,
        dilation_pixels=int(cfg_train.rgb_loss_mask_dilation),
    )
    target_images = images_01 * original_masks
    safe_background_mask = 1.0 - expanded_masks

    rec_loss = compute_reconstruction_loss(
        predicted=rendered.reshape(-1, *rendered.shape[3:]),
        ground_truth=target_images.reshape(-1, *target_images.shape[3:]),
        ssim_weight=float(cfg_train.ssim_weight),
        loss_mask=expanded_masks.reshape(-1, *expanded_masks.shape[3:]),
    )
    occupancy_loss = (occupancy_alpha * safe_background_mask).mean()
    point_stats = compute_temporal_point_losses(
        pc_sequence=pc_sequence[:timesteps],
        depths=depths,
        intrinsics=intrinsics_t,
        c2w=c2w_t,
        splatter_cfg=splatter_cfg,
        cfg_train=cfg_train,
        masks=masks,
    )
    delta_smooth_loss = decoded["delta01_map"].abs().mean() + decoded["delta12_map"].abs().mean()
    delta01_xyz = _delta_map_to_xyz(decoded["delta01_map"], vae.gaussians_per_pixel)
    delta12_xyz = _delta_map_to_xyz(decoded["delta12_map"], vae.gaussians_per_pixel)
    delta_magnitude = torch.stack(
        [delta01_xyz.norm(dim=-1), delta12_xyz.norm(dim=-1)],
        dim=1,
    )

    out_dict: Dict[str, Any] = {
        "rec_loss": rec_loss,
        "rec_self": rec_loss,
        "occupancy_loss": occupancy_loss,
        "mask_pixel_ratio": original_masks.mean(),
        "expanded_mask_pixel_ratio": expanded_masks.mean(),
        "mean_opacity": _mean_valid_opacity(pc0),
        "delta_smooth_loss": delta_smooth_loss,
        "delta01_mean": delta01_xyz.abs().mean(),
        "delta12_mean": delta12_xyz.abs().mean(),
        "delta_magnitude_mean": delta_magnitude.mean(),
        **point_stats,
    }

    if return_renders:
        out_dict["target_images_self"] = target_images
        out_dict["target_masks_self"] = original_masks
        out_dict["expanded_masks_self"] = expanded_masks
        out_dict["rendered_self"] = rendered
        out_dict["occupancy_alpha_self"] = occupancy_alpha
        out_dict["source_indices"] = source_indices.detach().cpu()
        out_dict["gaussian_pc"] = {k: v.detach() for k, v in pc0.items() if torch.is_tensor(v)}
        out_dict["gaussian_pc_sequence"] = [
            {k: v.detach() for k, v in pc.items() if torch.is_tensor(v)} for pc in pc_sequence
        ]
        out_dict["source_c2w"] = source_c2w.detach()
        out_dict["source_intrinsics"] = source_intrinsics.detach()
        out_dict["base_map"] = decoded["base_map"].detach()
        out_dict["delta01_map"] = decoded["delta01_map"].detach()
        out_dict["delta12_map"] = decoded["delta12_map"].detach()
        out_dict["delta_magnitude"] = delta_magnitude.detach()
    return out_dict
