from __future__ import annotations

import random
from typing import Any, Dict

import torch

from models.gaussian.motion import activate_motion_map, render_translation_flow_sequence, translate_gaussians
from models.gaussian.parameterization import DirectSplatterToGaussians, SplatterConfig
from models.gaussian.rendering import render_rgb_depth
from models.splattervae.config import TEMPORAL_WINDOW
from models.splattervae.model import SplatterVAE
from models.training.config import TrainConfig
from models.training.losses import (
    build_target_dynamic_scores,
    compute_balanced_silhouette_loss,
    compute_global_local_depth_loss,
    compute_optical_flow_loss,
    compute_reconstruction_loss,
    compute_union_frustum_loss,
)
from utils.camera_tensor_utils import gather_camera_rows


def encode_per_view_sequence_batch(
    vae: SplatterVAE,
    images: torch.Tensor,
    optical_flows: torch.Tensor,
) -> Dict[str, torch.Tensor]:
    """Run the one allowed encoder pass over independent camera sequences."""
    if images.dim() != 6 or images.shape[1] != TEMPORAL_WINDOW:
        raise ValueError(f"Expected images as (B,3,A,3,H,W), got {tuple(images.shape)}.")
    batch, timesteps, views, channels, height, width = images.shape
    expected_flow = (batch, 3, views, 2, height, width)
    if tuple(optical_flows.shape) != expected_flow:
        raise ValueError(f"Expected optical flows as {expected_flow}, got {tuple(optical_flows.shape)}.")
    view_sequences = images.permute(0, 2, 1, 3, 4, 5).reshape(
        batch * views, timesteps, 1, channels, height, width
    )
    view_flows = optical_flows.permute(0, 2, 1, 3, 4, 5).reshape(
        batch * views, 3, 1, 2, height, width
    )
    latents = vae.encode_pretraining(view_sequences, view_flows)
    return {
        "s_inv_by_view": latents["s_inv"].view(batch, views, -1).contiguous(),
        "z_dep_by_view": latents["z_dep_all"][:, 0].view(batch, views, -1).contiguous(),
        "inv_mask_by_view": latents["inv_mask"][:, 0].view(batch, views, -1).contiguous(),
        "dep_mask_by_view": latents["dep_mask"][:, 0].view(batch, views, -1).contiguous(),
    }


def apply_source_mask_to_gaussians(
    pc: Dict[str, torch.Tensor], source_masks: torch.Tensor
) -> Dict[str, torch.Tensor]:
    mask = source_masks.bool()
    if mask.dim() != 4 or mask.shape[1] != 1:
        raise ValueError(f"Expected source mask as (B,1,H,W), got {tuple(mask.shape)}.")
    flat = mask.flatten(2).squeeze(1)
    if pc["xyz"].shape[1] % flat.shape[1] != 0:
        raise ValueError("Source segmentation pixels do not align with dense Gaussians.")
    flat = flat.repeat_interleave(pc["xyz"].shape[1] // flat.shape[1], dim=1)
    output = dict(pc)
    output["valid_mask"] = pc["valid_mask"] & flat
    output["opacity"] = pc["opacity"] * flat[..., None].to(pc["opacity"].dtype)
    return output


def _expand_camera_time(camera: torch.Tensor) -> torch.Tensor:
    if camera.dim() == 4:
        return camera[:, None].expand(-1, TEMPORAL_WINDOW, -1, -1, -1).contiguous()
    if camera.dim() == 5 and camera.shape[1] == TEMPORAL_WINDOW:
        return camera
    raise ValueError(f"Expected camera tensor with or without T=3, got {tuple(camera.shape)}.")


def _gather_source_time0(values: torch.Tensor, source_indices: torch.Tensor) -> torch.Tensor:
    return gather_camera_rows(values[:, 0], source_indices)


def _render_sequence(
    pc_sequence: list[Dict[str, torch.Tensor]],
    w2c: torch.Tensor,
    intrinsics: torch.Tensor,
    background: torch.Tensor,
    splatter_cfg: SplatterConfig,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    keys = ("xyz", "scaling", "rotation", "opacity", "features_dc", "features_rest", "valid_mask")
    stacked = {key: torch.stack([pc[key] for pc in pc_sequence], 1).contiguous() for key in keys}
    result = render_rgb_depth(stacked, w2c, intrinsics, background, splatter_cfg)
    return result["render"], result["depth"], result["alpha"]


def _local_depth_patch_size(cfg: TrainConfig, training: bool) -> int:
    minimum = int(cfg.local_depth_min_patch_size)
    maximum = int(cfg.local_depth_max_patch_size)
    if minimum <= 0 or maximum < minimum:
        raise ValueError("Invalid local depth patch-size range.")
    return random.randint(minimum, maximum) if training else int(round((minimum + maximum) / 2.0))


def _weighted_mean(values: torch.Tensor, weights: torch.Tensor, time_idx: int) -> torch.Tensor:
    selected_values = values[:, time_idx]
    selected_weights = weights[:, time_idx].to(values.dtype)
    return (selected_values * selected_weights).sum() / selected_weights.sum().clamp_min(1.0)


def _timestep_render_losses(
    rendered: torch.Tensor,
    rendered_depth: torch.Tensor,
    rendered_alpha: torch.Tensor,
    target_images: torch.Tensor,
    target_depths: torch.Tensor,
    target_masks: torch.Tensor,
    dynamic_scores: torch.Tensor,
    cfg: TrainConfig,
    patch_size: int,
) -> list[Dict[str, torch.Tensor]]:
    batch, timesteps, views = rendered.shape[:3]
    flat_rendered = rendered.flatten(0, 2)
    flat_target = target_images.flatten(0, 2)
    flat_mask = target_masks.flatten(0, 2)
    flat_depth = rendered_depth.flatten(0, 2)
    flat_target_depth = target_depths.flatten(0, 2)
    flat_alpha = rendered_alpha.flatten(0, 2)
    flat_dynamic = dynamic_scores.flatten(0, 2)

    rgb, rgb_weights = compute_reconstruction_loss(
        flat_rendered,
        flat_target,
        ssim_weight=float(cfg.ssim_weight),
        loss_mask=flat_mask,
        dynamic_score=flat_dynamic,
        dynamic_region_weight=float(cfg.dynamic_region_weight),
        return_per_render=True,
    )
    sil_fg, sil_bg, _sil, sil_fg_valid, sil_bg_valid = compute_balanced_silhouette_loss(
        flat_alpha,
        flat_mask,
        dynamic_score=flat_dynamic,
        dynamic_region_weight=float(cfg.dynamic_region_weight),
        return_per_render=True,
    )
    global_depth, local_depth, global_valid, local_valid_count = compute_global_local_depth_loss(
        flat_depth,
        flat_target_depth,
        flat_mask,
        patch_size=patch_size,
        min_valid_pixels=int(cfg.local_depth_min_valid_pixels),
        dynamic_score=flat_dynamic,
        dynamic_region_weight=float(cfg.dynamic_region_weight),
        return_per_render=True,
    )
    shape = (batch, timesteps, views)
    rgb = rgb.view(shape); rgb_weights = rgb_weights.view(shape)
    sil_fg = sil_fg.view(shape); sil_bg = sil_bg.view(shape)
    sil_fg_valid = sil_fg_valid.view(shape); sil_bg_valid = sil_bg_valid.view(shape)
    global_depth = global_depth.view(shape); global_valid = global_valid.view(shape)
    local_depth = local_depth.view(shape); local_valid_count = local_valid_count.view(shape)

    losses: list[Dict[str, torch.Tensor]] = []
    for time_idx in range(TEMPORAL_WINDOW):
        rgb_t = _weighted_mean(rgb, rgb_weights, time_idx)
        sil_fg_t = _weighted_mean(sil_fg, sil_fg_valid, time_idx)
        sil_bg_t = _weighted_mean(sil_bg, sil_bg_valid, time_idx)
        silhouette_t = 0.5 * (sil_fg_t + sil_bg_t)
        global_t = _weighted_mean(global_depth, global_valid, time_idx)
        local_t = _weighted_mean(local_depth, local_valid_count, time_idx)
        depth_t = global_t + float(cfg.local_depth_weight) * local_t
        render_t = (
            float(cfg.rec_weight) * rgb_t
            + float(cfg.silhouette_weight) * silhouette_t
            + float(cfg.global_depth_weight) * depth_t
        )
        losses.append({
            "rgb_loss": rgb_t,
            "silhouette_foreground_loss": sil_fg_t,
            "silhouette_background_loss": sil_bg_t,
            "silhouette_loss": silhouette_t,
            "global_depth_loss": global_t,
            "local_depth_loss": local_t,
            "depth_loss": depth_t,
            "render_loss": render_t,
        })
    return losses


def _masked_mean(values: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    weights = valid.to(values.dtype)
    return (values * weights).sum() / weights.sum().clamp_min(1.0)


def compute_reconstruction_and_renders(
    splatter_to_gaussians: DirectSplatterToGaussians,
    splatter_cfg: SplatterConfig,
    raw_base_map: torch.Tensor,
    raw_motion_map: torch.Tensor,
    motion_translation_max: float,
    images_01: torch.Tensor,
    optical_flows: torch.Tensor,
    depths: torch.Tensor,
    masks: torch.Tensor,
    intrinsics: torch.Tensor,
    c2w: torch.Tensor,
    w2c: torch.Tensor,
    bg: torch.Tensor,
    cfg_train: TrainConfig,
    source_indices: torch.Tensor,
    temporal_ramp: float,
    training: bool,
    return_renders: bool = False,
    compute_diagnostics: bool = False,
) -> Dict[str, Any]:
    """Activate FP32 maps, render once, and compute all fixed pretraining losses."""
    if images_01.dim() != 6 or images_01.shape[1] != TEMPORAL_WINDOW:
        raise ValueError(f"Expected exactly three RGB frames, got {tuple(images_01.shape)}.")
    batch, _, views = images_01.shape[:3]
    expected_mask = (*images_01.shape[:3], 1, *images_01.shape[-2:])
    if tuple(masks.shape) != expected_mask or tuple(depths.shape) != expected_mask:
        raise ValueError(f"Depths and masks are mandatory with shape {expected_mask}.")
    expected_flow = (batch, 3, views, 2, *images_01.shape[-2:])
    if tuple(optical_flows.shape) != expected_flow:
        raise ValueError(f"Expected teacher flows as {expected_flow}, got {tuple(optical_flows.shape)}.")
    raw_base_map = raw_base_map.float()
    raw_motion_map = raw_motion_map.float()
    activated_motion = activate_motion_map(raw_motion_map, motion_translation_max)
    target_masks = masks.bool()
    target_masks_float = target_masks.float()
    depths = depths.float()
    intrinsics_t = _expand_camera_time(intrinsics).float()
    c2w_t = _expand_camera_time(c2w).float()
    w2c_t = _expand_camera_time(w2c).float()
    source_indices = source_indices.to(device=images_01.device, dtype=torch.long)
    if source_indices.shape != (batch,):
        raise ValueError(f"Expected source_indices shape {(batch,)}, got {tuple(source_indices.shape)}.")

    source_intrinsics = _gather_source_time0(intrinsics_t, source_indices)
    source_c2w = _gather_source_time0(c2w_t, source_indices)
    pc0 = splatter_to_gaussians(
        splatter_map=raw_base_map,
        motion_map=activated_motion,
        source_cameras_view_to_world=source_c2w,
        intrinsics=source_intrinsics,
    )
    pc0 = apply_source_mask_to_gaussians(pc0, _gather_source_time0(target_masks, source_indices))
    pc1 = translate_gaussians(pc0, pc0["delta_xyz_01"])
    pc2 = translate_gaussians(pc1, pc0["delta_xyz_12"])
    pc_sequence = [pc0, pc1, pc2]
    rendered, rendered_depth, rendered_alpha = _render_sequence(
        pc_sequence, w2c_t, intrinsics_t, bg, splatter_cfg
    )
    rendered_flows, flow_coverages, flow_valid_masks, flow_source_alphas = (
        render_translation_flow_sequence(pc0, w2c_t, intrinsics_t, splatter_cfg)
    )
    flow_foreground_masks = torch.stack(
        (target_masks[:, 0], target_masks[:, 1], target_masks[:, 0]), dim=1
    )
    flow_loss, flow_metrics = compute_optical_flow_loss(
        predicted_flows=rendered_flows,
        target_flows=optical_flows,
        rendered_coverage=flow_coverages,
        predicted_valid_mask=flow_valid_masks,
        foreground_mask=flow_foreground_masks,
        alpha_threshold=float(cfg_train.flow_alpha_threshold),
        smooth_l1_beta=float(cfg_train.flow_smooth_l1_beta),
    )
    dynamic_scores = build_target_dynamic_scores(optical_flows)
    target_images = images_01.float() * target_masks_float
    timestep_losses = _timestep_render_losses(
        rendered,
        rendered_depth,
        rendered_alpha,
        target_images,
        depths,
        target_masks,
        dynamic_scores,
        cfg_train,
        _local_depth_patch_size(cfg_train, training),
    )
    frustum_loss, frustum_per_timestep = compute_union_frustum_loss(
        pc0["xyz"],
        pc0["delta_xyz_01"],
        pc0["delta_xyz_12"],
        pc0["valid_mask"],
        w2c_t,
        intrinsics_t,
        image_height=int(splatter_cfg.data.img_height),
        image_width=int(splatter_cfg.data.img_width),
        near_plane=float(splatter_cfg.data.znear),
        far_plane=float(splatter_cfg.data.zfar),
        temporal_ramp=float(temporal_ramp),
    )
    output: Dict[str, Any] = {
        "timestep_losses": timestep_losses,
        "flow_loss": flow_loss,
        "frustum_loss": frustum_loss,
        "frustum_loss_per_timestep": frustum_per_timestep,
        **flow_metrics,
    }
    for time_idx, losses_t in enumerate(timestep_losses):
        for name, value in losses_t.items():
            output[f"{name}_t{time_idx}"] = value
    for name in timestep_losses[0]:
        output[name] = torch.stack([item[name] for item in timestep_losses]).mean()
    output["rec_loss"] = output["rgb_loss"]

    if compute_diagnostics or return_renders:
        valid = pc0["valid_mask"]
        output.update({
            "mean_valid_gaussian_opacity": _masked_mean(pc0["opacity"].squeeze(-1), valid),
            "translation_01_mean": _masked_mean(pc0["delta_xyz_01"].norm(dim=-1), valid),
            "translation_12_mean": _masked_mean(pc0["delta_xyz_12"].norm(dim=-1), valid),
        })
    if return_renders:
        output.update({
            "target_images_self": target_images,
            "target_masks_self": target_masks_float,
            "target_depths_self": depths,
            "target_optical_flows": optical_flows.detach(),
            "rendered_self": rendered,
            "rendered_expected_depth_self": rendered_depth,
            "rendered_alpha_self": rendered_alpha,
            "rendered_optical_flows": rendered_flows,
            "rendered_flow_alpha": flow_source_alphas,
            "rendered_flow_coverage": flow_coverages,
            "rendered_flow_valid_mask": flow_valid_masks,
            "source_indices": source_indices.detach().cpu(),
            "gaussian_pc": {key: value.detach() for key, value in pc0.items() if torch.is_tensor(value)},
            "gaussian_pc_sequence": [
                {key: value.detach() for key, value in pc.items() if torch.is_tensor(value)}
                for pc in pc_sequence
            ],
            "source_c2w": source_c2w.detach(),
            "source_intrinsics": source_intrinsics.detach(),
            "base_map": raw_base_map.detach(),
            "motion_map": activated_motion.detach(),
        })
    return output
