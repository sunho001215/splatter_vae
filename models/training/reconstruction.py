from __future__ import annotations

import random
from typing import Any, Dict

import torch

from models.gaussian.motion import (
    activate_motion_parameters,
    construct_chronological_gaussian_sequence,
    render_translation_flow_sequence,
)
from models.gaussian.parameterization import (
    ACTIVE_GAUSSIAN_OPACITY_THRESHOLD,
    SplatterConfig,
    WorldSpaceGaussianParameterization,
)
from models.gaussian.rendering import render_dngaussian_depths, render_rgb
from models.splattervae.config import TEMPORAL_WINDOW
from models.splattervae.model import SplatterVAE
from models.training.config import TrainConfig
from models.training.losses import (
    build_target_dynamic_scores,
    compute_balanced_silhouette_loss,
    compute_global_local_depth_loss,
    compute_optical_flow_loss,
    compute_reconstruction_loss,
    compute_visibility_loss,
)
from utils.camera_tensor_utils import gather_camera_rows


def encode_per_view_sequence_batch(
    vae: SplatterVAE,
    images: torch.Tensor,
    optical_flows: torch.Tensor,
) -> Dict[str, torch.Tensor]:
    """Encode each camera using either the temporal sequence or t0-only ablation."""
    expected_timesteps = TEMPORAL_WINDOW if vae.temporal_modeling else 1
    if images.dim() != 6 or images.shape[1] != expected_timesteps:
        raise ValueError(
            f"Expected images with T={expected_timesteps} as (B,T,A,3,H,W), "
            f"got {tuple(images.shape)}."
        )
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
        "inv_mask_by_view": latents["inv_mask"][:, 0].view(batch, views, -1).contiguous(),
    }


def _expand_camera_time(camera: torch.Tensor) -> torch.Tensor:
    if camera.dim() == 4:
        return camera[:, None].expand(-1, TEMPORAL_WINDOW, -1, -1, -1).contiguous()
    if camera.dim() == 5 and camera.shape[1] == TEMPORAL_WINDOW:
        return camera
    raise ValueError(f"Expected camera tensor with or without T=3, got {tuple(camera.shape)}.")


def _gather_source_time0(values: torch.Tensor, source_indices: torch.Tensor) -> torch.Tensor:
    return gather_camera_rows(values[:, 0], source_indices)


def _stack_gaussian_sequence(
    pc_sequence: list[Dict[str, torch.Tensor]],
) -> Dict[str, torch.Tensor]:
    keys = ("xyz", "scaling", "rotation", "opacity", "features_dc", "features_rest", "valid_mask")
    return {key: torch.stack([pc[key] for pc in pc_sequence], 1).contiguous() for key in keys}


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


def _resolve_loss_masks(
    segmentation_masks: torch.Tensor,
    use_segmentation_mask: bool,
) -> torch.Tensor:
    """Use configured segmentation masks or supervise every image pixel."""
    masks = segmentation_masks.bool()
    return masks if use_segmentation_mask else torch.ones_like(masks)


def _depth_loss_components(
    rendered_depth: torch.Tensor,
    target_depths: torch.Tensor,
    target_masks: torch.Tensor,
    dynamic_scores: torch.Tensor,
    cfg: TrainConfig,
    patch_size: int,
    shape: tuple[int, int, int],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    components = compute_global_local_depth_loss(
        rendered_depth.flatten(0, 2),
        target_depths.flatten(0, 2),
        target_masks.flatten(0, 2),
        patch_size=patch_size,
        min_valid_pixels=int(cfg.local_depth_min_valid_pixels),
        dynamic_score=dynamic_scores.flatten(0, 2),
        dynamic_region_weight=float(cfg.dynamic_region_weight),
        return_per_render=True,
    )
    return tuple(component.view(shape) for component in components)


def _timestep_render_losses(
    rendered: torch.Tensor,
    hard_depth: torch.Tensor,
    soft_depth: torch.Tensor,
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
    if cfg.use_segmentation_mask:
        sil_fg, sil_bg, _sil, sil_fg_valid, sil_bg_valid = (
            compute_balanced_silhouette_loss(
                flat_alpha,
                flat_mask,
                dynamic_score=flat_dynamic,
                dynamic_region_weight=float(cfg.dynamic_region_weight),
                return_per_render=True,
            )
        )
    else:
        sil_fg = torch.zeros_like(rgb)
        sil_bg = torch.zeros_like(rgb)
        sil_fg_valid = torch.ones_like(rgb)
        sil_bg_valid = torch.ones_like(rgb)
    shape = (batch, timesteps, views)
    (
        hard_global,
        hard_local,
        hard_global_valid,
        hard_local_valid,
    ) = _depth_loss_components(
        hard_depth,
        target_depths,
        target_masks,
        dynamic_scores,
        cfg,
        patch_size,
        shape,
    )
    (
        soft_global,
        soft_local,
        soft_global_valid,
        soft_local_valid,
    ) = _depth_loss_components(
        soft_depth,
        target_depths,
        target_masks,
        dynamic_scores,
        cfg,
        patch_size,
        shape,
    )
    rgb = rgb.view(shape); rgb_weights = rgb_weights.view(shape)
    sil_fg = sil_fg.view(shape); sil_bg = sil_bg.view(shape)
    sil_fg_valid = sil_fg_valid.view(shape); sil_bg_valid = sil_bg_valid.view(shape)

    losses: list[Dict[str, torch.Tensor]] = []
    for time_idx in range(timesteps):
        rgb_t = _weighted_mean(rgb, rgb_weights, time_idx)
        sil_fg_t = _weighted_mean(sil_fg, sil_fg_valid, time_idx)
        sil_bg_t = _weighted_mean(sil_bg, sil_bg_valid, time_idx)
        silhouette_t = 0.5 * (sil_fg_t + sil_bg_t)
        hard_global_t = _weighted_mean(hard_global, hard_global_valid, time_idx)
        hard_local_t = _weighted_mean(hard_local, hard_local_valid, time_idx)
        hard_depth_t = hard_global_t + float(cfg.local_depth_weight) * hard_local_t
        soft_global_t = _weighted_mean(soft_global, soft_global_valid, time_idx)
        soft_local_t = _weighted_mean(soft_local, soft_local_valid, time_idx)
        soft_depth_t = soft_global_t + float(cfg.local_depth_weight) * soft_local_t
        global_t = (
            float(cfg.hard_depth_weight) * hard_global_t
            + float(cfg.soft_depth_weight) * soft_global_t
        )
        local_t = (
            float(cfg.hard_depth_weight) * hard_local_t
            + float(cfg.soft_depth_weight) * soft_local_t
        )
        depth_t = (
            float(cfg.hard_depth_weight) * hard_depth_t
            + float(cfg.soft_depth_weight) * soft_depth_t
        )
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
            "hard_global_depth_loss": hard_global_t,
            "hard_local_depth_loss": hard_local_t,
            "hard_depth_loss": hard_depth_t,
            "soft_global_depth_loss": soft_global_t,
            "soft_local_depth_loss": soft_local_t,
            "soft_depth_loss": soft_depth_t,
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
    splatter_to_gaussians: WorldSpaceGaussianParameterization,
    splatter_cfg: SplatterConfig,
    raw_gaussian_params: torch.Tensor,
    raw_motion_params: torch.Tensor | None,
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
    """Activate predictions and compute batched RGB and DNGaussian depth losses.

    ``source_indices`` selects only which invariant feature was decoded upstream
    and which camera is displayed. No source camera or source mask enters Gaussian
    construction.
    """
    if images_01.dim() != 6 or images_01.shape[1] != TEMPORAL_WINDOW:
        raise ValueError(f"Expected exactly three RGB frames, got {tuple(images_01.shape)}.")
    batch, _, views = images_01.shape[:3]
    expected_mask = (*images_01.shape[:3], 1, *images_01.shape[-2:])
    if tuple(masks.shape) != expected_mask or tuple(depths.shape) != expected_mask:
        raise ValueError(f"Depths and masks are mandatory with shape {expected_mask}.")
    expected_flow = (batch, 3, views, 2, *images_01.shape[-2:])
    if tuple(optical_flows.shape) != expected_flow:
        raise ValueError(f"Expected teacher flows as {expected_flow}, got {tuple(optical_flows.shape)}.")
    temporal_modeling = raw_motion_params is not None
    num_render_timesteps = TEMPORAL_WINDOW if temporal_modeling else 1
    raw_gaussian_params = raw_gaussian_params.float()
    activated_motion = (
        activate_motion_parameters(raw_motion_params.float(), motion_translation_max)
        if temporal_modeling else None
    )
    images_01 = images_01[:, :num_render_timesteps]
    target_masks = _resolve_loss_masks(
        masks[:, :num_render_timesteps],
        cfg_train.use_segmentation_mask,
    )
    target_masks_float = target_masks.float()
    depths = depths[:, :num_render_timesteps].float()
    intrinsics_t = _expand_camera_time(intrinsics).float()[:, :num_render_timesteps]
    c2w_t = _expand_camera_time(c2w).float()[:, :num_render_timesteps]
    w2c_t = _expand_camera_time(w2c).float()[:, :num_render_timesteps]
    source_indices = source_indices.to(device=images_01.device, dtype=torch.long)
    if source_indices.shape != (batch,):
        raise ValueError(f"Expected source_indices shape {(batch,)}, got {tuple(source_indices.shape)}.")

    anchor_pc = splatter_to_gaussians(
        gaussian_parameters=raw_gaussian_params,
        motion_parameters=activated_motion,
    )
    pc_sequence = (
        construct_chronological_gaussian_sequence(
            anchor_pc, cfg_train.temporal_anchor
        )
        if temporal_modeling
        else [anchor_pc]
    )
    stacked_pc = _stack_gaussian_sequence(pc_sequence)
    rgb_result = render_rgb(
        stacked_pc, w2c_t, intrinsics_t, bg, splatter_cfg
    )
    rendered = rgb_result["render"]
    rendered_alpha = rgb_result["alpha"]
    depth_result = render_dngaussian_depths(
        stacked_pc,
        w2c_t,
        intrinsics_t,
        splatter_cfg,
        hard_opacity=float(cfg_train.hard_depth_opacity),
    )
    hard_depth = depth_result["hard_depth"]
    soft_depth = depth_result["soft_depth"]
    flow_metrics: Dict[str, torch.Tensor] = {}
    if temporal_modeling:
        rendered_flows, flow_coverages, flow_valid_masks, flow_source_alphas = render_translation_flow_sequence(
            anchor_pc,
            w2c_t,
            intrinsics_t,
            splatter_cfg,
            temporal_anchor=cfg_train.temporal_anchor,
        )
        flow_foreground_masks = torch.stack(
            (target_masks[:, 0], target_masks[:, 1], target_masks[:, 0]), dim=1
        )
        flow_loss, flow_metrics = compute_optical_flow_loss(
            predicted_flows=rendered_flows, target_flows=optical_flows,
            rendered_coverage=flow_coverages,
            predicted_valid_mask=flow_valid_masks,
            foreground_mask=flow_foreground_masks,
            alpha_threshold=float(cfg_train.flow_alpha_threshold),
            smooth_l1_beta=float(cfg_train.flow_smooth_l1_beta),
        )
    dynamic_scores = build_target_dynamic_scores(optical_flows)[:, :num_render_timesteps]
    target_images = images_01.float() * target_masks_float
    timestep_losses = _timestep_render_losses(
        rendered, hard_depth, soft_depth, rendered_alpha,
        target_images, depths, target_masks, dynamic_scores, cfg_train,
        _local_depth_patch_size(cfg_train, training),
    )
    visibility_loss, visibility_per_timestep = compute_visibility_loss(
        anchor_pc["xyz"],
        anchor_pc.get("delta_xyz_01"),
        anchor_pc.get("delta_xyz_12"),
        anchor_pc["valid_mask"],
        w2c_t,
        intrinsics_t,
        image_height=int(splatter_cfg.data.img_height),
        image_width=int(splatter_cfg.data.img_width),
        near_plane=float(splatter_cfg.data.znear),
        far_plane=float(splatter_cfg.data.zfar),
        temporal_ramp=float(temporal_ramp),
        temporal_anchor=cfg_train.temporal_anchor,
    )
    output: Dict[str, Any] = {
        "timestep_losses": timestep_losses,
        "visibility_loss": visibility_loss,
        "visibility_loss_per_timestep": visibility_per_timestep,
    }
    if temporal_modeling:
        output.update({"flow_loss": flow_loss, **flow_metrics})
    for time_idx, losses_t in enumerate(timestep_losses):
        for name, value in losses_t.items():
            output[f"{name}_t{time_idx}"] = value
    for name in timestep_losses[0]:
        output[name] = torch.stack([item[name] for item in timestep_losses]).mean()
    output["rec_loss"] = output["rgb_loss"]

    if compute_diagnostics or return_renders:
        valid = anchor_pc["valid_mask"]
        opacity = anchor_pc["opacity"].squeeze(-1)
        output["mean_valid_gaussian_opacity"] = _masked_mean(opacity, valid)
        active = valid & (opacity >= ACTIVE_GAUSSIAN_OPACITY_THRESHOLD)
        output["active_gaussian_fraction"] = active.to(opacity.dtype).sum() / valid.to(opacity.dtype).sum().clamp_min(1.0)
        if temporal_modeling:
            output["translation_01_mean"] = _masked_mean(
                anchor_pc["delta_xyz_01"].norm(dim=-1), valid
            )
            output["translation_12_mean"] = _masked_mean(
                anchor_pc["delta_xyz_12"].norm(dim=-1), valid
            )
    if return_renders:
        source_c2w = _gather_source_time0(c2w_t, source_indices)
        source_intrinsics = _gather_source_time0(intrinsics_t, source_indices)
        render_payload: Dict[str, Any] = {
            "target_images_self": target_images,
            "target_masks_self": target_masks_float,
            "target_depths_self": depths,
            "target_optical_flows": optical_flows.detach(),
            "rendered_self": rendered,
            "rendered_hard_depth_self": hard_depth,
            "rendered_soft_depth_self": soft_depth,
            "rendered_alpha_self": rendered_alpha,
            "source_indices": source_indices.detach().cpu(),
            "gaussian_pc_anchor": {
                key: value.detach()
                for key, value in anchor_pc.items()
                if torch.is_tensor(value)
            },
            "gaussian_pc_sequence": [
                {key: value.detach() for key, value in pc.items() if torch.is_tensor(value)}
                for pc in pc_sequence
            ],
            "source_c2w": source_c2w.detach(),
            "source_intrinsics": source_intrinsics.detach(),
            "raw_gaussian_params": raw_gaussian_params.detach(),
        }
        # Legacy diagnostics used gaussian_pc for the directly decoded set.
        render_payload["gaussian_pc"] = render_payload["gaussian_pc_anchor"]
        if temporal_modeling:
            render_payload.update({
                "rendered_optical_flows": rendered_flows,
                "rendered_flow_alpha": flow_source_alphas,
                "rendered_flow_coverage": flow_coverages,
                "rendered_flow_valid_mask": flow_valid_masks,
                "motion_parameters": activated_motion.detach(),
            })
        output.update(render_payload)
    return output
