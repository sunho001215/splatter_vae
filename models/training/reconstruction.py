from __future__ import annotations

from collections.abc import Callable
from dataclasses import replace
from typing import Any

import torch

from models.gaussian.geometry import project_gaussian_centers
from models.gaussian.motion import (
    activate_motion_parameters,
    construct_chronological_gaussian_sequence,
    render_middle_frame_translation_flow,
)
from models.gaussian.parameterization import (
    ACTIVE_GAUSSIAN_OPACITY_THRESHOLD,
    SplatterConfig,
    WorldSpaceGaussianParameterization,
)
from models.gaussian.rendering import render_rgb_expected_depth
from models.training.config import TrainConfig
from models.training.losses import (
    compute_optical_flow_loss,
    compute_visibility_loss,
    confidence_weighted_metric_depth_l1,
    gaussian_regularization,
    masked_rgb_reconstruction_losses,
    scale_invariant_log_depth_loss,
)


def _stack_gaussian_sequence(
    sequence: list[dict[str, torch.Tensor]],
) -> dict[str, torch.Tensor]:
    keys = (
        "xyz",
        "scaling",
        "rotation",
        "opacity",
        "features_dc",
        "features_rest",
        "valid_mask",
    )
    return {
        key: torch.stack([gaussians[key] for gaussians in sequence], dim=1).contiguous()
        for key in keys
    }


def _masked_mean(values: torch.Tensor, validity: torch.Tensor) -> torch.Tensor:
    weights = validity.to(values.dtype)
    return (values * weights).sum() / weights.sum().clamp_min(1.0)


def compute_droid_reconstruction(
    parameterization: WorldSpaceGaussianParameterization,
    splatter_config: SplatterConfig,
    prediction: dict[str, torch.Tensor],
    batch: dict[str, Any],
    train_config: TrainConfig,
    *,
    motion_translation_max: float,
    background_color: torch.Tensor,
    return_renders: bool = False,
    novel_view_enabled: bool = False,
    stage_runner: Callable[[str, Callable[[], Any]], Any] | None = None,
) -> dict[str, Any]:
    """Render current-anchored Gaussians into both real exterior histories."""
    run_stage = stage_runner or (lambda _name, callable_: callable_())
    raw_gaussians = prediction["raw_gaussian_params"].float()
    activated_motion = activate_motion_parameters(
        prediction["raw_motion_params"], motion_translation_max
    )
    anchor_pc = run_stage(
        "gaussian_parameterization_and_dynamics",
        lambda: parameterization(
            gaussian_parameters=raw_gaussians,
            motion_parameters=activated_motion,
        ),
    )
    sequence = construct_chronological_gaussian_sequence(anchor_pc, "current")
    temporal_pc = _stack_gaussian_sequence(sequence)
    w2c = batch["target_w2c"].float()
    intrinsics = batch["target_K"].float()
    render = run_stage(
        "real_rgb_depth_render",
        lambda: render_rgb_expected_depth(
            temporal_pc, w2c, intrinsics, background_color, splatter_config
        ),
    )
    target_rgb = batch["target_rgb"].float()
    image_validity = batch["target_image_validity"].bool()
    rgb_l1, dssim, rgb_metrics = run_stage(
        "real_rgb_losses",
        lambda: masked_rgb_reconstruction_losses(
            render["rgb"], target_rgb, image_validity
        ),
    )

    confidence = batch["target_depth_confidence"].float()
    depth_validity = (
        batch["target_depth_validity"].bool()
        & image_validity
        & render["depth_validity"].bool()
        & (confidence >= float(train_config.depth_confidence_threshold))
    )
    calibration_validity = batch["calibration_validity"].bool()
    while calibration_validity.dim() < depth_validity.dim():
        calibration_validity = calibration_validity.unsqueeze(-1)
    depth_validity = depth_validity & calibration_validity
    metric_depth, depth_metrics = run_stage(
        "metric_depth_loss",
        lambda: confidence_weighted_metric_depth_l1(
            render["expected_depth"],
            batch["target_depth"],
            confidence,
            depth_validity,
        ),
    )
    scale_invariant_depth = run_stage(
        "scale_invariant_depth_loss",
        lambda: scale_invariant_log_depth_loss(
            render["expected_depth"],
            batch["target_depth"],
            confidence,
            depth_validity,
            mean_weight=float(train_config.scale_invariant_mean_weight),
        ),
    )

    predicted_flow, flow_coverage, predicted_flow_validity, flow_alpha = (
        run_stage(
            "gaussian_flow_render",
            lambda: render_middle_frame_translation_flow(
                anchor_pc,
                w2c,
                intrinsics,
                splatter_config,
                temporal_anchor="current",
            ),
        )
    )
    flow_loss, flow_metrics = run_stage(
        "flow_loss",
        lambda: compute_optical_flow_loss(
            predicted_flow,
            batch["target_flow"],
            flow_coverage,
            predicted_flow_validity,
            batch["target_flow_validity"],
            batch.get("target_flow_confidence"),
            pair_weights=train_config.flow_pair_weights,
            alpha_threshold=float(train_config.flow_alpha_threshold),
            smooth_l1_beta=float(train_config.flow_smooth_l1_beta),
        ),
    )
    visibility, visibility_by_time = run_stage(
        "visibility_loss",
        lambda: compute_visibility_loss(
            anchor_pc["xyz"],
            anchor_pc["delta_xyz_01"],
            anchor_pc["delta_xyz_12"],
            anchor_pc["valid_mask"],
            w2c,
            intrinsics,
            image_height=int(splatter_config.data.img_height),
            image_width=int(splatter_config.data.img_width),
            near_plane=float(splatter_config.data.znear),
            far_plane=float(splatter_config.data.zfar),
        ),
    )
    gaussian_reg, gaussian_metrics = run_stage(
        "gaussian_regularization_loss",
        lambda: gaussian_regularization(
            anchor_pc, prediction.get("child_offsets")
        ),
    )
    novel_view_loss = rgb_l1.new_zeros(())
    novel_metrics: dict[str, torch.Tensor] = {
        "novel_rgb_l1_loss": novel_view_loss.detach(),
        "novel_dssim_loss": novel_view_loss.detach(),
        "novel_support_fraction": novel_view_loss.detach(),
    }
    novel_render = None
    if novel_view_enabled:
        required_novel = (
            "novel_rgb",
            "novel_K",
            "novel_w2c",
            "novel_support_mask",
        )
        missing_novel = [name for name in required_novel if name not in batch]
        if missing_novel:
            raise KeyError(
                "Enabled LagerNVS supervision is missing online fields "
                f"{missing_novel}."
            )
        novel_config = replace(
            splatter_config,
            data=replace(
                splatter_config.data,
                img_height=256,
                img_width=256,
            ),
        )
        novel_render = run_stage(
            "virtual_camera_render",
            lambda: render_rgb_expected_depth(
                anchor_pc,
                batch["novel_w2c"].float()[:, None],
                batch["novel_K"].float()[:, None],
                background_color,
                novel_config,
            ),
        )
        support = batch["novel_support_mask"].bool()[:, None]
        pixel_weights = torch.where(
            support,
            support.new_full(
                support.shape,
                float(train_config.novel_view_supported_weight),
                dtype=torch.float32,
            ),
            support.new_full(
                support.shape,
                float(train_config.novel_view_unsupported_weight),
                dtype=torch.float32,
            ),
        )
        novel_rgb_l1, novel_dssim, _novel_rgb_metrics = run_stage(
            "novel_view_loss",
            lambda: masked_rgb_reconstruction_losses(
                novel_render["rgb"],
                batch["novel_rgb"].float()[:, None],
                pixel_weights,
            ),
        )
        novel_view_loss = (
            float(train_config.rgb_l1_weight) * novel_rgb_l1
            + float(train_config.ssim_weight) * novel_dssim
        )
        novel_metrics = {
            "novel_rgb_l1_loss": novel_rgb_l1.detach(),
            "novel_dssim_loss": novel_dssim.detach(),
            "novel_support_fraction": support.float().mean().detach(),
        }
    total = (
        float(train_config.rgb_l1_weight) * rgb_l1
        + float(train_config.ssim_weight) * dssim
        + float(train_config.metric_depth_weight) * metric_depth
        + float(train_config.scale_invariant_depth_weight) * scale_invariant_depth
        + float(train_config.flow_weight) * flow_loss
        + float(train_config.visibility_weight) * visibility
        + float(train_config.gaussian_regularization_weight) * gaussian_reg
        + float(train_config.novel_view_rgb_weight) * novel_view_loss
    )
    opacity = anchor_pc["opacity"].squeeze(-1)
    valid_gaussians = anchor_pc["valid_mask"]
    active = valid_gaussians & (opacity >= ACTIVE_GAUSSIAN_OPACITY_THRESHOLD)
    current_pixels, current_depths, current_finite = project_gaussian_centers(
        anchor_pc["xyz"][:, None], w2c[:, 2:3], intrinsics[:, 2:3]
    )
    current_u, current_v = current_pixels.unbind(-1)
    in_frustum = (
        (
            current_finite
            & (current_depths > float(splatter_config.data.znear))
            & (current_depths < float(splatter_config.data.zfar))
            & (current_u >= 0)
            & (current_u < int(splatter_config.data.img_width))
            & (current_v >= 0)
            & (current_v < int(splatter_config.data.img_height))
        )
        .any(dim=2)
        .squeeze(1)
    )
    output: dict[str, Any] = {
        "loss": total,
        "rgb_l1_loss": rgb_l1,
        "dssim_loss": dssim,
        "metric_depth_loss": metric_depth,
        "scale_invariant_depth_loss": scale_invariant_depth,
        "flow_loss": flow_loss,
        "visibility_loss": visibility,
        "gaussian_regularization_loss": gaussian_reg,
        "novel_view_loss": novel_view_loss,
        "visibility_loss_by_time": visibility_by_time.detach(),
        "mean_opacity": _masked_mean(opacity, valid_gaussians).detach(),
        "active_gaussian_fraction": (
            active.float().sum() / valid_gaussians.float().sum().clamp_min(1.0)
        ).detach(),
        "out_of_frustum_fraction": (
            1.0
            - (in_frustum & valid_gaussians).float().sum()
            / valid_gaussians.float().sum().clamp_min(1.0)
        ).detach(),
        "rendered_visible_pixel_fraction": (render["alpha"] > 0.01)
        .float()
        .mean()
        .detach(),
        "gaussian_scale_mean": _masked_mean(
            anchor_pc["scaling"].mean(dim=-1), valid_gaussians
        ).detach(),
        "motion_01_mean": _masked_mean(
            anchor_pc["delta_xyz_01"].norm(dim=-1), valid_gaussians
        ).detach(),
        "motion_12_mean": _masked_mean(
            anchor_pc["delta_xyz_12"].norm(dim=-1), valid_gaussians
        ).detach(),
        "parent_xyz_x_mean": prediction["parent_centers"][..., 0]
        .float()
        .mean()
        .detach(),
        "parent_xyz_y_mean": prediction["parent_centers"][..., 1]
        .float()
        .mean()
        .detach(),
        "parent_xyz_z_mean": prediction["parent_centers"][..., 2]
        .float()
        .mean()
        .detach(),
        "child_offset_mean": prediction["child_offsets"]
        .float()
        .norm(dim=-1)
        .mean()
        .detach(),
        "child_offset_max": prediction["child_offsets"]
        .float()
        .norm(dim=-1)
        .amax()
        .detach(),
        **rgb_metrics,
        **depth_metrics,
        **flow_metrics,
        **gaussian_metrics,
        **novel_metrics,
    }
    if return_renders:
        output.update(
            {
                "rendered_rgb": render["rgb"],
                "rendered_expected_depth": render["expected_depth"],
                "rendered_alpha": render["alpha"],
                "rendered_flow": predicted_flow,
                "rendered_flow_coverage": flow_coverage,
                "rendered_flow_validity": predicted_flow_validity,
                "rendered_flow_alpha": flow_alpha,
                "gaussian_pc_anchor": anchor_pc,
                "gaussian_pc_sequence": sequence,
            }
        )
        if novel_render is not None:
            output.update(
                {
                    "novel_rendered_rgb": novel_render["rgb"],
                    "novel_rendered_expected_depth": novel_render[
                        "expected_depth"
                    ],
                    "novel_rendered_alpha": novel_render["alpha"],
                }
            )
    return output
