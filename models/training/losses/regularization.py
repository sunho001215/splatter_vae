from __future__ import annotations

import torch

from models.gaussian.geometry import (
    VISIBILITY_MARGIN_PIXELS,
    visibility_loss_per_timestep,
)
from models.gaussian.motion import construct_chronological_gaussian_sequence


def compute_visibility_loss(
    anchor_xyz: torch.Tensor,
    delta_xyz_01: torch.Tensor | None,
    delta_xyz_12: torch.Tensor | None,
    valid_mask: torch.Tensor,
    world_view_transform: torch.Tensor,
    intrinsics: torch.Tensor,
    *,
    image_height: int,
    image_width: int,
    near_plane: float,
    far_plane: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    if delta_xyz_01 is None or delta_xyz_12 is None:
        raise ValueError(
            "DROID temporal Gaussians require both chronological motion deltas."
        )
    sequence = construct_chronological_gaussian_sequence(
        {"xyz": anchor_xyz, "delta_xyz_01": delta_xyz_01, "delta_xyz_12": delta_xyz_12},
        "current",
    )
    xyz = torch.stack([item["xyz"] for item in sequence], dim=1)
    validity = valid_mask[:, None].expand(-1, 3, -1)
    per_timestep = visibility_loss_per_timestep(
        xyz,
        validity,
        world_view_transform,
        intrinsics,
        image_height=image_height,
        image_width=image_width,
        near_plane=near_plane,
        far_plane=far_plane,
        margin_pixels=VISIBILITY_MARGIN_PIXELS,
    )
    return per_timestep.mean(), per_timestep


def gaussian_regularization(
    gaussian_pc: dict[str, torch.Tensor],
    child_offsets: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    valid = gaussian_pc["valid_mask"].float()
    scales = gaussian_pc["scaling"].float().clamp_min(1.0e-8)
    log_scales = torch.log(scales)
    anisotropy = (
        (log_scales - log_scales.mean(dim=-1, keepdim=True)).square().mean(dim=-1)
    )
    anisotropy_loss = (anisotropy * valid).sum() / valid.sum().clamp_min(1.0)
    opacity = gaussian_pc["opacity"].float().squeeze(-1).clamp(1.0e-6, 1.0 - 1.0e-6)
    opacity_entropy = -(
        opacity * torch.log(opacity) + (1.0 - opacity) * torch.log1p(-opacity)
    )
    opacity_entropy_loss = (opacity_entropy * valid).sum() / valid.sum().clamp_min(1.0)
    offset_loss = anisotropy_loss.new_zeros(())
    if child_offsets is not None:
        offset_loss = child_offsets.float().square().sum(dim=-1).mean()
    total = anisotropy_loss + 0.1 * opacity_entropy_loss + 0.1 * offset_loss
    return total, {
        "gaussian_scale_anisotropy": anisotropy_loss.detach(),
        "gaussian_opacity_entropy": opacity_entropy_loss.detach(),
        "gaussian_child_offset_squared": offset_loss.detach(),
    }
