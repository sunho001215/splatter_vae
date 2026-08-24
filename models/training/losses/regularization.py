from __future__ import annotations

from typing import Optional

import torch

from models.gaussian.geometry import VISIBILITY_MARGIN_PIXELS, visibility_loss_per_timestep
from models.gaussian.motion import construct_chronological_gaussian_sequence
from models.splattervae.temporal import (
    combine_temporal_anchor_losses,
    temporal_anchor_index,
)


def compute_visibility_loss(
    anchor_xyz: torch.Tensor,
    delta_xyz_01: Optional[torch.Tensor],
    delta_xyz_12: Optional[torch.Tensor],
    valid_mask: torch.Tensor,
    world_view_transform: torch.Tensor,
    intrinsics: torch.Tensor,
    *,
    image_height: int,
    image_width: int,
    near_plane: float,
    far_plane: float,
    temporal_ramp: float,
    temporal_anchor: str = "t0",
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute temporal mean-over-cameras visibility regularization."""
    if delta_xyz_01 is None or delta_xyz_12 is None:
        if delta_xyz_01 is not None or delta_xyz_12 is not None:
            raise ValueError("Both temporal deltas must be provided together.")
        xyz = anchor_xyz[:, None]
        validity = valid_mask[:, None]
        per_timestep = visibility_loss_per_timestep(
            xyz, validity, world_view_transform[:, :1], intrinsics[:, :1],
            image_height=image_height, image_width=image_width,
            near_plane=near_plane, far_plane=far_plane,
            margin_pixels=VISIBILITY_MARGIN_PIXELS,
        )
        return per_timestep[0], per_timestep
    detached_anchor_pc = {
        "xyz": anchor_xyz.detach(),
        "delta_xyz_01": delta_xyz_01,
        "delta_xyz_12": delta_xyz_12,
    }
    chronological_pc = construct_chronological_gaussian_sequence(
        detached_anchor_pc, temporal_anchor
    )
    chronological_xyz = [pc["xyz"] for pc in chronological_pc]
    chronological_xyz[temporal_anchor_index(temporal_anchor)] = anchor_xyz
    xyz = torch.stack(chronological_xyz, dim=1)
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
    combined = combine_temporal_anchor_losses(
        per_timestep, temporal_anchor, temporal_ramp
    )
    return combined, per_timestep
