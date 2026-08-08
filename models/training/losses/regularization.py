from __future__ import annotations

from typing import Optional

import torch

from models.gaussian.geometry import FRUSTUM_MARGIN_PIXELS, union_frustum_loss_per_timestep


def compute_union_frustum_loss(
    base_xyz: torch.Tensor,
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
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute t0-only or ramped temporal union-frustum regularization."""
    if delta_xyz_01 is None or delta_xyz_12 is None:
        if delta_xyz_01 is not None or delta_xyz_12 is not None:
            raise ValueError("Both temporal deltas must be provided together.")
        xyz = base_xyz[:, None]
        validity = valid_mask[:, None]
        per_timestep = union_frustum_loss_per_timestep(
            xyz, validity, world_view_transform[:, :1], intrinsics[:, :1],
            image_height=image_height, image_width=image_width,
            near_plane=near_plane, far_plane=far_plane,
            margin_pixels=FRUSTUM_MARGIN_PIXELS,
        )
        return per_timestep[0], per_timestep
    detached_base = base_xyz.detach()
    xyz = torch.stack(
        (
            base_xyz,
            detached_base + delta_xyz_01,
            detached_base + delta_xyz_01 + delta_xyz_12,
        ),
        dim=1,
    )
    validity = valid_mask[:, None].expand(-1, 3, -1)
    per_timestep = union_frustum_loss_per_timestep(
        xyz,
        validity,
        world_view_transform,
        intrinsics,
        image_height=image_height,
        image_width=image_width,
        near_plane=near_plane,
        far_plane=far_plane,
        margin_pixels=FRUSTUM_MARGIN_PIXELS,
    )
    combined = per_timestep[0] + float(temporal_ramp) * 0.5 * (per_timestep[1] + per_timestep[2])
    return combined, per_timestep
