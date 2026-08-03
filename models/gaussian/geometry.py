from __future__ import annotations

import torch

FRUSTUM_MARGIN_PIXELS = 4


def project_gaussian_centers(
    xyz: torch.Tensor,
    world_view_transform: torch.Tensor,
    intrinsics: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Project ``(B,T,N,3)`` world centers into ``(B,T,A,N)`` cameras in FP32."""
    xyz = xyz.float()
    w2c = world_view_transform.float()
    camera_k = intrinsics.float()
    if xyz.dim() != 4 or xyz.shape[-1] != 3:
        raise ValueError(f"Expected xyz as (B,T,N,3), got {tuple(xyz.shape)}.")
    if w2c.dim() != 5 or w2c.shape[:2] != xyz.shape[:2] or w2c.shape[-2:] != (4, 4):
        raise ValueError("World-to-camera matrices must align with the center batch and timesteps.")
    if camera_k.shape != (*w2c.shape[:3], 3, 3):
        raise ValueError("Intrinsics must align with world-to-camera matrices.")
    finite_world = torch.isfinite(xyz).all(-1)
    safe_xyz = torch.nan_to_num(xyz, nan=0.0, posinf=0.0, neginf=0.0)
    camera_xyz = torch.einsum("btaij,btnj->btani", w2c[..., :3, :3], safe_xyz)
    camera_xyz = camera_xyz + w2c[..., :3, 3].unsqueeze(-2)
    homogeneous = torch.einsum("btaij,btanj->btani", camera_k, camera_xyz)
    denominator = homogeneous[..., 2]
    safe_denominator = torch.where(denominator.abs() > 1.0e-8, denominator, torch.ones_like(denominator))
    pixel_xy = homogeneous[..., :2] / safe_denominator[..., None]
    depth = camera_xyz[..., 2]
    finite = (
        finite_world[:, :, None]
        & torch.isfinite(camera_xyz).all(-1)
        & torch.isfinite(pixel_xy).all(-1)
        & (denominator.abs() > 1.0e-8)
    )
    return pixel_xy, depth, finite


def union_frustum_loss_per_timestep(
    xyz: torch.Tensor,
    valid_mask: torch.Tensor,
    world_view_transform: torch.Tensor,
    intrinsics: torch.Tensor,
    *,
    image_height: int,
    image_width: int,
    near_plane: float,
    far_plane: float,
    margin_pixels: int = FRUSTUM_MARGIN_PIXELS,
) -> torch.Tensor:
    """Return the all-camera union-frustum penalty for each timestep."""
    if float(far_plane) <= float(near_plane):
        raise ValueError("far_plane must be greater than near_plane.")
    if valid_mask.shape != xyz.shape[:-1]:
        raise ValueError("Gaussian validity must align with xyz.")
    pixel_xy, depth, finite = project_gaussian_centers(xyz, world_view_transform, intrinsics)
    u, v = pixel_xy.unbind(-1)
    margin = float(margin_pixels)
    du = (
        torch.relu(-margin - u)
        + torch.relu(u - (float(image_width - 1) + margin))
    ) / float(image_width)
    dv = (
        torch.relu(-margin - v)
        + torch.relu(v - (float(image_height - 1) + margin))
    ) / float(image_height)
    dz = (
        torch.relu(float(near_plane) - depth)
        + torch.relu(depth - float(far_plane))
    ) / float(far_plane - near_plane)
    violation = du + dv + dz
    violation = torch.where(finite, violation, torch.ones_like(violation))
    union_violation = violation.amin(dim=2)
    weights = valid_mask.to(dtype=union_violation.dtype)
    per_batch_timestep = (union_violation * weights).sum(-1) / weights.sum(-1).clamp_min(1.0)
    return per_batch_timestep.mean(0)
