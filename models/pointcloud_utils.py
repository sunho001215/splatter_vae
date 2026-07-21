from __future__ import annotations

from typing import Optional, Tuple

import torch


@torch.no_grad()
def sample_points(
    points: torch.Tensor,
    mask: torch.Tensor,
    max_points: Optional[int],
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Detach and randomly cap a masked point set for qualitative logging."""
    points = points.detach()
    valid = mask.detach().to(device=points.device, dtype=torch.bool) & torch.isfinite(points).all(dim=-1)
    if not bool(valid.any()):
        return points.new_zeros((1, 3)), torch.zeros((1,), device=points.device, dtype=torch.bool)

    selected = points[valid]
    if max_points is not None and int(max_points) > 0 and selected.shape[0] > int(max_points):
        indices = torch.randperm(selected.shape[0], device=points.device)[: int(max_points)]
        selected = selected[indices]
    return selected, torch.ones((selected.shape[0],), device=points.device, dtype=torch.bool)


@torch.no_grad()
def depths_to_world_point_cloud(
    depths: Optional[torch.Tensor],
    intrinsics: torch.Tensor,
    c2w: torch.Tensor,
    masks: Optional[torch.Tensor] = None,
    max_points_per_view: Optional[int] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Back-project detached, valid foreground depths from every camera.

    Depth is interpreted as OpenCV camera-z depth.  A separate validity tensor is
    not required: finite positive samples define depth validity, and ``masks``
    supplies the exact foreground region.  Each camera is capped independently
    before its points are concatenated so multi-camera logging stays balanced.
    """
    if depths is None:
        device = intrinsics.device
        return (
            torch.zeros((intrinsics.shape[0], 1, 3), device=device),
            torch.zeros((intrinsics.shape[0], 1), device=device, dtype=torch.bool),
        )

    depths = depths.detach()
    intrinsics = intrinsics.detach().to(device=depths.device, dtype=depths.dtype)
    c2w = c2w.detach().to(device=depths.device, dtype=depths.dtype)
    if depths.dim() != 5 or depths.shape[2] != 1:
        raise ValueError(f"Expected depths as (B,A,1,H,W), got {tuple(depths.shape)}.")

    bsz, num_views, _, height, width = depths.shape
    device = depths.device
    dtype = depths.dtype
    foreground = None
    if masks is not None:
        foreground = masks.detach().to(device=device, dtype=torch.bool)
        if foreground.shape != depths.shape:
            raise ValueError(
                f"Depths and masks must share (B,A,1,H,W), got {tuple(depths.shape)} and {tuple(foreground.shape)}."
            )

    ys, xs = torch.meshgrid(
        torch.arange(height, device=device, dtype=dtype) + 0.5,
        torch.arange(width, device=device, dtype=dtype) + 0.5,
        indexing="ij",
    )

    point_lists: list[torch.Tensor] = []
    mask_lists: list[torch.Tensor] = []
    per_view_cap = None if max_points_per_view is None else max(1, int(max_points_per_view))
    for batch_idx in range(bsz):
        view_points: list[torch.Tensor] = []
        for view_idx in range(num_views):
            depth = depths[batch_idx, view_idx, 0]
            valid = torch.isfinite(depth) & (depth > 0.0)
            if foreground is not None:
                valid &= foreground[batch_idx, view_idx, 0]
            if not bool(valid.any()):
                continue

            k = intrinsics[batch_idx, view_idx]
            z = depth[valid]
            x = (xs[valid] - k[0, 2]) / k[0, 0].clamp_min(1.0e-6) * z
            y = (ys[valid] - k[1, 2]) / k[1, 1].clamp_min(1.0e-6) * z
            points_camera = torch.stack([x, y, z], dim=-1)
            rotation = c2w[batch_idx, view_idx, :3, :3]
            translation = c2w[batch_idx, view_idx, :3, 3]
            points_world = points_camera @ rotation.transpose(0, 1) + translation.view(1, 3)
            points_world = points_world[torch.isfinite(points_world).all(dim=-1)]
            if per_view_cap is not None and points_world.shape[0] > per_view_cap:
                indices = torch.randperm(points_world.shape[0], device=device)[:per_view_cap]
                points_world = points_world[indices]
            if points_world.numel() > 0:
                view_points.append(points_world)

        if view_points:
            selected = torch.cat(view_points, dim=0)
            selected_mask = torch.ones((selected.shape[0],), device=device, dtype=torch.bool)
        else:
            selected = depths.new_zeros((1, 3))
            selected_mask = torch.zeros((1,), device=device, dtype=torch.bool)
        point_lists.append(selected)
        mask_lists.append(selected_mask)

    max_count = max(points.shape[0] for points in point_lists)
    padded_points: list[torch.Tensor] = []
    padded_masks: list[torch.Tensor] = []
    for points, valid in zip(point_lists, mask_lists):
        pad_count = max_count - points.shape[0]
        if pad_count > 0:
            points = torch.cat([points, points.new_zeros((pad_count, 3))], dim=0)
            valid = torch.cat([valid, torch.zeros((pad_count,), device=device, dtype=torch.bool)], dim=0)
        padded_points.append(points)
        padded_masks.append(valid)
    return torch.stack(padded_points, dim=0), torch.stack(padded_masks, dim=0)
