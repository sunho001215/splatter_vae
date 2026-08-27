from __future__ import annotations

import torch
import torch.nn.functional as F

from .camera import LAGERNVS_IMAGE_SIZE


def unproject_xlens_depth(
    depth: torch.Tensor,
    K: torch.Tensor,
    c2w: torch.Tensor,
    validity: torch.Tensor,
    confidence: torch.Tensor | None = None,
    *,
    confidence_threshold: float = 0.0,
    stride: int = 2,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Unproject X-Lens z-depth from two sources into robot-base coordinates."""

    if depth.dim() != 4 or depth.shape[1] != 1:
        raise ValueError("Depth must have shape (V,1,H,W).")
    views, _, height, width = depth.shape
    if K.shape != (views, 3, 3) or c2w.shape != (views, 4, 4):
        raise ValueError("Coverage cameras must align with source depth views.")
    if validity.shape != depth.shape:
        raise ValueError("Coverage depth validity must match depth.")
    step = int(stride)
    if step <= 0:
        raise ValueError("Coverage sampling stride must be positive.")
    u = torch.arange(0, width, step, device=depth.device, dtype=torch.float32) + 0.5
    v = torch.arange(0, height, step, device=depth.device, dtype=torch.float32) + 0.5
    vv, uu = torch.meshgrid(v, u, indexing="ij")
    z = depth[:, 0, ::step, ::step].float()
    x = (uu[None] - K[:, 0, 2, None, None]) * z / K[:, 0, 0, None, None]
    y = (vv[None] - K[:, 1, 2, None, None]) * z / K[:, 1, 1, None, None]
    camera_points = torch.stack((x, y, z), dim=-1)
    world = (
        torch.einsum("vij,vhwj->vhwi", c2w[:, :3, :3].float(), camera_points)
        + c2w[:, None, None, :3, 3].float()
    )
    valid = validity[:, 0, ::step, ::step].bool() & torch.isfinite(world).all(-1)
    if confidence is not None:
        valid &= confidence[:, 0, ::step, ::step].float() >= float(
            confidence_threshold
        )
        point_confidence = confidence[:, 0, ::step, ::step].float()[valid]
    else:
        point_confidence = torch.ones(valid.sum(), device=depth.device)
    return world[valid], point_confidence


def project_world_support(
    points_world: torch.Tensor,
    target_c2w: torch.Tensor,
    target_K: torch.Tensor,
    *,
    height: int = LAGERNVS_IMAGE_SIZE,
    width: int = LAGERNVS_IMAGE_SIZE,
    dilation_kernel: int = 5,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Project source points with a z-buffer into a candidate target view."""

    support = torch.zeros(1, height, width, dtype=torch.bool, device=points_world.device)
    depth_map = torch.full(
        (height * width,), float("inf"), device=points_world.device, dtype=torch.float32
    )
    if points_world.numel() == 0:
        return support, depth_map.reshape(1, height, width)
    w2c = torch.linalg.inv(target_c2w.float())
    camera = torch.einsum("ij,nj->ni", w2c[:3, :3], points_world.float()) + w2c[
        None, :3, 3
    ]
    z = camera[:, 2]
    u = target_K[0, 0] * camera[:, 0] / z.clamp_min(1.0e-8) + target_K[0, 2]
    v = target_K[1, 1] * camera[:, 1] / z.clamp_min(1.0e-8) + target_K[1, 2]
    x_index = torch.round(u - 0.5).long()
    y_index = torch.round(v - 0.5).long()
    valid = (
        torch.isfinite(camera).all(-1)
        & (z > 1.0e-4)
        & (x_index >= 0)
        & (x_index < width)
        & (y_index >= 0)
        & (y_index < height)
    )
    if valid.any():
        linear = y_index[valid] * width + x_index[valid]
        depth_map.scatter_reduce_(0, linear, z[valid], reduce="amin", include_self=True)
    finite = torch.isfinite(depth_map).reshape(1, 1, height, width)
    kernel = int(dilation_kernel)
    if kernel <= 0 or kernel % 2 == 0:
        raise ValueError("Coverage dilation must be a positive odd integer.")
    if kernel > 1:
        finite = F.max_pool2d(
            finite.float(), kernel_size=kernel, stride=1, padding=kernel // 2
        ) > 0.5
    return finite[0], depth_map.reshape(1, height, width)


def source_coverage_from_xlens(
    depth: torch.Tensor,
    confidence: torch.Tensor,
    validity: torch.Tensor,
    source_K: torch.Tensor,
    source_c2w: torch.Tensor,
    target_K: torch.Tensor,
    target_c2w: torch.Tensor,
    *,
    confidence_threshold: float = 0.0,
    sample_stride: int = 2,
    dilation_kernel: int = 5,
) -> dict[str, torch.Tensor]:
    points, point_confidence = unproject_xlens_depth(
        depth,
        source_K,
        source_c2w,
        validity,
        confidence,
        confidence_threshold=confidence_threshold,
        stride=sample_stride,
    )
    support, zbuffer = project_world_support(
        points,
        target_c2w,
        target_K,
        dilation_kernel=dilation_kernel,
    )
    camera_center = target_c2w[:3, 3]
    minimum_geometry_distance = (
        (points - camera_center).norm(dim=-1).amin()
        if points.numel()
        else camera_center.new_tensor(float("inf"))
    )
    return {
        "support_mask": support,
        "coverage_fraction": support.float().mean(),
        "target_zbuffer": zbuffer,
        "minimum_geometry_distance": minimum_geometry_distance,
        "source_point_count": torch.tensor(
            points.shape[0], device=points.device, dtype=torch.long
        ),
        "source_confidence_mean": (
            point_confidence.mean()
            if point_confidence.numel()
            else points.new_zeros(())
        ),
        "points_world": points,
    }
