from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import torch
import torch.nn.functional as F

from gsplat.rendering import rasterization


@dataclass
class SplatterDataConfig:
    """Camera and image configuration used by the Gaussian renderer."""

    img_height: int = 128
    img_width: int = 128
    znear: float = 0.1
    zfar: float = 2.0
    white_background: bool = False
    inverted_x: bool = False
    inverted_y: bool = False
    category: str = "generic"


@dataclass
class SplatterModelConfig:
    """Point-proposal, voxelization, PointNeXt, and Gaussian-head config."""

    max_sh_degree: int = 1
    points_per_pixel: int = 2
    point_offset_scale: float = 0.05

    voxel_size: float = 0.02
    voxelization_type: str = "trilinear_soft"
    active_voxel_threshold: float = 1.0e-4

    pointnet_channels: int = 128
    pointnet_depth: int = 4
    pointnet_neighbors: int = 16

    gaussians_per_voxel: int = 1
    gaussian_scale_max: float = 0.05
    gaussian_local_offset_scale: float = 0.5


@dataclass
class SplatterConfig:
    data: SplatterDataConfig
    model: SplatterModelConfig


def render_predicted(
    pc: Dict[str, torch.Tensor],
    world_view_transform: torch.Tensor,
    intrinsics: torch.Tensor,
    bg_color: torch.Tensor,
    cfg: SplatterConfig,
    scaling_modifier: float = 1.0,
    override_color: Optional[torch.Tensor] = None,
    packed: bool = False,
    render_mode: str = "RGB",
) -> Dict[str, torch.Tensor]:
    """Render a batch of 3D Gaussians with gsplat.

    The renderer-facing dictionary contract is intentionally unchanged:
    ``xyz``, ``scaling``, ``rotation``, ``opacity``, ``features_dc``, and
    ``features_rest`` are all batch-first tensors.
    """
    device = pc["xyz"].device
    if device.type != "cuda":
        raise RuntimeError(
            "gsplat rasterization requires CUDA tensors, but Gaussian tensors are on "
            f"{device}. Run with --device cuda, or skip render-dependent code paths."
        )

    world_view_transform = world_view_transform.to(device=device, dtype=pc["xyz"].dtype)
    intrinsics = intrinsics.to(device=device, dtype=pc["xyz"].dtype)
    height = int(cfg.data.img_height)
    width = int(cfg.data.img_width)

    means = torch.nan_to_num(pc["xyz"], nan=0.0, posinf=0.0, neginf=0.0).clamp(-1.0e3, 1.0e3)
    scale_max = max(
        float(getattr(cfg.model, "gaussian_scale_max", getattr(cfg.model, "scale_max", 0.05))),
        1.0e-5,
    )
    scales = pc["scaling"] * float(scaling_modifier)
    scales = torch.nan_to_num(scales, nan=1.0e-4, posinf=scale_max, neginf=1.0e-4).clamp(1.0e-5, scale_max)
    quats = F.normalize(torch.nan_to_num(pc["rotation"], nan=0.0, posinf=0.0, neginf=0.0), dim=-1, eps=1.0e-6)
    opacities = torch.nan_to_num(pc["opacity"].squeeze(-1), nan=0.0, posinf=1.0, neginf=0.0).clamp(0.0, 1.0)

    valid_mask = pc.get("valid_mask", None)
    if valid_mask is not None:
        opacities = opacities * valid_mask.to(device=device, dtype=opacities.dtype)

    if override_color is not None:
        colors = override_color.to(device=device, dtype=pc["xyz"].dtype)
        sh_degree = None
    else:
        features_dc = pc["features_dc"]
        features_rest = pc.get("features_rest", None)
        if features_rest is not None and features_rest.numel() > 0:
            colors = torch.cat([features_dc, features_rest], dim=2)
            sh_degree = int(cfg.model.max_sh_degree)
        else:
            colors = features_dc
            sh_degree = 0

    if bg_color.dim() == 1:
        batch, views = world_view_transform.shape[:2]
        backgrounds = bg_color.to(device=device, dtype=pc["xyz"].dtype).view(1, 1, 3).expand(batch, views, 3)
    else:
        backgrounds = bg_color.to(device=device, dtype=pc["xyz"].dtype)

    render_colors, render_alphas, meta = rasterization(
        means=means,
        quats=quats,
        scales=scales,
        opacities=opacities,
        colors=colors,
        viewmats=world_view_transform,
        Ks=intrinsics,
        width=width,
        height=height,
        near_plane=float(cfg.data.znear),
        far_plane=float(cfg.data.zfar),
        sh_degree=sh_degree,
        backgrounds=backgrounds,
        packed=packed,
        render_mode=render_mode,
    )

    if render_mode in {"D", "ED", "d", "Ed"}:
        rendered_image = None
        rendered_depth = render_colors.permute(0, 1, 4, 2, 3).contiguous()
    elif render_colors.shape[-1] > 3:
        rendered_image = render_colors[..., :3].permute(0, 1, 4, 2, 3).contiguous()
        rendered_depth = render_colors[..., 3:4].permute(0, 1, 4, 2, 3).contiguous()
    else:
        rendered_image = render_colors.permute(0, 1, 4, 2, 3).contiguous()
        rendered_depth = None

    rendered_alpha = render_alphas.permute(0, 1, 4, 2, 3).contiguous()
    radii = meta.get("radii", None)

    return {
        "render": rendered_image,
        "depth": rendered_depth,
        "alpha": rendered_alpha,
        "viewspace_points": None,
        "visibility_filter": radii > 0 if radii is not None else None,
        "radii": radii,
    }


def default_splatter_channels(points_per_pixel: int = 2) -> int:
    """Return decoder channels for the pixel-wise point proposal map.

    Each pixel predicts ``points_per_pixel`` proposals, and each proposal stores
    metric-depth logits, a 3D camera-space offset, and a confidence logit.
    """
    return int(points_per_pixel) * 5
