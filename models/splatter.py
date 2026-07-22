from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import torch
import torch.nn.functional as F

from gsplat import spherical_harmonics
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
    """Direct decoder-to-3D-Gaussian configuration."""

    max_sh_degree: int = 1
    gaussians_per_pixel: int = 1

    depth_min: Optional[float] = None
    depth_max: Optional[float] = None
    depth_activation: str = "inverse_depth"
    depth_ordering: str = "sort"
    depth_scale: float = 1.0
    depth_bias: float = 0.0

    gaussian_offset_scale: float = 0.05
    offset_scale: float = 1.0
    offset_bias: float = 0.0

    gaussian_scale_min: float = 1.0e-4
    gaussian_scale_max: float = 0.05
    scale_scale: float = 1.0
    scale_bias: float = 0.0

    rotation_scale: float = 1.0
    rotation_bias: float = 0.0

    opacity_scale: float = 1.0
    opacity_bias: float = 0.0

    color_scale: float = 1.0
    color_bias: float = 0.0

    sh_rest_scale: float = 0.1
    sh_rest_raw_scale: float = 1.0
    sh_rest_raw_bias: float = 0.0


@dataclass
class SplatterConfig:
    data: SplatterDataConfig
    model: SplatterModelConfig


def gaussian_params_per_gaussian(max_sh_degree: int = 1) -> int:
    """Channels predicted for one Gaussian at one pixel."""
    sh_bases = (int(max_sh_degree) + 1) ** 2
    sh_rest = max(0, sh_bases - 1) * 3
    return 1 + 3 + 3 + 4 + 1 + 3 + sh_rest


def default_splatter_channels(gaussians_per_pixel: int = 1, max_sh_degree: int = 1) -> int:
    """Return decoder channels for direct pixel-wise 3D Gaussian prediction."""
    return int(gaussians_per_pixel) * gaussian_params_per_gaussian(max_sh_degree=max_sh_degree)


def _depth_render_mode(render_mode: str) -> bool:
    return str(render_mode).upper() in {"D", "ED"}


def _rgb_depth_render_mode(render_mode: str) -> bool:
    return str(render_mode).upper() in {"RGB+D", "RGB+ED"}


def render_predicted(
    pc: Dict[str, torch.Tensor],
    world_view_transform: torch.Tensor,
    intrinsics: torch.Tensor,
    bg_color: torch.Tensor,
    cfg: SplatterConfig,
    scaling_modifier: float = 1.0,
    override_color: Optional[torch.Tensor] = None,
    override_opacity: Optional[float | torch.Tensor] = None,
    detach_xyz: bool = False,
    detach_scale_rotation: bool = False,
    packed: bool = False,
    render_mode: str = "RGB",
) -> Dict[str, torch.Tensor]:
    """Render a batch of 3D Gaussians with gsplat."""
    device = pc["xyz"].device
    if device.type != "cuda":
        raise RuntimeError(
            "gsplat rasterization requires CUDA tensors, but Gaussian tensors are on "
            f"{device}. Run with --device cuda, or skip render-dependent code paths."
        )

    dtype = pc["xyz"].dtype
    world_view_transform = world_view_transform.to(device=device, dtype=dtype)
    intrinsics = intrinsics.to(device=device, dtype=dtype)
    height = int(cfg.data.img_height)
    width = int(cfg.data.img_width)

    xyz = pc["xyz"].detach() if detach_xyz else pc["xyz"]
    means = torch.nan_to_num(xyz, nan=0.0, posinf=0.0, neginf=0.0).clamp(-1.0e3, 1.0e3)

    scale_max = max(float(cfg.model.gaussian_scale_max), 1.0e-5)
    scale_min = min(max(float(cfg.model.gaussian_scale_min), 1.0e-8), scale_max)
    scaling = pc["scaling"].detach() if detach_scale_rotation else pc["scaling"]
    scales = scaling * float(scaling_modifier)
    scales = torch.nan_to_num(scales, nan=scale_min, posinf=scale_max, neginf=scale_min).clamp(scale_min, scale_max)

    rotation = pc["rotation"].detach() if detach_scale_rotation else pc["rotation"]
    quats = F.normalize(torch.nan_to_num(rotation, nan=0.0, posinf=0.0, neginf=0.0), dim=-1, eps=1.0e-6)

    if override_opacity is None:
        opacity_tensor = pc["opacity"].squeeze(-1)
    elif torch.is_tensor(override_opacity):
        opacity_tensor = override_opacity.to(device=device, dtype=dtype)
        if opacity_tensor.ndim == 0:
            opacity_tensor = opacity_tensor.expand(pc["xyz"].shape[:-1])
        elif opacity_tensor.shape[-1] == 1:
            opacity_tensor = opacity_tensor.squeeze(-1)
    else:
        opacity_tensor = torch.full(pc["xyz"].shape[:-1], float(override_opacity), device=device, dtype=dtype)
    opacities = torch.nan_to_num(opacity_tensor, nan=0.0, posinf=1.0, neginf=0.0).clamp(0.0, 1.0)

    valid_mask = pc.get("valid_mask", None)
    if valid_mask is not None:
        opacities = opacities * valid_mask.to(device=device, dtype=opacities.dtype)

    if _depth_render_mode(render_mode) and override_color is None:
        colors = None
        sh_degree = None
    elif override_color is not None:
        colors = override_color.to(device=device, dtype=dtype)
        sh_degree = None
    else:
        features_dc = pc["features_dc"]
        features_rest = pc.get("features_rest", None)
        if features_rest is not None and features_rest.numel() > 0:
            colors = torch.cat([features_dc, features_rest], dim=-2)
            sh_degree = int(cfg.model.max_sh_degree)
        else:
            colors = features_dc
            sh_degree = 0

    camera_shape = world_view_transform.shape[:-2]
    if bg_color.dim() == 1:
        backgrounds = bg_color.to(device=device, dtype=dtype).view(
            *((1,) * len(camera_shape)), 3
        ).expand(*camera_shape, 3)
    else:
        backgrounds = bg_color.to(device=device, dtype=dtype).expand(*camera_shape, 3)

    # gsplat 1.5.x accepts arbitrary Gaussian/camera batch dimensions for
    # activated colors, but its SH helper requires coefficients shaped [N,K,D].
    # Flatten all batch-camera-Gaussian rows for one external SH evaluation,
    # then render the camera-conditioned RGB values with sh_degree=None.
    if sh_degree is not None and colors is not None and colors.dim() > 3:
        gaussian_batch_shape = means.shape[:-2]
        if colors.shape[:-3] != gaussian_batch_shape:
            raise ValueError(
                f"SH coefficient batch shape {colors.shape[:-3]} does not match means {gaussian_batch_shape}."
            )
        camera_count = world_view_transform.shape[-3]
        rotation_w2c = world_view_transform[..., :3, :3]
        translation_w2c = world_view_transform[..., :3, 3]
        camera_positions = -torch.matmul(
            rotation_w2c.transpose(-1, -2), translation_w2c.unsqueeze(-1)
        ).squeeze(-1)
        directions = F.normalize(
            means.unsqueeze(-3) - camera_positions.unsqueeze(-2),
            p=2,
            dim=-1,
            eps=1.0e-8,
        )
        camera_coefficients = colors.unsqueeze(-4).expand(
            *gaussian_batch_shape,
            camera_count,
            colors.shape[-3],
            colors.shape[-2],
            colors.shape[-1],
        )
        flat_colors = spherical_harmonics(
            sh_degree,
            directions.reshape(-1, 3),
            camera_coefficients.reshape(-1, colors.shape[-2], colors.shape[-1]),
        )
        colors = torch.clamp_min(flat_colors + 0.5, 0.0).view(
            *gaussian_batch_shape,
            camera_count,
            means.shape[-2],
            colors.shape[-1],
        )
        sh_degree = None

    raster_kwargs = dict(
        width=width,
        height=height,
        near_plane=float(cfg.data.znear),
        far_plane=float(cfg.data.zfar),
        packed=packed,
    )

    render_colors, render_alphas, meta = rasterization(
        means=means,
        quats=quats,
        scales=scales,
        opacities=opacities,
        colors=colors,
        viewmats=world_view_transform,
        Ks=intrinsics,
        backgrounds=backgrounds,
        sh_degree=sh_degree,
        render_mode=render_mode,
        **raster_kwargs,
    )

    if _depth_render_mode(render_mode):
        rendered_image = None
        rendered_depth = render_colors.movedim(-1, -3).contiguous()
    elif _rgb_depth_render_mode(render_mode):
        if render_colors.shape[-1] != 4:
            raise RuntimeError(
                f"{render_mode} rendering returned {render_colors.shape[-1]} channels; expected RGB plus one depth channel."
            )
        rendered_image = render_colors[..., :3].movedim(-1, -3).contiguous()
        rendered_depth = render_colors[..., 3:4].movedim(-1, -3).contiguous()
    else:
        rendered_image = render_colors.movedim(-1, -3).contiguous()
        rendered_depth = None

    rendered_alpha = render_alphas.movedim(-1, -3).contiguous()
    radii = meta.get("radii", None)
    visibility_filter = (radii > 0) if torch.is_tensor(radii) else None

    return {
        "render": rendered_image,
        "depth": rendered_depth,
        "alpha": rendered_alpha,
        "viewspace_points": None,
        "visibility_filter": visibility_filter,
        "radii": radii,
    }
