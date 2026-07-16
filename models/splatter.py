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
    return render_mode in {"D", "ED", "d", "Ed"}


def render_predicted(
    pc: Dict[str, torch.Tensor],
    world_view_transform: torch.Tensor,
    intrinsics: torch.Tensor,
    bg_color: torch.Tensor,
    cfg: SplatterConfig,
    scaling_modifier: float = 1.0,
    override_color: Optional[torch.Tensor] = None,
    override_opacity: Optional[float | torch.Tensor] = None,
    occupancy_opacity: Optional[float | torch.Tensor] = None,
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
            opacity_tensor = opacity_tensor.expand(pc["xyz"].shape[:2])
        elif opacity_tensor.shape[-1] == 1:
            opacity_tensor = opacity_tensor.squeeze(-1)
    else:
        opacity_tensor = torch.full(pc["xyz"].shape[:2], float(override_opacity), device=device, dtype=dtype)
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
            colors = torch.cat([features_dc, features_rest], dim=2)
            sh_degree = int(cfg.model.max_sh_degree)
        else:
            colors = features_dc
            sh_degree = 0

    if bg_color.dim() == 1:
        batch, views = world_view_transform.shape[:2]
        backgrounds = bg_color.to(device=device, dtype=dtype).view(1, 1, 3).expand(batch, views, 3)
    else:
        backgrounds = bg_color.to(device=device, dtype=dtype)

    raster_kwargs = dict(
        width=width,
        height=height,
        near_plane=float(cfg.data.znear),
        far_plane=float(cfg.data.zfar),
        packed=packed,
    )

    def rasterize_path(
        path_means: torch.Tensor,
        path_quats: torch.Tensor,
        path_scales: torch.Tensor,
        path_opacities: torch.Tensor,
        path_colors: Optional[torch.Tensor],
        path_backgrounds: torch.Tensor,
        path_sh_degree: Optional[int],
        path_render_mode: str,
    ) -> tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        kwargs = dict(raster_kwargs, sh_degree=path_sh_degree, render_mode=path_render_mode)
        # gsplat accepts batched Gaussians for activated RGB colors, but SH
        # coefficients must be passed as an unbatched [N, K, D] tensor. The
        # decoder predicts SH coefficients per sample, so render each sample
        # independently and stack back to the usual [B, V, H, W, C] layout.
        if path_sh_degree is not None and path_means.dim() == 3:
            color_items = []
            alpha_items = []
            meta_items = []
            for batch_idx in range(path_means.shape[0]):
                item_colors = None if path_colors is None else path_colors[batch_idx]
                item_bg = path_backgrounds[batch_idx] if path_backgrounds.dim() >= 3 else path_backgrounds
                item_colors_out, item_alphas, item_meta = rasterization(
                    means=path_means[batch_idx],
                    quats=path_quats[batch_idx],
                    scales=path_scales[batch_idx],
                    opacities=path_opacities[batch_idx],
                    colors=item_colors,
                    viewmats=world_view_transform[batch_idx],
                    Ks=intrinsics[batch_idx],
                    backgrounds=item_bg,
                    **kwargs,
                )
                color_items.append(item_colors_out)
                alpha_items.append(item_alphas)
                meta_items.append(item_meta)
            out_colors = torch.stack(color_items, dim=0)
            out_alphas = torch.stack(alpha_items, dim=0)
            out_meta: Dict[str, torch.Tensor] = {}
            for key in set().union(*(item.keys() for item in meta_items)):
                values = [item.get(key) for item in meta_items]
                if all(torch.is_tensor(value) for value in values):
                    try:
                        out_meta[key] = torch.stack(values, dim=0)
                    except RuntimeError:
                        out_meta[key] = values
                else:
                    out_meta[key] = values
            return out_colors, out_alphas, out_meta

        return rasterization(
            means=path_means,
            quats=path_quats,
            scales=path_scales,
            opacities=path_opacities,
            colors=path_colors,
            viewmats=world_view_transform,
            Ks=intrinsics,
            backgrounds=path_backgrounds,
            **kwargs,
        )

    render_colors, render_alphas, meta = rasterize_path(
        path_means=means,
        path_quats=quats,
        path_scales=scales,
        path_opacities=opacities,
        path_colors=colors,
        path_backgrounds=backgrounds,
        path_sh_degree=sh_degree,
        path_render_mode=render_mode,
    )

    occupancy_alphas = None
    if occupancy_opacity is not None:
        if torch.is_tensor(occupancy_opacity):
            occupancy_opacity_tensor = occupancy_opacity.to(device=device, dtype=dtype)
            if occupancy_opacity_tensor.ndim == 0:
                occupancy_opacity_tensor = occupancy_opacity_tensor.expand(pc["xyz"].shape[:2])
            elif occupancy_opacity_tensor.shape[-1] == 1:
                occupancy_opacity_tensor = occupancy_opacity_tensor.squeeze(-1)
        else:
            occupancy_opacity_tensor = torch.full(
                pc["xyz"].shape[:2],
                float(occupancy_opacity),
                device=device,
                dtype=dtype,
            )
        occupancy_opacities = torch.nan_to_num(
            occupancy_opacity_tensor,
            nan=0.0,
            posinf=1.0,
            neginf=0.0,
        ).clamp(0.0, 1.0)
        if valid_mask is not None:
            occupancy_opacities = occupancy_opacities * valid_mask.to(device=device, dtype=occupancy_opacities.dtype)
        occupancy_colors = means.new_zeros((*means.shape[:-1], 3))
        _, occupancy_alphas, _ = rasterize_path(
            path_means=means,
            path_quats=quats.detach(),
            path_scales=scales.detach(),
            path_opacities=occupancy_opacities,
            path_colors=occupancy_colors,
            path_backgrounds=torch.zeros_like(backgrounds),
            path_sh_degree=None,
            path_render_mode="RGB",
        )

    if _depth_render_mode(render_mode):
        rendered_image = None
        rendered_depth = render_colors.permute(0, 1, 4, 2, 3).contiguous()
    elif render_colors.shape[-1] > 3:
        rendered_image = render_colors[..., :3].permute(0, 1, 4, 2, 3).contiguous()
        rendered_depth = render_colors[..., 3:4].permute(0, 1, 4, 2, 3).contiguous()
    else:
        rendered_image = render_colors.permute(0, 1, 4, 2, 3).contiguous()
        rendered_depth = None

    rendered_alpha = render_alphas.permute(0, 1, 4, 2, 3).contiguous()
    occupancy_alpha = None
    if occupancy_alphas is not None:
        occupancy_alpha = occupancy_alphas.permute(0, 1, 4, 2, 3).contiguous()
    radii = meta.get("radii", None)
    visibility_filter = (radii > 0) if torch.is_tensor(radii) else None

    return {
        "render": rendered_image,
        "depth": rendered_depth,
        "alpha": rendered_alpha,
        "occupancy_alpha": occupancy_alpha,
        "viewspace_points": None,
        "visibility_filter": visibility_filter,
        "radii": radii,
    }
