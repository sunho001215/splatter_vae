from __future__ import annotations

from typing import Dict

import torch
import torch.nn.functional as F

from .parameterization import SplatterConfig


def _load_gsplat_ops():
    try:
        from gsplat import spherical_harmonics
        from gsplat.rendering import rasterization
    except Exception as exc:
        raise RuntimeError(
            "gsplat could not be loaded. Rebuild its CUDA extension against the installed PyTorch."
        ) from exc
    return spherical_harmonics, rasterization


def _activated_camera_colors(
    pc: Dict[str, torch.Tensor],
    means: torch.Tensor,
    world_view_transform: torch.Tensor,
    sh_degree: int,
    spherical_harmonics,
) -> torch.Tensor:
    coefficients = torch.cat((pc["features_dc"].float(), pc["features_rest"].float()), dim=-2)
    gaussian_batch_shape = means.shape[:-2]
    if coefficients.shape[:-3] != gaussian_batch_shape:
        raise ValueError("SH coefficient batch dimensions must match Gaussian means.")
    camera_count = world_view_transform.shape[-3]
    rotation = world_view_transform[..., :3, :3]
    translation = world_view_transform[..., :3, 3]
    camera_positions = -torch.matmul(rotation.transpose(-1, -2), translation[..., None]).squeeze(-1)
    directions = F.normalize(means.unsqueeze(-3) - camera_positions.unsqueeze(-2), dim=-1, eps=1.0e-8)
    camera_coefficients = coefficients.unsqueeze(-4).expand(
        *gaussian_batch_shape,
        camera_count,
        coefficients.shape[-3],
        coefficients.shape[-2],
        coefficients.shape[-1],
    )
    evaluated = spherical_harmonics(
        sh_degree,
        directions.reshape(-1, 3),
        camera_coefficients.reshape(-1, coefficients.shape[-2], coefficients.shape[-1]),
    )
    return evaluated.add(0.5).clamp_min(0.0).view(
        *gaussian_batch_shape, camera_count, means.shape[-2], coefficients.shape[-1]
    )


def _render(
    pc: Dict[str, torch.Tensor],
    world_view_transform: torch.Tensor,
    intrinsics: torch.Tensor,
    bg_color: torch.Tensor,
    cfg: SplatterConfig,
    *,
    with_expected_depth: bool,
) -> Dict[str, torch.Tensor]:
    if pc["xyz"].device.type != "cuda":
        raise RuntimeError("gsplat rasterization requires CUDA tensors.")
    spherical_harmonics, rasterization = _load_gsplat_ops()
    # Gaussian conversion and all renderer inputs stay FP32 even when the neural
    # prediction path runs under BF16 autocast.
    means = torch.nan_to_num(pc["xyz"].float(), nan=0.0, posinf=0.0, neginf=0.0).clamp(-1.0e3, 1.0e3)
    w2c = world_view_transform.float()
    camera_k = intrinsics.float()
    minimum = max(float(cfg.model.gaussian_scale_min), 1.0e-8)
    maximum = max(float(cfg.model.gaussian_scale_max), minimum)
    scales = torch.nan_to_num(pc["scaling"].float(), nan=minimum, posinf=maximum, neginf=minimum).clamp(minimum, maximum)
    quaternions = F.normalize(
        torch.nan_to_num(pc["rotation"].float(), nan=0.0, posinf=0.0, neginf=0.0),
        dim=-1,
        eps=1.0e-6,
    )
    opacities = torch.nan_to_num(pc["opacity"].float().squeeze(-1), nan=0.0, posinf=1.0, neginf=0.0).clamp(0.0, 1.0)
    opacities = opacities * pc["valid_mask"].to(opacities.dtype)
    colors = _activated_camera_colors(
        pc, means, w2c, int(cfg.model.max_sh_degree), spherical_harmonics
    )
    camera_shape = w2c.shape[:-2]
    if bg_color.dim() == 1:
        backgrounds = bg_color.float().view(*((1,) * len(camera_shape)), 3).expand(*camera_shape, 3)
    else:
        backgrounds = bg_color.float().expand(*camera_shape, 3)
    rendered, alpha, meta = rasterization(
        means=means,
        quats=quaternions,
        scales=scales,
        opacities=opacities,
        colors=colors,
        viewmats=w2c,
        Ks=camera_k,
        backgrounds=backgrounds,
        width=int(cfg.data.img_width),
        height=int(cfg.data.img_height),
        near_plane=float(cfg.data.znear),
        far_plane=float(cfg.data.zfar),
        packed=False,
        segmented=False,
        sh_degree=None,
        render_mode="RGB+ED" if with_expected_depth else "RGB",
        sparse_grad=False,
        absgrad=False,
    )
    if with_expected_depth:
        if rendered.shape[-1] != 4:
            raise RuntimeError("RGB+ED rasterization must return four channels.")
        image = rendered[..., :3].movedim(-1, -3).contiguous()
        depth = rendered[..., 3:4].movedim(-1, -3).contiguous()
    else:
        image = rendered.movedim(-1, -3).contiguous()
        depth = None
    radii = meta.get("radii")
    return {
        "render": image,
        "depth": depth,
        "alpha": alpha.movedim(-1, -3).contiguous(),
        "viewspace_points": None,
        "visibility_filter": (radii > 0) if torch.is_tensor(radii) else None,
        "radii": radii,
    }


def render_rgb_depth(
    pc: Dict[str, torch.Tensor],
    world_view_transform: torch.Tensor,
    intrinsics: torch.Tensor,
    bg_color: torch.Tensor,
    cfg: SplatterConfig,
) -> Dict[str, torch.Tensor]:
    """Render RGB, alpha, and alpha-normalized expected depth in one batched call."""
    return _render(pc, world_view_transform, intrinsics, bg_color, cfg, with_expected_depth=True)


def render_rgb(
    pc: Dict[str, torch.Tensor],
    world_view_transform: torch.Tensor,
    intrinsics: torch.Tensor,
    bg_color: torch.Tensor,
    cfg: SplatterConfig,
) -> Dict[str, torch.Tensor]:
    """Render RGB and alpha for deterministic visualization utilities."""
    return _render(pc, world_view_transform, intrinsics, bg_color, cfg, with_expected_depth=False)
