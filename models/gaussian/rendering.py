from __future__ import annotations

from typing import Dict

import torch
import torch.nn.functional as F

from .parameterization import SplatterConfig


HARD_DEPTH_VARIANT = 0
SOFT_DEPTH_VARIANT = 1


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
) -> Dict[str, torch.Tensor]:
    if pc["xyz"].device.type != "cuda":
        raise RuntimeError("gsplat rasterization requires CUDA tensors.")
    spherical_harmonics, rasterization = _load_gsplat_ops()
    # Gaussian conversion and all renderer inputs stay FP32 even when the neural
    # prediction path runs under BF16 autocast.
    means = torch.nan_to_num(pc["xyz"].float(), nan=0.0, posinf=0.0, neginf=0.0).clamp(-1.0e3, 1.0e3)
    w2c = world_view_transform.float()
    camera_k = intrinsics.float()
    minimum = min(float(value) for value in cfg.model.scale_min)
    maximum = max(float(value) for value in cfg.model.scale_max)
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
        render_mode="RGB",
        sparse_grad=False,
        absgrad=False,
    )
    image = rendered.movedim(-1, -3).contiguous()
    radii = meta.get("radii")
    return {
        "render": image,
        "alpha": alpha.movedim(-1, -3).contiguous(),
        "viewspace_points": None,
        "visibility_filter": (radii > 0) if torch.is_tensor(radii) else None,
        "radii": radii,
    }


def render_rgb(
    pc: Dict[str, torch.Tensor],
    world_view_transform: torch.Tensor,
    intrinsics: torch.Tensor,
    bg_color: torch.Tensor,
    cfg: SplatterConfig,
) -> Dict[str, torch.Tensor]:
    """Render RGB and alpha for deterministic visualization utilities."""
    return _render(pc, world_view_transform, intrinsics, bg_color, cfg)


def _dngaussian_depth_inputs(
    pc: Dict[str, torch.Tensor],
    hard_opacity: float,
) -> Dict[str, torch.Tensor]:
    """Build batched hard/soft inputs with DNGaussian's gradient isolation."""
    required = ("xyz", "scaling", "rotation", "opacity", "valid_mask")
    missing = [key for key in required if key not in pc]
    if missing:
        raise KeyError(f"DNGaussian depth rendering requires Gaussian fields {missing}.")
    if not 0.0 < float(hard_opacity) <= 1.0:
        raise ValueError("DNGaussian hard opacity must be in (0, 1].")

    xyz = pc["xyz"].float()
    if xyz.dim() < 3 or xyz.shape[-1] != 3:
        raise ValueError(f"Expected Gaussian centers as (...,N,3), got {tuple(xyz.shape)}.")
    batch_dims = xyz.shape[:-2]
    gaussian_count = xyz.shape[-2]
    expected_shapes = {
        "scaling": (*batch_dims, gaussian_count, 3),
        "rotation": (*batch_dims, gaussian_count, 4),
        "opacity": (*batch_dims, gaussian_count, 1),
        "valid_mask": (*batch_dims, gaussian_count),
    }
    for name, expected in expected_shapes.items():
        if tuple(pc[name].shape) != expected:
            raise ValueError(
                f"Expected {name} shape {expected}, got {tuple(pc[name].shape)}."
            )

    # Insert the hard/soft variant directly after the Gaussian batch dimensions.
    # gsplat treats it as another batch axis, so every camera is still rasterized
    # together in one call.
    variant_dim = len(batch_dims)
    live_means = torch.nan_to_num(
        xyz, nan=0.0, posinf=0.0, neginf=0.0
    ).clamp(-1.0e3, 1.0e3)
    detached_scales = torch.nan_to_num(
        pc["scaling"].float().detach(), nan=1.0e-8, posinf=1.0e3, neginf=1.0e-8
    ).clamp(1.0e-8, 1.0e3)
    detached_quaternions = F.normalize(
        torch.nan_to_num(
            pc["rotation"].float().detach(),
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        ),
        dim=-1,
        eps=1.0e-6,
    )
    valid = pc["valid_mask"].detach().to(device=xyz.device, dtype=torch.bool)
    soft_opacities = torch.nan_to_num(
        pc["opacity"].float().squeeze(-1),
        nan=0.0,
        posinf=1.0,
        neginf=0.0,
    ).clamp(0.0, 1.0)
    soft_opacities = soft_opacities * valid.to(soft_opacities.dtype)
    hard_opacities = (
        torch.ones_like(soft_opacities.detach())
        * float(hard_opacity)
        * valid.to(soft_opacities.dtype)
    )

    return {
        # Hard: only centers are live. Soft: only opacities are live.
        "means": torch.stack(
            (live_means, live_means.detach()), dim=variant_dim
        ).contiguous(),
        "quats": torch.stack(
            (detached_quaternions, detached_quaternions), dim=variant_dim
        ).contiguous(),
        "scales": torch.stack(
            (detached_scales, detached_scales), dim=variant_dim
        ).contiguous(),
        "opacities": torch.stack(
            (hard_opacities, soft_opacities), dim=variant_dim
        ).contiguous(),
    }


def render_dngaussian_depths(
    pc: Dict[str, torch.Tensor],
    world_view_transform: torch.Tensor,
    intrinsics: torch.Tensor,
    cfg: SplatterConfig,
    *,
    hard_opacity: float = 0.95,
) -> Dict[str, torch.Tensor]:
    """Render DNGaussian hard and soft accumulated depths in one batched call.

    The hard variant fixes every valid Gaussian's opacity to hard_opacity and
    sends gradients only to its center. The soft variant uses the predicted
    opacity and detaches center, scale, and rotation. D intentionally matches
    DNGaussian's accumulated-depth equations; ED would divide out coverage
    and weaken the opacity-only regularizer.
    """
    if pc["xyz"].device.type != "cuda":
        raise RuntimeError("gsplat rasterization requires CUDA tensors.")
    _spherical_harmonics, rasterization = _load_gsplat_ops()
    inputs = _dngaussian_depth_inputs(pc, hard_opacity)
    variant_dim = pc["xyz"].dim() - 2
    w2c = world_view_transform.float().detach().unsqueeze(variant_dim)
    camera_k = intrinsics.float().detach().unsqueeze(variant_dim)
    expanded_camera_shape = (
        *world_view_transform.shape[:variant_dim],
        2,
        *world_view_transform.shape[variant_dim:],
    )
    w2c = w2c.expand(expanded_camera_shape).contiguous()
    expanded_intrinsics_shape = (
        *intrinsics.shape[:variant_dim],
        2,
        *intrinsics.shape[variant_dim:],
    )
    camera_k = camera_k.expand(expanded_intrinsics_shape).contiguous()

    rendered_depth, rendered_alpha, _meta = rasterization(
        means=inputs["means"],
        quats=inputs["quats"],
        scales=inputs["scales"],
        opacities=inputs["opacities"],
        colors=None,
        viewmats=w2c,
        Ks=camera_k,
        backgrounds=None,
        width=int(cfg.data.img_width),
        height=int(cfg.data.img_height),
        near_plane=float(cfg.data.znear),
        far_plane=float(cfg.data.zfar),
        packed=False,
        segmented=False,
        sh_degree=None,
        render_mode="D",
        sparse_grad=False,
        absgrad=False,
    )
    del _meta
    if rendered_depth.shape[-1] != 1:
        raise RuntimeError("D rasterization must return exactly one depth channel.")
    depth = rendered_depth.movedim(-1, -3).contiguous()
    alpha = rendered_alpha.movedim(-1, -3).contiguous()
    return {
        "hard_depth": depth.select(variant_dim, HARD_DEPTH_VARIANT),
        "soft_depth": depth.select(variant_dim, SOFT_DEPTH_VARIANT),
        "hard_alpha": alpha.select(variant_dim, HARD_DEPTH_VARIANT),
        "soft_alpha": alpha.select(variant_dim, SOFT_DEPTH_VARIANT),
    }
