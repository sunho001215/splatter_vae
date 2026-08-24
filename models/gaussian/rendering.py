from __future__ import annotations

import torch
import torch.nn.functional as F

from .parameterization import SplatterConfig


def _load_gsplat_ops():
    try:
        from gsplat import spherical_harmonics
        from gsplat.rendering import rasterization
    except Exception as exc:
        raise RuntimeError(
            "gsplat with CUDA support is required for Gaussian rasterization. "
            "Install/build it against the active PyTorch environment."
        ) from exc
    return spherical_harmonics, rasterization


def _camera_colors(
    pc: dict[str, torch.Tensor],
    means: torch.Tensor,
    w2c: torch.Tensor,
    sh_degree: int,
    spherical_harmonics,
) -> torch.Tensor:
    coefficients = torch.cat(
        (pc["features_dc"].float(), pc["features_rest"].float()), dim=-2
    )
    gaussian_batch_shape = means.shape[:-2]
    if coefficients.shape[:-3] != gaussian_batch_shape:
        raise ValueError(
            "SH coefficients and Gaussian means have different batch dimensions."
        )
    camera_count = w2c.shape[-3]
    rotation = w2c[..., :3, :3]
    translation = w2c[..., :3, 3]
    camera_positions = -torch.matmul(
        rotation.transpose(-1, -2), translation[..., None]
    ).squeeze(-1)
    directions = F.normalize(
        means.unsqueeze(-3) - camera_positions.unsqueeze(-2), dim=-1, eps=1.0e-8
    )
    camera_coefficients = coefficients.unsqueeze(-4).expand(
        *gaussian_batch_shape,
        camera_count,
        coefficients.shape[-3],
        coefficients.shape[-2],
        coefficients.shape[-1],
    )
    colors = spherical_harmonics(
        int(sh_degree),
        directions.reshape(-1, 3),
        camera_coefficients.reshape(-1, coefficients.shape[-2], 3),
    )
    return (
        colors.add(0.5)
        .clamp(0.0, 1.0)
        .view(*gaussian_batch_shape, camera_count, means.shape[-2], 3)
    )


def render_rgb_expected_depth(
    pc: dict[str, torch.Tensor],
    world_view_transform: torch.Tensor,
    intrinsics: torch.Tensor,
    background_color: torch.Tensor,
    cfg: SplatterConfig,
) -> dict[str, torch.Tensor]:
    """Rasterize RGB and expected metric depth together using gsplat ``RGB+ED``.

    Neural predictions may arrive in BF16, but all camera, projection, and
    rasterization geometry is explicitly converted to FP32.
    """
    required = (
        "xyz",
        "scaling",
        "rotation",
        "opacity",
        "features_dc",
        "features_rest",
        "valid_mask",
    )
    missing = [name for name in required if name not in pc]
    if missing:
        raise KeyError(f"Gaussian rendering is missing fields {missing}.")
    if pc["xyz"].device.type != "cuda":
        raise RuntimeError("gsplat rasterization requires CUDA tensors.")
    gaussian_batch = pc["xyz"].shape[:-2]
    if world_view_transform.shape[:-3] != gaussian_batch:
        raise ValueError(
            "Camera batch dimensions must match Gaussian batch dimensions."
        )
    if world_view_transform.shape[-2:] != (4, 4):
        raise ValueError("world_view_transform must end in 4x4 matrices.")
    if intrinsics.shape != (*world_view_transform.shape[:-2], 3, 3):
        raise ValueError("Intrinsics must align with every rendered camera.")

    spherical_harmonics, rasterization = _load_gsplat_ops()
    means = torch.nan_to_num(pc["xyz"].float(), nan=0.0, posinf=0.0, neginf=0.0).clamp(
        -1.0e3, 1.0e3
    )
    w2c = world_view_transform.float()
    camera_k = intrinsics.float()
    minimum = min(float(value) for value in cfg.model.scale_min)
    maximum = max(float(value) for value in cfg.model.scale_max)
    scales = torch.nan_to_num(
        pc["scaling"].float(), nan=minimum, posinf=maximum, neginf=minimum
    ).clamp(minimum, maximum)
    rotations = F.normalize(
        torch.nan_to_num(pc["rotation"].float(), nan=0.0, posinf=0.0, neginf=0.0),
        dim=-1,
        eps=1.0e-6,
    )
    opacities = torch.nan_to_num(
        pc["opacity"].float().squeeze(-1), nan=0.0, posinf=1.0, neginf=0.0
    ).clamp(0.0, 1.0)
    opacities = opacities * pc["valid_mask"].to(opacities.dtype)
    colors = _camera_colors(
        pc, means, w2c, int(cfg.model.max_sh_degree), spherical_harmonics
    )
    camera_shape = w2c.shape[:-2]
    if background_color.dim() == 1:
        backgrounds = (
            background_color.float()
            .view(*((1,) * len(camera_shape)), 3)
            .expand(*camera_shape, 3)
        )
    else:
        backgrounds = background_color.float().expand(*camera_shape, 3)
    rendered, alpha, metadata = rasterization(
        means=means,
        quats=rotations,
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
        render_mode="RGB+ED",
        sparse_grad=False,
        absgrad=False,
    )
    if rendered.shape[-1] != 4:
        raise RuntimeError(
            f"RGB+ED rasterization returned {rendered.shape[-1]} channels, expected four."
        )
    alpha_nchw = alpha.movedim(-1, -3).contiguous()
    expected_depth = rendered[..., 3:4].movedim(-1, -3).contiguous()
    expected_depth = torch.nan_to_num(expected_depth, nan=0.0, posinf=0.0, neginf=0.0)
    radii = metadata.get("radii")
    return {
        "rgb": rendered[..., :3].movedim(-1, -3).contiguous(),
        "expected_depth": expected_depth,
        "alpha": alpha_nchw,
        "depth_validity": (alpha_nchw > 1.0e-6) & torch.isfinite(expected_depth),
        "visibility_filter": (radii > 0) if torch.is_tensor(radii) else None,
        "radii": radii,
    }


def render_rgb(
    pc: dict[str, torch.Tensor],
    world_view_transform: torch.Tensor,
    intrinsics: torch.Tensor,
    background_color: torch.Tensor,
    cfg: SplatterConfig,
) -> dict[str, torch.Tensor]:
    """Visualization wrapper around the one-pass RGB plus expected-depth renderer."""
    output = render_rgb_expected_depth(
        pc, world_view_transform, intrinsics, background_color, cfg
    )
    return {**output, "render": output["rgb"]}
