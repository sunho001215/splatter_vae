from __future__ import annotations

from typing import Dict

import torch
import torch.nn.functional as F

from .geometry import project_gaussian_centers
from .parameterization import SplatterConfig


def _load_rasterization():
    try:
        from gsplat.rendering import rasterization
    except Exception as exc:
        raise RuntimeError(
            "gsplat could not be loaded. Rebuild its CUDA extension against the installed "
            "PyTorch before rendering optical flow."
        ) from exc
    return rasterization


def activate_motion_map(raw_motion_map: torch.Tensor, max_translation: float) -> torch.Tensor:
    """Activate the six raw translation channels in FP32."""
    if raw_motion_map.dim() != 4 or raw_motion_map.shape[1] % 6 != 0:
        raise ValueError(f"Expected raw motion map as (B,6*G,H,W), got {tuple(raw_motion_map.shape)}.")
    if float(max_translation) <= 0.0:
        raise ValueError("max_translation must be positive.")
    return torch.tanh(raw_motion_map.float()) * float(max_translation)


def translate_gaussians(
    pc: Dict[str, torch.Tensor], xyz_delta: torch.Tensor
) -> Dict[str, torch.Tensor]:
    """Translate centers while retaining all non-positional Gaussian attributes."""
    if xyz_delta.shape != pc["xyz"].shape:
        raise ValueError("Dense translation residuals must align with Gaussian centers.")
    output = dict(pc)
    output["xyz"] = (pc["xyz"] + xyz_delta).contiguous()
    return output

def _valid_flow_features(flow: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    valid = valid.detach() & torch.isfinite(flow).all(dim=-1)
    valid_float = valid.unsqueeze(-1).to(dtype=flow.dtype)
    safe_flow = torch.nan_to_num(flow, nan=0.0, posinf=0.0, neginf=0.0)
    return torch.cat((safe_flow * valid_float, valid_float), dim=-1)


def _normalize_flow_signal(
    signal: torch.Tensor,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if signal.shape[-1] != 3:
        raise ValueError(f"Expected a three-channel flow signal, got {tuple(signal.shape)}.")
    numerator = signal[..., :2]
    coverage = torch.nan_to_num(
        signal[..., 2:3], nan=0.0, posinf=0.0, neginf=0.0
    ).clamp_min(0.0)
    finite = torch.isfinite(numerator).all(dim=-1, keepdim=True)
    normalized = torch.nan_to_num(
        numerator / coverage.clamp_min(float(eps)),
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )
    valid = (coverage > float(eps)) & finite & torch.isfinite(normalized).all(
        dim=-1, keepdim=True
    )
    normalized = torch.where(valid, normalized, torch.zeros_like(normalized))
    return (
        normalized.movedim(-1, -3).contiguous(),
        coverage.detach().movedim(-1, -3).contiguous(),
        valid.detach().movedim(-1, -3).contiguous(),
    )


def render_translation_flow_sequence(
    pc0: Dict[str, torch.Tensor],
    world_view_transform: torch.Tensor,
    intrinsics: torch.Tensor,
    cfg: SplatterConfig,
    eps: float = 1.0e-6,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Render translation-only flows 01, 12, and 02 in one gsplat call.

    Source geometry, opacity, visibility, and projection endpoints are detached.
    Gradients pass only through the target endpoints constructed from the dense
    translation residuals: 01 to delta 01, 12 to delta 12, and 02 to both.
    """
    required = (
        "xyz",
        "scaling",
        "rotation",
        "opacity",
        "valid_mask",
        "delta_xyz_01",
        "delta_xyz_12",
    )
    missing = [key for key in required if key not in pc0]
    if missing:
        raise KeyError(f"Flow rendering requires Gaussian fields {missing}.")

    xyz0 = pc0["xyz"]
    if xyz0.dim() != 3 or xyz0.shape[-1] != 3:
        raise ValueError(f"Expected base centers as (B,N,3), got {tuple(xyz0.shape)}.")
    if pc0["delta_xyz_01"].shape != xyz0.shape or pc0["delta_xyz_12"].shape != xyz0.shape:
        raise ValueError("Dense translation residuals must align with base Gaussian centers.")
    if world_view_transform.dim() != 5 or world_view_transform.shape[1] != 3:
        raise ValueError(
            f"Expected world-to-camera matrices as (B,3,A,4,4), got {tuple(world_view_transform.shape)}."
        )
    if intrinsics.shape != (*world_view_transform.shape[:3], 3, 3):
        raise ValueError("Intrinsics must match the three-timestep camera batch.")
    if xyz0.device.type != "cuda":
        raise RuntimeError("Optical-flow rasterization requires CUDA tensors.")

    device = xyz0.device
    dtype = xyz0.dtype
    w2c = world_view_transform.to(device=device, dtype=dtype).detach()
    camera_k = intrinsics.to(device=device, dtype=dtype).detach()

    delta01 = pc0["delta_xyz_01"]
    delta12 = pc0["delta_xyz_12"]
    xyz0_source = xyz0.detach()
    xyz1_live = xyz0_source + delta01
    xyz1_source = xyz1_live.detach()
    xyz2_from_1 = xyz1_source + delta12
    xyz2_accumulated = xyz0_source + delta01 + delta12

    # Project all distinct endpoints together; timestep-1 source projection is
    # the detached version of the already computed timestep-1 target endpoint.
    projection_xyz = torch.stack(
        (xyz0_source, xyz1_live, xyz2_from_1, xyz2_accumulated), dim=1
    )
    projection_w2c = torch.stack(
        (w2c[:, 0], w2c[:, 1], w2c[:, 2], w2c[:, 2]), dim=1
    )
    projection_k = torch.stack(
        (camera_k[:, 0], camera_k[:, 1], camera_k[:, 2], camera_k[:, 2]), dim=1
    )
    projected, projected_depth, projected_finite = project_gaussian_centers(
        projection_xyz, projection_w2c, projection_k
    )
    projection_valid = (
        projected_finite
        & (projected_depth > float(cfg.data.znear))
        & (projected_depth < float(cfg.data.zfar))
    )
    projected0 = projected[:, 0].detach()
    projected1 = projected[:, 1]
    projected1_source = projected1.detach()
    projected2_from_1 = projected[:, 2]
    projected2_accumulated = projected[:, 3]

    valid0 = projection_valid[:, 0].detach()
    valid1 = projection_valid[:, 1].detach()
    valid2_from_1 = projection_valid[:, 2].detach()
    valid2_accumulated = projection_valid[:, 3].detach()
    base_valid = pc0["valid_mask"].detach().to(device=device, dtype=torch.bool)[:, None, :]

    valid01 = valid0 & valid1 & base_valid
    valid12 = valid1 & valid2_from_1 & base_valid
    valid02 = valid0 & valid2_accumulated & base_valid
    features01 = _valid_flow_features(projected1 - projected0, valid01)
    features12 = _valid_flow_features(projected2_from_1 - projected1_source, valid12)
    features02 = _valid_flow_features(projected2_accumulated - projected0, valid02)

    flow_features = features01.new_zeros(
        features01.shape[0], 2, *features01.shape[1:-1], 6
    )
    flow_features[:, 0, ..., 0:3] = features01
    flow_features[:, 0, ..., 3:6] = features02
    flow_features[:, 1, ..., 0:3] = features12

    scale_min = max(float(cfg.model.gaussian_scale_min), 1.0e-8)
    scale_max = max(float(cfg.model.gaussian_scale_max), scale_min)
    base_scales = torch.nan_to_num(
        pc0["scaling"].detach(),
        nan=scale_min,
        posinf=scale_max,
        neginf=scale_min,
    ).clamp(scale_min, scale_max)
    base_quaternions = F.normalize(
        torch.nan_to_num(
            pc0["rotation"].detach(), nan=0.0, posinf=0.0, neginf=0.0
        ),
        dim=-1,
        eps=1.0e-6,
    )
    base_opacities = torch.nan_to_num(
        pc0["opacity"].detach().squeeze(-1),
        nan=0.0,
        posinf=1.0,
        neginf=0.0,
    ).clamp(0.0, 1.0)
    base_opacities = base_opacities * pc0["valid_mask"].detach().to(dtype=dtype)

    source_means = torch.nan_to_num(
        torch.stack((xyz0_source, xyz1_source), dim=1),
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    ).clamp(-1.0e3, 1.0e3)
    source_scales = base_scales[:, None].expand(-1, 2, -1, -1).contiguous()
    source_quaternions = base_quaternions[:, None].expand(-1, 2, -1, -1).contiguous()
    source_opacities = base_opacities[:, None].expand(-1, 2, -1).contiguous()
    source_w2c = torch.stack((w2c[:, 0], w2c[:, 1]), dim=1).contiguous()
    source_k = torch.stack((camera_k[:, 0], camera_k[:, 1]), dim=1).contiguous()

    camera_shape = source_w2c.shape[:-2]
    backgrounds = torch.zeros(*camera_shape, 6, device=device, dtype=dtype)
    rasterization = _load_rasterization()
    rendered_features, rendered_alpha, _meta = rasterization(
        means=source_means,
        quats=source_quaternions,
        scales=source_scales,
        opacities=source_opacities,
        colors=flow_features,
        viewmats=source_w2c,
        Ks=source_k,
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
    del _meta

    flow01, coverage01, valid_pixels01 = _normalize_flow_signal(
        rendered_features[:, 0, ..., 0:3], eps
    )
    flow12, coverage12, valid_pixels12 = _normalize_flow_signal(
        rendered_features[:, 1, ..., 0:3], eps
    )
    flow02, coverage02, valid_pixels02 = _normalize_flow_signal(
        rendered_features[:, 0, ..., 3:6], eps
    )
    flows = torch.stack((flow01, flow12, flow02), dim=1)
    coverages = torch.stack((coverage01, coverage12, coverage02), dim=1)
    valid_masks = torch.stack((valid_pixels01, valid_pixels12, valid_pixels02), dim=1)

    source_alpha = rendered_alpha.detach().movedim(-1, -3).contiguous()
    pair_alphas = torch.stack(
        (source_alpha[:, 0], source_alpha[:, 1], source_alpha[:, 0]), dim=1
    )
    return flows, coverages, valid_masks, pair_alphas
