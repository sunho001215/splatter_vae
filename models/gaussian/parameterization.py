from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


ACTIVE_GAUSSIAN_OPACITY_THRESHOLD = 0.05
NUM_GAUSSIANS = 4096


@dataclass
class SplatterDataConfig:
    """Camera and image configuration shared by Gaussian rendering."""

    img_height: int = 128
    img_width: int = 128
    znear: float = 0.1
    zfar: float = 2.0
    white_background: bool = False


@dataclass
class SplatterModelConfig:
    """Anchor-free world-space Gaussian activation settings.

    The ROI was selected from metric foreground depth back-projected through all
    six cameras over all eight task datasets. The combined robust 0.1--99.9
    percentile range was x=[-0.455,0.501], y=[0.044,1.000], z=[-0.001,0.698].
    These configured limits retain the existing workspace safety margin while
    excluding camera locations. Every Gaussian can use this complete ROI.
    """

    max_sh_degree: int = 1
    world_bounds_min: Sequence[float] = (-0.65, -0.05, -0.05)
    world_bounds_max: Sequence[float] = (0.70, 1.10, 0.80)
    scale_min: Sequence[float] = (0.0005, 0.0005, 0.0005)
    scale_max: Sequence[float] = (0.05, 0.05, 0.05)
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
    """Raw attributes per Gaussian: absolute XYZ, scale, quaternion, opacity, and SH."""

    sh_bases = (int(max_sh_degree) + 1) ** 2
    return 3 + 3 + 4 + 1 + 3 + max(0, sh_bases - 1) * 3


def _xyz_vector(
    values: Sequence[float],
    *,
    name: str,
    device: torch.device,
) -> torch.Tensor:
    tensor = torch.as_tensor(tuple(values), dtype=torch.float32, device=device)
    if tensor.shape != (3,):
        raise ValueError(f"{name} must contain exactly three values, got {tuple(tensor.shape)}.")
    if not torch.isfinite(tensor).all():
        raise ValueError(f"{name} must contain only finite values.")
    return tensor


class WorldSpaceGaussianParameterization(nn.Module):
    """Activate 4096 anchor-free predictions in a shared world-space ROI."""

    def __init__(self, cfg: SplatterConfig):
        super().__init__()
        self.cfg = cfg
        world_min = _xyz_vector(
            cfg.model.world_bounds_min, name="world_bounds_min", device=torch.device("cpu")
        )
        world_max = _xyz_vector(
            cfg.model.world_bounds_max, name="world_bounds_max", device=torch.device("cpu")
        )
        scale_min = _xyz_vector(
            cfg.model.scale_min, name="scale_min", device=torch.device("cpu")
        )
        scale_max = _xyz_vector(
            cfg.model.scale_max, name="scale_max", device=torch.device("cpu")
        )
        if not torch.all(world_max > world_min):
            raise ValueError("Every world_bounds_max value must exceed world_bounds_min.")
        if not torch.all(scale_min > 0.0) or not torch.all(scale_max >= scale_min):
            raise ValueError("Scale bounds must satisfy 0 < scale_min <= scale_max per axis.")
        self.register_buffer("world_center", 0.5 * (world_min + world_max), persistent=False)
        self.register_buffer("world_half_extent", 0.5 * (world_max - world_min), persistent=False)
        self.register_buffer("scale_min", scale_min, persistent=False)
        self.register_buffer("scale_max", scale_max, persistent=False)

    @property
    def params_per_gaussian(self) -> int:
        return gaussian_params_per_gaussian(int(self.cfg.model.max_sh_degree))

    @property
    def num_gaussians(self) -> int:
        return NUM_GAUSSIANS

    def forward(
        self,
        *,
        gaussian_parameters: torch.Tensor,
        motion_parameters: Optional[torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        params = gaussian_parameters.float()
        if params.dim() != 3:
            raise ValueError(f"Expected Gaussian parameters as (B,N,P), got {tuple(params.shape)}.")
        batch, _count, _channels = params.shape
        expected = (batch, self.num_gaussians, self.params_per_gaussian)
        if tuple(params.shape) != expected:
            raise ValueError(f"Expected Gaussian parameters {expected}, got {tuple(params.shape)}.")
        if motion_parameters is not None:
            motion_parameters = motion_parameters.float()
            expected_motion = (batch, self.num_gaussians, 6)
            if tuple(motion_parameters.shape) != expected_motion:
                raise ValueError(
                    f"Expected activated motion {expected_motion}, got {tuple(motion_parameters.shape)}."
                )

        cursor = 0
        xyz_raw = params[..., cursor:cursor + 3]
        cursor += 3
        scale_raw = params[..., cursor:cursor + 3]
        cursor += 3
        rotation_raw = params[..., cursor:cursor + 4]
        cursor += 4
        opacity_raw = params[..., cursor:cursor + 1]
        cursor += 1
        dc_raw = params[..., cursor:cursor + 3]
        cursor += 3
        rest_raw = params[..., cursor:]

        xyz = (
            self.world_center.float().view(1, 1, 3)
            + self.world_half_extent.float().view(1, 1, 3) * torch.tanh(xyz_raw)
        )
        scale_unit = torch.sigmoid(
            scale_raw * float(self.cfg.model.scale_scale) + float(self.cfg.model.scale_bias)
        )
        scaling = (
            self.scale_min.float().view(1, 1, 3)
            + scale_unit
            * (self.scale_max.float() - self.scale_min.float()).view(1, 1, 3)
        )

        rotation_logits = (
            rotation_raw * float(self.cfg.model.rotation_scale)
            + float(self.cfg.model.rotation_bias)
        )
        identity = rotation_logits.new_tensor((1.0, 0.0, 0.0, 0.0)).view(1, 1, 4)
        rotation = F.normalize(rotation_logits + identity, dim=-1, eps=1.0e-6)
        opacity = torch.sigmoid(
            opacity_raw * float(self.cfg.model.opacity_scale)
            + float(self.cfg.model.opacity_bias)
        )
        features_dc = (
            dc_raw * float(self.cfg.model.color_scale) + float(self.cfg.model.color_bias)
        ).unsqueeze(-2)
        rest_bases = (int(self.cfg.model.max_sh_degree) + 1) ** 2 - 1
        rest_logits = (
            rest_raw * float(self.cfg.model.sh_rest_raw_scale)
            + float(self.cfg.model.sh_rest_raw_bias)
        )
        features_rest = float(self.cfg.model.sh_rest_scale) * torch.tanh(
            rest_logits.view(batch, self.num_gaussians, rest_bases, 3)
        )

        valid = (
            torch.isfinite(xyz).all(-1)
            & torch.isfinite(scaling).all(-1)
            & torch.isfinite(rotation).all(-1)
            & torch.isfinite(opacity).all(-1)
            & torch.isfinite(features_dc).all(dim=(-1, -2))
            & torch.isfinite(features_rest).all(dim=(-1, -2))
        )
        opacity = opacity * valid[..., None].to(opacity.dtype)
        output = {
            "xyz": xyz.contiguous(),
            "scaling": scaling.contiguous(),
            "rotation": rotation.contiguous(),
            "opacity": opacity.contiguous(),
            "features_dc": features_dc.contiguous(),
            "features_rest": features_rest.contiguous(),
            "valid_mask": valid.contiguous(),
        }
        if motion_parameters is not None:
            output["delta_xyz_01"] = motion_parameters[..., 0:3].contiguous()
            output["delta_xyz_12"] = motion_parameters[..., 3:6].contiguous()
        return output
