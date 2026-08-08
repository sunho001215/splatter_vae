from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class SplatterDataConfig:
    """Camera and image configuration shared by Gaussian conversion/rendering."""

    img_height: int = 128
    img_width: int = 128
    znear: float = 0.1
    zfar: float = 2.0
    white_background: bool = False


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
    sh_bases = (int(max_sh_degree) + 1) ** 2
    return 1 + 3 + 3 + 4 + 1 + 3 + max(0, sh_bases - 1) * 3


def default_splatter_channels(gaussians_per_pixel: int = 1, max_sh_degree: int = 1) -> int:
    return int(gaussians_per_pixel) * gaussian_params_per_gaussian(max_sh_degree)


class DirectSplatterToGaussians(nn.Module):
    """Activate FP32 decoder maps and construct renderer-ready world Gaussians."""

    def __init__(self, cfg: SplatterConfig):
        super().__init__()
        self.cfg = cfg
        height = int(cfg.data.img_height)
        width = int(cfg.data.img_width)
        g = int(cfg.model.gaussians_per_pixel)
        ys, xs = torch.meshgrid(
            torch.arange(height, dtype=torch.float32) + 0.5,
            torch.arange(width, dtype=torch.float32) + 0.5,
            indexing="ij",
        )
        self.register_buffer("pixel_center_x", xs.flatten().repeat_interleave(g)[None], persistent=False)
        self.register_buffer("pixel_center_y", ys.flatten().repeat_interleave(g)[None], persistent=False)

    @property
    def params_per_gaussian(self) -> int:
        return gaussian_params_per_gaussian(int(self.cfg.model.max_sh_degree))

    @property
    def expected_channels(self) -> int:
        return int(self.cfg.model.gaussians_per_pixel) * self.params_per_gaussian

    def forward(
        self,
        *,
        splatter_map: torch.Tensor,
        motion_map: Optional[torch.Tensor],
        source_cameras_view_to_world: torch.Tensor,
        intrinsics: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        # Activation, back-projection, and camera transforms are deliberately FP32.
        splatter_map = splatter_map.float()
        if motion_map is not None:
            motion_map = motion_map.float()
        c2w = source_cameras_view_to_world.float()
        intrinsics = intrinsics.float()
        if splatter_map.dim() != 4:
            raise ValueError(f"Expected splatter map as (B,C,H,W), got {tuple(splatter_map.shape)}.")
        batch, channels, height, width = splatter_map.shape
        if channels != self.expected_channels:
            raise ValueError(
                f"Decoder produced {channels} channels; expected {self.expected_channels}."
            )
        g = int(self.cfg.model.gaussians_per_pixel)
        expected_motion = (batch, g * 6, height, width)
        if motion_map is not None and tuple(motion_map.shape) != expected_motion:
            raise ValueError(f"Expected activated motion map {expected_motion}, got {tuple(motion_map.shape)}.")

        p = self.params_per_gaussian
        params = splatter_map.view(batch, g, p, height, width)
        params = params.permute(0, 3, 4, 1, 2).reshape(batch, height * width * g, p).contiguous()
        motion = None
        if motion_map is not None:
            motion = motion_map.view(batch, g, 6, height, width)
            motion = motion.permute(0, 3, 4, 1, 2).reshape(
                batch, height * width * g, 6
            ).contiguous()

        depth_raw = params[..., 0:1]
        depth = self._activate_depth(depth_raw)
        ordering = str(self.cfg.model.depth_ordering).lower()
        if ordering in {"sort", "sorted", "ascending"} and g > 1:
            depth_grid = depth.view(batch, height * width, g, 1)
            order = torch.argsort(depth_grid.squeeze(-1), dim=2, stable=True)
            params = params.view(batch, height * width, g, p).gather(
                2, order[..., None].expand(batch, height * width, g, p)
            ).reshape(batch, height * width * g, p).contiguous()
            if motion is not None:
                motion = motion.view(batch, height * width, g, 6).gather(
                    2, order[..., None].expand(batch, height * width, g, 6)
                ).reshape(batch, height * width * g, 6).contiguous()
            depth = depth_grid.gather(2, order[..., None]).reshape(batch, height * width * g, 1)
        elif ordering not in {"none", "off", "false", "sort", "sorted", "ascending"}:
            raise ValueError(f"Unknown depth_ordering={self.cfg.model.depth_ordering!r}.")

        cursor = 1
        offset_raw = params[..., cursor:cursor + 3]; cursor += 3
        scale_raw = params[..., cursor:cursor + 3]; cursor += 3
        rotation_raw = params[..., cursor:cursor + 4]; cursor += 4
        opacity_raw = params[..., cursor:cursor + 1]; cursor += 1
        dc_raw = params[..., cursor:cursor + 3]; cursor += 3
        rest_raw = params[..., cursor:]

        offset = torch.tanh(offset_raw * self.cfg.model.offset_scale + self.cfg.model.offset_bias)
        offset = offset * self.cfg.model.gaussian_offset_scale
        scale_min = max(float(self.cfg.model.gaussian_scale_min), 1.0e-8)
        scale_max = max(float(self.cfg.model.gaussian_scale_max), scale_min)
        scaling = torch.sigmoid(scale_raw * self.cfg.model.scale_scale + self.cfg.model.scale_bias)
        scaling = scaling * (scale_max - scale_min) + scale_min
        rotation_logits = rotation_raw * self.cfg.model.rotation_scale + self.cfg.model.rotation_bias
        identity = rotation_logits.new_tensor([1.0, 0.0, 0.0, 0.0]).view(1, 1, 4)
        rotation = F.normalize(rotation_logits + identity, dim=-1, eps=1.0e-6)
        opacity = torch.sigmoid(opacity_raw * self.cfg.model.opacity_scale + self.cfg.model.opacity_bias)
        features_dc = (dc_raw * self.cfg.model.color_scale + self.cfg.model.color_bias).unsqueeze(-2)
        rest_bases = (int(self.cfg.model.max_sh_degree) + 1) ** 2 - 1
        rest_logits = rest_raw * self.cfg.model.sh_rest_raw_scale + self.cfg.model.sh_rest_raw_bias
        features_rest = self.cfg.model.sh_rest_scale * torch.tanh(
            rest_logits.view(batch, height * width * g, rest_bases, 3)
        )

        xyz_camera = self._pixel_depth_to_camera_points(depth, offset, intrinsics, height, width, g)
        xyz_world = torch.einsum("bij,bnj->bni", c2w[:, :3, :3], xyz_camera) + c2w[:, None, :3, 3]
        valid = torch.isfinite(xyz_world).all(-1) & torch.isfinite(scaling).all(-1)
        opacity = opacity * valid[..., None].to(opacity.dtype)
        output = {
            "xyz": xyz_world.contiguous(),
            "scaling": scaling.contiguous(),
            "rotation": rotation.contiguous(),
            "opacity": opacity.contiguous(),
            "features_dc": features_dc.contiguous(),
            "features_rest": features_rest.contiguous(),
            "valid_mask": valid.contiguous(),
            "source_depth": depth.contiguous(),
        }
        if motion is not None:
            output["delta_xyz_01"] = motion[..., 0:3].contiguous()
            output["delta_xyz_12"] = motion[..., 3:6].contiguous()
        return output

    def _activate_depth(self, raw: torch.Tensor) -> torch.Tensor:
        minimum = float(self.cfg.data.znear if self.cfg.model.depth_min is None else self.cfg.model.depth_min)
        maximum = float(self.cfg.data.zfar if self.cfg.model.depth_max is None else self.cfg.model.depth_max)
        if maximum <= minimum:
            raise ValueError("depth_max must be greater than depth_min.")
        unit = torch.sigmoid(raw * self.cfg.model.depth_scale + self.cfg.model.depth_bias)
        mode = str(self.cfg.model.depth_activation).lower()
        if mode in {"inverse_depth", "inv_depth", "disparity"}:
            inverse = 1.0 / maximum + (1.0 / minimum - 1.0 / maximum) * unit
            return inverse.clamp_min(1.0e-8).reciprocal()
        if mode in {"linear", "metric", "sigmoid"}:
            return minimum + (maximum - minimum) * unit
        raise ValueError(f"Unknown depth_activation={self.cfg.model.depth_activation!r}.")

    def _pixel_depth_to_camera_points(
        self, depth: torch.Tensor, offset: torch.Tensor, intrinsics: torch.Tensor,
        height: int, width: int, gaussians_per_pixel: int,
    ) -> torch.Tensor:
        points = height * width * gaussians_per_pixel
        if self.pixel_center_x.shape[-1] != points:
            raise ValueError("Cached pixel grid does not match the decoder output grid.")
        xs = self.pixel_center_x.expand(depth.shape[0], -1)
        ys = self.pixel_center_y.expand(depth.shape[0], -1)
        fx = intrinsics[:, 0, 0, None].clamp_min(1.0e-6)
        fy = intrinsics[:, 1, 1, None].clamp_min(1.0e-6)
        cx = intrinsics[:, 0, 2, None]
        cy = intrinsics[:, 1, 2, None]
        z = depth.squeeze(-1)
        return torch.stack(((xs - cx) / fx * z, (ys - cy) / fy * z, z), -1) + offset
