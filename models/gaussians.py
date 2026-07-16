from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.splatter import SplatterConfig, gaussian_params_per_gaussian


class DirectSplatterToGaussians(nn.Module):
    """Convert a decoder Gaussian map into a renderer-ready 3DGS dictionary."""

    def __init__(self, cfg: SplatterConfig):
        super().__init__()
        self.cfg = cfg

    @property
    def params_per_gaussian(self) -> int:
        return gaussian_params_per_gaussian(int(self.cfg.model.max_sh_degree))

    @property
    def expected_channels(self) -> int:
        return int(self.cfg.model.gaussians_per_pixel) * self.params_per_gaussian

    def forward(
        self,
        splatter_map: Optional[torch.Tensor] = None,
        source_cameras_view_to_world: Optional[torch.Tensor] = None,
        intrinsics: Optional[torch.Tensor] = None,
        activate_output: bool = True,
        proposal_map: Optional[torch.Tensor] = None,
        **_: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        if splatter_map is None:
            splatter_map = proposal_map
        if splatter_map is None:
            raise ValueError("Expected splatter_map or proposal_map.")
        if source_cameras_view_to_world is None or intrinsics is None:
            raise ValueError("source_cameras_view_to_world and intrinsics are required.")
        if splatter_map.dim() != 4:
            raise ValueError(f"Expected splatter map as (B,C,H,W), got {tuple(splatter_map.shape)}.")

        batch, channels, height, width = splatter_map.shape
        expected = self.expected_channels
        if channels != expected:
            raise ValueError(
                f"Decoder produced {channels} channels, but direct Gaussian conversion expects {expected} "
                f"({self.cfg.model.gaussians_per_pixel} gaussians/pixel x {self.params_per_gaussian} params)."
            )

        device = splatter_map.device
        dtype = splatter_map.dtype
        g = int(self.cfg.model.gaussians_per_pixel)
        p = self.params_per_gaussian
        params = splatter_map.view(batch, g, p, height, width)
        params = params.permute(0, 3, 4, 1, 2).reshape(batch, height * width * g, p).contiguous()

        cursor = 0
        depth_raw = params[..., cursor: cursor + 1]
        cursor += 1
        offset_raw = params[..., cursor: cursor + 3]
        cursor += 3
        scale_raw = params[..., cursor: cursor + 3]
        cursor += 3
        rotation_raw = params[..., cursor: cursor + 4]
        cursor += 4
        opacity_raw = params[..., cursor: cursor + 1]
        cursor += 1
        features_dc_raw = params[..., cursor: cursor + 3]
        cursor += 3
        features_rest_raw = params[..., cursor:]

        if activate_output:
            depth = self._activate_depth(depth_raw)

            depth_ordering = str(self.cfg.model.depth_ordering).lower()
            if depth_ordering in {"sort", "sorted", "ascending"} and g > 1:
                depth_grid = depth.view(batch, height * width, g, 1)
                order = torch.argsort(depth_grid.squeeze(-1), dim=2, stable=True)
                gather_param = order.view(batch, height * width, g, 1).expand(batch, height * width, g, p)
                params = params.view(batch, height * width, g, p).gather(dim=2, index=gather_param)
                params = params.reshape(batch, height * width * g, p).contiguous()
                depth = depth_grid.gather(dim=2, index=order.unsqueeze(-1)).reshape(batch, height * width * g, 1)
                cursor = 1
                offset_raw = params[..., cursor: cursor + 3]
                cursor += 3
                scale_raw = params[..., cursor: cursor + 3]
                cursor += 3
                rotation_raw = params[..., cursor: cursor + 4]
                cursor += 4
                opacity_raw = params[..., cursor: cursor + 1]
                cursor += 1
                features_dc_raw = params[..., cursor: cursor + 3]
                cursor += 3
                features_rest_raw = params[..., cursor:]
            elif depth_ordering not in {"none", "off", "false", "sort", "sorted", "ascending"}:
                raise ValueError(f"Unknown depth_ordering={self.cfg.model.depth_ordering!r}.")

            offset = torch.tanh(offset_raw * float(self.cfg.model.offset_scale) + float(self.cfg.model.offset_bias))
            offset = offset * float(self.cfg.model.gaussian_offset_scale)

            scale_min = max(float(self.cfg.model.gaussian_scale_min), 1.0e-8)
            scale_max = max(float(self.cfg.model.gaussian_scale_max), scale_min)
            scale_logits = scale_raw * float(self.cfg.model.scale_scale) + float(self.cfg.model.scale_bias)
            scaling = torch.sigmoid(scale_logits) * (scale_max - scale_min) + scale_min

            rotation_logits = rotation_raw * float(self.cfg.model.rotation_scale) + float(self.cfg.model.rotation_bias)
            identity = rotation_logits.new_tensor([1.0, 0.0, 0.0, 0.0]).view(1, 1, 4)
            rotation = F.normalize(rotation_logits + identity, dim=-1, eps=1.0e-6)

            opacity_logits = opacity_raw * float(self.cfg.model.opacity_scale) + float(self.cfg.model.opacity_bias)
            opacity = torch.sigmoid(opacity_logits)

            color_logits = features_dc_raw * float(self.cfg.model.color_scale) + float(self.cfg.model.color_bias)
            # With sh_degree enabled, gsplat interprets colors as spherical-harmonic
            # coefficients. Match Splatter Image by predicting the DC coefficient
            # directly instead of sigmoid-activating it as RGB.
            features_dc = color_logits.unsqueeze(-2)

            sh_rest_bases = (int(self.cfg.model.max_sh_degree) + 1) ** 2 - 1
            sh_rest_logits = features_rest_raw * float(self.cfg.model.sh_rest_raw_scale) + float(self.cfg.model.sh_rest_raw_bias)
            features_rest = float(self.cfg.model.sh_rest_scale) * torch.tanh(
                sh_rest_logits.view(batch, height * width * g, sh_rest_bases, 3)
            )
        else:
            depth = depth_raw
            offset = offset_raw
            scaling = scale_raw
            rotation = rotation_raw
            opacity = opacity_raw
            features_dc = features_dc_raw.unsqueeze(-2)
            sh_rest_bases = (int(self.cfg.model.max_sh_degree) + 1) ** 2 - 1
            features_rest = features_rest_raw.view(batch, height * width * g, sh_rest_bases, 3)

        xyz_camera = self._pixel_depth_to_camera_points(depth, offset, intrinsics, height, width, g, dtype, device)
        c2w = source_cameras_view_to_world.to(device=device, dtype=dtype)
        rot = c2w[:, :3, :3]
        trans = c2w[:, :3, 3]
        xyz_world = torch.einsum("bij,bnj->bni", rot, xyz_camera) + trans[:, None, :]

        valid_mask = torch.isfinite(xyz_world).all(dim=-1) & torch.isfinite(scaling).all(dim=-1)
        opacity = opacity * valid_mask.unsqueeze(-1).to(dtype)

        return {
            "xyz": xyz_world.contiguous(),
            "scaling": scaling.contiguous(),
            "rotation": rotation.contiguous(),
            "opacity": opacity.contiguous(),
            "features_dc": features_dc.contiguous(),
            "features_rest": features_rest.contiguous(),
            "valid_mask": valid_mask.contiguous(),
            "source_depth": depth.contiguous(),
        }

    def _activate_depth(self, depth_raw: torch.Tensor) -> torch.Tensor:
        depth_min = float(self.cfg.data.znear if self.cfg.model.depth_min is None else self.cfg.model.depth_min)
        depth_max = float(self.cfg.data.zfar if self.cfg.model.depth_max is None else self.cfg.model.depth_max)
        if depth_max <= depth_min:
            raise ValueError(f"depth_max must be greater than depth_min, got {depth_min} and {depth_max}.")

        depth_unit = torch.sigmoid(depth_raw * float(self.cfg.model.depth_scale) + float(self.cfg.model.depth_bias))
        activation = str(self.cfg.model.depth_activation).lower()
        if activation in {"inverse_depth", "inv_depth", "disparity"}:
            min_inv = 1.0 / depth_max
            max_inv = 1.0 / depth_min
            inv_depth = min_inv + (max_inv - min_inv) * depth_unit
            return 1.0 / inv_depth.clamp_min(1.0e-8)
        if activation in {"linear", "metric", "sigmoid"}:
            return depth_unit * (depth_max - depth_min) + depth_min
        raise ValueError(f"Unknown depth_activation={self.cfg.model.depth_activation!r}.")

    def _pixel_depth_to_camera_points(
        self,
        depth: torch.Tensor,
        offset: torch.Tensor,
        intrinsics: torch.Tensor,
        height: int,
        width: int,
        gaussians_per_pixel: int,
        dtype: torch.dtype,
        device: torch.device,
    ) -> torch.Tensor:
        ys, xs = torch.meshgrid(
            torch.arange(height, device=device, dtype=dtype) + 0.5,
            torch.arange(width, device=device, dtype=dtype) + 0.5,
            indexing="ij",
        )
        xs = xs.reshape(1, height * width, 1).expand(depth.shape[0], height * width, gaussians_per_pixel)
        ys = ys.reshape(1, height * width, 1).expand(depth.shape[0], height * width, gaussians_per_pixel)
        xs = xs.reshape(depth.shape[0], height * width * gaussians_per_pixel)
        ys = ys.reshape(depth.shape[0], height * width * gaussians_per_pixel)

        k = intrinsics.to(device=device, dtype=dtype)
        fx = k[:, 0, 0].view(depth.shape[0], 1).clamp_min(1.0e-6)
        fy = k[:, 1, 1].view(depth.shape[0], 1).clamp_min(1.0e-6)
        cx = k[:, 0, 2].view(depth.shape[0], 1)
        cy = k[:, 1, 2].view(depth.shape[0], 1)
        z = depth.squeeze(-1)
        x_cam = (xs - cx) / fx * z
        y_cam = (ys - cy) / fy * z
        return torch.stack((x_cam, y_cam, z), dim=-1) + offset
