from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.splatter import SplatterConfig, gaussian_params_per_gaussian


class DirectSplatterToGaussians(nn.Module):
    """Convert a decoder Gaussian map into a renderer-ready 3DGS dictionary.

    Each pixel predicts ``gaussians_per_pixel`` Gaussian records containing
    depth, camera-space local offset, scale, rotation, opacity, and SH color.
    The source camera is used only to lift centers into world space.
    """

    def __init__(self, cfg: SplatterConfig):
        super().__init__()
        self.cfg = cfg
        if int(cfg.model.max_sh_degree) != 1:
            raise ValueError("DirectSplatterToGaussians expects splatter.model.max_sh_degree=1.")

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
        depth_raw = params[..., cursor: cursor + 1]; cursor += 1
        offset_raw = params[..., cursor: cursor + 3]; cursor += 3
        scale_raw = params[..., cursor: cursor + 3]; cursor += 3
        rotation_raw = params[..., cursor: cursor + 4]; cursor += 4
        opacity_raw = params[..., cursor: cursor + 1]; cursor += 1
        features_dc_raw = params[..., cursor: cursor + 3]; cursor += 3
        features_rest_raw = params[..., cursor:]

        if activate_output:
            znear = float(self.cfg.data.znear)
            zfar = float(self.cfg.data.zfar)
            depth = torch.sigmoid(depth_raw) * (zfar - znear) + znear
            offset = torch.tanh(offset_raw) * float(self.cfg.model.gaussian_offset_scale)
            scale_min = max(float(self.cfg.model.gaussian_scale_min), 1.0e-8)
            scale_max = max(float(self.cfg.model.gaussian_scale_max), scale_min)
            scaling = torch.sigmoid(scale_raw) * (scale_max - scale_min) + scale_min
            identity = rotation_raw.new_tensor([1.0, 0.0, 0.0, 0.0]).view(1, 1, 4)
            rotation = F.normalize(rotation_raw + identity, dim=-1, eps=1.0e-6)
            opacity = torch.sigmoid(opacity_raw)
            features_dc = torch.sigmoid(features_dc_raw).unsqueeze(-2)
            sh_rest_bases = (int(self.cfg.model.max_sh_degree) + 1) ** 2 - 1
            features_rest = 0.1 * torch.tanh(features_rest_raw.view(batch, height * width * g, sh_rest_bases, 3))
        else:
            depth = depth_raw
            offset = offset_raw
            scaling = scale_raw
            rotation = rotation_raw
            opacity = opacity_raw
            features_dc = features_dc_raw.unsqueeze(-2)
            sh_rest_bases = (int(self.cfg.model.max_sh_degree) + 1) ** 2 - 1
            features_rest = features_rest_raw.view(batch, height * width * g, sh_rest_bases, 3)

        ys, xs = torch.meshgrid(
            torch.arange(height, device=device, dtype=dtype) + 0.5,
            torch.arange(width, device=device, dtype=dtype) + 0.5,
            indexing="ij",
        )
        xs = xs.reshape(1, height * width, 1).expand(batch, height * width, g).reshape(batch, height * width * g)
        ys = ys.reshape(1, height * width, 1).expand(batch, height * width, g).reshape(batch, height * width * g)

        k = intrinsics.to(device=device, dtype=dtype)
        fx = k[:, 0, 0].view(batch, 1).clamp_min(1.0e-6)
        fy = k[:, 1, 1].view(batch, 1).clamp_min(1.0e-6)
        cx = k[:, 0, 2].view(batch, 1)
        cy = k[:, 1, 2].view(batch, 1)
        z = depth.squeeze(-1)
        x_cam = (xs - cx) / fx * z
        y_cam = (ys - cy) / fy * z
        xyz_camera = torch.stack((x_cam, y_cam, z), dim=-1) + offset

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
