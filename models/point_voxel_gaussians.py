from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.splatter import SplatterConfig
from utils.ray_utils import build_ray_dirs_from_intrinsics


class SparseGridPointNeXtBlock(nn.Module):
    """A lightweight PointNeXt-style block over active voxel-grid neighbors."""

    def __init__(self, channels: int, neighbor_offsets: torch.Tensor):
        super().__init__()
        self.register_buffer("neighbor_offsets", neighbor_offsets.to(dtype=torch.long), persistent=False)
        self.mlp = nn.Sequential(
            nn.LayerNorm(2 * channels),
            nn.Linear(2 * channels, channels),
            nn.GELU(),
            nn.Linear(channels, channels),
        )

    def _linear_hash(self, shifted: torch.Tensor, dims: torch.Tensor) -> torch.Tensor:
        return shifted[:, 0] + shifted[:, 1] * dims[0] + shifted[:, 2] * dims[0] * dims[1]

    def _neighbor_mean_one(self, features: torch.Tensor, voxel_indices: torch.Tensor) -> torch.Tensor:
        num_voxels = int(features.shape[0])
        if num_voxels <= 1:
            return features

        mins = voxel_indices.amin(dim=0)
        shifted = voxel_indices - mins
        dims = shifted.amax(dim=0) + 3
        active_hash = self._linear_hash(shifted, dims)
        sorted_hash, order = torch.sort(active_hash)

        queries = voxel_indices[:, None, :] + self.neighbor_offsets.to(voxel_indices.device).view(1, -1, 3)
        shifted_q = queries - mins.view(1, 1, 3)
        inside = (shifted_q >= 0).all(dim=-1) & (shifted_q < dims.view(1, 1, 3)).all(dim=-1)
        query_hash = (
            shifted_q[..., 0]
            + shifted_q[..., 1] * dims[0]
            + shifted_q[..., 2] * dims[0] * dims[1]
        )
        query_hash = torch.where(inside, query_hash, torch.zeros_like(query_hash))

        pos = torch.searchsorted(sorted_hash, query_hash.reshape(-1))
        pos_safe = pos.clamp(max=max(num_voxels - 1, 0))
        matched = (pos < num_voxels) & (sorted_hash[pos_safe] == query_hash.reshape(-1)) & inside.reshape(-1)
        neighbor_rows = order[pos_safe]
        neighbor_feat = features[neighbor_rows].view(num_voxels, -1, features.shape[-1])
        weights = matched.to(features.dtype).view(num_voxels, -1, 1)
        denom = weights.sum(dim=1).clamp_min(1.0)
        return (neighbor_feat * weights).sum(dim=1) / denom

    def forward(self, features: torch.Tensor, voxel_indices: torch.Tensor, valid_mask: torch.Tensor) -> torch.Tensor:
        out = torch.zeros_like(features)
        for batch_idx in range(features.shape[0]):
            mask = valid_mask[batch_idx].to(dtype=torch.bool)
            count = int(mask.sum().item())
            if count == 0:
                continue
            active_features = features[batch_idx, :count]
            active_indices = voxel_indices[batch_idx, :count]
            neighbor_mean = self._neighbor_mean_one(active_features, active_indices)
            block_in = torch.cat([active_features, neighbor_mean - active_features], dim=-1)
            out[batch_idx, :count] = active_features + self.mlp(block_in)
        return out * valid_mask.unsqueeze(-1).to(out.dtype)


class SparseGridPointNeXt(nn.Module):
    """Small sparse-grid PointNeXt refinement stack for voxel features."""

    def __init__(self, channels: int, depth: int, neighbors: int):
        super().__init__()
        offsets = []
        for dz in (-1, 0, 1):
            for dy in (-1, 0, 1):
                for dx in (-1, 0, 1):
                    offsets.append((dx, dy, dz))
        offsets = sorted(offsets, key=lambda item: (item[0] * item[0] + item[1] * item[1] + item[2] * item[2], item))
        offsets_t = torch.tensor(offsets[: max(1, min(int(neighbors), len(offsets)))], dtype=torch.long)
        self.blocks = nn.ModuleList(
            SparseGridPointNeXtBlock(channels=channels, neighbor_offsets=offsets_t)
            for _ in range(max(1, int(depth)))
        )

    def forward(self, features: torch.Tensor, voxel_indices: torch.Tensor, valid_mask: torch.Tensor) -> torch.Tensor:
        x = features * valid_mask.unsqueeze(-1).to(features.dtype)
        for block in self.blocks:
            x = block(x, voxel_indices, valid_mask)
        return x


class PointVoxelToGaussians(nn.Module):
    """Convert decoder point proposals into renderable 3D Gaussian dictionaries.

    Pipeline:
      point proposal map -> raw world point cloud -> trilinear soft voxelization
      -> sparse-grid PointNeXt -> Gaussian head.
    """

    def __init__(self, cfg: SplatterConfig, z_inv_dim: int):
        super().__init__()
        self.cfg = cfg
        self.z_inv_dim = int(z_inv_dim)
        if int(cfg.model.max_sh_degree) != 1:
            raise ValueError("PointVoxelToGaussians expects splatter.model.max_sh_degree=1.")
        if str(cfg.model.voxelization_type).lower() != "trilinear_soft":
            raise ValueError("Only voxelization_type='trilinear_soft' is supported.")

        in_dim = self.z_inv_dim + 8
        channels = int(cfg.model.pointnet_channels)
        self.voxel_feature_projection = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, channels),
            nn.GELU(),
            nn.Linear(channels, channels),
        )
        self.pointnext = SparseGridPointNeXt(
            channels=channels,
            depth=int(cfg.model.pointnet_depth),
            neighbors=int(cfg.model.pointnet_neighbors),
        )

        self.gaussians_per_voxel = int(cfg.model.gaussians_per_voxel)
        sh_rest = 3 * (((int(cfg.model.max_sh_degree) + 1) ** 2) - 1)
        self.params_per_gaussian = 3 + 3 + 4 + 1 + 3 + sh_rest
        self.gaussian_head = nn.Sequential(
            nn.LayerNorm(channels),
            nn.Linear(channels, channels),
            nn.GELU(),
            nn.Linear(channels, self.gaussians_per_voxel * self.params_per_gaussian),
        )

        corner_offsets = torch.tensor(
            [
                [0, 0, 0],
                [1, 0, 0],
                [0, 1, 0],
                [1, 1, 0],
                [0, 0, 1],
                [1, 0, 1],
                [0, 1, 1],
                [1, 1, 1],
            ],
            dtype=torch.long,
        )
        self.register_buffer("corner_offsets", corner_offsets, persistent=False)

    @property
    def proposal_channels(self) -> int:
        return int(self.cfg.model.points_per_pixel) * 5

    def _pad_float(self, tensors: list[torch.Tensor], pad_value: float = 0.0) -> torch.Tensor:
        if not tensors:
            raise ValueError("Cannot pad an empty tensor list.")
        max_items = max(int(t.shape[0]) for t in tensors)
        padded = []
        for tensor in tensors:
            pad_len = max_items - int(tensor.shape[0])
            if pad_len > 0:
                pad = tensor.new_full((pad_len, *tensor.shape[1:]), float(pad_value))
                tensor = torch.cat([tensor, pad], dim=0)
            padded.append(tensor)
        return torch.stack(padded, dim=0)

    def _pad_bool(self, tensors: list[torch.Tensor]) -> torch.Tensor:
        max_items = max(int(t.shape[0]) for t in tensors)
        padded = []
        for tensor in tensors:
            pad_len = max_items - int(tensor.shape[0])
            if pad_len > 0:
                pad = torch.zeros((pad_len,), device=tensor.device, dtype=torch.bool)
                tensor = torch.cat([tensor, pad], dim=0)
            padded.append(tensor)
        return torch.stack(padded, dim=0)

    def _proposal_to_world_points(
        self,
        proposal_map: torch.Tensor,
        source_cameras_view_to_world: torch.Tensor,
        intrinsics: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        batch, channels, height, width = proposal_map.shape
        points_per_pixel = int(self.cfg.model.points_per_pixel)
        expected_channels = points_per_pixel * 5
        if channels != expected_channels:
            raise ValueError(f"Expected proposal map with {expected_channels} channels, got {channels}.")
        if (height, width) != (int(self.cfg.data.img_height), int(self.cfg.data.img_width)):
            raise ValueError(
                f"Expected proposal map size {(self.cfg.data.img_height, self.cfg.data.img_width)}, "
                f"got {(height, width)}."
            )

        proposal = proposal_map.view(batch, points_per_pixel, 5, height, width)
        proposal = proposal.permute(0, 3, 4, 1, 2).reshape(batch, height * width, points_per_pixel, 5)
        depth_logits = proposal[..., 0]
        offset_raw = proposal[..., 1:4]
        confidence_logits = proposal[..., 4:5]

        znear = float(self.cfg.data.znear)
        zfar = float(self.cfg.data.zfar)
        depth = torch.sigmoid(depth_logits) * (zfar - znear) + znear
        offset = torch.tanh(offset_raw) * float(self.cfg.model.point_offset_scale)
        confidence = torch.sigmoid(confidence_logits)

        ray_dirs = build_ray_dirs_from_intrinsics(
            H=height,
            W=width,
            fx=intrinsics[:, 0, 0],
            fy=intrinsics[:, 1, 1],
            cx=intrinsics[:, 0, 2],
            cy=intrinsics[:, 1, 2],
            device=proposal_map.device,
            inverted_x=bool(self.cfg.data.inverted_x),
            inverted_y=bool(self.cfg.data.inverted_y),
        )
        ray_dirs = ray_dirs.permute(0, 2, 3, 1).reshape(batch, height * width, 1, 3)
        xyz_camera = ray_dirs * depth.unsqueeze(-1) + offset

        rot_c2w = source_cameras_view_to_world[:, :3, :3]
        trans_c2w = source_cameras_view_to_world[:, :3, 3]
        xyz_world = torch.matmul(xyz_camera.reshape(batch, -1, 3), rot_c2w.transpose(-1, -2))
        xyz_world = xyz_world + trans_c2w[:, None, :]
        xyz_world = torch.nan_to_num(xyz_world, nan=0.0, posinf=0.0, neginf=0.0)
        confidence = torch.nan_to_num(confidence.reshape(batch, -1, 1), nan=0.0, posinf=1.0, neginf=0.0)
        valid = torch.isfinite(xyz_world).all(dim=-1) & torch.isfinite(confidence.squeeze(-1))

        return {
            "raw_points": xyz_world.contiguous(),
            "raw_confidence": confidence.contiguous(),
            "raw_valid_mask": valid.contiguous(),
            "proposal_depth": depth.reshape(batch, -1, 1).contiguous(),
        }

    def _voxelize_one(
        self,
        points: torch.Tensor,
        confidence: torch.Tensor,
        valid_mask: torch.Tensor,
        z_inv_global: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        device = points.device
        dtype = points.dtype
        voxel_size = float(self.cfg.model.voxel_size)
        if voxel_size <= 0.0:
            raise ValueError(f"voxel_size must be positive, got {voxel_size}.")

        valid = valid_mask & torch.isfinite(points).all(dim=-1) & torch.isfinite(confidence.squeeze(-1))
        if not bool(valid.any()):
            valid = torch.ones_like(valid, dtype=torch.bool)

        points_v = points[valid]
        conf_v = confidence[valid, 0].clamp(0.0, 1.0)
        scaled = points_v / voxel_size
        base = torch.floor(scaled).to(dtype=torch.long)
        frac = (scaled - base.to(dtype=dtype)).clamp(0.0, 1.0)

        offsets = self.corner_offsets.to(device=device)
        corner_indices = base[:, None, :] + offsets.view(1, 8, 3)
        corner_frac = torch.where(offsets.view(1, 8, 3).bool(), frac[:, None, :], 1.0 - frac[:, None, :])
        tri_weights = corner_frac.prod(dim=-1).clamp_min(0.0)
        conf_weights = tri_weights * conf_v[:, None]

        flat_indices = corner_indices.reshape(-1, 3)
        flat_tri_weights = tri_weights.reshape(-1)
        flat_conf_weights = conf_weights.reshape(-1)
        flat_points = points_v[:, None, :].expand(-1, 8, 3).reshape(-1, 3)

        unique_indices, inverse = torch.unique(flat_indices, dim=0, return_inverse=True)
        num_voxels = int(unique_indices.shape[0])
        centers = unique_indices.to(dtype=dtype) * voxel_size

        mass = points.new_zeros(num_voxels)
        point_count = points.new_zeros(num_voxels)
        mass.index_add_(0, inverse, flat_conf_weights.to(dtype))
        point_count.index_add_(0, inverse, flat_tri_weights.to(dtype))

        center_for_contrib = centers[inverse]
        delta = flat_points - center_for_contrib
        weighted_delta = delta * flat_conf_weights.to(dtype).unsqueeze(-1)
        weighted_delta2 = delta.pow(2) * flat_conf_weights.to(dtype).unsqueeze(-1)

        delta_sum = points.new_zeros(num_voxels, 3)
        delta2_sum = points.new_zeros(num_voxels, 3)
        delta_sum.index_add_(0, inverse, weighted_delta)
        delta2_sum.index_add_(0, inverse, weighted_delta2)

        denom = mass.clamp_min(1.0e-8).unsqueeze(-1)
        mean_offset = delta_sum / denom
        variance = (delta2_sum / denom - mean_offset.pow(2)).clamp_min(0.0)
        mean_confidence = mass / point_count.clamp_min(1.0e-8)

        active = mass > float(self.cfg.model.active_voxel_threshold)
        if not bool(active.any()):
            active = mass == mass.max()

        active_mass = mass[active]
        z_inv = z_inv_global.to(device=device, dtype=dtype).view(1, -1).expand(int(active.sum().item()), -1)
        voxel_features = torch.cat(
            [
                z_inv,
                active_mass.unsqueeze(-1),
                mean_confidence[active].unsqueeze(-1),
                mean_offset[active],
                variance[active],
            ],
            dim=-1,
        )
        total_mass = flat_conf_weights.to(dtype).sum().clamp_min(1.0e-8)
        coverage = active_mass.sum() / total_mass

        return {
            "centers": centers[active],
            "indices": unique_indices[active],
            "features": voxel_features,
            "mass": active_mass.unsqueeze(-1),
            "mean_confidence": mean_confidence[active].unsqueeze(-1),
            "point_count": point_count[active].unsqueeze(-1),
            "coverage": coverage.view(()),
        }

    def _voxelize(
        self,
        raw_points: torch.Tensor,
        raw_confidence: torch.Tensor,
        raw_valid_mask: torch.Tensor,
        z_inv_global: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        per_batch = [
            self._voxelize_one(raw_points[i], raw_confidence[i], raw_valid_mask[i], z_inv_global[i])
            for i in range(raw_points.shape[0])
        ]
        valid_masks = [torch.ones(item["centers"].shape[0], device=raw_points.device, dtype=torch.bool) for item in per_batch]
        return {
            "centers": self._pad_float([item["centers"] for item in per_batch], 0.0),
            "indices": self._pad_float([item["indices"].to(raw_points.dtype) for item in per_batch], 0.0).to(torch.long),
            "features": self._pad_float([item["features"] for item in per_batch], 0.0),
            "mass": self._pad_float([item["mass"] for item in per_batch], 0.0),
            "mean_confidence": self._pad_float([item["mean_confidence"] for item in per_batch], 0.0),
            "point_count": self._pad_float([item["point_count"] for item in per_batch], 0.0),
            "valid_mask": self._pad_bool(valid_masks),
            "coverage": torch.stack([item["coverage"] for item in per_batch], dim=0),
        }

    def _head_to_gaussians(self, centers: torch.Tensor, features: torch.Tensor, valid_mask: torch.Tensor) -> Dict[str, torch.Tensor]:
        batch, num_voxels, _ = centers.shape
        g = self.gaussians_per_voxel
        params = self.gaussian_head(features).view(batch, num_voxels, g, self.params_per_gaussian)

        cursor = 0
        local_offset = params[..., cursor: cursor + 3]
        cursor += 3
        scaling_raw = params[..., cursor: cursor + 3]
        cursor += 3
        rotation_raw = params[..., cursor: cursor + 4]
        cursor += 4
        opacity_raw = params[..., cursor: cursor + 1]
        cursor += 1
        features_dc_raw = params[..., cursor: cursor + 3]
        cursor += 3
        features_rest_raw = params[..., cursor:]

        voxel_size = float(self.cfg.model.voxel_size)
        local_scale = float(self.cfg.model.gaussian_local_offset_scale) * voxel_size
        xyz = centers[:, :, None, :] + torch.tanh(local_offset) * local_scale
        scale_max = max(float(self.cfg.model.gaussian_scale_max), 1.0e-5)
        scaling = torch.sigmoid(scaling_raw) * scale_max + 1.0e-5

        identity_bias = rotation_raw.new_tensor([1.0, 0.0, 0.0, 0.0]).view(1, 1, 1, 4)
        rotation = F.normalize(rotation_raw + identity_bias, dim=-1, eps=1.0e-6)
        opacity = torch.sigmoid(opacity_raw)
        features_dc = torch.sigmoid(features_dc_raw).unsqueeze(-2)
        features_rest = 0.1 * torch.tanh(features_rest_raw.view(batch, num_voxels, g, 3, 3))

        gaussian_valid = valid_mask[:, :, None].expand(batch, num_voxels, g).contiguous()
        opacity = opacity * gaussian_valid.unsqueeze(-1).to(opacity.dtype)

        return {
            "xyz": xyz.reshape(batch, num_voxels * g, 3).contiguous(),
            "scaling": scaling.reshape(batch, num_voxels * g, 3).contiguous(),
            "rotation": rotation.reshape(batch, num_voxels * g, 4).contiguous(),
            "opacity": opacity.reshape(batch, num_voxels * g, 1).contiguous(),
            "features_dc": features_dc.reshape(batch, num_voxels * g, 1, 3).contiguous(),
            "features_rest": features_rest.reshape(batch, num_voxels * g, 3, 3).contiguous(),
            "valid_mask": gaussian_valid.reshape(batch, num_voxels * g).contiguous(),
        }

    def forward(
        self,
        proposal_map: torch.Tensor,
        z_inv: torch.Tensor,
        source_cameras_view_to_world: torch.Tensor,
        intrinsics: torch.Tensor,
        activate_output: bool = True,
    ) -> Dict[str, torch.Tensor]:
        del activate_output
        z_inv_global = torch.nan_to_num(z_inv, nan=0.0, posinf=0.0, neginf=0.0).mean(dim=1)
        proposal = self._proposal_to_world_points(
            proposal_map=proposal_map,
            source_cameras_view_to_world=source_cameras_view_to_world,
            intrinsics=intrinsics,
        )
        voxels = self._voxelize(
            raw_points=proposal["raw_points"],
            raw_confidence=proposal["raw_confidence"],
            raw_valid_mask=proposal["raw_valid_mask"],
            z_inv_global=z_inv_global,
        )
        voxel_features = self.voxel_feature_projection(voxels["features"])
        refined = self.pointnext(voxel_features, voxels["indices"], voxels["valid_mask"])
        gaussians = self._head_to_gaussians(voxels["centers"], refined, voxels["valid_mask"])

        gaussians.update(
            {
                "raw_points": proposal["raw_points"],
                "raw_confidence": proposal["raw_confidence"],
                "raw_valid_mask": proposal["raw_valid_mask"],
                "proposal_depth": proposal["proposal_depth"],
                "voxel_centers": voxels["centers"],
                "voxel_features": voxels["features"],
                "voxel_valid_mask": voxels["valid_mask"],
                "voxel_mass": voxels["mass"],
                "voxel_mean_confidence": voxels["mean_confidence"],
                "voxel_point_count": voxels["point_count"],
                "point_to_voxel_coverage": voxels["coverage"],
            }
        )
        return gaussians
