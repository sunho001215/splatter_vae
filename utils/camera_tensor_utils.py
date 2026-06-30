from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn.functional as F


def gather_camera_rows(values: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
    batch_ids = torch.arange(values.shape[0], device=values.device)
    return values[batch_ids, indices]


def gather_target_cameras(values: torch.Tensor, target_indices: torch.Tensor) -> torch.Tensor:
    trailing_shape = values.shape[2:]
    gather_index = target_indices.view(
        target_indices.shape[0],
        target_indices.shape[1],
        *([1] * len(trailing_shape)),
    ).expand(target_indices.shape[0], target_indices.shape[1], *trailing_shape)
    return torch.gather(values, dim=1, index=gather_index)


def target_indices_excluding_source(source_indices: torch.Tensor, num_views: int) -> torch.Tensor:
    if num_views < 2:
        raise ValueError("Source-plus-target reconstruction requires at least two camera viewpoints.")
    all_views = torch.arange(num_views, device=source_indices.device).view(1, num_views)
    all_views = all_views.expand(source_indices.shape[0], num_views)
    keep_target = all_views != source_indices.view(-1, 1)
    return all_views[keep_target].view(source_indices.shape[0], num_views - 1)


def mask_or_ones(images_01: torch.Tensor, masks: Optional[torch.Tensor]) -> torch.Tensor:
    if masks is None:
        return torch.ones((*images_01.shape[:2], 1, *images_01.shape[-2:]), device=images_01.device, dtype=images_01.dtype)
    return masks.to(device=images_01.device, dtype=images_01.dtype).clamp(0.0, 1.0)


def look_at_c2w_opencv(eye: torch.Tensor, target: torch.Tensor, up: torch.Tensor) -> torch.Tensor:
    z_axis = F.normalize(target - eye, dim=0, eps=1.0e-6)
    up = F.normalize(up, dim=0, eps=1.0e-6)
    y_down = F.normalize(-up, dim=0, eps=1.0e-6)
    x_axis = F.normalize(torch.cross(y_down, z_axis, dim=0), dim=0, eps=1.0e-6)
    y_axis = F.normalize(torch.cross(z_axis, x_axis, dim=0), dim=0, eps=1.0e-6)
    mat = torch.eye(4, device=eye.device, dtype=eye.dtype)
    mat[:3, 0] = x_axis
    mat[:3, 1] = y_axis
    mat[:3, 2] = z_axis
    mat[:3, 3] = eye
    return mat


def invert_4x4_torch(mat: torch.Tensor) -> torch.Tensor:
    rot = mat[:3, :3]
    trans = mat[:3, 3]
    out = torch.eye(4, device=mat.device, dtype=mat.dtype)
    out[:3, :3] = rot.transpose(0, 1)
    out[:3, 3] = -(rot.transpose(0, 1) @ trans)
    return out


def trajectory_w2c(
    base_c2w: torch.Tensor,
    trajectory: str,
    num_frames: int,
    amplitude: float,
    focus_distance: float,
) -> torch.Tensor:
    device = base_c2w.device
    dtype = base_c2w.dtype
    eye0 = base_c2w[:3, 3]
    right = F.normalize(base_c2w[:3, 0], dim=0, eps=1.0e-6)
    up = F.normalize(-base_c2w[:3, 1], dim=0, eps=1.0e-6)
    forward = F.normalize(base_c2w[:3, 2], dim=0, eps=1.0e-6)
    target = eye0 + forward * float(focus_distance)
    frames = []
    t_values = torch.linspace(-1.0, 1.0, steps=max(2, int(num_frames)), device=device, dtype=dtype)
    for t in t_values:
        if trajectory == "lateral":
            eye = eye0 + right * (float(amplitude) * t)
        elif trajectory == "circular":
            angle = t * math.pi
            eye = eye0 + right * (float(amplitude) * torch.sin(angle)) + up * (float(amplitude) * 0.5 * torch.cos(angle))
        else:
            raise ValueError(f"Unknown validation trajectory: {trajectory}")
        frames.append(invert_4x4_torch(look_at_c2w_opencv(eye, target, up)))
    return torch.stack(frames, dim=0)
