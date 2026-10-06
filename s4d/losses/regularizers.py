"""Visibility regulariser: keep Gaussian centres in front of the cameras and inside the frusta."""

from __future__ import annotations

import torch

from s4d.geometry import project

MARGIN_PIXELS = 4.0


def visibility_loss(
    xyz_seq: torch.Tensor, w2c: torch.Tensor, K: torch.Tensor, height: int, width: int, near: float, far: float
) -> torch.Tensor:
    """xyz_seq (B,S,N,3), cameras (B,V,…) -> per-state violation (S,) averaged over cameras and Gaussians."""
    B, S, N, _ = xyz_seq.shape
    V = w2c.shape[1]
    pts = xyz_seq.float()[:, :, None].expand(B, S, V, N, 3)
    uv, z = project(pts, K.float()[:, None].expand(B, S, V, 3, 3), w2c.float()[:, None].expand(B, S, V, 4, 4))
    u, v = uv.unbind(-1)
    du = (torch.relu(-MARGIN_PIXELS - u) + torch.relu(u - (width + MARGIN_PIXELS))) / width
    dv = (torch.relu(-MARGIN_PIXELS - v) + torch.relu(v - (height + MARGIN_PIXELS))) / height
    dz = (torch.relu(near - z) + torch.relu(z - far)) / (far - near)
    violation = torch.nan_to_num(du + dv + dz, nan=1.0, posinf=1.0, neginf=1.0)
    return violation.mean(dim=(0, 2, 3))
