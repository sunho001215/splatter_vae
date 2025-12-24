# utils/ray_utils.py

import torch
import numpy as np
import random
from datetime import datetime
import sys


def build_ray_dirs_from_intrinsics(
    H: int,
    W: int,
    fx,
    fy,
    cx,
    cy,
    device: torch.device,
    inverted_x: bool = False,
    inverted_y: bool = False,
) -> torch.Tensor:
    """
    Construct per-pixel ray directions in **OpenCV-style** camera coordinates:

        - +x: right
        - +y: down (image convention)
        - +z: forward

    For pixel centers (u, v) with:
        u in [0, W-1], v in [0, H-1]

        x = (u + 0.5 - cx) / fx
        y = (v + 0.5 - cy) / fy
        z = 1

    This implementation is **fully differentiable** w.r.t. fx, fy, cx, cy,
    as long as they are passed as torch.Tensors with requires_grad=True.

    Args:
        H, W: image size
        fx, fy, cx, cy:
            - scalar (Python float, int, or 0-dim torch.Tensor), or
            - 1D torch.Tensor of shape (B,) for batched intrinsics.
        device: torch.device
        inverted_x, inverted_y: optional axis flips.

    Returns:
        ray_dirs: (B, 3, H, W) if fx is batched (B,), otherwise (1, 3, H, W)
    """
    # Convert intrinsics to tensors on the correct device (keep grad if any)
    fx_t = torch.as_tensor(fx, device=device, dtype=torch.float32)
    fy_t = torch.as_tensor(fy, device=device, dtype=torch.float32)
    cx_t = torch.as_tensor(cx, device=device, dtype=torch.float32)
    cy_t = torch.as_tensor(cy, device=device, dtype=torch.float32)

    # Ensure a batch dimension: (B,)
    if fx_t.dim() == 0:
        fx_t = fx_t.view(1)
        fy_t = fy_t.view(1)
        cx_t = cx_t.view(1)
        cy_t = cy_t.view(1)

    assert fx_t.shape == fy_t.shape == cx_t.shape == cy_t.shape, \
        "fx, fy, cx, cy must have the same shape (either scalar or (B,))."

    B = fx_t.shape[0]

    # Pixel centers in [0, W), [0, H)
    u = torch.arange(W, device=device, dtype=torch.float32) + 0.5   # (W,)
    v = torch.arange(H, device=device, dtype=torch.float32) + 0.5   # (H,)

    # Grid with image convention: x horizontal, y vertical
    # xs: (H, W) ~ u, ys: (H, W) ~ v
    ys, xs = torch.meshgrid(
        v, u, indexing="ij"
    )  # ys(H,W) = v, xs(H,W) = u

    # Expand to (B, H, W) for broadcasting with per-batch intrinsics
    xs = xs.unsqueeze(0).expand(B, H, W)
    ys = ys.unsqueeze(0).expand(B, H, W)

    # Reshape intrinsics to (B, 1, 1)
    fx_t = fx_t.view(B, 1, 1)
    fy_t = fy_t.view(B, 1, 1)
    cx_t = cx_t.view(B, 1, 1)
    cy_t = cy_t.view(B, 1, 1)

    # Normalized directions (still differentiable in fx, fy, cx, cy)
    x = (xs - cx_t) / fx_t
    y = (ys - cy_t) / fy_t

    if inverted_x:
        x = -x
    if inverted_y:
        y = -y

    # OpenCV-style camera: forward direction is +Z
    z = torch.ones_like(x)

    # Stack as (B, 3, H, W)
    ray_dirs = torch.stack([x, y, z], dim=1)  # (B,3,H,W)
    return ray_dirs
