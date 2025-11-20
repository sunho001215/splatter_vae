import torch
import sys
from datetime import datetime
import numpy as np
import random


def build_ray_dirs_from_intrinsics(
    H: int,
    W: int,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
    device: torch.device,
    inverted_x: bool = False,
    inverted_y: bool = False,
) -> torch.Tensor:
    """
    Construct per-pixel ray directions in camera coordinates (OpenGL-style).

    For pixel centers (u, v) with:
        u in [0, W-1], v in [0, H-1]

    We define (OpenGL right-handed camera, looking along -Z):

        x = (u + 0.5 - cx) / fx
        y = -(v + 0.5 - cy) / fy
        z = -1

    So (x, y, z) is the unnormalized ray direction, with:
        +x right, +y up, -z forward (into the screen).

    Args:
        H, W: image size
        fx, fy, cx, cy: intrinsics
        inverted_x, inverted_y: optional axis flips for datasets.

    Returns:
        ray_dirs: (1, 3, H, W)
    """
    # Pixel centers in [0, W), [0, H)
    u = torch.arange(W, device=device, dtype=torch.float32) + 0.5  # (W,)
    v = torch.arange(H, device=device, dtype=torch.float32) + 0.5  # (H,)

    # Meshgrid with image convention: x horizontal, y vertical
    grid_u, grid_v = torch.meshgrid(u, v, indexing="xy")  # (W,H), (W,H)

    # Normalize by intrinsics, flip y to make +y up in camera space
    x = (grid_u - cx) / fx
    y = -(grid_v - cy) / fy

    if inverted_x:
        x = -x
    if inverted_y:
        y = -y

    # OpenGL-style camera: forward direction is -Z
    z = -torch.ones_like(x)

    # Stack as (3,H,W) then add batch dim
    ray_dirs = torch.stack([x, y, z], dim=0).unsqueeze(0)  # (1,3,H,W)
    return ray_dirs


def build_full_proj_matrix(
    K: torch.Tensor,
    H: int,
    W: int,
    znear: float,
    zfar: float,
    device: torch.device,
) -> torch.Tensor:
    """
    Build a 4x4 perspective projection matrix from intrinsics K and
    near/far planes.

    This maps camera/view coordinates (x, y, z) into clip coordinates.
    Here we use a standard OpenGL-like projection that is compatible
    with how diff_gaussian_rasterization is usually used:

        - x, y are normalized to [-1, 1] using fx, fy, cx, cy
        - z is mapped to [-1, 1] using znear, zfar

    NOTE:
      - You can swap this out for the exact projection used in your
        original Gaussian Splatting / Splatter setup if needed.
    """
    fx = K[0, 0]
    fy = K[1, 1]
    cx = K[0, 2]
    cy = K[1, 2]

    # Normalized device coordinates scaling (width, height)
    # Derived from: x_ndc = 2 * ((u - cx) / W) / (fx / W) - 1, etc.
    # Final form:
    #   tan(fovx / 2) = (W / 2) / fx
    #   tan(fovy / 2) = (H / 2) / fy
    # is handled separately in render_predicted, but projmatrix still
    # needs a reasonable perspective transform on z.
    #
    # Here we just put a standard perspective block on z.
    proj = torch.zeros((4, 4), dtype=torch.float32, device=device)

    # Horizontal / vertical scaling are not critical here (tanfovx/tanfovy
    # inside GaussianRasterizer are what really matters), but we keep them
    # consistent:
    proj[0, 0] = 2.0 * fx / W
    proj[1, 1] = 2.0 * fy / H
    proj[0, 2] = 1.0 - 2.0 * cx / W
    proj[1, 2] = 2.0 * cy / H - 1.0

    # Depth mapping
    proj[2, 2] = (zfar + znear) / (znear - zfar)
    proj[2, 3] = 2 * zfar * znear / (znear - zfar)
    proj[3, 2] = -1.0

    return proj


def camera_center_from_viewmatrix(T: torch.Tensor) -> torch.Tensor:
    """
    Compute camera center in "world" coordinates from a world->view matrix.

    If T = [R | t] (3x4), where x_view = R x_world + t, then the
    camera center C in world coordinates satisfies:
        R C + t = 0 -> C = -R^T t

    Args:
        T: (4,4) or (B,4,4) world->view matrix.

    Returns:
        center: (3,) or (B,3) camera center in world coordinates.
    """
    if T.dim() == 2:
        R = T[:3, :3]
        t = T[:3, 3]
        C = - R.transpose(0, 1) @ t
        return C
    elif T.dim() == 3:
        R = T[:, :3, :3]       # (B,3,3)
        t = T[:, :3, 3:4]      # (B,3,1)
        C = - torch.bmm(R.transpose(1, 2), t)  # (B,3,1)
        return C.squeeze(-1)  # (B,3)
    else:
        raise ValueError("T must be (4,4) or (B,4,4).")