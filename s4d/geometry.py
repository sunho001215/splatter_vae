"""Pinhole geometry helpers (torch, any device). OpenCV convention: x right, y down, z forward.

Pixel (i, j) (column i, row j) has continuous image coordinates (i + 0.5, j + 0.5).
"""

from __future__ import annotations

import torch


def pixel_centers(height: int, width: int, device=None, dtype=torch.float32) -> torch.Tensor:
    """Return ``(H, W, 2)`` continuous pixel-centre coordinates ``(u, v)``."""
    v, u = torch.meshgrid(
        torch.arange(height, device=device, dtype=dtype) + 0.5,
        torch.arange(width, device=device, dtype=dtype) + 0.5,
        indexing="ij",
    )
    return torch.stack((u, v), dim=-1)


def lift_depth(depth: torch.Tensor, K: torch.Tensor, c2w: torch.Tensor) -> torch.Tensor:
    """Lift camera-z depth ``(..., H, W)`` with ``K (..., 3, 3)`` and ``c2w (..., 4, 4)`` to world ``(..., H, W, 3)``."""
    H, W = depth.shape[-2:]
    uv = pixel_centers(H, W, device=depth.device, dtype=depth.dtype)
    fx, fy = K[..., 0, 0], K[..., 1, 1]
    cx, cy = K[..., 0, 2], K[..., 1, 2]
    x = (uv[..., 0] - cx[..., None, None]) / fx[..., None, None] * depth
    y = (uv[..., 1] - cy[..., None, None]) / fy[..., None, None] * depth
    cam = torch.stack((x, y, depth), dim=-1)
    R = c2w[..., :3, :3]
    t = c2w[..., :3, 3]
    return torch.einsum("...ij,...hwj->...hwi", R, cam) + t[..., None, None, :]


def project(xyz_world: torch.Tensor, K: torch.Tensor, w2c: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Project world points ``(..., P, 3)`` into continuous pixel coords ``(..., P, 2)`` and camera-z depth ``(..., P)``."""
    R = w2c[..., :3, :3]
    t = w2c[..., :3, 3]
    cam = torch.einsum("...ij,...pj->...pi", R, xyz_world) + t[..., None, :]
    z = cam[..., 2]
    safe_z = torch.where(z.abs() > 1e-8, z, torch.full_like(z, 1e-8))
    u = K[..., 0, 0, None] * cam[..., 0] / safe_z + K[..., 0, 2, None]
    v = K[..., 1, 1, None] * cam[..., 1] / safe_z + K[..., 1, 2, None]
    return torch.stack((u, v), dim=-1), z


def invert_se3(T: torch.Tensor) -> torch.Tensor:
    R = T[..., :3, :3]
    t = T[..., :3, 3]
    out = torch.zeros_like(T)
    Rt = R.transpose(-1, -2)
    out[..., :3, :3] = Rt
    out[..., :3, 3] = -(Rt @ t[..., None])[..., 0]
    out[..., 3, 3] = 1.0
    return out


def quat_wxyz_to_matrix(q: torch.Tensor) -> torch.Tensor:
    """Unit quaternion ``(..., 4)`` in (w, x, y, z) order to rotation matrix ``(..., 3, 3)``."""
    q = q / q.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    w, x, y, z = q.unbind(-1)
    return torch.stack(
        (
            torch.stack((1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)), -1),
            torch.stack((2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)), -1),
            torch.stack((2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)), -1),
        ),
        dim=-2,
    )


def rigid_body_displacement(
    xyz: torch.Tensor,
    body_id: torch.Tensor,
    xpos_a: torch.Tensor,
    xquat_a: torch.Tensor,
    xpos_b: torch.Tensor,
    xquat_b: torch.Tensor,
) -> torch.Tensor:
    """Displacement of world points attached to rigid bodies between pose sets a and b.

    xyz      (N, P, 3) world points at time a
    body_id  (N, P)    body index per point (must be < number of bodies)
    xpos_*   (N, B, 3), xquat_* (N, B, 4) wxyz body poses at times a and b
    returns  (N, P, 3) with X_b - X_a where X_b = R_b R_a^T (X_a - p_a) + p_b
    """
    Ra = quat_wxyz_to_matrix(xquat_a)
    Rb = quat_wxyz_to_matrix(xquat_b)
    rel_R = Rb @ Ra.transpose(-1, -2)  # (N, B, 3, 3)
    rel_t = xpos_b - torch.einsum("nbij,nbj->nbi", rel_R, xpos_a)  # (N, B, 3)
    idx = body_id.long()
    R = torch.gather(rel_R, 1, idx[..., None, None].expand(-1, -1, 3, 3))
    t = torch.gather(rel_t, 1, idx[..., None].expand(-1, -1, 3))
    moved = torch.einsum("npij,npj->npi", R, xyz) + t
    return moved - xyz
