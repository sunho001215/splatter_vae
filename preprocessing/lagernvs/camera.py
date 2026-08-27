from __future__ import annotations

import math

import torch
import torch.nn.functional as F

LAGERNVS_IMAGE_SIZE = 256
DROID_DISPLAY_HEIGHT = 180
DROID_DISPLAY_WIDTH = 320
DEFAULT_CANONICAL_FOCAL_PX = 186.5
CAMERA_SCALE_MULTIPLIER = 1.35


def canonical_intrinsics(
    leading_shape: tuple[int, ...],
    *,
    focal_px: float = DEFAULT_CANONICAL_FOCAL_PX,
    size: int = LAGERNVS_IMAGE_SIZE,
    device: torch.device | str | None = None,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    if not math.isfinite(float(focal_px)) or float(focal_px) <= 0.0:
        raise ValueError("LagerNVS canonical focal length must be positive and finite.")
    K = torch.zeros(*leading_shape, 3, 3, device=device, dtype=dtype)
    K[..., 0, 0] = float(focal_px)
    K[..., 1, 1] = float(focal_px)
    K[..., 0, 2] = float(size) / 2.0
    K[..., 1, 2] = float(size) / 2.0
    K[..., 2, 2] = 1.0
    return K


def _resampling_grid(
    source_K: torch.Tensor,
    target_K: torch.Tensor,
    *,
    source_height: int,
    source_width: int,
    target_height: int,
    target_width: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    if source_K.shape != target_K.shape or source_K.shape[-2:] != (3, 3):
        raise ValueError("Source and target intrinsics must have aligned 3x3 shapes.")
    leading = source_K.shape[:-2]
    u = torch.arange(target_width, device=source_K.device, dtype=torch.float32) + 0.5
    v = torch.arange(target_height, device=source_K.device, dtype=torch.float32) + 0.5
    vv, uu = torch.meshgrid(v, u, indexing="ij")
    pixels = torch.stack((uu, vv, torch.ones_like(uu)), dim=-1)
    pixels = pixels.reshape(1, target_height * target_width, 3).expand(
        math.prod(leading), -1, -1
    )
    homography = source_K.float().reshape(-1, 3, 3) @ torch.linalg.inv(
        target_K.float().reshape(-1, 3, 3)
    )
    source = torch.bmm(pixels, homography.transpose(1, 2))
    source_xy = source[..., :2] / source[..., 2:3].clamp_min(1.0e-8)
    source_xy = source_xy.reshape(*leading, target_height, target_width, 2)
    grid = torch.empty_like(source_xy)
    grid[..., 0] = 2.0 * source_xy[..., 0] / float(source_width) - 1.0
    grid[..., 1] = 2.0 * source_xy[..., 1] / float(source_height) - 1.0
    valid = (
        (source_xy[..., 0] >= 0.5)
        & (source_xy[..., 0] <= source_width - 0.5)
        & (source_xy[..., 1] >= 0.5)
        & (source_xy[..., 1] <= source_height - 0.5)
    )
    return grid, valid


def resample_pinhole_images(
    images: torch.Tensor,
    source_K: torch.Tensor,
    target_K: torch.Tensor,
    *,
    output_height: int,
    output_width: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    if images.shape[:-3] != source_K.shape[:-2] or images.shape[-3] <= 0:
        raise ValueError("Images and pinhole cameras must share their leading dimensions.")
    source_height, source_width = images.shape[-2:]
    channels = int(images.shape[-3])
    grid, inside = _resampling_grid(
        source_K,
        target_K,
        source_height=source_height,
        source_width=source_width,
        target_height=int(output_height),
        target_width=int(output_width),
    )
    flat = images.float().reshape(-1, channels, source_height, source_width)
    if images.dtype == torch.uint8:
        flat = flat / 255.0
    sampled = F.grid_sample(
        flat,
        grid.reshape(-1, output_height, output_width, 2),
        mode="bilinear",
        padding_mode="zeros",
        align_corners=False,
    )
    validity = F.grid_sample(
        torch.ones(
            flat.shape[0], 1, source_height, source_width, device=flat.device
        ),
        grid.reshape(-1, output_height, output_width, 2),
        mode="bilinear",
        padding_mode="zeros",
        align_corners=False,
    )
    validity = (validity >= 1.0 - 1.0e-6) & inside.reshape(
        -1, 1, output_height, output_width
    )
    leading = images.shape[:-3]
    return (
        sampled.reshape(*leading, channels, output_height, output_width).contiguous(),
        validity.reshape(*leading, 1, output_height, output_width).contiguous(),
    )


def canonicalize_droid_views(
    images: torch.Tensor,
    source_K: torch.Tensor,
    *,
    focal_px: float = DEFAULT_CANONICAL_FOCAL_PX,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Deterministically resample raw DROID views into one common 256x256 K."""

    target_K = canonical_intrinsics(
        source_K.shape[:-2],
        focal_px=focal_px,
        device=source_K.device,
        dtype=torch.float32,
    )
    canonical, validity = resample_pinhole_images(
        images,
        source_K,
        target_K,
        output_height=LAGERNVS_IMAGE_SIZE,
        output_width=LAGERNVS_IMAGE_SIZE,
    )
    return canonical, target_K, validity


def normalize_lagernvs_poses(
    c2w: torch.Tensor, *, num_conditioning_views: int = 2
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Match LagerNVS camera-based normalization exactly, batched."""

    if c2w.dim() != 4 or c2w.shape[-2:] != (4, 4):
        raise ValueError("LagerNVS c2w poses must have shape (B,V,4,4).")
    relative = torch.linalg.inv(c2w[:, :1]) @ c2w
    source_radius = relative[:, :num_conditioning_views, :3, 3].norm(dim=-1)
    scene_scale = CAMERA_SCALE_MULTIPLIER * source_radius.amax(dim=1).clamp_min(1.0e-6)
    normalized = relative.clone()
    normalized[..., :3, 3] /= scene_scale[:, None, None]
    camera_scale = normalized[:, :num_conditioning_views, :3, 3].norm(dim=-1).amax(
        dim=1
    )
    scene_scale_ratio = relative[..., :3, 3].norm(dim=-1).amax(dim=1) / scene_scale
    return normalized, camera_scale, scene_scale_ratio


def matrix_to_quaternion_xyzw(matrix: torch.Tensor) -> torch.Tensor:
    """Stable scalar-last quaternion conversion matching LagerNVS/VGGT."""

    if matrix.shape[-2:] != (3, 3):
        raise ValueError("Rotation matrices must end in 3x3.")
    m00, m01, m02, m10, m11, m12, m20, m21, m22 = matrix.reshape(
        *matrix.shape[:-2], 9
    ).unbind(-1)
    q_abs = torch.sqrt(
        torch.clamp(
            torch.stack(
                (
                    1 + m00 + m11 + m22,
                    1 + m00 - m11 - m22,
                    1 - m00 + m11 - m22,
                    1 - m00 - m11 + m22,
                ),
                dim=-1,
            ),
            min=0.0,
        )
    )
    candidates = torch.stack(
        (
            torch.stack((q_abs[..., 0] ** 2, m21 - m12, m02 - m20, m10 - m01), -1),
            torch.stack((m21 - m12, q_abs[..., 1] ** 2, m10 + m01, m02 + m20), -1),
            torch.stack((m02 - m20, m10 + m01, q_abs[..., 2] ** 2, m12 + m21), -1),
            torch.stack((m10 - m01, m20 + m02, m21 + m12, q_abs[..., 3] ** 2), -1),
        ),
        dim=-2,
    )
    candidates = candidates / (2.0 * q_abs[..., None].clamp_min(0.1))
    selector = F.one_hot(q_abs.argmax(-1), num_classes=4).bool()
    rijk = candidates[selector].reshape(*matrix.shape[:-2], 4)
    xyzw = rijk[..., (1, 2, 3, 0)]
    xyzw = F.normalize(xyzw, dim=-1, eps=1.0e-8)
    return torch.where(xyzw[..., 3:4] < 0, -xyzw, xyzw)


def quaternion_xyzw_to_matrix(quaternion: torch.Tensor) -> torch.Tensor:
    q = F.normalize(quaternion, dim=-1, eps=1.0e-8)
    x, y, z, w = q.unbind(-1)
    two = 2.0
    return torch.stack(
        (
            1 - two * (y * y + z * z),
            two * (x * y - z * w),
            two * (x * z + y * w),
            two * (x * y + z * w),
            1 - two * (x * x + z * z),
            two * (y * z - x * w),
            two * (x * z - y * w),
            two * (y * z + x * w),
            1 - two * (x * x + y * y),
        ),
        dim=-1,
    ).reshape(*q.shape[:-1], 3, 3)


def camera_tokens(
    normalized_c2w: torch.Tensor,
    K: torch.Tensor,
    camera_scale: torch.Tensor,
) -> torch.Tensor:
    height = width = LAGERNVS_IMAGE_SIZE
    fx, fy = K[..., 0, 0], K[..., 1, 1]
    fov_h = 2.0 * torch.atan((height / 2.0) / fy)
    fov_w = 2.0 * torch.atan((width / 2.0) / fx)
    pose = torch.cat(
        (
            normalized_c2w[..., :3, 3],
            matrix_to_quaternion_xyzw(normalized_c2w[..., :3, :3]),
            fov_h[..., None],
            fov_w[..., None],
        ),
        dim=-1,
    )
    scales = torch.stack((camera_scale, torch.zeros_like(camera_scale)), dim=-1)
    scales = scales[:, None].expand(-1, normalized_c2w.shape[1], -1)
    return torch.cat((pose, scales), dim=-1)


def plucker_rays(c2w: torch.Tensor, K: torch.Tensor, height: int, width: int) -> torch.Tensor:
    if c2w.shape[:-2] != K.shape[:-2]:
        raise ValueError("Plucker camera poses and intrinsics must align.")
    leading = c2w.shape[:-2]
    u = torch.arange(width, device=c2w.device, dtype=torch.float32) + 0.5
    v = torch.arange(height, device=c2w.device, dtype=torch.float32) + 0.5
    vv, uu = torch.meshgrid(v, u, indexing="ij")
    pixels = torch.stack((uu, vv, torch.ones_like(uu)), dim=-1)
    pixels = pixels.reshape(1, height * width, 3).expand(math.prod(leading), -1, -1)
    local = torch.bmm(
        pixels, torch.linalg.inv(K.float().reshape(-1, 3, 3)).transpose(1, 2)
    )
    local = F.normalize(local, dim=-1, eps=1.0e-8)
    rotation = c2w[..., :3, :3].float().reshape(-1, 3, 3)
    direction = torch.bmm(local, rotation.transpose(1, 2))
    origin = c2w[..., :3, 3].float().reshape(-1, 1, 3).expand_as(direction)
    moment = torch.cross(origin, direction, dim=-1)
    rays = torch.cat((moment, direction), dim=-1)
    return rays.reshape(*leading, height, width, 6).movedim(-1, -3).contiguous()


def lager_to_display_plane(
    canonical_image: torch.Tensor,
    canonical_K: torch.Tensor,
    *,
    display_focal_px: float | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Map square canonical output into 320x180 geometry without stretching."""

    if display_focal_px is None:
        display_focal_px = float(canonical_K[..., 0, 0].float().mean()) * (
            DROID_DISPLAY_HEIGHT / LAGERNVS_IMAGE_SIZE
        )
    display_K = canonical_K.clone().float()
    display_K[..., 0, 0] = float(display_focal_px)
    display_K[..., 1, 1] = float(display_focal_px)
    display_K[..., 0, 2] = DROID_DISPLAY_WIDTH / 2.0
    display_K[..., 1, 2] = DROID_DISPLAY_HEIGHT / 2.0
    display, valid = resample_pinhole_images(
        canonical_image,
        canonical_K,
        display_K,
        output_height=DROID_DISPLAY_HEIGHT,
        output_width=DROID_DISPLAY_WIDTH,
    )
    return display, display_K, valid
