from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation, Slerp


def interpolate_camera_pose(
    c2w_a: np.ndarray, c2w_b: np.ndarray, alpha: float
) -> np.ndarray:
    """Interpolate base-frame camera poses with linear translation and quaternion SLERP."""
    if not 0.0 <= float(alpha) <= 1.0:
        raise ValueError("Camera interpolation alpha must lie in [0,1].")
    a = np.asarray(c2w_a, dtype=np.float64)
    b = np.asarray(c2w_b, dtype=np.float64)
    if a.shape != (4, 4) or b.shape != (4, 4):
        raise ValueError("Camera poses must be 4x4.")
    rotations = Rotation.from_matrix(np.stack((a[:3, :3], b[:3, :3])))
    rotation = Slerp((0.0, 1.0), rotations)([float(alpha)]).as_matrix()[0]
    output = np.eye(4, dtype=np.float64)
    output[:3, :3] = rotation
    output[:3, 3] = (1.0 - float(alpha)) * a[:3, 3] + float(alpha) * b[:3, 3]
    return output


def interpolate_intrinsics(
    K_a: np.ndarray, K_b: np.ndarray, alpha: float
) -> np.ndarray:
    a = np.asarray(K_a, dtype=np.float64)
    b = np.asarray(K_b, dtype=np.float64)
    if a.shape != (3, 3) or b.shape != (3, 3):
        raise ValueError("Pinhole intrinsics must be 3x3.")
    output = (1.0 - float(alpha)) * a + float(alpha) * b
    output[0, 1] = output[1, 0] = 0.0
    output[2] = (0.0, 0.0, 1.0)
    return output


def warp_rgbd_to_camera(
    rgb: np.ndarray,
    depth: np.ndarray,
    confidence: np.ndarray,
    validity: np.ndarray,
    source_K: np.ndarray,
    source_c2w: np.ndarray,
    target_K: np.ndarray,
    target_c2w: np.ndarray,
    *,
    znear: float = 0.01,
    zfar: float = 10.0,
    zbuffer_tolerance_m: float = 1.0e-3,
) -> dict[str, np.ndarray]:
    """Forward-project one calibrated X-Lens RGB-D view with a nearest z-buffer."""
    image = np.asarray(rgb)
    z = np.asarray(depth, dtype=np.float64)
    conf = np.asarray(confidence, dtype=np.float64)
    valid = np.asarray(validity, dtype=bool)
    height, width = z.shape
    if (
        image.shape != (height, width, 3)
        or conf.shape != z.shape
        or valid.shape != z.shape
    ):
        raise ValueError("RGB, depth, confidence, and validity must share one grid.")
    source_K = np.asarray(source_K, dtype=np.float64)
    source_c2w = np.asarray(source_c2w, dtype=np.float64)
    target_K = np.asarray(target_K, dtype=np.float64)
    target_w2c = np.linalg.inv(np.asarray(target_c2w, dtype=np.float64))
    y, x = np.meshgrid(
        np.arange(height, dtype=np.float64) + 0.5,
        np.arange(width, dtype=np.float64) + 0.5,
        indexing="ij",
    )
    source_points = np.stack(
        (
            (x - source_K[0, 2]) / source_K[0, 0] * z,
            (y - source_K[1, 2]) / source_K[1, 1] * z,
            z,
        ),
        axis=-1,
    )
    world = source_points @ source_c2w[:3, :3].T + source_c2w[:3, 3]
    target = world @ target_w2c[:3, :3].T + target_w2c[:3, 3]
    target_z = target[..., 2]
    projected_x = (
        target_K[0, 0] * target[..., 0] / np.maximum(target_z, 1.0e-9)
        + target_K[0, 2]
        - 0.5
    )
    projected_y = (
        target_K[1, 1] * target[..., 1] / np.maximum(target_z, 1.0e-9)
        + target_K[1, 2]
        - 0.5
    )
    pixel_x = np.rint(projected_x).astype(np.int64)
    pixel_y = np.rint(projected_y).astype(np.int64)
    valid &= (
        np.isfinite(world).all(axis=-1)
        & np.isfinite(target).all(axis=-1)
        & np.isfinite(conf)
        & (z > 0.0)
        & (target_z > float(znear))
        & (target_z < float(zfar))
        & (pixel_x >= 0)
        & (pixel_x < width)
        & (pixel_y >= 0)
        & (pixel_y < height)
    )
    flat_index = (pixel_y[valid] * width + pixel_x[valid]).astype(np.int64)
    projected_depth = target_z[valid]
    zbuffer = np.full(height * width, np.inf, dtype=np.float64)
    np.minimum.at(zbuffer, flat_index, projected_depth)
    front = projected_depth <= zbuffer[flat_index] + float(zbuffer_tolerance_m)
    flat_index = flat_index[front]
    source_colors = image[valid][front].astype(np.float64)
    source_confidence = conf[valid][front]
    source_depth = projected_depth[front]
    count = np.zeros(height * width, dtype=np.float64)
    color_sum = np.zeros((height * width, 3), dtype=np.float64)
    confidence_sum = np.zeros(height * width, dtype=np.float64)
    depth_sum = np.zeros(height * width, dtype=np.float64)
    np.add.at(count, flat_index, 1.0)
    np.add.at(color_sum, flat_index, source_colors)
    np.add.at(confidence_sum, flat_index, source_confidence)
    np.add.at(depth_sum, flat_index, source_depth)
    supported = count > 0
    output_rgb = np.zeros((height * width, 3), dtype=np.float32)
    output_depth = np.zeros(height * width, dtype=np.float32)
    output_confidence = np.zeros(height * width, dtype=np.float32)
    output_rgb[supported] = (color_sum[supported] / count[supported, None]).astype(
        np.float32
    )
    output_depth[supported] = (depth_sum[supported] / count[supported]).astype(
        np.float32
    )
    output_confidence[supported] = (
        confidence_sum[supported] / count[supported]
    ).astype(np.float32)
    return {
        "rgb": output_rgb.reshape(height, width, 3),
        "depth": output_depth.reshape(height, width),
        "confidence": output_confidence.reshape(height, width),
        "validity": supported.reshape(height, width),
    }


def fuse_two_warps(
    warp_a: dict[str, np.ndarray],
    warp_b: dict[str, np.ndarray],
    *,
    relative_depth_tolerance: float = 0.05,
    absolute_depth_tolerance_m: float = 0.03,
) -> dict[str, np.ndarray]:
    valid_a = np.asarray(warp_a["validity"], dtype=bool)
    valid_b = np.asarray(warp_b["validity"], dtype=bool)
    if valid_a.shape != valid_b.shape:
        raise ValueError("Two geometric warps must share one target grid.")
    rgb_a = np.asarray(warp_a["rgb"], dtype=np.float32)
    rgb_b = np.asarray(warp_b["rgb"], dtype=np.float32)
    depth_a = np.asarray(warp_a["depth"], dtype=np.float32)
    depth_b = np.asarray(warp_b["depth"], dtype=np.float32)
    confidence_a = np.clip(np.asarray(warp_a["confidence"], dtype=np.float32), 0.0, 1.0)
    confidence_b = np.clip(np.asarray(warp_b["confidence"], dtype=np.float32), 0.0, 1.0)
    overlap = valid_a & valid_b
    depth_difference = np.abs(depth_a - depth_b)
    relative_depth_difference = depth_difference / np.maximum(
        np.minimum(depth_a, depth_b), 1.0e-6
    )
    rgb_difference = np.abs(rgb_a - rgb_b).mean(axis=-1) / 255.0
    compatible = overlap & (
        (depth_difference <= float(absolute_depth_tolerance_m))
        | (relative_depth_difference <= float(relative_depth_tolerance))
    )
    choose_a = valid_a & ~valid_b
    choose_b = valid_b & ~valid_a
    incompatible = overlap & ~compatible
    choose_a |= incompatible & (depth_a <= depth_b)
    choose_b |= incompatible & (depth_b < depth_a)
    weight_a = confidence_a + 1.0e-3
    weight_b = confidence_b + 1.0e-3
    total_weight = weight_a + weight_b
    fused_rgb = np.zeros_like(rgb_a)
    fused_depth = np.zeros_like(depth_a)
    fused_confidence = np.zeros_like(confidence_a)
    fused_rgb[choose_a] = rgb_a[choose_a]
    fused_rgb[choose_b] = rgb_b[choose_b]
    fused_depth[choose_a] = depth_a[choose_a]
    fused_depth[choose_b] = depth_b[choose_b]
    fused_confidence[choose_a] = confidence_a[choose_a]
    fused_confidence[choose_b] = confidence_b[choose_b]
    fused_rgb[compatible] = (
        rgb_a[compatible] * weight_a[compatible, None]
        + rgb_b[compatible] * weight_b[compatible, None]
    ) / total_weight[compatible, None]
    fused_depth[compatible] = (
        depth_a[compatible] * weight_a[compatible]
        + depth_b[compatible] * weight_b[compatible]
    ) / total_weight[compatible]
    fused_confidence[compatible] = 0.5 * (
        confidence_a[compatible] + confidence_b[compatible]
    )
    validity = valid_a | valid_b
    source_ids = np.full(validity.shape, -1, dtype=np.int8)
    source_ids[choose_a] = 0
    source_ids[choose_b] = 1
    source_ids[compatible] = 2
    return {
        "rgb": fused_rgb,
        "depth": fused_depth,
        "confidence": fused_confidence,
        "validity": validity,
        "overlap": overlap,
        "compatible_overlap": compatible,
        "source_camera_ids": source_ids,
        "depth_disagreement": depth_difference,
        "relative_depth_disagreement": relative_depth_difference,
        "rgb_disagreement": rgb_difference,
    }


def synthetic_confidence(
    fused_warp: dict[str, np.ndarray],
    generated_rgb: np.ndarray,
) -> tuple[np.ndarray, float]:
    validity = np.asarray(fused_warp["validity"], dtype=bool)
    observed_confidence = np.clip(
        np.asarray(fused_warp["confidence"], np.float32), 0.0, 1.0
    )
    generated = np.asarray(generated_rgb, dtype=np.float32)
    warp_rgb = np.asarray(fused_warp["rgb"], dtype=np.float32)
    observed_consistency = np.exp(-np.abs(generated - warp_rgb).mean(axis=-1) / 32.0)
    depth_agreement = np.exp(
        -np.asarray(fused_warp["relative_depth_disagreement"], np.float32) / 0.10
    )
    pixel = np.full(validity.shape, 0.10, dtype=np.float32)
    pixel[validity] = (
        0.45 * observed_confidence[validity]
        + 0.35 * observed_consistency[validity]
        + 0.20 * depth_agreement[validity]
    )
    coverage = float(validity.mean())
    view_confidence = coverage * float(
        pixel[validity].mean() if validity.any() else 0.0
    )
    return np.clip(pixel, 0.0, 1.0), view_confidence
