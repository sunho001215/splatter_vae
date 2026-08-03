from __future__ import annotations

from contextlib import nullcontext
from itertools import islice
from typing import Any, Dict, Optional

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

import wandb

from models.training.schedules import temporal_loss_ramp
from models.training.reconstruction import (
    compute_reconstruction_and_renders,
    encode_per_view_sequence_batch,
)
from models.gaussian.parameterization import DirectSplatterToGaussians
from models.training.losses import compute_view_structured_representation_losses
from models.gaussian.parameterization import SplatterConfig
from models.gaussian.rendering import render_rgb
from models.training.config import TrainConfig
from models.splattervae.model import SplatterVAE


FLOW_VISUALIZATION_MAX_MAGNITUDE_PIXELS = 32.0
FLOW_TRACK_POINT_COUNT = 12
FLOW_TRACK_PREFERRED_COVERAGE = 0.05
NOVEL_VIEW_ORBIT_FRAMES = 24
NOVEL_VIEW_ORBIT_FPS = 8
# dataset/metaworld/config.yaml uses this look-at point for every generated view.
METAWORLD_SCENE_CENTER = (0.0, 0.6, 0.0)
MASK_OVERLAY_ALPHA = 0.55
MASK_OVERLAY_COLOR = (1.0, 0.15, 0.15)
FLOW_TRACK_COLORS = (
    (1.00, 0.20, 0.20),
    (0.20, 0.85, 0.25),
    (0.20, 0.45, 1.00),
    (1.00, 0.80, 0.10),
    (0.95, 0.25, 0.85),
    (0.10, 0.85, 0.90),
    (1.00, 0.50, 0.10),
    (0.55, 0.25, 0.95),
    (0.55, 0.90, 0.10),
    (1.00, 0.35, 0.60),
    (0.10, 0.65, 0.55),
    (0.85, 0.65, 0.20),
    (0.35, 0.75, 1.00),
    (0.75, 0.40, 0.15),
    (0.65, 0.65, 1.00),
    (0.20, 0.95, 0.65),
)



def _circular_look_at_w2c(
    source_c2w: torch.Tensor,
    scene_center: torch.Tensor,
    num_frames: int,
) -> torch.Tensor:
    """Build an OpenCV camera orbit whose first frame is the source viewpoint."""
    if source_c2w.shape != (4, 4):
        raise ValueError(f"Expected one source c2w matrix, got {tuple(source_c2w.shape)}.")
    frame_count = max(1, int(num_frames))
    source_c2w = source_c2w.float()
    center = scene_center.to(device=source_c2w.device, dtype=torch.float32).reshape(3)
    source_position = source_c2w[:3, 3]
    source_offset = source_position - center
    angles = torch.arange(frame_count, device=source_c2w.device, dtype=torch.float32)
    angles = angles * (2.0 * torch.pi / frame_count)
    cos_angle, sin_angle = angles.cos(), angles.sin()
    positions = torch.stack(
        (
            center[0] + cos_angle * source_offset[0] - sin_angle * source_offset[1],
            center[1] + sin_angle * source_offset[0] + cos_angle * source_offset[1],
            center[2].expand_as(angles) + source_offset[2],
        ),
        dim=-1,
    )

    forward = F.normalize(center[None] - positions, dim=-1, eps=1.0e-8)
    world_up = source_c2w.new_tensor((0.0, 0.0, 1.0)).expand_as(forward)
    right = torch.linalg.cross(forward, world_up, dim=-1)
    nearly_vertical = right.norm(dim=-1, keepdim=True) < 1.0e-6
    fallback_up = source_c2w.new_tensor((0.0, 1.0, 0.0)).expand_as(forward)
    right = torch.where(nearly_vertical, torch.linalg.cross(forward, fallback_up, dim=-1), right)
    right = F.normalize(right, dim=-1, eps=1.0e-8)
    down = F.normalize(torch.linalg.cross(forward, right, dim=-1), dim=-1, eps=1.0e-8)

    # OpenCV c2w columns are camera right, down, and forward. Construct the
    # inverse analytically to avoid a batched matrix inversion.
    rotation_c2w = torch.stack((right, down, forward), dim=-1)
    rotation_w2c = rotation_c2w.transpose(-1, -2)
    w2c = torch.eye(4, device=source_c2w.device, dtype=torch.float32).expand(frame_count, -1, -1).clone()
    w2c[:, :3, :3] = rotation_w2c
    w2c[:, :3, 3] = -torch.einsum("fij,fj->fi", rotation_w2c, positions)
    return w2c


def _novel_view_orbit_video(
    rec_out: Dict[str, Any],
    background: torch.Tensor,
    splatter_cfg: SplatterConfig,
) -> wandb.Video:
    """Render one source-anchored full orbit in one batched RGB-only call."""
    sample_pc = {key: value[:1] for key, value in rec_out["gaussian_pc"].items()}
    source_c2w = rec_out["source_c2w"][0]
    source_intrinsics = rec_out["source_intrinsics"][0]
    scene_center = source_c2w.new_tensor(METAWORLD_SCENE_CENTER)
    orbit_w2c = _circular_look_at_w2c(
        source_c2w,
        scene_center,
        NOVEL_VIEW_ORBIT_FRAMES,
    )
    orbit_intrinsics = source_intrinsics.view(1, 1, 3, 3).expand(
        1, NOVEL_VIEW_ORBIT_FRAMES, 3, 3
    )
    rendered = render_rgb(
        sample_pc,
        orbit_w2c.unsqueeze(0),
        orbit_intrinsics,
        background,
        splatter_cfg,
    )["render"][0]
    frames_u8 = (
        torch.nan_to_num(rendered, nan=0.0, posinf=1.0, neginf=0.0)
        .clamp(0.0, 1.0)
        .mul(255.0)
        .round()
        .to(torch.uint8)
        .cpu()
        .numpy()
    )
    return wandb.Video(
        frames_u8,
        fps=NOVEL_VIEW_ORBIT_FPS,
        format="gif",
        caption=(
            "Novel-view RGB orbit from the source pose around the Meta-World "
            "scene center [0.0, 0.6, 0.0]"
        ),
    )


def _patch_mask_to_pixels(mask: torch.Tensor, patch_size: int, height: int, width: int) -> torch.Tensor:
    grid_h, grid_w = height // patch_size, width // patch_size
    pixels = mask.view(grid_h, grid_w).repeat_interleave(patch_size, 0).repeat_interleave(patch_size, 1)
    return pixels[:height, :width].float().unsqueeze(0)


def _overlay_patch_mask(
    image: torch.Tensor,
    mask: torch.Tensor,
    patch_size: int,
) -> torch.Tensor:
    height, width = image.shape[-2:]
    alpha = (
        _patch_mask_to_pixels(mask, patch_size, height, width)
        .to(device=image.device, dtype=image.dtype)
        .mul(MASK_OVERLAY_ALPHA)
    )
    color = image.new_tensor(MASK_OVERLAY_COLOR).view(3, 1, 1)
    return image * (1.0 - alpha) + color * alpha


def _flow_to_rgb(
    flow: torch.Tensor,
    magnitude_scale: float = FLOW_VISUALIZATION_MAX_MAGNITUDE_PIXELS,
) -> torch.Tensor:
    x, y = flow.float().unbind(dim=0)
    hue = (torch.atan2(y, x) / (2.0 * torch.pi) + 1.0) % 1.0
    saturation = (
        (x.square() + y.square())
        .sqrt()
        .div(max(float(magnitude_scale), 1.0e-6))
        .clamp(0.0, 1.0)
    )
    value = torch.ones_like(hue)
    sector = torch.floor(hue * 6.0).to(torch.long) % 6
    fraction = hue * 6.0 - torch.floor(hue * 6.0)
    p = value * (1.0 - saturation)
    q = value * (1.0 - fraction * saturation)
    t = value * (1.0 - (1.0 - fraction) * saturation)
    choices = (
        torch.stack((value, t, p)),
        torch.stack((q, value, p)),
        torch.stack((p, value, t)),
        torch.stack((p, q, value)),
        torch.stack((t, p, value)),
        torch.stack((value, p, q)),
    )
    output = torch.zeros_like(choices[0])
    for index, choice in enumerate(choices):
        output = torch.where((sector == index).unsqueeze(0), choice, output)
    return output



def _sample_image_field(
    field: torch.Tensor,
    points_xy: torch.Tensor,
    *,
    mode: str = "bilinear",
) -> torch.Tensor:
    """Sample a ``(C,H,W)`` field at floating-point pixel coordinates."""
    if field.dim() != 3:
        raise ValueError(f"Expected a (C,H,W) field, got {tuple(field.shape)}.")
    if points_xy.dim() != 2 or points_xy.shape[-1] != 2:
        raise ValueError(f"Expected points as (N,2), got {tuple(points_xy.shape)}.")
    if points_xy.shape[0] == 0:
        return field.new_empty((0, field.shape[0]))
    height, width = field.shape[-2:]
    x = points_xy[:, 0]
    y = points_xy[:, 1]
    x_normalized = torch.zeros_like(x) if width == 1 else 2.0 * x / (width - 1) - 1.0
    y_normalized = torch.zeros_like(y) if height == 1 else 2.0 * y / (height - 1) - 1.0
    grid = torch.stack((x_normalized, y_normalized), dim=-1).view(1, -1, 1, 2)
    sampled = F.grid_sample(
        field.unsqueeze(0),
        grid,
        mode=mode,
        padding_mode="zeros",
        align_corners=True,
    )
    return sampled[0, :, :, 0].transpose(0, 1)


def _points_inside_image(points_xy: torch.Tensor, height: int, width: int) -> torch.Tensor:
    return (
        torch.isfinite(points_xy).all(dim=-1)
        & (points_xy[:, 0] >= 0.0)
        & (points_xy[:, 0] <= width - 1)
        & (points_xy[:, 1] >= 0.0)
        & (points_xy[:, 1] <= height - 1)
    )


def _flow_trajectory_candidates(
    rec_out: Dict[str, Any],
    view_idx: int,
    coverage_threshold: float,
    target_count: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build full predicted/reference tracks before uniform spatial sampling."""
    predicted_flow = rec_out["rendered_optical_flows"][0, :, view_idx].float()
    teacher_flow = rec_out["target_optical_flows"][0, :, view_idx].float()
    coverage = rec_out["rendered_flow_coverage"][0, :, view_idx].float()
    rendered_valid = rec_out["rendered_flow_valid_mask"][0, :, view_idx].bool()
    foreground = rec_out["target_masks_self"][0, 0, view_idx, 0].bool()
    height, width = foreground.shape

    minimum_coverage = max(float(coverage_threshold), 1.0e-6)
    initial_valid = (
        foreground
        & rendered_valid[0, 0]
        & (coverage[0, 0] > minimum_coverage)
        & torch.isfinite(predicted_flow[0]).all(dim=0)
        & torch.isfinite(teacher_flow[0]).all(dim=0)
    )
    y, x = torch.where(initial_valid)
    if x.numel() == 0:
        empty_tracks = predicted_flow.new_empty((0, 3, 2))
        return empty_tracks, empty_tracks.clone()
    point0 = torch.stack((x, y), dim=-1).to(dtype=predicted_flow.dtype)

    predicted01 = _sample_image_field(predicted_flow[0], point0)
    teacher01 = _sample_image_field(teacher_flow[0], point0)
    predicted1 = point0 + predicted01
    teacher1 = point0 + teacher01
    predicted12 = _sample_image_field(predicted_flow[1], predicted1)
    teacher12 = _sample_image_field(teacher_flow[1], teacher1)
    predicted2 = predicted1 + predicted12
    teacher2 = teacher1 + teacher12
    coverage12 = _sample_image_field(coverage[1], predicted1)[:, 0]
    valid12 = _sample_image_field(
        rendered_valid[1].to(dtype=predicted_flow.dtype),
        predicted1,
        mode="nearest",
    )[:, 0] > 0.5

    full_valid = (
        _points_inside_image(predicted1, height, width)
        & _points_inside_image(predicted2, height, width)
        & _points_inside_image(teacher1, height, width)
        & _points_inside_image(teacher2, height, width)
        & torch.isfinite(predicted12).all(dim=-1)
        & torch.isfinite(teacher12).all(dim=-1)
        & valid12
        & (coverage12 > minimum_coverage)
    )
    preferred_coverage = max(minimum_coverage, FLOW_TRACK_PREFERRED_COVERAGE)
    coverage01 = _sample_image_field(coverage[0], point0)[:, 0]
    preferred = full_valid & (coverage01 >= preferred_coverage) & (
        coverage12 >= preferred_coverage
    )
    minimum_preferred = min(8, int(target_count))
    selected_valid = (
        preferred
        if int(preferred.sum().item()) >= minimum_preferred
        else full_valid
    )
    predicted_tracks = torch.stack((point0, predicted1, predicted2), dim=1)
    teacher_tracks = torch.stack((point0, teacher1, teacher2), dim=1)
    return predicted_tracks[selected_valid], teacher_tracks[selected_valid]


def _select_track_indices(
    points_xy: torch.Tensor,
    max_points: int,
) -> torch.Tensor:
    """Select image-plane points with greedy farthest-point sampling only."""
    count = int(points_xy.shape[0])
    target = min(max(1, int(max_points)), count)
    if count <= target:
        return torch.arange(count, device=points_xy.device)

    selected = torch.zeros(count, device=points_xy.device, dtype=torch.bool)
    minimum_distance = torch.full(
        (count,), float("inf"), device=points_xy.device, dtype=points_xy.dtype
    )
    centroid = points_xy.mean(dim=0, keepdim=True)
    current = (points_xy - centroid).square().sum(dim=-1).argmin()
    for _ in range(target):
        selected[current] = True
        distance = (points_xy - points_xy[current]).square().sum(dim=-1)
        minimum_distance = torch.minimum(minimum_distance, distance)
        minimum_distance[selected] = -1.0
        current = minimum_distance.argmax()
    return torch.where(selected)[0]


def _paint_pixels(
    image: torch.Tensor,
    x: torch.Tensor,
    y: torch.Tensor,
    color: torch.Tensor,
    radius: int,
) -> None:
    height, width = image.shape[-2:]
    for offset_y in range(-radius, radius + 1):
        for offset_x in range(-radius, radius + 1):
            target_x = x + offset_x
            target_y = y + offset_y
            valid = (
                (target_x >= 0)
                & (target_x < width)
                & (target_y >= 0)
                & (target_y < height)
            )
            image[:, target_y[valid], target_x[valid]] = color[:, None]


def _draw_track_line(
    image: torch.Tensor,
    start_xy: torch.Tensor,
    end_xy: torch.Tensor,
    color: torch.Tensor,
    *,
    dashed: bool,
) -> None:
    delta = end_xy - start_xy
    steps = max(2, int(torch.ceil(delta.abs().amax()).item()) + 1)
    interpolation = torch.linspace(0.0, 1.0, steps=steps)
    points = start_xy[None] + interpolation[:, None] * delta[None]
    if dashed:
        points = points[(torch.arange(steps) % 5) < 3]
    rounded = points.round().to(torch.long)
    _paint_pixels(image, rounded[:, 0], rounded[:, 1], color, radius=1)


def _draw_track_marker(
    image: torch.Tensor,
    point_xy: torch.Tensor,
    color: torch.Tensor,
    *,
    cross: bool,
) -> None:
    center = point_xy.round().to(torch.long)
    if cross:
        offsets = torch.tensor(
            ((-2, 0), (-1, 0), (0, 0), (1, 0), (2, 0), (0, -2), (0, -1), (0, 1), (0, 2))
        )
    else:
        offsets = torch.tensor(
            ((-1, -1), (0, -1), (1, -1), (-1, 0), (0, 0), (1, 0), (-1, 1), (0, 1), (1, 1))
        )
    points = center[None] + offsets
    _paint_pixels(image, points[:, 0], points[:, 1], color, radius=0)


def _trajectory_frames(
    images: torch.Tensor,
    trajectories: torch.Tensor,
    colors: torch.Tensor,
    *,
    dashed: bool,
) -> list[torch.Tensor]:
    panels = []
    for time_idx in range(3):
        panel = images[time_idx].detach().cpu().float().clone()
        for track_idx in range(trajectories.shape[0]):
            trajectory = trajectories[track_idx]
            color = colors[track_idx]
            for segment_idx in range(time_idx):
                _draw_track_line(
                    panel,
                    trajectory[segment_idx],
                    trajectory[segment_idx + 1],
                    color,
                    dashed=dashed,
                )
            for point_idx in range(time_idx + 1):
                _draw_track_marker(
                    panel,
                    trajectory[point_idx],
                    color,
                    cross=dashed,
                )
        panels.append(panel)
    return panels


def _point_trajectory_panel(
    images: torch.Tensor,
    rec_out: Dict[str, Any],
    view_idx: int,
    coverage_threshold: float,
) -> wandb.Image:
    predicted, teacher = _flow_trajectory_candidates(
        rec_out,
        view_idx,
        coverage_threshold,
        FLOW_TRACK_POINT_COUNT,
    )
    indices = _select_track_indices(predicted[:, 0], FLOW_TRACK_POINT_COUNT)
    predicted = predicted[indices].detach().cpu()
    teacher = teacher[indices].detach().cpu()
    colors = torch.tensor(FLOW_TRACK_COLORS, dtype=torch.float32)[: predicted.shape[0]]

    ground_truth_rgb = images[0, :, view_idx]
    rendered_rgb = rec_out["rendered_self"][0, :, view_idx]
    teacher_panels = _trajectory_frames(
        ground_truth_rgb, teacher, colors, dashed=True
    )
    predicted_panels = _trajectory_frames(
        rendered_rgb, predicted, colors, dashed=False
    )
    return _wandb_image_grid(
        teacher_panels + predicted_panels,
        nrow=3,
        caption=(
            f"flow point tracks ({predicted.shape[0]} uniformly distributed points) | "
            "columns: t0/t1/t2 | top: ground-truth RGB with SEA-RAFT dashed "
            "tracks/cross markers | bottom: rendered RGB with predicted solid "
            "tracks/square markers | colors and t0 query points match across rows"
        ),
    )


def _depth_rgb(depth: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    valid = mask.bool() & torch.isfinite(depth) & (depth > 0)
    if not bool(valid.any()):
        return torch.zeros(3, *depth.shape[-2:], device=depth.device)
    values = depth[valid]
    low = torch.quantile(values, 0.02)
    high = torch.quantile(values, 0.98)
    normalized = ((depth - low) / (high - low).clamp_min(1.0e-6)).clamp(0.0, 1.0)
    normalized = torch.where(valid, normalized, torch.zeros_like(normalized))
    return normalized.expand(3, -1, -1)


def _make_grid(images: list[torch.Tensor], nrow: int, padding: int = 2) -> torch.Tensor:
    if not images:
        raise ValueError("Cannot create an image grid from an empty list.")
    columns = max(1, min(int(nrow), len(images)))
    rows = (len(images) + columns - 1) // columns
    channels, height, width = images[0].shape
    grid = images[0].new_zeros(
        channels,
        rows * height + max(0, rows - 1) * padding,
        columns * width + max(0, columns - 1) * padding,
    )
    for index, image in enumerate(images):
        if image.shape != (channels, height, width):
            raise ValueError("All qualitative panel images must share shape.")
        row, column = divmod(index, columns)
        y = row * (height + padding); x = column * (width + padding)
        grid[:, y:y + height, x:x + width] = image
    return grid


def _wandb_image_grid(
    images: list[torch.Tensor],
    *,
    nrow: int,
    caption: str,
) -> wandb.Image:
    """Create a finite [0, 1] grid before W&B infers the image value range."""
    safe_images = [
        torch.nan_to_num(image.detach().cpu().float(), nan=0.0, posinf=1.0, neginf=0.0)
        .clamp(0.0, 1.0)
        for image in images
    ]
    grid = _make_grid(safe_images, nrow=nrow, padding=2)
    return wandb.Image(grid, caption=caption)


def _input_panel(
    images: torch.Tensor,
    flows: torch.Tensor,
    inv_mask: torch.Tensor,
    dep_mask: torch.Tensor,
    patch_size: int,
    view_idx: int,
) -> wandb.Image:
    if not 0 <= view_idx < images.shape[2]:
        raise ValueError(f"View index {view_idx} is outside [0, {images.shape[2]}).")
    rgb = [images[0, time_idx, view_idx].detach().cpu() for time_idx in range(3)]
    colored_flows = [_flow_to_rgb(flows[0, pair_idx, view_idx]).cpu() for pair_idx in range(3)]
    invariant_overlays = [
        _overlay_patch_mask(image, inv_mask[0, view_idx].cpu(), patch_size)
        for image in rgb
    ]
    dependent_overlay = _overlay_patch_mask(
        rgb[0], dep_mask[0, view_idx].cpu(), patch_size
    )
    placeholder = torch.ones_like(rgb[0])
    items = (
        rgb
        + colored_flows
        + invariant_overlays
        + [dependent_overlay, placeholder, placeholder]
    )
    return _wandb_image_grid(
        list(items),
        nrow=3,
        caption=(
            f"source view {view_idx} | RGB t0/t1/t2 | SEA-RAFT color flow 01/12/02 "
            f"(hue=direction, saturation=0-{FLOW_VISUALIZATION_MAX_MAGNITUDE_PIXELS:g} px, "
            "white=zero) | invariant mask overlays t0/t1/t2 | dependent mask overlay t0 "
            "(red=masked)"
        ),
    )


def _reconstruction_panel(rec_out: Dict[str, Any], view_idx: int) -> wandb.Image:
    items = []
    targets = rec_out["target_images_self"][0, :, view_idx]
    predicted = rec_out["rendered_self"][0, :, view_idx]
    target_masks = rec_out["target_masks_self"][0, :, view_idx]
    predicted_alpha = rec_out["rendered_alpha_self"][0, :, view_idx]
    predicted_depth = rec_out["rendered_expected_depth_self"][0, :, view_idx]
    target_depths = rec_out["target_depths_self"]
    items.extend(image.detach().cpu() for image in targets)
    items.extend(image.detach().cpu() for image in predicted)
    for time_idx in range(3):
        target_depth = target_depths[0, time_idx, view_idx]
        items.append(_depth_rgb(target_depth, target_masks[time_idx]).detach().cpu())
    for time_idx in range(3):
        items.append(_depth_rgb(predicted_depth[time_idx], target_masks[time_idx]).detach().cpu())
    items.extend(mask.expand(3, -1, -1).detach().cpu() for mask in target_masks)
    items.extend(alpha.expand(3, -1, -1).detach().cpu() for alpha in predicted_alpha)
    return _wandb_image_grid(
        items,
        nrow=3,
        caption="target/predicted RGB, depth, mask/alpha",
    )


def _reference_points(
    depth: torch.Tensor,
    mask: torch.Tensor,
    intrinsics: torch.Tensor,
    c2w: torch.Tensor,
) -> torch.Tensor:
    height, width = depth.shape[-2:]
    ys, xs = torch.meshgrid(
        torch.arange(height, device=depth.device, dtype=depth.dtype) + 0.5,
        torch.arange(width, device=depth.device, dtype=depth.dtype) + 0.5,
        indexing="ij",
    )
    z = depth[0]
    x = (xs - intrinsics[0, 2]) / intrinsics[0, 0].clamp_min(1.0e-6) * z
    y = (ys - intrinsics[1, 2]) / intrinsics[1, 1].clamp_min(1.0e-6) * z
    camera = torch.stack((x, y, z), dim=-1)
    valid = mask[0].bool() & torch.isfinite(camera).all(-1) & (z > 0)
    camera = camera[valid]
    return torch.einsum("ij,nj->ni", c2w[:3, :3], camera) + c2w[:3, 3]


def _evenly_sample_points(points: torch.Tensor, max_points: int) -> torch.Tensor:
    if points.shape[0] <= max_points:
        return points
    indices = torch.linspace(
        0,
        points.shape[0] - 1,
        steps=max_points,
        device=points.device,
        dtype=torch.float64,
    ).round().to(torch.long)
    return points[indices]


def _camera_matrix_at(
    camera: torch.Tensor,
    time_idx: int,
    view_idx: int,
) -> torch.Tensor:
    sample = camera[0]
    if sample.dim() == 3:
        return sample[view_idx]
    if sample.dim() == 4:
        return sample[time_idx, view_idx]
    raise ValueError(f"Expected batched camera matrices with optional time, got {camera.shape}.")


def _combined_pointcloud_payload(
    rec_out: Dict[str, Any],
    depths: torch.Tensor,
    masks: torch.Tensor,
    intrinsics: torch.Tensor,
    c2w: torch.Tensor,
    max_points: int,
    prefix: str,
) -> Dict[str, Any]:
    payload: Dict[str, Any] = {}
    num_views = int(depths.shape[2])
    per_view_max_points = max(1, int(max_points) // max(1, num_views))
    for time_idx, pc in enumerate(rec_out["gaussian_pc_sequence"][:3]):
        valid = pc["valid_mask"][0]
        predicted = _evenly_sample_points(pc["xyz"][0, valid], int(max_points))
        references = [
            _evenly_sample_points(
                _reference_points(
                    depths[0, time_idx, view_idx],
                    masks[0, time_idx, view_idx],
                    _camera_matrix_at(intrinsics, time_idx, view_idx),
                    _camera_matrix_at(c2w, time_idx, view_idx),
                ),
                per_view_max_points,
            )
            for view_idx in range(num_views)
        ]
        reference = torch.cat(references, dim=0)
        pred_rgb = torch.tensor([255.0, 64.0, 64.0], device=predicted.device).expand(predicted.shape[0], -1)
        ref_rgb = torch.tensor([64.0, 192.0, 255.0], device=reference.device).expand(reference.shape[0], -1)
        combined = torch.cat(
            (torch.cat((predicted, pred_rgb), dim=-1), torch.cat((reference, ref_rgb), dim=-1)),
            dim=0,
        )
        payload[f"{prefix}/pointcloud_overlay_t{time_idx}"] = wandb.Object3D(
            combined.cpu().numpy()
        )
    return payload


def _move_batch(batch: Dict[str, Any], device: torch.device) -> Dict[str, Any]:
    images_u8 = batch["images"].to(device, non_blocking=True)
    images_01 = images_u8.float().div_(255.0)
    return {
        "images": images_01.mul(2.0).sub(1.0),
        "optical_flows": batch["optical_flows"].to(
            device=device, dtype=torch.float32, non_blocking=True
        ),
        "depths": batch["depths"].to(device=device, dtype=torch.float32, non_blocking=True),
        "masks": batch["masks"].to(device=device, non_blocking=True),
        "K": batch["K"].to(device=device, dtype=torch.float32, non_blocking=True),
        "c2w": batch["c2w"].to(device=device, dtype=torch.float32, non_blocking=True),
        "w2c": batch["w2c"].to(device=device, dtype=torch.float32, non_blocking=True),
    }


@torch.no_grad()
def _evaluate_batch(
    batch: Dict[str, Any],
    vae: SplatterVAE,
    converter: DirectSplatterToGaussians,
    splatter_cfg: SplatterConfig,
    cfg_train: TrainConfig,
    background: torch.Tensor,
    device: torch.device,
    global_step: int,
    return_renders: bool,
) -> tuple[Dict[str, torch.Tensor], Dict[str, Any], Dict[str, torch.Tensor], Dict[str, Any]]:
    moved = _move_batch(batch, device)
    images = moved["images"]
    flows = moved["optical_flows"]
    autocast_context = (
        torch.autocast(device_type="cuda", dtype=torch.bfloat16)
        if device.type == "cuda"
        else nullcontext()
    )
    with autocast_context:
        view_latents = encode_per_view_sequence_batch(vae, images, flows)
        batch_size = images.shape[0]
        source_indices = torch.randint(images.shape[2], (batch_size,), device=device)
        batch_ids = torch.arange(batch_size, device=device)
        raw_outputs = vae.predict_raw_maps(
            view_latents["s_inv_by_view"][batch_ids, source_indices],
            view_latents["z_dep_by_view"][batch_ids, source_indices],
        )
    view_latents["s_inv_by_view"] = view_latents["s_inv_by_view"].float()
    view_latents["z_dep_by_view"] = view_latents["z_dep_by_view"].float()
    ramp = temporal_loss_ramp(global_step, cfg_train.temporal_loss_ramp_steps)
    rec_out = compute_reconstruction_and_renders(
        splatter_to_gaussians=converter,
        splatter_cfg=splatter_cfg,
        raw_base_map=raw_outputs["raw_base_map"].float(),
        raw_motion_map=raw_outputs["raw_motion_map"].float(),
        motion_translation_max=float(vae.motion_translation_max),
        images_01=(images + 1.0) * 0.5,
        optical_flows=flows,
        depths=moved["depths"],
        masks=moved["masks"],
        intrinsics=moved["K"],
        c2w=moved["c2w"],
        w2c=moved["w2c"],
        bg=background,
        cfg_train=cfg_train,
        source_indices=source_indices,
        temporal_ramp=ramp,
        training=False,
        return_renders=return_renders,
        compute_diagnostics=True,
    )
    inv_con, dep_con, inv_cons, dep_cons = compute_view_structured_representation_losses(
        view_latents["s_inv_by_view"], view_latents["z_dep_by_view"], cfg_train.temperature
    )
    representation_loss = (
        cfg_train.inv_contrastive_weight * inv_con
        + cfg_train.inv_consistency_weight * inv_cons
        + cfg_train.dep_contrastive_weight * dep_con
        + cfg_train.dep_consistency_weight * dep_cons
    )
    render_loss = rec_out["render_loss_t0"] + ramp * 0.5 * (
        rec_out["render_loss_t1"] + rec_out["render_loss_t2"]
    )
    total = (
        render_loss
        + ramp * cfg_train.flow_weight * rec_out["flow_loss"]
        + representation_loss
        + cfg_train.frustum_weight * rec_out["frustum_loss"]
    )
    metrics = {
        "val/core/total_loss": total,
        "val/core/render_loss": render_loss,
        "val/core/flow_loss": rec_out["flow_loss"],
        "val/core/representation_loss": representation_loss,
        "val/render/t0": rec_out["render_loss_t0"],
        "val/render/t1": rec_out["render_loss_t1"],
        "val/render/t2": rec_out["render_loss_t2"],
        "val/flow/epe_01": rec_out["flow_epe_01"],
        "val/flow/epe_12": rec_out["flow_epe_12"],
        "val/flow/epe_02": rec_out["flow_epe_02"],
        "val/flow/visible_fraction": rec_out["flow_visible_fraction"],
        "val/motion/translation_01_mean": rec_out["translation_01_mean"],
        "val/motion/translation_12_mean": rec_out["translation_12_mean"],
        "val/components/rgb": rec_out["rgb_loss"],
        "val/components/silhouette": rec_out["silhouette_loss"],
        "val/components/global_depth": rec_out["global_depth_loss"],
        "val/components/local_depth": rec_out["local_depth_loss"],
        "val/components/frustum": rec_out["frustum_loss"],
        "val/gaussian/mean_opacity": rec_out["mean_valid_gaussian_opacity"],
    }
    return metrics, rec_out, view_latents, moved


@torch.no_grad()
def validate_and_log_wandb(
    vae: SplatterVAE,
    splatter_cfg: SplatterConfig,
    splatter_to_gaussians: DirectSplatterToGaussians,
    valid_dataloader: DataLoader,
    device: torch.device,
    bg: torch.Tensor,
    cfg_train: TrainConfig,
    global_step: int,
) -> None:
    if wandb.run is None:
        return
    previous_mode = vae.training
    vae.eval()
    sums: Dict[str, float] = {}
    count = 0
    first_result = None
    max_batches = max(1, int(cfg_train.val_num_batches))
    for batch_index, batch in enumerate(islice(valid_dataloader, max_batches)):
        result = _evaluate_batch(
            batch,
            vae,
            splatter_to_gaussians,
            splatter_cfg,
            cfg_train,
            bg,
            device,
            global_step,
            return_renders=(batch_index == 0),
        )
        metrics, _rec_out, _latents, _moved = result
        for key, value in metrics.items():
            sums[key] = sums.get(key, 0.0) + float(value.item())
        if batch_index == 0:
            first_result = result
        count += 1
    if count == 0 or first_result is None:
        vae.train(previous_mode)
        return

    ramp = temporal_loss_ramp(global_step, cfg_train.temporal_loss_ramp_steps)
    log_values: Dict[str, Any] = {key: value / count for key, value in sums.items()}
    log_values.update({"global_step": global_step, "val/temporal_loss_ramp": ramp})
    _metrics, rec_out, latents, moved = first_result
    prefix = "val/qualitative/random_subset"
    source_view_idx = int(rec_out["source_indices"][0].item())
    validation_images = (moved["images"] + 1.0) * 0.5
    log_values.update({
        f"{prefix}/input_and_masks": _input_panel(
            validation_images,
            moved["optical_flows"],
            latents["inv_mask_by_view"],
            latents["dep_mask_by_view"],
            vae.patch_h,
            source_view_idx,
        ),
        f"{prefix}/reconstruction": _reconstruction_panel(rec_out, source_view_idx),
        f"{prefix}/flow_point_trajectories": _point_trajectory_panel(
            validation_images,
            rec_out,
            source_view_idx,
            float(cfg_train.flow_alpha_threshold),
        ),
        f"{prefix}/novel_view_orbit": _novel_view_orbit_video(
            rec_out,
            bg,
            splatter_cfg,
        ),
    })
    log_values.update(
        _combined_pointcloud_payload(
            rec_out,
            moved["depths"],
            moved["masks"],
            moved["K"],
            moved["c2w"],
            int(cfg_train.val_pointcloud_max_points),
            prefix,
        )
    )
    wandb.log(log_values, step=global_step)
    vae.train(previous_mode)
