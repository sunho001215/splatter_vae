from __future__ import annotations

from typing import Any, Dict, Optional, Sequence

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.utils import make_grid

import wandb

from models.losses import (
    compute_dependent_view_consistency_loss,
    compute_state_consistency_loss,
    compute_view_structured_contrastive_losses,
)
from models.splatter import SplatterConfig, render_predicted
from models.gaussians import DirectSplatterToGaussians
from models.pointcloud_utils import depths_to_world_point_cloud, sample_points
from models.reconstruction import (
    compute_reconstruction_and_renders,
    encode_per_view_sequence_batch,
)
from models.train_config import TrainConfig
from models.vae import SplatterVAE
from utils.camera_tensor_utils import trajectory_w2c


def make_wandb_named_image_panel(named_images: list[tuple[str, torch.Tensor]], max_vis: int) -> wandb.Image:
    max_vis = max(1, int(max_vis))
    rows = []
    names = []
    for name, images in named_images:
        rows.append(images[:max_vis].detach().cpu().clamp(0.0, 1.0))
        names.append(name)
    grid = make_grid(torch.cat(rows, dim=0), nrow=max_vis, padding=2)
    return wandb.Image(grid, caption=" | ".join(names))


def make_wandb_input_image_panel(images_01: torch.Tensor, max_vis: int) -> wandb.Image:
    if images_01.dim() == 5:
        images_01 = images_01[:, None]
    num_times_to_show = min(images_01.shape[1], 3)
    num_views_to_show = min(images_01.shape[2], 3)
    panel_items: list[tuple[str, torch.Tensor]] = []
    for time_idx in range(num_times_to_show):
        for view_idx in range(num_views_to_show):
            panel_items.append((f"input_t{time_idx}_cam{view_idx}", images_01[:, time_idx, view_idx]))
    return make_wandb_named_image_panel(panel_items, max_vis=max_vis)


def render_validation_trajectory_videos(
    pc: Dict[str, torch.Tensor],
    source_c2w: torch.Tensor,
    source_intrinsics: torch.Tensor,
    splatter_cfg: SplatterConfig,
    bg: torch.Tensor,
    cfg_train: TrainConfig,
) -> Dict[str, wandb.Video]:
    if not bool(cfg_train.val_render_trajectory_videos):
        return {}
    if pc["xyz"].device.type != "cuda":
        return {}

    first_pc = {k: v[:1] for k, v in pc.items() if torch.is_tensor(v) and v.shape[0] == pc["xyz"].shape[0]}
    videos: Dict[str, wandb.Video] = {}
    for trajectory in ("lateral", "circular"):
        w2c = trajectory_w2c(
            base_c2w=source_c2w[0],
            trajectory=trajectory,
            num_frames=int(cfg_train.val_video_frames),
            amplitude=float(cfg_train.val_video_amplitude),
            focus_distance=float(cfg_train.val_video_focus_distance),
        ).unsqueeze(0)
        intrinsics = source_intrinsics[0].view(1, 1, 3, 3).expand(1, w2c.shape[1], 3, 3)
        render = render_predicted(
            pc=first_pc,
            world_view_transform=w2c,
            intrinsics=intrinsics,
            bg_color=bg,
            cfg=splatter_cfg,
            render_mode="RGB+ED",
        )["render"][0]
        video = (render.detach().cpu().clamp(0.0, 1.0).numpy() * 255.0).astype("uint8")
        videos[f"val/video_{trajectory}"] = wandb.Video(video, fps=int(cfg_train.val_video_fps), format="mp4")
    return videos


def sample_object3d_array(
    points: torch.Tensor,
    mask: torch.Tensor,
    max_points: int,
    rgb: tuple[float, float, float],
) -> Optional[torch.Tensor]:
    selected, selected_mask = sample_points(points, mask, max(1, int(max_points)))
    if not bool(selected_mask.any()):
        return None
    colors = selected.new_tensor(rgb).view(1, 3).expand(selected.shape[0], 3)
    return torch.cat([selected, colors], dim=-1).detach().cpu().float()


def _colored_object3d_array(points: torch.Tensor, rgb: tuple[float, float, float]) -> Optional[torch.Tensor]:
    points = points.detach()
    points = points[torch.isfinite(points).all(dim=-1)]
    if points.numel() == 0:
        return None
    colors = points.new_tensor(rgb).view(1, 3).expand(points.shape[0], 3)
    return torch.cat([points, colors], dim=-1).cpu().float()


def _control_trajectory_object3d(
    control_xyz_sequence: torch.Tensor,
    control_valid_mask: torch.Tensor,
) -> Optional[torch.Tensor]:
    valid = control_valid_mask.detach().to(dtype=torch.bool)
    sequence = control_xyz_sequence.detach()[:3, valid]
    if sequence.numel() == 0:
        return None

    time_colors = (
        (255.0, 60.0, 60.0),
        (255.0, 230.0, 40.0),
        (40.0, 235.0, 100.0),
    )
    segment_colors = ((20.0, 210.0, 255.0), (255.0, 140.0, 20.0))
    items: list[torch.Tensor] = []
    for time_idx in range(sequence.shape[0]):
        colored = _colored_object3d_array(sequence[time_idx], time_colors[min(time_idx, 2)])
        if colored is not None:
            items.append(colored)
    for time_idx in range(sequence.shape[0] - 1):
        start = sequence[time_idx]
        end = sequence[time_idx + 1]
        for step in range(1, 5):
            fraction = float(step) / 5.0
            points = start.lerp(end, fraction)
            colored = _colored_object3d_array(points, segment_colors[min(time_idx, 1)])
            if colored is not None:
                items.append(colored)
    return torch.cat(items, dim=0) if items else None


def make_wandb_pointcloud_payload(
    rec_out: Dict[str, Any],
    depths: Optional[torch.Tensor],
    intrinsics: torch.Tensor,
    c2w: torch.Tensor,
    splatter_cfg: SplatterConfig,
    cfg_train: TrainConfig,
    masks: Optional[torch.Tensor],
) -> Dict[str, wandb.Object3D]:
    del splatter_cfg  # Back-projection uses the target depth's own positive/finite validity.
    if not bool(cfg_train.val_log_pointclouds):
        return {}
    max_points = max(1, int(cfg_train.val_pointcloud_max_points))
    pc = rec_out.get("gaussian_pc", None)
    if pc is None:
        return {}
    pc_sequence = rec_out.get("gaussian_pc_sequence", [pc])
    num_times = min(len(pc_sequence), 3)
    payload: Dict[str, wandb.Object3D] = {}

    control_sequence = rec_out.get("control_xyz_sequence", None)
    control_valid = rec_out.get("control_valid_mask", None)
    for time_idx in range(num_times):
        pc_t = pc_sequence[time_idx]
        pred_mask_t = pc_t.get(
            "valid_mask",
            torch.ones(pc_t["xyz"].shape[:2], device=pc_t["xyz"].device, dtype=torch.bool),
        )
        pred_t = sample_object3d_array(
            points=pc_t["xyz"][0],
            mask=pred_mask_t[0],
            max_points=max_points,
            rgb=(255.0, 80.0, 80.0),
        )
        if pred_t is not None:
            payload[f"val/pointcloud_pred/t{time_idx}"] = wandb.Object3D(
                pred_t.numpy(),
                caption=f"t{time_idx}: valid predicted dense Gaussian centers, sampled to <= {max_points}",
            )

        if time_idx == 0 and pred_t is not None and control_sequence is not None and control_valid is not None:
            controls_t0 = _colored_object3d_array(
                control_sequence[0, 0][control_valid[0].to(dtype=torch.bool)],
                (40.0, 210.0, 255.0),
            )
            if controls_t0 is not None:
                payload["val/pointcloud_controls_and_dense/t0"] = wandb.Object3D(
                    torch.cat([pred_t, controls_t0], dim=0).numpy(),
                    caption="t0: dense Gaussian centers red; persistent sparse motion controls cyan",
                )

        if depths is None or masks is None:
            continue
        depth_time_count = depths.shape[1] if depths.dim() == 6 else 1
        if time_idx >= depth_time_count:
            continue
        depths_t = depths[:1, time_idx] if depths.dim() == 6 else depths[:1]
        intrinsics_t = intrinsics[:1, time_idx] if intrinsics.dim() == 5 else intrinsics[:1]
        c2w_t = c2w[:1, time_idx] if c2w.dim() == 5 else c2w[:1]
        masks_t = None if masks is None else (masks[:1, time_idx] if masks.dim() == 6 else masks[:1])
        points_per_view = max(1, max_points // max(1, depths_t.shape[1]))
        reference_points, reference_mask = depths_to_world_point_cloud(
            depths=depths_t,
            intrinsics=intrinsics_t,
            c2w=c2w_t,
            masks=masks_t,
            max_points_per_view=points_per_view,
        )
        reference_t = sample_object3d_array(
            points=reference_points[0],
            mask=reference_mask[0],
            max_points=max_points,
            rgb=(80.0, 220.0, 120.0),
        )
        if reference_t is not None:
            payload[f"val/pointcloud_reference/t{time_idx}"] = wandb.Object3D(
                reference_t.numpy(),
                caption=(
                    f"t{time_idx}: exact-mask foreground depth back-projected from all target cameras; "
                    "pseudo-depth reference geometry may be scale-ambiguous"
                ),
            )
        if pred_t is not None and reference_t is not None:
            payload[f"val/pointcloud_overlay/t{time_idx}"] = wandb.Object3D(
                torch.cat([pred_t, reference_t], dim=0).numpy(),
                caption=f"t{time_idx}: predicted dense centers red; multi-camera depth reference green",
            )

    if control_sequence is not None and control_valid is not None:
        trajectories = _control_trajectory_object3d(control_sequence[0], control_valid[0])
        if trajectories is not None:
            payload["val/pointcloud_control_trajectories"] = wandb.Object3D(
                trajectories.numpy(),
                caption=(
                    "Persistent sparse control trajectories: t0 red, t1 yellow, t2 green; "
                    "t0->t1 cyan and t1->t2 orange"
                ),
            )
    return payload


def _farthest_point_indices(points: torch.Tensor, mask: torch.Tensor, max_points: int) -> torch.Tensor:
    valid_idx = torch.nonzero(mask.to(dtype=torch.bool), as_tuple=False).flatten()
    if valid_idx.numel() == 0:
        return valid_idx
    if valid_idx.numel() <= max_points:
        return valid_idx
    pts = points[valid_idx].float()
    first = torch.argmax((pts - pts.mean(dim=0, keepdim=True)).pow(2).sum(dim=-1))
    selected = [first]
    min_dist = (pts - pts[first]).pow(2).sum(dim=-1)
    for _ in range(1, int(max_points)):
        next_idx = torch.argmax(min_dist)
        selected.append(next_idx)
        min_dist = torch.minimum(min_dist, (pts - pts[next_idx]).pow(2).sum(dim=-1))
    return valid_idx[torch.stack(selected)]


def _camera_at_time(camera: torch.Tensor, batch_idx: int, time_idx: int, view_idx: int) -> torch.Tensor:
    if camera.dim() == 4:
        return camera[batch_idx, view_idx]
    if camera.dim() == 5:
        return camera[batch_idx, time_idx, view_idx]
    raise ValueError(f"Expected camera tensor as (B,A,...) or (B,T,A,...), got {tuple(camera.shape)}.")


def _project_world_points(
    points: torch.Tensor,
    w2c: torch.Tensor,
    intrinsics: torch.Tensor,
    height: int,
    width: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    rot = w2c[:3, :3].to(device=points.device, dtype=points.dtype)
    trans = w2c[:3, 3].to(device=points.device, dtype=points.dtype)
    k = intrinsics.to(device=points.device, dtype=points.dtype)
    cam = points @ rot.transpose(0, 1) + trans.view(1, 3)
    z = cam[:, 2].clamp_min(1.0e-6)
    xy = torch.stack(
        [
            k[0, 0] * cam[:, 0] / z + k[0, 2],
            k[1, 1] * cam[:, 1] / z + k[1, 2],
        ],
        dim=-1,
    )
    visible = (
        torch.isfinite(xy).all(dim=-1)
        & torch.isfinite(cam).all(dim=-1)
        & (cam[:, 2] > 1.0e-6)
        & (xy[:, 0] >= 0.0)
        & (xy[:, 0] <= float(width - 1))
        & (xy[:, 1] >= 0.0)
        & (xy[:, 1] <= float(height - 1))
    )
    return xy, visible


def _select_track_indices(
    xys: Sequence[torch.Tensor],
    visible: Sequence[torch.Tensor],
    valid_mask: torch.Tensor,
    max_tracks: int,
) -> torch.Tensor:
    common_visible = valid_mask.to(dtype=torch.bool).clone()
    for is_visible in visible:
        common_visible &= is_visible.to(dtype=torch.bool)
    if not bool(common_visible.any()):
        return torch.empty(0, device=valid_mask.device, dtype=torch.long)

    candidate_idx = torch.nonzero(common_visible, as_tuple=False).flatten()
    if candidate_idx.numel() > max_tracks:
        motion = torch.zeros((xys[0].shape[0],), device=xys[0].device, dtype=xys[0].dtype)
        for time_idx in range(1, len(xys)):
            motion = motion + (xys[time_idx] - xys[time_idx - 1]).norm(dim=-1)
        top_count = min(candidate_idx.numel(), max_tracks * 6)
        candidate_idx = candidate_idx[torch.topk(motion[candidate_idx], k=top_count).indices]

    subset_mask = torch.zeros_like(valid_mask, dtype=torch.bool)
    subset_mask[candidate_idx] = True
    return _farthest_point_indices(xys[0], subset_mask, max_tracks)


def _draw_disk(canvas: torch.Tensor, xy: torch.Tensor, color: torch.Tensor, radius: int = 2) -> None:
    height, width = canvas.shape[-2:]
    cx = int(round(float(xy[0])))
    cy = int(round(float(xy[1])))
    for yy in range(cy - radius, cy + radius + 1):
        if yy < 0 or yy >= height:
            continue
        for xx in range(cx - radius, cx + radius + 1):
            if xx < 0 or xx >= width:
                continue
            if (xx - cx) * (xx - cx) + (yy - cy) * (yy - cy) <= radius * radius:
                canvas[:, yy, xx] = color


def _draw_line(canvas: torch.Tensor, xy0: torch.Tensor, xy1: torch.Tensor, color: torch.Tensor, width: int = 1) -> None:
    dx = float(xy1[0] - xy0[0])
    dy = float(xy1[1] - xy0[1])
    steps = max(1, int(round(max(abs(dx), abs(dy)))))
    for step in range(steps + 1):
        alpha = float(step) / float(steps)
        xy = xy0 * (1.0 - alpha) + xy1 * alpha
        _draw_disk(canvas, xy, color, radius=width)


def _draw_endpoint_marker(canvas: torch.Tensor, xy: torch.Tensor, color: torch.Tensor) -> None:
    outline = color.new_tensor([0.0, 0.0, 0.0])
    halo = color.new_tensor([1.0, 1.0, 1.0])
    _draw_disk(canvas, xy, outline, radius=5)
    _draw_disk(canvas, xy, halo, radius=4)
    _draw_disk(canvas, xy, color, radius=3)


_TRACK_COLORS = (
    (0.95, 0.15, 0.15),
    (0.10, 0.75, 1.00),
    (0.15, 0.90, 0.30),
    (1.00, 0.65, 0.05),
    (0.75, 0.25, 1.00),
    (1.00, 0.20, 0.70),
    (0.20, 0.95, 0.85),
    (0.65, 0.85, 0.10),
    (0.35, 0.45, 1.00),
    (0.95, 0.45, 0.25),
)
_FLOW_ALPHA_THRESHOLD = 1.0e-2


def _make_temporal_canvas(frames: torch.Tensor, pad: int = 6) -> torch.Tensor:
    num_times, _, height, width = frames.shape
    canvas = torch.ones((3, height, num_times * width + (num_times - 1) * pad), dtype=torch.float32)
    for time_idx in range(num_times):
        x0 = time_idx * (width + pad)
        canvas[:, :, x0 : x0 + width] = frames[time_idx].detach().cpu().clamp(0.0, 1.0)
    return canvas


def _draw_trajectory_set(
    frames: torch.Tensor,
    points_by_time: Sequence[torch.Tensor],
    valid_by_time: Sequence[torch.Tensor],
) -> torch.Tensor:
    canvas = _make_temporal_canvas(frames)
    num_times = len(points_by_time)
    width = frames.shape[-1]
    pad = 6
    num_tracks = points_by_time[0].shape[0]
    for track_idx in range(num_tracks):
        color = torch.tensor(_TRACK_COLORS[track_idx % len(_TRACK_COLORS)], dtype=torch.float32)
        panel_points: list[torch.Tensor] = []
        for time_idx in range(num_times):
            point = points_by_time[time_idx][track_idx].detach().cpu().clone()
            point[0] += time_idx * (width + pad)
            panel_points.append(point)
        for time_idx in range(num_times - 1):
            if bool(valid_by_time[time_idx][track_idx]) and bool(valid_by_time[time_idx + 1][track_idx]):
                _draw_line(canvas, panel_points[time_idx], panel_points[time_idx + 1], color, width=2)
        for time_idx in range(num_times):
            if bool(valid_by_time[time_idx][track_idx]):
                _draw_endpoint_marker(canvas, panel_points[time_idx], color)
    return canvas.clamp(0.0, 1.0)


def _make_track_comparison_image(
    gt_frames: Optional[torch.Tensor],
    rendered_frames: torch.Tensor,
    points_by_time: Sequence[torch.Tensor],
    valid_by_time: Sequence[torch.Tensor],
    caption: str,
) -> wandb.Image:
    rows: list[torch.Tensor] = []
    if gt_frames is not None:
        rows.append(_draw_trajectory_set(gt_frames, points_by_time, valid_by_time))
    rows.append(_draw_trajectory_set(rendered_frames, points_by_time, valid_by_time))
    grid = make_grid(torch.stack(rows, dim=0), nrow=1, padding=8, pad_value=1.0)
    row_description = "Top row: ground truth; bottom row: rendered." if gt_frames is not None else "Rendered row."
    return wandb.Image(grid, caption=f"{caption} {row_description}")


def make_wandb_sparse_control_trajectory_payload(
    rec_out: Dict[str, Any],
    intrinsics: torch.Tensor,
    w2c: torch.Tensor,
    cfg_train: TrainConfig,
) -> Dict[str, wandb.Image]:
    control_sequence = rec_out.get("control_xyz_sequence", None)
    control_valid = rec_out.get("control_valid_mask", None)
    rendered_images = rec_out.get("rendered_self", None)
    gt_images = rec_out.get("target_images_self", None)
    if control_sequence is None or control_valid is None or rendered_images is None:
        return {}

    batch_idx = 0
    num_times = min(control_sequence.shape[1], rendered_images.shape[1], 3)
    if num_times < 2:
        return {}
    num_views = rendered_images.shape[2]
    source_indices = rec_out.get("source_indices", None)
    view_idx = int(source_indices[batch_idx].item()) if source_indices is not None else 0
    view_idx = max(0, min(view_idx, num_views - 1))
    rendered_frames = rendered_images[batch_idx, :num_times, view_idx].detach().cpu().clamp(0.0, 1.0)
    gt_frames = None
    if gt_images is not None:
        gt_frames = gt_images[batch_idx, :num_times, view_idx].detach().cpu().clamp(0.0, 1.0)
    _, height, width = rendered_frames[0].shape

    xys: list[torch.Tensor] = []
    visible: list[torch.Tensor] = []
    for time_idx in range(num_times):
        xy_t, visible_t = _project_world_points(
            points=control_sequence[batch_idx, time_idx],
            w2c=_camera_at_time(w2c, batch_idx, time_idx, view_idx),
            intrinsics=_camera_at_time(intrinsics, batch_idx, time_idx, view_idx),
            height=height,
            width=width,
        )
        xys.append(xy_t.detach().cpu())
        visible.append(visible_t.detach().cpu())

    max_tracks = max(1, min(int(getattr(cfg_train, "val_track_max_points", 8)), 10))
    indices = _select_track_indices(
        xys=xys,
        visible=visible,
        valid_mask=control_valid[batch_idx].detach().cpu(),
        max_tracks=max_tracks,
    )
    if indices.numel() == 0:
        return {}
    selected_points = [xy[indices] for xy in xys]
    selected_visible = [is_visible[indices] for is_visible in visible]
    image = _make_track_comparison_image(
        gt_frames=gt_frames,
        rendered_frames=rendered_frames,
        points_by_time=selected_points,
        valid_by_time=selected_visible,
        caption=(
            "Sparse control trajectories — internal motion representation. "
            f"Persistent control IDs: {indices.tolist()}."
        ),
    )
    return {"val/sparse_control_trajectories": image}


def _render_alpha_weighted_transition_flow(
    pc_t: Dict[str, torch.Tensor],
    pc_next: Dict[str, torch.Tensor],
    w2c_t: torch.Tensor,
    intrinsics_t: torch.Tensor,
    w2c_next: torch.Tensor,
    intrinsics_next: torch.Tensor,
    height: int,
    width: int,
    splatter_cfg: SplatterConfig,
) -> tuple[torch.Tensor, torch.Tensor]:
    xyz_t = pc_t["xyz"][0]
    xyz_next = pc_next["xyz"][0]
    xy_t, visible_t = _project_world_points(xyz_t, w2c_t, intrinsics_t, height, width)
    xy_next, visible_next = _project_world_points(xyz_next, w2c_next, intrinsics_next, height, width)
    displacement = xy_next - xy_t
    base_valid = pc_t.get(
        "valid_mask",
        torch.ones(pc_t["xyz"].shape[:2], device=xyz_t.device, dtype=torch.bool),
    )[0].to(dtype=torch.bool)
    next_valid = pc_next.get(
        "valid_mask",
        torch.ones(pc_next["xyz"].shape[:2], device=xyz_t.device, dtype=torch.bool),
    )[0].to(dtype=torch.bool)
    pair_valid = base_valid & next_valid & visible_t & visible_next & torch.isfinite(displacement).all(dim=-1)

    flow_pc = {
        key: value[:1]
        for key, value in pc_t.items()
        if torch.is_tensor(value) and value.ndim > 0 and value.shape[0] == pc_t["xyz"].shape[0]
    }
    flow_pc["valid_mask"] = pair_valid.unsqueeze(0)
    feature = torch.cat([displacement, displacement.new_zeros((displacement.shape[0], 1))], dim=-1)
    feature = torch.nan_to_num(feature, nan=0.0, posinf=0.0, neginf=0.0).unsqueeze(0)
    rendered = render_predicted(
        pc=flow_pc,
        world_view_transform=w2c_t.reshape(1, 1, 4, 4),
        intrinsics=intrinsics_t.reshape(1, 1, 3, 3),
        bg_color=xyz_t.new_zeros(3),
        cfg=splatter_cfg,
        override_color=feature,
        render_mode="RGB",
    )
    accumulated_flow = rendered["render"][0, 0, :2]
    alpha = rendered["alpha"][0, 0]
    normalized_flow = accumulated_flow / alpha.clamp_min(1.0e-6)
    normalized_flow = torch.where(alpha > _FLOW_ALPHA_THRESHOLD, normalized_flow, torch.zeros_like(normalized_flow))
    return normalized_flow, alpha


def _bilinear_sample(field: torch.Tensor, xy: torch.Tensor) -> torch.Tensor:
    if xy.numel() == 0:
        return field.new_zeros((0, field.shape[0]))
    height, width = field.shape[-2:]
    x = 2.0 * xy[:, 0] / float(max(1, width - 1)) - 1.0
    y = 2.0 * xy[:, 1] / float(max(1, height - 1)) - 1.0
    grid = torch.stack([x, y], dim=-1).view(1, -1, 1, 2)
    sampled = F.grid_sample(
        field.unsqueeze(0),
        grid,
        mode="bilinear",
        padding_mode="zeros",
        align_corners=True,
    )
    return sampled[0, :, :, 0].transpose(0, 1).contiguous()


def _points_inside_image(xy: torch.Tensor, height: int, width: int) -> torch.Tensor:
    return (
        torch.isfinite(xy).all(dim=-1)
        & (xy[:, 0] >= 0.0)
        & (xy[:, 0] <= float(width - 1))
        & (xy[:, 1] >= 0.0)
        & (xy[:, 1] <= float(height - 1))
    )


def _sample_foreground_queries(
    foreground_mask: torch.Tensor,
    alpha: torch.Tensor,
    max_queries: int,
) -> torch.Tensor:
    height, width = foreground_mask.shape
    ys, xs = torch.meshgrid(
        torch.arange(height, device=foreground_mask.device, dtype=alpha.dtype),
        torch.arange(width, device=foreground_mask.device, dtype=alpha.dtype),
        indexing="ij",
    )
    coordinates = torch.stack([xs, ys], dim=-1).reshape(-1, 2)
    candidate = (
        foreground_mask.to(dtype=torch.bool)
        & torch.isfinite(alpha[0])
        & (alpha[0] > _FLOW_ALPHA_THRESHOLD)
    ).reshape(-1)
    indices = _farthest_point_indices(coordinates, candidate, max(1, int(max_queries)))
    return coordinates[indices]


def make_wandb_gaussian_flow_track_payload(
    rec_out: Dict[str, Any],
    intrinsics: torch.Tensor,
    w2c: torch.Tensor,
    splatter_cfg: SplatterConfig,
    cfg_train: TrainConfig,
) -> Dict[str, wandb.Image]:
    pc_sequence = rec_out.get("gaussian_pc_sequence", None)
    rendered_images = rec_out.get("rendered_self", None)
    gt_images = rec_out.get("target_images_self", None)
    target_masks = rec_out.get("target_masks_self", None)
    if not pc_sequence or len(pc_sequence) < 3 or rendered_images is None or target_masks is None:
        return {}
    if pc_sequence[0]["xyz"].device.type != "cuda":
        return {}

    batch_idx = 0
    num_times = min(len(pc_sequence), rendered_images.shape[1], 3)
    num_views = rendered_images.shape[2]
    source_indices = rec_out.get("source_indices", None)
    view_idx = int(source_indices[batch_idx].item()) if source_indices is not None else 0
    view_idx = max(0, min(view_idx, num_views - 1))
    _, height, width = rendered_images.shape[-3:]

    flows: list[torch.Tensor] = []
    alphas: list[torch.Tensor] = []
    for time_idx in range(num_times - 1):
        flow, alpha = _render_alpha_weighted_transition_flow(
            pc_t=pc_sequence[time_idx],
            pc_next=pc_sequence[time_idx + 1],
            w2c_t=_camera_at_time(w2c, batch_idx, time_idx, view_idx),
            intrinsics_t=_camera_at_time(intrinsics, batch_idx, time_idx, view_idx),
            w2c_next=_camera_at_time(w2c, batch_idx, time_idx + 1, view_idx),
            intrinsics_next=_camera_at_time(intrinsics, batch_idx, time_idx + 1, view_idx),
            height=height,
            width=width,
            splatter_cfg=splatter_cfg,
        )
        flows.append(flow)
        alphas.append(alpha)

    max_queries = max(1, min(int(getattr(cfg_train, "val_track_max_points", 8)), 10))
    query0 = _sample_foreground_queries(
        foreground_mask=target_masks[batch_idx, 0, view_idx, 0],
        alpha=alphas[0],
        max_queries=max_queries,
    )
    if query0.numel() == 0:
        return {}

    points_by_time: list[torch.Tensor] = [query0]
    valid_by_time: list[torch.Tensor] = [torch.ones(query0.shape[0], device=query0.device, dtype=torch.bool)]
    current = query0
    current_valid = valid_by_time[0]
    for flow, alpha in zip(flows, alphas):
        sampled_flow = _bilinear_sample(flow, current)
        sampled_alpha = _bilinear_sample(alpha, current)[:, 0]
        next_points = current + sampled_flow
        next_valid = (
            current_valid
            & torch.isfinite(sampled_flow).all(dim=-1)
            & (sampled_alpha > _FLOW_ALPHA_THRESHOLD)
            & _points_inside_image(next_points, height, width)
        )
        points_by_time.append(next_points)
        valid_by_time.append(next_valid)
        current = next_points
        current_valid = next_valid

    rendered_frames = rendered_images[batch_idx, :num_times, view_idx].detach().cpu().clamp(0.0, 1.0)
    gt_frames = None
    if gt_images is not None:
        gt_frames = gt_images[batch_idx, :num_times, view_idx].detach().cpu().clamp(0.0, 1.0)
    image = _make_track_comparison_image(
        gt_frames=gt_frames,
        rendered_frames=rendered_frames,
        points_by_time=[points.detach().cpu() for points in points_by_time],
        valid_by_time=[valid.detach().cpu() for valid in valid_by_time],
        caption=(
            "Alpha-weighted Gaussian-flow point tracks — image-space tracking result. "
            "Per-Gaussian projected displacement is splatted with learned scale, rotation, opacity and visibility, "
            "then divided by accumulated alpha; tracks stop when alpha/visibility is insufficient."
        ),
    )
    return {"val/alpha_weighted_gaussian_flow_tracks": image}


def _depth_visualization(depth: torch.Tensor, foreground_mask: torch.Tensor) -> torch.Tensor:
    """Robustly normalize each foreground depth image for qualitative display."""
    if depth.dim() != 4 or depth.shape[1] != 1:
        raise ValueError(f"Expected depth as (B,1,H,W), got {tuple(depth.shape)}.")
    mask = foreground_mask.to(device=depth.device, dtype=torch.bool)
    if mask.shape != depth.shape:
        raise ValueError(f"Depth/mask visualization shapes differ: {tuple(depth.shape)} and {tuple(mask.shape)}.")
    output = torch.zeros_like(depth, dtype=torch.float32)
    depth_float = depth.float()
    for batch_idx in range(depth.shape[0]):
        valid = mask[batch_idx, 0] & torch.isfinite(depth_float[batch_idx, 0]) & (depth_float[batch_idx, 0] > 0.0)
        values = depth_float[batch_idx, 0][valid]
        if values.numel() == 0:
            continue
        if values.numel() == 1:
            lower = values[0]
            upper = values[0] + 1.0
        else:
            lower = torch.quantile(values, 0.02)
            upper = torch.quantile(values, 0.98)
            if bool((upper - lower).abs() < 1.0e-6):
                upper = lower + 1.0
        normalized = (depth_float[batch_idx, 0] - lower) / (upper - lower).clamp_min(1.0e-6)
        output[batch_idx, 0] = torch.where(valid, normalized.clamp(0.0, 1.0), torch.zeros_like(normalized))
    return output.expand(-1, 3, -1, -1).contiguous()


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

    prev_vae_mode = vae.training
    prev_converter_mode = splatter_to_gaussians.training
    vae.eval()
    splatter_to_gaussians.eval()

    scalar_keys = [
        "val/rgb_reconstruction_loss",
        "val/silhouette_foreground_loss",
        "val/silhouette_background_loss",
        "val/silhouette_loss",
        "val/global_depth_loss",
        "val/local_depth_loss",
        "val/depth_loss",
        "val/control_motion01_mean",
        "val/control_motion12_mean",
        "val/dense_motion_mean",
        "val/mean_valid_gaussian_opacity",
        "val/inv_contrastive_loss",
        "val/inv_consistency_loss",
        "val/dep_contrastive_loss",
        "val/dep_consistency_loss",
    ]
    scalar_sums = {key: 0.0 for key in scalar_keys}
    num_eval_batches = 0
    image_payload: Dict[str, Any] = {}

    for batch_idx, batch in enumerate(valid_dataloader):
        if cfg_train.val_num_batches > 0 and batch_idx >= cfg_train.val_num_batches:
            break

        images = batch["images"].to(device, non_blocking=True)
        depths = batch.get("depths", None)
        if depths is not None:
            depths = depths.to(device, non_blocking=True)
        masks = batch.get("masks", None)
        if masks is not None:
            masks = masks.to(device, non_blocking=True)
        intrinsics = batch["K"].to(device, non_blocking=True)
        c2w = batch["c2w"].to(device, non_blocking=True)
        w2c = batch["w2c"].to(device, non_blocking=True)
        images_01 = (images + 1.0) * 0.5

        view_latents, _view_inv_embed_loss, _view_dep_embed_loss = encode_per_view_sequence_batch(
            vae=vae,
            images=images,
        )
        inv_contrastive_loss, dep_contrastive_loss = compute_view_structured_contrastive_losses(
            s_inv_by_view=view_latents["s_inv_by_view"],
            z_dep_by_view=view_latents["z_dep_by_view"],
            temperature=cfg_train.temperature,
        )
        if cfg_train.inv_consistency_weight > 0.0 or cfg_train.dep_consistency_weight > 0.0:
            consistency_latents, _inv_embed_loss_aug, _dep_embed_loss_aug = encode_per_view_sequence_batch(
                vae=vae,
                images=images,
            )
            inv_consistency_loss = compute_state_consistency_loss(
                view_latents["s_inv_by_view"].flatten(0, 1),
                consistency_latents["s_inv_by_view"].flatten(0, 1),
            )
            dep_consistency_loss = compute_dependent_view_consistency_loss(
                view_latents["z_dep_by_view"],
                consistency_latents["z_dep_by_view"],
            )
        else:
            inv_consistency_loss = images.new_zeros(())
            dep_consistency_loss = images.new_zeros(())

        rec_out = compute_reconstruction_and_renders(
            vae=vae,
            splatter_to_gaussians=splatter_to_gaussians,
            splatter_cfg=splatter_cfg,
            images_01=images_01,
            intrinsics=intrinsics,
            c2w=c2w,
            w2c=w2c,
            bg=bg,
            cfg_train=cfg_train,
            depths=depths,
            masks=masks,
            return_renders=(num_eval_batches == 0),
        )

        metric_map = {
            "val/rgb_reconstruction_loss": rec_out["rgb_loss"],
            "val/silhouette_foreground_loss": rec_out["silhouette_foreground_loss"],
            "val/silhouette_background_loss": rec_out["silhouette_background_loss"],
            "val/silhouette_loss": rec_out["silhouette_loss"],
            "val/global_depth_loss": rec_out["global_depth_loss"],
            "val/local_depth_loss": rec_out["local_depth_loss"],
            "val/depth_loss": rec_out["depth_loss"],
            "val/control_motion01_mean": rec_out["control_motion01_mean"],
            "val/control_motion12_mean": rec_out["control_motion12_mean"],
            "val/dense_motion_mean": rec_out["dense_motion_mean"],
            "val/mean_valid_gaussian_opacity": rec_out["mean_valid_gaussian_opacity"],
            "val/inv_contrastive_loss": inv_contrastive_loss,
            "val/inv_consistency_loss": inv_consistency_loss,
            "val/dep_contrastive_loss": dep_contrastive_loss,
            "val/dep_consistency_loss": dep_consistency_loss,
        }
        for key, value in metric_map.items():
            scalar_sums[key] += float(value.item())

        if num_eval_batches == 0:
            image_payload["val/input_images"] = make_wandb_input_image_panel(
                images_01=images_01,
                max_vis=cfg_train.val_max_vis,
            )
            rendered_self = rec_out["rendered_self"]
            rendered_depth = rec_out["rendered_expected_depth_self"]
            rendered_alpha = rec_out["rendered_alpha_self"]
            target_images = rec_out["target_images_self"]
            target_masks = rec_out["target_masks_self"]
            target_depths = rec_out.get("target_depths_self", None)
            num_times_to_show = min(rendered_self.shape[1], 3)
            num_views_to_show = min(rendered_self.shape[2], 3)
            panel_items: list[tuple[str, torch.Tensor]] = []
            for time_idx in range(num_times_to_show):
                for view_slot in range(num_views_to_show):
                    exact_mask = target_masks[:, time_idx, view_slot]
                    mask_rgb = exact_mask.expand(-1, 3, -1, -1).contiguous()
                    alpha_rgb = rendered_alpha[:, time_idx, view_slot].expand(-1, 3, -1, -1).contiguous()
                    target_depth = (
                        torch.zeros_like(rendered_depth[:, time_idx, view_slot])
                        if target_depths is None
                        else target_depths[:, time_idx, view_slot]
                    )
                    target_depth_vis = _depth_visualization(target_depth, exact_mask)
                    rendered_depth_vis = _depth_visualization(rendered_depth[:, time_idx, view_slot], exact_mask)
                    prefix = f"t{time_idx}_cam{view_slot}"
                    panel_items.extend(
                        [
                            (f"ground_truth_rgb_{prefix}", target_images[:, time_idx, view_slot]),
                            (f"rendered_rgb_{prefix}", rendered_self[:, time_idx, view_slot]),
                            (f"target_depth_robust_norm_{prefix}", target_depth_vis),
                            (f"rendered_expected_depth_robust_norm_{prefix}", rendered_depth_vis),
                            (f"ground_truth_segmentation_mask_{prefix}", mask_rgb),
                            (f"rendered_alpha_{prefix}", alpha_rgb),
                        ]
                    )
            image_payload["val/render_summary"] = make_wandb_named_image_panel(
                panel_items,
                max_vis=cfg_train.val_max_vis,
            )
            image_payload.update(
                make_wandb_pointcloud_payload(
                    rec_out=rec_out,
                    depths=depths,
                    intrinsics=intrinsics,
                    c2w=c2w,
                    splatter_cfg=splatter_cfg,
                    cfg_train=cfg_train,
                    masks=masks,
                )
            )
            image_payload.update(
                make_wandb_sparse_control_trajectory_payload(
                    rec_out=rec_out,
                    intrinsics=intrinsics,
                    w2c=w2c,
                    cfg_train=cfg_train,
                )
            )
            image_payload.update(
                make_wandb_gaussian_flow_track_payload(
                    rec_out=rec_out,
                    intrinsics=intrinsics,
                    w2c=w2c,
                    splatter_cfg=splatter_cfg,
                    cfg_train=cfg_train,
                )
            )
            image_payload.update(
                render_validation_trajectory_videos(
                    pc=rec_out["gaussian_pc"],
                    source_c2w=rec_out["source_c2w"],
                    source_intrinsics=rec_out["source_intrinsics"],
                    splatter_cfg=splatter_cfg,
                    bg=bg,
                    cfg_train=cfg_train,
                )
            )

        num_eval_batches += 1

    if num_eval_batches == 0:
        vae.train(prev_vae_mode)
        splatter_to_gaussians.train(prev_converter_mode)
        return

    log_dict: Dict[str, Any] = {key: value / float(num_eval_batches) for key, value in scalar_sums.items()}
    log_dict["global_step"] = global_step
    log_dict.update({key: value for key, value in image_payload.items() if not key.startswith("mask_")})
    wandb.log(log_dict, step=global_step)

    vae.train(prev_vae_mode)
    splatter_to_gaussians.train(prev_converter_mode)
