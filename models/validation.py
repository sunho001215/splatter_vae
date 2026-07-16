from __future__ import annotations

from typing import Any, Dict, Optional, Sequence

import torch
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
from models.point_losses import depths_to_world_point_cloud, sample_points
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
            render_mode="RGB",
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


def make_wandb_pointcloud_payload(
    rec_out: Dict[str, Any],
    depths: Optional[torch.Tensor],
    intrinsics: torch.Tensor,
    c2w: torch.Tensor,
    splatter_cfg: SplatterConfig,
    cfg_train: TrainConfig,
    masks: Optional[torch.Tensor],
) -> Dict[str, wandb.Object3D]:
    if not bool(cfg_train.val_log_pointclouds):
        return {}
    max_points = max(1, int(cfg_train.val_pointcloud_max_points))
    pc = rec_out.get("gaussian_pc", None)
    if pc is None:
        return {}
    pc_sequence = rec_out.get("gaussian_pc_sequence", [pc])

    pred_mask = pc.get("valid_mask", torch.ones(pc["xyz"].shape[:2], device=pc["xyz"].device, dtype=torch.bool))
    pred = sample_object3d_array(
        points=pc["xyz"][0],
        mask=pred_mask[0],
        max_points=max_points,
        rgb=(255.0, 80.0, 80.0),
    )
    payload: Dict[str, wandb.Object3D] = {}
    if pred is not None:
        payload["val/pointcloud_pred_gaussians"] = wandb.Object3D(
            pred.numpy(),
            caption=f"Predicted Gaussian centers at t0, sampled to <= {max_points}",
        )

    if depths is None:
        return payload

    num_times = min(len(pc_sequence), depths.shape[1] if depths.dim() == 6 else 1, 3)
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

        depths_t = depths[:1, time_idx] if depths.dim() == 6 else depths[:1]
        intrinsics_t = intrinsics[:1, time_idx] if intrinsics.dim() == 5 else intrinsics[:1]
        c2w_t = c2w[:1, time_idx] if c2w.dim() == 5 else c2w[:1]
        masks_t = None if masks is None else (masks[:1, time_idx] if masks.dim() == 6 else masks[:1])
        gt_points, gt_mask = depths_to_world_point_cloud(
            depths=depths_t,
            intrinsics=intrinsics_t,
            c2w=c2w_t,
            splatter_cfg=splatter_cfg,
            masks=masks_t,
        )
        gt_t = sample_object3d_array(
            points=gt_points[0],
            mask=gt_mask[0],
            max_points=max_points,
            rgb=(80.0, 220.0, 120.0),
        )
        if time_idx == 0 and gt_t is not None:
            payload["val/pointcloud_gt_masked_depth"] = wandb.Object3D(
                gt_t.numpy(),
                caption=f"GT masked depth point cloud at t0, sampled to <= {max_points}",
            )

        if pred_t is not None and gt_t is not None:
            overlay = torch.cat([pred_t, gt_t], dim=0)
            payload[f"val/pointcloud_pred_gt_overlay/t{time_idx}"] = wandb.Object3D(
                overlay.numpy(),
                caption=f"t{time_idx}: predicted Gaussian centers red; masked GT depth points green",
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


def make_wandb_temporal_track_payload(
    rec_out: Dict[str, Any],
    intrinsics: torch.Tensor,
    w2c: torch.Tensor,
    cfg_train: TrainConfig,
) -> Dict[str, wandb.Image]:
    pc_sequence = rec_out.get("gaussian_pc_sequence", None)
    rendered_images = rec_out.get("rendered_self", None)
    gt_images = rec_out.get("target_images_self", None)
    if not pc_sequence or rendered_images is None:
        return {}

    max_tracks = max(1, min(int(getattr(cfg_train, "val_track_max_points", 8)), 10))
    batch_idx = 0
    num_times = min(len(pc_sequence), rendered_images.shape[1], 3)
    num_views = rendered_images.shape[2]
    source_indices = rec_out.get("source_indices", None)
    view_idx = int(source_indices[batch_idx].item()) if source_indices is not None else 0
    view_idx = max(0, min(view_idx, num_views - 1))

    frames = rendered_images[batch_idx, :num_times, view_idx].detach().cpu().clamp(0.0, 1.0)
    gt_frames = None
    if gt_images is not None:
        gt_frames = gt_images[batch_idx, :num_times, view_idx].detach().cpu().clamp(0.0, 1.0)
    _, height, width = frames[0].shape
    pc0 = pc_sequence[0]
    pred_mask = pc0.get("valid_mask", torch.ones(pc0["xyz"].shape[:2], device=pc0["xyz"].device, dtype=torch.bool))
    xys = []
    visible = []
    for time_idx in range(num_times):
        xy_t, visible_t = _project_world_points(
            points=pc_sequence[time_idx]["xyz"][batch_idx],
            w2c=_camera_at_time(w2c, batch_idx, time_idx, view_idx),
            intrinsics=_camera_at_time(intrinsics, batch_idx, time_idx, view_idx),
            height=height,
            width=width,
        )
        xys.append(xy_t.detach().cpu())
        visible.append(visible_t.detach().cpu())

    indices = _select_track_indices(
        xys=xys,
        visible=visible,
        valid_mask=pred_mask[batch_idx].detach().cpu(),
        max_tracks=max_tracks,
    )
    if indices.numel() == 0:
        return {}

    segment_colors = [
        torch.tensor([0.05, 0.85, 1.00], dtype=torch.float32),
        torch.tensor([1.00, 0.55, 0.05], dtype=torch.float32),
    ]
    marker_colors = [
        torch.tensor([1.00, 0.10, 0.10], dtype=torch.float32),
        torch.tensor([1.00, 0.95, 0.10], dtype=torch.float32),
        torch.tensor([0.10, 1.00, 0.25], dtype=torch.float32),
    ]
    pad = 6
    row_gap = 8
    base_canvas = torch.ones((3, height, num_times * width + (num_times - 1) * pad), dtype=torch.float32)
    for time_idx in range(num_times):
        x0 = time_idx * (width + pad)
        base_canvas[:, :, x0 : x0 + width] = frames[time_idx]

    track_rows = []
    captions = []
    if gt_frames is not None:
        gt_canvas = torch.ones_like(base_canvas)
        for time_idx in range(num_times):
            x0 = time_idx * (width + pad)
            gt_canvas[:, :, x0 : x0 + width] = gt_frames[time_idx]
        track_rows.append(gt_canvas.clamp(0.0, 1.0))

    for track_slot, gaussian_idx in enumerate(indices.tolist()):
        canvas = base_canvas.clone()
        panel_points = []
        for time_idx in range(num_times):
            xy = xys[time_idx][gaussian_idx].clone()
            xy[0] = xy[0] + time_idx * (width + pad)
            panel_points.append(xy)

        for time_idx in range(num_times - 1):
            color = segment_colors[min(time_idx, len(segment_colors) - 1)]
            _draw_line(canvas, panel_points[time_idx], panel_points[time_idx + 1], color, width=2)
        for time_idx, xy in enumerate(panel_points):
            color = marker_colors[min(time_idx, len(marker_colors) - 1)]
            _draw_endpoint_marker(canvas, xy, color)

        track_rows.append(canvas.clamp(0.0, 1.0))
        captions.append(f"row {track_slot}: gaussian {gaussian_idx}")

    grid = make_grid(torch.stack(track_rows, dim=0), nrow=1, padding=row_gap, pad_value=1.0)
    return {
        "val/gaussian_track_overlay": wandb.Image(
            grid.clamp(0.0, 1.0),
            caption=(
                f"Ground-truth images on the top row, then predicted Gaussian tracks projected into camera {view_idx}; "
                "each following row is one Gaussian over rendered images. "
                "Cyan: t0->t1, orange: t1->t2. Markers are red/yellow/green for t0/t1/t2. "
                + " | ".join(captions)
            ),
        )
    }


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
        "val/rec_loss",
        "val/occupancy_loss",
        "val/inv_contrastive_loss",
        "val/inv_consistency_loss",
        "val/dep_contrastive_loss",
        "val/dep_consistency_loss",
        "val/point_chamfer_loss",
        "val/delta_smooth_loss",
        "val/delta_magnitude_mean",
        "val/gt_point_count_mean",
        "val/mean_opacity",
        "val/mask_pixel_ratio",
        "val/expanded_mask_pixel_ratio",
    ]
    for time_idx in range(3):
        scalar_keys.append(f"val/point_chamfer_loss_t{time_idx}")
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
            "val/rec_loss": rec_out["rec_loss"],
            "val/occupancy_loss": rec_out["occupancy_loss"],
            "val/inv_contrastive_loss": inv_contrastive_loss,
            "val/inv_consistency_loss": inv_consistency_loss,
            "val/dep_contrastive_loss": dep_contrastive_loss,
            "val/dep_consistency_loss": dep_consistency_loss,
            "val/point_chamfer_loss": rec_out["point_chamfer_loss"],
            "val/delta_smooth_loss": rec_out["delta_smooth_loss"],
            "val/delta_magnitude_mean": rec_out["delta_magnitude_mean"],
            "val/gt_point_count_mean": rec_out["gt_point_count_mean"],
            "val/mean_opacity": rec_out["mean_opacity"],
            "val/mask_pixel_ratio": rec_out["mask_pixel_ratio"],
            "val/expanded_mask_pixel_ratio": rec_out["expanded_mask_pixel_ratio"],
        }
        for time_idx in range(3):
            metric_map[f"val/point_chamfer_loss_t{time_idx}"] = rec_out.get(
                f"point_chamfer_loss_t{time_idx}",
                images.new_zeros(()),
            )
        for key, value in metric_map.items():
            scalar_sums[key] += float(value.item())

        if num_eval_batches == 0:
            image_payload["val/input_images"] = make_wandb_input_image_panel(
                images_01=images_01,
                max_vis=cfg_train.val_max_vis,
            )
            num_times_to_show = min(rec_out["rendered_self"].shape[1], 3)
            num_views_to_show = min(rec_out["rendered_self"].shape[2], 3)
            panel_items: list[tuple[str, torch.Tensor]] = []
            for time_idx in range(num_times_to_show):
                for view_slot in range(num_views_to_show):
                    panel_items.append((f"gt_t{time_idx}_cam{view_slot}", rec_out["target_images_self"][:, time_idx, view_slot]))
                for view_slot in range(num_views_to_show):
                    panel_items.append((f"render_t{time_idx}_cam{view_slot}", rec_out["rendered_self"][:, time_idx, view_slot]))
            image_payload["val/render_summary"] = make_wandb_named_image_panel(panel_items, max_vis=cfg_train.val_max_vis)
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
                make_wandb_temporal_track_payload(
                    rec_out=rec_out,
                    intrinsics=intrinsics,
                    w2c=w2c,
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
