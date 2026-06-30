from __future__ import annotations

from typing import Any, Dict, Optional

import torch
from torch.utils.data import DataLoader
from torchvision.utils import make_grid

import wandb

from models.losses import compute_all_camera_contrastive_losses, compute_latent_consistency_loss
from models.splatter import SplatterConfig, render_predicted
from models.splatter_gaussians import DirectSplatterToGaussians
from models.splatter_point_losses import depths_to_world_point_cloud, sample_points
from models.splatter_reconstruction import compute_reconstruction_and_renders, encode_all_camera_batch
from models.splatter_train_config import TrainConfig
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
            caption=f"Predicted Gaussian centers, sampled to <= {max_points}",
        )

    if depths is None:
        return payload
    gt_points, gt_mask = depths_to_world_point_cloud(
        depths=depths[:1],
        intrinsics=intrinsics[:1],
        c2w=c2w[:1],
        splatter_cfg=splatter_cfg,
        masks=None if masks is None else masks[:1],
    )
    gt = sample_object3d_array(
        points=gt_points[0],
        mask=gt_mask[0],
        max_points=max_points,
        rgb=(80.0, 220.0, 120.0),
    )
    if gt is not None:
        payload["val/pointcloud_gt_masked_depth"] = wandb.Object3D(
            gt.numpy(),
            caption=f"GT masked depth point cloud, sampled to <= {max_points}",
        )

    if pred is not None and gt is not None:
        overlay = torch.cat([pred, gt], dim=0)
        payload["val/pointcloud_pred_gt_overlay"] = wandb.Object3D(
            overlay.numpy(),
            caption="Predicted Gaussian centers red; masked GT depth points green",
        )
    return payload


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
        "val/inv_contrastive_loss",
        "val/inv_consistency_loss",
        "val/dep_contrastive_loss",
        "val/dep_consistency_loss",
        "val/point_chamfer_loss",
        "val/gt_point_count_mean",
        "val/mean_opacity",
        "val/mask_pixel_ratio",
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

        latents, _inv_vq_loss, _dep_vq_loss = encode_all_camera_batch(vae=vae, images=images)
        inv_contrastive_loss, dep_contrastive_loss = compute_all_camera_contrastive_losses(
            z_inv=latents["z_inv"],
            z_dep=latents["z_dep"],
            temperature=cfg_train.temperature,
        )
        inv_consistency_loss = compute_latent_consistency_loss(latents["z_inv"], mode="state")
        dep_consistency_loss = compute_latent_consistency_loss(latents["z_dep"], mode="view")

        rec_out = compute_reconstruction_and_renders(
            vae=vae,
            splatter_to_gaussians=splatter_to_gaussians,
            splatter_cfg=splatter_cfg,
            images_01=images_01,
            z_inv=latents["z_inv"],
            z_dep=latents["z_dep"],
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
            "val/inv_contrastive_loss": inv_contrastive_loss,
            "val/inv_consistency_loss": inv_consistency_loss,
            "val/dep_contrastive_loss": dep_contrastive_loss,
            "val/dep_consistency_loss": dep_consistency_loss,
            "val/point_chamfer_loss": rec_out["point_chamfer_loss"],
            "val/gt_point_count_mean": rec_out["gt_point_count_mean"],
            "val/mean_opacity": rec_out["mean_opacity"],
            "val/mask_pixel_ratio": rec_out["mask_pixel_ratio"],
        }
        for key, value in metric_map.items():
            scalar_sums[key] += float(value.item())

        if num_eval_batches == 0:
            num_targets_to_show = min(rec_out["rendered_self"].shape[1], 6)
            panel_items: list[tuple[str, torch.Tensor]] = []
            for view_slot in range(num_targets_to_show):
                view_name = "source" if view_slot == 0 else f"target{view_slot}"
                panel_items.append((f"gt_{view_name}", rec_out["target_images_self"][:, view_slot]))
            for view_slot in range(num_targets_to_show):
                view_name = "source" if view_slot == 0 else f"target{view_slot}"
                panel_items.append((f"render_{view_name}", rec_out["rendered_self"][:, view_slot]))
            panel_items = [(name, image) for name, image in panel_items if not name.startswith("mask_")]
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
