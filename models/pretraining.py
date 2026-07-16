from __future__ import annotations

import os
from typing import Optional

import torch
from torch.utils.data import DataLoader

import wandb

from models.losses import (
    compute_dependent_view_consistency_loss,
    compute_state_consistency_loss,
    compute_view_structured_contrastive_losses,
)
from models.splatter import SplatterConfig
from models.gaussians import DirectSplatterToGaussians
from models.reconstruction import (
    compute_reconstruction_and_renders,
    encode_per_view_sequence_batch,
)
from models.train_config import TrainConfig
from models.validation import validate_and_log_wandb
from models.vae import SplatterVAE
from utils.training_utils import (
    compute_scheduled_lr,
    normalize_lr_schedule,
    resolve_lr_total_steps,
    set_optimizer_lr,
)


def _build_converter(splatter_cfg: SplatterConfig, device: torch.device) -> DirectSplatterToGaussians:
    return DirectSplatterToGaussians(splatter_cfg).to(device)


def train_splatter_vae(
    vae: SplatterVAE,
    splatter_cfg: SplatterConfig,
    train_dataloader: DataLoader,
    cfg_train: TrainConfig,
    valid_dataloader: Optional[DataLoader] = None,
    resume_ckpt: Optional[str] = None,
):
    """Train SplatterVAE with direct Gaussians, masked rendering, and masked Chamfer supervision."""
    device = torch.device(cfg_train.device)
    vae.to(device)
    splatter_to_gaussians = _build_converter(splatter_cfg, device)
    optimizer = torch.optim.Adam(list(vae.parameters()), lr=cfg_train.lr)

    lr_total_steps = resolve_lr_total_steps(cfg_train, train_dataloader)
    lr_schedule = normalize_lr_schedule(cfg_train.lr_schedule)
    if lr_schedule != "constant":
        print(
            f"[LR] schedule={lr_schedule}, peak_lr={cfg_train.lr:g}, min_lr={cfg_train.min_lr:g}, "
            f"warmup_steps={cfg_train.lr_warmup_steps}, total_steps={lr_total_steps}"
        )
    bg = torch.ones(3, device=device) if splatter_cfg.data.white_background else torch.zeros(3, device=device)
    start_epoch = 0
    global_step = 0

    if resume_ckpt is not None and os.path.isfile(resume_ckpt):
        ckpt = torch.load(resume_ckpt, map_location="cpu")
        vae.load_state_dict(ckpt["vae_state_dict"])
        try:
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        except ValueError as exc:
            print(f"[Resume] Skipping optimizer state because parameter groups changed: {exc}")
        start_epoch = int(ckpt["epoch"])
        global_step = int(ckpt["global_step"])

    os.makedirs(cfg_train.ckpt_dir, exist_ok=True)

    epoch = start_epoch
    while True:
        for step, batch in enumerate(train_dataloader):
            if cfg_train.max_global_steps is not None and global_step >= cfg_train.max_global_steps:
                print(f"[Stop] Reached max_global_steps={cfg_train.max_global_steps}.")
                return

            current_lr = compute_scheduled_lr(cfg_train, global_step, lr_total_steps)
            set_optimizer_lr(optimizer, current_lr)
            vae.train()
            splatter_to_gaussians.train()

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

            optimizer.zero_grad(set_to_none=True)
            view_latents, _view_inv_embed_loss, _view_dep_embed_loss = encode_per_view_sequence_batch(
                vae=vae,
                images=images,
            )
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
                return_renders=False,
            )
            rec_loss = rec_out["rec_loss"]
            occupancy_loss = rec_out["occupancy_loss"]
            point_chamfer_loss = rec_out["point_chamfer_loss"]
            delta_smooth_loss = rec_out["delta_smooth_loss"]

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
                inv_consistency_loss = rec_loss.new_zeros(())
                dep_consistency_loss = rec_loss.new_zeros(())

            rec_weight = float(cfg_train.rec_weight)
            total_loss = (
                rec_weight * rec_loss
                + cfg_train.occupancy_weight * occupancy_loss
                + cfg_train.point_chamfer_weight * point_chamfer_loss
                + cfg_train.delta_smooth_weight * delta_smooth_loss
                + cfg_train.inv_contrastive_weight * inv_contrastive_loss
                + cfg_train.inv_consistency_weight * inv_consistency_loss
                + cfg_train.dep_contrastive_weight * dep_contrastive_loss
                + cfg_train.dep_consistency_weight * dep_consistency_loss
            )

            finite_terms = {
                "rec_loss": rec_loss,
                "occupancy_loss": occupancy_loss,
                "point_chamfer_loss": point_chamfer_loss,
                "delta_smooth_loss": delta_smooth_loss,
                "inv_contrastive_loss": inv_contrastive_loss,
                "inv_consistency_loss": inv_consistency_loss,
                "dep_contrastive_loss": dep_contrastive_loss,
                "dep_consistency_loss": dep_consistency_loss,
                "total_loss": total_loss,
            }
            bad_terms = [name for name, value in finite_terms.items() if not torch.isfinite(value).all()]
            if bad_terms:
                print(
                    f"[Warn] Non-finite loss at global_step={global_step} "
                    f"(bad={bad_terms}). Skipping optimizer step."
                )
                if wandb.run is not None:
                    wandb.log({"global_step": global_step, "train/nonfinite_batch": 1.0, "train/lr": current_lr}, step=global_step)
                global_step += 1
                continue

            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(list(vae.parameters()), max_norm=5.0)
            optimizer.step()

            if step % 250 == 0:
                print(
                    f"[Epoch {epoch + 1} | Step {step} | Global {global_step}] "
                    f"Loss={total_loss.item():.4f} lr={current_lr:.2e} "
                    f"(rgb={rec_loss.item():.4f}, rgb_w={rec_weight:.4f}, "
                    f"occupancy={occupancy_loss.item():.4f}, "
                    f"gaussian_chamfer={point_chamfer_loss.item():.4f}, "
                    f"delta={delta_smooth_loss.item():.4f}, inv_con={inv_contrastive_loss.item():.4f}, "
                    f"dep_con={dep_contrastive_loss.item():.4f}, mask={rec_out['mask_pixel_ratio'].item():.3f})"
                )
                if wandb.run is not None:
                    timestep_log = {}
                    for time_idx in range(3):
                        chamfer_key = f"point_chamfer_loss_t{time_idx}"
                        if chamfer_key in rec_out:
                            timestep_log[f"train/{chamfer_key}"] = rec_out[chamfer_key].item()
                    wandb.log(
                        {
                            "train/total_loss": total_loss.item(),
                            "train/lr": current_lr,
                            "train/rec_loss": rec_loss.item(),
                            "train/rec_weight": rec_weight,
                            "train/rec_loss_weighted": rec_weight * rec_loss.item(),
                            "train/occupancy_loss": occupancy_loss.item(),
                            "train/occupancy_weight": float(cfg_train.occupancy_weight),
                            "train/occupancy_loss_weighted": (cfg_train.occupancy_weight * occupancy_loss).item(),
                            "train/occupancy_fixed_opacity": float(cfg_train.occupancy_fixed_opacity),
                            "train/rgb_loss_mask_dilation": float(cfg_train.rgb_loss_mask_dilation),
                            "train/point_chamfer_loss": point_chamfer_loss.item(),
                            "train/point_chamfer_loss_weighted": (cfg_train.point_chamfer_weight * point_chamfer_loss).item(),
                            "train/delta_smooth_loss": delta_smooth_loss.item(),
                            "train/delta_smooth_loss_weighted": (cfg_train.delta_smooth_weight * delta_smooth_loss).item(),
                            "train/delta01_mean": rec_out["delta01_mean"].item(),
                            "train/delta12_mean": rec_out["delta12_mean"].item(),
                            "train/delta_magnitude_mean": rec_out["delta_magnitude_mean"].item(),
                            "train/gt_point_count_mean": rec_out["gt_point_count_mean"].item(),
                            "train/inv_contrastive_loss": inv_contrastive_loss.item(),
                            "train/inv_consistency_loss": inv_consistency_loss.item(),
                            "train/dep_contrastive_loss": dep_contrastive_loss.item(),
                            "train/dep_consistency_loss": dep_consistency_loss.item(),
                            "train/mean_opacity": rec_out["mean_opacity"].item(),
                            "train/mask_pixel_ratio": rec_out["mask_pixel_ratio"].item(),
                            "train/expanded_mask_pixel_ratio": rec_out["expanded_mask_pixel_ratio"].item(),
                            "train/point_chamfer_weight": float(cfg_train.point_chamfer_weight),
                            "train/delta_smooth_weight": float(cfg_train.delta_smooth_weight),
                            "global_step": global_step,
                            **timestep_log,
                        },
                        step=global_step,
                    )

            if valid_dataloader is not None and cfg_train.eval_every > 0 and global_step > 0 and global_step % cfg_train.eval_every == 0:
                validate_and_log_wandb(
                    vae=vae,
                    splatter_cfg=splatter_cfg,
                    splatter_to_gaussians=splatter_to_gaussians,
                    valid_dataloader=valid_dataloader,
                    device=device,
                    bg=bg,
                    cfg_train=cfg_train,
                    global_step=global_step,
                )

            if cfg_train.save_every > 0 and global_step > 0 and global_step % cfg_train.save_every == 0:
                ckpt_path = os.path.join(cfg_train.ckpt_dir, f"step_{global_step:08d}.pth")
                ckpt = {
                    "epoch": epoch,
                    "global_step": global_step,
                    "vae_state_dict": vae.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                }
                torch.save(ckpt, ckpt_path)
                print(f"[Checkpoint] Saved checkpoint to {ckpt_path}")

            global_step += 1

        epoch += 1
        if cfg_train.max_global_steps is None and epoch >= cfg_train.num_epochs:
            print(f"[Stop] Reached num_epochs={cfg_train.num_epochs}.")
            break
