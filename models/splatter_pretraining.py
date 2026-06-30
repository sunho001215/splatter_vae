from __future__ import annotations

import os
from typing import Optional

import torch
from torch.utils.data import DataLoader

import wandb

from models.losses import compute_all_camera_contrastive_losses, compute_latent_consistency_loss
from models.splatter import SplatterConfig
from models.splatter_gaussians import DirectSplatterToGaussians
from models.splatter_reconstruction import compute_reconstruction_and_renders, encode_all_camera_batch
from models.splatter_train_config import TrainConfig
from models.splatter_validation import validate_and_log_wandb
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
            latents, inv_vq_loss, dep_vq_loss = encode_all_camera_batch(vae=vae, images=images)
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
                return_renders=False,
            )
            rec_loss = rec_out["rec_loss"]
            point_chamfer_loss = rec_out["point_chamfer_loss"]

            inv_contrastive_loss, dep_contrastive_loss = compute_all_camera_contrastive_losses(
                z_inv=latents["z_inv"],
                z_dep=latents["z_dep"],
                temperature=cfg_train.temperature,
            )
            inv_consistency_loss = compute_latent_consistency_loss(latents["z_inv"], mode="state")
            dep_consistency_loss = compute_latent_consistency_loss(latents["z_dep"], mode="view")

            vq_loss = inv_vq_loss + dep_vq_loss
            rec_weight = float(cfg_train.rec_weight)
            total_loss = (
                rec_weight * rec_loss
                + cfg_train.point_chamfer_weight * point_chamfer_loss
                + cfg_train.vq_weight * vq_loss
                + cfg_train.inv_contrastive_weight * inv_contrastive_loss
                + cfg_train.inv_consistency_weight * inv_consistency_loss
                + cfg_train.dep_contrastive_weight * dep_contrastive_loss
                + cfg_train.dep_consistency_weight * dep_consistency_loss
            )

            finite_terms = {
                "rec_loss": rec_loss,
                "point_chamfer_loss": point_chamfer_loss,
                "vq_loss": vq_loss,
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
                    f"gaussian_chamfer={point_chamfer_loss.item():.4f}, "
                    f"vq={vq_loss.item():.4f}, inv_con={inv_contrastive_loss.item():.4f}, "
                    f"dep_con={dep_contrastive_loss.item():.4f}, mask={rec_out['mask_pixel_ratio'].item():.3f})"
                )
                if wandb.run is not None:
                    wandb.log(
                        {
                            "train/total_loss": total_loss.item(),
                            "train/lr": current_lr,
                            "train/rec_loss": rec_loss.item(),
                            "train/rec_weight": rec_weight,
                            "train/rec_background_weight": float(cfg_train.rec_background_weight),
                            "train/rec_loss_weighted": rec_weight * rec_loss.item(),
                            "train/point_chamfer_loss": point_chamfer_loss.item(),
                            "train/point_chamfer_loss_weighted": (cfg_train.point_chamfer_weight * point_chamfer_loss).item(),
                            "train/gt_point_count_mean": rec_out["gt_point_count_mean"].item(),
                            "train/vq_loss": vq_loss.item(),
                            "train/inv_vq_loss": inv_vq_loss.item(),
                            "train/dep_vq_loss": dep_vq_loss.item(),
                            "train/inv_contrastive_loss": inv_contrastive_loss.item(),
                            "train/inv_consistency_loss": inv_consistency_loss.item(),
                            "train/dep_contrastive_loss": dep_contrastive_loss.item(),
                            "train/dep_consistency_loss": dep_consistency_loss.item(),
                            "train/mean_opacity": rec_out["mean_opacity"].item(),
                            "train/mask_pixel_ratio": rec_out["mask_pixel_ratio"].item(),
                            "train/point_chamfer_weight": float(cfg_train.point_chamfer_weight),
                            "global_step": global_step,
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
