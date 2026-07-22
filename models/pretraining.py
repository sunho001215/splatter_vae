from __future__ import annotations

import os
from typing import Optional

import torch
from torch.utils.data import DataLoader

import wandb

from models.losses import compute_view_structured_representation_losses
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
    """Train SplatterVAE with sparse motion controls and unified rendering supervision."""
    device = torch.device(cfg_train.device)
    vae.to(device)
    splatter_to_gaussians = _build_converter(splatter_cfg, device)
    trainable_parameters = list(vae.parameters())
    optimizer_kwargs = {"lr": cfg_train.lr}
    if device.type == "cuda":
        optimizer_kwargs["fused"] = True
    try:
        optimizer = torch.optim.Adam(trainable_parameters, **optimizer_kwargs)
    except (TypeError, RuntimeError):
        optimizer_kwargs.pop("fused", None)
        optimizer = torch.optim.Adam(trainable_parameters, **optimizer_kwargs)

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
            batch_size, num_views = view_latents["s_inv_by_view"].shape[:2]
            source_indices = torch.randint(num_views, (batch_size,), device=device)
            batch_indices = torch.arange(batch_size, device=device)
            s_inv_source = view_latents["s_inv_by_view"][batch_indices, source_indices]
            z_dep_source = view_latents["z_dep_by_view"][batch_indices, source_indices]
            should_log = step % 250 == 0
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
                s_inv_source=s_inv_source,
                z_dep_source=z_dep_source,
                source_indices=source_indices,
                depths=depths,
                masks=masks,
                return_renders=False,
                compute_diagnostics=bool(should_log and wandb.run is not None),
            )
            rgb_loss = rec_out["rgb_loss"]
            silhouette_loss = rec_out["silhouette_loss"]
            global_depth_loss = rec_out["global_depth_loss"]
            local_depth_loss = rec_out["local_depth_loss"]
            depth_loss = rec_out["depth_loss"]

            (
                inv_contrastive_loss,
                dep_contrastive_loss,
                inv_consistency_loss,
                dep_consistency_loss,
            ) = compute_view_structured_representation_losses(
                s_inv_by_view=view_latents["s_inv_by_view"],
                z_dep_by_view=view_latents["z_dep_by_view"],
                temperature=cfg_train.temperature,
            )

            rec_weight = float(cfg_train.rec_weight)
            render_loss = (
                rec_weight * rgb_loss
                + float(cfg_train.silhouette_weight) * silhouette_loss
                + float(cfg_train.global_depth_weight) * depth_loss
            )
            total_loss = (
                render_loss
                + cfg_train.inv_contrastive_weight * inv_contrastive_loss
                + cfg_train.inv_consistency_weight * inv_consistency_loss
                + cfg_train.dep_contrastive_weight * dep_contrastive_loss
                + cfg_train.dep_consistency_weight * dep_consistency_loss
            )

            finite_terms = {
                "rgb_loss": rgb_loss,
                "silhouette_loss": silhouette_loss,
                "global_depth_loss": global_depth_loss,
                "local_depth_loss": local_depth_loss,
                "depth_loss": depth_loss,
                "render_loss": render_loss,
                "inv_contrastive_loss": inv_contrastive_loss,
                "inv_consistency_loss": inv_consistency_loss,
                "dep_contrastive_loss": dep_contrastive_loss,
                "dep_consistency_loss": dep_consistency_loss,
                "total_loss": total_loss,
            }
            if not torch.isfinite(total_loss).all():
                bad_terms = [
                    name
                    for name, value in finite_terms.items()
                    if not torch.isfinite(value.detach()).all().item()
                ]
                print(
                    f"[Warn] Non-finite loss at global_step={global_step} "
                    f"(bad={bad_terms}). Skipping optimizer step."
                )
                if wandb.run is not None:
                    wandb.log({"global_step": global_step, "train/nonfinite_batch": 1.0, "train/lr": current_lr}, step=global_step)
                global_step += 1
                continue

            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(
                trainable_parameters,
                max_norm=5.0,
                foreach=True,
            )
            optimizer.step()

            if should_log:
                print(
                    f"[Epoch {epoch + 1} | Step {step} | Global {global_step}] "
                    f"Loss={total_loss.item():.4f} lr={current_lr:.2e} "
                    f"(rgb={rgb_loss.item():.4f}, silhouette={silhouette_loss.item():.4f}, "
                    f"depth={depth_loss.item():.4f}, inv_con={inv_contrastive_loss.item():.4f}, "
                    f"dep_con={dep_contrastive_loss.item():.4f})"
                )
                if wandb.run is not None:
                    wandb.log(
                        {
                            "train/total_loss": total_loss.item(),
                            "train/render_loss": render_loss.item(),
                            "train/lr": current_lr,
                            "train/rgb_reconstruction_loss": rgb_loss.item(),
                            "train/silhouette_foreground_loss": rec_out["silhouette_foreground_loss"].item(),
                            "train/silhouette_background_loss": rec_out["silhouette_background_loss"].item(),
                            "train/silhouette_loss": silhouette_loss.item(),
                            "train/global_depth_loss": global_depth_loss.item(),
                            "train/local_depth_loss": local_depth_loss.item(),
                            "train/depth_loss": depth_loss.item(),
                            "train/control_motion01_mean": rec_out["control_motion01_mean"].item(),
                            "train/control_motion12_mean": rec_out["control_motion12_mean"].item(),
                            "train/dense_motion_mean": rec_out["dense_motion_mean"].item(),
                            "train/mean_valid_gaussian_opacity": rec_out["mean_valid_gaussian_opacity"].item(),
                            "train/inv_contrastive_loss": inv_contrastive_loss.item(),
                            "train/inv_consistency_loss": inv_consistency_loss.item(),
                            "train/dep_contrastive_loss": dep_contrastive_loss.item(),
                            "train/dep_consistency_loss": dep_consistency_loss.item(),
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
