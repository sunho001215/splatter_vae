from __future__ import annotations

import os
from dataclasses import asdict
from contextlib import nullcontext
from typing import Dict, Optional

import torch
from torch.utils.data import DataLoader

import wandb

from models.training.reconstruction import (
    compute_reconstruction_and_renders,
    encode_per_view_sequence_batch,
)
from models.gaussian.parameterization import DirectSplatterToGaussians
from models.training.losses import compute_view_structured_representation_losses
from models.gaussian.parameterization import SplatterConfig
from models.training.config import TrainConfig
from models.splattervae.model import SplatterVAE
from models.training.schedules import compute_scheduled_lr, resolve_lr_total_steps, temporal_loss_ramp


OPTIMIZER_GROUPS = (
    "invariant_encoder",
    "dependent_encoder",
    "decoder_backbone",
    "base_gaussian_head",
    "dense_motion_head",
)


def _parameter_groups(vae: SplatterVAE) -> list[Dict]:
    prefixes = {
        "invariant_encoder": (
            "invariant_encoder.",
            "state_token",
            "temporal_embed",
            "state_norm.",
            "invariant_encoder_output_proj.",
        ),
        "dependent_encoder": (
            "dependent_encoder.",
            "dep_token",
            "dep_norm.",
            "dependent_encoder_output_proj.",
        ),
        "decoder_backbone": (
            "spatial_queries",
            "decoder_backbone.",
            "decoder_dpt_backbone.",
        ),
        "base_gaussian_head": ("base_gaussian_head.",),
        "dense_motion_head": ("dense_motion_head.",),
    }
    grouped = {name: [] for name in OPTIMIZER_GROUPS}
    unmatched = []
    for parameter_name, parameter in vae.named_parameters():
        if not parameter.requires_grad:
            continue
        matches = [
            group_name
            for group_name, group_prefixes in prefixes.items()
            if any(
                parameter_name == prefix or parameter_name.startswith(prefix)
                for prefix in group_prefixes
            )
        ]
        if len(matches) != 1:
            unmatched.append((parameter_name, matches))
        else:
            grouped[matches[0]].append(parameter)
    if unmatched:
        raise RuntimeError(f"Could not uniquely assign optimizer parameters: {unmatched}")
    if any(not grouped[name] for name in OPTIMIZER_GROUPS):
        raise RuntimeError("Every single-stage optimizer module group must be non-empty.")
    return [{"name": name, "params": grouped[name]} for name in OPTIMIZER_GROUPS]


def _build_optimizer_and_scheduler(
    vae: SplatterVAE,
    cfg_train: TrainConfig,
    train_dataloader: DataLoader,
    device: torch.device,
) -> tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR, list[torch.nn.Parameter]]:
    groups = _parameter_groups(vae)
    for group in groups:
        group["lr"] = float(cfg_train.lr)
    optimizer_kwargs = {"lr": float(cfg_train.lr)}
    if device.type == "cuda":
        optimizer_kwargs["fused"] = True
    try:
        optimizer = torch.optim.Adam(groups, **optimizer_kwargs)
    except (TypeError, RuntimeError):
        optimizer_kwargs.pop("fused", None)
        optimizer = torch.optim.Adam(groups, **optimizer_kwargs)
    total_steps = resolve_lr_total_steps(cfg_train, train_dataloader)
    peak_lr = max(float(cfg_train.lr), 1.0e-12)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=lambda step: compute_scheduled_lr(cfg_train, step, total_steps) / peak_lr,
    )
    parameters = [parameter for group in groups for parameter in group["params"]]
    return optimizer, scheduler, parameters


def _representation_loss(
    view_latents: Dict[str, torch.Tensor],
    cfg_train: TrainConfig,
) -> tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    inv_con, dep_con, inv_cons, dep_cons = compute_view_structured_representation_losses(
        s_inv_by_view=view_latents["s_inv_by_view"],
        z_dep_by_view=view_latents["z_dep_by_view"],
        temperature=cfg_train.temperature,
    )
    total = (
        cfg_train.inv_contrastive_weight * inv_con
        + cfg_train.inv_consistency_weight * inv_cons
        + cfg_train.dep_contrastive_weight * dep_con
        + cfg_train.dep_consistency_weight * dep_cons
    )
    return total, {
        "inv_contrastive": inv_con,
        "inv_consistency": inv_cons,
        "dep_contrastive": dep_con,
        "dep_consistency": dep_cons,
    }


def _core_log_values(
    rec_out: Dict,
    total_loss: torch.Tensor,
    render_loss: torch.Tensor,
    representation_loss: torch.Tensor,
    representation: Dict[str, torch.Tensor],
    ramp: float,
    optimizer: torch.optim.Optimizer,
    global_step: int,
) -> Dict[str, float | int]:
    return {
        "global_step": global_step,
        "train/core/total_loss": total_loss.item(),
        "train/core/render_loss": render_loss.item(),
        "train/core/flow_loss": rec_out["flow_loss"].item(),
        "train/core/representation_loss": representation_loss.item(),
        "train/render/t0": rec_out["render_loss_t0"].item(),
        "train/render/t1": rec_out["render_loss_t1"].item(),
        "train/render/t2": rec_out["render_loss_t2"].item(),
        "train/flow/epe_01": rec_out["flow_epe_01"].item(),
        "train/flow/epe_12": rec_out["flow_epe_12"].item(),
        "train/flow/epe_02": rec_out["flow_epe_02"].item(),
        "train/flow/visible_fraction": rec_out["flow_visible_fraction"].item(),
        "train/motion/translation_01_mean": rec_out["translation_01_mean"].item(),
        "train/motion/translation_12_mean": rec_out["translation_12_mean"].item(),
        "train/components/rgb": rec_out["rgb_loss"].item(),
        "train/components/silhouette": rec_out["silhouette_loss"].item(),
        "train/components/global_depth": rec_out["global_depth_loss"].item(),
        "train/components/local_depth": rec_out["local_depth_loss"].item(),
        "train/components/frustum": rec_out["frustum_loss"].item(),
        "train/representation/inv_contrastive": representation["inv_contrastive"].item(),
        "train/representation/inv_consistency": representation["inv_consistency"].item(),
        "train/representation/dep_contrastive": representation["dep_contrastive"].item(),
        "train/representation/dep_consistency": representation["dep_consistency"].item(),
        "train/gaussian/mean_opacity": rec_out["mean_valid_gaussian_opacity"].item(),
        "train/temporal_loss_ramp": float(ramp),
        "train/lr": float(optimizer.param_groups[0]["lr"]),
    }


def train_splatter_vae(
    vae: SplatterVAE,
    splatter_cfg: SplatterConfig,
    train_dataloader: DataLoader,
    cfg_train: TrainConfig,
    valid_dataloader: Optional[DataLoader] = None,
    resume_ckpt: Optional[str] = None,
) -> None:
    """Train every module jointly with one optimizer from the first batch."""
    device = torch.device(cfg_train.device)
    vae.to(device)
    converter = DirectSplatterToGaussians(splatter_cfg).to(device)
    optimizer, scheduler, trainable_parameters = _build_optimizer_and_scheduler(
        vae, cfg_train, train_dataloader, device
    )
    background = (
        torch.ones(3, device=device)
        if splatter_cfg.data.white_background
        else torch.zeros(3, device=device)
    )
    start_epoch = 0
    global_step = 0
    if resume_ckpt is not None and os.path.isfile(resume_ckpt):
        checkpoint = torch.load(resume_ckpt, map_location="cpu")
        vae.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        start_epoch = int(checkpoint.get("epoch", 0))
        global_step = int(checkpoint.get("global_step", 0))
        print(f"[Resume] global_step={global_step}, temporal_ramp={temporal_loss_ramp(global_step, cfg_train.temporal_loss_ramp_steps):.4f}")

    os.makedirs(cfg_train.ckpt_dir, exist_ok=True)
    epoch = start_epoch
    while True:
        for batch_idx, batch in enumerate(train_dataloader):
            if cfg_train.max_global_steps is not None and global_step >= cfg_train.max_global_steps:
                print(f"[Stop] Reached max_global_steps={cfg_train.max_global_steps}.")
                return
            vae.train()
            images_u8 = batch["images"].to(device, non_blocking=True)
            images_01 = images_u8.float().div_(255.0)
            images = images_01.mul(2.0).sub(1.0)
            optical_flows = batch["optical_flows"].to(
                device=device, dtype=torch.float32, non_blocking=True
            )
            depths = batch["depths"].to(device=device, dtype=torch.float32, non_blocking=True)
            masks = batch["masks"].to(device=device, non_blocking=True)
            intrinsics = batch["K"].to(device=device, dtype=torch.float32, non_blocking=True)
            c2w = batch["c2w"].to(device=device, dtype=torch.float32, non_blocking=True)
            w2c = batch["w2c"].to(device=device, dtype=torch.float32, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            autocast_context = (
                torch.autocast(device_type="cuda", dtype=torch.bfloat16)
                if device.type == "cuda"
                else nullcontext()
            )
            with autocast_context:
                view_latents = encode_per_view_sequence_batch(vae, images, optical_flows)
                batch_size, views = view_latents["s_inv_by_view"].shape[:2]
                source_indices = torch.randint(views, (batch_size,), device=device)
                batch_ids = torch.arange(batch_size, device=device)
                raw_outputs = vae.predict_raw_maps(
                    view_latents["s_inv_by_view"][batch_ids, source_indices],
                    view_latents["z_dep_by_view"][batch_ids, source_indices],
                )
            raw_base_map = raw_outputs["raw_base_map"].float()
            raw_motion_map = raw_outputs["raw_motion_map"].float()
            view_latents["s_inv_by_view"] = view_latents["s_inv_by_view"].float()
            view_latents["z_dep_by_view"] = view_latents["z_dep_by_view"].float()
            ramp = temporal_loss_ramp(global_step, cfg_train.temporal_loss_ramp_steps)
            should_log = global_step % max(1, int(cfg_train.scalar_log_every)) == 0
            rec_out = compute_reconstruction_and_renders(
                splatter_to_gaussians=converter,
                splatter_cfg=splatter_cfg,
                raw_base_map=raw_base_map,
                raw_motion_map=raw_motion_map,
                motion_translation_max=float(vae.motion_translation_max),
                images_01=images_01,
                optical_flows=optical_flows,
                depths=depths,
                masks=masks,
                intrinsics=intrinsics,
                c2w=c2w,
                w2c=w2c,
                bg=background,
                cfg_train=cfg_train,
                source_indices=source_indices,
                temporal_ramp=ramp,
                training=True,
                compute_diagnostics=should_log,
            )
            future_render = 0.5 * (rec_out["render_loss_t1"] + rec_out["render_loss_t2"])
            render_loss = rec_out["render_loss_t0"] + ramp * future_render
            representation_loss, representation = _representation_loss(view_latents, cfg_train)
            total_loss = (
                render_loss
                + ramp * float(cfg_train.flow_weight) * rec_out["flow_loss"]
                + representation_loss
                + float(cfg_train.frustum_weight) * rec_out["frustum_loss"]
            )
            if not torch.isfinite(total_loss):
                print(f"[Warn] Non-finite loss at global_step={global_step}; skipping batch.")
                global_step += 1
                continue
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable_parameters, max_norm=5.0, foreach=True)
            optimizer.step()
            scheduler.step()

            if should_log:
                print(
                    f"[Epoch {epoch + 1} | Batch {batch_idx} | Global {global_step}] "
                    f"ramp={ramp:.3f} loss={total_loss.item():.4f} "
                    f"render=({rec_out['render_loss_t0'].item():.4f}, "
                    f"{rec_out['render_loss_t1'].item():.4f}, {rec_out['render_loss_t2'].item():.4f}) "
                    f"flow={rec_out['flow_loss'].item():.4f}"
                )
                if wandb.run is not None:
                    wandb.log(
                        _core_log_values(
                            rec_out,
                            total_loss,
                            render_loss,
                            representation_loss,
                            representation,
                            ramp,
                            optimizer,
                            global_step,
                        ),
                        step=global_step,
                    )

            completed_steps = global_step + 1
            if (
                valid_dataloader is not None
                and cfg_train.eval_every > 0
                and completed_steps % cfg_train.eval_every == 0
            ):
                from models.training.validation import validate_and_log_wandb

                validate_and_log_wandb(
                    vae=vae,
                    splatter_cfg=splatter_cfg,
                    splatter_to_gaussians=converter,
                    valid_dataloader=valid_dataloader,
                    device=device,
                    bg=background,
                    cfg_train=cfg_train,
                    global_step=completed_steps,
                )
            if cfg_train.save_every > 0 and completed_steps % cfg_train.save_every == 0:
                checkpoint_path = os.path.join(
                    cfg_train.ckpt_dir,
                    f"step_{completed_steps:08d}.pth",
                )
                torch.save(
                    {
                        "epoch": epoch,
                        "global_step": completed_steps,
                        "model_state_dict": vae.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "scheduler_state_dict": scheduler.state_dict(),
                        "configuration": {
                            "train": asdict(cfg_train),
                            "splatter": asdict(splatter_cfg),
                        },
                    },
                    checkpoint_path,
                )
                print(f"[Checkpoint] Saved {checkpoint_path}")
            global_step = completed_steps
        epoch += 1
        if cfg_train.max_global_steps is None and epoch >= cfg_train.num_epochs:
            print(f"[Stop] Reached num_epochs={cfg_train.num_epochs}.")
            return
