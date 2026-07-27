from __future__ import annotations

import os
from typing import Dict, Optional

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
    STAGE_PARAMETER_GROUPS,
    combine_stage_render_losses,
    compute_stage_group_lrs,
    resolve_training_stage,
    set_optimizer_group_lrs,
    stage_schedule_settings,
    validate_stage_schedule,
)


def _build_converter(splatter_cfg: SplatterConfig, device: torch.device) -> DirectSplatterToGaussians:
    return DirectSplatterToGaussians(splatter_cfg).to(device)


def _staged_parameter_groups(vae: SplatterVAE) -> list[Dict]:
    """Partition every trainable VAE parameter into one stable optimizer group."""
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
        "decoder": (
            "spatial_queries",
            "decoder_backbone.",
            "decoder.",
        ),
        "motion_head": ("motion_head.",),
    }
    grouped = {name: [] for name in STAGE_PARAMETER_GROUPS}
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
            continue
        grouped[matches[0]].append(parameter)
    if unmatched:
        raise RuntimeError(f"Could not uniquely assign staged parameters: {unmatched}")
    if any(not grouped[name] for name in STAGE_PARAMETER_GROUPS):
        empty = [name for name in STAGE_PARAMETER_GROUPS if not grouped[name]]
        raise RuntimeError(f"Staged optimizer groups are empty: {empty}")

    flattened = [parameter for name in STAGE_PARAMETER_GROUPS for parameter in grouped[name]]
    if len({id(parameter) for parameter in flattened}) != len(flattened):
        raise RuntimeError("A parameter was assigned to more than one staged optimizer group.")
    expected = [parameter for parameter in vae.parameters() if parameter.requires_grad]
    if {id(parameter) for parameter in flattened} != {id(parameter) for parameter in expected}:
        raise RuntimeError("Staged optimizer groups do not cover every trainable VAE parameter.")
    return [{"name": name, "params": grouped[name]} for name in STAGE_PARAMETER_GROUPS]


def _build_staged_optimizer(
    vae: SplatterVAE,
    cfg_train: TrainConfig,
    device: torch.device,
) -> tuple[torch.optim.Optimizer, list[torch.nn.Parameter]]:
    stage = resolve_training_stage(cfg_train, 0)
    group_lrs = compute_stage_group_lrs(cfg_train, stage)
    parameter_groups = _staged_parameter_groups(vae)
    for group in parameter_groups:
        group["lr"] = group_lrs[group["name"]]
    trainable_parameters = [
        parameter for group in parameter_groups for parameter in group["params"]
    ]
    optimizer_kwargs = {"lr": 0.0}
    if device.type == "cuda":
        optimizer_kwargs["fused"] = True
    try:
        optimizer = torch.optim.Adam(parameter_groups, **optimizer_kwargs)
    except (TypeError, RuntimeError):
        optimizer_kwargs.pop("fused", None)
        optimizer = torch.optim.Adam(parameter_groups, **optimizer_kwargs)
    return optimizer, trainable_parameters


def _assert_resume_stage_settings(ckpt: Dict, cfg_train: TrainConfig) -> None:
    saved = ckpt.get("stage_schedule_settings", None)
    if saved is None:
        print("[Resume] Legacy checkpoint has no staged schedule metadata; using the current config.")
        return
    current = stage_schedule_settings(cfg_train)
    mismatches = {
        key: (saved.get(key), current.get(key))
        for key in current
        if saved.get(key) != current.get(key)
    }
    if mismatches:
        raise ValueError(
            "Checkpoint stage schedule differs from the current config; refusing an ambiguous resume: "
            f"{mismatches}"
        )


def train_splatter_vae(
    vae: SplatterVAE,
    splatter_cfg: SplatterConfig,
    train_dataloader: DataLoader,
    cfg_train: TrainConfig,
    valid_dataloader: Optional[DataLoader] = None,
    resume_ckpt: Optional[str] = None,
):
    """Train with base pretraining, detached-anchor motion, and joint fine-tuning."""
    validate_stage_schedule(cfg_train)
    device = torch.device(cfg_train.device)
    vae.to(device)
    splatter_to_gaussians = _build_converter(splatter_cfg, device)
    optimizer, trainable_parameters = _build_staged_optimizer(vae, cfg_train, device)
    print(
        "[Curriculum] "
        f"Stage 1: [0,{cfg_train.stage1_end_step}), "
        f"Stage 2: [{cfg_train.stage1_end_step},{cfg_train.stage2_end_step}), "
        f"Stage 3: [{cfg_train.stage2_end_step},{cfg_train.max_global_steps}); "
        f"temporal ramp={cfg_train.stage2_temporal_ramp_steps} steps"
    )

    bg = torch.ones(3, device=device) if splatter_cfg.data.white_background else torch.zeros(3, device=device)
    start_epoch = 0
    global_step = 0

    if resume_ckpt is not None and os.path.isfile(resume_ckpt):
        ckpt = torch.load(resume_ckpt, map_location="cpu")
        _assert_resume_stage_settings(ckpt, cfg_train)
        vae.load_state_dict(ckpt["vae_state_dict"])
        try:
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        except ValueError as exc:
            print(f"[Resume] Skipping optimizer state because parameter groups changed: {exc}")
        start_epoch = int(ckpt["epoch"])
        global_step = int(ckpt.get("next_global_step", ckpt["global_step"]))
        resumed_stage = resolve_training_stage(cfg_train, global_step)
        print(
            f"[Resume] next_global_step={global_step}, stage={resumed_stage.name}, "
            f"local_stage_step={resumed_stage.local_step}, "
            f"temporal_weight={resumed_stage.temporal_weight:.4f}"
        )

    os.makedirs(cfg_train.ckpt_dir, exist_ok=True)

    epoch = start_epoch
    previous_stage_index: Optional[int] = None
    while True:
        for step, batch in enumerate(train_dataloader):
            if cfg_train.max_global_steps is not None and global_step >= cfg_train.max_global_steps:
                print(f"[Stop] Reached max_global_steps={cfg_train.max_global_steps}.")
                return

            stage = resolve_training_stage(cfg_train, global_step)
            group_lrs = compute_stage_group_lrs(cfg_train, stage)
            set_optimizer_group_lrs(optimizer, group_lrs)
            if previous_stage_index != stage.index:
                print(
                    f"[Curriculum] Entering Stage {stage.index} ({stage.name}) at global_step={global_step}; "
                    f"local_step={stage.local_step}, LRs={group_lrs}"
                )
                previous_stage_index = stage.index

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
            should_log = global_step % 250 == 0
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
                training_stage=stage.index,
            )
            timestep_losses = rec_out["timestep_losses"]
            render_loss = combine_stage_render_losses(
                [losses_t["render_loss"] for losses_t in timestep_losses],
                stage,
            )

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
            representation_loss = (
                cfg_train.inv_contrastive_weight * inv_contrastive_loss
                + cfg_train.inv_consistency_weight * inv_consistency_loss
                + cfg_train.dep_contrastive_weight * dep_contrastive_loss
                + cfg_train.dep_consistency_weight * dep_consistency_loss
            )
            total_loss = render_loss + representation_loss

            finite_terms = {
                "render_loss": render_loss,
                "representation_loss": representation_loss,
                "inv_contrastive_loss": inv_contrastive_loss,
                "inv_consistency_loss": inv_consistency_loss,
                "dep_contrastive_loss": dep_contrastive_loss,
                "dep_consistency_loss": dep_consistency_loss,
                "total_loss": total_loss,
            }
            for time_idx, losses_t in enumerate(timestep_losses):
                finite_terms.update(
                    {f"{name}_t{time_idx}": value for name, value in losses_t.items()}
                )
            if not torch.isfinite(total_loss).all():
                bad_terms = [
                    name
                    for name, value in finite_terms.items()
                    if not torch.isfinite(value.detach()).all().item()
                ]
                print(
                    f"[Warn] Non-finite loss at global_step={global_step} "
                    f"(stage={stage.name}, bad={bad_terms}). Skipping optimizer step."
                )
                if wandb.run is not None:
                    wandb.log(
                        {
                            "global_step": global_step,
                            "train/nonfinite_batch": 1.0,
                            "train/stage": stage.index,
                        },
                        step=global_step,
                    )
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
                time_summary = ", ".join(
                    f"t{time_idx}={losses_t['render_loss'].item():.4f}"
                    for time_idx, losses_t in enumerate(timestep_losses)
                )
                print(
                    f"[Epoch {epoch + 1} | Batch {step} | Global {global_step}] "
                    f"Stage={stage.index}:{stage.name} local={stage.local_step} "
                    f"temporal={stage.temporal_weight:.3f} loss={total_loss.item():.4f} "
                    f"render({time_summary})"
                )
                if wandb.run is not None:
                    log_values: Dict[str, float | int | str] = {
                        "train/total_loss": total_loss.item(),
                        "train/render_loss": render_loss.item(),
                        "train/representation_loss": representation_loss.item(),
                        "train/stage": stage.index,
                        "train/stage_name": stage.name,
                        "train/stage_local_step": stage.local_step,
                        "train/temporal_loss_weight": stage.temporal_weight,
                        "train/inv_contrastive_loss": inv_contrastive_loss.item(),
                        "train/inv_consistency_loss": inv_consistency_loss.item(),
                        "train/dep_contrastive_loss": dep_contrastive_loss.item(),
                        "train/dep_consistency_loss": dep_consistency_loss.item(),
                        "train/mean_valid_gaussian_opacity": rec_out[
                            "mean_valid_gaussian_opacity"
                        ].item(),
                        "global_step": global_step,
                    }
                    for group_name, lr in group_lrs.items():
                        log_values[f"train/lr/{group_name}"] = lr
                    for time_idx, losses_t in enumerate(timestep_losses):
                        for name, value in losses_t.items():
                            log_values[f"train/{name}/t{time_idx}"] = value.item()
                    for name in (
                        "control_motion01_mean",
                        "control_motion12_mean",
                        "dense_motion_mean",
                        "motion_near_bound_fraction",
                    ):
                        if name in rec_out:
                            log_values[f"train/{name}"] = rec_out[name].item()
                    wandb.log(log_values, step=global_step)

            completed_steps = global_step + 1
            if (
                valid_dataloader is not None
                and cfg_train.eval_every > 0
                and completed_steps % cfg_train.eval_every == 0
            ):
                validate_and_log_wandb(
                    vae=vae,
                    splatter_cfg=splatter_cfg,
                    splatter_to_gaussians=splatter_to_gaussians,
                    valid_dataloader=valid_dataloader,
                    device=device,
                    bg=bg,
                    cfg_train=cfg_train,
                    global_step=completed_steps,
                )

            if cfg_train.save_every > 0 and completed_steps % cfg_train.save_every == 0:
                ckpt_path = os.path.join(
                    cfg_train.ckpt_dir, f"step_{completed_steps:08d}.pth"
                )
                resume_stage = resolve_training_stage(cfg_train, completed_steps)
                ckpt = {
                    "epoch": epoch,
                    "global_step": completed_steps,
                    "next_global_step": completed_steps,
                    "current_stage": resume_stage.name,
                    "current_stage_index": resume_stage.index,
                    "local_stage_step": resume_stage.local_step,
                    "completed_stage": stage.name,
                    "completed_local_stage_step": stage.local_step,
                    "stage_schedule_settings": stage_schedule_settings(cfg_train),
                    "vae_state_dict": vae.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                }
                torch.save(ckpt, ckpt_path)
                print(f"[Checkpoint] Saved checkpoint to {ckpt_path}")

            global_step = completed_steps

        epoch += 1
        if cfg_train.max_global_steps is None and epoch >= cfg_train.num_epochs:
            print(f"[Stop] Reached num_epochs={cfg_train.num_epochs}.")
            break
