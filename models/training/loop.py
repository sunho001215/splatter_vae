from __future__ import annotations

import os
import random
from collections.abc import Callable, Mapping
from contextlib import nullcontext
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.distributed as dist
from torch import nn
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader

from models.gaussian.parameterization import (
    SplatterConfig,
    WorldSpaceGaussianParameterization,
)
from models.splattervae.config import SPLATTERVAE_ARCHITECTURE
from models.splattervae.model import SplatterVAE
from models.training.config import TrainConfig
from models.training.distributed import (
    DistributedContext,
    distributed_barrier,
    move_to_device,
    reduce_scalar_metrics,
    unwrap_model,
)
from models.training.losses import cross_view_info_nce
from models.training.reconstruction import compute_droid_reconstruction
from models.training.schedules import (
    cosine_learning_rate,
    resolve_total_optimizer_steps,
    resolve_warmup_steps,
)
from models.training.validation import evaluate_droid


@dataclass
class TrainingState:
    epoch: int = 0
    next_batch_in_epoch: int = 0
    global_step: int = 0


def build_optimizer(
    model: SplatterVAE,
    config: TrainConfig,
    *,
    effective_global_batch: int,
) -> tuple[torch.optim.Optimizer, dict[str, float]]:
    encoder_lr, decoder_lr = config.learning_rates(effective_global_batch)
    groups = [
        {
            "name": "encoder",
            "params": list(model.encoder.parameters()),
            "lr": encoder_lr,
            "lr_multiplier": 1.0,
        },
        {
            "name": "gaussian_decoder",
            "params": list(model.gaussian_decoder.parameters()),
            "lr": decoder_lr,
            "lr_multiplier": float(config.decoder_lr_multiplier),
        },
        {
            "name": "contrastive_projector",
            "params": list(model.contrastive_projector.parameters()),
            "lr": decoder_lr,
            "lr_multiplier": float(config.decoder_lr_multiplier),
        },
    ]
    parameter_ids = [id(parameter) for group in groups for parameter in group["params"]]
    trainable_ids = [
        id(parameter) for parameter in model.parameters() if parameter.requires_grad
    ]
    if len(parameter_ids) != len(set(parameter_ids)) or set(parameter_ids) != set(
        trainable_ids
    ):
        raise RuntimeError(
            "Optimizer groups must cover every trainable model parameter exactly once."
        )
    kwargs: dict[str, Any] = {
        "lr": encoder_lr,
        "betas": (float(config.adam_beta1), float(config.adam_beta2)),
        "weight_decay": float(config.weight_decay),
    }
    if torch.cuda.is_available():
        kwargs["fused"] = True
    try:
        optimizer = torch.optim.AdamW(groups, **kwargs)
    except (TypeError, RuntimeError):
        kwargs.pop("fused", None)
        optimizer = torch.optim.AdamW(groups, **kwargs)
    return optimizer, {"encoder_lr": encoder_lr, "decoder_lr": decoder_lr}


def _set_learning_rates(
    optimizer: torch.optim.Optimizer,
    config: TrainConfig,
    step: int,
    total_steps: int,
    warmup_steps: int,
    peak_encoder_lr: float,
) -> tuple[float, float]:
    encoder_lr = cosine_learning_rate(
        step,
        peak_lr=peak_encoder_lr,
        min_lr=float(config.min_lr),
        total_steps=total_steps,
        warmup_steps=warmup_steps,
    )
    for group in optimizer.param_groups:
        group["lr"] = encoder_lr * float(group["lr_multiplier"])
    decoder_lr = encoder_lr * float(config.decoder_lr_multiplier)
    return encoder_lr, decoder_lr


def _checkpoint_payload(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    state: TrainingState,
    config: TrainConfig,
    rng_by_rank: list[dict[str, Any]],
) -> dict[str, Any]:
    unwrapped = unwrap_model(model)
    assert isinstance(unwrapped, SplatterVAE)
    return {
        "checkpoint_schema_version": 2,
        "architecture": SPLATTERVAE_ARCHITECTURE,
        "decoder_configuration": unwrapped.decoder_configuration(),
        "parameter_counts": unwrapped.parameter_counts(),
        "model": unwrapped.state_dict(),
        "optimizer": optimizer.state_dict(),
        "training_state": asdict(state),
        "train_config": asdict(config),
        "world_size": len(rng_by_rank),
        "rng_by_rank": rng_by_rank,
    }


def _capture_rng_state() -> dict[str, Any]:
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda": torch.cuda.get_rng_state() if torch.cuda.is_available() else None,
    }


def save_checkpoint(
    path: str | os.PathLike[str],
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    state: TrainingState,
    config: TrainConfig,
    context: DistributedContext,
) -> None:
    destination = Path(path).expanduser().resolve(strict=False)
    local_rng = _capture_rng_state()
    if dist.is_initialized():
        rng_by_rank: list[dict[str, Any] | None] = [None] * dist.get_world_size()
        dist.all_gather_object(rng_by_rank, local_rng)
        if any(item is None for item in rng_by_rank):
            raise RuntimeError(
                "Failed to gather distributed RNG states for checkpointing."
            )
        gathered_rng = [item for item in rng_by_rank if item is not None]
    else:
        gathered_rng = [local_rng]
    if context.is_main:
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_suffix(destination.suffix + ".partial")
        torch.save(
            _checkpoint_payload(model, optimizer, state, config, gathered_rng),
            temporary,
        )
        os.replace(temporary, destination)
    distributed_barrier()


def load_checkpoint(
    path: str | os.PathLike[str],
    model: nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
) -> TrainingState:
    checkpoint = torch.load(
        Path(path).expanduser().resolve(), map_location="cpu", weights_only=False
    )
    if checkpoint.get("architecture") != SPLATTERVAE_ARCHITECTURE:
        raise ValueError(
            f"Checkpoint architecture {checkpoint.get('architecture')!r} is not {SPLATTERVAE_ARCHITECTURE!r}."
        )
    unwrapped = unwrap_model(model)
    assert isinstance(unwrapped, SplatterVAE)
    if checkpoint.get("decoder_configuration") != unwrapped.decoder_configuration():
        raise ValueError(
            "Checkpoint grouped-decoder configuration does not match this run."
        )
    unwrapped.load_state_dict(checkpoint["model"], strict=True)
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint["optimizer"])
    rng_by_rank = checkpoint.get("rng_by_rank")
    if not isinstance(rng_by_rank, list) or not rng_by_rank:
        raise ValueError("DROID checkpoint does not contain per-rank RNG state.")
    current_world_size = dist.get_world_size() if dist.is_initialized() else 1
    if len(rng_by_rank) != current_world_size:
        raise ValueError(
            "Exact distributed resume requires the checkpoint world size to match: "
            f"saved={len(rng_by_rank)}, current={current_world_size}."
        )
    rank = dist.get_rank() if dist.is_initialized() else 0
    rng = rng_by_rank[rank]
    if rng:
        random.setstate(rng["python"])
        np.random.set_state(rng["numpy"])
        torch.set_rng_state(rng["torch_cpu"])
        if torch.cuda.is_available() and rng.get("torch_cuda") is not None:
            torch.cuda.set_rng_state(rng["torch_cuda"])
    return TrainingState(**checkpoint.get("training_state", {}))


def _feature_diagnostics(
    prediction: Mapping[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    cls = prediction["cls_tokens_by_view"].float()
    patches = prediction["current_patch_tokens_by_view"].float()
    return {
        "cls_feature_norm": cls.norm(dim=-1).mean().detach(),
        "patch_feature_norm": patches.norm(dim=-1).mean().detach(),
        "patch_token_variance": patches.var(dim=(0, 1, 2), unbiased=False)
        .mean()
        .detach(),
    }


def _all_ranks_finite(value: torch.Tensor) -> bool:
    finite = torch.isfinite(value.detach()).to(dtype=torch.int32)
    if dist.is_initialized():
        dist.all_reduce(finite, op=dist.ReduceOp.MIN)
    return bool(finite.item())


def train_droid(
    model: nn.Module,
    splatter_config: SplatterConfig,
    train_loader: DataLoader,
    validation_loader: DataLoader | None,
    config: TrainConfig,
    context: DistributedContext,
    *,
    per_gpu_logical_batch: int,
    logger: Any | None = None,
    visualization_callback: Callable[[int, dict[str, Any]], None] | None = None,
) -> TrainingState:
    unwrapped = unwrap_model(model)
    if not isinstance(unwrapped, SplatterVAE):
        raise TypeError("train_droid expects a SplatterVAE, optionally wrapped in DDP.")
    effective_batch = (
        int(per_gpu_logical_batch)
        * int(config.gradient_accumulation_steps)
        * int(context.world_size)
    )
    optimizer, peak_lrs = build_optimizer(
        unwrapped, config, effective_global_batch=effective_batch
    )
    parameterization = WorldSpaceGaussianParameterization(splatter_config).to(
        context.device
    )
    background = (
        torch.ones(3, device=context.device)
        if splatter_config.data.white_background
        else torch.zeros(3, device=context.device)
    )
    total_steps = resolve_total_optimizer_steps(config, len(train_loader))
    warmup_steps = resolve_warmup_steps(config, total_steps)
    state = TrainingState()
    if config.resume_checkpoint:
        distributed_barrier()
        state = load_checkpoint(config.resume_checkpoint, model, optimizer)
        distributed_barrier()
    checkpoint_root = Path(config.checkpoint_dir).expanduser().resolve(strict=False)
    if context.is_main:
        startup = {
            "per_gpu_logical_batch": int(per_gpu_logical_batch),
            "gradient_accumulation": int(config.gradient_accumulation_steps),
            "world_size": int(context.world_size),
            "effective_global_batch": effective_batch,
            "encoder_lr": peak_lrs["encoder_lr"],
            "decoder_lr": peak_lrs["decoder_lr"],
            "warmup_steps": warmup_steps,
            "total_steps": total_steps,
            **unwrapped.parameter_counts(),
            "position_initialization": unwrapped.position_initialization_diagnostics(),
        }
        print(f"[DROID startup] {startup}", flush=True)
        if logger is not None:
            logger.log(
                {f"startup/{key}": value for key, value in startup.items()},
                step=state.global_step,
            )

    optimizer.zero_grad(set_to_none=True)
    for epoch in range(state.epoch, int(config.num_epochs)):
        dataset = train_loader.dataset
        if hasattr(dataset, "set_epoch"):
            dataset.set_epoch(epoch)
        sampler = train_loader.sampler
        if hasattr(sampler, "set_epoch"):
            sampler.set_epoch(epoch)
        start_batch = state.next_batch_in_epoch if epoch == state.epoch else 0
        for batch_index, cpu_batch in enumerate(train_loader):
            if batch_index < start_batch:
                continue
            if state.global_step >= total_steps:
                break
            batch = move_to_device(cpu_batch, context.device)
            accumulation = int(config.gradient_accumulation_steps)
            should_step = ((batch_index + 1) % accumulation == 0) or (
                batch_index + 1 == len(train_loader)
            )
            synchronization = (
                model.no_sync()
                if isinstance(model, DistributedDataParallel) and not should_step
                else nullcontext()
            )
            with synchronization:
                autocast = (
                    torch.autocast("cuda", dtype=torch.bfloat16)
                    if config.bf16 and context.device.type == "cuda"
                    else nullcontext()
                )
                with autocast:
                    prediction = model(
                        batch["representation_histories"],
                        batch["representation_flows"],
                        batch["representation_validity"],
                    )
                    contrastive, contrast_metrics = cross_view_info_nce(
                        prediction["projected_cls_by_view"],
                        config.contrastive_temperature,
                    )
                reconstruction = compute_droid_reconstruction(
                    parameterization,
                    splatter_config,
                    prediction,
                    batch,
                    config,
                    motion_translation_max=unwrapped.motion_translation_max,
                    background_color=background,
                    return_renders=False,
                    synthetic_enabled=(
                        config.novel_view_enabled
                        and state.global_step >= int(config.novel_view_warmup_steps)
                        and random.random() < float(config.novel_view_probability)
                    ),
                )
                total_loss = (
                    reconstruction["loss"]
                    + float(config.contrastive_weight) * contrastive.float()
                )
                if not _all_ranks_finite(total_loss):
                    raise FloatingPointError(
                        f"Non-finite distributed loss at optimizer step {state.global_step}."
                    )
                (total_loss / accumulation).backward()
            state.next_batch_in_epoch = batch_index + 1
            if not should_step:
                continue
            encoder_lr, decoder_lr = _set_learning_rates(
                optimizer,
                config,
                state.global_step,
                total_steps,
                warmup_steps,
                peak_lrs["encoder_lr"],
            )
            gradient_norm = torch.nn.utils.clip_grad_norm_(
                unwrapped.parameters(), float(config.gradient_clip_norm), foreach=True
            )
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            state.global_step += 1
            scalar_values: dict[str, torch.Tensor | float] = {
                "train/loss": total_loss.detach(),
                "train/reconstruction_loss": reconstruction["loss"].detach(),
                "train/contrastive_loss": contrastive.detach(),
                "train/rgb_l1_loss": reconstruction["rgb_l1_loss"].detach(),
                "train/dssim_loss": reconstruction["dssim_loss"].detach(),
                "train/metric_depth_loss": reconstruction["metric_depth_loss"].detach(),
                "train/scale_invariant_depth_loss": reconstruction[
                    "scale_invariant_depth_loss"
                ].detach(),
                "train/flow_loss": reconstruction["flow_loss"].detach(),
                "train/visibility_loss": reconstruction["visibility_loss"].detach(),
                "train/synthetic_view_loss": reconstruction[
                    "synthetic_view_loss"
                ].detach(),
                "train/synthetic_applied_fraction": reconstruction[
                    "synthetic_applied_fraction"
                ].detach(),
                "train/gaussian_parent_x_mean": reconstruction["parent_xyz_x_mean"],
                "train/gaussian_parent_y_mean": reconstruction["parent_xyz_y_mean"],
                "train/gaussian_parent_z_mean": reconstruction["parent_xyz_z_mean"],
                "train/gaussian_child_offset_mean": reconstruction["child_offset_mean"],
                "train/gaussian_child_offset_max": reconstruction["child_offset_max"],
                "train/gaussian_scale_mean": reconstruction["gaussian_scale_mean"],
                "train/gaussian_mean_opacity": reconstruction["mean_opacity"],
                "train/gaussian_active_fraction": reconstruction[
                    "active_gaussian_fraction"
                ],
                "train/gaussian_out_of_frustum_fraction": reconstruction[
                    "out_of_frustum_fraction"
                ],
                "train/rendered_visible_pixel_fraction": reconstruction[
                    "rendered_visible_pixel_fraction"
                ],
                "train/motion_01_mean": reconstruction["motion_01_mean"],
                "train/motion_12_mean": reconstruction["motion_12_mean"],
                "train/peak_gpu_memory_gib": torch.cuda.max_memory_allocated(
                    context.device
                )
                / (1024**3),
                "train/gradient_norm": gradient_norm.detach(),
                "train/encoder_lr": encoder_lr,
                "train/decoder_lr": decoder_lr,
                **{f"train/{key}": value for key, value in contrast_metrics.items()},
                **{
                    f"train/{key}": value
                    for key, value in _feature_diagnostics(prediction).items()
                },
            }
            if state.global_step % max(1, int(config.scalar_log_every_steps)) == 0:
                reduced = reduce_scalar_metrics(scalar_values)
                if context.is_main:
                    print(f"[step {state.global_step}] {reduced}", flush=True)
                    if logger is not None:
                        logger.log(reduced, step=state.global_step)

            validation_due = (
                state.global_step % max(1, int(config.validation_every_steps)) == 0
            )
            visualization_due = (
                visualization_callback is not None
                and state.global_step % max(1, int(config.visualization_every_steps))
                == 0
            )
            if validation_loader is not None and (validation_due or visualization_due):
                validation_metrics, payload = evaluate_droid(
                    model,
                    parameterization,
                    splatter_config,
                    validation_loader,
                    config,
                    context,
                    background,
                )
                if context.is_main:
                    print(
                        f"[validation {state.global_step}] {validation_metrics}",
                        flush=True,
                    )
                    if logger is not None:
                        logger.log(validation_metrics, step=state.global_step)
                    if (
                        payload is not None
                        and visualization_callback is not None
                        and visualization_due
                    ):
                        visualization_callback(state.global_step, payload)
                model.train()

            if state.global_step % max(1, int(config.checkpoint_every_steps)) == 0:
                save_checkpoint(
                    checkpoint_root / f"step-{state.global_step:08d}.pt",
                    model,
                    optimizer,
                    state,
                    config,
                    context,
                )
                save_checkpoint(
                    checkpoint_root / "last.pt",
                    model,
                    optimizer,
                    state,
                    config,
                    context,
                )
        if state.global_step >= total_steps:
            break
        state.epoch = epoch + 1
        state.next_batch_in_epoch = 0
    save_checkpoint(
        checkpoint_root / "last.pt", model, optimizer, state, config, context
    )
    return state
