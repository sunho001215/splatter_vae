from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Sequence

import torch
from torch.utils.data import DataLoader

from models.train_config import TrainConfig


def normalize_lr_schedule(schedule: str) -> str:
    schedule = str(schedule or "constant").strip().lower().replace("-", "_")
    aliases = {
        "none": "constant",
        "off": "constant",
        "constant": "constant",
        "cosine": "warmup_cosine",
        "cosine_annealing": "warmup_cosine",
        "warmup_cosine": "warmup_cosine",
        "cosine_warmup": "warmup_cosine",
    }
    if schedule not in aliases:
        raise ValueError(f"Unknown lr_schedule={schedule!r}. Use one of: constant, warmup_cosine, cosine.")
    return aliases[schedule]


def resolve_lr_total_steps(cfg_train: TrainConfig, train_dataloader: DataLoader) -> int:
    if cfg_train.lr_total_steps is not None:
        total_steps = int(cfg_train.lr_total_steps)
    elif cfg_train.max_global_steps is not None:
        total_steps = int(cfg_train.max_global_steps)
    else:
        try:
            total_steps = int(cfg_train.num_epochs) * len(train_dataloader)
        except TypeError:
            total_steps = int(cfg_train.lr_warmup_steps) + 1
    return max(1, total_steps)


def compute_scheduled_lr(cfg_train: TrainConfig, global_step: int, total_steps: int) -> float:
    peak_lr = float(cfg_train.lr)
    schedule = normalize_lr_schedule(cfg_train.lr_schedule)
    if schedule == "constant":
        return peak_lr

    min_lr = float(cfg_train.min_lr)
    if min_lr > peak_lr:
        raise ValueError(f"min_lr ({min_lr}) must be <= lr ({peak_lr}).")

    step = max(0, int(global_step))
    warmup_steps = max(0, int(cfg_train.lr_warmup_steps))
    if warmup_steps > 0 and step < warmup_steps:
        return peak_lr * float(step + 1) / float(warmup_steps)

    decay_steps = max(1, int(total_steps) - warmup_steps)
    progress = min(1.0, max(0.0, float(step - warmup_steps) / float(decay_steps)))
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return min_lr + (peak_lr - min_lr) * cosine


def set_optimizer_lr(optimizer: torch.optim.Optimizer, lr: float) -> None:
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr


STAGE_PARAMETER_GROUPS = (
    "invariant_encoder",
    "dependent_encoder",
    "decoder",
    "motion_head",
)


@dataclass(frozen=True)
class TrainingStage:
    index: int
    name: str
    start_step: int
    end_step: int
    local_step: int
    duration: int
    temporal_weight: float


def stage_schedule_settings(cfg_train: TrainConfig) -> Dict[str, int | float | None]:
    """Serializable stage boundaries and LR settings stored in checkpoints."""
    keys = (
        "max_global_steps",
        "stage1_end_step",
        "stage2_end_step",
        "stage2_temporal_ramp_steps",
        "stage1_invariant_encoder_lr",
        "stage1_dependent_encoder_lr",
        "stage1_decoder_lr",
        "stage1_motion_lr",
        "stage1_warmup_steps",
        "stage1_min_lr",
        "stage2_invariant_encoder_lr",
        "stage2_dependent_encoder_lr",
        "stage2_decoder_lr",
        "stage2_motion_lr",
        "stage2_warmup_steps",
        "stage2_min_lr",
        "stage3_invariant_encoder_lr",
        "stage3_dependent_encoder_lr",
        "stage3_decoder_lr",
        "stage3_motion_lr",
        "stage3_warmup_steps",
        "stage3_min_lr",
    )
    return {key: getattr(cfg_train, key) for key in keys}


def validate_stage_schedule(cfg_train: TrainConfig) -> None:
    stage1_end = int(cfg_train.stage1_end_step)
    stage2_end = int(cfg_train.stage2_end_step)
    final_end = int(cfg_train.max_global_steps or (stage2_end + 250_000))
    if not (0 < stage1_end < stage2_end < final_end):
        raise ValueError(
            "Expected 0 < stage1_end_step < stage2_end_step < max_global_steps, got "
            f"{stage1_end}, {stage2_end}, and {final_end}."
        )
    if int(cfg_train.stage2_temporal_ramp_steps) < 0:
        raise ValueError("stage2_temporal_ramp_steps must be non-negative.")
    for stage_index in (1, 2, 3):
        warmup = int(getattr(cfg_train, f"stage{stage_index}_warmup_steps"))
        minimum = float(getattr(cfg_train, f"stage{stage_index}_min_lr"))
        if warmup < 0 or minimum < 0.0:
            raise ValueError(f"Stage {stage_index} warmup and minimum LR must be non-negative.")
        for group in STAGE_PARAMETER_GROUPS:
            field = "motion_lr" if group == "motion_head" else f"{group}_lr"
            peak = float(getattr(cfg_train, f"stage{stage_index}_{field}"))
            if peak < 0.0:
                raise ValueError(f"Stage {stage_index} {group} LR must be non-negative.")
            if peak > 0.0 and minimum > peak:
                raise ValueError(
                    f"Stage {stage_index} minimum LR {minimum:g} exceeds {group} peak LR {peak:g}."
                )


def resolve_training_stage(cfg_train: TrainConfig, global_step: int) -> TrainingStage:
    """Resolve the active curriculum stage from the next global optimizer step."""
    step = max(0, int(global_step))
    stage1_end = int(cfg_train.stage1_end_step)
    stage2_end = int(cfg_train.stage2_end_step)
    final_end = int(cfg_train.max_global_steps or (stage2_end + 250_000))
    if step < stage1_end:
        return TrainingStage(1, "base_gaussian_pretraining", 0, stage1_end, step, stage1_end, 0.0)
    if step < stage2_end:
        local_step = step - stage1_end
        ramp_steps = int(cfg_train.stage2_temporal_ramp_steps)
        temporal_weight = 1.0 if ramp_steps <= 0 else min(1.0, float(local_step) / float(ramp_steps))
        return TrainingStage(
            2,
            "base_anchored_motion",
            stage1_end,
            stage2_end,
            local_step,
            stage2_end - stage1_end,
            temporal_weight,
        )
    return TrainingStage(
        3,
        "joint_finetuning",
        stage2_end,
        final_end,
        step - stage2_end,
        max(1, final_end - stage2_end),
        1.0,
    )


def _local_warmup_cosine_lr(
    peak_lr: float,
    min_lr: float,
    local_step: int,
    duration: int,
    warmup_steps: int,
) -> float:
    peak = float(peak_lr)
    if peak <= 0.0:
        return 0.0
    step = max(0, int(local_step))
    warmup = max(0, int(warmup_steps))
    if warmup > 0 and step < warmup:
        return peak * float(step + 1) / float(warmup)
    decay_steps = max(1, int(duration) - warmup)
    progress = min(1.0, max(0.0, float(step - warmup) / float(decay_steps)))
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return float(min_lr) + (peak - float(min_lr)) * cosine


def compute_stage_group_lrs(cfg_train: TrainConfig, stage: TrainingStage) -> Dict[str, float]:
    prefix = f"stage{stage.index}"
    minimum = float(getattr(cfg_train, f"{prefix}_min_lr"))
    warmup = int(getattr(cfg_train, f"{prefix}_warmup_steps"))
    peaks = {
        "invariant_encoder": float(getattr(cfg_train, f"{prefix}_invariant_encoder_lr")),
        "dependent_encoder": float(getattr(cfg_train, f"{prefix}_dependent_encoder_lr")),
        "decoder": float(getattr(cfg_train, f"{prefix}_decoder_lr")),
        "motion_head": float(getattr(cfg_train, f"{prefix}_motion_lr")),
    }
    return {
        name: _local_warmup_cosine_lr(peak, minimum, stage.local_step, stage.duration, warmup)
        for name, peak in peaks.items()
    }


def set_optimizer_group_lrs(
    optimizer: torch.optim.Optimizer,
    group_lrs: Dict[str, float],
) -> None:
    seen = set()
    for param_group in optimizer.param_groups:
        name = str(param_group.get("name", ""))
        if name not in group_lrs:
            raise ValueError(f"Optimizer parameter group {name!r} has no staged LR.")
        param_group["lr"] = float(group_lrs[name])
        seen.add(name)
    missing = set(group_lrs) - seen
    if missing:
        raise ValueError(f"Optimizer is missing staged parameter groups: {sorted(missing)}")


def combine_stage_render_losses(
    timestep_render_losses: Sequence[torch.Tensor],
    stage: TrainingStage,
) -> torch.Tensor:
    """Apply the curriculum only after independent timestep losses exist."""
    if len(timestep_render_losses) < 1:
        raise ValueError("At least the t0 rendering loss is required.")
    if stage.index == 1:
        return timestep_render_losses[0]
    if len(timestep_render_losses) != 3:
        raise ValueError(f"Stage {stage.index} requires t0/t1/t2 losses, got {len(timestep_render_losses)}.")
    if stage.index == 2:
        future = 0.5 * (timestep_render_losses[1] + timestep_render_losses[2])
        return timestep_render_losses[0] + float(stage.temporal_weight) * future
    return torch.stack(tuple(timestep_render_losses)).mean()
