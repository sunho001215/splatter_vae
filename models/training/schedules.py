from __future__ import annotations

import math

from torch.utils.data import DataLoader

from models.training.config import TrainConfig


def normalize_lr_schedule(schedule: str) -> str:
    normalized = str(schedule or "constant").strip().lower().replace("-", "_")
    aliases = {
        "none": "constant",
        "off": "constant",
        "constant": "constant",
        "cosine": "warmup_cosine",
        "cosine_annealing": "warmup_cosine",
        "warmup_cosine": "warmup_cosine",
        "cosine_warmup": "warmup_cosine",
    }
    if normalized not in aliases:
        raise ValueError(f"Unknown lr_schedule={schedule!r}; use constant or cosine.")
    return aliases[normalized]


def resolve_lr_total_steps(cfg_train: TrainConfig, train_dataloader: DataLoader) -> int:
    if cfg_train.lr_total_steps is not None:
        total_steps = int(cfg_train.lr_total_steps)
    elif cfg_train.max_global_steps is not None:
        total_steps = int(cfg_train.max_global_steps)
    else:
        total_steps = int(cfg_train.num_epochs) * len(train_dataloader)
    return max(1, total_steps)


def compute_scheduled_lr(cfg_train: TrainConfig, global_step: int, total_steps: int) -> float:
    peak_lr = float(cfg_train.lr)
    if normalize_lr_schedule(cfg_train.lr_schedule) == "constant":
        return peak_lr
    minimum_lr = float(cfg_train.min_lr)
    if not 0.0 <= minimum_lr <= peak_lr:
        raise ValueError(f"Expected 0 <= min_lr <= lr, got {minimum_lr} and {peak_lr}.")
    step = max(0, int(global_step))
    warmup = max(0, int(cfg_train.lr_warmup_steps))
    if warmup > 0 and step < warmup:
        return peak_lr * float(step + 1) / float(warmup)
    decay_steps = max(1, int(total_steps) - warmup)
    progress = min(1.0, max(0.0, float(step - warmup) / float(decay_steps)))
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return minimum_lr + (peak_lr - minimum_lr) * cosine


def temporal_loss_ramp(global_step: int, ramp_steps: int) -> float:
    if int(ramp_steps) <= 0:
        return 1.0
    return min(1.0, max(0.0, float(global_step) / float(ramp_steps)))
