from __future__ import annotations

import math

from models.training.config import TrainConfig


def resolve_total_optimizer_steps(
    config: TrainConfig,
    batches_per_epoch: int,
) -> int:
    epoch_steps = math.ceil(
        int(batches_per_epoch) / int(config.gradient_accumulation_steps)
    )
    scheduled = max(1, int(config.num_epochs) * epoch_steps)
    return (
        min(scheduled, int(config.max_global_steps))
        if config.max_global_steps is not None
        else scheduled
    )


def resolve_warmup_steps(config: TrainConfig, total_steps: int) -> int:
    requested = max(
        int(config.warmup_steps),
        int(math.ceil(float(config.warmup_fraction) * int(total_steps))),
    )
    return min(max(0, requested), max(0, int(total_steps) - 1))


def cosine_learning_rate(
    step: int,
    *,
    peak_lr: float,
    min_lr: float,
    total_steps: int,
    warmup_steps: int,
) -> float:
    current = max(0, int(step))
    if warmup_steps > 0 and current < warmup_steps:
        return float(peak_lr) * float(current + 1) / float(warmup_steps)
    decay_steps = max(1, int(total_steps) - int(warmup_steps))
    progress = min(1.0, max(0.0, (current - int(warmup_steps)) / decay_steps))
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return float(min_lr) + (float(peak_lr) - float(min_lr)) * cosine
