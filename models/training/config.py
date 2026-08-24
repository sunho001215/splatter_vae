from __future__ import annotations

from dataclasses import dataclass


@dataclass
class TrainConfig:
    num_epochs: int = 100
    max_global_steps: int | None = 300_000
    reference_lr: float = 1.5e-4
    reference_batch_size: int = 256
    decoder_lr_multiplier: float = 2.0
    min_lr: float = 1.0e-6
    warmup_steps: int = 10_000
    warmup_fraction: float = 0.05
    weight_decay: float = 0.05
    adam_beta1: float = 0.9
    adam_beta2: float = 0.95
    gradient_clip_norm: float = 1.0
    gradient_accumulation_steps: int = 1
    bf16: bool = True
    tf32: bool = True

    rgb_l1_weight: float = 1.0
    ssim_weight: float = 0.2
    contrastive_weight: float = 1.0
    metric_depth_weight: float = 0.5
    scale_invariant_depth_weight: float = 0.25
    flow_weight: float = 0.1
    visibility_weight: float = 0.05
    gaussian_regularization_weight: float = 0.01
    synthetic_view_weight: float = 0.05

    contrastive_temperature: float = 0.1
    depth_confidence_threshold: float = 0.0
    scale_invariant_mean_weight: float = 1.0
    flow_pair_weights: tuple[float, float, float] = (0.4, 0.4, 0.2)
    flow_alpha_threshold: float = 0.01
    flow_smooth_l1_beta: float = 1.0

    novel_view_enabled: bool = False
    novel_view_probability: float = 0.20
    novel_view_warmup_steps: int = 50_000
    novel_view_minimum_confidence: float = 0.5
    novel_view_require_cached: bool = True
    novel_view_supervise_rgb: bool = True
    novel_view_supervise_metric_depth: bool = True
    novel_view_supervise_scale_invariant_depth: bool = True

    seed: int = 42
    checkpoint_dir: str = "/ws/data/ws/droid_splattervae/logs/checkpoints"
    resume_checkpoint: str | None = None
    validation_every_steps: int = 2_000
    checkpoint_every_steps: int = 5_000
    visualization_every_steps: int = 5_000
    scalar_log_every_steps: int = 50
    validation_batches: int = 8

    def __post_init__(self) -> None:
        positive = {
            "reference_lr": self.reference_lr,
            "reference_batch_size": self.reference_batch_size,
            "decoder_lr_multiplier": self.decoder_lr_multiplier,
            "gradient_clip_norm": self.gradient_clip_norm,
            "gradient_accumulation_steps": self.gradient_accumulation_steps,
        }
        if any(float(value) <= 0.0 for value in positive.values()):
            raise ValueError(f"Optimization values must be positive: {positive}")
        if not 0.0 <= float(self.min_lr) <= float(self.reference_lr):
            raise ValueError("min_lr must lie between zero and reference_lr.")
        if not 0.0 <= float(self.warmup_fraction) <= 1.0:
            raise ValueError("warmup_fraction must lie in [0,1].")
        if self.warmup_steps < 0 or self.num_epochs <= 0:
            raise ValueError(
                "Epoch count must be positive and warmup steps nonnegative."
            )
        weights = {
            name: value
            for name, value in self.__dict__.items()
            if name.endswith("_weight")
        }
        if any(float(value) < 0.0 for value in weights.values()):
            raise ValueError(f"Loss weights must be nonnegative: {weights}")
        if float(self.contrastive_temperature) <= 0.0:
            raise ValueError("contrastive_temperature must be positive.")
        if len(self.flow_pair_weights) != 3 or sum(self.flow_pair_weights) <= 0.0:
            raise ValueError("flow_pair_weights must contain three nonnegative values.")
        if not 0.0 <= float(self.novel_view_probability) <= 1.0:
            raise ValueError("novel_view_probability must lie in [0,1].")
        if self.novel_view_enabled and not any(
            (
                self.novel_view_supervise_rgb,
                self.novel_view_supervise_metric_depth,
                self.novel_view_supervise_scale_invariant_depth,
            )
        ):
            raise ValueError(
                "Enabled See3D training requires at least one synthetic supervision loss."
            )

    def learning_rates(self, effective_global_batch: int) -> tuple[float, float]:
        encoder_lr = (
            float(self.reference_lr)
            * int(effective_global_batch)
            / int(self.reference_batch_size)
        )
        return encoder_lr, encoder_lr * float(self.decoder_lr_multiplier)
