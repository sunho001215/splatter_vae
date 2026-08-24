from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from models.splattervae.temporal import validate_temporal_anchor


@dataclass
class TrainConfig:
    """Training hyperparameters for world-space tri-plane SplatterVAE pretraining."""

    num_epochs: int = 50
    max_global_steps: Optional[int] = 300_000
    lr: float = 5e-4
    lr_schedule: str = "constant"
    lr_warmup_steps: int = 0
    min_lr: float = 1e-4
    lr_total_steps: Optional[int] = None
    device: str = "cuda"

    temporal_anchor: str = "t0"
    temporal_loss_ramp_steps: int = 20_000
    flow_weight: float = 0.1
    flow_alpha_threshold: float = 0.01
    flow_smooth_l1_beta: float = 1.0
    dynamic_region_weight: float = 1.0
    visibility_weight: float = 0.5

    rec_weight: float = 1.0
    ssim_weight: float = 0.2
    silhouette_weight: float = 0.1
    global_depth_weight: float = 1.0
    hard_depth_weight: float = 1.0
    soft_depth_weight: float = 1.0
    hard_depth_opacity: float = 0.95
    local_depth_weight: float = 0.1
    local_depth_min_patch_size: int = 8
    local_depth_max_patch_size: int = 32
    local_depth_min_valid_pixels: int = 16

    inv_contrastive_weight: float = 1.0
    inv_consistency_weight: float = 0.5
    temperature: float = 0.1

    eval_every: int = 1000
    save_every: int = 5000
    ckpt_dir: str = "./checkpoints"
    resume_from_last: bool = False
    resume_from_checkpoint: Optional[str] = None

    val_num_batches: int = 2
    # Ground-truth depth points shown alongside all decoded predictions.
    val_pointcloud_max_points: int = 4096
    scalar_log_every: int = 250
    use_segmentation_mask: bool = True

    def __post_init__(self) -> None:
        self.temporal_anchor = validate_temporal_anchor(self.temporal_anchor)
        if float(self.hard_depth_weight) < 0.0:
            raise ValueError("hard_depth_weight must be non-negative.")
        if float(self.soft_depth_weight) < 0.0:
            raise ValueError("soft_depth_weight must be non-negative.")
        if not 0.0 < float(self.hard_depth_opacity) <= 1.0:
            raise ValueError("hard_depth_opacity must be in (0, 1].")
