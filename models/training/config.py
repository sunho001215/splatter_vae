from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass
class TrainConfig:
    """Training hyperparameters for direct-Gaussian SplatterVAE pretraining."""

    num_epochs: int = 50
    max_global_steps: Optional[int] = None
    lr: float = 1e-4
    lr_schedule: str = "constant"
    lr_warmup_steps: int = 0
    min_lr: float = 0.0
    lr_total_steps: Optional[int] = None
    device: str = "cuda"

    temporal_loss_ramp_steps: int = 20_000
    flow_weight: float = 0.1
    flow_alpha_threshold: float = 0.01
    flow_smooth_l1_beta: float = 1.0
    dynamic_region_weight: float = 1.0
    frustum_weight: float = 0.01

    rec_weight: float = 1.0
    ssim_weight: float = 0.2
    silhouette_weight: float = 0.1
    global_depth_weight: float = 0.1
    local_depth_weight: float = 1.0
    local_depth_min_patch_size: int = 8
    local_depth_max_patch_size: int = 32
    local_depth_min_valid_pixels: int = 16

    inv_contrastive_weight: float = 1.0
    inv_consistency_weight: float = 0.5
    dep_contrastive_weight: float = 0.1
    dep_consistency_weight: float = 0.1

    temperature: float = 0.1

    eval_every: int = 1000
    save_every: int = 5000
    ckpt_dir: str = "./checkpoints"
    resume_from_last: bool = False
    resume_from_checkpoint: Optional[str] = None

    val_num_batches: int = 2
    val_pointcloud_max_points: int = 4096
    scalar_log_every: int = 250
