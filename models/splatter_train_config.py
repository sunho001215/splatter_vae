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

    rec_weight: float = 1.0
    ssim_weight: float = 0.2

    hard_depth_weight: float = 1.0
    soft_depth_weight: float = 1.0
    depth_local_weight: float = 0.1
    depth_global_weight: float = 1.0
    depth_error_tolerance: float = 0.2
    depth_patch_min: int = 5
    depth_patch_max: int = 17
    depth_valid_min_ratio: float = 0.5
    hard_depth_opacity: float = 0.95
    hard_depth_start_step: int = 0
    soft_depth_start_step: int = 1000

    vq_weight: float = 0.25
    inv_contrastive_weight: float = 1.0
    inv_consistency_weight: float = 0.5
    dep_contrastive_weight: float = 0.1
    dep_consistency_weight: float = 0.1
    frustum_weight: float = 5.0e-3

    temperature: float = 0.1

    eval_every: int = 1000
    save_every: int = 5000
    ckpt_dir: str = "./checkpoints"
    resume_from_last: bool = False
    resume_from_checkpoint: Optional[str] = None

    val_num_batches: int = 2
    val_max_vis: int = 8
    val_render_trajectory_videos: bool = True
    val_video_frames: int = 48
    val_video_fps: int = 20
    val_video_amplitude: float = 0.12
    val_video_focus_distance: float = 1.0
