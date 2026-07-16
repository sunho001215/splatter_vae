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
    rgb_loss_mask_dilation: int = 3
    occupancy_weight: float = 1.0
    occupancy_fixed_opacity: float = 0.8

    point_chamfer_weight: float = 5.0
    chamfer_huber_delta: float = 0.05
    delta_smooth_weight: float = 1.0e-3

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
    val_max_vis: int = 8
    val_log_pointclouds: bool = True
    val_pointcloud_max_points: int = 4096
    val_track_max_points: int = 8
    val_render_trajectory_videos: bool = True
    val_video_frames: int = 48
    val_video_fps: int = 20
    val_video_amplitude: float = 0.12
    val_video_focus_distance: float = 1.0
