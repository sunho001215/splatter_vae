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
    silhouette_weight: float = 0.1
    global_depth_weight: float = 0.1
    local_depth_weight: float = 1.0
    local_depth_min_patch_size: int = 8
    local_depth_max_patch_size: int = 32
    local_depth_min_valid_pixels: int = 16

    num_motion_controls: int = 256
    motion_num_neighbors: int = 4
    motion_delta_max: float = 0.05

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
