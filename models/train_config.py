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

    # Three-stage base/motion curriculum. Stage end steps are exclusive.
    stage1_end_step: int = 150_000
    stage2_end_step: int = 250_000
    stage2_temporal_ramp_steps: int = 20_000

    stage1_invariant_encoder_lr: float = 3.0e-4
    stage1_dependent_encoder_lr: float = 3.0e-4
    stage1_decoder_lr: float = 3.0e-4
    stage1_motion_lr: float = 0.0
    stage1_warmup_steps: int = 10_000
    stage1_min_lr: float = 5.0e-5

    stage2_invariant_encoder_lr: float = 5.0e-5
    stage2_dependent_encoder_lr: float = 3.0e-5
    stage2_decoder_lr: float = 5.0e-5
    stage2_motion_lr: float = 3.0e-4
    stage2_warmup_steps: int = 5_000
    stage2_min_lr: float = 1.0e-5

    stage3_invariant_encoder_lr: float = 5.0e-5
    stage3_dependent_encoder_lr: float = 3.0e-5
    stage3_decoder_lr: float = 5.0e-5
    stage3_motion_lr: float = 1.5e-4
    stage3_warmup_steps: int = 5_000
    stage3_min_lr: float = 1.0e-5

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
    motion_delta_max: float = 0.5

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
