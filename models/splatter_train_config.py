from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass
class TrainConfig:
    """Training hyperparameters for SplatterVAE pretraining."""

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
    # ReViWo-style reconstruction coefficients.  The base self-reconstruction
    # term is always included once; these weights scale the three shuffled
    # latent variants in models/splatter_pretraining.py.
    shuffle_inv_rec_weight: float = 1.0
    shuffle_dep_rec_weight: float = 1.0
    shuffle_both_rec_weight: float = 1.0
    vq_weight: float = 0.25
    inv_contrastive_weight: float = 1.0
    inv_consistency_weight: float = 0.5
    dep_contrastive_weight: float = 0.1
    dep_consistency_weight: float = 0.1
    frustum_weight: float = 0.001

    temperature: float = 0.1

    eval_every: int = 1000
    save_every: int = 5000
    ckpt_dir: str = "./checkpoints"
    resume_from_last: bool = False

    val_num_batches: int = 2
    val_max_vis: int = 8
