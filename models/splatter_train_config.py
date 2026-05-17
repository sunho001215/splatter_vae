from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass
class TrainConfig:
    """Training hyperparameters for SplatterVAE pretraining."""

    num_epochs: int = 50
    max_global_steps: Optional[int] = None
    lr: float = 1e-4
    device: str = "cuda"

    rec_weight: float = 1.0
    ssim_weight: float = 0.2

    # Reconstruction still supervises the self render and the disentangling
    # shuffle renders. These weights scale the shuffled latent combinations.
    shuffle_inv_rec_weight: float = 1.0
    shuffle_dep_rec_weight: float = 1.0
    shuffle_both_rec_weight: float = 1.0

    kl_weight: float = 1.0e-4

    # VICReg is applied directly to z_inv_mu, with no projection head.
    inv_vicreg_weight: float = 1.0
    vicreg_sim_coeff: float = 25.0
    vicreg_std_coeff: float = 25.0
    vicreg_cov_coeff: float = 1.0
    vicreg_std_gamma: float = 1.0
    vicreg_eps: float = 1.0e-4

    # VCReg is applied to invariant vision-encoder intermediate patch tokens.
    inv_encoder_vcreg_weight: float = 0.1
    inv_encoder_vcreg_std_coeff: float = 25.0
    inv_encoder_vcreg_cov_coeff: float = 1.0
    inv_encoder_vcreg_std_gamma: float = 1.0
    inv_encoder_vcreg_eps: float = 1.0e-4
    inv_encoder_vcreg_cov_smooth_l1_delta: float = 1.0

    # z_dep_mu uses InfoNCE to identify camera viewpoint: positives are
    # same-camera/different-timestep pairs from one demo, negatives are
    # different-camera/same-timestep pairs.
    dep_infonce_weight: float = 1.0
    dep_infonce_temperature: float = 0.1

    # L_cross discourages shared covariance between z_inv_mu and z_dep_mu.
    cross_cov_weight: float = 1.0

    frustum_weight: float = 0.001

    eval_every: int = 1000
    save_every: int = 5000
    ckpt_dir: str = "./checkpoints"
    resume_from_last: bool = False

    val_num_batches: int = 2
    val_max_vis: int = 8
