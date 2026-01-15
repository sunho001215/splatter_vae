import os
import yaml
import math
import argparse
from dataclasses import dataclass
from typing import Optional, Dict, Any

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

import wandb
from torchvision.utils import make_grid

from models.vae import InvariantDependentSplatterVAE
from models.splatter import (
    VAESplatterToGaussians,
    SplatterConfig,
    SplatterDataConfig,
    SplatterModelConfig,
    render_predicted,
)
from models.losses import infonce_loss, compute_reconstruction_loss
from models.transformer import STTransConfig
from models.vae import CodebookConfig
from models.camera_predictor import CameraParamPredictor
from dataset.dataloader import build_train_valid_loaders_robosuite
from utils.general_utils import compute_intrinsics_errors, compute_rotation_translation_errors, set_random_seed

# -------------------------------------------------------------------------
# Small config for training hyperparameters
# -------------------------------------------------------------------------

@dataclass
class TrainConfig:
    # If max_global_steps is set, it will override num_epochs as the stopping condition
    num_epochs: int = 50
    max_global_steps: Optional[int] = None

    lr: float = 1e-4
    device: str = "cuda"
    # Loss weights
    rec_weight: float = 1.0
    ssim_weight: float = 0.2
    vq_weight: float = 0.25
    inv_contrastive_weight: float = 1.0
    dep_contrastive_weight: float = 0.1
    # Contrastive settings
    temperature: float = 0.1
    # How often to run validation rendering (in training steps)
    eval_every: int = 1000
    # How often to save checkpoints (in training steps)
    save_every: int = 5000
    # Where to save checkpoints
    ckpt_dir: str = "./checkpoints"
    resume_from_last: bool = False
    # Camera parameter prediction
    predict_camera_params: bool = True
    # Weight on T_ij * T_ji ≈ I pose cycle-consistency loss
    pose_cycle_weight: float = 1e-3
    # Step at which we start using cross-view & pose-related terms
    crossview_start_step: int = 10000

# -------------------------------------------------------------------------
# Shared helper functions
# -------------------------------------------------------------------------

def encode_triplet(
    vae: InvariantDependentSplatterVAE,
    image_i_t: torch.Tensor,
    image_j_t: torch.Tensor,
    image_i_t1: torch.Tensor,
):
    """
    Encode three images into invariant + dependent latents.

    Returns:
        z_inv_i_t, z_inv_j_t, z_inv_i_t1
        z_dep_i_t, z_dep_j_t, z_dep_i_t1
        inv_vq_loss, dep_vq_loss
    """
    z_inv_i_t, inv_loss_i,  z_dep_i_t,  dep_loss_i,  _ = vae.encode(image_i_t)
    z_inv_j_t, inv_loss_j,  z_dep_j_t,  dep_loss_j,  _ = vae.encode(image_j_t)
    z_inv_i_t1, inv_loss_i1, z_dep_i_t1, dep_loss_i1, _ = vae.encode(image_i_t1)

    # Average VQ loss across the three encodings
    inv_vq_loss = (inv_loss_i + inv_loss_j + inv_loss_i1) / 3.0
    dep_vq_loss = (dep_loss_i + dep_loss_j + dep_loss_i1) / 3.0

    return (
        z_inv_i_t, z_inv_j_t, z_inv_i_t1,
        z_dep_i_t, z_dep_j_t, z_dep_i_t1,
        inv_vq_loss, dep_vq_loss,
    )


def compute_contrastive_losses(
    z_inv_i_t: torch.Tensor,
    z_inv_j_t: torch.Tensor,
    z_inv_i_t1: torch.Tensor,
    z_dep_i_t: torch.Tensor,
    z_dep_j_t: torch.Tensor,
    z_dep_i_t1: torch.Tensor,
    temperature: float,
):
    """
    Compute invariant and dependent contrastive InfoNCE losses.

    Follows the same setup in both training and validation.
    """
    B = z_inv_i_t.shape[0]

    # Invariant branch: negatives mix temporal & cross-view features
    inv_neg_keys = torch.cat(
        [
            z_inv_i_t1.reshape(B, -1).unsqueeze(1),
            z_dep_i_t.reshape(B, -1).unsqueeze(1),
            z_dep_j_t.reshape(B, -1).unsqueeze(1),
        ],
        dim=1,
    )
    inv_contrastive_loss = infonce_loss(
        query=z_inv_i_t.reshape(B, -1),
        positive_keys=z_inv_j_t.reshape(B, -1),
        negative_keys=inv_neg_keys,
        temperature=temperature,
        negative_mode="mixed",
    )

    # Dependent branch: positives are same-camera next-time-step, negatives are other-view
    dep_contrastive_loss = infonce_loss(
        query=z_dep_i_t.reshape(B, -1),
        positive_keys=z_dep_i_t1.reshape(B, -1),
        negative_keys=z_dep_j_t.reshape(B, -1).unsqueeze(1),
        temperature=temperature,
        negative_mode="paired",
    )

    return inv_contrastive_loss, dep_contrastive_loss


def compute_reconstruction_and_renders(
    vae: InvariantDependentSplatterVAE,
    splatter_to_gaussians: VAESplatterToGaussians,
    splatter_cfg: SplatterConfig,
    image_i_t: torch.Tensor,
    image_j_t: torch.Tensor,
    gt_i_t_01: torch.Tensor,
    gt_j_t_01: torch.Tensor,
    z_inv_i_t: torch.Tensor,
    z_dep_i_t: torch.Tensor,
    z_dep_j_t: torch.Tensor,
    T_ij: torch.Tensor,
    K_i: torch.Tensor,
    K_j: torch.Tensor,
    bg: torch.Tensor,
    return_renders: bool = False,
    ssim_weight: float = 0.2,
) -> Dict[str, Any]:
    """
    Common reconstruction + rendering code used by both training and validation.

    Args:
        z_inv_i_t, z_dep_i_t, z_dep_j_t: latents
        T_ij, K_i, K_j: camera transforms/intrinsics (may be GT or predicted)
        gt_i_t_01, gt_j_t_01: ground truth images in [0,1]

    Returns:
        dict with:
            rec_loss, rec_loss_cross, rec_loss_self_i,
            rec_loss_shuffle, rec_loss_shuffle_j, rec_loss_shuffle_i_from_j,
            optionally renders for visualization.
    """
    device = image_i_t.device
    B, _, H, W = image_i_t.shape

    # ---------------------------------------------------------
    # 1) Decode image_i_t latents into Splatter Image (world == cam_i)
    # ---------------------------------------------------------
    splatter_i_t = vae.decode(z_inv_i_t, z_dep_i_t)  # (B, C_s, H, W)

    # Camera-to-world for cam_i is identity (we treat cam_i as world)
    source_cameras_view_to_world = torch.eye(4, device=device).view(1, 4, 4).repeat(B, 1, 1)
    source_cv2wT_quat = torch.tensor([1.0, 0.0, 0.0, 0.0], device=device).view(1, 4).repeat(B, 1)

    gaussian_pc = splatter_to_gaussians(
        splatter_i_t,
        source_cameras_view_to_world=source_cameras_view_to_world,
        source_cv2wT_quat=source_cv2wT_quat,
        intrinsics=K_i,
        activate_output=True,
    )

    # ---------------------------------------------------------
    # 2) Cross-view + self reconstruction in one batched call
    # ---------------------------------------------------------
    eye_4 = torch.eye(4, device=device).view(1, 1, 4, 4).repeat(B, 1, 1, 1)  # (B,1,4,4)
    T_ij_views = T_ij.view(B, 1, 4, 4)                                       # (B,1,4,4)

    world_view_ij = torch.cat([eye_4, T_ij_views], dim=1)   # (B,2,4,4)
    Ks_ij = torch.stack([K_i, K_j], dim=1)                  # (B,2,3,3)

    out_ij = render_predicted(
        pc=gaussian_pc,
        world_view_transform=world_view_ij,
        intrinsics=Ks_ij,
        bg_color=bg,
        cfg=splatter_cfg,
    )
    renders_ij = out_ij["render"]                           # (B,2,3,H,W)

    rendered_self_i = renders_ij[:, 0]                      # (B,3,H,W)
    rendered_from_j = renders_ij[:, 1]                      # (B,3,H,W)

    # Reconstruction losses vs GT
    rec_loss_self_i = compute_reconstruction_loss(rendered_self_i, gt_i_t_01, ssim_weight=ssim_weight)
    rec_loss_cross  = compute_reconstruction_loss(rendered_from_j, gt_j_t_01, ssim_weight=ssim_weight)

    # ---------------------------------------------------------
    # 3) Shuffle loss: world == cam_j
    # ---------------------------------------------------------
    splatter_shuffle_j = vae.decode(z_inv_i_t, z_dep_j_t)   # (B,C_s,H,W)

    source_cameras_view_to_world_j = torch.eye(4, device=device).view(1, 4, 4).repeat(B, 1, 1)
    source_cv2wT_quat_j = torch.tensor([1.0, 0.0, 0.0, 0.0], device=device).view(1, 4).repeat(B, 1)

    gaussian_pc_shuffle_j = splatter_to_gaussians(
        splatter_shuffle_j,
        source_cameras_view_to_world=source_cameras_view_to_world_j,
        source_cv2wT_quat=source_cv2wT_quat_j,
        intrinsics=K_j,
        activate_output=True,
    )

    # T_ji: invert T_ij (cam_j -> cam_i); view 0 = j, view 1 = i_from_j
    T_ji = torch.linalg.inv(T_ij)                           # (B,4,4)

    eye_4_j = torch.eye(4, device=device).view(1, 1, 4, 4).repeat(B, 1, 1, 1)
    T_ji_views = T_ji.view(B, 1, 4, 4)

    world_view_shuffle = torch.cat([eye_4_j, T_ji_views], dim=1)  # (B,2,4,4)
    Ks_shuffle = torch.stack([K_j, K_i], dim=1)                   # (B,2,3,3)

    out_shuffle = render_predicted(
        pc=gaussian_pc_shuffle_j,
        world_view_transform=world_view_shuffle,
        intrinsics=Ks_shuffle,
        bg_color=bg,
        cfg=splatter_cfg,
    )
    renders_shuffle = out_shuffle["render"]                 # (B,2,3,H,W)

    rendered_shuffle_j        = renders_shuffle[:, 0]       # (B,3,H,W)
    rendered_shuffle_i_from_j = renders_shuffle[:, 1]       # (B,3,H,W)

    rec_loss_shuffle_j = compute_reconstruction_loss(rendered_shuffle_j, gt_j_t_01, ssim_weight=ssim_weight)
    rec_loss_shuffle_i_from_j = compute_reconstruction_loss(rendered_shuffle_i_from_j, gt_i_t_01, ssim_weight=ssim_weight)

    rec_loss_shuffle = 0.5 * (rec_loss_shuffle_j + rec_loss_shuffle_i_from_j)

    # Combine all reconstruction terms
    rec_loss = (rec_loss_cross + rec_loss_self_i + rec_loss_shuffle) / 3.0

    out_dict: Dict[str, Any] = dict(
        rec_loss=rec_loss,
        rec_loss_cross=rec_loss_cross,
        rec_loss_self_i=rec_loss_self_i,
        rec_loss_shuffle=rec_loss_shuffle,
        rec_loss_shuffle_j=rec_loss_shuffle_j,
        rec_loss_shuffle_i_from_j=rec_loss_shuffle_i_from_j,
    )

    if return_renders:
        out_dict.update(
            dict(
                rendered_self_i=rendered_self_i,
                rendered_from_j=rendered_from_j,
                rendered_shuffle_j=rendered_shuffle_j,
                rendered_shuffle_i_from_j=rendered_shuffle_i_from_j,
            )
        )

    return out_dict


def predict_cameras_from_latents(
    camera_predictor: Optional[CameraParamPredictor],
    z_dep_i_t: torch.Tensor,
    z_dep_j_t: torch.Tensor,
):
    """
    Predict T_ij, T_ji, K_i, K_j from dependent latents, or return Nones
    if no camera_predictor is provided.

    Here we assume the predictor takes flattened view-dependent latents.
    """
    if camera_predictor is None:
        return None, None, None, None

    B = z_dep_i_t.shape[0]
    z_dep_i_flat = z_dep_i_t.reshape(B, -1)
    z_dep_j_flat = z_dep_j_t.reshape(B, -1)

    T_ij_pred, T_ji_pred, K_i_pred, K_j_pred = camera_predictor(
        z_dep_i_flat, z_dep_j_flat
    )
    return T_ij_pred, T_ji_pred, K_i_pred, K_j_pred


# -------------------------------------------------------------------------
# Helper: render validation samples, camera errors, and log to wandb
# -------------------------------------------------------------------------

@torch.no_grad()
def log_validation_images(
    vae: InvariantDependentSplatterVAE,
    splatter_cfg: SplatterConfig,
    splatter_to_gaussians: VAESplatterToGaussians,
    valid_dataloader: DataLoader,
    device: torch.device,
    bg: torch.Tensor,
    cfg_train: TrainConfig,
    camera_predictor: Optional[CameraParamPredictor],
    global_step: int,
    max_vis: int = 4,
):
    """
    Take one batch from valid_dataloader and:

      - Encode images into invariant & dependent latents
      - Render:
          1) rendered_i_from_i    : image_i_t → i-view (self-view)
          2) rendered_j_from_i    : image_i_t → j-view (cross-view)
          3) rendered_i_from_shuf : shuffled latents, rendered to i-view
          4) rendered_j_from_shuf : shuffled latents, rendered to j-view
          5) gt_i_t, gt_j_t       : ground truth images

      - Compute contrastive losses on the full batch
      - If camera_predictor is provided:
          - compute rotation & translation errors between T_ij_pred and T_ij_gt
          - compute intrinsics errors between K_pred and K_gt for i and j

    All images and scalars are logged to wandb.
    """
    # Put models in eval mode (no dropout, no grad)
    vae.eval()
    splatter_to_gaussians.eval()
    if camera_predictor is not None:
        camera_predictor.eval()

    # Get a single validation batch
    try:
        batch = next(iter(valid_dataloader))
    except StopIteration:
        return

    # -------------------- Move batch to device --------------------
    image_i_t  = batch["image_i_t"].to(device)   # (B,3,H,W) in [-1,1]
    image_j_t  = batch["image_j_t"].to(device)   # (B,3,H,W)
    image_i_t1 = batch["image_i_t1"].to(device)  # (B,3,H,W)

    # Ground-truth camera parameters (used for metrics, and as fallback if no predictor)
    T_ij_gt = batch["T_ij"].to(device)           # (B,4,4) cam_i -> cam_j
    K_i_gt  = batch["K_i"].to(device)            # (B,3,3)
    K_j_gt  = batch["K_j"].to(device)            # (B,3,3)

    B, _, H, W = image_i_t.shape
    n_vis = min(max_vis, B)

    # Convert original images from [-1,1] to [0,1]
    gt_i_t_01 = (image_i_t + 1.0) * 0.5   # (B,3,H,W)
    gt_j_t_01 = (image_j_t + 1.0) * 0.5   # (B,3,H,W)

    # -------------------- Encode triplet into latents --------------------
    (
        z_inv_i_t, z_inv_j_t, z_inv_i_t1,
        z_dep_i_t, z_dep_j_t, z_dep_i_t1,
        inv_vq_loss, dep_vq_loss,
    ) = encode_triplet(
        vae, image_i_t, image_j_t, image_i_t1
    )

    # -------------------- Contrastive losses on full batch --------------------
    inv_contrastive_loss, dep_contrastive_loss = compute_contrastive_losses(
        z_inv_i_t, z_inv_j_t, z_inv_i_t1,
        z_dep_i_t, z_dep_j_t, z_dep_i_t1,
        temperature=cfg_train.temperature,
    )

    # Metrics (initialized as None; filled only if predictor is available)
    angle_err_deg = None
    pos_err = None
    fx_i_err = fy_i_err = cx_i_err = cy_i_err = None
    fx_j_err = fy_j_err = cx_j_err = cy_j_err = None

    if camera_predictor is not None:
        # Predict cameras from view-dependent latents
        T_ij_pred, T_ji_pred, K_i_pred, K_j_pred = predict_cameras_from_latents(
            camera_predictor=camera_predictor,
            z_dep_i_t=z_dep_i_t,
            z_dep_j_t=z_dep_j_t,
        )

        # Compute camera pose errors (pred vs GT)
        angle_err_deg, pos_err = compute_rotation_translation_errors(
            T_pred=T_ij_pred,
            T_gt=T_ij_gt,
        )

        # Compute intrinsics errors (per view)
        fx_i_err, fy_i_err, cx_i_err, cy_i_err = compute_intrinsics_errors(
            K_pred=K_i_pred,
            K_gt=K_i_gt,
        )
        fx_j_err, fy_j_err, cx_j_err, cy_j_err = compute_intrinsics_errors(
            K_pred=K_j_pred,
            K_gt=K_j_gt,
        )

    # ----------------------------------------------------------------------
    # Reconstruction & rendering using *chosen* cameras (predicted or GT)
    # ----------------------------------------------------------------------
    rec_out = compute_reconstruction_and_renders(
        vae=vae,
        splatter_to_gaussians=splatter_to_gaussians,
        splatter_cfg=splatter_cfg,
        image_i_t=image_i_t,
        image_j_t=image_j_t,
        gt_i_t_01=gt_i_t_01,
        gt_j_t_01=gt_j_t_01,
        z_inv_i_t=z_inv_i_t,
        z_dep_i_t=z_dep_i_t,
        z_dep_j_t=z_dep_j_t,
        T_ij=T_ij_gt,
        K_i=K_i_gt,
        K_j=K_j_gt,
        bg=bg,
        return_renders=True,
        ssim_weight=cfg_train.ssim_weight,
    )

    # -------------------- Grab reconstruction losses --------------------
    rec_loss                  = rec_out["rec_loss"]
    rec_loss_cross            = rec_out["rec_loss_cross"]
    rec_loss_self_i           = rec_out["rec_loss_self_i"]
    rec_loss_shuffle          = rec_out["rec_loss_shuffle"]
    rec_loss_shuffle_j        = rec_out["rec_loss_shuffle_j"]
    rec_loss_shuffle_i_from_j = rec_out["rec_loss_shuffle_i_from_j"]

    # Grab the rendered images for visualization (first n_vis only)
    renders_i_from_i    = rec_out["rendered_self_i"][:n_vis]           # i -> i
    renders_j_from_i    = rec_out["rendered_from_j"][:n_vis]           # i -> j
    renders_j_from_shuf = rec_out["rendered_shuffle_j"][:n_vis]        # shuffled, j-view
    renders_i_from_shuf = rec_out["rendered_shuffle_i_from_j"][:n_vis] # shuffled, i-view
    gt_i_vis            = gt_i_t_01[:n_vis].detach()
    gt_j_vis            = gt_j_t_01[:n_vis].detach()

    # -------------------- Make image grids for wandb --------------------
    grid_i_from_i    = make_grid(renders_i_from_i.cpu(),    nrow=n_vis, normalize=True, value_range=(0, 1))
    grid_j_from_i    = make_grid(renders_j_from_i.cpu(),    nrow=n_vis, normalize=True, value_range=(0, 1))
    grid_i_from_shuf = make_grid(renders_i_from_shuf.cpu(), nrow=n_vis, normalize=True, value_range=(0, 1))
    grid_j_from_shuf = make_grid(renders_j_from_shuf.cpu(), nrow=n_vis, normalize=True, value_range=(0, 1))
    grid_gt_i        = make_grid(gt_i_vis.cpu(),            nrow=n_vis, normalize=True, value_range=(0, 1))
    grid_gt_j        = make_grid(gt_j_vis.cpu(),            nrow=n_vis, normalize=True, value_range=(0, 1))

    # -------------------- Prepare wandb log dict --------------------
    log_dict: Dict[str, Any] = {
        "val/render_i_from_i":    wandb.Image(grid_i_from_i),
        "val/render_j_from_i":    wandb.Image(grid_j_from_i),
        "val/render_i_from_shuf": wandb.Image(grid_i_from_shuf),
        "val/render_j_from_shuf": wandb.Image(grid_j_from_shuf),
        "val/gt_i_t":             wandb.Image(grid_gt_i),
        "val/gt_j_t":             wandb.Image(grid_gt_j),
        "val/inv_contrastive_loss": inv_contrastive_loss.item(),
        "val/dep_contrastive_loss": dep_contrastive_loss.item(),
        "val/rec_loss":              rec_loss.item(),
        "val/rec_loss_cross":        rec_loss_cross.item(),
        "val/rec_loss_self_i":       rec_loss_self_i.item(),
        "val/rec_loss_shuffle":      rec_loss_shuffle.item(),
        "val/rec_loss_shuffle_j":    rec_loss_shuffle_j.item(),
        "val/rec_loss_shuffle_i_from_j": rec_loss_shuffle_i_from_j.item(),
    }

    # Add camera error metrics only if predictor is present
    if camera_predictor is not None:
        log_dict.update(
            {
                "val/camera_angle_error_deg": angle_err_deg.item(),
                "val/camera_pos_error": pos_err.item(),
                "val/intrin_focus_error": 0.5 * (
                    fx_i_err.item() + fy_i_err.item()
                    + fx_j_err.item() + fy_j_err.item()
                ),
                "val/intrin_center_error": 0.5 * (
                    cx_i_err.item() + cy_i_err.item()
                    + cx_j_err.item() + cy_j_err.item()
                ),
            }
        )

    # -------------------- Log to wandb --------------------
    wandb.log(log_dict, step=global_step)

# -------------------------------------------------------------------------
# Main training loop (now with wandb + validation renders + checkpoints)
# -------------------------------------------------------------------------

def train_splatter_vae(
    vae: InvariantDependentSplatterVAE,
    splatter_cfg: SplatterConfig,
    train_dataloader: DataLoader,
    valid_dataloader: DataLoader,
    cfg_train: TrainConfig,
    camera_predictor: Optional[CameraParamPredictor] = None,
    resume_ckpt: Optional[str] = None,
):
    """
    Training loop with batched multi-view rendering via gsplat.

    For each batch:

      A) Cross-view + self reconstruction (from camera i)
      B) Shuffle loss (swap dependent latents)
      C) Contrastive + VQ losses
      D) Pose cycle-consistency loss for predicted cameras (if enabled)
      E) Periodic validation renders (also logs camera errors)
      F) Periodic checkpoint saving

    All scalar metrics are logged to wandb.
    """
    device = torch.device(cfg_train.device)
    vae.to(device)

    # Build Splatter -> Gaussian converter
    splatter_data_cfg = splatter_cfg.data
    splatter_to_gaussians = VAESplatterToGaussians(splatter_cfg).to(device)

    # Move camera predictor if provided
    if camera_predictor is not None:
        camera_predictor.to(device)

    # Optimizer: VAE + splatter_to_gaussians (+ camera_predictor if present)
    params = list(vae.parameters()) + list(splatter_to_gaussians.parameters())
    if camera_predictor is not None:
        params += list(camera_predictor.parameters())
    optimizer = torch.optim.Adam(params, lr=cfg_train.lr)

    # Resume from checkpoint
    start_epoch = 0
    global_step = 0
    if resume_ckpt is not None and os.path.isfile(resume_ckpt):
        print(f"[Resume] Loading checkpoint from: {resume_ckpt}")
        ckpt = torch.load(resume_ckpt, map_location=device, weight_only=False)

        vae.load_state_dict(ckpt["vae_state_dict"])
        splatter_to_gaussians.load_state_dict(ckpt["splatter_to_gaussians_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])

        # Camera predictor state is optional (for older checkpoints)
        if camera_predictor is not None and "camera_predictor_state_dict" in ckpt:
            camera_predictor.load_state_dict(ckpt["camera_predictor_state_dict"])

        start_epoch = ckpt.get("epoch", 0)
        global_step = ckpt.get("global_step", 0)
        print(f"[Resume] Resuming from epoch={start_epoch}, global_step={global_step}")

    # Background color (white or black)
    if splatter_data_cfg.white_background:
        bg = torch.ones(3, dtype=torch.float32, device=device)
    else:
        bg = torch.zeros(3, dtype=torch.float32, device=device)

    # Ensure checkpoint directory exists
    os.makedirs(cfg_train.ckpt_dir, exist_ok=True)

    epoch = start_epoch
    while True:
        for step, batch in enumerate(train_dataloader):

            # -----------------------------------------
            # Early exit if we've hit max_global_steps
            # -----------------------------------------
            if cfg_train.max_global_steps is not None and global_step >= cfg_train.max_global_steps:
                print(f"[Stop] Reached max_global_steps={cfg_train.max_global_steps}.")
                print("Training finished.")
                return

            # ---------------------------------------------------------
            # 0. Set models to train mode
            # ---------------------------------------------------------
            vae.train()
            splatter_to_gaussians.train()
            if camera_predictor is not None:
                camera_predictor.train()

            # ---------------------------------------------------------
            # 1. Move batch to device
            # ---------------------------------------------------------
            image_i_t  = batch["image_i_t"].to(device)   # (B,3,H,W)
            image_j_t  = batch["image_j_t"].to(device)   # (B,3,H,W)
            image_i_t1 = batch["image_i_t1"].to(device)  # (B,3,H,W)

            # Ground-truth camera params (only used for validation/metrics, not for training)
            T_ij_gt = batch["T_ij"].to(device)           # (B,4,4)
            K_i_gt  = batch["K_i"].to(device)            # (B,3,3)
            K_j_gt  = batch["K_j"].to(device)            # (B,3,3)

            B, _, H, W = image_i_t.shape

            # Images are in [-1,1]; convert to [0,1]
            gt_i_t_01 = (image_i_t + 1.0) * 0.5   # (B,3,H,W)
            gt_j_t_01 = (image_j_t + 1.0) * 0.5

            # ---------------------------------------------------------
            # 2. Encode three images into invariant + view-dependent latents
            # ---------------------------------------------------------
            (
                z_inv_i_t, z_inv_j_t, z_inv_i_t1,
                z_dep_i_t, z_dep_j_t, z_dep_i_t1,
                inv_vq_loss, dep_vq_loss,
            ) = encode_triplet(
                vae, image_i_t, image_j_t, image_i_t1
            )

            # ---------------------------------------------------------
            # 3. Predict camera parameters (if enabled)
            # ---------------------------------------------------------
            if camera_predictor is not None:
                T_ij_pred, T_ji_pred, K_i_pred, K_j_pred = predict_cameras_from_latents(
                    camera_predictor=camera_predictor,
                    z_dep_i_t=z_dep_i_t,
                    z_dep_j_t=z_dep_j_t,
                )
                if global_step < cfg_train.crossview_start_step:
                    T_ij_train = T_ij_pred.detach()
                    K_i_train  = K_i_pred.detach()
                    K_j_train  = K_j_pred.detach()
                else:
                    T_ij_train = T_ij_pred
                    K_i_train  = K_i_pred
                    K_j_train  = K_j_pred
            else:
                T_ij_train = T_ij_gt
                K_i_train  = K_i_gt
                K_j_train  = K_j_gt
                T_ji_pred  = None
                T_ij_pred  = None

            # ---------------------------------------------------------
            # 4. Reconstruction / rendering (shared helper)
            # ---------------------------------------------------------
            rec_out = compute_reconstruction_and_renders(
                vae=vae,
                splatter_to_gaussians=splatter_to_gaussians,
                splatter_cfg=splatter_cfg,
                image_i_t=image_i_t,
                image_j_t=image_j_t,
                gt_i_t_01=gt_i_t_01,
                gt_j_t_01=gt_j_t_01,
                z_inv_i_t=z_inv_i_t,
                z_dep_i_t=z_dep_i_t,
                z_dep_j_t=z_dep_j_t,
                T_ij=T_ij_train,
                K_i=K_i_train,
                K_j=K_j_train,
                bg=bg,
                return_renders=False,
                ssim_weight=cfg_train.ssim_weight,
            )

            rec_loss                  = rec_out["rec_loss"]
            rec_loss_cross            = rec_out["rec_loss_cross"]
            rec_loss_self_i           = rec_out["rec_loss_self_i"]
            rec_loss_shuffle          = rec_out["rec_loss_shuffle"]
            rec_loss_shuffle_j        = rec_out["rec_loss_shuffle_j"]
            rec_loss_shuffle_i_from_j = rec_out["rec_loss_shuffle_i_from_j"]

            # ---------------------------------------------------------
            # 5. Contrastive losses
            # ---------------------------------------------------------
            inv_contrastive_loss, dep_contrastive_loss = compute_contrastive_losses(
                z_inv_i_t, z_inv_j_t, z_inv_i_t1,
                z_dep_i_t, z_dep_j_t, z_dep_i_t1,
                temperature=cfg_train.temperature,
            )

            # ---------------------------------------------------------
            # 6. VQ loss
            # ---------------------------------------------------------
            vq_loss = inv_vq_loss + dep_vq_loss

            # ---------------------------------------------------------
            # 7. Pose cycle-consistency loss
            # ---------------------------------------------------------
            if camera_predictor is not None and T_ij_pred is not None and T_ji_pred is not None:
                comp_ij = torch.matmul(T_ij_pred, T_ji_pred)  # (B,4,4)
                I = torch.eye(4, device=device, dtype=comp_ij.dtype).unsqueeze(0)
                pose_cycle_loss = ((comp_ij - I) ** 2).mean()
            else:
                pose_cycle_loss = torch.tensor(0.0, device=device)

            # ---------------------------------------------------------
            # 8. Curriculum on loss terms
            # ---------------------------------------------------------
            if camera_predictor is not None and global_step < cfg_train.crossview_start_step:
                rec_loss_active = 0.5 * (rec_loss_self_i + rec_loss_shuffle_j)
                pose_cycle_weight = 0.0
            else:
                rec_loss_active = (
                    rec_loss_self_i
                    + rec_loss_shuffle_j
                    + rec_loss_cross
                    + rec_loss_shuffle_i_from_j
                ) / 4.0
                pose_cycle_weight = cfg_train.pose_cycle_weight

            total_loss = (
                cfg_train.rec_weight * rec_loss_active
                + cfg_train.vq_weight * vq_loss
                + cfg_train.inv_contrastive_weight * inv_contrastive_loss
                + cfg_train.dep_contrastive_weight * dep_contrastive_loss
                + pose_cycle_weight * pose_cycle_loss
            )

            optimizer.zero_grad(set_to_none=True)
            total_loss.backward()
            optimizer.step()

            # ---------------------------------------------------------
            # 9. wandb logging
            # ---------------------------------------------------------
            wandb.log(
                {
                    "train/total_loss": total_loss.item(),
                    "train/rec_loss_active": rec_loss_active.item(),
                    "train/rec_loss_all": rec_loss.item(),
                    "train/rec_cross": rec_loss_cross.item(),
                    "train/rec_self_i": rec_loss_self_i.item(),
                    "train/rec_shuffle": rec_loss_shuffle.item(),
                    "train/rec_shuffle_j": rec_loss_shuffle_j.item(),
                    "train/rec_shuffle_i_from_j": rec_loss_shuffle_i_from_j.item(),
                    "train/vq_loss": vq_loss.item(),
                    "train/inv_vq_loss": inv_vq_loss.item(),
                    "train/dep_vq_loss": dep_vq_loss.item(),
                    "train/inv_contrastive_loss": inv_contrastive_loss.item(),
                    "train/dep_contrastive_loss": dep_contrastive_loss.item(),
                    "train/pose_cycle_loss": pose_cycle_loss.item(),
                    "train/epoch": epoch,
                    "train/step_in_epoch": step,
                    "train/global_step": global_step,
                },
                step=global_step,
            )

            if step % 50 == 0:
                print(
                    f"[Epoch {epoch+1} | Step {step} | Global {global_step}] "
                    f"Loss={total_loss.item():.4f} "
                    f"(rec={rec_loss.item():.4f}, "
                    f"vq={vq_loss.item():.4f}, "
                    f"inv_con={inv_contrastive_loss.item():.4f}, "
                    f"dep_con={dep_contrastive_loss.item():.4f}, "
                    f"rec_cross={rec_loss_cross.item():.4f}, "
                    f"rec_self_i={rec_loss_self_i.item():.4f}, "
                    f"rec_shuffle_j={rec_loss_shuffle_j.item():.4f}, "
                    f"rec_shuffle_i_from_j={rec_loss_shuffle_i_from_j.item():.4f})"
                )

            # ---------------------------------------------------------
            # 10. Periodic validation
            # ---------------------------------------------------------
            if (
                cfg_train.eval_every > 0
                and global_step > 0
                and global_step % cfg_train.eval_every == 0
            ):
                log_validation_images(
                    vae=vae,
                    splatter_cfg=splatter_cfg,
                    splatter_to_gaussians=splatter_to_gaussians,
                    valid_dataloader=valid_dataloader,
                    device=device,
                    bg=bg,
                    cfg_train=cfg_train,
                    camera_predictor=camera_predictor,
                    global_step=global_step,
                )

            # ---------------------------------------------------------
            # 11. Periodic checkpoint saving
            # ---------------------------------------------------------
            if (
                cfg_train.save_every > 0
                and global_step > 0
                and global_step % cfg_train.save_every == 0
            ):
                ckpt_path = os.path.join(
                    cfg_train.ckpt_dir,
                    f"step_{global_step:08d}.pth",
                )

                ckpt = {
                    "epoch": epoch,
                    "global_step": global_step,
                    "vae_state_dict": vae.state_dict(),
                    "splatter_to_gaussians_state_dict": splatter_to_gaussians.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "splatter_cfg": splatter_cfg,
                    "train_cfg": cfg_train,
                }
                if camera_predictor is not None:
                    ckpt["camera_predictor_state_dict"] = camera_predictor.state_dict()

                torch.save(ckpt, ckpt_path)
                print(f"[Checkpoint] Saved checkpoint to {ckpt_path}")

            global_step += 1

        # Completed one pass over train_dataloader
        epoch += 1

        # Fallback: if max_global_steps is not set, stop by num_epochs
        if cfg_train.max_global_steps is None and epoch >= cfg_train.num_epochs:
            print(f"[Stop] Reached num_epochs={cfg_train.num_epochs}.")
            break

    print("Training finished.")



# -------------------------------------------------------------------------
# main(): wiring everything together
# -------------------------------------------------------------------------

def main():
    """
    Entry point for training the splatter VAE.

    - Reads all hyperparameters and paths from a single YAML config.
    - Builds train/valid dataloaders from manifests + cameras.json
    - Configures Splatter (camera / Gaussian) parameters
    - Instantiates a ReViWo-style InvariantDependentSplatterVAE
      whose decoder outputs a Splatter Image instead of RGB
    - Optionally instantiates a camera predictor to learn T_ij & intrinsics
    - Runs training with wandb logging and periodic checkpoints
    """

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to YAML config file."
    )
    cli_args = parser.parse_args()

    # -------------------------------------------------
    # 0) Load YAML config
    # -------------------------------------------------
    with open(cli_args.config, "r") as f:
        cfg = yaml.safe_load(f)

    # -------------------------------------------------
    # 1) Dataset config + dataloaders (UPDATED for MimicGen HDF5)
    # -------------------------------------------------
    ds_cfg = cfg.get("dataset", {})

    dataset_path = ds_cfg.get("hdf5_path", None)
    if dataset_path is None:
        raise ValueError('Config "dataset.hdf5_path" must be provided for MimicGen HDF5 loading.')

    batch_size = ds_cfg.get("batch_size", 128)
    num_workers = ds_cfg.get("num_workers", 8)
    pin_memory = ds_cfg.get("pin_memory", True)

    # Optional split controls
    train_ratio = float(ds_cfg.get("train_ratio", 0.90))
    seed = int(ds_cfg.get("seed", 42))
    num_episodes = ds_cfg.get("num_episodes", None)
    max_frames_per_demo = ds_cfg.get("max_frames_per_demo", None)

    # Set random seed for reproducibility
    set_random_seed(seed)

    # Build dataloaders
    train_loader, valid_loader = build_train_valid_loaders_robosuite(
        dataset_path=dataset_path,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        train_ratio=train_ratio,
        seed=seed,
        num_episodes=num_episodes,
        max_frames_per_demo=max_frames_per_demo,
    )

    # Peek a batch to get input resolution (H, W) (same as before)
    sample_batch = next(iter(train_loader))
    _, _, H, W = sample_batch["image_i_t"].shape
    print(f"[Info] Training image resolution: H={H}, W={W}")

    # -------------------------------------------------
    # 2) SplatterConfig (camera / Gaussian setup) from YAML
    # -------------------------------------------------
    spl_cfg = cfg.get("splatter", {})
    spl_data_cfg_dict = spl_cfg.get("data", {})
    spl_model_cfg_dict = spl_cfg.get("model", {})

    splatter_data_cfg = SplatterDataConfig(**spl_data_cfg_dict)
    # Attach resolution attributes needed by VAESplatterToGaussians
    splatter_data_cfg.img_height = H
    splatter_data_cfg.img_width = W

    splatter_model_cfg = SplatterModelConfig(**spl_model_cfg_dict)

    splatter_cfg = SplatterConfig(
        data=splatter_data_cfg,
        model=splatter_model_cfg,
    )

    # Number of channels required by the splatter decoder output
    tmp_converter = VAESplatterToGaussians(splatter_cfg)
    splatter_channels = tmp_converter.num_splatter_channels()
    del tmp_converter

    # -------------------------------------------------
    # 3) Transformer / Swin + codebook + VAE config
    # -------------------------------------------------
    # Codebook configs (invariant / dependent)
    cb_cfg = cfg.get("codebook", {})
    inv_cb_cfg_dict = cb_cfg.get("invariant", {})
    dep_cb_cfg_dict = cb_cfg.get("dependent", {})

    inv_cb_cfg = CodebookConfig(**inv_cb_cfg_dict)
    dep_cb_cfg = CodebookConfig(**dep_cb_cfg_dict)

    # Model-level config (fusion style, VQ switches, etc.)
    model_cfg = cfg.get("model", {})
    fusion_style = model_cfg.get("fusion_style", "cat")
    use_dependent_vq = model_cfg.get("use_dependent_vq", True)
    is_dependent_ae = model_cfg.get("is_dependent_ae", False)
    use_invariant_vq = model_cfg.get("use_invariant_vq", True)
    is_invariant_ae = model_cfg.get("is_invariant_ae", False)

    # Swin backbone configuration (all architecture hyper-parameters live here)
    swin_cfg_dict = cfg.get("swin", {})

    # If 'patch_size' is not explicitly provided in the 'swin' section,
    # fall back to the previous grid_tokens_h/w interface:
    if "patch_size" not in swin_cfg_dict:
        grid_tokens_h = model_cfg.get("grid_tokens_h", 8)
        grid_tokens_w = model_cfg.get("grid_tokens_w", 8)

        assert H % grid_tokens_h == 0 and W % grid_tokens_w == 0, (
            f"Expected H divisible by grid_tokens_h and W by grid_tokens_w, "
            f"got H={H}, W={W}, grid={grid_tokens_h}x{grid_tokens_w}"
        )
        patch_size = (H // grid_tokens_h, W // grid_tokens_w)
        # Store the derived patch_size back into the Swin config
        swin_cfg_dict["patch_size"] = list(patch_size)
    else:
        # Normalise possible int / list formats
        ps = swin_cfg_dict["patch_size"]
        if isinstance(ps, int):
            swin_cfg_dict["patch_size"] = [ps, ps]
        else:
            assert len(ps) == 2, "swin.patch_size must be an int or length-2 list/tuple."

    # Instantiate the Swin-based multi-view VAE
    vae = InvariantDependentSplatterVAE(
        swin_cfg=swin_cfg_dict,
        invariant_cb_config=inv_cb_cfg,
        dependent_cb_config=dep_cb_cfg,
        img_height=H,
        img_width=W,
        splatter_channels=splatter_channels,
        fusion_style=fusion_style,
        use_dependent_vq=use_dependent_vq,
        is_dependent_ae=is_dependent_ae,
        use_invariant_vq=use_invariant_vq,
        is_invariant_ae=is_invariant_ae,
    )

    # -------------------------------------------------
    # 4) Training config + camera predictor from YAML
    # -------------------------------------------------
    train_cfg_dict = cfg.get("train", {})
    cfg_train = TrainConfig(**train_cfg_dict)

    # Instantiate camera predictor if enabled in config
    camera_predictor: Optional[CameraParamPredictor] = None
    if cfg_train.predict_camera_params:
        # dependent latent is flattened: (T * D_dep)
        dep_latent_dim = dep_cb_cfg.embed_dim * vae.n_tokens_per_frame
        camera_predictor = CameraParamPredictor(
            dep_latent_dim=dep_latent_dim,
            img_height=H,
            img_width=W,
        )
        print(f"[Info] Using CameraParamPredictor with dep_latent_dim={dep_latent_dim}")

    # -------------------------------------------------
    # 5) wandb init from YAML
    # -------------------------------------------------
    wandb_cfg = cfg.get("wandb", {})
    wandb_project = wandb_cfg.get("project", "MimicGen-SplatterVAE")
    wandb_entity = wandb_cfg.get("entity", None)

    wandb.init(
        project=wandb_project,
        entity=wandb_entity,
        config={
            # Log the most important hyperparams
            "dataset_path": dataset_path,
            "batch_size": batch_size,
            "num_workers": num_workers,
            "pin_memory": pin_memory,

            "num_epochs": cfg_train.num_epochs,
            "lr": cfg_train.lr,
            "device": cfg_train.device,
            "eval_every": cfg_train.eval_every,
            "save_every": cfg_train.save_every,
            "ckpt_dir": cfg_train.ckpt_dir,

            "img_height": H,
            "img_width": W,

            "predict_camera_params": cfg_train.predict_camera_params,
            "pose_cycle_weight": cfg_train.pose_cycle_weight,
            "crossview_start_step": cfg_train.crossview_start_step,

            "splatter_data": spl_data_cfg_dict,
            "splatter_model": spl_model_cfg_dict,
            "swin_transformer": swin_cfg_dict,
            "codebook_invariant": inv_cb_cfg_dict,
            "codebook_dependent": dep_cb_cfg_dict,
            "model": model_cfg,
        },
    )
    print(f"[wandb] Logging to project: {wandb_project}")

    # -------------------------------------------------
    # 6) Run training (optionally resuming from last ckpt)
    # -------------------------------------------------
    resume_ckpt = None
    if os.path.isdir(cfg_train.ckpt_dir) and cfg_train.resume_from_last:
        ckpt_files = [
            f for f in os.listdir(cfg_train.ckpt_dir)
            if f.endswith(".pth") and f.startswith("step_")
        ]
        if ckpt_files:
            ckpt_files.sort()  # "step_00001000.pth" ... lexicographically sorted = by step
            resume_ckpt = os.path.join(cfg_train.ckpt_dir, ckpt_files[-1])
            print(f"[Resume] Found latest checkpoint: {resume_ckpt}")

    train_splatter_vae(
        vae=vae,
        splatter_cfg=splatter_cfg,
        train_dataloader=train_loader,
        valid_dataloader=valid_loader,
        cfg_train=cfg_train,
        camera_predictor=camera_predictor,
        resume_ckpt=resume_ckpt,
    )


if __name__ == "__main__":
    main()
