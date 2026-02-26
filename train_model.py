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

# -------------------------------------------------------------------------
# Shared helper functions
# -------------------------------------------------------------------------

def encode_quartet(
    vae: InvariantDependentSplatterVAE,
    image_a: torch.Tensor,  # A = (s0, view i)
    image_b: torch.Tensor,  # B = (s0, view j)
    image_c: torch.Tensor,  # C = (s1, view i)
    image_d: torch.Tensor,  # D = (s1, view j)
):
    """
    Encode the full 2x2 grid.

    Returns invariant / dependent latents for:
        A = (s0, i), B = (s0, j), C = (s1, i), D = (s1, j)

    This gives us exactly the four corners needed for:
        - self      : A <- (z_s[A], z_v[A])
        - swap_view : B <- (z_s[A], z_v[B])
        - swap_state: C <- (z_s[C], z_v[A])
        - swap_both : D <- (z_s[C], z_v[B])
    """
    z_inv_a, inv_loss_a, z_dep_a, dep_loss_a, _ = vae.encode(image_a)
    z_inv_b, inv_loss_b, z_dep_b, dep_loss_b, _ = vae.encode(image_b)
    z_inv_c, inv_loss_c, z_dep_c, dep_loss_c, _ = vae.encode(image_c)
    z_inv_d, inv_loss_d, z_dep_d, dep_loss_d, _ = vae.encode(image_d)

    # Average VQ losses across all four encodings
    inv_vq_loss = 0.25 * (inv_loss_a + inv_loss_b + inv_loss_c + inv_loss_d)
    dep_vq_loss = 0.25 * (dep_loss_a + dep_loss_b + dep_loss_c + dep_loss_d)

    return (
        z_inv_a, z_inv_b, z_inv_c, z_inv_d,
        z_dep_a, z_dep_b, z_dep_c, z_dep_d,
        inv_vq_loss, dep_vq_loss,
    )


def compute_contrastive_losses(
    z_inv_a: torch.Tensor,
    z_inv_b: torch.Tensor,
    z_inv_c: torch.Tensor,
    z_inv_d: torch.Tensor,
    z_dep_a: torch.Tensor,
    z_dep_b: torch.Tensor,
    z_dep_c: torch.Tensor,
    z_dep_d: torch.Tensor,
    temperature: float,
):
    """
    Contrastive layout on the 2x2 grid:

        A = (s0, i)   B = (s0, j)
        C = (s1, i)   D = (s1, j)

    Invariant branch:
        - positives are same-state / different-view  (A<->B, C<->D)
        - negatives are different-state

    Dependent branch:
        - positives are same-view / different-state  (A<->C, B<->D)
        - negatives are different-view
    """
    Bsz = z_inv_a.shape[0]

    # Flatten once for InfoNCE
    a_s = z_inv_a.reshape(Bsz, -1)
    b_s = z_inv_b.reshape(Bsz, -1)
    c_s = z_inv_c.reshape(Bsz, -1)
    d_s = z_inv_d.reshape(Bsz, -1)

    a_v = z_dep_a.reshape(Bsz, -1)
    b_v = z_dep_b.reshape(Bsz, -1)
    c_v = z_dep_c.reshape(Bsz, -1)
    d_v = z_dep_d.reshape(Bsz, -1)

    # ---------------------------
    # Invariant: same row positives
    # ---------------------------
    inv_loss_row0 = infonce_loss(
        query=a_s,
        positive_keys=b_s,
        negative_keys=torch.stack([c_s, d_s], dim=1),
        temperature=temperature,
        negative_mode="paired",
    )
    inv_loss_row1 = infonce_loss(
        query=c_s,
        positive_keys=d_s,
        negative_keys=torch.stack([a_s, b_s], dim=1),
        temperature=temperature,
        negative_mode="paired",
    )
    inv_contrastive_loss = 0.5 * (inv_loss_row0 + inv_loss_row1)

    # ---------------------------
    # Dependent: same column positives
    # ---------------------------
    dep_loss_col0 = infonce_loss(
        query=a_v,
        positive_keys=c_v,
        negative_keys=torch.stack([b_v, d_v], dim=1),
        temperature=temperature,
        negative_mode="paired",
    )
    dep_loss_col1 = infonce_loss(
        query=b_v,
        positive_keys=d_v,
        negative_keys=torch.stack([a_v, c_v], dim=1),
        temperature=temperature,
        negative_mode="paired",
    )
    dep_contrastive_loss = 0.5 * (dep_loss_col0 + dep_loss_col1)

    return inv_contrastive_loss, dep_contrastive_loss


def _render_two_views_from_latents(
    vae: InvariantDependentSplatterVAE,
    splatter_to_gaussians: VAESplatterToGaussians,
    splatter_cfg: SplatterConfig,
    z_inv: torch.Tensor,
    z_dep: torch.Tensor,
    K_src: torch.Tensor,
    K_tgt: torch.Tensor,
    T_src_to_tgt: torch.Tensor,   # (B,4,4): source camera -> target camera
    bg: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Decode a splatter from (z_inv, z_dep), where z_dep defines the *source/native*
    camera frame, then render it in:
        - source/native view
        - target/other view using the known relative transform T_src_to_tgt
    """
    device = z_inv.device
    dtype = K_src.dtype
    B = z_inv.shape[0]

    # ---------------------------------------------------------
    # 1) Decode splatter in the native/source frame implied by z_dep
    # ---------------------------------------------------------
    splatter = vae.decode(z_inv, z_dep)

    # ---------------------------------------------------------
    # 2) Treat the native/source frame as the local "world" for Gaussian creation
    # ---------------------------------------------------------
    eye_4 = torch.eye(4, device=device, dtype=dtype).view(1, 4, 4).repeat(B, 1, 1)
    eye_q = torch.tensor([1.0, 0.0, 0.0, 0.0], device=device, dtype=dtype).view(1, 4).repeat(B, 1)

    gaussian_pc = splatter_to_gaussians(
        splatter,
        source_cameras_view_to_world=eye_4,
        source_cv2wT_quat=eye_q,
        intrinsics=K_src,
        activate_output=True,
    )

    # ---------------------------------------------------------
    # 3) Render the same Gaussian scene in two views:
    #    view 0 = source/native
    #    view 1 = target/other camera
    # ---------------------------------------------------------
    world_views = torch.stack([eye_4, T_src_to_tgt], dim=1)   # (B,2,4,4)
    Ks = torch.stack([K_src, K_tgt], dim=1)                   # (B,2,3,3)

    out = render_predicted(
        pc=gaussian_pc,
        world_view_transform=world_views,
        intrinsics=Ks,
        bg_color=bg,
        cfg=splatter_cfg,
    )

    renders = out["render"]  # (B,2,3,H,W)
    rendered_src = renders[:, 0]   # source/native view
    rendered_tgt = renders[:, 1]   # target/other view

    return rendered_src, rendered_tgt


def compute_reconstruction_and_renders(
    vae: InvariantDependentSplatterVAE,
    splatter_to_gaussians: VAESplatterToGaussians,
    splatter_cfg: SplatterConfig,
    gt_a_01: torch.Tensor,  # A = (s0, i)
    gt_b_01: torch.Tensor,  # B = (s0, j)
    gt_c_01: torch.Tensor,  # C = (s1, i)
    gt_d_01: torch.Tensor,  # D = (s1, j)
    z_inv_a: torch.Tensor,
    z_inv_c: torch.Tensor,
    z_dep_a: torch.Tensor,
    z_dep_b: torch.Tensor,
    K_i: torch.Tensor,
    K_j: torch.Tensor,
    T_ij: torch.Tensor,     # NEW: known relative camera transform (i -> j)
    bg: torch.Tensor,
    return_renders: bool = False,
    ssim_weight: float = 0.2,
) -> Dict[str, Any]:
    """
    2x2 reconstruction targets with *both* native-view and cross-view rendering.

        A = (s0, i)
        B = (s0, j)
        C = (s1, i)
        D = (s1, j)

    Latent pairings:
        self       = (z_s[A], z_v[A])  -> native i (A), cross j (B)
        swap_view  = (z_s[A], z_v[B])  -> native j (B), cross i (A)
        swap_state = (z_s[C], z_v[A])  -> native i (C), cross j (D)
        swap_both  = (z_s[C], z_v[B])  -> native j (D), cross i (C)

    This adds explicit 3D consistency across viewpoints using T_ij / T_ji.
    """
    # Invert once: j -> i
    T_ji = torch.linalg.inv(T_ij)

    # ---------------------------------------------------------
    # 1) self = (state s0, view i)
    #    Native should match A, cross-view should match B
    # ---------------------------------------------------------
    rendered_self_i, rendered_self_j_from_i = _render_two_views_from_latents(
        vae=vae,
        splatter_to_gaussians=splatter_to_gaussians,
        splatter_cfg=splatter_cfg,
        z_inv=z_inv_a,
        z_dep=z_dep_a,
        K_src=K_i,
        K_tgt=K_j,
        T_src_to_tgt=T_ij,   # i -> j
        bg=bg,
    )

    # ---------------------------------------------------------
    # 2) swap_view = (state s0, view j)
    #    Native should match B, cross-view should match A
    # ---------------------------------------------------------
    rendered_swap_view_j, rendered_swap_view_i_from_j = _render_two_views_from_latents(
        vae=vae,
        splatter_to_gaussians=splatter_to_gaussians,
        splatter_cfg=splatter_cfg,
        z_inv=z_inv_a,
        z_dep=z_dep_b,
        K_src=K_j,
        K_tgt=K_i,
        T_src_to_tgt=T_ji,   # j -> i
        bg=bg,
    )

    # ---------------------------------------------------------
    # 3) swap_state = (state s1, view i)
    #    Native should match C, cross-view should match D
    # ---------------------------------------------------------
    rendered_swap_state_i, rendered_swap_state_j_from_i = _render_two_views_from_latents(
        vae=vae,
        splatter_to_gaussians=splatter_to_gaussians,
        splatter_cfg=splatter_cfg,
        z_inv=z_inv_c,
        z_dep=z_dep_a,
        K_src=K_i,
        K_tgt=K_j,
        T_src_to_tgt=T_ij,   # i -> j
        bg=bg,
    )

    # ---------------------------------------------------------
    # 4) swap_both = (state s1, view j)
    #    Native should match D, cross-view should match C
    # ---------------------------------------------------------
    rendered_swap_both_j, rendered_swap_both_i_from_j = _render_two_views_from_latents(
        vae=vae,
        splatter_to_gaussians=splatter_to_gaussians,
        splatter_cfg=splatter_cfg,
        z_inv=z_inv_c,
        z_dep=z_dep_b,
        K_src=K_j,
        K_tgt=K_i,
        T_src_to_tgt=T_ji,   # j -> i
        bg=bg,
    )

    # ---------------------------------------------------------
    # 5) Native-view reconstruction losses
    # ---------------------------------------------------------
    rec_self = compute_reconstruction_loss(rendered_self_i, gt_a_01, ssim_weight=ssim_weight)
    rec_swap_view = compute_reconstruction_loss(rendered_swap_view_j, gt_b_01, ssim_weight=ssim_weight)
    rec_swap_state = compute_reconstruction_loss(rendered_swap_state_i, gt_c_01, ssim_weight=ssim_weight)
    rec_swap_both = compute_reconstruction_loss(rendered_swap_both_j, gt_d_01, ssim_weight=ssim_weight)

    rec_native_loss = 0.25 * (
        rec_self + rec_swap_view + rec_swap_state + rec_swap_both
    )

    # ---------------------------------------------------------
    # 6) Cross-view reconstruction losses
    #    (same decoded 3D Gaussian scene, rendered into the other camera)
    # ---------------------------------------------------------
    rec_self_cross = compute_reconstruction_loss(rendered_self_j_from_i, gt_b_01, ssim_weight=ssim_weight)
    rec_swap_view_cross = compute_reconstruction_loss(rendered_swap_view_i_from_j, gt_a_01, ssim_weight=ssim_weight)
    rec_swap_state_cross = compute_reconstruction_loss(rendered_swap_state_j_from_i, gt_d_01, ssim_weight=ssim_weight)
    rec_swap_both_cross = compute_reconstruction_loss(rendered_swap_both_i_from_j, gt_c_01, ssim_weight=ssim_weight)

    rec_cross_loss = 0.25 * (
        rec_self_cross + rec_swap_view_cross + rec_swap_state_cross + rec_swap_both_cross
    )

    # ---------------------------------------------------------
    # 7) Final reconstruction loss:
    #    equal weight on native-view and cross-view supervision
    # ---------------------------------------------------------
    rec_loss = 0.5 * (rec_native_loss + rec_cross_loss)

    out_dict: Dict[str, Any] = {
        "rec_loss": rec_loss,

        # grouped losses
        "rec_native_loss": rec_native_loss,
        "rec_cross_loss": rec_cross_loss,

        # native-view terms
        "rec_self": rec_self,
        "rec_swap_view": rec_swap_view,
        "rec_swap_state": rec_swap_state,
        "rec_swap_both": rec_swap_both,

        # cross-view terms
        "rec_self_cross": rec_self_cross,
        "rec_swap_view_cross": rec_swap_view_cross,
        "rec_swap_state_cross": rec_swap_state_cross,
        "rec_swap_both_cross": rec_swap_both_cross,
    }

    if return_renders:
        out_dict.update(
            {
                # native renders
                "rendered_self_i": rendered_self_i,
                "rendered_swap_view_j": rendered_swap_view_j,
                "rendered_swap_state_i": rendered_swap_state_i,
                "rendered_swap_both_j": rendered_swap_both_j,

                # cross-view renders
                "rendered_self_j_from_i": rendered_self_j_from_i,
                "rendered_swap_view_i_from_j": rendered_swap_view_i_from_j,
                "rendered_swap_state_j_from_i": rendered_swap_state_j_from_i,
                "rendered_swap_both_i_from_j": rendered_swap_both_i_from_j,
            }
        )

    return out_dict

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
    global_step: int,
    max_vis: int = 4,
):
    """
    Validation for the 2x2 setup.
    No camera predictor, no pose metrics.
    """
    # ---------------------------------------------------------
    # 1. Set models to evaluation mode
    # ---------------------------------------------------------
    vae.eval()
    splatter_to_gaussians.eval()

    # ---------------------------------------------------------
    # 2. Load a single validation batch
    # ---------------------------------------------------------
    try:
        batch = next(iter(valid_dataloader))
    except StopIteration:
        return

    # ---------------------------------------------------------
    # 3. Extract and move batch data to device
    # A=(s0,i), B=(s0,j), C=(s1,i), D=(s1,j)
    # ---------------------------------------------------------
    image_a = batch["image_i_t"].to(device)
    image_b = batch["image_j_t"].to(device)
    image_c = batch["image_i_t1"].to(device)
    image_d = batch["image_j_t1"].to(device)

    T_ij = batch["T_ij"].to(device)
    K_i = batch["K_i"].to(device)
    K_j = batch["K_j"].to(device)

    # ---------------------------------------------------------
    # 4. Convert ground-truth images from [-1,1] to [0,1]
    # ---------------------------------------------------------
    gt_a_01 = (image_a + 1.0) * 0.5
    gt_b_01 = (image_b + 1.0) * 0.5
    gt_c_01 = (image_c + 1.0) * 0.5
    gt_d_01 = (image_d + 1.0) * 0.5

    # ---------------------------------------------------------
    # 5. Encode all four images into invariant and dependent latents
    # ---------------------------------------------------------
    (
        z_inv_a, z_inv_b, z_inv_c, z_inv_d,
        z_dep_a, z_dep_b, z_dep_c, z_dep_d,
        _, _,
    ) = encode_quartet(vae, image_a, image_b, image_c, image_d)

    # ---------------------------------------------------------
    # 6. Compute contrastive losses for validation monitoring
    # ---------------------------------------------------------
    inv_contrastive_loss, dep_contrastive_loss = compute_contrastive_losses(
        z_inv_a, z_inv_b, z_inv_c, z_inv_d,
        z_dep_a, z_dep_b, z_dep_c, z_dep_d,
        temperature=cfg_train.temperature,
    )

    # ---------------------------------------------------------
    # 7. Compute reconstructions and rendered outputs
    # ---------------------------------------------------------
    rec_out = compute_reconstruction_and_renders(
        vae=vae,
        splatter_to_gaussians=splatter_to_gaussians,
        splatter_cfg=splatter_cfg,
        gt_a_01=gt_a_01,
        gt_b_01=gt_b_01,
        gt_c_01=gt_c_01,
        gt_d_01=gt_d_01,
        z_inv_a=z_inv_a,
        z_inv_c=z_inv_c,
        z_dep_a=z_dep_a,
        z_dep_b=z_dep_b,
        K_i=K_i,
        K_j=K_j,
        T_ij=T_ij,   # NEW
        bg=bg,
        return_renders=True,
        ssim_weight=cfg_train.ssim_weight,
    )

    # ---------------------------------------------------------
    # 8. Limit number of samples to visualize
    # ---------------------------------------------------------
    n_vis = min(max_vis, image_a.shape[0])

    # ---------------------------------------------------------
    # 9. Create grid visualizations for rendered outputs
    # ---------------------------------------------------------
    # Native-view renders
    grid_self_i = make_grid(
        rec_out["rendered_self_i"][:n_vis].cpu(),
        nrow=n_vis, normalize=True, value_range=(0, 1)
    )
    grid_swap_view_j = make_grid(
        rec_out["rendered_swap_view_j"][:n_vis].cpu(),
        nrow=n_vis, normalize=True, value_range=(0, 1)
    )
    grid_swap_state_i = make_grid(
        rec_out["rendered_swap_state_i"][:n_vis].cpu(),
        nrow=n_vis, normalize=True, value_range=(0, 1)
    )
    grid_swap_both_j = make_grid(
        rec_out["rendered_swap_both_j"][:n_vis].cpu(),
        nrow=n_vis, normalize=True, value_range=(0, 1)
    )

    # Cross-view renders
    grid_self_j_from_i = make_grid(
        rec_out["rendered_self_j_from_i"][:n_vis].cpu(),
        nrow=n_vis, normalize=True, value_range=(0, 1)
    )
    grid_swap_view_i_from_j = make_grid(
        rec_out["rendered_swap_view_i_from_j"][:n_vis].cpu(),
        nrow=n_vis, normalize=True, value_range=(0, 1)
    )
    grid_swap_state_j_from_i = make_grid(
        rec_out["rendered_swap_state_j_from_i"][:n_vis].cpu(),
        nrow=n_vis, normalize=True, value_range=(0, 1)
    )
    grid_swap_both_i_from_j = make_grid(
        rec_out["rendered_swap_both_i_from_j"][:n_vis].cpu(),
        nrow=n_vis, normalize=True, value_range=(0, 1)
    )

    # ---------------------------------------------------------
    # 10. Create grid visualizations for ground-truth images
    # ---------------------------------------------------------
    grid_gt_a = make_grid(gt_a_01[:n_vis].cpu(), nrow=n_vis, normalize=True, value_range=(0, 1))
    grid_gt_b = make_grid(gt_b_01[:n_vis].cpu(), nrow=n_vis, normalize=True, value_range=(0, 1))
    grid_gt_c = make_grid(gt_c_01[:n_vis].cpu(), nrow=n_vis, normalize=True, value_range=(0, 1))
    grid_gt_d = make_grid(gt_d_01[:n_vis].cpu(), nrow=n_vis, normalize=True, value_range=(0, 1))

    # ---------------------------------------------------------
    # 11. Log validation metrics and visualizations to wandb
    # ---------------------------------------------------------
    wandb.log(
        {
            # -----------------------------------------------------
            # Native-view renders
            # -----------------------------------------------------
            "val/render_self_i": wandb.Image(grid_self_i),                 # should match A
            "val/render_swap_view_j": wandb.Image(grid_swap_view_j),       # should match B
            "val/render_swap_state_i": wandb.Image(grid_swap_state_i),     # should match C
            "val/render_swap_both_j": wandb.Image(grid_swap_both_j),       # should match D

            # -----------------------------------------------------
            # Cross-view renders
            # -----------------------------------------------------
            "val/render_self_j_from_i": wandb.Image(grid_self_j_from_i),               # A rendered into j -> should match B
            "val/render_swap_view_i_from_j": wandb.Image(grid_swap_view_i_from_j),     # B rendered into i -> should match A
            "val/render_swap_state_j_from_i": wandb.Image(grid_swap_state_j_from_i),   # C rendered into j -> should match D
            "val/render_swap_both_i_from_j": wandb.Image(grid_swap_both_i_from_j),     # D rendered into i -> should match C

            # Ground truth targets
            "val/gt_a_s0_view_i": wandb.Image(grid_gt_a),
            "val/gt_b_s0_view_j": wandb.Image(grid_gt_b),
            "val/gt_c_s1_view_i": wandb.Image(grid_gt_c),
            "val/gt_d_s1_view_j": wandb.Image(grid_gt_d),

            # Grouped losses
            "val/rec_loss": rec_out["rec_loss"].item(),
            "val/rec_native_loss": rec_out["rec_native_loss"].item(),
            "val/rec_cross_loss": rec_out["rec_cross_loss"].item(),

            # Native-view losses
            "val/rec_self": rec_out["rec_self"].item(),
            "val/rec_swap_view": rec_out["rec_swap_view"].item(),
            "val/rec_swap_state": rec_out["rec_swap_state"].item(),
            "val/rec_swap_both": rec_out["rec_swap_both"].item(),

            # Cross-view losses
            "val/rec_self_cross": rec_out["rec_self_cross"].item(),
            "val/rec_swap_view_cross": rec_out["rec_swap_view_cross"].item(),
            "val/rec_swap_state_cross": rec_out["rec_swap_state_cross"].item(),
            "val/rec_swap_both_cross": rec_out["rec_swap_both_cross"].item(),

            # Contrastive monitoring
            "val/inv_contrastive_loss": inv_contrastive_loss.item(),
            "val/dep_contrastive_loss": dep_contrastive_loss.item(),
        },
        step=global_step,
    )

# -------------------------------------------------------------------------
# Main training loop (now with wandb + validation renders + checkpoints)
# -------------------------------------------------------------------------

def train_splatter_vae(
    vae: InvariantDependentSplatterVAE,
    splatter_cfg: SplatterConfig,
    train_dataloader: DataLoader,
    valid_dataloader: DataLoader,
    cfg_train: TrainConfig,
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

    # Optimizer: VAE + splatter_to_gaussians 
    params = list(vae.parameters()) + list(splatter_to_gaussians.parameters())
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

            # ---------------------------------------------------------
            # 1. Move batch to device
            # ---------------------------------------------------------
            image_a = batch["image_i_t"].to(device)    # A = (s0, i)
            image_b = batch["image_j_t"].to(device)    # B = (s0, j)
            image_c = batch["image_i_t1"].to(device)   # C = (s1, i)
            image_d = batch["image_j_t1"].to(device)   # D = (s1, j)

            # Ground-truth camera params (only used for validation/metrics, not for training)
            T_ij = batch["T_ij"].to(device)
            K_i = batch["K_i"].to(device)
            K_j = batch["K_j"].to(device)

            # Images are in [-1,1]; convert to [0,1]
            gt_a_01 = (image_a + 1.0) * 0.5
            gt_b_01 = (image_b + 1.0) * 0.5
            gt_c_01 = (image_c + 1.0) * 0.5
            gt_d_01 = (image_d + 1.0) * 0.5

            # ---------------------------------------------------------
            # 2. Encode three images into invariant + view-dependent latents
            # ---------------------------------------------------------
            (
                z_inv_a, z_inv_b, z_inv_c, z_inv_d,
                z_dep_a, z_dep_b, z_dep_c, z_dep_d,
                inv_vq_loss, dep_vq_loss,
            ) = encode_quartet(vae, image_a, image_b, image_c, image_d)

            # ---------------------------------------------------------
            # 3. Reconstruction / rendering
            # ---------------------------------------------------------
            rec_out = compute_reconstruction_and_renders(
                vae=vae,
                splatter_to_gaussians=splatter_to_gaussians,
                splatter_cfg=splatter_cfg,
                gt_a_01=gt_a_01,
                gt_b_01=gt_b_01,
                gt_c_01=gt_c_01,
                gt_d_01=gt_d_01,
                z_inv_a=z_inv_a,
                z_inv_c=z_inv_c,
                z_dep_a=z_dep_a,
                z_dep_b=z_dep_b,
                K_i=K_i,
                K_j=K_j,
                T_ij=T_ij,   # NEW: enables explicit cross-view rendering supervision
                bg=bg,
                return_renders=False,
                ssim_weight=cfg_train.ssim_weight,
            )

            rec_loss = rec_out["rec_loss"]

            # Grouped losses
            rec_native_loss = rec_out["rec_native_loss"]
            rec_cross_loss = rec_out["rec_cross_loss"]

            # Native-view losses
            rec_self = rec_out["rec_self"]
            rec_swap_view = rec_out["rec_swap_view"]
            rec_swap_state = rec_out["rec_swap_state"]
            rec_swap_both = rec_out["rec_swap_both"]

            # Cross-view losses
            rec_self_cross = rec_out["rec_self_cross"]
            rec_swap_view_cross = rec_out["rec_swap_view_cross"]
            rec_swap_state_cross = rec_out["rec_swap_state_cross"]
            rec_swap_both_cross = rec_out["rec_swap_both_cross"]

            # ---------------------------------------------------------
            # 4. Contrastive losses
            # ---------------------------------------------------------
            inv_contrastive_loss, dep_contrastive_loss = compute_contrastive_losses(
                z_inv_a, z_inv_b, z_inv_c, z_inv_d,
                z_dep_a, z_dep_b, z_dep_c, z_dep_d,
                temperature=cfg_train.temperature,
            )

            # ---------------------------------------------------------
            # 5. VQ loss
            # ---------------------------------------------------------
            vq_loss = inv_vq_loss + dep_vq_loss

            # ---------------------------------------------------------
            # 6. Total loss with weights
            # ---------------------------------------------------------
            total_loss = (
                cfg_train.rec_weight * rec_loss
                + cfg_train.vq_weight * vq_loss
                + cfg_train.inv_contrastive_weight * inv_contrastive_loss
                + cfg_train.dep_contrastive_weight * dep_contrastive_loss
            )

            optimizer.zero_grad(set_to_none=True)
            total_loss.backward()
            optimizer.step()

            # ---------------------------------------------------------
            # 7. wandb logging
            # ---------------------------------------------------------
            wandb.log(
                {
                    "train/total_loss": total_loss.item(),
                    "train/rec_loss": rec_loss.item(),

                    # Grouped reconstruction losses
                    "train/rec_native_loss": rec_native_loss.item(),
                    "train/rec_cross_loss": rec_cross_loss.item(),

                    # Native-view losses
                    "train/rec_self": rec_self.item(),
                    "train/rec_swap_view": rec_swap_view.item(),
                    "train/rec_swap_state": rec_swap_state.item(),
                    "train/rec_swap_both": rec_swap_both.item(),

                    # Cross-view losses
                    "train/rec_self_cross": rec_self_cross.item(),
                    "train/rec_swap_view_cross": rec_swap_view_cross.item(),
                    "train/rec_swap_state_cross": rec_swap_state_cross.item(),
                    "train/rec_swap_both_cross": rec_swap_both_cross.item(),

                    "train/vq_loss": vq_loss.item(),
                    "train/inv_vq_loss": inv_vq_loss.item(),
                    "train/dep_vq_loss": dep_vq_loss.item(),
                    "train/inv_contrastive_loss": inv_contrastive_loss.item(),
                    "train/dep_contrastive_loss": dep_contrastive_loss.item(),
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
                    f"native={rec_native_loss.item():.4f}, "
                    f"cross={rec_cross_loss.item():.4f}, "
                    f"self={rec_self.item():.4f}, "
                    f"swap_view={rec_swap_view.item():.4f}, "
                    f"swap_state={rec_swap_state.item():.4f}, "
                    f"swap_both={rec_swap_both.item():.4f}, "
                    f"self_cross={rec_self_cross.item():.4f}, "
                    f"swap_view_cross={rec_swap_view_cross.item():.4f}, "
                    f"swap_state_cross={rec_swap_state_cross.item():.4f}, "
                    f"swap_both_cross={rec_swap_both_cross.item():.4f}, "
                    f"vq={vq_loss.item():.4f}, "
                    f"inv_con={inv_contrastive_loss.item():.4f}, "
                    f"dep_con={dep_contrastive_loss.item():.4f})"
                )

            # ---------------------------------------------------------
            # 8. Periodic validation
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
                    global_step=global_step,
                )

            # ---------------------------------------------------------
            # 9. Periodic checkpoint saving
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
                }

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
    - Instantiates a InvariantDependentSplatterVAE whose decoder outputs a Splatter Image instead of RGB
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

    # -------------------------------------------------
    # 5) wandb init from YAML
    # -------------------------------------------------
    wandb_cfg = cfg.get("wandb", {})
    wandb_project = wandb_cfg.get("project", "MimicGen-SplatterVAE")
    wandb_entity = wandb_cfg.get("entity", None)
    wamdb_run_name = wandb_cfg.get("run_name", None)

    wandb.init(
        project=wandb_project,
        entity=wandb_entity,
        name=wamdb_run_name,
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
        resume_ckpt=resume_ckpt,
    )


if __name__ == "__main__":
    main()
