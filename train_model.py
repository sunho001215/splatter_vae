from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.utils import make_grid

import wandb

from models.splatter import VAESplatterToGaussians, default_splatter_channels, render_predicted
from models.vae import SplatterVAE
from models.losses import compute_reconstruction_loss, infonce_loss


# ============================================================================
# Configuration
# ============================================================================

@dataclass
class TrainConfig:
    """Training hyperparameters and configuration."""
    num_epochs: int = 50
    max_global_steps: Optional[int] = None
    lr: float = 1e-4
    device: str = "cuda"

    # Keep the original loss weights.
    rec_weight: float = 1.0
    ssim_weight: float = 0.2
    vq_weight: float = 0.25
    inv_contrastive_weight: float = 1.0
    dep_contrastive_weight: float = 0.1
    frustum_weight: float = 0.001
    temperature: float = 0.1

    eval_every: int = 1000
    save_every: int = 5000
    ckpt_dir: str = "./checkpoints"
    resume_from_last: bool = False

    val_num_batches: int = -1
    val_max_vis: int = 8


# ============================================================================
# Encoding Functions
# ============================================================================

def encode_quartet(
    vae: SplatterVAE,
    image_a: torch.Tensor,
    image_b: torch.Tensor,
    image_c: torch.Tensor,
    image_d: torch.Tensor,
):
    """Encode the four corners of the 2x2 state-view grid.
    
    Encodes four images and computes VQ losses for both invariant and depth components.
    
    Returns:
        Tuple containing latent codes (z_inv_a, z_inv_b, z_inv_c, z_inv_d,
        z_dep_a, z_dep_b, z_dep_c, z_dep_d) and VQ losses.
    """
    z_inv_a, inv_loss_a, z_dep_a, dep_loss_a, _ = vae.encode(image_a)
    z_inv_b, inv_loss_b, z_dep_b, dep_loss_b, _ = vae.encode(image_b)
    z_inv_c, inv_loss_c, z_dep_c, dep_loss_c, _ = vae.encode(image_c)
    z_inv_d, inv_loss_d, z_dep_d, dep_loss_d, _ = vae.encode(image_d)

    inv_vq_loss = 0.25 * (inv_loss_a + inv_loss_b + inv_loss_c + inv_loss_d)
    dep_vq_loss = 0.25 * (dep_loss_a + dep_loss_b + dep_loss_c + dep_loss_d)

    return (
        z_inv_a, z_inv_b, z_inv_c, z_inv_d,
        z_dep_a, z_dep_b, z_dep_c, z_dep_d,
        inv_vq_loss, dep_vq_loss,
    )


# ============================================================================
# Contrastive Learning Functions
# ============================================================================

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
    """Compute contrastive losses for invariant and depth components.
    
    Uses InfoNCE loss with specific positive/negative pairing:
    - Invariant loss: rows contrast (same view, different state)
    - Depth loss: columns contrast (same state, different view)
    
    Returns:
        Tuple of (inv_contrastive_loss, dep_contrastive_loss)
    """
    bsz = z_inv_a.shape[0]

    # Reshape latent codes for contrastive loss computation
    a_s = z_inv_a.reshape(bsz, -1)
    b_s = z_inv_b.reshape(bsz, -1)
    c_s = z_inv_c.reshape(bsz, -1)
    d_s = z_inv_d.reshape(bsz, -1)

    a_v = z_dep_a.reshape(bsz, -1)
    b_v = z_dep_b.reshape(bsz, -1)
    c_v = z_dep_c.reshape(bsz, -1)
    d_v = z_dep_d.reshape(bsz, -1)

    # Invariant/state contrastive loss: same view, different states
    inv_loss_row0 = infonce_loss(
        query=a_s,
        positive_keys=b_s,
        negative_keys=torch.stack([c_s, d_s], dim=1),
        temperature=temperature,
        negative_mode="mixed",
    )
    inv_loss_row1 = infonce_loss(
        query=c_s,
        positive_keys=d_s,
        negative_keys=torch.stack([a_s, b_s], dim=1),
        temperature=temperature,
        negative_mode="mixed",
    )
    inv_contrastive_loss = 0.5 * (inv_loss_row0 + inv_loss_row1)

    # Depth/view contrastive loss: same state, different views
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


# ============================================================================
# Camera Transformation Functions
# ============================================================================

def camera_center_from_world_view(world_view: torch.Tensor) -> torch.Tensor:
    """
    world_view is a standard world->camera homogeneous transform:
        [ R  t ]
        [ 0  1 ]
    Camera center in world coordinates is inv(world_view)[:3, 3].
    """
    cam_to_world = torch.linalg.inv(world_view)
    return cam_to_world[:, :3, 3].contiguous()


# ============================================================================
# Rendering Functions
# ============================================================================

def _compute_soft_image_region_penalty(
    xyz_world: torch.Tensor,           # (B, N, 3)
    world_view_transform: torch.Tensor,  # (B, V, 4, 4), world -> camera
    intrinsics: torch.Tensor,            # (B, V, 3, 3)
    img_h: int,
    img_w: int,
    eps: float = 1e-6,
) -> Dict[str, torch.Tensor]:
    """
    Softly penalize Gaussian centers whose projected pixels fall outside
    [0, W-1] x [0, H-1].
    """
    B, N, _ = xyz_world.shape
    V = world_view_transform.shape[1]
    device = xyz_world.device
    dtype = xyz_world.dtype

    # World xyz -> homogeneous coordinates.
    ones = torch.ones((B, N, 1), device=device, dtype=dtype)
    xyz_world_h = torch.cat([xyz_world, ones], dim=-1)  # (B, N, 4)

    # Project world points into each camera frame.
    # Result: (B, V, N, 4)
    xyz_cam_h = torch.einsum("bvij,bnj->bvni", world_view_transform, xyz_world_h)
    xyz_cam = xyz_cam_h[..., :3]  # (B, V, N, 3)

    x = xyz_cam[..., 0]
    y = xyz_cam[..., 1]
    z = xyz_cam[..., 2]

    # Numerical safeguard only. This is not a z-regularizer.
    z_safe = z.clamp_min(eps)

    fx = intrinsics[..., 0, 0].unsqueeze(-1)  # (B, V, 1)
    fy = intrinsics[..., 1, 1].unsqueeze(-1)
    cx = intrinsics[..., 0, 2].unsqueeze(-1)
    cy = intrinsics[..., 1, 2].unsqueeze(-1)

    # Perspective projection.
    u = fx * (x / z_safe) + cx  # (B, V, N)
    v = fy * (y / z_safe) + cy  # (B, V, N)

    max_u = float(max(img_w - 1, 1))
    max_v = float(max(img_h - 1, 1))

    # Soft amount by which the projected center is outside the image box.
    # Zero if inside; positive if outside.
    u_over = F.relu(-u) / max_u + F.relu(u - max_u) / max_u
    v_over = F.relu(-v) / max_v + F.relu(v - max_v) / max_v
    per_view_penalty = u_over + v_over  # (B, V, N)

    # Boolean inactivity mask: outside image region in that camera frame.
    outside_mask = (u < 0.0) | (u > max_u) | (v < 0.0) | (v > max_v)

    stats: Dict[str, torch.Tensor] = {
        # Average penalty over batch, views, and gaussians.
        "frustum_loss": per_view_penalty.mean(),

        # Mean inactive ratio over both views.
        "inactive_ratio_mean": outside_mask.float().mean(),
    }

    # Log source / target separately when V == 2 (the usual case here).
    if V >= 1:
        stats["inactive_ratio_src"] = outside_mask[:, 0].float().mean()
    if V >= 2:
        stats["inactive_ratio_tgt"] = outside_mask[:, 1].float().mean()

    return stats


def _render_two_views_from_latents(
    vae: SplatterVAE,
    splatter_to_gaussians: VAESplatterToGaussians,
    splatter_cfg: SplatterConfig,
    z_inv: torch.Tensor,
    z_dep: torch.Tensor,
    k_src: torch.Tensor,
    k_tgt: torch.Tensor,
    t_src_to_tgt: torch.Tensor,   # (B,4,4)
    bg: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
    """
    Direct rendering path:
      decode one splatter image,
      convert it to a K-depth-ordered Gaussian scene,
      render the same scene from source and target cameras.
    """
    device = z_inv.device
    dtype = k_src.dtype
    b = z_inv.shape[0]

    # 1) Decode one direct splatter image from (z_inv, z_dep)
    splatter = vae.decode(z_inv, z_dep)

    # 2) Use the native/source frame as the local world frame
    eye_4 = torch.eye(4, device=device, dtype=dtype).view(1, 4, 4).repeat(b, 1, 1)
    eye_q = torch.tensor([1.0, 0.0, 0.0, 0.0], device=device, dtype=dtype).view(1, 4).repeat(b, 1)

    gaussian_pc = splatter_to_gaussians(
        splatter=splatter,
        source_cameras_view_to_world=eye_4,
        source_cv2wT_quat=eye_q,
        intrinsics=k_src,
        activate_output=True,
    )

    # 3) Render the same Gaussian scene in source and target views
    world_views = torch.stack([eye_4, t_src_to_tgt], dim=1)  # (B,2,4,4)
    intrinsics = torch.stack([k_src, k_tgt], dim=1)          # (B,2,3,3)

    out = render_predicted(
        pc=gaussian_pc,
        world_view_transform=world_views,
        intrinsics=intrinsics,
        bg_color=bg,
        cfg=splatter_cfg,
    )

    # 4) Soft image-plane frustum penalty for both source and target frames
    frustum_stats = _compute_soft_image_region_penalty(
        xyz_world=gaussian_pc["xyz"],
        world_view_transform=world_views,
        intrinsics=intrinsics,
        img_h=splatter_cfg.data.img_height,
        img_w=splatter_cfg.data.img_width,
    )

    return out["render"][:, 0], out["render"][:, 1], frustum_stats


# ============================================================================
# Loss Computation Functions
# ============================================================================

def compute_reconstruction_and_renders(
    vae: SplatterVAE,
    splatter_to_gaussians: VAESplatterToGaussians,
    splatter_cfg: SplatterConfig,
    gt_a_01: torch.Tensor,
    gt_b_01: torch.Tensor,
    gt_c_01: torch.Tensor,
    gt_d_01: torch.Tensor,
    z_inv_a: torch.Tensor,
    z_inv_c: torch.Tensor,
    z_dep_a: torch.Tensor,
    z_dep_b: torch.Tensor,
    k_i: torch.Tensor,
    k_j: torch.Tensor,
    t_ij: torch.Tensor,
    bg: torch.Tensor,
    return_renders: bool = False,
    ssim_weight: float = 0.2,
) -> Dict[str, Any]:
    """
    Same quartet supervision as before, but using direct K-Gaussian splatter rendering.
    """
    t_ji = torch.linalg.inv(t_ij)

    rendered_self_i, rendered_self_j_from_i, frustum_self = _render_two_views_from_latents(
        vae=vae,
        splatter_to_gaussians=splatter_to_gaussians,
        splatter_cfg=splatter_cfg,
        z_inv=z_inv_a,
        z_dep=z_dep_a,
        k_src=k_i,
        k_tgt=k_j,
        t_src_to_tgt=t_ij,
        bg=bg,
    )

    rendered_swap_view_j, rendered_swap_view_i_from_j, frustum_swap_view = _render_two_views_from_latents(
        vae=vae,
        splatter_to_gaussians=splatter_to_gaussians,
        splatter_cfg=splatter_cfg,
        z_inv=z_inv_a,
        z_dep=z_dep_b,
        k_src=k_j,
        k_tgt=k_i,
        t_src_to_tgt=t_ji,
        bg=bg,
    )

    rendered_swap_state_i, rendered_swap_state_j_from_i, frustum_swap_state = _render_two_views_from_latents(
        vae=vae,
        splatter_to_gaussians=splatter_to_gaussians,
        splatter_cfg=splatter_cfg,
        z_inv=z_inv_c,
        z_dep=z_dep_a,
        k_src=k_i,
        k_tgt=k_j,
        t_src_to_tgt=t_ij,
        bg=bg,
    )

    rendered_swap_both_j, rendered_swap_both_i_from_j, frustum_swap_both = _render_two_views_from_latents(
        vae=vae,
        splatter_to_gaussians=splatter_to_gaussians,
        splatter_cfg=splatter_cfg,
        z_inv=z_inv_c,
        z_dep=z_dep_b,
        k_src=k_j,
        k_tgt=k_i,
        t_src_to_tgt=t_ji,
        bg=bg,
    )

    # Native-view losses
    rec_self = compute_reconstruction_loss(rendered_self_i, gt_a_01, ssim_weight=ssim_weight)
    rec_swap_view = compute_reconstruction_loss(rendered_swap_view_j, gt_b_01, ssim_weight=ssim_weight)
    rec_swap_state = compute_reconstruction_loss(rendered_swap_state_i, gt_c_01, ssim_weight=ssim_weight)
    rec_swap_both = compute_reconstruction_loss(rendered_swap_both_j, gt_d_01, ssim_weight=ssim_weight)

    rec_native_loss = 0.25 * (rec_self + rec_swap_view + rec_swap_state + rec_swap_both)

    # Cross-view losses
    rec_self_cross = compute_reconstruction_loss(rendered_self_j_from_i, gt_b_01, ssim_weight=ssim_weight)
    rec_swap_view_cross = compute_reconstruction_loss(rendered_swap_view_i_from_j, gt_a_01, ssim_weight=ssim_weight)
    rec_swap_state_cross = compute_reconstruction_loss(rendered_swap_state_j_from_i, gt_d_01, ssim_weight=ssim_weight)
    rec_swap_both_cross = compute_reconstruction_loss(rendered_swap_both_i_from_j, gt_c_01, ssim_weight=ssim_weight)

    rec_cross_loss = 0.25 * (
        rec_self_cross + rec_swap_view_cross + rec_swap_state_cross + rec_swap_both_cross
    )

    rec_loss = 0.5 * (rec_native_loss + rec_cross_loss)

    # Average the soft image-frustum penalty over the 4 decoded scenes.
    frustum_loss = 0.25 * (
        frustum_self["frustum_loss"]
        + frustum_swap_view["frustum_loss"]
        + frustum_swap_state["frustum_loss"]
        + frustum_swap_both["frustum_loss"]
    )

    inactive_ratio_mean = 0.25 * (
        frustum_self["inactive_ratio_mean"]
        + frustum_swap_view["inactive_ratio_mean"]
        + frustum_swap_state["inactive_ratio_mean"]
        + frustum_swap_both["inactive_ratio_mean"]
    )

    inactive_ratio_src = 0.25 * (
        frustum_self["inactive_ratio_src"]
        + frustum_swap_view["inactive_ratio_src"]
        + frustum_swap_state["inactive_ratio_src"]
        + frustum_swap_both["inactive_ratio_src"]
    )

    inactive_ratio_tgt = 0.25 * (
        frustum_self["inactive_ratio_tgt"]
        + frustum_swap_view["inactive_ratio_tgt"]
        + frustum_swap_state["inactive_ratio_tgt"]
        + frustum_swap_both["inactive_ratio_tgt"]
    )

    out_dict: Dict[str, Any] = {
        "rec_loss": rec_loss,
        "rec_native_loss": rec_native_loss,
        "rec_cross_loss": rec_cross_loss,
        "rec_self": rec_self,
        "rec_swap_view": rec_swap_view,
        "rec_swap_state": rec_swap_state,
        "rec_swap_both": rec_swap_both,
        "rec_self_cross": rec_self_cross,
        "rec_swap_view_cross": rec_swap_view_cross,
        "rec_swap_state_cross": rec_swap_state_cross,
        "rec_swap_both_cross": rec_swap_both_cross,

        # New frustum-related outputs
        "frustum_loss": frustum_loss,
        "inactive_ratio_mean": inactive_ratio_mean,
        "inactive_ratio_src": inactive_ratio_src,
        "inactive_ratio_tgt": inactive_ratio_tgt,
    }

    if return_renders:
        out_dict.update(
            {
                "rendered_self_i": rendered_self_i,
                "rendered_swap_view_j": rendered_swap_view_j,
                "rendered_swap_state_i": rendered_swap_state_i,
                "rendered_swap_both_j": rendered_swap_both_j,
                "rendered_self_j_from_i": rendered_self_j_from_i,
                "rendered_swap_view_i_from_j": rendered_swap_view_i_from_j,
                "rendered_swap_state_j_from_i": rendered_swap_state_j_from_i,
                "rendered_swap_both_i_from_j": rendered_swap_both_i_from_j,
            }
        )

    return out_dict


# ============================================================================
# Validation / Visualization Helpers
# ============================================================================

def _make_wandb_image_grid(images_01: torch.Tensor, max_vis: int = 4) -> wandb.Image:
    """Convert a batch of images in [0, 1] into a single wandb image grid.

    Args:
        images_01: Tensor of shape (B, 3, H, W) in [0, 1].
        max_vis: Maximum number of samples from the batch to visualize.

    Returns:
        wandb.Image containing a grid.
    """
    images_01 = images_01.detach().float().cpu().clamp(0.0, 1.0)
    n_vis = max(1, min(int(max_vis), images_01.shape[0]))
    grid = make_grid(images_01[:n_vis], nrow=n_vis, padding=2)
    return wandb.Image(grid)


@torch.no_grad()
def validate_and_log_wandb(
    vae: SplatterVAE,
    splatter_cfg: SplatterConfig,
    splatter_to_gaussians: VAESplatterToGaussians,
    valid_dataloader: Optional[DataLoader],
    device: torch.device,
    bg: torch.Tensor,
    cfg_train: TrainConfig,
    global_step: int,
):
    """Perform validation on the provided dataloader and log metrics and images to wandb."""

    # No validation loader or no active wandb run -> nothing to do.
    if valid_dataloader is None or wandb.run is None:
        return

    # Preserve the current train/eval modes so we can restore them afterwards.
    prev_vae_mode = vae.training
    prev_splatter_mode = splatter_to_gaussians.training

    vae.eval()
    splatter_to_gaussians.eval()

    # Scalar accumulators (averaged across validation batches).
    scalar_sums = {
        "val/rec_loss": 0.0,
        "val/rec_native_loss": 0.0,
        "val/rec_cross_loss": 0.0,
        "val/rec_self": 0.0,
        "val/rec_swap_view": 0.0,
        "val/rec_swap_state": 0.0,
        "val/rec_swap_both": 0.0,
        "val/rec_self_cross": 0.0,
        "val/rec_swap_view_cross": 0.0,
        "val/rec_swap_state_cross": 0.0,
        "val/rec_swap_both_cross": 0.0,
        "val/inv_contrastive_loss": 0.0,
        "val/dep_contrastive_loss": 0.0,
        "val/frustum_loss": 0.0,
        "val/inactive_pct_mean": 0.0,
        "val/inactive_pct_src": 0.0,
        "val/inactive_pct_tgt": 0.0,
    }

    num_eval_batches = 0
    image_payload: Dict[str, wandb.Image] = {}

    for batch_idx, batch in enumerate(valid_dataloader):
        # Optional early stop for faster periodic validation.
        if cfg_train.val_num_batches > 0 and batch_idx >= cfg_train.val_num_batches:
            break

        # ------------------------------------------------------------
        # 1) Move batch to device
        # ------------------------------------------------------------
        image_a = batch["image_i_t"].to(device)
        image_b = batch["image_j_t"].to(device)
        image_c = batch["image_i_t1"].to(device)
        image_d = batch["image_j_t1"].to(device)
        t_ij = batch["T_ij"].to(device)
        k_i = batch["K_i"].to(device)
        k_j = batch["K_j"].to(device)

        # Convert ground truth from [-1, 1] to [0, 1].
        gt_a_01 = (image_a + 1.0) * 0.5
        gt_b_01 = (image_b + 1.0) * 0.5
        gt_c_01 = (image_c + 1.0) * 0.5
        gt_d_01 = (image_d + 1.0) * 0.5

        # ------------------------------------------------------------
        # 2) Encode quartet and compute contrastive losses
        # ------------------------------------------------------------
        (
            z_inv_a, z_inv_b, z_inv_c, z_inv_d,
            z_dep_a, z_dep_b, z_dep_c, z_dep_d,
            _inv_vq_loss, _dep_vq_loss,   # validation logging is kept identical to the original setup
        ) = encode_quartet(vae, image_a, image_b, image_c, image_d)

        inv_contrastive_loss, dep_contrastive_loss = compute_contrastive_losses(
            z_inv_a, z_inv_b, z_inv_c, z_inv_d,
            z_dep_a, z_dep_b, z_dep_c, z_dep_d,
            temperature=cfg_train.temperature,
        )

        # ------------------------------------------------------------
        # 3) Reconstruction + rendering
        #    Only the first validation batch stores rendered images;
        #    later batches only contribute to scalar averages.
        # ------------------------------------------------------------
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
            k_i=k_i,
            k_j=k_j,
            t_ij=t_ij,
            bg=bg,
            return_renders=(num_eval_batches == 0),
            ssim_weight=cfg_train.ssim_weight,
        )

        # ------------------------------------------------------------
        # 4) Accumulate scalar metrics
        # ------------------------------------------------------------
        scalar_sums["val/rec_loss"] += float(rec_out["rec_loss"].item())
        scalar_sums["val/rec_native_loss"] += float(rec_out["rec_native_loss"].item())
        scalar_sums["val/rec_cross_loss"] += float(rec_out["rec_cross_loss"].item())

        scalar_sums["val/rec_self"] += float(rec_out["rec_self"].item())
        scalar_sums["val/rec_swap_view"] += float(rec_out["rec_swap_view"].item())
        scalar_sums["val/rec_swap_state"] += float(rec_out["rec_swap_state"].item())
        scalar_sums["val/rec_swap_both"] += float(rec_out["rec_swap_both"].item())

        scalar_sums["val/rec_self_cross"] += float(rec_out["rec_self_cross"].item())
        scalar_sums["val/rec_swap_view_cross"] += float(rec_out["rec_swap_view_cross"].item())
        scalar_sums["val/rec_swap_state_cross"] += float(rec_out["rec_swap_state_cross"].item())
        scalar_sums["val/rec_swap_both_cross"] += float(rec_out["rec_swap_both_cross"].item())

        scalar_sums["val/inv_contrastive_loss"] += float(inv_contrastive_loss.item())
        scalar_sums["val/dep_contrastive_loss"] += float(dep_contrastive_loss.item())

        scalar_sums["val/frustum_loss"] += float(rec_out["frustum_loss"].item())
        scalar_sums["val/inactive_pct_mean"] += float(100.0 * rec_out["inactive_ratio_mean"].item())
        scalar_sums["val/inactive_pct_src"] += float(100.0 * rec_out["inactive_ratio_src"].item())
        scalar_sums["val/inactive_pct_tgt"] += float(100.0 * rec_out["inactive_ratio_tgt"].item())

        # ------------------------------------------------------------
        # 5) Save image grids from the first validation batch only
        # ------------------------------------------------------------
        if num_eval_batches == 0:
            image_payload = {
                # Native-view renders
                "val/render_self_i": _make_wandb_image_grid(
                    rec_out["rendered_self_i"], max_vis=cfg_train.val_max_vis
                ),
                "val/render_swap_view_j": _make_wandb_image_grid(
                    rec_out["rendered_swap_view_j"], max_vis=cfg_train.val_max_vis
                ),
                "val/render_swap_state_i": _make_wandb_image_grid(
                    rec_out["rendered_swap_state_i"], max_vis=cfg_train.val_max_vis
                ),
                "val/render_swap_both_j": _make_wandb_image_grid(
                    rec_out["rendered_swap_both_j"], max_vis=cfg_train.val_max_vis
                ),

                # Cross-view renders
                "val/render_self_j_from_i": _make_wandb_image_grid(
                    rec_out["rendered_self_j_from_i"], max_vis=cfg_train.val_max_vis
                ),
                "val/render_swap_view_i_from_j": _make_wandb_image_grid(
                    rec_out["rendered_swap_view_i_from_j"], max_vis=cfg_train.val_max_vis
                ),
                "val/render_swap_state_j_from_i": _make_wandb_image_grid(
                    rec_out["rendered_swap_state_j_from_i"], max_vis=cfg_train.val_max_vis
                ),
                "val/render_swap_both_i_from_j": _make_wandb_image_grid(
                    rec_out["rendered_swap_both_i_from_j"], max_vis=cfg_train.val_max_vis
                ),

                # Ground-truth targets
                "val/gt_a_s0_view_i": _make_wandb_image_grid(
                    gt_a_01, max_vis=cfg_train.val_max_vis
                ),
                "val/gt_b_s0_view_j": _make_wandb_image_grid(
                    gt_b_01, max_vis=cfg_train.val_max_vis
                ),
                "val/gt_c_s1_view_i": _make_wandb_image_grid(
                    gt_c_01, max_vis=cfg_train.val_max_vis
                ),
                "val/gt_d_s1_view_j": _make_wandb_image_grid(
                    gt_d_01, max_vis=cfg_train.val_max_vis
                ),
            }

        num_eval_batches += 1

    # Nothing was evaluated.
    if num_eval_batches == 0:
        vae.train(prev_vae_mode)
        splatter_to_gaussians.train(prev_splatter_mode)
        return

    # Average scalar metrics across validation batches.
    log_dict: Dict[str, Any] = {
        key: value / float(num_eval_batches) for key, value in scalar_sums.items()
    }

    # Add the visualization grids from the first validation batch.
    log_dict.update(image_payload)

    # Upload everything to wandb in one call.
    wandb.log(log_dict, step=global_step)

    # Restore original modes.
    vae.train(prev_vae_mode)
    splatter_to_gaussians.train(prev_splatter_mode)


# ============================================================================
# Main Training Loop
# ============================================================================

def train_splatter_vae(
    vae: SplatterVAE,
    splatter_cfg: SplatterConfig,
    train_dataloader: DataLoader,
    cfg_train: TrainConfig,
    valid_dataloader: Optional[DataLoader] = None,
    resume_ckpt: Optional[str] = None,
):
    """Training loop for splatter VAE.
    
    Combines VAE reconstruction, VQ loss, and contrastive learning objectives.
    
    Expected batch keys:
        image_i_t, image_j_t, image_i_t1, image_j_t1, T_ij, K_i, K_j
    """
    # ========================================================================
    # Initialization
    # ========================================================================
    device = torch.device(cfg_train.device)
    vae.to(device)

    # Splatter-to-Gaussian converter
    splatter_to_gaussians = VAESplatterToGaussians(splatter_cfg).to(device)

    # Direct converter has no trainable child predictor, so only optimize the VAE
    optimizer = torch.optim.Adam(list(vae.parameters()), lr=cfg_train.lr)

    bg = (
        torch.ones(3, device=device)
        if splatter_cfg.data.white_background
        else torch.zeros(3, device=device)
    )

    start_epoch = 0
    global_step = 0
    
    # ========================================================================
    # Checkpoint Loading
    # ========================================================================
    if resume_ckpt is not None and os.path.isfile(resume_ckpt):
        ckpt = torch.load(resume_ckpt, map_location="cpu")
        vae.load_state_dict(ckpt["vae_state_dict"])
        splatter_to_gaussians.load_state_dict(ckpt["splatter_to_gaussians_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        start_epoch = int(ckpt["epoch"])
        global_step = int(ckpt["global_step"])

    os.makedirs(cfg_train.ckpt_dir, exist_ok=True)
    
    # ========================================================================
    # Main Training Loop
    # ========================================================================
    epoch = start_epoch
    while True:
        for step, batch in enumerate(train_dataloader):
            # Check termination condition
            if cfg_train.max_global_steps is not None and global_step >= cfg_train.max_global_steps:
                print(f"[Stop] Reached max_global_steps={cfg_train.max_global_steps}.")
                return

            vae.train()
            splatter_to_gaussians.train()

            # ================================================================
            # Data Loading and Preprocessing
            # ================================================================
            image_a = batch["image_i_t"].to(device)
            image_b = batch["image_j_t"].to(device)
            image_c = batch["image_i_t1"].to(device)
            image_d = batch["image_j_t1"].to(device)
            t_ij = batch["T_ij"].to(device)
            k_i = batch["K_i"].to(device)
            k_j = batch["K_j"].to(device)

            # Convert images from [-1, 1] to [0, 1] range
            gt_a_01 = (image_a + 1.0) * 0.5
            gt_b_01 = (image_b + 1.0) * 0.5
            gt_c_01 = (image_c + 1.0) * 0.5
            gt_d_01 = (image_d + 1.0) * 0.5

            # ================================================================
            # Forward Pass: Encoding
            # ================================================================
            (
                z_inv_a, z_inv_b, z_inv_c, z_inv_d,
                z_dep_a, z_dep_b, z_dep_c, z_dep_d,
                inv_vq_loss, dep_vq_loss,
            ) = encode_quartet(vae, image_a, image_b, image_c, image_d)

            # ================================================================
            # Forward Pass: Reconstruction
            # ================================================================
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
                k_i=k_i,
                k_j=k_j,
                t_ij=t_ij,
                bg=bg,
                return_renders=False,
                ssim_weight=cfg_train.ssim_weight,
            )
            rec_loss = rec_out["rec_loss"]
            frustum_loss = rec_out["frustum_loss"]

            # ================================================================
            # Forward Pass: Contrastive Learning
            # ================================================================
            inv_contrastive_loss, dep_contrastive_loss = compute_contrastive_losses(
                z_inv_a, z_inv_b, z_inv_c, z_inv_d,
                z_dep_a, z_dep_b, z_dep_c, z_dep_d,
                temperature=cfg_train.temperature,
            )
            
            # ================================================================
            # Loss Aggregation
            # ================================================================
            vq_loss = inv_vq_loss + dep_vq_loss
            total_loss = (
                cfg_train.rec_weight * rec_loss
                + cfg_train.vq_weight * vq_loss
                + cfg_train.inv_contrastive_weight * inv_contrastive_loss
                + cfg_train.dep_contrastive_weight * dep_contrastive_loss
                + cfg_train.frustum_weight * frustum_loss
            )

            # ================================================================
            # Backward Pass and Optimization
            # ================================================================
            optimizer.zero_grad(set_to_none=True)
            total_loss.backward()
            optimizer.step()

            # ================================================================
            # Logging
            # ================================================================
            if wandb.run is not None:
                wandb.log(
                    {
                        "train/total_loss": total_loss.item(),
                        "train/rec_loss": rec_loss.item(),
                        "train/rec_native_loss": rec_out["rec_native_loss"].item(),
                        "train/rec_cross_loss": rec_out["rec_cross_loss"].item(),
                        "train/rec_self": rec_out["rec_self"].item(),
                        "train/rec_swap_view": rec_out["rec_swap_view"].item(),
                        "train/rec_swap_state": rec_out["rec_swap_state"].item(),
                        "train/rec_swap_both": rec_out["rec_swap_both"].item(),
                        "train/rec_self_cross": rec_out["rec_self_cross"].item(),
                        "train/rec_swap_view_cross": rec_out["rec_swap_view_cross"].item(),
                        "train/rec_swap_state_cross": rec_out["rec_swap_state_cross"].item(),
                        "train/rec_swap_both_cross": rec_out["rec_swap_both_cross"].item(),
                        "train/vq_loss": vq_loss.item(),
                        "train/inv_vq_loss": inv_vq_loss.item(),
                        "train/dep_vq_loss": dep_vq_loss.item(),
                        "train/inv_contrastive_loss": inv_contrastive_loss.item(),
                        "train/dep_contrastive_loss": dep_contrastive_loss.item(),
                        "train/frustum_loss": frustum_loss.item(),
                        "train/frustum_loss_weighted": (cfg_train.frustum_weight * frustum_loss).item(),
                        "train/inactive_pct_mean": 100.0 * rec_out["inactive_ratio_mean"].item(),
                        "train/inactive_pct_src": 100.0 * rec_out["inactive_ratio_src"].item(),
                        "train/inactive_pct_tgt": 100.0 * rec_out["inactive_ratio_tgt"].item(),
                        "train/frustum_weight": float(cfg_train.frustum_weight),
                        "train/global_step": global_step,
                    },
                    step=global_step,
                )

            if step % 50 == 0:
                print(
                    f"[Epoch {epoch+1} | Step {step} | Global {global_step}] "
                    f"Loss={total_loss.item():.4f} "
                    f"(rec={rec_loss.item():.4f}, "
                    f"native={rec_out['rec_native_loss'].item():.4f}, "
                    f"cross={rec_out['rec_cross_loss'].item():.4f}, "
                    f"vq={vq_loss.item():.4f}, "
                    f"inv_con={inv_contrastive_loss.item():.4f}, "
                    f"dep_con={dep_contrastive_loss.item():.4f})"
                )
            
            # ================================================================
            # Periodic validation
            # ================================================================
            if (
                valid_dataloader is not None
                and cfg_train.eval_every > 0
                and global_step > 0
                and global_step % cfg_train.eval_every == 0
            ):
                validate_and_log_wandb(
                    vae=vae,
                    splatter_cfg=splatter_cfg,
                    splatter_to_gaussians=splatter_to_gaussians,
                    valid_dataloader=valid_dataloader,
                    device=device,
                    bg=bg,
                    cfg_train=cfg_train,
                    global_step=global_step,
                )

            # ================================================================
            # Checkpointing
            # ================================================================
            if cfg_train.save_every > 0 and global_step > 0 and global_step % cfg_train.save_every == 0:
                ckpt_path = os.path.join(cfg_train.ckpt_dir, f"step_{global_step:08d}.pth")
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

        epoch += 1
        if cfg_train.max_global_steps is None and epoch >= cfg_train.num_epochs:
            print(f"[Stop] Reached num_epochs={cfg_train.num_epochs}.")
            break


def main():
    import argparse
    import glob
    import yaml

    from dataset.dataloader import build_train_valid_loaders_robosuite
    from utils.general_utils import set_random_seed

    from models.splatter import (
        SplatterConfig,
        SplatterDataConfig,
        SplatterModelConfig,
    )
    from models.vae import CodebookConfig

    # ---------------------------------------------------------------------
    # 1) Parse CLI
    # ---------------------------------------------------------------------
    parser = argparse.ArgumentParser(description="Train Hierarchical SplatterVAE")
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to YAML config file.",
    )
    cli_args = parser.parse_args()

    # ---------------------------------------------------------------------
    # 2) Load YAML
    # ---------------------------------------------------------------------
    with open(cli_args.config, "r") as f:
        cfg = yaml.safe_load(f)

    # ---------------------------------------------------------------------
    # 3) Dataset / dataloaders
    # ---------------------------------------------------------------------
    ds_cfg = cfg.get("dataset", {})
    dataset_path = ds_cfg.get("hdf5_path", None)
    if dataset_path is None:
        raise ValueError('Config field "dataset.hdf5_path" is required.')

    batch_size = int(ds_cfg.get("batch_size", 32))
    num_workers = int(ds_cfg.get("num_workers", 8))
    pin_memory = bool(ds_cfg.get("pin_memory", True))
    train_ratio = float(ds_cfg.get("train_ratio", 0.90))
    seed = int(ds_cfg.get("seed", 42))
    num_episodes = ds_cfg.get("num_episodes", None)
    max_frames_per_demo = ds_cfg.get("max_frames_per_demo", None)

    set_random_seed(seed)

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

    # Infer image resolution from one training batch.
    sample_batch = next(iter(train_loader))
    _, _, H, W = sample_batch["image_i_t"].shape
    print(f"[Info] Training image resolution: H={H}, W={W}")

    # ---------------------------------------------------------------------
    # 4) Build hierarchical SplatterConfig
    # ---------------------------------------------------------------------
    spl_cfg = cfg.get("splatter", {})
    spl_data_cfg_dict = dict(spl_cfg.get("data", {}))
    spl_model_cfg_dict = dict(spl_cfg.get("model", {}))

    # Override / fill spatial size from the actual batch.
    spl_data_cfg_dict["img_height"] = H
    spl_data_cfg_dict["img_width"] = W

    splatter_data_cfg = SplatterDataConfig(**spl_data_cfg_dict)
    splatter_model_cfg = SplatterModelConfig(**spl_model_cfg_dict)
    splatter_cfg = SplatterConfig(
        data=splatter_data_cfg,
        model=splatter_model_cfg,
    )

    # ---------------------------------------------------------------------
    # 5) Build TrainConfig
    # ---------------------------------------------------------------------
    train_cfg_dict = cfg.get("train", {})
    cfg_train = TrainConfig(**train_cfg_dict)

    # ---------------------------------------------------------------------
    # 6) Build codebook configs
    # ---------------------------------------------------------------------
    cb_cfg = cfg.get("codebook", {})
    inv_cb_cfg = CodebookConfig(**cb_cfg.get("invariant", {}))
    dep_cb_cfg = CodebookConfig(**cb_cfg.get("dependent", {}))

    # ---------------------------------------------------------------------
    # 7) Build VAE
    # ---------------------------------------------------------------------
    vit_cfg = dict(cfg.get("vit", {}))
    model_cfg = dict(cfg.get("model", {}))

    # Derive splatter_channels from max_sh_degree and num_gaussians_per_pixel.
    max_sh_degree = int(cfg.get("splatter", {}).get("model", {}).get("max_sh_degree", 1))
    num_gaussians_per_pixel = int(
        cfg.get("splatter", {}).get("model", {}).get("num_gaussians_per_pixel", 5)
    )

    splatter_channels = int(
        cfg.get("splatter", {}).get(
            "splatter_channels",
            default_splatter_channels(
                max_sh_degree=max_sh_degree,
                num_gaussians_per_pixel=num_gaussians_per_pixel,
            ),
        )
    )

    vae = SplatterVAE(
        vit_cfg=vit_cfg,
        invariant_cb_config=inv_cb_cfg,
        dependent_cb_config=dep_cb_cfg,
        img_height=H,
        img_width=W,
        splatter_channels=splatter_channels,
        fusion_style=str(model_cfg.get("fusion_style", "cat")),
        use_dependent_vq=bool(model_cfg.get("use_dependent_vq", True)),
        is_dependent_ae=bool(model_cfg.get("is_dependent_ae", True)),
        use_invariant_vq=bool(model_cfg.get("use_invariant_vq", True)),
        is_invariant_ae=bool(model_cfg.get("is_invariant_ae", True)),
        dep_input_mask_ratio=float(model_cfg.get("dep_input_mask_ratio", 0.95)),
        dep_mask_eval=bool(model_cfg.get("dep_mask_eval", True)),
        dpt_features=int(vit_cfg.get("dpt_features", 256)),
    )

    # ---------------------------------------------------------------------
    # 8) wandb init
    # ---------------------------------------------------------------------
    wandb_cfg = cfg.get("wandb", {})
    wandb_enabled = bool(wandb_cfg.get("enabled", True))

    if wandb_enabled:
        wandb.init(
            project=wandb_cfg.get("project", "splattervae"),
            entity=wandb_cfg.get("entity", None),
            name=wandb_cfg.get("run_name", None),
            config=cfg,
        )
        print(f"[wandb] Logging to project: {wandb_cfg.get('project', 'splattervae')}")

    # ---------------------------------------------------------------------
    # 9) Optional resume-from-last
    # ---------------------------------------------------------------------
    resume_ckpt = None
    if cfg_train.resume_from_last and os.path.isdir(cfg_train.ckpt_dir):
        ckpt_candidates = sorted(glob.glob(os.path.join(cfg_train.ckpt_dir, "step_*.pth")))
        if len(ckpt_candidates) > 0:
            resume_ckpt = ckpt_candidates[-1]
            print(f"[Resume] Found latest checkpoint: {resume_ckpt}")

    # ---------------------------------------------------------------------
    # 10) Launch training
    # ---------------------------------------------------------------------
    train_splatter_vae(
        vae=vae,
        splatter_cfg=splatter_cfg,
        train_dataloader=train_loader,
        cfg_train=cfg_train,
        valid_dataloader=valid_loader,
        resume_ckpt=resume_ckpt,
    )

    print("Training finished.")


if __name__ == "__main__":
    main()