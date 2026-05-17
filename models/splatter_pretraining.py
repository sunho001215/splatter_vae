from __future__ import annotations

import os
from typing import Any, Dict, Optional

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.utils import make_grid

import wandb

from models.losses import (
    compute_reconstruction_loss,
    compute_splattervae_representation_losses,
)
from models.splatter import SplatterConfig, VAESplatterToGaussians, render_predicted
from models.splatter_train_config import TrainConfig
from models.vae import SplatterVAE


def encode_temporal_pair_batch(
    vae: SplatterVAE,
    image_i_t: torch.Tensor,
    image_j_t: torch.Tensor,
    image_i_tk: torch.Tensor,
    image_j_tk: torch.Tensor,
) -> tuple[Dict[str, Any], torch.Tensor, torch.Tensor]:
    """Encode two random views at t and the same views at temporal index tk.

    Args:
        image_i_t, image_j_t: two randomly sampled camera views from the same
            demo/timestep. They share view-invariant content.
        image_i_tk, image_j_tk: the same camera viewpoints from a different
            timestep in the same demo. They provide same-view features for
            temporal shuffling.

    Returns:
        A dict of compact state vectors keyed by image name, plus
        invariant/dependent beta-VAE KL losses averaged over the four images.
    """
    bsz, channels, height, width = image_i_t.shape
    images = torch.stack((image_i_t, image_j_t, image_i_tk, image_j_tk), dim=1)
    flat_images = images.reshape(bsz * 4, channels, height, width).contiguous()

    z_inv, inv_kl_loss, z_dep, dep_kl_loss, stats = vae.encode(flat_images)

    z_inv = z_inv.reshape(bsz, 4, *z_inv.shape[1:]).contiguous()
    z_dep = z_dep.reshape(bsz, 4, *z_dep.shape[1:]).contiguous()
    z_inv_mu = stats["z_inv_mu"].reshape(bsz, 4, -1).contiguous()
    z_dep_mu = stats["z_dep_mu"].reshape(bsz, 4, -1).contiguous()

    # Intermediate VCReg should not push same-state cross-view features apart.
    # Use only the fixed camera-i sequence: h_i_t and h_i_tk.
    z_inv_encoder_hidden_states = []
    for hidden in stats["z_inv_encoder_hidden_states"]:
        hidden = hidden.reshape(bsz, 4, *hidden.shape[1:]).contiguous()
        hidden_i = hidden[:, [0, 2]].reshape(bsz * 2, *hidden.shape[2:]).contiguous()
        z_inv_encoder_hidden_states.append(hidden_i)

    latents: Dict[str, Any] = {
        # Stochastic VAE samples are kept for decoding/rendering.
        "z_inv_i_t": z_inv[:, 0],
        "z_inv_j_t": z_inv[:, 1],
        "z_inv_i_tk": z_inv[:, 2],
        "z_inv_j_tk": z_inv[:, 3],
        "z_dep_i_t": z_dep[:, 0],
        "z_dep_j_t": z_dep[:, 1],
        "z_dep_i_tk": z_dep[:, 2],
        "z_dep_j_tk": z_dep[:, 3],
        # Deterministic means are used for representation regularization.
        "z_inv_mu_i_t": z_inv_mu[:, 0],
        "z_inv_mu_j_t": z_inv_mu[:, 1],
        "z_inv_mu_i_tk": z_inv_mu[:, 2],
        "z_inv_mu_j_tk": z_inv_mu[:, 3],
        "z_dep_mu_i_t": z_dep_mu[:, 0],
        "z_dep_mu_j_t": z_dep_mu[:, 1],
        "z_dep_mu_i_tk": z_dep_mu[:, 2],
        "z_dep_mu_j_tk": z_dep_mu[:, 3],
        # VCReg is applied only to h_i_t and h_i_tk intermediate features.
        "z_inv_encoder_hidden_states": z_inv_encoder_hidden_states,
    }
    return latents, inv_kl_loss, dep_kl_loss


def _decode_latent_kwargs(latents: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """Select only sampled latents for rendering; *_mu tensors are loss-only."""
    return {
        "z_inv_i_t": latents["z_inv_i_t"],
        "z_inv_j_t": latents["z_inv_j_t"],
        "z_inv_i_tk": latents["z_inv_i_tk"],
        "z_inv_j_tk": latents["z_inv_j_tk"],
        "z_dep_i_t": latents["z_dep_i_t"],
        "z_dep_j_t": latents["z_dep_j_t"],
        "z_dep_i_tk": latents["z_dep_i_tk"],
        "z_dep_j_tk": latents["z_dep_j_tk"],
    }


def compute_representation_losses(
    latents: Dict[str, Any],
    cfg_train: TrainConfig,
) -> Dict[str, torch.Tensor]:
    """Bind the shared SplatterVAE representation loss to TrainConfig fields."""
    return compute_splattervae_representation_losses(
        latents,
        vicreg_sim_coeff=cfg_train.vicreg_sim_coeff,
        vicreg_std_coeff=cfg_train.vicreg_std_coeff,
        vicreg_cov_coeff=cfg_train.vicreg_cov_coeff,
        vicreg_std_gamma=cfg_train.vicreg_std_gamma,
        vicreg_eps=cfg_train.vicreg_eps,
        dep_infonce_temperature=cfg_train.dep_infonce_temperature,
        inv_encoder_vcreg_std_coeff=cfg_train.inv_encoder_vcreg_std_coeff,
        inv_encoder_vcreg_cov_coeff=cfg_train.inv_encoder_vcreg_cov_coeff,
        inv_encoder_vcreg_std_gamma=cfg_train.inv_encoder_vcreg_std_gamma,
        inv_encoder_vcreg_eps=cfg_train.inv_encoder_vcreg_eps,
        inv_encoder_vcreg_cov_smooth_l1_delta=(
            cfg_train.inv_encoder_vcreg_cov_smooth_l1_delta
        ),
    )


def _compute_soft_image_region_penalty(
    xyz_world: torch.Tensor,
    world_view_transform: torch.Tensor,
    intrinsics: torch.Tensor,
    img_h: int,
    img_w: int,
    min_depth: float = 1e-3,
    penalty_cap: float = 100.0,
    source_view_indices: Optional[torch.Tensor] = None,
) -> Dict[str, torch.Tensor]:
    """Softly penalize Gaussian centers outside the image region or valid depth."""
    device = xyz_world.device

    xyz_world_f = xyz_world.float()
    world_view_f = world_view_transform.float()
    intrinsics_f = intrinsics.float()

    b, n, _ = xyz_world_f.shape
    ones = torch.ones((b, n, 1), device=device, dtype=xyz_world_f.dtype)
    xyz_world_h = torch.cat([xyz_world_f, ones], dim=-1)

    xyz_cam_h = torch.einsum("bvij,bnj->bvni", world_view_f, xyz_world_h)
    xyz_cam = xyz_cam_h[..., :3]

    x = xyz_cam[..., 0]
    y = xyz_cam[..., 1]
    z = xyz_cam[..., 2]

    min_z = max(float(min_depth), 1e-3)
    valid_depth = torch.isfinite(z) & (z > min_z)
    z_for_projection = torch.where(valid_depth, z, torch.ones_like(z))

    fx = intrinsics_f[..., 0, 0].unsqueeze(-1)
    fy = intrinsics_f[..., 1, 1].unsqueeze(-1)
    cx = intrinsics_f[..., 0, 2].unsqueeze(-1)
    cy = intrinsics_f[..., 1, 2].unsqueeze(-1)

    u = fx * (x / z_for_projection) + cx
    v = fy * (y / z_for_projection) + cy

    finite_projection = torch.isfinite(u) & torch.isfinite(v)
    u = torch.nan_to_num(u, nan=0.0, posinf=0.0, neginf=0.0)
    v = torch.nan_to_num(v, nan=0.0, posinf=0.0, neginf=0.0)

    max_u = float(max(img_w - 1, 1))
    max_v = float(max(img_h - 1, 1))

    u_over = F.relu(-u) / max_u + F.relu(u - max_u) / max_u
    v_over = F.relu(-v) / max_v + F.relu(v - max_v) / max_v
    image_penalty = u_over + v_over
    image_penalty = torch.where(
        valid_depth & finite_projection,
        image_penalty,
        torch.zeros_like(image_penalty),
    )

    z_clean = torch.nan_to_num(z, nan=-min_z, posinf=min_z, neginf=-min_z)
    depth_penalty = F.relu(min_z - z_clean) / min_z
    depth_penalty = torch.where(
        torch.isfinite(z),
        depth_penalty,
        torch.full_like(depth_penalty, float(penalty_cap)),
    )

    per_view_penalty = (image_penalty + depth_penalty).clamp(max=float(penalty_cap))
    outside_mask = (
        (~valid_depth)
        | (~finite_projection)
        | (u < 0.0)
        | (u > max_u)
        | (v < 0.0)
        | (v > max_v)
    )

    stats: Dict[str, torch.Tensor] = {
        "frustum_loss": per_view_penalty.mean(),
        "inactive_ratio_mean": outside_mask.float().mean(),
        "invalid_depth_ratio_mean": (~valid_depth).float().mean(),
        "nonfinite_projection_ratio_mean": (~finite_projection).float().mean(),
    }
    if source_view_indices is not None:
        num_views = world_view_transform.shape[1]
        src_idx = source_view_indices.to(device=device, dtype=torch.long).view(-1, 1, 1)
        src_idx = src_idx.expand(-1, 1, outside_mask.shape[-1])
        source_outside = outside_mask.gather(dim=1, index=src_idx).squeeze(1)
        stats["inactive_ratio_src"] = source_outside.float().mean()

        if num_views > 1:
            target_view_ids = torch.arange(num_views, device=device).view(1, num_views, 1)
            non_source_mask = target_view_ids != source_view_indices.to(device=device).view(-1, 1, 1)
            non_source_mask = non_source_mask.expand_as(outside_mask)
            stats["inactive_ratio_tgt"] = outside_mask.masked_select(non_source_mask).float().mean()
        else:
            stats["inactive_ratio_tgt"] = stats["inactive_ratio_src"]
    else:
        if world_view_transform.shape[1] >= 1:
            stats["inactive_ratio_src"] = outside_mask[:, 0].float().mean()
        if world_view_transform.shape[1] >= 2:
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
    t_src_to_tgt: torch.Tensor,
    bg: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
    """Decode one splatter per latent pair and render source/target views."""
    device = z_inv.device
    dtype = k_src.dtype
    bsz = z_inv.shape[0]

    splatter = vae.decode(z_inv, z_dep)
    eye_4 = torch.eye(4, device=device, dtype=dtype).view(1, 4, 4).repeat(bsz, 1, 1)
    eye_q = torch.tensor([1.0, 0.0, 0.0, 0.0], device=device, dtype=dtype).view(1, 4).repeat(bsz, 1)

    gaussian_pc = splatter_to_gaussians(
        splatter=splatter,
        source_cameras_view_to_world=eye_4,
        source_cv2wT_quat=eye_q,
        intrinsics=k_src,
        activate_output=True,
    )

    world_views = torch.stack((eye_4, t_src_to_tgt), dim=1)
    intrinsics = torch.stack((k_src, k_tgt), dim=1)

    out = render_predicted(
        pc=gaussian_pc,
        world_view_transform=world_views,
        intrinsics=intrinsics,
        bg_color=bg,
        cfg=splatter_cfg,
    )
    frustum_stats = _compute_soft_image_region_penalty(
        xyz_world=gaussian_pc["xyz"],
        world_view_transform=world_views,
        intrinsics=intrinsics,
        img_h=splatter_cfg.data.img_height,
        img_w=splatter_cfg.data.img_width,
        min_depth=splatter_cfg.data.znear,
    )
    return out["render"][:, 0], out["render"][:, 1], frustum_stats


def compute_reconstruction_and_renders(
    vae: SplatterVAE,
    splatter_to_gaussians: VAESplatterToGaussians,
    splatter_cfg: SplatterConfig,
    image_i_t_01: torch.Tensor,
    image_j_t_01: torch.Tensor,
    image_i_tk_01: torch.Tensor,
    image_j_tk_01: torch.Tensor,
    z_inv_i_t: torch.Tensor,
    z_inv_j_t: torch.Tensor,
    z_inv_i_tk: torch.Tensor,
    z_inv_j_tk: torch.Tensor,
    z_dep_i_t: torch.Tensor,
    z_dep_j_t: torch.Tensor,
    z_dep_i_tk: torch.Tensor,
    z_dep_j_tk: torch.Tensor,
    k_i: torch.Tensor,
    k_j: torch.Tensor,
    t_ij: torch.Tensor,
    bg: torch.Tensor,
    cfg_train: TrainConfig,
    return_renders: bool = False,
) -> Dict[str, Any]:
    """Compute the pasted-code quartet rendering loss.

    Rendering now uses exactly four decoded latent combinations per batch item:
      1. self:      z_inv_i_t  + z_dep_i_t, source i -> target j
      2. swap_view: z_inv_i_t  + z_dep_j_t, source j -> target i
      3. swap_time: z_inv_i_tk + z_dep_i_t, source i -> target j
      4. swap_both: z_inv_i_tk + z_dep_j_t, source j -> target i
    """
    _ = (z_inv_j_t, z_inv_j_tk, z_dep_i_tk, z_dep_j_tk)
    t_ji = torch.linalg.inv(t_ij)

    # Batch the four render terms into one decode/render call.
    z_inv_all = torch.cat((z_inv_i_t, z_inv_i_t, z_inv_i_tk, z_inv_i_tk), dim=0)
    z_dep_all = torch.cat((z_dep_i_t, z_dep_j_t, z_dep_i_t, z_dep_j_t), dim=0)
    k_src_all = torch.cat((k_i, k_j, k_i, k_j), dim=0)
    k_tgt_all = torch.cat((k_j, k_i, k_j, k_i), dim=0)
    t_src_to_tgt_all = torch.cat((t_ij, t_ji, t_ij, t_ji), dim=0)

    rendered_src_all, rendered_tgt_all, frustum_all = _render_two_views_from_latents(
        vae=vae,
        splatter_to_gaussians=splatter_to_gaussians,
        splatter_cfg=splatter_cfg,
        z_inv=z_inv_all,
        z_dep=z_dep_all,
        k_src=k_src_all,
        k_tgt=k_tgt_all,
        t_src_to_tgt=t_src_to_tgt_all,
        bg=bg,
    )

    (
        rendered_self_i,
        rendered_swap_view_j,
        rendered_swap_state_i,
        rendered_swap_both_j,
    ) = rendered_src_all.chunk(4, dim=0)
    (
        rendered_self_j_from_i,
        rendered_swap_view_i_from_j,
        rendered_swap_state_j_from_i,
        rendered_swap_both_i_from_j,
    ) = rendered_tgt_all.chunk(4, dim=0)

    # Native/source-view losses supervise the camera frame used to decode the
    # splatter. Cross-view losses supervise the same Gaussian scene after
    # applying the paired relative camera transform.
    rec_self = compute_reconstruction_loss(
        rendered_self_i,
        image_i_t_01,
        ssim_weight=cfg_train.ssim_weight,
    )
    rec_swap_view = compute_reconstruction_loss(
        rendered_swap_view_j,
        image_j_t_01,
        ssim_weight=cfg_train.ssim_weight,
    )
    rec_swap_state = compute_reconstruction_loss(
        rendered_swap_state_i,
        image_i_tk_01,
        ssim_weight=cfg_train.ssim_weight,
    )
    rec_swap_both = compute_reconstruction_loss(
        rendered_swap_both_j,
        image_j_tk_01,
        ssim_weight=cfg_train.ssim_weight,
    )
    rec_self_cross = compute_reconstruction_loss(
        rendered_self_j_from_i,
        image_j_t_01,
        ssim_weight=cfg_train.ssim_weight,
    )
    rec_swap_view_cross = compute_reconstruction_loss(
        rendered_swap_view_i_from_j,
        image_i_t_01,
        ssim_weight=cfg_train.ssim_weight,
    )
    rec_swap_state_cross = compute_reconstruction_loss(
        rendered_swap_state_j_from_i,
        image_j_tk_01,
        ssim_weight=cfg_train.ssim_weight,
    )
    rec_swap_both_cross = compute_reconstruction_loss(
        rendered_swap_both_i_from_j,
        image_i_tk_01,
        ssim_weight=cfg_train.ssim_weight,
    )

    # Keep the original shuffle reconstruction supervision, but make the
    # weighting explicit. z_inv carries state/time, z_dep carries view/camera;
    # the three shuffle weights control state, view, and joint shuffles.
    w_inv = float(cfg_train.shuffle_inv_rec_weight)
    w_dep = float(cfg_train.shuffle_dep_rec_weight)
    w_both = float(cfg_train.shuffle_both_rec_weight)
    rec_norm = max(1.0 + w_inv + w_dep + w_both, 1.0e-6)
    rec_native_loss = (
        rec_self
        + w_dep * rec_swap_view
        + w_inv * rec_swap_state
        + w_both * rec_swap_both
    ) / rec_norm
    rec_cross_loss = (
        rec_self_cross
        + w_dep * rec_swap_view_cross
        + w_inv * rec_swap_state_cross
        + w_both * rec_swap_both_cross
    ) / rec_norm
    rec_loss = 0.1 * rec_native_loss + 0.9 * rec_cross_loss

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
        "frustum_loss": frustum_all["frustum_loss"],
        "inactive_ratio_mean": frustum_all["inactive_ratio_mean"],
        "inactive_ratio_src": frustum_all["inactive_ratio_src"],
        "inactive_ratio_tgt": frustum_all["inactive_ratio_tgt"],
        "invalid_depth_ratio_mean": frustum_all["invalid_depth_ratio_mean"],
        "nonfinite_projection_ratio_mean": frustum_all["nonfinite_projection_ratio_mean"],
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
                "gt_i_t": image_i_t_01,
                "gt_j_t": image_j_t_01,
                "gt_i_tk": image_i_tk_01,
                "gt_j_tk": image_j_tk_01,
            }
        )
    return out_dict


def _make_wandb_named_image_panel(
    named_images_01: list[tuple[str, torch.Tensor]],
    max_vis: int = 4,
) -> wandb.Image:
    """Pack multiple named image batches into one W&B media panel.

    W&B creates media panels from logged media keys. Logging every render under
    its own key makes the workspace noisy and can leave many nearly identical
    auto-created panels. A single summary key keeps validation visualization in
    one stable panel while the caption records the row order.
    """
    if not named_images_01:
        raise ValueError("named_images_01 must contain at least one image batch.")

    n_vis = max(1, min(int(max_vis), *(images.shape[0] for _name, images in named_images_01)))
    rows = []
    row_names = []
    for name, images in named_images_01:
        rows.append(images.detach().float().cpu().clamp(0.0, 1.0)[:n_vis])
        row_names.append(name)

    grid = make_grid(torch.cat(rows, dim=0), nrow=n_vis, padding=2)
    return wandb.Image(grid, caption="rows: " + " | ".join(row_names))


@torch.inference_mode()
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
    if valid_dataloader is None or wandb.run is None:
        return

    prev_vae_mode = vae.training
    prev_splatter_mode = splatter_to_gaussians.training

    vae.eval()
    splatter_to_gaussians.eval()

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
        "val/kl_loss": 0.0,
        "val/inv_kl_loss": 0.0,
        "val/dep_kl_loss": 0.0,
        "val/inv_vicreg_loss": 0.0,
        "val/inv_vicreg_invariance_loss": 0.0,
        "val/inv_vicreg_variance_loss": 0.0,
        "val/inv_vicreg_covariance_loss": 0.0,
        "val/inv_encoder_vcreg_loss": 0.0,
        "val/inv_encoder_vcreg_variance_loss": 0.0,
        "val/inv_encoder_vcreg_covariance_loss": 0.0,
        "val/inv_encoder_vcreg_std_mean": 0.0,
        "val/inv_encoder_vcreg_std_min": 0.0,
        "val/dep_infonce_loss": 0.0,
        "val/cross_cov_loss": 0.0,
        "val/z_inv_std_mean": 0.0,
        "val/z_inv_std_min": 0.0,
        "val/frustum_loss": 0.0,
        "val/inactive_pct_mean": 0.0,
        "val/inactive_pct_src": 0.0,
        "val/inactive_pct_tgt": 0.0,
        "val/invalid_depth_pct_mean": 0.0,
        "val/nonfinite_projection_pct_mean": 0.0,
    }

    num_eval_batches = 0
    image_payload: Dict[str, wandb.Image] = {}

    for batch_idx, batch in enumerate(valid_dataloader):
        if cfg_train.val_num_batches > 0 and batch_idx >= cfg_train.val_num_batches:
            break

        image_i_t = batch["image_i_t"].to(device, non_blocking=True)
        image_j_t = batch["image_j_t"].to(device, non_blocking=True)
        image_i_tk = batch["image_i_tk"].to(device, non_blocking=True)
        image_j_tk = batch["image_j_tk"].to(device, non_blocking=True)
        k_i = batch["K_i"].to(device, non_blocking=True)
        k_j = batch["K_j"].to(device, non_blocking=True)
        t_ij = batch["T_ij"].to(device, non_blocking=True)

        latents, inv_kl_loss, dep_kl_loss = encode_temporal_pair_batch(
            vae=vae,
            image_i_t=image_i_t,
            image_j_t=image_j_t,
            image_i_tk=image_i_tk,
            image_j_tk=image_j_tk,
        )

        rep_losses = compute_representation_losses(latents=latents, cfg_train=cfg_train)

        rec_out = compute_reconstruction_and_renders(
            vae=vae,
            splatter_to_gaussians=splatter_to_gaussians,
            splatter_cfg=splatter_cfg,
            image_i_t_01=(image_i_t + 1.0) * 0.5,
            image_j_t_01=(image_j_t + 1.0) * 0.5,
            image_i_tk_01=(image_i_tk + 1.0) * 0.5,
            image_j_tk_01=(image_j_tk + 1.0) * 0.5,
            **_decode_latent_kwargs(latents),
            k_i=k_i,
            k_j=k_j,
            t_ij=t_ij,
            bg=bg,
            cfg_train=cfg_train,
            return_renders=(num_eval_batches == 0),
        )

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
        scalar_sums["val/kl_loss"] += float((inv_kl_loss + dep_kl_loss).item())
        scalar_sums["val/inv_kl_loss"] += float(inv_kl_loss.item())
        scalar_sums["val/dep_kl_loss"] += float(dep_kl_loss.item())
        scalar_sums["val/inv_vicreg_loss"] += float(rep_losses["inv_vicreg_loss"].item())
        scalar_sums["val/inv_vicreg_invariance_loss"] += float(
            rep_losses["inv_vicreg_invariance_loss"].item()
        )
        scalar_sums["val/inv_vicreg_variance_loss"] += float(
            rep_losses["inv_vicreg_variance_loss"].item()
        )
        scalar_sums["val/inv_vicreg_covariance_loss"] += float(
            rep_losses["inv_vicreg_covariance_loss"].item()
        )
        scalar_sums["val/inv_encoder_vcreg_loss"] += float(
            rep_losses["inv_encoder_vcreg_loss"].item()
        )
        scalar_sums["val/inv_encoder_vcreg_variance_loss"] += float(
            rep_losses["inv_encoder_vcreg_variance_loss"].item()
        )
        scalar_sums["val/inv_encoder_vcreg_covariance_loss"] += float(
            rep_losses["inv_encoder_vcreg_covariance_loss"].item()
        )
        scalar_sums["val/inv_encoder_vcreg_std_mean"] += float(
            rep_losses["inv_encoder_vcreg_std_mean"].item()
        )
        scalar_sums["val/inv_encoder_vcreg_std_min"] += float(
            rep_losses["inv_encoder_vcreg_std_min"].item()
        )
        scalar_sums["val/dep_infonce_loss"] += float(rep_losses["dep_infonce_loss"].item())
        scalar_sums["val/cross_cov_loss"] += float(rep_losses["cross_cov_loss"].item())
        scalar_sums["val/z_inv_std_mean"] += float(rep_losses["z_inv_std_mean"].item())
        scalar_sums["val/z_inv_std_min"] += float(rep_losses["z_inv_std_min"].item())
        scalar_sums["val/frustum_loss"] += float(rec_out["frustum_loss"].item())
        scalar_sums["val/inactive_pct_mean"] += float(100.0 * rec_out["inactive_ratio_mean"].item())
        scalar_sums["val/inactive_pct_src"] += float(100.0 * rec_out["inactive_ratio_src"].item())
        scalar_sums["val/inactive_pct_tgt"] += float(100.0 * rec_out["inactive_ratio_tgt"].item())
        scalar_sums["val/invalid_depth_pct_mean"] += float(100.0 * rec_out["invalid_depth_ratio_mean"].item())
        scalar_sums["val/nonfinite_projection_pct_mean"] += float(
            100.0 * rec_out["nonfinite_projection_ratio_mean"].item()
        )

        if num_eval_batches == 0:
            image_payload = {
                "val/render_summary": _make_wandb_named_image_panel(
                    [
                        ("gt_i_t", rec_out["gt_i_t"]),
                        ("gt_j_t", rec_out["gt_j_t"]),
                        ("gt_i_tk", rec_out["gt_i_tk"]),
                        ("gt_j_tk", rec_out["gt_j_tk"]),
                        ("render_self_i", rec_out["rendered_self_i"]),
                        ("render_swap_view_j", rec_out["rendered_swap_view_j"]),
                        ("render_swap_state_i", rec_out["rendered_swap_state_i"]),
                        ("render_swap_both_j", rec_out["rendered_swap_both_j"]),
                        ("render_self_j_from_i", rec_out["rendered_self_j_from_i"]),
                        ("render_swap_view_i_from_j", rec_out["rendered_swap_view_i_from_j"]),
                        ("render_swap_state_j_from_i", rec_out["rendered_swap_state_j_from_i"]),
                        ("render_swap_both_i_from_j", rec_out["rendered_swap_both_i_from_j"]),
                    ],
                    max_vis=cfg_train.val_max_vis,
                ),
            }

        num_eval_batches += 1

    if num_eval_batches == 0:
        vae.train(prev_vae_mode)
        splatter_to_gaussians.train(prev_splatter_mode)
        return

    log_dict: Dict[str, Any] = {
        key: value / float(num_eval_batches) for key, value in scalar_sums.items()
    }
    log_dict["global_step"] = global_step
    log_dict.update(image_payload)
    wandb.log(log_dict, step=global_step)

    vae.train(prev_vae_mode)
    splatter_to_gaussians.train(prev_splatter_mode)


def train_splatter_vae(
    vae: SplatterVAE,
    splatter_cfg: SplatterConfig,
    train_dataloader: DataLoader,
    cfg_train: TrainConfig,
    valid_dataloader: Optional[DataLoader] = None,
    resume_ckpt: Optional[str] = None,
):
    """Train SplatterVAE with rendering, beta-VAE, VICReg, cross-covariance, and frustum losses."""
    device = torch.device(cfg_train.device)
    vae.to(device)

    splatter_to_gaussians = VAESplatterToGaussians(splatter_cfg).to(device)

    optimizer = torch.optim.Adam(vae.parameters(), lr=cfg_train.lr)
    bg = (
        torch.ones(3, device=device)
        if splatter_cfg.data.white_background
        else torch.zeros(3, device=device)
    )

    start_epoch = 0
    global_step = 0

    if resume_ckpt is not None and os.path.isfile(resume_ckpt):
        ckpt = torch.load(resume_ckpt, map_location="cpu")
        vae.load_state_dict(ckpt["vae_state_dict"])
        splatter_to_gaussians.load_state_dict(ckpt["splatter_to_gaussians_state_dict"])
        try:
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        except ValueError as exc:
            print(f"[Resume] Skipping optimizer state because parameter groups changed: {exc}")
        start_epoch = int(ckpt["epoch"])
        global_step = int(ckpt["global_step"])

    os.makedirs(cfg_train.ckpt_dir, exist_ok=True)

    epoch = start_epoch
    while True:
        for step, batch in enumerate(train_dataloader):
            if cfg_train.max_global_steps is not None and global_step >= cfg_train.max_global_steps:
                print(f"[Stop] Reached max_global_steps={cfg_train.max_global_steps}.")
                return

            vae.train()
            splatter_to_gaussians.train()

            # Move the sampled two-view temporal batch to device. Shapes:
            # image_*: (B,3,H,W), K_*: (B,3,3), T_ij: (B,4,4).
            image_i_t = batch["image_i_t"].to(device, non_blocking=True)
            image_j_t = batch["image_j_t"].to(device, non_blocking=True)
            image_i_tk = batch["image_i_tk"].to(device, non_blocking=True)
            image_j_tk = batch["image_j_tk"].to(device, non_blocking=True)
            k_i = batch["K_i"].to(device, non_blocking=True)
            k_j = batch["K_j"].to(device, non_blocking=True)
            t_ij = batch["T_ij"].to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            latents, inv_kl_loss, dep_kl_loss = encode_temporal_pair_batch(
                vae=vae,
                image_i_t=image_i_t,
                image_j_t=image_j_t,
                image_i_tk=image_i_tk,
                image_j_tk=image_j_tk,
            )

            rec_out = compute_reconstruction_and_renders(
                vae=vae,
                splatter_to_gaussians=splatter_to_gaussians,
                splatter_cfg=splatter_cfg,
                image_i_t_01=(image_i_t + 1.0) * 0.5,
                image_j_t_01=(image_j_t + 1.0) * 0.5,
                image_i_tk_01=(image_i_tk + 1.0) * 0.5,
                image_j_tk_01=(image_j_tk + 1.0) * 0.5,
                **_decode_latent_kwargs(latents),
                k_i=k_i,
                k_j=k_j,
                t_ij=t_ij,
                bg=bg,
                cfg_train=cfg_train,
                return_renders=False,
            )
            rec_loss = rec_out["rec_loss"]
            frustum_loss = rec_out["frustum_loss"]

            rep_losses = compute_representation_losses(latents=latents, cfg_train=cfg_train)

            kl_loss = inv_kl_loss + dep_kl_loss
            total_loss = (
                cfg_train.rec_weight * rec_loss
                + cfg_train.kl_weight * kl_loss
                + cfg_train.inv_vicreg_weight * rep_losses["inv_vicreg_loss"]
                + cfg_train.inv_encoder_vcreg_weight * rep_losses["inv_encoder_vcreg_loss"]
                + cfg_train.dep_infonce_weight * rep_losses["dep_infonce_loss"]
                + cfg_train.cross_cov_weight * rep_losses["cross_cov_loss"]
                + cfg_train.frustum_weight * frustum_loss
            )

            finite_terms = {
                "rec_loss": rec_loss,
                "kl_loss": kl_loss,
                "inv_vicreg_loss": rep_losses["inv_vicreg_loss"],
                "inv_encoder_vcreg_loss": rep_losses["inv_encoder_vcreg_loss"],
                "dep_infonce_loss": rep_losses["dep_infonce_loss"],
                "cross_cov_loss": rep_losses["cross_cov_loss"],
                "frustum_loss": frustum_loss,
                "total_loss": total_loss,
            }
            bad_terms = [name for name, value in finite_terms.items() if not torch.isfinite(value).all()]
            if bad_terms:
                print(
                    f"[Warn] Non-finite loss detected at global_step={global_step} "
                    f"(bad={bad_terms}, "
                    f"inactive_pct={100.0 * rec_out['inactive_ratio_mean'].item():.2f}, "
                    f"invalid_depth_pct={100.0 * rec_out['invalid_depth_ratio_mean'].item():.2f}, "
                    f"nonfinite_proj_pct={100.0 * rec_out['nonfinite_projection_ratio_mean'].item():.2f}). "
                    "Skipping optimizer step."
                )
                if wandb.run is not None:
                    wandb.log(
                        {"global_step": global_step, "train/nonfinite_batch": 1.0},
                        step=global_step,
                    )
                global_step += 1
                continue

            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(vae.parameters(), max_norm=5.0)
            optimizer.step()

            if step % 250 == 0:
                print(
                    f"[Epoch {epoch+1} | Step {step} | Global {global_step}] "
                    f"Loss={total_loss.item():.4f} "
                    f"(rec={rec_loss.item():.4f}, "
                    f"native={rec_out['rec_native_loss'].item():.4f}, "
                    f"cross={rec_out['rec_cross_loss'].item():.4f}, "
                    f"self={rec_out['rec_self'].item():.4f}, "
                    f"swap_view={rec_out['rec_swap_view'].item():.4f}, "
                    f"swap_state={rec_out['rec_swap_state'].item():.4f}, "
                    f"swap_both={rec_out['rec_swap_both'].item():.4f}, "
                    f"kl={kl_loss.item():.4f}, "
                    f"inv_vicreg={rep_losses['inv_vicreg_loss'].item():.4f}, "
                    f"inv_sim={rep_losses['inv_vicreg_invariance_loss'].item():.4f}, "
                    f"inv_std={rep_losses['inv_vicreg_variance_loss'].item():.4f}, "
                    f"inv_cov={rep_losses['inv_vicreg_covariance_loss'].item():.4f}, "
                    f"enc_vc={rep_losses['inv_encoder_vcreg_loss'].item():.4f}, "
                    f"enc_std={rep_losses['inv_encoder_vcreg_std_mean'].item():.4f}, "
                    f"dep_nce={rep_losses['dep_infonce_loss'].item():.4f}, "
                    f"cross_cov={rep_losses['cross_cov_loss'].item():.4f}, "
                    f"z_inv_std={rep_losses['z_inv_std_mean'].item():.4f}, "
                    f"frustum={frustum_loss.item():.4f})"
                )
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
                            "train/shuffle_inv_rec_weight": float(cfg_train.shuffle_inv_rec_weight),
                            "train/shuffle_dep_rec_weight": float(cfg_train.shuffle_dep_rec_weight),
                            "train/shuffle_both_rec_weight": float(cfg_train.shuffle_both_rec_weight),
                            "train/kl_loss": kl_loss.item(),
                            "train/inv_kl_loss": inv_kl_loss.item(),
                            "train/dep_kl_loss": dep_kl_loss.item(),
                            "train/kl_weight": float(cfg_train.kl_weight),
                            "train/inv_vicreg_loss": rep_losses["inv_vicreg_loss"].item(),
                            "train/inv_vicreg_weight": float(cfg_train.inv_vicreg_weight),
                            "train/inv_vicreg_invariance_loss": (
                                rep_losses["inv_vicreg_invariance_loss"].item()
                            ),
                            "train/inv_vicreg_variance_loss": (
                                rep_losses["inv_vicreg_variance_loss"].item()
                            ),
                            "train/inv_vicreg_covariance_loss": (
                                rep_losses["inv_vicreg_covariance_loss"].item()
                            ),
                            "train/vicreg_sim_coeff": float(cfg_train.vicreg_sim_coeff),
                            "train/vicreg_std_coeff": float(cfg_train.vicreg_std_coeff),
                            "train/vicreg_cov_coeff": float(cfg_train.vicreg_cov_coeff),
                            "train/vicreg_std_gamma": float(cfg_train.vicreg_std_gamma),
                            "train/inv_encoder_vcreg_loss": rep_losses[
                                "inv_encoder_vcreg_loss"
                            ].item(),
                            "train/inv_encoder_vcreg_weight": float(
                                cfg_train.inv_encoder_vcreg_weight
                            ),
                            "train/inv_encoder_vcreg_variance_loss": rep_losses[
                                "inv_encoder_vcreg_variance_loss"
                            ].item(),
                            "train/inv_encoder_vcreg_covariance_loss": rep_losses[
                                "inv_encoder_vcreg_covariance_loss"
                            ].item(),
                            "train/inv_encoder_vcreg_std_mean": rep_losses[
                                "inv_encoder_vcreg_std_mean"
                            ].item(),
                            "train/inv_encoder_vcreg_std_min": rep_losses[
                                "inv_encoder_vcreg_std_min"
                            ].item(),
                            "train/inv_encoder_vcreg_std_coeff": float(
                                cfg_train.inv_encoder_vcreg_std_coeff
                            ),
                            "train/inv_encoder_vcreg_cov_coeff": float(
                                cfg_train.inv_encoder_vcreg_cov_coeff
                            ),
                            "train/inv_encoder_vcreg_std_gamma": float(
                                cfg_train.inv_encoder_vcreg_std_gamma
                            ),
                            "train/inv_encoder_vcreg_cov_smooth_l1_delta": float(
                                cfg_train.inv_encoder_vcreg_cov_smooth_l1_delta
                            ),
                            "train/dep_infonce_loss": rep_losses["dep_infonce_loss"].item(),
                            "train/dep_infonce_weight": float(cfg_train.dep_infonce_weight),
                            "train/dep_infonce_temperature": float(
                                cfg_train.dep_infonce_temperature
                            ),
                            "train/cross_cov_loss": rep_losses["cross_cov_loss"].item(),
                            "train/cross_cov_weight": float(cfg_train.cross_cov_weight),
                            "train/z_inv_std_mean": rep_losses["z_inv_std_mean"].item(),
                            "train/z_inv_std_min": rep_losses["z_inv_std_min"].item(),
                            "train/frustum_loss": frustum_loss.item(),
                            "train/frustum_loss_weighted": (cfg_train.frustum_weight * frustum_loss).item(),
                            "train/inactive_pct_mean": 100.0 * rec_out["inactive_ratio_mean"].item(),
                            "train/inactive_pct_src": 100.0 * rec_out["inactive_ratio_src"].item(),
                            "train/inactive_pct_tgt": 100.0 * rec_out["inactive_ratio_tgt"].item(),
                            "train/invalid_depth_pct_mean": 100.0 * rec_out["invalid_depth_ratio_mean"].item(),
                            "train/nonfinite_projection_pct_mean": (
                                100.0 * rec_out["nonfinite_projection_ratio_mean"].item()
                            ),
                            "train/frustum_weight": float(cfg_train.frustum_weight),
                            "global_step": global_step,
                        },
                        step=global_step,
                    )

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
