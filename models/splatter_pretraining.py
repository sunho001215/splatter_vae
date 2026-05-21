from __future__ import annotations

import math
import os
from typing import Any, Dict, Optional

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.utils import make_grid

import wandb

from models.losses import (
    compute_reconstruction_loss,
    infonce_loss,
)
from models.splatter import SplatterConfig, VAESplatterToGaussians, render_predicted
from models.splatter_train_config import TrainConfig
from models.vae import SplatterVAE


def _normalize_lr_schedule(schedule: str) -> str:
    schedule = str(schedule or "constant").strip().lower().replace("-", "_")
    aliases = {
        "none": "constant",
        "off": "constant",
        "constant": "constant",
        "cosine": "warmup_cosine",
        "cosine_annealing": "warmup_cosine",
        "warmup_cosine": "warmup_cosine",
        "cosine_warmup": "warmup_cosine",
    }
    if schedule not in aliases:
        raise ValueError(
            f"Unknown lr_schedule={schedule!r}. "
            "Use one of: constant, warmup_cosine, cosine."
        )
    return aliases[schedule]


def _resolve_lr_total_steps(cfg_train: TrainConfig, train_dataloader: DataLoader) -> int:
    if cfg_train.lr_total_steps is not None:
        total_steps = int(cfg_train.lr_total_steps)
    elif cfg_train.max_global_steps is not None:
        total_steps = int(cfg_train.max_global_steps)
    else:
        try:
            total_steps = int(cfg_train.num_epochs) * len(train_dataloader)
        except TypeError:
            total_steps = int(cfg_train.lr_warmup_steps) + 1
    return max(1, total_steps)


def _compute_scheduled_lr(cfg_train: TrainConfig, global_step: int, total_steps: int) -> float:
    peak_lr = float(cfg_train.lr)
    schedule = _normalize_lr_schedule(cfg_train.lr_schedule)
    if schedule == "constant":
        return peak_lr

    min_lr = float(cfg_train.min_lr)
    if peak_lr < 0.0:
        raise ValueError(f"lr must be non-negative, got {peak_lr}.")
    if min_lr < 0.0:
        raise ValueError(f"min_lr must be non-negative, got {min_lr}.")
    if min_lr > peak_lr:
        raise ValueError(f"min_lr ({min_lr}) must be <= lr ({peak_lr}).")

    step = max(0, int(global_step))
    warmup_steps = max(0, int(cfg_train.lr_warmup_steps))
    if warmup_steps > 0 and step < warmup_steps:
        return peak_lr * float(step + 1) / float(warmup_steps)

    decay_steps = max(1, int(total_steps) - warmup_steps)
    progress = min(1.0, max(0.0, float(step - warmup_steps) / float(decay_steps)))
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return min_lr + (peak_lr - min_lr) * cosine


def _set_optimizer_lr(optimizer: torch.optim.Optimizer, lr: float) -> None:
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr


def encode_temporal_pair_batch(
    vae: SplatterVAE,
    image_i_t: torch.Tensor,
    image_j_t: torch.Tensor,
    image_i_tk: torch.Tensor,
    image_j_tk: torch.Tensor,
) -> tuple[Dict[str, torch.Tensor], torch.Tensor, torch.Tensor]:
    """Encode two random views at t and the same views at temporal index tk.

    Args:
        image_i_t, image_j_t: two randomly sampled camera views from the same
            demo/timestep. They share view-invariant content.
        image_i_tk, image_j_tk: the same camera viewpoints from a different
            timestep in the same demo. They provide same-view features for
            temporal shuffling.

    Returns:
        A dict of token tensors keyed by image name, plus invariant/dependent
        VQ losses averaged over the four encoded images.
    """
    bsz, channels, height, width = image_i_t.shape
    images = torch.stack((image_i_t, image_j_t, image_i_tk, image_j_tk), dim=1)
    flat_images = images.reshape(bsz * 4, channels, height, width).contiguous()

    z_inv, inv_vq_loss, z_dep, dep_vq_loss, _ = vae.encode(flat_images)

    z_inv = z_inv.reshape(bsz, 4, *z_inv.shape[1:]).contiguous()
    z_dep = z_dep.reshape(bsz, 4, *z_dep.shape[1:]).contiguous()
    latents = {
        "z_inv_i_t": z_inv[:, 0],
        "z_inv_j_t": z_inv[:, 1],
        "z_inv_i_tk": z_inv[:, 2],
        "z_inv_j_tk": z_inv[:, 3],
        "z_dep_i_t": z_dep[:, 0],
        "z_dep_j_t": z_dep[:, 1],
        "z_dep_i_tk": z_dep[:, 2],
        "z_dep_j_tk": z_dep[:, 3],
    }
    return latents, inv_vq_loss, dep_vq_loss


def _flatten_latent_tokens(z: torch.Tensor) -> torch.Tensor:
    """Flatten one token tensor to the 2D shape required by infonce_loss."""
    z = torch.nan_to_num(z, nan=0.0, posinf=0.0, neginf=0.0)
    return z.reshape(z.shape[0], -1)


def compute_invariant_variance_loss(
    z_inv_i_t: torch.Tensor,
    z_inv_j_t: torch.Tensor,
    z_inv_i_tk: torch.Tensor,
    z_inv_j_tk: torch.Tensor,
    gamma: float,
    eps: float = 1e-4,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Penalize invariant features whose batch std is close to collapse."""
    z = torch.cat(
        (
            _flatten_latent_tokens(z_inv_i_t),
            _flatten_latent_tokens(z_inv_j_t),
            _flatten_latent_tokens(z_inv_i_tk),
            _flatten_latent_tokens(z_inv_j_tk),
        ),
        dim=0,
    )
    z = z - z.mean(dim=0, keepdim=True)
    std = torch.sqrt(z.var(dim=0, unbiased=False) + eps)
    loss = F.relu(float(gamma) - std).mean()
    return loss, std.mean(), std.min()


def compute_contrastive_losses(
    z_inv_i_t: torch.Tensor,
    z_inv_j_t: torch.Tensor,
    z_inv_i_tk: torch.Tensor,
    z_inv_j_tk: torch.Tensor,
    z_dep_i_t: torch.Tensor,
    z_dep_j_t: torch.Tensor,
    z_dep_i_tk: torch.Tensor,
    z_dep_j_tk: torch.Tensor,
    temperature: float,
):
    """Compute two-view temporal contrastive losses.

    The loss is intentionally built from image_{i,j}_t and image_{i,j}_tk
    within each sampled demo item, rather than shuffling across batch rows whose
    camera IDs may differ.

    - Invariant positives: same demo/timestep, different camera.
    - Dependent positives: same camera, different timestep.
    """
    zi_t = _flatten_latent_tokens(z_inv_i_t)
    zj_t = _flatten_latent_tokens(z_inv_j_t)
    zi_tk = _flatten_latent_tokens(z_inv_i_tk)
    zj_tk = _flatten_latent_tokens(z_inv_j_tk)

    # For z_inv, the positive is the other camera at the same timestep. The
    # paired negatives are both same-demo camera views at the other timestep.
    inv_query = torch.cat((zi_t, zj_t, zi_tk, zj_tk), dim=0)
    inv_positive = torch.cat((zj_t, zi_t, zj_tk, zi_tk), dim=0)
    inv_negative = torch.cat(
        (
            torch.stack((zi_tk, zj_tk), dim=1),
            torch.stack((zi_tk, zj_tk), dim=1),
            torch.stack((zi_t, zj_t), dim=1),
            torch.stack((zi_t, zj_t), dim=1),
        ),
        dim=0,
    )
    inv_contrastive_loss = infonce_loss(
        query=inv_query,
        positive_keys=inv_positive,
        negative_keys=inv_negative,
        temperature=temperature,
        negative_mode="mixed",
    )

    zdi_t = _flatten_latent_tokens(z_dep_i_t)
    zdj_t = _flatten_latent_tokens(z_dep_j_t)
    zdi_tk = _flatten_latent_tokens(z_dep_i_tk)
    zdj_tk = _flatten_latent_tokens(z_dep_j_tk)

    # For z_dep, the positive is the same camera at tk/t. The paired negatives
    # are both temporal states from the other sampled camera, avoiding
    # cross-batch comparisons between inconsistent random camera pairs.
    dep_query = torch.cat((zdi_t, zdi_tk, zdj_t, zdj_tk), dim=0)
    dep_positive = torch.cat((zdi_tk, zdi_t, zdj_tk, zdj_t), dim=0)
    dep_negative = torch.cat(
        (
            torch.stack((zdj_t, zdj_tk), dim=1),
            torch.stack((zdj_t, zdj_tk), dim=1),
            torch.stack((zdi_t, zdi_tk), dim=1),
            torch.stack((zdi_t, zdi_tk), dim=1),
        ),
        dim=0,
    )
    dep_contrastive_loss = infonce_loss(
        query=dep_query,
        positive_keys=dep_positive,
        negative_keys=dep_negative,
        temperature=temperature,
        negative_mode="paired",
    )

    return inv_contrastive_loss, dep_contrastive_loss


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
    rec_native_loss = 0.25 * (rec_self + rec_swap_view + rec_swap_state + rec_swap_both)

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
    rec_cross_loss = 0.25 * (
        rec_self_cross
        + rec_swap_view_cross
        + rec_swap_state_cross
        + rec_swap_both_cross
    )
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
        "val/inv_contrastive_loss": 0.0,
        "val/inv_variance_loss": 0.0,
        "val/z_inv_std_mean": 0.0,
        "val/z_inv_std_min": 0.0,
        "val/dep_contrastive_loss": 0.0,
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

        latents, _inv_vq_loss, _dep_vq_loss = encode_temporal_pair_batch(
            vae=vae,
            image_i_t=image_i_t,
            image_j_t=image_j_t,
            image_i_tk=image_i_tk,
            image_j_tk=image_j_tk,
        )

        inv_contrastive_loss, dep_contrastive_loss = compute_contrastive_losses(
            **latents,
            temperature=cfg_train.temperature,
        )
        inv_variance_loss, z_inv_std_mean, z_inv_std_min = compute_invariant_variance_loss(
            latents["z_inv_i_t"],
            latents["z_inv_j_t"],
            latents["z_inv_i_tk"],
            latents["z_inv_j_tk"],
            gamma=cfg_train.inv_variance_gamma,
        )

        rec_out = compute_reconstruction_and_renders(
            vae=vae,
            splatter_to_gaussians=splatter_to_gaussians,
            splatter_cfg=splatter_cfg,
            image_i_t_01=(image_i_t + 1.0) * 0.5,
            image_j_t_01=(image_j_t + 1.0) * 0.5,
            image_i_tk_01=(image_i_tk + 1.0) * 0.5,
            image_j_tk_01=(image_j_tk + 1.0) * 0.5,
            **latents,
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
        scalar_sums["val/inv_contrastive_loss"] += float(inv_contrastive_loss.item())
        scalar_sums["val/inv_variance_loss"] += float(inv_variance_loss.item())
        scalar_sums["val/z_inv_std_mean"] += float(z_inv_std_mean.item())
        scalar_sums["val/z_inv_std_min"] += float(z_inv_std_min.item())
        scalar_sums["val/dep_contrastive_loss"] += float(dep_contrastive_loss.item())
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
    """Train SplatterVAE with two-view temporal rendering, contrastive, VQ, and frustum losses."""
    device = torch.device(cfg_train.device)
    vae.to(device)

    splatter_to_gaussians = VAESplatterToGaussians(splatter_cfg).to(device)

    optimizer = torch.optim.Adam(vae.parameters(), lr=cfg_train.lr)
    lr_total_steps = _resolve_lr_total_steps(cfg_train, train_dataloader)
    lr_schedule = _normalize_lr_schedule(cfg_train.lr_schedule)
    if lr_schedule != "constant":
        print(
            f"[LR] schedule={lr_schedule}, peak_lr={cfg_train.lr:g}, "
            f"min_lr={cfg_train.min_lr:g}, warmup_steps={cfg_train.lr_warmup_steps}, "
            f"total_steps={lr_total_steps}"
        )
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

            current_lr = _compute_scheduled_lr(cfg_train, global_step, lr_total_steps)
            _set_optimizer_lr(optimizer, current_lr)

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

            latents, inv_vq_loss, dep_vq_loss = encode_temporal_pair_batch(
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
                **latents,
                k_i=k_i,
                k_j=k_j,
                t_ij=t_ij,
                bg=bg,
                cfg_train=cfg_train,
                return_renders=False,
            )
            rec_loss = rec_out["rec_loss"]
            frustum_loss = rec_out["frustum_loss"]

            inv_contrastive_loss, dep_contrastive_loss = compute_contrastive_losses(
                **latents,
                temperature=cfg_train.temperature,
            )
            inv_variance_loss, z_inv_std_mean, z_inv_std_min = compute_invariant_variance_loss(
                latents["z_inv_i_t"],
                latents["z_inv_j_t"],
                latents["z_inv_i_tk"],
                latents["z_inv_j_tk"],
                gamma=cfg_train.inv_variance_gamma,
            )

            vq_loss = inv_vq_loss + dep_vq_loss
            total_loss = (
                cfg_train.rec_weight * rec_loss
                + cfg_train.vq_weight * vq_loss
                + cfg_train.inv_contrastive_weight * inv_contrastive_loss
                + cfg_train.inv_variance_weight * inv_variance_loss
                + cfg_train.dep_contrastive_weight * dep_contrastive_loss
                + cfg_train.frustum_weight * frustum_loss
            )

            finite_terms = {
                "rec_loss": rec_loss,
                "vq_loss": vq_loss,
                "inv_contrastive_loss": inv_contrastive_loss,
                "inv_variance_loss": inv_variance_loss,
                "dep_contrastive_loss": dep_contrastive_loss,
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
                        {
                            "global_step": global_step,
                            "train/nonfinite_batch": 1.0,
                            "train/lr": current_lr,
                        },
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
                    f"lr={current_lr:.2e} "
                    f"(rec={rec_loss.item():.4f}, "
                    f"native={rec_out['rec_native_loss'].item():.4f}, "
                    f"cross={rec_out['rec_cross_loss'].item():.4f}, "
                    f"self={rec_out['rec_self'].item():.4f}, "
                    f"swap_view={rec_out['rec_swap_view'].item():.4f}, "
                    f"swap_state={rec_out['rec_swap_state'].item():.4f}, "
                    f"swap_both={rec_out['rec_swap_both'].item():.4f}, "
                    f"vq={vq_loss.item():.4f}, "
                    f"inv_con={inv_contrastive_loss.item():.4f}, "
                    f"inv_var={inv_variance_loss.item():.4f}, "
                    f"z_inv_std={z_inv_std_mean.item():.4f}, "
                    f"dep_con={dep_contrastive_loss.item():.4f}, "
                    f"frustum={frustum_loss.item():.4f})"
                )
                if wandb.run is not None:
                    wandb.log(
                        {
                            "train/total_loss": total_loss.item(),
                            "train/lr": current_lr,
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
                            "train/inv_contrastive_weight": float(cfg_train.inv_contrastive_weight),
                            "train/inv_variance_loss": inv_variance_loss.item(),
                            "train/inv_variance_weight": float(cfg_train.inv_variance_weight),
                            "train/inv_variance_gamma": float(cfg_train.inv_variance_gamma),
                            "train/z_inv_std_mean": z_inv_std_mean.item(),
                            "train/z_inv_std_min": z_inv_std_min.item(),
                            "train/dep_contrastive_loss": dep_contrastive_loss.item(),
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
