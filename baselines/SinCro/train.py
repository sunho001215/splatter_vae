"""
Training SinCro with your custom HDF5 dataset.

Matches the original MV_run_nerf.py training logic:
  1. NeRF reconstructs ALL remaining views (V - num_ref_view).
     N_rand rays are sampled from the pooled B*(V-ref_V)*H*W pixels.
  2. Latent is tiled across all views before selecting remain_view_index.
  3. Fine-network loss (img_loss0 / extras['rgb0']) is included.
  4. Contrastive weight is hardcoded 0.0004.
  5. Negative sample encoder forward is wrapped in torch.no_grad().
  6. Checkpoints save all weights (network_fn, network_fine, latent_embed,
     optimizer) for full resume.
  7. Evaluation renders full-resolution images for all views and logs
     rendered vs GT images to wandb.
"""

import os
import sys
import math
import argparse
import time as time_module
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import yaml
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split

import wandb
from torchvision.utils import make_grid
from einops import rearrange, repeat

# Ensure repo root is on sys.path so absolute imports work when run as a script.
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

# Import SinCro modules
from baselines.SinCro.sincro.MV_run_nerf import (
    create_nerf,
    render,
    img2mse,
    mse2psnr,
    get_rays,
    get_rays_np,
)
from baselines.SinCro.sincro.MV_mae_encoder import MaskedViTEncoder

from baselines.SinCro.dataloader import (
    RobosuiteSinCroDatasetConfig,
    RobosuiteSinCroSequenceDataset,
)

# -------------------------------------------------------------------------
# YAML config dataclasses
# -------------------------------------------------------------------------

@dataclass
class DatasetConfig:
    hdf5_path: str
    num_views: int = 6
    sequence_length: int = 3
    max_episodes: Optional[int] = None
    max_frames_per_demo: Optional[int] = None
    temporal_stride: int = 3
    num_workers: int = 4
    pin_memory: bool = True
    batch_size: int = 8
    train_ratio: float = 0.96


@dataclass
class SinCroModelConfig:
    # NeRF MLP
    netdepth: int = 8
    netwidth: int = 256
    netdepth_fine: int = 8
    netwidth_fine: int = 256
    N_rand: int = 2048
    N_samples: int = 64
    N_importance: int = 128
    multires: int = 10
    multires_views: int = 4
    i_embed: int = 0
    use_viewdirs: bool = True
    raw_noise_std: float = 0.0
    white_bkgd: bool = False
    perturb: float = 1.0
    lindisp: bool = False
    # MAE / ViT encoder
    img_size: int = 128
    patch_size: int = 16
    embed_dim: int = 256
    vit_depth: int = 4
    vit_num_heads: int = 4
    vit_mlp_dim: int = 1024
    decoder_depth: int = 2
    decoder_num_heads: int = 2
    decoder_mlp_dim: int = 1024
    decoder_output_dim: int = 256
    # SinCro-specific
    time_interval: int = 3
    mask_ratio: float = 0.75
    num_views: int = 6
    # NeRF near / far
    near: float = 0.1
    far: float = 2.5
    chunk: int = 1024 * 32
    netchunk: int = 1024 * 64
    # Contrastive margin (from peg.txt: enc_contrastive_margin = 0.2)
    enc_contrastive_margin: float = 0.2
    # Precrop
    precrop_iters: int = 2000
    precrop_frac: float = 0.5


@dataclass
class TrainConfig:
    num_epochs: int = 50
    max_global_steps: Optional[int] = 300001
    lrate: float = 5e-4
    lrate_decay: int = 500   # in 1000 steps -- peg.txt uses 500
    device: str = "cuda"
    # Logging / eval / ckpt
    eval_every: int = 1000
    save_every: int = 5000
    i_print: int = 100
    i_img: int = 500
    ckpt_dir: str = "./checkpoints_sincro"
    resume_from_last: bool = False
    seed: int = 42


@dataclass
class WandbConfig:
    project: str = "SinCro-MetaWorld"
    entity: Optional[str] = None
    run_name: Optional[str] = None


@dataclass
class ExperimentConfig:
    basedir: str = "./logs/SinCro/metaworld/"
    expname: str = "sincro_metaworld"


# -------------------------------------------------------------------------
# SimpleArgs: shim for create_nerf()
# -------------------------------------------------------------------------

class SimpleArgs:
    """Tiny shim so we can call create_nerf(args, ...) without touching SinCro code."""

    def __init__(
        self,
        model_cfg: SinCroModelConfig,
        train_cfg: TrainConfig,
        dataset_cfg: DatasetConfig,
        exp_cfg: ExperimentConfig,
    ):
        # NeRF MLP
        self.netdepth = model_cfg.netdepth
        self.netwidth = model_cfg.netwidth
        self.netdepth_fine = model_cfg.netdepth_fine
        self.netwidth_fine = model_cfg.netwidth_fine
        self.N_rand = model_cfg.N_rand
        self.N_samples = model_cfg.N_samples
        self.N_importance = model_cfg.N_importance
        self.multires = model_cfg.multires
        self.multires_views = model_cfg.multires_views
        self.i_embed = model_cfg.i_embed
        self.use_viewdirs = model_cfg.use_viewdirs
        self.raw_noise_std = model_cfg.raw_noise_std
        self.white_bkgd = model_cfg.white_bkgd
        self.perturb = model_cfg.perturb
        self.lindisp = model_cfg.lindisp

        # MAE / ViT
        self.img_size = model_cfg.img_size
        self.patch_size = model_cfg.patch_size
        self.embed_dim = model_cfg.embed_dim
        self.vit_depth = model_cfg.vit_depth
        self.vit_num_heads = model_cfg.vit_num_heads
        self.vit_mlp_dim = model_cfg.vit_mlp_dim
        self.decoder_depth = model_cfg.decoder_depth
        self.decoder_num_heads = model_cfg.decoder_num_heads
        self.decoder_mlp_dim = model_cfg.decoder_mlp_dim
        self.decoder_output_dim = model_cfg.decoder_output_dim
        self.vit_encoder_mlp_dim = model_cfg.vit_mlp_dim
        self.vit_decoder_mlp_dim = model_cfg.decoder_mlp_dim

        # SinCro-specific bits
        self.time_interval = model_cfg.time_interval
        self.num_view = model_cfg.num_views
        self.batch_size = dataset_cfg.batch_size
        self.lrate = train_cfg.lrate

        # Misc flags for create_nerf
        self.no_reload = True
        self.ft_path = None
        self.dataset_type = "metaworld"
        self.basedir = exp_cfg.basedir
        self.expname = exp_cfg.expname
        self.N_rgb = 0
        self.no_ndc = True
        self.render_only = False
        self.render_test = False
        self.render_factor = 1
        self.precrop_iters = model_cfg.precrop_iters
        self.precrop_frac = model_cfg.precrop_frac
        self.N_iters = train_cfg.max_global_steps or 300001
        self.i_embed_views = 0
        self.i_embed_state = -1
        self.chunk = model_cfg.chunk
        self.netchunk = model_cfg.netchunk
        self.lr_decay = train_cfg.lrate_decay
        self.use_mae = True
        self.mask_ratio = model_cfg.mask_ratio
        self.gamma = 1.0
        self.log_wandb = False
        self.enc_contrastive_margin = model_cfg.enc_contrastive_margin
        self.render_pose_path = None
        self.render_episode = None


# -------------------------------------------------------------------------
# Helpers
# -------------------------------------------------------------------------

def update_learning_rate(optimizer, train_cfg, global_step):
    decay_rate = 0.1
    decay_steps = train_cfg.lrate_decay * 1000
    new_lrate = train_cfg.lrate * (decay_rate ** (global_step / decay_steps))
    for param_group in optimizer.param_groups:
        param_group["lr"] = new_lrate
    return new_lrate


def distance(x1, x2):
    diff = torch.abs(x1 - x2)
    return torch.pow(diff, 2).sum(dim=1)


def to8b(x):
    """Convert float [0,1] array/tensor to uint8 [0,255]."""
    return (255 * np.clip(x, 0, 1)).astype(np.uint8)


# -------------------------------------------------------------------------
# Encode a batch through ViT (shared between train and eval)
# -------------------------------------------------------------------------

def encode_sincro(
    images: torch.Tensor,
    latent_embed: MaskedViTEncoder,
    model_cfg: SinCroModelConfig,
    mask_ratio: float = 0.0,
    view_index: Optional[int] = None,
    ref_view_indices: Optional[List[int]] = None,
):
    """
    Run SinCro image encoder + state encoder on a batch of images.

    Args:
        images: [B, T, H, V, W, C]
        mask_ratio: 0.0 for eval, model_cfg.mask_ratio for training
        view_index, ref_view_indices: if None, randomly chosen

    Returns:
        latent: raw state encoder output
        anchor_latent: [B, feat_dim]
        positive_latent: [B, feat_dim]
        view_index: int  (the primary view chosen)
        ref_view_indices: list[int]
    """
    B, T, H, V, W, C = images.shape

    batch_images_for_vit = images.reshape(B, T, H, V * W, C)
    per_view = torch.split(batch_images_for_vit, W, dim=3)

    num_ref_view = V // 3
    if view_index is None or ref_view_indices is None:
        indices = np.random.choice(V, size=1 + num_ref_view, replace=False)
        view_index = int(indices[0])
        ref_view_indices = indices[1:].tolist()

    primary_images = per_view[view_index]
    ref_images = torch.cat([per_view[idx] for idx in ref_view_indices], dim=3)

    # Primary encoder (with masking)
    latent, mask, ids_restore = latent_embed.SinCro_image_encoder(
        primary_images, mask_ratio, T, is_ref=False
    )

    # Reference encoder (no masking)
    ref_views_list = torch.split(ref_images, W, dim=3)
    ref_for_encoder = torch.cat(ref_views_list, dim=0)
    ref_latent, _, _ = latent_embed.SinCro_image_encoder(
        ref_for_encoder, 0, T, is_ref=True
    )
    ref_latent = rearrange(
        ref_latent[:, 1:, :], "b (t hw) d -> b t hw d", t=T
    )[:, -1]
    ref_latent = rearrange(ref_latent, "(v b) hw d -> b (v hw) d", b=B)

    # State encoder
    latent, mask, ids_restore = latent_embed.SinCro_state_encoder(
        latent, ref_latent, mask, ids_restore
    )
    anchor_latent = latent_embed.input_feature
    positive_latent = latent_embed.ref_feature

    return latent, anchor_latent, positive_latent, view_index, ref_view_indices


# -------------------------------------------------------------------------
# Forward pass -- training (matches original MV_run_nerf.py)
# -------------------------------------------------------------------------

def forward_sincro_batch(
    batch: Dict[str, torch.Tensor],
    latent_embed: MaskedViTEncoder,
    render_kwargs_train: Dict[str, Any],
    args: SimpleArgs,
    model_cfg: SinCroModelConfig,
    global_step: int = 0,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """
    One forward pass matching the original MV_run_nerf.py training loop.
    Pools rays from B*(V-ref_V) images, samples N_rand total.
    """

    images = batch["images"]   # [B, T, H, V, W, C]
    K_mats = batch["K"]        # [B, V, 3, 3]
    c2w = batch["c2w"]         # [B, V, 4, 4]

    device = images.device
    B, T, H, V, W, C = images.shape
    N_rand = args.N_rand
    assert T == model_cfg.time_interval

    # ==================================================================
    # 1-2) Encode (primary + reference -> state encoder)
    # ==================================================================
    latent, anchor_latent, positive_latent, view_index, ref_view_indices = \
        encode_sincro(images, latent_embed, model_cfg,
                      mask_ratio=model_cfg.mask_ratio)

    # ==================================================================
    # 3) Negative encoding (torch.no_grad, like original)
    # ==================================================================
    negative_primary_imgs = torch.roll(images, shifts=1, dims=0)
    with torch.no_grad():
        _, neg_anchor, _, _, _ = encode_sincro(
            negative_primary_imgs, latent_embed, model_cfg,
            mask_ratio=model_cfg.mask_ratio,
            view_index=view_index, ref_view_indices=ref_view_indices,
        )
    negative_latent = neg_anchor  # [B, feat_dim]

    # ==================================================================
    # 4) Tile latent across all V views (original pattern)
    # ==================================================================
    latent_dim = render_kwargs_train["network_fn"].latent_dim
    latent_seq = latent.reshape(B, T, 1, -1)
    latent_seq = repeat(latent_seq, "b t v d -> b t (v mv) d", mv=V)
    latent_seq = latent_seq.permute(1, 0, 2, 3)   # [T, B, V, dim]
    latent_seq = latent_seq.reshape(T, B * V, -1)  # [T, BV, dim]
    assert latent_seq.shape[-1] == latent_dim

    # ==================================================================
    # 5) Build rays for all views
    # ==================================================================
    K_single = K_mats[0]   # [V, 3, 3]
    c2w_single = c2w[0]    # [V, 4, 4]

    rays_per_view = []
    for v in range(V):
        rays_o, rays_d = get_rays(H, W, K_single[v], c2w_single[v, :3, :4], device=device)
        rays_per_view.append(torch.stack([rays_o, rays_d], dim=0))
    rays_all = torch.stack(rays_per_view, dim=0)  # [V, 2, H, W, 3]

    tiled_rays = rays_all.unsqueeze(0).expand(B, -1, -1, -1, -1, -1)
    tiled_rays = tiled_rays.reshape(B * V, 2, H, W, 3)

    # ==================================================================
    # 6) NeRF rendering (last timestep only, remaining views)
    # ==================================================================
    remain_view_index = np.delete(np.arange(V), ref_view_indices)
    t = T - 1

    images_at_t = images[:, t].permute(0, 2, 1, 3, 4)  # [B, V, H, W, C]
    images_at_t = images_at_t.reshape(B * V, 1, H, W, C)

    rays_rgb = torch.cat([tiled_rays, images_at_t], dim=1)  # [B*V, 3, H, W, C]
    rays_rgb = rays_rgb.reshape(B, V, 3, H, W, C)[:, remain_view_index]
    rays_rgb = rays_rgb.reshape(-1, 3, H, W, C)

    # Precrop
    if global_step < args.precrop_iters:
        dH = int(H // 2 * args.precrop_frac)
        dW = int(W // 2 * args.precrop_frac)
        rays_rgb = rays_rgb[:, :, H // 2 - dH: H // 2 + dH, W // 2 - dW: W // 2 + dW]
        tile_H, tile_W = 2 * dH, 2 * dW
    else:
        tile_H, tile_W = H, W

    rays_rgb = rays_rgb.permute(0, 2, 3, 1, 4).reshape(-1, 3, 3).float()

    random_shuffle_indices = np.random.randint(rays_rgb.shape[0], size=N_rand)
    rays_rgb = rays_rgb[random_shuffle_indices]

    batch_data = rays_rgb.permute(1, 0, 2)
    batch_rays, target_s = batch_data[:2], batch_data[2]

    # Per-ray latent
    latent_at_t = latent_seq[t]  # [B*V, dim]
    latent_at_t = torch.tile(latent_at_t[:, None, :], (1, tile_H * tile_W, 1))
    latent_at_t = latent_at_t.reshape(B, V, tile_H * tile_W, latent_dim)[:, remain_view_index]
    latent_at_t = latent_at_t.reshape(-1, latent_dim)[random_shuffle_indices]

    # ==================================================================
    # 7) Core NeRF rendering
    # ==================================================================
    rgb, disp, depth, acc, extras = render(
        H, W, K_single[0], chunk=args.chunk,
        rays=batch_rays, verbose=False, retraw=True,
        latent=latent_at_t, args=args,
        **render_kwargs_train,
    )

    # ==================================================================
    # 8) Losses
    # ==================================================================
    img_loss = img2mse(rgb, target_s)
    psnr = mse2psnr(img_loss, device)

    img_loss0 = torch.tensor(0.0, device=device)
    psnr0 = torch.tensor(0.0, device=device)
    if "rgb0" in extras:
        img_loss0 = img2mse(extras["rgb0"], target_s)
        psnr0 = mse2psnr(img_loss0, device)

    d_positive = distance(anchor_latent, positive_latent)
    d_negative = distance(anchor_latent, negative_latent)
    contrastive_loss_raw = torch.clamp(
        args.enc_contrastive_margin + d_positive - d_negative, min=0.0
    ).mean()
    contrastive_loss = 0.0004 * contrastive_loss_raw

    loss = img_loss + img_loss0 + contrastive_loss

    stats = {
        "loss": loss.item(),
        "img_loss": img_loss.item(),
        "img_loss0": img_loss0.item(),
        "psnr": psnr.item(),
        "psnr0": psnr0.item(),
        "contrastive_loss": contrastive_loss_raw.item(),
    }
    return loss, stats


# -------------------------------------------------------------------------
# Render a full-resolution image for a single view
# (matches original: render(H, W, K, c2w=c2w[:3,:4], ..., test_mode=True))
# -------------------------------------------------------------------------

@torch.no_grad()
def render_full_image(
    H: int, W: int,
    K: torch.Tensor,           # [3, 3]
    c2w: torch.Tensor,         # [4, 4]
    latent: torch.Tensor,      # [1, dim] or [dim]
    args: SimpleArgs,
    render_kwargs: Dict[str, Any],
) -> np.ndarray:
    """
    Render a full H x W image from a single viewpoint, returning uint8 numpy.
    Uses render_kwargs_test (perturb=0, raw_noise_std=0) for clean output.
    """
    if latent.dim() == 1:
        latent = latent.unsqueeze(0)

    rgb, disp, depth, acc, extras = render(
        H, W, K,
        chunk=args.chunk,
        c2w=c2w[:3, :4],
        latent=latent,
        args=args,
        test_mode=True,
        **render_kwargs,
    )
    # rgb: [H, W, 3]
    return to8b(rgb.cpu().numpy())


# -------------------------------------------------------------------------
# Validation: full-res rendering for all views, logged to wandb
# -------------------------------------------------------------------------

@torch.no_grad()
def run_validation(
    dataloader: DataLoader,
    latent_embed: MaskedViTEncoder,
    render_kwargs_test: Dict[str, Any],
    args: SimpleArgs,
    model_cfg: SinCroModelConfig,
    train_cfg: TrainConfig,
    global_step: int,
):
    """
    Evaluation matching the original i_testset block:
    - Pick one batch from val set (use first sample, B=1).
    - Encode with mask_ratio=0 (no masking, like original test-time).
    - Render full H x W for ALL viewpoints.
    - Compute per-view PSNR.
    - Log rendered vs GT images to wandb.
    """
    latent_embed.eval()
    device = torch.device(train_cfg.device)

    # Get one batch
    try:
        batch = next(iter(dataloader))
    except StopIteration:
        return
    batch = {k: v.to(device) for k, v in batch.items()}

    images = batch["images"]   # [B, T, H, V, W, C]
    K_mats = batch["K"]        # [B, V, 3, 3]
    c2w_all = batch["c2w"]     # [B, V, 4, 4]

    B, T, H, V, W, C = images.shape

    # Use first sample only (B=1 like original test)
    images_0 = images[0:1]     # [1, T, H, V, W, C]
    K_0 = K_mats[0]            # [V, 3, 3]
    c2w_0 = c2w_all[0]         # [V, 4, 4]

    # ------------------------------------------------------------------
    # Encode with mask_ratio=0 (like original: m = 0.0 during test)
    # ------------------------------------------------------------------
    latent, _, _, primary_idx, ref_indices = encode_sincro(
        images_0, latent_embed, model_cfg, mask_ratio=0.0,
    )

    # Reshape latent: [1, T, 1, dim] -> take last timestep -> [1, dim]
    test_latent = latent.reshape(1, T, 1, -1)[:, -1, 0]  # [1, dim]

    # ------------------------------------------------------------------
    # Render full-res image for every viewpoint
    # (matches original: for v in test_recon_view_index)
    # ------------------------------------------------------------------
    rendered_views = []   # list of [H, W, 3] uint8
    gt_views = []         # list of [H, W, 3] uint8
    per_view_psnr = []

    for v in range(V):
        # Render full image
        rendered_np = render_full_image(
            H, W, K_0[v], c2w_0[v],
            latent=test_latent,
            args=args,
            render_kwargs=render_kwargs_test,
        )  # [H, W, 3] uint8

        # GT: last timestep, view v
        gt_np = (images_0[0, -1, :, v, :, :].cpu().numpy() * 255).astype(np.uint8)

        rendered_views.append(rendered_np)
        gt_views.append(gt_np)

        # Per-view PSNR
        rendered_f = rendered_np.astype(np.float32) / 255.0
        gt_f = gt_np.astype(np.float32) / 255.0
        mse = np.mean((rendered_f - gt_f) ** 2)
        psnr_v = -10.0 * np.log10(mse + 1e-8)
        per_view_psnr.append(psnr_v)

    # ------------------------------------------------------------------
    # Build wandb log
    # ------------------------------------------------------------------
    log_dict = {
        "val/mean_psnr": float(np.mean(per_view_psnr)),
    }

    # Per-view PSNR scalars
    for v in range(V):
        log_dict[f"val/psnr_view{v}"] = per_view_psnr[v]

    # Comparison grid: stack [GT | Rendered] for each view vertically
    pair_images = []
    for v in range(V):
        pair = np.concatenate([gt_views[v], rendered_views[v]], axis=1)  # [H, 2W, 3]
        pair_images.append(pair)
    comparison_grid = np.concatenate(pair_images, axis=0)  # [V*H, 2W, 3]
    log_dict["val/gt_vs_rendered_all_views"] = wandb.Image(
        comparison_grid,
        caption=f"Left=GT, Right=Rendered | primary={primary_idx} | step={global_step}",
    )

    # Individual per-view images
    for v in range(V):
        log_dict[f"val/rendered_view{v}"] = wandb.Image(
            rendered_views[v],
            caption=f"View {v}, PSNR={per_view_psnr[v]:.2f}",
        )
        log_dict[f"val/gt_view{v}"] = wandb.Image(
            gt_views[v],
            caption=f"GT View {v}",
        )

    wandb.log(log_dict, step=global_step)

    mean_psnr = float(np.mean(per_view_psnr))
    print(
        f"[Eval @ step {global_step}] mean_psnr={mean_psnr:.2f}, "
        f"per_view={[f'{p:.2f}' for p in per_view_psnr]}"
    )


# -------------------------------------------------------------------------
# Checkpoint save / load
# -------------------------------------------------------------------------

def save_checkpoint(path, global_step, epoch, render_kwargs_train, latent_embed, optimizer):
    """Save ALL model parameters for full resume."""
    ckpt = {
        "global_step": global_step,
        "epoch": epoch,
        "network_fn_state_dict": render_kwargs_train["network_fn"].state_dict(),
        "network_fine_state_dict": render_kwargs_train["network_fine"].state_dict(),
        "latent_embed_state_dict": latent_embed.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
    }
    torch.save(ckpt, path)
    print(f"[Checkpoint] Saved: {path}")


def load_checkpoint(path, render_kwargs_train, latent_embed, optimizer):
    """Load ALL model parameters for resuming training."""
    ckpt = torch.load(path, map_location="cpu")
    render_kwargs_train["network_fn"].load_state_dict(ckpt["network_fn_state_dict"])
    render_kwargs_train["network_fine"].load_state_dict(ckpt["network_fine_state_dict"])
    latent_embed.load_state_dict(ckpt["latent_embed_state_dict"])
    optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    print(f"[Resume] Loaded checkpoint: {path}")
    print(f"         global_step={ckpt['global_step']}, epoch={ckpt['epoch']}")
    return ckpt["global_step"], ckpt["epoch"]


# -------------------------------------------------------------------------
# Main
# -------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="YAML config path.")
    args_cli = parser.parse_args()

    with open(args_cli.config, "r") as f:
        cfg_yaml = yaml.safe_load(f)

    ds_cfg = DatasetConfig(**cfg_yaml.get("dataset", {}))
    model_cfg = SinCroModelConfig(**cfg_yaml.get("model", {}))
    train_cfg = TrainConfig(**cfg_yaml.get("train", {}))
    wandb_cfg = WandbConfig(**cfg_yaml.get("wandb", {}))
    exp_cfg = ExperimentConfig(**cfg_yaml.get("experiment", {}))

    model_cfg.num_views = ds_cfg.num_views

    torch.manual_seed(train_cfg.seed)
    np.random.seed(train_cfg.seed)

    device = torch.device(train_cfg.device if torch.cuda.is_available() else "cpu")

    import baselines.SinCro.sincro.MV_run_nerf as sincro_nerf
    sincro_nerf.device = device

    # ------------------------------------------------------------------
    # Dataset
    # ------------------------------------------------------------------
    ds_config = RobosuiteSinCroDatasetConfig(
        hdf5_path=ds_cfg.hdf5_path,
        num_views=ds_cfg.num_views,
        sequence_length=ds_cfg.sequence_length,
        max_episodes=ds_cfg.max_episodes,
        max_frames_per_demo=ds_cfg.max_frames_per_demo,
        temporal_stride=ds_cfg.temporal_stride,
    )
    full_dataset = RobosuiteSinCroSequenceDataset(ds_config)

    num_train = int(len(full_dataset) * ds_cfg.train_ratio)
    num_val = len(full_dataset) - num_train
    train_dataset, val_dataset = random_split(full_dataset, [num_train, num_val])

    train_loader = DataLoader(
        train_dataset, batch_size=ds_cfg.batch_size, shuffle=True,
        num_workers=ds_cfg.num_workers, pin_memory=ds_cfg.pin_memory,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=ds_cfg.batch_size, shuffle=False,
        num_workers=ds_cfg.num_workers, pin_memory=ds_cfg.pin_memory,
        drop_last=False,
    )

    # ------------------------------------------------------------------
    # Model
    # ------------------------------------------------------------------
    simple_args = SimpleArgs(model_cfg, train_cfg, ds_cfg, exp_cfg)
    (
        render_kwargs_train,
        render_kwargs_test,
        start_step,
        grad_vars,
        optimizer,
        latent_embed,
    ) = create_nerf(simple_args, simple_args.basedir, simple_args.expname)

    latent_embed.to(device)

    # ------------------------------------------------------------------
    # Resume
    # ------------------------------------------------------------------
    global_step = 0
    start_epoch = 0
    os.makedirs(train_cfg.ckpt_dir, exist_ok=True)

    if train_cfg.resume_from_last:
        ckpt_files = [
            f for f in os.listdir(train_cfg.ckpt_dir)
            if f.endswith(".tar") and f.startswith("step_")
            and "encoder" not in f and "final" not in f
        ]
        if ckpt_files:
            ckpt_files.sort()
            resume_path = os.path.join(train_cfg.ckpt_dir, ckpt_files[-1])
            global_step, start_epoch = load_checkpoint(
                resume_path, render_kwargs_train, latent_embed, optimizer
            )

    # ------------------------------------------------------------------
    # wandb
    # ------------------------------------------------------------------
    run_name = wandb_cfg.run_name or exp_cfg.expname
    wandb.init(
        project=wandb_cfg.project, entity=wandb_cfg.entity, name=run_name,
        config={
            "dataset": ds_cfg.__dict__,
            "model": model_cfg.__dict__,
            "train": train_cfg.__dict__,
            "experiment": exp_cfg.__dict__,
        },
    )

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------
    epoch = start_epoch

    while True:
        for step_in_epoch, batch in enumerate(train_loader):
            if (
                train_cfg.max_global_steps is not None
                and global_step >= train_cfg.max_global_steps
            ):
                print(f"[Stop] Reached max_global_steps={train_cfg.max_global_steps}.")
                wandb.finish()
                return

            latent_embed.train()
            batch = {k: v.to(device) for k, v in batch.items()}

            # Forward
            loss, stats = forward_sincro_batch(
                batch, latent_embed, render_kwargs_train,
                simple_args, model_cfg, global_step=global_step,
            )

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            new_lr = update_learning_rate(optimizer, train_cfg, global_step)

            # ------ Logging (every i_print steps) ------
            if global_step % train_cfg.i_print == 0:
                log_dict = {
                    "train/loss": stats["loss"],
                    "train/img_loss": stats["img_loss"],
                    "train/psnr": stats["psnr"],
                    "train/contrastive_loss": stats["contrastive_loss"],
                    "train/lr": new_lr,
                    "train/epoch": epoch,
                    "train/global_step": global_step,
                }
                if stats["img_loss0"] > 0:
                    log_dict["train/img_loss0"] = stats["img_loss0"]
                    log_dict["train/psnr0"] = stats["psnr0"]

                wandb.log(log_dict, step=global_step)

                print(
                    f"[Epoch {epoch+1} | Step {step_in_epoch} | Global {global_step}] "
                    f"loss={stats['loss']:.4f}, img={stats['img_loss']:.4f}, "
                    f"contrast={stats['contrastive_loss']:.4f}, psnr={stats['psnr']:.2f}"
                )

            # ------ Evaluation with full-res rendering ------
            if (
                train_cfg.eval_every > 0
                and global_step > 0
                and global_step % train_cfg.eval_every == 0
            ):
                run_validation(
                    val_loader, latent_embed, render_kwargs_test,
                    simple_args, model_cfg, train_cfg, global_step,
                )
                latent_embed.train()  # switch back to training mode

            # ------ Checkpoint ------
            if (
                train_cfg.save_every > 0
                and global_step > 0
                and global_step % train_cfg.save_every == 0
            ):
                ckpt_path = os.path.join(
                    train_cfg.ckpt_dir, f"step_{global_step:08d}.tar"
                )
                save_checkpoint(
                    ckpt_path, global_step, epoch,
                    render_kwargs_train, latent_embed, optimizer,
                )
                # Also save encoder separately (like original)
                enc_path = os.path.join(
                    train_cfg.ckpt_dir, f"step_{global_step:08d}_encoder.tar"
                )
                torch.save(latent_embed.state_dict(), enc_path)

            global_step += 1

        epoch += 1
        if train_cfg.max_global_steps is None and epoch >= train_cfg.num_epochs:
            print(f"[Stop] Reached num_epochs={train_cfg.num_epochs}.")
            break

    # Final checkpoint
    final_path = os.path.join(train_cfg.ckpt_dir, f"step_{global_step:08d}_final.tar")
    save_checkpoint(
        final_path, global_step, epoch,
        render_kwargs_train, latent_embed, optimizer,
    )
    wandb.finish()
    print("Training finished.")


if __name__ == "__main__":
    main()