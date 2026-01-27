# sincro_train_robosuite.py
"""
Training SinCro with your custom robosuite HDF5 dataset.

Key ideas:
  - We *do not* modify the original SinCro modules.
    We import them and wire them up here.
  - All hyperparameters and paths are controlled by a YAML file.
  - All losses and sample visualizations are logged to wandb.
"""

import os
import sys
import math
import argparse
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import yaml
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split

import wandb
from torchvision.utils import make_grid
from einops import rearrange

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
    sequence_length: int = 9
    max_episodes: Optional[int] = None
    max_frames_per_demo: Optional[int] = None
    temporal_stride: int = 3
    num_workers: int = 4
    pin_memory: bool = True
    batch_size: int = 4
    train_ratio: float = 0.9


@dataclass
class SinCroModelConfig:
    # These mirror the important args in MV_run_nerf.config_parser
    netdepth: int = 8
    netwidth: int = 256
    netdepth_fine: int = 8
    netwidth_fine: int = 256
    N_rand: int = 2048
    N_samples: int = 64
    N_importance: int = 64
    multires: int = 10
    multires_views: int = 4
    i_embed: int = 0
    use_viewdirs: bool = True
    raw_noise_std: float = 0.0
    white_bkgd: bool = True
    perturb: float = 1.0
    lindisp: bool = False
    # MAE / ViT encoder
    img_size: int = 224
    patch_size: int = 16
    embed_dim: int = 384
    vit_depth: int = 8
    vit_num_heads: int = 6
    vit_mlp_dim: int = 1024
    decoder_depth: int = 2
    decoder_num_heads: int = 2
    decoder_mlp_dim: int = 1024
    decoder_output_dim: int = 64
    # SinCro-specific
    time_interval: int = 3          # used to build anchor/positive/negative
    mask_ratio: float = 0.75        # MAE masking ratio
    num_views: int = 6              # must match dataset.num_views
    # NeRF radius / depth range
    near: float = 0.0
    far: float = 1.0
    chunk: int = 1024 * 32
    netchunk: int = 1024 * 64


@dataclass
class TrainConfig:
    num_epochs: int = 50
    max_global_steps: Optional[int] = None
    lrate: float = 5e-4
    lrate_decay: int = 250  # in 1000 steps, like SinCro original
    device: str = "cuda"
    # Loss weights
    rec_weight: float = 1.0
    contrastive_weight: float = 0.1
    # Logging / eval / ckpt
    eval_every: int = 1000
    save_every: int = 5000
    ckpt_dir: str = "./checkpoints_sincro"
    resume_from_last: bool = False
    seed: int = 42


@dataclass
class WandbConfig:
    project: str = "SinCro-Robosuite"
    entity: Optional[str] = None
    run_name: Optional[str] = None


@dataclass
class ExperimentConfig:
    basedir: str = "./logs_sincro"
    expname: str = "sincro_robosuite"


# -------------------------------------------------------------------------
# Helper: build Args namespace compatible with SinCro's create_nerf()
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
        # NeRF stuff
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

        # SinCro-specific bits needed by create_nerf
        self.time_interval = model_cfg.time_interval
        self.num_view = model_cfg.num_views
        self.batch_size = dataset_cfg.batch_size

        # Training hyperparams
        self.lrate = train_cfg.lrate

        # Misc flags used inside create_nerf (simplified)
        self.no_reload = True
        self.ft_path = None
        self.dataset_type = "robosuite"  # just a label; we don't use load_mae_data
        self.basedir = exp_cfg.basedir
        self.expname = exp_cfg.expname

        # Some other options that create_nerf expects; we set safe defaults
        self.N_rgb = 0
        self.no_ndc = True
        self.render_only = False
        self.render_test = False
        self.render_factor = 1
        self.precrop_iters = 0
        self.precrop_frac = 1.0
        self.N_iters = 100000
        self.i_embed_views = 0
        self.i_embed_state = -1
        self.chunk = model_cfg.chunk
        self.netchunk = model_cfg.netchunk
        self.lr_decay = train_cfg.lrate_decay
        self.use_mae = True
        self.mask_ratio = model_cfg.mask_ratio
        self.gamma = 1.0
        self.log_wandb = False  # we'll do wandb manually here

        # Extra placeholders that the original code sometimes references
        self.render_pose_path = None
        self.render_episode = None


# -------------------------------------------------------------------------
# Helper: schedule learning rate like original NeRF / SinCro
# -------------------------------------------------------------------------

def update_learning_rate(optimizer: torch.optim.Optimizer, train_cfg: TrainConfig, global_step: int):
    """Cosine-ish decay as in NeRF / SinCro."""
    if train_cfg.max_global_steps is None:
        return  # optional

    decay_rate = 0.1
    decay_steps = train_cfg.lrate_decay * 1000
    new_lrate = train_cfg.lrate * (decay_rate ** (global_step / decay_steps))

    for param_group in optimizer.param_groups:
        param_group["lr"] = new_lrate


# -------------------------------------------------------------------------
# Forward pass helper for a single batch
# -------------------------------------------------------------------------

def forward_sincro_batch(
    batch: Dict[str, torch.Tensor],
    latent_embed: MaskedViTEncoder,
    render_kwargs_train: Dict[str, Any],
    args: SimpleArgs,
    model_cfg: SinCroModelConfig,
    full_image: bool = False,
) -> Tuple[torch.Tensor, Dict[str, float], Dict[str, torch.Tensor]]:
    """
    One forward pass of SinCro-style training.

    This now follows the original SinCro behavior:
      - Concatenate views horizontally.
      - Randomly choose 1 primary view for the masked branch.
      - Choose V/3 reference views for the reference branch.
      - Run SinCro_image_encoder(..., is_ref=False) on primary views.
      - Run SinCro_image_encoder(..., is_ref=True) on reference views.
      - Fuse them in SinCro_state_encoder(...) before losses.

    If full_image is False:
      - Use N_rand random rays per image (memory-friendly, for training).

    If full_image is True:
      - Render the entire H×W image, but in ray chunks to keep memory bounded
        (for validation).
    """

    images = batch["images"]  # [B,T,H,V,W,3]
    K = batch["K"]            # [B,V,3,3]
    c2w = batch["c2w"]        # [B,V,4,4]

    device = images.device
    B, T, H, V, W, C = images.shape

    # In the original SinCro code, the temporal length equals time_interval.
    assert (
        T == model_cfg.time_interval
    ), f"Expected sequence length T={T} to match time_interval={model_cfg.time_interval}."

    # ------------------------------------------------------------------
    # 1) Prepare MAE / ViT inputs: primary vs reference views
    # ------------------------------------------------------------------
    batch_images_for_vit = images.reshape(B, T, H, V * W, C)
    per_view = torch.split(batch_images_for_vit, W, dim=3)

    num_ref_view = V // 3
    assert num_ref_view >= 1, "SinCro expects at least 3 views (1 primary + 2 reference)."

    view_indices = np.random.choice(V, size=1 + num_ref_view, replace=False)
    primary_view_idx = int(view_indices[0])
    ref_view_indices = view_indices[1:].tolist()

    primary_images = per_view[primary_view_idx]  # [B, T, H, W, 3]
    ref_images = torch.cat(
        [per_view[idx] for idx in ref_view_indices],
        dim=3,
    )  # [B, T, H, num_ref_view*W, 3]

    mask_ratio = model_cfg.mask_ratio

    # ------------------------------------------------------------------
    # 2) SinCro MAE encoders: image encoder + state encoder
    # ------------------------------------------------------------------
    latent, mask, ids_restore = latent_embed.SinCro_image_encoder(
        primary_images,
        mask_ratio=mask_ratio,
        T=T,
        is_ref=False,
    )

    ref_views = torch.split(ref_images, W, dim=3)   # list len=num_ref_view, each [B,T,H,W,3]
    ref_for_encoder = torch.cat(ref_views, dim=0)   # [num_ref_view * B, T, H, W, 3]

    ref_latent, _, _ = latent_embed.SinCro_image_encoder(
        ref_for_encoder,
        mask_ratio=0.0,
        T=T,
        is_ref=True,
    )

    ref_latent = rearrange(
        ref_latent[:, 1:, :],   # drop CLS
        "b (t hw) d -> b t hw d",
        t=T,
    )[:, -1]                    # [num_ref_view * B, H'*W', D]

    ref_latent = rearrange(
        ref_latent,
        "(v b) hw d -> b (v hw) d",
        b=B,
    )                           # [B, num_ref_view * H'*W', D]

    latent, mask, ids_restore = latent_embed.SinCro_state_encoder(
        latent,
        ref_latent,
        mask,
        ids_restore,
    )

    latent_seq = latent.view(B, T, -1)   # [B, T, D_eff]

    # ------------------------------------------------------------------
    # 3) Contrastive loss (anchor = first frame, positive = last frame)
    # ------------------------------------------------------------------
    def l2_normalize(x, dim=-1, eps=1e-6):
        return x / (x.norm(dim=dim, keepdim=True) + eps)

    anchor_flat = l2_normalize(latent_seq[:, 0], dim=-1)   # [B, D]
    pos_flat = l2_normalize(latent_seq[:, -1], dim=-1)     # [B, D]

    temperature = 0.1
    logits_pos = (anchor_flat * pos_flat).sum(dim=-1, keepdim=True)  # [B, 1]
    logits_neg = anchor_flat @ pos_flat.t()                          # [B, B]
    logits = torch.cat([logits_pos, logits_neg], dim=1) / temperature
    labels = torch.zeros(B, dtype=torch.long, device=device)
    contrastive_loss = F.cross_entropy(logits, labels)

    scene_latent = latent_seq[:, -1]  # [B, D]  # used to condition NeRF

    # ------------------------------------------------------------------
    # 4) Reconstruction via NeRF (random target view)
    # ------------------------------------------------------------------
    # Randomly choose a reconstruction view index for each sample
    recon_view_indices = torch.randint(
        low=0,
        high=V,
        size=(B,),
        device=device,
    )  # [B]

    # Build per-sample target RGB for the chosen view at time=0
    target_full = []
    for b in range(B):
        v_idx = recon_view_indices[b].item()
        target_b = images[b, 0, :, v_idx, :, :]      # [H, W, 3]
        target_full.append(target_b.reshape(H * W, 3))  # [HW, 3]
    target_rgb = torch.stack(target_full, dim=0)     # [B, HW, 3]

    if not full_image:
        # --------------------------------------------------------------
        # Training path: sample N_rand rays per image (memory-friendly)
        # --------------------------------------------------------------
        N_pixels = H * W
        N_rand = min(args.N_rand, N_pixels)

        rec_losses = []
        pred_full = []  # used only for visualization (sparse image)

        for b in range(B):
            v_idx = recon_view_indices[b].item()
            K_bv = K[b, v_idx]       # [3, 3]
            c2w_bv = c2w[b, v_idx]   # [4, 4]

            latent_b = scene_latent[b:b+1]   # [1, D]

            rays_o, rays_d = get_rays(
                H,
                W,
                K_bv,
                c2w_bv,
                device=device,
            )  # [H, W, 3] each

            rays_o = rays_o.reshape(-1, 3)   # [HW, 3]
            rays_d = rays_d.reshape(-1, 3)   # [HW, 3]

            # Randomly pick N_rand pixels
            select_inds = torch.randperm(N_pixels, device=device)[:N_rand]

            rays_o_sel = rays_o[select_inds]     # [N_rand, 3]
            rays_d_sel = rays_d[select_inds]     # [N_rand, 3]
            rays = torch.stack([rays_o_sel, rays_d_sel], dim=0)  # [2, N_rand, 3]

            target_full_b = target_rgb[b]        # [HW, 3]
            target_sel = target_full_b[select_inds]  # [N_rand, 3]

            rgb_b, disp_b, depth_b, acc_b, extras_b = render(
                H=H,
                W=W,
                K=K_bv,
                chunk=model_cfg.chunk,
                rays=rays,          # sample rays only
                near=model_cfg.near,
                far=model_cfg.far,
                latent=latent_b,
                args=args,
                test_mode=None,
                **render_kwargs_train,
            )  # rgb_b: [N_rand, 3]

            rec_b = img2mse(rgb_b, target_sel)
            rec_losses.append(rec_b)

            # "Sparse" full image for logging
            pred_full_b = torch.zeros_like(target_full_b)  # [HW, 3]
            pred_full_b[select_inds] = rgb_b
            pred_full.append(pred_full_b)

        rec_loss = torch.stack(rec_losses).mean()
        psnr = mse2psnr(rec_loss, device)
        rgb_map = torch.stack(pred_full, dim=0)  # [B, HW, 3]

    else:
        # --------------------------------------------------------------
        # Validation path: render full H×W image in ray chunks
        # --------------------------------------------------------------
        rec_losses = []
        rgb_full = []

        # Use a smaller chunk than training to reduce peak memory.
        val_chunk = max(1024, model_cfg.chunk // 4)

        for b in range(B):
            v_idx = recon_view_indices[b].item()
            K_bv = K[b, v_idx]
            c2w_bv = c2w[b, v_idx]
            latent_b = scene_latent[b:b+1]   # [1, D]

            # Let MV_run_nerf.render do the ray chunking internally.
            rgb_b, disp_b, depth_b, acc_b, extras_b = render(
                H=H,
                W=W,
                K=K_bv,
                chunk=val_chunk,      # smaller chunk => more chunks, less memory
                c2w=c2w_bv,
                near=model_cfg.near,
                far=model_cfg.far,
                latent=latent_b,
                args=args,
                test_mode=None,
                **render_kwargs_train,
            )  # rgb_b: [H, W, 3]

            rgb_b = rgb_b.view(H * W, 3)           # [HW, 3]
            rgb_full.append(rgb_b)

            target_full_b = target_rgb[b]          # [HW, 3]
            rec_b = img2mse(rgb_b, target_full_b)  # full-image reconstruction loss
            rec_losses.append(rec_b)

        rec_loss = torch.stack(rec_losses).mean()
        psnr = mse2psnr(rec_loss, device)
        rgb_map = torch.stack(rgb_full, dim=0)  # [B, HW, 3]

    # ------------------------------------------------------------------
    # 5) Package losses, stats, and visuals
    # ------------------------------------------------------------------
    losses = {
        "rec_loss": rec_loss,
        "contrastive_loss": contrastive_loss,
    }

    stats = {
        "psnr": psnr.item(),
        "rec_loss": rec_loss.item(),
        "contrastive_loss": contrastive_loss.item(),
    }

    rgb_vis = rgb_map.view(B, H, W, 3).permute(0, 3, 1, 2).contiguous()
    target_vis = target_rgb.view(B, H, W, 3).permute(0, 3, 1, 2).contiguous()

    visuals = {
        "pred_rgb": rgb_vis,
        "gt_rgb": target_vis,
    }

    total_loss = rec_loss + contrastive_loss
    return total_loss, stats, visuals


# -------------------------------------------------------------------------
# Validation: log a few images and scalar metrics
# -------------------------------------------------------------------------

@torch.no_grad()
def run_validation(
    dataloader: DataLoader,
    latent_embed: MaskedViTEncoder,
    render_kwargs_train: Dict[str, Any],
    args: SimpleArgs,
    model_cfg: SinCroModelConfig,
    train_cfg: TrainConfig,
    global_step: int,
    max_vis_batches: int = 1,
):
    latent_embed.eval()

    all_psnr = []
    all_rec_loss = []
    all_contrastive = []

    vis_pred = None
    vis_gt = None

    for b_idx, batch in enumerate(dataloader):
        if b_idx >= max_vis_batches:
            break

        batch = {k: v.to(train_cfg.device) for k, v in batch.items()}
        loss, stats, visuals = forward_sincro_batch(
            batch,
            latent_embed,
            render_kwargs_train,
            args,
            model_cfg,
            full_image=True
        )
        all_psnr.append(stats["psnr"])
        all_rec_loss.append(stats["rec_loss"])
        all_contrastive.append(stats["contrastive_loss"])

        if vis_pred is None:
            vis_pred = visuals["pred_rgb"]
            vis_gt = visuals["gt_rgb"]

    if len(all_psnr) == 0:
        return

    mean_psnr = float(np.mean(all_psnr))
    mean_rec = float(np.mean(all_rec_loss))
    mean_contrastive = float(np.mean(all_contrastive))

    # Build grids for wandb
    grid_pred = make_grid(
        vis_pred.cpu(),
        nrow=min(vis_pred.size(0), 4),
        normalize=True,
        value_range=(0.0, 1.0),
    )
    grid_gt = make_grid(
        vis_gt.cpu(),
        nrow=min(vis_gt.size(0), 4),
        normalize=True,
        value_range=(0.0, 1.0),
    )

    wandb.log(
        {
            "val/psnr": mean_psnr,
            "val/rec_loss": mean_rec,
            "val/contrastive_loss": mean_contrastive,
            "val/pred_rgb": wandb.Image(grid_pred),
            "val/gt_rgb": wandb.Image(grid_gt),
        },
        step=global_step,
    )


# -------------------------------------------------------------------------
# Main training loop
# -------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="YAML config path.")
    args_cli = parser.parse_args()

    with open(args_cli.config, "r") as f:
        cfg_yaml = yaml.safe_load(f)

    # Parse YAML sections into dataclasses
    ds_cfg = DatasetConfig(**cfg_yaml.get("dataset", {}))
    model_cfg = SinCroModelConfig(**cfg_yaml.get("model", {}))
    train_cfg = TrainConfig(**cfg_yaml.get("train", {}))
    wandb_cfg = WandbConfig(**cfg_yaml.get("wandb", {}))
    exp_cfg = ExperimentConfig(**cfg_yaml.get("experiment", {}))

    # Ensure consistency
    model_cfg.num_views = ds_cfg.num_views

    # Seed
    torch.manual_seed(train_cfg.seed)
    np.random.seed(train_cfg.seed)

    # Set device
    device = torch.device(train_cfg.device if torch.cuda.is_available() else "cpu")

    # Import SinCro's MV_run_nerf and set device
    import baselines.SinCro.sincro.MV_run_nerf  as sincro_nerf
    sincro_nerf.device = device

    # ------------------------------------------------------------------
    # Build dataset + dataloaders
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
    train_dataset, val_dataset = random_split(
        full_dataset, [num_train, num_val]
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=ds_cfg.batch_size,
        shuffle=True,
        num_workers=ds_cfg.num_workers,
        pin_memory=ds_cfg.pin_memory,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=ds_cfg.batch_size,
        shuffle=False,
        num_workers=ds_cfg.num_workers,
        pin_memory=ds_cfg.pin_memory,
        drop_last=False,
    )

    # ------------------------------------------------------------------
    # Build SinCro model (NeRF + MaskedViTEncoder) via create_nerf()
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
    # All model params already in optimizer from create_nerf

    # ------------------------------------------------------------------
    # wandb init
    # ------------------------------------------------------------------
    run_name = wandb_cfg.run_name or exp_cfg.expname
    wandb.init(
        project=wandb_cfg.project,
        entity=wandb_cfg.entity,
        name=run_name,
        config={
            "dataset": ds_cfg.__dict__,
            "model": model_cfg.__dict__,
            "train": train_cfg.__dict__,
            "experiment": exp_cfg.__dict__,
        },
    )

    os.makedirs(train_cfg.ckpt_dir, exist_ok=True)

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------
    global_step = start_step
    epoch = 0

    while True:
        for step_in_epoch, batch in enumerate(train_loader):
            if (
                train_cfg.max_global_steps is not None
                and global_step >= train_cfg.max_global_steps
            ):
                print(f"[Stop] Reached max_global_steps={train_cfg.max_global_steps}.")
                return

            latent_embed.train()

            # Move batch to device
            batch = {k: v.to(device) for k, v in batch.items()}

            # Forward
            total_loss, stats, visuals = forward_sincro_batch(
                batch,
                latent_embed,
                render_kwargs_train,
                simple_args,
                model_cfg,
            )

            optimizer.zero_grad(set_to_none=True)
            total_loss.backward()
            optimizer.step()

            # Learning rate schedule
            update_learning_rate(optimizer, train_cfg, global_step)

            # ------ wandb logging ------
            log_dict = {
                "train/total_loss": total_loss.item(),
                "train/rec_loss": stats["rec_loss"],
                "train/contrastive_loss": stats["contrastive_loss"],
                "train/psnr": stats["psnr"],
                "train/epoch": epoch,
                "train/step_in_epoch": step_in_epoch,
                "train/global_step": global_step,
            }

            # Optionally log a small grid of predicted vs GT images
            if global_step % 200 == 0:
                pred_grid = make_grid(
                    visuals["pred_rgb"].cpu(),
                    nrow=min(visuals["pred_rgb"].size(0), 4),
                    normalize=True,
                    value_range=(0.0, 1.0),
                )
                gt_grid = make_grid(
                    visuals["gt_rgb"].cpu(),
                    nrow=min(visuals["gt_rgb"].size(0), 4),
                    normalize=True,
                    value_range=(0.0, 1.0),
                )
                log_dict.update(
                    {
                        "train/pred_rgb": wandb.Image(pred_grid),
                        "train/gt_rgb": wandb.Image(gt_grid),
                    }
                )

            wandb.log(log_dict, step=global_step)

            # ------ periodic validation ------
            if (
                train_cfg.eval_every > 0
                and global_step > 0
                and global_step % train_cfg.eval_every == 0
            ):
                run_validation(
                    val_loader,
                    latent_embed,
                    render_kwargs_train,
                    simple_args,
                    model_cfg,
                    train_cfg,
                    global_step,
                )

            # ------ periodic checkpoint ------
            if (
                train_cfg.save_every > 0
                and global_step > 0
                and global_step % train_cfg.save_every == 0
            ):
                ckpt_path = os.path.join(
                    train_cfg.ckpt_dir,
                    f"step_{global_step:08d}.pth",
                )
                ckpt = {
                    "epoch": epoch,
                    "global_step": global_step,
                    "latent_embed_state_dict": latent_embed.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                }
                # You can also add NeRF weights (in grad_vars) here if needed.
                torch.save(ckpt, ckpt_path)
                print(f"[Checkpoint] Saved: {ckpt_path}")

            if global_step % 100 == 0:
                print(
                    f"[Epoch {epoch+1} | Step {step_in_epoch} | Global {global_step}] "
                    f"total={total_loss.item():.4f}, "
                    f"rec={stats['rec_loss']:.4f}, "
                    f"contrastive={stats['contrastive_loss']:.4f}, "
                    f"psnr={stats['psnr']:.2f}"
                )

            global_step += 1

        epoch += 1
        if train_cfg.max_global_steps is None and epoch >= train_cfg.num_epochs:
            print(f"[Stop] Reached num_epochs={train_cfg.num_epochs}.")
            break

    print("Training finished.")


if __name__ == "__main__":
    main()
