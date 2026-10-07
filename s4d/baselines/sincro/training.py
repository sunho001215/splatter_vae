"""SinCro training step, encoding and validation rendering.

Copied verbatim from the reference repository (sunho001215/splatter_vae @ c0abf56, ``baselines/SinCro/train.py``),
whose trainer reproduces the original ``MV_run_nerf.py`` training logic; only imports differ. The splatter4d entry
point is ``scripts/train_sincro.py``.
"""

from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass

import numpy as np
import torch
from einops import rearrange, repeat

from s4d.baselines.sincro.mae_encoder import MaskedViTEncoder
from s4d.baselines.sincro.nerf import render
from s4d.baselines.sincro.nerf_helpers import get_rays, img2mse, mse2psnr

@dataclass
class DatasetConfig:
    hdf5_path: str = ""
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
    num_ref_views: int = 2
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
    val_random_sample: bool = True
    val_vis_nrow: int = 6


@dataclass
class ExperimentConfig:
    basedir: str = "./logs/SinCro/metaworld/"
    expname: str = "sincro_metaworld"


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
        self.num_ref_views = model_cfg.num_ref_views
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

    num_ref_view = int(model_cfg.num_ref_views)
    if num_ref_view != 2:
        raise ValueError("SinCro currently expects exactly two reference views.")
    if num_ref_view + 1 > V:
        raise ValueError(f"Need at least {num_ref_view + 1} views, got V={V}.")

    if view_index is None or ref_view_indices is None:
        indices = np.random.choice(V, size=1 + num_ref_view, replace=False)
        view_index = int(indices[0])
        ref_view_indices = indices[1:].tolist()
    else:
        ref_view_indices = [int(idx) for idx in ref_view_indices]
        if len(ref_view_indices) != num_ref_view:
            raise ValueError(
                f"Expected {num_ref_view} reference views, got {len(ref_view_indices)}."
            )

    if not (0 <= int(view_index) < V):
        raise ValueError(f"Primary view_index={view_index} out of range for V={V}.")
    if any(idx < 0 or idx >= V for idx in ref_view_indices):
        raise ValueError(f"Reference view indices {ref_view_indices} out of range for V={V}.")
    if int(view_index) in ref_view_indices:
        raise ValueError("Primary view must be distinct from reference views.")
    if len(set(ref_view_indices)) != len(ref_view_indices):
        raise ValueError(f"Reference view indices must be unique: {ref_view_indices}.")

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

