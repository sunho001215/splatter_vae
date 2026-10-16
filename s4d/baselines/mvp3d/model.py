"""3D-MVP masked multiview pretraining (Qi et al., "3D-MVP: 3D Multiview Pretraining for Manipulation", CVPR 2025,
arXiv 2406.18158; no official code): encoder E, lightweight MAE decoder D_MAE, per-view RGB masking and the
reconstruction loss (§3.2, Eq. 4-6).

E is RVT-1's MVT (``rvt_derived.MVTEncoder``, built from RVT's own layer classes). Everything here that is not RVT
is 3D-MVP and cites the paragraph it implements; choices the paper leaves open are marked "unspecified".
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
from einops import rearrange
from torch import nn

from s4d.baselines.mvp3d.rvt_derived import MVTEncoder, multiview_layers, rvt_layers

EXPORT_FORMAT = "splatter4d-baseline-mvp3d-v1"
RGB = slice(3, 6)  # RGB channels of the 10-channel virtual images
RGBD = slice(3, 7)  # reconstruction target: RGB + depth (§3.2 "reconstruct the multiview RGB-D images")

# Scene bounds [x_min, y_min, z_min, x_max, y_max, z_max] (RVT SCENE_BOUNDS order), metres, Meta-World world frame.
# Derived from the data (docs/baselines_mvp3d.md): objects/gripper over the 8 tasks span x [-0.48, 0.57],
# y [0.24, 1.06], z [0.00, 0.54]; the arm reaches z 0.75; the table top is x [-0.66, 0.66], y [0.20, 0.98], z ~0.
SCENE_BOUNDS = (-0.6, 0.15, -0.05, 0.6, 1.15, 0.75)


@dataclass
class MVP3DConfig:
    """Encoder, decoder and masking. Defaults = the size-matched configuration (scaled to at most SinCro's encoder
    size by user decision; see ``ORIGINAL`` for the paper's)."""

    # E: RVT-1 MVT (§3.2 "Implementation details": 8 layers, 8 heads, hidden 1024, patch 10; RVT-1: 220 px, 64 conv
    # channels, GEGLU x4 MLP, attention dropout 0.1, lrelu)
    img_size: int = 48
    img_patch_size: int = 12
    im_channels: int = 64
    attn_dim: int = 256
    attn_heads: int = 8
    attn_dim_head: int = 32
    depth: int = 8
    attn_dropout: float = 0.1
    activation: str = "lrelu"
    num_img: int = 5
    # D_MAE: "a multiview transformer with 2 layers and 8 attention heads" (v1: same hidden size as E)
    decoder_depth: int = 2
    decoder_dim: int = 256
    decoder_heads: int = 8
    decoder_dim_head: int = 32
    # Masking: 75 % of the tokens of each view (§3.2), RGB channels only (Table 2 / §4.4 "RGB" strategy)
    mask_ratio: float = 0.75
    mask_fill: float = 0.0  # unspecified; masked RGB is set to 0, the value of empty (no-point) pixels


ORIGINAL = MVP3DConfig(img_size=220, img_patch_size=10, attn_dim=1024, attn_dim_head=128, decoder_dim=1024,
                       decoder_dim_head=128)
RVT1 = MVP3DConfig(img_size=220, img_patch_size=11, attn_dim=512, attn_dim_head=64, decoder_dim=512, decoder_dim_head=64)


def build_encoder(cfg: MVP3DConfig | dict) -> MVTEncoder:
    cfg = cfg if isinstance(cfg, MVP3DConfig) else MVP3DConfig(**cfg)
    return MVTEncoder(
        img_size=cfg.img_size,
        img_patch_size=cfg.img_patch_size,
        im_channels=cfg.im_channels,
        attn_dim=cfg.attn_dim,
        attn_heads=cfg.attn_heads,
        attn_dim_head=cfg.attn_dim_head,
        depth=cfg.depth,
        attn_dropout=cfg.attn_dropout,
        activation=cfg.activation,
        num_img=cfg.num_img,
    )


class MAEDecoder(nn.Module):
    """D_MAE (§3.2, Eq. 5): z -> reconstructed RGB-D of all five views.

    "A separate, lightweight MAE decoder": a multiview transformer of RVT layers, 2 layers with 8 heads. Unspecified
    details, chosen as in MAE (He et al., 2022): a linear embedding of z, a learned positional embedding (RVT's
    randn-initialised form), a final LayerNorm and a linear per-patch pixel head; the layers keep E's within-view
    then cross-view split (1 + 1). No mask tokens: with RGB-only masking every token stays in E (its geometry is
    visible), so z has all 5N tokens."""

    def __init__(self, cfg: MVP3DConfig, out_channels: int = 4):
        super().__init__()
        if cfg.decoder_depth % 2:
            raise ValueError("the decoder keeps the within-view / cross-view split: use an even decoder_depth")
        self.num_img, self.patch = cfg.num_img, cfg.img_patch_size
        self.num_pat_img = cfg.img_size // cfg.img_patch_size
        self.out_channels = out_channels
        tokens = self.num_img * self.num_pat_img**2
        self.decoder_embed = nn.Linear(cfg.attn_dim, cfg.decoder_dim)
        self.decoder_pos_encoding = nn.Parameter(torch.randn(1, tokens, cfg.decoder_dim))
        self.layers = rvt_layers(cfg.decoder_dim, cfg.decoder_heads, cfg.decoder_dim_head, cfg.attn_dropout,
                                 cfg.decoder_depth)
        self.decoder_norm = nn.LayerNorm(cfg.decoder_dim)
        self.decoder_pred = nn.Linear(cfg.decoder_dim, self.patch * self.patch * out_channels)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        bs = z.shape[0]
        x = self.decoder_embed(z) + self.decoder_pos_encoding
        x = multiview_layers(self.layers, x, bs, self.num_img, self.num_pat_img**2)
        x = self.decoder_pred(self.decoder_norm(x))
        g, p = self.num_pat_img, self.patch
        return rearrange(x, "b (v gh gw) (ph pw c) -> b v c (gh ph) (gw pw)", v=self.num_img, gh=g, gw=g, ph=p, pw=p)


def random_view_mask(
    bs: int, num_img: int, num_pat_img: int, ratio: float, generator: torch.Generator | None = None, device=None
) -> torch.Tensor:
    """§3.2 "we randomly mask out a subset of the visual tokens for each view": an independent random subset per view
    and sample, of size L - int(L (1 - ratio)) for L tokens per view (MAE's ``random_masking`` count).
    -> bool (bs, num_img, num_pat_img, num_pat_img), True = masked."""
    L = num_pat_img * num_pat_img
    len_keep = int(L * (1 - ratio))
    noise = torch.rand(bs, num_img, L, generator=generator, device=device)
    rank = noise.argsort(dim=-1).argsort(dim=-1)
    return (rank >= len_keep).view(bs, num_img, num_pat_img, num_pat_img)


def mask_rgb(images: torch.Tensor, mask: torch.Tensor, patch: int, fill: float = 0.0) -> torch.Tensor:
    """I' (Eq. 4) with the "RGB" masking strategy (Table 2, §4.4): the RGB channels of the masked patches are set to
    ``fill``; xyz, depth and pixel-location channels are kept. images (bs, 5, 10, S, S), mask (bs, 5, g, g)."""
    pixels = mask.repeat_interleave(patch, dim=-2).repeat_interleave(patch, dim=-1)[:, :, None]  # (bs, 5, 1, S, S)
    out = images.clone()
    out[:, :, RGB] = torch.where(pixels, torch.full_like(images[:, :, RGB], fill), images[:, :, RGB])
    return out


def reconstruction_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Eq. 6: L = 1/(5WH) sum_i sum_p ||I_i(p) - I~_i(p)||_2^2 over all pixels of the five views (the squared norm runs
    over the reconstructed RGB-D channels), averaged over the batch. pred/target (bs, 5, C, H, W)."""
    return (pred - target).pow(2).sum(dim=2).mean()


class MVP3D(nn.Module):
    """E + D_MAE trained end to end with Eq. 6 on masked virtual images (§3.2 "Pretraining encoder E")."""

    def __init__(self, cfg: MVP3DConfig):
        super().__init__()
        self.cfg = cfg
        self.encoder = build_encoder(cfg)
        self.decoder = MAEDecoder(cfg)

    def forward(self, images: torch.Tensor, mask: torch.Tensor) -> dict[str, torch.Tensor]:
        masked = mask_rgb(images, mask, self.cfg.img_patch_size, self.cfg.mask_fill)
        z = self.encoder(masked)  # Eq. 4
        pred = self.decoder(z)  # Eq. 5
        target = images[:, :, RGBD]
        loss = reconstruction_loss(pred, target)
        with torch.no_grad():
            err = (pred - target).pow(2)
            parts = {"loss_rgb": err[:, :, :3].sum(2).mean(), "loss_depth": err[:, :, 3].mean()}
        return {"loss": loss, "pred": pred, "masked": masked, **parts}


def save_export(path: str | Path, encoder: MVTEncoder, model_cfg: dict, render_cfg: dict, step: int,
                random_init: bool = False) -> None:
    """Encoder-only export: weights, the config needed to rebuild it and to render its input. 3D-MVP pretrains on
    single time steps, so there is no pretraining frame spacing (``frame_spacing`` None)."""
    payload = {
        "format": EXPORT_FORMAT,
        "model_cfg": dict(model_cfg),
        "render_cfg": dict(render_cfg),
        "frame_spacing": None,
        "pretraining_frames": 1,
        "state_dict": encoder.state_dict(),
        "step": int(step),
        "random_init": bool(random_init),
    }
    tmp = Path(path).with_suffix(".tmp")
    torch.save(payload, tmp)
    tmp.replace(path)


def load_export(path: str | Path) -> tuple[dict, MVTEncoder]:
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if payload.get("format") != EXPORT_FORMAT:
        raise ValueError(f"{path} is not a 3D-MVP encoder export")
    encoder = build_encoder(payload["model_cfg"])
    encoder.load_state_dict(payload["state_dict"], strict=True)
    return payload, encoder.eval().requires_grad_(False)

