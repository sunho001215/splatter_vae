"""Frame-blind ViT encoder with temporal embedding, tube masking, and K learnable state tokens."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
DINOV2_VITS14_URL = "https://dl.fbaipublicfiles.com/dinov2/dinov2_vits14/dinov2_vits14_pretrain.pth"


@dataclass
class EncoderConfig:
    image_height: int = 128
    image_width: int = 128
    patch_size: int = 16
    width: int = 128
    depth: int = 8
    heads: int = 8
    mlp_ratio: float = 4.0
    num_slots: int = 1
    slot_dim: int = 256
    num_frames: int = 3
    mask_ratio: float = 0.5
    motion_patch_threshold: float = 0.05
    drop_path: float = 0.1
    layerscale_init: float | None = None
    use_cls_token: bool = False
    backbone: str = "scratch"  # or "dinov2_vits14"
    normalization: str = "signed"  # "signed" -> [-1,1]; "imagenet" -> ImageNet mean/std
    pretrained_path: str | None = None

    @classmethod
    def dinov2_vits14(cls, **overrides) -> EncoderConfig:
        base = dict(
            image_height=140,
            image_width=252,
            patch_size=14,
            width=384,
            depth=12,
            heads=6,
            layerscale_init=1e-5,
            use_cls_token=True,
            backbone="dinov2_vits14",
            normalization="imagenet",
            drop_path=0.0,
        )
        base.update(overrides)
        return cls(**base)

    @property
    def grid(self) -> tuple[int, int]:
        return self.image_height // self.patch_size, self.image_width // self.patch_size

    @property
    def num_patches(self) -> int:
        return self.grid[0] * self.grid[1]


class Attention(nn.Module):
    def __init__(self, dim: int, heads: int):
        super().__init__()
        self.heads = heads
        self.qkv = nn.Linear(dim, 3 * dim)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, C = x.shape
        qkv = self.qkv(x).view(B, N, 3, self.heads, C // self.heads).permute(2, 0, 3, 1, 4)
        out = F.scaled_dot_product_attention(qkv[0], qkv[1], qkv[2])
        return self.proj(out.transpose(1, 2).reshape(B, N, C))


class Mlp(nn.Module):
    def __init__(self, dim: int, hidden: int):
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden)
        self.fc2 = nn.Linear(hidden, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(F.gelu(self.fc1(x)))


class LayerScale(nn.Module):
    def __init__(self, dim: int, init: float):
        super().__init__()
        self.gamma = nn.Parameter(torch.full((dim,), float(init)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.gamma


class Block(nn.Module):
    """Pre-norm ViT block; attribute names follow DINOv2 so its weights load directly."""

    def __init__(self, dim: int, heads: int, mlp_ratio: float, drop_path: float, layerscale_init: float | None):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, eps=1e-6)
        self.attn = Attention(dim, heads)
        self.ls1 = LayerScale(dim, layerscale_init) if layerscale_init else nn.Identity()
        self.norm2 = nn.LayerNorm(dim, eps=1e-6)
        self.mlp = Mlp(dim, int(dim * mlp_ratio))
        self.ls2 = LayerScale(dim, layerscale_init) if layerscale_init else nn.Identity()
        self.drop_path = float(drop_path)

    def _drop(self, x: torch.Tensor) -> torch.Tensor:
        if not self.training or self.drop_path == 0.0:
            return x
        keep = 1.0 - self.drop_path
        mask = x.new_empty(x.shape[0], 1, 1).bernoulli_(keep)
        return x * mask / keep

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self._drop(self.ls1(self.attn(self.norm1(x))))
        return x + self._drop(self.ls2(self.mlp(self.norm2(x))))


def sample_tube_mask(patch_scores: torch.Tensor, mask_ratio: float, threshold: float) -> torch.Tensor:
    """Visible-patch mask (True = visible) of shape (B, N) shared across time.

    Half of the visible patches are the highest-scoring moving patches (score > threshold),
    the rest are uniform among the remaining patches. Without motion everything is random.
    """
    B, N = patch_scores.shape
    keep = max(1, min(N, int(round(N * (1.0 - mask_ratio)))))
    rank = torch.argsort(torch.argsort(patch_scores, dim=1, descending=True), dim=1)
    motion_pick = (rank < keep // 2) & (patch_scores > threshold)
    priority = torch.where(motion_pick, torch.full_like(patch_scores, 2.0), torch.rand_like(patch_scores))
    ids = torch.topk(priority, keep, dim=1).indices
    visible = torch.zeros(B, N, dtype=torch.bool, device=patch_scores.device)
    visible.scatter_(1, ids, True)
    return visible


class Encoder(nn.Module):
    def __init__(self, cfg: EncoderConfig):
        super().__init__()
        self.cfg = cfg
        D = cfg.width
        self.patch_embed = nn.Conv2d(3, D, cfg.patch_size, cfg.patch_size)
        self.pos_embed = nn.Parameter(torch.zeros(1, cfg.num_patches, D))
        self.temporal_embed = nn.Parameter(torch.zeros(1, cfg.num_frames, D))
        self.state_tokens = nn.Parameter(torch.zeros(1, cfg.num_slots, D))
        self.cls_token = nn.Parameter(torch.zeros(1, 1, D)) if cfg.use_cls_token else None
        dpr = torch.linspace(0, cfg.drop_path, cfg.depth).tolist()
        self.blocks = nn.ModuleList(
            Block(D, cfg.heads, cfg.mlp_ratio, dpr[i], cfg.layerscale_init) for i in range(cfg.depth)
        )
        self.norm = nn.LayerNorm(D, eps=1e-6)
        self.slot_proj = nn.Linear(D, cfg.slot_dim)
        if cfg.normalization == "imagenet":
            mean, std = torch.tensor(IMAGENET_MEAN), torch.tensor(IMAGENET_STD)
        else:
            mean, std = torch.full((3,), 0.5), torch.full((3,), 0.5)
        self.register_buffer("mean", mean.view(1, 3, 1, 1), persistent=False)
        self.register_buffer("std", std.view(1, 3, 1, 1), persistent=False)
        for p in (self.pos_embed, self.temporal_embed, self.state_tokens):
            nn.init.trunc_normal_(p, std=0.02)
        if self.cls_token is not None:
            nn.init.trunc_normal_(self.cls_token, std=0.02)
        if cfg.backbone == "dinov2_vits14":
            load_dinov2_vits14(self, cfg.pretrained_path)

    @property
    def state_dim(self) -> int:
        return self.cfg.num_slots * self.cfg.slot_dim

    def patch_scores(self, motion_score: torch.Tensor) -> torch.Tensor:
        """(B,T,1,H,W) pixel scores -> (B,N) max over time and patch."""
        B, T = motion_score.shape[:2]
        pooled = F.max_pool2d(motion_score.flatten(0, 1).float(), self.cfg.patch_size)
        return pooled.view(B, T, -1).amax(1)

    def forward(
        self, images: torch.Tensor, motion_score: torch.Tensor | None = None, mask_ratio: float | None = None
    ) -> dict[str, torch.Tensor]:
        """images (B,T,3,H,W) float in [0,1]. Returns slots (B,K,Ds), patch_tokens, visible mask."""
        cfg = self.cfg
        B, T, C, H, W = images.shape
        if (H, W) != (cfg.image_height, cfg.image_width) or T != cfg.num_frames:
            raise ValueError(
                f"encoder expects (T,H,W)=({cfg.num_frames},{cfg.image_height},{cfg.image_width}), got {(T, H, W)}"
            )
        x = (images.flatten(0, 1).float() - self.mean) / self.std
        tokens = self.patch_embed(x).flatten(2).transpose(1, 2).view(B, T, cfg.num_patches, -1)
        tokens = tokens + self.pos_embed[:, None] + self.temporal_embed[:, :T, None]
        ratio = cfg.mask_ratio if mask_ratio is None else mask_ratio
        if ratio > 0.0 and self.training:
            scores = (
                self.patch_scores(motion_score)
                if motion_score is not None
                else torch.zeros(B, cfg.num_patches, device=images.device)
            )
            visible = sample_tube_mask(scores, ratio, cfg.motion_patch_threshold)
            n_vis = int(visible.sum(1)[0])
            ids = visible.nonzero()[:, 1].view(B, n_vis)
            tokens = torch.gather(tokens, 2, ids[:, None, :, None].expand(B, T, n_vis, tokens.shape[-1]))
        else:
            visible = torch.ones(B, cfg.num_patches, dtype=torch.bool, device=images.device)
        tokens = tokens.reshape(B, -1, tokens.shape[-1])
        prefix = [self.state_tokens.expand(B, -1, -1)]
        if self.cls_token is not None:
            prefix.insert(0, self.cls_token.expand(B, -1, -1))
        x = torch.cat(prefix + [tokens], dim=1)
        for block in self.blocks:
            x = block(x)
        x = self.norm(x)
        start = 1 if self.cls_token is not None else 0
        slots = self.slot_proj(x[:, start : start + cfg.num_slots])
        return {
            "slots": slots,
            "patch_tokens": x[:, start + cfg.num_slots :],
            "visible": visible,
            "cls": x[:, 0] if self.cls_token is not None else None,
        }

    @torch.no_grad()
    def policy_state(self, images: torch.Tensor) -> torch.Tensor:
        """(B,T,3,H,W) uint8 or float [0,1] from one camera -> (B, K*Ds) flattened slots, eval mode."""
        was_training = self.training
        self.eval()
        if self.cfg.num_frames == 1 and images.shape[1] == 3:
            images = images[:, :1]
        if images.dtype == torch.uint8:
            images = images.float() / 255.0
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=images.is_cuda):
            slots = self.forward(images, mask_ratio=0.0)["slots"]
        self.train(was_training)
        return slots.flatten(1).float()


def load_dinov2_vits14(encoder: Encoder, path: str | None = None) -> None:
    """Load DINOv2 ViT-S/14 weights (local path or download) with interpolated position embeddings."""
    if path:
        state = torch.load(path, map_location="cpu", weights_only=True)
    else:
        state = torch.hub.load_state_dict_from_url(DINOV2_VITS14_URL, map_location="cpu", weights_only=True)
    state = {k.replace("patch_embed.proj.", "patch_embed."): v for k, v in state.items() if not k.startswith("mask_token")}
    pos = state.pop("pos_embed")  # (1, 1 + 37*37, 384)
    cls_pos, patch_pos = pos[:, :1], pos[:, 1:]
    side = int(round(patch_pos.shape[1] ** 0.5))
    gh, gw = encoder.cfg.grid
    grid = patch_pos.view(1, side, side, -1).permute(0, 3, 1, 2)
    grid = F.interpolate(grid, size=(gh, gw), mode="bicubic", align_corners=False)
    state["pos_embed"] = grid.permute(0, 2, 3, 1).reshape(1, gh * gw, -1)
    state["cls_token"] = state["cls_token"] + cls_pos
    missing, unexpected = encoder.load_state_dict(state, strict=False)
    allowed_missing = {"temporal_embed", "state_tokens", "slot_proj.weight", "slot_proj.bias"}
    bad = [k for k in missing if k not in allowed_missing]
    if bad or unexpected:
        raise RuntimeError(f"DINOv2 load mismatch: missing={bad} unexpected={list(unexpected)}")


def load_encoder(path: str | Path) -> Encoder:
    """Rebuild only the frozen RGB encoder from an encoder-only export on CPU.

    Move the returned module and RGB history to the policy's device. No decoder,
    cameras, renderer, or pretrained download is needed at inference.
    """
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if payload.get("format") != "splatter4d-encoder-v1":
        raise ValueError("not a splatter4d encoder-only export")
    config = dict(payload["encoder_config"])
    config["backbone"] = "scratch"
    config["pretrained_path"] = None
    encoder = Encoder(EncoderConfig(**config))
    encoder.load_state_dict(payload["state_dict"], strict=True)
    return encoder.eval().requires_grad_(False)
