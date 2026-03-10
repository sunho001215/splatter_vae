from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# -----------------------------------------------------------------------------
# Small helpers
# -----------------------------------------------------------------------------


class DropPath(nn.Module):
    """Stochastic depth.

    This is a tiny local implementation so the code does not depend on timm.
    """

    def __init__(self, drop_prob: float = 0.0):
        super().__init__()
        self.drop_prob = float(drop_prob)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.drop_prob == 0.0 or not self.training:
            return x
        keep_prob = 1.0 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()
        return x.div(keep_prob) * random_tensor


class RMSNorm(nn.Module):
    """RMSNorm over the last dimension.

    RMSNorm is a drop-in alternative to LayerNorm that normalizes by the root
    mean square without subtracting the mean.
    """

    def __init__(self, dim: int, eps: float = 1e-8):
        super().__init__()
        self.eps = float(eps)
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        scale = torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return x * scale * self.weight


class LayerScale(nn.Module):
    """Per-channel residual scaling from CaiT / "Going Deeper with Image Transformers".

    The layer is initialized with a small value so the residual branch starts
    near zero and grows during optimization.
    """

    def __init__(self, dim: int, init_value: float = 1e-5):
        super().__init__()
        self.gamma = nn.Parameter(init_value * torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.gamma


class SwishGLU(nn.Module):
    """Feed-forward block using the SwiGLU / SwishGLU gating pattern.

    Input -> linear(2 * hidden) -> split -> swish(gate) * value -> linear(out)
    """

    def __init__(self, dim: int, hidden_dim: int, out_dim: int | None = None, dropout: float = 0.0):
        super().__init__()
        out_dim = dim if out_dim is None else out_dim
        self.fc1 = nn.Linear(dim, 2 * hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, out_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        value, gate = self.fc1(x).chunk(2, dim=-1)
        x = value * F.silu(gate)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x


class MultiHeadSelfAttention(nn.Module):
    """Standard ViT multi-head self attention over patch tokens."""

    def __init__(
        self,
        dim: int,
        num_heads: int,
        qkv_bias: bool = True,
        attn_dropout: float = 0.0,
        proj_dropout: float = 0.0,
    ):
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError(f"dim={dim} must be divisible by num_heads={num_heads}")
        self.dim = int(dim)
        self.num_heads = int(num_heads)
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(dim, 3 * dim, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_dropout)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, n, c = x.shape
        qkv = self.qkv(x).reshape(b, n, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, B, heads, N, head_dim)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        out = attn @ v
        out = out.transpose(1, 2).reshape(b, n, c)
        out = self.proj(out)
        out = self.proj_drop(out)
        return out


class TransformerBlock(nn.Module):
    """ViT block with RMSNorm, SwishGLU and LayerScale."""

    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        dropout: float = 0.0,
        attn_dropout: float = 0.0,
        drop_path: float = 0.0,
        layerscale_init: float = 1e-5,
    ):
        super().__init__()
        hidden_dim = int(dim * mlp_ratio)
        self.norm1 = RMSNorm(dim)
        self.attn = MultiHeadSelfAttention(
            dim=dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            attn_dropout=attn_dropout,
            proj_dropout=dropout,
        )
        self.ls1 = LayerScale(dim, init_value=layerscale_init)
        self.drop_path1 = DropPath(drop_path)

        self.norm2 = RMSNorm(dim)
        self.ffn = SwishGLU(dim=dim, hidden_dim=hidden_dim, out_dim=dim, dropout=dropout)
        self.ls2 = LayerScale(dim, init_value=layerscale_init)
        self.drop_path2 = DropPath(drop_path)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x))))
        x = x + self.drop_path2(self.ls2(self.ffn(self.norm2(x))))
        return x


# -----------------------------------------------------------------------------
# Patch embedding + ViT backbones
# -----------------------------------------------------------------------------


@dataclass
class ViTSmallConfig:
    img_height: int = 128
    img_width: int = 128
    patch_size: int = 16
    in_chans: int = 3
    embed_dim: int = 384
    depth: int = 12
    num_heads: int = 6
    mlp_ratio: float = 4.0
    qkv_bias: bool = True
    dropout: float = 0.0
    attn_dropout: float = 0.0
    drop_path_rate: float = 0.0
    layerscale_init: float = 1e-5
    selected_layers: Tuple[int, int, int, int] = (2, 5, 8, 11)


class PatchEmbed(nn.Module):
    """Image -> patch tokens using the original ViT stem style.

    The original ViT paper uses a linear projection of flattened patches.
    The equivalent and more efficient PyTorch implementation is a Conv2d with
    kernel_size == stride == patch_size.
    """

    def __init__(self, img_height: int, img_width: int, patch_size: int, in_chans: int, embed_dim: int):
        super().__init__()
        if img_height % patch_size != 0 or img_width % patch_size != 0:
            raise ValueError("Image size must be divisible by patch size.")
        self.img_height = int(img_height)
        self.img_width = int(img_width)
        self.patch_size = int(patch_size)
        self.grid_h = img_height // patch_size
        self.grid_w = img_width // patch_size
        self.num_patches = self.grid_h * self.grid_w
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Tuple[int, int]]:
        b, c, h, w = x.shape
        if h != self.img_height or w != self.img_width:
            raise ValueError(
                f"PatchEmbed expected {(self.img_height, self.img_width)} but got {(h, w)}"
            )
        x = self.proj(x)                # (B, D, Gh, Gw)
        gh, gw = x.shape[2], x.shape[3]
        x = x.flatten(2).transpose(1, 2).contiguous()  # (B, N, D)
        return x, (gh, gw)


class ViTBackbone(nn.Module):
    """Image ViT backbone that returns final tokens and chosen intermediate blocks."""

    def __init__(self, cfg: ViTSmallConfig):
        super().__init__()
        self.cfg = cfg
        self.patch_embed = PatchEmbed(
            img_height=cfg.img_height,
            img_width=cfg.img_width,
            patch_size=cfg.patch_size,
            in_chans=cfg.in_chans,
            embed_dim=cfg.embed_dim,
        )
        self.num_patches = self.patch_embed.num_patches
        self.grid_size = (self.patch_embed.grid_h, self.patch_embed.grid_w)
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches, cfg.embed_dim))
        self.pos_drop = nn.Dropout(cfg.dropout)

        dpr = torch.linspace(0, cfg.drop_path_rate, cfg.depth).tolist()
        self.blocks = nn.ModuleList(
            [
                TransformerBlock(
                    dim=cfg.embed_dim,
                    num_heads=cfg.num_heads,
                    mlp_ratio=cfg.mlp_ratio,
                    qkv_bias=cfg.qkv_bias,
                    dropout=cfg.dropout,
                    attn_dropout=cfg.attn_dropout,
                    drop_path=dpr[i],
                    layerscale_init=cfg.layerscale_init,
                )
                for i in range(cfg.depth)
            ]
        )
        self.norm = RMSNorm(cfg.embed_dim)
        self.selected_layers = tuple(cfg.selected_layers)

        nn.init.trunc_normal_(self.pos_embed, std=0.02)

    @property
    def embed_dim(self) -> int:
        return self.cfg.embed_dim

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, List[torch.Tensor], Tuple[int, int]]:
        tokens, grid_size = self.patch_embed(x)
        tokens = self.pos_drop(tokens + self.pos_embed)

        hidden_states: List[torch.Tensor] = []
        for idx, blk in enumerate(self.blocks):
            tokens = blk(tokens)
            if idx in self.selected_layers:
                hidden_states.append(tokens)

        tokens = self.norm(tokens)
        # Also normalize the DPT skip states to match decoder inputs.
        hidden_states = [self.norm(h) for h in hidden_states]
        return tokens, hidden_states, grid_size


class TokenTransformer(nn.Module):
    """Transformer that operates directly on a token grid instead of an image.

    This is used as the fused latent backbone before the DPT-style dense decoder.
    It keeps the grid resolution fixed, just like the original ViT / DPT backbone.
    """

    def __init__(self, num_tokens: int, embed_dim: int, depth: int, num_heads: int, mlp_ratio: float,
                 qkv_bias: bool = True, dropout: float = 0.0, attn_dropout: float = 0.0,
                 drop_path_rate: float = 0.0, layerscale_init: float = 1e-5,
                 selected_layers: Sequence[int] = (2, 5, 8, 11)):
        super().__init__()
        self.num_tokens = int(num_tokens)
        self.embed_dim = int(embed_dim)
        self.pos_embed = nn.Parameter(torch.zeros(1, num_tokens, embed_dim))
        self.pos_drop = nn.Dropout(dropout)
        dpr = torch.linspace(0, drop_path_rate, depth).tolist()
        self.blocks = nn.ModuleList(
            [
                TransformerBlock(
                    dim=embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    dropout=dropout,
                    attn_dropout=attn_dropout,
                    drop_path=dpr[i],
                    layerscale_init=layerscale_init,
                )
                for i in range(depth)
            ]
        )
        self.norm = RMSNorm(embed_dim)
        self.selected_layers = tuple(selected_layers)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

    def forward(self, tokens: torch.Tensor) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        if tokens.shape[1] != self.num_tokens:
            raise ValueError(f"TokenTransformer expected {self.num_tokens} tokens, got {tokens.shape[1]}")
        x = self.pos_drop(tokens + self.pos_embed)
        hidden_states: List[torch.Tensor] = []
        for idx, blk in enumerate(self.blocks):
            x = blk(x)
            if idx in self.selected_layers:
                hidden_states.append(x)
        x = self.norm(x)
        hidden_states = [self.norm(h) for h in hidden_states]
        return x, hidden_states


# -----------------------------------------------------------------------------
# DPT-style dense decoder
# -----------------------------------------------------------------------------


class ResidualConvUnit(nn.Module):
    """DPT / MiDaS style local refinement block."""

    def __init__(self, features: int):
        super().__init__()
        self.conv1 = nn.Conv2d(features, features, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(features, features, kernel_size=3, stride=1, padding=1)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.act(x)
        x = self.conv1(x)
        x = self.act(x)
        x = self.conv2(x)
        return x + residual


class FeatureFusionBlock(nn.Module):
    """Feature fusion block matching the spirit of the DPT."""

    def __init__(self, features: int):
        super().__init__()
        self.res_unit1 = ResidualConvUnit(features)
        self.res_unit2 = ResidualConvUnit(features)

    def forward(self, x: torch.Tensor, skip: torch.Tensor | None = None) -> torch.Tensor:
        if skip is not None:
            x = x + self.res_unit1(skip)
        x = self.res_unit2(x)
        x = F.interpolate(x, scale_factor=2.0, mode="bilinear", align_corners=True)
        return x


class TokenReassembleBlock(nn.Module):
    """Reassemble token sequences into image-like feature maps at four DPT scales.

    DPT takes fixed-resolution ViT tokens and converts them into a pyramid by
    reshaping the token grid and then using scale-specific projections / resampling.
    """

    def __init__(self, in_dim: int, out_dim: int, scale_factor: float):
        super().__init__()
        self.proj = nn.Conv2d(in_dim, out_dim, kernel_size=1)
        self.refine = nn.Conv2d(out_dim, out_dim, kernel_size=3, stride=1, padding=1)
        self.scale_factor = float(scale_factor)

        if scale_factor > 1.0:
            # For 4x and 2x maps we use transposed conv upsampling.
            kernel = int(2 * scale_factor)
            stride = int(scale_factor)
            padding = int(scale_factor // 2)
            self.resample = nn.ConvTranspose2d(out_dim, out_dim, kernel_size=kernel, stride=stride, padding=padding)
        elif scale_factor == 1.0:
            self.resample = nn.Identity()
        elif scale_factor == 0.5:
            self.resample = nn.Conv2d(out_dim, out_dim, kernel_size=3, stride=2, padding=1)
        else:
            raise ValueError("Supported DPT reassembly scales are {4, 2, 1, 0.5}.")

    def forward(self, tokens: torch.Tensor, grid_size: Tuple[int, int]) -> torch.Tensor:
        b, n, c = tokens.shape
        gh, gw = grid_size
        if n != gh * gw:
            raise ValueError(f"Expected {gh * gw} tokens from grid {grid_size}, got {n}")
        x = tokens.transpose(1, 2).reshape(b, c, gh, gw).contiguous()
        x = self.proj(x)
        x = self.resample(x)
        x = self.refine(x)
        return x


class DPTHead(nn.Module):
    """DPT-style dense prediction head.

    Inputs are the four hidden states selected from the fused latent transformer.
    They are reassembled into a spatial pyramid and progressively fused to produce
    a full-resolution splatter image.
    """

    def __init__(self, in_dim: int, features: int, out_channels: int):
        super().__init__()
        self.reassemble_1 = TokenReassembleBlock(in_dim, features, scale_factor=4.0)
        self.reassemble_2 = TokenReassembleBlock(in_dim, features, scale_factor=2.0)
        self.reassemble_3 = TokenReassembleBlock(in_dim, features, scale_factor=1.0)
        self.reassemble_4 = TokenReassembleBlock(in_dim, features, scale_factor=0.5)

        self.refinenet4 = FeatureFusionBlock(features)
        self.refinenet3 = FeatureFusionBlock(features)
        self.refinenet2 = FeatureFusionBlock(features)
        self.refinenet1 = FeatureFusionBlock(features)

        self.output_conv = nn.Sequential(
            nn.Conv2d(features, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, out_channels, kernel_size=1, stride=1, padding=0),
        )

    def forward(self, hidden_states: Sequence[torch.Tensor], grid_size: Tuple[int, int],
                output_size: Tuple[int, int]) -> torch.Tensor:
        if len(hidden_states) != 4:
            raise ValueError(f"DPTHead expects 4 hidden states, got {len(hidden_states)}")

        layer_1 = self.reassemble_1(hidden_states[0], grid_size)
        layer_2 = self.reassemble_2(hidden_states[1], grid_size)
        layer_3 = self.reassemble_3(hidden_states[2], grid_size)
        layer_4 = self.reassemble_4(hidden_states[3], grid_size)

        path_4 = self.refinenet4(layer_4)
        path_3 = self.refinenet3(path_4, layer_3)
        path_2 = self.refinenet2(path_3, layer_2)
        path_1 = self.refinenet1(path_2, layer_1)

        out = self.output_conv(path_1)
        out = F.interpolate(out, size=output_size, mode="bilinear", align_corners=True)
        return out
