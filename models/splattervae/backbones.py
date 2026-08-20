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
        input_dtype = x.dtype
        x_float = x.float()
        scale = torch.rsqrt(x_float.square().mean(dim=-1, keepdim=True) + self.eps)
        normalized = (x_float * scale).to(input_dtype)
        return normalized * self.weight.to(input_dtype)


class LayerScale(nn.Module):
    """Per-channel residual scaling from CaiT / "Going Deeper with Image Transformers".

    The layer is initialized with a small value so the residual branch starts
    near zero and grows during optimization.
    """

    def __init__(self, dim: int, init_value: float = 1e-5):
        super().__init__()
        self.gamma = nn.Parameter(init_value * torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.gamma.to(x.dtype)


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

        out = F.scaled_dot_product_attention(
            q,
            k,
            v,
            dropout_p=self.attn_drop.p if self.training else 0.0,
            is_causal=False,
            scale=self.scale,
        )
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
    """Image ViT backbone with a CLS token."""

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

        # ViT CLS token + positional embedding for (CLS + patches)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, cfg.embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, 1 + self.num_patches, cfg.embed_dim))
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

        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

    @property
    def embed_dim(self) -> int:
        return self.cfg.embed_dim

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, List[torch.Tensor], Tuple[int, int]]:
        patch_tokens, grid_size = self.patch_embed(x)  # (B, N, D)
        b = patch_tokens.shape[0]

        # Prepend CLS token
        cls_tokens = self.cls_token.to(patch_tokens.dtype).expand(b, -1, -1)
        tokens = torch.cat([cls_tokens, patch_tokens], dim=1)
        tokens = self.pos_drop(tokens + self.pos_embed.to(patch_tokens.dtype))

        hidden_states: List[torch.Tensor] = []
        for idx, blk in enumerate(self.blocks):
            tokens = blk(tokens)
            if idx in self.selected_layers:
                # Keep the CLS token for state readout
                hidden_states.append(tokens)

        # Final norm only for returned backbone output.
        tokens = self.norm(tokens)

        # Return patch tokens, hidden states, and grid size.
        return tokens[:, 1:, :], hidden_states, grid_size
