import math
import random
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
from torch.nn import functional as F

# -------------------------------------------------------------------
# Config with support for non-square images
# -------------------------------------------------------------------

@dataclass
class STTransConfig:
    block_size: int = 4*4
    vocab_size: int = 0
    n_tokens_per_frame: int = 4*4
    n_layer: int = 8
    n_head: int = 8
    n_embed: int = 256
    dropout: float = 0.1
    bias: bool = False 
    mask_rate: float = None

    def update(self, overrides: dict) -> "STTransConfig":
        """Return a new config with `overrides` applied."""
        d = self.__dict__.copy()
        d.update(overrides)
        return STTransConfig(**d)


# -------------------------------------------------------------------
# Basic LayerNorm (same spirit as nanoGPT / ReViWo)
# -------------------------------------------------------------------

class LayerNorm(nn.Module):
    """
    LayerNorm but with an optional bias.

    PyTorch's nn.LayerNorm doesn't allow turning the bias off directly, so
    this small wrapper replicates the functionality.
    """

    def __init__(self, ndim: int, bias: bool):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.layer_norm(x, self.weight.shape, self.weight, self.bias, 1e-5)


# -------------------------------------------------------------------
# Spatial self-attention over tokens in a frame
# -------------------------------------------------------------------

class SCausalSelfAttention(nn.Module):
    """
    Spatial self-attention over tokens.

    This operates over a flattened grid of image tokens
    (patch features) for each frame. There is no temporal dimension here;
    "causal" refers to how they originally adapted a GPT-like block.

    Args (from config):
        n_embed:            embedding dimension (feature dim of tokens)
        n_head:             number of attention heads
        n_tokens_per_frame: number of spatial tokens per frame (H_p * W_p)
        dropout:            dropout probability
        bias:               whether Linear / LayerNorm layers use bias
        mask_rate:          optional random masking rate over *keys* per frame
    """

    def __init__(self, config) -> None:
        super().__init__()
        assert config.n_embed % config.n_head == 0
        self.n_head = config.n_head
        self.n_embed = config.n_embed
        self.dropout = config.dropout
        self.n_tokens_per_frame = config.n_tokens_per_frame
        self.mask_rate = config.mask_rate

        # Joint QKV projection as in GPT/nanoGPT.
        self.c_attn = nn.Linear(config.n_embed, 3 * config.n_embed, bias=config.bias)
        # Output projection back to n_embed.
        self.c_proj = nn.Linear(config.n_embed, config.n_embed, bias=config.bias)

        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, T, C) where

        Returns:
            y: (B, T, C) same shape as input
        """
        B, T, C = x.size()
        assert T == self.n_tokens_per_frame, (
            f"Expected T == n_tokens_per_frame = {self.n_tokens_per_frame}, "
            f"but got T = {T}"
        )

        # 1) Compute Q, K, V for all heads in one go.
        #    c_attn projects from C -> 3*C, then we split on the last dim.
        q, k, v = self.c_attn(x).split(self.n_embed, dim=2)

        head_dim = C // self.n_head

        # Reshape to (B', n_head, T, head_dim)
        # Here B' is just B; in the original code, they often use
        # B = batch_size * num_frames.
        k = k.view(B, T, self.n_head, head_dim).transpose(1, 2)  # (B, n_head, T, head_dim)
        q = q.view(B, T, self.n_head, head_dim).transpose(1, 2)  # (B, n_head, T, head_dim)
        v = v.view(B, T, self.n_head, head_dim).transpose(1, 2)  # (B, n_head, T, head_dim)

        # 2) Attention computation using PyTorch 2.x scaled_dot_product_attention
        #    NOTE: this attention is *not* spatially causal (no future mask),
        #          so each token can see all other tokens in the same frame.
        if self.training:
            if self.mask_rate is None:
                # No custom masking; standard self-attention.
                y = torch.nn.functional.scaled_dot_product_attention(
                    q, k, v,
                    dropout_p=self.dropout,
                    is_causal=False,
                )
            else:
                # Random masking over *keys* for each forward pass:
                # choose a subset of token indices to "drop" in attention.
                num_mask = int(self.mask_rate * self.n_tokens_per_frame)
                if num_mask > 0:
                    mask_ids = random.sample(
                        list(range(self.n_tokens_per_frame)), k=num_mask
                    )
                    # attn_mask shape: (B, n_head, T, T), True = keep, False = mask out.
                    attn_mask = torch.ones(
                        (q.shape[0], q.shape[1], self.n_tokens_per_frame, self.n_tokens_per_frame),
                        device=q.device,
                        dtype=torch.bool,
                    )
                    attn_mask[:, :, :, mask_ids] = False
                else:
                    attn_mask = None

                y = torch.nn.functional.scaled_dot_product_attention(
                    q, k, v,
                    dropout_p=self.dropout,
                    is_causal=False,
                    attn_mask=attn_mask,
                )
        else:
            # No dropout at eval time.
            y = torch.nn.functional.scaled_dot_product_attention(
                q, k, v,
                dropout_p=0.0,
                is_causal=False,
            )

        # 3) Merge heads back: (B, n_head, T, head_dim) -> (B, T, C)
        y = y.transpose(1, 2).contiguous().view(B, T, C)

        # 4) Output projection + residual dropout.
        y = self.resid_dropout(self.c_proj(y))
        return y


# -------------------------------------------------------------------
# Feedforward MLP block (as in GPT / ReViWo)
# -------------------------------------------------------------------

class MLP(nn.Module):
    """
    Standard Transformer MLP: Linear → GELU → Linear → Dropout.
    """

    def __init__(self, config):
        super().__init__()
        self.c_fc   = nn.Linear(config.n_embed, 4 * config.n_embed, bias=config.bias)
        self.gelu   = nn.GELU()
        self.c_proj = nn.Linear(4 * config.n_embed, config.n_embed, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x


# -------------------------------------------------------------------
# One Transformer block: LN → attention → residual, then LN → MLP → residual
# -------------------------------------------------------------------

class Block(nn.Module):
    """
    Single Transformer block using:
      - LayerNorm + SCausalSelfAttention + residual
      - LayerNorm + MLP + residual
    """

    def __init__(self, config):
        super().__init__()
        self.ln_1  = LayerNorm(config.n_embed, bias=config.bias)
        self.s_attn = SCausalSelfAttention(config)
        self.ln_2  = LayerNorm(config.n_embed, bias=config.bias)
        self.mlp   = MLP(config)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.s_attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


# -------------------------------------------------------------------
# Full Spatial Transformer (no token embedding here; operates on features)
# -------------------------------------------------------------------

class STransformer(nn.Module):
    """
    Spatial Transformer over per-frame image tokens, as in ReViWo.

    This module:
      - adds 1D positional embeddings over the token sequence (flattened grid)
      - applies a stack of Transformer blocks (Block)
      - returns either:
          * per-token features, or
          * (optionally) logits over a codebook if vocab_size > 0

    Important:
      - It is agnostic to whether the image is square or not.
      - As long as your encoder outputs T = H_p * W_p tokens per frame
        and you set `n_tokens_per_frame = T` in the config, this works.
    """

    def __init__(self, config):
        super().__init__()
        assert config.block_size is not None
        assert config.n_tokens_per_frame is not None
        assert config.block_size >= config.n_tokens_per_frame

        self.config = config

        # Positional embeddings over the sequence length (block_size).
        self.transformer = nn.ModuleDict(
            dict(
                wpe = nn.Embedding(config.block_size, config.n_embed),
                drop = nn.Dropout(config.dropout),
                h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
                ln_f = LayerNorm(config.n_embed, bias=config.bias),
            )
        )

        # Optional "head": in ReViWo this is often a codebook classifier (vocab_size).
        if config.vocab_size is not None and config.vocab_size != 0:
            self.lm_head = nn.Linear(config.n_embed, config.vocab_size, bias=False)
        else:
            # Identity if you just want the features back.
            self.lm_head = nn.Identity()

        # Initialize all weights similar to GPT / ReViWo.
        self.apply(self._init_weights)

        # Slightly smaller init for residual projections (GPT-2 trick).
        for name, p in self.named_parameters():
            if name.endswith("c_proj.weight"):
                nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * config.n_layer))

    def get_num_params(self, non_embedding: bool = True) -> int:
        """
        Count model parameters. If non_embedding=True (default),
        positional embeddings are not counted.
        """
        n_params = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n_params -= self.transformer.wpe.weight.numel()
        return n_params

    def _init_weights(self, module: nn.Module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B_total, T, C)
               - T must be <= block_size and == n_tokens_per_frame.
               - For image tokens, T = H_p * W_p where
                 H_p = H / patch_size, W_p = W / patch_size.

        Returns:
            logits_or_features: (B_total, T, vocab_size) if vocab_size > 0,
                                else (B_total, T, C).
        """
        device = x.device
        B, T, C = x.size()
        assert T <= self.config.block_size, (
            f"Cannot forward sequence length {T}, "
            f"block_size is only {self.config.block_size}"
        )

        # 1) Positional indices [0, 1, ..., T-1]
        pos = torch.arange(0, T, dtype=torch.long, device=device)  # (T,)

        # 2) Positional embedding + dropout
        pos_emb = self.transformer.wpe(pos)  # (T, n_embed)
        x = x + pos_emb                      # broadcast over batch
        x = self.transformer.drop(x)

        # 3) Transformer blocks
        for block in self.transformer.h:
            x = block(x)

        # 4) Final LayerNorm
        x = self.transformer.ln_f(x)  # (B, T, C)

        # 5) Optional head: codebook logits or identity
        logits = self.lm_head(x)      # (B, T, vocab_size) or (B, T, C)
        return logits