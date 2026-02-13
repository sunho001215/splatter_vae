from __future__ import annotations
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


def _ffn(d_model: int, ff_mult: int, dropout: float) -> nn.Module:
    hidden = d_model * ff_mult
    return nn.Sequential(
        nn.Linear(d_model, hidden),
        nn.GELU(),
        nn.Dropout(dropout),
        nn.Linear(hidden, d_model),
        nn.Dropout(dropout),
    )


class EncoderBlock(nn.Module):
    """Bidirectional self-attention block (no causal mask)."""

    def __init__(self, d_model: int, n_heads: int, ff_mult: int, dropout: float):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.ln2 = nn.LayerNorm(d_model)
        self.ff = _ffn(d_model, ff_mult, dropout)

    def forward(self, x: torch.Tensor, key_padding_mask: torch.Tensor | None = None) -> torch.Tensor:
        # x: (B, N, D)
        h = self.ln1(x)
        attn_out, _ = self.attn(h, h, h, key_padding_mask=key_padding_mask, need_weights=False)
        x = x + attn_out
        x = x + self.ff(self.ln2(x))
        return x


class DecoderBlock(nn.Module):
    """
    Bidirectional self-attn over action tokens, plus cross-attn to encoded observations.

    This mirrors the paper description:
      - decoder takes action token sequence
      - cross-attends to observation latents
      - conditioned on timestep embedding
    """

    def __init__(self, d_model: int, n_heads: int, ff_mult: int, dropout: float):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.self_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)

        self.ln2 = nn.LayerNorm(d_model)
        self.cross_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)

        self.ln3 = nn.LayerNorm(d_model)
        self.ff = _ffn(d_model, ff_mult, dropout)

    def forward(
        self,
        x: torch.Tensor,                  # (B, T, D)
        memory: torch.Tensor,             # (B, N, D)
        memory_kpm: torch.Tensor | None = None,
    ) -> torch.Tensor:
        h = self.ln1(x)
        self_out, _ = self.self_attn(h, h, h, need_weights=False)
        x = x + self_out

        h = self.ln2(x)
        cross_out, _ = self.cross_attn(h, memory, memory, key_padding_mask=memory_kpm, need_weights=False)
        x = x + cross_out

        x = x + self.ff(self.ln3(x))
        return x
