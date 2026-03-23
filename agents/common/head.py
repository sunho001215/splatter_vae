from __future__ import annotations

import torch
import torch.nn as nn

from .encoders import _weight_init


class TemporalGRUHead(nn.Module):
    """
    Small temporal head for trainable non-SinCro encoders.

    Input:
        frame_feats: (B, T, D_in)

    Output:
        fused: (B, D_out)

    Design:
      - shared per-frame backbone encodes each RGB frame
      - GRU models short temporal dynamics across the stacked frames
      - final MLP projects the last hidden state
    """

    def __init__(
        self,
        in_dim: int,
        hidden_dim: int = 256,
        out_dim: int = 256,
        num_layers: int = 1,
    ):
        super().__init__()
        self.out_dim = int(out_dim)

        self.gru = nn.GRU(
            input_size=in_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
        )

        self.proj = nn.Sequential(
            nn.Linear(hidden_dim, out_dim),
            nn.LayerNorm(out_dim),
            nn.Tanh(),
        )

        # Init GRU weights explicitly, since _weight_init only covers Linear/Conv
        for name, param in self.gru.named_parameters():
            if "weight_ih" in name:
                nn.init.xavier_uniform_(param.data)
            elif "weight_hh" in name:
                nn.init.orthogonal_(param.data)
            elif "bias" in name:
                param.data.zero_()

        self.proj.apply(_weight_init)

    def forward(self, frame_feats: torch.Tensor) -> torch.Tensor:
        # frame_feats: (B, T, D_in)
        _, h_n = self.gru(frame_feats)

        # Last GRU layer hidden state: (B, hidden_dim)
        h_last = h_n[-1]

        # Small projection for downstream RL heads
        return self.proj(h_last).contiguous()


class SmallPostEncoderMLPHead(nn.Module):
    """
    Tiny post-backbone MLP head.

    Use case:
      - backbone already outputs one fused vector (e.g. SinCro)
      - we still want a small learnable encoder-side adapter before RL heads
    """

    def __init__(
        self,
        in_dim: int,
        hidden_dim: int = 256,
        out_dim: int = 256,
        num_layers: int = 2,
        use_tanh: bool = True,
    ):
        super().__init__()

        if num_layers < 1:
            raise ValueError(f"num_layers must be >= 1, got {num_layers}")
        self.out_dim = int(out_dim)

        layers = []
        prev_dim = in_dim

        # Hidden layers
        for _ in range(num_layers - 1):
            layers += [
                nn.Linear(prev_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.ReLU(inplace=True),
            ]
            prev_dim = hidden_dim

        # Final projection
        layers += [
            nn.Linear(prev_dim, out_dim),
            nn.LayerNorm(out_dim),
        ]
        if use_tanh:
            layers.append(nn.Tanh())

        self.net = nn.Sequential(*layers)
        self.apply(_weight_init)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 2:
            x = x.flatten(1)
        return self.net(x).contiguous()