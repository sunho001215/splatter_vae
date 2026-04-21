from __future__ import annotations

import torch
import torch.nn as nn

from .encoders import _weight_init


class FrameMLPStackHead(nn.Module):
    """
    Shared temporal adapter for frozen non-SinCro encoders.
    """

    def __init__(
        self,
        in_dim: int,
        frame_stack: int,
        per_frame_hidden_dim: int = 256,
        per_frame_out_dim: int = 128,
        stacked_hidden_dim: int = 256,
        out_dim: int = 256,
        stacked_num_layers: int = 2,
        use_tanh: bool = True,
    ):
        super().__init__()
        if frame_stack < 1:
            raise ValueError(f"frame_stack must be >= 1, got {frame_stack}")
        if stacked_num_layers < 1:
            raise ValueError(f"stacked_num_layers must be >= 1, got {stacked_num_layers}")

        self.frame_stack = int(frame_stack)
        self.per_frame_out_dim = int(per_frame_out_dim)
        self.out_dim = int(out_dim)

        self.per_frame_mlp = nn.Sequential(
            nn.Linear(in_dim, per_frame_hidden_dim),
            nn.LayerNorm(per_frame_hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(per_frame_hidden_dim, per_frame_out_dim),
            nn.LayerNorm(per_frame_out_dim),
            nn.ReLU(inplace=True),
        )

        layers = []
        prev_dim = self.frame_stack * self.per_frame_out_dim
        for _ in range(stacked_num_layers - 1):
            layers += [
                nn.Linear(prev_dim, stacked_hidden_dim),
                nn.LayerNorm(stacked_hidden_dim),
                nn.ReLU(inplace=True),
            ]
            prev_dim = stacked_hidden_dim

        layers += [nn.Linear(prev_dim, out_dim), nn.LayerNorm(out_dim)]
        if use_tanh:
            layers.append(nn.Tanh())

        self.stack_mlp = nn.Sequential(*layers)
        self.apply(_weight_init)

    def forward(self, frame_feats: torch.Tensor) -> torch.Tensor:
        if frame_feats.ndim != 3:
            raise ValueError(f"Expected frame_feats as (B,T,D), got {tuple(frame_feats.shape)}")

        B, T, D = frame_feats.shape
        if T != self.frame_stack:
            raise ValueError(f"Expected T == frame_stack == {self.frame_stack}, got T={T}")

        x = frame_feats.reshape(B * T, D)
        x = self.per_frame_mlp(x)
        x = x.reshape(B, T * self.per_frame_out_dim)
        return self.stack_mlp(x).contiguous()


class SmallPostEncoderMLPHead(nn.Module):
    """Shared post-backbone adapter for fused-vector frozen encoders."""

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
        for _ in range(num_layers - 1):
            layers += [
                nn.Linear(prev_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.ReLU(inplace=True),
            ]
            prev_dim = hidden_dim

        layers += [nn.Linear(prev_dim, out_dim), nn.LayerNorm(out_dim)]
        if use_tanh:
            layers.append(nn.Tanh())

        self.net = nn.Sequential(*layers)
        self.apply(_weight_init)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 2:
            x = x.flatten(1)
        return self.net(x).contiguous()