from __future__ import annotations

import torch
import torch.nn as nn

from .encoders import _weight_init


class FrameMLPStackHead(nn.Module):
    """
    Temporal adapter for frozen non-SinCro encoders.

    Structure:
      1) apply the SAME small MLP independently to each frame feature
      2) stack / flatten all processed frame features
      3) apply another MLP over the stacked vector

    Input:
        frame_feats: (B, T, D_in)

    Output:
        fused: (B, D_out)
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

        # --------------------------------------------------------------
        # 1) Shared per-frame MLP
        #    Applied to every frame feature independently.
        # --------------------------------------------------------------
        self.per_frame_mlp = nn.Sequential(
            nn.Linear(in_dim, per_frame_hidden_dim),
            nn.LayerNorm(per_frame_hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(per_frame_hidden_dim, per_frame_out_dim),
            nn.LayerNorm(per_frame_out_dim),
            nn.ReLU(inplace=True),
        )

        # --------------------------------------------------------------
        # 2) MLP over the stacked frame-wise features
        # --------------------------------------------------------------
        layers = []
        prev_dim = self.frame_stack * self.per_frame_out_dim

        for _ in range(stacked_num_layers - 1):
            layers += [
                nn.Linear(prev_dim, stacked_hidden_dim),
                nn.LayerNorm(stacked_hidden_dim),
                nn.ReLU(inplace=True),
            ]
            prev_dim = stacked_hidden_dim

        layers += [
            nn.Linear(prev_dim, out_dim),
            nn.LayerNorm(out_dim),
        ]
        if use_tanh:
            layers.append(nn.Tanh())

        self.stack_mlp = nn.Sequential(*layers)

        self.apply(_weight_init)

    def forward(self, frame_feats: torch.Tensor) -> torch.Tensor:
        if frame_feats.ndim != 3:
            raise ValueError(f"Expected frame_feats as (B,T,D), got {tuple(frame_feats.shape)}")

        B, T, D = frame_feats.shape
        if T != self.frame_stack:
            raise ValueError(
                f"Expected T == frame_stack == {self.frame_stack}, got T={T}"
            )

        # Apply the same MLP to each frame independently
        x = frame_feats.reshape(B * T, D)
        x = self.per_frame_mlp(x)  # (B*T, per_frame_out_dim)

        # Stack all processed frame features into one vector per sample
        x = x.reshape(B, T * self.per_frame_out_dim)

        # Final fusion / projection
        return self.stack_mlp(x).contiguous()


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