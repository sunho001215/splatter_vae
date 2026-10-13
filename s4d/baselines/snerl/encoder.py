"""Frozen SNeRL encoder for DrM: one RL time step of 3 RGB views with their cameras -> the 63-d scene latent.

Multi-view RGB input (privileged): the policy sees three simultaneous cameras with known extrinsics, fixed for an
episode (``configs/rl/encoders/snerl.yaml`` requests ``env: {views: 3, cameras: true}``).
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn

from s4d.baselines.snerl.data import normalise_c2w
from s4d.baselines.snerl.model import encode_batch, load_export


class SNeRLFrameEncoder(nn.Module):
    """Per-frame backbone: ``rgb`` (N, V, 3, H, W) in [0, 1] and OpenCV ``c2w`` (N, V, 4, 4) -> (N, 63), the LayerNorm
    output that conditions the NeRF decoder in pretraining. The whole encoder is frozen (the official RL code trains
    its fc/mlp/LayerNorm and freezes only the convolutions; our protocol freezes every pretrained encoder)."""

    is_trainable = False

    def __init__(self, cfg: dict[str, Any]):
        super().__init__()
        payload, self.encoder = load_export(cfg["vision"]["export_path"])
        self.observation = {"views": int(payload["num_views"]), "depth": False, "cameras": True}
        self.lookat, self.scale = tuple(payload["pose"]["lookat"]), float(payload["pose"]["scale"])
        self.image_size = int(payload["image_size"])
        self.random_init = bool(payload["random_init"])

    def forward(self, obs: dict[str, torch.Tensor]) -> torch.Tensor:
        rgb, c2w = obs["rgb"].float(), obs["c2w"].float()
        if rgb.shape[-2:] != (self.image_size, self.image_size) or rgb.shape[1] != self.observation["views"]:
            views = self.observation["views"]
            raise ValueError(f"SNeRL expects {views} views of {self.image_size}^2 pixels, got {tuple(rgb.shape)}")
        poses = normalise_c2w(c2w, self.lookat, self.scale).flatten(-2)  # (N, V, 16), as in pretraining
        return encode_batch(self.encoder, rgb, poses)
