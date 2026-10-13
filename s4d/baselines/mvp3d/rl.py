"""Frozen 3D-MVP encoder as a DrM vision backbone (labels: "single RGB-D + extrinsics (privileged)",
"reimplemented").

Per frame: the current camera's RGB-D image with its K and c2w -> point cloud -> RVT-1's five virtual views
(``renderer.render_rgbd``, the pretraining renderer and scene bounds from the export) -> frozen E without masking ->
RVT's per-view spatial max-pool of E's last layer, concatenated over the views (5 x width). ``FrameMLPStackHead``
stacks it over the 3-frame history (``s4d/rl/encoders.py``). The paper fine-tunes E with RVT's action decoder; a
frozen-feature recipe is not in the paper, so the read-out is RVT's own global image feature, taken from z.
"""

from __future__ import annotations

from typing import Any

import torch
from torch import nn

from s4d.baselines.mvp3d.model import load_export
from s4d.baselines.mvp3d.renderer import render_rgbd


class MVP3DFrameEncoder(nn.Module):
    is_trainable = False
    observation = {"views": 1, "depth": True, "cameras": True}

    def __init__(self, cfg: dict[str, Any]):
        super().__init__()
        payload, self.encoder = load_export(cfg["vision"]["export_path"])
        self.model_cfg, self.render_cfg = dict(payload["model_cfg"]), dict(payload["render_cfg"])
        self.bounds = tuple(float(b) for b in self.render_cfg["bounds"])
        self.radius = float(self.render_cfg["radius"])
        self.img_size = int(self.model_cfg["img_size"])

    def virtual_images(self, obs: dict[str, torch.Tensor]) -> torch.Tensor:
        """obs: rgb (N, V, 3, H, W) in [0, 1], depth (N, V, H, W) metres, K (N, V, 3, 3), c2w (N, V, 4, 4) OpenCV."""
        return render_rgbd(obs["rgb"], obs["depth"], obs["K"], obs["c2w"], self.bounds, self.img_size, self.radius)

    def forward(self, obs: dict[str, torch.Tensor]) -> torch.Tensor:
        """-> (N, 5 * attn_dim) per-frame features."""
        z = self.encoder(self.virtual_images(obs))
        return self.encoder.global_feature(z).contiguous()
