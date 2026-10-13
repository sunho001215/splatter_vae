"""Frozen DynaMo encoder for DrM: the official online ResNet-18 trunk, one 512-d feature per RGB frame.

The policy sees one RGB camera, as for every single-view method; per-frame features are stacked over the 3-frame
history by the shared ``FrameMLPStackHead`` (as for ReViWo). The encoder runs in eval mode, so BatchNorm uses the
running statistics accumulated in pretraining, as the official downstream code does
(``utils/inference.py::embed_trajectory_dataset`` under ``eval_mode``).
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn


class DynaMoFrameEncoder(nn.Module):
    """``vision.export_path``: an export written by ``scripts/train_dynamo.py`` (``encoder.pt``, or ``encoder_init.pt``
    for the same encoder at its random initialisation)."""

    is_trainable = False

    def __init__(self, cfg: dict[str, Any]):
        super().__init__()
        from s4d.baselines.dynamo.model import load_export

        payload, self.encoder = load_export(cfg["vision"]["export_path"])
        self.frame_spacing, self.random_init = int(payload["frame_spacing"]), bool(payload["random_init"])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (N, 3, H, W) float in [0, 1] -> (N, 512); ImageNet normalisation happens inside the official module."""
        return self.encoder(x).contiguous()
