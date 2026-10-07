"""Policy-side vision encoders and projection heads, ported from the reference ``agents/common``.

``ConvNet`` is the end-to-end DrM/DrQ-v2 convolutional encoder (CNN baseline). Frozen pretrained encoders cache
features in replay and feed a small trainable projection head, exactly as in the reference.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
import torch.nn as nn


def weight_init(m: nn.Module) -> None:
    if isinstance(m, nn.Linear):
        nn.init.orthogonal_(m.weight.data)
        if m.bias is not None:
            m.bias.data.zero_()
    elif isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
        nn.init.orthogonal_(m.weight.data, nn.init.calculate_gain("relu"))
        if m.bias is not None:
            m.bias.data.zero_()


class ConvNet(nn.Module):
    """Official DrM ``Encoder`` (identical to DrQ-v2): stacked frames as channels, no internal augmentation."""

    is_trainable = True

    def __init__(self, cfg: dict[str, Any]):
        super().__init__()
        frame_stack = int(cfg["env"]["frame_stack"])
        h, w = int(cfg["env"]["image_height"]), int(cfg["env"]["image_width"])
        self.convnet = nn.Sequential(
            nn.Conv2d(3 * frame_stack, 32, 3, stride=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, 3, stride=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, 3, stride=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, 3, stride=1),
            nn.ReLU(inplace=True),
        )
        self.apply(weight_init)
        with torch.no_grad():
            self.repr_dim = int(self.convnet(torch.zeros(1, 3 * frame_stack, h, w)).flatten(1).shape[-1])

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.convnet(obs.float() / 255.0 - 0.5).flatten(1).contiguous()


class Splatter4DStateEncoder(nn.Module):
    """Frozen splatter4d encoder: one chronological RGB history -> flattened state slots."""

    is_trainable = False
    returns_sequence_state = True

    def __init__(self, cfg: dict[str, Any]):
        super().__init__()
        from s4d.model.encoder import load_encoder

        self.encoder = load_encoder(cfg["vision"]["export_path"])
        self.num_frames = int(self.encoder.cfg.num_frames)

    def forward(self, frames: torch.Tensor) -> torch.Tensor:
        """frames: (B,T,3,H,W) float in [0,1], oldest first. The T=1 ablation sees only the newest frame."""
        if self.num_frames == 1:
            frames = frames[:, -1:]
        return self.encoder.policy_state(frames)


class FrameMLPStackHead(nn.Module):
    """Temporal adapter for frozen per-frame encoders (reference ``FrameMLPStackHead``)."""

    def __init__(
        self,
        in_dim: int,
        frame_stack: int,
        per_frame_hidden_dim: int,
        per_frame_out_dim: int,
        stacked_hidden_dim: int,
        out_dim: int,
    ):
        super().__init__()
        self.frame_stack, self.per_frame_out_dim, self.out_dim = frame_stack, per_frame_out_dim, out_dim
        self.per_frame_mlp = nn.Sequential(
            nn.Linear(in_dim, per_frame_hidden_dim),
            nn.LayerNorm(per_frame_hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(per_frame_hidden_dim, per_frame_out_dim),
            nn.LayerNorm(per_frame_out_dim),
            nn.ReLU(inplace=True),
        )
        self.stack_mlp = nn.Sequential(
            nn.Linear(frame_stack * per_frame_out_dim, stacked_hidden_dim),
            nn.LayerNorm(stacked_hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(stacked_hidden_dim, out_dim),
            nn.LayerNorm(out_dim),
            nn.Tanh(),
        )
        self.apply(weight_init)

    def forward(self, frame_feats: torch.Tensor) -> torch.Tensor:
        b, t, d = frame_feats.shape
        if t != self.frame_stack:
            raise ValueError(f"expected T == frame_stack == {self.frame_stack}, got {t}")
        x = self.per_frame_mlp(frame_feats.reshape(b * t, d)).reshape(b, t * self.per_frame_out_dim)
        return self.stack_mlp(x).contiguous()


class SmallPostEncoderMLPHead(nn.Module):
    """Projection for frozen encoders that already return one fused vector (reference head)."""

    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int):
        super().__init__()
        self.out_dim = out_dim
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, out_dim),
            nn.LayerNorm(out_dim),
            nn.Tanh(),
        )
        self.apply(weight_init)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x.flatten(1)).contiguous()


BACKBONES = {"convnet": ConvNet, "splatter4d": Splatter4DStateEncoder}


class VisionEncoderAdapter(nn.Module):
    """Pixels or cached features -> policy representation (reference ``VisionEncoderAdapter``)."""

    def __init__(self, cfg: dict[str, Any]):
        super().__init__()
        self.vision_key = str(cfg["vision"]["encoder_type"])
        self.frame_stack = int(cfg["env"]["frame_stack"])
        h, w = int(cfg["env"]["image_height"]), int(cfg["env"]["image_width"])
        self.backbone = BACKBONES[self.vision_key](cfg)
        self.backbone_trainable = bool(self.backbone.is_trainable)
        self.stack_feature = bool(getattr(self.backbone, "returns_sequence_state", False))
        if not self.backbone_trainable:
            self.backbone.eval().requires_grad_(False)
        self.replay_atom_is_feature = not self.backbone_trainable
        self.replay_atom_is_stack_feature = self.replay_atom_is_feature and self.stack_feature
        self.replay_atom_frame_stack = 1 if self.replay_atom_is_stack_feature else self.frame_stack
        self.replay_atom_dtype = np.dtype(np.float16 if self.replay_atom_is_feature else np.uint8)
        self.proj_head = None
        if self.backbone_trainable:
            self.replay_atom_shape = (3, h, w)
            self.repr_dim = int(self.backbone.repr_dim)
            return
        proj = int(cfg["agent"]["feature_dim"])
        out_shape = self._feature_shape(h, w)
        if self.stack_feature:
            self.replay_atom_shape = out_shape
            self.proj_head = SmallPostEncoderMLPHead(int(np.prod(out_shape)), proj, proj)
        else:
            self.replay_atom_shape = out_shape[1:]
            self.proj_head = FrameMLPStackHead(out_shape[-1], self.frame_stack, proj, min(proj, 256), proj, proj)
        self.repr_dim = int(self.proj_head.out_dim)

    @torch.no_grad()
    def _feature_shape(self, h: int, w: int) -> tuple[int, ...]:
        device = next(self.backbone.parameters()).device
        if self.stack_feature:
            return tuple(self.backbone(torch.zeros(1, self.frame_stack, 3, h, w, device=device)).shape[1:])
        dim = int(self.backbone(torch.zeros(1, 3, h, w, device=device)).flatten(1).shape[-1])
        return (self.frame_stack, dim)

    def set_update_mode(self) -> None:
        super().train(True)
        if not self.backbone_trainable:
            self.backbone.eval()

    def set_act_mode(self) -> None:
        super().train(False)

    @torch.no_grad()
    def extract_cacheable_feature(self, obs: torch.Tensor) -> torch.Tensor:
        """(B, 3*T, H, W) uint8 channel-stacked history -> fp16 cached feature."""
        if self.backbone_trainable:
            raise RuntimeError("extract_cacheable_feature is only for frozen encoders.")
        b, c, h, w = obs.shape
        if c != 3 * self.frame_stack:
            raise ValueError(f"expected {3 * self.frame_stack} stacked RGB channels, got {c}")
        frames = (obs.float() / 255.0).reshape(b, self.frame_stack, 3, h, w)
        if self.stack_feature:
            return self.backbone(frames).to(torch.float16).contiguous()
        feat = self.backbone(frames.reshape(b * self.frame_stack, 3, h, w)).flatten(1)
        return feat.view(b, self.frame_stack, -1).to(torch.float16).contiguous()

    def forward(self, obs: torch.Tensor, *, is_feature: bool) -> torch.Tensor:
        if self.backbone_trainable:
            return self.backbone(obs)
        if not is_feature:
            obs = self.extract_cacheable_feature(obs)
        return self.proj_head(obs.float())
