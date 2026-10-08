"""Content-blind grouped Gaussian set decoder: slots -> (scene group, dynamic group) Gaussians.

The decoder never sees pixels, depth, or cameras. Parent tokens are FiLM-conditioned
by the mean slot, refined by L blocks of (self-attention, cross-attention to the K
slots, MLP), then expanded into children around parent centres.

Screened options (review item 4, all off by default): ``state_concat`` concatenates the full K-slot state to every
parent token before the first block; ``conditioning="adaln_zero"`` replaces FiLM by DiT-style adaptive LayerNorm
(shift, scale and a zero-initialised gate per sub-layer, from the mean slot); ``anchor_fourier`` adds Fourier
features of each parent's anchor position to its token.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import torch
import torch.nn as nn
import torch.nn.functional as F

from s4d.model.gaussians import DYNAMIC_GROUP, SCENE_GROUP, GaussianSet


@dataclass
class GroupConfig:
    parents: int
    children: int
    offset_scale: float
    child_radius: float
    scale_min: float
    scale_max: float
    anchor_mean: tuple = (0.0, 0.6, 0.1)
    anchor_std: tuple = (0.2, 0.2, 0.1)


@dataclass
class DecoderConfig:
    slot_dim: int = 256
    dim: int = 128
    depth: int = 2
    heads: int = 4
    mlp_ratio: float = 4.0
    motion_max: float = 0.25
    scale_act_bias: float = -1.0
    scene: GroupConfig = field(default_factory=lambda: GroupConfig(768, 8, 0.15, 0.06, 0.001, 0.08))
    dynamic: GroupConfig = field(default_factory=lambda: GroupConfig(256, 8, 0.4, 0.04, 0.0005, 0.03))
    single_group: bool = False  # ablation (c): one static-budget group that also carries motion
    num_slots: int = 1  # K, needed to size the state concatenation
    state_concat: bool = False
    conditioning: str = "film"  # "film" | "adaln_zero"
    anchor_fourier: int = 0  # number of frequencies (0 = off)


class DecoderBlock(nn.Module):
    def __init__(self, dim: int, heads: int, mlp_ratio: float):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.self_attn = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        self.cross_attn = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.norm3 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(nn.Linear(dim, int(dim * mlp_ratio)), nn.GELU(), nn.Linear(int(dim * mlp_ratio), dim))

    def forward(self, parents: torch.Tensor, slots: torch.Tensor) -> torch.Tensor:
        h = self.norm1(parents)
        parents = parents + self.self_attn(h, h, h, need_weights=False)[0]
        parents = parents + self.cross_attn(self.norm2(parents), slots, slots, need_weights=False)[0]
        return parents + self.mlp(self.norm3(parents))


class AdaLNZeroBlock(nn.Module):
    """DecoderBlock with adaptive LayerNorm: each sub-layer input is modulated by (shift, scale) and its output scaled
    by a gate, all regressed from the conditioning vector by a zero-initialised layer (so every block starts as the
    identity)."""

    def __init__(self, dim: int, heads: int, mlp_ratio: float):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.self_attn = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.norm2 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.cross_attn = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.norm3 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.mlp = nn.Sequential(nn.Linear(dim, int(dim * mlp_ratio)), nn.GELU(), nn.Linear(int(dim * mlp_ratio), dim))
        self.modulation = nn.Sequential(nn.SiLU(), nn.Linear(dim, 9 * dim))
        nn.init.zeros_(self.modulation[1].weight)
        nn.init.zeros_(self.modulation[1].bias)

    def forward(self, parents: torch.Tensor, slots: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        s1, c1, g1, s2, c2, g2, s3, c3, g3 = self.modulation(cond)[:, None].chunk(9, dim=-1)
        h = self.norm1(parents) * (1.0 + c1) + s1
        parents = parents + g1 * self.self_attn(h, h, h, need_weights=False)[0]
        h = self.norm2(parents) * (1.0 + c2) + s2
        parents = parents + g2 * self.cross_attn(h, slots, slots, need_weights=False)[0]
        h = self.norm3(parents) * (1.0 + c3) + s3
        return parents + g3 * self.mlp(h)


def fourier_features(x: torch.Tensor, frequencies: int) -> torch.Tensor:
    """(..., 3) metres -> (..., 6 * frequencies): sin and cos of 2^k * pi * x, k = 0..frequencies-1."""
    scales = math.pi * 2.0 ** torch.arange(frequencies, device=x.device, dtype=x.dtype)
    angles = (x[..., None] * scales).flatten(-2)
    return torch.cat((angles.sin(), angles.cos()), -1)


class GroupHeads(nn.Module):
    """Per-group anchors, parent tokens, and output heads."""

    def __init__(self, g: GroupConfig, dim: int, dynamic: bool):
        super().__init__()
        self.g, self.dynamic = g, dynamic
        self.parent_tokens = nn.Parameter(torch.randn(g.parents, dim) * 0.02)
        anchors = torch.randn(g.parents, 3) * torch.tensor(g.anchor_std) + torch.tensor(g.anchor_mean)
        self.anchors = nn.Parameter(anchors)
        self.parent_pos = nn.Linear(dim, 3)
        self.child_ids = nn.Parameter(torch.randn(g.children, dim) * 0.02)
        self.child_mlp = nn.Sequential(nn.Linear(2 * dim, dim), nn.SiLU(), nn.Linear(dim, dim), nn.SiLU())
        self.xyz_head = nn.Linear(dim, 3)
        self.attr_head = nn.Linear(dim, 3 + 4 + 1 + 3)
        self.motion_head = nn.Linear(dim, 6) if dynamic else None
        nn.init.zeros_(self.parent_pos.weight)
        nn.init.zeros_(self.parent_pos.bias)
        nn.init.normal_(self.xyz_head.weight, std=1e-3)
        nn.init.zeros_(self.xyz_head.bias)
        nn.init.zeros_(self.attr_head.bias)
        if self.motion_head is not None:
            nn.init.zeros_(self.motion_head.weight)
            nn.init.zeros_(self.motion_head.bias)

    def forward(self, parents: torch.Tensor, motion_max: float, scale_act_bias: float) -> dict[str, torch.Tensor]:
        B, P, D = parents.shape
        g = self.g
        centers = self.anchors[None] + g.offset_scale * self.parent_pos(parents)  # (B,P,3)
        feats = self.child_mlp(
            torch.cat((parents[:, :, None].expand(B, P, g.children, D), self.child_ids[None, None].expand(B, P, -1, -1)), -1)
        )
        feats = feats.flatten(1, 2)  # (B, P*C, D)
        xyz = centers[:, :, None] + g.child_radius * torch.tanh(self.xyz_head(feats).view(B, P, g.children, 3))
        attr = self.attr_head(feats).float()
        scales = g.scale_min + torch.sigmoid(attr[..., 0:3] + scale_act_bias) * (g.scale_max - g.scale_min)
        quats = F.normalize(attr[..., 3:7] + attr.new_tensor([1.0, 0.0, 0.0, 0.0]), dim=-1)
        out = {
            "xyz": xyz.flatten(1, 2).float(),
            "scales": scales,
            "quats": quats,
            "opacity": torch.sigmoid(attr[..., 7]),
            "rgb": torch.sigmoid(attr[..., 8:11]),
            "parent_centers": centers.float(),
        }
        if self.motion_head is not None:
            motion = torch.tanh(self.motion_head(feats).float()) * motion_max
            out["delta01"], out["delta12"] = motion[..., :3], motion[..., 3:]
        else:
            out["delta01"] = out["delta12"] = torch.zeros_like(out["xyz"])
        return out


class GaussianDecoder(nn.Module):
    def __init__(self, cfg: DecoderConfig):
        super().__init__()
        self.cfg = cfg
        D = cfg.dim
        if cfg.conditioning == "film":
            self.film = nn.Sequential(nn.Linear(cfg.slot_dim, D), nn.SiLU(), nn.Linear(D, 2 * D))
            self.blocks = nn.ModuleList(DecoderBlock(D, cfg.heads, cfg.mlp_ratio) for _ in range(cfg.depth))
        elif cfg.conditioning == "adaln_zero":
            self.cond = nn.Sequential(nn.Linear(cfg.slot_dim, D), nn.SiLU(), nn.Linear(D, D))
            self.blocks = nn.ModuleList(AdaLNZeroBlock(D, cfg.heads, cfg.mlp_ratio) for _ in range(cfg.depth))
        else:
            raise ValueError(f"unknown decoder conditioning {cfg.conditioning!r}")
        self.slot_proj = nn.Linear(cfg.slot_dim, D)
        self.token_norm = nn.LayerNorm(D)
        if cfg.anchor_fourier > 0:
            self.anchor_proj = nn.Linear(6 * cfg.anchor_fourier, D)
            nn.init.normal_(self.anchor_proj.weight, std=0.02)
            nn.init.zeros_(self.anchor_proj.bias)
        if cfg.state_concat:  # Linear([token, state]) starting as the identity on the token
            self.state_cat = nn.Linear(D + cfg.num_slots * cfg.slot_dim, D)
            with torch.no_grad():
                self.state_cat.weight.zero_()
                self.state_cat.weight[:, :D].copy_(torch.eye(D))
                nn.init.normal_(self.state_cat.weight[:, D:], std=0.02)
                self.state_cat.bias.zero_()
        if cfg.single_group:
            total = cfg.scene.parents + cfg.dynamic.parents
            merged = GroupConfig(
                total,
                cfg.scene.children,
                cfg.dynamic.offset_scale,
                cfg.scene.child_radius,
                cfg.dynamic.scale_min,
                cfg.scene.scale_max,
                cfg.scene.anchor_mean,
                cfg.scene.anchor_std,
            )
            self.groups = nn.ModuleList([GroupHeads(merged, D, dynamic=True)])
            group_ids = [DYNAMIC_GROUP]
        else:
            self.groups = nn.ModuleList([GroupHeads(cfg.scene, D, dynamic=False), GroupHeads(cfg.dynamic, D, dynamic=True)])
            group_ids = [SCENE_GROUP, DYNAMIC_GROUP]
        self.register_buffer(
            "group",
            torch.cat(
                [torch.full((h.g.parents * h.g.children,), gid, dtype=torch.long) for h, gid in zip(self.groups, group_ids)]
            ),
            persistent=False,
        )

    @torch.no_grad()
    def set_anchor_statistics(self, stats: dict) -> None:
        """Re-initialise anchors from workspace statistics {"scene": {"mean","std"}, "dynamic": {...}}."""
        names = ["scene"] if self.cfg.single_group else ["scene", "dynamic"]
        for head, name in zip(self.groups, names):
            s = stats[name]
            like = {"dtype": torch.float32, "device": head.anchors.device}
            mean, std = torch.tensor(s["mean"], **like), torch.tensor(s["std"], **like)
            head.anchors.copy_(torch.randn_like(head.anchors) * std + mean)

    def forward(self, slots: torch.Tensor) -> GaussianSet:
        """slots (B,K,Ds) -> GaussianSet. No camera or image arguments by design."""
        if slots.dim() != 2 + 1:
            raise ValueError(f"slots must be (B,K,Ds), got {tuple(slots.shape)}")
        B = slots.shape[0]
        kv = self.slot_proj(slots)
        tokens = []
        for h in self.groups:
            token = self.token_norm(h.parent_tokens)
            if self.cfg.anchor_fourier > 0:
                token = token + self.anchor_proj(fourier_features(h.anchors, self.cfg.anchor_fourier))
            tokens.append(token[None].expand(B, -1, -1))
        parents = torch.cat(tokens, 1)
        if self.cfg.conditioning == "film":
            gamma, beta = self.film(slots.mean(1)).chunk(2, dim=-1)
            parents = parents * (1.0 + gamma[:, None]) + beta[:, None]
        if self.cfg.state_concat:
            state = slots.flatten(1)[:, None].expand(B, parents.shape[1], -1)
            parents = self.state_cat(torch.cat((parents, state.to(parents.dtype)), -1))
        if self.cfg.conditioning == "film":
            for block in self.blocks:
                parents = block(parents, kv)
        else:
            cond = self.cond(slots.mean(1))
            for block in self.blocks:
                parents = block(parents, kv, cond)
        outputs, start = [], 0
        for head in self.groups:
            outputs.append(head(parents[:, start : start + head.g.parents], self.cfg.motion_max, self.cfg.scale_act_bias))
            start += head.g.parents
        cat = {
            k: torch.cat([o[k] for o in outputs], 1)
            for k in ("xyz", "scales", "quats", "opacity", "rgb", "delta01", "delta12")
        }
        return GaussianSet(
            cat["xyz"], cat["scales"], cat["quats"], cat["opacity"], cat["rgb"], cat["delta01"], cat["delta12"], self.group
        )
