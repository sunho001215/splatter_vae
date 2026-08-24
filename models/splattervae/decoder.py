from __future__ import annotations

import math
from collections.abc import Sequence

import torch
import torch.nn.functional as F
from torch import nn

from models.gaussian.parameterization import NUM_GAUSSIANS

from .backbones import LayerScale, MultiHeadSelfAttention, RMSNorm, SwishGLU


class MultiHeadCrossAttention(nn.Module):
    def __init__(self, dimension: int, num_heads: int, qkv_bias: bool = True):
        super().__init__()
        if int(dimension) % int(num_heads):
            raise ValueError(
                "Cross-attention dimension must be divisible by num_heads."
            )
        self.dimension = int(dimension)
        self.num_heads = int(num_heads)
        self.head_dimension = self.dimension // self.num_heads
        self.query = nn.Linear(self.dimension, self.dimension, bias=qkv_bias)
        self.key_value = nn.Linear(self.dimension, 2 * self.dimension, bias=qkv_bias)
        self.output = nn.Linear(self.dimension, self.dimension)

    def forward(self, queries: torch.Tensor, memory: torch.Tensor) -> torch.Tensor:
        batch, query_count, _ = queries.shape
        memory_count = memory.shape[1]
        query = (
            self.query(queries)
            .view(batch, query_count, self.num_heads, self.head_dimension)
            .transpose(1, 2)
        )
        key, value = (
            self.key_value(memory)
            .view(batch, memory_count, 2, self.num_heads, self.head_dimension)
            .permute(2, 0, 3, 1, 4)
            .unbind(0)
        )
        attended = F.scaled_dot_product_attention(query, key, value, is_causal=False)
        attended = attended.transpose(1, 2).reshape(batch, query_count, self.dimension)
        return self.output(attended)


class GaussianSlotBlock(nn.Module):
    def __init__(
        self,
        dimension: int,
        num_heads: int,
        hidden_dimension: int,
        *,
        use_self_attention: bool,
        layerscale_initial_value: float = 1.0e-5,
    ):
        super().__init__()
        self.use_self_attention = bool(use_self_attention)
        if self.use_self_attention:
            self.self_norm = RMSNorm(dimension)
            self.self_attention = MultiHeadSelfAttention(dimension, num_heads)
            self.self_scale = LayerScale(dimension, layerscale_initial_value)
        self.cross_query_norm = RMSNorm(dimension)
        self.cross_memory_norm = RMSNorm(dimension)
        self.cross_attention = MultiHeadCrossAttention(dimension, num_heads)
        self.cross_scale = LayerScale(dimension, layerscale_initial_value)
        self.feedforward_norm = RMSNorm(dimension)
        self.feedforward = SwishGLU(dimension, hidden_dimension)
        self.feedforward_scale = LayerScale(dimension, layerscale_initial_value)

    def forward(self, slots: torch.Tensor, memory: torch.Tensor) -> torch.Tensor:
        if self.use_self_attention:
            slots = slots + self.self_scale(self.self_attention(self.self_norm(slots)))
        slots = slots + self.cross_scale(
            self.cross_attention(
                self.cross_query_norm(slots), self.cross_memory_norm(memory)
            )
        )
        return slots + self.feedforward_scale(
            self.feedforward(self.feedforward_norm(slots))
        )


class GaussianSlotDecoder(nn.Module):
    """Content-condition 256 Gaussian group identities by cross-attending to ViT tokens."""

    motion_parameters_per_gaussian = 6

    def __init__(
        self,
        *,
        encoder_dimension: int,
        gaussian_parameters_per_gaussian: int,
        num_groups: int = 256,
        gaussians_per_group: int = 8,
        dimension: int = 256,
        depth: int = 2,
        num_heads: int = 8,
        swiglu_hidden_dimension: int = 512,
        use_self_attention: bool = True,
        global_center: Sequence[float],
        anchor_initial_spread: float,
        parent_displacement_scale: float,
        child_radius: float,
    ):
        super().__init__()
        self.num_groups = int(num_groups)
        self.gaussians_per_group = int(gaussians_per_group)
        self.dimension = int(dimension)
        self.depth = int(depth)
        self.num_heads = int(num_heads)
        self.swiglu_hidden_dimension = int(swiglu_hidden_dimension)
        if self.num_groups != 256 or self.gaussians_per_group != 8:
            raise ValueError(
                "DROID grouped Gaussian decoding requires exactly 256 groups x 8 children."
            )
        if self.num_groups * self.gaussians_per_group != NUM_GAUSSIANS:
            raise ValueError(f"Decoder must produce exactly {NUM_GAUSSIANS} Gaussians.")
        if self.depth <= 0 or self.dimension % self.num_heads:
            raise ValueError(
                "Decoder depth must be positive and dimension divisible by heads."
            )
        if int(gaussian_parameters_per_gaussian) <= 3:
            raise ValueError(
                "Gaussian output width must include renderer attributes after XYZ."
            )
        center = torch.as_tensor(tuple(global_center), dtype=torch.float32)
        if center.shape != (3,) or not torch.isfinite(center).all():
            raise ValueError("global_center must contain three finite coordinates.")
        for name, value in (
            ("anchor_initial_spread", anchor_initial_spread),
            ("parent_displacement_scale", parent_displacement_scale),
            ("child_radius", child_radius),
        ):
            if not math.isfinite(float(value)) or float(value) <= 0.0:
                raise ValueError(f"{name} must be positive and finite.")
        self.global_center = tuple(float(value) for value in center)
        self.anchor_initial_spread = float(anchor_initial_spread)
        self.parent_displacement_scale = float(parent_displacement_scale)
        self.child_radius = float(child_radius)
        self.memory_projection = nn.Linear(int(encoder_dimension), self.dimension)
        self.slot_queries = nn.Parameter(torch.empty(self.num_groups, self.dimension))
        self.group_anchors = nn.Parameter(torch.empty(self.num_groups, 3))
        self.blocks = nn.ModuleList(
            GaussianSlotBlock(
                self.dimension,
                self.num_heads,
                self.swiglu_hidden_dimension,
                use_self_attention=use_self_attention,
            )
            for _ in range(self.depth)
        )
        self.output_norm = RMSNorm(self.dimension)
        self.parent_position_head = nn.Linear(self.dimension, 3)
        self.child_identities = nn.Parameter(
            torch.empty(self.gaussians_per_group, self.dimension)
        )
        self.child_expansion = nn.Sequential(
            nn.Linear(2 * self.dimension, self.dimension),
            nn.SiLU(),
            nn.Linear(self.dimension, self.dimension),
            nn.SiLU(),
        )
        self.child_position_head = nn.Linear(self.dimension, 3)
        self.attribute_head = nn.Linear(
            self.dimension, int(gaussian_parameters_per_gaussian) - 3
        )
        self.motion_head = nn.Linear(
            self.dimension, self.motion_parameters_per_gaussian
        )
        nn.init.trunc_normal_(self.slot_queries, std=0.02)
        nn.init.trunc_normal_(self.child_identities, std=0.02)
        with torch.no_grad():
            nn.init.normal_(self.group_anchors, std=self.anchor_initial_spread)
            self.group_anchors.add_(center)
        nn.init.zeros_(self.parent_position_head.weight)
        nn.init.zeros_(self.parent_position_head.bias)
        nn.init.normal_(self.child_position_head.weight, std=1.0e-3)
        nn.init.zeros_(self.child_position_head.bias)
        nn.init.zeros_(self.attribute_head.bias)
        nn.init.zeros_(self.motion_head.weight)
        nn.init.zeros_(self.motion_head.bias)

    def configuration(self) -> dict[str, object]:
        return {
            "num_groups": self.num_groups,
            "gaussians_per_group": self.gaussians_per_group,
            "dimension": self.dimension,
            "depth": self.depth,
            "num_heads": self.num_heads,
            "swiglu_hidden_dimension": self.swiglu_hidden_dimension,
            "global_center": list(self.global_center),
            "anchor_initial_spread": self.anchor_initial_spread,
            "parent_displacement_scale": self.parent_displacement_scale,
            "child_radius": self.child_radius,
            "conditioning": "multi_token_cross_attention",
        }

    def forward(self, encoder_tokens: torch.Tensor) -> dict[str, torch.Tensor]:
        if encoder_tokens.dim() != 3:
            raise ValueError(
                f"Gaussian decoder expects multiple encoder tokens (B,M,D), got {tuple(encoder_tokens.shape)}."
            )
        if encoder_tokens.shape[1] < 2:
            raise ValueError("Gaussian decoder refuses single-token conditioning.")
        memory = self.memory_projection(encoder_tokens)
        groups = (
            self.slot_queries.to(memory.dtype)
            .unsqueeze(0)
            .expand(memory.shape[0], -1, -1)
        )
        for block in self.blocks:
            groups = block(groups, memory)
        groups = self.output_norm(groups)
        parent_displacements = self.parent_displacement_scale * torch.tanh(
            self.parent_position_head(groups)
        )
        parent_centers = self.group_anchors[None] + parent_displacements
        group_features = groups[:, :, None].expand(-1, -1, self.gaussians_per_group, -1)
        child_identity = self.child_identities[None, None].expand(
            memory.shape[0], self.num_groups, -1, -1
        )
        children = self.child_expansion(
            torch.cat((group_features, child_identity), dim=-1)
        )
        child_offsets = self.child_radius * torch.tanh(
            self.child_position_head(children)
        )
        world_xyz = parent_centers[:, :, None] + child_offsets
        flat_children = children.flatten(1, 2)
        raw_parameters = torch.cat(
            (world_xyz.flatten(1, 2), self.attribute_head(flat_children)), dim=-1
        )
        return {
            "raw_gaussian_params": raw_parameters.contiguous(),
            "raw_motion_params": self.motion_head(flat_children).contiguous(),
            "group_tokens": groups.contiguous(),
            "parent_centers": parent_centers.contiguous(),
            "parent_displacements": parent_displacements.contiguous(),
            "child_offsets": child_offsets.contiguous(),
        }
