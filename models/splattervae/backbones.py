from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn


class DropPath(nn.Module):
    def __init__(self, drop_probability: float = 0.0):
        super().__init__()
        self.drop_probability = float(drop_probability)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        if self.drop_probability == 0.0 or not self.training:
            return values
        keep_probability = 1.0 - self.drop_probability
        shape = (values.shape[0],) + (1,) * (values.ndim - 1)
        keep = (
            torch.rand(shape, device=values.device, dtype=values.dtype)
            < keep_probability
        )
        return values * keep / keep_probability


class RMSNorm(nn.Module):
    def __init__(self, dimension: int, epsilon: float = 1.0e-8):
        super().__init__()
        self.epsilon = float(epsilon)
        self.weight = nn.Parameter(torch.ones(int(dimension)))

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        dtype = values.dtype
        normalized = values.float() * torch.rsqrt(
            values.float().square().mean(dim=-1, keepdim=True) + self.epsilon
        )
        return normalized.to(dtype) * self.weight.to(dtype)


class LayerScale(nn.Module):
    def __init__(self, dimension: int, initial_value: float = 1.0e-5):
        super().__init__()
        self.gamma = nn.Parameter(torch.full((int(dimension),), float(initial_value)))

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return values * self.gamma.to(values.dtype)


class SwishGLU(nn.Module):
    def __init__(
        self,
        dimension: int,
        hidden_dimension: int,
        output_dimension: int | None = None,
        dropout: float = 0.0,
    ):
        super().__init__()
        output_dimension = int(
            dimension if output_dimension is None else output_dimension
        )
        self.input_projection = nn.Linear(int(dimension), 2 * int(hidden_dimension))
        self.output_projection = nn.Linear(int(hidden_dimension), output_dimension)
        self.dropout = nn.Dropout(float(dropout))

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        value, gate = self.input_projection(values).chunk(2, dim=-1)
        values = value * F.silu(gate)
        return self.dropout(self.output_projection(self.dropout(values)))


class MultiHeadSelfAttention(nn.Module):
    def __init__(
        self,
        dimension: int,
        num_heads: int,
        *,
        qkv_bias: bool = True,
        attention_dropout: float = 0.0,
        projection_dropout: float = 0.0,
    ):
        super().__init__()
        if int(dimension) % int(num_heads):
            raise ValueError("Attention dimension must be divisible by num_heads.")
        self.dimension = int(dimension)
        self.num_heads = int(num_heads)
        self.head_dimension = self.dimension // self.num_heads
        self.qkv = nn.Linear(self.dimension, 3 * self.dimension, bias=bool(qkv_bias))
        self.output_projection = nn.Linear(self.dimension, self.dimension)
        self.attention_dropout = float(attention_dropout)
        self.projection_dropout = nn.Dropout(float(projection_dropout))

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        batch, tokens, dimension = values.shape
        qkv = (
            self.qkv(values)
            .view(batch, tokens, 3, self.num_heads, self.head_dimension)
            .permute(2, 0, 3, 1, 4)
        )
        query, key, value = qkv.unbind(0)
        attended = F.scaled_dot_product_attention(
            query,
            key,
            value,
            dropout_p=self.attention_dropout if self.training else 0.0,
            is_causal=False,
        )
        attended = attended.transpose(1, 2).reshape(batch, tokens, dimension)
        return self.projection_dropout(self.output_projection(attended))


class TransformerBlock(nn.Module):
    def __init__(
        self,
        dimension: int,
        num_heads: int,
        swiglu_hidden_dimension: int,
        *,
        qkv_bias: bool,
        dropout: float,
        attention_dropout: float,
        drop_path: float,
        layerscale_initial_value: float,
    ):
        super().__init__()
        self.attention_norm = RMSNorm(dimension)
        self.attention = MultiHeadSelfAttention(
            dimension,
            num_heads,
            qkv_bias=qkv_bias,
            attention_dropout=attention_dropout,
            projection_dropout=dropout,
        )
        self.attention_scale = LayerScale(dimension, layerscale_initial_value)
        self.attention_drop_path = DropPath(drop_path)
        self.feedforward_norm = RMSNorm(dimension)
        self.feedforward = SwishGLU(dimension, swiglu_hidden_dimension, dropout=dropout)
        self.feedforward_scale = LayerScale(dimension, layerscale_initial_value)
        self.feedforward_drop_path = DropPath(drop_path)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        values = values + self.attention_drop_path(
            self.attention_scale(self.attention(self.attention_norm(values)))
        )
        return values + self.feedforward_drop_path(
            self.feedforward_scale(self.feedforward(self.feedforward_norm(values)))
        )


@dataclass(frozen=True)
class ViTSmallConfig:
    image_size: int = 224
    patch_size: int = 16
    input_channels: int = 3
    embed_dimension: int = 384
    depth: int = 12
    num_heads: int = 6
    swiglu_hidden_dimension: int = 1024
    qkv_bias: bool = True
    dropout: float = 0.0
    attention_dropout: float = 0.0
    drop_path_rate: float = 0.1
    layerscale_initial_value: float = 1.0e-5
    temporal_window: int = 3

    def __post_init__(self) -> None:
        if self.image_size != 224 or self.patch_size != 16:
            raise ValueError(
                "The DROID encoder requires 224x224 inputs and 16x16 patches."
            )
        if self.embed_dimension != 384 or self.depth != 12 or self.num_heads != 6:
            raise ValueError(
                "The reusable DROID encoder is fixed at ViT-S scale (384/12/6)."
            )
        if self.swiglu_hidden_dimension != 1024:
            raise ValueError(
                "ViT-S with SwiGLU uses hidden_dimension=1024 to preserve capacity."
            )
        if self.temporal_window != 3:
            raise ValueError("DROID temporal histories contain exactly three frames.")


class PatchEmbed(nn.Module):
    def __init__(self, config: ViTSmallConfig):
        super().__init__()
        self.image_size = config.image_size
        self.patch_size = config.patch_size
        self.grid_size = config.image_size // config.patch_size
        self.num_patches = self.grid_size**2
        self.projection = nn.Conv2d(
            config.input_channels,
            config.embed_dimension,
            kernel_size=config.patch_size,
            stride=config.patch_size,
        )

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        if images.shape[-2:] != (self.image_size, self.image_size):
            raise ValueError(
                f"PatchEmbed expected {self.image_size}x{self.image_size}, got {tuple(images.shape[-2:])}."
            )
        return self.projection(images).flatten(2).transpose(1, 2).contiguous()


class TemporalViTEncoder(nn.Module):
    """Jointly encode two historical frames and the current frame.

    A tube mask retains the same spatial patch identities at all three times.
    At inference masking is disabled and the returned current-frame grid always
    contains exactly 14x14 tokens.
    """

    def __init__(
        self,
        config: ViTSmallConfig,
        *,
        masking_ratio: float = 0.60,
        motion_visible_fraction: float = 0.50,
    ):
        super().__init__()
        if not 0.0 <= float(masking_ratio) < 1.0:
            raise ValueError("masking_ratio must lie in [0,1).")
        if not 0.0 <= float(motion_visible_fraction) <= 1.0:
            raise ValueError("motion_visible_fraction must lie in [0,1].")
        self.config = config
        self.masking_ratio = float(masking_ratio)
        self.motion_visible_fraction = float(motion_visible_fraction)
        self.patch_embed = PatchEmbed(config)
        self.num_patches = self.patch_embed.num_patches
        self.grid_size = self.patch_embed.grid_size
        self.cls_token = nn.Parameter(torch.zeros(1, 1, config.embed_dimension))
        self.cls_position = nn.Parameter(torch.zeros(1, 1, config.embed_dimension))
        self.spatial_position = nn.Parameter(
            torch.zeros(1, self.num_patches, config.embed_dimension)
        )
        self.temporal_embedding = nn.Parameter(
            torch.zeros(1, config.temporal_window, 1, config.embed_dimension)
        )
        self.position_dropout = nn.Dropout(config.dropout)
        drop_paths = torch.linspace(0.0, config.drop_path_rate, config.depth).tolist()
        self.blocks = nn.ModuleList(
            TransformerBlock(
                config.embed_dimension,
                config.num_heads,
                config.swiglu_hidden_dimension,
                qkv_bias=config.qkv_bias,
                dropout=config.dropout,
                attention_dropout=config.attention_dropout,
                drop_path=drop_paths[index],
                layerscale_initial_value=config.layerscale_initial_value,
            )
            for index in range(config.depth)
        )
        self.norm = RMSNorm(config.embed_dimension)
        for parameter in (
            self.cls_token,
            self.cls_position,
            self.spatial_position,
            self.temporal_embedding,
        ):
            nn.init.trunc_normal_(parameter, std=0.02)

    @property
    def embed_dimension(self) -> int:
        return self.config.embed_dimension

    def flow_patch_scores(self, flows: torch.Tensor) -> torch.Tensor:
        if flows.dim() != 5 or flows.shape[2] != 2:
            raise ValueError(
                f"Expected temporal flows as (B,pairs,2,H,W), got {tuple(flows.shape)}."
            )
        magnitude = torch.linalg.vector_norm(flows.float(), dim=2).amax(
            dim=1, keepdim=True
        )
        pooled = F.max_pool2d(
            magnitude,
            kernel_size=self.config.patch_size,
            stride=self.config.patch_size,
        )
        return pooled.flatten(1)

    def _patch_validity(
        self, image_validity: torch.Tensor | None, batch: int
    ) -> torch.Tensor:
        if image_validity is None:
            return torch.ones(
                batch, self.num_patches, dtype=torch.bool, device=self.cls_token.device
            )
        expected = (
            batch,
            self.config.temporal_window,
            1,
            self.config.image_size,
            self.config.image_size,
        )
        if tuple(image_validity.shape) != expected:
            raise ValueError(
                f"Expected image validity {expected}, got {tuple(image_validity.shape)}."
            )
        current = image_validity[:, -1].float()
        pooled = F.avg_pool2d(
            current,
            kernel_size=self.config.patch_size,
            stride=self.config.patch_size,
        )
        return pooled.flatten(1) > 0.0

    def _sample_visible_patch_ids(
        self,
        scores: torch.Tensor,
        valid_patches: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch, patch_count = scores.shape
        keep_count = max(1, round(patch_count * (1.0 - self.masking_ratio)))
        motion_count = min(
            keep_count, round(keep_count * self.motion_visible_fraction)
        )
        uniform_count = keep_count - motion_count
        ranked_scores = torch.nan_to_num(
            scores.detach().float(),
            nan=float("-inf"),
            posinf=1.0e9,
            neginf=float("-inf"),
        ).masked_fill(~valid_patches, float("-inf"))
        motion_ids = (
            ranked_scores.topk(motion_count, dim=-1, sorted=False).indices
            if motion_count
            else torch.empty(batch, 0, dtype=torch.long, device=scores.device)
        )
        available = valid_patches.clone()
        if motion_count:
            available.scatter_(1, motion_ids, False)
        random_priority = torch.rand(
            batch, patch_count, device=scores.device
        ).masked_fill(~available, float("-inf"))
        available_count = available.sum(dim=1)
        if bool((available_count < uniform_count).any()):
            # This can only occur for unusually small geometric validity masks.
            fallback = torch.ones_like(available)
            if motion_count:
                fallback.scatter_(1, motion_ids, False)
            random_priority = torch.where(
                available,
                random_priority,
                torch.rand_like(random_priority).masked_fill(~fallback, float("-inf")),
            )
        uniform_ids = (
            random_priority.topk(uniform_count, dim=-1, sorted=False).indices
            if uniform_count
            else torch.empty(batch, 0, dtype=torch.long, device=scores.device)
        )
        visible_ids = torch.cat((motion_ids, uniform_ids), dim=1).sort(dim=1).values
        mask = torch.ones(batch, patch_count, dtype=torch.bool, device=scores.device)
        mask.scatter_(1, visible_ids, False)
        return visible_ids, mask

    def forward(
        self,
        histories: torch.Tensor,
        *,
        optical_flows: torch.Tensor | None = None,
        image_validity: torch.Tensor | None = None,
        apply_mask: bool,
    ) -> dict[str, torch.Tensor]:
        if histories.dim() != 5:
            raise ValueError(
                f"Expected histories as (B,3,3,224,224), got {tuple(histories.shape)}."
            )
        batch, timesteps, channels, height, width = histories.shape
        expected = (
            batch,
            self.config.temporal_window,
            self.config.input_channels,
            self.config.image_size,
            self.config.image_size,
        )
        if tuple(histories.shape) != expected:
            raise ValueError(
                f"Expected histories {expected}, got {tuple(histories.shape)}."
            )
        patches = self.patch_embed(
            histories.reshape(batch * timesteps, channels, height, width)
        )
        patches = patches.view(batch, timesteps, self.num_patches, self.embed_dimension)
        patches = (
            patches
            + self.spatial_position[:, None].to(patches.dtype)
            + self.temporal_embedding.to(patches.dtype)
        )
        if apply_mask:
            if optical_flows is None:
                raise ValueError("Masked pretraining requires MEMFOF optical flow.")
            scores = self.flow_patch_scores(optical_flows)
            visible_ids, mask = self._sample_visible_patch_ids(
                scores, self._patch_validity(image_validity, batch)
            )
        else:
            visible_ids = torch.arange(
                self.num_patches, device=histories.device
            ).expand(batch, -1)
            mask = torch.zeros(
                batch, self.num_patches, dtype=torch.bool, device=histories.device
            )
        gather_index = visible_ids[:, None, :, None].expand(
            batch, timesteps, visible_ids.shape[1], self.embed_dimension
        )
        visible = torch.gather(patches, dim=2, index=gather_index)
        visible_per_frame = visible.shape[2]
        visible = visible.flatten(1, 2)
        cls = self.cls_token.to(visible.dtype).expand(batch, -1, -1)
        cls = cls + self.cls_position.to(visible.dtype)
        tokens = self.position_dropout(torch.cat((cls, visible), dim=1))
        for block in self.blocks:
            tokens = block(tokens)
        tokens = self.norm(tokens)
        current_start = 1 + (timesteps - 1) * visible_per_frame
        current = tokens[:, current_start : current_start + visible_per_frame]
        return {
            "cls_token": tokens[:, 0].contiguous(),
            "current_patch_tokens": current.contiguous(),
            "decoder_tokens": tokens.contiguous(),
            "visible_patch_ids": visible_ids.contiguous(),
            "patch_mask": mask.contiguous(),
        }

    def inference_features(self, histories: torch.Tensor) -> dict[str, torch.Tensor]:
        """Reusable encoder-only API with masking and pretraining heads removed."""
        output = self.forward(histories, apply_mask=False)
        patches = output["current_patch_tokens"]
        if patches.shape[1:] != (self.num_patches, self.embed_dimension):
            raise RuntimeError(
                "Encoder inference must retain the complete current-frame patch grid; "
                f"got {tuple(patches.shape)}."
            )
        return {
            "cls_token": output["cls_token"],
            "patch_tokens": patches,
        }


class ContrastiveProjector(nn.Module):
    def __init__(
        self,
        input_dimension: int = 384,
        hidden_dimension: int = 1024,
        output_dimension: int = 256,
    ):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dimension, hidden_dimension),
            nn.SiLU(),
            nn.Linear(hidden_dimension, output_dimension),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.network(features), dim=-1, eps=1.0e-6)
