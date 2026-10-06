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

    def forward(
        self,
        values: torch.Tensor,
        token_validity: torch.Tensor | None = None,
    ) -> torch.Tensor:
        batch, tokens, dimension = values.shape
        if token_validity is not None:
            if token_validity.shape != (batch, tokens):
                raise ValueError(
                    f"Attention validity {tuple(token_validity.shape)} does not match {(batch, tokens)}."
                )
            if not token_validity[:, 0].all() or not token_validity.any(dim=1).all():
                raise ValueError(
                    "Every attention sequence must contain a valid CLS token."
                )
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
            attn_mask=(
                None if token_validity is None else token_validity[:, None, None, :]
            ),
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

    def forward(
        self,
        values: torch.Tensor,
        token_validity: torch.Tensor | None = None,
    ) -> torch.Tensor:
        values = values + self.attention_drop_path(
            self.attention_scale(
                self.attention(self.attention_norm(values), token_validity)
            )
        )
        values = values + self.feedforward_drop_path(
            self.feedforward_scale(self.feedforward(self.feedforward_norm(values)))
        )
        if token_validity is not None:
            values = values.masked_fill(~token_validity[..., None], 0.0)
        return values


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

    def motion_patch_scores(self, motion: torch.Tensor) -> torch.Tensor:
        if motion.dim() != 4 or motion.shape[1] != 1:
            raise ValueError(
                f"Expected middle-frame motion as (B,1,H,W), got {tuple(motion.shape)}."
            )
        pooled = F.max_pool2d(
            torch.nan_to_num(motion.float(), nan=0.0, posinf=0.0, neginf=0.0),
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
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        batch, patch_count = scores.shape
        valid_counts = valid_patches.sum(dim=1)
        if bool((valid_counts == 0).any()):
            raise ValueError(
                "Every sample must contain at least one geometrically valid patch."
            )
        keep_counts = torch.clamp(
            torch.round(valid_counts.float() * (1.0 - self.masking_ratio)).long(),
            min=1,
        )
        maximum_keep = int(keep_counts.max().item())
        visible_ids = torch.full(
            (batch, maximum_keep), -1, dtype=torch.long, device=scores.device
        )
        visible_validity = torch.zeros(
            batch, maximum_keep, dtype=torch.bool, device=scores.device
        )
        ssl_mask = torch.zeros(
            batch, patch_count, dtype=torch.bool, device=scores.device
        )
        for sample in range(batch):
            keep = int(keep_counts[sample])
            valid_ids = torch.nonzero(valid_patches[sample], as_tuple=False).flatten()
            valid_scores = torch.nan_to_num(
                scores[sample, valid_ids].detach().float(),
                nan=0.0,
                posinf=1.0e9,
                neginf=0.0,
            )
            motion_is_degenerate = float(valid_scores.amax().item()) <= 1.0e-8
            motion_count = (
                0
                if motion_is_degenerate
                else min(keep, round(keep * self.motion_visible_fraction))
            )
            if motion_count:
                motion_local = valid_scores.topk(motion_count, sorted=False).indices
                motion_ids = valid_ids[motion_local]
            else:
                motion_ids = valid_ids.new_empty(0)
            available = valid_patches[sample].clone()
            if motion_ids.numel():
                available[motion_ids] = False
            random_count = keep - int(motion_ids.numel())
            priorities = torch.rand(patch_count, device=scores.device).masked_fill(
                ~available, float("-inf")
            )
            random_ids = priorities.topk(random_count, sorted=False).indices
            selected = torch.cat((motion_ids, random_ids)).sort().values
            visible_ids[sample, :keep] = selected
            visible_validity[sample, :keep] = True
            ssl_mask[sample] = valid_patches[sample]
            ssl_mask[sample, selected] = False
        return visible_ids, visible_validity, ssl_mask, keep_counts

    def forward(
        self,
        histories: torch.Tensor,
        *,
        motion_maps: torch.Tensor | None = None,
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
            if motion_maps is None:
                raise ValueError(
                    "Masked pretraining requires cached middle-frame motion."
                )
            patch_validity = self._patch_validity(image_validity, batch)
            scores = self.motion_patch_scores(motion_maps)
            visible_ids, visible_validity, mask, keep_counts = (
                self._sample_visible_patch_ids(scores, patch_validity)
            )
        else:
            patch_validity = self._patch_validity(image_validity, batch)
            keep_counts = patch_validity.sum(dim=1)
            maximum_keep = int(keep_counts.max().item())
            visible_ids = torch.full(
                (batch, maximum_keep), -1, dtype=torch.long, device=histories.device
            )
            visible_validity = torch.zeros(
                batch, maximum_keep, dtype=torch.bool, device=histories.device
            )
            for sample in range(batch):
                selected = torch.nonzero(
                    patch_validity[sample], as_tuple=False
                ).flatten()
                visible_ids[sample, : selected.numel()] = selected
                visible_validity[sample, : selected.numel()] = True
            mask = torch.zeros_like(patch_validity)
        safe_visible_ids = visible_ids.clamp_min(0)
        gather_index = safe_visible_ids[:, None, :, None].expand(
            batch, timesteps, visible_ids.shape[1], self.embed_dimension
        )
        visible = torch.gather(patches, dim=2, index=gather_index)
        visible = visible.masked_fill(~visible_validity[:, None, :, None], 0.0)
        visible_per_frame = visible.shape[2]
        visible = visible.flatten(1, 2)
        cls = self.cls_token.to(visible.dtype).expand(batch, -1, -1)
        cls = cls + self.cls_position.to(visible.dtype)
        tokens = self.position_dropout(torch.cat((cls, visible), dim=1))
        token_validity = torch.cat(
            (
                torch.ones(batch, 1, dtype=torch.bool, device=histories.device),
                visible_validity[:, None].expand(-1, timesteps, -1).reshape(batch, -1),
            ),
            dim=1,
        )
        for block in self.blocks:
            tokens = block(tokens, token_validity)
        tokens = self.norm(tokens)
        tokens = tokens.masked_fill(~token_validity[..., None], 0.0)
        current_start = 1 + (timesteps - 1) * visible_per_frame
        current = tokens[:, current_start : current_start + visible_per_frame]
        return {
            "cls_token": tokens[:, 0].contiguous(),
            "current_patch_tokens": current.contiguous(),
            "decoder_tokens": tokens.contiguous(),
            "visible_patch_ids": visible_ids.contiguous(),
            "visible_patch_validity": visible_validity.contiguous(),
            "decoder_token_validity": token_validity.contiguous(),
            "patch_mask": mask.contiguous(),
            "patch_validity": patch_validity.contiguous(),
            "num_visible_patches": keep_counts.contiguous(),
        }

    def inference_features(
        self,
        histories: torch.Tensor,
        image_validity: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """Reusable encoder-only API with masking and pretraining heads removed."""
        output = self.forward(
            histories, image_validity=image_validity, apply_mask=False
        )
        selected = output["current_patch_tokens"]
        patches = selected.new_zeros(
            histories.shape[0], self.num_patches, self.embed_dimension
        )
        for sample in range(histories.shape[0]):
            count = int(output["num_visible_patches"][sample])
            patches[sample, output["visible_patch_ids"][sample, :count]] = selected[
                sample, :count
            ]
        return {
            "cls_token": output["cls_token"],
            "patch_tokens": patches,
            "patch_validity": output["patch_validity"],
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
