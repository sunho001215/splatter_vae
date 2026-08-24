from __future__ import annotations

from contextlib import nullcontext
import math
from typing import Dict, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.gaussian.parameterization import NUM_GAUSSIANS
from .backbones import MultiHeadSelfAttention, RMSNorm, ViTBackbone, ViTSmallConfig
from .config import SPLATTERVAE_ARCHITECTURE, TEMPORAL_WINDOW


class ParentTransformerBlock(nn.Module):
    """Ordinary pre-norm Transformer block for the 256 parent identities."""

    def __init__(self, dim: int, num_heads: int, mlp_ratio: float):
        super().__init__()
        hidden_dim = int(dim * mlp_ratio)
        self.norm1 = nn.LayerNorm(dim)
        self.attn = MultiHeadSelfAttention(
            dim=dim,
            num_heads=num_heads,
            qkv_bias=True,
            attn_dropout=0.0,
            proj_dropout=0.0,
        )
        self.norm2 = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, dim),
        )

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        tokens = tokens + self.attn(self.norm1(tokens))
        return tokens + self.ffn(self.norm2(tokens))


class SplatterVAE(nn.Module):
    """Invariant-only parent-anchored grouped Gaussian set model."""

    architecture_id = SPLATTERVAE_ARCHITECTURE
    motion_params_per_gaussian = 6

    def __init__(
        self,
        vit_cfg: Dict,
        img_height: int,
        img_width: int,
        gaussian_params_per_gaussian: int,
        inv_tube_mask_ratio: float = 0.50,
        tube_mask_per_view: bool = True,
        state_dim: int = 256,
        flow_patch_threshold_pixels: float = 0.5,
        temporal_modeling: bool = True,
        motion_translation_max: float = 0.5,
        decoder_num_parent_tokens: int = 256,
        decoder_gaussians_per_parent: int = 8,
        decoder_dim: int = 128,
        decoder_depth: int = 2,
        decoder_num_heads: int = 4,
        decoder_mlp_ratio: float = 4.0,
        decoder_global_center: Sequence[float] = (0.0, 0.5, 0.1),
        decoder_anchor_init_std: float = 0.15,
        decoder_parent_offset_scale: float = 0.1,
        decoder_child_radius: float = 0.05,
    ):
        super().__init__()
        self.img_height = int(img_height)
        self.img_width = int(img_width)
        patch_size = int(vit_cfg.get("patch_size", 16))
        if self.img_height % patch_size != 0 or self.img_width % patch_size != 0:
            raise ValueError("Image size must be divisible by ViT patch size.")
        self.patch_h = patch_size
        self.patch_w = patch_size

        enc_cfg = ViTSmallConfig(
            img_height=self.img_height,
            img_width=self.img_width,
            patch_size=patch_size,
            in_chans=int(vit_cfg.get("in_chans", 3)),
            embed_dim=int(vit_cfg.get("embed_dim", 384)),
            depth=int(vit_cfg.get("depth", 12)),
            num_heads=int(vit_cfg.get("num_heads", 6)),
            mlp_ratio=float(vit_cfg.get("mlp_ratio", 4.0)),
            qkv_bias=bool(vit_cfg.get("qkv_bias", True)),
            dropout=float(vit_cfg.get("dropout", 0.0)),
            attn_dropout=float(vit_cfg.get("attn_dropout", 0.0)),
            drop_path_rate=float(vit_cfg.get("drop_path_rate", 0.0)),
            layerscale_init=float(vit_cfg.get("layerscale_init", 1e-5)),
        )
        self.invariant_encoder = ViTBackbone(enc_cfg)
        self.n_tokens_per_frame = self.invariant_encoder.num_patches
        self.grid_size = self.invariant_encoder.grid_size
        latent_dim = enc_cfg.embed_dim

        self.state_dim = int(state_dim)
        self.inv_tube_mask_ratio = float(inv_tube_mask_ratio)
        self.tube_mask_per_view = bool(tube_mask_per_view)
        self.flow_patch_threshold_pixels = float(flow_patch_threshold_pixels)
        self.temporal_modeling = bool(temporal_modeling)
        self.motion_translation_max = float(motion_translation_max)
        if self.temporal_modeling and self.motion_translation_max <= 0.0:
            raise ValueError("Motion translation bound must be positive.")

        self.state_token = nn.Parameter(torch.zeros(1, 1, latent_dim))
        self.temporal_embed = nn.Parameter(torch.zeros(1, TEMPORAL_WINDOW, 1, 1, latent_dim))
        self.state_norm = RMSNorm(latent_dim)
        self.invariant_encoder_output_proj = nn.Linear(latent_dim, self.state_dim, bias=True)

        self.num_parent_tokens = int(decoder_num_parent_tokens)
        self.gaussians_per_parent = int(decoder_gaussians_per_parent)
        self.decoder_dim = int(decoder_dim)
        self.decoder_depth = int(decoder_depth)
        self.decoder_num_heads = int(decoder_num_heads)
        self.decoder_mlp_ratio = float(decoder_mlp_ratio)
        configured_global_center = tuple(
            float(value) for value in decoder_global_center
        )
        global_center = torch.tensor(configured_global_center, dtype=torch.float32)
        if global_center.shape != (3,) or not torch.isfinite(global_center).all():
            raise ValueError("decoder_global_center must contain three finite values.")
        self.decoder_global_center = configured_global_center
        self.decoder_anchor_init_std = float(decoder_anchor_init_std)
        self.decoder_parent_offset_scale = float(decoder_parent_offset_scale)
        self.decoder_child_radius = float(decoder_child_radius)
        if self.num_parent_tokens != 256 or self.gaussians_per_parent != 8:
            raise ValueError("The grouped decoder requires exactly 256 parents and 8 children per parent.")
        if self.num_parent_tokens * self.gaussians_per_parent != NUM_GAUSSIANS:
            raise ValueError(f"The grouped decoder must produce exactly {NUM_GAUSSIANS} Gaussians.")
        if self.decoder_depth != 2:
            raise ValueError("The grouped decoder requires exactly two parent Transformer blocks.")
        if (
            self.decoder_dim <= 0
            or self.decoder_num_heads <= 0
            or self.decoder_dim % self.decoder_num_heads != 0
        ):
            raise ValueError("decoder_dim must be positive and divisible by decoder_num_heads.")
        if not math.isfinite(self.decoder_mlp_ratio) or self.decoder_mlp_ratio <= 0.0:
            raise ValueError("decoder_mlp_ratio must be positive.")
        if not math.isfinite(self.decoder_anchor_init_std) or self.decoder_anchor_init_std <= 0.0:
            raise ValueError("decoder_anchor_init_std must be positive.")
        if (
            not math.isfinite(self.decoder_parent_offset_scale)
            or self.decoder_parent_offset_scale <= 0.0
        ):
            raise ValueError("decoder_parent_offset_scale must be positive.")
        if not math.isfinite(self.decoder_child_radius) or self.decoder_child_radius <= 0.0:
            raise ValueError("decoder_child_radius must be positive.")
        if int(gaussian_params_per_gaussian) <= 3:
            raise ValueError("Gaussian prediction width must include XYZ and renderer attributes.")

        self.num_gaussians = NUM_GAUSSIANS
        self.parent_anchor = nn.Parameter(
            torch.empty(self.num_parent_tokens, 3)
        )
        self.parent_tokens = nn.Parameter(
            torch.empty(self.num_parent_tokens, self.decoder_dim)
        )
        self.parent_token_norm = nn.LayerNorm(self.decoder_dim)
        self.decoder_film = nn.Sequential(
            nn.Linear(self.state_dim, self.decoder_dim),
            nn.SiLU(),
            nn.Linear(self.decoder_dim, 2 * self.decoder_dim),
        )
        self.parent_transformer = nn.ModuleList(
            ParentTransformerBlock(
                dim=self.decoder_dim,
                num_heads=self.decoder_num_heads,
                mlp_ratio=self.decoder_mlp_ratio,
            )
            for _ in range(2)
        )
        self.parent_position_head = nn.Linear(self.decoder_dim, 3)
        self.child_identities = nn.Parameter(
            torch.empty(self.gaussians_per_parent, self.decoder_dim)
        )
        self.child_expansion_mlp = nn.Sequential(
            nn.Linear(2 * self.decoder_dim, self.decoder_dim),
            nn.SiLU(),
            nn.Linear(self.decoder_dim, self.decoder_dim),
            nn.SiLU(),
        )

        self.xyz_head = nn.Linear(self.decoder_dim, 3)
        self.gaussian_attribute_head = nn.Linear(
            self.decoder_dim, int(gaussian_params_per_gaussian) - 3
        )
        # Parent motion starts exactly at zero, while tiny child offsets break
        # symmetry without scattering children away from their parent anchor.
        nn.init.zeros_(self.parent_position_head.weight)
        nn.init.zeros_(self.parent_position_head.bias)
        nn.init.normal_(self.xyz_head.weight, mean=0.0, std=1.0e-3)
        nn.init.zeros_(self.xyz_head.bias)
        nn.init.zeros_(self.gaussian_attribute_head.bias)

        self.dense_motion_head = None
        if self.temporal_modeling:
            self.dense_motion_head = nn.Linear(
                self.decoder_dim, self.motion_params_per_gaussian
            )
            nn.init.zeros_(self.dense_motion_head.weight)
            nn.init.zeros_(self.dense_motion_head.bias)

        nn.init.trunc_normal_(self.state_token, std=0.02)
        nn.init.trunc_normal_(self.temporal_embed, std=0.02)
        with torch.no_grad():
            nn.init.normal_(
                self.parent_anchor, mean=0.0, std=self.decoder_anchor_init_std
            )
            self.parent_anchor.add_(global_center)
        nn.init.trunc_normal_(self.parent_tokens, std=0.02)
        nn.init.trunc_normal_(self.child_identities, std=0.02)

    def decoder_configuration(self) -> Dict[str, object]:
        """Return the architecture-sensitive decoder settings stored in checkpoints."""
        return {
            "num_parent_tokens": self.num_parent_tokens,
            "gaussians_per_parent": self.gaussians_per_parent,
            "dim": self.decoder_dim,
            "depth": self.decoder_depth,
            "num_heads": self.decoder_num_heads,
            "mlp_ratio": self.decoder_mlp_ratio,
            "global_center": list(self.decoder_global_center),
            "anchor_init_std": self.decoder_anchor_init_std,
            "parent_offset_scale": self.decoder_parent_offset_scale,
            "child_radius": self.decoder_child_radius,
        }

    def validate_checkpoint_decoder_configuration(self, checkpoint: Dict) -> None:
        """Reject checkpoints whose parent-anchor geometry differs from this model."""
        found = checkpoint.get("decoder_configuration")
        expected = self.decoder_configuration()
        if not isinstance(found, dict):
            raise ValueError(
                "Checkpoint is missing parent-anchor decoder configuration metadata."
            )
        mismatches = {
            key: {"checkpoint": found.get(key), "configured": value}
            for key, value in expected.items()
            if found.get(key) != value
        }
        if mismatches:
            raise ValueError(
                "Checkpoint decoder configuration does not match the current model: "
                f"{mismatches}."
            )

    def _validate_pretraining_inputs(self, images: torch.Tensor, optical_flows: torch.Tensor) -> None:
        if images.dim() != 6:
            raise ValueError(f"Expected pretraining images as (B,T,A,3,H,W), got {tuple(images.shape)}.")
        batch, timesteps, views, _channels, height, width = images.shape
        expected_timesteps = TEMPORAL_WINDOW if self.temporal_modeling else 1
        expected_images = (batch, expected_timesteps, views, 3, self.img_height, self.img_width)
        if tuple(images.shape) != expected_images:
            raise ValueError(f"Expected pretraining images as {expected_images}, got {tuple(images.shape)}.")
        expected_flows = (batch, 3, views, 2, height, width)
        if tuple(optical_flows.shape) != expected_flows:
            raise ValueError(f"Expected optical flows as {expected_flows}, got {tuple(optical_flows.shape)}.")
        if not images.is_floating_point() or not optical_flows.is_floating_point():
            raise TypeError("Neural pretraining inputs must be floating-point tensors.")

    def flow_patch_scores(self, optical_flows: torch.Tensor) -> torch.Tensor:
        """Pool max teacher-flow magnitude to the ViT grid as (B,A,N)."""
        if optical_flows.dim() != 6 or optical_flows.shape[1] != 3 or optical_flows.shape[3] != 2:
            raise ValueError(f"Expected optical flows as (B,3,A,2,H,W), got {tuple(optical_flows.shape)}.")
        batch, _pairs, num_views, _channels, height, width = optical_flows.shape
        magnitude = optical_flows.float().square().sum(dim=3).clamp_min(0.0).sqrt()
        dynamic_pixels = magnitude.amax(dim=1)
        pooled = F.max_pool2d(
            dynamic_pixels.reshape(batch * num_views, 1, height, width),
            kernel_size=(self.patch_h, self.patch_w),
            stride=(self.patch_h, self.patch_w),
        )
        return pooled.flatten(1).view(batch, num_views, self.n_tokens_per_frame).to(optical_flows.dtype)

    def _sample_dynamic_visible_patch_ids(
        self, patch_scores: torch.Tensor, mask_ratio: float
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        num_patches = patch_scores.shape[-1]
        ratio = min(max(float(mask_ratio), 0.0), 0.99)
        keep_count = max(1, min(num_patches, int(round(num_patches * (1.0 - ratio)))))
        scores = patch_scores.detach()
        dynamic = scores > self.flow_patch_threshold_pixels
        priority = torch.where(dynamic, 2.0 + scores, torch.rand_like(scores))
        ids_keep = torch.topk(priority, k=keep_count, dim=-1, largest=True, sorted=False).indices
        mask = torch.ones_like(patch_scores, dtype=torch.bool)
        mask.scatter_(dim=-1, index=ids_keep, value=False)
        return ids_keep, mask

    def _encode_invariant_branch(
        self, images: torch.Tensor, optical_flows: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        bsz, timesteps, num_views, channels, height, width = images.shape
        flat_images = images.reshape(bsz * timesteps * num_views, channels, height, width).contiguous()
        patch_tokens, _ = self.invariant_encoder.patch_embed(flat_images)
        patch_tokens = patch_tokens.view(bsz, timesteps, num_views, self.n_tokens_per_frame, -1)
        spatial = self.invariant_encoder.pos_embed[:, 1:].to(patch_tokens.dtype).view(
            1, 1, 1, self.n_tokens_per_frame, -1
        )
        patch_tokens = patch_tokens + spatial + self.temporal_embed[:, :timesteps].to(patch_tokens.dtype)
        dynamic_scores = self.flow_patch_scores(optical_flows)
        if self.tube_mask_per_view:
            ids_keep, inv_mask = self._sample_dynamic_visible_patch_ids(
                dynamic_scores, self.inv_tube_mask_ratio
            )
        else:
            ids_keep, shared_mask = self._sample_dynamic_visible_patch_ids(
                dynamic_scores.amax(dim=1, keepdim=True), self.inv_tube_mask_ratio
            )
            ids_keep = ids_keep.expand(bsz, num_views, -1).contiguous()
            inv_mask = shared_mask.expand(bsz, num_views, -1).contiguous()
        gather_index = ids_keep[:, None, :, :, None].expand(
            bsz, timesteps, num_views, ids_keep.shape[-1], patch_tokens.shape[-1]
        )
        visible = torch.gather(patch_tokens, dim=3, index=gather_index)
        visible = visible.reshape(bsz, timesteps * num_views * ids_keep.shape[-1], -1)
        tokens = torch.cat((self.state_token.to(visible.dtype).expand(bsz, -1, -1), visible), dim=1)
        tokens = self.invariant_encoder.pos_drop(tokens)
        for block in self.invariant_encoder.blocks:
            tokens = block(tokens)
        tokens = self.invariant_encoder.norm(tokens)
        state = self.invariant_encoder_output_proj(self.state_norm(tokens[:, 0]))
        return state.contiguous(), inv_mask

    def encode_pretraining(
        self, images: torch.Tensor, optical_flows: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """Encode with teacher-flow-guided tube masking; no dependent branch exists."""
        self._validate_pretraining_inputs(images, optical_flows)
        state, inv_mask = self._encode_invariant_branch(images, optical_flows)
        return {"s_inv": state, "inv_mask": inv_mask}

    def _encode_invariant_all_patches(self, images: torch.Tensor) -> torch.Tensor:
        if images.dim() != 6:
            raise ValueError(f"Expected policy images as (B,T,A,3,H,W), got {tuple(images.shape)}.")
        batch, timesteps, views, channels, height, width = images.shape
        expected_timesteps = TEMPORAL_WINDOW if self.temporal_modeling else 1
        expected = (batch, expected_timesteps, views, 3, self.img_height, self.img_width)
        if tuple(images.shape) != expected:
            raise ValueError(f"Expected policy images as {expected}, got {tuple(images.shape)}.")
        flat = images.reshape(batch * timesteps * views, channels, height, width).contiguous()
        patch_tokens, _ = self.invariant_encoder.patch_embed(flat)
        patch_tokens = patch_tokens.view(batch, timesteps, views, self.n_tokens_per_frame, -1)
        spatial = self.invariant_encoder.pos_embed[:, 1:].to(patch_tokens.dtype).view(
            1, 1, 1, self.n_tokens_per_frame, -1
        )
        tokens = patch_tokens + spatial + self.temporal_embed[:, :timesteps].to(patch_tokens.dtype)
        tokens = tokens.reshape(batch, timesteps * views * self.n_tokens_per_frame, -1)
        tokens = torch.cat((self.state_token.to(tokens.dtype).expand(batch, -1, -1), tokens), dim=1)
        tokens = self.invariant_encoder.pos_drop(tokens)
        for block in self.invariant_encoder.blocks:
            tokens = block(tokens)
        tokens = self.invariant_encoder.norm(tokens)
        return self.invariant_encoder_output_proj(self.state_norm(tokens[:, 0])).contiguous()

    @torch.no_grad()
    def inference_features(self, images: torch.Tensor) -> Dict[str, torch.Tensor]:
        context = (
            torch.autocast(device_type="cuda", dtype=torch.bfloat16)
            if images.device.type == "cuda" else nullcontext()
        )
        with context:
            state = self._encode_invariant_all_patches(images)
        return {"s_inv": state.float()}

    def decode_grouped_features(
        self, s_inv: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return conditioned parents, coordinated parents, and child features."""
        if s_inv.dim() != 2 or s_inv.shape[-1] != self.state_dim:
            raise ValueError(f"Expected s_inv as (B,{self.state_dim}), got {tuple(s_inv.shape)}.")
        batch = s_inv.shape[0]
        base = self.parent_token_norm(self.parent_tokens).unsqueeze(0).expand(batch, -1, -1)
        gamma, beta = self.decoder_film(s_inv).chunk(2, dim=-1)
        conditioned = base * (1.0 + gamma[:, None, :]) + beta[:, None, :]
        parents = conditioned
        for block in self.parent_transformer:
            parents = block(parents)
        parent_expanded = parents[:, :, None, :].expand(
            batch, self.num_parent_tokens, self.gaussians_per_parent, self.decoder_dim
        )
        children = self.child_identities[None, None, :, :].expand(
            batch, self.num_parent_tokens, self.gaussians_per_parent, self.decoder_dim
        )
        child_features = self.child_expansion_mlp(
            torch.cat((parent_expanded, children), dim=-1)
        )
        return conditioned, parents, child_features

    def _decode_world_positions(
        self, parents: torch.Tensor, child_features: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Decode unbounded parent centers and bounded parent-local child offsets."""
        parent_displacements = (
            self.decoder_parent_offset_scale * self.parent_position_head(parents)
        )
        parent_centers = self.parent_anchor[None] + parent_displacements
        child_offsets = self.decoder_child_radius * torch.tanh(
            self.xyz_head(child_features)
        )
        world_xyz = parent_centers[:, :, None, :] + child_offsets
        return parent_centers, parent_displacements, world_xyz

    def predict_gaussian_parameters(self, s_inv: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Decode only s_inv into 2048 global Gaussians and translations."""
        _conditioned, parents, child_features = self.decode_grouped_features(s_inv)
        _parent_centers, _parent_displacements, world_xyz = (
            self._decode_world_positions(parents, child_features)
        )
        shared = child_features.flatten(1, 2)
        raw_gaussian_params = torch.cat(
            (world_xyz.flatten(1, 2), self.gaussian_attribute_head(shared)), dim=-1
        )
        outputs = {"raw_gaussian_params": raw_gaussian_params.contiguous()}
        if self.temporal_modeling:
            if self.dense_motion_head is None:
                raise RuntimeError("Temporal modeling is enabled without a dense motion head.")
            outputs["raw_motion_params"] = self.dense_motion_head(shared).contiguous()
        return outputs

    @torch.no_grad()
    def position_initialization_diagnostics(self) -> Dict[str, object]:
        """Summarize canonical anchors and zero-state parent/child positions."""
        state = self.parent_anchor.new_zeros(1, self.state_dim)
        _conditioned, parents, child_features = self.decode_grouped_features(state)
        parent_centers, parent_displacements, world_xyz = self._decode_world_positions(
            parents, child_features
        )
        child_offsets = world_xyz - parent_centers[:, :, None, :]
        anchor_mean = self.parent_anchor.mean(0)
        anchor_std = self.parent_anchor.std(0, unbiased=False)
        configured_center = self.parent_anchor.new_tensor(self.decoder_global_center)
        return {
            "parent_anchor_mean": tuple(float(value) for value in anchor_mean),
            "parent_anchor_std": tuple(float(value) for value in anchor_std),
            "parent_anchor_center_error": float(
                torch.linalg.vector_norm(anchor_mean - configured_center)
            ),
            "parent_displacement_max": float(parent_displacements.abs().max()),
            "child_offset_rms": float(child_offsets.square().mean().sqrt()),
            "child_offset_max": float(torch.linalg.vector_norm(child_offsets, dim=-1).max()),
            "xyz_min": tuple(float(value) for value in world_xyz.amin(dim=(0, 1, 2))),
            "xyz_max": tuple(float(value) for value in world_xyz.amax(dim=(0, 1, 2))),
        }

    def decode_sequence(self, s_inv: torch.Tensor) -> Dict[str, torch.Tensor]:
        return self.predict_gaussian_parameters(s_inv)

    @torch.no_grad()
    def policy_state(self, images: torch.Tensor) -> torch.Tensor:
        context = (
            torch.autocast(device_type="cuda", dtype=torch.bfloat16)
            if images.device.type == "cuda" else nullcontext()
        )
        with context:
            state = self._encode_invariant_all_patches(images)
        return state.float()
