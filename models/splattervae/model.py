from __future__ import annotations

from contextlib import nullcontext
from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .backbones import DPTBackbone, DPTOutputHead, RMSNorm, TransformerBlock, ViTBackbone, ViTSmallConfig
from .config import TEMPORAL_WINDOW


class FiLMTokenTransformer(nn.Module):
    """Fixed-grid token transformer with per-layer FiLM conditioning."""

    def __init__(
        self,
        num_tokens: int,
        embed_dim: int,
        condition_dim: int,
        depth: int,
        num_heads: int,
        mlp_ratio: float,
        qkv_bias: bool = True,
        dropout: float = 0.0,
        attn_dropout: float = 0.0,
        drop_path_rate: float = 0.0,
        layerscale_init: float = 1e-5,
        selected_layers: Sequence[int] = (2, 5, 8, 11),
    ):
        super().__init__()
        self.num_tokens = int(num_tokens)
        self.embed_dim = int(embed_dim)
        self.condition_dim = int(condition_dim)

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, 1 + num_tokens, embed_dim))
        self.pos_drop = nn.Dropout(dropout)

        dpr = torch.linspace(0, drop_path_rate, depth).tolist()
        self.blocks = nn.ModuleList(
            [
                TransformerBlock(
                    dim=embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    dropout=dropout,
                    attn_dropout=attn_dropout,
                    drop_path=dpr[i],
                    layerscale_init=layerscale_init,
                )
                for i in range(depth)
            ]
        )
        self.film_mlps = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(condition_dim, 4 * embed_dim),
                    nn.SiLU(),
                    nn.Linear(4 * embed_dim, 2 * embed_dim),
                )
                for _ in range(depth)
            ]
        )
        self.norm = RMSNorm(embed_dim)
        self.selected_layers = tuple(int(idx) for idx in selected_layers)

        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        for mlp in self.film_mlps:
            final = mlp[-1]
            nn.init.zeros_(final.weight)
            nn.init.zeros_(final.bias)

    def forward(self, tokens: torch.Tensor, condition: torch.Tensor) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        if tokens.shape[1] != self.num_tokens:
            raise ValueError(f"FiLMTokenTransformer expected {self.num_tokens} tokens, got {tokens.shape[1]}")
        if condition.dim() != 2 or condition.shape[0] != tokens.shape[0]:
            raise ValueError(
                f"Expected condition as (B,C) with B={tokens.shape[0]}, got {tuple(condition.shape)}."
            )

        batch = tokens.shape[0]
        cls_tokens = self.cls_token.to(tokens.dtype).expand(batch, -1, -1)
        x = torch.cat([cls_tokens, tokens], dim=1)
        x = self.pos_drop(x + self.pos_embed.to(tokens.dtype))

        hidden_states: List[torch.Tensor] = []
        for idx, block in enumerate(self.blocks):
            x = block(x)
            gamma, beta = self.film_mlps[idx](condition).chunk(2, dim=-1)
            gamma = gamma.unsqueeze(1)
            beta = beta.unsqueeze(1)
            x = torch.cat([x[:, :1], (1.0 + gamma) * x[:, 1:] + beta], dim=1)
            if idx in self.selected_layers:
                hidden_states.append(x)

        x = self.norm(x)
        return x[:, 1:, :], hidden_states


class SplatterVAE(nn.Module):
    """Joint three-frame encoder and dense per-Gaussian motion decoder."""

    motion_params_per_gaussian = 6

    def __init__(
        self,
        vit_cfg: Dict,
        img_height: int,
        img_width: int,
        splatter_channels: int,
        dep_mask_eval: bool = True,
        dpt_features: int = 256,
        inv_tube_mask_ratio: float = 0.50,
        dep_mask_ratio: float = 0.75,
        tube_mask_per_view: bool = True,
        state_dim: int = 256,
        view_dim: Optional[int] = None,
        decoder_condition_mode: str = "concat",
        gaussians_per_pixel: int = 1,
        flow_patch_threshold_pixels: float = 0.5,
        motion_translation_max: float = 0.5,
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
            selected_layers=tuple(vit_cfg.get("selected_layers", (2, 5, 8, 11))),
        )
        if len(enc_cfg.selected_layers) != 4:
            raise ValueError("DPT requires exactly four selected transformer layers.")

        self.invariant_encoder = ViTBackbone(enc_cfg)
        self.dependent_encoder = ViTBackbone(enc_cfg)
        self.n_tokens_per_frame = self.invariant_encoder.num_patches
        self.grid_size = self.invariant_encoder.grid_size
        latent_dim = enc_cfg.embed_dim

        self.state_dim = int(state_dim)
        self.view_dim = int(self.state_dim if view_dim is None else view_dim)
        self.decoder_condition_mode = str(decoder_condition_mode).lower()
        if self.decoder_condition_mode not in {"concat", "add"}:
            raise ValueError(
                f"decoder_condition_mode must be 'concat' or 'add', got {decoder_condition_mode!r}."
            )
        if self.decoder_condition_mode == "add" and self.state_dim != self.view_dim:
            raise ValueError(
                f"decoder_condition_mode='add' requires state_dim == view_dim, "
                f"got state_dim={self.state_dim} and view_dim={self.view_dim}."
            )

        self.inv_tube_mask_ratio = float(inv_tube_mask_ratio)
        self.dep_mask_ratio = float(dep_mask_ratio)
        self.tube_mask_per_view = bool(tube_mask_per_view)
        self.dep_mask_eval = bool(dep_mask_eval)
        self.splatter_channels = int(splatter_channels)
        self.gaussians_per_pixel = int(gaussians_per_pixel)
        self.flow_patch_threshold_pixels = float(flow_patch_threshold_pixels)
        self.motion_translation_max = float(motion_translation_max)
        if self.motion_translation_max <= 0.0:
            raise ValueError("Motion translation bound must be positive.")

        self.state_token = nn.Parameter(torch.zeros(1, 1, latent_dim))
        self.dep_token = nn.Parameter(torch.zeros(1, 1, 1, latent_dim))
        self.temporal_embed = nn.Parameter(torch.zeros(1, TEMPORAL_WINDOW, 1, 1, latent_dim))

        self.state_norm = RMSNorm(latent_dim)
        self.dep_norm = RMSNorm(latent_dim)
        self.invariant_encoder_output_proj = nn.Linear(latent_dim, self.state_dim, bias=True)
        self.dependent_encoder_output_proj = nn.Linear(latent_dim, self.view_dim, bias=True)

        condition_dim = self.state_dim + self.view_dim if self.decoder_condition_mode == "concat" else self.state_dim
        self.spatial_queries = nn.Parameter(torch.zeros(1, self.n_tokens_per_frame, latent_dim))
        self.decoder_backbone = FiLMTokenTransformer(
            num_tokens=self.n_tokens_per_frame,
            embed_dim=latent_dim,
            condition_dim=condition_dim,
            depth=int(vit_cfg.get("decoder_depth", enc_cfg.depth)),
            num_heads=int(vit_cfg.get("decoder_num_heads", enc_cfg.num_heads)),
            mlp_ratio=float(vit_cfg.get("decoder_mlp_ratio", enc_cfg.mlp_ratio)),
            qkv_bias=bool(vit_cfg.get("qkv_bias", True)),
            dropout=float(vit_cfg.get("dropout", 0.0)),
            attn_dropout=float(vit_cfg.get("attn_dropout", 0.0)),
            drop_path_rate=float(vit_cfg.get("decoder_drop_path_rate", enc_cfg.drop_path_rate)),
            layerscale_init=float(vit_cfg.get("layerscale_init", 1e-5)),
            selected_layers=tuple(vit_cfg.get("decoder_selected_layers", enc_cfg.selected_layers)),
        )
        dpt_width = int(vit_cfg.get("dpt_features", dpt_features))
        self.decoder_dpt_backbone = DPTBackbone(
            in_dim=latent_dim,
            features=dpt_width,
            readout_type=str(vit_cfg.get("dpt_readout_type", "project")),
        )
        self.base_gaussian_head = DPTOutputHead(
            features=dpt_width,
            out_channels=self.splatter_channels,
        )
        self.dense_motion_head = DPTOutputHead(
            features=dpt_width,
            out_channels=self.gaussians_per_pixel * self.motion_params_per_gaussian,
        )
        nn.init.zeros_(self.dense_motion_head.final_conv.weight)
        nn.init.zeros_(self.dense_motion_head.final_conv.bias)

        nn.init.trunc_normal_(self.state_token, std=0.02)
        nn.init.trunc_normal_(self.dep_token, std=0.02)
        nn.init.trunc_normal_(self.temporal_embed, std=0.02)
        nn.init.trunc_normal_(self.spatial_queries, std=0.02)

    def _validate_pretraining_inputs(
        self, images: torch.Tensor, optical_flows: torch.Tensor
    ) -> None:
        if images.dim() != 6:
            raise ValueError(
                f"Expected pretraining images as (B,3,A,3,H,W), got {tuple(images.shape)}."
            )
        batch, timesteps, views, channels, height, width = images.shape
        expected_images = (batch, TEMPORAL_WINDOW, views, 3, self.img_height, self.img_width)
        if tuple(images.shape) != expected_images:
            raise ValueError(f"Expected pretraining images as {expected_images}, got {tuple(images.shape)}.")
        expected_flows = (batch, 3, views, 2, height, width)
        if tuple(optical_flows.shape) != expected_flows:
            raise ValueError(f"Expected optical flows as {expected_flows}, got {tuple(optical_flows.shape)}.")
        if not images.is_floating_point() or not optical_flows.is_floating_point():
            raise TypeError("Neural pretraining inputs must be floating-point GPU tensors.")

    def _sample_visible_patch_ids(
        self,
        leading_shape: Tuple[int, ...],
        num_patches: int,
        mask_ratio: float,
        device: torch.device,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        ratio = min(max(float(mask_ratio), 0.0), 0.99)
        keep_count = max(1, min(num_patches, int(round(num_patches * (1.0 - ratio)))))
        noise = torch.rand(*leading_shape, num_patches, device=device)
        ids_keep = torch.topk(
            noise,
            k=keep_count,
            dim=-1,
            largest=False,
            sorted=False,
        ).indices
        mask = torch.ones(*leading_shape, num_patches, device=device, dtype=torch.bool)
        mask.scatter_(dim=-1, index=ids_keep, value=False)
        return ids_keep, mask

    def flow_patch_scores(self, optical_flows: torch.Tensor) -> torch.Tensor:
        """Pool max pairwise flow magnitude to the ViT grid as ``(B,A,N)``."""
        if optical_flows.dim() != 6 or optical_flows.shape[1] != 3 or optical_flows.shape[3] != 2:
            raise ValueError(f"Expected optical flows as (B,3,A,2,H,W), got {tuple(optical_flows.shape)}.")
        batch, _pairs, num_views, _channels, height, width = optical_flows.shape
        flow_magnitude = optical_flows.float().square().sum(dim=3).clamp_min(0.0).sqrt()
        dynamic_pixels = flow_magnitude.amax(dim=1)
        pooled = F.max_pool2d(
            dynamic_pixels.reshape(batch * num_views, 1, height, width),
            kernel_size=(self.patch_h, self.patch_w),
            stride=(self.patch_h, self.patch_w),
        )
        return pooled.flatten(1).view(batch, num_views, self.n_tokens_per_frame).to(optical_flows.dtype)

    def _sample_dynamic_visible_patch_ids(
        self,
        patch_scores: torch.Tensor,
        mask_ratio: float,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Vectorized dynamic-priority selection for invariant tube masking."""
        num_patches = patch_scores.shape[-1]
        ratio = min(max(float(mask_ratio), 0.0), 0.99)
        keep_count = max(1, min(num_patches, int(round(num_patches * (1.0 - ratio)))))
        scores = patch_scores.detach()
        dynamic = scores > self.flow_patch_threshold_pixels
        random_priority = torch.rand_like(scores)
        dynamic_priority = 2.0 + scores
        priority = torch.where(dynamic, dynamic_priority, random_priority)
        ids_keep = torch.topk(
            priority,
            k=keep_count,
            dim=-1,
            largest=True,
            sorted=False,
        ).indices
        mask = torch.ones_like(patch_scores, dtype=torch.bool)
        mask.scatter_(dim=-1, index=ids_keep, value=False)
        return ids_keep, mask

    def _encode_invariant_branch(
        self,
        images: torch.Tensor,
        optical_flows: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        bsz, timesteps, num_views, channels, height, width = images.shape
        flat_images = images.reshape(bsz * timesteps * num_views, channels, height, width).contiguous()
        patch_tokens, _ = self.invariant_encoder.patch_embed(flat_images)
        patch_tokens = patch_tokens.view(bsz, timesteps, num_views, self.n_tokens_per_frame, -1)

        spatial = self.invariant_encoder.pos_embed[:, 1:].to(patch_tokens.dtype).view(1, 1, 1, self.n_tokens_per_frame, -1)
        temporal = self.temporal_embed[:, :timesteps].to(patch_tokens.dtype)
        patch_tokens = patch_tokens + spatial + temporal

        if self.tube_mask_per_view:
            ids_keep, inv_mask = self._sample_dynamic_visible_patch_ids(
                self.flow_patch_scores(optical_flows),
                self.inv_tube_mask_ratio,
            )
        else:
            dynamic_scores = self.flow_patch_scores(optical_flows)
            shared_scores = dynamic_scores.amax(dim=1, keepdim=True)
            ids_keep, shared_mask = self._sample_dynamic_visible_patch_ids(
                shared_scores,
                self.inv_tube_mask_ratio,
            )
            ids_keep = ids_keep.expand(bsz, num_views, -1).contiguous()
            inv_mask = shared_mask.expand(bsz, num_views, -1).contiguous()

        gather_index = ids_keep[:, None, :, :, None].expand(
            bsz,
            timesteps,
            num_views,
            ids_keep.shape[-1],
            patch_tokens.shape[-1],
        )
        visible_tokens = torch.gather(patch_tokens, dim=3, index=gather_index)
        visible_tokens = visible_tokens.reshape(bsz, timesteps * num_views * ids_keep.shape[-1], -1)

        state_token = self.state_token.to(visible_tokens.dtype).expand(bsz, -1, -1)
        tokens = torch.cat([state_token, visible_tokens], dim=1)
        tokens = self.invariant_encoder.pos_drop(tokens)

        for block in self.invariant_encoder.blocks:
            tokens = block(tokens)
        tokens = self.invariant_encoder.norm(tokens)
        s_inv = self.invariant_encoder_output_proj(self.state_norm(tokens[:, 0]))
        return s_inv.contiguous(), inv_mask

    def _encode_dependent_branch(self, first_timestep_images: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        bsz, num_views, channels, height, width = first_timestep_images.shape
        flat_images = first_timestep_images.reshape(bsz * num_views, channels, height, width).contiguous()
        patch_tokens, _ = self.dependent_encoder.patch_embed(flat_images)
        patch_tokens = patch_tokens.view(bsz, num_views, self.n_tokens_per_frame, -1)

        spatial = self.dependent_encoder.pos_embed[:, 1:].to(patch_tokens.dtype).view(1, 1, self.n_tokens_per_frame, -1)
        patch_tokens = patch_tokens + spatial

        mask_ratio = self.dep_mask_ratio
        if (not self.training) and (not self.dep_mask_eval):
            mask_ratio = 0.0
        ids_keep, dep_mask = self._sample_visible_patch_ids(
            (bsz, num_views),
            self.n_tokens_per_frame,
            mask_ratio,
            first_timestep_images.device,
        )
        gather_index = ids_keep[..., None].expand(bsz, num_views, ids_keep.shape[-1], patch_tokens.shape[-1])
        visible_tokens = torch.gather(patch_tokens, dim=2, index=gather_index)

        dep_token = self.dep_token.to(visible_tokens.dtype).expand(bsz, num_views, -1, -1)
        tokens = torch.cat([dep_token, visible_tokens], dim=2)
        tokens = tokens.reshape(bsz * num_views, 1 + ids_keep.shape[-1], -1)
        tokens = self.dependent_encoder.pos_drop(tokens)

        for block in self.dependent_encoder.blocks:
            tokens = block(tokens)
        tokens = self.dependent_encoder.norm(tokens)
        dep = self.dependent_encoder_output_proj(self.dep_norm(tokens[:, 0]))
        dep = dep.view(bsz, num_views, self.view_dim).contiguous()
        return dep, dep_mask

    def encode_pretraining(
        self,
        images: torch.Tensor,
        optical_flows: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """Encode the fixed three-frame training input with teacher-flow masking."""
        self._validate_pretraining_inputs(images, optical_flows)
        s_inv, inv_mask = self._encode_invariant_branch(images, optical_flows)
        z_dep_all, dep_mask = self._encode_dependent_branch(images[:, 0])
        return {
            "s_inv": s_inv,
            "z_dep_all": z_dep_all,
            "inv_mask": inv_mask,
            "dep_mask": dep_mask,
        }

    def _encode_invariant_all_patches(self, images: torch.Tensor) -> torch.Tensor:
        if images.dim() != 6:
            raise ValueError(f"Expected policy images as (B,3,A,3,H,W), got {tuple(images.shape)}.")
        batch, timesteps, views, channels, height, width = images.shape
        expected = (batch, TEMPORAL_WINDOW, views, 3, self.img_height, self.img_width)
        if tuple(images.shape) != expected:
            raise ValueError(f"Expected policy images as {expected}, got {tuple(images.shape)}.")
        flat = images.reshape(batch * timesteps * views, channels, height, width).contiguous()
        patch_tokens, _ = self.invariant_encoder.patch_embed(flat)
        patch_tokens = patch_tokens.view(
            batch, timesteps, views, self.n_tokens_per_frame, -1
        )
        spatial = self.invariant_encoder.pos_embed[:, 1:].to(patch_tokens.dtype).view(
            1, 1, 1, self.n_tokens_per_frame, -1
        )
        tokens = patch_tokens + spatial + self.temporal_embed.to(patch_tokens.dtype)
        tokens = tokens.reshape(batch, timesteps * views * self.n_tokens_per_frame, -1)
        tokens = torch.cat((self.state_token.to(tokens.dtype).expand(batch, -1, -1), tokens), dim=1)
        tokens = self.invariant_encoder.pos_drop(tokens)
        for block in self.invariant_encoder.blocks:
            tokens = block(tokens)
        tokens = self.invariant_encoder.norm(tokens)
        return self.invariant_encoder_output_proj(self.state_norm(tokens[:, 0])).contiguous()

    def _encode_dependent_all_patches(self, first_timestep_images: torch.Tensor) -> torch.Tensor:
        if first_timestep_images.dim() != 5:
            raise ValueError("Expected first-timestep images as (B,A,3,H,W).")
        batch, views, channels, height, width = first_timestep_images.shape
        flat = first_timestep_images.reshape(batch * views, channels, height, width).contiguous()
        patch_tokens, _ = self.dependent_encoder.patch_embed(flat)
        patch_tokens = patch_tokens + self.dependent_encoder.pos_embed[:, 1:].to(patch_tokens.dtype)
        tokens = torch.cat(
            (self.dep_token.to(patch_tokens.dtype).expand(batch, views, -1, -1).reshape(batch * views, 1, -1), patch_tokens),
            dim=1,
        )
        tokens = self.dependent_encoder.pos_drop(tokens)
        for block in self.dependent_encoder.blocks:
            tokens = block(tokens)
        tokens = self.dependent_encoder.norm(tokens)
        dep = self.dependent_encoder_output_proj(self.dep_norm(tokens[:, 0]))
        return dep.view(batch, views, self.view_dim).contiguous()

    @torch.no_grad()
    def inference_features(self, images: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Deterministic fixed-window inference with fixed CUDA BF16 neural forward."""
        context = (
            torch.autocast(device_type="cuda", dtype=torch.bfloat16)
            if images.device.type == "cuda"
            else nullcontext()
        )
        with context:
            s_inv = self._encode_invariant_all_patches(images)
            z_dep_all = self._encode_dependent_all_patches(images[:, 0])
        return {"s_inv": s_inv.float(), "z_dep_all": z_dep_all.float()}

    def _build_decoder_condition(self, s_inv: torch.Tensor, z_dep_source: torch.Tensor) -> torch.Tensor:
        if self.decoder_condition_mode == "concat":
            return torch.cat([s_inv, z_dep_source], dim=-1)
        if s_inv.shape != z_dep_source.shape:
            raise ValueError(
                f"decoder_condition_mode='add' requires matching runtime shapes, "
                f"got s_inv={tuple(s_inv.shape)} and z_dep_source={tuple(z_dep_source.shape)}."
            )
        return s_inv + z_dep_source

    def predict_raw_maps(
        self, s_inv: torch.Tensor, z_dep_source: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """Run the learned decoder and return unactivated neural output maps."""
        if s_inv.dim() != 2 or z_dep_source.dim() != 2:
            raise ValueError("Decoder state and source-view features must both be rank-two tensors.")
        condition = self._build_decoder_condition(s_inv, z_dep_source)
        queries = self.spatial_queries.to(s_inv.dtype).expand(s_inv.shape[0], -1, -1)
        _, hidden_states = self.decoder_backbone(queries, condition)
        shared_features = self.decoder_dpt_backbone(
            hidden_states=hidden_states, grid_size=self.grid_size
        )
        output_size = (self.img_height, self.img_width)
        return {
            "raw_base_map": self.base_gaussian_head(shared_features, output_size).contiguous(),
            "raw_motion_map": self.dense_motion_head(shared_features, output_size).contiguous(),
        }

    def decode_sequence(
        self, s_inv: torch.Tensor, z_dep_source: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """Explicit deterministic decoder interface for visualization utilities."""
        return self.predict_raw_maps(s_inv, z_dep_source)

    @torch.no_grad()
    def policy_state(self, images: torch.Tensor) -> torch.Tensor:
        """Return deterministic all-patch policy features with fixed CUDA BF16 forward."""
        context = (
            torch.autocast(device_type="cuda", dtype=torch.bfloat16)
            if images.device.type == "cuda"
            else nullcontext()
        )
        with context:
            state = self._encode_invariant_all_patches(images)
        return state.float()

    def save_checkpoint(self, checkpoint_file: str) -> None:
        torch.save(self.state_dict(), checkpoint_file)

    def load_checkpoint(self, checkpoint_file: str) -> None:
        state = torch.load(checkpoint_file, map_location="cpu")
        self.load_state_dict(state)
