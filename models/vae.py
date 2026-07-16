from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn

from .vision_transformer import DPTHead, RMSNorm, TransformerBlock, ViTBackbone, ViTSmallConfig


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
        cls_tokens = self.cls_token.expand(batch, -1, -1)
        x = torch.cat([cls_tokens, tokens], dim=1)
        x = self.pos_drop(x + self.pos_embed)

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
    """Joint spatiotemporal multi-view SplatterVAE.

    The invariant branch consumes a 3-frame multi-view sequence and returns one
    state vector ``s_inv``. The dependent branch consumes only the first
    timestep and returns per-view anchor vectors. The decoder reconstructs dense
    Gaussian maps from learnable spatial queries modulated by FiLM from
    either concat(s_inv, z_dep_source) or s_inv + z_dep_source.
    """

    def __init__(
        self,
        vit_cfg: Dict,
        img_height: int,
        img_width: int,
        splatter_channels: int,
        dep_mask_eval: bool = True,
        dpt_features: int = 256,
        temporal_window: int = 3,
        inv_tube_mask_ratio: float = 0.50,
        dep_mask_ratio: float = 0.75,
        tube_mask_per_view: bool = True,
        state_dim: int = 256,
        view_dim: Optional[int] = None,
        use_single_state_vector: bool = True,
        dependent_uses_first_timestep_only: bool = True,
        use_temporal_delta_decoder: bool = True,
        decoder_condition_mode: str = "concat",
        gaussians_per_pixel: int = 1,
        delta_xyz_scale: float = 0.05,
        **_: object,
    ):
        super().__init__()
        if not bool(use_single_state_vector):
            raise ValueError("The migrated SplatterVAE requires use_single_state_vector=true.")
        if not bool(dependent_uses_first_timestep_only):
            raise ValueError("The migrated SplatterVAE requires dependent_uses_first_timestep_only=true.")
        if not bool(use_temporal_delta_decoder):
            raise ValueError("The migrated SplatterVAE requires use_temporal_delta_decoder=true.")

        self.img_height = int(img_height)
        self.img_width = int(img_width)
        self.temporal_window = int(temporal_window)
        if self.temporal_window <= 0:
            raise ValueError(f"temporal_window must be positive, got {temporal_window}.")

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
        self.delta_channels = 3 * self.gaussians_per_pixel
        self.delta_xyz_scale = float(delta_xyz_scale)

        self.state_token = nn.Parameter(torch.zeros(1, 1, latent_dim))
        self.dep_token = nn.Parameter(torch.zeros(1, 1, 1, latent_dim))
        self.temporal_embed = nn.Parameter(torch.zeros(1, self.temporal_window, 1, 1, latent_dim))

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
        self.decoder = DPTHead(
            in_dim=latent_dim,
            features=int(vit_cfg.get("dpt_features", dpt_features)),
            out_channels=self.splatter_channels + 2 * self.delta_channels,
            readout_type=str(vit_cfg.get("dpt_readout_type", "project")),
        )

        nn.init.trunc_normal_(self.state_token, std=0.02)
        nn.init.trunc_normal_(self.dep_token, std=0.02)
        nn.init.trunc_normal_(self.temporal_embed, std=0.02)
        nn.init.trunc_normal_(self.spatial_queries, std=0.02)

    def _zero_loss(self, reference: torch.Tensor) -> torch.Tensor:
        return reference.new_zeros(())

    def _coerce_sequence(self, images: torch.Tensor) -> torch.Tensor:
        if images.dim() == 4:
            images = images[:, None, None].expand(-1, self.temporal_window, 1, -1, -1, -1).contiguous()
        elif images.dim() == 5:
            images = images[:, None].expand(-1, self.temporal_window, -1, -1, -1, -1).contiguous()
        elif images.dim() != 6:
            raise ValueError(
                f"Expected images as (B,T,A,3,H,W), (B,A,3,H,W), or (B,3,H,W), got {tuple(images.shape)}."
            )

        _bsz, timesteps, num_views, channels, height, width = images.shape
        if channels != 3:
            raise ValueError(f"SplatterVAE expects RGB inputs, got {channels} channels.")
        if height != self.img_height or width != self.img_width:
            raise ValueError(f"Expected image size {(self.img_height, self.img_width)}, got {(height, width)}.")
        if timesteps > self.temporal_window:
            images = images[:, : self.temporal_window].contiguous()
        elif timesteps < self.temporal_window:
            pad = images[:, -1:].expand(-1, self.temporal_window - timesteps, -1, -1, -1, -1)
            images = torch.cat([images, pad], dim=1).contiguous()
        return images

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
        ids_keep = torch.argsort(noise, dim=-1)[..., :keep_count]
        mask = torch.ones(*leading_shape, num_patches, device=device, dtype=torch.bool)
        mask.scatter_(dim=-1, index=ids_keep, value=False)
        return ids_keep, mask

    def _encode_invariant_branch(self, images: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        bsz, timesteps, num_views, channels, height, width = images.shape
        flat_images = images.reshape(bsz * timesteps * num_views, channels, height, width).contiguous()
        patch_tokens, _ = self.invariant_encoder.patch_embed(flat_images)
        patch_tokens = patch_tokens.view(bsz, timesteps, num_views, self.n_tokens_per_frame, -1)

        spatial = self.invariant_encoder.pos_embed[:, 1:].view(1, 1, 1, self.n_tokens_per_frame, -1)
        temporal = self.temporal_embed[:, :timesteps]
        patch_tokens = patch_tokens + spatial + temporal

        if self.tube_mask_per_view:
            ids_keep, inv_mask = self._sample_visible_patch_ids(
                (bsz, num_views),
                self.n_tokens_per_frame,
                self.inv_tube_mask_ratio,
                images.device,
            )
        else:
            ids_keep, shared_mask = self._sample_visible_patch_ids(
                (bsz, 1),
                self.n_tokens_per_frame,
                self.inv_tube_mask_ratio,
                images.device,
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

        state_token = self.state_token.expand(bsz, -1, -1)
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

        spatial = self.dependent_encoder.pos_embed[:, 1:].view(1, 1, self.n_tokens_per_frame, -1)
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

        dep_token = self.dep_token.expand(bsz, num_views, -1, -1)
        tokens = torch.cat([dep_token, visible_tokens], dim=2)
        tokens = tokens.reshape(bsz * num_views, 1 + ids_keep.shape[-1], -1)
        tokens = self.dependent_encoder.pos_drop(tokens)

        for block in self.dependent_encoder.blocks:
            tokens = block(tokens)
        tokens = self.dependent_encoder.norm(tokens)
        dep = self.dependent_encoder_output_proj(self.dep_norm(tokens[:, 0]))
        dep = dep.view(bsz, num_views, self.view_dim).contiguous()
        return dep, dep_mask

    def encode_sequence(
        self,
        images: torch.Tensor,
        source_indices: Optional[torch.Tensor] = None,
    ) -> Tuple[Dict[str, torch.Tensor], torch.Tensor, torch.Tensor]:
        """Encode a multi-view temporal sequence into state and view anchors."""
        images = self._coerce_sequence(images)
        s_inv, inv_mask = self._encode_invariant_branch(images)
        z_dep_all, dep_mask = self._encode_dependent_branch(images[:, 0])

        latents: Dict[str, torch.Tensor] = {
            "s_inv": s_inv,
            "z_dep_all": z_dep_all,
            "inv_mask": inv_mask,
            "dep_mask": dep_mask,
        }
        if source_indices is not None:
            batch_ids = torch.arange(s_inv.shape[0], device=s_inv.device)
            latents["z_dep_source"] = z_dep_all[batch_ids, source_indices.to(device=s_inv.device)]

        zero = self._zero_loss(s_inv)
        return latents, zero, zero

    def encode(
        self,
        x: torch.Tensor,
        deterministic_invariant: bool = False,
        deterministic_dependent: bool = False,
    ):
        """Compatibility wrapper returning the single state and first-view anchor."""
        del deterministic_invariant, deterministic_dependent
        latents, inv_loss, dep_loss = self.encode_sequence(x)
        s_inv = latents["s_inv"]
        z_dep = latents["z_dep_all"][:, 0]
        indices = (
            torch.zeros((s_inv.shape[0], 1), device=s_inv.device, dtype=torch.long),
            torch.zeros((s_inv.shape[0], 1), device=s_inv.device, dtype=torch.long),
        )
        return s_inv, inv_loss, z_dep, dep_loss, indices

    def _build_decoder_condition(self, s_inv: torch.Tensor, z_dep_source: torch.Tensor) -> torch.Tensor:
        if self.decoder_condition_mode == "concat":
            return torch.cat([s_inv, z_dep_source], dim=-1)
        if s_inv.shape != z_dep_source.shape:
            raise ValueError(
                f"decoder_condition_mode='add' requires matching runtime shapes, "
                f"got s_inv={tuple(s_inv.shape)} and z_dep_source={tuple(z_dep_source.shape)}."
            )
        return s_inv + z_dep_source

    def decode_sequence(self, s_inv: torch.Tensor, z_dep_source: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Decode state and first-frame view anchor into base and delta maps."""
        if s_inv.dim() != 2:
            s_inv = s_inv.flatten(1)
        if z_dep_source.dim() != 2:
            z_dep_source = z_dep_source.flatten(1)
        condition = self._build_decoder_condition(s_inv, z_dep_source)
        queries = self.spatial_queries.expand(s_inv.shape[0], -1, -1)
        _, hidden_states = self.decoder_backbone(queries, condition)
        dense = self.decoder(
            hidden_states=hidden_states,
            grid_size=self.grid_size,
            output_size=(self.img_height, self.img_width),
        ).contiguous()

        base_map = dense[:, : self.splatter_channels]
        delta01_raw = dense[:, self.splatter_channels : self.splatter_channels + self.delta_channels]
        delta12_raw = dense[:, self.splatter_channels + self.delta_channels :]
        delta01_map = torch.tanh(delta01_raw) * self.delta_xyz_scale
        delta12_map = torch.tanh(delta12_raw) * self.delta_xyz_scale
        return {
            "base_map": base_map.contiguous(),
            "delta01_map": delta01_map.contiguous(),
            "delta12_map": delta12_map.contiguous(),
        }

    def decode(self, z_inv: torch.Tensor, z_dep: torch.Tensor) -> torch.Tensor:
        """Compatibility wrapper returning only the t0/base splatter map."""
        if z_inv.dim() > 2:
            z_inv = z_inv.mean(dim=1)
        if z_dep.dim() > 2:
            z_dep = z_dep.mean(dim=1)
        return self.decode_sequence(z_inv, z_dep)["base_map"]

    def forward(self, x: torch.Tensor):
        latents, inv_loss, dep_loss = self.encode_sequence(x)
        z_dep_source = latents["z_dep_all"][:, 0]
        decoded = self.decode_sequence(latents["s_inv"], z_dep_source)
        return decoded, inv_loss + dep_loss

    @torch.no_grad()
    def policy_state(self, images: torch.Tensor) -> torch.Tensor:
        """Return only the downstream policy representation ``s_inv``."""
        latents, _, _ = self.encode_sequence(images)
        return latents["s_inv"]

    def save_checkpoint(self, checkpoint_file: str) -> None:
        torch.save(self.state_dict(), checkpoint_file)

    def load_checkpoint(self, checkpoint_file: str) -> None:
        state = torch.load(checkpoint_file, map_location="cpu")
        self.load_state_dict(state)
