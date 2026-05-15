from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Tuple

import torch
import torch.nn as nn

from .vision_transformer import DPTHead, TokenTransformer, ViTBackbone, ViTSmallConfig

@dataclass
class CodebookConfig:
    """Legacy codebook config kept for ReViWo compatibility."""

    # ----- VQ params -----
    n_embed: int = 512
    embed_dim: int = 64
    beta: float = 0.25

    # ----- quantizer selection -----
    quantizer: str = "vq"                     # "vq" or "fsq"
    fsq_levels: Tuple[int, ...] = field(default_factory=tuple)

class AttentionStatePooler(nn.Module):
    """Pool patch tokens into one compact state vector with a learned query."""

    def __init__(self, dim: int, num_heads: int, mlp_ratio: float = 2.0, dropout: float = 0.0):
        super().__init__()
        self.query = nn.Parameter(torch.zeros(1, 1, dim))
        self.token_norm = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        hidden_dim = int(round(dim * float(mlp_ratio)))
        self.out_norm = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
        )
        nn.init.trunc_normal_(self.query, std=0.02)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        bsz = tokens.shape[0]
        tokens = self.token_norm(tokens)
        query = self.query.expand(bsz, -1, -1)
        pooled, _ = self.attn(query=query, key=tokens, value=tokens, need_weights=False)
        pooled = pooled.squeeze(1)
        return self.out_norm(pooled + self.mlp(pooled))


class SplatterVAE(nn.Module):
    """Dual-branch continuous beta-VAE with a compact-state decoder.

    Notes:
        - Each branch pools ViT patch tokens into one Gaussian state vector.
        - state_dim controls the invariant branch; dep_state_dim optionally
          bottlenecks the dependent camera/view branch.
        - The decoder uses one fused compact vector to generate patch tokens.
        - The decoder outputs a *parent* Splatter Image only.
        - The parent image is later converted to Gaussians and expanded with
          target-conditioned child Gaussians.
    """

    def __init__(
        self,
        vit_cfg: Dict,
        img_height: int,
        img_width: int,
        splatter_channels: int,
        fusion_style: str = "cat",
        state_dim: int = 256,
        dep_state_dim: int | None = None,
        state_token_dim: int | None = None,
        state_pool_heads: int = 4,
        state_pool_mlp_ratio: float = 2.0,
        decoder_token_hidden_dim: int | None = None,
        dep_input_mask_ratio: float = 0.95,
        dep_mask_eval: bool = True,
        dpt_features: int = 256,
    ):
        super().__init__()

        # -------------------------------------------------------------
        # Image / patch setup
        # -------------------------------------------------------------
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
        latent_dim = enc_cfg.embed_dim
        self.latent_dim = int(latent_dim)
        self.inv_state_dim = int(state_dim)
        self.dep_state_dim = int(dep_state_dim or state_dim)
        # Keep state_dim as the invariant state size for existing RL/config code.
        self.state_dim = self.inv_state_dim
        state_token_dim = int(state_token_dim or state_dim)
        if state_token_dim % int(state_pool_heads) != 0:
            raise ValueError(
                f"state_token_dim={state_token_dim} must be divisible by state_pool_heads={state_pool_heads}."
            )

        # -------------------------------------------------------------
        # Project and pool encoder tokens into compact Gaussian states
        # -------------------------------------------------------------
        self.invariant_encoder_output_proj = nn.Sequential(
            nn.Linear(latent_dim, state_token_dim, bias=True),
            nn.LayerNorm(state_token_dim),
            nn.GELU(),
            nn.Linear(state_token_dim, state_token_dim, bias=True),
        )
        self.dependent_encoder_output_proj = nn.Sequential(
            nn.Linear(latent_dim, state_token_dim, bias=True),
            nn.LayerNorm(state_token_dim),
            nn.GELU(),
            nn.Linear(state_token_dim, state_token_dim, bias=True),
        )
        self.invariant_state_pool = AttentionStatePooler(
            dim=state_token_dim,
            num_heads=int(state_pool_heads),
            mlp_ratio=float(state_pool_mlp_ratio),
            dropout=float(vit_cfg.get("dropout", 0.0)),
        )
        self.dependent_state_pool = AttentionStatePooler(
            dim=state_token_dim,
            num_heads=int(state_pool_heads),
            mlp_ratio=float(state_pool_mlp_ratio),
            dropout=float(vit_cfg.get("dropout", 0.0)),
        )
        self.invariant_mu = nn.Linear(state_token_dim, self.inv_state_dim)
        self.invariant_logvar = nn.Linear(state_token_dim, self.inv_state_dim)
        self.dependent_mu = nn.Linear(state_token_dim, self.dep_state_dim)
        self.dependent_logvar = nn.Linear(state_token_dim, self.dep_state_dim)

        # -------------------------------------------------------------
        # Fusion of invariant + dependent compact states before the decoder
        # -------------------------------------------------------------
        self.fusion_style = fusion_style
        if fusion_style == "plus":
            if self.inv_state_dim != self.dep_state_dim:
                raise ValueError(
                    "fusion_style='plus' requires state_dim and dep_state_dim to match. "
                    "Use fusion_style='cat' when the dependent branch is smaller."
                )
            decoder_in_dim = self.inv_state_dim
        elif fusion_style == "cat":
            decoder_in_dim = self.inv_state_dim + self.dep_state_dim
        else:
            raise NotImplementedError(f"Unknown fusion_style={fusion_style}")

        self.decoder_state_mlp = nn.Sequential(
            nn.Linear(decoder_in_dim, latent_dim, bias=True),
            nn.LayerNorm(latent_dim),
            nn.GELU(),
            nn.Linear(latent_dim, latent_dim, bias=True),
            nn.LayerNorm(latent_dim),
        )
        self.decoder_patch_query = nn.Parameter(torch.zeros(1, self.n_tokens_per_frame, latent_dim))
        self.decoder_2d_pos = nn.Parameter(torch.zeros(1, self.n_tokens_per_frame, latent_dim))
        decoder_token_hidden_dim = int(decoder_token_hidden_dim or (2 * latent_dim))
        self.decoder_token_mlp = nn.Sequential(
            nn.Linear(2 * latent_dim, decoder_token_hidden_dim, bias=True),
            nn.GELU(),
            nn.Linear(decoder_token_hidden_dim, latent_dim, bias=True),
            nn.LayerNorm(latent_dim),
        )
        nn.init.trunc_normal_(self.decoder_patch_query, std=0.02)
        nn.init.trunc_normal_(self.decoder_2d_pos, std=0.02)

        # -------------------------------------------------------------
        # Token transformer + DPT dense head
        # -------------------------------------------------------------
        self.decoder_backbone = TokenTransformer(
            num_tokens=self.n_tokens_per_frame,
            embed_dim=latent_dim,
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
            out_channels=int(splatter_channels),
            readout_type=str(vit_cfg.get("dpt_readout_type", "project")),
        )

        # -------------------------------------------------------------
        # Misc flags / helper tokens
        # -------------------------------------------------------------
        self.splatter_channels = int(splatter_channels)

        # Patch-aligned random masking for the dependent branch input
        self.dep_input_mask_ratio = float(dep_input_mask_ratio)
        self.dep_mask_eval = bool(dep_mask_eval)
        self.dep_mask_token = nn.Parameter(torch.zeros(1, 1, 1, 3, self.patch_h, self.patch_w))
        nn.init.normal_(self.dep_mask_token, mean=0.0, std=0.02)

    # ------------------------------------------------------------------
    # Input masking for the dependent branch
    # ------------------------------------------------------------------
    def _mask_dependent_input_patches(self, x: torch.Tensor) -> torch.Tensor:
        """Patch-aligned random masking on raw pixels for the dependent branch.

        The view-dependent branch is encouraged to learn compact, view-specific
        information instead of simply copying the RGB input.
        """
        if self.dep_input_mask_ratio <= 0.0:
            return x
        if (not self.training) and (not self.dep_mask_eval):
            return x

        b, c, h, w = x.shape
        gh = h // self.patch_h
        gw = w // self.patch_w

        x_patches = x.view(b, c, gh, self.patch_h, gw, self.patch_w)
        x_patches = x_patches.permute(0, 2, 4, 1, 3, 5).contiguous()

        keep = (torch.rand(b, gh, gw, 1, 1, 1, device=x.device) >= self.dep_input_mask_ratio).to(dtype=x.dtype)
        mask_token = self.dep_mask_token.to(dtype=x.dtype).expand(b, gh, gw, -1, -1, -1)
        x_patches = x_patches * keep + mask_token * (1.0 - keep)

        x_masked = x_patches.permute(0, 3, 1, 4, 2, 5).contiguous()
        x_masked = x_masked.view(b, c, h, w)
        return x_masked

    # ------------------------------------------------------------------
    # Encoding
    # ------------------------------------------------------------------
    @staticmethod
    def _kl_loss(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        logvar = logvar.clamp(-20.0, 6.0)
        kl = -0.5 * (1.0 + logvar - mu.pow(2) - logvar.exp())
        return kl.sum(dim=-1).mean()

    @staticmethod
    def _sample_state(mu: torch.Tensor, logvar: torch.Tensor, deterministic: bool, training: bool) -> torch.Tensor:
        if deterministic or not training:
            return mu
        std = torch.exp(0.5 * logvar.clamp(-20.0, 6.0))
        eps = torch.randn_like(std)
        return mu + eps * std

    def _check_input_size(self, x: torch.Tensor) -> None:
        _b, _c, h, w = x.shape
        if h != self.img_height or w != self.img_width:
            raise ValueError(f"Expected input size {(self.img_height, self.img_width)}, got {(h, w)}")

    def encode_invariant_pooled_state(self, x: torch.Tensor) -> torch.Tensor:
        """Return the invariant branch output immediately after attention pooling."""
        self._check_input_size(x)
        h_inv_tokens, _, _ = self.invariant_encoder(x)
        h_inv = self.invariant_encoder_output_proj(h_inv_tokens)
        return self.invariant_state_pool(h_inv).contiguous()

    def encode_dependent_pooled_state(self, x: torch.Tensor) -> torch.Tensor:
        """Return the dependent branch output immediately after attention pooling."""
        self._check_input_size(x)
        x_dep_masked = self._mask_dependent_input_patches(x)
        h_dep_tokens, _, _ = self.dependent_encoder(x_dep_masked)
        h_dep = self.dependent_encoder_output_proj(h_dep_tokens)
        return self.dependent_state_pool(h_dep).contiguous()

    def encode_invariant_state(self, x: torch.Tensor, deterministic: bool = True) -> torch.Tensor:
        """Encode only the invariant branch into the Gaussian state used by the decoder."""
        pooled = self.encode_invariant_pooled_state(x)
        mu = self.invariant_mu(pooled)
        logvar = self.invariant_logvar(pooled)
        return self._sample_state(mu, logvar, deterministic=deterministic, training=self.training).contiguous()

    def encode_dependent_state(self, x: torch.Tensor, deterministic: bool = True) -> torch.Tensor:
        """Encode only the dependent branch into the Gaussian state used by the decoder."""
        pooled = self.encode_dependent_pooled_state(x)
        mu = self.dependent_mu(pooled)
        logvar = self.dependent_logvar(pooled)
        return self._sample_state(mu, logvar, deterministic=deterministic, training=self.training).contiguous()

    def encode(
        self,
        x: torch.Tensor,
        deterministic_invariant: bool = False,
        deterministic_dependent: bool = False,
    ):
        """Encode one RGB image into invariant + dependent compact Gaussian states."""
        self._check_input_size(x)

        # Invariant branch sees the original image.
        h_inv_tokens, _, _ = self.invariant_encoder(x)

        # Dependent branch sees the masked image.
        x_dep_masked = self._mask_dependent_input_patches(x)
        h_dep_tokens, _, _ = self.dependent_encoder(x_dep_masked)

        if h_inv_tokens.shape[1] != self.n_tokens_per_frame:
            raise ValueError("Unexpected token count from invariant encoder.")

        h_inv = self.invariant_encoder_output_proj(h_inv_tokens)
        h_dep = self.dependent_encoder_output_proj(h_dep_tokens)
        inv_pooled = self.invariant_state_pool(h_inv)
        dep_pooled = self.dependent_state_pool(h_dep)

        z_inv_mu = self.invariant_mu(inv_pooled)
        z_inv_logvar = self.invariant_logvar(inv_pooled)
        z_dep_mu = self.dependent_mu(dep_pooled)
        z_dep_logvar = self.dependent_logvar(dep_pooled)

        z_inv = self._sample_state(
            z_inv_mu,
            z_inv_logvar,
            deterministic=deterministic_invariant,
            training=self.training,
        )
        z_dep = self._sample_state(
            z_dep_mu,
            z_dep_logvar,
            deterministic=deterministic_dependent,
            training=self.training,
        )

        inv_kl_loss = self._kl_loss(z_inv_mu, z_inv_logvar)
        dep_kl_loss = self._kl_loss(z_dep_mu, z_dep_logvar)
        stats = {
            "z_inv_mu": z_inv_mu,
            "z_inv_logvar": z_inv_logvar,
            "z_dep_mu": z_dep_mu,
            "z_dep_logvar": z_dep_logvar,
        }
        return z_inv.contiguous(), inv_kl_loss, z_dep.contiguous(), dep_kl_loss, stats

    # ------------------------------------------------------------------
    # Fusion
    # ------------------------------------------------------------------
    def fusion(self, z_inv: torch.Tensor, z_dep: torch.Tensor) -> torch.Tensor:
        if self.fusion_style == "plus":
            return z_inv + z_dep
        if self.fusion_style == "cat":
            return torch.cat([z_inv, z_dep], dim=-1)
        raise NotImplementedError(f"Unknown fusion_style={self.fusion_style}")

    # ------------------------------------------------------------------
    # Decoding
    # ------------------------------------------------------------------
    def decode(self, z_inv: torch.Tensor, z_dep: torch.Tensor) -> torch.Tensor:
        """Decode one fused compact state into a parent-only Splatter Image."""
        fused_state = self.fusion(z_inv, z_dep)
        state_embed = self.decoder_state_mlp(fused_state)

        bsz = state_embed.shape[0]
        patch_seed = (self.decoder_patch_query + self.decoder_2d_pos).expand(bsz, -1, -1)
        state_tokens = state_embed.unsqueeze(1).expand(-1, self.n_tokens_per_frame, -1)
        dec_tokens = self.decoder_token_mlp(torch.cat((state_tokens, patch_seed), dim=-1))
        _, hidden_states = self.decoder_backbone(dec_tokens)

        grid_h = self.img_height // self.patch_h
        grid_w = self.img_width // self.patch_w
        splatter = self.decoder(
            hidden_states=hidden_states,
            grid_size=(grid_h, grid_w),
            output_size=(self.img_height, self.img_width),
        )
        return splatter.contiguous()

    # ------------------------------------------------------------------
    # Convenience forward
    # ------------------------------------------------------------------
    def forward(self, x: torch.Tensor):
        z_inv, inv_kl_loss, z_dep, dep_kl_loss, _ = self.encode(
            x,
            deterministic_invariant=False,
            deterministic_dependent=False,
        )
        splatter = self.decode(z_inv, z_dep)
        total_kl_loss = inv_kl_loss + dep_kl_loss
        return splatter, total_kl_loss

    # ------------------------------------------------------------------
    # Checkpoint helpers
    # ------------------------------------------------------------------
    def save_checkpoint(self, checkpoint_file: str):
        torch.save(self.state_dict(), checkpoint_file)

    def load_checkpoint(self, checkpoint_file: str):
        state = torch.load(checkpoint_file, map_location="cpu")
        self.load_state_dict(state)
