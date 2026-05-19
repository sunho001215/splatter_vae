from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Tuple

import torch
import torch.nn as nn

from vector_quantize_pytorch import FSQ, VectorQuantize

from .vision_transformer import DPTHead, TokenTransformer, ViTBackbone, ViTSmallConfig

@dataclass
class CodebookConfig:
    """Codebook / quantizer hyper-parameters used by the invariant / dependent branches."""

    # ----- VQ params -----
    n_embed: int = 512
    embed_dim: int = 64
    beta: float = 0.25

    # ----- quantizer selection -----
    quantizer: str = "vq"                     # "vq" or "fsq"
    fsq_levels: Tuple[int, ...] = field(default_factory=tuple)

    def __post_init__(self):
        self.quantizer = str(self.quantizer).lower()
        self.n_embed = int(self.n_embed)
        self.embed_dim = int(self.embed_dim)
        self.beta = float(self.beta)
        self.fsq_levels = tuple(int(level) for level in self.fsq_levels)

        if self.quantizer not in {"vq", "fsq"}:
            raise ValueError(f"Unknown quantizer type: {self.quantizer}")
        if self.embed_dim <= 0:
            raise ValueError(f"embed_dim must be positive, got {self.embed_dim}")
        if self.quantizer == "vq":
            if self.n_embed <= 0:
                raise ValueError(f"n_embed must be positive for VQ, got {self.n_embed}")
            if self.beta < 0:
                raise ValueError(f"beta must be non-negative for VQ, got {self.beta}")
        elif len(self.fsq_levels) == 0:
            raise ValueError("FSQ selected, but fsq_levels is empty.")

class SplatterVAE(nn.Module):
    """ReViWo-style dual-encoder VAE with a ViT+DPT generator.

    Notes:
        - The encode/decode interface mirrors the original implementation.
        - The decoder outputs a *parent* Splatter Image only.
        - The parent image is later converted to Gaussians and expanded with
          target-conditioned child Gaussians.
    """

    def __init__(
        self,
        vit_cfg: Dict,
        invariant_cb_config: CodebookConfig,
        dependent_cb_config: CodebookConfig,
        img_height: int,
        img_width: int,
        splatter_channels: int,
        fusion_style: str = "cat",
        use_dependent_vq: bool = True,
        is_dependent_ae: bool = True,
        use_invariant_vq: bool = True,
        is_invariant_ae: bool = True,
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

        # -------------------------------------------------------------
        # Project encoder outputs into codebook spaces
        # -------------------------------------------------------------
        self.invariant_encoder_output_proj = nn.Linear(latent_dim, invariant_cb_config.embed_dim, bias=True)
        self.dependent_encoder_output_proj = nn.Linear(latent_dim, dependent_cb_config.embed_dim, bias=True)

        # -------------------------------------------------------------
        # Fusion of invariant + dependent embeddings before the decoder
        # -------------------------------------------------------------
        self.fusion_style = fusion_style
        if fusion_style == "plus":
            if invariant_cb_config.embed_dim != dependent_cb_config.embed_dim:
                raise ValueError("fusion_style='plus' requires equal invariant/dependent dims.")
            decoder_in_dim = invariant_cb_config.embed_dim
        elif fusion_style == "cat":
            decoder_in_dim = invariant_cb_config.embed_dim + dependent_cb_config.embed_dim
        else:
            raise NotImplementedError(f"Unknown fusion_style={fusion_style}")

        self.decoder_input_proj = nn.Linear(decoder_in_dim, latent_dim, bias=True)

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
        # VQ or Gaussian latent heads
        # -------------------------------------------------------------
        # Keep track of which discrete quantizer each branch uses
        self.invariant_quantizer_type = str(invariant_cb_config.quantizer).lower()
        self.dependent_quantizer_type = str(dependent_cb_config.quantizer).lower()
        # Build the appropriate quantizer heads based on config:
        self.invariant_output_head = self._build_token_quantizer(
            invariant_cb_config,
            use_discrete_quantizer=use_invariant_vq,
        )
        self.dependent_output_head = self._build_token_quantizer(
            dependent_cb_config,
            use_discrete_quantizer=use_dependent_vq,
        )
        self.dependent_output_final_proj = nn.Identity()

        # -------------------------------------------------------------
        # Misc flags / helper tokens
        # -------------------------------------------------------------
        self.splatter_channels = int(splatter_channels)
        self.use_dependent_vq = bool(use_dependent_vq)
        self.is_dependent_ae = bool(is_dependent_ae)
        self.use_invariant_vq = bool(use_invariant_vq)
        self.is_invariant_ae = bool(is_invariant_ae)

        # Patch-aligned random masking for the dependent branch input
        self.dep_input_mask_ratio = float(dep_input_mask_ratio)
        self.dep_mask_eval = bool(dep_mask_eval)
        self.dep_mask_token = nn.Parameter(torch.zeros(1, 1, 1, 3, self.patch_h, self.patch_w))
        nn.init.normal_(self.dep_mask_token, mean=0.0, std=0.02)

    # ------------------------------------------------------------------
    # Quantizer builders and runners
    # ------------------------------------------------------------------

    def _build_token_quantizer(self, cb_config: CodebookConfig, use_discrete_quantizer: bool) -> nn.Module:
        """
        Build either:
        - VQ head
        - FSQ head
        - Gaussian head (if use_discrete_quantizer=False)

        Safe FSQ path here follows lucidrains' README usage:
        FSQ(levels=[...]) with input last-dim == len(levels)
        """
        if not use_discrete_quantizer:
            # Original Gaussian / AE path
            return nn.Linear(cb_config.embed_dim, 2 * cb_config.embed_dim)

        quantizer = str(cb_config.quantizer).lower()

        if quantizer == "vq":
            return VectorQuantize(
                dim=cb_config.embed_dim,
                codebook_size=cb_config.n_embed,
                commitment_weight=cb_config.beta,
                kmeans_init=True,
                kmeans_iters=10,
                threshold_ema_dead_code=2,
            )

        if quantizer == "fsq":
            if len(cb_config.fsq_levels) == 0:
                raise ValueError("FSQ selected, but fsq_levels is empty.")

            # Safe / explicit choice:
            # for this patch, make embed_dim exactly equal to len(levels)
            # e.g. L = [6, 5, 5]  ->  embed_dim = 3
            if cb_config.embed_dim != len(cb_config.fsq_levels):
                raise ValueError(
                    f"FSQ requires embed_dim == len(fsq_levels) in this implementation. "
                    f"Got embed_dim={cb_config.embed_dim}, len(fsq_levels)={len(cb_config.fsq_levels)}."
                )

            return FSQ(levels=list(cb_config.fsq_levels))

        raise ValueError(f"Unknown quantizer type: {cb_config.quantizer}")


    def _run_discrete_quantizer(
        self,
        quantizer_module: nn.Module,
        x: torch.Tensor,
        quantizer_type: str,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Normalize the interface across VQ and FSQ.

        Returns:
            z_q      : quantized tokens
            indices  : token indices
            aux_loss : scalar tensor

        Notes:
        - VQ returns (quantized, indices, aux_loss)
        - FSQ returns (quantized, indices)
        - For FSQ, aux_loss is zero by design in this wrapper
        """
        quantizer_type = str(quantizer_type).lower()

        if quantizer_type == "vq":
            z_q, indices, aux_loss = quantizer_module(x)
            return z_q, indices, aux_loss.mean()

        if quantizer_type == "fsq":
            z_q, indices = quantizer_module(x)
            aux_loss = x.new_zeros(())
            return z_q, indices, aux_loss

        raise ValueError(f"Unknown quantizer type: {quantizer_type}")

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
    def encode(
        self,
        x: torch.Tensor,
        deterministic_invariant: bool = False,
        deterministic_dependent: bool = False,
    ):
        """Encode one RGB image into invariant + dependent token sequences."""
        b, c, h, w = x.shape
        if h != self.img_height or w != self.img_width:
            raise ValueError(
                f"Expected input size {(self.img_height, self.img_width)}, got {(h, w)}"
            )

        # Invariant branch sees the original image.
        h_inv_tokens, _, _ = self.invariant_encoder(x)

        # Dependent branch sees the masked image.
        x_dep_masked = self._mask_dependent_input_patches(x)
        h_dep_tokens, _, _ = self.dependent_encoder(x_dep_masked)

        if h_inv_tokens.shape[1] != self.n_tokens_per_frame:
            raise ValueError("Unexpected token count from invariant encoder.")

        h_inv = self.invariant_encoder_output_proj(h_inv_tokens)
        h_dep = self.dependent_encoder_output_proj(h_dep_tokens)

        # Invariant branch.
        if self.use_invariant_vq:
            z_inv, invariant_encoding_indices, inv_embed_loss = self._run_discrete_quantizer(
                self.invariant_output_head,
                h_inv,
                self.invariant_quantizer_type,
            )
        else:
            z_inv_output = self.invariant_output_head(h_inv)
            z_inv_mu = z_inv_output[:, :, : h_inv.shape[-1]]
            z_inv_sigma = torch.exp(z_inv_output[:, :, h_inv.shape[-1] :].clamp(-20, 2))
            inv_embed_loss = -0.5 * torch.mean(
                1 + torch.log(z_inv_sigma ** 2) - z_inv_mu ** 2 - z_inv_sigma ** 2
            )
            dist = torch.distributions.Normal(z_inv_mu, z_inv_sigma)
            z_inv = z_inv_mu if deterministic_invariant else (dist.rsample() if self.training else dist.sample())
            invariant_encoding_indices = torch.zeros(
                (z_inv.shape[0], z_inv.shape[1]),
                device=z_inv.device,
                dtype=torch.long,
            )

        # Dependent branch.
        if self.use_dependent_vq:
            z_dep, dependent_encoding_indices, dep_embed_loss = self._run_discrete_quantizer(
                self.dependent_output_head,
                h_dep,
                self.dependent_quantizer_type,
            )
            z_dep = self.dependent_output_final_proj(z_dep)
        else:
            z_dep_output = self.dependent_output_head(h_dep)
            z_dep_mu = z_dep_output[:, :, : h_dep.shape[-1]]
            z_dep_sigma = torch.exp(z_dep_output[:, :, h_dep.shape[-1] :].clamp(-20, 2))
            dep_embed_loss = -0.5 * torch.mean(
                1 + torch.log(z_dep_sigma ** 2) - z_dep_mu ** 2 - z_dep_sigma ** 2
            )
            dist = torch.distributions.Normal(z_dep_mu, z_dep_sigma)
            z_dep = z_dep_mu if deterministic_dependent else (dist.rsample() if self.training else dist.sample())
            dependent_encoding_indices = torch.zeros(
                (z_dep.shape[0], z_dep.shape[1]),
                device=z_dep.device,
                dtype=torch.long,
            )

        return z_inv, inv_embed_loss, z_dep, dep_embed_loss, (
            dependent_encoding_indices,
            invariant_encoding_indices,
        )

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
        """Decode latent tokens into the vanilla (parent-only) Splatter Image."""
        quant = self.fusion(z_inv, z_dep)
        dec_tokens = self.decoder_input_proj(quant)
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
        z_inv, inv_embed_loss, z_dep, dep_embed_loss, _ = self.encode(
            x,
            deterministic_invariant=False,
            deterministic_dependent=False,
        )
        splatter = self.decode(z_inv, z_dep)
        total_embed_loss = inv_embed_loss + dep_embed_loss
        return splatter, total_embed_loss

    # ------------------------------------------------------------------
    # Checkpoint helpers
    # ------------------------------------------------------------------
    def save_checkpoint(self, checkpoint_file: str):
        torch.save(self.state_dict(), checkpoint_file)

    def load_checkpoint(self, checkpoint_file: str):
        state = torch.load(checkpoint_file, map_location="cpu")
        self.load_state_dict(state)
        for head in [self.invariant_output_head, self.dependent_output_head]:
            if isinstance(head, VectorQuantize) and hasattr(head, "kmeans_init"):
                head.kmeans_init = False