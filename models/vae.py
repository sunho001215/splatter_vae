import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from einops.layers.torch import Rearrange
from dataclasses import dataclass

from vector_quantize_pytorch import VectorQuantize
from .swin_transformer import SwinTransformerV2, SwinTransformerV2Decoder

# -------------------------------------------------------------------------
# Small configs to mimic ReViWo's ViT-style encoders/decoder
# -------------------------------------------------------------------------

@dataclass
class CodebookConfig:
    """
    Simple container for VQ codebook hyperparameters.
    Matches ReViWo: codebook size 512, embedding dim 64.
    """
    n_embed: int = 512   # codebook size (K)
    embed_dim: int = 64  # embedding dim (D)
    beta: float = 0.25     # commitment cost

# -----------------------------------------------------------------------------
# Invariant/Dependent VAE that outputs Splatter Image instead of RGB
# -----------------------------------------------------------------------------

class InvariantDependentSplatterVAE(nn.Module):
    """
    ReViWo-style multi-view VAE, but:

      - encoders are named invariant_encoder and dependent_encoder
      - works with non-square images (img_height, img_width)
      - decoder outputs a Splatter Image (per-pixel Gaussian parameters),
        not an RGB image.

    Structure (same as original, just renamed):

      invariant_encoder: SwinTransformerV2 for view-invariant representation
      dependent_encoder: SwinTransformerV2 for view-dependent representation
      decoder          : SwinTransformerV2Decoder over fused tokens

      invariant_output_head: VQ or Gaussian head for invariant branch
      dependent_output_head: VQ or Gaussian head for dependent branch
    """

class InvariantDependentSplatterVAE(nn.Module):
    """
    ReViWo-style multi-view VAE, now using Swin Transformers for the
    invariant / dependent encoders and a Swin-based decoder that directly
    produces a Splatter Image.

    The Swin architecture (embed_dim, depths, num_heads, patch_size, ...)
    is fully controlled by the YAML file via the `swin` section.
    """

    def __init__(
        self,
        swin_cfg,
        invariant_cb_config,
        dependent_cb_config,
        img_height: int,
        img_width: int,
        splatter_channels: int,
        fusion_style: str = "cat",
        use_dependent_vq: bool = True,
        is_dependent_ae: bool = True,
        use_invariant_vq: bool = True,
        is_invariant_ae: bool = True,
    ):
        super().__init__()

        # -------------------------------------------------------------
        # Image + patch configuration (from YAML)
        # -------------------------------------------------------------
        self.img_height = img_height
        self.img_width = img_width

        patch_size_cfg = swin_cfg.get("patch_size", 4)
        if isinstance(patch_size_cfg, int):
            patch_h = patch_w = patch_size_cfg
        else:
            assert len(patch_size_cfg) == 2, "swin.patch_size must be int or (h, w)."
            patch_h, patch_w = patch_size_cfg

        assert img_height % patch_h == 0, f"img_height={img_height} not divisible by patch_h={patch_h}"
        assert img_width % patch_w == 0, f"img_width={img_width} not divisible by patch_w={patch_w}"

        self.patch_h = patch_h
        self.patch_w = patch_w

        # Swin hyperparameters (all can be overridden in YAML)
        embed_dim = swin_cfg.get("embed_dim", 96)
        depths = swin_cfg.get("depths", [2, 2, 6, 2])
        num_heads = swin_cfg.get("num_heads", [3, 6, 12, 24])
        window_size = swin_cfg.get("window_size", 7)
        mlp_ratio = swin_cfg.get("mlp_ratio", 4.0)
        qkv_bias = swin_cfg.get("qkv_bias", True)
        drop_rate = swin_cfg.get("drop_rate", 0.0)
        attn_drop_rate = swin_cfg.get("attn_drop_rate", 0.0)
        drop_path_rate = swin_cfg.get("drop_path_rate", 0.1)
        ape = swin_cfg.get("ape", False)
        patch_norm = swin_cfg.get("patch_norm", True)
        use_checkpoint = swin_cfg.get("use_checkpoint", False)
        pretrained_window_sizes = swin_cfg.get(
            "pretrained_window_sizes",
            [0 for _ in range(len(depths))],
        )

        # -------------------------------------------------------------
        # Swin encoders for invariant and dependent representations
        # -------------------------------------------------------------
        self.invariant_encoder = SwinTransformerV2(
            img_size=(img_height, img_width),
            patch_size=(patch_h, patch_w),
            in_chans=3,
            num_classes=0,  # no classification head
            embed_dim=embed_dim,
            depths=depths,
            num_heads=num_heads,
            window_size=window_size,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            drop_rate=drop_rate,
            attn_drop_rate=attn_drop_rate,
            drop_path_rate=drop_path_rate,
            norm_layer=nn.LayerNorm,
            ape=ape,
            patch_norm=patch_norm,
            use_checkpoint=use_checkpoint,
            pretrained_window_sizes=pretrained_window_sizes,
        )

        self.dependent_encoder = SwinTransformerV2(
            img_size=(img_height, img_width),
            patch_size=(patch_h, patch_w),
            in_chans=3,
            num_classes=0,
            embed_dim=embed_dim,
            depths=depths,
            num_heads=num_heads,
            window_size=window_size,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            drop_rate=drop_rate,
            attn_drop_rate=attn_drop_rate,
            drop_path_rate=drop_path_rate,
            norm_layer=nn.LayerNorm,
            ape=ape,
            patch_norm=patch_norm,
            use_checkpoint=use_checkpoint,
            pretrained_window_sizes=pretrained_window_sizes,
        )

        # Swin bottleneck channel dimension
        latent_dim = self.invariant_encoder.num_features

        # Swin bottleneck spatial resolution (H_lat, W_lat)
        patches_resolution = self.invariant_encoder.patches_resolution
        num_layers = self.invariant_encoder.num_layers
        self.latent_patches_h = patches_resolution[0] // (2 ** (num_layers - 1))
        self.latent_patches_w = patches_resolution[1] // (2 ** (num_layers - 1))

        # Number of tokens per frame at the bottleneck (used by camera predictor)
        self.n_tokens_per_frame = self.latent_patches_h * self.latent_patches_w

        # -------------------------------------------------------------
        # Swin decoder: fused latents -> Splatter Image
        # -------------------------------------------------------------
        self.decoder = SwinTransformerV2Decoder(
            img_size=(img_height, img_width),
            patch_size=(patch_h, patch_w),
            out_chans=splatter_channels,
            embed_dim=embed_dim,
            depths=depths,
            num_heads=num_heads,
            window_size=window_size,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            drop_rate=drop_rate,
            attn_drop_rate=attn_drop_rate,
            drop_path_rate=drop_path_rate,
            norm_layer=nn.LayerNorm,
            patch_norm=patch_norm,
            use_checkpoint=use_checkpoint,
            pretrained_window_sizes=pretrained_window_sizes,
        )

        # -------------------------------------------------------------
        # Project encoder outputs into VQ codebook embedding spaces
        # -------------------------------------------------------------
        self.invariant_encoder_output_proj = nn.Linear(
            latent_dim,
            invariant_cb_config.embed_dim,
            bias=True,
        )
        self.dependent_encoder_output_proj = nn.Linear(
            latent_dim,
            dependent_cb_config.embed_dim,
            bias=True,
        )

        # -------------------------------------------------------------
        # Fusion + projection into Swin bottleneck dim
        # -------------------------------------------------------------
        self.fusion_style = fusion_style
        if fusion_style == "plus":
            assert (
                invariant_cb_config.embed_dim == dependent_cb_config.embed_dim
            ), "fusion_style='plus' requires equal invariant and dependent embed_dim."
            decoder_in_dim = invariant_cb_config.embed_dim
        elif fusion_style == "cat":
            decoder_in_dim = (
                invariant_cb_config.embed_dim + dependent_cb_config.embed_dim
            )
        else:
            raise NotImplementedError(f"Unknown fusion_style: {fusion_style}")

        self.decoder_input_proj = nn.Linear(
            decoder_in_dim,
            latent_dim,
            bias=True,
        )

        # -------------------------------------------------------------
        # VQ or Gaussian heads (same logic as before)
        # -------------------------------------------------------------
        self.invariant_output_head = (
            VectorQuantize(
                dim=invariant_cb_config.embed_dim,
                codebook_size=invariant_cb_config.n_embed,
                commitment_weight=invariant_cb_config.beta,
                use_cosine_sim=True,
                kmeans_init=True,
                kmeans_iters=10,
                threshold_ema_dead_code=2,
            )
            if use_invariant_vq
            else nn.Linear(
                invariant_cb_config.embed_dim,
                2 * invariant_cb_config.embed_dim,
            )
        )

        self.dependent_output_head = (
            VectorQuantize(
                dim=dependent_cb_config.embed_dim,
                codebook_size=dependent_cb_config.n_embed,
                commitment_weight=dependent_cb_config.beta,
                use_cosine_sim=True,
                kmeans_init=True,
                kmeans_iters=10,
                threshold_ema_dead_code=2,
            )
            if use_dependent_vq
            else nn.Linear(
                dependent_cb_config.embed_dim,
                2 * dependent_cb_config.embed_dim,
            )
        )

        # Same as original: optional final projection on dependent branch
        self.dependent_output_final_proj = nn.Identity()

        # Store flags
        self.splatter_channels = splatter_channels
        self.use_dependent_vq = use_dependent_vq
        self.is_dependent_ae = is_dependent_ae
        self.use_invariant_vq = use_invariant_vq
        self.is_invariant_ae = is_invariant_ae

    # ------------------------------------------------------------------
    # Encoding: produce invariant & dependent embeddings (with VQ or Gaussian)
    # ------------------------------------------------------------------
    def encode(
        self,
        x: torch.Tensor,
        deterministic_invariant: bool = False,
        deterministic_dependent: bool = False,
    ):
        """
        x: (B, 3, H, W) RGB image(s)

        Returns:
            z_inv: (B, T, D_inv)
            inv_embed_loss: scalar
            z_dep: (B, T, D_dep)
            dep_embed_loss: scalar
            (dependent_indices, invariant_indices)
        """
        B, C, H, W = x.shape
        assert H == self.img_height and W == self.img_width, (
            f"Expected input size ({self.img_height}, {self.img_width}), "
            f"got ({H}, {W})"
        )

        # ---------------------------------------------------------
        # Swin encoders: image -> bottleneck token sequences
        # ---------------------------------------------------------
        # (B, T_latent, latent_dim), T_latent = n_tokens_per_frame
        h_inv_tokens = self.invariant_encoder(x)
        h_dep_tokens = self.dependent_encoder(x)

        _, T_latent, _ = h_inv_tokens.shape
        assert T_latent == self.n_tokens_per_frame, (
            f"Expected {self.n_tokens_per_frame} latent tokens, "
            f"but encoder returned {T_latent}."
        )

        # Project to codebook embedding spaces
        h_inv = self.invariant_encoder_output_proj(h_inv_tokens)  # (B, T, D_inv)
        h_dep = self.dependent_encoder_output_proj(h_dep_tokens)  # (B, T, D_dep)

        # -------------------------
        # Invariant branch: VQ or Gaussian
        # -------------------------
        if self.use_invariant_vq:
            z_inv, invariant_encoding_indices, inv_aux_loss = self.invariant_output_head(
                h_inv
            )
            inv_embed_loss = inv_aux_loss.mean()
        else:
            z_inv_output = self.invariant_output_head(h_inv)  # (B, T, 2*D_inv)
            D_half = z_inv_output.shape[-1] // 2
            z_inv_mu = torch.tanh(z_inv_output[..., :D_half])
            if self.is_invariant_ae or deterministic_invariant:
                z_inv = z_inv_mu
                inv_embed_loss = torch.tensor(
                    0.0, dtype=torch.float32, device=z_inv.device
                )
            else:
                z_inv_sigma = torch.exp(z_inv_output[..., D_half:].clamp(-20, 2))
                inv_embed_loss = -0.5 * torch.mean(
                    1 + torch.log(z_inv_sigma**2) - z_inv_mu**2 - z_inv_sigma**2
                )
                dist = torch.distributions.Normal(z_inv_mu, z_inv_sigma)
                z_inv = dist.rsample() if self.training else dist.sample()
            invariant_encoding_indices = torch.zeros((6,), device=z_inv.device)

        # -------------------------
        # Dependent branch: VQ or Gaussian
        # -------------------------
        if self.use_dependent_vq:
            z_dep, dependent_encoding_indices, dep_aux_loss = self.dependent_output_head(
                h_dep
            )
            z_dep = self.dependent_output_final_proj(z_dep)
            dep_embed_loss = dep_aux_loss.mean()
        else:
            z_dep_output = torch.tanh(self.dependent_output_head(h_dep))  # (B, T, 2*D_dep)
            D_half = z_dep_output.shape[-1] // 2
            z_dep_mu = z_dep_output[..., :D_half]
            if self.is_dependent_ae or deterministic_dependent:
                z_dep = z_dep_mu
                dep_embed_loss = torch.tensor(
                    0.0, dtype=torch.float32, device=z_dep.device
                )
            else:
                z_dep_sigma = torch.exp(z_dep_output[..., D_half:].clamp(-20, 2))
                dep_embed_loss = -0.5 * torch.mean(
                    1 + torch.log(z_dep_sigma**2) - z_dep_mu**2 - z_dep_sigma**2
                )
                dist = torch.distributions.Normal(z_dep_mu, z_dep_sigma)
                z_dep = dist.rsample() if self.training else dist.sample()
            dependent_encoding_indices = torch.zeros((6,), device=z_dep.device)

        return z_inv, inv_embed_loss, z_dep, dep_embed_loss, (
            dependent_encoding_indices,
            invariant_encoding_indices,
        )

    # ------------------------------------------------------------------
    # Fusion of invariant & dependent embeddings
    # ------------------------------------------------------------------
    def fusion(self, z_inv: torch.Tensor, z_dep: torch.Tensor) -> torch.Tensor:
        """
        z_inv: (B, T, D_inv)
        z_dep: (B, T, D_dep)

        Returns:
            quant: (B, T, D_fused)
        """
        if self.fusion_style == "plus":
            return z_inv + z_dep
        elif self.fusion_style == "cat":
            return torch.cat((z_inv, z_dep), dim=-1)
        else:
            raise NotImplementedError(f"Unknown fusion_style: {self.fusion_style}")

    # ------------------------------------------------------------------
    # Decoding: from latents to Splatter Image
    # ------------------------------------------------------------------
    def decode(self, z_inv: torch.Tensor, z_dep: torch.Tensor) -> torch.Tensor:
        """
        Decode fused embeddings into a Splatter Image.

        Inputs:
            z_inv: (B, T, D_inv)
            z_dep: (B, T, D_dep)

        Returns:
            splatter: (B, splatter_channels, H, W)
        """
        # Fuse invariant + dependent latents
        quant = self.fusion(z_inv, z_dep)  # (B, T, D_fused)

        # Project into Swin bottleneck channel dimension
        dec_tokens = self.decoder_input_proj(quant)  # (B, T, latent_dim)

        # Swin decoder handles upsampling + de-patchifying
        splatter = self.decoder(dec_tokens).contiguous()  # (B, C_s, H, W)
        return splatter

    # ------------------------------------------------------------------
    # Optional forward shortcut (e.g. for reconstruction training)
    # ------------------------------------------------------------------
    def forward(self, x: torch.Tensor):
        """
        Convenience forward that:
            1) encodes x into z_inv, z_dep,
            2) decodes to a Splatter Image.

        Returns:
            splatter: (B, splatter_channels, H, W)
            total_embed_loss: scalar = inv_embed_loss + dep_embed_loss
        """
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