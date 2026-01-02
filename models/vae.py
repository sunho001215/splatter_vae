import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from einops.layers.torch import Rearrange
from dataclasses import dataclass

from vector_quantize_pytorch import VectorQuantize
from .transformer import STransformer 

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

      invariant_encoder: STransformer for view-invariant representation
      dependent_encoder: STransformer for view-dependent representation
      decoder          : STransformer over fused tokens

      invariant_output_head: VQ or Gaussian head for invariant branch
      dependent_output_head: VQ or Gaussian head for dependent branch
    """

    def __init__(
        self,
        invariant_encoder_config,
        dependent_encoder_config,
        decoder_config,
        invariant_cb_config,
        dependent_cb_config,
        img_height: int,
        img_width: int,
        patch_size,
        splatter_channels: int,
        fusion_style: str = "cat",
        use_dependent_vq: bool = True,
        is_dependent_ae: bool = True,
        use_invariant_vq: bool = True,
        is_invariant_ae: bool = True,
    ):
        super().__init__()

        # -------------------------------------------------------------
        # Handle non-square image + (possibly) non-square patch size
        # -------------------------------------------------------------
        if isinstance(patch_size, int):
            patch_h = patch_w = patch_size
        else:
            assert (
                len(patch_size) == 2
            ), "patch_size must be int or (patch_h, patch_w) tuple."
            patch_h, patch_w = patch_size

        assert img_height % patch_h == 0, "img_height must be divisible by patch_h."
        assert img_width % patch_w == 0, "img_width must be divisible by patch_w."

        n_patches_h = img_height // patch_h
        n_patches_w = img_width // patch_w
        n_tokens_per_frame = n_patches_h * n_patches_w

        # -------------------------------------------------------------
        # Update transformer configs (same style as original ReViWo)
        # -------------------------------------------------------------
        invariant_encoder_config = invariant_encoder_config.update(
            {
                "n_tokens_per_frame": n_tokens_per_frame,
                "block_size": n_tokens_per_frame,
                "vocab_size": 0,  # STransformer returns features directly
            }
        )

        dependent_encoder_config = dependent_encoder_config.update(
            {
                "n_tokens_per_frame": n_tokens_per_frame,
                "block_size": n_tokens_per_frame,
                "vocab_size": 0,
            }
        )

        decoder_config = decoder_config.update(
            {
                "n_tokens_per_frame": n_tokens_per_frame,
                "block_size": n_tokens_per_frame,
                "vocab_size": 0,
            }
        )

        # -------------------------------------------------------------
        # Transformers: invariant encoder, dependent encoder, decoder
        # -------------------------------------------------------------
        self.invariant_encoder = STransformer(invariant_encoder_config)
        self.dependent_encoder = STransformer(dependent_encoder_config)
        self.decoder = STransformer(decoder_config)

        # -------------------------------------------------------------
        # Shared patch embedding for both encoders
        #   (B, 3, H, W) -> (B, T, C), T = n_tokens_per_frame
        # -------------------------------------------------------------
        self.to_patch_embed = nn.Sequential(
            nn.Conv2d(
                in_channels=3,
                out_channels=invariant_encoder_config.n_embed,
                kernel_size=(patch_h, patch_w),
                stride=(patch_h, patch_w),
            ),
            # Result is (B, C, n_patches_h, n_patches_w)
            Rearrange("b c h w -> b (h w) c"),
        )

        # Project encoder outputs into VQ codebook embedding spaces
        self.invariant_encoder_output_proj = nn.Linear(
            invariant_encoder_config.n_embed,
            invariant_cb_config.embed_dim,
            bias=invariant_encoder_config.bias,
        )
        self.dependent_encoder_output_proj = nn.Linear(
            dependent_encoder_config.n_embed,
            dependent_cb_config.embed_dim,
            bias=dependent_encoder_config.bias,
        )

        # -------------------------------------------------------------
        # Fusion + decoder input projection
        # -------------------------------------------------------------
        # z_inv: (B, T, D_inv)  and  z_dep: (B, T, D_dep)
        # fusion_style:
        #   - 'plus': D_inv == D_dep, element-wise sum -> D_inv
        #   - 'cat' : concat -> (D_inv + D_dep)
        # Then map to decoder_config.n_embed for decoder transformer.
        # -------------------------------------------------------------
        if fusion_style == "plus":
            self.decoder_input_proj = nn.Linear(
                invariant_cb_config.embed_dim,
                decoder_config.n_embed,
                bias=decoder_config.bias,
            )
        elif fusion_style == "cat":
            self.decoder_input_proj = nn.Linear(
                invariant_cb_config.embed_dim + dependent_cb_config.embed_dim,
                decoder_config.n_embed,
                bias=decoder_config.bias,
            )
        else:
            raise NotImplementedError(f"Unknown fusion_style: {fusion_style}")

        # -------------------------------------------------------------
        # Decoder output: Splatter Image instead of RGB
        #
        #   - Input: decoder tokens (B, T, C_dec)
        #   - Reshape back to spatial grid (B, C_dec, n_patches_h, n_patches_w)
        #   - ConvTranspose2d to (B, splatter_channels, H, W)
        # -------------------------------------------------------------
        self.to_splatter = nn.Sequential(
            # (B, T, C) -> (B, C, n_patches_h, n_patches_w)
            Rearrange(
                "b (h w) c -> b c h w",
                h=n_patches_h,
                w=n_patches_w,
            ),
            # De-patchify: conv transpose to original H, W resolution
            nn.ConvTranspose2d(
                in_channels=decoder_config.n_embed,
                out_channels=splatter_channels,
                kernel_size=(patch_h, patch_w),
                stride=(patch_h, patch_w),
            ),
        )

        # -------------------------------------------------------------
        # VQ or Gaussian heads for invariant & dependent branches
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
                threshold_ema_dead_code=2
            )
            if use_dependent_vq
            else nn.Linear(
                dependent_cb_config.embed_dim,
                2 * dependent_cb_config.embed_dim,
            )
        )

        # Same as original latent_output_final_proj: Identity by default
        self.dependent_output_final_proj = nn.Identity()

        # -------------------------------------------------------------
        # Store basic config info
        # -------------------------------------------------------------
        self.img_height = img_height
        self.img_width = img_width
        self.patch_h = patch_h
        self.patch_w = patch_w
        self.n_patches_h = n_patches_h
        self.n_patches_w = n_patches_w
        self.n_tokens_per_frame = n_tokens_per_frame
        self.splatter_channels = splatter_channels

        self.fusion_style = fusion_style
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
            z_inv: (B, T, D_inv)  - invariant embedding (quantized or sampled)
            inv_embed_loss: scalar
            z_dep: (B, T, D_dep)  - dependent embedding (quantized or sampled)
            dep_embed_loss: scalar
            (dependent_encoding_indices, invariant_encoding_indices): codebook indices
        """
        B, C, H, W = x.shape
        assert H == self.img_height and W == self.img_width, (
            f"Expected input size ({self.img_height}, {self.img_width}), "
            f"got ({H}, {W})"
        )

        # Shared patch embedding: (B, 3, H, W) -> (B, T, C_embed)
        patch_embed = self.to_patch_embed(x).contiguous()  # (B, T, C_enc)

        # Invariant / dependent transformer encoders
        h_inv = self.invariant_encoder_output_proj(self.invariant_encoder(patch_embed))  # (B, T, D_inv)
        h_dep = self.dependent_encoder_output_proj(self.dependent_encoder(patch_embed))  # (B, T, D_dep)

        # -------------------------
        # Invariant branch: VQ or Gaussian
        # -------------------------
        if self.use_invariant_vq:
            z_inv, invariant_encoding_indices, inv_aux_loss = self.invariant_output_head(h_inv)
            inv_embed_loss = inv_aux_loss.mean()
        else:
            z_inv_output = self.invariant_output_head(h_inv)  # (B, T, 2*D_inv)
            D_half = z_inv_output.shape[-1] // 2
            z_inv_mu = torch.tanh(z_inv_output[..., :D_half])
            if self.is_invariant_ae or deterministic_invariant:
                z_inv = z_inv_mu
                inv_embed_loss = torch.tensor(0.0, dtype=torch.float32, device=z_inv.device)
            else:
                z_inv_sigma = torch.exp(z_inv_output[..., D_half:].clamp(-20, 2))
                inv_embed_loss = -0.5 * torch.mean(
                    1 + torch.log(z_inv_sigma ** 2) - z_inv_mu ** 2 - z_inv_sigma ** 2
                )
                dist = torch.distributions.Normal(z_inv_mu, z_inv_sigma)
                z_inv = dist.rsample() if self.training else dist.sample()
            invariant_encoding_indices = torch.zeros((6,), device=z_inv.device)

        # -------------------------
        # Dependent branch: VQ or Gaussian
        # -------------------------
        if self.use_dependent_vq:
            z_dep, dependent_encoding_indices, dep_aux_loss = self.dependent_output_head(h_dep)
            z_dep = self.dependent_output_final_proj(z_dep)
            dep_embed_loss = dep_aux_loss.mean()
        else:
            z_dep_output = torch.tanh(self.dependent_output_head(h_dep))  # (B, T, 2*D_dep)
            D_half = z_dep_output.shape[-1] // 2
            z_dep_mu = z_dep_output[..., :D_half]
            if self.is_dependent_ae or deterministic_dependent:
                z_dep = z_dep_mu
                dep_embed_loss = torch.tensor(0.0, dtype=torch.float32, device=z_dep.device)
            else:
                z_dep_sigma = torch.exp(z_dep_output[..., D_half:].clamp(-20, 2))
                dep_embed_loss = -0.5 * torch.mean(
                    1 + torch.log(z_dep_sigma ** 2) - z_dep_mu ** 2 - z_dep_sigma ** 2
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
                      H, W are img_height, img_width.
        """
        # Fuse invariant + dependent
        quant = self.fusion(z_inv, z_dep)  # (B, T, D_fused)

        # Map to decoder transformer embedding dimension
        dec_tokens = self.decoder_input_proj(quant)  # (B, T, n_embed_dec)

        # Transformer over tokens
        dec_tokens = self.decoder(dec_tokens)  # (B, T, n_embed_dec)

        # Convert token grid back into spatial Splatter Image
        splatter = self.to_splatter(dec_tokens).contiguous()  # (B, C_s, H, W)
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