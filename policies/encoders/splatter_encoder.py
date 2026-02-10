from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

import torch
import torch.nn as nn
import yaml


def make_splatter_custom_backbone(
    *,
    dp_config: Any,
    cfg_raw: dict,
    base_dir: Path,
    device: torch.device,
) -> Optional[nn.Module]:
    """
    Build a SplatterVAE-based custom vision backbone for modified LeRobot DiffusionPolicy.

    Enable via diffusion config.json:
      - "vision_backbone": "splatter_vae"
      - "splatter_vae_train_yaml": "splatter_train.yaml"
      - "splatter_vae_filename": "step_00050000.pth"   (must contain "vae_state_dict")

    Output feature dimension:
      - The flattened output of the view-invariant encoder branch (B, N_tokens * D_inv).

    Freezing:
      - The backbone is fully frozen (no gradients, output is detached).
    """
    # 1) Check switch
    vision_backbone = str(cfg_raw.get("vision_backbone", "resnet18")).lower()
    if vision_backbone not in ("splatter_vae", "splattervae", "splatter"):
        raise ValueError(f"Invalid vision_backbone for SplatterVAE: {vision_backbone}")
    
    # 2) Resolve paths (absolute allowed; otherwise relative to base_dir)
    yaml_name = cfg_raw.get("splatter_vae_train_yaml", "splatter_train.yaml")
    ckpt_name = cfg_raw.get("splatter_vae_filename", "step_latest.pth")

    train_yaml_path = Path(yaml_name)
    if not train_yaml_path.is_absolute():
        train_yaml_path = base_dir / train_yaml_path

    vae_ckpt_path = Path(ckpt_name)
    if not vae_ckpt_path.is_absolute():
        vae_ckpt_path = base_dir / vae_ckpt_path

    if not train_yaml_path.exists():
        raise FileNotFoundError(f"Missing Splatter train YAML: {train_yaml_path}")
    if not vae_ckpt_path.exists():
        raise FileNotFoundError(f"Missing Splatter checkpoint: {vae_ckpt_path}")

    # 3) Avoid per-camera encoder module sharing issues
    if bool(getattr(dp_config, "use_separate_rgb_encoder_per_camera", False)):
        dp_config.use_separate_rgb_encoder_per_camera = False

    # 4) Determine image resolution that reaches the backbone
    #    (DiffusionRgbEncoder will apply crop_shape before calling custom backbone)
    crop_shape = getattr(dp_config, "crop_shape", None)
    if crop_shape is not None:
        img_h, img_w = int(crop_shape[0]), int(crop_shape[1])
    else:
        # image_features: dict[str, FeatureSpec(shape=(C,H,W), ...)]
        images_shape = next(iter(dp_config.image_features.values())).shape
        img_h, img_w = int(images_shape[1]), int(images_shape[2])

    # 5) Rebuild VAE architecture from training YAML (same logic as training code)
    from models.splatter import (
        SplatterConfig,
        SplatterDataConfig,
        SplatterModelConfig,
        VAESplatterToGaussians,
    )
    from models.vae import InvariantDependentSplatterVAE, CodebookConfig

    with open(train_yaml_path, "r") as f:
        train_cfg = yaml.safe_load(f)

    # --- splatter cfg ---
    spl_cfg = train_cfg.get("splatter", {})
    spl_data = spl_cfg.get("data", {})
    spl_model = spl_cfg.get("model", {})

    splatter_data_cfg = SplatterDataConfig(**spl_data)
    splatter_data_cfg.img_height = img_h
    splatter_data_cfg.img_width = img_w
    splatter_model_cfg = SplatterModelConfig(**spl_model)
    splatter_cfg = SplatterConfig(data=splatter_data_cfg, model=splatter_model_cfg)

    # --- splatter_channels ---
    tmp = VAESplatterToGaussians(splatter_cfg)
    splatter_channels = int(tmp.num_splatter_channels())
    del tmp

    # --- codebooks ---
    cb = train_cfg.get("codebook", {})
    inv_cb = CodebookConfig(**cb.get("invariant", {}))
    dep_cb = CodebookConfig(**cb.get("dependent", {}))

    # --- model flags ---
    model_cfg = train_cfg.get("model", {})
    fusion_style = model_cfg.get("fusion_style", "cat")
    use_dependent_vq = bool(model_cfg.get("use_dependent_vq", True))
    is_dependent_ae = bool(model_cfg.get("is_dependent_ae", False))
    use_invariant_vq = bool(model_cfg.get("use_invariant_vq", True))
    is_invariant_ae = bool(model_cfg.get("is_invariant_ae", False))

    # --- swin cfg + patch_size fallback ---
    swin_cfg = dict(train_cfg.get("swin", {}))
    if "patch_size" not in swin_cfg:
        grid_h = int(model_cfg.get("grid_tokens_h", 8))
        grid_w = int(model_cfg.get("grid_tokens_w", 8))
        if img_h % grid_h != 0 or img_w % grid_w != 0:
            raise ValueError(
                f"(H,W)=({img_h},{img_w}) must be divisible by (grid_tokens_h,grid_tokens_w)=({grid_h},{grid_w})."
            )
        swin_cfg["patch_size"] = [img_h // grid_h, img_w // grid_w]
    else:
        ps = swin_cfg["patch_size"]
        if isinstance(ps, int):
            swin_cfg["patch_size"] = [ps, ps]

    vae = InvariantDependentSplatterVAE(
        swin_cfg=swin_cfg,
        invariant_cb_config=inv_cb,
        dependent_cb_config=dep_cb,
        img_height=img_h,
        img_width=img_w,
        splatter_channels=splatter_channels,
        fusion_style=fusion_style,
        use_dependent_vq=use_dependent_vq,
        is_dependent_ae=is_dependent_ae,
        use_invariant_vq=use_invariant_vq,
        is_invariant_ae=is_invariant_ae,
    ).to(device)

    # Freeze VAE
    vae.eval()
    for p in vae.parameters():
        p.requires_grad = False

    # 6) Load only tensors safely (your checkpoint dict is weights-only friendly)
    ckpt = torch.load(str(vae_ckpt_path), map_location="cpu", weights_only=True)
    if "vae_state_dict" not in ckpt:
        raise KeyError(f"{vae_ckpt_path} has no 'vae_state_dict'")
    vae.load_state_dict(ckpt["vae_state_dict"], strict=True)

    # 7) Backbone: frozen token encoder + learnable MLP projection
    class SplatterBackbone(nn.Module):
        """
        Frozen VAE tokens (B, N, D_token)
            -> LayerNorm
            -> + trainable positional embedding (nn.Embedding) on KV tokens
            -> Attention pooling with learnable queries (B, Q, attn_dim)
            -> flatten pooled queries (B, Q*attn_dim)
            -> LN + Linear -> global_cond (B, final_proj_dim)
        """

        # -----------------------------
        # Cross-Attention Block
        # -----------------------------
        class CrossAttentionBlock(nn.Module):
            """
            Cross-attention block:

                q = q + Attn(LN(q), LN(kv))
                q = q + FFN(LN(q))
            """
            def __init__(
                self,
                *,
                q_dim: int,
                kv_dim: int,
                num_heads: int,
                mlp_ratio: float = 2.0,
                dropout: float = 0.0,
            ):
                super().__init__()
                self.q_ln = nn.LayerNorm(q_dim)
                self.kv_ln = nn.LayerNorm(kv_dim)

                # Allow q_dim != kv_dim via kdim/vdim.
                self.attn = nn.MultiheadAttention(
                    embed_dim=q_dim,
                    num_heads=num_heads,
                    dropout=dropout,
                    batch_first=True,
                    kdim=kv_dim,
                    vdim=kv_dim,
                )
                self.attn_drop = nn.Dropout(dropout)

                self.ffn_ln = nn.LayerNorm(q_dim)
                hidden = int(q_dim * mlp_ratio)
                self.ffn = nn.Sequential(
                    nn.Linear(q_dim, hidden),
                    nn.GELU(),
                    nn.Linear(hidden, q_dim),
                )
                self.ffn_drop = nn.Dropout(dropout)

            def forward(
                self,
                q: torch.Tensor,  # (B, Q, q_dim)
                kv: torch.Tensor,  # (B, N, kv_dim)
                key_padding_mask: Optional[torch.Tensor] = None,  # (B, N), True => ignore token
            ) -> torch.Tensor:
                # Pre-norm
                q_norm = self.q_ln(q)
                kv_norm = self.kv_ln(kv)

                # Cross-attention
                attn_out, _ = self.attn(
                    query=q_norm,
                    key=kv_norm,
                    value=kv_norm,
                    key_padding_mask=key_padding_mask,
                    need_weights=False,
                )
                q = q + self.attn_drop(attn_out)  # residual

                # FFN
                ffn_out = self.ffn(self.ffn_ln(q))
                q = q + self.ffn_drop(ffn_out)  # residual
                return q

        def __init__(
            self,
            vae_module: nn.Module,
            img_h: int,
            img_w: int,
            *,
            num_queries: int = 4,
            attn_dim: int = 128,         # query token dim (internal pooled representation)
            final_proj_dim: int = 128,   # FINAL output dim to the diffusion policy
            num_heads: int = 8,
            num_layers: int = 2,
            mlp_ratio: float = 2.0,
            dropout: float = 0.0,
        ):
            super().__init__()

            # -----------------------
            # Frozen VAE (no grads)
            # -----------------------
            self.vae = vae_module
            self.vae.eval()
            self.vae.requires_grad_(False)

            self.use_vq = hasattr(self.vae, "invariant_output_head") and (
                self.vae.invariant_output_head is not None
            )

            # Infer token shape using a dummy input so we can size the positional embedding table.
            with torch.no_grad():
                device = next(self.vae.parameters()).device
                dummy = torch.zeros(1, 3, img_h, img_w, device=device)
                tok = self._encode_invariant_tokens(dummy)  # (1, N, D_token)
                self.num_tokens = int(tok.shape[1])
                self.d_token = int(tok.shape[2])

            # -----------------------
            # Pooling configuration
            # -----------------------
            self.num_queries = int(num_queries)
            self.attn_dim = int(attn_dim)
            self.final_proj_dim = int(final_proj_dim)

            # Normalize frozen tokens before adding positional embedding.
            self.token_norm = nn.LayerNorm(self.d_token)

            # -----------------------
            # Positional embedding (nn.Embedding)
            # -----------------------
            # This is trainable and adds an explicit "token order / location" signal to KV tokens.
            self.kv_pos_embed = nn.Embedding(self.num_tokens, self.d_token)
            nn.init.normal_(self.kv_pos_embed.weight, mean=0.0, std=0.02)

            # Precompute position indices [0..N-1] as a buffer.
            # We slice this in forward in case N is smaller (should normally match exactly).
            self.register_buffer(
                "pos_ids",
                torch.arange(self.num_tokens, dtype=torch.long).unsqueeze(0),  # (1, N)
                persistent=False,
            )

            self.kv_pos_drop = nn.Dropout(dropout)

            # -----------------------
            # Learnable query tokens
            # -----------------------
            # (1, Q, attn_dim) -> expanded to (B, Q, attn_dim) in forward.
            self.query = nn.Parameter(torch.randn(1, self.num_queries, self.attn_dim) * 0.02)

            # Stacked cross-attention blocks
            self.blocks = nn.ModuleList([
                SplatterBackbone.CrossAttentionBlock(
                    q_dim=self.attn_dim,
                    kv_dim=self.d_token,
                    num_heads=int(num_heads),
                    mlp_ratio=float(mlp_ratio),
                    dropout=float(dropout),
                )
                for _ in range(int(num_layers))
            ])

            # -----------------------
            # Final output head: (B, Q*attn_dim) -> (B, final_proj_dim)
            # -----------------------
            self.out_ln = nn.LayerNorm(self.num_queries * self.attn_dim)
            self.out_proj = nn.Linear(self.num_queries * self.attn_dim, self.final_proj_dim, bias=True)

            # Expose final conditioning dimension
            self.feature_dim = self.final_proj_dim

        def _encode_invariant_tokens(self, x01: torch.Tensor) -> torch.Tensor:
            """
            Frozen token extraction from VAE.
            Returns: (B, N, D_token)
            """
            # Match your existing preprocessing: [0,1] -> [-1,1]
            x_vae = x01 * 2.0 - 1.0

            tokens = self.vae.invariant_encoder(x_vae)              # (B, N, *)
            emb = self.vae.invariant_encoder_output_proj(tokens)    # (B, N, D_token)

            # Optional VQ/output head
            if self.use_vq:
                vq_out = self.vae.invariant_output_head(emb)
                emb = vq_out[0] if isinstance(vq_out, (tuple, list)) else vq_out

            return emb

        def forward(
            self,
            x: torch.Tensor,
            key_padding_mask: Optional[torch.Tensor] = None,
        ) -> torch.Tensor:
            """
            Args:
                x: (B,3,H,W) in [0,1]
                key_padding_mask: optional (B, N_tokens), True => ignore token in attention.
                                Usually None for dense vision tokens.

            Returns:
                global_cond: (B, final_proj_dim)
            """
            # 1) Get frozen tokens from VAE (no grads)
            with torch.no_grad():
                kv = self._encode_invariant_tokens(x)  # (B, N, D_token)
            kv = kv.detach()

            B, N, D = kv.shape

            # Sanity: positional embedding table was built for a fixed N (based on img_h/img_w).
            # If N changes, you must rebuild or use a different PE strategy.
            if N > self.num_tokens:
                raise ValueError(
                    f"Got N_tokens={N}, but positional embedding was built for num_tokens={self.num_tokens}. "
                    f"Token count must be consistent."
                )

            # 2) Normalize KV tokens
            kv = self.token_norm(kv)  # (B, N, D_token)

            # 3) Add trainable positional embedding to KV tokens
            #    pos_ids: (1, N) -> pe: (1, N, D_token) -> broadcast to (B, N, D_token)
            pos_ids = self.pos_ids[:, :N].to(kv.device)          # (1, N)
            pe = self.kv_pos_embed(pos_ids).to(kv.dtype)         # (1, N, D_token)
            kv = self.kv_pos_drop(kv + pe)                       # (B, N, D_token)

            # 4) Initialize learnable query tokens for the batch
            q = self.query.expand(B, -1, -1)  # (B, Q, attn_dim)

            # 5) Stacked cross-attention pooling
            for blk in self.blocks:
                q = blk(q, kv, key_padding_mask=key_padding_mask)  # (B, Q, attn_dim)

            # 6) Flatten pooled queries and project to final global conditioning vector
            q_flat = q.reshape(B, self.num_queries * self.attn_dim)  # (B, Q*attn_dim)
            q_flat = self.out_ln(q_flat)
            global_cond = self.out_proj(q_flat)                      # (B, final_proj_dim)

            return global_cond

    return SplatterBackbone(vae, img_h, img_w).to(device)
