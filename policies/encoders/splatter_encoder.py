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

    # 7) Backbone: output is flattened invariant encoder output (B, N_tokens * D_inv)
    class SplatterBackbone(nn.Module):
        """
        Input : (B,3,H,W) float in [0,1] (cropping already handled upstream by DiffusionRgbEncoder)
        Output: (B, N_tokens * D_inv) where D_inv is invariant embedding dim after output_proj/VQ (if used)
        """
        def __init__(self, vae_module: nn.Module, img_h: int, img_w: int):
            super().__init__()
            self.vae = vae_module
            self.use_vq = hasattr(self.vae, "invariant_output_head") and (self.vae.invariant_output_head is not None)

            # Infer feature_dim with a dry run (depends on token count)
            with torch.no_grad():
                dummy = torch.zeros(1, 3, img_h, img_w, device=device)
                emb = self._encode_invariant(dummy)  # (1, N, D)
                self.feature_dim = int(emb.shape[1] * emb.shape[2])

            # Freeze everything in this module (extra safety)
            self.eval()
            for p in self.parameters():
                p.requires_grad = False

        def _encode_invariant(self, x01: torch.Tensor) -> torch.Tensor:
            # Convert [0,1] -> [-1,1]
            x_vae = x01 * 2.0 - 1.0
            tokens = self.vae.invariant_encoder(x_vae)                 # (B, N, *)
            emb = self.vae.invariant_encoder_output_proj(tokens)       # (B, N, D)
            if self.use_vq:
                vq_out = self.vae.invariant_output_head(emb)
                emb = vq_out[0] if isinstance(vq_out, (tuple, list)) else vq_out
            return emb

        @torch.no_grad()
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            # Fully frozen: no graph, no gradients, output detached
            with torch.no_grad():
                emb = self._encode_invariant(x)            # (B, N, D)
                feat = emb.reshape(emb.shape[0], -1)       # (B, N*D)
            return feat

    return SplatterBackbone(vae, img_h, img_w).to(device)
