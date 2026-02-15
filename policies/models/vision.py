# vision.py
from __future__ import annotations

from typing import Any, Dict

import torch
import torch.nn as nn
import torchvision
import torchvision.transforms.functional as TF

from models.vae import InvariantDependentSplatterVAE, CodebookConfig

# ----------------------------
# ResNet50
# ----------------------------
class ResNet50Tokens(nn.Module):
    """
    ResNet50 -> feature map -> flatten into tokens.
    This matches the ALOHA Unleashed description: use stage-4 output and flatten.
    """
    def __init__(self, cfg: Dict[str, Any]):
        super().__init__()
        import torchvision

        pretrained = bool(cfg.get("pretrained", True))
        self.out_stage = int(cfg.get("out_stage", 4))
        self.flatten = bool(cfg.get("flatten", True))
        self.image_height = int(cfg.get("img_height", 200))
        self.image_width = int(cfg.get("img_width", 200))

        # Normalize the input images to imagenet stats
        self.register_buffer(
            "img_mean",
            torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1),
            persistent=False,
        )
        self.register_buffer(
            "img_std",
            torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1),
            persistent=False,
        )

        # torchvision resnet50
        backbone = torchvision.models.resnet50(
            weights=(torchvision.models.ResNet50_Weights.IMAGENET1K_V1 if pretrained else None)
        )

        # Build "stage-4 output" trunk:
        # conv1/bn1/relu/maxpool + layer1..layer3
        self.stem = nn.Sequential(
            backbone.conv1, backbone.bn1, backbone.relu, backbone.maxpool
        )
        self.layer1 = backbone.layer1
        self.layer2 = backbone.layer2
        self.layer3 = backbone.layer3

        self.out_channels = 1024

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, 3, H, W)
        returns:
          tokens: (B, N, C) where N = H' * W'
        """
        # Normalize
        x = (x - self.img_mean) / self.img_std

        # Crop
        if self.training:
            # Random crop
            i, j, h, w = torchvision.transforms.RandomCrop.get_params(
                x, output_size=(self.image_height, self.image_width)
            )
            x = TF.crop(x, i, j, h, w)
        else:
            # Center crop
            x = TF.center_crop(x, (self.image_height, self.image_width))

        # Forward through ResNet50 trunk
        h = self.stem(x)
        h = self.layer1(h)
        h = self.layer2(h)
        h = self.layer3(h)  # stage-4 output

        if not self.flatten:
            return h  # (B, C, H', W')

        # Flatten into tokens
        B, C, Hp, Wp = h.shape
        tokens = h.flatten(2).transpose(1, 2).contiguous()  # (B, N, C)
        return tokens


# --------------------------------------
# SplatterVAE
# --------------------------------------
class SplatterVAEInvariantCodebookTokens(nn.Module):
    """
    Loads pre-trained InvariantDependentSplatterVAE and returns
    the *VQ codebook output* for the invariant branch (quantized embeddings).

    Workflow:
      - build from YAML (needs swin + codebook configs)
      - load checkpoint weights
      - freeze (no updates during policy training)
      - use output of codebook (quantized vectors), not pre-VQ features
    """

    def __init__(self, cfg: Dict[str, Any]):
        super().__init__()

        sv_cfg = cfg["splatter_vae"]
        
        # Feature source
        self.feature_source = str(sv_cfg.get("feature_source", "codebook"))

        # -----------------------------
        # 1) Instantiate model skeleton
        # -----------------------------
        img_h = int(cfg["img_height"])
        img_w = int(cfg["img_width"])

        swin_cfg: Dict[str, Any] = dict(sv_cfg["swin"])
        model_cfg: Dict[str, Any] = dict(sv_cfg["model"])
        cb_cfg: Dict[str, Any] = dict(sv_cfg["codebook"])

        inv_cb = CodebookConfig(**cb_cfg["invariant"])
        dep_cb = CodebookConfig(**cb_cfg["dependent"])

        # splatter_channels is only needed for decoder output shape
        splatter_channels = int(sv_cfg.get("splatter_channels", 96))
        if splatter_channels <= 0:
            raise ValueError(
                "vision.splatter_vae.splatter_channels must be set (or add your helper to compute it)."
            )

        self.vae = InvariantDependentSplatterVAE(
            swin_cfg=swin_cfg,
            invariant_cb_config=inv_cb,
            dependent_cb_config=dep_cb,
            img_height=img_h,
            img_width=img_w,
            splatter_channels=splatter_channels,
            fusion_style=str(model_cfg.get("fusion_style", "cat")),
            use_dependent_vq=bool(model_cfg.get("use_dependent_vq", True)),
            is_dependent_ae=bool(model_cfg.get("is_dependent_ae", True)),
            use_invariant_vq=bool(model_cfg.get("use_invariant_vq", True)),
            is_invariant_ae=bool(model_cfg.get("is_invariant_ae", True)),
        )

        # --------------------------------
        # 2) Load checkpoint (pretrained)
        # --------------------------------
        ckpt_path = sv_cfg["checkpoint_path"]
        state = torch.load(ckpt_path, map_location="cpu")

        # Support either:
        #  - raw state_dict
        #  - checkpoint dict containing state_dict
        state_dict = state.get("vae_state_dict", state)

        # If your training saved with prefixes (e.g., "module."), strip here if needed.
        self.vae.load_state_dict(state_dict, strict=True)

        # -----------------------
        # 3) Freeze permanently
        # -----------------------
        self.vae.eval()
        for p in self.vae.parameters():
            p.requires_grad = False

        # Output dim
        if self.feature_source == "swin":
            self.out_channels = self.vae.invariant_encoder.num_features
        else: # feature_source in ["pre_vq", "codebook"]
            self.out_channels = inv_cb.embed_dim

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, 3, H, W)

        returns:
          inv_tokens: (B, N, D_cb)

        Implementation detail:
          We explicitly route through:
            invariant_encoder -> invariant_output_head
          so we get the *VQ codebook output* (quantized vectors).
        """
        # Rescale to [-1, 1] as SplatterVAE expects
        x = x * 2.0 - 1.0

        # Get invariant encoder features
        h_inv = self.vae.invariant_encoder(x)

        if self.feature_source == "swin":
            tokens = h_inv
        else:
            h_proj =  self.vae.invariant_encoder_output_proj(h_inv)
            if self.feature_source == "pre_vq":
                tokens = h_proj
            else:
                z_inv, inv_indices, inv_aux_loss = self.vae.invariant_output_head(h_proj)
                tokens = z_inv

        return tokens.contiguous()


# ----------------------------
# Factory
# ----------------------------
def build_vision_encoder(cfg: Dict[str, Any]) -> nn.Module:
    enc_type = cfg["vision"].get("encoder_type", cfg["vision"].get("name"))
    if enc_type == "resnet50":
        return ResNet50Tokens(cfg["vision"]["resnet50"])
    elif enc_type == "splatter_vae":
        return SplatterVAEInvariantCodebookTokens(cfg["vision"])
    else:
        raise ValueError(f"Unknown vision.encoder_type={enc_type}")
