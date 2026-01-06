from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Tuple, Optional

import numpy as np

# Torch is required for DINOv2
import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class DinoOutputSpec:
    patch_size: int
    grid_h: int
    grid_w: int
    feat_dim: int
    mode: str  # "patch" or "pixel"


class DINOv2DenseExtractor:
    """
    Dense DINOv2 feature extractor.

    Default:
      - model: ViT-S/14 (small)
      - mode: "patch" => output shape (B, grid_h, grid_w, D)
        where grid_h ~ H/14, grid_w ~ W/14

    Optional:
      - mode: "pixel" => output shape (B, H, W, D) by bilinear upsampling
        WARNING: this is extremely large for storage.
    """

    def __init__(
        self,
        device: str = "cuda",
        model_source: Literal["torchhub", "hf"] = "torchhub",
        mode: Literal["patch", "pixel"] = "patch",
        amp: bool = True,
    ):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.model_source = model_source
        self.mode = mode
        self.amp = amp and (self.device.type == "cuda")

        self.model, self.patch_size, self.feat_dim = self._load_model()
        self.model.eval()

        # Standard ImageNet normalization (common for ViT backbones)
        self.mean = torch.tensor([0.485, 0.456, 0.406], device=self.device).view(1, 3, 1, 1)
        self.std = torch.tensor([0.229, 0.224, 0.225], device=self.device).view(1, 3, 1, 1)

    def _load_model(self):
        """
        Two options:
        - torchhub: torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14')
        - hf: transformers Dinov2Model.from_pretrained('facebook/dinov2-small')
        """
        if self.model_source == "hf":
            try:
                from transformers import Dinov2Model
            except ImportError as e:
                raise ImportError("transformers not installed; install it or use model_source='torchhub'") from e

            model = Dinov2Model.from_pretrained("facebook/dinov2-small").to(self.device)
            patch = int(model.config.patch_size)
            dim = int(model.config.hidden_size)
            return model, patch, dim

        # torchhub path (official dinov2 repo)
        model = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14").to(self.device)
        # dinov2 hub models typically have patch_size attribute
        patch = int(getattr(model, "patch_size", 14))
        # feature dim depends on backbone; for vits14 it's commonly 384
        # We infer it by a quick forward on a dummy input of size (1,3,patch*2,patch*2)
        with torch.no_grad():
            x = torch.zeros((1, 3, patch * 2, patch * 2), device=self.device)
            # get_intermediate_layers is typical on dinov2 ViT impl; fallback to forward_features otherwise
            if hasattr(model, "get_intermediate_layers"):
                y = model.get_intermediate_layers(x, n=1, return_class_token=False)[0]  # (1, N, D)
                dim = int(y.shape[-1])
            else:
                y = model(x)
                dim = int(y.shape[-1])
        return model, patch, dim

    def _preprocess_uint8(self, imgs_uint8: np.ndarray) -> Tuple[torch.Tensor, int, int, int, int]:
        """
        imgs_uint8: (B,H,W,3) uint8
        Returns:
          x: float tensor (B,3,Hpad,Wpad)
          H,W original, Hpad,Wpad padded to multiple of patch_size
        """
        assert imgs_uint8.dtype == np.uint8
        assert imgs_uint8.ndim == 4 and imgs_uint8.shape[-1] == 3

        B, H, W, _ = imgs_uint8.shape
        ps = self.patch_size

        Hpad = int(np.ceil(H / ps) * ps)
        Wpad = int(np.ceil(W / ps) * ps)

        # Convert to torch float in [0,1]
        x = torch.from_numpy(imgs_uint8).to(self.device).float() / 255.0  # (B,H,W,3)
        x = x.permute(0, 3, 1, 2).contiguous()  # (B,3,H,W)

        # Pad bottom/right to multiples of patch size (avoid resizing)
        pad_h = Hpad - H
        pad_w = Wpad - W
        if pad_h > 0 or pad_w > 0:
            x = F.pad(x, (0, pad_w, 0, pad_h), mode="replicate")

        # Normalize
        x = (x - self.mean) / self.std
        return x, H, W, Hpad, Wpad

    @torch.no_grad()
    def extract(self, imgs_uint8: np.ndarray) -> Tuple[np.ndarray, DinoOutputSpec]:
        """
        imgs_uint8: (B,H,W,3) uint8
        Returns:
          feats:
            - mode="patch": (B, grid_h, grid_w, D) float16
            - mode="pixel": (B, H, W, D) float16 (WARNING: huge)
        """
        x, H, W, Hpad, Wpad = self._preprocess_uint8(imgs_uint8)

        ps = self.patch_size
        grid_h = Hpad // ps
        grid_w = Wpad // ps

        # Forward
        autocast_ctx = torch.autocast(device_type="cuda", dtype=torch.float16) if self.amp else torch.autocast("cpu")
        with autocast_ctx if self.amp else torch.no_grad():
            if self.model_source == "hf":
                # HF Dinov2Model returns last_hidden_state with [CLS] + patch tokens
                out = self.model(pixel_values=x)
                tokens = out.last_hidden_state  # (B, 1+N, D)
                patch_tokens = tokens[:, 1:, :]  # drop CLS
            else:
                # torchhub dinov2 ViT commonly supports get_intermediate_layers
                if hasattr(self.model, "get_intermediate_layers"):
                    patch_tokens = self.model.get_intermediate_layers(
                        x, n=1, return_class_token=False
                    )[0]  # (B, N, D)
                else:
                    # Fallback: try forward_features
                    if hasattr(self.model, "forward_features"):
                        feats = self.model.forward_features(x)
                        # This path is model-dependent; if you hit this, adapt as needed.
                        raise RuntimeError("Model forward_features path is not standardized; use get_intermediate_layers.")
                    raise RuntimeError("DINOv2 model does not expose get_intermediate_layers; use HF mode.")

        # Reshape tokens -> grid
        B = patch_tokens.shape[0]
        D = patch_tokens.shape[-1]
        patch_tokens = patch_tokens.reshape(B, grid_h, grid_w, D)  # (B,gh,gw,D)

        if self.mode == "patch":
            feats = patch_tokens.to(torch.float16).cpu().numpy()
            spec = DinoOutputSpec(patch_size=ps, grid_h=grid_h, grid_w=grid_w, feat_dim=D, mode="patch")
            return feats, spec

        # mode == "pixel": upsample to (Hpad,Wpad) then crop to (H,W)
        # Convert to (B,D,gh,gw)
        fm = patch_tokens.permute(0, 3, 1, 2).contiguous()
        up = F.interpolate(fm, size=(Hpad, Wpad), mode="bilinear", align_corners=False)
        up = up[:, :, :H, :W]  # crop padding
        up = up.permute(0, 2, 3, 1).contiguous()  # (B,H,W,D)
        feats = up.to(torch.float16).cpu().numpy()
        spec = DinoOutputSpec(patch_size=ps, grid_h=H, grid_w=W, feat_dim=D, mode="pixel")
        return feats, spec
