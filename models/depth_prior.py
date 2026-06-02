from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class DepthPriorConfig:
    """Configuration for the frozen monocular depth prior."""

    enabled: bool = False
    provider: str = "unidepth"
    repo: str = "lpiccinelli-eth/UniDepth"
    version: str = "v2"
    backbone: str = "vits14"
    pretrained: bool = True
    trust_repo: bool = True
    force_reload: bool = False
    input_scale: float = 255.0
    use_as_encoder_input: bool = True


class UniDepthPrior(nn.Module):
    """Frozen UniDepth wrapper returning metric camera-z depth maps.

    Flash3D uses UniDepth as a frozen depth foundation model.  This wrapper
    keeps the dependency lazy: importing the training code does not require the
    UniDepth package or torch hub cache until the prior is explicitly enabled.
    """

    def __init__(self, cfg: DepthPriorConfig):
        super().__init__()
        self.cfg = cfg
        backbone = self._normalize_backbone(cfg.backbone)
        self.model = torch.hub.load(
            cfg.repo,
            "UniDepth",
            version=cfg.version,
            backbone=backbone,
            pretrained=bool(cfg.pretrained),
            trust_repo=bool(cfg.trust_repo),
            force_reload=bool(cfg.force_reload),
        )
        self.model.eval()
        for param in self.model.parameters():
            param.requires_grad_(False)

    @staticmethod
    def _normalize_backbone(backbone: str) -> str:
        aliases = {
            "cnvnxtl": "cnvnxtl",
            "convnextl": "cnvnxtl",
            "convnext-l": "cnvnxtl",
            "convnext_large": "cnvnxtl",
            "vits14": "vits14",
            "vit-s14": "vits14",
            "vit_small14": "vits14",
            "vitb14": "vitb14",
            "vit-b14": "vitb14",
            "vit_base14": "vitb14",
            "vitl14": "vitl14",
            "vit-l14": "vitl14",
            "vit_large14": "vitl14",
        }
        key = str(backbone).strip().lower()
        return aliases.get(key, key)

    def train(self, mode: bool = True):  # type: ignore[override]
        super().train(False)
        self.model.eval()
        return self

    @torch.inference_mode()
    def forward(
        self,
        images_01: torch.Tensor,
        intrinsics: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Predict depth for ``(B,3,H,W)`` images in ``[0, 1]``."""
        if images_01.dim() != 4:
            raise ValueError(f"Expected images as (B,3,H,W), got {tuple(images_01.shape)}.")
        if images_01.shape[1] != 3:
            raise ValueError(f"Expected RGB images, got shape {tuple(images_01.shape)}.")
        if intrinsics is not None and intrinsics.shape[0] != images_01.shape[0]:
            raise ValueError(
                f"Expected intrinsics batch {images_01.shape[0]}, got {tuple(intrinsics.shape)}."
            )

        depths = []
        for idx in range(images_01.shape[0]):
            image = images_01[idx].detach().clamp(0.0, 1.0) * float(self.cfg.input_scale)
            camera = intrinsics[idx].detach() if intrinsics is not None else None
            pred = self._infer_one(image, camera)
            depth = self._extract_depth(pred, image.shape[-2:])
            depths.append(depth)
        return torch.stack(depths, dim=0).to(device=images_01.device, dtype=images_01.dtype)

    def _infer_one(self, image: torch.Tensor, intrinsics: Optional[torch.Tensor]) -> Any:
        if hasattr(self.model, "infer"):
            if intrinsics is None:
                return self.model.infer(image)
            try:
                return self.model.infer(image, intrinsics)
            except TypeError:
                try:
                    return self.model.infer(image, camera=intrinsics)
                except TypeError:
                    return self.model.infer(image, K=intrinsics)

        data = {"image": image.unsqueeze(0)}
        if intrinsics is not None:
            data["K"] = intrinsics.unsqueeze(0)
        return self.model(data, {})

    def _extract_depth(self, prediction: Any, output_hw: tuple[int, int]) -> torch.Tensor:
        if isinstance(prediction, dict):
            for key in ("depth", "depths", "metric_depth"):
                if key in prediction:
                    depth = prediction[key]
                    break
            else:
                raise KeyError(f"UniDepth prediction did not contain a depth key: {prediction.keys()}")
        else:
            depth = prediction

        if not torch.is_tensor(depth):
            depth = torch.as_tensor(depth)
        depth = depth.detach()
        if depth.dim() == 4:
            depth = depth[0]
        if depth.dim() == 2:
            depth = depth.unsqueeze(0)
        elif depth.dim() == 3 and depth.shape[0] != 1:
            depth = depth[:1]
        elif depth.dim() != 3:
            raise ValueError(f"Unexpected UniDepth depth shape {tuple(depth.shape)}.")

        if depth.shape[-2:] != output_hw:
            depth = F.interpolate(
                depth.unsqueeze(0),
                size=output_hw,
                mode="bilinear",
                align_corners=False,
            ).squeeze(0)
        return depth.contiguous()


def build_depth_prior_estimator(cfg: DepthPriorConfig) -> Optional[UniDepthPrior]:
    if not bool(cfg.enabled):
        return None
    provider = str(cfg.provider).strip().lower()
    if provider in {"dataset", "gt", "ground_truth", "sim", "simulator"}:
        return None
    if provider != "unidepth":
        raise ValueError(f"Unsupported depth prior provider: {cfg.provider!r}.")
    return UniDepthPrior(cfg)
