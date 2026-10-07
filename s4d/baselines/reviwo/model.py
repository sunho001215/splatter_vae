"""Build the ReViWo model from its config section and rebuild it from an export, for the frozen RL encoder."""

from __future__ import annotations

from pathlib import Path

import torch

from s4d.baselines.reviwo.configs import CodebookConfig, STTransConfig
from s4d.baselines.reviwo.multiview_vae import MultiViewBetaVAE

EXPORT_FORMAT = "splatter4d-baseline-reviwo-v1"


def build_model(reviwo_cfg: dict, img_size: int) -> MultiViewBetaVAE:
    """Reference ``baselines/ReViWo/train.py::main`` model construction from the ``reviwo`` config section."""
    return MultiViewBetaVAE(
        view_encoder_config=STTransConfig(**reviwo_cfg.get("view_encoder", {})),
        latent_encoder_config=STTransConfig(**reviwo_cfg.get("latent_encoder", {})),
        decoder_config=STTransConfig(**reviwo_cfg.get("decoder", {})),
        view_cb_config=CodebookConfig(**reviwo_cfg.get("view_codebook", {})),
        latent_cb_config=CodebookConfig(**reviwo_cfg.get("latent_codebook", {})),
        img_size=img_size,
        patch_size=reviwo_cfg.get("patch_size", 16),
        fusion_style=reviwo_cfg.get("fusion_style", "plus"),
        use_latent_vq=reviwo_cfg.get("use_latent_vq", True),
        is_latent_ae=reviwo_cfg.get("is_latent_ae", False),
        use_view_vq=reviwo_cfg.get("use_view_vq", True),
        is_view_ae=reviwo_cfg.get("is_view_ae", False),
    )


def disable_kmeans_init(model: MultiViewBetaVAE) -> None:
    """Trained codebooks must never be re-initialised by k-means (the reference RL wrapper does the same)."""
    for module in model.modules():
        if hasattr(module, "init_kmeans"):
            module.init_kmeans = False


def save_export(path: Path, model: MultiViewBetaVAE, reviwo_cfg: dict, img_size: int, step: int) -> None:
    """Export for the RL side: weights and the config needed to rebuild the model (the encoder half is used)."""
    payload = {
        "format": EXPORT_FORMAT,
        "reviwo_cfg": dict(reviwo_cfg),
        "img_size": int(img_size),
        "state_dict": model.state_dict(),
        "step": int(step),
    }
    tmp = Path(path).with_suffix(".tmp")
    torch.save(payload, tmp)
    tmp.replace(path)


def load_export(path: str | Path) -> tuple[dict, MultiViewBetaVAE]:
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if payload.get("format") != EXPORT_FORMAT:
        raise ValueError(f"{path} is not a ReViWo encoder export")
    model = build_model(payload["reviwo_cfg"], payload["img_size"])
    model.load_state_dict(payload["state_dict"], strict=True)
    disable_kmeans_init(model)
    return payload, model.eval().requires_grad_(False)
