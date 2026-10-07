"""Rebuild the SinCro state encoder (``MaskedViTEncoder``) from an export, for the frozen RL encoder."""

from __future__ import annotations

from pathlib import Path

import torch

from s4d.baselines.sincro.mae_encoder import MaskedViTEncoder

EXPORT_FORMAT = "splatter4d-baseline-sincro-v1"


def build_encoder(model_cfg: dict, device: torch.device | str) -> MaskedViTEncoder:
    """Same constructor arguments as ``create_nerf`` (SimpleArgs maps vit/decoder mlp dims and ``num_view``)."""
    return MaskedViTEncoder(
        img_size=model_cfg["img_size"],
        patch_size=model_cfg["patch_size"],
        embed_dim=model_cfg["embed_dim"],
        depth=model_cfg["vit_depth"],
        num_heads=model_cfg["vit_num_heads"],
        num_view=model_cfg["num_views"],
        device=device,
        time_interval=model_cfg["time_interval"],
        decoder_depth=model_cfg["decoder_depth"],
        decoder_num_heads=model_cfg["decoder_num_heads"],
        decoder_output_dim=model_cfg["decoder_output_dim"],
        batch_size=1,
        vit_encoder_mlp_dim=model_cfg["vit_mlp_dim"],
        vit_decoder_mlp_dim=model_cfg["decoder_mlp_dim"],
    ).to(device)


def save_export(path: Path, encoder: MaskedViTEncoder, model_cfg: dict, frame_spacing: int, step: int) -> None:
    """Encoder-only export: weights, the model config needed to rebuild it, and the pretraining frame spacing."""
    payload = {
        "format": EXPORT_FORMAT,
        "model_cfg": dict(model_cfg),
        "frame_spacing": int(frame_spacing),
        "state_dict": encoder.state_dict(),
        "step": int(step),
    }
    tmp = Path(path).with_suffix(".tmp")
    torch.save(payload, tmp)
    tmp.replace(path)


def load_export(path: str | Path) -> tuple[dict, MaskedViTEncoder]:
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if payload.get("format") != EXPORT_FORMAT:
        raise ValueError(f"{path} is not a SinCro encoder export")
    encoder = build_encoder(payload["model_cfg"], "cpu")
    encoder.load_state_dict(payload["state_dict"], strict=True)
    return payload, encoder.eval().requires_grad_(False)
