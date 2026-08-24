from __future__ import annotations

import copy
from contextlib import nullcontext
import json
from dataclasses import fields
from typing import Any, Dict, Tuple

import h5py
import torch

from models.gaussian.motion import activate_motion_parameters
from models.gaussian.parameterization import (
    WorldSpaceGaussianParameterization,
    SplatterConfig,
    SplatterDataConfig,
    SplatterModelConfig,
    gaussian_params_per_gaussian,
)
from models.splattervae.config import SPLATTERVAE_ARCHITECTURE
from models.splattervae.model import SplatterVAE


def image_size_from_demo(dataset_path: str, demo_key: str) -> Tuple[int, int]:
    with h5py.File(dataset_path, "r") as f:
        first_cam = json.loads(f["data"][demo_key].attrs["camera_names"])[0]
        return tuple(f["data"][demo_key]["obs"][f"{first_cam}_rgb"].shape[1:3])


def _filter_dataclass_kwargs(values: Dict[str, Any], cls: type) -> Dict[str, Any]:
    allowed = {field.name for field in fields(cls)}
    return {key: value for key, value in values.items() if key in allowed}


def build_splatter_config(cfg: Dict[str, Any], img_height: int, img_width: int) -> SplatterConfig:
    renderer_cfg = dict(cfg.get("renderer", {}))
    gaussian_cfg = dict(cfg.get("gaussian", {}))
    renderer_cfg["img_height"] = int(img_height)
    renderer_cfg["img_width"] = int(img_width)
    return SplatterConfig(
        data=SplatterDataConfig(
            **_filter_dataclass_kwargs(renderer_cfg, SplatterDataConfig)
        ),
        model=SplatterModelConfig(
            **_filter_dataclass_kwargs(gaussian_cfg, SplatterModelConfig)
        ),
    )


def splatter_channels_from_config(cfg: Dict[str, Any], spl_cfg: SplatterConfig) -> int:
    del cfg
    return gaussian_params_per_gaussian(int(spl_cfg.model.max_sh_degree))


def _checkpoint_state_dict(payload: Dict[str, Any]) -> Dict[str, torch.Tensor]:
    state: Any = payload
    for key in ("vae_state_dict", "model_state_dict", "state_dict"):
        if isinstance(state, dict) and key in state and isinstance(state[key], dict):
            state = state[key]
            break
    if any(k.startswith("module.") for k in state):
        state = {k.replace("module.", "", 1): v for k, v in state.items()}
    return state


def adapt_config_to_checkpoint(cfg: Dict[str, Any], ckpt_path: str) -> Dict[str, Any]:
    """Reject pixel-aligned checkpoints instead of silently adapting them."""
    payload = torch.load(ckpt_path, map_location="cpu")
    architecture = payload.get("architecture") if isinstance(payload, dict) else None
    if architecture != SPLATTERVAE_ARCHITECTURE:
        raise ValueError(
            "Visualization requires an invariant grouped Gaussian set checkpoint; "
            f"expected {SPLATTERVAE_ARCHITECTURE!r}, found {architecture!r}."
        )
    return copy.deepcopy(cfg)


def build_splattervae(
    cfg: Dict[str, Any], img_height: int, img_width: int, splatter_channels: int
) -> SplatterVAE:
    model_cfg = dict(cfg.get("model", {}))
    vit_cfg = dict(cfg.get("vit", {}))
    masking_cfg = dict(model_cfg.get("masking", {}))
    decoder_cfg = dict(model_cfg.get("decoder", {}))
    motion_cfg = dict(model_cfg.get("motion", {}))
    return SplatterVAE(
        vit_cfg=vit_cfg,
        img_height=img_height,
        img_width=img_width,
        gaussian_params_per_gaussian=splatter_channels,
        inv_tube_mask_ratio=float(masking_cfg.get("inv_tube_mask_ratio", 0.50)),
        tube_mask_per_view=bool(masking_cfg.get("tube_mask_per_view", True)),
        state_dim=int(model_cfg.get("state_dim", 256)),
        flow_patch_threshold_pixels=float(
            masking_cfg.get("flow_patch_threshold_pixels", 0.5)
        ),
        motion_translation_max=float(motion_cfg.get("translation_max", 0.5)),
        temporal_modeling=bool(model_cfg.get("temporal_modeling", True)),
        decoder_num_parent_tokens=int(decoder_cfg.get("num_parent_tokens", 256)),
        decoder_gaussians_per_parent=int(
            decoder_cfg.get("gaussians_per_parent", 8)
        ),
        decoder_dim=int(decoder_cfg.get("dim", 128)),
        decoder_depth=int(decoder_cfg.get("depth", 2)),
        decoder_num_heads=int(decoder_cfg.get("num_heads", 4)),
        decoder_mlp_ratio=float(decoder_cfg.get("mlp_ratio", 4.0)),
        decoder_global_center=decoder_cfg.get(
            "global_center", [0.0, 0.5, 0.1]
        ),
        decoder_anchor_init_std=float(decoder_cfg.get("anchor_init_std", 0.15)),
        decoder_parent_offset_scale=float(
            decoder_cfg.get("parent_offset_scale", 0.1)
        ),
        decoder_child_radius=float(decoder_cfg.get("child_radius", 0.05)),
    )


def load_vae_state_dict(vae: SplatterVAE, ckpt_path: str) -> None:
    payload = torch.load(ckpt_path, map_location="cpu")
    if not isinstance(payload, dict):
        raise ValueError("SplatterVAE checkpoint payload must be a dictionary.")
    vae.validate_checkpoint_decoder_configuration(payload)
    vae.load_state_dict(_checkpoint_state_dict(payload), strict=True)


def load_converter_state_dict(converter: WorldSpaceGaussianParameterization, ckpt_path: str) -> None:
    del converter, ckpt_path


def build_visualization_models(
    cfg: Dict[str, Any],
    dataset_path: str,
    reference_demo: str,
    ckpt_path: str,
    device: torch.device,
):
    cfg = adapt_config_to_checkpoint(cfg, ckpt_path)
    img_height, img_width = image_size_from_demo(dataset_path, reference_demo)
    spl_cfg = build_splatter_config(cfg, img_height, img_width)
    splatter_channels = splatter_channels_from_config(cfg, spl_cfg)
    vae = build_splattervae(cfg, img_height, img_width, splatter_channels)
    load_vae_state_dict(vae, ckpt_path)

    converter = WorldSpaceGaussianParameterization(spl_cfg)
    load_converter_state_dict(converter, ckpt_path)
    vae.to(device).eval()
    converter.to(device).eval()
    return vae, converter, spl_cfg


def fixed_window_from_single_image(
    images: torch.Tensor, temporal_modeling: bool = True
) -> torch.Tensor:
    """Expand normalized ``(B,3,H,W)`` input in a visualization-only utility."""
    if images.dim() != 4 or images.shape[1] != 3:
        raise ValueError(f"Expected normalized images as (B,3,H,W), got {tuple(images.shape)}.")
    window = 3 if temporal_modeling else 1
    return images[:, None, None].expand(
        -1, window, 1, -1, -1, -1
    ).contiguous()


@torch.no_grad()
def decode_single_image_gaussians(
    vae: SplatterVAE,
    converter: WorldSpaceGaussianParameterization,
    images: torch.Tensor,
) -> tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
    sequence = fixed_window_from_single_image(images, vae.temporal_modeling)
    context = (
        torch.autocast(device_type="cuda", dtype=torch.bfloat16)
        if images.device.type == "cuda" else nullcontext()
    )
    with context:
        features = vae.inference_features(sequence)
        raw = vae.predict_gaussian_parameters(features["s_inv"])
    motion = (
        activate_motion_parameters(raw["raw_motion_params"].float(), vae.motion_translation_max)
        if vae.temporal_modeling else None
    )
    pc = converter(
        gaussian_parameters=raw["raw_gaussian_params"].float(),
        motion_parameters=motion,
    )
    return pc, features
