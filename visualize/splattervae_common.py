from __future__ import annotations

import json
from typing import Any, Dict, Tuple

import h5py
import torch

from models.splatter import (
    SplatterConfig,
    SplatterDataConfig,
    SplatterModelConfig,
    VAESplatterToGaussians,
    default_splatter_channels,
)
from models.vae import CodebookConfig, SplatterVAE


def image_size_from_demo(dataset_path: str, demo_key: str) -> Tuple[int, int]:
    with h5py.File(dataset_path, "r") as f:
        first_cam = json.loads(f["data"][demo_key].attrs["camera_names"])[0]
        return tuple(f["data"][demo_key]["obs"][f"{first_cam}_rgb"].shape[1:3])


def build_splatter_config(cfg: Dict[str, Any], img_height: int, img_width: int) -> SplatterConfig:
    spl_cfg = cfg.get("splatter", {})
    spl_data_cfg = dict(spl_cfg.get("data", {}))
    spl_model_cfg = dict(spl_cfg.get("model", {}))
    spl_data_cfg["img_height"] = int(img_height)
    spl_data_cfg["img_width"] = int(img_width)
    return SplatterConfig(
        data=SplatterDataConfig(**spl_data_cfg),
        model=SplatterModelConfig(**spl_model_cfg),
    )


def splatter_channels_from_config(cfg: Dict[str, Any], spl_cfg: SplatterConfig) -> int:
    return int(
        cfg.get("splatter", {}).get(
            "splatter_channels",
            default_splatter_channels(
                max_sh_degree=int(spl_cfg.model.max_sh_degree),
                num_gaussians_per_pixel=int(spl_cfg.model.num_gaussians_per_pixel),
            ),
        )
    )


def build_splattervae(cfg: Dict[str, Any], img_height: int, img_width: int, splatter_channels: int) -> SplatterVAE:
    cb_cfg = cfg.get("codebook", {})
    inv_cb = CodebookConfig(**cb_cfg.get("invariant", {}))
    dep_cb = CodebookConfig(**cb_cfg.get("dependent", {}))
    model_cfg = dict(cfg.get("model", {}))
    vit_cfg = dict(cfg.get("vit", {}))

    return SplatterVAE(
        vit_cfg=vit_cfg,
        invariant_cb_config=inv_cb,
        dependent_cb_config=dep_cb,
        img_height=img_height,
        img_width=img_width,
        splatter_channels=splatter_channels,
        fusion_style=str(model_cfg.get("fusion_style", "cat")),
        use_dependent_vq=bool(model_cfg.get("use_dependent_vq", True)),
        is_dependent_ae=bool(model_cfg.get("is_dependent_ae", True)),
        use_invariant_vq=bool(model_cfg.get("use_invariant_vq", True)),
        is_invariant_ae=bool(model_cfg.get("is_invariant_ae", True)),
        dep_input_mask_ratio=float(model_cfg.get("dep_input_mask_ratio", 0.95)),
        dep_mask_eval=bool(model_cfg.get("dep_mask_eval", True)),
        dpt_features=int(vit_cfg.get("dpt_features", 256)),
    )


def load_vae_state_dict(vae: SplatterVAE, ckpt_path: str) -> None:
    state = torch.load(ckpt_path, map_location="cpu")
    for key in ("vae_state_dict", "model_state_dict", "state_dict"):
        if isinstance(state, dict) and key in state and isinstance(state[key], dict):
            state = state[key]
            break
    if any(k.startswith("module.") for k in state):
        state = {k.replace("module.", "", 1): v for k, v in state.items()}
    vae.load_state_dict(state, strict=True)


def build_visualization_models(
    cfg: Dict[str, Any],
    dataset_path: str,
    reference_demo: str,
    ckpt_path: str,
    device: torch.device,
):
    img_height, img_width = image_size_from_demo(dataset_path, reference_demo)
    spl_cfg = build_splatter_config(cfg, img_height, img_width)
    splatter_channels = splatter_channels_from_config(cfg, spl_cfg)
    vae = build_splattervae(cfg, img_height, img_width, splatter_channels)
    load_vae_state_dict(vae, ckpt_path)

    converter = VAESplatterToGaussians(spl_cfg)
    vae.to(device).eval()
    converter.to(device).eval()
    return vae, converter, spl_cfg
