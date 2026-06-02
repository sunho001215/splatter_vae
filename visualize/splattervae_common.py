from __future__ import annotations

import copy
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
                isotropic=bool(spl_cfg.model.isotropic),
                depth_parameterization=str(getattr(spl_cfg.model, "depth_parameterization", "absolute")),
            ),
        )
    )


def _checkpoint_state_dict(ckpt_path: str) -> Dict[str, torch.Tensor]:
    state = torch.load(ckpt_path, map_location="cpu")
    for key in ("vae_state_dict", "model_state_dict", "state_dict"):
        if isinstance(state, dict) and key in state and isinstance(state[key], dict):
            state = state[key]
            break
    if any(k.startswith("module.") for k in state):
        state = {k.replace("module.", "", 1): v for k, v in state.items()}
    return state


def adapt_config_to_checkpoint(cfg: Dict[str, Any], ckpt_path: str) -> Dict[str, Any]:
    """Return a visualization config whose architecture matches ``ckpt_path``.

    The active training config may have moved on, e.g. RGB-D/depth-prior, while
    an older checkpoint is RGB/absolute-depth.  Visualization should follow the
    checkpoint tensor shapes so loading is strict and the Gaussian splitter uses
    the right channel layout.
    """
    cfg = copy.deepcopy(cfg)
    state = _checkpoint_state_dict(ckpt_path)

    patch = state.get("invariant_encoder.patch_embed.proj.weight")
    if patch is not None and patch.ndim == 4:
        cfg.setdefault("vit", {})["in_chans"] = int(patch.shape[1])

    out_channels = None
    for key in (
        "decoder.output_conv.4.weight",
        "decoder.scratch.output_conv2.4.weight",
        "decoder.head.4.weight",
    ):
        weight = state.get(key)
        if weight is not None and weight.ndim >= 1:
            out_channels = int(weight.shape[0])
            break

    if out_channels is not None:
        splatter_cfg = cfg.setdefault("splatter", {})
        model_cfg = splatter_cfg.setdefault("model", {})
        max_sh_degree = int(model_cfg.get("max_sh_degree", 1))
        num_gaussians = int(model_cfg.get("num_gaussians_per_pixel", 5))
        isotropic = bool(model_cfg.get("isotropic", False))
        absolute_channels = default_splatter_channels(
            max_sh_degree=max_sh_degree,
            num_gaussians_per_pixel=num_gaussians,
            isotropic=isotropic,
            depth_parameterization="absolute",
        )
        depth_prior_channels = default_splatter_channels(
            max_sh_degree=max_sh_degree,
            num_gaussians_per_pixel=num_gaussians,
            isotropic=isotropic,
            depth_parameterization="depth_prior",
        )
        if out_channels == absolute_channels:
            model_cfg["depth_parameterization"] = "absolute"
        elif out_channels == depth_prior_channels:
            model_cfg["depth_parameterization"] = "depth_prior"
        splatter_cfg["splatter_channels"] = out_channels

    return cfg


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
    vae.load_state_dict(_checkpoint_state_dict(ckpt_path), strict=True)


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

    converter = VAESplatterToGaussians(spl_cfg)
    vae.to(device).eval()
    converter.to(device).eval()
    return vae, converter, spl_cfg
