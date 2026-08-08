from __future__ import annotations

import copy
from contextlib import nullcontext
import json
from dataclasses import fields
from typing import Any, Dict, Tuple

import h5py
import torch

from models.gaussian.motion import activate_motion_map
from models.gaussian.parameterization import (
    DirectSplatterToGaussians,
    SplatterConfig,
    SplatterDataConfig,
    SplatterModelConfig,
    default_splatter_channels,
)
from models.splattervae.model import SplatterVAE


def image_size_from_demo(dataset_path: str, demo_key: str) -> Tuple[int, int]:
    with h5py.File(dataset_path, "r") as f:
        first_cam = json.loads(f["data"][demo_key].attrs["camera_names"])[0]
        return tuple(f["data"][demo_key]["obs"][f"{first_cam}_rgb"].shape[1:3])


def _filter_dataclass_kwargs(values: Dict[str, Any], cls: type) -> Dict[str, Any]:
    allowed = {field.name for field in fields(cls)}
    return {key: value for key, value in values.items() if key in allowed}


def build_splatter_config(cfg: Dict[str, Any], img_height: int, img_width: int) -> SplatterConfig:
    spl_cfg = cfg.get("splatter", {})
    spl_data_cfg = dict(spl_cfg.get("data", {}))
    spl_model_cfg = dict(spl_cfg.get("model", {}))
    spl_data_cfg["img_height"] = int(img_height)
    spl_data_cfg["img_width"] = int(img_width)
    return SplatterConfig(
        data=SplatterDataConfig(**_filter_dataclass_kwargs(spl_data_cfg, SplatterDataConfig)),
        model=SplatterModelConfig(**_filter_dataclass_kwargs(spl_model_cfg, SplatterModelConfig)),
    )


def splatter_channels_from_config(cfg: Dict[str, Any], spl_cfg: SplatterConfig) -> int:
    return int(
        cfg.get("splatter", {}).get(
            "splatter_channels",
            default_splatter_channels(
                gaussians_per_pixel=int(spl_cfg.model.gaussians_per_pixel),
                max_sh_degree=int(spl_cfg.model.max_sh_degree),
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
    cfg.setdefault("model", {})["temporal_modeling"] = any(
        key.startswith("dense_motion_head.") for key in state
    )
    is_temporal_state = any(
        key.startswith("spatial_queries") or key.startswith("decoder_backbone.film_mlps")
        for key in state
    )

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

    if out_channels is not None and not is_temporal_state:
        splatter_cfg = cfg.setdefault("splatter", {})
        model_cfg = splatter_cfg.setdefault("model", {})
        max_sh_degree = int(model_cfg.get("max_sh_degree", 1))
        params_per_gaussian = default_splatter_channels(gaussians_per_pixel=1, max_sh_degree=max_sh_degree)
        if out_channels % params_per_gaussian == 0:
            model_cfg["gaussians_per_pixel"] = max(1, out_channels // params_per_gaussian)
        splatter_cfg["splatter_channels"] = out_channels

    return cfg


def build_splattervae(cfg: Dict[str, Any], img_height: int, img_width: int, splatter_channels: int) -> SplatterVAE:
    model_cfg = dict(cfg.get("model", {}))
    vit_cfg = dict(cfg.get("vit", {}))
    spl_model_cfg = cfg.get("splatter", {}).get("model", {})
    gaussians_per_pixel = int(spl_model_cfg.get("gaussians_per_pixel", 1))

    return SplatterVAE(
        vit_cfg=vit_cfg,
        img_height=img_height,
        img_width=img_width,
        splatter_channels=splatter_channels,
        dep_mask_eval=bool(model_cfg.get("dep_mask_eval", True)),
        dpt_features=int(vit_cfg.get("dpt_features", 256)),
        inv_tube_mask_ratio=float(model_cfg.get("inv_tube_mask_ratio", 0.50)),
        dep_mask_ratio=float(model_cfg.get("dep_mask_ratio", 0.75)),
        tube_mask_per_view=bool(model_cfg.get("tube_mask_per_view", True)),
        state_dim=int(model_cfg.get("state_dim", 256)),
        view_dim=model_cfg.get("view_dim", None),
        gaussians_per_pixel=gaussians_per_pixel,
        flow_patch_threshold_pixels=float(model_cfg.get("flow_patch_threshold_pixels", 0.5)),
        motion_translation_max=float(model_cfg.get("motion_translation_max", 0.5)),
        temporal_modeling=bool(model_cfg.get("temporal_modeling", True)),
    )


def load_vae_state_dict(vae: SplatterVAE, ckpt_path: str) -> None:
    vae.load_state_dict(_checkpoint_state_dict(ckpt_path), strict=True)


def load_converter_state_dict(converter: DirectSplatterToGaussians, ckpt_path: str) -> None:
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

    converter = DirectSplatterToGaussians(spl_cfg)
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
    converter: DirectSplatterToGaussians,
    images: torch.Tensor,
    intrinsics: torch.Tensor,
    source_c2w: torch.Tensor,
) -> tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
    sequence = fixed_window_from_single_image(images, vae.temporal_modeling)
    context = (
        torch.autocast(device_type="cuda", dtype=torch.bfloat16)
        if images.device.type == "cuda"
        else nullcontext()
    )
    with context:
        features = vae.inference_features(sequence)
        raw = vae.predict_raw_maps(features["s_inv"], features["z_dep_all"][:, 0])
    motion = (
        activate_motion_map(raw["raw_motion_map"].float(), vae.motion_translation_max)
        if vae.temporal_modeling else None
    )
    pc = converter(
        splatter_map=raw["raw_base_map"].float(),
        motion_map=motion,
        source_cameras_view_to_world=source_c2w.float(),
        intrinsics=intrinsics.float(),
    )
    return pc, features
