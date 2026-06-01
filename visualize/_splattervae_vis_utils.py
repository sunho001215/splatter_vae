from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import h5py
import numpy as np
import torch
import yaml

from models.vae import CodebookConfig, SplatterVAE


def image_to_tensor(img_rgb: np.ndarray) -> torch.Tensor:
    img = img_rgb.astype(np.float32) / 255.0
    return torch.from_numpy(img * 2.0 - 1.0).permute(2, 0, 1)


def invert_4x4(m: np.ndarray) -> np.ndarray:
    r = m[:3, :3]
    t = m[:3, 3:4]
    out = np.eye(4, dtype=np.float32)
    out[:3, :3] = r.T
    out[:3, 3:4] = -(r.T @ t)
    return out


def sort_demo_keys(keys: Sequence[str]) -> List[str]:
    def key_fn(x: str) -> int:
        try:
            return int(x.replace("demo", ""))
        except Exception:
            return 10**9

    return sorted(keys, key=key_fn)


def load_cfg(cfg_path: str | Path) -> Dict[str, Any]:
    with open(cfg_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def infer_demo_image_size(dataset_path: str | Path, demo_key: str) -> Tuple[int, int]:
    with h5py.File(dataset_path, "r") as f:
        demo = f["data"][demo_key]
        cam_names = json.loads(demo.attrs["camera_names"])
        if not cam_names:
            raise ValueError(f"Demo '{demo_key}' does not contain any cameras.")
        first_cam = cam_names[0]
        h, w = demo["obs"][f"{first_cam}_rgb"].shape[1:3]
    return int(h), int(w)


def load_camera_mats(demo: h5py.Group, cam_names: Sequence[str]) -> Dict[str, Dict[str, np.ndarray]]:
    intr = np.asarray(demo["camera_params"]["intrinsics"], dtype=np.float32)
    world_t_cam = np.asarray(demo["camera_params"]["extrinsics_world_T_cam"], dtype=np.float32)
    mats: Dict[str, Dict[str, np.ndarray]] = {}
    s = np.diag([1.0, -1.0, -1.0, 1.0]).astype(np.float32)
    for idx, cam in enumerate(cam_names):
        k = intr[idx]
        w2c_gl = invert_4x4(world_t_cam[idx])
        w2c = s @ w2c_gl
        c2w = invert_4x4(w2c)
        mats[cam] = {"K": k, "w2c": w2c, "c2w": c2w}
    return mats


def _default_splatter_channels(max_sh_degree: int = 1, num_gaussians_per_pixel: int = 5) -> int:
    if max_sh_degree not in (0, 1):
        raise ValueError("SplatterVAE visualizers support max_sh_degree in {0, 1}.")
    k = int(num_gaussians_per_pixel)
    sh_rest = 0 if int(max_sh_degree) == 0 else 3 * (((int(max_sh_degree) + 1) ** 2) - 1)
    channels_per_gaussian = 1 + 3 + 1 + 3 + 4 + 3 + sh_rest
    return int(k * channels_per_gaussian)


def _normalize_vit_cfg(
    vit_cfg: Dict[str, Any],
    model_cfg: Dict[str, Any],
    img_height: int,
    img_width: int,
) -> Dict[str, Any]:
    vit_cfg = dict(vit_cfg)
    patch_size = vit_cfg.get("patch_size")

    if patch_size is None:
        gh = model_cfg.get("grid_tokens_h")
        gw = model_cfg.get("grid_tokens_w")
        if gh is None or gw is None:
            raise KeyError("Config must define vit.patch_size or legacy model.grid_tokens_h/grid_tokens_w.")
        patch_h = int(img_height) // int(gh)
        patch_w = int(img_width) // int(gw)
        if patch_h != patch_w:
            raise ValueError(
                "SplatterVAE now expects a square patch size. "
                f"Got derived patch sizes {(patch_h, patch_w)} from image size {(img_height, img_width)}."
            )
        patch_size = patch_h
    elif isinstance(patch_size, (list, tuple)):
        if len(patch_size) != 2:
            raise ValueError(f"Expected vit.patch_size to have length 2, got {patch_size}.")
        patch_h, patch_w = int(patch_size[0]), int(patch_size[1])
        if patch_h != patch_w:
            raise ValueError(
                "SplatterVAE now expects a square patch size. "
                f"Got vit.patch_size={(patch_h, patch_w)}."
            )
        patch_size = patch_h
    else:
        patch_size = int(patch_size)

    vit_cfg["patch_size"] = patch_size

    for key in ("selected_layers", "decoder_selected_layers"):
        if key in vit_cfg and isinstance(vit_cfg[key], list):
            vit_cfg[key] = tuple(int(v) for v in vit_cfg[key])

    return vit_cfg


def _select_checkpoint_subdict(
    state: Dict[str, Any],
    preferred_keys: Sequence[str] = ("vae_state_dict", "model_state_dict", "state_dict"),
) -> Dict[str, torch.Tensor]:
    state_dict: Dict[str, Any] = state
    for key in preferred_keys:
        if key in state and isinstance(state[key], dict):
            state_dict = state[key]
            break

    return {
        k[len("module."):] if k.startswith("module.") else k: v
        for k, v in state_dict.items()
    }


def build_splatter_vae(
    cfg: Dict[str, Any],
    img_height: int,
    img_width: int,
    ckpt_path: str | Path,
    device: torch.device,
) -> SplatterVAE:
    model_cfg = dict(cfg.get("model", {}))
    vit_cfg = dict(cfg.get("vit", {}))
    if not vit_cfg:
        vit_cfg = dict(cfg.get("swin", {}))
    if not vit_cfg:
        raise KeyError("Config must contain a `vit` section for the current SplatterVAE.")

    vit_cfg = _normalize_vit_cfg(vit_cfg, model_cfg, img_height=img_height, img_width=img_width)

    cb_cfg = dict(cfg.get("codebook", {}))
    inv_cb_cfg = CodebookConfig(**cb_cfg.get("invariant", {}))
    dep_cb_cfg = CodebookConfig(**cb_cfg.get("dependent", {}))

    splatter_cfg = dict(cfg.get("splatter", {}))
    splatter_model_cfg = dict(splatter_cfg.get("model", {}))
    splatter_channels = int(
        splatter_cfg.get(
            "splatter_channels",
            _default_splatter_channels(
                max_sh_degree=int(splatter_model_cfg.get("max_sh_degree", 1)),
                num_gaussians_per_pixel=int(splatter_model_cfg.get("num_gaussians_per_pixel", 5)),
            ),
        )
    )

    vae = SplatterVAE(
        vit_cfg=vit_cfg,
        invariant_cb_config=inv_cb_cfg,
        dependent_cb_config=dep_cb_cfg,
        img_height=int(img_height),
        img_width=int(img_width),
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

    state = torch.load(ckpt_path, map_location="cpu")
    state_dict = _select_checkpoint_subdict(state if isinstance(state, dict) else {"state_dict": state})
    vae.load_state_dict(state_dict, strict=True)
    vae.to(device).eval()
    return vae


def build_vae_for_demo(
    cfg: Dict[str, Any],
    dataset_path: str | Path,
    demo_key: str,
    ckpt_path: str | Path,
    device: torch.device,
) -> Tuple[SplatterVAE, Tuple[int, int]]:
    img_height, img_width = infer_demo_image_size(dataset_path, demo_key)
    vae = build_splatter_vae(
        cfg,
        img_height=img_height,
        img_width=img_width,
        ckpt_path=ckpt_path,
        device=device,
    )
    return vae, (img_height, img_width)


def build_render_models_for_demo(
    cfg: Dict[str, Any],
    dataset_path: str | Path,
    demo_key: str,
    ckpt_path: str | Path,
    device: torch.device,
):
    from models.splatter import SplatterConfig, SplatterDataConfig, SplatterModelConfig, VAESplatterToGaussians

    img_height, img_width = infer_demo_image_size(dataset_path, demo_key)
    spl_data_cfg_dict = dict(cfg.get("splatter", {}).get("data", {}))
    spl_data_cfg_dict["img_height"] = int(img_height)
    spl_data_cfg_dict["img_width"] = int(img_width)

    spl_cfg = SplatterConfig(
        data=SplatterDataConfig(**spl_data_cfg_dict),
        model=SplatterModelConfig(**dict(cfg.get("splatter", {}).get("model", {}))),
    )
    converter = VAESplatterToGaussians(spl_cfg).to(device).eval()
    vae = build_splatter_vae(
        cfg,
        img_height=img_height,
        img_width=img_width,
        ckpt_path=ckpt_path,
        device=device,
    )
    return vae, converter, spl_cfg, (img_height, img_width)
