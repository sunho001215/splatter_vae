from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
    
import argparse
import json
import math
from pathlib import Path
from typing import Dict, List, Tuple

import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml

from models.splatter import (
    SplatterConfig,
    SplatterDataConfig,
    SplatterModelConfig,
    VAESplatterToGaussians,
    render_predicted,
)
from models.vae import CodebookConfig, InvariantDependentSplatterVAE


# ---------- small helpers ----------
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


def sort_demo_keys(keys: List[str]) -> List[str]:
    def key_fn(x: str) -> int:
        try:
            return int(x.replace("demo", ""))
        except Exception:
            return 10**9
    return sorted(keys, key=key_fn)


def to_01(x: torch.Tensor) -> np.ndarray:
    x = x.detach().cpu().float()
    if x.min() < 0:
        x = (x + 1.0) * 0.5
    return x.clamp(0.0, 1.0).permute(1, 2, 0).numpy()


def normalize_preview(x: torch.Tensor) -> np.ndarray:
    y = x.detach().cpu().float()
    c = min(3, y.shape[0])
    y = y[:c]
    if c < 3:
        y = torch.cat([y, y[-1:].repeat(3 - c, 1, 1)], dim=0)
    y = y - y.amin(dim=(1, 2), keepdim=True)
    y = y / (y.amax(dim=(1, 2), keepdim=True) + 1e-8)
    return y.permute(1, 2, 0).numpy()


def load_cfg(cfg_path: str) -> dict:
    with open(cfg_path, "r") as f:
        return yaml.safe_load(f)


def load_camera_mats(demo: h5py.Group, cam_names: List[str]) -> Dict[str, Dict[str, np.ndarray]]:
    intr = np.asarray(demo["camera_params"]["intrinsics"], dtype=np.float32)
    world_t_cam = np.asarray(demo["camera_params"]["extrinsics_world_T_cam"], dtype=np.float32)
    mats = {}
    s = np.diag([1.0, -1.0, -1.0, 1.0]).astype(np.float32)
    for idx, cam in enumerate(cam_names):
        k = intr[idx]
        w2c_gl = invert_4x4(world_t_cam[idx])
        w2c = s @ w2c_gl
        c2w = invert_4x4(w2c)
        mats[cam] = {"K": k, "w2c": w2c, "c2w": c2w}
    return mats


def build_models(cfg: dict, dataset_path: str, reference_demo: str, ckpt_path: str, device: torch.device):
    with h5py.File(dataset_path, "r") as f:
        first_cam = json.loads(f["data"][reference_demo].attrs["camera_names"])[0]
        h, w = f["data"][reference_demo]["obs"][f"{first_cam}_rgb"].shape[1:3]

    spl_data = SplatterDataConfig(**cfg.get("splatter", {}).get("data", {}))
    spl_data.img_height = h
    spl_data.img_width = w
    spl_model = SplatterModelConfig(**cfg.get("splatter", {}).get("model", {}))
    spl_cfg = SplatterConfig(data=spl_data, model=spl_model)

    converter = VAESplatterToGaussians(spl_cfg)
    splatter_channels = converter.num_splatter_channels()

    inv_cb = CodebookConfig(**cfg.get("codebook", {}).get("invariant", {}))
    dep_cb = CodebookConfig(**cfg.get("codebook", {}).get("dependent", {}))
    model_cfg = cfg.get("model", {})
    swin_cfg = dict(cfg.get("swin", {}))

    if "patch_size" not in swin_cfg:
        gh = int(model_cfg.get("grid_tokens_h", 8))
        gw = int(model_cfg.get("grid_tokens_w", 8))
        swin_cfg["patch_size"] = [h // gh, w // gw]
    elif isinstance(swin_cfg["patch_size"], int):
        ps = int(swin_cfg["patch_size"])
        swin_cfg["patch_size"] = [ps, ps]

    vae = InvariantDependentSplatterVAE(
        swin_cfg=swin_cfg,
        invariant_cb_config=inv_cb,
        dependent_cb_config=dep_cb,
        img_height=h,
        img_width=w,
        splatter_channels=splatter_channels,
        fusion_style=model_cfg.get("fusion_style", "cat"),
        use_dependent_vq=bool(model_cfg.get("use_dependent_vq", True)),
        is_dependent_ae=bool(model_cfg.get("is_dependent_ae", False)),
        use_invariant_vq=bool(model_cfg.get("use_invariant_vq", True)),
        is_invariant_ae=bool(model_cfg.get("is_invariant_ae", False)),
    )

    state = torch.load(ckpt_path, map_location="cpu")
    state_dict = state.get("vae_state_dict", state)
    if any(k.startswith("module.") for k in state_dict):
        state_dict = {k.replace("module.", "", 1): v for k, v in state_dict.items()}
    vae.load_state_dict(state_dict, strict=True)

    vae.to(device).eval()
    converter.to(device).eval()
    return vae, converter, spl_cfg


def validate_source(demo: h5py.Group, cam: str, timestep: int, tag: str):
    cams = json.loads(demo.attrs["camera_names"])
    if cam not in cams:
        raise ValueError(f"{tag}: camera '{cam}' not found in this demo. Available: {cams}")
    t_max = demo["obs"][f"{cam}_rgb"].shape[0]
    if not (0 <= timestep < t_max):
        raise ValueError(f"{tag}: timestep must be in [0, {t_max - 1}]")


def read_source(demo: h5py.Group, cam: str, timestep: int) -> np.ndarray:
    return np.asarray(demo["obs"][f"{cam}_rgb"][timestep], dtype=np.uint8)


def choose_default_cam(cams: List[str], fallback: str | None = None) -> str:
    if fallback is not None and fallback in cams:
        return fallback
    return cams[0]


def save_summary(inv_img, dep_img, render_dep, save_path: Path,
                 inv_desc: str, dep_desc: str, dep_view_desc: str):
    titles = [
        f"z_inv source\n{inv_desc}",
        f"z_dep source\n{dep_desc}",
        "decoded splatter preview",
        f"rendered at z_dep viewpoint\n{dep_view_desc}",
    ]
    imgs = [inv_img, dep_img, render_dep]
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    for ax, title, img in zip(axes, titles, imgs):
        ax.imshow(np.clip(img, 0.0, 1.0))
        ax.set_title(title, fontsize=10)
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(save_path, dpi=200)
    plt.close(fig)


def save_all_views(renders: np.ndarray, cam_names: List[str], save_path: Path):
    n = len(cam_names)
    cols = min(3, n)
    rows = math.ceil(n / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 4 * rows))
    axes = np.asarray(axes).reshape(-1)
    for ax in axes[n:]:
        ax.axis("off")
    for ax, cam, img in zip(axes, cam_names, renders):
        ax.imshow(np.clip(img, 0.0, 1.0))
        ax.set_title(cam)
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(save_path, dpi=200)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(
        description="Mix z_inv and z_dep from different timesteps/episodes to stress-test disentanglement."
    )
    ap.add_argument("--config", required=True, help="Training YAML config.")
    ap.add_argument("--dataset", required=True, help="Training HDF5 dataset.")
    ap.add_argument("--ckpt", required=True, help="SplatterVAE checkpoint.")
    ap.add_argument("--out_dir", default="view_swap_outputs")

    # z_inv source
    ap.add_argument("--demo_inv", default=None, help="Demo used to extract z_inv (default: first demo)")
    ap.add_argument("--timestep_inv", type=int, default=0, help="Timestep used to extract z_inv")
    ap.add_argument("--cam_inv", default=None, help="Camera used to extract z_inv")

    # z_dep source
    ap.add_argument("--demo_dep", default=None, help="Demo used to extract z_dep (default: same as demo_inv)")
    ap.add_argument("--timestep_dep", type=int, default=None, help="Timestep used to extract z_dep (default: different from timestep_inv when possible)")
    ap.add_argument("--cam_dep", default=None, help="Camera used to extract z_dep")

    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    cfg = load_cfg(args.config)

    with h5py.File(args.dataset, "r") as f:
        demo_keys = sort_demo_keys(list(f["data"].keys()))
        demo_inv_key = args.demo_inv or demo_keys[0]
        demo_dep_key = args.demo_dep or demo_inv_key

        demo_inv = f["data"][demo_inv_key]
        demo_dep = f["data"][demo_dep_key]
        inv_cams = json.loads(demo_inv.attrs["camera_names"])
        dep_cams = json.loads(demo_dep.attrs["camera_names"])

        cam_inv = choose_default_cam(inv_cams, args.cam_inv)
        if args.cam_dep is not None:
            cam_dep = choose_default_cam(dep_cams, args.cam_dep)
        else:
            # Prefer a different camera when the two sources are from the same episode.
            candidates = [c for c in dep_cams if c != cam_inv]
            cam_dep = candidates[0] if candidates else dep_cams[0]

        t_max_dep = demo_dep["obs"][f"{cam_dep}_rgb"].shape[0]
        if args.timestep_dep is None:
            timestep_dep = 1 if (demo_dep_key == demo_inv_key and args.timestep_inv == 0 and t_max_dep > 1) else 0
            if demo_dep_key == demo_inv_key and timestep_dep == args.timestep_inv and t_max_dep > 1:
                timestep_dep = min(args.timestep_inv + 1, t_max_dep - 1)
        else:
            timestep_dep = args.timestep_dep

        validate_source(demo_inv, cam_inv, args.timestep_inv, "z_inv source")
        validate_source(demo_dep, cam_dep, timestep_dep, "z_dep source")

        if demo_inv_key == demo_dep_key and args.timestep_inv == timestep_dep:
            raise ValueError(
                "Use different sources: z_inv and z_dep must come from different timesteps or different demos."
            )

    # Model/image size is taken from the invariant-source demo.
    vae, converter, spl_cfg = build_models(cfg, args.dataset, demo_inv_key, args.ckpt, device)

    with h5py.File(args.dataset, "r") as f:
        demo_inv = f["data"][demo_inv_key]
        demo_dep = f["data"][demo_dep_key]
        dep_cams = json.loads(demo_dep.attrs["camera_names"])
        dep_cam_mats = load_camera_mats(demo_dep, dep_cams)

        img_inv_u8 = read_source(demo_inv, cam_inv, args.timestep_inv)
        img_dep_u8 = read_source(demo_dep, cam_dep, timestep_dep)

    x_inv = image_to_tensor(img_inv_u8).unsqueeze(0).to(device)
    x_dep = image_to_tensor(img_dep_u8).unsqueeze(0).to(device)

    with torch.no_grad():
        z_inv, _, _, _, _ = vae.encode(
            x_inv,
            deterministic_invariant=True,
            deterministic_dependent=True,
        )
        _, _, z_dep, _, _ = vae.encode(
            x_dep,
            deterministic_invariant=True,
            deterministic_dependent=True,
        )

        # The output should keep the content/scene cues from z_inv and adopt the viewpoint cues from z_dep.
        splatter = vae.decode(z_inv, z_dep)

        eye = torch.eye(4, dtype=torch.float32, device=device).unsqueeze(0)
        quat_identity = torch.tensor([[1.0, 0.0, 0.0, 0.0]], dtype=torch.float32, device=device)
        k_dep = torch.from_numpy(dep_cam_mats[cam_dep]["K"]).unsqueeze(0).to(device)

        gaussian_pc = converter(
            splatter,
            source_cameras_view_to_world=eye,
            source_cv2wT_quat=quat_identity,
            intrinsics=k_dep,
            activate_output=True,
        )

        # Anchor the decoded splatter to the z_dep camera frame, then render to all training viewpoints from demo_dep.
        world_views, ks = [], []
        for cam in dep_cams:
            t_dep_to_target = dep_cam_mats[cam]["w2c"] @ dep_cam_mats[cam_dep]["c2w"]
            world_views.append(torch.from_numpy(t_dep_to_target))
            ks.append(torch.from_numpy(dep_cam_mats[cam]["K"]))

        world_view_transform = torch.stack(world_views, dim=0).unsqueeze(0).to(device)
        intrinsics = torch.stack(ks, dim=0).unsqueeze(0).to(device)
        bg = (
            torch.ones(3, dtype=torch.float32, device=device)
            if spl_cfg.data.white_background
            else torch.zeros(3, dtype=torch.float32, device=device)
        )

        renders = render_predicted(
            pc=gaussian_pc,
            world_view_transform=world_view_transform,
            intrinsics=intrinsics,
            bg_color=bg,
            cfg=spl_cfg,
        )["render"][0]

    render_np = [to_01(r) for r in renders]
    dep_render = render_np[dep_cams.index(cam_dep)]

    stem = (
        f"inv_{demo_inv_key}_t{args.timestep_inv:04d}_{cam_inv}"
        f"__dep_{demo_dep_key}_t{timestep_dep:04d}_{cam_dep}"
    )
    summary_path = out_dir / f"{stem}_summary.png"
    all_views_path = out_dir / f"{stem}_all_dep_demo_views.png"

    save_summary(
        inv_img=img_inv_u8.astype(np.float32) / 255.0,
        dep_img=img_dep_u8.astype(np.float32) / 255.0,
        render_dep=dep_render,
        save_path=summary_path,
        inv_desc=f"{demo_inv_key}, t={args.timestep_inv}, {cam_inv}",
        dep_desc=f"{demo_dep_key}, t={timestep_dep}, {cam_dep}",
        dep_view_desc=f"{demo_dep_key}, {cam_dep}",
    )
    save_all_views(render_np, dep_cams, all_views_path)

    print(f"z_inv source : demo={demo_inv_key}, timestep={args.timestep_inv}, camera={cam_inv}")
    print(f"z_dep source : demo={demo_dep_key}, timestep={timestep_dep}, camera={cam_dep}")
    print(f"Saved: {summary_path}")
    print(f"Saved: {all_views_path}")


if __name__ == "__main__":
    main()
