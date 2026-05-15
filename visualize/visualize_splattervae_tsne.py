from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import argparse
import json
from typing import List, Tuple

import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml
from sklearn.manifold import TSNE

from models.vae import SplatterVAE
from visualize.splattervae_common import (
    build_splatter_config,
    build_splattervae,
    image_size_from_demo,
    load_vae_state_dict,
    splatter_channels_from_config,
)


# ---------- small helpers ----------
def image_to_tensor(img_rgb: np.ndarray) -> torch.Tensor:
    img = img_rgb.astype(np.float32) / 255.0
    return torch.from_numpy(img * 2.0 - 1.0).permute(2, 0, 1)


def sort_demo_keys(keys: List[str]) -> List[str]:
    def key_fn(x: str) -> int:
        try:
            return int(x.replace("demo", ""))
        except Exception:
            return 10**9
    return sorted(keys, key=key_fn)


def safe_perplexity(n: int, wanted: int) -> int:
    if n <= 3:
        return max(1, n - 1)
    return max(2, min(wanted, n - 1, (n - 1) // 3))


def run_tsne(x: np.ndarray, perplexity: int, seed: int) -> np.ndarray:
    tsne = TSNE(
        n_components=2,
        perplexity=safe_perplexity(len(x), perplexity),
        init="pca",
        learning_rate="auto",
        random_state=seed,
    )
    return tsne.fit_transform(x)


def load_cfg(cfg_path: str) -> dict:
    with open(cfg_path, "r") as f:
        return yaml.safe_load(f)


def build_vae(cfg: dict, dataset_path: str, demo_key: str, ckpt_path: str, device: torch.device):
    h, w = image_size_from_demo(dataset_path, demo_key)
    spl_cfg = build_splatter_config(cfg, h, w)
    splatter_channels = splatter_channels_from_config(cfg, spl_cfg)
    vae = build_splattervae(cfg, h, w, splatter_channels)
    load_vae_state_dict(vae, ckpt_path)
    vae.to(device).eval()
    return vae, (h, w)


def encode_image_states(vae: SplatterVAE, img_u8: np.ndarray, device: torch.device):
    x = image_to_tensor(img_u8).unsqueeze(0).to(device)
    with torch.no_grad():
        # t-SNE should inspect the current compact state interface, not the old
        # token/codebook path. The invariant feature matches DrQ-v2 input.
        z_inv = vae.encode_invariant_pooled_state(x)
        z_dep = vae.encode_dependent_pooled_state(x)
    return z_inv[0].detach().cpu().numpy(), z_dep[0].detach().cpu().numpy()


# ---------- plots ----------
def plot_dep_tsne(emb2d: np.ndarray, cam_labels: List[str], save_path: Path):
    plt.figure(figsize=(8, 7))
    uniq = list(dict.fromkeys(cam_labels))
    cmap = plt.cm.get_cmap("tab10", len(uniq))
    for i, cam in enumerate(uniq):
        idx = [k for k, c in enumerate(cam_labels) if c == cam]
        pts = emb2d[idx]
        plt.scatter(pts[:, 0], pts[:, 1], s=16, alpha=0.8, label=cam, color=cmap(i))
    plt.title("t-SNE of view-dependent pooled states (all cameras, one trajectory)")
    plt.xlabel("t-SNE 1")
    plt.ylabel("t-SNE 2")
    plt.legend(markerscale=1.5)
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()


def plot_inv_tsne(emb2d: np.ndarray, cam_labels: List[str], times: List[int], cams: Tuple[str, str], save_path: Path):
    markers = {cams[0]: "o", cams[1]: "^"}
    t = np.asarray(times, dtype=np.float32)

    vmin = float(t.min()) if t.size else 0.0
    vmax = float(t.max()) if t.size else 1.0
    if vmax <= vmin:
        vmax = vmin + 1.0
    norm = plt.Normalize(vmin=vmin, vmax=vmax)

    fig, ax = plt.subplots(figsize=(8, 7))

    for cam in cams:
        idx = np.asarray([k for k, c in enumerate(cam_labels) if c == cam], dtype=np.int64)
        if idx.size == 0:
            continue
        pts = emb2d[idx]
        ax.scatter(
            pts[:, 0],
            pts[:, 1],
            s=24,
            alpha=0.9,
            marker=markers[cam],
            c=t[idx],
            cmap="viridis",
            norm=norm,
            label=cam,
        )

    sm = plt.cm.ScalarMappable(cmap="viridis", norm=norm)
    sm.set_array([])
    fig.colorbar(sm, ax=ax, label="trajectory timestep")

    ax.set_title("t-SNE of view-invariant pooled states (two cameras, same trajectory)")
    ax.set_xlabel("t-SNE 1")
    ax.set_ylabel("t-SNE 2")
    ax.legend(markerscale=1.5)

    fig.tight_layout()
    fig.savefig(save_path, dpi=200)
    plt.close(fig)


# ---------- main ----------
def main():
    ap = argparse.ArgumentParser(description="Visualize compact SplatterVAE pooled states with t-SNE.")
    ap.add_argument("--config", required=True, help="Training YAML config.")
    ap.add_argument("--dataset", required=True, help="Training HDF5 dataset.")
    ap.add_argument("--ckpt", required=True, help="SplatterVAE checkpoint.")
    ap.add_argument("--out_dir", default="tsne_outputs")
    ap.add_argument("--demo", default=None, help="demo key (default: first demo)")
    ap.add_argument("--cam_a", default=None, help="first camera for invariant plot")
    ap.add_argument("--cam_b", default=None, help="second camera for invariant plot")
    ap.add_argument("--max_steps", type=int, default=None, help="optional trajectory truncation")
    ap.add_argument("--perplexity", type=int, default=30)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    torch.manual_seed(int(args.seed))
    cfg = load_cfg(args.config)

    with h5py.File(args.dataset, "r") as f:
        demo_keys = sort_demo_keys(list(f["data"].keys()))
        demo_key = args.demo or demo_keys[0]
        demo = f["data"][demo_key]
        cam_names = json.loads(demo.attrs["camera_names"])
        t_len = int(demo.attrs.get("num_samples", demo["obs"][f"{cam_names[0]}_rgb"].shape[0]))
        if args.max_steps is not None:
            t_len = min(t_len, args.max_steps)

    cam_a = args.cam_a or cam_names[0]
    cam_b = args.cam_b or cam_names[1]
    if cam_a == cam_b:
        raise ValueError("cam_a and cam_b must be different")

    vae, _ = build_vae(cfg, args.dataset, demo_key, args.ckpt, device)

    dep_feats, dep_cams = [], []
    inv_feats, inv_cams, inv_times = [], [], []

    with h5py.File(args.dataset, "r") as f:
        obs = f["data"][demo_key]["obs"]
        for t in range(t_len):
            # view-dependent: all cameras across the trajectory
            for cam in cam_names:
                img = np.asarray(obs[f"{cam}_rgb"][t], dtype=np.uint8)
                _, z_dep = encode_image_states(vae, img, device)
                dep_feats.append(z_dep)
                dep_cams.append(cam)

            # view-invariant: two chosen cameras, same trajectory and same timesteps
            for cam in (cam_a, cam_b):
                img = np.asarray(obs[f"{cam}_rgb"][t], dtype=np.uint8)
                z_inv, _ = encode_image_states(vae, img, device)
                inv_feats.append(z_inv)
                inv_cams.append(cam)
                inv_times.append(t)

    dep_feats = np.stack(dep_feats, axis=0)
    inv_feats = np.stack(inv_feats, axis=0)

    dep_2d = run_tsne(dep_feats, args.perplexity, args.seed)
    inv_2d = run_tsne(inv_feats, args.perplexity, args.seed)

    dep_path = out_dir / f"{demo_key}_view_dependent_tsne.png"
    inv_path = out_dir / f"{demo_key}_{cam_a}_{cam_b}_view_invariant_tsne.png"
    plot_dep_tsne(dep_2d, dep_cams, dep_path)
    plot_inv_tsne(inv_2d, inv_cams, inv_times, (cam_a, cam_b), inv_path)

    print(f"Saved: {dep_path}")
    print(f"Saved: {inv_path}")


if __name__ == "__main__":
    main()
