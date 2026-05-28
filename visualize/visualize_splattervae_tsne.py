from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import numpy as np
import torch
import torch.nn.functional as F
import yaml
from sklearn.manifold import TSNE

from visualize.splattervae_common import (
    build_splatter_config,
    build_splattervae,
    image_size_from_demo,
    load_vae_state_dict,
    splatter_channels_from_config,
)


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


def load_cfg(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def build_vae(cfg: dict, dataset_path: str, demo_key: str, ckpt_path: str, device: torch.device):
    h, w = image_size_from_demo(dataset_path, demo_key)
    spl_cfg = build_splatter_config(cfg, h, w)
    vae = build_splattervae(cfg, h, w, splatter_channels_from_config(cfg, spl_cfg))
    load_vae_state_dict(vae, ckpt_path)
    vae.to(device).eval()
    return vae, (h, w)


def pooled_latent(z: torch.Tensor, mode: str) -> torch.Tensor:
    if mode == "flatten":
        return z.reshape(z.shape[0], -1)
    if mode == "mean":
        return z.mean(dim=1)
    raise ValueError(f"Unknown pool mode: {mode}")


@torch.no_grad()
def encode_trajectories(
    vae,
    dataset_path: str,
    demo_key: str,
    cameras: List[str],
    num_steps: int,
    batch_size: int,
    pool: str,
    device: torch.device,
) -> Tuple[np.ndarray, np.ndarray, List[str], List[int]]:
    inv_feats = []
    dep_feats = []
    labels = []
    times = []

    with h5py.File(dataset_path, "r") as f:
        obs = f["data"][demo_key]["obs"]
        items = [(cam, t) for cam in cameras for t in range(num_steps)]
        for start in range(0, len(items), batch_size):
            chunk = items[start : start + batch_size]
            x = torch.stack(
                [image_to_tensor(np.asarray(obs[f"{cam}_rgb"][t], dtype=np.uint8)) for cam, t in chunk],
                dim=0,
            ).to(device)
            z_inv, _, z_dep, _, _ = vae.encode(
                x,
                deterministic_invariant=True,
                deterministic_dependent=True,
            )
            inv_feats.append(pooled_latent(z_inv, pool).detach().cpu())
            dep_feats.append(pooled_latent(z_dep, pool).detach().cpu())
            labels.extend([cam for cam, _t in chunk])
            times.extend([int(t) for _cam, t in chunk])

    inv = torch.cat(inv_feats, dim=0).numpy()
    dep = torch.cat(dep_feats, dim=0).numpy()
    return inv, dep, labels, times


def preprocess_features(x: np.ndarray, normalize: str) -> np.ndarray:
    if normalize == "none":
        return x
    if normalize == "l2":
        x_t = torch.from_numpy(x).float()
        return F.normalize(x_t, dim=-1, eps=1e-6).numpy()
    raise ValueError(f"Unknown normalize mode: {normalize}")


def safe_perplexity(n: int, requested: int) -> int:
    if n <= 3:
        return max(1, n - 1)
    return max(2, min(int(requested), n - 1, (n - 1) // 3))


def run_tsne(x: np.ndarray, perplexity: int, seed: int, metric: str) -> np.ndarray:
    tsne = TSNE(
        n_components=2,
        perplexity=safe_perplexity(len(x), perplexity),
        init="pca" if metric == "euclidean" else "random",
        learning_rate="auto",
        metric=metric,
        random_state=seed,
    )
    return tsne.fit_transform(x)


def load_camera_samples(
    dataset_path: str,
    demo_key: str,
    cameras: List[str],
    sample_step: int,
) -> Dict[str, np.ndarray]:
    samples = {}
    with h5py.File(dataset_path, "r") as f:
        obs = f["data"][demo_key]["obs"]
        for cam in cameras:
            samples[cam] = np.asarray(obs[f"{cam}_rgb"][sample_step], dtype=np.uint8)
    return samples


def axis_limits(points: np.ndarray, pad_ratio: float = 0.08) -> Tuple[Tuple[float, float], Tuple[float, float]]:
    x_min, y_min = points.min(axis=0)
    x_max, y_max = points.max(axis=0)
    x_pad = max(1e-6, (x_max - x_min) * pad_ratio)
    y_pad = max(1e-6, (y_max - y_min) * pad_ratio)
    return (float(x_min - x_pad), float(x_max + x_pad)), (float(y_min - y_pad), float(y_max + y_pad))


def parse_camera_groups(cameras: List[str], spec: str | None) -> Dict[str, int]:
    """Map each camera to a color group.

    The group spec is a semicolon-separated list such as
    ``cam0,cam2,cam4;cam1,cam3,cam5``.  Cameras omitted from the spec get their
    own singleton group.
    """
    if not spec:
        return {cam: i for i, cam in enumerate(cameras)}

    group_by_cam: Dict[str, int] = {}
    for group_id, chunk in enumerate(spec.split(";")):
        members = [item.strip() for item in chunk.split(",") if item.strip()]
        for cam in members:
            if cam not in cameras:
                raise ValueError(f"Camera group references unknown camera {cam!r}; available={cameras}.")
            if cam in group_by_cam:
                raise ValueError(f"Camera {cam!r} appears in more than one camera group.")
            group_by_cam[cam] = group_id

    next_group = 0 if not group_by_cam else max(group_by_cam.values()) + 1
    for cam in cameras:
        if cam not in group_by_cam:
            group_by_cam[cam] = next_group
            next_group += 1
    return group_by_cam


def make_marker_map(cameras: List[str]) -> Dict[str, str]:
    markers = ["o", "^", "s", "D", "P", "X", "v", "<", ">", "h", "*", "p"]
    return {cam: markers[i % len(markers)] for i, cam in enumerate(cameras)}


def make_group_color_map(group_by_cam: Dict[str, int]) -> Dict[int, Tuple[float, float, float, float]]:
    group_ids = sorted(set(group_by_cam.values()))

    # Do not sample tab10/tab20 with ``N=len(group_ids)`` because with only two
    # groups that can pick visually similar endpoints.  Instead, use a
    # hand-ordered high-contrast sequence so group color scales remain clearly
    # separated, e.g. blue vs red/orange for two groups.
    high_contrast = [
        "#1f77b4",  # blue
        "#d62728",  # red
        "#2ca02c",  # green
        "#9467bd",  # purple
        "#ff7f0e",  # orange
        "#17becf",  # cyan
        "#8c564b",  # brown
        "#e377c2",  # pink
        "#7f7f7f",  # gray
        "#bcbd22",  # olive
    ]
    if len(group_ids) > len(high_contrast):
        tab20 = list(plt.get_cmap("tab20").colors)
        high_contrast.extend(matplotlib.colors.to_hex(c) for c in tab20)

    return {
        group_id: matplotlib.colors.to_rgba(high_contrast[i % len(high_contrast)])
        for i, group_id in enumerate(group_ids)
    }


def sort_cameras_by_group(cameras: List[str], group_by_cam: Dict[str, int]) -> List[str]:
    """Place cameras from the same group next to each other in the figure."""
    original_index = {cam: i for i, cam in enumerate(cameras)}
    return sorted(cameras, key=lambda cam: (group_by_cam[cam], original_index[cam]))


def make_group_timestep_cmap(base_color: Tuple[float, float, float, float]) -> matplotlib.colors.Colormap:
    base = np.asarray(base_color[:3], dtype=np.float32)
    white = np.ones(3, dtype=np.float32)
    black = np.zeros(3, dtype=np.float32)

    # Use a wider light-to-dark ramp so timesteps within each camera group are
    # easier to distinguish than with a subtle white-to-base gradient.
    start = white * 0.88 + base * 0.12
    mid = base
    end = black * 0.35 + base * 0.65
    colors = [
        tuple(start.tolist() + [1.0]),
        tuple(mid.tolist() + [1.0]),
        tuple(end.tolist() + [1.0]),
    ]
    return matplotlib.colors.LinearSegmentedColormap.from_list("group_timestep", colors)


def timestep_shaded_colors(
    base_color: Tuple[float, float, float, float],
    times: np.ndarray,
    t_min: float,
    t_max: float,
) -> np.ndarray:
    norm = matplotlib.colors.Normalize(vmin=t_min, vmax=t_max if t_max > t_min else t_min + 1.0)
    rgba = make_group_timestep_cmap(base_color)(norm(times.astype(np.float32)))
    rgba[:, 3] = 0.88
    return rgba


def add_group_colorbars(
    ax,
    group_colors: Dict[int, Tuple[float, float, float, float]],
    t_min: float,
    t_max: float,
) -> None:
    group_ids = sorted(group_colors)
    if not group_ids:
        return

    norm = matplotlib.colors.Normalize(vmin=t_min, vmax=t_max if t_max > t_min else t_min + 1.0)
    bar_height = min(28.0, 82.0 / len(group_ids))
    gap = 4.0
    total_height = len(group_ids) * bar_height + (len(group_ids) - 1) * gap
    top = 50.0 + total_height / 2.0

    for i, group_id in enumerate(group_ids):
        y0 = top - (i + 1) * bar_height - i * gap
        cax = inset_axes(
            ax,
            width="3.0%",
            height=f"{bar_height}%",
            loc="lower left",
            bbox_to_anchor=(1.025, y0 / 100.0, 1.0, 1.0),
            bbox_transform=ax.transAxes,
            borderpad=0.0,
        )
        sm = matplotlib.cm.ScalarMappable(norm=norm, cmap=make_group_timestep_cmap(group_colors[group_id]))
        cbar = ax.figure.colorbar(sm, cax=cax, orientation="vertical")
        cbar.outline.set_linewidth(0.6)
        cbar.ax.tick_params(labelsize=7, width=0.5, length=2)
        cbar.ax.set_title(f"G{group_id + 1}", fontsize=8, color=group_colors[group_id], pad=3)
    # Place the shared colorbar label just outside the colorbar tick labels.
    # 1.125 keeps it close without overlapping the stacked bars.
    ax.text(
        1.125,
        0.5,
        "timestep",
        transform=ax.transAxes,
        rotation=90,
        va="center",
        ha="center",
        fontsize=8,
        color="0.28",
        clip_on=False,
    )


def plot_embedding_panel(
    ax,
    emb: np.ndarray,
    labels: List[str],
    times: List[int],
    cameras: List[str],
    group_by_cam: Dict[str, int],
    group_colors: Dict[int, Tuple[float, float, float, float]],
    markers: Dict[str, str],
    title: str,
) -> None:
    labels_np = np.asarray(labels)
    times_np = np.asarray(times)
    t_min = float(times_np.min()) if len(times_np) else 0.0
    t_max = float(times_np.max()) if len(times_np) else 1.0

    add_group_colorbars(ax, group_colors, t_min, t_max)

    for cam in cameras:
        idx = np.where(labels_np == cam)[0]
        order = idx[np.argsort(times_np[idx])]
        pts = emb[order]
        cam_times = times_np[order]
        colors = timestep_shaded_colors(group_colors[group_by_cam[cam]], cam_times, t_min, t_max)
        ax.scatter(
            pts[:, 0],
            pts[:, 1],
            c=colors,
            s=34,
            marker=markers[cam],
            alpha=0.92,
            label=cam,
            edgecolors="black",
            linewidths=0.28,
        )

    xlim, ylim = axis_limits(emb)
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_title(title, fontsize=14, fontweight="bold", pad=10)
    ax.set_xlabel("t-SNE 1", fontsize=11)
    ax.set_ylabel("t-SNE 2", fontsize=11)
    ax.grid(True, linewidth=0.45, alpha=0.25)
    ax.tick_params(labelsize=9)
    for spine in ax.spines.values():
        spine.set_linewidth(0.8)
        spine.set_alpha(0.55)


def save_paper_figure(
    inv_emb: np.ndarray,
    dep_emb: np.ndarray,
    labels: List[str],
    times: List[int],
    cameras: List[str],
    samples: Dict[str, np.ndarray],
    group_by_cam: Dict[str, int],
    demo_key: str,
    sample_step: int,
    out_path: Path,
) -> None:
    num_cams = len(cameras)
    if num_cams < 2:
        raise ValueError("At least two cameras are required for the paper t-SNE figure.")

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "axes.titleweight": "bold",
            "axes.linewidth": 0.8,
            "figure.dpi": 160,
            "savefig.dpi": 300,
        }
    )
    group_colors = make_group_color_map(group_by_cam)
    display_cameras = sort_cameras_by_group(cameras, group_by_cam)
    markers = make_marker_map(display_cameras)

    fig_w = max(12.0, 2.15 * num_cams)
    fig = plt.figure(figsize=(fig_w, 7.8), constrained_layout=False)

    # Add a dedicated spacer column.  With two camera groups, place the spacer
    # exactly between the groups so the sample images are visually grouped too.
    group_ids = sorted(set(group_by_cam.values()))
    if len(group_ids) == 2:
        mid = sum(1 for cam in display_cameras if group_by_cam[cam] == group_ids[0])
        mid = max(1, min(mid, num_cams - 1))
    else:
        mid = max(1, num_cams // 2)
    width_ratios = [1.0] * mid + [0.34] + [1.0] * (num_cams - mid)
    grid = GridSpec(
        2,
        num_cams + 1,
        figure=fig,
        height_ratios=[1.05, 3.3],
        width_ratios=width_ratios,
        hspace=0.34,
        wspace=0.46,
    )

    for i, cam in enumerate(display_cameras):
        col = i if i < mid else i + 1
        ax_img = fig.add_subplot(grid[0, col])
        ax_img.imshow(samples[cam])
        ax_img.set_xticks([])
        ax_img.set_yticks([])
        cam_color = group_colors[group_by_cam[cam]]
        ax_img.set_title(cam, color=cam_color, fontsize=12, fontweight="bold", pad=6)
        for spine in ax_img.spines.values():
            spine.set_color(cam_color)
            spine.set_linewidth(2.0)

    ax_inv = fig.add_subplot(grid[1, :mid])
    ax_dep = fig.add_subplot(grid[1, mid + 1:])
    plot_embedding_panel(ax_inv, inv_emb, labels, times, display_cameras, group_by_cam, group_colors, markers, "View-Invariant Feature")
    plot_embedding_panel(ax_dep, dep_emb, labels, times, display_cameras, group_by_cam, group_colors, markers, "View-Dependent Feature")

    handles, legend_labels = ax_dep.get_legend_handles_labels()
    fig.legend(
        handles,
        legend_labels,
        loc="lower center",
        ncol=min(num_cams, 6),
        frameon=False,
        fontsize=13,
        markerscale=1.25,
        handletextpad=0.55,
        columnspacing=1.15,
        bbox_to_anchor=(0.5, 0.012),
    )
    fig.subplots_adjust(top=0.94, bottom=0.12, left=0.055, right=0.94)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Create a paper-style SplatterVAE t-SNE diagnostic figure.")
    parser.add_argument("--config", required=True, help="Training YAML config.")
    parser.add_argument("--dataset", required=True, help="Multi-view HDF5 dataset.")
    parser.add_argument("--ckpt", required=True, help="SplatterVAE checkpoint.")
    parser.add_argument("--out", default=None, help="Output image path. Defaults under paper_tsne_outputs/.")
    parser.add_argument("--demo", default=None, help="Demo key. Defaults to the first demo.")
    parser.add_argument("--max_steps", type=int, default=100, help="Maximum trajectory steps to plot.")
    parser.add_argument("--sample_step", type=int, default=None, help="Timestep used for camera sample images.")
    parser.add_argument("--pool", choices=["flatten", "mean"], default="flatten")
    parser.add_argument("--normalize", choices=["l2", "none"], default="l2")
    parser.add_argument("--metric", choices=["euclidean", "cosine"], default="euclidean")
    parser.add_argument("--perplexity", type=int, default=30)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument(
        "--camera_groups",
        default=None,
        help="Semicolon-separated camera groups for color hues, e.g. 'cam0,cam2,cam4;cam1,cam3,cam5'.",
    )
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    cfg = load_cfg(args.config)
    device = torch.device(args.device)

    with h5py.File(args.dataset, "r") as f:
        demo_keys = sort_demo_keys(list(f["data"].keys()))
        demo_key = args.demo or demo_keys[0]
        demo = f["data"][demo_key]
        cameras = json.loads(demo.attrs["camera_names"])
        camera_num = cfg.get("dataset", {}).get("camera_num", None)
        if camera_num is not None:
            cameras = cameras[: int(camera_num)]
        t_len = int(demo.attrs.get("num_samples", demo["obs"][f"{cameras[0]}_rgb"].shape[0]))
        num_steps = min(int(args.max_steps), t_len)

    if num_steps < 2:
        raise ValueError(f"Need at least two timesteps for trajectory plotting, got {num_steps}.")
    sample_step = args.sample_step if args.sample_step is not None else num_steps // 2
    sample_step = max(0, min(int(sample_step), t_len - 1))

    vae, _ = build_vae(cfg, args.dataset, demo_key, args.ckpt, device)
    inv_feats, dep_feats, labels, times = encode_trajectories(
        vae=vae,
        dataset_path=args.dataset,
        demo_key=demo_key,
        cameras=cameras,
        num_steps=num_steps,
        batch_size=int(args.batch_size),
        pool=args.pool,
        device=device,
    )
    inv_feats = preprocess_features(inv_feats, args.normalize)
    dep_feats = preprocess_features(dep_feats, args.normalize)

    inv_emb = run_tsne(inv_feats, args.perplexity, args.seed, args.metric)
    dep_emb = run_tsne(dep_feats, args.perplexity, args.seed, args.metric)
    samples = load_camera_samples(args.dataset, demo_key, cameras, sample_step)
    group_by_cam = parse_camera_groups(cameras, args.camera_groups)

    if args.out is None:
        out_path = Path("paper_tsne_outputs") / f"{demo_key}_splattervae_paper_tsne.png"
    else:
        out_path = Path(args.out)
    save_paper_figure(inv_emb, dep_emb, labels, times, cameras, samples, group_by_cam, demo_key, sample_step, out_path)
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
