from __future__ import annotations

"""Export temporal SplatterVAE t-SNE plots and publication input assets.

This is a visualization-only entry point.  It deliberately uses the model's
deterministic all-patch inference API and never imports or changes the training
loop.  Encoder trajectories use every adjacent (t, t+1, t+2) window in the
first demonstration.  Publication assets use one high-motion triplet selected
from the temporal strides used during training.
"""

import argparse
import gc
import json
import sys
from pathlib import Path
from typing import Any, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
from sklearn.manifold import TSNE

from dataset.dataloader import _coerce_seg_ids, _segmentation_mask_from_selectors
from visualize._splattervae_vis_utils import load_cfg, sort_demo_keys
from visualize.splattervae_common import (
    adapt_config_to_checkpoint,
    build_splatter_config,
    build_splattervae,
    image_size_from_demo,
    load_vae_state_dict,
    splatter_channels_from_config,
)


ENVIRONMENTS = (
    "button-press-wall",
    "drawer-open",
    "door-open",
    "hammer",
    "peg-unplug-side",
    "handle-press",
    "plate-slide",
    "stick-push",
)
TRAINING_STRIDES = (3, 6, 9)
FLOW_COLOR_MAX_MAGNITUDE_PIXELS = 32.0
DEPTH_LOW_PERCENTILE = 1.0
DEPTH_HIGH_PERCENTILE = 99.0


def _first_demo(dataset_path: Path) -> str:
    with h5py.File(dataset_path, "r") as handle:
        demos = sort_demo_keys(list(handle["data"].keys()))
    if not demos:
        raise ValueError(f"Dataset contains no demonstrations: {dataset_path}")
    return demos[0]


def _build_vae(config_path: Path, dataset_path: Path, checkpoint_path: Path, demo_key: str, device: torch.device):
    cfg = adapt_config_to_checkpoint(load_cfg(config_path), str(checkpoint_path))
    height, width = image_size_from_demo(str(dataset_path), demo_key)
    splatter_cfg = build_splatter_config(cfg, height, width)
    vae = build_splattervae(
        cfg,
        height,
        width,
        splatter_channels_from_config(cfg, splatter_cfg),
    )
    load_vae_state_dict(vae, str(checkpoint_path))
    return vae.to(device).eval(), cfg


def _latest_temporal_checkpoint(checkpoint_root: Path, environment: str) -> Path | None:
    candidates = sorted((checkpoint_root / environment).glob("**/step_*.pth"))
    return candidates[-1] if candidates else None


def _safe_perplexity(num_samples: int, requested: float) -> float:
    if num_samples < 3:
        raise ValueError(f"t-SNE needs at least three samples, got {num_samples}.")
    return float(max(2.0, min(float(requested), num_samples - 1.0, (num_samples - 1.0) / 3.0)))


def _tsne(features: np.ndarray, perplexity: float, seed: int) -> np.ndarray:
    normalized = F.normalize(torch.from_numpy(features).float(), dim=-1, eps=1.0e-6).numpy()
    return TSNE(
        n_components=2,
        perplexity=_safe_perplexity(len(normalized), perplexity),
        init="pca",
        learning_rate="auto",
        metric="euclidean",
        random_state=int(seed),
    ).fit_transform(normalized)


@torch.inference_mode()
def _encode_adjacent_windows(
    vae,
    dataset_path: Path,
    demo_key: str,
    cameras: Sequence[str],
    batch_size: int,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    invariant: list[torch.Tensor] = []
    dependent: list[torch.Tensor] = []
    camera_ids: list[int] = []
    starts: list[int] = []

    with h5py.File(dataset_path, "r") as handle:
        obs = handle["data"][demo_key]["obs"]
        for camera_id, camera in enumerate(cameras):
            rgb = np.asarray(obs[f"{camera}_rgb"], dtype=np.uint8)
            num_windows = int(rgb.shape[0]) - 2
            if num_windows <= 0:
                continue
            for first in range(0, num_windows, batch_size):
                chunk_starts = np.arange(first, min(first + batch_size, num_windows), dtype=np.int64)
                windows = np.stack([rgb[t : t + 3] for t in chunk_starts], axis=0)
                images = torch.from_numpy(windows).to(device=device, non_blocking=True)
                images = images.permute(0, 1, 4, 2, 3).unsqueeze(2)
                images = images.float().div_(255.0).mul_(2.0).sub_(1.0)
                encoded = vae.inference_features(images)
                invariant.append(encoded["s_inv"].detach().cpu())
                dependent.append(encoded["z_dep_all"][:, 0].detach().cpu())
                camera_ids.extend([camera_id] * len(chunk_starts))
                starts.extend(chunk_starts.tolist())

    if not invariant:
        raise ValueError(f"No adjacent three-frame windows in {dataset_path}:{demo_key}.")
    return (
        torch.cat(invariant).numpy(),
        torch.cat(dependent).numpy(),
        np.asarray(camera_ids, dtype=np.int16),
        np.asarray(starts, dtype=np.int32),
    )


def _plot_embedding(
    embedding: np.ndarray,
    camera_ids: np.ndarray,
    starts: np.ndarray,
    cameras: Sequence[str],
    title: str,
    output_path: Path,
) -> None:
    markers = ("o", "^", "s", "D", "P", "X", "v", "<", ">", "h")
    norm = matplotlib.colors.Normalize(vmin=float(starts.min()), vmax=float(max(starts.max(), starts.min() + 1)))
    cmap = plt.get_cmap("viridis")
    fig, axis = plt.subplots(figsize=(9.2, 7.5), dpi=160)
    for camera_id, camera in enumerate(cameras):
        index = np.flatnonzero(camera_ids == camera_id)
        order = index[np.argsort(starts[index])]
        points = embedding[order]
        axis.plot(points[:, 0], points[:, 1], color="0.72", linewidth=0.55, alpha=0.55, zorder=1)
        axis.scatter(
            points[:, 0],
            points[:, 1],
            c=starts[order],
            cmap=cmap,
            norm=norm,
            marker=markers[camera_id % len(markers)],
            s=28,
            edgecolors="black",
            linewidths=0.25,
            alpha=0.88,
            label=camera,
            zorder=2,
        )
    scalar = matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap)
    colorbar = fig.colorbar(scalar, ax=axis, pad=0.02)
    colorbar.set_label("Sliding-window start timestep")
    axis.set_title(title, fontsize=15, fontweight="bold")
    axis.set_xlabel("t-SNE 1")
    axis.set_ylabel("t-SNE 2")
    axis.grid(alpha=0.2, linewidth=0.5)
    axis.legend(loc="best", frameon=False, ncol=2)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def export_tsne(
    environment: str,
    dataset_path: Path,
    config_path: Path,
    checkpoint_path: Path,
    output_dir: Path,
    batch_size: int,
    perplexity: float,
    seed: int,
    device: torch.device,
) -> dict[str, Any]:
    demo_key = _first_demo(dataset_path)
    vae, cfg = _build_vae(config_path, dataset_path, checkpoint_path, demo_key, device)
    if not vae.temporal_modeling:
        raise ValueError(f"Checkpoint is not a temporal SplatterVAE model: {checkpoint_path}")
    with h5py.File(dataset_path, "r") as handle:
        available = json.loads(handle["data"][demo_key].attrs["camera_names"])
    configured = cfg.get("dataset", {}).get("views", available)
    cameras = [camera for camera in configured if camera in available]
    inv, dep, camera_ids, starts = _encode_adjacent_windows(
        vae, dataset_path, demo_key, cameras, batch_size, device
    )
    inv_embedding = _tsne(inv, perplexity, seed)
    dep_embedding = _tsne(dep, perplexity, seed)

    tsne_dir = output_dir / environment / "tsne"
    _plot_embedding(
        inv_embedding,
        camera_ids,
        starts,
        cameras,
        f"{environment}: view-invariant encoder",
        tsne_dir / "view_invariant_tsne.png",
    )
    _plot_embedding(
        dep_embedding,
        camera_ids,
        starts,
        cameras,
        f"{environment}: view-dependent encoder",
        tsne_dir / "view_dependent_tsne.png",
    )
    np.savez_compressed(
        tsne_dir / "embeddings_and_features.npz",
        invariant_features=inv.astype(np.float16),
        dependent_features=dep.astype(np.float16),
        invariant_tsne=inv_embedding.astype(np.float32),
        dependent_tsne=dep_embedding.astype(np.float32),
        camera_ids=camera_ids,
        camera_names=np.asarray(cameras),
        window_start=starts,
        window_indices=np.stack((starts, starts + 1, starts + 2), axis=1),
    )
    result = {
        "status": "created",
        "demo": demo_key,
        "checkpoint": str(checkpoint_path),
        "num_adjacent_windows": int(len(starts)),
        "num_cameras": len(cameras),
        "view_invariant_plot": str(tsne_dir / "view_invariant_tsne.png"),
        "view_dependent_plot": str(tsne_dir / "view_dependent_tsne.png"),
    }
    del vae
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return result


def _top_motion_score(flow: np.ndarray, fraction: float = 0.02) -> np.ndarray:
    magnitude = np.linalg.norm(np.nan_to_num(flow.astype(np.float32), copy=False), axis=-1)
    flattened = magnitude.reshape(magnitude.shape[0], -1)
    count = max(1, int(round(flattened.shape[1] * fraction)))
    partition = np.partition(flattened, flattened.shape[1] - count, axis=1)
    return partition[:, -count:].mean(axis=1)


def _select_training_triplet(demo: h5py.Group, cameras: Sequence[str]) -> dict[str, Any]:
    num_frames = int(demo.attrs["num_samples"])
    best: dict[str, Any] | None = None
    for camera in cameras:
        flow_group = demo["optical_flow"][camera]
        for stride in TRAINING_STRIDES:
            gap_name = f"gap_{stride}"
            double_name = f"gap_{2 * stride}"
            if gap_name not in flow_group or double_name not in flow_group:
                continue
            pair = np.asarray(flow_group[gap_name], dtype=np.float16)
            accumulated = np.asarray(flow_group[double_name], dtype=np.float16)
            pair_scores = _top_motion_score(pair)
            accumulated_scores = _top_motion_score(accumulated)
            for start in range(max(0, num_frames - 2 * stride)):
                if start + stride >= len(pair_scores) or start >= len(accumulated_scores):
                    continue
                score01 = float(pair_scores[start])
                score12 = float(pair_scores[start + stride])
                score02 = float(accumulated_scores[start])
                # Reward visible motion in both consecutive transitions rather
                # than selecting a triplet with only one isolated large jump.
                score = 0.35 * score01 + 0.35 * score12 + 0.15 * score02 + 0.15 * min(score01, score12)
                if best is None or score > best["selection_score"]:
                    best = {
                        "camera": camera,
                        "stride": stride,
                        "start": start,
                        "indices": [start, start + stride, start + 2 * stride],
                        "selection_score": score,
                        "top_2pct_motion_pixels": {"01": score01, "12": score12, "02": score02},
                    }
    if best is None:
        raise ValueError(f"No valid training triplet with strides {TRAINING_STRIDES} in {demo.name}.")
    return best


def _save_rgb(path: Path, array: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.asarray(array, dtype=np.uint8)).save(path)


def _depth_limits(depths: np.ndarray) -> tuple[float, float]:
    valid = depths[np.isfinite(depths) & (depths > 0.0)]
    if valid.size == 0:
        return 0.0, 1.0
    low, high = np.percentile(valid, [DEPTH_LOW_PERCENTILE, DEPTH_HIGH_PERCENTILE])
    if not np.isfinite(high) or high <= low:
        high = low + 1.0
    return float(low), float(high)


def _colorize_depth(depth: np.ndarray, low: float, high: float) -> np.ndarray:
    valid = np.isfinite(depth) & (depth > 0.0)
    normalized = np.clip((depth.astype(np.float32) - low) / max(high - low, 1.0e-6), 0.0, 1.0)
    rgb = plt.get_cmap("turbo_r")(normalized)[..., :3]
    rgb[~valid] = 0.0
    return np.rint(rgb * 255.0).astype(np.uint8)


def _colorize_flow(flow: np.ndarray, max_magnitude: float = FLOW_COLOR_MAX_MAGNITUDE_PIXELS) -> np.ndarray:
    flow = np.nan_to_num(flow.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    x, y = flow[..., 0], flow[..., 1]
    hue = np.mod(np.arctan2(y, x) / (2.0 * np.pi) + 1.0, 1.0)
    saturation = np.clip(np.sqrt(x * x + y * y) / max(max_magnitude, 1.0e-6), 0.0, 1.0)
    hsv = np.stack((hue, saturation, np.ones_like(hue)), axis=-1)
    rgb = matplotlib.colors.hsv_to_rgb(hsv)
    return np.rint(rgb * 255.0).astype(np.uint8)


def _save_depth_legend(path: Path, low: float, high: float) -> None:
    gradient = np.linspace(0.0, 1.0, 512, dtype=np.float32)[None]
    fig, axis = plt.subplots(figsize=(6.0, 1.1), dpi=150)
    axis.imshow(gradient, aspect="auto", cmap="turbo_r", extent=(low, high, 0, 1))
    axis.set_yticks([])
    axis.set_xlabel("metric depth (m)")
    fig.tight_layout()
    fig.savefig(path, dpi=240, bbox_inches="tight")
    plt.close(fig)


def _save_flow_legend(path: Path, max_magnitude: float) -> None:
    size = 401
    coordinate = np.linspace(-max_magnitude, max_magnitude, size, dtype=np.float32)
    x, y = np.meshgrid(coordinate, coordinate)
    flow = np.stack((x, y), axis=-1)
    rgb = _colorize_flow(flow, max_magnitude)
    radius = np.sqrt(x * x + y * y)
    rgb[radius > max_magnitude] = 255
    _save_rgb(path, rgb)


def _segmentation_selectors(config_path: Path, environment: str, demo_key: str, dataset_path: Path):
    selected = load_cfg(config_path)["dataset"]["selected_seg_ids"]
    if isinstance(selected, dict):
        for key in (environment, demo_key, dataset_path.stem, dataset_path.name, "default", "*"):
            if key in selected:
                selected = selected[key]
                break
        else:
            raise KeyError(f"No segmentation selector matches {environment}:{demo_key}.")
    selectors = _coerce_seg_ids(selected)
    if not selectors:
        raise ValueError(f"Empty segmentation selector for {environment}:{demo_key}.")
    return selectors


def export_figure_assets(
    environment: str,
    dataset_path: Path,
    config_path: Path,
    output_dir: Path,
) -> dict[str, Any]:
    demo_key = _first_demo(dataset_path)
    asset_dir = output_dir / environment / "figure_assets"
    asset_dir.mkdir(parents=True, exist_ok=True)
    selectors = _segmentation_selectors(config_path, environment, demo_key, dataset_path)
    with h5py.File(dataset_path, "r") as handle:
        demo = handle["data"][demo_key]
        cameras = json.loads(demo.attrs["camera_names"])
        selection = _select_training_triplet(demo, cameras)
        camera = selection["camera"]
        stride = int(selection["stride"])
        start = int(selection["start"])
        indices = selection["indices"]
        obs = demo["obs"]
        rgbs = np.asarray(obs[f"{camera}_rgb"][indices], dtype=np.uint8)
        depths = np.asarray(obs[f"{camera}_depth"][indices], dtype=np.float32)
        segmentations = np.asarray(obs[f"{camera}_seg"][indices], dtype=np.int32)
        segmentation_types = np.asarray(obs[f"{camera}_seg_type"][indices], dtype=np.int32)
        masks = _segmentation_mask_from_selectors(segmentations, segmentation_types, selectors)
        flow_group = demo["optical_flow"][camera]
        flows = {
            "01": np.asarray(flow_group[f"gap_{stride}"][start], dtype=np.float32),
            "12": np.asarray(flow_group[f"gap_{stride}"][start + stride], dtype=np.float32),
            "02": np.asarray(flow_group[f"gap_{2 * stride}"][start], dtype=np.float32),
        }

    depth_low, depth_high = _depth_limits(depths)
    for timestep in range(3):
        _save_rgb(asset_dir / f"rgb_t{timestep}.png", rgbs[timestep])
        _save_rgb(
            asset_dir / f"segmentation_roi_t{timestep}.png",
            np.repeat((masks[timestep].astype(np.uint8) * 255)[..., None], 3, axis=-1),
        )
        _save_rgb(
            asset_dir / f"rgb_roi_t{timestep}.png",
            np.where(masks[timestep, ..., None], rgbs[timestep], 0),
        )
        _save_rgb(
            asset_dir / f"depth_t{timestep}_color.png",
            _colorize_depth(depths[timestep], depth_low, depth_high),
        )
    for pair, flow in flows.items():
        _save_rgb(asset_dir / f"optical_flow_{pair}.png", _colorize_flow(flow))
    _save_depth_legend(asset_dir / "depth_colormap_legend.png", depth_low, depth_high)
    _save_flow_legend(asset_dir / "optical_flow_colorwheel.png", FLOW_COLOR_MAX_MAGNITUDE_PIXELS)

    metadata = {
        "environment": environment,
        "dataset": str(dataset_path),
        "demo": demo_key,
        **selection,
        "depth_colormap": "turbo_r (shared limits across t0/t1/t2; invalid depth is black)",
        "depth_limits_meters": [depth_low, depth_high],
        "flow_colormap": "HSV: hue=direction, saturation=magnitude, white=zero flow",
        "flow_max_magnitude_pixels": FLOW_COLOR_MAX_MAGNITUDE_PIXELS,
        "segmentation_selectors": [
            f"{obj_type}:{obj_id}" if obj_type is not None else str(obj_id)
            for obj_type, obj_id in selectors
        ],
        "roi_pixel_fraction": [float(mask.mean()) for mask in masks],
    }
    with (asset_dir / "metadata.json").open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)
    return {"status": "created", "directory": str(asset_dir), **metadata}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--environments", nargs="+", default=list(ENVIRONMENTS), choices=ENVIRONMENTS)
    parser.add_argument("--dataset-root", type=Path, default=Path("/home/ws/data/metaworld/pre-training"))
    parser.add_argument(
        "--config-root",
        type=Path,
        default=PROJECT_ROOT / "config" / "splattervae" / "metaworld" / "temporal",
    )
    parser.add_argument(
        "--checkpoint-root",
        type=Path,
        default=PROJECT_ROOT / "checkpoints" / "metaworld" / "encoders" / "SplatterVAE-temporal",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "visualization_outputs" / "splattervae_temporal_analysis",
    )
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--perplexity", type=float, default=30.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--skip-tsne", action="store_true")
    parser.add_argument("--skip-assets", action="store_true")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    manifest_path = args.output_dir / "manifest.json"
    manifest: dict[str, Any] = {
        "settings": {
            "adjacent_tsne_window": [0, 1, 2],
            "asset_training_strides": list(TRAINING_STRIDES),
            "flow_color_max_magnitude_pixels": FLOW_COLOR_MAX_MAGNITUDE_PIXELS,
            "seed": int(args.seed),
            "device": str(device),
        },
        "environments": {},
    }
    if manifest_path.exists():
        with manifest_path.open("r", encoding="utf-8") as handle:
            previous_manifest = json.load(handle)
        manifest["environments"].update(previous_manifest.get("environments", {}))
    for environment in args.environments:
        print(f"\n[{environment}]", flush=True)
        dataset_path = args.dataset_root / f"{environment}.hdf5"
        config_path = args.config_root / f"{environment}.yaml"
        record: dict[str, Any] = dict(manifest["environments"].get(environment, {}))
        try:
            if not args.skip_assets:
                record["figure_assets"] = export_figure_assets(
                    environment, dataset_path, config_path, args.output_dir
                )
                print(f"  figure assets: {record['figure_assets']['directory']}", flush=True)
        except Exception as error:
            record["figure_assets"] = {"status": "failed", "error": repr(error)}
            print(f"  figure assets failed: {error}", flush=True)

        if not args.skip_tsne:
            checkpoint_path = _latest_temporal_checkpoint(args.checkpoint_root, environment)
            if checkpoint_path is None:
                record["tsne"] = {
                    "status": "missing_checkpoint",
                    "searched_under": str(args.checkpoint_root / environment),
                }
                print("  t-SNE skipped: no temporal checkpoint found", flush=True)
            else:
                try:
                    record["tsne"] = export_tsne(
                        environment,
                        dataset_path,
                        config_path,
                        checkpoint_path,
                        args.output_dir,
                        int(args.batch_size),
                        float(args.perplexity),
                        int(args.seed),
                        device,
                    )
                    print(
                        f"  t-SNE: {record['tsne']['num_adjacent_windows']} adjacent windows",
                        flush=True,
                    )
                except Exception as error:
                    record["tsne"] = {"status": "failed", "error": repr(error)}
                    print(f"  t-SNE failed: {error}", flush=True)
        manifest["environments"][environment] = record
        with manifest_path.open("w", encoding="utf-8") as handle:
            json.dump(manifest, handle, indent=2)

    print(f"\nManifest: {manifest_path}", flush=True)


if __name__ == "__main__":
    main()
