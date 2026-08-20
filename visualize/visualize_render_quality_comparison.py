from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from visualize.metaworld_camera_utils import choose_demo_key, dataset_camera_mats, read_rgb, sort_demo_keys
from visualize.render_quality_models import (
    LPIPSMetric,
    ReViWoRenderer,
    SinCroRenderer,
    SplatterVAERenderer,
    mse_uint8,
    ssim_torch,
)

METHOD_ORDER = ["sincro", "splattervae", "reviwo"]
METHOD_LABELS = {"sincro": "SinCro", "splattervae": "SplatterVAE", "reviwo": "ReViWo"}
FIGURE_COLUMNS = ["Input Image", "Ground Truth", "SinCro", "SplatterVAE", "ReViWo"]


@dataclass(frozen=True)
class SceneSpec:
    demo: str
    timestep: int
    source_cam: str
    target_cam: str


def parse_methods(value: str) -> list[str]:
    methods = [item.strip().lower() for item in value.split(",") if item.strip()]
    unknown = [m for m in methods if m not in METHOD_ORDER]
    if unknown:
        raise ValueError(f"Unknown methods {unknown}; expected some of {METHOD_ORDER}.")
    return [m for m in METHOD_ORDER if m in set(methods)]


def parse_scene_spec(value: str) -> SceneSpec:
    parts = [part.strip() for part in value.split(",")]
    if len(parts) != 4:
        raise ValueError(
            "Each --scenes entry must be 'demo,timestep,source_cam,target_cam', "
            f"got {value!r}."
        )
    demo, timestep, source_cam, target_cam = parts
    return SceneSpec(demo=demo, timestep=int(timestep), source_cam=source_cam, target_cam=target_cam)


def parse_scenes(value: str | None) -> list[SceneSpec]:
    if not value:
        return []
    return [parse_scene_spec(chunk) for chunk in value.split(";") if chunk.strip()]


def default_scenes(
    dataset_path: str,
    requested_demo: str | None,
    timestep: int,
    source_cam: str,
    target_cam: str,
    num_scenes: int,
) -> list[SceneSpec]:
    with h5py.File(dataset_path, "r") as f:
        demo_keys = sort_demo_keys(list(f["data"].keys()))
        if not demo_keys:
            raise ValueError(f"No demos found in {dataset_path}.")
        if requested_demo is not None:
            if requested_demo not in demo_keys:
                raise ValueError(f"Demo {requested_demo!r} not found in {dataset_path}.")
            demo_keys = [requested_demo]

        scenes: list[SceneSpec] = []
        for idx in range(max(1, int(num_scenes))):
            demo_key = demo_keys[idx % len(demo_keys)]
            demo = f["data"][demo_key]
            cam_a, cam_b = source_cam, target_cam
            if f"{cam_a}_rgb" not in demo["obs"] or f"{cam_b}_rgb" not in demo["obs"]:
                available = sorted(k[:-4] for k in demo["obs"].keys() if k.endswith("_rgb"))
                if len(available) < 2:
                    raise ValueError(f"Demo {demo_key} needs at least two RGB cameras, found {available}.")
                cam_a, cam_b = available[0], available[1]
            t_len = int(demo["obs"][f"{cam_a}_rgb"].shape[0])
            if idx == 0:
                t = int(timestep)
            else:
                t = int(round((idx + 1) * (t_len - 1) / (max(1, int(num_scenes)) + 1)))
            scenes.append(SceneSpec(demo=demo_key, timestep=max(0, min(t, t_len - 1)), source_cam=cam_a, target_cam=cam_b))
    return scenes


def load_sincro_sequence(demo: h5py.Group, cam: str, timestep: int, length: int, use_history: bool) -> np.ndarray:
    if not use_history:
        return read_rgb(demo, cam, timestep)
    t = int(timestep)
    start = max(0, t - int(length) + 1)
    frames = [read_rgb(demo, cam, idx) for idx in range(start, t + 1)]
    while len(frames) < int(length):
        frames.insert(0, frames[0])
    return np.stack(frames, axis=0)


def format_metric(value: float | None, fmt: str = ".4f") -> str:
    if value is None or not np.isfinite(value):
        return "n/a"
    return format(float(value), fmt)


def format_metric_block(metrics: dict[str, float | None]) -> str:
    return (
        f"LPIPS {format_metric(metrics.get('lpips'))}\n"
        f"SSIM {format_metric(metrics.get('ssim'))}\n"
        f"MSE {format_metric(metrics.get('mse'))}"
    )


def load_scene(dataset_path: str, scene: SceneSpec) -> dict:
    with h5py.File(dataset_path, "r") as f:
        if scene.demo not in f["data"]:
            raise ValueError(f"Demo {scene.demo!r} not found in dataset.")
        demo = f["data"][scene.demo]
        obs = demo["obs"]
        for cam in (scene.source_cam, scene.target_cam):
            if f"{cam}_rgb" not in obs:
                raise ValueError(f"Camera {cam!r} not found in demo {scene.demo}.")
        t_len = int(obs[f"{scene.source_cam}_rgb"].shape[0])
        if scene.timestep < 0 or scene.timestep >= t_len:
            raise ValueError(f"Scene {scene}: timestep must be in [0, {t_len - 1}].")
        source = read_rgb(demo, scene.source_cam, scene.timestep)
        target = read_rgb(demo, scene.target_cam, scene.timestep)
        cv_mats = dataset_camera_mats(demo, [scene.source_cam, scene.target_cam], opencv=True)
        gl_mats = dataset_camera_mats(demo, [scene.source_cam, scene.target_cam], opencv=False)
    return {"source": source, "target": target, "cv_mats": cv_mats, "gl_mats": gl_mats}


def save_figure(scene_rows: list[dict], out_path: Path) -> None:
    if not scene_rows:
        raise ValueError("No scene rows were produced.")
    plt.rcParams.update({"font.family": "DejaVu Sans", "axes.titleweight": "bold"})
    n = len(scene_rows)
    fig, axes = plt.subplots(n, len(FIGURE_COLUMNS), figsize=(15.8, max(3.0, 3.05 * n)), squeeze=False)
    for ax, title in zip(axes[0], FIGURE_COLUMNS):
        ax.set_title(title, fontsize=13.5, pad=9)

    for row_idx, row in enumerate(scene_rows):
        scene: SceneSpec = row["scene"]
        images = [row["input"], row["gt"], row["predictions"].get("sincro"), row["predictions"].get("splattervae"), row["predictions"].get("reviwo")]
        for col_idx, img in enumerate(images):
            ax = axes[row_idx, col_idx]
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_linewidth(0.8)
                spine.set_color("#d1d5db")
            if img is None:
                ax.imshow(np.full_like(row["gt"], 244, dtype=np.uint8))
                ax.text(0.5, 0.5, "not run", ha="center", va="center", transform=ax.transAxes, fontsize=11, color="#6b7280")
            else:
                ax.imshow(np.asarray(img, dtype=np.uint8))
            if col_idx >= 2:
                method = METHOD_ORDER[col_idx - 2]
                metrics = row["metrics"].get(method)
                if metrics is not None:
                    ax.text(
                        0.5,
                        -0.13,
                        format_metric_block(metrics),
                        transform=ax.transAxes,
                        ha="center",
                        va="top",
                        fontsize=9.6,
                        linespacing=1.25,
                        color="#111827",
                    )

        axes[row_idx, 0].set_ylabel(
            f"Scene {row_idx + 1}\n{scene.demo}\nt={scene.timestep}\n{scene.source_cam}->{scene.target_cam}",
            fontsize=10.2,
            fontweight="bold",
            rotation=0,
            labelpad=46,
            va="center",
            ha="right",
        )

    fig.subplots_adjust(top=0.91, bottom=0.08, left=0.105, right=0.992, wspace=0.055, hspace=0.42)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=260, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare reconstruction/rendering quality in a 3-row paper-style layout.")
    parser.add_argument("--dataset", required=True, help="Multi-view HDF5 dataset.")
    parser.add_argument("--demo", default=None)
    parser.add_argument("--timestep", type=int, default=0)
    parser.add_argument("--source_cam", default="cam0")
    parser.add_argument("--target_cam", default="cam1")
    parser.add_argument("--num_scenes", type=int, default=3, help="Default number of scene rows when --scenes is omitted.")
    parser.add_argument(
        "--scenes",
        default=None,
        help="Semicolon-separated scene specs: 'demo,timestep,source_cam,target_cam;demo,timestep,source_cam,target_cam'.",
    )
    parser.add_argument("--methods", default="splattervae,sincro,reviwo", help="Comma-separated methods to run.")
    parser.add_argument("--splatter_config", default="config/splattervae/metaworld/temporal-no-segmentation-mask/button-press-wall.yaml")
    parser.add_argument("--splatter_ckpt", default=None)
    parser.add_argument("--sincro_config", default="agents/drqv2/config/sincro/button-press-wall.yaml")
    parser.add_argument("--sincro_ckpt", default=None, help="Full SinCro NeRF checkpoint.")
    parser.add_argument("--reviwo_config", default="agents/drqv2/config/reviwo/button-press-wall.yaml")
    parser.add_argument("--reviwo_ckpt", default=None)
    parser.add_argument("--sincro_sequence_from_dataset", action="store_true")
    parser.add_argument("--out", default="outputs/render_quality_comparison.png")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--require_lpips", action="store_true")
    args = parser.parse_args()

    methods = parse_methods(args.methods)
    scenes = parse_scenes(args.scenes)
    if not scenes:
        scenes = default_scenes(args.dataset, args.demo, args.timestep, args.source_cam, args.target_cam, args.num_scenes)
    if len(scenes) != 3:
        print(f"Warning: producing {len(scenes)} scene rows; pass exactly three --scenes entries for a 3-row figure.")

    device = torch.device(args.device)
    lpips_metric = LPIPSMetric(device, required=bool(args.require_lpips))

    renderers: dict[str, object] = {}
    if "splattervae" in methods:
        renderers["splattervae"] = SplatterVAERenderer(args.splatter_config, args.dataset, scenes[0].demo, args.splatter_ckpt, device)
    if "sincro" in methods:
        renderers["sincro"] = SinCroRenderer(args.sincro_config, args.sincro_ckpt, device)
    if "reviwo" in methods:
        renderers["reviwo"] = ReViWoRenderer(args.reviwo_config, args.reviwo_ckpt, device)

    scene_rows: list[dict] = []
    metrics_out: dict[str, object] = {"dataset": args.dataset, "scenes": []}

    for scene in scenes:
        data = load_scene(args.dataset, scene)
        source = data["source"]
        target = data["target"]
        cv_mats = data["cv_mats"]
        gl_mats = data["gl_mats"]
        preds: dict[str, np.ndarray] = {}
        metrics: dict[str, dict[str, float | None]] = {}
        notes: dict[str, str] = {}

        if "sincro" in renderers:
            renderer = renderers["sincro"]
            assert isinstance(renderer, SinCroRenderer)
            with h5py.File(args.dataset, "r") as f:
                demo = f["data"][scene.demo]
                seq = load_sincro_sequence(
                    demo,
                    scene.source_cam,
                    scene.timestep,
                    renderer.args.time_interval,
                    args.sincro_sequence_from_dataset,
                )
            preds["sincro"] = renderer.render_from_source(
                seq,
                gl_mats[scene.target_cam]["K"],
                gl_mats[scene.target_cam]["c2w"],
                height=target.shape[0],
                width=target.shape[1],
            )
            notes["sincro"] = "dataset frame history" if args.sincro_sequence_from_dataset else "source frame repeated over time window"

        if "splattervae" in renderers:
            renderer = renderers["splattervae"]
            assert isinstance(renderer, SplatterVAERenderer)
            preds["splattervae"] = renderer.render_from_source(
                source,
                cv_mats[scene.source_cam]["K"],
                cv_mats[scene.source_cam]["c2w"],
                cv_mats[scene.target_cam]["K"],
                cv_mats[scene.target_cam]["w2c"],
            )

        if "reviwo" in renderers:
            renderer = renderers["reviwo"]
            assert isinstance(renderer, ReViWoRenderer)
            preds["reviwo"] = renderer.reconstruct_with_target_view(source, target)
            notes["reviwo"] = "target image supplies the view branch only"

        for method, pred in preds.items():
            metrics[method] = {
                "mse": mse_uint8(pred, target),
                "ssim": ssim_torch(pred, target, device=device if device.type == "cuda" else torch.device("cpu")),
                "lpips": lpips_metric(pred, target),
            }

        scene_rows.append({"scene": scene, "input": source, "gt": target, "predictions": preds, "metrics": metrics, "notes": notes})
        metrics_out["scenes"].append(
            {
                "demo": scene.demo,
                "timestep": scene.timestep,
                "source_cam": scene.source_cam,
                "target_cam": scene.target_cam,
                "methods": {
                    METHOD_LABELS[m]: (metrics[m] | {"note": notes.get(m, "")})
                    for m in METHOD_ORDER
                    if m in metrics
                },
            }
        )

    out_path = Path(args.out)
    save_figure(scene_rows, out_path)
    metrics_path = out_path.with_suffix(".json")
    metrics_path.write_text(json.dumps(metrics_out, indent=2))
    print(f"Saved: {out_path}")
    print(f"Saved: {metrics_path}")


if __name__ == "__main__":
    main()
