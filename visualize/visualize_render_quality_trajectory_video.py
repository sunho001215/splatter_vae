from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:
    import mujoco_mig_setup  # noqa: F401
except Exception:
    pass

import h5py
import mujoco
import numpy as np
import torch

from visualize.metaworld_camera_utils import (
    choose_demo_key,
    dataset_camera_mats,
    hstack_panels,
    labeled_panel,
    load_yaml,
    lookat_up_from_drq,
    make_metaworld_env,
    pose_mats,
    read_rgb,
    read_state,
    render_pose_with_renderer,
    set_env_state_from_flat,
    trajectory_poses,
    write_mp4,
    unwrap_mujoco,
)
from visualize.render_quality_models import SinCroRenderer, SplatterVAERenderer, mse_uint8, ssim_torch


def parse_methods(value: str) -> list[str]:
    return [item.strip().lower() for item in value.split(",") if item.strip()]


def parse_vec3(value: str | None) -> np.ndarray | None:
    if value is None or not str(value).strip():
        return None
    parts = [float(item.strip()) for item in str(value).split(",") if item.strip()]
    if len(parts) != 3:
        raise ValueError(f"Expected a comma-separated 3D vector, got {value!r}.")
    return np.asarray(parts, dtype=np.float64)


def load_sincro_sequence(demo: h5py.Group, cam: str, timestep: int, length: int, use_history: bool) -> np.ndarray:
    if not use_history:
        return read_rgb(demo, cam, timestep)
    t = int(timestep)
    start = max(0, t - int(length) + 1)
    frames = [read_rgb(demo, cam, idx) for idx in range(start, t + 1)]
    while len(frames) < int(length):
        frames.insert(0, frames[0])
    return np.stack(frames, axis=0)


def metric_subtitle(pred: np.ndarray, gt: np.ndarray | None, device: torch.device) -> str:
    if gt is None:
        return "disturbed target view"
    mse = mse_uint8(pred, gt)
    ssim = ssim_torch(pred, gt, device=device if device.type == "cuda" else torch.device("cpu"))
    return f"MSE {mse:.4f} | SSIM {ssim:.3f}"


def run_one_trajectory(args, traj_name: str, cfg: dict, source: np.ndarray, state: np.ndarray | None, cv_source_mats: dict, sincro_sequence: np.ndarray | None) -> dict:
    methods = parse_methods(args.methods)
    device = torch.device(args.device)
    h = int(cfg["env"].get("image_height", source.shape[0]))
    w = int(cfg["env"].get("image_width", source.shape[1]))
    lookat, _up = lookat_up_from_drq(cfg)
    orbit_center = parse_vec3(args.orbit_center)
    render_lookat = orbit_center if traj_name == "orbit" and orbit_center is not None else lookat
    poses = trajectory_poses(
        cfg,
        base_camera=args.base_camera,
        trajectory=traj_name,
        num_frames=int(args.num_frames),
        lateral_amplitude=float(args.lateral_amplitude),
        circular_azimuth_deg=float(args.circular_azimuth_deg),
        circular_elevation_deg=float(args.circular_elevation_deg),
        orbit_center=orbit_center,
        orbit_degrees=float(args.orbit_degrees),
    )

    gt_frames = None
    env = None
    renderer = None
    if bool(args.include_gt) and state is not None:
        env = make_metaworld_env(cfg, seed=int(args.seed))
        env.reset(seed=int(args.seed))
        set_env_state_from_flat(env, state)
        model, data = unwrap_mujoco(env)
        renderer = mujoco.Renderer(model, height=h, width=w)
        gt_frames = [render_pose_with_renderer(renderer, data, pose, render_lookat) for pose in poses]

    splatter = None
    splatter_pc = None
    if "splattervae" in methods:
        splatter = SplatterVAERenderer(args.splatter_config, args.dataset, args.demo_key, args.splatter_ckpt, device)
        splatter_pc = splatter.encode_source(source, cv_source_mats["K"], cv_source_mats["c2w"])

    sincro = None
    sincro_latent = None
    if "sincro" in methods:
        sincro = SinCroRenderer(args.sincro_config, args.sincro_ckpt, device)
        if sincro_sequence is None:
            sincro_sequence = source
        sincro_latent = sincro.encode_single_view(sincro_sequence)

    frames = []
    for idx, pose in enumerate(poses):
        mats = pose_mats(pose, h, w)
        gt = None if gt_frames is None else gt_frames[idx]
        panels = [labeled_panel(source, f"Input {args.source_cam}", f"{args.demo_key}, t={args.timestep}")]
        if gt is not None:
            panels.append(labeled_panel(gt, "MuJoCo GT", pose.name))
        if splatter is not None and splatter_pc is not None:
            pred = splatter.render_pc(splatter_pc, mats["K"], mats["w2c"])
            panels.append(labeled_panel(pred, "SplatterVAE", metric_subtitle(pred, gt, device)))
        if sincro is not None and sincro_latent is not None:
            pred = sincro.render_latent(sincro_latent, mats["K"], mats["c2w_gl"], h, w)
            panels.append(labeled_panel(pred, "SinCro", metric_subtitle(pred, gt, device)))
        frames.append(hstack_panels(panels, pad=8))

    if renderer is not None:
        renderer.close()
    if env is not None:
        env.close()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{args.demo_key}_t{int(args.timestep):06d}_{args.source_cam}_to_{args.base_camera}_{traj_name}.mp4"
    write_mp4(out_path, frames, fps=float(args.fps))
    return {
        "trajectory": traj_name,
        "path": str(out_path),
        "num_frames": len(frames),
        "gt_included": gt_frames is not None,
        "methods": methods,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate disturbed-viewpoint rendering-quality videos for SplatterVAE and SinCro.")
    parser.add_argument("--drq_config", required=True, help="DrQ-v2 config that defines env.cameras/lookat/up.")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--demo", default=None)
    parser.add_argument("--timestep", type=int, default=0)
    parser.add_argument("--source_cam", default="cam0")
    parser.add_argument("--base_camera", default="cam1", help="Training camera to perturb for the rendering viewpoint.")
    parser.add_argument("--trajectory", choices=["lateral", "circular", "orbit", "both"], default="both")
    parser.add_argument("--methods", default="splattervae,sincro")
    parser.add_argument("--splatter_config", default="config/metaworld/button-press-wall.yaml")
    parser.add_argument("--splatter_ckpt", default=None)
    parser.add_argument("--sincro_config", default=None, help="Defaults to --drq_config.")
    parser.add_argument("--sincro_ckpt", default=None, help="Full SinCro NeRF checkpoint.")
    parser.add_argument("--sincro_sequence_from_dataset", action="store_true")
    parser.add_argument("--num_frames", type=int, default=72)
    parser.add_argument("--fps", type=float, default=20.0)
    parser.add_argument("--lateral_amplitude", type=float, default=0.12)
    parser.add_argument("--circular_azimuth_deg", type=float, default=10.0)
    parser.add_argument("--circular_elevation_deg", type=float, default=6.0)
    parser.add_argument("--orbit_center", default=None, help="Comma-separated center for full-orbit rendering, e.g. 0,0.6,0.")
    parser.add_argument("--orbit_degrees", type=float, default=360.0, help="Azimuth sweep for --trajectory orbit.")
    parser.add_argument("--include_gt", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--out_dir", default="outputs/render_quality_videos")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    cfg = load_yaml(args.drq_config)
    args.sincro_config = args.sincro_config or args.drq_config
    args.demo_key = choose_demo_key(args.dataset, args.demo)

    with h5py.File(args.dataset, "r") as f:
        demo = f["data"][args.demo_key]
        t_len = demo["obs"][f"{args.source_cam}_rgb"].shape[0]
        if args.timestep < 0 or args.timestep >= t_len:
            raise ValueError(f"timestep must be in [0, {t_len - 1}], got {args.timestep}.")
        source = read_rgb(demo, args.source_cam, args.timestep)
        state = read_state(demo, args.timestep)
        cv_mats = dataset_camera_mats(demo, [args.source_cam], opencv=True)[args.source_cam]
        sincro_sequence = None
        if "sincro" in parse_methods(args.methods):
            # The SinCro renderer has not been constructed yet, so use the config value directly when possible.
            sc_cfg = load_yaml(args.sincro_config).get("vision", {}).get("sincro", {})
            time_interval = int(sc_cfg.get("time_interval", cfg["env"].get("frame_stack", 3)))
            sincro_sequence = load_sincro_sequence(demo, args.source_cam, args.timestep, time_interval, args.sincro_sequence_from_dataset)

    trajectories = ["lateral", "circular"] if args.trajectory == "both" else [args.trajectory]
    results = []
    for traj in trajectories:
        results.append(run_one_trajectory(args, traj, cfg, source, state, cv_mats, sincro_sequence))

    manifest_path = Path(args.out_dir) / f"{args.demo_key}_t{int(args.timestep):06d}_{args.source_cam}_to_{args.base_camera}_manifest.json"
    manifest_path.write_text(json.dumps({"args": vars(args), "videos": results}, indent=2, default=str))
    print(f"Saved manifest: {manifest_path}")
    for item in results:
        print(f"Saved: {item['path']}")


if __name__ == "__main__":
    main()
