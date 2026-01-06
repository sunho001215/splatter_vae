# ./collect_demos.py
"""
Demo collection (10 Hz) using SpaceMouse teleoperation + async HDF5 writer.

Key fix vs your current version:
- We DO NOT pass custom camera_names to suite.make() with use_camera_obs=True,
  because robosuite validates camera names during env initialization.
- Instead:
    1) Create env with use_camera_obs=False
    2) Inject custom cameras into MJCF via reset_from_xml_string
    3) Render RGB + segmentation each step via env.sim.render(camera_name=..., segmentation=...)

robosuite binding_utils.MjSim.render supports camera_name and segmentation flags. :contentReference[oaicite:1]{index=1}
"""

from __future__ import annotations

import argparse
import datetime as dt
import os
import subprocess
from typing import Any, Dict, List, Tuple

import numpy as np
import robosuite as suite

from demo_collector.async_hdf5_writer import AsyncHDF5Writer, StepPacket
from demo_collector.camera_config import load_camera_json, apply_custom_cameras_to_env
from demo_collector.camera_params import get_camera_intrinsics, get_camera_extrinsics_world_T_cam
from demo_collector.device_factory import make_teleop_device
from demo_collector.viz import MultiCamVisualizer


# ------------------------- helpers -------------------------

def get_repo_version() -> str:
    """Best-effort repo / version string."""
    try:
        pkg_dir = os.path.dirname(suite.__file__)
        sha = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=pkg_dir,
            stderr=subprocess.DEVNULL,
        ).decode("utf-8").strip()
        return sha
    except Exception:
        return getattr(suite, "__version__", "unknown")


def flatten_action_for_storage(action: Any) -> np.ndarray:
    """
    Storage helper only (does not affect env.step()).
    - np.ndarray -> flatten
    - dict of arrays -> concat in sorted key order
    """
    if isinstance(action, dict):
        parts = []
        for k in sorted(action.keys()):
            parts.append(np.asarray(action[k]).ravel())
        return np.concatenate(parts, axis=0) if parts else np.zeros((0,), dtype=np.float32)
    return np.asarray(action).ravel()


def _to_uint8_image(x: Any) -> np.ndarray:
    """Ensure RGB is uint8 HxWx3."""
    arr = np.asarray(x)
    if arr.dtype == np.uint8:
        return arr
    if np.issubdtype(arr.dtype, np.floating):
        arr = np.clip(arr, 0.0, 1.0)
        return (arr * 255.0).round().astype(np.uint8)
    return arr.astype(np.uint8)


def _to_int32_mask(x: Any) -> np.ndarray:
    """
    Ensure segmentation is int32 HxW.
    Notes:
      - Some render paths may return HxWx2 (objtype, objid). If so, we keep objid by default.
      - Some may return HxWx1.
    """
    arr = np.asarray(x)
    if arr.ndim == 3 and arr.shape[-1] == 2:
        # Common MuJoCo segmentation buffer: (objtype, objid)
        arr = arr[..., 1]
    elif arr.ndim == 3 and arr.shape[-1] == 1:
        arr = arr[..., 0]
    if arr.ndim != 2:
        raise RuntimeError(f"Segmentation must be HxW (or HxWx1/HxWx2). Got {arr.shape}")
    return arr.astype(np.int32)


def render_rgb(sim, camera_name: str, H: int, W: int) -> np.ndarray:
    """
    Render RGB via robosuite's MjSim.render wrapper.
    Robustly handle if some backends return (rgb, depth) tuple.
    """
    out = sim.render(width=W, height=H, camera_name=camera_name, depth=False, segmentation=False)
    if isinstance(out, tuple):
        out = out[0]
    return _to_uint8_image(out)


def render_segmentation(sim, camera_name: str, H: int, W: int) -> np.ndarray:
    """
    Render segmentation via MjSim.render(segmentation=True).
    Different MuJoCo bindings / renderers may return:
      - seg only
      - (rgb, seg)
      - (rgb, depth, seg)
    We always take the *last* element if a tuple is returned.
    """
    out = sim.render(width=W, height=H, camera_name=camera_name, depth=False, segmentation=True)
    if isinstance(out, tuple):
        out = out[-1]
    return _to_int32_mask(out)


# ------------------------- main -------------------------


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--env", type=str, required=True, choices=["Stack", "PickPlace", "NutAssembly", "Wipe"])
    parser.add_argument("--robot", type=str, default="Panda")
    parser.add_argument("--output", type=str, default="demo.hdf5")

    # 6 user cameras
    parser.add_argument("--camera_json", type=str, required=True)
    parser.add_argument("--img_h", type=int, default=224)
    parser.add_argument("--img_w", type=int, default=224)

    # Teleop
    parser.add_argument("--device", type=str, default="spacemouse", choices=["spacemouse", "keyboard", "dualsense", "mjgui"])
    parser.add_argument("--pos_sensitivity", type=float, default=1.0)
    parser.add_argument("--rot_sensitivity", type=float, default=1.0)

    # 10 Hz sampling: one env.step == one sample
    parser.add_argument("--control_freq", type=float, default=10.0)
    parser.add_argument("--horizon", type=int, default=500)
    parser.add_argument("--num_demos", type=int, default=10)

    # DINO options (writer thread)
    parser.add_argument("--dino_source", type=str, default="torchhub", choices=["torchhub", "hf"])
    parser.add_argument("--dino_mode", type=str, default="patch", choices=["patch", "pixel"])
    parser.add_argument("--dino_device", type=str, default="cuda")

    args = parser.parse_args()

    cams = load_camera_json(args.camera_json)
    camera_names = [c.name for c in cams]
    H, W = args.img_h, args.img_w

    # Visualizer
    viz = MultiCamVisualizer(
        camera_names=camera_names, H=H, W=W,
    )

    # Create env WITHOUT camera observables first (use_camera_obs=False).
    # This prevents robosuite from validating camera_names that don't exist yet.
    env = suite.make(
        args.env,
        robots=[args.robot],
        has_renderer=True,
        has_offscreen_renderer=True,
        use_camera_obs=False,
        control_freq=args.control_freq,
        horizon=args.horizon,
        hard_reset=False,
        renderer="mujoco",
    )

    # Inject user-defined cameras, rebuild sim/model from modified XML
    apply_custom_cameras_to_env(env, cams)

    # Create SpaceMouse AFTER camera injection (safe either way, but clean)
    device = make_teleop_device(
        args.device,
        env,
        pos_sensitivity=args.pos_sensitivity,
        rot_sensitivity=args.rot_sensitivity,
    )

    # Writer init (robosuite-like root attrs)
    now = dt.datetime.now()
    root_attrs = {
        "date": now.strftime("%Y-%m-%d"),
        "time": now.strftime("%H:%M:%S"),
        "repository_version": get_repo_version(),
        "env": args.env,
    }

    writer = AsyncHDF5Writer(
        output_path=args.output,
        camera_names=camera_names,
        compression="lzf",
        max_queue=256,
        dino_device=args.dino_device,
        dino_source=args.dino_source,
        dino_mode=args.dino_mode,
    )
    writer.start(root_attrs)

    # ---------------------- collect demos ----------------------
    for demo_i in range(args.num_demos):
        obs = env.reset()
        done = False
        t = 0

        # Model xml (robosuite demo format)
        model_xml = env.model.get_xml() if hasattr(env.model, "get_xml") else str(env.model)

        # Camera params (after reset so sim.data is valid)
        intrinsics = {cam: get_camera_intrinsics(env.sim, cam, H, W) for cam in camera_names}
        extrinsics = {cam: get_camera_extrinsics_world_T_cam(env.sim, cam) for cam in camera_names}

        demo_name = f"demo{demo_i + 1}"  # matches "demo1", "demo2", ...
        writer.begin_demo(
            demo_name=demo_name,
            model_xml=model_xml,
            camera_intrinsics=intrinsics,
            camera_extrinsics=extrinsics,
            image_hw=(H, W),
            extra_demo_attrs={
                "control_freq": float(args.control_freq),
                "camera_json": os.path.basename(args.camera_json),
                "teleop_device": args.device,
                "note": "RGB/seg captured via sim.render (not use_camera_obs)",
            },
        )

        while not done and t < args.horizon:
            # Teleop -> action
            action = device.input2action(goal_update_mode="target")

            if action is None:
                # Device asked for reset. We keep the same demo group but restart the episode stream.
                # If you prefer to discard partial demos, you’d need an 'abort_demo' command in the writer.
                print("[teleop] reset triggered from device; resetting env and continuing same demo group.")
                obs = env.reset()
                t = 0
                continue

            # Step sim
            obs, reward, done, info = env.step(action)

            # 1) state
            state = env.sim.get_state().flatten().copy()

            # 2) action (flatten for storage)
            action_flat = flatten_action_for_storage(action).astype(np.float32).copy()

            # 3) RGB + segmentation via sim.render for each custom cam
            rgb_by_cam: Dict[str, np.ndarray] = {}
            seg_by_cam: Dict[str, np.ndarray] = {}

            for cam in camera_names:
                rgb_by_cam[cam] = render_rgb(env.sim, cam, H, W)
                seg_by_cam[cam] = render_segmentation(env.sim, cam, H, W)

            # 4) enqueue (writer thread will compute DINO + write HDF5)
            writer.enqueue_step(
                StepPacket(
                    state=state,
                    action=action_flat,
                    rgb_by_cam=rgb_by_cam,
                    seg_by_cam=seg_by_cam,
                    t=float(t),
                )
            )

            # 5) visualize
            key = viz.update(rgb_by_cam, seg_by_cam, step_i=t)

            t += 1

        writer.end_demo()
        print(f"[demo] finished {demo_name} (steps attempted={t})")

    viz.close()
    writer.close()
    env.close()
    print(f"[DONE] Wrote {args.num_demos} demos to {args.output}. Dropped steps: {writer.dropped_steps}")


if __name__ == "__main__":
    main()
