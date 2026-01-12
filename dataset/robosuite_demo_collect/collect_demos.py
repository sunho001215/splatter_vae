from __future__ import annotations

import time
import argparse
import datetime as dt
import os
import subprocess
from typing import Any, Dict

import numpy as np
import robosuite as suite
import h5py
import re

from demo_collector.async_hdf5_writer import AsyncHDF5Writer, StepPacket
from demo_collector.camera_config import apply_custom_cameras_to_env, make_spherical_cameras
from demo_collector.camera_params import get_camera_intrinsics, get_camera_extrinsics_world_T_cam
from demo_collector.device_factory import make_teleop_device
from demo_collector.render import render_rgb, render_segmentation
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

# ------------------------- main -------------------------

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--env", type=str, required=True, choices=["Stack", "PickPlace", "NutAssembly", "Wipe"])
    parser.add_argument("--robot", type=str, default="Panda")
    parser.add_argument("--output", type=str, default="demo.hdf5")

    # 6 cameras from spherical inputs: r (1), theta (2), phi (3) => 6 cams
    parser.add_argument("--cam_r", type=float, required=True, help="Radius for spherical camera placement")
    parser.add_argument("--cam_theta", type=float, nargs=2, required=True, help="2 elevation angles (deg by default)")
    parser.add_argument("--cam_phi", type=float, nargs=3, required=True, help="3 azimuth angles (deg by default)")
    parser.add_argument("--cam_fovy", type=float, default=45.0, help="Vertical field-of-view (deg)")
    parser.add_argument(
        "--cam_lookat",
        type=float,
        nargs=3,
        default=None,
        help="Look-at center (world xyz). Default: env.sim.model.stat.center if available, else (0,0,0)",
    )
    parser.add_argument("--cam_up", type=float, nargs=3, default=(0.0, 0.0, 1.0), help="World up vector (xyz)")
    parser.add_argument("--cam_angles_in_rad", action="store_true", help="If set, theta/phi are in radians")

    parser.add_argument("--img_h", type=int, default=224)
    parser.add_argument("--img_w", type=int, default=224)

    # Teleop
    parser.add_argument("--device", type=str, default="spacemouse",
                        choices=["spacemouse", "keyboard", "dualsense", "mjgui"])
    parser.add_argument("--pos_sensitivity", type=float, default=1.0)
    parser.add_argument("--rot_sensitivity", type=float, default=1.0)

    # 10 Hz sampling: one env.step == one sample
    parser.add_argument("--control_freq", type=float, default=10.0)
    parser.add_argument("--horizon", type=int, default=3000)
    
    # Demo
    parser.add_argument("--num_demos", type=int, default=120)
    parser.add_argument("--start_demo", type=int, default=1)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--overwrite_output", action="store_true")
    parser.add_argument("--stop_key", type=str, default="q")
    parser.add_argument("--overwrite_existing_demos", action="store_true")
    parser.add_argument("--auto_skip_existing", action="store_true", default=True)

    args = parser.parse_args()

    H, W = args.img_h, args.img_w

    # Decide file open mode for the writer.
    # - If output exists:
    #     * --resume  => open with "r+" and append
    #     * else      => error unless --overwrite_output
    output_exists = os.path.exists(args.output)
    if output_exists and (not args.resume) and (not args.overwrite_output):
        raise FileExistsError(
            f"Output already exists: {args.output}. "
            f"Use --resume to append, or --overwrite_output to overwrite."
        )
    file_mode = "w"
    if args.overwrite_output:
        file_mode = "w"
    elif args.resume and output_exists:
        file_mode = "r+"

    # ----------------- env creation -----------------

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

    # Initialize sim so we can choose a default lookat
    _ = env.reset()

    # ----------------- camera configuration -----------------

    if args.cam_lookat is None:
        try:
            lookat = np.array(env.sim.model.stat.center, dtype=np.float64)
        except Exception:
            lookat = np.zeros(3, dtype=np.float64)
    else:
        lookat = np.array(args.cam_lookat, dtype=np.float64)

    # 6 spherical cameras that will be saved to the dataset
    cams = make_spherical_cameras(
        r=float(args.cam_r),
        theta_list=list(args.cam_theta),
        phi_list=list(args.cam_phi),
        lookat=lookat.tolist(),
        up=list(args.cam_up),
        degrees=(not args.cam_angles_in_rad),
        fovy=float(args.cam_fovy),
        name_prefix="cam",
    )
    # Cameras that are actually written to disk (data cameras)
    data_camera_names = [c.name for c in cams]

    # Name of robosuite's default wrist / eye-in-hand camera
    # (this exists for single-arm robots as "robot0_eye_in_hand")
    WRIST_CAM_NAME = "robot0_eye_in_hand"

    # Visualization cameras = 6 data cams + 1 wrist cam
    # We will render all of these on-screen,
    # but only data_camera_names will be stored in the HDF5 dataset.
    viz_camera_names = data_camera_names + [WRIST_CAM_NAME]

    # Visualizer now uses the *viz* camera list (includes wrist)
    viz = MultiCamVisualizer(camera_names=viz_camera_names, H=H, W=W)

    # Inject the 6 custom spherical cameras into the MJCF and rebuild sim
    # (this does NOT remove default cameras like "robot0_eye_in_hand")
    apply_custom_cameras_to_env(env, cams)

    # ----------------- teleop device -----------------

    device = make_teleop_device(
        args.device,
        env,
        pos_sensitivity=args.pos_sensitivity,
        rot_sensitivity=args.rot_sensitivity,
    )

    # ----------------- writer init -----------------

    now = dt.datetime.now()
    root_attrs = {
        "date": now.strftime("%Y-%m-%d"),
        "time": now.strftime("%H:%M:%S"),
        "repository_version": get_repo_version(),
        "env": args.env,
        "camera_r": float(args.cam_r),
        "camera_theta": str(list(args.cam_theta)),
        "camera_phi": str(list(args.cam_phi)),
        "camera_fovy": float(args.cam_fovy),
        "camera_lookat": str(lookat.tolist()),
        "camera_up": str(list(args.cam_up)),
        "camera_angles_in_rad": bool(args.cam_angles_in_rad),
    }

    # IMPORTANT:
    #   AsyncHDF5Writer gets only the data_camera_names.
    #   It will *ignore* the wrist camera, so wrist images are never saved.
    writer = AsyncHDF5Writer(
        output_path=args.output,
        camera_names=data_camera_names,   # <--- only the 6 spherical cams
        compression=None,
        max_queue=1024,
        file_mode=file_mode,
    )
    writer.start(root_attrs)

    # ---------------------- collect demos ----------------------
    dt_sim = 1.0 / args.control_freq

    demo_idx = int(args.start_demo)  # 1-indexed
    stop_requested = False

    while demo_idx <= int(args.num_demos):
        obs = env.reset()
        t = 0
        aborted = False  # True => discard this demo and retry same demo_idx

        # Model xml (robosuite demo format)
        model_xml = env.model.get_xml() if hasattr(env.model, "get_xml") else str(env.model)

        # Camera params (only for data cameras, since only they are saved)
        intrinsics = {cam: get_camera_intrinsics(env.sim, cam, H, W) for cam in data_camera_names}
        extrinsics = {cam: get_camera_extrinsics_world_T_cam(env.sim, cam) for cam in data_camera_names}

        demo_name = f"demo{demo_idx}"  # keep demo numbering stable across retries/resume.
        writer.begin_demo(
            demo_name=demo_name,
            model_xml=model_xml,
            camera_intrinsics=intrinsics,
            camera_extrinsics=extrinsics,
            image_hw=(H, W),
            extra_demo_attrs={
                "control_freq": float(args.control_freq),
                "teleop_device": args.device,
                "note": "RGB/seg captured via sim.render (not use_camera_obs); cameras from spherical inputs",
            },
        )

        # Stop when task succeeds OR horizon reached
        while not env._check_success() and t < args.horizon:
            loop_start = time.perf_counter()

            # Get action from device
            action = device.input2action(goal_update_mode="target")

            if action is None:
                # SpaceMouse reset => discard the entire current episode (demo group)
                print("[teleop] reset triggered: aborting current demo and retrying same demo index.")
                aborted = True
                break

            obs, reward, done, info = env.step(action)

            # 1) state
            state = env.sim.get_state().flatten().copy()

            # 2) action (flatten for storage)
            action_flat = flatten_action_for_storage(action).astype(np.float32).copy()

            # 3) RGB + segmentation via sim.render
            #    We build two dicts:
            #      - full_rgb / full_seg: includes wrist cam (for visualization)
            #      - data_rgb / data_seg: only data cams (for writer / HDF5)
            full_rgb_by_cam: Dict[str, np.ndarray] = {}
            full_seg_by_cam: Dict[str, np.ndarray] = {}

            # Render all data cameras (these are saved + visualized)
            for cam in data_camera_names:
                full_rgb_by_cam[cam] = render_rgb(env.sim, cam, H, W)
                full_seg_by_cam[cam] = render_segmentation(env, cam, H, W)

            # Render wrist view ONLY for visualization (not stored)
            try:
                wrist_rgb = render_rgb(env.sim, WRIST_CAM_NAME, H, W)
                wrist_seg = render_segmentation(env, WRIST_CAM_NAME, H, W)
                full_rgb_by_cam[WRIST_CAM_NAME] = wrist_rgb
                full_seg_by_cam[WRIST_CAM_NAME] = wrist_seg
            except Exception as e:
                # If the robot / env combo does not have this camera, skip it gracefully
                pass

            # Subset dicts that only contain data cameras for writing
            data_rgb_by_cam = {cam: full_rgb_by_cam[cam] for cam in data_camera_names}
            data_seg_by_cam = {cam: full_seg_by_cam[cam] for cam in data_camera_names}

            # 4) enqueue (writer thread will write HDF5)
            writer.enqueue_step(
                StepPacket(
                    state=state,
                    action=action_flat,
                    rgb_by_cam=data_rgb_by_cam,   # <--- only 6 cams
                    seg_by_cam=data_seg_by_cam,   # <--- only 6 cams
                    obs=obs,
                    t=float(t),
                )
            )

            # 5) visualize (uses full set including wrist)
            key = viz.update(full_rgb_by_cam, full_seg_by_cam, step_i=t)

            # Stop collection on keyboard input from the OpenCV window.
            # Press 'q' (default) or ESC to stop. Current demo is discarded.
            if key in (ord(args.stop_key.lower()), ord(args.stop_key.upper()), 27):
                print("[user] stop requested: aborting current demo and exiting.")
                aborted = True
                stop_requested = True
                break

            t += 1

            # 6) real-time throttling
            elapsed = time.perf_counter() - loop_start
            sleep_time = dt_sim - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)

        if aborted:
            # Delete the demo group from disk (no partial episodes)
            writer.abort_demo()
            if stop_requested:
                break
            # Retry the SAME demo_idx (demo number is preserved)
            continue

        writer.end_demo()
        print(f"[demo] finished {demo_name} (steps attempted={t})")
        demo_idx += 1

    viz.close()
    writer.close()
    env.close()
    print(f"[DONE] Finished up to demo{demo_idx - 1}. Output: {args.output}. Dropped steps: {writer.dropped_steps}")


if __name__ == "__main__":
    main()
