# collect_metaworld_demos.py
from __future__ import annotations

import argparse
import os
from pathlib import Path
import numpy as np
from tqdm import tqdm

import gymnasium as gym
import metaworld  # noqa: F401

from demo_collector.config import load_config
from demo_collector.camera_math import (
    spherical_camera_pose,
    intrinsics_from_fovy,
    extrinsics_world_T_cam,
)
from demo_collector.render_mujoco import MujocoMultiCameraRenderer
from demo_collector.policy_factory import RandomPolicy, make_scripted_policy
from demo_collector.hdf5_writier import HDF5DemoWriter, DemoMeta
from demo_collector.viz import MultiCamVisualizer
from dino_postprocess import add_dino_features_inplace


def _unwrap_mujoco(env):
    e = env.unwrapped
    # Gymnasium mujoco-style
    if hasattr(e, "model") and hasattr(e, "data"):
        return e.model, e.data
    # Older mujoco_py-style (best effort)
    if hasattr(e, "sim"):
        sim = e.sim
        return sim.model, sim.data
    raise RuntimeError("Could not find MuJoCo model/data on env.unwrapped")


def _get_state(env) -> np.ndarray:
    """Concatenate qpos and qvel for storage (best-effort across wrappers)."""
    e = env.unwrapped
    if hasattr(e, "data"):
        qpos = np.asarray(e.data.qpos).ravel()
        qvel = np.asarray(e.data.qvel).ravel()
        return np.concatenate([qpos, qvel], axis=0)
    if hasattr(e, "sim"):
        qpos = np.asarray(e.sim.data.qpos).ravel()
        qvel = np.asarray(e.sim.data.qvel).ravel()
        return np.concatenate([qpos, qvel], axis=0)
    return np.zeros((0,), dtype=np.float64)


def _step_env(env, action):
    """Support Gymnasium and old Gym signatures."""
    out = env.step(action)
    # Gymnasium: (obs, reward, terminated, truncated, info)
    if len(out) == 5:
        obs, reward, terminated, truncated, info = out
        done = bool(terminated or truncated)
        return obs, float(reward), done, info
    # Old gym: (obs, reward, done, info)
    obs, reward, done, info = out
    return obs, float(reward), bool(done), info


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()

    cfg = load_config(args.config)

    # Create Meta-World env via Gym API
    env = gym.make(cfg.metaworld.benchmark_id, env_name=cfg.metaworld.env_name, seed=cfg.metaworld.seed)
    model, data = _unwrap_mujoco(env)

    # Choose look-at center (default: model.stat.center if present)
    if cfg.render.lookat is None:
        try:
            lookat = np.array(model.stat.center, dtype=np.float64)
        except Exception:
            lookat = np.zeros(3, dtype=np.float64)
    else:
        lookat = np.array(cfg.render.lookat, dtype=np.float64)

    up = np.array(cfg.render.up, dtype=np.float64)

    # Build camera poses + intr/extr
    cam_poses = []
    cam_intr = {}
    cam_extr = {}
    for c in cfg.render.cameras:
        pose = spherical_camera_pose(
            name=c.name,
            r=c.r,
            theta_deg=c.theta,
            phi_deg=c.phi,
            lookat=lookat,
            up=up,
            fovy_deg=c.fovy,
        )
        cam_poses.append(pose)
        cam_intr[c.name] = intrinsics_from_fovy(c.fovy, cfg.render.height, cfg.render.width)
        cam_extr[c.name] = extrinsics_world_T_cam(pose.pos, pose.quat_wxyz)

    renderer = MujocoMultiCameraRenderer(
        model,
        data,
        cameras=cam_poses,
        height=cfg.render.height,
        width=cfg.render.width,
        enable_seg=cfg.segmentation.enabled,
        save_objtype=cfg.segmentation.save_objtype,
    )

    writer = HDF5DemoWriter(cfg.output.path, cfg.output.mode, cfg.output.compression)

    # Global stop flag (e.g. user presses 'q')
    stop_requested = False

    def _make_demo_visualizer(demo_name: str) -> MultiCamVisualizer | None:
        """Create a per-demo visualizer (so we can optionally save per-demo mp4)."""
        if not cfg.visualize.enabled:
            return None

        save_path = None
        if cfg.visualize.save_video_dir:
            out_dir = Path(cfg.visualize.save_video_dir)
            out_dir.mkdir(parents=True, exist_ok=True)
            save_path = str(out_dir / f"{demo_name}.mp4")

        return MultiCamVisualizer(
            camera_names=[p.name for p in cam_poses],
            H=cfg.render.height,
            W=cfg.render.width,
            window_name=cfg.visualize.window_name,
            ncols=int(cfg.visualize.ncols),
            pad=int(cfg.visualize.pad),
            show_window=bool(cfg.visualize.show_window),
            save_video_path=save_path,
            video_fps=float(cfg.visualize.video_fps),
        )

    def run_policy_block(policy_type: str, num: int, policy_obj, policy_name: str):
        nonlocal lookat, stop_requested

        for _ in tqdm(range(num), desc=f"{policy_type} demos"):
            if stop_requested:
                break

            demo_idx = writer.next_demo_index()
            demo_name = f"demo{demo_idx}"
            seed = cfg.metaworld.seed + demo_idx

            obs, info = env.reset(seed=seed)
            obs = np.asarray(obs).ravel().astype(np.float32)

            # Create per-demo visualizer (optional)
            viz = _make_demo_visualizer(demo_name)

            meta = DemoMeta(
                env_id=cfg.metaworld.benchmark_id,
                env_name=cfg.metaworld.env_name,
                seed=seed,
                policy_type=policy_type,
                policy_name=policy_name,
                camera_names=[p.name for p in cam_poses],
                model_file="unknown",
            )
            writer.begin_demo(
                demo_name,
                meta,
                H=cfg.render.height,
                W=cfg.render.width,
                camera_intrinsics=cam_intr,
                camera_extrinsics=cam_extr,
                extra_attrs={"max_steps": cfg.metaworld.max_steps},
            )

            for t in range(cfg.metaworld.max_steps):
                # Policy action
                action = policy_obj.get_action(obs)
                action = np.asarray(action).ravel().astype(np.float32)

                # Step
                next_obs, reward, done, info = _step_env(env, action)
                next_obs = np.asarray(next_obs).ravel().astype(np.float32)

                # Render AFTER stepping so images align with next_obs (same as your current script)
                rend = renderer.render_all(lookat=lookat)

                # Ensure seg dict exists for downstream (writer + viz)
                if rend.seg_id_by_cam is None:
                    seg_for_step = {p.name: np.zeros((cfg.render.height, cfg.render.width), np.int32) for p in cam_poses}
                else:
                    seg_for_step = rend.seg_id_by_cam

                # Success heuristic from Meta-World info
                success = bool(info.get("success", False) or info.get("is_success", False))

                # -------- live visualization (optional) --------
                if viz is not None and (t % max(int(cfg.visualize.every_n_steps), 1) == 0):
                    key = viz.update(
                        rend.rgb_by_cam,
                        seg_for_step,
                        step_i=t,
                        overlay_lines=[
                            f"{demo_name} | {policy_type}:{policy_name}",
                            f"reward={reward:.3f} success={int(success)} done={int(done)}",
                            f"MUJOCO_GL={os.environ.get('MUJOCO_GL', '')}",
                        ],
                    )

                    # Press stop_key (default 'q') to stop collection early
                    if key != -1 and chr(key).lower() == str(cfg.visualize.stop_key).lower():
                        stop_requested = True
                        break

                # -------- write step to dataset --------
                state = _get_state(env)
                writer.append_step(
                    state=state,
                    action=action,
                    reward=reward,
                    done=done,
                    success=success,
                    obs_vec=next_obs,
                    rgb_by_cam=rend.rgb_by_cam,
                    seg_id_by_cam=seg_for_step,
                    seg_type_by_cam=rend.seg_type_by_cam,
                )

                obs = next_obs

                # Task-based termination
                if cfg.metaworld.terminate_on_success and success:
                    print(f"Demo {demo_name} succeeded at step {t}, terminating episode.")
                    break
                if done:
                    print(f"Demo {demo_name} ended at step {t} with done=True.")
                    break

            writer.end_demo()

            if viz is not None:
                viz.close()

    # ---- Collect scripted demos ----
    if cfg.policies.scripted.enabled and cfg.policies.scripted.num_demos > 0:
        pol, polinfo = make_scripted_policy(cfg.metaworld.env_name, cfg.policies.scripted.policy_class)
        run_policy_block("scripted", cfg.policies.scripted.num_demos, pol, polinfo.policy_name)

    # ---- Collect random demos ----
    if (not stop_requested) and cfg.policies.random.enabled and cfg.policies.random.num_demos > 0:
        pol = RandomPolicy(env.action_space)
        run_policy_block("random", cfg.policies.random.num_demos, pol, "RandomPolicy")

    writer.close()
    env.close()

    # Postprocess DINO into same HDF5
    if cfg.dino.enabled:
        add_dino_features_inplace(
            cfg.output.path,
            model_name=cfg.dino.model,
            image_size=cfg.dino.image_size,
            batch_size=cfg.dino.batch_size,
            device=cfg.dino.device,
            out_dtype=cfg.dino.dtype,
        )


if __name__ == "__main__":
    main()