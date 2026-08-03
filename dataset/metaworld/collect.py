from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

# Set default MuJoCo rendering backend to EGL for headless environments
os.environ.setdefault("MUJOCO_GL", "egl")

# Add project root to sys.path for imports
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import mujoco_mig_setup

import numpy as np
from tqdm import tqdm

import gymnasium as gym
import metaworld  # noqa: F401
import mujoco

from dataset.metaworld.collector.config import load_config
from dataset.metaworld.collector.camera import (
    spherical_camera_pose,
    intrinsics_from_fovy,
    extrinsics_world_T_cam,
)
from dataset.metaworld.collector.renderer import MujocoMultiCameraRenderer
from dataset.metaworld.collector.policy import make_scripted_policy
from dataset.metaworld.collector.hdf5_writer import HDF5DemoWriter, DemoMeta
from dataset.metaworld.collector.actions import (
    build_action_generator,
    build_episode_configs,
    episode_config_metadata,
)
from dataset.metaworld.collector.visualization import MultiCamVisualizer
from dataset.metaworld.tools.dino_postprocess import add_dino_features_inplace


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




def _mujoco_segmentation_objects(model) -> list[dict]:
    rows = [{"id": -1, "type": -1, "name": "background", "type_name": "background", "body_id": -1, "body_name": "", "body_path": ""}]

    def body_path(body_id: int) -> str:
        names = []
        current = int(body_id)
        seen = set()
        while current >= 0 and current not in seen:
            seen.add(current)
            names.append(mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, current) or f"body_{current}")
            parent = int(model.body_parentid[current]) if current > 0 else -1
            current = parent
        return "/".join(reversed(names))
    object_types = (
        (mujoco.mjtObj.mjOBJ_GEOM, "ngeom"),
        (mujoco.mjtObj.mjOBJ_BODY, "nbody"),
        (mujoco.mjtObj.mjOBJ_SITE, "nsite"),
        (mujoco.mjtObj.mjOBJ_CAMERA, "ncam"),
        (mujoco.mjtObj.mjOBJ_LIGHT, "nlight"),
    )
    for obj_type, count_attr in object_types:
        count = int(getattr(model, count_attr, 0))
        type_id = int(obj_type)
        type_name = obj_type.name.replace("mjOBJ_", "").lower()
        for obj_id in range(count):
            name = mujoco.mj_id2name(model, obj_type, obj_id) or f"{type_name}_{obj_id}"
            body_id = -1
            body_name = ""
            if obj_type == mujoco.mjtObj.mjOBJ_GEOM:
                body_id = int(model.geom_bodyid[obj_id])
            elif obj_type == mujoco.mjtObj.mjOBJ_SITE:
                body_id = int(model.site_bodyid[obj_id])
            elif obj_type == mujoco.mjtObj.mjOBJ_BODY:
                body_id = int(obj_id)
            if body_id >= 0:
                body_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body_id) or f"body_{body_id}"
            rows.append({
                "id": int(obj_id),
                "type": type_id,
                "name": str(name),
                "type_name": type_name,
                "body_id": int(body_id),
                "body_name": str(body_name),
                "body_path": body_path(body_id) if body_id >= 0 else "",
            })
    return rows

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
    ap = argparse.ArgumentParser(
        description="Collect one transactional LZF-compressed Meta-World dataset."
    )
    ap.add_argument("--config", required=True)
    ap.add_argument("--env-name", help="Override metaworld.env_name from the YAML.")
    ap.add_argument("--output-path", help="Override output.path from the YAML.")
    ap.add_argument("--max-steps", type=int, help="Override metaworld.max_steps.")
    args = ap.parse_args()

    cfg = load_config(args.config)
    if args.env_name is not None:
        cfg.metaworld.env_name = args.env_name
    if args.output_path is not None:
        cfg.output.path = args.output_path
    if args.max_steps is not None:
        if args.max_steps <= 0:
            raise ValueError("--max-steps must be positive.")
        cfg.metaworld.max_steps = args.max_steps
    if cfg.output.compression != "lzf":
        raise ValueError("Meta-World collection requires output.compression: lzf.")

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
        enable_depth=cfg.render.save_depth,
    )
    segmentation_objects = _mujoco_segmentation_objects(model) if cfg.segmentation.enabled else None

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

    def run_policy_block(
        mode: str,
        episode_configs: list,
        policy_obj,
        policy_name: str,
        mode_seed_offset: int,
    ) -> None:
        nonlocal lookat, stop_requested

        for episode_config in tqdm(episode_configs, desc=f"{mode} demos"):
            if stop_requested:
                break

            demo_idx = writer.next_demo_index()
            demo_name = f"demo{demo_idx}"
            seed = cfg.metaworld.seed + demo_idx
            action_rng = np.random.default_rng(
                cfg.metaworld.seed + mode_seed_offset + demo_idx
            )
            action_generator = build_action_generator(
                episode_config,
                env.action_space.low,
                env.action_space.high,
                action_rng,
            )

            obs, info = env.reset(seed=seed)
            obs = np.asarray(obs).ravel().astype(np.float32)
            viz = _make_demo_visualizer(demo_name)

            meta = DemoMeta(
                env_id=cfg.metaworld.benchmark_id,
                env_name=cfg.metaworld.env_name,
                seed=seed,
                policy_type=mode,
                policy_name=policy_name,
                camera_names=[pose.name for pose in cam_poses],
                model_file="unknown",
            )
            extra_attrs = {
                "max_steps": cfg.metaworld.max_steps,
                "save_depth": bool(cfg.render.save_depth),
                "segmentation_enabled": bool(cfg.segmentation.enabled),
                "segmentation_save_objtype": bool(cfg.segmentation.save_objtype),
                **episode_config_metadata(episode_config),
            }
            writer.begin_demo(
                demo_name,
                meta,
                H=cfg.render.height,
                W=cfg.render.width,
                camera_intrinsics=cam_intr,
                camera_extrinsics=cam_extr,
                extra_attrs=extra_attrs,
                save_depth=cfg.render.save_depth,
                segmentation_objects=segmentation_objects,
            )

            frames_saved = 0
            for timestep in range(cfg.metaworld.max_steps):
                # Every mode receives the current scripted action. The selected
                # generator is solely responsible for transforming it.
                expert_action = np.asarray(
                    policy_obj.get_action(obs), dtype=np.float32
                ).reshape(4)
                action = action_generator.get_action(
                    expert_action=expert_action,
                    timestep=timestep,
                )

                # Store state/observation at t before applying action_t. The
                # synchronous writer completes compression before the next step.
                state = _get_state(env)
                obs_to_store = obs.copy()
                rend = renderer.render_all(lookat=lookat)
                next_obs, reward, done, info = _step_env(env, action)
                next_obs = np.asarray(next_obs).ravel().astype(np.float32)

                if rend.seg_id_by_cam is None:
                    seg_for_step = {
                        pose.name: np.zeros(
                            (cfg.render.height, cfg.render.width), np.int32
                        )
                        for pose in cam_poses
                    }
                else:
                    seg_for_step = rend.seg_id_by_cam

                success = bool(
                    info.get("success", False) or info.get("is_success", False)
                )
                if viz is not None and (
                    timestep % max(int(cfg.visualize.every_n_steps), 1) == 0
                ):
                    key = viz.update(
                        rend.rgb_by_cam,
                        seg_for_step,
                        step_i=timestep,
                        overlay_lines=[
                            f"{demo_name} | {mode}:{policy_name}",
                            f"reward={reward:.3f} success={int(success)} done={int(done)}",
                            f"MUJOCO_GL={os.environ.get('MUJOCO_GL', '')}",
                        ],
                    )
                    if (
                        key != -1
                        and chr(key).lower()
                        == str(cfg.visualize.stop_key).lower()
                    ):
                        stop_requested = True
                        break

                writer.append_step(
                    state=state,
                    action=action,
                    reward=reward,
                    done=done,
                    success=success,
                    obs_vec=obs_to_store,
                    rgb_by_cam=rend.rgb_by_cam,
                    seg_id_by_cam=seg_for_step,
                    depth_by_cam=rend.depth_by_cam,
                    seg_type_by_cam=rend.seg_type_by_cam,
                )
                frames_saved += 1
                obs = next_obs

                if (
                    cfg.metaworld.terminate_on_success
                    and success
                    and action_generator.allow_success_termination(timestep)
                ):
                    print(
                        f"Demo {demo_name} succeeded at step {timestep}, "
                        "terminating episode."
                    )
                    break
                if done:
                    print(
                        f"Demo {demo_name} ended at step {timestep} with done=True."
                    )
                    break

            writer.end_demo(expected_frames=frames_saved)
            if viz is not None:
                viz.close()

    modes = (
        ("expert_guided", cfg.collection.expert_guided.num_demos),
        ("perturb_recover", cfg.collection.perturb_recover.num_demos),
        ("smooth_random", cfg.collection.smooth_random.num_demos),
    )
    policy, policy_info = make_scripted_policy(
        cfg.metaworld.env_name,
        cfg.collection.scripted_policy_class,
    )
    expected_demos = sum(count for _mode, count in modes)
    try:
        for mode_index, (mode, expected_mode_demos) in enumerate(modes):
            episode_configs = build_episode_configs(
                mode,
                cfg.collection,
                cfg.metaworld.max_steps,
                np.random.default_rng(cfg.metaworld.seed + 10_000 * mode_index),
            )
            if len(episode_configs) != expected_mode_demos:
                raise RuntimeError(
                    f"{mode} produced {len(episode_configs)} configurations; "
                    f"expected {expected_mode_demos}."
                )
            run_policy_block(
                mode,
                episode_configs,
                policy,
                policy_info.policy_name,
                mode_seed_offset=100_000 * (mode_index + 1),
            )
        writer.close(expected_demos=expected_demos)
    except BaseException:
        writer.abort()
        raise
    finally:
        renderer.close()
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
