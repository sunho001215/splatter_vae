from __future__ import annotations

import argparse
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

    # Create Meta-World env via Gym API (per Meta-World docs). :contentReference[oaicite:15]{index=15}
    env = gym.make(cfg.metaworld.benchmark_id, env_name=cfg.metaworld.env_name, seed=cfg.metaworld.seed)

    model, data = _unwrap_mujoco(env)

    # Lookat default: try model.stat.center (similar to your robosuite code)
    if cfg.render.lookat is None:
        lookat = None
        try:
            lookat = np.array(model.stat.center, dtype=np.float64)
        except Exception:
            lookat = np.zeros(3, dtype=np.float64)
    else:
        lookat = np.array(cfg.render.lookat, dtype=np.float64)

    up = np.array(cfg.render.up, dtype=np.float64)

    # Build camera poses
    cam_poses = []
    cam_intr = {}
    cam_extr = {}
    for c in cfg.render.cameras:
        pose = spherical_camera_pose(
            name=c.name, r=c.r, theta_deg=c.theta, phi_deg=c.phi,
            lookat=lookat, up=up, fovy_deg=c.fovy
        )
        cam_poses.append(pose)
        cam_intr[c.name] = intrinsics_from_fovy(c.fovy, cfg.render.height, cfg.render.width)
        cam_extr[c.name] = extrinsics_world_T_cam(pose.pos, pose.quat_wxyz)

    renderer = MujocoMultiCameraRenderer(
        model, data,
        cameras=cam_poses,
        height=cfg.render.height,
        width=cfg.render.width,
        enable_seg=cfg.segmentation.enabled,
        save_objtype=cfg.segmentation.save_objtype,
    )

    writer = HDF5DemoWriter(cfg.output.path, cfg.output.mode, cfg.output.compression)

    def run_policy_block(policy_type: str, num: int, policy_obj, policy_name: str):
        nonlocal lookat
        for k in tqdm(range(num), desc=f"{policy_type} demos"):
            demo_idx = writer.next_demo_index()
            demo_name = f"demo{demo_idx}"
            seed = cfg.metaworld.seed + demo_idx

            obs, info = env.reset(seed=seed)
            obs = np.asarray(obs).ravel().astype(np.float32)

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
                demo_name, meta,
                H=cfg.render.height, W=cfg.render.width,
                camera_intrinsics=cam_intr,
                camera_extrinsics=cam_extr,
                extra_attrs={"max_steps": cfg.metaworld.max_steps},
            )

            for t in range(cfg.metaworld.max_steps):
                action = policy_obj.get_action(obs)
                action = np.asarray(action).ravel().astype(np.float32)

                next_obs, reward, done, info = _step_env(env, action)
                next_obs = np.asarray(next_obs).ravel().astype(np.float32)

                # Render AFTER stepping, so images align with next_obs.
                rend = renderer.render_all(lookat=lookat)

                state = _get_state(env)
                success = bool(info.get("success", False) or info.get("is_success", False))

                writer.append_step(
                    state=state,
                    action=action,
                    reward=reward,
                    done=done,
                    success=success,
                    obs_vec=next_obs,
                    rgb_by_cam=rend.rgb_by_cam,
                    seg_id_by_cam=rend.seg_id_by_cam if rend.seg_id_by_cam is not None else {p.name: np.zeros((cfg.render.height, cfg.render.width), np.int32) for p in cam_poses},
                    seg_type_by_cam=rend.seg_type_by_cam,
                )

                obs = next_obs

                if cfg.metaworld.terminate_on_success and success:
                    print(f"Demo {demo_name} succeeded at step {t}, terminating episode.")
                    break
                if done:
                    print(f"Demo {demo_name} ended at step {t} with done=True.")
                    break

            writer.end_demo()

    # Scripted
    if cfg.policies.scripted.enabled and cfg.policies.scripted.num_demos > 0:
        pol, polinfo = make_scripted_policy(cfg.metaworld.env_name, cfg.policies.scripted.policy_class)
        run_policy_block("scripted", cfg.policies.scripted.num_demos, pol, polinfo.policy_name)

    # Random
    if cfg.policies.random.enabled and cfg.policies.random.num_demos > 0:
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
