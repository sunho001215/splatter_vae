from __future__ import annotations

"""
Compact Meta-World training script for DrM.
"""

import argparse
import os
import random
import sys
from collections import deque
from pathlib import Path
from typing import Any, Deque, Dict, Tuple

import numpy as np
import torch
import yaml

# Headless MuJoCo rendering.
os.environ.setdefault("MUJOCO_GL", "egl")

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import gymnasium as gym
import metaworld
import mujoco
import cv2
import wandb

from agents.drm.drm_metaworld import DrMMetaWorldAgent
from dataset.metaworld_demo_collect.demo_collector.camera_math import spherical_camera_pose
from agents.drm.replay_buffer import make_replay_loader, ReplayBufferStorage

# -----------------------------------------------------------------------------
# Small helpers
# -----------------------------------------------------------------------------

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

# -----------------------------------------------------------------------------
# Meta-World environment wrapper with a single configurable camera
# -----------------------------------------------------------------------------

class MetaWorldSingleCameraEnv:
    """
    Single-task Meta-World environment that returns pixel observations only.

    Camera configuration follows the same spherical style used in the provided
    Meta-World demo collection utilities, but we keep only one camera.
    """

    def __init__(self, cfg: Dict[str, Any], seed: int) -> None:
        env_cfg = cfg["env"]
        self.env_name = str(env_cfg["env_name"])
        self.image_height = int(env_cfg["image_height"])
        self.image_width = int(env_cfg["image_width"])
        self.frame_stack = int(env_cfg.get("frame_stack", 3))
        self.action_repeat = int(env_cfg.get("action_repeat", 1))
        self.max_episode_steps = int(env_cfg.get("max_episode_steps", 250))
        self.camera_cfg = dict(env_cfg["camera"])

        # Match the demo collector more closely:
        # - if lookat is omitted / null, use model.stat.center
        # - keep "up" as a world-space vector
        raw_lookat = env_cfg.get("lookat", None)
        self.up = np.asarray(env_cfg.get("up", [0.0, 0.0, 1.0]), dtype=np.float64)

        # Meta-World v3 via Gymnasium API
        self.benchmark_id = str(env_cfg.get("benchmark_id", "Meta-World/MT1"))
        gym_env_name = self.env_name if self.env_name.endswith("-v3") else f"{self.env_name}-v3"

        self._env = gym.make(self.benchmark_id, env_name=gym_env_name, seed=seed)
        self._model, self._data = self._unwrap_mujoco(self._env)

        # Default lookat follows the collector script
        if raw_lookat is None:
            try:
                self.lookat = np.array(self._model.stat.center, dtype=np.float64)
            except Exception:
                self.lookat = np.zeros(3, dtype=np.float64)
        else:
            self.lookat = np.asarray(raw_lookat, dtype=np.float64)

        # Direct MuJoCo renderer
        self._renderer = mujoco.Renderer(
            self._model,
            height=self.image_height,
            width=self.image_width,
        )

        self._base_seed = int(seed)
        self._reset_count = 0

        base_env = self._env.unwrapped
        if hasattr(base_env, "_freeze_rand_vec"):
            base_env._freeze_rand_vec = False

        self.action_space = self._env.action_space
        self.frames: Deque[np.ndarray] = deque(maxlen=self.frame_stack)
        self.episode_step = 0

        # Optional OpenCV visualization
        vis_cfg = dict(cfg.get("visualization", {}))
        self.enable_cv2_vis = bool(vis_cfg.get("enabled", False))
        self.vis_window_name = str(vis_cfg.get("window_name", f"MetaWorld-{self.env_name}"))
        self.vis_wait_key = int(vis_cfg.get("wait_key", 1))
        self.vis_scale = float(vis_cfg.get("scale", 1.0))
        self._vis_window_initialized = False

        if self.enable_cv2_vis and cv2 is None:
            raise ImportError(
                "visualization.enabled=True, but OpenCV (`cv2`) is not available."
            )

    @property
    def obs_shape(self) -> Tuple[int, int, int]:
        return (3 * self.frame_stack, self.image_height, self.image_width)

    @property
    def action_shape(self) -> Tuple[int, ...]:
        return tuple(self.action_space.shape)

    def _unwrap_mujoco(self, env):
        e = env.unwrapped
        # Gymnasium mujoco-style
        if hasattr(e, "model") and hasattr(e, "data"):
            return e.model, e.data
        # Older mujoco-py style fallback
        if hasattr(e, "sim"):
            return e.sim.model, e.sim.data
        raise RuntimeError("Could not find MuJoCo model/data on env.unwrapped")

    def _make_free_camera(self) -> "mujoco.MjvCamera":
        """
        Build the same kind of free/orbit camera used in the data collection code:
        spherical camera placement -> MjvCamera(lookat, distance, azimuth, elevation).
        """
        pose = spherical_camera_pose(
            name=str(self.camera_cfg["name"]),
            r=float(self.camera_cfg["r"]),
            theta_deg=float(self.camera_cfg["theta"]),
            phi_deg=float(self.camera_cfg["phi"]),
            lookat=self.lookat,
            up=self.up,
            fovy_deg=float(self.camera_cfg["fovy"]),
        )

        cam = mujoco.MjvCamera()
        cam.type = mujoco.mjtCamera.mjCAMERA_FREE
        cam.lookat[:] = self.lookat.astype(np.float64)

        rel = pose.pos.astype(np.float64) - self.lookat.astype(np.float64)
        cam.distance = float(np.linalg.norm(rel))

        # azimuth: angle in XY plane from +X
        cam.azimuth = float(np.degrees(np.arctan2(rel[1], rel[0])))

        # elevation: angle above XY plane
        xy = np.sqrt(rel[0] ** 2 + rel[1] ** 2)
        cam.elevation = float(np.degrees(np.arctan2(rel[2], xy)))

        return cam

    def _show_frame_cv2(self, frame_chw: np.ndarray) -> None:
        """
        Show the current RGB frame with OpenCV.

        frame_chw: (3, H, W), uint8, RGB
        """
        if not self.enable_cv2_vis:
            return

        # Convert CHW RGB -> HWC RGB
        img = np.transpose(frame_chw, (1, 2, 0))

        # Optional resize for easier viewing
        if self.vis_scale != 1.0:
            new_w = max(1, int(img.shape[1] * self.vis_scale))
            new_h = max(1, int(img.shape[0] * self.vis_scale))
            interp = cv2.INTER_NEAREST if self.vis_scale >= 1.0 else cv2.INTER_AREA
            img = cv2.resize(img, (new_w, new_h), interpolation=interp)

        # OpenCV expects BGR
        img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

        if not self._vis_window_initialized:
            cv2.namedWindow(self.vis_window_name, cv2.WINDOW_NORMAL)
            self._vis_window_initialized = True

        cv2.imshow(self.vis_window_name, img_bgr)
        cv2.waitKey(max(1, self.vis_wait_key))

    def _render(self) -> np.ndarray:
        """
        Render through mujoco.Renderer directly, matching the demo collector.
        This avoids Gymnasium wrapper render-mode issues and does not require
        a named camera to exist in the XML.
        """
        cam = self._make_free_camera()

        self._renderer.update_scene(self._data, camera=cam)
        img = self._renderer.render()

        if img is None:
            raise RuntimeError("MuJoCo renderer returned None.")

        img = np.asarray(img, dtype=np.uint8)
        if img.ndim != 3 or img.shape[-1] != 3:
            raise RuntimeError(f"Unexpected rendered image shape: {img.shape}")

        # Convert HWC -> CHW for the agent
        frame_chw = np.transpose(img, (2, 0, 1)).copy()

        # OpenCV visualization
        self._show_frame_cv2(frame_chw)

        return frame_chw

    def _stacked_obs(self) -> np.ndarray:
        return np.concatenate(list(self.frames), axis=0)

    def reset(self) -> np.ndarray:
        reset_seed = self._base_seed + self._reset_count
        self._reset_count += 1

        try:
            self._env.reset(seed=reset_seed)
        except TypeError:
            self._env.reset()

        self.episode_step = 0
        frame = self._render()
        self.frames.clear()
        for _ in range(self.frame_stack):
            self.frames.append(frame)
        return self._stacked_obs()

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, Dict[str, Any]]:
        total_reward = 0.0
        success = 0.0
        terminated = False
        truncated = False

        for _ in range(self.action_repeat):
            step_out = self._env.step(action)
            if len(step_out) == 5:
                _, reward, terminated, truncated, info = step_out
            else:
                _, reward, terminated, info = step_out
                truncated = False

            total_reward += float(reward)
            success = max(success, float(info.get("success", 0.0)))
            self.episode_step += 1

            if terminated or truncated or self.episode_step >= self.max_episode_steps:
                break

        done = bool(terminated or truncated or self.episode_step >= self.max_episode_steps)
        frame = self._render()
        self.frames.append(frame)
        return self._stacked_obs(), total_reward, done, {"success": success}

    def close(self) -> None:
        # Close the OpenCV window if it was created.
        if self.enable_cv2_vis and cv2 is not None and self._vis_window_initialized:
            try:
                cv2.destroyWindow(self.vis_window_name)
            except Exception:
                pass
            self._vis_window_initialized = False

        # Close the Gym env.
        try:
            self._env.close()
        except Exception:
            pass


# -----------------------------------------------------------------------------
# Evaluation
# -----------------------------------------------------------------------------

def evaluate(env: MetaWorldSingleCameraEnv, agent: DrMMetaWorldAgent, num_episodes: int, step: int) -> Dict[str, float]:
    returns = []
    successes = []

    for _ in range(num_episodes):
        obs = env.reset()
        done = False
        episode_return = 0.0
        episode_success = 0.0

        while not done:
            action = agent.act(obs, step=step, eval_mode=True)
            obs, reward, done, info = env.step(action)
            episode_return += float(reward)
            episode_success = max(episode_success, float(info["success"]))

        returns.append(episode_return)
        successes.append(episode_success)

    return {
        "eval/success_rate": float(np.mean(successes)),
        "eval/cumulative_reward": float(np.mean(returns)),
    }


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path to the single YAML config.")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        cfg: Dict[str, Any] = yaml.safe_load(f)

    # Fill image sizes into the vision config so `build_vision_encoder(...)`
    # can be reused without touching the existing codebase.
    vision_cfg = cfg.setdefault("vision", {})
    vision_cfg.setdefault("img_height", int(cfg["env"]["image_height"]))
    vision_cfg.setdefault("img_width", int(cfg["env"]["image_width"]))
    if vision_cfg.get("encoder_type", vision_cfg.get("name", "resnet50")) == "resnet50":
        res_cfg = vision_cfg.setdefault("resnet50", {})
        res_cfg.setdefault("img_height", int(cfg["env"]["image_height"]))
        res_cfg.setdefault("img_width", int(cfg["env"]["image_width"]))

    seed = int(cfg.get("seed", 0))
    set_seed(seed)

    device = torch.device(cfg.get("device", "cuda" if torch.cuda.is_available() else "cpu"))

    train_env = MetaWorldSingleCameraEnv(cfg, seed=seed)
    eval_env = MetaWorldSingleCameraEnv(cfg, seed=seed + 1)

    agent = DrMMetaWorldAgent(
        cfg=cfg,
        obs_shape=train_env.obs_shape,
        action_shape=train_env.action_shape,
        device=device,
    )

    tcfg = cfg["train"]
    replay_dir = Path(tcfg.get("replay_dir", "buffer"))
    replay_storage = ReplayBufferStorage(
        replay_dir=replay_dir,
        action_shape=train_env.action_shape,
    )

    replay_loader, replay_buffer = make_replay_loader(
        replay_dir=replay_dir,
        max_size=int(tcfg.get("replay_size", 500_000)),
        batch_size=int(tcfg.get("batch_size", 256)),
        num_workers=int(tcfg.get("replay_num_workers", 2)),
        save_snapshot=bool(tcfg.get("save_replay_snapshot", False)),
        nstep=int(tcfg.get("nstep", 3)),
        discount=float(tcfg.get("discount", 0.99)),
        fetch_every=int(tcfg.get("replay_fetch_every", 1000)),
    )
    replay_iter = None

    use_wandb = bool(cfg.get("wandb", {}).get("enabled", False)) and wandb is not None
    if use_wandb:
        wandb.init(
            project=str(cfg["wandb"].get("project", "drm-metaworld")),
            name=str(cfg["wandb"].get("name", f"{cfg['env']['env_name']}-{cfg['vision'].get('encoder_type', 'resnet50')}")),
            config=cfg,
        )

    num_train_steps = int(tcfg.get("num_train_steps", 1_000_000))
    seed_steps = int(tcfg.get("seed_steps", 4_000))
    batch_size = int(tcfg.get("batch_size", 256))
    update_every_steps = int(tcfg.get("update_every_steps", 2))
    gradient_steps = int(tcfg.get("gradient_steps", 1))
    eval_every_steps = int(tcfg.get("eval_every_steps", 10_000))
    log_every_steps = int(tcfg.get("log_every_steps", 1_000))
    num_eval_episodes = int(tcfg.get("num_eval_episodes", 10))
    discount = float(tcfg.get("discount", 0.99))

    ckpt_every = int(tcfg.get("checkpoint_every_steps", 0))
    ckpt_dir = Path(tcfg.get("checkpoint_dir", "checkpoints"))
    if ckpt_every > 0:
        ckpt_dir.mkdir(parents=True, exist_ok=True)

    rolling_success: Deque[float] = deque(maxlen=int(tcfg.get("rolling_window", 20)))
    rolling_return: Deque[float] = deque(maxlen=int(tcfg.get("rolling_window", 20)))

    obs = train_env.reset()
    replay_storage.add_initial(obs)
    episode_return = 0.0
    episode_success = 0.0
    update_metrics: Dict[str, float] = {}

    for step in range(1, num_train_steps + 1):
        if step <= seed_steps:
            action = train_env.action_space.sample().astype(np.float32)
        else:
            action = agent.act(obs, step=step, eval_mode=False).astype(np.float32)

        next_obs, reward, done, info = train_env.step(action)
        transition_discount = 0.0 if done else discount

        replay_storage.add(
            action=action,
            reward=reward,
            discount=transition_discount,
            next_obs=next_obs,
            done=done,
        )

        obs = next_obs
        episode_return += float(reward)
        episode_success = max(episode_success, float(info["success"]))

        # Standard off-policy updates
        if step > seed_steps and len(replay_storage) >= batch_size and step % update_every_steps == 0:
            # Lazily create the replay iterator only after enough data exists.
            # This matches the intended DrM usage much better.
            if replay_iter is None:
                replay_iter = iter(replay_loader)

            for _ in range(gradient_steps):
                update_metrics = agent.update(replay_iter, step)

        # Per-step logging (includes dormant ratio).
        if step % log_every_steps == 0:
            log_data = {"step": step}
            if update_metrics:
                log_data.update({f"train/{k}": v for k, v in update_metrics.items()})
            if rolling_success:
                log_data["train/rolling_success_rate"] = float(np.mean(rolling_success))
                log_data["train/rolling_cumulative_reward"] = float(np.mean(rolling_return))

            if use_wandb:
                wandb.log(log_data, step=step)
            else:
                print(log_data)

        # Periodic evaluation.
        if step % eval_every_steps == 0:
            eval_metrics = evaluate(eval_env, agent, num_episodes=num_eval_episodes, step=step)
            eval_metrics["step"] = step
            if use_wandb:
                wandb.log(eval_metrics, step=step)
            else:
                print(eval_metrics)

        # Optional checkpointing.
        if ckpt_every > 0 and step % ckpt_every == 0:
            ckpt_path = ckpt_dir / f"step_{step:08d}.pt"
            torch.save(
                {
                    "step": step,
                    "cfg": cfg,
                    "agent": agent.state_dict(),
                },
                ckpt_path,
            )

        # Episode boundary.
        if done:
            rolling_success.append(episode_success)
            rolling_return.append(episode_return)
            episode_log = {
                "step": step,
                "train/episode_success_rate": float(episode_success),
                "train/cumulative_reward": float(episode_return),
                "train/rolling_success_rate": float(np.mean(rolling_success)),
                "train/rolling_cumulative_reward": float(np.mean(rolling_return)),
                "train/dormant_ratio": float(update_metrics.get("dormant_ratio", agent.dormant_ratio)),
            }

            if use_wandb:
                wandb.log(episode_log, step=step)
            else:
                print(episode_log)

            obs = train_env.reset()

            replay_storage.add_initial(obs)

            episode_return = 0.0
            episode_success = 0.0

    train_env.close()
    eval_env.close()
    
    if use_wandb:
        wandb.finish()


if __name__ == "__main__":
    main()
