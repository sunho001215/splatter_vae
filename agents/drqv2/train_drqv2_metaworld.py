from __future__ import annotations

import argparse
import os
import re
import sys
from pathlib import Path

# Set default MuJoCo rendering backend to EGL for headless environments
os.environ.setdefault("MUJOCO_GL", "egl")

# Add project root to sys.path for imports
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import mujoco_mig_setup

import random
from collections import deque
from typing import Any, Deque, Dict, List, Optional, Tuple

import cv2
import gymnasium as gym
import metaworld
import mujoco
import numpy as np
import torch
import wandb
import yaml

from agents.drqv2.drqv2_metaworld import DrQv2MetaWorldAgent
from agents.drqv2.replay_buffer import MemmapReplayBufferStorage, make_memmap_replay_loader
from dataset.metaworld_demo_collect.demo_collector.camera_math import spherical_camera_pose


def set_seed(seed: int) -> None:
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _format_step_tag(step: int) -> str:
    if step > 0 and step % 1000 == 0:
        return f"{step // 1000}k"
    return str(step)


def _checkpoint_step_from_path(path: str) -> Optional[int]:
    match = re.search(r"step[_-]?(\d+)", path, flags=re.IGNORECASE)
    return int(match.group(1)) if match else None


def _encoder_checkpoint_step(cfg: Dict[str, Any]) -> Optional[int]:
    vision_cfg = cfg.get("vision", {})
    if not isinstance(vision_cfg, dict):
        return None

    encoder_type = str(vision_cfg.get("encoder_type", vision_cfg.get("name", ""))).lower()
    section_keys = [
        encoder_type,
        encoder_type.replace("_", ""),
        "reviwo",
        "sincro",
        "splatter_vae",
        "splattervae",
    ]
    for key in dict.fromkeys(section_keys):
        section = vision_cfg.get(key)
        if isinstance(section, dict) and section.get("checkpoint_path") is not None:
            step = _checkpoint_step_from_path(str(section["checkpoint_path"]))
            if step is not None:
                return step

    for section in vision_cfg.values():
        if isinstance(section, dict) and section.get("checkpoint_path") is not None:
            step = _checkpoint_step_from_path(str(section["checkpoint_path"]))
            if step is not None:
                return step
    return None


def _wandb_run_name(cfg: Dict[str, Any], wandb_cfg: Dict[str, Any]) -> str:
    default_name = f"{cfg['env']['env_name']}-{cfg['vision'].get('encoder_type', 'convnet')}"
    return re.sub(r"[_-]seed\d+$", "", str(wandb_cfg.get("name", default_name)), flags=re.IGNORECASE)


def _wandb_tags(cfg: Dict[str, Any], wandb_cfg: Dict[str, Any]) -> List[str]:
    tags: List[str] = []

    def add_tag(tag: str) -> None:
        if tag and tag not in tags:
            tags.append(tag)

    explicit_tags = wandb_cfg.get("tags") or []
    if isinstance(explicit_tags, str):
        explicit_tags = [explicit_tags]
    for tag in explicit_tags:
        add_tag(str(tag))

    seed = cfg.get("seed")
    if seed is not None:
        add_tag(f"seed_{seed}")

    feature_dim = cfg.get("agent", {}).get("feature_dim")
    if feature_dim is not None:
        add_tag(f"feature_dim_{feature_dim}")

    frame_stack = cfg.get("env", {}).get("frame_stack")
    if frame_stack is not None:
        add_tag(f"frame_stack_{frame_stack}")

    checkpoint_step = _encoder_checkpoint_step(cfg)
    if checkpoint_step is not None:
        add_tag(f"checkpoint_step_{_format_step_tag(checkpoint_step)}")

    return tags


class MetaWorldSingleCameraEnv:
    """
    A wrapper for MetaWorld environments with single-camera rendering and frame stacking.

    Returns:
        obs_pixels: (3*T, H, W) uint8
        proprio:    (P,) float32

    Default proprio_indices=[0,1,2,3] which usually correspond to tcp xyz + gripper opening.
    Adjust in YAML if needed.
    """

    def __init__(self, cfg: Dict[str, Any], seed: int) -> None:
        env_cfg = cfg["env"]
        self.env_name = str(env_cfg["env_name"])
        self.image_height = int(env_cfg["image_height"])
        self.image_width = int(env_cfg["image_width"])
        self.frame_stack = int(env_cfg.get("frame_stack", 3))
        self.action_repeat = int(env_cfg.get("action_repeat", 1))
        self.max_episode_steps = int(env_cfg.get("max_episode_steps", 250))
        self.proprio_indices = list(env_cfg.get("proprio_indices", [0, 1, 2, 3]))

        raw_cameras = env_cfg.get("cameras", None)
        self.camera_cfgs = [dict(env_cfg["camera"])] if raw_cameras is None else [dict(cam) for cam in raw_cameras]
        self.num_cameras = len(self.camera_cfgs)
        self.current_camera_index = 0
        self.current_camera_cfg = dict(self.camera_cfgs[0])
        self._camera_rng = random.Random(int(seed) + 12345)

        raw_lookat = env_cfg.get("lookat", [0.0, 0.6, 0.0])
        self.up = np.asarray(env_cfg.get("up", [0.0, 0.0, 1.0]), dtype=np.float64)
        self.benchmark_id = str(env_cfg.get("benchmark_id", "Meta-World/MT1"))
        gym_env_name = self.env_name if self.env_name.endswith("-v3") else f"{self.env_name}-v3"
        self._env = gym.make(self.benchmark_id, env_name=gym_env_name, seed=seed)
        self._model, self._data = self._unwrap_mujoco(self._env)
        self.lookat = np.asarray(raw_lookat, dtype=np.float64) if raw_lookat is not None else np.array(self._model.stat.center, dtype=np.float64)

        self._renderer = mujoco.Renderer(self._model, height=self.image_height, width=self.image_width)
        self._base_seed = int(seed)
        self._reset_count = 0
        base_env = self._env.unwrapped
        if hasattr(base_env, "_freeze_rand_vec"):
            base_env._freeze_rand_vec = False

        self.action_space = self._env.action_space
        self.frames: Deque[np.ndarray] = deque(maxlen=self.frame_stack)
        self.episode_step = 0
        self._last_proprio = np.zeros((len(self.proprio_indices),), dtype=np.float32)

        vis_cfg = dict(cfg.get("visualization", {}))
        self.enable_cv2_vis = bool(vis_cfg.get("enabled", False))
        self.vis_window_name = str(vis_cfg.get("window_name", f"MetaWorld-{self.env_name}"))
        self.vis_wait_key = int(vis_cfg.get("wait_key", 1))
        self.vis_scale = float(vis_cfg.get("scale", 1.0))
        self._vis_window_initialized = False

    @property
    def obs_shape(self) -> Tuple[int, int, int]:
        """Shape of the observation (pixels)."""
        return (3 * self.frame_stack, self.image_height, self.image_width)

    @property
    def proprio_shape(self) -> Tuple[int, ...]:
        """Shape of the proprioceptive observation."""
        return (len(self.proprio_indices),)

    @property
    def action_shape(self) -> Tuple[int, ...]:
        """Shape of the action space."""
        return tuple(self.action_space.shape)

    def _unwrap_mujoco(self, env):
        """Unwrap the environment to access MuJoCo model and data."""
        e = env.unwrapped
        if hasattr(e, "model") and hasattr(e, "data"):
            return e.model, e.data
        if hasattr(e, "sim"):
            return e.sim.model, e.sim.data
        raise RuntimeError("Could not find MuJoCo model/data on env.unwrapped")

    def _extract_proprio_from_state_obs(self, state_obs: np.ndarray) -> np.ndarray:
        """Extract proprioceptive features from state observation."""
        state_obs = np.asarray(state_obs, dtype=np.float32).reshape(-1)
        if max(self.proprio_indices) >= len(state_obs):
            raise ValueError(f"proprio_indices={self.proprio_indices} out of range for state obs shape {state_obs.shape}")
        return state_obs[self.proprio_indices].astype(np.float32).copy()

    def _set_episode_camera(self, camera_index: Optional[int] = None) -> None:
        """Set the camera for the current episode."""
        self.current_camera_index = int(self._camera_rng.randrange(self.num_cameras)) if camera_index is None else int(camera_index)
        if self.current_camera_index < 0 or self.current_camera_index >= self.num_cameras:
            raise ValueError(f"camera_index out of range: {self.current_camera_index}")
        self.current_camera_cfg = dict(self.camera_cfgs[self.current_camera_index])

    def get_camera_name(self, camera_index: int) -> str:
        """Get the name of a camera by index."""
        return str(self.camera_cfgs[int(camera_index)].get("name", f"camera_{camera_index}"))

    def get_current_camera_name(self) -> str:
        """Get the name of the current camera."""
        return str(self.current_camera_cfg.get("name", f"camera_{self.current_camera_index}"))

    def get_last_frame(self) -> np.ndarray:
        """Get the last rendered frame."""
        return self.frames[-1].copy()

    def _make_free_camera(self) -> "mujoco.MjvCamera":
        """Create a free camera based on current configuration."""
        cam_cfg = self.current_camera_cfg
        pose = spherical_camera_pose(
            name=str(cam_cfg["name"]),
            r=float(cam_cfg["r"]),
            theta_deg=float(cam_cfg["theta"]),
            phi_deg=float(cam_cfg["phi"]),
            lookat=self.lookat,
            up=self.up,
            fovy_deg=float(cam_cfg["fovy"])
        )
        cam = mujoco.MjvCamera()
        cam.type = mujoco.mjtCamera.mjCAMERA_FREE
        cam.lookat[:] = self.lookat.astype(np.float64)
        rel = pose.pos.astype(np.float64) - self.lookat.astype(np.float64)
        cam.distance = float(np.linalg.norm(rel))
        cam.azimuth = float(np.degrees(np.arctan2(rel[1], rel[0])))
        cam.elevation = float(np.degrees(np.arctan2(rel[2], np.sqrt(rel[0] ** 2 + rel[1] ** 2))))
        return cam

    def _show_frame_cv2(self, frame_chw: np.ndarray) -> None:
        """Display the frame using OpenCV if visualization is enabled."""
        if not self.enable_cv2_vis:
            return
        img = np.transpose(frame_chw, (1, 2, 0))
        if self.vis_scale != 1.0:
            new_w = max(1, int(img.shape[1] * self.vis_scale))
            new_h = max(1, int(img.shape[0] * self.vis_scale))
            img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_NEAREST if self.vis_scale >= 1.0 else cv2.INTER_AREA)
        img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        if not self._vis_window_initialized:
            cv2.namedWindow(self.vis_window_name, cv2.WINDOW_NORMAL)
            self._vis_window_initialized = True
        cv2.imshow(self.vis_window_name, img_bgr)
        cv2.waitKey(max(1, self.vis_wait_key))

    def _render(self) -> np.ndarray:
        """Render the current frame."""
        self._renderer.update_scene(self._data, camera=self._make_free_camera())
        img = np.asarray(self._renderer.render(), dtype=np.uint8)
        frame_chw = np.transpose(img, (2, 0, 1)).copy()
        self._show_frame_cv2(frame_chw)
        return frame_chw

    def _stacked_obs(self) -> np.ndarray:
        """Stack the frames for observation."""
        return np.concatenate(list(self.frames), axis=0)

    def reset(self, camera_index: Optional[int] = None) -> tuple[np.ndarray, np.ndarray]:
        """Reset the environment and return initial observation."""
        reset_seed = self._base_seed + self._reset_count
        self._reset_count += 1
        state_obs, _info = self._env.reset(seed=reset_seed)
        self._set_episode_camera(camera_index)
        self._last_proprio = self._extract_proprio_from_state_obs(state_obs)
        self.episode_step = 0
        frame = self._render()
        self.frames.clear()
        for _ in range(self.frame_stack):
            self.frames.append(frame)
        return self._stacked_obs(), self._last_proprio.copy()

    def step(self, action: np.ndarray) -> tuple[np.ndarray, np.ndarray, float, bool, Dict[str, Any]]:
        """Step the environment with the given action."""
        total_reward = 0.0
        success = 0.0
        terminated = False
        truncated = False
        last_state_obs = None
        for _ in range(self.action_repeat):
            last_state_obs, reward, terminated, truncated, info = self._env.step(action)
            total_reward += float(reward)
            success = max(success, float(info.get("success", 0.0)))
            self.episode_step += 1
            if terminated or truncated or self.episode_step >= self.max_episode_steps:
                break
        self._last_proprio = self._extract_proprio_from_state_obs(last_state_obs)
        frame = self._render()
        self.frames.append(frame)
        done = bool(terminated or truncated or self.episode_step >= self.max_episode_steps)
        return self._stacked_obs(), self._last_proprio.copy(), total_reward, done, {
            "success": success,
            "camera_index": int(self.current_camera_index),
            "camera_name": self.get_current_camera_name()
        }

    def close(self) -> None:
        """Close the environment and visualization windows."""
        if self.enable_cv2_vis and self._vis_window_initialized:
            try:
                cv2.destroyWindow(self.vis_window_name)
            except Exception:
                pass
        try:
            self._env.close()
        except Exception:
            pass


@torch.inference_mode()
def obs_to_policy_input(agent: DrQv2MetaWorldAgent, obs_pixels: np.ndarray) -> np.ndarray:
    """Convert stacked observation pixels to the agent's acting format."""
    if agent.use_pixels:
        return np.asarray(obs_pixels, dtype=np.uint8)
    feat = agent.encoder.extract_cacheable_feature(torch.as_tensor(obs_pixels, device=agent.device).unsqueeze(0))
    return feat.squeeze(0).cpu().numpy()


@torch.inference_mode()
def obs_to_replay_atom(
    agent: DrQv2MetaWorldAgent,
    obs_pixels: np.ndarray,
    latest_frame: np.ndarray,
    policy_obs: np.ndarray | None = None,
) -> np.ndarray:
    """Convert the latest RGB frame to the compact memmap replay atom."""
    if agent.encoder.replay_atom_is_feature:
        if policy_obs is not None:
            if agent.encoder.replay_atom_is_stack_feature:
                return np.asarray(policy_obs, dtype=agent.encoder.replay_atom_dtype)
            return np.asarray(policy_obs[-1], dtype=agent.encoder.replay_atom_dtype)
        if agent.encoder.replay_atom_is_stack_feature:
            feat = agent.encoder.extract_cacheable_feature(torch.as_tensor(obs_pixels, device=agent.device).unsqueeze(0))
        else:
            feat = agent.encoder.extract_single_frame_feature(
                torch.as_tensor(latest_frame, device=agent.device).unsqueeze(0)
            )
        return feat.squeeze(0).cpu().numpy()
    return np.asarray(latest_frame, dtype=np.uint8)


def evaluate(env: MetaWorldSingleCameraEnv, agent: DrQv2MetaWorldAgent, num_episodes: int, step: int, *, log_videos: bool = False, video_fps: int = 15) -> Dict[str, Any]:
    """Evaluate the agent on the environment."""
    num_cameras = int(env.num_cameras)
    base = num_episodes // num_cameras
    rem = num_episodes % num_cameras
    per_camera_counts = [base + (1 if i < rem else 0) for i in range(num_cameras)]
    metrics: Dict[str, Any] = {}
    rets = []
    succs = []
    for cam_idx in range(num_cameras):
        n = int(per_camera_counts[cam_idx])
        if n <= 0:
            continue
        cam_returns = []
        cam_successes = []
        for ep_idx in range(n):
            obs_pixels, proprio = env.reset(camera_index=cam_idx)
            obs_for_policy = obs_to_policy_input(agent, obs_pixels)
            done = False
            ep_ret = 0.0
            ep_succ = 0.0
            frames = [env.get_last_frame()] if log_videos and ep_idx == 0 else []
            while not done:
                action = agent.act(obs_for_policy, proprio, step=step, eval_mode=True)
                next_pixels, next_proprio, reward, done, info = env.step(action)
                obs_for_policy = obs_to_policy_input(agent, next_pixels)
                proprio = next_proprio
                ep_ret += float(reward)
                ep_succ = max(ep_succ, float(info["success"]))
                if log_videos and ep_idx == 0:
                    frames.append(env.get_last_frame())
            cam_returns.append(ep_ret)
            cam_successes.append(ep_succ)
            if log_videos and ep_idx == 0 and len(frames) > 0:
                metrics[f"eval/view_{cam_idx}_video"] = wandb.Video(np.stack(frames, axis=0), fps=video_fps, format="mp4")
        ret = float(np.mean(cam_returns))
        succ = float(np.mean(cam_successes))
        rets.append(ret)
        succs.append(succ)
        metrics[f"eval/view_{cam_idx}_cumulative_reward"] = ret
        metrics[f"eval/view_{cam_idx}_success_rate"] = succ
    metrics["eval/cumulative_reward"] = float(np.mean(rets))
    metrics["eval/success_rate"] = float(np.mean(succs))
    return metrics


def main() -> None:
    """Main training loop."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()
    with open(args.config, "r") as f:
        cfg: Dict[str, Any] = yaml.safe_load(f)
    cfg.setdefault("vision", {}).setdefault("img_height", int(cfg["env"]["image_height"]))
    cfg["vision"].setdefault("img_width", int(cfg["env"]["image_width"]))
    seed = int(cfg.get("seed", 0))
    set_seed(seed)
    device = torch.device(cfg.get("device", "cuda" if torch.cuda.is_available() else "cpu"))
    cfg["device"] = str(device)

    train_env = MetaWorldSingleCameraEnv(cfg, seed=seed)
    eval_env = MetaWorldSingleCameraEnv(cfg, seed=seed + 1)
    agent = DrQv2MetaWorldAgent(cfg, train_env.obs_shape, train_env.action_shape, train_env.proprio_shape, device)

    tcfg = cfg["train"]
    replay_dir = Path(tcfg.get("replay_dir", "buffer"))
    nstep = int(tcfg.get("nstep", 3))
    replay_storage = MemmapReplayBufferStorage(
        replay_dir,
        agent.encoder.replay_atom_shape,
        agent.encoder.replay_atom_dtype,
        train_env.proprio_shape,
        train_env.action_shape,
        int(tcfg.get("replay_size", 500_000)),
        int(agent.encoder.replay_atom_frame_stack),
        nstep,
    )
    replay_loader, replay_buffer = make_memmap_replay_loader(
        replay_dir,
        agent.encoder.replay_atom_shape,
        agent.encoder.replay_atom_dtype,
        train_env.proprio_shape,
        train_env.action_shape,
        int(tcfg.get("replay_size", 500_000)),
        int(tcfg.get("batch_size", 256)),
        int(tcfg.get("replay_num_workers", 2)),
        int(agent.encoder.replay_atom_frame_stack),
        nstep,
        float(tcfg.get("discount", 0.99)),
    )
    replay_iter = None

    wandb_cfg = cfg.get("wandb", {})
    use_wandb = bool(wandb_cfg.get("enabled", True))
    if use_wandb:
        wandb.init(
            project=str(wandb_cfg.get("project", "drqv2-metaworld")),
            name=_wandb_run_name(cfg, wandb_cfg),
            config=cfg,
            tags=_wandb_tags(cfg, wandb_cfg)
        )

    num_train_steps = int(tcfg.get("num_train_steps", 1_000_000))
    seed_steps = int(tcfg.get("seed_steps", 4_000))
    batch_size = int(tcfg.get("batch_size", 256))
    update_every_steps = int(tcfg.get("update_every_steps", 2))
    gradient_steps = int(tcfg.get("gradient_steps", 1))
    eval_every_steps = int(tcfg.get("eval_every_steps", 10_000))
    log_every_steps = int(tcfg.get("log_every_steps", 1_000))
    num_eval_episodes = int(tcfg.get("num_eval_episodes", 10))
    ckpt_every = int(tcfg.get("checkpoint_every_steps", 0))
    ckpt_dir = Path(tcfg.get("checkpoint_dir", "checkpoints"))
    if ckpt_every > 0:
        ckpt_dir.mkdir(parents=True, exist_ok=True)

    rolling_success: Deque[float] = deque(maxlen=int(tcfg.get("rolling_window", 20)))
    rolling_return: Deque[float] = deque(maxlen=int(tcfg.get("rolling_window", 20)))
    obs_pixels, proprio = train_env.reset()
    obs_for_policy = obs_to_policy_input(agent, obs_pixels)
    obs_atom = obs_to_replay_atom(agent, obs_pixels, train_env.get_last_frame(), obs_for_policy)
    replay_storage.add_initial(obs_atom, proprio)
    episode_return = 0.0
    episode_success = 0.0
    update_metrics: Dict[str, float] = {}

    for step in range(1, num_train_steps + 1):
        action = (
            train_env.action_space.sample().astype(np.float32)
            if step <= seed_steps
            else agent.act(obs_for_policy, proprio, step=step, eval_mode=False).astype(np.float32)
        )
        next_pixels, next_proprio, reward, done, info = train_env.step(action)
        next_for_policy = obs_to_policy_input(agent, next_pixels)
        next_atom = obs_to_replay_atom(agent, next_pixels, train_env.get_last_frame(), next_for_policy)
        # Store the environment continuation flag here. ReplayBuffer applies the
        # algorithmic discount when it builds n-step returns.
        replay_storage.add(action, reward, 0.0 if done else 1.0, next_atom, next_proprio, done)
        obs_pixels, obs_for_policy, proprio = next_pixels, next_for_policy, next_proprio
        episode_return += float(reward)
        episode_success = max(episode_success, float(info["success"]))

        if step > seed_steps and len(replay_storage) >= batch_size and step % update_every_steps == 0:
            if replay_iter is None:
                replay_iter = iter(replay_loader)
            for _ in range(gradient_steps):
                update_metrics = agent.update(replay_iter, step)

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

        if step % eval_every_steps == 0:
            eval_metrics = evaluate(
                eval_env,
                agent,
                num_episodes=num_eval_episodes,
                step=step,
                log_videos=use_wandb,
                video_fps=int(cfg.get("eval_video_fps", 30))
            )
            eval_metrics["step"] = step
            if use_wandb:
                wandb.log(eval_metrics, step=step)
            else:
                print({k: v for k, v in eval_metrics.items() if not k.endswith("_video")})

        if ckpt_every > 0 and step % ckpt_every == 0:
            torch.save(
                {"step": step, "cfg": cfg, "agent": agent.state_dict()},
                ckpt_dir / f"step_{step:08d}.pt"
            )

        if done:
            rolling_success.append(episode_success)
            rolling_return.append(episode_return)
            episode_log = {
                "step": step,
                "train/episode_success_rate": float(episode_success),
                "train/cumulative_reward": float(episode_return),
                "train/rolling_success_rate": float(np.mean(rolling_success)),
                "train/rolling_cumulative_reward": float(np.mean(rolling_return))
            }
            if use_wandb:
                wandb.log(episode_log, step=step)
            else:
                print(episode_log)
            obs_pixels, proprio = train_env.reset()
            obs_for_policy = obs_to_policy_input(agent, obs_pixels)
            obs_atom = obs_to_replay_atom(agent, obs_pixels, train_env.get_last_frame(), obs_for_policy)
            replay_storage.add_initial(obs_atom, proprio)
            episode_return = 0.0
            episode_success = 0.0

    train_env.close()
    eval_env.close()
    if use_wandb:
        wandb.finish()


if __name__ == "__main__":
    main()
