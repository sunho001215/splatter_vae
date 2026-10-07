"""Single-camera Meta-World environment for RL, ported from the reference ``MetaWorldSingleCameraEnv``.

Images are rendered with the same free cameras, field of view and resolution as the pretraining data
(``s4d.data.metaworld.cameras``). A camera *path* is a list of free-camera poses indexed by render call:
a fixed camera is a one-pose path, and the reference lateral/circular perturbation trajectories loop.

Every method receives the reference observation: the last 3 rendered frames, one per agent step,
so consecutive frames are ``action_repeat`` = 2 simulator steps apart. Pretraining strides include 2.
"""

from __future__ import annotations

import math
import random
from collections import deque
from dataclasses import dataclass

import numpy as np

from s4d.data.metaworld.cameras import EVAL_CAMERAS, FOVY_DEG, LOOKAT, RADIUS, TRAIN_CAMERAS


@dataclass(frozen=True)
class FreeCamera:
    azimuth: float
    elevation: float
    distance: float = RADIUS


def orbit_camera(theta: float, phi: float) -> FreeCamera:
    """Rig convention: ``theta`` is the downward tilt, so MuJoCo elevation is ``-theta``."""
    return FreeCamera(azimuth=float(phi), elevation=-float(theta))


TRAIN_PATHS = {f"train{i}": [orbit_camera(t, p)] for i, (t, p) in enumerate(TRAIN_CAMERAS)}
HELDOUT_PATHS = {f"eval{i}": [orbit_camera(t, p)] for i, (t, p) in enumerate(EVAL_CAMERAS)}


def _forward(camera: FreeCamera) -> np.ndarray:
    az, el = math.radians(camera.azimuth), math.radians(camera.elevation)
    return np.array([math.cos(el) * math.cos(az), math.cos(el) * math.sin(az), math.sin(el)])


def _camera_at(position: np.ndarray) -> FreeCamera:
    relative = np.asarray(LOOKAT) - position
    distance = float(np.linalg.norm(relative))
    return FreeCamera(
        azimuth=math.degrees(math.atan2(relative[1], relative[0])),
        elevation=math.degrees(math.asin(np.clip(relative[2] / distance, -1.0, 1.0))),
        distance=distance,
    )


def lateral_offsets(num_frames: int, amplitude: float) -> np.ndarray:
    """Reference reciprocating path: center -> right -> center -> left -> center."""
    phase = np.linspace(0.0, 4.0, num_frames)
    return amplitude * np.where(phase < 1.0, phase, np.where(phase < 3.0, 2.0 - phase, phase - 4.0))


def trajectory_path(
    kind: str,
    base: str = "train1",
    num_frames: int = 72,
    lateral_amplitude: float = 0.12,
    circular_azimuth_deg: float = 10.0,
    circular_elevation_deg: float = 6.0,
) -> list[FreeCamera]:
    """Reference ``evaluate_camera_trajectory.py`` defaults: base cam1, 72 poses, 0.12 m, +/-10 deg az, +/-6 deg el."""
    theta, phi = TRAIN_CAMERAS[int(base.removeprefix("train"))]
    camera = orbit_camera(theta, phi)
    if kind == "lateral":
        forward = _forward(camera)
        position = np.asarray(LOOKAT) - camera.distance * forward
        right = np.cross(forward, [0.0, 0.0, 1.0])
        right /= np.linalg.norm(right)
        return [_camera_at(position + offset * right) for offset in lateral_offsets(num_frames, lateral_amplitude)]
    if kind == "circular":
        angles = 2.0 * math.pi * np.arange(num_frames) / num_frames
        return [
            orbit_camera(theta + circular_elevation_deg * math.sin(a), phi + circular_azimuth_deg * math.cos(a))
            for a in angles
        ]
    raise ValueError(f"unknown trajectory {kind!r}")


def check_pretraining_spacing(strides, num_frames: int, action_repeat: int) -> None:
    """The RL frame spacing (``action_repeat`` simulator steps) must be a pretraining stride of the encoder."""
    if num_frames > 1 and action_repeat not in [int(s) for s in strides or ()]:
        raise ValueError(f"RL frames are {action_repeat} simulator steps apart, outside pretraining strides {strides}")


class MetaWorldCameraEnv:
    """Returns channel-stacked uint8 pixels (3*T,H,W) and proprio (P,), like the reference wrapper."""

    def __init__(
        self,
        task: str,
        seed: int,
        *,
        image_size: int,
        frame_stack: int,
        action_repeat: int,
        max_episode_steps: int,
        proprio_indices=(0, 1, 2, 3),
        renderer=None,
    ):
        """``renderer``: share another environment's MuJoCo renderer (same task). Each renderer holds its own copy of
        the scene textures (~0.44 GB of GPU memory); poses come from this env's ``MjData``, so shared rendering is
        pixel-identical up to the renderer's own +/-1 intensity noise (checked on all eight tasks)."""
        import gymnasium as gym
        import metaworld  # noqa: F401  (registers Meta-World/MT1)
        import mujoco

        self._mujoco = mujoco
        self.env = gym.make("Meta-World/MT1", env_name=f"{task}-v3", seed=seed)
        base = self.env.unwrapped
        if hasattr(base, "_freeze_rand_vec"):
            base._freeze_rand_vec = False
        self.model, self.data = base.model, base.data
        self.model.vis.global_.fovy = FOVY_DEG
        self.owns_renderer = renderer is None
        self.renderer = mujoco.Renderer(self.model, height=image_size, width=image_size) if renderer is None else renderer
        self.frame_stack = int(frame_stack)
        self.action_repeat, self.max_episode_steps = int(action_repeat), int(max_episode_steps)
        self.proprio_indices = list(proprio_indices)
        self.history: deque[np.ndarray] = deque(maxlen=self.frame_stack)
        self.action_space = self.env.action_space
        self.action_space.seed(seed + 23456)
        self.base_seed, self.reset_count = int(seed), 0
        self.path: list[FreeCamera] = []
        self.render_count = 0
        self.episode_step = 0

    @property
    def action_dim(self) -> int:
        return int(self.action_space.shape[0])

    @property
    def proprio_dim(self) -> int:
        return len(self.proprio_indices)

    def _render(self) -> np.ndarray:
        pose = self.path[self.render_count % len(self.path)]
        self.render_count += 1
        cam = self._mujoco.MjvCamera()
        cam.type = self._mujoco.mjtCamera.mjCAMERA_FREE
        cam.lookat[:] = np.asarray(LOOKAT, dtype=np.float64)
        cam.distance, cam.azimuth, cam.elevation = pose.distance, pose.azimuth, pose.elevation
        self.renderer.update_scene(self.data, camera=cam)
        frame = np.transpose(self.renderer.render(), (2, 0, 1)).copy()
        self.history.append(frame)
        return frame

    def latest_frame(self) -> np.ndarray:
        return self.history[-1].copy()

    def stacked(self) -> np.ndarray:
        return np.concatenate(list(self.history), axis=0)

    def _proprio(self, state_obs) -> np.ndarray:
        return np.asarray(state_obs, dtype=np.float32).reshape(-1)[self.proprio_indices].copy()

    def reset(self, path: list[FreeCamera], seed: int | None = None) -> tuple[np.ndarray, np.ndarray]:
        """Training: reference behaviour. Evaluation: pass an explicit episode seed.

        Meta-World 3.0 ignores the ``reset`` seed, and the MT1 task wrapper re-freezes the random vector on every
        reset, so an episode uses one of the 50 MT1 configurations fixed by the construction seed, drawn from the
        env RNG in reset order; the hand also starts from wherever the previous episode left it. With an explicit
        seed the env RNG is reseeded (it selects the configuration) and the simulator data is reset to the model
        defaults first, so the start state depends only on (construction seed, episode seed).
        """
        if seed is not None:
            self.env.unwrapped.seed(int(seed))
            self._mujoco.mj_resetData(self.model, self.data)
        state_obs, _ = self.env.reset(seed=self.base_seed + self.reset_count if seed is None else int(seed))
        self.reset_count += 1
        self.path, self.render_count, self.episode_step = list(path), 0, 0
        self.history.clear()
        frame = self._render()
        while len(self.history) < self.history.maxlen:
            self.history.append(frame)
        return self.stacked(), self._proprio(state_obs)

    def step(self, action: np.ndarray):
        total_reward, success, done = 0.0, 0.0, False
        for _ in range(self.action_repeat):
            state_obs, reward, terminated, truncated, info = self.env.step(action)
            total_reward += float(reward)
            success = max(success, float(info.get("success", 0.0)))
            self.episode_step += 1
            done = bool(terminated or truncated or self.episode_step >= self.max_episode_steps)
            if done:
                break
        self._render()
        return self.stacked(), self._proprio(state_obs), total_reward, done, {"success": success}

    def close(self) -> None:
        if self.owns_renderer:
            self.renderer.close()
        self.env.close()


def env_kwargs(cfg: dict) -> dict:
    """Constructor arguments of ``MetaWorldCameraEnv`` from a resolved RL config."""
    return {k: cfg["env"][k] for k in ("image_size", "frame_stack", "action_repeat", "max_episode_steps", "proprio_indices")}


class TrainCameraSampler:
    """Reference behaviour: each training episode uses one training camera drawn uniformly."""

    def __init__(self, seed: int):
        self.rng = random.Random(int(seed) + 12345)
        self.names = list(TRAIN_PATHS)

    def __call__(self) -> list[FreeCamera]:
        return TRAIN_PATHS[self.names[self.rng.randrange(len(self.names))]]
