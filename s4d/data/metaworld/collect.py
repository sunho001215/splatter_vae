"""Meta-World collection: multi-camera RGB, metric depth, body-id maps, body poses, states.

Import order matters for GPU isolation: callers must run ``enforce_allowed_gpus``
and import ``mujoco_mig_setup`` before importing this module (it imports mujoco).
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import gymnasium as gym
import h5py
import metaworld  # noqa: F401  (registers Meta-World/MT1)
import mujoco
import numpy as np
from metaworld.policies import ENV_POLICY_MAP

from s4d.data.metaworld.cameras import FOVY_DEG, camera_rig, mujoco_free_camera

VERSION = "splatter4d-metaworld-v1"
DEPTH_UNIT_M = 1.0e-4  # uint16 depth in 0.1 mm
DEPTH_MAX_M = 6.5
BACKGROUND_BODY = 65535
NUM_EPISODES = 250


@dataclass
class CollectConfig:
    task: str
    output: str
    seed: int = 0
    height: int = 128
    width: int = 128
    max_steps: int = 450
    num_episodes: int = NUM_EPISODES
    # expert-guided: 30 demos at each noise strength
    expert_noise_levels: tuple = (0.0, 0.05, 0.15, 0.30, 0.50)
    expert_per_level: int = 30
    expert_noise_smoothing: float = 0.85
    expert_hold_steps: tuple = (8, 16)
    # perturb-and-recover
    perturb_episodes: int = 60
    perturb_start_fraction: tuple = (0.25, 0.55)
    perturb_duration: tuple = (8, 24)
    perturb_strength: tuple = (0.30, 0.70)
    perturb_hold_steps: tuple = (4, 10)
    perturb_min_recovery: int = 10
    # smooth random
    random_episodes: int = 40
    random_xyz_smoothing: float = 0.80
    random_gripper_smoothing: float = 0.95
    random_xyz_hold: tuple = (6, 14)
    random_gripper_hold: tuple = (30, 60)
    compression: dict = field(default_factory=lambda: {"compression": "gzip", "compression_opts": 4})

    @property
    def env_name(self) -> str:
        return f"{self.task}-v3"


# --------------------------------------------------------------------------- actions
class _SmoothNoise:
    def __init__(self, dim: int, smoothing: float, hold: int, rng: np.random.Generator):
        self.smoothing, self.hold, self.rng = smoothing, max(1, hold), rng
        self.target = rng.uniform(-1.0, 1.0, size=dim).astype(np.float32)
        self.value = self.target.copy()

    def sample(self, t: int) -> np.ndarray:
        if t > 0 and t % self.hold == 0:
            self.target = self.rng.uniform(-1.0, 1.0, size=self.target.shape).astype(np.float32)
        self.value = self.smoothing * self.value + (1.0 - self.smoothing) * self.target
        return self.value


class EpisodePlan:
    """One episode's action policy: ``mode`` in {expert_guided, perturb_recover, smooth_random}."""

    def __init__(self, mode: str, params: dict, rng: np.random.Generator):
        self.mode, self.params, self.rng = mode, params, rng
        if mode == "expert_guided":
            self.noise = _SmoothNoise(3, params["smoothing"], params["hold"], rng)
        elif mode == "perturb_recover":
            self.noise = _SmoothNoise(3, params["smoothing"], params["hold"], rng)
        elif mode == "smooth_random":
            self.xyz = _SmoothNoise(3, params["xyz_smoothing"], params["xyz_hold"], rng)
            self.grip = _SmoothNoise(1, params["gripper_smoothing"], params["gripper_hold"], rng)
        else:
            raise ValueError(mode)

    def action(self, expert: np.ndarray, t: int) -> np.ndarray:
        a = np.asarray(expert, dtype=np.float32).reshape(4).copy()
        if self.mode == "expert_guided":
            beta = self.params["strength"]
            if beta > 0:
                a[:3] = (1 - beta) * a[:3] + beta * self.noise.sample(t)
        elif self.mode == "perturb_recover":
            s, d = self.params["start"], self.params["duration"]
            if s <= t < s + d:
                beta = self.params["strength"]
                a[:3] = (1 - beta) * a[:3] + beta * self.noise.sample(t - s)
        else:
            a = np.concatenate([self.xyz.sample(t), self.grip.sample(t)])
        return np.clip(a, -1.0, 1.0).astype(np.float32)

    def may_stop_on_success(self, t: int) -> bool:
        if self.mode == "perturb_recover":
            return t + 1 >= self.params["start"] + self.params["duration"] + self.params["min_recovery"]
        return True


def episode_plans(cfg: CollectConfig, rng: np.random.Generator) -> list[EpisodePlan]:
    plans = []
    for level in cfg.expert_noise_levels:
        for _ in range(cfg.expert_per_level):
            plans.append(
                EpisodePlan(
                    "expert_guided",
                    {
                        "strength": float(level),
                        "smoothing": cfg.expert_noise_smoothing,
                        "hold": int(rng.integers(cfg.expert_hold_steps[0], cfg.expert_hold_steps[1] + 1)),
                    },
                    rng,
                )
            )
    lo, hi = (int(round(f * cfg.max_steps)) for f in cfg.perturb_start_fraction)
    latest = max(0, cfg.max_steps - cfg.perturb_duration[1] - cfg.perturb_min_recovery)
    lo, hi = min(lo, latest), min(max(lo, hi), latest)
    for _ in range(cfg.perturb_episodes):
        plans.append(
            EpisodePlan(
                "perturb_recover",
                {
                    "start": int(rng.integers(lo, hi + 1)),
                    "duration": int(rng.integers(cfg.perturb_duration[0], cfg.perturb_duration[1] + 1)),
                    "strength": float(rng.uniform(*cfg.perturb_strength)),
                    "smoothing": cfg.expert_noise_smoothing,
                    "hold": int(rng.integers(cfg.perturb_hold_steps[0], cfg.perturb_hold_steps[1] + 1)),
                    "min_recovery": cfg.perturb_min_recovery,
                },
                rng,
            )
        )
    for _ in range(cfg.random_episodes):
        plans.append(
            EpisodePlan(
                "smooth_random",
                {
                    "xyz_smoothing": cfg.random_xyz_smoothing,
                    "gripper_smoothing": cfg.random_gripper_smoothing,
                    "xyz_hold": int(rng.integers(cfg.random_xyz_hold[0], cfg.random_xyz_hold[1] + 1)),
                    "gripper_hold": int(rng.integers(cfg.random_gripper_hold[0], cfg.random_gripper_hold[1] + 1)),
                },
                rng,
            )
        )
    if len(plans) != cfg.num_episodes:
        raise ValueError(f"episode mixture yields {len(plans)} episodes, expected {cfg.num_episodes}")
    order = rng.permutation(len(plans))
    return [plans[i] for i in order]


# --------------------------------------------------------------------------- env + render
class MetaworldScene:
    """Meta-World env plus three MuJoCo renderers (RGB, depth, segmentation) and the camera rig."""

    def __init__(self, task: str, seed: int, height: int, width: int):
        self.env = gym.make("Meta-World/MT1", env_name=f"{task}-v3", seed=seed)
        self.env.reset(seed=seed)
        self.model = self.env.unwrapped.model
        self.data = self.env.unwrapped.data
        base = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "base")
        if (
            base < 0
            or not np.allclose(self.data.xpos[base], 0, atol=1e-7)
            or not np.allclose(self.data.xquat[base], [1, 0, 0, 0], atol=1e-7)
        ):
            raise ValueError("native simulator world is not the robot-base world")
        self.model.vis.global_.fovy = FOVY_DEG
        self.policy = ENV_POLICY_MAP[f"{task}-v3"]()
        self.rig = camera_rig(height, width)
        self.cams = [mujoco_free_camera(a, e) for a, e in zip(self.rig["azimuth"], self.rig["elevation"])]
        self.rgb_r = mujoco.Renderer(self.model, height=height, width=width)
        self.depth_r = mujoco.Renderer(self.model, height=height, width=width)
        self.depth_r.enable_depth_rendering()
        self.seg_r = mujoco.Renderer(self.model, height=height, width=width)
        self.seg_r.enable_segmentation_rendering()
        self.body_names = [
            mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_BODY, i) or f"body{i}" for i in range(self.model.nbody)
        ]

    def close(self) -> None:
        for r in (self.rgb_r, self.depth_r, self.seg_r):
            r.close()
        self.env.close()

    def render(self, cam_indices) -> tuple[np.ndarray, np.ndarray]:
        """RGB uint8 (N,H,W,3) and depth uint16 (N,H,W) for the given camera indices."""
        rgb, depth = [], []
        for i in cam_indices:
            self.rgb_r.update_scene(self.data, camera=self.cams[i])
            rgb.append(self.rgb_r.render().copy())
            self.depth_r.update_scene(self.data, camera=self.cams[i])
            depth.append(encode_depth(self.depth_r.render()))
        return np.stack(rgb), np.stack(depth)

    def render_body_ids(self, cam_indices) -> np.ndarray:
        out = []
        for i in cam_indices:
            self.seg_r.update_scene(self.data, camera=self.cams[i])
            out.append(segmentation_to_body_id(self.seg_r.render(), self.model))
        return np.stack(out)

    def body_poses(self) -> tuple[np.ndarray, np.ndarray]:
        return self.data.xpos.copy().astype(np.float32), self.data.xquat.copy().astype(np.float32)

    def state(self) -> tuple[np.ndarray, np.ndarray]:
        return self.data.qpos.copy(), self.data.qvel.copy()


def encode_depth(depth_m: np.ndarray) -> np.ndarray:
    d = np.asarray(depth_m, dtype=np.float64)
    valid = np.isfinite(d) & (d > 0) & (d < DEPTH_MAX_M)
    out = np.where(valid, np.round(d / DEPTH_UNIT_M), 0.0)
    return out.clip(0, 65534).astype(np.uint16)


def segmentation_to_body_id(seg: np.ndarray, model) -> np.ndarray:
    """MuJoCo segmentation (H,W,2)=(objid, objtype) -> uint16 body id (BACKGROUND_BODY elsewhere)."""
    objid, objtype = seg[..., 0].astype(np.int64), seg[..., 1].astype(np.int64)
    body = np.full(objid.shape, BACKGROUND_BODY, dtype=np.int64)
    geom = (objtype == int(mujoco.mjtObj.mjOBJ_GEOM)) & (objid >= 0)
    body[geom] = np.asarray(model.geom_bodyid)[objid[geom]]
    site = (objtype == int(mujoco.mjtObj.mjOBJ_SITE)) & (objid >= 0)
    body[site] = np.asarray(model.site_bodyid)[objid[site]]
    return body.astype(np.uint16)


# --------------------------------------------------------------------------- episode rollout
def rollout(
    scene: MetaworldScene, plan: EpisodePlan, seed: int, max_steps: int, train_cams: np.ndarray, all_cams: np.ndarray
) -> dict:
    env = scene.env
    obs, _ = env.reset(seed=seed)
    obs = np.asarray(obs, dtype=np.float32).ravel()
    buf: dict[str, list] = {
        k: [] for k in ("rgb", "depth", "body_id", "xpos", "xquat", "obs", "qpos", "qvel", "action", "reward", "success")
    }
    success_step = -1
    for t in range(max_steps):
        expert = np.asarray(scene.policy.get_action(obs), dtype=np.float32).reshape(4)
        action = plan.action(expert, t)
        rgb, depth = scene.render(all_cams)
        body_id = scene.render_body_ids(train_cams)
        xpos, xquat = scene.body_poses()
        qpos, qvel = scene.state()
        next_obs, reward, terminated, truncated, info = env.step(action)
        success = bool(info.get("success", 0.0) > 0.5)
        for k, v in (
            ("rgb", rgb),
            ("depth", depth),
            ("body_id", body_id),
            ("xpos", xpos),
            ("xquat", xquat),
            ("obs", obs.copy()),
            ("qpos", qpos),
            ("qvel", qvel),
            ("action", action),
            ("reward", np.float32(reward)),
            ("success", np.uint8(success)),
        ):
            buf[k].append(v)
        obs = np.asarray(next_obs, dtype=np.float32).ravel()
        if success and success_step < 0:
            success_step = t
        if success and plan.may_stop_on_success(t):
            break
        if terminated or truncated:
            break
    out = {k: np.stack(v) for k, v in buf.items()}
    out["success_step"] = success_step
    return out


def write_episode(group: h5py.Group, ep: dict, plan: EpisodePlan, seed: int, comp: dict) -> None:
    T, V, H, W, _ = ep["rgb"].shape
    group.attrs.update(
        {
            "mode": plan.mode,
            "seed": int(seed),
            "length": int(T),
            "success_step": int(ep["success_step"]),
            "params": json.dumps(plan.params),
        }
    )
    group.create_dataset("rgb", data=ep["rgb"], chunks=(1, 1, H, W, 3), **comp)
    group.create_dataset("depth", data=ep["depth"], chunks=(1, 1, H, W), shuffle=True, **comp)
    group.create_dataset("body_id", data=ep["body_id"], chunks=(1, 1, H, W), shuffle=True, **comp)
    for key in ("xpos", "xquat", "obs", "qpos", "qvel", "action", "reward", "success"):
        group.create_dataset(key, data=ep[key])


def collect_task(cfg: CollectConfig, *, log=print, max_episodes: int | None = None) -> Path:
    """Collect one task into ``cfg.output`` (written atomically via ``.incomplete``)."""
    output = Path(cfg.output)
    allowed = Path("/home/ws/data/metaworld/splatter4d_v1").resolve()
    if allowed not in output.resolve().parents:
        raise ValueError("collection output must be under /home/ws/data/metaworld/splatter4d_v1")
    if output.exists() or output.with_name(output.name + ".incomplete").exists():
        raise FileExistsError(f"refusing to overwrite {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    tmp = output.with_name(output.name + ".incomplete")
    scene = MetaworldScene(cfg.task, cfg.seed, cfg.height, cfg.width)
    rig = scene.rig
    train_cams = np.where(rig["is_train"])[0]
    all_cams = np.arange(len(rig["names"]))
    plans = episode_plans(cfg, np.random.default_rng(cfg.seed))
    if max_episodes is not None:
        if not 0 < max_episodes <= len(plans):
            raise ValueError("episode limit must be in [1,250]")
        plans = plans[:max_episodes]
    start = time.time()
    try:
        with h5py.File(tmp, "w") as f:
            f.attrs.update(
                {
                    "version": VERSION,
                    "task": cfg.task,
                    "env_name": cfg.env_name,
                    "height": cfg.height,
                    "width": cfg.width,
                    "depth_unit_m": DEPTH_UNIT_M,
                    "background_body": BACKGROUND_BODY,
                    "train_cameras": json.dumps([rig["names"][i] for i in train_cams]),
                    "eval_cameras": json.dumps([rig["names"][i] for i in all_cams if i not in set(train_cams)]),
                    "body_names": json.dumps(scene.body_names),
                    "config": json.dumps(asdict(cfg)),
                    "mujoco_version": mujoco.__version__,
                    "complete": False,
                    "num_episodes": len(plans),
                    "world_frame": "robot_base",
                    "dt_seconds": float(scene.env.unwrapped.dt),
                }
            )
            cams = f.create_group("cameras")
            cams.create_dataset("names", data=np.asarray(rig["names"], dtype=h5py.string_dtype()))
            for key in ("K", "c2w", "w2c", "is_train", "azimuth", "elevation"):
                cams.create_dataset(key, data=rig[key])
            episodes = f.create_group("episodes")
            for idx, plan in enumerate(plans):
                seed = cfg.seed * 100_000 + idx
                ep = rollout(scene, plan, seed, cfg.max_steps, train_cams, all_cams)
                write_episode(episodes.create_group(f"ep{idx:03d}"), ep, plan, seed, cfg.compression)
                f.flush()
                log(
                    f"[{cfg.task}] episode {idx + 1}/{len(plans)} mode={plan.mode} len={ep['rgb'].shape[0]} "
                    f"success_step={ep['success_step']} elapsed={time.time() - start:.0f}s"
                )
            f.attrs["complete"] = True
        os.replace(tmp, output)
    finally:
        scene.close()
    return output
