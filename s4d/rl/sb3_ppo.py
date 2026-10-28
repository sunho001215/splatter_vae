"""PPO pilot: Stable-Baselines3 v2.9.0 PPO on the shared reference protocol (user directive 2026-10-28).

Upstream: github.com/DLR-RM/stable-baselines3 @ 8908708 (v2.9.0, MIT; submodule ``third_party/stable-baselines3``),
imported unchanged (its core dependencies gymnasium, numpy, torch and cloudpickle are the project's). PPO, its rollout
buffer, GAE, clipping, value/entropy losses, optimiser, VecNormalize and the time-limit bootstrap
(``on_policy_algorithm.py:236-245``: value of ``terminal_observation`` when ``TimeLimit.truncated``) are SB3's.

Thin adapters:
- ``ProtocolEnv``: a gymnasium env around ``MetaWorldCameraEnv`` (Meta-World v3, 128 px, 3-frame history one agent step
  apart, action repeat 2, 125 agent steps truncated, the episode's training camera(s) from ``TrainCameraSampler``);
  observations are the raw stack (or the multi-view / depth / camera dict) plus the shared 4-d proprioception.
  ``n_envs`` copies run in SB3 ``SubprocVecEnv`` workers (spawned, one EGL renderer each, MuJoCo guard as the evaluation
  workers); environment i uses construction / camera seed ``seed + 10000 i`` (environment 0 is DrM's / SAC's).
- frozen encoders (csw4, SinCro, ReViWo, DynaMo, SNeRL, 3D-MVP A/B): ``FeatureVecEnv`` runs the frozen backbone of the
  registered export once per environment step in the main process (``FrozenFeatures``, as SAC and DrM), batched over
  the environments; the policy observes [flattened feature, proprioception] (``MlpPolicy``). Terminal observations used
  for the time-limit bootstrap are encoded the same way.
- end-to-end CNN control: SB3 ``MultiInputPolicy`` (``CombinedExtractor``: SB3's NatureCNN on the 9-channel 128-px stack,
  images / 255, plus the proprioception), trained end to end by the PPO loss.
- ``VecNormalize`` (observations and rewards) for every arm: frozen arms normalise [feature, proprioception], the CNN arm
  its proprioception (images are scaled by the policy). Evaluation uses the snapshot's observation statistics.
"""

from __future__ import annotations

import importlib
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[2]
UPSTREAM = REPO / "third_party/stable-baselines3"
SEED_STRIDE = 10000


def load_sb3():
    if str(UPSTREAM) not in sys.path:
        sys.path.insert(0, str(UPSTREAM))
    sb3 = importlib.import_module("stable_baselines3")
    if not str(Path(sb3.__file__).resolve()).startswith(str(UPSTREAM)) or sb3.__version__ != "2.9.0":
        raise RuntimeError(f"stable_baselines3 must be the pinned v2.9.0 of {UPSTREAM}, got {sb3.__file__} {sb3.__version__}")
    return sb3


def is_cnn(cfg: dict) -> bool:
    return str(cfg["vision"]["encoder_type"]) == "convnet"


def raw_spaces(cfg: dict, env_kwargs: dict, proprio_dim: int):
    """gymnasium observation space of one environment's raw observation (stack or dict) + proprioception."""
    from gymnasium import spaces

    t, size = int(env_kwargs["frame_stack"]), int(env_kwargs["image_size"])
    views = int(env_kwargs.get("views", 1))
    plain = views == 1 and not env_kwargs.get("depth") and not env_kwargs.get("cameras")
    inf = np.float32(np.inf)
    out = {"proprio": spaces.Box(-inf, inf, (proprio_dim,), np.float32)}
    if plain:
        out["pixels"] = spaces.Box(0, 255, (3 * t, size, size), np.uint8)
    else:
        out["rgb"] = spaces.Box(0, 255, (t, views, 3, size, size), np.uint8)
        if env_kwargs.get("depth"):
            out["depth"] = spaces.Box(0.0, inf, (t, views, size, size), np.float32)
        if env_kwargs.get("cameras"):
            out["K"] = spaces.Box(-inf, inf, (t, views, 3, 3), np.float32)
            out["c2w"] = spaces.Box(-inf, inf, (t, views, 4, 4), np.float32)
    return spaces.Dict(out), plain


def _protocol_env_class():
    import gymnasium as gym

    class ProtocolEnv(gym.Env):
        """One training environment of the reference protocol (gymnasium API for SB3)."""

        metadata = {"render_modes": []}

        def __init__(self, task: str, seed: int, env_kwargs: dict, proprio_dim: int, cfg: dict):
            from s4d.rl.env import MetaWorldCameraEnv, TrainCameraSampler

            self.env = MetaWorldCameraEnv(task, seed, **env_kwargs)
            self.sampler = TrainCameraSampler(seed, int(env_kwargs.get("views", 1)))
            self.observation_space, self.plain = raw_spaces(cfg, env_kwargs, proprio_dim)
            self.action_space = gym.spaces.Box(-1.0, 1.0, (self.env.action_dim,), np.float32)

        def _observe(self, obs, proprio) -> dict:
            out = {"proprio": np.asarray(proprio, np.float32)}
            if self.plain:
                out["pixels"] = obs
            else:
                out.update({k: np.asarray(v, self.observation_space[k].dtype) for k, v in obs.items()})
            return out

        def reset(self, *, seed=None, options=None):
            # the protocol's own reset schedule (MetaWorldCameraEnv: construction seed + reset count); SB3's reset
            # seeds are not used, so training episodes match the single-environment protocol of DrM / SAC
            obs, proprio = self.env.reset(self.sampler())
            return self._observe(obs, proprio), {}

        def step(self, action):
            obs, proprio, reward, done, info = self.env.step(np.asarray(action, np.float32))
            # Meta-World only truncates: terminated False, truncated at the protocol's time limit
            return self._observe(obs, proprio), float(reward), False, bool(done), {"success": float(info["success"])}

        def close(self):
            self.env.close()

    return ProtocolEnv


def make_env_fn(task: str, seed: int, env_kwargs: dict, proprio_dim: int, cfg: dict):
    def build():
        from _bootstrap import guard_mujoco  # scripts/ is on the spawned worker's inherited sys.path

        guard_mujoco()
        return _protocol_env_class()(task, seed, env_kwargs, proprio_dim, cfg)

    return build


def _to_device(obs: dict, device) -> dict:
    return {k: torch.as_tensor(np.asarray(v), device=device) for k, v in obs.items() if k != "proprio"}


class FrozenFeatureExtractor:
    """Raw batched observations -> [flattened frozen feature, proprioception] (float32 numpy)."""

    def __init__(self, cfg: dict, device: torch.device):
        from s4d.rl.snerl_sac import FrozenFeatures

        self.features, self.device = FrozenFeatures(cfg, device), device
        self.dim = int(np.prod(self.features.atom_shape)) * (1 if self.features.mode == "stack"
                                                              else int(cfg["env"]["frame_stack"]))

    @torch.no_grad()
    def __call__(self, obs: dict) -> np.ndarray:
        raw = _to_device(obs, self.device)
        inputs = raw["pixels"] if "pixels" in raw else raw
        feature = self.features(inputs).float().flatten(1).cpu().numpy()
        return np.concatenate([feature, np.asarray(obs["proprio"], np.float32)], axis=1).astype(np.float32)


def feature_vec_env(venv, extractor: FrozenFeatureExtractor, proprio_dim: int):
    from gymnasium import spaces
    from stable_baselines3.common.vec_env import VecEnvWrapper

    class FeatureVecEnv(VecEnvWrapper):
        def __init__(self):
            inf = np.float32(np.inf)
            super().__init__(venv, observation_space=spaces.Box(-inf, inf, (extractor.dim + proprio_dim,), np.float32))

        def reset(self):
            return extractor(self.venv.reset())

        def step_wait(self):
            obs, rewards, dones, infos = self.venv.step_wait()
            for info in infos:  # the time-limit bootstrap evaluates V(terminal observation): encode it identically
                if info.get("terminal_observation") is not None:
                    terminal = {k: np.asarray(v)[None] for k, v in info["terminal_observation"].items()}
                    info["terminal_observation"] = extractor(terminal)[0]
            return extractor(obs), rewards, dones, infos

    return FeatureVecEnv()


def policy_kwargs(cfg: dict) -> dict:
    width = list(cfg["agent"]["net_arch"])
    return {"net_arch": {"pi": width, "vf": width}, "activation_fn": torch.nn.Tanh}


def normalizer_state(vecnorm) -> dict:
    """VecNormalize observation statistics needed to act (numpy, weights-only loadable)."""
    rms = vecnorm.obs_rms
    if isinstance(rms, dict):
        stats = {k: {"mean": torch.as_tensor(v.mean), "var": torch.as_tensor(v.var)} for k, v in rms.items()}
    else:
        stats = {"": {"mean": torch.as_tensor(rms.mean), "var": torch.as_tensor(rms.var)}}
    return {"obs_rms": stats, "clip_obs": float(vecnorm.clip_obs), "epsilon": float(vecnorm.epsilon),
            "norm_obs": bool(vecnorm.norm_obs)}


class PPOPolicy:
    """Evaluation policy of a PPO snapshot (``s4d.rl.algorithms``): the SB3 policy network with the snapshot's
    observation statistics, acting with its deterministic mean (SB3 ``predict(deterministic=True)``, clipped)."""

    recurrent = False

    def __init__(self, cfg: dict, action_dim: int, proprio_dim: int, device: torch.device):
        load_sb3()
        from gymnasium import spaces
        from stable_baselines3.common.policies import ActorCriticPolicy, MultiInputActorCriticPolicy

        from s4d.rl.env import env_kwargs

        self.cfg, self.device, self.proprio_dim = cfg, device, int(proprio_dim)
        self.use_pixels = is_cnn(cfg)
        action_space = spaces.Box(-1.0, 1.0, (int(action_dim),), np.float32)
        if self.use_pixels:
            space, _ = raw_spaces(cfg, env_kwargs(cfg), proprio_dim)
            cls, self.extractor = MultiInputActorCriticPolicy, None
        else:
            self.extractor = FrozenFeatureExtractor(cfg, device)
            inf = np.float32(np.inf)
            space = spaces.Box(-inf, inf, (self.extractor.dim + proprio_dim,), np.float32)
            cls = ActorCriticPolicy
        self.policy = cls(space, action_space, lambda _: 0.0, **policy_kwargs(cfg)).to(device)
        self.policy.set_training_mode(False)
        self.encoder = self
        self.norm: dict[str, Any] = {}

    @torch.no_grad()
    def extract_cacheable_feature(self, obs):
        return self.extractor.features(obs)

    def _normalize(self, key: str, x: np.ndarray) -> np.ndarray:
        stats = self.norm["obs_rms"][key]
        mean, var = stats["mean"].cpu().numpy(), stats["var"].cpu().numpy()  # snapshots load on the device
        clip = self.norm["clip_obs"]
        return np.clip((x - mean) / np.sqrt(var + self.norm["epsilon"]), -clip, clip).astype(np.float32)

    def act(self, inputs, proprio, step: int, eval_mode: bool) -> np.ndarray:
        proprio = np.asarray(proprio, np.float32).reshape(len(inputs), -1)
        if self.use_pixels:
            obs = {"pixels": inputs.cpu().numpy(), "proprio": self._normalize("proprio", proprio)}
        else:
            feature = inputs.float().flatten(1).cpu().numpy()
            obs = self._normalize("", np.concatenate([feature, proprio], axis=1))
        actions, _ = self.policy.predict(obs, deterministic=eval_mode)
        return np.asarray(actions, np.float32)

    def train(self, training: bool = True) -> None:
        self.policy.set_training_mode(False)

    def load_policy(self, policy: dict) -> None:
        self.policy.load_state_dict(policy["policy"])
        self.norm = policy["normalizer"]
