"""SNeRL-code SAC: the official SNeRL downstream agent on the shared reference protocol (user directive 2026-10-28).

Upstream: github.com/jayLEE0301/snerl_official @ d01533b (submodule ``third_party/snerl``). ``snerl/curl_sac.py``
(``CurlSacAgent``, ``Actor``, ``Critic``, ``CURL``: SAC losses, targets, entropy tuning, optimisers and update schedule),
``snerl/encoder.py`` (``MeanEncoder``, ``PixelEncoder``, ``make_encoder``) and ``snerl/utils.py`` (``soft_update_params``,
``random_crop``) are imported unchanged. ``utils.py`` imports gym, cv2 and skimage at module level for helpers this
port never calls (``FrameStack``, video, buffer saving); missing ones are replaced by inert stub modules during that
import only. Upstream ``train.py`` (MuJoCo-py Meta-World v2 wrapper, tensorboard logger, 2-frame replay) is replaced by
``scripts/train_sac.py`` on our environment, replay and evaluation (see its docstring for the loop correspondence).

Thin encoder adapters, registered for this process before the agent is built (upstream selects encoders by type):
- ``pixel`` (end-to-end CNN control): upstream ``PixelEncoder``; the only change is that the 128-px convolution output
  size is accepted (upstream maps 64/84/100 px and raises for 128 although its README runs ``--image_size 128``; 57 is
  ``OUT_DIM_128`` of the same file). With ``pixel`` the agent trains CURL as upstream (``sample_cpc``; random crop
  ``pre_transform_image_size`` 128 -> ``image_size`` 128, i.e. the IDENTITY at the README size; reported as such).
- ``nerf`` (SNeRL): upstream ``MeanEncoder`` with the weights of our SNeRL export (identical layer names) instead of
  ``./encoder_pretrained/<env>/snerl.tar`` and the episode's real normalised camera poses (as in our SNeRL pretraining)
  instead of the hard-coded poses of upstream's fixed cameras. Convolutions frozen under ``no_grad``; fc/mlp1/mlp2/ln
  initialised from pretraining and trained by the actor and critic losses separately (upstream).
- ``frozen`` (csw4 and the other frozen exports): the frozen backbone runs once per environment step (its feature is
  cached in replay, exactly as upstream would recompute it: frozen, deterministic, no augmentation for non-pixel
  encoders); a trainable head with MeanEncoder's layer sequence (fc -> mlp1 -> mlp2 -> LayerNorm, no nonlinearity in
  between; tanh for the actor, logits for the critic) maps it to the same 63 per frame (per-frame features) or 63 x 3
  (one feature for the 3-frame history), like the 63-per-frame SNeRL and pixel encoders.
All adapters append the shared proprioception after the upstream representation (upstream has none; our protocol gives
every method the same 4-d hand state). CURL (pixel) encodes pixels only, so its bilinear W keeps upstream's size.
"""

from __future__ import annotations

import importlib
import importlib.util
import sys
import types
from pathlib import Path
from typing import Any, NamedTuple

import numpy as np
import torch
import torch.nn as nn

REPO = Path(__file__).resolve().parents[2]
UPSTREAM = REPO / "third_party/snerl/snerl"
UPSTREAM_MODULES = ("encoder", "utils", "curl_sac")


def _stub(name: str) -> types.ModuleType:
    module = types.ModuleType(name)
    module.__s4d_stub__ = True
    if name == "gym":
        module.Wrapper = object  # utils.FrameStack subclasses gym.Wrapper at import time; never instantiated here
    if name == "skimage.util.shape":
        def view_as_windows(*_args, **_kwargs):
            raise RuntimeError("skimage is not installed; upstream view_as_windows is not used by this port")
        module.view_as_windows = view_as_windows
    return module


def load_upstream() -> dict[str, types.ModuleType]:
    """Import upstream ``encoder``, ``utils`` and ``curl_sac`` unchanged, isolated under ``snerl_upstream.*`` names."""
    if "snerl_upstream.curl_sac" in sys.modules:
        return {name: sys.modules[f"snerl_upstream.{name}"] for name in UPSTREAM_MODULES}
    displaced = {name: sys.modules.pop(name) for name in UPSTREAM_MODULES if name in sys.modules}
    missing = {top for top in ("gym", "cv2", "skimage") if top not in sys.modules and importlib.util.find_spec(top) is None}
    stubs = [name for name in ("gym", "cv2", "skimage", "skimage.util", "skimage.util.shape")
             if name.split(".")[0] in missing]
    for name in stubs:
        sys.modules[name] = _stub(name)
    sys.path.insert(0, str(UPSTREAM))
    try:
        modules = {name: importlib.import_module(name) for name in UPSTREAM_MODULES}
    finally:
        sys.path.remove(str(UPSTREAM))
        for name in UPSTREAM_MODULES:
            sys.modules.pop(name, None)
        sys.modules.update(displaced)
        for name in stubs:
            sys.modules.pop(name, None)
    for name, module in modules.items():
        sys.modules[f"snerl_upstream.{name}"] = module
    return modules


class Packed(NamedTuple):
    """Encoder input for the adapters: ``x`` pixels / SNeRL views / cached features, ``proprio`` (B, P), ``poses``
    (SNeRL only: (B, V, 16) for a static episode, or (B, T, V, 16) when the cameras move)."""

    x: torch.Tensor
    proprio: torch.Tensor
    poses: torch.Tensor | None = None

    def __len__(self) -> int:  # upstream encoders take the batch size from len(obs)
        return int(self.x.shape[0])


# Construction context of the adapters: upstream ``make_encoder`` passes only its own positional arguments.
_CONTEXT: dict[str, Any] = {}


def _register(up: dict[str, types.ModuleType]) -> None:
    enc = up["encoder"]
    if getattr(enc, "__s4d_registered__", False):
        return

    class PixelEncoder128(enc.PixelEncoder):
        """Upstream ``PixelEncoder`` accepting 128-px inputs; appends proprioception to the policy representation."""

        def __init__(self, obs_shape, feature_dim, num_layers=2, num_filters=32, output_logits=False,
                     log_encoder=False, multiview=3, frame_stack=3):
            width = obs_shape[-1]
            if width == 128:  # upstream __init__ raises for 128: build at 100 px, then resize fc to the 128-px output
                super().__init__((obs_shape[0], 100, 100), feature_dim, num_layers, num_filters, output_logits,
                                 log_encoder, multiview, frame_stack)
                self.width, self.obs_shape = width, obs_shape
                self.fc = nn.Linear(num_filters * enc.OUT_DIM_128[num_layers] ** 2, int(feature_dim / frame_stack))
            else:
                super().__init__(obs_shape, feature_dim, num_layers, num_filters, output_logits, log_encoder,
                                 multiview, frame_stack)
            self.representation_dim = int(feature_dim)
            self.feature_dim = self.representation_dim + int(_CONTEXT["proprio_dim"])

        def forward(self, obs, detach=False):
            if isinstance(obs, Packed):
                return torch.cat([super().forward(obs.x, detach=detach), obs.proprio], dim=-1)
            return super().forward(obs, detach=detach)  # CURL: pixel representation only

        def log(self, L, step, log_freq):
            pass

    class SNeRLMeanEncoder(enc.MeanEncoder):
        """Upstream ``MeanEncoder`` loading our SNeRL export and taking the episode's camera poses."""

        def __init__(self, obs_shape, feature_dim, multiview, frame_stack, encoder_name, finetune_encoder=False,
                     log_encoder=False, output_logits=False, env_name=None):
            nn.Module.__init__(self)
            assert len(obs_shape) == 3 and feature_dim / frame_stack == 63
            if finetune_encoder or log_encoder:
                raise NotImplementedError  # upstream raises as well
            self.obs_shape, self.num_layers, self.num_filters = obs_shape, 4, 32
            self.output_logits, self.multiview, self.frame_stack = output_logits, multiview, frame_stack
            self.encoder_name, self.finetune_encoder, self.log_encoder = encoder_name, finetune_encoder, log_encoder
            self.width, self.env_name, self.outputs = obs_shape[-1], env_name, None
            out_dim = enc.OUT_DIM_128[self.num_layers] if self.width == 128 else enc.OUT_DIM_64[self.num_layers]
            self.convs = nn.ModuleList([nn.Conv2d(3, self.num_filters, 3, stride=2)])
            for _ in range(self.num_layers - 1):
                self.convs.append(nn.Conv2d(self.num_filters, self.num_filters, 3, stride=1))
            self.fc = nn.Linear(self.num_filters * out_dim * out_dim, int(feature_dim / frame_stack))
            self.ln = nn.LayerNorm(int(feature_dim / frame_stack))
            self.mlp1 = nn.Linear(int(feature_dim / frame_stack) + 16, 63)
            self.mlp2 = nn.Linear(63, 63)
            self.load_state_dict(_CONTEXT["snerl_state_dict"], strict=True)  # upstream: torch.load(snerl.tar)
            for param in self.parameters():
                param.requires_grad = False
            for module in (self.fc, self.mlp1, self.mlp2, self.ln):
                for param in module.parameters():
                    param.requires_grad = True
            self.representation_dim = int(feature_dim)
            self.feature_dim = self.representation_dim + int(_CONTEXT["proprio_dim"])

        def forward(self, obs, obs_pose=None, detach=False):
            b, t, v = len(obs), self.frame_stack, self.multiview
            poses = obs.poses if obs.poses.dim() == 4 else obs.poses[:, None].expand(b, t, v, 16)
            x = obs.x.reshape(b, t * v * 3, self.width, self.width).float()
            out = super().forward(x, obs_pose=poses.reshape(b * t, v, 16).float(), detach=detach)
            return torch.cat([out, obs.proprio], dim=-1)

        def log(self, L, step, log_freq):
            pass

    class FrozenFeatureEncoder(nn.Module):
        """Cached frozen-backbone feature -> trainable head with MeanEncoder's layers (fc, mlp1, mlp2, LayerNorm)."""

        def __init__(self, obs_shape, feature_dim, num_layers, num_filters, output_logits=False, log_encoder=False,
                     multiview=1, frame_stack=3):
            super().__init__()
            mode, dim = _CONTEXT["feature"]
            self.per_frame, self.frame_stack, self.output_logits = mode == "frame", int(frame_stack), output_logits
            width = int(feature_dim / frame_stack) if self.per_frame else int(feature_dim)
            self.fc = nn.Linear(int(dim), width)
            self.mlp1 = nn.Linear(width, width)
            self.mlp2 = nn.Linear(width, width)
            self.ln = nn.LayerNorm(width)
            self.representation_dim = int(feature_dim)
            self.feature_dim = self.representation_dim + int(_CONTEXT["proprio_dim"])

        def forward(self, obs, detach=False):
            x = obs.x.float()  # cached fp16 features; the backbone is frozen, so `detach` has nothing to cut
            if self.per_frame:
                x = x.reshape(len(obs) * self.frame_stack, -1)
            h = self.ln(self.mlp2(self.mlp1(self.fc(x)))).reshape(len(obs), -1)
            return torch.cat([h if self.output_logits else torch.tanh(h), obs.proprio], dim=-1)

        def copy_conv_weights_from(self, source):
            pass  # upstream ties convolutions only; the frozen backbone is outside the agent

        def log(self, L, step, log_freq):
            pass

    enc._AVAILABLE_ENCODERS["pixel"] = PixelEncoder128
    enc._AVAILABLE_ENCODERS["frozen"] = FrozenFeatureEncoder
    enc.MeanEncoder = SNeRLMeanEncoder  # make_encoder builds 'nerf' through this module attribute
    enc.__s4d_registered__ = True


class Logger:
    """Duck-typed upstream logger ``L``: scalars of the latest update are kept for train.jsonl; histograms, parameter
    and image logs (tensorboard only upstream) are dropped."""

    def __init__(self):
        self.scalars: dict[str, float] = {}

    def log(self, key, value, step, n=1):
        self.scalars[key] = float(value.item() if torch.is_tensor(value) else value)

    def log_histogram(self, *args, **kwargs):
        pass

    def log_param(self, *args, **kwargs):
        pass

    def log_image(self, *args, **kwargs):
        pass


def family(cfg: dict) -> str:
    vision = str(cfg["vision"]["encoder_type"])
    return {"convnet": "pixel", "snerl": "nerf"}.get(vision, "frozen")


class FrozenFeatures:
    """The frozen backbone of a non-SNeRL export (``s4d.rl.encoders.VisionEncoderAdapter``: identical loading and
    feature extraction as the DrM runs; its own projection head is not used)."""

    def __init__(self, cfg: dict, device: torch.device):
        from s4d.rl.encoders import VisionEncoderAdapter

        adapter_cfg = {**cfg, "agent": {"feature_dim": 256}}  # only sizes the unused DrM head
        self.adapter = VisionEncoderAdapter(adapter_cfg).to(device)
        self.adapter.eval()
        self.atom_shape, self.atom_dtype = self.adapter.replay_atom_shape, self.adapter.replay_atom_dtype
        self.atom_frame_stack = self.adapter.replay_atom_frame_stack
        self.mode = "stack" if self.adapter.replay_atom_is_stack_feature else "frame"
        self.dim = int(np.prod(self.atom_shape))

    def __call__(self, obs):
        return self.adapter.extract_cacheable_feature(obs)


class SNeRLViews:
    """Policy input of the SNeRL encoder: the raw views (uint8) and their normalised flattened OpenGL poses."""

    def __init__(self, payload: dict):
        self.lookat, self.scale = tuple(payload["pose"]["lookat"]), float(payload["pose"]["scale"])

    def poses(self, c2w: torch.Tensor) -> torch.Tensor:
        from s4d.baselines.snerl.data import normalise_c2w

        return normalise_c2w(c2w.float(), self.lookat, self.scale).flatten(-2).float()

    def __call__(self, obs: dict) -> Packed:
        return Packed(obs["rgb"], torch.empty(0), self.poses(obs["c2w"]))  # (B, T, V, 3, H, W), (B, T, V, 16)


class SNeRLSAC:
    """``CurlSacAgent`` with the family adapter; the interface of ``s4d.rl.agent.DrMAgent`` used by training and
    ``s4d.rl.evaluate`` (``device``, ``use_pixels``, ``encoder.extract_cacheable_feature``, ``act``)."""

    def __init__(self, cfg: dict, action_dim: int, proprio_dim: int, device: torch.device):
        up = load_upstream()
        _register(up)
        self.cfg, self.device, self.proprio_dim = cfg, device, int(proprio_dim)
        self.family = family(cfg)
        a = cfg["agent"]
        frames, views = int(cfg["env"]["frame_stack"]), int(cfg["env"].get("views", 1))
        feature_dim = int(a["per_frame_feature"]) * frames
        _CONTEXT.clear()
        _CONTEXT["proprio_dim"] = self.proprio_dim
        self.use_pixels = self.family == "pixel"
        if self.family == "pixel":
            obs_shape, self.features = (3 * views * frames, int(a["image_size"]), int(a["image_size"])), None
        elif self.family == "nerf":
            payload = torch.load(cfg["vision"]["export_path"], map_location="cpu", weights_only=True)
            if payload.get("format") != "splatter4d-baseline-snerl-v1" or int(payload["num_views"]) != views:
                raise ValueError("the SNeRL family needs a SNeRL export with as many views as the environment")
            _CONTEXT["snerl_state_dict"] = payload["state_dict"]
            size = int(cfg["env"]["image_size"])
            obs_shape, self.features = (3 * views * frames, size, size), SNeRLViews(payload)
        else:
            self.features = FrozenFeatures(cfg, device)
            _CONTEXT["feature"] = (self.features.mode, self.features.dim)  # dim: one frame's (or the stack's) feature
            obs_shape = (self.features.dim,)
        self.encoder = self  # evaluate.policy_inputs calls agent.encoder.extract_cacheable_feature
        self.agent = up["curl_sac"].CurlSacAgent(
            obs_shape=obs_shape, action_shape=(int(action_dim),), device=device, hidden_dim=int(a["hidden_dim"]),
            discount=float(a["discount"]), init_temperature=float(a["init_temperature"]), alpha_lr=float(a["alpha_lr"]),
            alpha_beta=float(a["alpha_beta"]), actor_lr=float(a["actor_lr"]), actor_beta=float(a["actor_beta"]),
            actor_log_std_min=float(a["actor_log_std_min"]), actor_log_std_max=float(a["actor_log_std_max"]),
            actor_update_freq=int(a["actor_update_freq"]), critic_lr=float(a["critic_lr"]),
            critic_beta=float(a["critic_beta"]), critic_tau=float(a["critic_tau"]),
            critic_target_update_freq=int(a["critic_target_update_freq"]), encoder_type=self.family,
            encoder_feature_dim=feature_dim, encoder_lr=float(a["encoder_lr"]), encoder_tau=float(a["encoder_tau"]),
            num_layers=int(a["num_layers"]), num_filters=int(a["num_filters"]),
            cpc_update_freq=int(a["cpc_update_freq"]), log_interval=int(cfg["train"]["log_every_steps"]),
            detach_encoder=bool(a["detach_encoder"]), curl_latent_dim=int(a["curl_latent_dim"]),
            multiview=views, frame_stack=frames, encoder_name="snerl" if self.family == "nerf" else None,
            env_name=cfg["task"],
        )
        self.logger = Logger()
        self.utils = up["utils"]

    # ------------------------------------------------------------------ policy inputs and acting
    @torch.no_grad()
    def extract_cacheable_feature(self, obs):
        """Batched env observation (already on the device) -> the encoder input the replay stores per state."""
        if self.family in ("frozen", "nerf"):
            return self.features(obs)
        raise RuntimeError("the pixel family consumes pixels directly")

    def packed(self, inputs, proprio) -> Packed:
        proprio = torch.as_tensor(np.asarray(proprio), device=self.device, dtype=torch.float32).reshape(len(inputs), -1)
        if self.family == "nerf":
            return Packed(inputs.x.float(), proprio, inputs.poses)
        return Packed(inputs.float(), proprio)

    @torch.no_grad()
    def act(self, inputs, proprio, step: int, eval_mode: bool) -> np.ndarray:
        """Deterministic mean (upstream ``select_action``) for evaluation, a policy sample (``sample_action``) for
        training; batched over environments. Uniform actions before ``init_steps`` are drawn by the training loop."""
        self.agent.train(False)  # upstream acts under utils.eval_mode(agent)
        mu, pi, _, _ = self.agent.actor(self.packed(inputs, proprio), compute_pi=not eval_mode, compute_log_pi=False)
        return (mu if eval_mode else pi).cpu().numpy()

    def train(self, training: bool = True) -> None:
        self.agent.train(training)

    # ------------------------------------------------------------------ updates
    def update(self, batches, step: int) -> dict[str, float]:
        self.agent.train(True)
        self.agent.update(_UpstreamBatches(self, batches), self.logger, step)
        return dict(self.logger.scalars)

    # ------------------------------------------------------------------ state
    def _modules(self) -> dict[str, Any]:
        out = {"actor": self.agent.actor, "critic": self.agent.critic, "critic_target": self.agent.critic_target,
               "actor_optimizer": self.agent.actor_optimizer, "critic_optimizer": self.agent.critic_optimizer,
               "log_alpha_optimizer": self.agent.log_alpha_optimizer}
        if hasattr(self.agent, "CURL"):
            out |= {"CURL": self.agent.CURL, "encoder_optimizer": self.agent.encoder_optimizer,
                    "cpc_optimizer": self.agent.cpc_optimizer}
        return out

    def state_dict(self) -> dict:
        return {**{k: m.state_dict() for k, m in self._modules().items()}, "log_alpha": self.agent.log_alpha.detach().cpu()}

    def load_state_dict(self, state: dict) -> None:
        for k, m in self._modules().items():
            m.load_state_dict(state[k])
        with torch.no_grad():
            self.agent.log_alpha.copy_(state["log_alpha"].to(self.agent.log_alpha.device))

    def policy_state(self) -> dict:
        return {"actor": self.agent.actor.state_dict()}

    def load_policy(self, policy: dict) -> None:
        self.agent.actor.load_state_dict(policy["actor"])


class _UpstreamBatches:
    """Upstream ``ReplayBuffer`` sampling interface over our replay iterator (1-step, uniform over stored transitions
    as ``np.random.randint`` upstream; ``not_done`` = the time-limit continuation, 1.0 as upstream's done_bool 0)."""

    def __init__(self, sac: SNeRLSAC, batches):
        self.sac, self.batches = sac, batches

    def _next(self):
        obs, proprio, action, reward, not_done, next_obs, next_proprio = (
            t.to(self.sac.device, non_blocking=True) for t in next(self.batches))
        return obs, proprio.float(), action.float(), reward.float(), not_done.float(), next_obs, next_proprio.float()

    def _pack(self, x, proprio) -> Packed:
        sac = self.sac
        if sac.family == "nerf":
            p = sac.proprio_dim
            views = int(sac.cfg["env"]["views"])
            b, t = x.shape[0], int(sac.cfg["env"]["frame_stack"])
            size = int(sac.cfg["env"]["image_size"])
            return Packed(x.reshape(b, t, views, 3, size, size).float(), proprio[:, :p],
                          proprio[:, p:].reshape(b, views, 16))
        return Packed(x.float(), proprio)

    def sample_proprio(self):
        obs, proprio, action, reward, not_done, next_obs, next_proprio = self._next()
        return self._pack(obs, proprio), action, reward, self._pack(next_obs, next_proprio), not_done

    def sample_cpc(self):
        obs, proprio, action, reward, not_done, next_obs, next_proprio = self._next()
        size = int(self.sac.cfg["agent"]["image_size"])
        random_crop = self.sac.utils.random_crop
        obs, next_obs = obs.float(), next_obs.float()
        pos = obs.clone()
        obs, next_obs, pos = random_crop(obs, size), random_crop(next_obs, size), random_crop(pos, size)
        cpc_kwargs = dict(obs_anchor=obs, obs_pos=pos, time_anchor=None, time_pos=None)
        return Packed(obs, proprio), action, reward, Packed(next_obs, next_proprio), not_done, cpc_kwargs
