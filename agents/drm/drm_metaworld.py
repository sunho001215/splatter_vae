from __future__ import annotations

"""
Compact DrM agent for pixel-based Meta-World.
"""

import math
import re
from collections import defaultdict
from copy import deepcopy
from typing import Any, Dict, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import distributions as pyd

from agents.common.encoders import build_vision_encoder


# -----------------------------------------------------------------------------
# Small utility helpers 
# -----------------------------------------------------------------------------

def weight_init(module: nn.Module) -> None:
    """Orthogonal init used by DrM / DrQ style implementations."""
    if isinstance(module, nn.Linear):
        nn.init.orthogonal_(module.weight.data)
        if module.bias is not None:
            module.bias.data.zero_()
    elif isinstance(module, (nn.Conv2d, nn.ConvTranspose2d)):
        gain = nn.init.calculate_gain("relu")
        nn.init.orthogonal_(module.weight.data, gain)
        if module.bias is not None:
            module.bias.data.zero_()


def soft_update_params(net: nn.Module, target_net: nn.Module, tau: float) -> None:
    """Polyak averaging for the target critic."""
    for param, target_param in zip(net.parameters(), target_net.parameters()):
        target_param.data.copy_(tau * param.data + (1.0 - tau) * target_param.data)


def schedule(schedule_str: str | float | int, step: int) -> float:
    """Small schedule parser matching the official code style."""
    try:
        return float(schedule_str)
    except (TypeError, ValueError):
        pass

    match = re.match(r"linear\((.+),(.+),(.+)\)", str(schedule_str))
    if match:
        start, end, duration = [float(x) for x in match.groups()]
        mix = np.clip(step / duration, 0.0, 1.0)
        return float((1.0 - mix) * start + mix * end)

    match = re.match(r"step_linear\((.+),(.+),(.+),(.+),(.+)\)", str(schedule_str))
    if match:
        start, mid, duration1, end, duration2 = [float(x) for x in match.groups()]
        if step <= duration1:
            mix = np.clip(step / duration1, 0.0, 1.0)
            return float((1.0 - mix) * start + mix * mid)
        mix = np.clip((step - duration1) / duration2, 0.0, 1.0)
        return float((1.0 - mix) * mid + mix * end)

    raise NotImplementedError(f"Unsupported schedule: {schedule_str}")


class TruncatedNormal(pyd.Normal):
    """Simple clipped Gaussian used by the official DrM actor."""

    def __init__(self, loc: torch.Tensor, scale: torch.Tensor, low: float = -1.0, high: float = 1.0, eps: float = 1e-6):
        super().__init__(loc, scale, validate_args=False)
        self.low = low
        self.high = high
        self.eps = eps

    def _clamp(self, x: torch.Tensor) -> torch.Tensor:
        clipped = torch.clamp(x, self.low + self.eps, self.high - self.eps)
        return x - x.detach() + clipped.detach()

    def sample(self, clip: float | None = None, sample_shape: torch.Size = torch.Size()) -> torch.Tensor:
        shape = self._extended_shape(sample_shape)
        eps = torch.randn(shape, dtype=self.loc.dtype, device=self.loc.device)
        eps = eps * self.scale
        if clip is not None:
            eps = torch.clamp(eps, -clip, clip)
        x = self.loc + eps
        return self._clamp(x)


class _LinearOutputHook:
    """Stores intermediate linear outputs for dormant-ratio computation."""

    def __init__(self) -> None:
        self.outputs: list[torch.Tensor] = []

    def __call__(self, module: nn.Module, module_in: tuple[torch.Tensor, ...], module_out: torch.Tensor) -> None:
        del module, module_in
        self.outputs.append(module_out)


def calc_dormant_ratio(model: nn.Module, *inputs: torch.Tensor, percentage: float = 0.025) -> float:
    """
    Dormant-ratio metric from the DrM paper / official code.

    We measure the fraction of neurons whose mean activation is below
    `percentage * average_neuron_activation` for linear layers.
    """
    hooks: list[_LinearOutputHook] = []
    handles = []
    total_neurons = 0
    dormant_neurons = 0

    linear_modules = [m for m in model.modules() if isinstance(m, nn.Linear)]
    for module in linear_modules:
        hook = _LinearOutputHook()
        hooks.append(hook)
        handles.append(module.register_forward_hook(hook))

    with torch.no_grad():
        model(*inputs)

    for module, hook in zip(linear_modules, hooks):
        for out in hook.outputs:
            mean_out = out.abs().mean(0)
            avg = mean_out.mean()
            dormant = (mean_out < avg * percentage).nonzero(as_tuple=True)[0]
            total_neurons += module.weight.shape[0]
            dormant_neurons += int(len(dormant))

    for handle in handles:
        handle.remove()

    if total_neurons == 0:
        return 0.0
    return float(dormant_neurons / total_neurons)


@torch.no_grad()
def perturb(module: nn.Module, optimizer: torch.optim.Optimizer, factor: float) -> None:
    """
    DrM-style periodic perturbation for linear layers.

    We keep it on the policy/value heads only. This keeps the frozen SplatterVAE
    untouched and avoids destabilizing a large visual backbone.
    """
    linear_keys = [name for name, mod in module.named_modules() if isinstance(mod, nn.Linear)]
    if not linear_keys:
        return

    noise_model = deepcopy(module)
    noise_model.apply(weight_init)

    for name, param in module.named_parameters():
        if any(key in name for key in linear_keys):
            noise = noise_model.state_dict()[name] * (1.0 - factor)
            param.data = param.data * factor + noise

    optimizer.state = defaultdict(dict)


# -----------------------------------------------------------------------------
# Vision adapter
# -----------------------------------------------------------------------------

class VisionEncoderAdapter(nn.Module):
    """
    Wraps the configurable vision encoder from `policies.models.vision`.

    Observations are frame-stacked RGB images with shape (B, 3 * frame_stack, H, W).
    We run the selected encoder on each RGB frame independently and concatenate all
    token outputs into one flat state vector.

    This directly satisfies the requested SplatterVAE behavior:
    use the *concatenated VQ output* of the view-invariant encoder as the state.
    """

    def __init__(self, full_cfg: Dict[str, Any]):
        super().__init__()
        self.full_cfg = full_cfg
        self.vision_name = str(full_cfg["vision"].get("encoder_type", full_cfg["vision"].get("name", "resnet50")))
        self.frame_stack = int(full_cfg["env"].get("frame_stack", 1))
        self.backbone = build_vision_encoder(full_cfg)

        # ResNet50 baseline is trainable end-to-end. SplatterVAE stays frozen.
        self.is_trainable = self.vision_name == "resnet50"
        if not self.is_trainable:
            self.backbone.eval()
            for p in self.backbone.parameters():
                p.requires_grad = False

        height = int(full_cfg["env"]["image_height"])
        width = int(full_cfg["env"]["image_width"])
        dummy = torch.zeros(1, 3, height, width)
        prev_mode = self.backbone.training
        self.backbone.eval()
        with torch.no_grad():
            flat_dim = int(self.backbone(dummy).flatten(1).shape[-1])
        self.backbone.train(prev_mode)
        self.single_frame_dim = flat_dim
        self.repr_dim = self.single_frame_dim * self.frame_stack

    def set_update_mode(self) -> None:
        """Enable augmentation only where requested."""
        if self.is_trainable:
            # In `ResNet50Tokens`, train() triggers random crop.
            self.backbone.train(True)
        else:
            # Frozen SplatterVAE must stay deterministic.
            self.backbone.eval()

    def set_act_mode(self) -> None:
        """Action selection should be deterministic (no crop noise)."""
        self.backbone.eval()

    def _to_unit_float(self, x: torch.Tensor) -> torch.Tensor:
        x = x.float()
        if x.max() > 1.0:
            x = x / 255.0
        return x

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        if obs.ndim == 3:
            obs = obs.unsqueeze(0)

        obs = self._to_unit_float(obs)
        batch_size, channels, height, width = obs.shape
        expected = 3 * self.frame_stack
        if channels != expected:
            raise ValueError(f"Expected {expected} channels (= 3 * frame_stack), got {channels}.")

        frames = obs.view(batch_size, self.frame_stack, 3, height, width)
        pieces = []
        for frame_idx in range(self.frame_stack):
            tokens = self.backbone(frames[:, frame_idx])
            pieces.append(tokens.flatten(1))
        return torch.cat(pieces, dim=-1)


# -----------------------------------------------------------------------------
# DrM heads 
# -----------------------------------------------------------------------------

class Actor(nn.Module):
    def __init__(self, repr_dim: int, action_shape: Tuple[int, ...], feature_dim: int, hidden_dim: int):
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(repr_dim, feature_dim),
            nn.LayerNorm(feature_dim),
            nn.Tanh(),
        )
        self.policy = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, action_shape[0]),
        )
        self.apply(weight_init)

    def forward(self, obs_repr: torch.Tensor, std: float) -> TruncatedNormal:
        h = self.trunk(obs_repr)
        mu = torch.tanh(self.policy(h))
        std_t = torch.ones_like(mu) * std
        return TruncatedNormal(mu, std_t)


class Critic(nn.Module):
    def __init__(self, repr_dim: int, action_shape: Tuple[int, ...], feature_dim: int, hidden_dim: int):
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(repr_dim, feature_dim),
            nn.LayerNorm(feature_dim),
            nn.Tanh(),
        )
        self.q1 = nn.Sequential(
            nn.Linear(feature_dim + action_shape[0], hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 1),
        )
        self.q2 = nn.Sequential(
            nn.Linear(feature_dim + action_shape[0], hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 1),
        )
        self.apply(weight_init)

    def forward(self, obs_repr: torch.Tensor, action: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.trunk(obs_repr)
        h = torch.cat([h, action], dim=-1)
        return self.q1(h), self.q2(h)


class ValueNet(nn.Module):
    def __init__(self, repr_dim: int, feature_dim: int, hidden_dim: int):
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(repr_dim, feature_dim),
            nn.LayerNorm(feature_dim),
            nn.Tanh(),
        )
        self.v = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 1),
        )
        self.apply(weight_init)

    def forward(self, obs_repr: torch.Tensor) -> torch.Tensor:
        return self.v(self.trunk(obs_repr))


# -----------------------------------------------------------------------------
# Main agent
# -----------------------------------------------------------------------------

class DrMMetaWorldAgent:
    """DrM agent with a pluggable visual backbone for Meta-World."""

    def __init__(self, cfg: Dict[str, Any], obs_shape: Tuple[int, ...], action_shape: Tuple[int, ...], device: torch.device):
        del obs_shape  # repr dim is inferred directly from the selected visual encoder.
        self.cfg = cfg
        self.device = device
        acfg = cfg["agent"]

        # Hyperparameters
        self.critic_target_tau = float(acfg.get("critic_target_tau", 0.01))
        self.dormant_threshold = float(acfg.get("dormant_threshold", 0.025))
        self.target_dormant_ratio = float(acfg.get("target_dormant_ratio", 0.10))
        self.dormant_temp = float(acfg.get("dormant_temp", 10.0))
        self.target_lambda = float(acfg.get("target_lambda", 0.7))
        self.lambda_temp = float(acfg.get("lambda_temp", 10.0))
        self.dormant_perturb_interval = int(acfg.get("dormant_perturb_interval", 2000))
        self.min_perturb_factor = float(acfg.get("min_perturb_factor", 0.2))
        self.max_perturb_factor = float(acfg.get("max_perturb_factor", 0.95))
        self.perturb_rate = float(acfg.get("perturb_rate", 0.8))
        self.num_expl_steps = int(cfg["train"].get("seed_steps", 4000))
        self.stddev_type = str(acfg.get("stddev_type", "awake"))
        self.stddev_schedule = acfg.get("stddev_schedule", "linear(1.0,0.1,100000)")
        self.stddev_clip = float(acfg.get("stddev_clip", 0.3))
        self.expectile = float(acfg.get("expectile", 0.7))
        self.feature_dim = int(acfg.get("feature_dim", 256))
        self.hidden_dim = int(acfg.get("hidden_dim", 256))
        self.lr = float(acfg.get("lr", 1e-4))

        self.dormant_ratio = 1.0
        self.awaken_step: int | None = None

        # Visual encoder
        self.encoder = VisionEncoderAdapter(cfg).to(device)

        # DrM heads
        repr_dim = self.encoder.repr_dim
        self.actor = Actor(repr_dim, action_shape, self.feature_dim, self.hidden_dim).to(device)
        self.critic = Critic(repr_dim, action_shape, self.feature_dim, self.hidden_dim).to(device)
        self.critic_target = Critic(repr_dim, action_shape, self.feature_dim, self.hidden_dim).to(device)
        self.value_predictor = ValueNet(repr_dim, self.feature_dim, self.hidden_dim).to(device)
        self.critic_target.load_state_dict(self.critic.state_dict())

        # Optimizers
        self.encoder_opt = torch.optim.Adam(self.encoder.parameters(), lr=self.lr) if self.encoder.is_trainable else None
        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=self.lr)
        self.critic_opt = torch.optim.Adam(self.critic.parameters(), lr=self.lr)
        self.value_opt = torch.optim.Adam(self.value_predictor.parameters(), lr=self.lr)

        self.train(True)
        self.critic_target.train(True)

    # ------------------------------------------------------------------
    # DrM schedules
    # ------------------------------------------------------------------
    @property
    def dormant_stddev(self) -> float:
        return 1.0 / (1.0 + math.exp(-self.dormant_temp * (self.dormant_ratio - self.target_dormant_ratio)))

    def stddev(self, step: int) -> float:
        if self.stddev_type == "max":
            return max(schedule(self.stddev_schedule, step), self.dormant_stddev)
        if self.stddev_type == "dormant":
            return self.dormant_stddev
        if self.stddev_type == "awake":
            if self.awaken_step is None:
                return self.dormant_stddev
            return max(self.dormant_stddev, schedule(self.stddev_schedule, step - self.awaken_step))
        # Fallback: allow plain DrQ-style schedules as well.
        return schedule(self.stddev_schedule, step)

    def perturb_factor(self) -> float:
        val = 1.0 - self.perturb_rate * self.dormant_ratio
        return float(min(max(self.min_perturb_factor, val), self.max_perturb_factor))

    @property
    def lambda_(self) -> float:
        denom = 1.0 + math.exp(self.lambda_temp * (self.dormant_ratio - self.target_dormant_ratio))
        return float(self.target_lambda / denom)

    # ------------------------------------------------------------------
    # API helpers
    # ------------------------------------------------------------------
    def train(self, mode: bool = True) -> None:
        self.training = mode
        self.actor.train(mode)
        self.critic.train(mode)
        self.value_predictor.train(mode)
        if mode:
            self.encoder.set_update_mode()
        else:
            self.encoder.set_act_mode()

    @torch.no_grad()
    def act(self, obs: np.ndarray | torch.Tensor, step: int, eval_mode: bool) -> np.ndarray:
        # Action selection is deterministic with respect to preprocessing.
        prev_mode = self.training
        self.train(False)

        obs_t = torch.as_tensor(obs, device=self.device)
        obs_repr = self.encoder(obs_t)
        dist = self.actor(obs_repr, self.stddev(step))
        if eval_mode:
            action = dist.mean
        else:
            action = dist.sample(clip=None)
            if step < self.num_expl_steps:
                action.uniform_(-1.0, 1.0)

        self.train(prev_mode)
        return action.cpu().numpy()[0]

    def state_dict(self) -> Dict[str, Any]:
        payload = {
            "encoder": self.encoder.state_dict(),
            "actor": self.actor.state_dict(),
            "critic": self.critic.state_dict(),
            "critic_target": self.critic_target.state_dict(),
            "value_predictor": self.value_predictor.state_dict(),
            "actor_opt": self.actor_opt.state_dict(),
            "critic_opt": self.critic_opt.state_dict(),
            "value_opt": self.value_opt.state_dict(),
            "dormant_ratio": self.dormant_ratio,
            "awaken_step": self.awaken_step,
        }
        if self.encoder_opt is not None:
            payload["encoder_opt"] = self.encoder_opt.state_dict()
        return payload

    def load_state_dict(self, payload: Dict[str, Any]) -> None:
        self.encoder.load_state_dict(payload["encoder"])
        self.actor.load_state_dict(payload["actor"])
        self.critic.load_state_dict(payload["critic"])
        self.critic_target.load_state_dict(payload["critic_target"])
        self.value_predictor.load_state_dict(payload["value_predictor"])
        self.actor_opt.load_state_dict(payload["actor_opt"])
        self.critic_opt.load_state_dict(payload["critic_opt"])
        self.value_opt.load_state_dict(payload["value_opt"])
        if self.encoder_opt is not None and "encoder_opt" in payload:
            self.encoder_opt.load_state_dict(payload["encoder_opt"])
        self.dormant_ratio = float(payload.get("dormant_ratio", 1.0))
        self.awaken_step = payload.get("awaken_step", None)

    # ------------------------------------------------------------------
    # Internal update pieces
    # ------------------------------------------------------------------
    def _encode_batch(self, obs: torch.Tensor, next_obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        self.encoder.set_update_mode()
        obs_repr = self.encoder(obs)
        with torch.no_grad():
            next_repr = self.encoder(next_obs)
        return obs_repr, next_repr

    def _update_value(self, obs_repr_detached: torch.Tensor, action: torch.Tensor) -> Dict[str, float]:
        q1, q2 = self.critic(obs_repr_detached, action)
        q = torch.min(q1, q2)
        v = self.value_predictor(obs_repr_detached)

        err = v - q
        sign = (err > 0).float()
        weight = (1.0 - sign) * self.expectile + sign * (1.0 - self.expectile)
        value_loss = (weight * (err ** 2)).mean()

        self.value_opt.zero_grad(set_to_none=True)
        value_loss.backward()
        self.value_opt.step()

        return {"value_loss": float(value_loss.item())}

    def _update_critic(
        self,
        obs_repr: torch.Tensor,
        action: torch.Tensor,
        reward: torch.Tensor,
        discount: torch.Tensor,
        next_repr: torch.Tensor,
        step: int,
    ) -> Dict[str, float]:
        with torch.no_grad():
            next_dist = self.actor(next_repr, self.stddev(step))
            next_action = next_dist.sample(clip=self.stddev_clip)
            target_q1, target_q2 = self.critic_target(next_repr, next_action)
            target_v_explore = torch.min(target_q1, target_q2)
            target_v_exploit = self.value_predictor(next_repr)
            target_v = self.lambda_ * target_v_exploit + (1.0 - self.lambda_) * target_v_explore
            target_q = reward + discount * target_v

        q1, q2 = self.critic(obs_repr, action)
        critic_loss = F.mse_loss(q1, target_q) + F.mse_loss(q2, target_q)

        if self.encoder_opt is not None:
            self.encoder_opt.zero_grad(set_to_none=True)
        self.critic_opt.zero_grad(set_to_none=True)
        critic_loss.backward()
        self.critic_opt.step()
        if self.encoder_opt is not None:
            self.encoder_opt.step()

        return {
            "critic_loss": float(critic_loss.item()),
            "critic_q": float(torch.min(q1, q2).mean().item()),
            "target_q": float(target_q.mean().item()),
        }

    def _update_actor(self, obs_repr_detached: torch.Tensor, step: int) -> Dict[str, float]:
        dist = self.actor(obs_repr_detached, self.stddev(step))
        action = dist.sample(clip=self.stddev_clip)
        q1, q2 = self.critic(obs_repr_detached, action)
        q = torch.min(q1, q2)
        actor_loss = -q.mean()

        self.actor_opt.zero_grad(set_to_none=True)
        actor_loss.backward()
        self.actor_opt.step()

        return {
            "actor_loss": float(actor_loss.item()),
            "actor_entropy": float(dist.entropy().sum(dim=-1).mean().item()),
        }

    def _maybe_perturb(self) -> None:
        factor = self.perturb_factor()
        perturb(self.actor, self.actor_opt, factor)
        perturb(self.critic, self.critic_opt, factor)
        perturb(self.critic_target, self.critic_opt, factor)
        perturb(self.value_predictor, self.value_opt, factor)

    # ------------------------------------------------------------------
    # Public update
    # ------------------------------------------------------------------
    def update(self, batch: tuple[torch.Tensor, ...], step: int) -> Dict[str, float]:
        if step > 0 and step % self.dormant_perturb_interval == 0:
            self._maybe_perturb()

        obs, action, reward, discount, next_obs = batch
        obs_repr, next_repr = self._encode_batch(obs, next_obs)

        # Dormant ratio is measured on the actor MLP, as in the official code.
        self.dormant_ratio = calc_dormant_ratio(
            self.actor,
            obs_repr.detach(),
            self.dormant_stddev,
            percentage=self.dormant_threshold,
        )

        if self.awaken_step is None and step > self.num_expl_steps and self.dormant_ratio < self.target_dormant_ratio:
            self.awaken_step = step

        metrics: Dict[str, float] = {
            "dormant_ratio": float(self.dormant_ratio),
            "batch_reward": float(reward.mean().item()),
            "stddev": float(self.stddev(step)),
            "lambda": float(self.lambda_),
        }
        metrics.update(self._update_value(obs_repr.detach(), action))
        metrics.update(self._update_critic(obs_repr, action, reward, discount, next_repr, step))
        metrics.update(self._update_actor(obs_repr.detach(), step))
        soft_update_params(self.critic, self.critic_target, self.critic_target_tau)
        return metrics
