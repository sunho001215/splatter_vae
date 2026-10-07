"""DrQ-v2 agent ported from the reference ``agents/drqv2/drqv2_metaworld.py`` (same updates and defaults)."""

from __future__ import annotations

import re
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import distributions as pyd

from s4d.rl.encoders import VisionEncoderAdapter, weight_init


def soft_update_params(net: nn.Module, target_net: nn.Module, tau: float) -> None:
    for param, target_param in zip(net.parameters(), target_net.parameters()):
        target_param.data.copy_(tau * param.data + (1.0 - tau) * target_param.data)


def schedule(spec: str | float, step: int) -> float:
    try:
        return float(spec)
    except (TypeError, ValueError):
        pass
    match = re.match(r"linear\((.+),(.+),(.+)\)", str(spec))
    if not match:
        raise NotImplementedError(f"unsupported schedule: {spec}")
    start, end, duration = (float(x) for x in match.groups())
    mix = float(np.clip(step / duration, 0.0, 1.0))
    return (1.0 - mix) * start + mix * end


class RandomShiftsAug(nn.Module):
    def __init__(self, pad: int):
        super().__init__()
        self.pad = int(pad)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        n, _, h, w = x.size()
        if h != w:
            raise ValueError(f"RandomShiftsAug expects square images, got H={h}, W={w}")
        x = F.pad(x.float(), (self.pad,) * 4, "replicate")
        eps = 1.0 / (h + 2 * self.pad)
        arange = torch.linspace(-1.0 + eps, 1.0 - eps, h + 2 * self.pad, device=x.device, dtype=x.dtype)[:h]
        arange = arange.unsqueeze(0).repeat(h, 1).unsqueeze(2)
        base_grid = torch.cat([arange, arange.transpose(1, 0)], dim=2).unsqueeze(0).repeat(n, 1, 1, 1)
        shift = torch.randint(0, 2 * self.pad + 1, size=(n, 1, 1, 2), device=x.device).to(x.dtype)
        shift *= 2.0 / (h + 2 * self.pad)
        return F.grid_sample(x, base_grid + shift, padding_mode="zeros", align_corners=False)


class TruncatedNormal(pyd.Normal):
    def __init__(self, loc, scale, low=-1.0, high=1.0, eps=1e-6):
        super().__init__(loc, scale, validate_args=False)
        self.low, self.high, self.eps = low, high, eps

    def _clamp(self, x):
        clipped = torch.clamp(x, self.low + self.eps, self.high - self.eps)
        return x - x.detach() + clipped.detach()

    def sample(self, clip=None, sample_shape=torch.Size()):  # noqa: B008
        eps = torch.randn(self._extended_shape(sample_shape), dtype=self.loc.dtype, device=self.loc.device) * self.scale
        if clip is not None:
            eps = torch.clamp(eps, -clip, clip)
        return self._clamp(self.loc + eps)


class Actor(nn.Module):
    def __init__(self, repr_dim, proprio_dim, action_dim, feature_dim, hidden_dim):
        super().__init__()
        self.trunk = nn.Sequential(nn.Linear(repr_dim + proprio_dim, feature_dim), nn.LayerNorm(feature_dim), nn.Tanh())
        self.policy = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, action_dim),
        )
        self.apply(weight_init)

    def forward(self, obs_repr, proprio, std: float) -> TruncatedNormal:
        mu = torch.tanh(self.policy(self.trunk(torch.cat([obs_repr, proprio], dim=-1))))
        return TruncatedNormal(mu, torch.ones_like(mu) * std)


class Critic(nn.Module):
    def __init__(self, repr_dim, proprio_dim, action_dim, feature_dim, hidden_dim):
        super().__init__()
        self.trunk = nn.Sequential(nn.Linear(repr_dim + proprio_dim, feature_dim), nn.LayerNorm(feature_dim), nn.Tanh())

        def q():
            return nn.Sequential(
                nn.Linear(feature_dim + action_dim, hidden_dim),
                nn.ReLU(inplace=True),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(inplace=True),
                nn.Linear(hidden_dim, 1),
            )

        self.q1, self.q2 = q(), q()
        self.apply(weight_init)

    def forward(self, obs_repr, proprio, action):
        h = torch.cat([self.trunk(torch.cat([obs_repr, proprio], dim=-1)), action], dim=-1)
        return self.q1(h), self.q2(h)


class DrQv2Agent:
    def __init__(self, cfg: dict[str, Any], action_dim: int, proprio_dim: int, device: torch.device):
        acfg = cfg["agent"]
        self.device = device
        self.critic_target_tau = float(acfg["critic_target_tau"])
        self.stddev_schedule = acfg["stddev_schedule"]
        self.stddev_clip = float(acfg["stddev_clip"])
        feature_dim, hidden_dim, lr = int(acfg["feature_dim"]), int(acfg["hidden_dim"]), float(acfg["lr"])
        self.encoder = VisionEncoderAdapter(cfg).to(device)
        self.use_pixels = self.encoder.backbone_trainable
        self.augment_pixels = bool(acfg.get("augment_pixels", self.use_pixels))
        repr_dim = self.encoder.repr_dim
        self.actor = Actor(repr_dim, proprio_dim, action_dim, feature_dim, hidden_dim).to(device)
        self.critic = Critic(repr_dim, proprio_dim, action_dim, feature_dim, hidden_dim).to(device)
        self.critic_target = Critic(repr_dim, proprio_dim, action_dim, feature_dim, hidden_dim).to(device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        encoder_params = [p for p in self.encoder.parameters() if p.requires_grad]
        self.encoder_opt = torch.optim.Adam(encoder_params, lr=lr) if encoder_params else None
        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=lr)
        self.critic_opt = torch.optim.Adam(self.critic.parameters(), lr=lr)
        self.aug = RandomShiftsAug(pad=int(acfg.get("random_shift_pad", 4)))
        self.train(True)

    def stddev(self, step: int) -> float:
        return schedule(self.stddev_schedule, step)

    def train(self, mode: bool = True) -> None:
        self.training = mode
        self.actor.train(mode)
        self.critic.train(mode)
        self.critic_target.train(mode)
        self.encoder.set_update_mode() if mode else self.encoder.set_act_mode()

    @torch.no_grad()
    def act(self, obs, proprio, step: int, eval_mode: bool) -> np.ndarray:
        """Batched acting: ``obs`` is (N, ...) pixels or cached features, ``proprio`` is (N, P)."""
        prev = self.training
        self.train(False)
        obs_t = torch.as_tensor(obs, device=self.device)
        proprio_t = torch.as_tensor(proprio, device=self.device, dtype=torch.float32)
        dist = self.actor(self.encoder(obs_t, is_feature=not self.use_pixels), proprio_t, self.stddev(step))
        action = dist.mean if eval_mode else dist.sample(clip=None)
        self.train(prev)
        return action.cpu().numpy()

    def state_dict(self) -> dict[str, Any]:
        payload = {
            name: getattr(self, name).state_dict()
            for name in ("encoder", "actor", "critic", "critic_target", "actor_opt", "critic_opt")
        }
        if self.encoder_opt is not None:
            payload["encoder_opt"] = self.encoder_opt.state_dict()
        return payload

    def load_state_dict(self, payload: dict[str, Any]) -> None:
        for name in ("encoder", "actor", "critic", "critic_target", "actor_opt", "critic_opt"):
            getattr(self, name).load_state_dict(payload[name])
        if self.encoder_opt is not None:
            self.encoder_opt.load_state_dict(payload["encoder_opt"])

    def _encode_batch(self, obs, next_obs):
        self.encoder.set_update_mode()
        if self.use_pixels:
            obs_repr = self.encoder(self.aug(obs.float()) if self.augment_pixels else obs, is_feature=False)
            with torch.no_grad():
                next_repr = self.encoder(self.aug(next_obs.float()) if self.augment_pixels else next_obs, is_feature=False)
        else:
            obs_repr = self.encoder(obs, is_feature=True)
            with torch.no_grad():
                next_repr = self.encoder(next_obs, is_feature=True)
        return obs_repr, next_repr

    def update(self, replay_iter, step: int) -> dict[str, float]:
        obs, proprio, action, reward, discount, next_obs, next_proprio = (
            x.to(self.device, non_blocking=True) for x in next(replay_iter)
        )
        if not self.use_pixels:
            obs, next_obs = obs.float(), next_obs.float()
        proprio, next_proprio = proprio.float().flatten(1), next_proprio.float().flatten(1)
        action, reward, discount = action.float(), reward.float(), discount.float()
        obs_repr, next_repr = self._encode_batch(obs, next_obs)
        metrics = {"batch_reward": float(reward.mean()), "stddev": self.stddev(step)}

        with torch.no_grad():
            next_action = self.actor(next_repr, next_proprio, self.stddev(step)).sample(clip=self.stddev_clip)
            target_q = reward + discount * torch.min(*self.critic_target(next_repr, next_proprio, next_action))
        q1, q2 = self.critic(obs_repr, proprio, action)
        critic_loss = F.mse_loss(q1, target_q) + F.mse_loss(q2, target_q)
        if self.encoder_opt is not None:
            self.encoder_opt.zero_grad(set_to_none=True)
        self.critic_opt.zero_grad(set_to_none=True)
        critic_loss.backward()
        self.critic_opt.step()
        if self.encoder_opt is not None:
            self.encoder_opt.step()
        metrics.update(critic_loss=float(critic_loss), critic_q1=float(q1.mean()), target_q=float(target_q.mean()))

        obs_repr = obs_repr.detach()
        dist = self.actor(obs_repr, proprio, self.stddev(step))
        actor_loss = -torch.min(*self.critic(obs_repr, proprio, dist.sample(clip=self.stddev_clip))).mean()
        self.actor_opt.zero_grad(set_to_none=True)
        actor_loss.backward()
        self.actor_opt.step()
        metrics.update(actor_loss=float(actor_loss), actor_entropy=float(dist.entropy().sum(dim=-1).mean()))
        soft_update_params(self.critic, self.critic_target, self.critic_target_tau)
        return metrics
