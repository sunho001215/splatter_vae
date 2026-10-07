"""DrM (Xu et al., ICLR 2024) for Meta-World, ported from the official ``agents/drm_mw.py`` and ``utils.py``
(github.com/XuGW-Kevin/DrM, commit 989732d68d3eed986ddfc9809909a3c3171ca049).

Kept from the official Meta-World agent: actor, dropout+LayerNorm twin critic and value network; dormant ratio of
the actor's Linear outputs; periodic shrink-and-perturb with a dormant-ratio factor and optimizer-state reset;
"awake" exploration noise; expectile value regression and the blended (exploit/explore) critic target.
Deviations, all documented in ``docs/RL_PROTOCOL.md``: the network trunks also take the reference proprio vector;
frozen pretrained encoders feed cached latents through a trainable projection head, and only the actor, critic,
critic target and value network are perturbed for them (the frozen encoder stays bit-identical).
"""

from __future__ import annotations

import math
import re
from collections import defaultdict
from copy import deepcopy
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


def dormant_ratio(model: nn.Module, *inputs, percentage: float = 0.025) -> float:
    """Official ``utils.cal_dormant_ratio``: share of Linear output units whose batch-mean |activation| is below
    ``percentage`` times the layer's mean over units (all Linear layers, including the output layer)."""
    outputs: list[tuple[nn.Linear, torch.Tensor]] = []
    handles = [
        module.register_forward_hook(lambda m, _i, out: outputs.append((m, out)))
        for module in model.modules()
        if isinstance(module, nn.Linear)
    ]
    try:
        with torch.no_grad():
            model(*inputs)
    finally:
        for handle in handles:
            handle.remove()
    total = dormant = 0
    for module, out in outputs:
        mean_output = out.abs().mean(0)
        dormant += int((mean_output < mean_output.mean() * percentage).sum())
        total += module.weight.shape[0]
    return dormant / total


def perturb(net: nn.Module, optimizer: torch.optim.Optimizer, factor: float) -> None:
    """Official ``utils.perturb``: shrink-and-perturb every parameter whose name contains a Linear module name
    (``factor * old + (1 - factor) * fresh_init``), keep all other parameters, and reset the optimizer state."""
    linear_keys = [name for name, mod in net.named_modules() if isinstance(mod, nn.Linear)]
    fresh = deepcopy(net)
    fresh.apply(weight_init)
    fresh_state = fresh.state_dict()
    for name, param in net.named_parameters():
        if any(key in name for key in linear_keys):
            param.data = param.data * factor + fresh_state[name] * (1 - factor)
    optimizer.state = defaultdict(dict)


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


def trunk(in_dim: int, feature_dim: int) -> nn.Sequential:
    return nn.Sequential(nn.Linear(in_dim, feature_dim), nn.LayerNorm(feature_dim), nn.Tanh())


class Actor(nn.Module):
    def __init__(self, repr_dim, proprio_dim, action_dim, feature_dim, hidden_dim):
        super().__init__()
        self.trunk = trunk(repr_dim + proprio_dim, feature_dim)
        self.policy = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, action_dim),
        )
        self.apply(weight_init)

    def forward(self, obs, proprio, std) -> TruncatedNormal:
        mu = torch.tanh(self.policy(self.trunk(torch.cat([obs, proprio], dim=-1))))
        return TruncatedNormal(mu, torch.ones_like(mu) * std)


class Critic(nn.Module):
    """Official Meta-World critic: Dropout(0.01) and LayerNorm after each hidden Linear."""

    def __init__(self, repr_dim, proprio_dim, action_dim, feature_dim, hidden_dim):
        super().__init__()
        self.trunk = trunk(repr_dim + proprio_dim, feature_dim)

        def q() -> nn.Sequential:
            return nn.Sequential(
                nn.Linear(feature_dim + action_dim, hidden_dim),
                nn.Dropout(0.01),
                nn.LayerNorm(hidden_dim),
                nn.ReLU(inplace=True),
                nn.Linear(hidden_dim, hidden_dim),
                nn.Dropout(0.01),
                nn.LayerNorm(hidden_dim),
                nn.ReLU(inplace=True),
                nn.Linear(hidden_dim, 1),
            )

        self.Q1, self.Q2 = q(), q()
        self.apply(weight_init)

    def forward(self, obs, proprio, action):
        h = torch.cat([self.trunk(torch.cat([obs, proprio], dim=-1)), action], dim=-1)
        return self.Q1(h), self.Q2(h)


class VNetwork(nn.Module):
    def __init__(self, repr_dim, proprio_dim, feature_dim, hidden_dim):
        super().__init__()
        self.trunk = trunk(repr_dim + proprio_dim, feature_dim)
        self.V = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 1),
        )
        self.apply(weight_init)

    def forward(self, obs, proprio):
        return self.V(self.trunk(torch.cat([obs, proprio], dim=-1)))


class DrMAgent:
    def __init__(self, cfg: dict[str, Any], action_dim: int, proprio_dim: int, device: torch.device):
        a = cfg["agent"]
        self.device = device
        self.critic_target_tau = float(a["critic_target_tau"])
        self.dormant_threshold = float(a["dormant_threshold"])
        self.target_dormant_ratio = float(a["target_dormant_ratio"])
        self.dormant_temp = float(a["dormant_temp"])
        self.dormant_perturb_interval = int(a["dormant_perturb_interval"])
        self.min_perturb_factor = float(a["min_perturb_factor"])
        self.max_perturb_factor = float(a["max_perturb_factor"])
        self.perturb_rate = float(a["perturb_rate"])
        self.target_lambda = float(a["target_lambda"])
        self.expectile = float(a["expectile"])
        self.num_expl_steps = int(a["num_expl_steps"])
        self.stddev_schedule = a["stddev_schedule"]
        self.stddev_clip = float(a["stddev_clip"])
        self.dormant_ratio = 1.0
        self.awaken_step: int | None = None
        feature_dim, hidden_dim, lr = int(a["feature_dim"]), int(a["hidden_dim"]), float(a["lr"])

        self.encoder = VisionEncoderAdapter(cfg).to(device)
        self.use_pixels = self.encoder.backbone_trainable
        # Official DrM augments every update; the reference gives frozen encoders no augmentation.
        self.augment_pixels = self.use_pixels
        repr_dim = self.encoder.repr_dim
        self.actor = Actor(repr_dim, proprio_dim, action_dim, feature_dim, hidden_dim).to(device)
        self.value_predictor = VNetwork(repr_dim, proprio_dim, feature_dim, hidden_dim).to(device)
        self.critic = Critic(repr_dim, proprio_dim, action_dim, feature_dim, hidden_dim).to(device)
        self.critic_target = Critic(repr_dim, proprio_dim, action_dim, feature_dim, hidden_dim).to(device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        encoder_params = [p for p in self.encoder.parameters() if p.requires_grad]
        self.encoder_opt = torch.optim.Adam(encoder_params, lr=lr) if encoder_params else None
        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=lr)
        self.critic_opt = torch.optim.Adam(self.critic.parameters(), lr=lr)
        self.predictor_opt = torch.optim.Adam(self.value_predictor.parameters(), lr=lr)
        self.aug = RandomShiftsAug(pad=4)
        self.train(True)
        self.critic_target.train(True)

    # ------------------------------------------------------------------ schedules
    @property
    def dormant_stddev(self) -> float:
        return 1 / (1 + math.exp(-self.dormant_temp * (self.dormant_ratio - self.target_dormant_ratio)))

    def stddev(self, step: int) -> float:
        """Official ``stddev_type: awake``: dormant-ratio noise until the ratio first drops below its target,
        then the larger of that and the linear schedule restarted at the awakening step."""
        if self.awaken_step is None:
            return self.dormant_stddev
        return max(self.dormant_stddev, schedule(self.stddev_schedule, step - self.awaken_step))

    @property
    def perturb_factor(self) -> float:
        return min(max(self.min_perturb_factor, 1 - self.perturb_rate * self.dormant_ratio), self.max_perturb_factor)

    # ------------------------------------------------------------------ acting
    def train(self, mode: bool = True) -> None:
        self.training = mode
        self.actor.train(mode)
        self.critic.train(mode)
        self.value_predictor.train(mode)
        self.encoder.set_update_mode() if mode else self.encoder.set_act_mode()

    @torch.no_grad()
    def act(self, obs, proprio, step: int, eval_mode: bool) -> np.ndarray:
        """Batched: ``obs`` is (N, ...) pixels or cached features, ``proprio`` (N, P). Uniform before
        ``num_expl_steps`` (official), deterministic mean in evaluation."""
        prev = self.training
        self.train(False)
        obs_t = torch.as_tensor(obs, device=self.device)
        proprio_t = torch.as_tensor(proprio, device=self.device, dtype=torch.float32)
        dist = self.actor(self.encoder(obs_t, is_feature=not self.use_pixels), proprio_t, self.stddev(step))
        if eval_mode:
            action = dist.mean
        else:
            action = dist.sample(clip=None)
            if step < self.num_expl_steps:
                action.uniform_(-1.0, 1.0)
        self.train(prev)
        return action.cpu().numpy()

    # ------------------------------------------------------------------ persistence
    NETS = ("encoder", "actor", "critic", "critic_target", "value_predictor")
    OPTS = ("actor_opt", "critic_opt", "predictor_opt")

    def state_dict(self) -> dict[str, Any]:
        payload = {name: getattr(self, name).state_dict() for name in (*self.NETS, *self.OPTS)}
        payload.update(dormant_ratio=self.dormant_ratio, awaken_step=self.awaken_step)
        if self.encoder_opt is not None:
            payload["encoder_opt"] = self.encoder_opt.state_dict()
        return payload

    def load_state_dict(self, payload: dict[str, Any]) -> None:
        for name in (*self.NETS, *self.OPTS):
            getattr(self, name).load_state_dict(payload[name])
        if self.encoder_opt is not None:
            self.encoder_opt.load_state_dict(payload["encoder_opt"])
        self.dormant_ratio, self.awaken_step = float(payload["dormant_ratio"]), payload["awaken_step"]

    # ------------------------------------------------------------------ learning
    def perturb(self) -> float:
        """Official ``DrMAgent.perturb``. The CNN encoder has no Linear layers, so as in the official code only its
        optimizer state is reset. Frozen encoders are not touched; their projection head's optimizer is reset."""
        factor = self.perturb_factor
        perturb(self.actor, self.actor_opt, factor)
        perturb(self.critic, self.critic_opt, factor)
        perturb(self.critic_target, self.critic_opt, factor)
        if self.use_pixels:
            perturb(self.encoder, self.encoder_opt, factor)
        elif self.encoder_opt is not None:
            self.encoder_opt.state = defaultdict(dict)
        perturb(self.value_predictor, self.predictor_opt, factor)
        return factor

    def update_predictor(self, obs, proprio, action) -> dict[str, float]:
        q1, q2 = self.critic(obs, proprio, action)
        vf_err = self.value_predictor(obs, proprio) - torch.min(q1, q2)
        vf_sign = (vf_err > 0).float()
        vf_weight = (1 - vf_sign) * self.expectile + vf_sign * (1 - self.expectile)
        predictor_loss = (vf_weight * vf_err**2).mean()
        self.predictor_opt.zero_grad(set_to_none=True)
        predictor_loss.backward()
        self.predictor_opt.step()
        return {"predictor_loss": float(predictor_loss)}

    def critic_target_value(self, next_obs, next_proprio, reward, discount, step: int) -> torch.Tensor:
        """Official blended target: lambda * V(s') + (1 - lambda) * min(Q'_1, Q'_2)(s', a'), a' ~ pi(s')."""
        with torch.no_grad():
            next_action = self.actor(next_obs, next_proprio, self.stddev(step)).sample(clip=self.stddev_clip)
            explore = torch.min(*self.critic_target(next_obs, next_proprio, next_action))
            exploit = self.value_predictor(next_obs, next_proprio)
            target_v = self.target_lambda * exploit + (1 - self.target_lambda) * explore
            return reward + discount * target_v

    def update_critic(self, obs, proprio, action, reward, discount, next_obs, next_proprio, step) -> dict[str, float]:
        target_q = self.critic_target_value(next_obs, next_proprio, reward, discount, step)
        q1, q2 = self.critic(obs, proprio, action)
        critic_loss = F.mse_loss(q1, target_q) + F.mse_loss(q2, target_q)
        if self.encoder_opt is not None:
            self.encoder_opt.zero_grad(set_to_none=True)
        self.critic_opt.zero_grad(set_to_none=True)
        critic_loss.backward()
        self.critic_opt.step()
        if self.encoder_opt is not None:
            self.encoder_opt.step()
        return {
            "critic_target_q": float(target_q.mean()),
            "critic_q1": float(q1.mean()),
            "critic_q2": float(q2.mean()),
            "critic_loss": float(critic_loss),
        }

    def update_actor(self, obs, proprio, step) -> dict[str, float]:
        dist = self.actor(obs, proprio, self.stddev(step))
        action = dist.sample(clip=self.stddev_clip)
        actor_loss = -torch.min(*self.critic(obs, proprio, action)).mean()
        self.actor_opt.zero_grad(set_to_none=True)
        actor_loss.backward()
        self.actor_opt.step()
        return {
            "actor_loss": float(actor_loss),
            "actor_logprob": float(dist.log_prob(action).sum(-1).mean()),
            "actor_ent": float(dist.entropy().sum(dim=-1).mean()),
        }

    def update(self, replay_iter, step: int) -> dict[str, float]:
        metrics: dict[str, float] = {}
        if step % self.dormant_perturb_interval == 0:
            metrics["perturb_factor"] = self.perturb()
        obs, proprio, action, reward, discount, next_obs, next_proprio = (
            x.to(self.device, non_blocking=True) for x in next(replay_iter)
        )
        proprio, next_proprio = proprio.float().flatten(1), next_proprio.float().flatten(1)
        action, reward, discount = action.float(), reward.float(), discount.float()
        self.encoder.set_update_mode()
        if self.use_pixels:
            if self.augment_pixels:
                obs, next_obs = self.aug(obs.float()), self.aug(next_obs.float())
            obs = self.encoder(obs, is_feature=False)
            with torch.no_grad():
                next_obs = self.encoder(next_obs, is_feature=False)
        else:
            obs = self.encoder(obs.float(), is_feature=True)
            with torch.no_grad():
                next_obs = self.encoder(next_obs.float(), is_feature=True)

        self.dormant_ratio = dormant_ratio(self.actor, obs.detach(), proprio, 0, percentage=self.dormant_threshold)
        if self.awaken_step is None and self.dormant_ratio < self.target_dormant_ratio:
            self.awaken_step = step
        metrics.update(
            batch_reward=float(reward.mean()),
            actor_dormant_ratio=self.dormant_ratio,
            critic_dormant_ratio=dormant_ratio(
                self.critic, obs.detach(), proprio, action, percentage=self.dormant_threshold
            ),
            stddev=self.stddev(step),
            target_lambda=self.target_lambda,
            awakened=float(self.awaken_step is not None),
        )
        metrics.update(self.update_predictor(obs.detach(), proprio, action))
        metrics.update(self.update_critic(obs, proprio, action, reward, discount, next_obs, next_proprio, step))
        metrics.update(self.update_actor(obs.detach(), proprio, step))
        soft_update_params(self.critic, self.critic_target, self.critic_target_tau)
        return metrics
