from __future__ import annotations

import re
from typing import Any, Dict, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import distributions as pyd

from agents.common.encoders import build_vision_encoder
from agents.common.head import FrameMLPStackHead, SmallPostEncoderMLPHead


def weight_init(module: nn.Module) -> None:
    """Initialize weights for linear and convolutional layers."""
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
    """Soft update target network parameters."""
    for param, target_param in zip(net.parameters(), target_net.parameters()):
        target_param.data.copy_(tau * param.data + (1.0 - tau) * target_param.data)


def schedule(schedule_str: str | float | int, step: int) -> float:
    """Parse and compute schedule value based on step."""
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


class RandomShiftsAug(nn.Module):
    """Random shifts augmentation for images."""

    def __init__(self, pad: int):
        super().__init__()
        self.pad = int(pad)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        n, c, h, w = x.size()
        if h != w:
            raise ValueError(f"RandomShiftsAug expects square images, got H={h}, W={w}")

        x = x.float()
        x = F.pad(x, (self.pad, self.pad, self.pad, self.pad), "replicate")

        eps = 1.0 / (h + 2 * self.pad)
        arange = torch.linspace(
            -1.0 + eps, 1.0 - eps, h + 2 * self.pad, device=x.device, dtype=x.dtype
        )[:h]
        arange = arange.unsqueeze(0).repeat(h, 1).unsqueeze(2)
        base_grid = torch.cat([arange, arange.transpose(1, 0)], dim=2)
        base_grid = base_grid.unsqueeze(0).repeat(n, 1, 1, 1)

        shift = torch.randint(
            0, 2 * self.pad + 1, size=(n, 1, 1, 2), device=x.device
        ).to(x.dtype)
        shift *= 2.0 / (h + 2 * self.pad)
        grid = base_grid + shift
        return F.grid_sample(x, grid, padding_mode="zeros", align_corners=False)


class TruncatedNormal(pyd.Normal):
    """Truncated normal distribution."""

    def __init__(
        self, loc: torch.Tensor, scale: torch.Tensor, low: float = -1.0, high: float = 1.0, eps: float = 1e-6
    ):
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


class VisionEncoderAdapter(nn.Module):
    """Adapter for vision encoders with projection heads."""

    def __init__(self, full_cfg: Dict[str, Any]):
        super().__init__()
        self.full_cfg = full_cfg
        self.vision_name = str(
            full_cfg["vision"].get("encoder_type", full_cfg["vision"].get("name", "convnet"))
        ).lower()
        self.frame_stack = int(full_cfg["env"].get("frame_stack", 1))

        self.backbone = build_vision_encoder(full_cfg)
        self.backbone_trainable = bool(
            getattr(self.backbone, "is_trainable", self.vision_name == "convnet")
        )
        if not self.backbone_trainable:
            self.backbone.eval()
            for p in self.backbone.parameters():
                p.requires_grad = False

        self.backbone_out_shape, self.single_frame_dim = self._infer_backbone_feature_shape(full_cfg)

        self.proj_head = None
        if not self.backbone_trainable:
            proj_dim = int(full_cfg["agent"].get("feature_dim", 256))
            if self.vision_name == "sincro":
                self.proj_head = SmallPostEncoderMLPHead(
                    in_dim=self.single_frame_dim, hidden_dim=proj_dim, out_dim=proj_dim
                )
            else:
                self.proj_head = FrameMLPStackHead(
                    in_dim=self.single_frame_dim,
                    frame_stack=self.frame_stack,
                    per_frame_hidden_dim=proj_dim,
                    per_frame_out_dim=min(proj_dim, 256),
                    stacked_hidden_dim=proj_dim,
                    out_dim=proj_dim,
                )

        if self.backbone_trainable:
            self.repr_dim = int(self.backbone.repr_dim)
            self.replay_obs_shape = (
                3 * self.frame_stack,
                int(full_cfg["env"]["image_height"]),
                int(full_cfg["env"]["image_width"])
            )
            self.replay_obs_dtype = np.uint8
        else:
            self.repr_dim = int(self.proj_head.out_dim)
            self.replay_obs_shape = tuple(self.backbone_out_shape)
            self.replay_obs_dtype = np.float16

    def _to_unit_float(self, x: torch.Tensor) -> torch.Tensor:
        """Convert tensor to unit float [0, 1]."""
        if x.dtype == torch.uint8:
            return x.float() / 255.0
        x = x.float()
        return x / 255.0 if x.max() > 1.0 else x

    def _infer_backbone_feature_shape(self, full_cfg: Dict[str, Any]) -> tuple[tuple[int, ...], int]:
        """Infer the output shape and dimension of the backbone."""
        h = int(full_cfg["env"]["image_height"])
        w = int(full_cfg["env"]["image_width"])
        try:
            device = next(self.backbone.parameters()).device
        except StopIteration:
            device = torch.device("cpu")

        prev_mode = self.backbone.training
        self.backbone.eval()
        with torch.no_grad():
            if self.vision_name == "convnet":
                dummy = torch.zeros(1, 3 * self.frame_stack, h, w, device=device, dtype=torch.uint8)
                feat = self.backbone(dummy)
                shape = tuple(feat.shape[1:]); dim = int(feat.shape[-1])
            elif self.vision_name == "sincro":
                dummy = torch.zeros(1, self.frame_stack, 3, h, w, device=device)
                feat = self.backbone(dummy)
                shape = tuple(feat.shape[1:]); dim = int(feat.flatten(1).shape[-1])
            else:
                dummy = torch.zeros(1, 3, h, w, device=device)
                feat = self.backbone(dummy)
                dim = int(feat.flatten(1).shape[-1])
                shape = (self.frame_stack, dim)
        self.backbone.train(prev_mode)
        return shape, dim

    def set_update_mode(self) -> None:
        """Set to update mode (train backbone if trainable)."""
        super().train(True)
        if not self.backbone_trainable:
            self.backbone.eval()

    def set_act_mode(self) -> None:
        """Set to act mode (eval backbone)."""
        super().train(False)
        self.backbone.eval()

    @torch.inference_mode()
    def extract_cacheable_feature(self, obs: torch.Tensor) -> torch.Tensor:
        """Extract cacheable features for frozen encoders."""
        if self.backbone_trainable:
            raise RuntimeError("extract_cacheable_feature is only for frozen encoders.")
        if obs.ndim == 3:
            obs = obs.unsqueeze(0)
        obs = self._to_unit_float(obs)

        if self.vision_name == "sincro":
            b, c, h, w = obs.shape
            feat = self.backbone(obs.view(b, self.frame_stack, 3, h, w))
            return feat.to(torch.float16).contiguous()

        b, c, h, w = obs.shape
        frames = obs.view(b, self.frame_stack, 3, h, w)
        feat = self.backbone(frames.reshape(b * self.frame_stack, 3, h, w).contiguous()).flatten(1)
        return feat.view(b, self.frame_stack, -1).to(torch.float16).contiguous()

    def forward_pixels(self, obs: torch.Tensor) -> torch.Tensor:
        """Forward pass for pixel observations."""
        if obs.ndim == 3:
            obs = obs.unsqueeze(0)
        if self.backbone_trainable:
            return self.backbone(obs)
        return self.forward_features(self.extract_cacheable_feature(obs).float())

    def forward_features(self, feat: torch.Tensor) -> torch.Tensor:
        """Forward pass for cached features."""
        if self.backbone_trainable:
            raise RuntimeError("forward_features is only for frozen encoders.")
        feat = feat.float()
        if self.vision_name == "sincro":
            return self.proj_head(feat)
        if feat.ndim != 3:
            raise ValueError(f"Expected non-SinCro cached feature as (B,T,D), got {tuple(feat.shape)}")
        return self.proj_head(feat)

    def forward(self, obs: torch.Tensor, *, is_feature: bool = False) -> torch.Tensor:
        """Unified forward pass."""
        return self.forward_features(obs) if is_feature else self.forward_pixels(obs)


class Actor(nn.Module):
    """Actor network for policy."""

    def __init__(
        self, repr_dim: int, proprio_dim: int, action_shape: Tuple[int, ...], feature_dim: int, hidden_dim: int
    ):
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(repr_dim + proprio_dim, feature_dim), nn.LayerNorm(feature_dim), nn.Tanh()
        )
        self.policy = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim), nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, action_shape[0]),
        )
        self.apply(weight_init)

    def forward(self, obs_repr: torch.Tensor, proprio: torch.Tensor, std: float) -> TruncatedNormal:
        h = self.trunk(torch.cat([obs_repr, proprio], dim=-1))
        mu = torch.tanh(self.policy(h))
        return TruncatedNormal(mu, torch.ones_like(mu) * std)


class Critic(nn.Module):
    """Critic network for Q-value estimation."""

    def __init__(
        self, repr_dim: int, proprio_dim: int, action_shape: Tuple[int, ...], feature_dim: int, hidden_dim: int
    ):
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(repr_dim + proprio_dim, feature_dim), nn.LayerNorm(feature_dim), nn.Tanh()
        )
        self.q1 = nn.Sequential(
            nn.Linear(feature_dim + action_shape[0], hidden_dim), nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(inplace=True), nn.Linear(hidden_dim, 1)
        )
        self.q2 = nn.Sequential(
            nn.Linear(feature_dim + action_shape[0], hidden_dim), nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(inplace=True), nn.Linear(hidden_dim, 1)
        )
        self.apply(weight_init)

    def forward(self, obs_repr: torch.Tensor, proprio: torch.Tensor, action: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.trunk(torch.cat([obs_repr, proprio], dim=-1))
        h = torch.cat([h, action], dim=-1)
        return self.q1(h), self.q2(h)


class DrQv2MetaWorldAgent:
    """DrQ-v2 agent for MetaWorld environments."""

    def __init__(
        self, cfg: Dict[str, Any], obs_shape: Tuple[int, ...], action_shape: Tuple[int, ...],
        proprio_shape: Tuple[int, ...], device: torch.device
    ):
        del obs_shape
        self.cfg = cfg
        self.device = device
        acfg = cfg["agent"]
        self.critic_target_tau = float(acfg.get("critic_target_tau", 0.01))
        self.update_every_steps = int(acfg.get("update_every_steps", 2))
        self.num_expl_steps = int(cfg["train"].get("seed_steps", 4000))
        self.stddev_schedule = acfg.get("stddev_schedule", "linear(1.0,0.1,100000)")
        self.stddev_clip = float(acfg.get("stddev_clip", 0.3))
        self.feature_dim = int(acfg.get("feature_dim", 256))
        self.hidden_dim = int(acfg.get("hidden_dim", 256))
        self.lr = float(acfg.get("lr", 1e-4))
        self.use_pixels = str(cfg["vision"].get("encoder_type", "convnet")).lower() == "convnet"

        self.encoder = VisionEncoderAdapter(cfg).to(device)
        proprio_dim = int(np.prod(proprio_shape))
        repr_dim = self.encoder.repr_dim
        self.actor = Actor(repr_dim, proprio_dim, action_shape, self.feature_dim, self.hidden_dim).to(device)
        self.critic = Critic(repr_dim, proprio_dim, action_shape, self.feature_dim, self.hidden_dim).to(device)
        self.critic_target = Critic(repr_dim, proprio_dim, action_shape, self.feature_dim, self.hidden_dim).to(device)
        self.critic_target.load_state_dict(self.critic.state_dict())

        self.encoder_opt = torch.optim.Adam(
            self.encoder.parameters(), lr=self.lr
        ) if self.encoder.backbone_trainable or self.encoder.proj_head is not None else None
        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=self.lr)
        self.critic_opt = torch.optim.Adam(self.critic.parameters(), lr=self.lr)
        self.aug = RandomShiftsAug(pad=int(acfg.get("random_shift_pad", 4)))

        self.train(True)
        self.critic_target.train(True)

    def stddev(self, step: int) -> float:
        """Get standard deviation for exploration."""
        return float(schedule(self.stddev_schedule, step))

    def train(self, mode: bool = True) -> None:
        """Set training mode."""
        self.training = mode
        self.actor.train(mode)
        self.critic.train(mode)
        self.critic_target.train(mode)
        self.encoder.set_update_mode() if mode else self.encoder.set_act_mode()

    @torch.no_grad()
    def act(
        self, obs: np.ndarray | torch.Tensor, proprio: np.ndarray | torch.Tensor, step: int, eval_mode: bool
    ) -> np.ndarray:
        """Sample action from policy."""
        prev_mode = self.training
        self.train(False)
        obs_t = torch.as_tensor(obs, device=self.device)
        if not self.use_pixels:
            if obs_t.ndim == len(self.encoder.replay_obs_shape):
                obs_t = obs_t.unsqueeze(0)
        proprio_t = torch.as_tensor(proprio, device=self.device, dtype=torch.float32).view(1, -1)
        obs_repr = self.encoder(obs_t, is_feature=not self.use_pixels)
        dist = self.actor(obs_repr, proprio_t, self.stddev(step))
        action = dist.mean if eval_mode else dist.sample(clip=None)
        if (not eval_mode) and step < self.num_expl_steps:
            action.uniform_(-1.0, 1.0)
        self.train(prev_mode)
        return action.cpu().numpy()[0]

    def state_dict(self) -> Dict[str, Any]:
        """Get state dict for saving."""
        payload = {
            "encoder": self.encoder.state_dict(),
            "actor": self.actor.state_dict(),
            "critic": self.critic.state_dict(),
            "critic_target": self.critic_target.state_dict(),
            "actor_opt": self.actor_opt.state_dict(),
            "critic_opt": self.critic_opt.state_dict()
        }
        if self.encoder_opt is not None:
            payload["encoder_opt"] = self.encoder_opt.state_dict()
        return payload

    def load_state_dict(self, payload: Dict[str, Any]) -> None:
        """Load state dict."""
        self.encoder.load_state_dict(payload["encoder"])
        self.actor.load_state_dict(payload["actor"])
        self.critic.load_state_dict(payload["critic"])
        self.critic_target.load_state_dict(payload["critic_target"])
        self.actor_opt.load_state_dict(payload["actor_opt"])
        self.critic_opt.load_state_dict(payload["critic_opt"])
        if self.encoder_opt is not None and "encoder_opt" in payload:
            self.encoder_opt.load_state_dict(payload["encoder_opt"])

    def _encode_batch(self, obs: torch.Tensor, proprio: torch.Tensor, next_obs: torch.Tensor, next_proprio: torch.Tensor):
        """Encode batch of observations."""
        self.encoder.set_update_mode()
        if self.use_pixels:
            obs_repr = self.encoder(self.aug(obs.float()), is_feature=False)
            with torch.no_grad():
                next_repr = self.encoder(self.aug(next_obs.float()), is_feature=False)
        else:
            obs_repr = self.encoder(obs, is_feature=True)
            with torch.no_grad():
                next_repr = self.encoder(next_obs, is_feature=True)
        return obs_repr, proprio, next_repr, next_proprio

    def _update_critic(
        self, obs_repr, proprio, action, reward, discount, next_repr, next_proprio, step: int
    ) -> Dict[str, float]:
        """Update critic network."""
        with torch.no_grad():
            next_action = self.actor(next_repr, next_proprio, self.stddev(step)).sample(clip=self.stddev_clip)
            target_q1, target_q2 = self.critic_target(next_repr, next_proprio, next_action)
            target_q = reward + discount * torch.min(target_q1, target_q2)

        q1, q2 = self.critic(obs_repr, proprio, action)
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
            "critic_q1": float(q1.mean().item()),
            "critic_q2": float(q2.mean().item()),
            "target_q": float(target_q.mean().item())
        }

    def _update_actor(self, obs_repr_detached: torch.Tensor, proprio: torch.Tensor, step: int) -> Dict[str, float]:
        """Update actor network."""
        dist = self.actor(obs_repr_detached, proprio, self.stddev(step))
        action = dist.sample(clip=self.stddev_clip)
        q1, q2 = self.critic(obs_repr_detached, proprio, action)
        actor_loss = -torch.min(q1, q2).mean()

        self.actor_opt.zero_grad(set_to_none=True)
        actor_loss.backward()
        self.actor_opt.step()
        return {
            "actor_loss": float(actor_loss.item()),
            "actor_entropy": float(dist.entropy().sum(dim=-1).mean().item())
        }

    def update(self, replay_iter, step: int) -> Dict[str, float]:
        """Perform update step."""
        metrics: Dict[str, float] = {}
        if step % self.update_every_steps != 0:
            return metrics

        obs, proprio, action, reward, discount, next_obs, next_proprio = next(replay_iter)
        tdtype = torch.float32
        todev = lambda x, dtype=None: (
            (x.to(self.device, non_blocking=True) if torch.is_tensor(x) else torch.as_tensor(x, device=self.device))
            .to(dtype=dtype) if dtype is not None else (
                x.to(self.device, non_blocking=True) if torch.is_tensor(x) else torch.as_tensor(x, device=self.device)
            )
        )
        obs = todev(obs, tdtype if not self.use_pixels else None)
        next_obs = todev(next_obs, tdtype if not self.use_pixels else None)
        proprio = todev(proprio, tdtype).view(obs.shape[0], -1)
        next_proprio = todev(next_proprio, tdtype).view(next_obs.shape[0], -1)
        action = todev(action, tdtype)
        reward = todev(reward, tdtype)
        discount = todev(discount, tdtype)

        obs_repr, proprio, next_repr, next_proprio = self._encode_batch(obs, proprio, next_obs, next_proprio)
        metrics["batch_reward"] = float(reward.mean().item())
        metrics["stddev"] = float(self.stddev(step))
        metrics.update(self._update_critic(obs_repr, proprio, action, reward, discount, next_repr, next_proprio, step))
        metrics.update(self._update_actor(obs_repr.detach(), proprio, step))
        soft_update_params(self.critic, self.critic_target, self.critic_target_tau)
        return metrics