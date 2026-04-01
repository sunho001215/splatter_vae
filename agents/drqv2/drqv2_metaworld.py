from __future__ import annotations

"""
Compact DrQ-v2 agent for pixel-based Meta-World.
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
    """Orthogonal init used by DrQ style implementations."""
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
    """Simple clipped Gaussian used by the official DrQ actor."""

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

# -----------------------------------------------------------------------------
# Vision adapter
# -----------------------------------------------------------------------------

class VisionEncoderAdapter(nn.Module):
    """
    Frame-stack -> representation adapter for DrQ.

    Rules:
      - backbone is built from build_vision_encoder(...)
      - backbone is never perturbed here
      - if backbone is frozen:
          * non-SinCro -> add small fusion/projection head
          * SinCro     -> add small post-backbone MLP head
      - only the small heads are considered perturbable
    """

    def __init__(self, full_cfg: Dict[str, Any]):
        super().__init__()
        self.full_cfg = full_cfg
        self.vision_name = str(
            full_cfg["vision"].get("encoder_type", full_cfg["vision"].get("name", "convnet"))
        ).lower()
        self.frame_stack = int(full_cfg["env"].get("frame_stack", 1))

        # ------------------------------------------------------------------
        # 1) Build backbone and freeze if pretrained
        # ------------------------------------------------------------------
        self.backbone = build_vision_encoder(full_cfg)
        self.backbone_trainable = bool(
            getattr(self.backbone, "is_trainable", self.vision_name == "convnet")
        )

        if not self.backbone_trainable:
            self.backbone.eval()
            for p in self.backbone.parameters():
                p.requires_grad = False

        # ------------------------------------------------------------------
        # 2) Infer backbone output dim once
        # ------------------------------------------------------------------
        self.single_frame_dim = self._infer_backbone_dim(full_cfg)

        # ------------------------------------------------------------------
        # 3) Adapter heads
        # ------------------------------------------------------------------
        self.actor_adapter = None
        self.critic_adapter = None

        if (not self.backbone_trainable) and (self.vision_name != "sincro"):
            from agents.common.head import FrameMLPStackHead

            self.actor_adapter = FrameMLPStackHead(
                in_dim=self.single_frame_dim,
                frame_stack=self.frame_stack,
            )
            self.critic_adapter = FrameMLPStackHead(
                in_dim=self.single_frame_dim,
                frame_stack=self.frame_stack,
            )
            self.repr_dim = self.actor_adapter.out_dim

        elif (not self.backbone_trainable) and (self.vision_name == "sincro"):
            from agents.common.head import SmallPostEncoderMLPHead

            self.actor_adapter = SmallPostEncoderMLPHead(
                in_dim=self.single_frame_dim,
            )
            self.critic_adapter = SmallPostEncoderMLPHead(
                in_dim=self.single_frame_dim,
            )
            self.repr_dim = self.actor_adapter.out_dim

        else:
            # No extra adapter:
            # - ConvNet still behaves like the original shared encoder path
            self.repr_dim = (
                self.single_frame_dim
                if self.vision_name == "sincro"
                else self.single_frame_dim * self.frame_stack
            )

        # ------------------------------------------------------------------
        # 4) DrQ-facing flags
        # ------------------------------------------------------------------
        self.has_separate_adapters = (
            self.actor_adapter is not None and self.critic_adapter is not None
        )

        # Keep this flag name for compatibility with the rest of the code.
        self.has_proj_head = self.has_separate_adapters

        # trainable:
        #   - full ConvNet backbone
        #   - or frozen backbone + separate actor/critic adapters
        self.is_trainable = self.backbone_trainable or self.has_separate_adapters
        self.can_cache_backbone = not self.backbone_trainable

    def actor_adapter_parameters(self):
        return [] if self.actor_adapter is None else self.actor_adapter.parameters()

    def critic_adapter_parameters(self):
        return [] if self.critic_adapter is None else self.critic_adapter.parameters()

    def _select_adapter(self, branch):
        if not self.has_separate_adapters:
            return None
        if branch == "actor":
            return self.actor_adapter
        if branch == "critic":
            return self.critic_adapter
        raise ValueError(...) 

    def _infer_backbone_dim(self, full_cfg: Dict[str, Any]) -> int:
        """Run one dummy forward pass to get backbone output dim."""
        h = int(full_cfg["env"]["image_height"])
        w = int(full_cfg["env"]["image_width"])

        try:
            device = next(self.backbone.parameters()).device
        except StopIteration:
            device = torch.device("cpu")

        prev_mode = self.backbone.training
        self.backbone.eval()

        with torch.no_grad():
            if self.vision_name == "sincro":
                dummy = torch.zeros(1, self.frame_stack, 3, h, w, device=device)
                dim = int(self.backbone(dummy).flatten(1).shape[-1])
            else:
                dummy = torch.zeros(1, 3, h, w, device=device)
                dim = int(self.backbone(dummy).flatten(1).shape[-1])

        self.backbone.train(prev_mode)
        return dim

    def set_update_mode(self) -> None:
        """Training mode: heads train, frozen backbone stays eval."""
        super().train(True)
        if not self.backbone_trainable:
            self.backbone.eval()

    def set_act_mode(self) -> None:
        """Act/eval mode: everything eval."""
        super().train(False)
        self.backbone.eval()

    def _to_unit_float(self, x: torch.Tensor) -> torch.Tensor:
        """Convert uint8 / [0,255] input to float [0,1]."""
        if x.dtype == torch.uint8:
            return x.float() / 255.0
        x = x.float()
        return x / 255.0 if x.max() > 1.0 else x

    def _encode_single(self, x: torch.Tensor) -> torch.Tensor:
        """Encode one RGB frame: (B,3,H,W) -> (B,D)."""
        if self.backbone_trainable:
            feat = self.backbone(x)
        else:
            with torch.no_grad():
                feat = self.backbone(x)
        return feat.flatten(1).contiguous()

    def _encode_sincro(self, x_seq: torch.Tensor) -> torch.Tensor:
        """Encode SinCro temporal input: (B,T,3,H,W) -> (B,D)."""
        if self.backbone_trainable:
            feat = self.backbone(x_seq)
        else:
            with torch.no_grad():
                feat = self.backbone(x_seq)
        return feat.flatten(1).contiguous()
    
    @torch.inference_mode()
    def extract_cacheable_backbone_feature(self, obs: torch.Tensor) -> torch.Tensor:
        if self.backbone_trainable:
            raise RuntimeError("Cache only frozen backbones.")

        if obs.ndim == 3:
            obs = obs.unsqueeze(0)

        obs = self._to_unit_float(obs)
        B, C, H, W = obs.shape

        if self.vision_name == "sincro":
            seq = obs.view(B, self.frame_stack, 3, H, W)
            feat = self._encode_sincro(seq)              # (B, D_backbone)
            return feat.to(torch.float16).contiguous()

        frames = obs.view(B, self.frame_stack, 3, H, W)
        frame_feats = self._encode_frames_batched(frames)  # (B, T, D_backbone)
        return frame_feats.to(torch.float16).contiguous()

    def forward_from_cached_backbone(
        self,
        cached_feat: torch.Tensor,
        branch: str | None = None,
    ) -> torch.Tensor:
        """
        Convert cached frozen-backbone features into the final RL representation.

        For frozen pretrained encoders:
          - actor branch uses actor_adapter
          - critic branch uses critic_adapter
        """
        cached_feat = cached_feat.float()
        adapter = self._select_adapter(branch)

        if self.vision_name == "sincro":
            return adapter(cached_feat) if adapter is not None else cached_feat

        if cached_feat.ndim != 3:
            raise ValueError(
                f"Expected cached non-SinCro feature as (B,T,D), got {tuple(cached_feat.shape)}"
            )

        return adapter(cached_feat) if adapter is not None else cached_feat.flatten(1).contiguous()

    def _encode_frames_batched(self, frames: torch.Tensor) -> torch.Tensor:
        """
        Vectorized per-frame encoding for all non-SinCro encoders.

        Args:
            frames: (B, T, 3, H, W)

        Returns:
            frame_feats: (B, T, D)
        """
        B, T, C, H, W = frames.shape

        # Special handling for ConvNet so we preserve "same crop across time"
        # while still doing one vectorized conv forward over (B*T) frames.
        if self.vision_name == "convnet":
            x = self.backbone.preprocess_batched_frames(frames)   # (B*T,3,h,w)
            feat = self.backbone.encode_preprocessed(x)           # (B*T,D)
            return feat.reshape(B, T, -1).contiguous()

        # Generic non-SinCro case:
        # flatten time into batch, run backbone once, then reshape back.
        x = frames.reshape(B * T, C, H, W).contiguous()

        if self.backbone_trainable:
            feat = self.backbone(x)
        else:
            with torch.no_grad():
                feat = self.backbone(x)

        feat = feat.flatten(1)  # (B*T, D)
        return feat.reshape(B, T, -1).contiguous()

    def forward(self, obs: torch.Tensor, branch: str | None = None) -> torch.Tensor:
        """
        Args:
            obs: (B, 3*T, H, W) or (3*T, H, W)
            branch:
              - "actor" / "critic" for frozen pretrained encoders
              - None for shared ConvNet path

        Returns:
            repr: (B, repr_dim)
        """
        if obs.ndim == 3:
            obs = obs.unsqueeze(0)

        obs = self._to_unit_float(obs)
        B, C, H, W = obs.shape
        exp_c = 3 * self.frame_stack
        if C != exp_c:
            raise ValueError(f"Expected {exp_c} channels (= 3 * frame_stack), got {C}.")

        adapter = self._select_adapter(branch)

        # --------------------------------------------------------------
        # SinCro: full temporal stack goes into backbone once
        # --------------------------------------------------------------
        if self.vision_name == "sincro":
            seq = obs.view(B, self.frame_stack, 3, H, W)
            feat = self._encode_sincro(seq)
            return adapter(feat) if adapter is not None else feat

        # --------------------------------------------------------------
        # Others: encode all frames in one batched backbone forward
        # --------------------------------------------------------------
        frames = obs.view(B, self.frame_stack, 3, H, W)
        frame_feats = self._encode_frames_batched(frames)  # (B, T, D)

        if adapter is not None:
            return adapter(frame_feats)

        # Shared path (e.g. ConvNet)
        return frame_feats.flatten(1).contiguous()

# -----------------------------------------------------------------------------
# DrQ heads 
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

# -----------------------------------------------------------------------------
# Main agent
# -----------------------------------------------------------------------------

class DrQv2MetaWorldAgent:
    """DrQ-v2 agent with a pluggable visual backbone for Meta-World."""

    def __init__(self, cfg: Dict[str, Any], obs_shape: Tuple[int, ...], action_shape: Tuple[int, ...], device: torch.device):
        del obs_shape  # repr dim is inferred directly from the selected visual encoder.
        self.cfg = cfg
        self.device = device
        acfg = cfg["agent"]

        # ------------------------------------------------------------------
        # Hyperparameters (DrQ-v2 style)
        # ------------------------------------------------------------------
        self.critic_target_tau = float(acfg.get("critic_target_tau", 0.01))
        self.update_every_steps = int(acfg.get("update_every_steps", 2))
        self.num_expl_steps = int(cfg["train"].get("seed_steps", 4000))
        self.stddev_schedule = acfg.get("stddev_schedule", "linear(1.0,0.1,100000)")
        self.stddev_clip = float(acfg.get("stddev_clip", 0.3))
        self.feature_dim = int(acfg.get("feature_dim", 256))
        self.hidden_dim = int(acfg.get("hidden_dim", 256))
        self.lr = float(acfg.get("lr", 1e-4))

        # Visual encoder
        self.encoder = VisionEncoderAdapter(cfg).to(device)

        # DrQ-v2-style heads: actor + twin critic + target critic
        repr_dim = self.encoder.repr_dim
        self.actor = Actor(repr_dim, action_shape, self.feature_dim, self.hidden_dim).to(device)
        self.critic = Critic(repr_dim, action_shape, self.feature_dim, self.hidden_dim).to(device)
        self.critic_target = Critic(repr_dim, action_shape, self.feature_dim, self.hidden_dim).to(device)
        self.critic_target.load_state_dict(self.critic.state_dict())

        # Optimizers
        self.encoder_opt = None
        self.actor_adapter_opt = None
        self.critic_adapter_opt = None

        if self.encoder.backbone_trainable:
            self.encoder_opt = torch.optim.Adam(self.encoder.backbone.parameters(), lr=self.lr)
        elif self.encoder.has_separate_adapters:
            self.actor_adapter_opt = torch.optim.Adam(
                self.encoder.actor_adapter_parameters(), lr=self.lr
            )
            self.critic_adapter_opt = torch.optim.Adam(
                self.encoder.critic_adapter_parameters(), lr=self.lr
            )

        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=self.lr)
        self.critic_opt = torch.optim.Adam(self.critic.parameters(), lr=self.lr)

        self.train(True)
        self.critic_target.train(True)

    # ------------------------------------------------------------------
    # DrQ-v2 Scheduler
    # ------------------------------------------------------------------
    def stddev(self, step: int) -> float:
        # DrQ-v2 uses only the exploration-noise schedule.
        return float(schedule(self.stddev_schedule, step))

    # ------------------------------------------------------------------
    # API helpers
    # ------------------------------------------------------------------
    def train(self, mode: bool = True) -> None:
        self.training = mode
        self.actor.train(mode)
        self.critic.train(mode)
        self.critic_target.train(mode)

        if hasattr(self.encoder, "set_update_mode"):
            if mode:
                self.encoder.set_update_mode()
            else:
                self.encoder.set_act_mode()
        else:
            self.encoder.train(mode)

    @torch.no_grad()
    def act(self, obs: np.ndarray | torch.Tensor, step: int, eval_mode: bool) -> np.ndarray:
        # Action selection is deterministic with respect to preprocessing.
        prev_mode = self.training
        self.train(False)

        obs_t = torch.as_tensor(obs, device=self.device)
        branch = "actor" if self.encoder.has_separate_adapters else None  # Frozen pretrained encoders use the actor-side adapter.
        obs_repr = self.encoder(obs_t, branch=branch)
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
            "actor_opt": self.actor_opt.state_dict(),
            "critic_opt": self.critic_opt.state_dict(),
        }
        if self.encoder_opt is not None:
            payload["encoder_opt"] = self.encoder_opt.state_dict()
        if self.actor_adapter_opt is not None:
            payload["actor_adapter_opt"] = self.actor_adapter_opt.state_dict()
        if self.critic_adapter_opt is not None:
            payload["critic_adapter_opt"] = self.critic_adapter_opt.state_dict()
        return payload

    def load_state_dict(self, payload: Dict[str, Any]) -> None:
        self.encoder.load_state_dict(payload["encoder"])
        self.actor.load_state_dict(payload["actor"])
        self.critic.load_state_dict(payload["critic"])
        self.critic_target.load_state_dict(payload["critic_target"])
        self.actor_opt.load_state_dict(payload["actor_opt"])
        self.critic_opt.load_state_dict(payload["critic_opt"])
        if self.encoder_opt is not None and "encoder_opt" in payload:
            self.encoder_opt.load_state_dict(payload["encoder_opt"])
        if self.actor_adapter_opt is not None and "actor_adapter_opt" in payload:
            self.actor_adapter_opt.load_state_dict(payload["actor_adapter_opt"])
        if self.critic_adapter_opt is not None and "critic_adapter_opt" in payload:
            self.critic_adapter_opt.load_state_dict(payload["critic_adapter_opt"])

    # ------------------------------------------------------------------
    # Internal update pieces
    # ------------------------------------------------------------------
    def _encode_single_input(
        self,
        obs=None,
        obs_feat=None,
        branch: str | None = None,
    ) -> torch.Tensor:
        """
        Encode one batch of observations.

        Modes:
          1) image mode: use obs
          2) cached mode: use obs_feat
        """
        self.encoder.set_update_mode()

        if obs_feat is not None:
            if not torch.is_tensor(obs_feat):
                obs_feat = torch.as_tensor(obs_feat, device=self.device)
            else:
                obs_feat = obs_feat.to(self.device, non_blocking=True)

            return self.encoder.forward_from_cached_backbone(obs_feat, branch=branch)

        if obs is None:
            raise ValueError("Need either obs or obs_feat.")

        return self.encoder(obs, branch=branch)

    def _encode_batch_for_critic(
        self,
        obs=None,
        next_obs=None,
        obs_feat=None,
        next_obs_feat=None,
    ):
        """
        Build the representations needed by the critic update.

        Shared ConvNet:
          - one shared representation path

        Frozen pretrained encoder:
          - critic representation for critic / target critic
          - actor representation for next-state policy sampling
            (important because target action should come from actor-side adapter)
        """
        if self.encoder.has_separate_adapters:
            obs_repr_critic = self._encode_single_input(
                obs=obs,
                obs_feat=obs_feat,
                branch="critic",
            )

            with torch.no_grad():
                next_repr_critic = self._encode_single_input(
                    obs=next_obs,
                    obs_feat=next_obs_feat,
                    branch="critic",
                )
                next_repr_actor = self._encode_single_input(
                    obs=next_obs,
                    obs_feat=next_obs_feat,
                    branch="actor",
                )

            return obs_repr_critic, next_repr_critic, next_repr_actor

        # Shared path (ConvNet or any no-adapter case)
        obs_repr = self._encode_single_input(obs=obs, obs_feat=obs_feat, branch=None)
        with torch.no_grad():
            next_repr = self._encode_single_input(obs=next_obs, obs_feat=next_obs_feat, branch=None)

        return obs_repr, next_repr, next_repr

    def _update_critic(
        self,
        obs_repr: torch.Tensor,
        action: torch.Tensor,
        reward: torch.Tensor,
        discount: torch.Tensor,
        next_repr_critic: torch.Tensor,
        next_repr_actor: torch.Tensor,
        step: int,
    ) -> Dict[str, float]:
        """
        Critic update.

        Important when using separate adapters:
          - next action is sampled from the ACTOR adapter output
          - target Q is evaluated from the CRITIC adapter output
        """
        with torch.no_grad():
            next_dist = self.actor(next_repr_actor, self.stddev(step))
            next_action = next_dist.sample(clip=self.stddev_clip)
            target_q1, target_q2 = self.critic_target(next_repr_critic, next_action)
            target_v = torch.min(target_q1, target_q2)
            target_q = reward + discount * target_v

        q1, q2 = self.critic(obs_repr, action)
        critic_loss = F.mse_loss(q1, target_q) + F.mse_loss(q2, target_q)

        if self.encoder_opt is not None:
            self.encoder_opt.zero_grad(set_to_none=True)
        if self.critic_adapter_opt is not None:
            self.critic_adapter_opt.zero_grad(set_to_none=True)
        self.critic_opt.zero_grad(set_to_none=True)

        critic_loss.backward()

        self.critic_opt.step()
        if self.critic_adapter_opt is not None:
            self.critic_adapter_opt.step()
        if self.encoder_opt is not None:
            self.encoder_opt.step()

        return {
            "critic_loss": float(critic_loss.item()),
            "critic_q1": float(q1.mean().item()),
            "critic_q2": float(q2.mean().item()),
            "target_q": float(target_q.mean().item()),
        }

    def _update_actor(
        self,
        step: int,
        obs=None,
        obs_feat=None,
        obs_repr_detached: torch.Tensor | None = None,
    ) -> Dict[str, float]:
        """
        Actor update.

        Shared ConvNet path:
        - use the detached shared representation (standard DrQ-v2 style)

        Frozen pretrained path with separate adapters:
        - actor uses the ACTOR adapter representation
        - critic value is evaluated on the CRITIC adapter representation
        - critic / critic-adapter are frozen during this update so only
            actor (+ actor adapter) receive gradients
        """
        # --------------------------------------------------------------
        # 1) Build actor-side and critic-side representations
        # --------------------------------------------------------------
        if self.encoder.has_separate_adapters:
            # Actor branch: should receive gradients from actor loss
            obs_repr_actor = self._encode_single_input(
                obs=obs,
                obs_feat=obs_feat,
                branch="actor",
            )

            # Critic branch: used only as critic input context, so do not let
            # actor loss update the critic adapter
            with torch.no_grad():
                obs_repr_critic = self._encode_single_input(
                    obs=obs,
                    obs_feat=obs_feat,
                    branch="critic",
                )
        else:
            if obs_repr_detached is None:
                raise ValueError("obs_repr_detached is required for the shared-encoder path.")

            # Shared DrQ-v2 path: actor uses detached encoder features
            obs_repr_actor = obs_repr_detached
            obs_repr_critic = obs_repr_detached

        # --------------------------------------------------------------
        # 2) Sample action from the current actor
        # --------------------------------------------------------------
        dist = self.actor(obs_repr_actor, self.stddev(step))
        action = dist.sample(clip=self.stddev_clip)

        # --------------------------------------------------------------
        # 3) Evaluate Q while freezing critic params
        #
        # Important:
        # - gradient must flow from Q -> action -> actor
        # - but should NOT update critic weights
        # --------------------------------------------------------------
        critic_requires_grad = [p.requires_grad for p in self.critic.parameters()]
        for p in self.critic.parameters():
            p.requires_grad_(False)

        q1, q2 = self.critic(obs_repr_critic, action)
        q = torch.min(q1, q2)
        actor_loss = -q.mean()

        # Restore critic grad flags
        for p, req in zip(self.critic.parameters(), critic_requires_grad):
            p.requires_grad_(req)

        # --------------------------------------------------------------
        # 4) Optimize actor (+ actor adapter, if present)
        # --------------------------------------------------------------
        if self.actor_adapter_opt is not None:
            self.actor_adapter_opt.zero_grad(set_to_none=True)
        self.actor_opt.zero_grad(set_to_none=True)

        actor_loss.backward()

        self.actor_opt.step()
        if self.actor_adapter_opt is not None:
            self.actor_adapter_opt.step()

        return {
            "actor_loss": float(actor_loss.item()),
            "actor_entropy": float(dist.entropy().sum(dim=-1).mean().item()),
        }
    
    # ------------------------------------------------------------------
    # Public update
    # ------------------------------------------------------------------
    def update(self, replay_iter, step: int) -> Dict[str, float]:
        """
        DrQ-v2 public update.
        """
        metrics: Dict[str, float] = {}

        if step % self.update_every_steps != 0:
            return metrics

        batch = next(replay_iter)
        if len(batch) == 5:
            obs, action, reward, discount, next_obs = batch
            obs_feat = next_obs_feat = None
        elif len(batch) == 7:
            obs, action, reward, discount, next_obs, obs_feat, next_obs_feat = batch
        else:
            raise ValueError(f"Unexpected replay batch length: {len(batch)}")

        def _to_device(x):
            if torch.is_tensor(x):
                return x.to(self.device, non_blocking=True)
            return torch.as_tensor(x, device=self.device)

        # Always move action/reward/discount
        action = _to_device(action)
        reward = _to_device(reward)
        discount = _to_device(discount)

        # Only move images if we are NOT using cached features
        if obs_feat is None or next_obs_feat is None:
            obs = _to_device(obs)
            next_obs = _to_device(next_obs)
        else:
            obs = None
            next_obs = None

        # Encode the batch and update critic and actor.
        obs_repr, next_repr_critic, next_repr_actor = self._encode_batch_for_critic(
            obs=obs,
            next_obs=next_obs,
            obs_feat=obs_feat,
            next_obs_feat=next_obs_feat,
        )

        metrics["batch_reward"] = float(reward.mean().item())
        metrics["stddev"] = float(self.stddev(step))

        metrics.update(
            self._update_critic(
                obs_repr,
                action,
                reward,
                discount,
                next_repr_critic,
                next_repr_actor,
                step,
            )
        )

        if self.encoder.has_separate_adapters:
            metrics.update(
                self._update_actor(
                    step=step,
                    obs=obs,
                    obs_feat=obs_feat,
                )
            )
        else:
            metrics.update(
                self._update_actor(
                    step=step,
                    obs_repr_detached=obs_repr.detach(),
                )
            )

        soft_update_params(self.critic, self.critic_target, self.critic_target_tau)
        return metrics