"""FlashSAC downstream RL (user directive 2026-10-29): the official FlashSAC agent with a visual adapter, no RL image
augmentation, on the shared reference protocol.

Upstream: github.com/Holiday-Robot/FlashSAC @ 87edc90 (submodule ``third_party/flashsac``, MIT; Kim, Lee et al.,
"FlashSAC: Fast and Stable Off-Policy Reinforcement Learning for High-Dimensional Robot Control", arXiv:2604.04539).
The public code is PyTorch and STATE-VECTOR ONLY: it contains no image encoder, no pixel replay and no augmentation.
Imported UNCHANGED (sha256-pinned in tests/test_flashsac.py) from the pinned checkout:
- ``flash_rl/agents/flashSAC/agent.py``: ``FlashSACConfig``, ``FlashSACAgent.__init__`` (target entropy from
  ``temp_target_sigma`` (agent.py:355), network/optimiser/cosine-LR construction ``_init_flashsac_networks``, grad
  scaler, truncated-zeta noise-repetition state, ``RewardNormalizer``), ``FlashSACAgent.sample_actions`` (exploration:
  noise repetition, agent.py:223-257, 424-457; evaluation: tanh(mean)) and ``_update_networks`` (actor -> temperature
  -> critic -> target EMA, actor update period, agent.py:260-323);
- ``network.py`` / ``layer.py``: ``FlashSACActor``, ``FlashSACDoubleCritic`` (inverted-residual blocks, pre-activation
  BatchNorm, post-RMSNorm, categorical value head), ``FlashSACTemperature``, unit-norm weight projection;
- ``update.py``: ``update_actor``, ``update_critic`` (cross-batch [s, s'] BatchNorm statistics, min-Q categorical
  projection target, entropy term), ``update_temperature``, ``update_target_network``;
- ``agents/utils/{network,reward_normalization,scheduler,distribution}.py``: ``Network`` (weight normalisation, EMA),
  ``RewardNormalizer`` (running discounted-return variance and G_max bound, eq. 6), warmup-cosine LR, safe tanh.
``flash_rl/types.py`` imports ``jax.numpy`` for a type alias only and ``flash_rl/agents/__init__.py`` imports omegaconf
for ``create_agent`` (not used); a missing package is replaced by an inert stub module during that import only. The
module-level ``@torch.compile`` helpers of upstream (zeta sampling, min-Q selection, categorical target, reward
statistics) stay compiled as upstream (inductor, one compile thread), except the one-off CPU construction of the
truncated-zeta CDF table at agent construction, which runs eagerly (no C++ compiler in the pool image); ``agent.use_compile`` (upstream: also compile the
networks) is a speed switch. Upstream ``train.py`` (hydra, its state-vector environment wrappers incl. the
``SawyerXYZEnv`` Meta-World v2 wrapper, tensorboard/wandb logger, in-agent ``TorchUniformBuffer``) is NOT used: the
loop is ``scripts/train_flashsac.py`` on our ``MetaWorldCameraEnv`` observation protocol, replay and evaluation.

VISUAL ADAPTER (ours; the paper's vision experiments (Sec. 5.3, Table 11) have no released code, so this is NOT a
verbatim reproduction of the published visual implementation). Paper: "a lightweight convolutional encoder (3
convolutional layers followed by a linear bottleneck)" matching "the standard DrQ-v2 configuration"; Table 11: encoder
4 layers, 32 channels, output feature dim 50, frame stack 3, n-step 3, batch 256, UTD 0.5, 1M replay. Implemented as:
- ``FeatureEncoder`` = backbone -> one linear bottleneck to ``agent.feature_dim`` (50) features; the protocol's 4-d
  proprioception is appended -> FlashSAC actor and critic input (54-d), identical for every family;
  * ``cnn``: backbone = our protocol's ``s4d.rl.encoders.ConvNet`` (the DrQ-v2 / DrM encoder: 4 convolutions, 32
    channels, 3x3, strides 2,1,1,1, ReLU: Table 11's 4 layers x 32 channels and DrQ-v2's standard; Sec. 5.3's "3
    convolutional layers" contradicts Table 11 and is not followed); input = the 9-channel 128-px uint8 stack,
    deterministic ``/255 - 0.5`` only; trained end to end;
  * frozen exports (csw4/splatter4d, SinCro, ReViWo, DynaMo, SNeRL, 3D-MVP A/B): backbone = the frozen export run once
    per environment step through ``s4d.rl.snerl_sac.FrozenFeatures`` (= ``VisionEncoderAdapter.extract_cacheable_feature``,
    the DrM/SAC/PPO/Dreamer loading and extraction incl. the multi-view / depth / camera dict observations), its fp16
    feature cached in replay (frozen, deterministic, no augmentation); the bottleneck takes the flattened feature
    (per-frame features of the 3 frames concatenated, as the CNN stacks frames as channels).
- Gradient routing (DrQ-v2 convention, the configuration the paper says it adopts): the encoder (ConvNet + bottleneck,
  or the bottleneck of a frozen family) is trained by the CRITIC loss only; the actor sees detached features; there is
  no target encoder: the target critic (upstream's EMA of the critic's FlashSAC layers) reads the online encoder's
  features. ``EncodedCritic`` puts the encoder in front of upstream's
  ``FlashSACDoubleCritic`` so that upstream ``update_critic`` (which concatenates [s, s'] for cross-batch BatchNorm
  under ``no_grad`` and then calls the critic) back-propagates its cross-entropy loss into the encoder; the encoder's
  parameters join the critic's Adam and cosine schedule. Per update the encoder runs once on s with its graph and once
  on s' without (the encoder does not change before the critic step): the critic loss reaches the encoder through
  the features of s, those of s' are constants (DrQ-v2's target side; they still enter the cross-batch BatchNorm
  statistics as upstream), and the actor inputs, the target critic and the critic inside the actor loss read the
  detached features.
- No stochastic image augmentation of any kind (no random shift / crop / resized crop / colour jitter / masking, no
  augmentation-based auxiliary loss). FlashSAC's action exploration (noise repetition, entropy target sigma 0.15) and
  numerical stabilisation (BatchNorm, RMSNorm, unit-norm projection, categorical critic, reward normalisation) are
  upstream's and unchanged.

Reward normalisation: upstream ``RewardNormalizer`` sees every raw agent-step reward (the sum over the 2 simulator
steps of action repeat) with ``truncated`` at the protocol's time limit, as ``process_transition``; batch rewards (our
replay's 3-step discounted sums of raw rewards, as upstream's n-step buffer) are scaled by it before the critic
update. All logged and evaluated returns are raw environment returns.
"""

from __future__ import annotations

import importlib
import importlib.util
import math
import os
import sys
import types
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

REPO = Path(__file__).resolve().parents[2]
SUBMODULE = "flashsac"
PINNED_COMMIT = "87edc9061150ae9e962dd84e6544e27a1554b3ab"
UPSTREAM_MODULES = {
    "agent": "flash_rl.agents.flashSAC.agent",
    "network": "flash_rl.agents.flashSAC.network",
    "layer": "flash_rl.agents.flashSAC.layer",
    "update": "flash_rl.agents.flashSAC.update",
    "nets": "flash_rl.agents.utils.network",
    "reward_normalization": "flash_rl.agents.utils.reward_normalization",
    "scheduler": "flash_rl.agents.utils.scheduler",
}
# Packages imported at module level by flash_rl/types.py (jax.numpy: a type alias) and flash_rl/agents/__init__.py
# (omegaconf: create_agent, not used); stubbed during the import only when not installed.
STUB_MODULES = ("jax", "jax.numpy", "omegaconf")


def upstream_dir() -> Path:
    """The clean pinned checkout (``s4d.third_party.source_tree`` refuses a missing, moved or dirty one)."""
    from s4d.third_party import source_tree

    return source_tree(SUBMODULE, repo=REPO)


def _stub(name: str) -> types.ModuleType:
    module = types.ModuleType(name)
    module.__s4d_stub__ = True
    if name == "jax.numpy":
        module.ndarray = type("ndarray", (), {})  # only referenced by the ``Tensor`` type alias
    if name == "omegaconf":
        class OmegaConf:  # noqa: D401  (create_agent's config conversion; never called by this port)
            @staticmethod
            def to_container(*_args, **_kwargs):
                raise RuntimeError("omegaconf is not installed; flash_rl.agents.create_agent is not used")

        module.OmegaConf = OmegaConf
    return module


def _missing(top: str) -> bool:
    if top in sys.modules:
        return sys.modules[top] is None
    return importlib.util.find_spec(top) is None


def load_upstream() -> dict[str, types.ModuleType]:
    """Import the upstream FlashSAC modules unchanged from the pinned checkout (package ``flash_rl``; stubs for missing
    jax / omegaconf during the import only; no bytecode written into the checkout)."""
    if all(name in sys.modules for name in UPSTREAM_MODULES.values()):
        return {key: sys.modules[name] for key, name in UPSTREAM_MODULES.items()}
    root = upstream_dir()
    missing = {top for top in ("jax", "omegaconf") if _missing(top)}
    stubs = [name for name in STUB_MODULES if name.split(".")[0] in missing]
    blocked = {name: sys.modules[name] for name in stubs if name in sys.modules}
    for name in stubs:
        sys.modules[name] = _stub(name)
    if "jax.numpy" in stubs:
        sys.modules["jax"].numpy = sys.modules["jax.numpy"]
    if "TORCHINDUCTOR_COMPILE_THREADS" not in os.environ:  # upstream's compiled helpers: no compile-worker pool
        import torch._inductor.config as inductor_config

        inductor_config.compile_threads = 1
    bytecode = sys.dont_write_bytecode
    sys.dont_write_bytecode = True
    sys.path.insert(0, str(root))
    try:
        modules = {key: importlib.import_module(name) for key, name in UPSTREAM_MODULES.items()}
    finally:
        sys.path.remove(str(root))
        sys.dont_write_bytecode = bytecode
        for name in stubs:
            sys.modules.pop(name, None)
        sys.modules.update(blocked)
    for key, module in modules.items():
        if not str(Path(module.__file__).resolve()).startswith(str(root.resolve())):
            raise RuntimeError(f"{UPSTREAM_MODULES[key]} was not imported from the pinned checkout {root}")
    return modules


def family(cfg: dict) -> str:
    """``pixel``: the end-to-end CNN (encoder ``cnn``); ``frozen``: every pretrained export (cached features)."""
    return "pixel" if str(cfg["vision"]["encoder_type"]) == "convnet" else "frozen"


def _module(net: nn.Module) -> nn.Module:
    return getattr(net, "_orig_mod", net)  # the uncompiled module of a torch.compile'd network


class FeatureEncoder(nn.Module):
    """Backbone (trainable ``ConvNet`` for the CNN; none for a frozen family, whose input is the cached feature) ->
    linear bottleneck to ``feature_dim`` features (Table 11: output feature dim 50)."""

    def __init__(self, backbone: nn.Module | None, in_dim: int, feature_dim: int):
        super().__init__()
        from s4d.rl.encoders import weight_init

        self.backbone = backbone
        self.bottleneck = nn.Linear(int(in_dim), int(feature_dim))
        self.bottleneck.apply(weight_init)  # DrQ-v2 / DrM initialisation (orthogonal, zero bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.backbone(x) if self.backbone is not None else x.float().flatten(1)
        return self.bottleneck(h)


class EncodedCritic(nn.Module):
    """``FeatureEncoder`` in front of upstream ``FlashSACDoubleCritic`` (the online critic of ``Network``).

    Rows of ``observations`` are packed by ``FlashSAC.update``: [encoder features (feature_dim), proprio, row index],
    rows 0..B-1 the batch's s and rows B..2B-1 its s'. With gradients enabled and the encoder trainable (upstream
    ``update_critic``'s online critic forward on [s; s']), the features of s are the encoder output that carries the
    gradient (``grad_features``, computed once per update), so the critic loss trains the encoder; the features of s'
    stay constants, as the target side of DrQ-v2. Otherwise (``update_actor`` freezes the critic with
    ``requires_grad_(False)``) every row uses the packed no-grad features."""

    def __init__(self, encoder: FeatureEncoder, core: nn.Module, feature_dim: int):
        super().__init__()
        self.encoder, self.core, self.feature_dim = encoder, core, int(feature_dim)
        self.grad_features: torch.Tensor | None = None  # encoder(s) with its graph, during FlashSAC.update only

    def forward(self, observations: torch.Tensor, actions: torch.Tensor, training: bool):
        x = observations[:, :-1]
        if torch.is_grad_enabled() and self.encoder.bottleneck.weight.requires_grad:
            if self.grad_features is None:
                raise RuntimeError("the encoder's gradient path only exists inside FlashSAC.update")
            rows = observations[:, -1].long()
            own = rows < self.grad_features.shape[0]
            grad = self.grad_features.to(x.dtype)[rows.clamp(max=self.grad_features.shape[0] - 1)]
            features = torch.where(own[:, None], grad, x[:, : self.feature_dim])
            x = torch.cat([features, x[:, self.feature_dim :]], dim=-1)
        return self.core(x, actions, training)


class FeatureCritic(nn.Module):
    """Upstream target ``FlashSACDoubleCritic`` reading the packed no-grad features (no target encoder)."""

    def __init__(self, core: nn.Module):
        super().__init__()
        self.core = core

    def forward(self, observations: torch.Tensor, actions: torch.Tensor, training: bool):
        return self.core(observations[:, :-1], actions, training)


def flashsac_config(up: dict, cfg: dict, device: torch.device):
    """Upstream ``FlashSACConfig`` from the run config, with the derived entries of upstream's yaml
    (configs/agent/flashSAC.yaml: critic support [-G_max, G_max]; warmup / decay steps = rate x interaction steps x
    UTD). Upstream's in-agent replay is never used (our ``Replay`` holds ``buffer_max_length`` transitions), so the
    agent's own ``TorchUniformBuffer`` is built with one CPU row."""
    a, t = cfg["agent"], cfg["train"]
    updates = int(t["num_train_steps"]) * float(t["updates_per_interaction_step"])
    return up["agent"].FlashSACConfig(
        seed=int(cfg["seed"]),
        normalize_reward=bool(a["normalize_reward"]),
        normalized_G_max=float(a["normalized_G_max"]),
        asymmetric_observation=False,
        device_type="cuda" if device.type == "cuda" else "cpu",
        buffer_max_length=1,
        buffer_min_length=int(a["buffer_min_length"]),
        buffer_device_type="cpu",
        sample_batch_size=int(a["sample_batch_size"]),
        learning_rate_init=float(a["learning_rate_init"]),
        learning_rate_peak=float(a["learning_rate_peak"]),
        learning_rate_end=float(a["learning_rate_end"]),
        learning_rate_warmup_rate=float(a["learning_rate_warmup_rate"]),
        learning_rate_warmup_step=int(float(a["learning_rate_warmup_rate"]) * updates),
        learning_rate_decay_rate=float(a["learning_rate_decay_rate"]),
        learning_rate_decay_step=int(float(a["learning_rate_decay_rate"]) * updates),
        actor_num_blocks=int(a["actor_num_blocks"]),
        actor_hidden_dim=int(a["actor_hidden_dim"]),
        actor_bc_alpha=float(a["actor_bc_alpha"]),
        actor_noise_zeta_mu=float(a["actor_noise_zeta_mu"]),
        actor_noise_zeta_max=int(a["actor_noise_zeta_max"]),
        actor_update_period=int(a["actor_update_period"]),
        critic_num_blocks=int(a["critic_num_blocks"]),
        critic_hidden_dim=int(a["critic_hidden_dim"]),
        critic_num_bins=int(a["critic_num_bins"]),
        critic_min_v=-float(a["normalized_G_max"]),
        critic_max_v=float(a["normalized_G_max"]),
        critic_target_update_tau=float(a["critic_target_update_tau"]),
        temp_initial_value=float(a["temp_initial_value"]),
        temp_target_sigma=float(a["temp_target_sigma"]),
        temp_target_entropy=None,  # set by FlashSACAgent.__init__ from temp_target_sigma (agent.py:355)
        gamma=float(a["gamma"]),
        n_step=int(a["n_step"]),
        use_compile=bool(a["use_compile"]),
        compile_mode=str(a["compile_mode"]),
        use_amp=bool(a["use_amp"]),
        load_optimizer=True,
        load_reward_normalizer=True,
    )


def _network_state(net) -> dict:
    """Upstream ``Network.save``'s payload, in memory."""
    return {"network_state_dict": _module(net.network).state_dict(),
            "optimizer_state_dict": net.optimizer.state_dict() if net.optimizer is not None else None,
            "scheduler_state_dict": net.scheduler.state_dict() if net.scheduler is not None else None,
            "update_step": net.update_step}


def _load_network_state(net, state: dict) -> None:
    _module(net.network).load_state_dict(state["network_state_dict"])
    if net.optimizer is not None:
        net.optimizer.load_state_dict(state["optimizer_state_dict"])
    if net.scheduler is not None:
        net.scheduler.load_state_dict(state["scheduler_state_dict"])
    net.update_step = state["update_step"]


class FlashSAC:
    """Upstream ``FlashSACAgent`` + the visual adapter; the interface of ``s4d.rl.agent.DrMAgent`` used by training
    and ``s4d.rl.evaluate`` (``device``, ``use_pixels``, ``encoder.extract_cacheable_feature``, ``act``, ``train``,
    ``load_policy``). Also the evaluation policy (``s4d.rl.algorithms``)."""

    def __init__(self, cfg: dict, action_dim: int, proprio_dim: int, device: torch.device):
        import gymnasium

        self.up = up = load_upstream()
        self.cfg, self.device, self.proprio_dim = cfg, device, int(proprio_dim)
        self.family = family(cfg)
        self.use_pixels = self.family == "pixel"
        self.encoder = self  # evaluate.policy_inputs calls agent.encoder.extract_cacheable_feature
        a = cfg["agent"]
        frames = int(cfg["env"]["frame_stack"])
        self.feature_dim = int(a["feature_dim"])
        if self.use_pixels:
            from s4d.rl.encoders import ConvNet

            backbone, self.features = ConvNet(cfg), None
            in_dim = int(backbone.repr_dim)
        else:
            from s4d.rl.snerl_sac import FrozenFeatures

            backbone, self.features = None, FrozenFeatures(cfg, device)
            in_dim = self.features.dim * (frames if self.features.mode == "frame" else 1)
        obs_dim = self.feature_dim + self.proprio_dim
        self.fcfg = flashsac_config(up, cfg, device)
        observation_space = gymnasium.spaces.Box(-np.inf, np.inf, (obs_dim,), np.float32)
        action_space = gymnasium.spaces.Box(-1.0, 1.0, (int(action_dim),), np.float32)
        # Construction runs eagerly: its one compiled helper, the truncated-zeta CDF table, is built on the CPU, and
        # inductor's CPU backend needs a C++ compiler, which the campaign's pool image lacks (it has gcc for Triton).
        # Eager gives the same table, which is moved to the device. Every helper upstream calls during acting and
        # updates runs on CUDA tensors and stays compiled (Triton).
        with torch.compiler.set_stance("force_eager"):
            self.agent = up["agent"].FlashSACAgent(observation_space, action_space, {}, self.fcfg)
        self.feature_encoder = FeatureEncoder(backbone, in_dim, self.feature_dim).to(device)
        critic = self.agent._critic
        critic.network = EncodedCritic(self.feature_encoder, critic.network, self.feature_dim)
        # the encoder joins the critic's Adam and its warmup-cosine schedule (rebuilt over both parameter groups
        # before any step; group 0 keeps its state)
        critic.optimizer.add_param_group({"params": list(self.feature_encoder.parameters())})
        critic.scheduler = torch.optim.lr_scheduler.LambdaLR(critic.optimizer, lr_lambda=critic.scheduler.lr_lambdas[0])
        self.agent._target_critic.network = FeatureCritic(self.agent._target_critic.network)
        self.update_info: dict[str, torch.Tensor] = {}

    # ------------------------------------------------------------------ policy inputs and acting
    @torch.no_grad()
    def extract_cacheable_feature(self, obs):
        """Batched env observation (already on the device) -> the frozen feature the replay stores per state."""
        if self.use_pixels:
            raise RuntimeError("the CNN family consumes pixels directly")
        return self.features(obs)

    def _amp(self):
        return torch.autocast(device_type=self.device.type, dtype=torch.float16, enabled=self.fcfg.use_amp)

    @torch.no_grad()
    def policy_observation(self, inputs: torch.Tensor, proprio) -> torch.Tensor:
        """[encoder features (feature_dim), proprioception]: the 54-d observation of upstream's actor."""
        proprio = torch.as_tensor(np.asarray(proprio), device=self.device, dtype=torch.float32).reshape(len(inputs), -1)
        with self._amp():
            features = self.feature_encoder(inputs).float()
        return torch.cat([features, proprio], dim=-1)

    @torch.no_grad()
    def act(self, inputs, proprio, step: int, eval_mode: bool) -> np.ndarray:
        """Upstream ``sample_actions``: deterministic tanh(mean) for evaluation; for training a policy sample with
        noise repetition (agent.py:240-257). Uniform actions before ``buffer_min_length`` come from the loop."""
        observation = self.policy_observation(inputs, proprio)
        return self.agent.sample_actions(int(step), {"next_observation": observation}, training=not eval_mode)

    def train(self, training: bool = True) -> None:
        """Upstream passes ``training`` explicitly to every forward; nothing to switch (no dropout, the adapter has no
        normalisation layers)."""

    # ------------------------------------------------------------------ learning
    def observe_reward(self, reward: float, done: bool) -> None:
        """Upstream ``process_transition``'s reward-statistics update with one environment's transition, as the
        gymnasium vector env hands it (float64 reward array; the time limit is a truncation, never a termination)."""
        if self.agent.reward_normalizer is None:
            return
        dev = self.agent._device
        self.agent.reward_normalizer.update_reward_stats(
            reward=torch.as_tensor(np.array([reward], dtype=np.float64), device=dev),
            terminated=torch.as_tensor(np.array([False]), device=dev),
            truncated=torch.as_tensor(np.array([bool(done)]), device=dev),
        )

    def reward_scale(self) -> float:
        """The current denominator of upstream ``_scale_reward`` (eq. 6), for logging."""
        norm = self.agent.reward_normalizer
        if norm is None:
            return 1.0
        return float(torch.maximum(torch.sqrt(norm.G_rms.var + norm.epsilon), norm.G_r_max / norm.G_max).item())

    def update(self, batches) -> dict[str, torch.Tensor]:
        """One upstream update (``FlashSACAgent.update``, agent.py:475-505) on a batch of our replay:
        (obs, proprio, action, reward, discount, next_obs, next_proprio) with n-step discounted raw rewards and
        discount gamma^n x continuation (1.0 at the time limit, so ``terminated`` is 0 as upstream at truncation)."""
        a = self.agent
        obs, proprio, action, reward, discount, next_obs, next_proprio = (
            t.to(a._device, non_blocking=True) for t in next(batches))
        b = obs.shape[0]
        # encoder(s) once with its graph (kept for the critic loss, see EncodedCritic), encoder(s') once without
        with self._amp():
            grad_features = self.feature_encoder(obs)
            with torch.no_grad():
                next_features = self.feature_encoder(next_obs)
        features = torch.cat([grad_features.detach(), next_features], dim=0).float()
        detached = torch.cat([features, torch.cat([proprio, next_proprio], dim=0).float()], dim=-1)
        rows = torch.arange(2 * b, device=a._device, dtype=torch.float32)[:, None]
        packed = torch.cat([detached, rows], dim=-1)
        batch = {
            "observation": packed[:b],
            "next_observation": packed[b:],
            "action": action.float(),
            "reward": reward.float().reshape(-1),
            "terminated": (discount.float().reshape(-1) <= 0).float(),
            "truncated": torch.zeros(b, device=a._device),
            "actor_observation": detached[:b],
            "actor_next_observation": detached[b:],
        }
        if a._cfg.normalize_reward:  # agent.py:488-491
            batch["reward"] = a.reward_normalizer.normalize_rewards(batch["reward"])
        critic = _module(a._critic.network)
        critic.grad_features = grad_features
        try:
            info = self.up["agent"]._update_networks(  # agent.py:494-505
                batch=batch,
                actor=a._actor,
                critic=a._critic,
                target_critic=a._target_critic,
                temperature=a._temperature,
                cfg=a._cfg,
                do_actor_update=(a._update_step % a._cfg.actor_update_period == 0),
                device=a._device,
                grad_scaler=a._grad_scaler,
            )
        finally:
            critic.grad_features = None
        a._update_step += 1
        self.update_info.update(info)  # tensors; converted to floats only when logged (upstream: every update)
        return info

    def metrics(self) -> dict[str, float]:
        return {k: float(v.item() if torch.is_tensor(v) else v) for k, v in self.update_info.items()}

    # ------------------------------------------------------------------ state
    def state_dict(self) -> dict:
        """Everything upstream ``save`` writes (actor, critic incl. our encoder, target critic, temperature, reward
        normaliser, update step, grad scaler) plus the noise-repetition state, for an exact resume."""
        a = self.agent
        norm = a.reward_normalizer
        return {
            "actor": _network_state(a._actor),
            "critic": _network_state(a._critic),
            "target_critic": _network_state(a._target_critic),
            "temperature": _network_state(a._temperature),
            "reward_normalizer": None if norm is None else {
                "G_r": norm.G_r, "G_r_max": norm.G_r_max, "G_rms_mean": norm.G_rms.mean, "G_rms_var": norm.G_rms.var,
                "G_rms_count": norm.G_rms.count},
            "update_step": a._update_step,
            "grad_scaler": a._grad_scaler.state_dict(),
            "noise": {"cached_noise": a._cached_noise, "count": a._cur_noise_repeat_count, "n": a._cur_noise_repeat_n},
        }

    def load_state_dict(self, state: dict) -> None:
        a = self.agent
        for key, net in (("actor", a._actor), ("critic", a._critic), ("target_critic", a._target_critic),
                         ("temperature", a._temperature)):
            _load_network_state(net, state[key])
        if a.reward_normalizer is not None:
            norm, saved, dev = a.reward_normalizer, state["reward_normalizer"], a._device
            norm.G_r, norm.G_r_max = saved["G_r"].to(dev), saved["G_r_max"].to(dev)
            norm.G_rms.mean, norm.G_rms.var = saved["G_rms_mean"].to(dev), saved["G_rms_var"].to(dev)
            norm.G_rms.count = saved["G_rms_count"].to(dev)
        a._update_step = int(state["update_step"])
        a._grad_scaler.load_state_dict(state["grad_scaler"])
        noise = state["noise"]
        a._cached_noise = noise["cached_noise"].to(a._device)
        a._cur_noise_repeat_count = noise["count"].to(a._device)
        a._cur_noise_repeat_n = noise["n"].to(a._device)

    def policy_state(self) -> dict:
        """What evaluation needs: the encoder (ConvNet + bottleneck, or the bottleneck of a frozen family) and the
        actor (incl. its BatchNorm running statistics)."""
        return {"encoder": self.feature_encoder.state_dict(), "actor": _module(self.agent._actor.network).state_dict()}

    def load_policy(self, policy: dict) -> None:
        self.feature_encoder.load_state_dict(policy["encoder"])
        _module(self.agent._actor.network).load_state_dict(policy["actor"])


def target_entropy(action_dim: int, sigma: float) -> float:
    """Upstream's unified entropy target (agent.py:355; paper eq. 7)."""
    return 0.5 * action_dim * math.log(2 * math.pi * math.e * sigma**2)

