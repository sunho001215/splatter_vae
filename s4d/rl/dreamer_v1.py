"""Dreamer v1 pilot (user directive 2026-10-28): frozen pretrained encoders vs. an end-to-end CNN under one Dreamer recipe.

Upstream: github.com/yusukeurakami/dreamer-pytorch @ b51af83 (submodule ``third_party/dreamer-pytorch``, MIT; Dreamer v1,
Hafner et al. 2019), the code base whose argument names SNeRL's Table 3 (ICML 2023) uses for its Dreamer experiments.
Imported UNCHANGED (sha256-pinned in tests/test_dreamer_v1.py) under unique module names ``dreamer_upstream.*``:
``models.py`` (``TransitionModel``, ``ObservationModel``, ``RewardModel``, ``ValueModel``, ``ActorModel``, ``Encoder``
incl. the symbolic and visual variants, ``bottle``), ``utils.py`` (``imagine_ahead``, ``lambda_return``,
``FreezeParameters``), ``env.py`` (``preprocess_observation_``: bit-depth quantisation + dequantisation noise) and
``memory.py`` (``ExperienceReplay._sample_idx`` / ``sample``). ``utils.py`` and ``env.py`` import cv2 and plotly at module
level for helpers this port never calls (plots, videos, the gym/dm_control wrappers); a missing package is replaced by an
inert stub module during that import only (as ``s4d.rl.snerl_sac.load_upstream``). ``main.py`` (argparse script with
tensorboardX, dm_control/gym environments) is not imported: its training loop is reproduced by ``DreamerAgent.update``
and ``scripts/train_dreamer.py`` with ``main.py:<line>`` references. Its non-default options stay off: latent
overshooting (``--overshooting-kl-beta`` 0, main.py:77-83, 391-455), the learning-rate schedule (0, main.py:97-103,
456-461) and the log-probability world-model loss (store_true, main.py:65-69, 362-377); they are not ported.

REPRESENTATION INTERFACE (identical for every arm except the encoder input):
- Decoder target: the current frame of the episode's (first) training camera, area-downsampled 128 -> 64 px (stored as
  uint8 after rounding), bit depth 5 with dequantisation noise (env.py:48-54 ``preprocess_observation_``, applied when a
  batch is drawn as memory.py:47-58 does), reconstructed by upstream's VISUAL ``ObservationModel`` (64x64 deconvolution
  decoder, models.py:147-169) with the summed-MSE loss (main.py:370-374). Reward model, KL terms (free nats 3, optional
  global KL) and the actor/value objectives are the same modules and losses for every arm.
- Frozen arms (csw4/splatter4d, SinCro, ReViWo, DynaMo, SNeRL, 3D-MVP A/B): the frozen backbone runs once per
  environment step on the protocol observation (3-frame 128-px history; SNeRL and 3D-MVP A: 3 views + cameras, 3D-MVP B:
  RGB-D + camera, per their encoder yaml) through ``s4d.rl.snerl_sac.FrozenFeatures`` =
  ``s4d.rl.encoders.VisionEncoderAdapter.extract_cacheable_feature`` (the DrM/SAC loading and extraction; SNeRL through
  ``SNeRLFrameEncoder``: its 63-d per-frame output for the 3 frames). Its flattened cached feature is the "symbolic
  observation" of upstream's SYMBOLIC ``Encoder`` (trainable 3-layer MLP -> ``embedding_size``, models.py:272-286),
  trained by the world-model loss. Replay stores one feature atom per step (a per-frame feature, or the joint feature of
  the whole history for history encoders) and rebuilds the 3-frame stack by index, the first frame repeated at an
  episode start exactly as the environment's own history.
- End-to-end CNN arm: upstream's VISUAL ``Encoder`` (models.py:289-311) on the 3-frame 64-px stack (the same
  downsampled frames, bit-depth preprocessed), its first convolution adapted to 9 input channels, nothing else changed;
  trained by the world-model loss. The newest frame of the stack and the decoder target share one dequantisation draw,
  as upstream's encoder input and target are one tensor (main.py:359, 371).
- Proprioception: the protocol's 4-d hand state, concatenated to the embedding for every arm (``TransitionModel`` built
  with ``embedding_size + 4``; upstream has none).
- Hyperparameters: SNeRL Table 3 where given (belief 128, state 30, action noise 0.3, batch 32, model lr 1e-3, actor and
  value lr 5e-5, discount 0.99, replay 100k transitions, imagination horizon 15, Adam eps 1e-7); embedding_size 1024 for
  every arm (Dreamer's CNN embedding; Table 3's 63 is SNeRL's feature size, which our frozen arms project from); the rest
  dreamer-pytorch defaults (hidden 200, chunk 50, free nats 3, lambda 0.95, grad clip 100, 5 seed episodes, bit depth 5).
  Upstream runs 100 updates per 500-agent-step episode (0.2 per agent step); kept with 25 updates per 125-step episode.
  Budget: 500k agent steps (1M simulator steps) for every arm. See configs/rl/algorithms/dreamer_v1.yaml.

Replay (``DreamerReplay``, a subclass of upstream ``ExperienceReplay`` reusing ``_sample_idx`` and ``sample``): row t =
(o_t, a_t, r_{t+1}, nonterminal_{t+1}) as memory.py:24-37 (the final observation of an episode is not stored, as
upstream); chunks of ``chunk_size`` consecutive rows drawn uniformly, possibly across episode boundaries, where the
nonterminal flag zeroes the latent state (models.py:92-94). The flag marks the time limit (our only episode end): Dreamer
v1 never predicts termination, imagination always continues and the lambda-returns bootstrap from the value, so the time
limit is bootstrapped through. Storage: disk memmaps ``runs/<id>/replay/*.memmap`` (lifecycle cleanup's default
patterns); each row also records its global index so that rows written after the last checkpoint by a crashed process
are never sampled after a resume.

Acting (main.py:266-289 ``update_belief_and_act``): posterior update from (previous posterior sample, previous action,
current embedding), then ``ActorModel.get_action`` (``det=True`` for evaluation: upstream's sampled mode; training: a
policy sample plus N(0, 0.3) noise clamped to [-1, 1]). Recurrent state per episode (``recurrent = True``,
``begin_episodes``, ``act(..., index=active positions)``) for ``s4d.rl.evaluate.evaluation_suite``. The posterior
sample and the sampled mode make evaluation stochastic, as upstream's test episodes.
"""

from __future__ import annotations

import importlib
import importlib.util
import sys
import types
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.distributions import Normal
from torch.distributions.kl import kl_divergence

REPO = Path(__file__).resolve().parents[2]
SUBMODULE = "dreamer-pytorch"
UPSTREAM_MODULES = ("env", "models", "utils", "memory")  # memory.py imports env at module level
PREFIX = "dreamer_upstream"
# Packages imported at module level by utils.py / env.py and only used by helpers this port never calls.
STUB_MODULES = ("cv2", "plotly", "plotly.graph_objs", "plotly.graph_objs.scatter")


def upstream_dir() -> Path:
    """The clean pinned checkout (``s4d.third_party.source_tree`` refuses a missing, moved or dirty one)."""
    from s4d.third_party import source_tree

    return source_tree(SUBMODULE, repo=REPO)


def _stub(name: str) -> types.ModuleType:
    module = types.ModuleType(name)
    module.__s4d_stub__ = True

    def unavailable(*_args, **_kwargs):
        raise RuntimeError(f"{name} is not installed; the dreamer-pytorch plotting/video helpers are not used")

    if name == "plotly":
        module.offline = types.SimpleNamespace(plot=unavailable)
    if name == "plotly.graph_objs":
        module.Scatter = unavailable
    if name == "plotly.graph_objs.scatter":
        module.Line = unavailable
    return module


def _missing(top: str) -> bool:
    if top in sys.modules:
        return sys.modules[top] is None  # an import of a None entry raises ImportError
    return importlib.util.find_spec(top) is None


def load_upstream() -> dict[str, types.ModuleType]:
    """Import upstream ``env``, ``models``, ``utils`` and ``memory`` unchanged, isolated under ``dreamer_upstream.*``:
    the generic names are importable only during the import (previous holders restored), stubs replace missing cv2 /
    plotly only during the import, and no bytecode is written into the pinned checkout."""
    if f"{PREFIX}.memory" in sys.modules:
        return {name: sys.modules[f"{PREFIX}.{name}"] for name in UPSTREAM_MODULES}
    root = str(upstream_dir())
    displaced = {name: sys.modules.pop(name) for name in UPSTREAM_MODULES if name in sys.modules}
    missing = {top for top in ("cv2", "plotly") if _missing(top)}
    stubs = [name for name in STUB_MODULES if name.split(".")[0] in missing]
    blocked = {name: sys.modules[name] for name in stubs if name in sys.modules}  # e.g. None: imports blocked
    for name in stubs:
        sys.modules[name] = _stub(name)
    bytecode = sys.dont_write_bytecode
    sys.dont_write_bytecode = True
    sys.path.insert(0, root)
    try:
        modules = {name: importlib.import_module(name) for name in UPSTREAM_MODULES}
    finally:
        sys.path.remove(root)
        sys.dont_write_bytecode = bytecode
        for name in UPSTREAM_MODULES:
            sys.modules.pop(name, None)
        sys.modules.update(displaced)
        for name in stubs:
            sys.modules.pop(name, None)
        sys.modules.update(blocked)
    for name, module in modules.items():
        sys.modules[f"{PREFIX}.{name}"] = module
    return modules


# ---------------------------------------------------------------------------------------------- upstream subclasses
_CLASSES: dict[str, type] = {}


def stack_visual_encoder_class() -> type:
    """Upstream ``VisualEncoder`` (models.py:289-311) whose first convolution takes ``channels`` (3 x frame stack)."""
    if "encoder" not in _CLASSES:
        models = load_upstream()["models"]

        class StackVisualEncoder(models.VisualEncoder):
            def __init__(self, embedding_size, activation_function="relu", channels=9):
                super().__init__(embedding_size, activation_function)  # scripted after this __init__ (ScriptMeta)
                self.conv1 = nn.Conv2d(int(channels), 32, 4, stride=2)  # upstream: nn.Conv2d(3, 32, 4, stride=2)
                self.modules = [self.conv1, self.conv2, self.conv3, self.conv4]  # upstream's FreezeParameters list

        _CLASSES["encoder"] = StackVisualEncoder
    return _CLASSES["encoder"]


def replay_class() -> type:
    """``DreamerReplay``: upstream ``ExperienceReplay`` with memmap storage of our row contents (see module doc)."""
    if "replay" not in _CLASSES:
        _CLASSES["replay"] = _make_replay_class(load_upstream()["memory"].ExperienceReplay)
    return _CLASSES["replay"]


REPLAY_FIELDS = ("observations", "features", "proprio", "actions", "rewards", "nonterminals", "episode_step",
                 "global_index")


def _make_replay_class(base: type) -> type:
    class DreamerReplay(base):
        """Rows as memory.py:24-37 (row t = o_t, a_t, r_{t+1}, nonterminal_{t+1}; ``idx``/``full``/``steps``/
        ``episodes`` counters) with o_t = (64-px target frame ``observations``, feature atom ``features`` (frozen arms;
        the CNN's atom is the frame itself), ``proprio``) plus ``episode_step`` (stack rebuild) and ``global_index``
        (resume). ``_sample_idx`` (uniform chunk start, never across the write index: memory.py:39-45) and ``sample``
        (memory.py:60-70) are upstream's; a chunk is redrawn while it touches a row not written in this run's
        checkpointed history (rows of a crashed process after its checkpoint) or a stack frame already overwritten, or
        while it crosses an episode boundary. The last rule is the one deviation from upstream sampling (Codex review
        2026-10-28): upstream's 1000-step DMC episodes let about 10 % of 50-step chunks cross a reset; our 125-step
        Meta-World episodes would let about 39 % pair an episode's last action and reward with the next episode's reset
        observation and camera, so chunks stay inside one episode (125 >= the chunk length 50)."""

        def __init__(self, directory, size, frame_shape, atom_shape, atom_dtype, atom_stack, proprio_dim, action_dim,
                     bit_depth, device, mode="w+"):
            # memory.py:8-22 allocates fixed in-RAM layouts; the same counters over memmaps here
            self.device, self.size, self.bit_depth = device, int(size), int(bit_depth)
            self.symbolic_env = False
            self.idx, self.full, self.steps, self.episodes = 0, False, 0, 0  # memory.py:19-21
            self.frame_shape, self.atom_stack = tuple(frame_shape), int(atom_stack)
            self.pixel_atoms = atom_shape is None  # end-to-end CNN: the stacked atoms are the target frames
            self.atom_shape = self.frame_shape if self.pixel_atoms else tuple(atom_shape)
            self.directory = Path(directory)
            self.directory.mkdir(parents=True, exist_ok=True)
            size = self.size
            specs = {
                "observations": (np.uint8, (size, *self.frame_shape)),
                "features": None if self.pixel_atoms else (np.dtype(atom_dtype), (size, *self.atom_shape)),
                "proprio": (np.float32, (size, int(proprio_dim))),
                "actions": (np.float32, (size, int(action_dim))),
                "rewards": (np.float32, (size,)),
                "nonterminals": (np.float32, (size, 1)),
                "episode_step": (np.int32, (size,)),
                "global_index": (np.int64, (size,)),
            }
            for name, spec in specs.items():
                array = None if spec is None else np.memmap(self.directory / f"{name}.memmap", dtype=spec[0], mode=mode,
                                                            shape=spec[1])
                setattr(self, name, array)
            if mode == "w+":
                self.global_index[:] = -1
            from s4d.rl.replay import advise_random  # uniformly sampled rows: no readahead (s4d.rl.replay)

            advise_random(self.observations)

        # -------------------------------------------------------------- writing
        def append(self, frame, atom, proprio, action, reward, done, episode_step):
            """memory.py:24-37 with our row contents (o_t's frame / atom / proprio, a_t, r_{t+1}, done)."""
            i = self.idx
            self.observations[i] = frame
            if not self.pixel_atoms:
                self.features[i] = atom
            self.proprio[i] = proprio
            self.actions[i] = action
            self.rewards[i] = reward
            self.nonterminals[i] = not done
            self.episode_step[i] = int(episode_step)
            self.global_index[i] = self.steps  # published last
            self.idx = (self.idx + 1) % self.size
            self.full = self.full or self.idx == 0
            self.steps, self.episodes = self.steps + 1, self.episodes + (1 if done else 0)

        def nbytes(self) -> int:
            return sum(getattr(self, name).nbytes for name in REPLAY_FIELDS if getattr(self, name) is not None)

        def checkpoint(self) -> dict:
            for name in REPLAY_FIELDS:
                if getattr(self, name) is not None:
                    getattr(self, name).flush()
            return {"idx": self.idx, "full": self.full, "steps": self.steps, "episodes": self.episodes}

        def restore(self, meta: dict) -> None:
            self.idx, self.full = int(meta["idx"]), bool(meta["full"])
            self.steps, self.episodes = int(meta["steps"]), int(meta["episodes"])

        # -------------------------------------------------------------- sampling
        def _live(self, idxs: np.ndarray) -> bool:
            g = self.global_index[idxs]
            first = int(g[0])
            if first < max(0, self.steps - self.size) or int(g[-1]) >= self.steps:
                return False
            if not np.array_equal(g, first + np.arange(len(idxs))):
                return False
            start = int(self.episode_step[idxs[0]])  # inside one episode (see the class docstring)
            if not np.array_equal(self.episode_step[idxs], start + np.arange(len(idxs))):
                return False
            back = min(self.atom_stack - 1, int(self.episode_step[idxs[0]]))  # stack frames before the chunk
            if back:
                offsets = np.arange(back, 0, -1)
                if not np.array_equal(self.global_index[(idxs[0] - offsets) % self.size], first - offsets):
                    return False
            return True

        def _sample_idx(self, L):
            while True:
                idxs = super()._sample_idx(L)  # memory.py:39-45, unchanged
                if self._live(idxs):
                    return idxs

        def _retrieve_batch(self, idxs, n, L):
            """Time-major (L, n) batch as memory.py:47-58: [target frames (uint8; empty for the CNN, whose targets are
            the atom window's newest rows), atom window (L + S - 1, n, *atom) with S = ``atom_stack``, stack index
            (L, n, S) into the window's time axis (oldest first, the episode's first frame repeated), proprio (L, n,
            P), actions (L, n, A), rewards (L, n), nonterminals (L, n, 1)]. Bit-depth preprocessing is applied on the
            device by the agent (``preprocess_observation_``, as memory.py:483)."""
            S = self.atom_stack
            vec_idxs = idxs.transpose().reshape(-1)  # memory.py:48
            window = np.concatenate([(idxs[:, :1] - np.arange(S - 1, 0, -1)[None]) % self.size, idxs], axis=1)
            source = self.observations if self.pixel_atoms else self.features
            atoms = source[window.transpose().reshape(-1)].reshape(L + S - 1, n, *self.atom_shape)
            steps = self.episode_step[vec_idxs].reshape(L, n, 1).astype(np.int64)
            back = np.arange(S - 1, -1, -1)[None, None, :]  # frames before the newest, oldest first
            stack_index = np.arange(L)[:, None, None] + (S - 1) - np.minimum(back, steps)
            frames = (np.empty((0,), np.uint8) if self.pixel_atoms
                      else self.observations[vec_idxs].reshape(L, n, *self.frame_shape))
            return (
                frames,
                atoms,
                stack_index,
                self.proprio[vec_idxs].reshape(L, n, -1),
                self.actions[vec_idxs].reshape(L, n, -1),
                self.rewards[vec_idxs].reshape(L, n),
                self.nonterminals[vec_idxs].reshape(L, n, 1),
            )

    return DreamerReplay


# ---------------------------------------------------------------------------------------------- image helpers
def downsample(frames: torch.Tensor, size: int) -> torch.Tensor:
    """(..., 3, H, W) uint8 -> (..., 3, size, size) uint8 by area averaging, rounded (the stored replay frames and the
    CNN's acting input use this one function, so training and acting see identical frames)."""
    lead, (c, h, w) = frames.shape[:-3], frames.shape[-3:]
    if h % size or w % size:
        raise ValueError(f"{h}x{w} frames are not an integer multiple of {size} px")
    x = F.interpolate(frames.reshape(-1, c, h, w).float(), size=(size, size), mode="area")
    return x.round_().clamp_(0, 255).to(torch.uint8).reshape(*lead, c, size, size)


def family(cfg: dict) -> str:
    return "pixel" if str(cfg["vision"]["encoder_type"]) == "convnet" else "frozen"


# ---------------------------------------------------------------------------------------------- agent
class DreamerAgent:
    """Upstream models, optimisers and losses (main.py:184-262, 340-504) with the arm's encoder; the interface of
    ``s4d.rl.evaluate`` (``device``, ``use_pixels``, ``encoder.extract_cacheable_feature``, ``act``, ``train``,
    ``load_policy``, ``recurrent`` / ``begin_episodes``). ``learn=False`` builds only what acting needs."""

    recurrent = True

    def __init__(self, cfg: dict, action_dim: int, proprio_dim: int, device: torch.device, learn: bool = True):
        up = load_upstream()
        self.models, self.utils, self.env_utils = up["models"], up["utils"], up["env"]
        self.cfg, self.device = cfg, torch.device(device)
        a, t = cfg["agent"], cfg["train"]
        self.action_dim, self.proprio_dim = int(action_dim), int(proprio_dim)
        self.family = family(cfg)
        self.use_pixels = self.family == "pixel"
        self.frame_stack = int(cfg["env"]["frame_stack"])
        self.size, self.bit_depth = int(a["observation_size"]), int(a["bit_depth"])
        if self.size != 64:
            raise ValueError("upstream's visual encoder and decoder are built for 64x64 observations")
        self.belief_size, self.state_size = int(a["belief_size"]), int(a["state_size"])
        self.hidden_size, self.embedding_size = int(a["hidden_size"]), int(a["embedding_size"])
        self.action_noise = float(a["action_noise"])
        self.encoder = self  # evaluate.policy_inputs calls agent.encoder.extract_cacheable_feature
        dense, cnn = str(a["dense_activation_function"]), str(a["cnn_activation_function"])
        if self.use_pixels:
            self.features = None
            self.atom_shape, self.atom_dtype, self.atom_stack = None, np.dtype(np.uint8), self.frame_stack
            self.obs_encoder = stack_visual_encoder_class()(self.embedding_size, cnn, 3 * self.frame_stack)
        else:
            from s4d.rl.snerl_sac import FrozenFeatures

            self.features = FrozenFeatures(cfg, self.device)
            self.atom_shape, self.atom_dtype = tuple(self.features.atom_shape), np.dtype(self.features.atom_dtype)
            self.atom_stack = int(self.features.atom_frame_stack)
            self.feature_size = int(np.prod(self.atom_shape)) * self.atom_stack
            # main.py:203-205: Encoder(symbolic, observation_size, embedding_size, cnn activation)
            self.obs_encoder = self.models.Encoder(True, self.feature_size, self.embedding_size, cnn)
        self.obs_encoder = self.obs_encoder.to(self.device)
        # main.py:184-191 (the embedding carries the shared proprioception), main.py:206-208
        self.transition_model = self.models.TransitionModel(
            self.belief_size, self.state_size, self.action_dim, self.hidden_size, self.embedding_size + self.proprio_dim,
            dense).to(self.device)
        self.actor_model = self.models.ActorModel(
            self.belief_size, self.state_size, self.hidden_size, self.action_dim, dense).to(self.device)
        self.learn = bool(learn)
        if self.learn:
            # main.py:192-202 (the VISUAL decoder for every arm: same 64-px target), main.py:209-211
            self.observation_model = self.models.ObservationModel(
                False, (3, self.size, self.size), self.belief_size, self.state_size, self.embedding_size,
                cnn).to(self.device)
            self.reward_model = self.models.RewardModel(
                self.belief_size, self.state_size, self.hidden_size, dense).to(self.device)
            self.value_model = self.models.ValueModel(
                self.belief_size, self.state_size, self.hidden_size, dense).to(self.device)
            self.param_list = (list(self.transition_model.parameters()) + list(self.observation_model.parameters())
                               + list(self.reward_model.parameters()) + list(self.obs_encoder.parameters()))  # :212-217
            eps = float(a["adam_epsilon"])
            self.model_optimizer = torch.optim.Adam(self.param_list, lr=float(a["model_learning_rate"]), eps=eps)
            self.actor_optimizer = torch.optim.Adam(self.actor_model.parameters(), lr=float(a["actor_learning_rate"]),
                                                    eps=eps)
            self.value_optimizer = torch.optim.Adam(self.value_model.parameters(), lr=float(a["value_learning_rate"]),
                                                    eps=eps)  # main.py:221-233
            self.batch_size, self.chunk_size = int(t["batch_size"]), int(t["chunk_size"])
            self.global_kl_beta = float(a["global_kl_beta"])
            self.global_prior = Normal(torch.zeros(self.batch_size, self.state_size, device=self.device),
                                       torch.ones(self.batch_size, self.state_size, device=self.device))  # :258-261
            self.free_nats = torch.full((1,), float(a["free_nats"]), device=self.device)  # main.py:262
            self.grad_clip_norm = float(a["grad_clip_norm"])
            self.planning_horizon, self.discount = int(a["planning_horizon"]), float(a["discount"])
            self.disclam = float(a["disclam"])
        self.begin_episodes(1)

    # ------------------------------------------------------------------ observations
    @torch.no_grad()
    def extract_cacheable_feature(self, obs):
        """Batched env observation (on the device) -> the frozen backbone's cached feature (frozen arms only)."""
        if self.features is None:
            raise RuntimeError("the end-to-end CNN arm consumes pixels directly")
        return self.features(obs)

    def preprocess(self, frames: torch.Tensor) -> torch.Tensor:
        """uint8 [0, 255] -> float [-0.5, 0.5] at the bit depth, plus dequantisation noise (env.py:48-54)."""
        x = frames.float()
        self.env_utils.preprocess_observation_(x, self.bit_depth)
        return x

    def target_frame(self, frame: np.ndarray) -> np.ndarray:
        """The newest 128-px frame of the first camera (3, H, W) uint8 -> the stored 64-px row frame."""
        return downsample(torch.from_numpy(np.ascontiguousarray(frame)), self.size).numpy()

    def pixel_stack(self, pixels: torch.Tensor) -> torch.Tensor:
        """(N, 3S, H, W) uint8 channel-stacked history -> (N, S, 3, 64, 64) uint8 (the replay's row frames)."""
        n, _, h, w = pixels.shape
        return downsample(pixels.reshape(n, self.frame_stack, 3, h, w), self.size)

    def encoder_input(self, inputs) -> torch.Tensor:
        """Acting: policy inputs (pixels or cached features) -> the upstream encoder's input."""
        if self.use_pixels:
            stack = self.pixel_stack(inputs)
            return self.preprocess(stack.reshape(len(stack), -1, self.size, self.size))
        return inputs.reshape(len(inputs), -1).float()

    def embed(self, enc_in: torch.Tensor, proprio: torch.Tensor) -> torch.Tensor:
        return torch.cat([self.obs_encoder(enc_in), proprio], dim=-1)

    # ------------------------------------------------------------------ acting (main.py:266-289)
    def begin_episodes(self, count: int) -> None:
        """Fresh recurrent state for ``count`` episodes: zero belief, state and previous action (main.py:533-537)."""
        count = int(count)
        self.belief = torch.zeros(count, self.belief_size, device=self.device)
        self.posterior_state = torch.zeros(count, self.state_size, device=self.device)
        self.prev_action = torch.zeros(count, self.action_dim, device=self.device)

    @torch.no_grad()
    def act(self, inputs, proprio, step: int, eval_mode: bool, index=None) -> np.ndarray:
        """One step for the episodes at positions ``index`` (default: all, in order) of the current
        ``begin_episodes`` set; the others' state is untouched. Evaluation: ``get_action(det=True)``; training: a policy
        sample plus N(0, action_noise), clamped to [-1, 1] (main.py:277-285)."""
        rows = torch.arange(len(inputs), device=self.device) if index is None else torch.as_tensor(
            list(index), dtype=torch.long, device=self.device)
        if len(rows) != len(inputs):
            raise ValueError(f"{len(inputs)} observations for {len(rows)} episode positions")
        proprio = torch.as_tensor(np.asarray(proprio), dtype=torch.float32, device=self.device).reshape(len(rows), -1)
        embedding = self.embed(self.encoder_input(inputs), proprio)
        belief, _, _, _, posterior_state, _, _ = self.transition_model(
            self.posterior_state[rows], self.prev_action[rows].unsqueeze(dim=0), self.belief[rows],
            embedding.unsqueeze(dim=0))  # main.py:271-276
        belief, posterior_state = belief.squeeze(dim=0), posterior_state.squeeze(dim=0)
        action = self.actor_model.get_action(belief, posterior_state, det=bool(eval_mode))  # main.py:278
        if not eval_mode:
            action = torch.clamp(Normal(action, self.action_noise).rsample(), -1, 1)  # main.py:281-284
        self.belief[rows], self.posterior_state[rows], self.prev_action[rows] = belief, posterior_state, action
        return action.cpu().numpy()

    def train(self, training: bool = True) -> None:
        for module in self._networks().values():
            module.train(training)

    # ------------------------------------------------------------------ learning (main.py:340-509)
    def make_replay(self, directory: Path, mode: str = "w+"):
        return replay_class()(
            directory, int(self.cfg["train"]["replay_size"]), (3, self.size, self.size), self.atom_shape,
            self.atom_dtype, self.atom_stack, self.proprio_dim, self.action_dim, self.bit_depth, self.device, mode)

    def _batch_inputs(self, frames, atoms, stack_index):
        """Encoder inputs (L, B, ...) and decoder targets (L, B, 3, 64, 64) of a sampled chunk batch."""
        S, L, B = self.atom_stack, stack_index.shape[0], stack_index.shape[1]
        if self.use_pixels:
            atoms = self.preprocess(atoms)  # one dequantisation draw for the encoder input and the target
            targets = atoms[S - 1:]
        else:
            targets = self.preprocess(frames)
        x = atoms if S == 1 else atoms[stack_index, torch.arange(B, device=atoms.device)[None, :, None]]
        if self.use_pixels:
            return x.reshape(L, B, 3 * S, self.size, self.size), targets
        return x.reshape(L, B, -1).float(), targets

    def update(self, replay) -> dict[str, float]:
        """One iteration of main.py's collect_interval loop: world model, then actor, then value."""
        bottle, utils = self.models.bottle, self.utils
        frames, atoms, stack_index, proprio, actions, rewards, nonterminals = replay.sample(
            self.batch_size, self.chunk_size)  # main.py:342-344
        enc_in, targets = self._batch_inputs(frames, atoms, stack_index)
        init_belief = torch.zeros(self.batch_size, self.belief_size, device=self.device)
        init_state = torch.zeros(self.batch_size, self.state_size, device=self.device)  # main.py:346-348
        embeddings = torch.cat([bottle(self.obs_encoder, (enc_in[1:],)), proprio[1:]], dim=2)
        (beliefs, prior_states, prior_means, prior_std_devs, posterior_states, posterior_means,
         posterior_std_devs) = self.transition_model(
            init_state, actions[:-1], init_belief, embeddings, nonterminals[:-1])  # main.py:350-360
        observation_loss = (
            F.mse_loss(bottle(self.observation_model, (beliefs, posterior_states)), targets[1:], reduction="none")
            .sum(dim=(2, 3, 4)).mean(dim=(0, 1)))  # main.py:369-374
        reward_loss = F.mse_loss(bottle(self.reward_model, (beliefs, posterior_states)), rewards[:-1],
                                 reduction="none").mean(dim=(0, 1))  # main.py:378-381
        div = kl_divergence(Normal(posterior_means, posterior_std_devs), Normal(prior_means, prior_std_devs)).sum(dim=2)
        kl_loss = torch.max(div, self.free_nats).mean(dim=(0, 1))  # main.py:383-386
        if self.global_kl_beta != 0:  # main.py:387-390
            kl_loss = kl_loss + self.global_kl_beta * kl_divergence(
                Normal(posterior_means, posterior_std_devs), self.global_prior).sum(dim=2).mean(dim=(0, 1))
        model_loss = observation_loss + reward_loss + kl_loss  # main.py:462
        self.model_optimizer.zero_grad()
        model_loss.backward()
        nn.utils.clip_grad_norm_(self.param_list, self.grad_clip_norm, norm_type=2)
        self.model_optimizer.step()  # main.py:464-467

        model_modules = (self.transition_model.modules + self.obs_encoder.modules + self.observation_model.modules
                         + self.reward_model.modules)  # main.py:337
        with torch.no_grad():
            actor_states, actor_beliefs = posterior_states.detach(), beliefs.detach()  # main.py:470-472
        with utils.FreezeParameters(model_modules):
            imagination_traj = utils.imagine_ahead(actor_states, actor_beliefs, self.actor_model, self.transition_model,
                                                   self.planning_horizon)  # main.py:473-476
        imged_beliefs, imged_prior_states, _, _ = imagination_traj
        with utils.FreezeParameters(model_modules + self.value_model.modules):
            imged_reward = bottle(self.reward_model, (imged_beliefs, imged_prior_states))
            value_pred = bottle(self.value_model, (imged_beliefs, imged_prior_states))  # main.py:478-480
        returns = utils.lambda_return(imged_reward, value_pred, bootstrap=value_pred[-1], discount=self.discount,
                                      lambda_=self.disclam)  # main.py:481-483
        actor_loss = -torch.mean(returns)
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        nn.utils.clip_grad_norm_(self.actor_model.parameters(), self.grad_clip_norm, norm_type=2)
        self.actor_optimizer.step()  # main.py:484-489

        with torch.no_grad():
            value_beliefs, value_prior_states = imged_beliefs.detach(), imged_prior_states.detach()
            target_return = returns.detach()  # main.py:492-495
        value_dist = Normal(bottle(self.value_model, (value_beliefs, value_prior_states)), 1)
        value_loss = -value_dist.log_prob(target_return).mean(dim=(0, 1))  # main.py:496-499
        self.value_optimizer.zero_grad()
        value_loss.backward()
        nn.utils.clip_grad_norm_(self.value_model.parameters(), self.grad_clip_norm, norm_type=2)
        self.value_optimizer.step()  # main.py:500-504
        return {"observation_loss": observation_loss.item(), "reward_loss": reward_loss.item(),
                "kl_loss": kl_loss.item(), "actor_loss": actor_loss.item(), "value_loss": value_loss.item(),
                "imagined_return": returns.mean().item()}  # main.py:506-509

    # ------------------------------------------------------------------ state
    def _networks(self) -> dict[str, nn.Module]:
        out = {"encoder": self.obs_encoder, "transition_model": self.transition_model, "actor_model": self.actor_model}
        if self.learn:
            out |= {"observation_model": self.observation_model, "reward_model": self.reward_model,
                    "value_model": self.value_model}
        return out

    def _modules(self) -> dict[str, Any]:
        out: dict[str, Any] = dict(self._networks())  # the names of main.py:668-679
        if self.learn:
            out |= {"model_optimizer": self.model_optimizer, "actor_optimizer": self.actor_optimizer,
                    "value_optimizer": self.value_optimizer}
        return out

    def state_dict(self) -> dict:
        return {k: m.state_dict() for k, m in self._modules().items()}

    def load_state_dict(self, state: dict) -> None:
        for k, m in self._modules().items():
            m.load_state_dict(state[k])

    def policy_state(self) -> dict:
        """Snapshot policy: ``world_model`` = what acting needs of the world model (encoder + transition model, one
        flat state dict with ``encoder.`` / ``transition_model.`` prefixes), ``actor`` = the actor's state dict."""
        world = {f"encoder.{k}": v for k, v in self.obs_encoder.state_dict().items()}
        world |= {f"transition_model.{k}": v for k, v in self.transition_model.state_dict().items()}
        return {"world_model": world, "actor": self.actor_model.state_dict()}

    def load_policy(self, policy: dict) -> None:
        for prefix, module in (("encoder.", self.obs_encoder), ("transition_model.", self.transition_model)):
            module.load_state_dict({k[len(prefix):]: v for k, v in policy["world_model"].items() if k.startswith(prefix)})
        self.actor_model.load_state_dict(policy["actor"])


class DreamerPolicy(DreamerAgent):
    """Evaluation policy (``s4d.rl.algorithms.evaluation_policy``): acting modules only."""

    def __init__(self, cfg: dict, action_dim: int, proprio_dim: int, device: torch.device):
        super().__init__(cfg, action_dim, proprio_dim, device, learn=False)
