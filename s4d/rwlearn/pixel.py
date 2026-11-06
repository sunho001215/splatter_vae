"""End-to-end CNN + FQL (condition ``e2e_cnn``; user decision 2026-11-06 ~11:20): a RANDOMLY INITIALISED CNN encoder
trained JOINTLY with FQL from the offline data (never frozen), with the official FQL pixel handling as the reference
(github.com/seohongpark/fql @ e8cd16e; line numbers below are of that commit):

* encoder: ``utils/encoders.py`` ``ImpalaEncoder`` as ``encoder_modules['impala_small']`` (``functools.partial(
  ImpalaEncoder, num_blocks=1)``, :106): stack sizes (16, 32, 32), width 1, no dropout, no layer norm, MLP (512,)
  with ``activate_final=True`` (:60-100). Each ``ResnetStack`` (:10-57): 3x3 conv (stride 1, SAME, Xavier-uniform
  kernel, zero bias) -> 3x3 max-pool stride 2 SAME -> ``num_blocks`` residual blocks (relu, conv, relu, conv, + input);
  after the stacks relu, flatten (H, W, C order), MLP. Input normalisation ``x.astype(float32) / 255.0`` (:84), nothing
  else (no mean / std).
* one encoder per network (``agents/fql.py`` :196-202: critic, actor_bc_flow, actor_onestep_flow); the target critic
  is a copy of the critic including its encoder (:226, :241-242) and is Polyak-updated with it; the BC-flow encoder is
  shared by the BC loss and the Euler flow target (:162-163, :230-232). Every gradient reaches the encoders exactly as in
  the official losses (``s4d.rwlearn.fql``): critic encoder through the critic loss, the actors' encoders through the
  BC-flow and distillation + Q losses; the critic inside the actor's Q term and the flow target are stop-gradient.
* frame stacking (``utils/datasets.py`` :71-85, ``--frame_stack=3`` in the README's pixel commands): observation =
  ``[ob[t-2], ob[t-1], ob[t]]`` concatenated on the CHANNEL axis, padded with the first frame of the episode;
  next observation = ``[ob[t-1], ob[t], ob[t+1]]``. Here the history is the same 3-step history every frozen-encoder
  condition uses (``s4d.rwlearn.fql_data.FRAME_STACK``), padded with the first frame of the recording run.
* augmentation (``utils/datasets.py`` :86-89, :100-110, ``--p_aug=0.5``): with probability ``p_aug`` per BATCH,
  random-crop every 4-D (image) array: edge padding 3 px, per-sample offsets in [0, 6] x [0, 6], the SAME offsets for
  a sample's observation and next observation; set on the training AND validation datasets (``main.py`` :85-88).

Matched with the other FQL conditions (docs/realworld_fql.md): task data, episode split, action contract, rewards,
10-d proprioception (concatenated to the encoder output, as the frozen conditions concatenate it to their adapter
output), 3-step history, viewpoint protocol (one random camera per sample, fixed through the history and the paired
next observation), the compact store's 128 x 128 centre-crop RGB, seeds, alpha 10 and the 1M-step budget.
Documented differences from the official visual OGBench runs: 128 px instead of 64 px images (the crop padding stays
the official 3 px), proprioception input, alpha 10 (user decision; official visual tasks 100-300), 1M steps
(official visual runs 500k).
"""

from __future__ import annotations

import json

import h5py
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from s4d.rwlearn import actions as A
from s4d.rwlearn.fql import MLP

FRAME_STACK = 3
P_AUG, AUG_PADDING = 0.5, 3
ACTION_CLIP_EPS = 1e-5  # envs/env_utils.py :138-146 (make_env_and_datasets(action_clip_eps=1e-5))
IMPALA_SMALL = {"stack_sizes": (16, 32, 32), "num_blocks": 1, "mlp_hidden_dims": (512,)}


def max_pool_same(x: torch.Tensor, k: int = 3, s: int = 2) -> torch.Tensor:
    """``flax.linen.max_pool(window (k, k), strides (s, s), padding='SAME')`` on NCHW: -inf padding with the extra
    element at the high end (``lax.padtype_to_pads``)."""
    pads = []
    for n in (x.shape[-1], x.shape[-2]):  # F.pad order: last dim first
        total = max((-(-n // s) - 1) * s + k - n, 0)
        pads += [total // 2, total - total // 2]
    return F.max_pool2d(F.pad(x, pads, value=float("-inf")), k, s)


def same_out(n: int, s: int = 2) -> int:
    return -(-n // s)


class ResnetStack(nn.Module):
    """``utils/encoders.py::ResnetStack`` (NCHW)."""

    def __init__(self, in_channels: int, features: int, num_blocks: int):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, features, 3, padding=1)
        self.blocks = nn.ModuleList(nn.ModuleList([nn.Conv2d(features, features, 3, padding=1),
                                                   nn.Conv2d(features, features, 3, padding=1)])
                                    for _ in range(num_blocks))
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight)  # flax xavier_uniform on (kh, kw, in, out): same fans
                nn.init.zeros_(m.bias)

    def forward(self, x):
        x = max_pool_same(self.conv(x))
        for a, b in self.blocks:
            x = x + b(F.relu(a(F.relu(x))))
        return x

    def convs(self):
        yield self.conv
        for a, b in self.blocks:
            yield a
            yield b


class ImpalaEncoder(nn.Module):
    """``ImpalaEncoder`` (official ``impala_small`` by default) on a channel-stacked uint8 image ``(B, H, W, C)``
    (NHWC, as the official datasets store it), plus optional proprioception concatenated to its output.

    ``forward`` takes the image tensor, or a dict with ``pixels`` (and ``proprio`` when ``proprio_dim`` > 0)."""

    def __init__(self, in_channels: int, image_hw=(128, 128), stack_sizes=(16, 32, 32), num_blocks: int = 1,
                 mlp_hidden_dims=(512,), proprio_dim: int = 0):
        super().__init__()
        chans = [int(in_channels), *[int(c) for c in stack_sizes]]
        self.stacks = nn.ModuleList(ResnetStack(a, b, int(num_blocks)) for a, b in zip(chans[:-1], chans[1:]))
        h, w = (int(v) for v in image_hw)
        for _ in stack_sizes:
            h, w = same_out(h), same_out(w)
        self.flat_dim = h * w * chans[-1]
        self.mlp = MLP(self.flat_dim, tuple(mlp_hidden_dims), activate_final=True, layer_norm=False)
        self.proprio_dim = int(proprio_dim)
        self.out_dim = int(mlp_hidden_dims[-1]) + self.proprio_dim
        self.config = {"in_channels": int(in_channels), "image_hw": [int(v) for v in image_hw],
                       "stack_sizes": [int(c) for c in stack_sizes], "num_blocks": int(num_blocks),
                       "mlp_hidden_dims": [int(d) for d in mlp_hidden_dims], "proprio_dim": self.proprio_dim}

    def forward(self, obs):
        pixels = obs["pixels"] if isinstance(obs, dict) else obs
        x = pixels.to(torch.float32) / 255.0  # :84
        x = x.permute(0, 3, 1, 2)  # NHWC -> NCHW
        for stack in self.stacks:
            x = stack(x)
        x = F.relu(x)  # :93
        x = x.permute(0, 2, 3, 1).reshape(x.shape[0], -1)  # flatten in the official (H, W, C) order (:96)
        x = self.mlp(x)  # MLP((512,), activate_final=True) (:98)
        if self.proprio_dim:
            x = torch.cat([x, obs["proprio"].to(x.dtype)], dim=-1)
        return x

    def official_slots(self, prefix: str):
        """(official flat name, tensor, kind) of every parameter, for loading / exporting official parameters."""
        for i, stack in enumerate(self.stacks):
            for j, conv in enumerate(stack.convs()):
                yield f"{prefix}/stack_blocks_{i}/Conv_{j}/kernel", conv.weight, "conv_kernel"
                yield f"{prefix}/stack_blocks_{i}/Conv_{j}/bias", conv.bias, "vector"
        for j, lin in enumerate(self.mlp.linears):
            yield f"{prefix}/MLP_0/Dense_{j}/kernel", lin.weight, "kernel"
            yield f"{prefix}/MLP_0/Dense_{j}/bias", lin.bias, "vector"


def stack_frames(frames: torch.Tensor) -> torch.Tensor:
    """(B, T, H, W, C) oldest first -> (B, H, W, T * C): the official channel-axis concatenation (datasets.py :84)."""
    B, T, H, W, C = frames.shape
    return frames.permute(0, 2, 3, 1, 4).reshape(B, H, W, T * C)


def random_crop(images: torch.Tensor, offsets: torch.Tensor, padding: int = AUG_PADDING) -> torch.Tensor:
    """``batched_random_crop`` (datasets.py :15-31): edge-pad ``padding`` px, crop the original size at per-sample
    (row, col) ``offsets`` in [0, 2 * padding]. Edge padding = index clamping, so uint8 stays exact."""
    B, H, W = images.shape[:3]
    dev = images.device
    rows = (torch.arange(H, device=dev)[None] + offsets[:, :1].to(dev) - padding).clamp(0, H - 1)
    cols = (torch.arange(W, device=dev)[None] + offsets[:, 1:2].to(dev) - padding).clamp(0, W - 1)
    return images[torch.arange(B, device=dev)[:, None, None], rows[:, :, None], cols[:, None, :]]


def augment(batch: dict, generator: torch.Generator, p_aug: float = P_AUG, padding: int = AUG_PADDING) -> dict:
    """``Dataset.sample``'s augmentation (:86-89, :100-110): ONE coin per batch; when it lands, every image of the
    observations and next observations is cropped, a sample's two images with the same offsets."""
    if p_aug is None or not float(torch.rand(1, generator=generator)) < p_aug:
        return batch
    n = batch["actions"].shape[0]
    offsets = torch.randint(0, 2 * padding + 1, (n, 2), generator=generator)
    for key in ("observations", "next_observations"):
        obs = dict(batch[key])
        obs["pixels"] = random_crop(obs["pixels"], offsets, padding)
        batch[key] = obs
    return batch


class OfflinePixelTransitions:
    """All transitions of an ``e2e_cnn`` FQL dataset (``s4d.rwlearn.fql_data.build_fql_dataset``) with the RGB frames
    of their camera timelines loaded ONCE per camera and frame from the training store (in memory, never written);
    observation histories and next observations are index arrays into those frames (no window copies)."""

    def __init__(self, path, store, episodes: list[str] | None = None, device="cpu", frames_device=None,
                 p_aug: float | None = P_AUG, action_clip_eps: float | None = ACTION_CLIP_EPS):
        from s4d.rwlearn.fql_data import FORMAT
        from s4d.rwlearn.store import open_store
        from s4d.rwlearn.transitions import history_indices

        self.path, self.p_aug = str(path), p_aug
        store = open_store(store)
        frames, prop, rows = [], [], []
        base = 0
        with h5py.File(self.path, "r") as f:
            if f.attrs.get("format") != FORMAT:
                raise ValueError(f"{path}: not a {FORMAT} file")
            self.attrs = {k: (v.decode() if isinstance(v, bytes) else v) for k, v in f.attrs.items()}
            if self.attrs.get("condition") != "e2e_cnn":
                raise ValueError(f"{path}: condition {self.attrs.get('condition')!r}, not e2e_cnn")
            self.contract = A.ActionContract.from_json(json.loads(self.attrs["contract"]))
            eps = episodes if episodes is not None else json.loads(self.attrs["episodes"])
            self.timelines = []
            for ep in eps:
                for name, g in f["episodes"][ep].items():
                    cams = json.loads(g.attrs["cameras"])
                    if len(cams) != 1:
                        raise ValueError(f"{ep}/{name}: e2e_cnn timelines are single-camera")
                    native = g["frames"][:][:, 0]
                    t_store = np.asarray(store.timestamps(ep, cams[0]), np.int64)
                    if native.max(initial=-1) >= len(t_store) or not np.array_equal(t_store[native], g["times"][:]):
                        raise ValueError(f"{ep}/{name}: store frames are not the dataset's frames")
                    run, ok = g["run"][:], g["transition_ok"][:].astype(bool)
                    ks = np.flatnonzero(ok)
                    rows.append({"obs": history_indices(run, ks, FRAME_STACK) + base,
                                 "next": history_indices(run, ks + 1, FRAME_STACK) + base,
                                 "action": g["action"][:][ks], "reward": g["reward"][:][ks], "mask": g["mask"][:][ks],
                                 "obs_k": ks + base, "next_k": ks + 1 + base})
                    frames.append(np.asarray(store.rgb(ep, cams[0], native), np.uint8))
                    prop.append(g["proprio"][:])
                    self.timelines.append((ep, name, base, len(native)))
                    base += len(native)
        if not rows:
            raise ValueError("no transitions")
        cat = {k: np.concatenate([r[k] for r in rows]) for k in rows[0]}
        dev = torch.device(device)
        self.frames = torch.from_numpy(np.concatenate(frames)).to(torch.device(frames_device or device))
        self.image_hw = tuple(self.frames.shape[1:3])
        self.proprio = torch.from_numpy(np.concatenate(prop)).to(dev)
        self.obs_idx = torch.from_numpy(cat["obs"]).to(dev)
        self.next_idx = torch.from_numpy(cat["next"]).to(dev)
        self.obs_k = torch.from_numpy(cat["obs_k"]).to(dev)
        self.next_k = torch.from_numpy(cat["next_k"]).to(dev)
        actions = cat["action"]
        if action_clip_eps is not None:
            actions = np.clip(actions, -1 + action_clip_eps, 1 - action_clip_eps)
        self.actions = torch.from_numpy(actions).to(dev).float()
        self.rewards = torch.from_numpy(cat["reward"]).to(dev).float()
        self.masks = torch.from_numpy(cat["mask"]).to(dev).float()
        self.size = len(self.actions)
        self.sequence = False
        self.feat_dim = None
        self.device = dev

    def _obs(self, hist, k):
        px = self.frames[hist.to(self.frames.device)].to(self.device)
        return {"pixels": stack_frames(px), "proprio": self.proprio[k]}

    def batch(self, idx: torch.Tensor) -> dict:
        idx = idx.to(self.device)
        return {"observations": self._obs(self.obs_idx[idx], self.obs_k[idx]), "actions": self.actions[idx],
                "rewards": self.rewards[idx], "masks": self.masks[idx],
                "next_observations": self._obs(self.next_idx[idx], self.next_k[idx])}

    def sample(self, batch_size: int, generator: torch.Generator, aug_generator: torch.Generator | None = None) -> dict:
        """Uniform over all (camera timeline, step) transitions (official ``Dataset.sample``), then the official
        augmentation drawn from ``aug_generator`` (default: ``generator``)."""
        batch = self.batch(torch.randint(self.size, (batch_size,), generator=generator))
        return augment(batch, aug_generator or generator, self.p_aug)

    def encoder_factory(self, **kw):
        cfg = {**IMPALA_SMALL, **kw}
        return lambda: ImpalaEncoder(3 * FRAME_STACK, self.image_hw, proprio_dim=A.PROPRIO_DIM, **cfg)


def encoder_from_config(cfg: dict) -> ImpalaEncoder:
    c = dict(cfg)
    return ImpalaEncoder(c["in_channels"], c["image_hw"], c["stack_sizes"], c["num_blocks"], c["mlp_hidden_dims"],
                         c["proprio_dim"])

