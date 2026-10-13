"""The official DynaMo optimisation step, learning-rate schedule, evaluation and epoch logging (``train.py``).

``train.py::Trainer`` cannot run here (hydra, accelerate and the env workspaces; its resume is broken), so its
logic is replicated line by line for the official single-process fp32 launch (``python3 train.py``, no accelerate
config). There ``accelerator.autocast()`` is a null context, ``accelerator.backward(loss, retain_graph=True)`` is
``loss.backward(retain_graph=True)``, ``accelerator.clip_grad_norm_`` is ``torch.nn.utils.clip_grad_norm_`` and
``gather_for_metrics`` is the identity. ``tests/test_baselines_dynamo.py`` runs the verbatim ``Trainer.train`` /
``adjust_lr`` source against these functions.
"""

from __future__ import annotations

import os
import random
from collections import OrderedDict

import numpy as np
import torch

from s4d.baselines.dynamo.model import DynaMoModules


def set_seed_everywhere(seed: int) -> None:
    """``utils/__init__.py:87-94`` (called at train.py:36): seeds, deterministic algorithms, cuBLAS workspace."""
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.use_deterministic_algorithms(True)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"


def epoch_lr(train_cfg: dict, epoch: int) -> float:
    """``Trainer.adjust_lr`` (train.py:371-379): half-cycle cosine after a linear warmup, per epoch (NumPy's cos, as
    upstream, so the values are bit-identical)."""
    ssl_lr, warmup, epochs = float(train_cfg["ssl_lr"]), int(train_cfg["warmup_epochs"]), int(train_cfg["num_epochs"])
    if epoch < warmup:
        return ssl_lr * epoch / warmup
    return ssl_lr * 0.5 * (1.0 + np.cos(np.pi * (epoch - warmup) / (epochs - warmup)))


def start_epoch(modules: DynaMoModules, train_cfg: dict, epoch: int) -> float | None:
    """Top of ``Trainer.train`` (train.py:170-173). Only the encoder and projector optimisers are scheduled
    (train.py:380-383); the forward-dynamics optimiser keeps ``ssl_lr`` throughout, as upstream."""
    lr = None
    if bool(train_cfg["use_lr_scheduling"]):
        lr = epoch_lr(train_cfg, epoch)
        for optim in (modules.encoder_optim, modules.projector_optim):
            for group in optim.param_groups:
                group["lr"] = lr
    modules.ssl.adjust_beta(epoch, int(train_cfg["num_epochs"]))
    return lr


def train_step(modules: DynaMoModules, obs: torch.Tensor, clip_grad_norm: float | None) -> dict[str, float]:
    """One iteration of the ``Trainer.train`` loop (train.py:180-209) on ``obs`` (N, T, V, 3, H, W) in [0, 1]."""
    _, _, ssl_loss, components = modules.ssl.forward(obs)
    ssl_loss.backward(retain_graph=True)
    if clip_grad_norm:
        torch.nn.utils.clip_grad_norm_(modules.encoder.parameters(), clip_grad_norm)
        torch.nn.utils.clip_grad_norm_(modules.projector.parameters(), clip_grad_norm)
        torch.nn.utils.clip_grad_norm_(modules.ssl.parameters(), clip_grad_norm)
    modules.encoder_optim.step()
    modules.projector_optim.step()
    modules.ssl.step()  # forward-dynamics AdamW step + zero_grad (+ EMA update when ema_beta is set)
    modules.encoder_optim.zero_grad(set_to_none=True)
    modules.projector_optim.zero_grad(set_to_none=True)
    return {k: float(v.detach().cpu().item()) for k, v in components.items()}


class EpochSampler(torch.utils.data.Sampler):
    """The shuffled window order of one epoch, resumable mid-epoch.

    Upstream shuffles with ``DataLoader(shuffle=True)`` (train.py:163), i.e. a fresh random permutation each epoch
    from the global torch RNG. Here each epoch's permutation comes from a generator seeded by ``(seed, epoch)``, so a
    resumed run continues the exact order from ``start``; it is equally a uniform random permutation per epoch.
    """

    def __init__(self, size: int, seed: int):
        self.size, self.seed, self.epoch, self.start = int(size), int(seed), 0, 0

    def set_epoch(self, epoch: int, start: int = 0) -> None:
        self.epoch, self.start = int(epoch), int(start)

    def permutation(self, epoch: int) -> torch.Tensor:
        seed = int(np.random.SeedSequence([self.seed, int(epoch)]).generate_state(1, np.uint32)[0])
        return torch.randperm(self.size, generator=torch.Generator().manual_seed(seed))

    def __iter__(self):
        return iter(self.permutation(self.epoch)[self.start :].tolist())

    def __len__(self) -> int:
        return max(self.size - self.start, 0)


class EpochLog:
    """``Trainer.log_append`` / ``flush_log`` (train.py:339-369): length-weighted means per epoch."""

    def __init__(self, sums: dict | None = None):
        self.components: OrderedDict[str, tuple[int, float]] = OrderedDict(
            (k, (int(c), float(s))) for k, (c, s) in (sums or {}).items()
        )

    def append(self, log_key: str, length: int, values: dict) -> None:
        for key, value in values.items():
            if isinstance(value, torch.Tensor):
                value = value.detach().cpu().item()
            name = f"{log_key}/{key}"
            count, total = self.components.get(name, (0, 0.0))
            self.components[name] = (count + length, total + length * value)

    def flush(self) -> dict[str, float]:
        out = {key: total / count for key, (count, total) in self.components.items()}
        self.components = OrderedDict()
        return out

    def state(self) -> dict:
        return {k: list(v) for k, v in self.components.items()}


@torch.no_grad()
def evaluate(modules: DynaMoModules, batches, log: EpochLog, to_obs) -> None:
    """``Trainer.eval`` (train.py:222-273) on the validation windows: encoder and projector in eval mode (BatchNorm
    running statistics), the SSL module (forward dynamics, dropout 0) left as is, as upstream. The env-specific
    ``workspace.run_offline_eval`` (train.py:212-220) has no Meta-World counterpart and is not run."""
    states = [modules.encoder.training, modules.projector.training]
    modules.encoder.train(False)
    modules.projector.train(False)
    try:
        for batch in batches:
            obs = to_obs(batch)
            obs_enc, obs_proj, _, components = modules.ssl.forward(obs)
            log.append("ssl_eval", len(obs), components)
            flat_enc = obs_enc.flatten(0, 2)  # "N T V E -> (N T V) E"
            log.append(
                "metrics",
                len(flat_enc),
                {"obs_enc_mean_std": flat_enc.std(dim=0).mean(), "obs_enc_mean_norm": flat_enc.norm(dim=-1).mean()},
            )
            flat_proj = obs_proj.flatten(0, 2)
            log.append(
                "metrics",
                len(flat_proj),
                {"obs_proj_mean_std": flat_proj.std(dim=0).mean(), "obs_proj_mean_norm": flat_proj.norm(dim=-1).mean()},
            )
    finally:
        modules.encoder.train(states[0])
        modules.projector.train(states[1])
