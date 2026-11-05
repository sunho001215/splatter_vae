"""Toy offline task for the FQL port's checks (identical to ``toy_bandit`` in
runs/setup/tools/realworld_fql_20261106/fql_reference_jax.py, which runs in a separate JAX environment; the reference
output records the dataset's sha256 so the torch side can verify it rebuilt the same data)."""

from __future__ import annotations

import hashlib

import numpy as np


def toy_bandit(n: int, seed: int) -> dict:
    """One-step bandit with a bimodal behaviour policy: context ~ N(0, I_4); the behaviour picks mode +1 or -1 with
    probability 1/2 and acts m * (0.5, 0.5) + 0.1 N(0, I_2) (clipped to [-1, 1]); reward -|a - (0.5, 0.5)|^2, so the
    +1 mode is better; every transition terminal (mask 0). BC samples both modes, FQL should prefer +1."""
    rng = np.random.default_rng(seed)
    obs = rng.normal(size=(n, 4)).astype(np.float32)
    mode = np.where(rng.random(n) < 0.5, 1.0, -1.0)[:, None]
    act = np.clip(mode * 0.5 + 0.1 * rng.normal(size=(n, 2)), -1, 1).astype(np.float32)
    rew = -np.sum((act - 0.5) ** 2, axis=1).astype(np.float32)
    return {"observations": obs, "actions": act, "rewards": rew, "masks": np.zeros(n, np.float32),
            "next_observations": obs.copy(), "terminals": np.ones(n, np.float32)}


def dataset_sha256(data: dict) -> str:
    h = hashlib.sha256()
    for k in sorted(data):
        h.update(k.encode())
        h.update(np.ascontiguousarray(data[k]).tobytes())
    return h.hexdigest()
