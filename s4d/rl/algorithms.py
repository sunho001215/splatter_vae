"""Evaluation policies of the downstream algorithms (user directive 2026-10-28), keyed by the run config's ``algorithm``.

Every policy offers the interface ``s4d.rl.evaluate`` uses: ``device``, ``use_pixels``,
``encoder.extract_cacheable_feature``, ``act(inputs, proprio, step, eval_mode[, index])``, ``train(mode)`` and
``load_policy(snapshot["policy"])``; recurrent policies also set ``recurrent = True`` and ``begin_episodes(count)``.
DrM keeps its own path in scripts/eval_rl.py.
"""

from __future__ import annotations

import torch


def evaluation_policy(cfg: dict, action_dim: int, proprio_dim: int, device: torch.device):
    algorithm = cfg.get("algorithm")
    if algorithm == "snerl_sac":
        from s4d.rl.snerl_sac import SNeRLSAC

        return SNeRLSAC(cfg, action_dim, proprio_dim, device)
    if algorithm == "sb3_ppo":
        from s4d.rl.sb3_ppo import PPOPolicy

        return PPOPolicy(cfg, action_dim, proprio_dim, device)
    if algorithm == "dreamer_v1":
        from s4d.rl.dreamer_v1 import DreamerPolicy

        return DreamerPolicy(cfg, action_dim, proprio_dim, device)
    raise ValueError(f"no evaluation policy for algorithm {algorithm!r}")
