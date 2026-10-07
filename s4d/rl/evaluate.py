"""Evaluation protocol: six training cameras (reference: 120 episodes), each held-out camera and both trajectories.

Episode reset seeds depend only on the run seed and the evaluation index, so for one seed every encoder
is evaluated on identical initial states and camera paths at every evaluation point.
"""

from __future__ import annotations

import numpy as np
import torch

from s4d.rl.env import HELDOUT_PATHS, TRAIN_PATHS, trajectory_path


@torch.no_grad()
def policy_inputs(agent, stacked) -> torch.Tensor:
    """(N, 3T, H, W) uint8 stacks -> pixels for the CNN agent, or cached features for frozen encoders."""
    pixels = torch.as_tensor(stacked, device=agent.device)
    return pixels if agent.use_pixels else agent.encoder.extract_cacheable_feature(pixels)


def evaluation_groups(cfg: dict) -> dict[str, tuple[list, int]]:
    groups = {name: (path, int(cfg["episodes_per_train_camera"])) for name, path in TRAIN_PATHS.items()}
    groups |= {name: (path, int(cfg["episodes_per_heldout_camera"])) for name, path in HELDOUT_PATHS.items()}
    groups |= {f"traj_{k}": (trajectory_path(k), int(cfg["episodes_per_trajectory"])) for k in ("lateral", "circular")}
    return groups


def episode_seeds(run_seed: int, eval_index: int, count: int) -> list[int]:
    return np.random.default_rng([int(run_seed), int(eval_index), 7]).integers(0, 2**31 - 1, size=count).tolist()


def evaluation_suite(pool, agent, step: int, eval_index: int, run_seed: int, cfg: dict) -> dict[str, float]:
    groups = evaluation_groups(cfg)
    names = [name for name, (_, n) in groups.items() for _ in range(n)]
    seeds = episode_seeds(run_seed, eval_index, len(names))

    def act(obs, proprio):
        return agent.act(policy_inputs(agent, obs), proprio, step=step, eval_mode=True)

    successes, returns = pool.run(act, [(groups[name][0], seed) for name, seed in zip(names, seeds)])
    metrics: dict[str, float] = {"episodes": len(names)}
    for name in groups:
        idx = [i for i, n in enumerate(names) if n == name]
        metrics[f"{name}_success"] = float(np.mean([successes[i] for i in idx]))
        metrics[f"{name}_return"] = float(np.mean([returns[i] for i in idx]))
    metrics["train_cameras_success"] = float(np.mean([metrics[f"{n}_success"] for n in TRAIN_PATHS]))
    metrics["heldout_cameras_success"] = float(np.mean([metrics[f"{n}_success"] for n in HELDOUT_PATHS]))
    metrics["trajectories_success"] = float(np.mean([metrics[f"traj_{k}_success"] for k in ("lateral", "circular")]))
    return metrics
