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
    """Camera groups and episode counts. ``train_episodes_total`` (light evaluation, task screening only) spreads a
    total over the training cameras instead of ``episodes_per_train_camera``; groups with zero episodes are skipped."""
    total = cfg.get("train_episodes_total")
    if total:
        base, extra = divmod(int(total), len(TRAIN_PATHS))
        train = {name: base + (i < extra) for i, name in enumerate(TRAIN_PATHS)}
    else:
        train = {name: int(cfg["episodes_per_train_camera"]) for name in TRAIN_PATHS}
    groups = {name: (path, train[name]) for name, path in TRAIN_PATHS.items()}
    groups |= {name: (path, int(cfg["episodes_per_heldout_camera"])) for name, path in HELDOUT_PATHS.items()}
    groups |= {f"traj_{k}": (trajectory_path(k), int(cfg["episodes_per_trajectory"])) for k in ("lateral", "circular")}
    return {name: group for name, group in groups.items() if group[1] > 0}


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
    for key, names in (
        ("train_cameras_success", list(TRAIN_PATHS)),
        ("heldout_cameras_success", list(HELDOUT_PATHS)),
        ("trajectories_success", ["traj_lateral", "traj_circular"]),
    ):
        present = [metrics[f"{n}_success"] for n in names if n in groups]
        if present:
            metrics[key] = float(np.mean(present))
    return metrics
