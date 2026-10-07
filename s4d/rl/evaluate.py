"""Batched evaluation over training cameras, held-out cameras and camera-perturbation trajectories.

A pool of independent environments steps in lockstep so the frozen encoder runs once per step for
the whole pool. Episode-to-environment assignment is deterministic, so for one seed every encoder
is evaluated on the same initial states and camera paths.
"""

from __future__ import annotations

import numpy as np
import torch

from s4d.rl.env import HELDOUT_PATHS, TRAIN_PATHS, trajectory_path


@torch.no_grad()
def policy_inputs(agent, stacked: np.ndarray) -> torch.Tensor:
    """(N, 3T, H, W) uint8 stacks -> pixels for the CNN agent, or cached features for frozen encoders."""
    pixels = torch.as_tensor(stacked, device=agent.device)
    return pixels if agent.use_pixels else agent.encoder.extract_cacheable_feature(pixels)


def run_episodes(envs, agent, paths, step: int) -> tuple[list[float], list[float]]:
    """Roll out one episode per camera path; returns (success, return) per episode, in order."""
    successes, returns = [0.0] * len(paths), [0.0] * len(paths)
    for start in range(0, len(paths), len(envs)):
        batch = list(range(start, min(start + len(envs), len(paths))))
        obs, proprio = zip(*(envs[k].reset(paths[i]) for k, i in enumerate(batch)))
        obs, proprio, active = list(obs), list(proprio), list(range(len(batch)))
        while active:
            inputs = policy_inputs(agent, np.stack([obs[k] for k in active]))
            actions = agent.act(inputs, np.stack([proprio[k] for k in active]), step=step, eval_mode=True)
            still = []
            for action, k in zip(actions, active):
                obs[k], proprio[k], reward, done, info = envs[k].step(action)
                returns[batch[k]] += reward
                successes[batch[k]] = max(successes[batch[k]], info["success"])
                if not done:
                    still.append(k)
            active = still
    return successes, returns


def evaluation_suite(envs, agent, step: int, cfg: dict) -> dict[str, float]:
    """Training cameras (reference protocol), each held-out camera, and the reference trajectories."""
    groups = {name: (path, cfg["episodes_per_train_camera"]) for name, path in TRAIN_PATHS.items()}
    groups |= {name: (path, cfg["episodes_per_heldout_camera"]) for name, path in HELDOUT_PATHS.items()}
    groups |= {f"traj_{kind}": (trajectory_path(kind), cfg["episodes_per_trajectory"]) for kind in ("lateral", "circular")}
    names = [name for name, (_, n) in groups.items() for _ in range(n)]
    successes, returns = run_episodes(envs, agent, [groups[name][0] for name in names], step)
    metrics: dict[str, float] = {}
    for name in groups:
        idx = [i for i, n in enumerate(names) if n == name]
        metrics[f"eval/{name}_success"] = float(np.mean([successes[i] for i in idx]))
        metrics[f"eval/{name}_return"] = float(np.mean([returns[i] for i in idx]))
    metrics["eval/train_cameras_success"] = float(np.mean([metrics[f"eval/{n}_success"] for n in TRAIN_PATHS]))
    metrics["eval/heldout_cameras_success"] = float(np.mean([metrics[f"eval/{n}_success"] for n in HELDOUT_PATHS]))
    return metrics
