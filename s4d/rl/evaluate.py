"""Evaluation protocol: six training cameras (reference: 120 episodes), each held-out camera and both trajectories.

Episode reset seeds depend only on the run seed and the evaluation index, so for one seed every encoder
is evaluated on identical initial states and camera paths at every evaluation point.
"""

from __future__ import annotations

import numpy as np
import torch

from s4d.rl.env import HELDOUT_PATHS, TRAIN_PATHS, camera_c2w, trajectory_path

# Multi-view encoders (e.g. SNeRL, 3 views): training group i uses training cameras (i, i+1, i+3) mod 6, a cyclic
# design in which every camera appears in exactly three groups. A viewpoint-shift group keeps two inputs of a training
# group and replaces its first camera by the shifted one: a held-out camera replaces its nearest training camera, a
# trajectory replaces its base camera (train1).
VIEW_OFFSETS = {3: (0, 1, 3)}


def _training_triple(index: int, views: int) -> list[str]:
    names = list(TRAIN_PATHS)
    return [names[(index + offset) % len(names)] for offset in VIEW_OFFSETS[views]]


def nearest_training_camera(path) -> int:
    """Index of the training camera whose viewing direction is closest to ``path``'s first pose."""
    forward = camera_c2w(path[0])[:3, 2]
    return int(max(range(len(TRAIN_PATHS)), key=lambda i: float(camera_c2w(TRAIN_PATHS[f"train{i}"][0])[:3, 2] @ forward)))


def multiview_groups(groups: dict, views: int) -> dict:
    """Single-camera groups -> groups of ``views`` camera paths (see ``VIEW_OFFSETS``)."""
    if views not in VIEW_OFFSETS:
        raise ValueError(f"no multi-view evaluation design for {views} views")
    out = {}
    for name, (path, count) in groups.items():
        if name in TRAIN_PATHS:
            out[name] = (tuple(TRAIN_PATHS[n] for n in _training_triple(int(name.removeprefix("train")), views)), count)
            continue
        base = 1 if name.startswith("traj_") else nearest_training_camera(path)
        triple = [TRAIN_PATHS[n] for n in _training_triple(base, views)]
        out[name] = (tuple([path, *triple[1:]]), count)
    return out


@torch.no_grad()
def policy_inputs(agent, stacked) -> torch.Tensor:
    """(N, 3T, H, W) uint8 stacks (or a dict of batched arrays from multi-view / depth / camera envs) -> pixels for
    the CNN agent, or cached features for frozen encoders."""
    if isinstance(stacked, dict):
        arrays = {k: torch.as_tensor(v, device=agent.device) for k, v in stacked.items()}
        return agent.encoder.extract_cacheable_feature(arrays)
    pixels = torch.as_tensor(stacked, device=agent.device)
    return pixels if agent.use_pixels else agent.encoder.extract_cacheable_feature(pixels)


def evaluation_groups(cfg: dict, views: int = 1) -> dict[str, tuple[list, int]]:
    """Camera groups and episode counts. ``train_episodes_total`` (light evaluation, task screening only) spreads a
    total over the training cameras instead of ``episodes_per_train_camera``; groups with zero episodes are skipped.
    With ``views`` > 1 every group holds one camera path per view (``multiview_groups``)."""
    total = cfg.get("train_episodes_total")
    if total:
        base, extra = divmod(int(total), len(TRAIN_PATHS))
        train = {name: base + (i < extra) for i, name in enumerate(TRAIN_PATHS)}
    else:
        train = {name: int(cfg["episodes_per_train_camera"]) for name in TRAIN_PATHS}
    groups = {name: (path, train[name]) for name, path in TRAIN_PATHS.items()}
    groups |= {name: (path, int(cfg["episodes_per_heldout_camera"])) for name, path in HELDOUT_PATHS.items()}
    groups |= {f"traj_{k}": (trajectory_path(k), int(cfg["episodes_per_trajectory"])) for k in ("lateral", "circular")}
    groups = {name: group for name, group in groups.items() if group[1] > 0}
    return groups if int(views) == 1 else multiview_groups(groups, int(views))


def episode_seeds(run_seed: int, eval_index: int, count: int) -> list[int]:
    return np.random.default_rng([int(run_seed), int(eval_index), 7]).integers(0, 2**31 - 1, size=count).tolist()


def evaluation_suite(pool, agent, step: int, eval_index: int, run_seed: int, cfg: dict, views: int = 1) -> dict[str, float]:
    groups = evaluation_groups(cfg, views)
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
