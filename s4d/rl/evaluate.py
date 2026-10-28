"""Evaluation protocol: six training cameras (reference: 120 episodes), each held-out camera and both trajectories.

Episode reset seeds depend only on the run seed and the evaluation index, so for one seed every encoder
is evaluated on identical initial states and camera paths at every evaluation point.

Camera suite v2 (evaluation only, 2026-10-29; ``s4d.rl.camera_suites``) adds its own groups -- the visibility-selected
held-out cameras vis0..vis3, the stronger moving paths ``traj_*_2x/3x`` and optional matched references ``ref_*`` --
with their own paired seed stream (``v2_episode_seeds``). The original groups, counts and seeds are unchanged.
"""

from __future__ import annotations

import numpy as np
import torch

from s4d.rl.camera_suites import STRONGER_TRAJECTORIES, V2_REFERENCES, V2_SEED_TAG
from s4d.rl.env import HELDOUT_PATHS, HELDOUT_V2_PATHS, TRAIN_PATHS, camera_c2w, heldout_v2_paths, trajectory_path

# Multi-view encoders (e.g. SNeRL, 3 views): training group i uses training cameras (i, i+1, i+3) mod 6, a cyclic
# design in which every camera appears in exactly three groups. A viewpoint-shift group keeps two inputs of a training
# group and replaces its first camera by the shifted one: a held-out camera replaces its nearest training camera, a
# trajectory replaces its base camera (train1). Camera suite v2 applies the same rule: vis* replaces its nearest
# training camera, traj_*_2x/3x replace train1, and the matched references ref_<group> use <group>'s inputs.
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
        key = name.removeprefix("ref_")  # v2 matched references follow the group they repeat
        if key in TRAIN_PATHS:
            out[name] = (tuple(TRAIN_PATHS[n] for n in _training_triple(int(key.removeprefix("train")), views)), count)
            continue
        base = 1 if key.startswith("traj_") else nearest_training_camera(path)
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


def evaluation_groups_v2(cfg: dict, views: int = 1, reference: str = "train+traj",
                         task: str | None = None) -> dict[str, tuple[list, int]]:
    """Camera suite v2 groups: [ref_train0..5,] vis0..3, [ref_traj_lateral, ref_traj_circular,] traj_*_2x, traj_*_3x.
    ``task`` applies the registered task-specific exceptions of the held-out cameras (``heldout_v2_paths``).

    Episode counts are the historical per-camera counts (``episodes_per_train_camera`` for the training-camera
    references, ``episodes_per_heldout_camera`` for vis*, ``episodes_per_trajectory`` for every path); the light
    ``train_episodes_total`` of task screening does not apply. ``reference``: ``none``, ``train`` (the six training
    cameras) or ``train+traj`` (also the 1x reference paths), re-run on the v2 seeds as a matched reference."""
    if reference not in V2_REFERENCES:
        raise ValueError(f"reference must be one of {V2_REFERENCES}")
    if not HELDOUT_V2_PATHS:
        raise ValueError("camera suite v2 is not frozen (s4d.rl.camera_suites.HELDOUT_V2_CAMERAS is empty)")
    n_train, n_cam, n_traj = (int(cfg[k]) for k in
                              ("episodes_per_train_camera", "episodes_per_heldout_camera", "episodes_per_trajectory"))
    groups: dict[str, tuple[list, int]] = {}
    if reference != "none":
        groups |= {f"ref_{name}": (path, n_train) for name, path in TRAIN_PATHS.items()}
    groups |= {name: (path, n_cam) for name, path in heldout_v2_paths(task).items()}
    if reference == "train+traj":
        groups |= {f"ref_traj_{k}": (trajectory_path(k), n_traj) for k in ("lateral", "circular")}
    groups |= {f"traj_{k}": (trajectory_path(k), n_traj) for k in STRONGER_TRAJECTORIES}
    groups = {name: group for name, group in groups.items() if group[1] > 0}
    return groups if int(views) == 1 else multiview_groups(groups, int(views))


def v2_episode_seeds(run_seed: int, eval_index: int, count: int) -> list[int]:
    """The paired v2 seed list of one evaluation point: every v2 group uses its first ``count`` entries."""
    return np.random.default_rng([int(run_seed), int(eval_index), 7, V2_SEED_TAG]).integers(
        0, 2**31 - 1, size=count).tolist()


def _rollout(pool, agent, step: int, groups: dict, names: list[str], seeds: list[int]) -> tuple[list, list]:
    if getattr(agent, "recurrent", False):  # Dreamer (2026-10-28): per-episode recurrent state kept by the agent
        class act:  # noqa: N801  (a callable with EnvPool's recurrent hooks)
            active: list[int] = []

            @staticmethod
            def begin_wave(count: int) -> None:
                agent.begin_episodes(count)

            def __call__(self, obs, proprio):
                return agent.act(policy_inputs(agent, obs), proprio, step=step, eval_mode=True, index=self.active)

        act = act()
    else:
        def act(obs, proprio):
            return agent.act(policy_inputs(agent, obs), proprio, step=step, eval_mode=True)

    return pool.run(act, [(groups[name][0], seed) for name, seed in zip(names, seeds)])


def _group_metrics(groups: dict, names: list[str], successes: list, returns: list) -> dict[str, float]:
    metrics: dict[str, float] = {"episodes": len(names)}
    for name in groups:
        idx = [i for i, n in enumerate(names) if n == name]
        metrics[f"{name}_success"] = float(np.mean([successes[i] for i in idx]))
        metrics[f"{name}_return"] = float(np.mean([returns[i] for i in idx]))
    return metrics


def evaluation_suite(pool, agent, step: int, eval_index: int, run_seed: int, cfg: dict, views: int = 1) -> dict[str, float]:
    groups = evaluation_groups(cfg, views)
    names = [name for name, (_, n) in groups.items() for _ in range(n)]
    seeds = episode_seeds(run_seed, eval_index, len(names))
    successes, returns = _rollout(pool, agent, step, groups, names, seeds)
    metrics = _group_metrics(groups, names, successes, returns)
    for key, names in (
        ("train_cameras_success", list(TRAIN_PATHS)),
        ("heldout_cameras_success", list(HELDOUT_PATHS)),
        ("trajectories_success", ["traj_lateral", "traj_circular"]),
    ):
        present = [metrics[f"{n}_success"] for n in names if n in groups]
        if present:
            metrics[key] = float(np.mean(present))
    return metrics


V2_CONDITIONS = {  # condition -> member groups (success and return are the means over present members)
    "heldout_v2": list(HELDOUT_V2_PATHS),
    "traj_2x": ["traj_lateral_2x", "traj_circular_2x"],
    "traj_3x": ["traj_lateral_3x", "traj_circular_3x"],
    "moving_v2": [f"traj_{k}" for k in STRONGER_TRAJECTORIES],
    "ref_train_cameras": [f"ref_{n}" for n in TRAIN_PATHS],
    "ref_trajectories": ["ref_traj_lateral", "ref_traj_circular"],
}


def evaluation_suite_v2(pool, agent, step: int, eval_index: int, run_seed: int, cfg: dict, views: int = 1,
                        reference: str = "train+traj", task: str | None = None) -> dict[str, float]:
    """Camera suite v2 at one evaluation point: group order as ``evaluation_groups_v2``; group g's k-th episode uses
    the k-th paired seed, so every group (and every method with this run seed) starts from the same states."""
    groups = evaluation_groups_v2(cfg, views, reference, task)
    names = [name for name, (_, n) in groups.items() for _ in range(n)]
    paired = v2_episode_seeds(run_seed, eval_index, max(n for _, n in groups.values()))
    seeds = [paired[k] for name, (_, n) in groups.items() for k in range(n)]
    successes, returns = _rollout(pool, agent, step, groups, names, seeds)
    metrics = _group_metrics(groups, names, successes, returns)
    for condition, members in V2_CONDITIONS.items():
        present = [n for n in members if n in groups]
        if present:
            metrics[f"{condition}_success"] = float(np.mean([metrics[f"{n}_success"] for n in present]))
            metrics[f"{condition}_return"] = float(np.mean([metrics[f"{n}_return"] for n in present]))
    return metrics
