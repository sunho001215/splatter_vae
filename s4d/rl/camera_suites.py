"""Versioned camera evaluation suites (user directive 2026-10-29). Evaluation only: nothing here is used by
pretraining or RL training, which keep the six training cameras and their data unchanged.

Suite ``v1`` is the original protocol (``s4d.rl.evaluate.evaluation_groups``): train0..5, the held-out cameras
eval0..3 and the reference lateral/circular trajectories, written to ``eval.jsonl``. Its held-out cameras often put
the robot's proximal links between the camera and the workspace; they stay unchanged and are reported separately as
an *occlusion-heavy stress test*.

Suite ``v2`` (``cameras_v2``) is written to its own file, ``eval_cameras_v2.jsonl``, with its own group names:

* ``heldout_v2``: four static cameras ``vis0..vis3`` on the training rig's orbit (look-at ``LOOKAT``, radius 1.0,
  fovy 45), selected before any policy was evaluated, by the pre-registered method-independent visibility rule of
  ``scripts/camera_visibility.py`` (rule: ``RULE_FILE``; frozen selection with its measurements: ``DEFINITION_FILE``).
* stronger moving cameras ``traj_{lateral,circular}_{2x,3x}``: the reference paths (base camera train1, 72 looping
  poses, same starting phase) with larger amplitudes. At a fixed period a larger amplitude also means a faster
  camera, so these are combined range-and-speed stress tests.
* optional matched references in the same pass, on the same episode seeds: ``ref_train0..5`` (the training cameras)
  and ``ref_traj_{lateral,circular}`` (the 1x reference paths).

This module is pure geometry (no MuJoCo import) so tests and the evaluator can use it anywhere.
"""

from __future__ import annotations

import math

import numpy as np

from s4d.data.metaworld.cameras import EVAL_CAMERAS, TRAIN_CAMERAS

SUITE_V2 = "cameras_v2"
HELDOUT_V2 = "heldout_v2"
RULE_FILE = "runs/setup/camera-suite-v2-rule-20261029.json"
DEFINITION_FILE = "runs/setup/camera-suite-v2-definition-20261029.json"

# Frozen 2026-10-29 by scripts/camera_visibility.py (DEFINITION_FILE): (tilt deg, azimuth deg) on the training orbit.
HELDOUT_V2_CAMERAS: tuple[tuple[float, float], ...] = ()
HELDOUT_V2_NAMES = tuple(f"vis{i}" for i in range(len(HELDOUT_V2_CAMERAS)))
# Task-specific exceptions (a task whose common-rig visibility fails the registered minimum): none unless listed here
# with the numbers that justify them in DEFINITION_FILE; applied identically to every method.
HELDOUT_V2_TASK_EXCEPTIONS: dict[str, dict[str, tuple[float, float]]] = {}

# Stronger moving-camera levels: (path kind, amplitudes passed to s4d.rl.env.trajectory_path). The 1x reference paths
# are 0.12 m lateral and +/-10 deg azimuth, +/-6 deg tilt; base camera, 72-pose period and starting phase unchanged.
STRONGER_TRAJECTORIES: dict[str, tuple[str, dict[str, float]]] = {
    "lateral_2x": ("lateral", {"lateral_amplitude": 0.24}),
    "circular_2x": ("circular", {"circular_azimuth_deg": 20.0, "circular_elevation_deg": 12.0}),
    "lateral_3x": ("lateral", {"lateral_amplitude": 0.36}),
    "circular_3x": ("circular", {"circular_azimuth_deg": 30.0, "circular_elevation_deg": 18.0}),
}

# Episode seeds of the v2 groups: one paired list per (run seed, evaluation index) from its own RNG stream, shared by
# every v2 group (a group with n episodes uses the first n), so cameras, severity levels and the matched references
# are compared on identical initial conditions. The v1 stream (``evaluate.episode_seeds``) is not touched.
V2_SEED_TAG = 20261029
V2_REFERENCES = ("none", "train", "train+traj")

SUITE_OUTPUTS = {"v1": "eval.jsonl", "v2": "eval_cameras_v2.jsonl"}
SUITE_PROVENANCE = {"v1": "eval_provenance", "v2": "eval_cameras_v2_provenance"}
SUITE_WANDB_SUFFIX = {"v1": "-eval", "v2": "-eval-cameras-v2"}
SUITE_WANDB_PREFIX = {"v1": "eval", "v2": "eval_cameras_v2"}


def view_direction(tilt_deg: float, azimuth_deg: float) -> np.ndarray:
    """Unit vector from the look-at point to a rig camera (MuJoCo free camera, elevation = -tilt)."""
    az, el = math.radians(azimuth_deg), math.radians(-tilt_deg)
    forward = np.array([math.cos(el) * math.cos(az), math.cos(el) * math.sin(az), math.sin(el)])
    return -forward


def angular_separation_deg(a: tuple[float, float], b: tuple[float, float]) -> float:
    """Angle between the viewing directions of two rig cameras given as (tilt, azimuth) in degrees."""
    cos = float(np.clip(view_direction(*a) @ view_direction(*b), -1.0, 1.0))
    return math.degrees(math.acos(cos))


def min_separation_deg(camera: tuple[float, float], others) -> float:
    return min((angular_separation_deg(camera, o) for o in others), default=math.inf)


def candidate_grid(tilts=range(25, 66, 5), azimuths=range(0, 360, 10)) -> list[tuple[float, float]]:
    return [(float(t), float(a)) for t in tilts for a in azimuths]


def reference_cameras() -> dict[str, tuple[float, float]]:
    out = {f"train{i}": tuple(map(float, c)) for i, c in enumerate(TRAIN_CAMERAS)}
    out |= {f"eval{i}": tuple(map(float, c)) for i, c in enumerate(EVAL_CAMERAS)}
    return out


def _rotation_angle_deg(Ra: np.ndarray, Rb: np.ndarray) -> float:
    cos = (np.trace(Ra.T @ Rb) - 1.0) / 2.0
    return math.degrees(math.acos(float(np.clip(cos, -1.0, 1.0))))


def path_kinematics(c2w: np.ndarray, lookat, *, action_repeat: int, sim_dt: float, episode_agent_steps: int) -> dict:
    """Pose ranges and speeds of a looping camera path given its per-pose camera-to-world transforms (P,4,4).

    The environment renders once per agent step, so consecutive poses are one agent step = ``action_repeat``
    simulator steps apart, and the path wraps from its last pose to its first. Angular speed is reported both as the
    change of viewing direction seen from the look-at point and as the camera's own rotation angle."""
    c2w = np.asarray(c2w, dtype=np.float64)
    P = len(c2w)
    positions, rotations = c2w[:, :3, 3], c2w[:, :3, :3]
    rel = positions - np.asarray(lookat, dtype=np.float64)
    distance = np.linalg.norm(rel, axis=1)
    elevation = np.degrees(np.arcsin(rel[:, 2] / distance))  # camera height angle above the look-at point (= tilt)
    azimuth = np.degrees(np.arctan2(-rel[:, 1], -rel[:, 0]))  # MuJoCo free-camera azimuth (forward = -rel)
    nxt = np.roll(np.arange(P), -1)
    translation = np.linalg.norm(positions[nxt] - positions, axis=1)
    dirs = rel / distance[:, None]
    view_angle = np.degrees(np.arccos(np.clip((dirs[nxt] * dirs).sum(1), -1.0, 1.0)))
    rotation = np.asarray([_rotation_angle_deg(rotations[i], rotations[j]) for i, j in zip(range(P), nxt)])
    agent_dt = action_repeat * sim_dt
    renders = episode_agent_steps + 1  # the reset frame plus one frame per agent step

    def stats(x, per):
        return {"max": float(x.max()), "mean": float(x.mean()), "per": per}

    return {
        "period_poses": P,
        "period_agent_steps": P,
        "period_sim_steps": P * action_repeat,
        "period_seconds": P * agent_dt,
        "episode_renders": renders,
        "episode_periods": renders / P,
        "phase_at_agent_step": "pose index = agent step mod period (the reset frame is pose 0)",
        "azimuth_deg": [float(azimuth.min()), float(azimuth.max())],
        "tilt_deg": [float(elevation.min()), float(elevation.max())],
        "distance_m": [float(distance.min()), float(distance.max())],
        "position_min_m": positions.min(0).tolist(),
        "position_max_m": positions.max(0).tolist(),
        "translation_m_per_agent_step": stats(translation, "agent step"),
        "translation_m_per_sim_step": stats(translation / action_repeat, "simulator step"),
        "translation_m_per_s": stats(translation / agent_dt, "second"),
        "view_angle_deg_per_agent_step": stats(view_angle, "agent step"),
        "view_angle_deg_per_sim_step": stats(view_angle / action_repeat, "simulator step"),
        "rotation_deg_per_agent_step": stats(rotation, "agent step"),
        "rotation_deg_per_sim_step": stats(rotation / action_repeat, "simulator step"),
        "rotation_deg_per_s": stats(rotation / agent_dt, "second"),
        "loop_step_m": float(translation[-1]),
        "loop_step_rotation_deg": float(rotation[-1]),
        "max_step_within_period_m": float(translation[:-1].max()),
    }


def reevaluation_steps(available, num_train_steps: int, every: int, which: str) -> list[int]:
    """Snapshot steps re-evaluated for a finished run: ``final`` (the last training step) or ``last5`` (the last five
    evaluation points, e.g. 960k..1M of a 1M-step run; their mean is the last-5 metric, the 1M one the final)."""
    if which not in ("final", "last5"):
        raise ValueError(which)
    count = 1 if which == "final" else 5
    wanted = [int(num_train_steps) - k * int(every) for k in reversed(range(count))]
    missing = sorted(set(wanted) - {int(s) for s in available})
    if missing:
        raise ValueError(f"finished run lacks snapshots {missing} needed for {which}")
    return wanted
