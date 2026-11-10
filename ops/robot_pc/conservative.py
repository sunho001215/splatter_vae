"""Conservative (TIGHTEN-ONLY) motion limits for the first controlled hardware tests (Directive J, P3).

Every value can only make the configured limits stricter: ``tighten`` takes the minimum of the configured value and
the profile value, so a profile can never loosen the collector-derived limits of configs/robot_pc/robot_pc.yaml, and
the controller's own protections (switch_controller rejects EE targets with log6 error > 0.2 m / 0.5 rad,
action_manager_base.cpp:197-208; holds the measured pose after target_dt 0.05 s) stay untouched. ``test_box``
intersects the task's workspace box with a box around the start pose, so a rollout's validated goals stay within a
small radius of where the robot started (a goal outside is REJECTED: no command, the controller holds).

Profile ``slow`` (P3 test 3, short policy rollout): linear 0.05 m/s (collector 0.3), angular 0.25 rad/s (1.5),
accelerations 0.5 m/s^2 / 2.5 rad/s^2 (4 / 15), lead envelope 0.02 m / 8 deg (0.04 / 20), goals within 0.08 m of the
start flange position. ``GRIPPER_TEST``: Robotiq speed 64/255 (collector 146), force 0 (minimum), closures 0..0.5.
"""

from __future__ import annotations

import copy
import dataclasses

import numpy as np

PROFILES = {
    "slow": {
        "executor": {"max_linear_speed": 0.05, "max_angular_speed": 0.25, "max_linear_accel": 0.5,
                     "max_angular_accel": 2.5, "lead_m": 0.02, "lead_rot_deg": 8.0},
        "test_radius_m": 0.08,
    },
}
GRIPPER_TEST = {"speed": 64, "force": 0, "max_closure": 0.5}
TIGHTENABLE = ("max_linear_speed", "max_angular_speed", "max_linear_accel", "max_angular_accel", "lead_m",
               "lead_rot_deg")


def tighten(cfg: dict, profile: str) -> tuple[dict, dict]:
    """(new cfg, {key: (configured, applied)}); the executor values become min(configured, profile)."""
    if profile not in PROFILES:
        raise ValueError(f"unknown motion profile {profile!r} (known: {sorted(PROFILES)})")
    out = copy.deepcopy(cfg)
    ex = out.setdefault("executor", {})
    applied = {}
    for key, value in PROFILES[profile]["executor"].items():
        if key not in TIGHTENABLE:
            raise ValueError(f"profile key {key} is not tighten-only")
        cur = float(ex.get(key, value))
        ex[key] = min(cur, float(value))
        applied[key] = (cur, ex[key])
    return out, applied


def test_box(limits, start_flange_pose7, radius_m: float):
    """ActionLimits with the workspace box intersected with start +- radius (never larger than the task box)."""
    p = np.asarray(start_flange_pose7, float)[:3]
    lo = np.maximum(np.asarray(limits.workspace_min_m, float), p - float(radius_m))
    hi = np.minimum(np.asarray(limits.workspace_max_m, float), p + float(radius_m))
    if np.any(lo >= hi) or np.any(p < lo) or np.any(p > hi):
        raise ValueError(f"start position {np.round(p, 4).tolist()} is outside the task workspace box "
                         f"{list(limits.workspace_min_m)}..{list(limits.workspace_max_m)}")
    return dataclasses.replace(limits, workspace_min_m=tuple(lo.tolist()), workspace_max_m=tuple(hi.tolist()))


def gripper_closures(targets) -> list:
    """Validated closure targets for the gripper test (0 .. GRIPPER_TEST max_closure)."""
    out = [float(t) for t in targets]
    if not out or any(not np.isfinite(t) or t < 0 or t > GRIPPER_TEST["max_closure"] for t in out):
        raise ValueError(f"gripper test closures must be in [0, {GRIPPER_TEST['max_closure']}]: {targets}")
    return out
