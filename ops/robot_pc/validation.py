"""Robot-side action validation: a policy result becomes an executable ``Command`` only if it is finite, inside the
contract ranges, bound to the measured pose of ITS observation, and not expired. numpy only. No per-step distance,
workspace box or speed limit (user directive K, 2026-11-10): the goal is the policy's; the executor never sends a
setpoint switch_controller would reject (its 0.2 m / 0.5 rad log6 guard).

Checks (``validate_action``), each failure an ``ActionRejected`` with a code:
* ``no_action``: the step returned no action (error / timeout / takeover);
* ``binding``: reference id, session, frame ids, observation time, pose / gripper sample times or the echoed measured
  pose differ from the request this client sent (an action is never re-applied to another pose);
* ``non_finite`` / ``range``: non-finite outputs, |action_norm| > 1 (the FQL actor's tanh range), malformed quaternion;
* ``gripper``: closure outside the configured range, or register != round(3 + 227 c) (recorder convention);
* ``expired`` / ``future``: observation older than ``max_action_age_ms`` (or stamped in the future) at validation time.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ops.robot_pc.poses import as_pose7, quat_angle

GRIPPER_OPEN_REGISTER, GRIPPER_SPAN = 3.0, 227.0  # conventions_json: "Robotiq position 3 + 227 * closure (DROID)"


class ActionRejected(ValueError):
    def __init__(self, code: str, detail: str):
        super().__init__(f"{code}: {detail}")
        self.code, self.detail = code, detail


@dataclass(frozen=True)
class ActionLimits:
    gripper_closure_min: float = 0.0
    gripper_closure_max: float = 1.0
    max_action_age_ms: float = 200.0  # observation (capture) age bound at validation and until execution
    quaternion_tolerance: float = 1e-6

    @classmethod
    def from_config(cls, d: dict) -> ActionLimits:
        unknown = set(d) - set(cls.__dataclass_fields__)
        if unknown:
            raise ValueError(f"unknown action limit keys {sorted(unknown)}")
        kw = {k: tuple(float(x) for x in v) if isinstance(v, (list, tuple)) else float(v) for k, v in d.items()}
        out = cls(**kw)
        values = [out.gripper_closure_min, out.gripper_closure_max, out.max_action_age_ms, out.quaternion_tolerance]
        if not all(np.isfinite(values)) or out.max_action_age_ms <= 0:
            raise ValueError("action limits must be finite and positive")
        if not 0 <= out.gripper_closure_min <= out.gripper_closure_max <= 1:
            raise ValueError("gripper closure range must lie in [0, 1]")
        return out


@dataclass(frozen=True)
class BoundRequest:
    """What this client sent: the observation and the measured samples the action must be bound to."""
    reference_id: str
    end_ids: dict
    observation_time_ns: int
    pose7: tuple
    pose_time_ns: int
    gripper_register: float
    gripper_time_ns: int


@dataclass(frozen=True)
class Command:
    """An executable command: ABSOLUTE flange goal in the base frame + gripper, bound to its observation."""
    reference_id: str
    session_id: str
    observation_time_ns: int
    expires_ns: int  # observation_time_ns + max_action_age_ms: never executed after this (client clock)
    measured_pose7: tuple
    goal_pose7: tuple  # execution_flange_pose (lead envelope around the measured pose); else flange_pose
    gripper_closure: float
    gripper_register: int
    translation_m: float
    rotation_deg: float

    def to_json(self) -> dict:
        return {k: (list(v) if isinstance(v, tuple) else v) for k, v in self.__dict__.items()}


def validate_action(result, request: BoundRequest, limits: ActionLimits, *, session_id: str, now_ns: int) -> Command:
    if not getattr(result, "ok", False):
        raise ActionRejected("no_action", f"{getattr(result, 'status', '?')}: {getattr(result, 'code', None)} "
                                          f"{getattr(result, 'detail', None)}")
    o = result.outputs
    ids = o.get("obs_ids") or {}
    if result.reference_id != request.reference_id or ids.get("reference_id") != request.reference_id:
        raise ActionRejected("binding", f"action of {result.reference_id!r}, request {request.reference_id!r}")
    if result.session_id != session_id or ids.get("session_id") != session_id:
        raise ActionRejected("binding", "action of another session")
    if any(ids.get("frame_ids", {}).get(c, [None])[-1] != int(f) for c, f in request.end_ids.items()):
        raise ActionRejected("binding", f"frame ids {ids.get('frame_ids')} do not end at {request.end_ids}")
    for key, want in (("observation_time_ns", request.observation_time_ns), ("pose_time_ns", request.pose_time_ns),
                      ("gripper_time_ns", request.gripper_time_ns)):
        if ids.get(key) != want:
            raise ActionRejected("binding", f"{key} {ids.get(key)} != requested {want}")
    measured = np.asarray(o.get("measured_pose7"), np.float64)
    if measured.shape != (7,) or not np.array_equal(measured, np.asarray(request.pose7, np.float64)):
        raise ActionRejected("binding", "the action is not bound to the measured pose sent with its observation")
    if not np.isfinite(measured).all() or abs(float(np.linalg.norm(measured[3:])) - 1.0) > 1e-3:
        raise ActionRejected("range", "measured pose is not finite with a unit quaternion")
    if float(o.get("measured_gripper_register", request.gripper_register)) != float(request.gripper_register):
        raise ActionRejected("binding", "the action is not bound to the gripper register sent with its observation")
    a = np.asarray(o.get("action_norm"), np.float64)
    goal = np.asarray(o.get("execution_flange_pose", o.get("flange_pose")), np.float64)
    c = float(o.get("gripper_closure", np.nan))
    if a.shape != (7,) or goal.shape != (7,) or not (np.isfinite(a).all() and np.isfinite(goal).all() and np.isfinite(c)):
        raise ActionRejected("non_finite", "action, goal or gripper not finite (or malformed)")
    if np.abs(a).max() > 1.0:
        raise ActionRejected("range", f"|action_norm| {np.abs(a).max():.6f} > 1 (contract range)")
    if abs(float(np.linalg.norm(goal[3:])) - 1.0) > limits.quaternion_tolerance:
        raise ActionRejected("range", "goal quaternion is not unit norm")
    goal = as_pose7(goal)
    t_m = float(np.linalg.norm(goal[:3] - measured[:3]))
    r_deg = float(np.degrees(quat_angle(goal[3:], measured[3:])))
    if not limits.gripper_closure_min <= c <= limits.gripper_closure_max:
        raise ActionRejected("gripper", f"closure {c:.4f} outside [{limits.gripper_closure_min}, "
                                        f"{limits.gripper_closure_max}]")
    register = int(o.get("gripper_register"))
    if register != int(np.rint(GRIPPER_OPEN_REGISTER + GRIPPER_SPAN * c)):
        raise ActionRejected("gripper", f"register {register} != round(3 + 227 * {c:.4f})")
    age_ms = (int(now_ns) - int(request.observation_time_ns)) / 1e6
    if age_ms < 0:
        raise ActionRejected("future", f"observation {-age_ms:.1f} ms in the future: frames not on this clock")
    if age_ms > limits.max_action_age_ms:
        raise ActionRejected("expired", f"observation {age_ms:.1f} ms old > {limits.max_action_age_ms} ms")
    return Command(str(request.reference_id), session_id, int(request.observation_time_ns),
                   int(request.observation_time_ns + limits.max_action_age_ms * 1e6), tuple(measured.tolist()),
                   tuple(goal.tolist()), c, register, t_m, r_deg)
