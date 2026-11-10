"""Local high-rate loop of the robot PC: the newest validated policy goal -> setpoints at the controller rate.

The policy runs at the camera rate (15 Hz); switch_controller wants a fresh target at its control rate (the
collector streams at 500 Hz; without a fresher action the controller holds the measured pose after ``target_dt``
0.05 s). ``HighRateExecutor.tick`` INTERPOLATES from the previously sent setpoint (the measured pose at the start of
an engagement) to the newest goal over one policy period (``interpolation_s``, linear position + slerp), then holds
the goal until a newer action arrives. There are NO speed, acceleration, lead or workspace caps here (user directive K,
2026-11-10): the goal is the policy's (the recorder's lead-limited flange target it was trained on).

Kept (each protects against a failure the controller itself does not handle):

* switch_controller compatibility: the controller REJECTS an EE target whose log6 error to the measured pose exceeds
  0.2 m / 0.5 rad and resets its command (action_manager_base.cpp:197-208); such a setpoint is never sent (counted
  ``controller_guard``; the controller keeps holding) - with a 5 % margin for the measured-pose age;
* robot-state health as the collector's ``state_ok`` (app.py:243-275): a measured sample older than
  ``state_timeout_s`` (0.05 s), a q/dq/tau signature bit-identical for > 0.25 s (FROZEN_STATE_S, app.py:54:
  switch_controller republishes its last state when stopped), a missing time / signature, a future or regressing state
  time, an invalid measured pose -> STOP, latched until ``rearm()``;
* expiry: after the command's ``expires_ns`` nothing is sent (the controller holds by itself after target_dt); a newer
  command starts again from the measured pose;
* a command is replaced only by one of a NEWER observation.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ops.robot_pc.poses import as_pose7, quat_angle, rotvec_between, slerp
from ops.robot_pc.validation import Command

CONTROLLER_REJECT_M, CONTROLLER_REJECT_RAD = 0.2, 0.5  # action_manager_base.cpp:207 (log6 norms, per ee)
CONTROLLER_MARGIN = 0.95
TARGET_DT_S = 0.05  # RobotAction target_dt (switch_controller.TARGET_DT_S): without a fresher action it holds


@dataclass(frozen=True)
class Setpoint:
    t_ns: int
    kind: str  # "track": toward the newest goal (an expired command produces no setpoint at all)
    pose7: tuple
    gripper_closure: float | None
    reference_id: str

    def to_json(self) -> dict:
        return {"t_ns": self.t_ns, "kind": self.kind, "pose7": list(self.pose7), "gripper_closure": self.gripper_closure,
                "reference_id": self.reference_id}


def log6_norms(desired7, current7) -> tuple[float, float]:
    """(|translation part|, |rotation part|) of pinocchio::log6(desired^-1 * current), as the controller computes."""
    d, c = as_pose7(desired7), as_pose7(current7)
    q_rel_vec = rotvec_between(d[3:], c[3:])  # base frame: R_c = exp(w) R_d
    R_d = _quat_to_mat(d[3:])
    w = R_d.T @ q_rel_vec  # rotation of desired^-1 current, in the desired frame
    p = R_d.T @ (c[:3] - d[:3])
    th = float(np.linalg.norm(w))
    W = np.array([[0, -w[2], w[1]], [w[2], 0, -w[0]], [-w[1], w[0], 0]])
    if th < 1e-3:  # series of (1 - th sin th / (2 (1 - cos th))) / th^2: no cancellation near 0
        c = 1.0 / 12.0 + th ** 2 / 720.0
    else:
        c = (1.0 - (th * np.sin(th)) / (2.0 * (1.0 - np.cos(th)))) / th ** 2
    v = (np.eye(3) - 0.5 * W + c * (W @ W)) @ p
    return float(np.linalg.norm(v)), th


def _quat_to_mat(q) -> np.ndarray:
    x, y, z, w = np.asarray(q, np.float64) / np.linalg.norm(q)
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


class HighRateExecutor:
    def __init__(self, sink, *, rate_hz: float = 500.0, interpolation_s: float = 1.0 / 15.0,
                 state_timeout_s: float = 0.05, frozen_state_s: float = 0.25):
        vals = (rate_hz, interpolation_s, state_timeout_s, frozen_state_s)
        if not all(np.isfinite(vals)) or min(vals) <= 0:
            raise ValueError("executor rate, interpolation time and state timeouts must be finite and positive")
        self.sink, self.period_ns = sink, int(1e9 / float(rate_hz))
        self.interp_ns = int(float(interpolation_s) * 1e9)
        self.state_timeout_ns, self.frozen_ns = int(state_timeout_s * 1e9), int(frozen_state_s * 1e9)
        self.command: Command | None = None
        self.stopped: str | None = None  # latched stop reason (stale / frozen / invalid state) until rearm()
        self._setpoint: np.ndarray | None = None
        self._from: np.ndarray | None = None  # start pose of the current interpolation segment
        self._seg_t0: int | None = None
        self._t_last: int | None = None
        self._sent_t: int | None = None  # time of the last emitted setpoint
        self.controller_hold_ns = int(TARGET_DT_S * 1e9)  # switch_controller holds the measured pose after this
        self._sig, self._sig_t, self._state_t = None, None, None
        self.counts = {"accepted": 0, "refused_older": 0, "track": 0, "expired": 0, "stopped": 0,
                       "controller_guard": 0}

    def accept(self, cmd: Command) -> bool:
        """Adopt a validated command only if it belongs to a NEWER observation; it starts a new interpolation."""
        if self.command is not None and cmd.observation_time_ns <= self.command.observation_time_ns:
            self.counts["refused_older"] += 1
            return False
        self.command = cmd
        self._from, self._seg_t0 = None, None  # set at the next tick (from the last setpoint / measured pose)
        self.counts["accepted"] += 1
        return True

    def rearm(self) -> None:
        """Explicitly clear a latched stop; a NEW command is required."""
        self.stopped, self.command, self._setpoint, self._from, self._seg_t0 = None, None, None, None, None
        self._sent_t = None
        self._t_last, self._sig, self._sig_t, self._state_t = None, None, None, None

    def _state_problem(self, now: int, state_t_ns, signature) -> str | None:
        if state_t_ns is None or signature is None:
            return "robot state without time / signature (frozen detection impossible)"
        t = int(state_t_ns)
        if t > now + 1_000_000:
            return "robot state stamped in the future (clock mismatch)"
        if now - t > self.state_timeout_ns:
            return "stale robot state"
        if self._sig_t is not None and t < self._state_t:
            return "robot state time went backwards"
        self._state_t = t
        if signature != self._sig:
            self._sig, self._sig_t = signature, t
        elif t - self._sig_t > self.frozen_ns:
            return "robot state frozen (switch_controller stopped?)"
        return None

    def _stop(self, why: str) -> None:
        self.stopped = why
        self.counts["stopped"] += 1

    def tick(self, now_ns: int, measured_pose7, *, state_t_ns: int | None, signature: bytes | None = None) -> \
            Setpoint | None:
        """One setpoint for the measured state sampled at ``state_t_ns``; None when nothing may be sent (no command,
        expired command, a setpoint the controller would reject, or a latched state stop)."""
        now = int(now_ns)
        if self.command is None or self.stopped is not None:
            return None
        problem = self._state_problem(now, state_t_ns, signature)
        try:
            measured = as_pose7(measured_pose7)
            if abs(float(np.linalg.norm(measured[3:])) - 1.0) > 1e-3:
                raise ValueError("bad quaternion")
        except ValueError as exc:
            problem = problem or f"invalid measured pose ({exc})"
        if problem is not None:
            self._stop(problem)
            return None
        self._t_last = now
        cmd = self.command
        if now > cmd.expires_ns:  # nothing of an expired command; the controller holds the measured pose itself
            self._setpoint, self._from, self._seg_t0 = None, None, None
            self.counts["expired"] += 1
            return None
        if self._from is None:
            # continue from the last SENT setpoint only while the controller still tracks it (sent within its
            # target_dt): after a pause (guard skips, expiry) it holds the MEASURED pose, so restart from there
            recent = self._sent_t is not None and now - self._sent_t <= self.controller_hold_ns
            self._from = self._setpoint if (self._setpoint is not None and recent) else measured
            self._seg_t0 = now
        goal = as_pose7(cmd.goal_pose7)
        a = 1.0 if self.interp_ns <= 0 else min(1.0, max(0.0, (now - self._seg_t0) / self.interp_ns))
        nxt = np.concatenate([self._from[:3] + a * (goal[:3] - self._from[:3]), slerp(self._from[3:], goal[3:], a)])
        dm, dr = log6_norms(nxt, measured)
        if not (np.isfinite(dm) and np.isfinite(dr)) or dm > CONTROLLER_MARGIN * CONTROLLER_REJECT_M or \
                dr > CONTROLLER_MARGIN * CONTROLLER_REJECT_RAD:
            self.counts["controller_guard"] += 1  # switch_controller would reject it and reset: never sent
            return None
        self._setpoint, self._sent_t = nxt, now
        sp = Setpoint(now, "track", tuple(nxt.tolist()), cmd.gripper_closure, cmd.reference_id)
        self.counts["track"] += 1
        self.sink.send(sp)
        return sp


def poses_far(a, b, max_m: float, max_rad: float) -> bool:
    a, b = np.asarray(a, np.float64), np.asarray(b, np.float64)
    return float(np.linalg.norm(a[:3] - b[:3])) > max_m or quat_angle(a[3:], b[3:]) > max_rad
