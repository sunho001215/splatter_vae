"""Local high-rate loop of the robot PC (design + implementation; it only ever feeds a ``CommandSink``).

The policy runs at the camera rate (15 Hz) on the server; the robot needs fresh setpoints at its controller rate
(the collector's control loop: 500 Hz, controller hold after ``target_dt`` 0.05 s). ``HighRateExecutor.tick`` turns
the newest validated ``Command`` (ABSOLUTE flange goal bound to its observation) into setpoints with the robot PC
collector's own command shaping (panda-teleop/teleop_collector/teleop_collector/teleop.py, read-only; lines cited in
docs/robot_pc.md):

* ``RateLimiter`` (teleop.py:154-190): velocity- AND acceleration-limited tracking - speed capped by ``max_speed`` and
  the braking speed sqrt(2 a d) (and d / dt), the velocity change per tick by ``a dt``; linear 0.3 m/s (site.yaml),
  4 m/s^2 (MAX_LINEAR_ACCEL, teleop.py:39), angular 1.5 rad/s, 15 rad/s^2 (MAX_ANGULAR_ACCEL, teleop.py:40);
* the goal is first clamped to 2 x the lead around the measured pose, the limiter output to 1 x the lead (0.04 m,
  20 deg) with the limiter state made consistent (``constrain``), exactly as ``TeleopMapper.step`` (teleop.py:254-275);
  the goal is never re-derived from a newer pose;
* robot-state health as the collector's ``state_ok`` (app.py:243-275): a measured sample older than
  ``state_timeout_s`` (0.05 s) or a state signature (q, dq, tau_ext) bit-identical for > 0.25 s (FROZEN_STATE_S,
  app.py:54; switch_controller republishes its last state when stopped) STOPS the loop: nothing more is sent and the
  stop is LATCHED until ``rearm()`` (the collector requires a new pedal press);
* a command is replaced only by one of a NEWER observation; after ``expires_ns`` the loop HOLDS at the measured pose
  and applies nothing of the expired command (the gripper keeps the last closure actually emitted);
* nothing is sent before the first command; the step uses the ELAPSED time since the previous tick (capped at 2
  periods); a repeated / earlier timestamp re-sends the previous setpoint without moving.

In this revision the only sinks are MockSink / DryRunSink; ``sinks.LiveSink`` is refused (``safety.EXECUTION_PERMITTED``).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ops.robot_pc.poses import as_pose7, rotate_left, rotvec_between, toward
from ops.robot_pc.validation import Command


@dataclass(frozen=True)
class Setpoint:
    t_ns: int
    kind: str  # "track" (toward the bound goal) | "hold" (command expired: stay at the measured pose)
    pose7: tuple
    gripper_closure: float | None  # None: no gripper target (hold before any tracked command)
    reference_id: str

    def to_json(self) -> dict:
        return {"t_ns": self.t_ns, "kind": self.kind, "pose7": list(self.pose7), "gripper_closure": self.gripper_closure,
                "reference_id": self.reference_id}


def _clamp(v: np.ndarray, limit: float) -> np.ndarray:  # teleop.clamp_norm
    n = float(np.linalg.norm(v))
    return v if n <= limit or n == 0.0 else v * (limit / n)


class RateLimiter:
    """numpy port of the collector's ``teleop.RateLimiter`` (velocity- and acceleration-limited goal tracking)."""

    def __init__(self, max_speed: float, max_accel: float, max_ang_speed: float, max_ang_accel: float):
        self.v_max, self.a_max, self.w_max, self.alpha_max = (float(x) for x in
                                                              (max_speed, max_accel, max_ang_speed, max_ang_accel))
        self.reset(np.array([0, 0, 0, 0, 0, 0, 1.0]))

    def reset(self, pose7) -> None:
        p = as_pose7(pose7)
        self.p, self.q = p[:3].copy(), p[3:] / np.linalg.norm(p[3:])
        self.v, self.w = np.zeros(3), np.zeros(3)

    @staticmethod
    def _limited_velocity(error, current_v, v_max, a_max, dt):
        dist = float(np.linalg.norm(error))
        desired = np.zeros(3) if dist < 1e-12 else error * (min(v_max, np.sqrt(2.0 * a_max * dist), dist / dt) / dist)
        return current_v + _clamp(desired - current_v, a_max * dt)

    def step(self, goal7, dt: float) -> np.ndarray:
        g = as_pose7(goal7)
        self.v = self._limited_velocity(g[:3] - self.p, self.v, self.v_max, self.a_max, dt)
        self.p = self.p + self.v * dt
        self.w = self._limited_velocity(rotvec_between(self.q, g[3:]), self.w, self.w_max, self.alpha_max, dt)
        self.q = rotate_left(self.q, self.w * dt)
        return np.concatenate([self.p, self.q])

    def constrain(self, pose7, dt: float) -> None:
        """State replaced after an external clamp, velocities kept consistent (teleop.RateLimiter.constrain)."""
        p = as_pose7(pose7)
        self.v = _clamp(self.v + (p[:3] - self.p) / dt, self.v_max)
        self.w = _clamp(self.w + rotvec_between(self.q, p[3:]) / dt, self.w_max)
        self.p, self.q = p[:3].copy(), p[3:] / np.linalg.norm(p[3:])


class HighRateExecutor:
    def __init__(self, sink, *, rate_hz: float = 500.0, max_linear_speed: float = 0.3, max_angular_speed: float = 1.5,
                 max_linear_accel: float = 4.0, max_angular_accel: float = 15.0, lead_m: float = 0.04,
                 lead_rot_deg: float = 20.0, state_timeout_s: float = 0.05, frozen_state_s: float = 0.25):
        vals = (rate_hz, max_linear_speed, max_angular_speed, max_linear_accel, max_angular_accel, lead_m,
                lead_rot_deg, state_timeout_s, frozen_state_s)
        if not all(np.isfinite(vals)) or min(vals) <= 0:
            raise ValueError("executor rates and limits must be finite and positive")
        self.sink, self.period_ns = sink, int(1e9 / float(rate_hz))
        self.limiter = RateLimiter(max_linear_speed, max_linear_accel, max_angular_speed, max_angular_accel)
        self.lead_m, self.lead_rad = float(lead_m), float(np.deg2rad(lead_rot_deg))
        self.state_timeout_ns, self.frozen_ns = int(state_timeout_s * 1e9), int(frozen_state_s * 1e9)
        self.command: Command | None = None
        self.stopped: str | None = None  # latched stop reason (stale / frozen state) until rearm()
        self._setpoint: np.ndarray | None = None
        self._t_last: int | None = None
        self._gripper: float | None = None  # last gripper closure actually emitted in a "track" setpoint
        self._sig, self._sig_t = None, None
        self.counts = {"accepted": 0, "refused_older": 0, "track": 0, "hold": 0, "stopped": 0, "resent": 0}

    def accept(self, cmd: Command) -> bool:
        """Adopt a validated command only if it belongs to a NEWER observation than the current one."""
        if self.command is not None and cmd.observation_time_ns <= self.command.observation_time_ns:
            self.counts["refused_older"] += 1
            return False
        self.command = cmd
        self.counts["accepted"] += 1
        return True

    def rearm(self) -> None:
        """Explicitly clear a latched stop (the collector's fresh pedal press); a NEW command is required."""
        self.stopped, self.command, self._setpoint, self._t_last, self._sig, self._sig_t = (None,) * 6

    def _state_problem(self, now: int, state_t_ns, signature) -> str | None:
        if state_t_ns is None or now - int(state_t_ns) > self.state_timeout_ns:
            return "stale robot state"
        if signature is not None:
            if signature != self._sig:
                self._sig, self._sig_t = signature, int(state_t_ns)
            elif int(state_t_ns) - self._sig_t > self.frozen_ns:
                return "robot state frozen (switch_controller stopped?)"
        return None

    def tick(self, now_ns: int, measured_pose7, *, state_t_ns: int | None, signature: bytes | None = None) -> \
            Setpoint | None:
        """One setpoint for the measured state sampled at ``state_t_ns`` (its q/dq/tau ``signature`` for the frozen
        check); None when nothing may be sent (no command yet, or a latched stop)."""
        now = int(now_ns)
        if self.command is None or self.stopped is not None:
            return None
        problem = self._state_problem(now, state_t_ns, signature)
        if problem is not None:
            self.stopped = problem
            self.counts["stopped"] += 1
            return None
        measured, cmd = as_pose7(measured_pose7), self.command
        if self._t_last is not None and now <= self._t_last:  # repeated / earlier tick: re-send, never move
            self.counts["resent"] += 1
            sp = Setpoint(now, "track" if now <= cmd.expires_ns else "hold", tuple(self._setpoint.tolist()),
                          self._gripper, cmd.reference_id)
            self.sink.send(sp)
            return sp
        dt = min(now - self._t_last, 2 * self.period_ns) / 1e9 if self._t_last is not None else self.period_ns / 1e9
        self._t_last = now
        if now > cmd.expires_ns:
            self._setpoint = measured
            self.limiter.reset(measured)
            sp = Setpoint(now, "hold", tuple(measured.tolist()), self._gripper, cmd.reference_id)
            self.counts["hold"] += 1
        else:
            if self._setpoint is None:
                self.limiter.reset(measured)  # engage at the measured pose with zero velocity
            goal = toward(measured, cmd.goal_pose7, 2 * self.lead_m, 2 * self.lead_rad)
            raw = self.limiter.step(goal, dt)
            nxt = toward(measured, raw, self.lead_m, self.lead_rad)
            if not np.array_equal(nxt, raw):
                self.limiter.constrain(nxt, dt)
            self._setpoint = nxt
            self._gripper = cmd.gripper_closure
            sp = Setpoint(now, "track", tuple(nxt.tolist()), cmd.gripper_closure, cmd.reference_id)
            self.counts["track"] += 1
        self.sink.send(sp)
        return sp
