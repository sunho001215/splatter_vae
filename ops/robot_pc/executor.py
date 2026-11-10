"""Local high-rate loop of the robot PC (design + implementation; it only ever feeds a ``CommandSink``).

The policy runs at the camera rate (15 Hz) on the server; the robot needs fresh setpoints at its controller rate
(the collector's control loop: 500 Hz, controller hold after ``target_dt`` 0.05 s). ``HighRateExecutor.tick`` turns
the newest validated ``Command`` (ABSOLUTE flange goal bound to its observation) into setpoints with the robot PC
collector's own command shaping (panda-teleop/teleop_collector/teleop_collector/teleop.py, read-only; lines cited in
docs/robot_pc.md):

* ``RateLimiter`` (teleop.py:154-190): velocity- AND acceleration-limited tracking - speed capped by ``max_speed`` and
  the braking speed (the discrete-time form, so no overshoot; the collector uses sqrt(2 a d)) and d / dt, the velocity
  change per tick by ``a dt``; linear 0.3 m/s (site.yaml),
  4 m/s^2 (MAX_LINEAR_ACCEL, teleop.py:39), angular 1.5 rad/s, 15 rad/s^2 (MAX_ANGULAR_ACCEL, teleop.py:40);
* the goal is clamped to the lead around the measured pose (0.04 m, 20 deg) so the limiter brakes INTO the envelope
  (the collector clamps the goal at 2 x lead and leashes the output at 1 x, ``TeleopMapper.step`` teleop.py:254-275,
  which can decelerate beyond its own limits); the output is still leashed at 1 x lead with ``constrain``, and if the
  EMITTED setpoint would then break the speed / acceleration limits (the measured pose runs away: robot not
  following), the loop stops (latched) instead of jumping; the goal is never re-derived from a newer pose;
* robot-state health as the collector's ``state_ok`` (app.py:243-275): a measured sample older than
  ``state_timeout_s`` (0.05 s) or a state signature (q, dq, tau_ext) bit-identical for > 0.25 s (FROZEN_STATE_S,
  app.py:54; switch_controller republishes its last state when stopped) STOPS the loop; so do (fail closed) a missing
  time or signature, a state stamped in the future, a state time going backwards and an invalid measured pose. Nothing
  more is sent and the stop is LATCHED until ``rearm()`` (the collector requires a new pedal press);
* a command is replaced only by one of a NEWER observation; after ``expires_ns`` the loop HOLDS at the measured pose
  and applies nothing of the expired command (the gripper keeps the last closure actually emitted);
* nothing is sent before the first command; the step uses the ELAPSED time since the previous tick (capped at 2
  periods); a repeated / earlier timestamp re-sends the previous setpoint without moving (only while it is still
  inside the lead envelope of the current measurement).

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


LEASH_TOL_M, LEASH_TOL_RAD = 1e-5, 1e-4  # the leash ignores sub-10-um / 0.006-deg numerical overshoot at the envelope


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
        """The collector's rule with the DISCRETE braking speed a dt (sqrt(1/4 + 2 d / (a dt^2)) - 1/2) instead of
        sqrt(2 a d): with velocity steps of at most a dt the goal is then reached without overshoot, so the lead leash
        never has to cut the velocity beyond the acceleration limit (the continuous rule overshoots by a few um)."""
        dist = float(np.linalg.norm(error))
        if dist < 1e-12:
            desired = np.zeros(3)
        else:
            brake = a_max * dt * (np.sqrt(0.25 + 2.0 * dist / (a_max * dt * dt)) - 0.5)
            desired = error * (min(v_max, brake, dist / dt) / dist)
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
        self._sig, self._sig_t, self._state_t = None, None, None
        self._kind, self._v_emit, self._w_emit = "track", np.zeros(3), np.zeros(3)
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
        self.stopped, self.command, self._setpoint, self._t_last = None, None, None, None
        self._sig, self._sig_t, self._state_t = None, None, None

    def _state_problem(self, now: int, state_t_ns, signature) -> str | None:
        """The collector's state health + fail-closed edge cases (no signature, future / regressing state times)."""
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

    def _within_rates(self, nxt: np.ndarray, dt: float) -> bool:
        """The EMITTED setpoint (after the lead clamp) must respect the speed and acceleration limits."""
        if self._setpoint is None:
            return True
        lim, eps = self.limiter, 1e-9
        v = (nxt[:3] - self._setpoint[:3]) / dt
        w = rotvec_between(self._setpoint[3:], nxt[3:]) / dt
        return bool(np.linalg.norm(v) <= lim.v_max + eps and np.linalg.norm(v - self._v_emit) <= lim.a_max * dt + eps
                    and np.linalg.norm(w) <= lim.w_max + 1e-6
                    and np.linalg.norm(w - self._w_emit) <= lim.alpha_max * dt + 1e-6)

    def tick(self, now_ns: int, measured_pose7, *, state_t_ns: int | None, signature: bytes | None = None) -> \
            Setpoint | None:
        """One setpoint for the measured state sampled at ``state_t_ns`` (its q/dq/tau ``signature`` for the frozen
        check); None when nothing may be sent: no command yet, or a LATCHED stop (stale / frozen / invalid state, or
        lead envelope and rate limits that cannot both hold, i.e. the robot does not follow)."""
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
        cmd = self.command
        if self._t_last is not None and now <= self._t_last:  # repeated / earlier tick: re-send, never move
            if self._setpoint is None or poses_far(self._setpoint, measured, self.lead_m, self.lead_rad):
                return None
            self.counts["resent"] += 1
            sp = Setpoint(now, self._kind, tuple(self._setpoint.tolist()), self._gripper, cmd.reference_id)
            self.sink.send(sp)
            return sp
        dt = min(now - self._t_last, 2 * self.period_ns) / 1e9 if self._t_last is not None else self.period_ns / 1e9
        self._t_last = now
        if now > cmd.expires_ns:  # the collector's "final hold at the measured pose": a stop, exempt from rates
            self._setpoint, self._kind = measured, "hold"
            self._v_emit, self._w_emit = np.zeros(3), np.zeros(3)
            self.limiter.reset(measured)
            sp = Setpoint(now, "hold", tuple(measured.tolist()), self._gripper, cmd.reference_id)
            self.counts["hold"] += 1
        else:
            if self._setpoint is None or self._kind == "hold":
                self.limiter.reset(measured)  # (re)engage at the measured pose with zero velocity
                self._setpoint, self._v_emit, self._w_emit = measured, np.zeros(3), np.zeros(3)
            goal = toward(measured, cmd.goal_pose7, self.lead_m, self.lead_rad)  # brake INTO the lead envelope
            raw = self.limiter.step(goal, dt)
            nxt = toward(measured, raw, self.lead_m + LEASH_TOL_M, self.lead_rad + LEASH_TOL_RAD)
            if poses_far(raw, nxt, 1e-12, 1e-12):
                self.limiter.constrain(nxt, dt)
            if not self._within_rates(nxt, dt):
                self._stop("lead envelope and rate limits conflict (robot not following the command)")
                return None
            self._v_emit = (nxt[:3] - self._setpoint[:3]) / dt
            self._w_emit = rotvec_between(self._setpoint[3:], nxt[3:]) / dt
            self._setpoint, self._kind = nxt, "track"
            self._gripper = cmd.gripper_closure
            sp = Setpoint(now, "track", tuple(nxt.tolist()), cmd.gripper_closure, cmd.reference_id)
            self.counts["track"] += 1
        self.sink.send(sp)
        return sp


def poses_far(a, b, max_m: float, max_rad: float) -> bool:
    a, b = np.asarray(a, np.float64), np.asarray(b, np.float64)
    from ops.robot_pc.poses import quat_angle

    return float(np.linalg.norm(a[:3] - b[:3])) > max_m or quat_angle(a[3:], b[3:]) > max_rad
