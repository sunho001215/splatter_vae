"""Local high-rate loop of the robot PC (design + implementation; it only ever feeds a ``CommandSink``).

The policy runs at the camera rate (15 Hz) on the server; the robot needs fresh setpoints at its controller rate
(the recorder's teleop loop: 500 Hz, controller ``command_timeout_s`` 0.05). ``HighRateExecutor.tick`` turns the
newest validated ``Command`` (ABSOLUTE flange goal bound to its observation) into setpoints with the recorder's own
safety behaviour, inferred from its config and recordings (docs/robot_pc.md):

* the goal is clamped to the lead envelope around the CURRENT measured pose (teleop.max_position_lead 0.04 m,
  max_rotation_lead_deg 20; every recorded command satisfies it), and the setpoint moves toward that clamped target by
  at most ``max_linear_speed`` / ``max_angular_speed`` x the elapsed tick time (teleop 0.3 m/s, 1.5 rad/s); the goal is
  never re-derived from a newer pose. (The collector's RateLimiter also bounds acceleration, 4 m/s^2 / 15 rad/s^2:
  not reproduced here; a future execute mode must add it.)
* a command is replaced only by one of a NEWER observation; after ``expires_ns`` (observation age > the configured
  max age) the loop HOLDS: the setpoint becomes the measured pose (no motion) and nothing of the expired command is
  applied (the gripper keeps the last closure actually emitted);
* nothing is sent before the first command (``tick`` returns None).

In this revision the only sinks are MockSink / DryRunSink; ``sinks.LiveSink`` is refused (``safety.EXECUTION_PERMITTED``).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ops.robot_pc.poses import as_pose7, toward
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


class HighRateExecutor:
    def __init__(self, sink, *, rate_hz: float = 500.0, max_linear_speed: float = 0.3, max_angular_speed: float = 1.5,
                 lead_m: float = 0.04, lead_rot_deg: float = 20.0):
        vals = (rate_hz, max_linear_speed, max_angular_speed, lead_m, lead_rot_deg)
        if not all(np.isfinite(vals)) or min(vals) <= 0:
            raise ValueError("executor rates and limits must be finite and positive")
        self.sink, self.period_ns = sink, int(1e9 / float(rate_hz))
        self.v_lin, self.v_ang = float(max_linear_speed), float(max_angular_speed)
        self.lead_m, self.lead_rad = float(lead_m), float(np.deg2rad(lead_rot_deg))
        self.command: Command | None = None
        self._setpoint: np.ndarray | None = None
        self._t_last: int | None = None
        self._gripper: float | None = None  # last gripper closure actually emitted in a "track" setpoint
        self.counts = {"accepted": 0, "refused_older": 0, "track": 0, "hold": 0}

    def accept(self, cmd: Command) -> bool:
        """Adopt a validated command only if it belongs to a NEWER observation than the current one."""
        if self.command is not None and cmd.observation_time_ns <= self.command.observation_time_ns:
            self.counts["refused_older"] += 1
            return False
        self.command = cmd
        self.counts["accepted"] += 1
        return True

    def tick(self, now_ns: int, measured_pose7) -> Setpoint | None:
        """One setpoint. Motion per tick is bounded by the speed limits x the ELAPSED time since the previous tick
        (capped at 2 periods; a repeated or earlier timestamp moves nothing); the lead-envelope clamp acts on the
        TARGET, so the setpoint itself never jumps. An expired command contributes nothing: hold at the measured pose
        with the last emitted gripper closure (None if no gripper target was ever emitted)."""
        if self.command is None:
            return None
        measured = as_pose7(measured_pose7)
        cmd, now = self.command, int(now_ns)
        dt_ns = 0 if self._t_last is None else min(max(0, now - self._t_last), 2 * self.period_ns)
        if self._t_last is None or now > self._t_last:
            self._t_last = now
        if now > cmd.expires_ns:
            sp = Setpoint(now, "hold", tuple(measured.tolist()), self._gripper, cmd.reference_id)
            self._setpoint = measured
            self.counts["hold"] += 1
        else:
            start = measured if self._setpoint is None else self._setpoint
            target = toward(measured, cmd.goal_pose7, self.lead_m, self.lead_rad)  # recorder's lead envelope
            dt = (dt_ns if self._setpoint is not None else self.period_ns) / 1e9
            nxt = toward(start, target, self.v_lin * dt, self.v_ang * dt)
            self._setpoint = nxt
            self._gripper = cmd.gripper_closure
            sp = Setpoint(now, "track", tuple(nxt.tolist()), cmd.gripper_closure, cmd.reference_id)
            self.counts["track"] += 1
        self.sink.send(sp)
        return sp
