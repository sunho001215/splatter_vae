"""Local high-rate loop of the robot PC (design + implementation; it only ever feeds a ``CommandSink``).

The policy runs at the camera rate (15 Hz) on the server; the robot needs fresh setpoints at its controller rate
(the recorder's teleop loop: 500 Hz, controller ``command_timeout_s`` 0.05). ``HighRateExecutor.tick`` turns the
newest validated ``Command`` (ABSOLUTE flange goal bound to its observation) into setpoints with the recorder's own
safety behaviour, inferred from its config and recordings (docs/robot_pc.md):

* the setpoint moves from the previous setpoint toward the goal by at most ``max_linear_speed`` / ``max_angular_speed``
  x tick period (teleop.max_linear_speed 0.3 m/s, max_angular_speed 1.5 rad/s), then is clamped to the lead envelope
  around the CURRENT measured pose (teleop.max_position_lead 0.04 m, max_rotation_lead_deg 20; every recorded command
  satisfies it). The goal itself is never re-derived from a newer pose: only clamped toward it;
* a command is replaced only by one of a NEWER observation; after ``expires_ns`` (observation age > the configured
  max age) the loop HOLDS: the setpoint becomes the measured pose (no motion), the gripper keeps its last command;
* nothing is sent before the first command (``tick`` returns None).

In this revision the only sinks are MockSink / DryRunSink; the live sink is UNBOUND (``sinks.LiveSink``).
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
    gripper_closure: float
    reference_id: str

    def to_json(self) -> dict:
        return {"t_ns": self.t_ns, "kind": self.kind, "pose7": list(self.pose7), "gripper_closure": self.gripper_closure,
                "reference_id": self.reference_id}


class HighRateExecutor:
    def __init__(self, sink, *, rate_hz: float = 500.0, max_linear_speed: float = 0.3, max_angular_speed: float = 1.5,
                 lead_m: float = 0.04, lead_rot_deg: float = 20.0):
        self.sink, self.period_s = sink, 1.0 / float(rate_hz)
        self.max_step_m, self.max_step_rad = float(max_linear_speed) * self.period_s, float(max_angular_speed) * \
            self.period_s
        self.lead_m, self.lead_rad = float(lead_m), float(np.deg2rad(lead_rot_deg))
        self.command: Command | None = None
        self._setpoint: np.ndarray | None = None
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
        if self.command is None:
            return None
        measured = as_pose7(measured_pose7)
        cmd = self.command
        if now_ns > cmd.expires_ns:
            sp = Setpoint(int(now_ns), "hold", tuple(measured.tolist()), cmd.gripper_closure, cmd.reference_id)
            self._setpoint = measured
            self.counts["hold"] += 1
        else:
            start = measured if self._setpoint is None else self._setpoint
            nxt = toward(start, cmd.goal_pose7, self.max_step_m, self.max_step_rad)
            nxt = toward(measured, nxt, self.lead_m, self.lead_rad)
            self._setpoint = nxt
            sp = Setpoint(int(now_ns), "track", tuple(nxt.tolist()), cmd.gripper_closure, cmd.reference_id)
            self.counts["track"] += 1
        self.sink.send(sp)
        return sp
