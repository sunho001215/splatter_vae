"""Following detection for the execution mode: is the robot actually tracking the streamed setpoints?

The executor leashes every setpoint to within ``lead_m`` / ``lead_rad`` of the MEASURED pose, so the tracking error can
never exceed the lead by more than the leash tolerance; an "error > lead + margin" rule could never fire. Under the
controller's impedance the error legitimately sits AT the lead while the robot moves (test 3, 2026-11-10 21:19,
inference_20261110_211938: lag ~0.42 s at 0.05 m/s -> ~19 mm of a 20 mm lead for > 1 s while the robot moved toward the
setpoint at 0.030-0.037 m/s). What distinguishes a robot that does not follow is the ABSENCE OF PROGRESS:

* ``mode not accepted``: since engagement the setpoint has moved away (>= ``cmd_motion_m`` / ``cmd_motion_rad``) but the
  robot has not moved at all (< ``still_m`` / ``still_rad`` from its engage pose) for ``window_s`` - the controller is
  not tracking (mode 3 not accepted, or a controller-side rejection);
* ``not progressing`` (position or rotation, separately): the error is >= ``err_frac`` x lead_m (position) /
  ``err_frac_rot`` x lead_rad (rotation) while the robot's velocity component TOWARD the setpoint stays below
  ``progress_frac`` x the profile's max speed for ``window_s`` - blocked, in contact, or far beyond the impedance lag.
  Rotation uses 0.9 x lead: in test 3 the orientation error drifted to ~5 deg of the 8 deg lead with ~0 rate toward the
  setpoint (impedance coupling during translation) while the robot tracked fine; only an orientation pinned at the
  leash without progress is a stall.

Velocities are finite differences of the measured flange over ``vel_window_s`` (host receipt times). Pure numpy, no I/O.
"""

from __future__ import annotations

import collections

import numpy as np

from ops.robot_pc.poses import as_pose7, quat_angle, rotvec_between


class FollowMonitor:
    def __init__(self, *, lead_m: float, lead_rad: float, max_speed: float, max_ang_speed: float,
                 window_s: float = 1.0, vel_window_s: float = 0.05, err_frac: float = 0.5, err_frac_rot: float = 0.9,
                 progress_frac: float = 0.1,
                 still_m: float = 0.002, still_rad: float = np.radians(0.5), cmd_motion_m: float = 0.005,
                 cmd_motion_rad: float = np.radians(2.0)):
        self.err_m, self.err_rad = err_frac * float(lead_m), err_frac_rot * float(lead_rad)
        self.v_min, self.w_min = progress_frac * float(max_speed), progress_frac * float(max_ang_speed)
        self.window_ns, self.vel_window_ns = int(window_s * 1e9), int(vel_window_s * 1e9)
        self.still_m, self.still_rad = float(still_m), float(still_rad)
        self.cmd_m, self.cmd_rad = float(cmd_motion_m), float(cmd_motion_rad)
        self.hist: collections.deque = collections.deque()
        self.engage = self.sp0 = None
        self.robot_moved = False
        self.since = {"mode": None, "pos": None, "rot": None}
        self.last: dict = {}

    def pause(self) -> None:
        """No setpoint this tick (expired command): the continuous-condition timers restart."""
        self.since = dict.fromkeys(self.since)

    def _timer(self, key: str, cond: bool, t: int) -> bool:
        if not cond:
            self.since[key] = None
            return False
        if self.since[key] is None:
            self.since[key] = t
        return t - self.since[key] >= self.window_ns

    def update(self, t_ns: int, setpoint7, measured7) -> str | None:
        """One sent setpoint and the measured flange used for it; returns a stop reason or None."""
        t, sp, m = int(t_ns), as_pose7(setpoint7), as_pose7(measured7)
        if self.engage is None:
            self.engage, self.sp0 = m, sp
        self.hist.append((t, m))
        while len(self.hist) > 2 and t - self.hist[1][0] >= self.vel_window_ns:
            self.hist.popleft()
        t_old, m_old = self.hist[0]
        dt = (t - t_old) / 1e9
        err = sp[:3] - m[:3]
        e = float(np.linalg.norm(err))
        r_err = rotvec_between(m[3:], sp[3:])
        er = float(np.linalg.norm(r_err))
        v_toward = w_toward = None
        if dt >= 0.5 * self.vel_window_ns / 1e9:
            v = (m[:3] - m_old[:3]) / dt
            w = rotvec_between(m_old[3:], m[3:]) / dt
            v_toward = float(v @ err / e) if e > 1e-9 else 0.0
            w_toward = float(w @ r_err / er) if er > 1e-9 else 0.0
        moved = float(np.linalg.norm(m[:3] - self.engage[:3])) >= self.still_m or \
            quat_angle(m[3:], self.engage[3:]) >= self.still_rad
        self.robot_moved = self.robot_moved or moved
        cmd_moved = float(np.linalg.norm(sp[:3] - self.sp0[:3])) >= self.cmd_m or \
            quat_angle(sp[3:], self.sp0[3:]) >= self.cmd_rad
        self.last = {"err_m": e, "err_rad": er, "v_toward": v_toward, "w_toward": w_toward,
                     "robot_moved": self.robot_moved}
        if self._timer("mode", not self.robot_moved and cmd_moved, t):
            return ("robot has not moved since engagement although the command moved "
                    f"{np.linalg.norm(sp[:3] - self.sp0[:3]) * 1e3:.1f} mm / "
                    f"{np.degrees(quat_angle(sp[3:], self.sp0[3:])):.1f} deg for "
                    f"{self.window_ns / 1e9:.1f} s: controller mode not accepted?")
        if v_toward is None:
            return None
        if self._timer("pos", e >= self.err_m and v_toward < self.v_min, t):
            return (f"robot not progressing toward the command: position error {e * 1e3:.1f} mm, speed toward it "
                    f"{v_toward * 1e3:.1f} mm/s < {self.v_min * 1e3:.1f} for {self.window_ns / 1e9:.1f} s "
                    "(blocked / contact?)")
        if self._timer("rot", er >= self.err_rad and w_toward < self.w_min, t):
            return (f"robot not progressing toward the commanded orientation: error {np.degrees(er):.1f} deg, "
                    f"rate toward it {np.degrees(w_toward):.2f} deg/s < {np.degrees(self.w_min):.2f} for "
                    f"{self.window_ns / 1e9:.1f} s (blocked / contact?)")
        return None
