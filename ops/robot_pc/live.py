"""The live session of the robot PC (Directive J): live RealSense cameras + switch_controller RobotState (read-only
SUB) + measured Robotiq state -> policy server (newest observation, 15 Hz) -> validation -> the local 500 Hz executor
-> DryRunSink (dry run: computed, validated, recorded, NOT sent) or LiveSink (execution: ControllerLink + gripper
command), with the whole session recorded (``recorder.SessionRecorder``).

Stops (any of them ends the session; in execution the link then holds at the measured pose and requests IDLE):
the operator (``stop`` event: Ctrl-C, ``s``, SIGTERM), a latched executor stop (stale / frozen / invalid state, rate
conflict), ``max_failed_steps`` consecutive policy steps without an action (server / route failure), a camera without
frames for ``idle_timeout_s``, the step / duration budget.
"""

from __future__ import annotations

import threading
import time

import numpy as np

from ops.robot_pc.loop import PolicyLoop
from ops.robot_pc.robot_state import RobotStateSource


class StaticGripper:
    """DRY RUN ONLY: a labelled fixed register when no measured gripper is available (never in execution)."""

    origin = "static latency-only (no measured gripper; dry run)"

    def __init__(self, register: float = 3.0):
        self.register = float(register)

    def at_or_before(self, t_ns: int):
        return int(t_ns) - 10_000_000, self.register

    def rows(self, after_ns: int, up_to_ns: int) -> list:
        return []


class LiveState(RobotStateSource):
    """The live state reader bound to a gripper source (measured worker, or the labelled static one)."""

    kind = "live"

    def __init__(self, reader, gripper):
        self.reader, self.gripper = reader, gripper
        reader.gripper = gripper
        self.label = f"switch_controller RobotState + {getattr(gripper, 'origin', '?')}"

    def bound_state(self, t_obs_ns: int):
        try:
            return self.reader.bound_state(t_obs_ns)
        except LookupError as exc:  # no gripper sample yet
            from ops.robot_pc.robot_state import StateUnavailable

            raise StateUnavailable(str(exc)) from exc

    def rows(self, after_ns: int, up_to_ns: int):
        return self.reader.rows(after_ns, up_to_ns)

    def latest(self):
        s = self.reader.samples
        return s[-1] if s else None


class LiveSession:
    def __init__(self, *, client, cameras, cams: list, state: LiveState, limits, executor, sink, recorder=None,
                 sent: bool = False, max_failed_steps: int = 5, idle_timeout_s: float = 2.0, on_status=None):
        self.client, self.cameras, self.cams, self.state = client, cameras, list(cams), state
        self.executor, self.sink, self.recorder, self.sent = executor, sink, recorder, bool(sent)
        self.loop = PolicyLoop(client, state, limits, _NullSink(), executor=executor)
        self.max_failed, self.idle_timeout_s, self.on_status = int(max_failed_steps), float(idle_timeout_s), on_status
        self.stop_event = threading.Event()
        self.stop_reason: str | None = None
        self.rows: list = []
        self._exec_thread: threading.Thread | None = None
        self._rec_robot_t = self._rec_grip_t = 0

    # ------------------------------------------------------------------------------------------------ executor
    def _executor_loop(self) -> None:
        period = self.executor.period_ns / 1e9
        while not self.stop_event.is_set():
            t0 = time.monotonic()
            s = self.state.latest()
            if s is not None:
                sp = self.executor.tick(time.monotonic_ns(), s["flange_pose7"], state_t_ns=s["t_ns"],
                                        signature=s["signature"])
                if sp is not None:
                    try:
                        self.sink.send(sp)
                    except Exception as exc:  # noqa: BLE001 - a failed send ends the session
                        self.stop(f"command send failed: {exc}")
                    if self.recorder is not None:
                        self.recorder.executed(sp.t_ns, sp.kind, sp.pose7, sp.gripper_closure, self.sent, "")
                if self.executor.stopped and not self.stop_event.is_set():
                    self.stop(f"executor stop: {self.executor.stopped}")
            time.sleep(max(0.0, period - (time.monotonic() - t0)))

    def stop(self, reason: str) -> None:
        if self.stop_reason is None:
            self.stop_reason = reason
            if self.recorder is not None:
                self.recorder.event(time.monotonic_ns(), f"stop: {reason}")
        self.stop_event.set()

    # ------------------------------------------------------------------------------------------------ recording
    def _record_streams(self) -> None:
        if self.recorder is None:
            return
        for s in [s for s in list(self.state.reader.samples) if s["t_ns"] > self._rec_robot_t]:
            self.recorder.robot_state(s["t_ns"], s["seq"], s["q"], s.get("dq"), s.get("tau"), s["flange_pose7"],
                                      s["ee_pose7"])
            self._rec_robot_t = s["t_ns"]
        g = self.state.gripper
        for t, st in [x for x in list(getattr(g, "samples", ())) if x[0] > self._rec_grip_t]:
            self.recorder.gripper(t, int(st.position), dict(st.__dict__))
            self._rec_grip_t = t

    # ------------------------------------------------------------------------------------------------ main loop
    def run(self, *, steps: int | None = None, duration_s: float | None = None) -> list:
        self._exec_thread = threading.Thread(target=self._executor_loop, name="executor-500hz", daemon=True)
        self._exec_thread.start()
        t_end = None if duration_s is None else time.monotonic() + float(duration_s)
        last_frame = dict.fromkeys(self.cams, time.monotonic())
        failed = 0
        try:
            while not self.stop_event.is_set():
                if (steps is not None and len(self.rows) >= steps) or (t_end is not None and time.monotonic() > t_end):
                    self.stop("budget reached")
                    break
                newest, acquire = None, 0.0
                for cam in self.cams:
                    for cap in self.cameras.drain(cam):
                        last_frame[cam] = time.monotonic()
                        t = cap["capture_ns"]
                        try:  # informational per-frame sample (the request binds its own)
                            fs = self.state.bound_state(t)
                            pose, grip = fs.pose7, fs.gripper_register
                        except Exception:  # noqa: BLE001 - no bindable state at this frame
                            pose, grip = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0), 0.0
                        self.client.add_capture(cam, cap, pose7=pose, gripper_register=grip,
                                                metadata={"state_source": "live"})
                        acquire = max(acquire, (cap["arrival_ns"] - t) / 1e6)
                        if self.recorder is not None:
                            if "native_rgb" in cap:
                                self.recorder.frame(cam, cap["frame_id"], t, cap["native_rgb"], None, None,
                                                    cap.get("metadata"))
                            self.recorder.observation(cam, cap["frame_id"], t, cap["rgb"], cap.get("depth"))
                        if cam == self.cams[0]:
                            newest = ({cam: cap["frame_id"]}, t, f"{cam}:{cap['frame_id']}")
                self._record_streams()
                stale = [c for c in self.cams if time.monotonic() - last_frame[c] > self.idle_timeout_s]
                if stale:
                    self.stop(f"cameras {stale} sent no frame for {self.idle_timeout_s} s")
                    break
                if newest is None:
                    time.sleep(0.001)
                    continue
                row = self.loop.step(*newest, acquire_ms=acquire)
                self.rows.append(row)
                if self.recorder is not None:
                    self.recorder.step(row)
                    if row.get("command"):
                        self.recorder.command(row["command"])
                failed = 0 if row.get("status") == "ok" else failed + 1
                if self.on_status is not None:
                    self.on_status(row)
                if row.get("terminal"):
                    self.stop("session taken over by another client")
                elif failed >= self.max_failed:
                    self.stop(f"{failed} consecutive steps without an action ({row.get('code') or row.get('status')})")
        except KeyboardInterrupt:
            self.stop("operator Ctrl-C")
        finally:
            self.stop_event.set()
            if self._exec_thread is not None:
                self._exec_thread.join(timeout=2.0)
            self._record_streams()
        return self.rows


class _NullSink:
    """Validated commands go to the executor (``PolicyLoop(executor=...)``); only its setpoints reach the real sink."""

    def send(self, item) -> None:
        pass


def still(state: LiveState, max_dq: float = 0.02) -> bool:
    s = state.latest()
    return s is not None and s.get("dq") is not None and float(np.abs(s["dq"]).max()) < max_dq
