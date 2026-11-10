"""The live session of the robot PC (Directive J): live RealSense cameras + switch_controller RobotState (read-only
SUB) + measured Robotiq state -> policy server (newest observation, 15 Hz) -> validation -> the local 500 Hz executor
-> DryRunSink (dry run: computed, validated, recorded, NOT sent; the executor's measured pose is then an ideal
follower of its own setpoints, since the real robot does not move - health checks still use the real stream) or
LiveSink (execution: ControllerLink + gripper command), with the whole session recorded (``recorder.SessionRecorder``).

Stops (any of them ends the session; in execution the link then holds at the measured pose and requests IDLE):
the operator (``stop`` event: Ctrl-C, ``s``, SIGTERM), a latched executor stop (stale / frozen / invalid state, rate
conflict), ``max_failed_steps`` consecutive policy steps without an action (server / route failure), a camera without
frames for ``idle_timeout_s``, the step / duration budget.
"""

from __future__ import annotations

import json
import logging
import threading
import time
from dataclasses import dataclass

import numpy as np

from ops.robot_pc.executor import Setpoint
from ops.robot_pc.loop import PolicyLoop
from ops.robot_pc.robot_state import RobotStateSource
from ops.robot_pc.safety import ExecutionRefused
from ops.robot_pc.sinks import LiveSink as _BaseLiveSink

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class LiveSetpoint(Setpoint):
    """The executor's setpoint with the expiry of the command that produced it."""
    expires_ns: int

    def to_json(self) -> dict:
        return {**super().to_json(), "expires_ns": self.expires_ns}


class SendFailure(RuntimeError):
    def __init__(self, outcomes):
        self.outcomes = outcomes
        super().__init__(json.dumps(outcomes))


class LiveSink(_BaseLiveSink):
    """Expiry-aware live transport; outcomes distinguish arm sends from gripper target queueing."""

    def send(self, item) -> dict:
        if getattr(item, "kind", None) != "track" or not hasattr(item, "expires_ns"):
            raise ExecutionRefused("LiveSink requires executor setpoints with command expiry")
        outcome = {"arm": "not sent", "gripper": "not queued"}
        try:
            self.link.send_flange_pose(item.pose7)
            outcome["arm"] = "sent"
        except Exception as exc:
            outcome["arm"] = f"{type(exc).__name__}: {exc}"
            raise SendFailure(outcome) from exc
        if self.gripper is not None and item.gripper_closure is not None:
            try:
                self.gripper.set_closure(item.gripper_closure, item.expires_ns)
                outcome["gripper"] = "target queued (serial writes recorded by gripper worker)"
            except Exception as exc:
                outcome["gripper"] = f"{type(exc).__name__}: {exc}"
                raise SendFailure(outcome) from exc
        self.sent += 1
        if self.log is not None:
            self.log(item, True)
        return outcome


class MotionShutdown:
    """Stop motion independently of policy/network/disk waits. Resource closes belong to the launcher."""

    def __init__(self, cancel, *, report=None):
        self.cancel, self.report = cancel, report or log.error
        self.session = self.gripper = self.link = self.state = None
        self._lock = threading.RLock()  # signal handlers may re-enter request() on the main thread
        self._thread = None
        self._done = threading.Event()

    def request(self) -> None:
        self.cancel.set()
        with self._lock:
            if self._thread is None:
                self._thread = threading.Thread(target=self._run, name="motion-shutdown", daemon=True)
                self._thread.start()

    def _run(self) -> None:
        try:
            self._release_motion()
        finally:
            self._done.set()

    def _release_motion(self) -> None:
        phases = []
        if self.session is not None:
            phases.append(self.session.join_executor)
        if self.gripper is not None:
            phases.extend([self.gripper.clear_target, self.gripper.stop])
        if self.link is not None:
            phases.append(lambda: self.link.release(self.state.latest, still=lambda: still(self.state)))
        for phase in phases:
            try:
                phase()
            except (Exception, KeyboardInterrupt) as exc:
                self.report(f"motion shutdown: {type(exc).__name__}: {exc}")

    def finish(self) -> None:
        self.request()
        # A Ctrl-C during join() can mark a still-running thread as stopped on CPython 3.10. Completion of the
        # actual release phases is the barrier before resource cleanup, even after another interrupt.
        while True:
            try:
                self._done.wait()
                self._thread.join()
                return
            except KeyboardInterrupt:
                self.report("Ctrl-C during shutdown: waiting for controller release")


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
                 sent: bool = False, max_failed_steps: int = 5, idle_timeout_s: float = 2.0, on_status=None,
                 stop_event=None, shutdown=None):
        spec = getattr(client, "spec", None) or {}
        if len(cams) != 1 or spec.get("views", 1) > 1 or len(spec.get("camera_ids", ())) > 1:
            raise ExecutionRefused("live multiview sync sets are not implemented; select a single-camera policy")
        self.client, self.cameras, self.cams, self.state = client, cameras, list(cams), state
        self.executor, self.sink, self.recorder, self.sent = executor, sink, recorder, bool(sent)
        self._command_lock = threading.Lock()
        self._accepted = 0
        self.loop = PolicyLoop(client, state, limits, _NullSink(), executor=self)
        self.max_failed, self.idle_timeout_s, self.on_status = int(max_failed_steps), float(idle_timeout_s), on_status
        self.stop_event = stop_event if stop_event is not None else threading.Event()
        self._stop_lock = threading.RLock()
        self.shutdown = shutdown or MotionShutdown(self.stop_event)
        self.shutdown.session, self.shutdown.state = self, state
        self.shutdown.gripper = state.gripper if hasattr(state.gripper, "stop") else None
        if self.sent:
            self.shutdown.link = sink.link
        self.stop_reason: str | None = None
        self.rows: list = []
        self._exec_thread: threading.Thread | None = None
        self._rec_robot_t = self._rec_grip_t = 0
        self.tick_periods_ns: list = []  # measured executor loop periods (local control rate evidence)
        self._gripper_health = None

    def accept(self, cmd) -> bool:
        with self._command_lock:
            if self.stop_event.is_set():
                return False
            accepted = self.executor.accept(cmd)
            self._accepted += bool(accepted)
            return accepted

    def _check_gripper(self) -> bool:
        g = self.state.gripper
        if not hasattr(g, "healthy"):
            return True
        healthy = g.healthy()
        s = g.latest
        # healthy() requires a <100 ms sample. Allow a missing reply up to 200 ms before latching comm loss.
        fatal = not healthy and (s is None or time.monotonic_ns() - s.t_ns >= 200_000_000 or
                                 s.fault != 0 or s.status != 3 or s.activated != 1)
        if self.sent and fatal:
            self.stop(f"gripper unhealthy: {g.error or 'no fresh activated status'}")
            return False
        if healthy != self._gripper_health:
            self._gripper_health = healthy
            if not healthy:
                log.warning("gripper unhealthy: %s", g.error or "no fresh activated status")
            if self.recorder is not None:
                self.recorder.event(time.monotonic_ns(), f"gripper healthy={healthy}: {g.error}")
        return True

    # ------------------------------------------------------------------------------------------------ executor
    def _executor_loop(self) -> None:
        try:
            self._executor_ticks()
        except BaseException as exc:
            self.stop(f"executor thread failed: {type(exc).__name__}: {exc}")

    def _executor_ticks(self) -> None:
        period = self.executor.period_ns / 1e9
        last = None
        next_t = time.monotonic()  # absolute deadlines: the mean rate stays at rate_hz despite sleep overshoot
        while not self.stop_event.is_set():
            t0 = time.monotonic()
            if last is not None:
                self.tick_periods_ns.append(int((t0 - last) * 1e9))
            last = t0
            if self.sent and not self.sink.link.peers_ok():
                self.stop("controller subscriber disconnected")
                break
            if not self._check_gripper():
                break
            s = self.state.latest()
            if s is not None:
                measured = s["flange_pose7"]
                if not self.sent and self.executor._setpoint is not None:  # dry run: nothing is sent, so the robot
                    measured = self.executor._setpoint  # cannot follow; an ideal follower (labelled) keeps the shaping
                with self._command_lock:
                    sp = self.executor.tick(time.monotonic_ns(), measured, state_t_ns=s["t_ns"],
                                            signature=s["signature"])
                    if sp is not None:
                        cmd = self.executor.command
                        if cmd is None or cmd.reference_id != sp.reference_id:
                            raise RuntimeError("setpoint has no matching current command")
                        sp = LiveSetpoint(sp.t_ns, sp.kind, sp.pose7, sp.gripper_closure, sp.reference_id,
                                          cmd.expires_ns)
                if sp is not None:
                    sent, detail = False, ""
                    try:
                        if self.stop_event.is_set():
                            break
                        outcome = self.sink.send(sp)
                        sent = self.sent
                        detail = json.dumps(outcome) if outcome is not None else ""
                    except Exception as exc:  # noqa: BLE001 - a failed send ends the session
                        detail = str(exc)
                        self.stop(f"command send failed: {exc}")
                    if self.recorder is not None:
                        self.recorder.executed(sp.t_ns, sp.kind, sp.pose7, sp.gripper_closure, sent, detail)
                if self.executor.stopped and not self.stop_event.is_set():
                    self.stop(f"executor stop: {self.executor.stopped}")
            next_t += period
            delay = next_t - time.monotonic()
            if delay < -period:  # overrun by more than a period: resynchronise instead of bursting
                next_t, delay = time.monotonic(), 0.0
            self.stop_event.wait(max(0.0, delay))

    def loop_stats(self) -> dict:
        """Executor (local control loop) and policy-step rates of this session, measured."""
        nominal = self.executor.period_ns / 1e6
        out = {"executor_nominal_hz": 1e3 / nominal, "executor_ticks": len(self.tick_periods_ns) + 1}
        if self.tick_periods_ns:
            p = np.asarray(self.tick_periods_ns, float) / 1e6
            out.update(executor_mean_hz=float(1e3 / p.mean()), period_ms_p50=float(np.median(p)),
                       period_ms_p99=float(np.percentile(p, 99)), period_ms_max=float(p.max()),
                       overruns_gt_1p5x=int((p > 1.5 * nominal).sum()), gaps_gt_10ms=int((p > 10).sum()))
        t = [r.get("t_step_ns") for r in self.rows if r.get("t_step_ns")]
        if len(t) > 1:
            out["policy_steps"], out["policy_mean_hz"] = len(t), float((len(t) - 1) / ((t[-1] - t[0]) / 1e9))
        return out

    def stop(self, reason: str) -> None:
        with self._stop_lock:
            first = self.stop_reason is None
            if first:
                self.stop_reason = reason
            self.stop_event.set()
        self.shutdown.request()  # never wait for the main loop, the recorder, or the policy server
        if first and self.recorder is not None:
            try:
                self.recorder.event(time.monotonic_ns(), f"stop: {reason}")
            except Exception:
                log.exception("could not record session stop: %s", reason)

    def join_executor(self, timeout: float = 1.0) -> None:
        t = self._exec_thread
        if t is not None and t is not threading.current_thread():
            t.join(timeout=timeout)
            if t.is_alive():
                log.error("executor thread did not exit within %.3f s; proceeding with release", timeout)

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
        if self.stop_event.is_set():
            self.stop("cancelled before streaming")
            self.shutdown.finish()
            return self.rows
        self._exec_thread = threading.Thread(target=self._executor_loop, name="executor-500hz", daemon=True)
        try:
            self._exec_thread.start()
            t_end = None if duration_s is None else time.monotonic() + float(duration_s)
            last_frame = dict.fromkeys(self.cams, time.monotonic())
            failed = 0
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
                accepted_before = self._accepted
                row = self.loop.step(*newest, acquire_ms=acquire)
                self.rows.append(row)
                if self.recorder is not None:
                    self.recorder.step(row)
                    if row.get("command"):
                        self.recorder.command(row["command"])
                failed = 0 if self._accepted > accepted_before else failed + 1
                if self.on_status is not None:
                    self.on_status(row)
                if row.get("terminal"):
                    self.stop("session taken over by another client")
                elif failed >= self.max_failed:
                    self.stop(f"{failed} consecutive steps without an action ({row.get('code') or row.get('status')})")
        except KeyboardInterrupt:
            self.stop("operator Ctrl-C")
        except BaseException as exc:
            self.stop(f"main loop failed: {type(exc).__name__}: {exc}")
            raise
        finally:
            self.stop(self.stop_reason or "session ended")
            self.shutdown.finish()
            self._record_streams()
            if self.recorder is not None:
                self.recorder.event(time.monotonic_ns(), "loop_stats " + json.dumps(self.loop_stats()))
        return self.rows


class _NullSink:
    """Validated commands go to the executor (``PolicyLoop(executor=...)``); only its setpoints reach the real sink."""

    def send(self, item) -> None:
        pass


def still(state: LiveState, max_dq: float = 0.02) -> bool:
    s = state.latest()
    return bool(s is not None and 0 <= time.monotonic_ns() - s["t_ns"] < 50_000_000 and
                s.get("dq") is not None and np.isfinite(s["dq"]).all() and
                np.asarray(s["dq"]).size > 0 and float(np.abs(s["dq"]).max()) < max_dq)
