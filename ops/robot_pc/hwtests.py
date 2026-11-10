"""Policy-free P3 gripper and zero-motion hold tests. Hardware ownership belongs to the launcher.

Rehearsals read measured streams but never authorize motion, queue a gripper target or construct a commander.
The hold rehearsal uses the same ideal follower as LiveSession; its health checks still use measured state.
"""

from __future__ import annotations

import json
import threading
import time

import numpy as np

from ops.robot_pc import live
from ops.robot_pc.gripper import POS_CLOSED, POS_OPEN, closure_to_position
from ops.robot_pc.poses import as_pose7, quat_angle
from ops.robot_pc.safety import ExecutionAuthorization, ExecutionRefused
from ops.robot_pc.validation import Command


class Cancelled(Exception):
    """The launcher's single cancellation event was set at a checkpoint."""


class _Discard:
    def send(self, item):
        pass


class HardwareTest:
    def __init__(self, test, *, parameters, cfg, state, rehearse, stop_event, shutdown,
                 recorder=None, cameras=None, cams=()):
        self.test, self.parameters, self.state = test, dict(parameters), state
        self.rehearse, self.sent = bool(rehearse), not rehearse
        self.stop_event, self.shutdown = stop_event, shutdown
        self.recorder, self.cameras, self.cams = recorder, cameras, list(cams)
        if self.shutdown.on_release is None:
            self.shutdown.on_release = self.event
        self.shutdown.session, self.shutdown.state = self, state
        self.executor = self.sink = self.link = None
        if test == "hold":
            from ops.robot_pc.executor import HighRateExecutor

            self.executor = HighRateExecutor(_Discard(), **(cfg.get("executor") or {}))
        if rehearse:
            from ops.robot_pc.sinks import DryRunSink

            self.sink = DryRunSink()
        self._command_lock, self._stop_lock = threading.Lock(), threading.RLock()
        self._exec_thread = None
        self.stop_reason = None
        self.abort_reasons = []
        self.completed = False
        self.start_pose = None
        self.targets = []
        self.held_s = self.max_drift_mm = self.max_drift_deg = self.max_abs_dq = 0.0
        self.tick_periods_ns = []
        self.rows = []  # loop_stats has no policy rows for a hardware test
        self._rec_robot_t = self._rec_grip_t = self._rec_write_t = 0
        self._started_ns = None
        self._hold_started = None
        self._verdict = None

    def event(self, text):
        if self.recorder is not None:
            self.recorder.event(time.monotonic_ns(), text)

    def checkpoint(self):
        if self.stop_event.is_set():
            raise Cancelled()

    def abort(self, reason):
        # Latch before starting release; recording must never delay motion shutdown.
        with self._stop_lock:
            first = reason not in self.abort_reasons
            if first:
                self.abort_reasons.append(reason)
            if self.stop_reason is None:
                self.stop_reason = reason
                if self._hold_started is not None:
                    self.held_s = time.monotonic() - self._hold_started
            self.stop_event.set()
        self.shutdown.request()
        if first:
            self.event("abort: " + reason)

    def _healthy_gripper(self, wait_s: float = 0.3):
        """Healthy now, or within ``wait_s`` (one late 50 Hz status read must not refuse a test)."""
        g = self.state.gripper
        deadline = time.monotonic() + wait_s
        while not g.healthy():
            self.checkpoint()  # a cancel during the wait is a cancel (130), never a refusal
            if time.monotonic() >= deadline:
                raise ExecutionRefused(f"gripper unhealthy: {g.error or 'no fresh activated status'}")
            time.sleep(0.01)

    def _gripper_not_fatal(self):
        """Hold test (gripper read-only, never commanded): LiveSession's rule - a single missed / bad status reply
        is tolerated; no status for 200 ms, a fault or a deactivated gripper aborts."""
        g = self.state.gripper
        s = g.latest
        if s is None or time.monotonic_ns() - s.t_ns >= 200_000_000 or s.fault != 0 or s.status != 3 or \
                s.activated != 1 or getattr(g, "error_latched", False):
            raise ExecutionRefused(f"gripper unhealthy: {g.error or 'no fresh activated status'}")

    def prepare(self):
        """Measured preconditions, before the operator is offered START."""
        self.checkpoint()
        # a RealSense start stalls the status stream once (~0.3 s measured on the robot PC): wait <= 1 s for health
        deadline = time.monotonic() + 2.5  # covers one reconnect of the worker (1 s back-off) after an I/O error
        while not self.state.gripper.healthy() and time.monotonic() < deadline:
            self.checkpoint()
            self.stop_event.wait(0.02)
        self._healthy_gripper()
        if self.test == "gripper":
            if self.state.gripper.latest.position > closure_to_position(0.05):
                raise ExecutionRefused("gripper must be essentially open (closure <= 0.05)")
            return
        deadline, since = time.monotonic() + 3.0, None
        while time.monotonic() < deadline:
            self.checkpoint()
            self._healthy_gripper()
            now = time.monotonic()
            since = (now if since is None else since) if live.still(self.state) else None
            self.record()
            if since is not None and now - since >= 0.5:
                return
            self.stop_event.wait(0.02)
        raise ExecutionRefused("robot must be still (fresh |dq| < 0.02) for 0.5 s")

    def capture_start(self):
        """Called immediately after ENTER (or after preparation in rehearsal)."""
        from ops.robot_pc.switch_controller import state_blockers

        self.checkpoint()
        self._healthy_gripper()
        blockers = state_blockers(list(self.state.reader.samples), time.monotonic_ns())
        if blockers:
            raise ExecutionRefused("; ".join(blockers))
        if self.test == "gripper":
            self.prepare()  # the fingers may have moved while waiting for ENTER
        else:
            if not live.still(self.state):
                raise ExecutionRefused("robot is no longer still at START")
            self.start_pose = tuple(as_pose7(self.state.latest()["flange_pose7"]).tolist())

    def record(self):
        # Identical measured stream binding to LiveSession; no observations or policy calls.
        if self.recorder is not None:
            for s in [s for s in list(self.state.reader.samples) if s["t_ns"] > self._rec_robot_t]:
                self.recorder.robot_state(s["t_ns"], s["seq"], s["q"], s.get("dq"), s.get("tau"),
                                          s["flange_pose7"], s["ee_pose7"])
                self._rec_robot_t = s["t_ns"]
            for t, s in [x for x in list(self.state.gripper.samples) if x[0] > self._rec_grip_t]:
                self.recorder.gripper(t, int(s.position), dict(s.__dict__))
                self._rec_grip_t = t
            for t, register in [x for x in list(self.state.gripper.sent) if x[0] > self._rec_write_t]:
                self.event("gripper_write " + json.dumps({"write_monotonic_ns": t, "target_register": register}))
                self._rec_write_t = t
        if self.cameras is not None:
            for cam in self.cams:
                for cap in self.cameras.drain(cam):
                    if self.recorder is not None:
                        self.recorder.frame(cam, cap["frame_id"], cap["capture_ns"], cap["native_rgb"],
                                            None, None, cap.get("metadata"))

    def run(self):
        self.checkpoint()
        self._started_ns = time.monotonic_ns()
        self.event("start " + json.dumps({"test": self.test, "rehearse": self.rehearse,
                                         "parameters": self.parameters, "start_flange_pose7": self.start_pose}))
        try:
            if self.test == "gripper":
                self._gripper_test()
            else:
                self._hold_test()
        except Cancelled:
            if self.stop_reason is None:
                self.abort("operator stop")
        except KeyboardInterrupt:
            self.abort("operator Ctrl-C")
        except Exception as exc:
            self.abort(f"{type(exc).__name__}: {exc}")
        finally:
            self.shutdown.finish()
        return self.finalize()

    def _gripper_test(self):
        g, dwell = self.state.gripper, self.parameters["dwell_s"]
        for closure in self.parameters["closures"]:
            self.checkpoint()
            self._healthy_gripper()
            started = time.monotonic_ns()
            row = {"target_register": closure_to_position(closure), "final_position": int(g.latest.position),
                   "reached": False, "time_to_reach_s": None, "object": int(g.latest.object), "faults": []}
            self.targets.append(row)
            self.event("target " + json.dumps({"closure": closure, "target_register": row["target_register"],
                                               "expires_ns": started + int((dwell + 0.5) * 1e9)}))
            if not self.rehearse:
                g.set_closure(closure, expires_monotonic_ns=started + int((dwell + 0.5) * 1e9))
            last_t = started - 1
            while True:
                self.checkpoint()
                samples = [s for t, s in list(g.samples) if t > last_t]
                if not samples or samples[-1].t_ns != g.latest.t_ns:
                    samples.append(g.latest)
                for s in samples:
                    last_t = max(last_t, s.t_ns)
                    row.update(final_position=int(s.position), object=int(s.object))
                    if s.fault not in row["faults"]:
                        row["faults"].append(int(s.fault))
                    if s.object in (1, 2):
                        g.clear_target()
                        self.abort("contact")
                        return
                    if s.fault or s.activated != 1 or s.status != 3:
                        g.clear_target()
                        self.abort(f"gripper unhealthy: fault 0x{s.fault:02X}, status {s.status}")
                        return
                    if not row["reached"] and abs(s.position - row["target_register"]) <= 8:
                        row["reached"] = True
                        row["time_to_reach_s"] = max(0.0, (s.t_ns - started) / 1e9)
                self._healthy_gripper()
                self.record()
                remaining = dwell - (time.monotonic_ns() - started) / 1e9
                if remaining <= 0:
                    break
                self.stop_event.wait(min(0.02, remaining))
        g.clear_target()  # the Robotiq itself holds its last position
        self.completed = True

    def _measure_hold(self, s):
        if s is None:
            raise ExecutionRefused("no robot state")
        measured = as_pose7(s["flange_pose7"])
        mm = float(np.linalg.norm(measured[:3] - self.start_pose[:3]) * 1e3)
        deg = float(np.degrees(quat_angle(measured[3:], self.start_pose[3:])))
        dq = np.asarray(s.get("dq"), float)
        if dq.shape != (7,) or not np.isfinite(dq).all():
            raise ExecutionRefused("invalid robot dq")
        speed = float(np.abs(dq).max())
        with self._stop_lock:
            self.max_drift_mm = max(self.max_drift_mm, mm)
            self.max_drift_deg = max(self.max_drift_deg, deg)
            self.max_abs_dq = max(self.max_abs_dq, speed)
        if mm > self.parameters["max_drift_mm"] or deg > self.parameters["max_drift_deg"]:
            raise ExecutionRefused(f"drift: {mm:.6f} mm / {deg:.6f} deg")
        if speed > self.parameters["max_dq"]:
            raise ExecutionRefused(f"dq: {speed:.6f} rad/s")
        return mm / 1e3, deg

    def _hold_test(self):
        self.checkpoint()
        if self.sent:
            authorization = getattr(self.sink, "authorization", None)
            if not isinstance(authorization, ExecutionAuthorization) or authorization.hardware_test != "hold":
                raise ExecutionRefused("hold test requires hardware_test == 'hold' authorization")
        if self.start_pose is None:
            raise ExecutionRefused("hold start pose has not been captured")
        started = self._hold_started = time.monotonic()
        # Seed the first command before starting the executor, avoiding an unmonitored commandless interval.
        self._feed_hold(0)
        self.checkpoint()
        self._exec_thread = threading.Thread(target=self._executor_loop, name="hwtest-executor-500hz", daemon=True)
        self._exec_thread.start()
        n = 1
        while not self.stop_event.is_set():
            self.held_s = time.monotonic() - started
            self.record()
            if self.held_s >= self.parameters["duration_s"]:
                self.completed = True
                return
            self.stop_event.wait(min(0.05, self.parameters["duration_s"] - self.held_s))
            self.checkpoint()
            self._feed_hold(n)
            n += 1

    def _feed_hold(self, n):
        self.checkpoint()
        self._gripper_not_fatal()
        s = self.state.latest()
        distance, angle = self._measure_hold(s)
        register = int(self.state.gripper.latest.position)
        closure = float(np.clip((register - POS_OPEN) / (POS_CLOSED - POS_OPEN), 0, 1))
        now = time.monotonic_ns()
        cmd = Command(f"hold:{n}", "hwtest-hold", now, now + 200_000_000,
                      tuple(s["flange_pose7"]), self.start_pose, closure, register, distance, angle)
        with self._command_lock:
            self.checkpoint()
            accepted = self.executor.accept(cmd)
        if not accepted:
            raise RuntimeError("hold command refused by executor")
        if self.recorder is not None:
            self.recorder.command(cmd.to_json())

    def _executor_loop(self):
        try:
            self._executor_ticks()
        except BaseException as exc:
            self.abort(f"executor thread failed: {type(exc).__name__}: {exc}")

    def _executor_ticks(self):
        # LiveSession's tick path: elapsed periods, measured health, expiry-aware setpoints, real send outcomes.
        period, last = self.executor.period_ns / 1e9, None
        next_t = time.monotonic()
        while not self.stop_event.is_set():
            t0 = time.monotonic()
            if last is not None:
                self.tick_periods_ns.append(int((t0 - last) * 1e9))
            last = t0
            if self.sent and not self.link.peers_ok():
                self.abort("controller subscriber disconnected")
                break
            self._gripper_not_fatal()
            s = self.state.latest()
            self._measure_hold(s)
            measured = s["flange_pose7"]
            if self.rehearse and self.executor._setpoint is not None:
                measured = self.executor._setpoint
            with self._command_lock:
                sp = self.executor.tick(time.monotonic_ns(), measured, state_t_ns=s["t_ns"], signature=s["signature"])
                if sp is not None:
                    cmd = self.executor.command
                    if cmd is None or cmd.reference_id != sp.reference_id:
                        raise RuntimeError("setpoint has no matching current command")
                    sp = live.LiveSetpoint(sp.t_ns, sp.kind, sp.pose7, sp.gripper_closure, sp.reference_id,
                                           cmd.expires_ns)
            if sp is not None:
                sent, detail = False, ""
                try:
                    if self.stop_event.is_set():
                        break
                    outcome = self.sink.send(sp)
                    sent = self.sent
                    detail = json.dumps(outcome) if outcome is not None else ""
                except Exception as exc:
                    detail = str(exc)
                    self.abort(f"command send failed: {exc}")
                if self.recorder is not None:
                    self.recorder.executed(sp.t_ns, sp.kind, sp.pose7, sp.gripper_closure, sent, detail)
            if self.executor.stopped and not self.stop_event.is_set():
                self.abort(f"executor stop: {self.executor.stopped}")
            next_t += period
            delay = next_t - time.monotonic()
            if delay < -period:
                next_t, delay = time.monotonic(), 0.0
            self.stop_event.wait(max(0, delay))

    def join_executor(self, timeout=1.0):
        live.LiveSession.join_executor(self, timeout)

    def finalize(self):
        """Called only after MotionShutdown.finish(), including when startup was cancelled."""
        if self._verdict is not None:
            return self._verdict
        self.record()
        g = self.state.gripper
        verdict = {"test": self.test, "rehearse": self.rehearse, "abort_reasons": list(self.abort_reasons)}
        clean = self.completed and not self.abort_reasons
        if self.test == "gripper":
            reached = len(self.targets) == len(self.parameters["closures"]) and all(r["reached"] for r in self.targets)
            holds = [list(h) for h in getattr(g, "holds", [])]
            hold_failures = [list(h) for h in getattr(g, "hold_failures", [])]
            verdict.update(targets=self.targets, writes=len(g.sent), status_reads=len(g.samples), all_reached=reached,
                           holds=holds, hold_failures=hold_failures,
                           # A rehearsal verifies read-only operation; it cannot demonstrate finger motion.
                           **{"pass": clean and not hold_failures and (len(g.sent) == 0 if self.rehearse else reached)})
            if self.rehearse:
                verdict["verdict"] = f"rehearsal: {len(g.samples)} status reads, {len(g.sent)} writes"
        else:
            stats = live.LiveSession.loop_stats(self)
            release = self.shutdown.release_outcome
            verdict.update(duration_held_s=self.held_s, max_drift_mm=self.max_drift_mm, max_drift_deg=self.max_drift_deg,
                           max_abs_dq=self.max_abs_dq, executor_counts=dict(self.executor.counts), loop_stats=stats,
                           actions_sent=0 if self.link is None else self.link.actions_sent,
                           mode_requests_sent=0 if self.link is None else self.link.mode_requests_sent,
                           release=release,
                           **{"pass": clean and (self.rehearse or
                                                  (isinstance(release, dict) and release.get("hold") == "sent" and
                                                   release.get("idle") == "sent"))})
            self.event("loop_stats " + json.dumps(stats))
        self.event("verdict " + json.dumps(verdict, allow_nan=False))
        self._verdict = verdict
        return verdict
