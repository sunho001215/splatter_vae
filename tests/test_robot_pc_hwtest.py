"""Conservative limits, hardware-test authorization and the hardware-test launcher (fakes only; nothing real)."""

from __future__ import annotations

import pytest

from ops.robot_pc import conservative
from ops.robot_pc.safety import ExecutionAuthorization, ExecutionRefused, Mode, authorize_execution, \
    authorize_hardware_test
from ops.robot_pc.validation import ActionLimits

BASE = {"executor": {"rate_hz": 500, "max_linear_speed": 0.3, "max_angular_speed": 1.5, "max_linear_accel": 4.0,
                     "max_angular_accel": 15.0, "lead_m": 0.04, "lead_rot_deg": 20.0, "state_timeout_s": 0.05,
                     "frozen_state_s": 0.25}}


def test_tighten_only_lowers():
    cfg, applied = conservative.tighten(BASE, "slow")
    ex = cfg["executor"]
    assert ex["max_linear_speed"] == 0.05 and ex["max_angular_speed"] == 0.25 and ex["lead_m"] == 0.02
    assert ex["rate_hz"] == 500 and ex["state_timeout_s"] == 0.05  # untouched
    assert BASE["executor"]["max_linear_speed"] == 0.3  # input not mutated
    assert applied["max_linear_speed"] == (0.3, 0.05)


def test_tighten_never_loosens():
    strict = {"executor": {**BASE["executor"], "max_linear_speed": 0.01, "lead_m": 0.005}}
    cfg, _ = conservative.tighten(strict, "slow")
    assert cfg["executor"]["max_linear_speed"] == 0.01 and cfg["executor"]["lead_m"] == 0.005


def test_tighten_unknown_profile():
    with pytest.raises(ValueError):
        conservative.tighten(BASE, "fast")


def test_test_box_intersects():
    lim = ActionLimits.from_config({"workspace_min_m": [0.257, -0.242, 0.425], "workspace_max_m": [0.621, 0.321, 0.694]})
    box = conservative.test_box(lim, [0.42, 0.0, 0.66, 0, 0, 0, 1], 0.08)
    assert box.workspace_min_m == pytest.approx((0.34, -0.08, 0.58))
    assert box.workspace_max_m == pytest.approx((0.50, 0.08, 0.694))  # clipped by the task box
    assert box.max_translation_m == lim.max_translation_m
    with pytest.raises(ValueError):
        conservative.test_box(lim, [0.9, 0.0, 0.6, 0, 0, 0, 1], 0.08)  # start outside the task box


def test_gripper_closures_bounded():
    assert conservative.gripper_closures([0.3, 0.0]) == [0.3, 0.0]
    for bad in ([0.8], [-0.1], [], [float("nan")]):
        with pytest.raises(ValueError):
            conservative.gripper_closures(bad)


def test_hardware_test_authorization():
    a = authorize_hardware_test(test="hold", cli_flag=True, operator_start=True)
    assert a.mode is Mode.EXECUTE and a.hardware_test == "hold"
    for kw in ({"test": "rollout", "cli_flag": True, "operator_start": True},
               {"test": "gripper", "cli_flag": False, "operator_start": True},
               {"test": "gripper", "cli_flag": True, "operator_start": False}):
        with pytest.raises(ExecutionRefused):
            authorize_hardware_test(**kw)
    with pytest.raises(ExecutionRefused):  # forged
        ExecutionAuthorization(Mode.EXECUTE, 0, (), None, hardware_test="hold")
    p = authorize_execution(cli_execute=True, operator_start=True, policy_info={"execution_eligible": True})
    assert p.hardware_test is None
    with pytest.raises(ExecutionRefused):
        authorize_execution(cli_execute=True, operator_start=True, policy_info={})


@pytest.fixture
def hw_env(monkeypatch, tmp_path):
    """All acquisition/transport is fake, including RTU serial, ZMQ, state and cameras.

    Keep the real GripperWorker, RobotiqRTU codec, ControllerLink, LiveSink and HighRateExecutor in the path.
    No test may fall through to a physical transport or a policy connection.
    """
    import collections
    import dataclasses
    import io
    import json
    import signal
    import threading
    import time
    from types import SimpleNamespace

    import numpy as np

    from ops.robot_pc import cameras, cli, controller, devices, hwtests, live, recorder, safety
    from ops.robot_pc import gripper as G
    from ops.robot_pc import switch_controller as sc
    from scripts.robot_pc import hwtest
    from test_robot_pc import FakeZmq, fake_robot_msg

    env = SimpleNamespace(problem=None, order=[], links=[], zs=[], workers=[], serials=[], recorders=[], sessions=[],
                          targets=[], authorizations=[], prompts=[], messages=[], handlers={}, started=None,
                          pose=(0.42, 0.01, 0.55, 0.0, 0.0, 0.0, 1.0), serial_writes=[], free_calls=[], camera_kw=[])
    real_link, real_rtu, real_worker = controller.ControllerLink, G.RobotiqRTU, G.GripperWorker
    real_test, real_auth = hwtests.HardwareTest, safety.authorize_hardware_test
    env.real_recorder = recorder.SessionRecorder
    epoch = time.monotonic_ns()

    def trigger_stop():
        env.handlers[signal.SIGTERM](signal.SIGTERM, None)

    def fake_signal(sig, handler):
        previous = env.handlers.get(sig, signal.SIG_DFL)
        env.handlers[sig] = handler
        return previous

    def forbidden(*a, **kw):
        pytest.fail("unexpected policy or physical transport construction")

    monkeypatch.setattr(hwtest.signal, "signal", fake_signal)
    monkeypatch.setattr(hwtest.sys, "stdin", io.StringIO(""))
    def say(message):
        env.messages.append(message)
        if env.problem == "ready_sigterm" and message.startswith("READY"):
            trigger_stop()

    monkeypatch.setattr(hwtest, "say", say)
    monkeypatch.setattr(cli, "load_config", lambda args: {**BASE, "root": str(tmp_path),
                                                          "cameras": {"native_hw": [4, 6], "fps": 15}})
    monkeypatch.setattr(cli, "connect", forbidden)
    monkeypatch.setattr(cli, "choose_policy", forbidden)
    monkeypatch.setattr(cli, "welcome", forbidden)
    monkeypatch.setattr(devices, "require_free", lambda **kw: env.free_calls.append(kw))
    monkeypatch.setattr(devices, "scan", lambda **kw: {"controllers": [] if env.problem == "no_controller" else [1],
                                                      "recorders": []})

    class Reader:
        def __init__(self, *, approved):
            assert approved is True
            self.rows = collections.deque(maxlen=4096)
            self.seq = 0
            self._sample_lock = threading.Lock()

        @property
        def samples(self):
            # Main-loop recording and release both read samples. Simulate the real reader's single writer.
            with self._sample_lock:
                return list(self._sample_rows())

        def _sample_rows(self):
            now = time.monotonic_ns()
            active = env.started is not None and (now - env.started) > 80_000_000
            if env.problem == "stale" and active:
                return self.rows
            last = int((now - epoch) // 1_000_000)
            while self.seq < last:
                self.seq += 1
                p = list(env.pose)
                dq = np.zeros(7)
                if active and env.problem == "drift":
                    p[0] += 0.006
                if active and env.problem == "rotation":
                    a = np.deg2rad(3) / 2
                    p[3:] = [0, 0, np.sin(a), np.cos(a)]
                if active and env.problem == "dq":
                    dq[0] = 0.06
                q = np.ones(7) + self.seq * 1e-12
                signature = b"frozen" if active and env.problem == "frozen" else q.tobytes()
                self.rows.append({"t_ns": epoch + self.seq * 1_000_000, "seq": self.seq, "q": q, "dq": dq,
                                  "tau": np.zeros(7), "flange_pose7": np.array(p),
                                  "ee_pose7": sc.ee_from_flange(p), "signature": signature})
            return self.rows

        def close(self):
            env.order.append("reader close")

    class Serial:
        def __init__(self, *a, **kw):
            assert kw["exclusive"] is True
            assert kw["write_timeout"] == 0.05
            self.position = 30 if env.problem == "not_open" else 3
            self.reply, self.requests, self.reads = b"", [], 0
            env.serials.append(self)

        def reset_input_buffer(self):
            pass

        def fileno(self):
            return 12345  # ioctl is injected; never a physical fd

        def write(self, req):
            self.requests.append(req)
            if req[1] == 3:
                self.reads += 1
                fault = 9 if env.problem == "preflight_fault" or \
                    (env.problem == "preflight_fault_clears" and self.reads < 3) else 0
                obj = 3
                if env.started is not None and env.serial_writes:
                    if env.problem == "contact":
                        obj = 2
                    if env.problem == "fault":
                        fault = 5
                status = 1 | 8 | (3 << 4) | (obj << 6)
                body = bytes([9, 3, 6, status, 0, fault, self.position, self.position, 0])
            else:
                assert req[1] == 0x10 and req[11:13] == bytes([64, 0])
                env.serial_writes.append(req[10])
                if env.problem != "unreached":
                    self.position = req[10]
                body = req[:6]
            self.reply = body + G.crc16_modbus(body)

        def read(self, n):
            return self.reply[:n]

        def close(self):
            env.order.append("serial close")

    class Worker(real_worker):
        def __init__(self, *a, **kw):
            super().__init__(*a, **kw)
            env.workers.append(self)

        def set_closure(self, closure, expires_monotonic_ns):
            env.targets.append((closure, expires_monotonic_ns, time.monotonic_ns()))
            super().set_closure(closure, expires_monotonic_ns)

        def clear_target(self):
            env.order.append("gripper clear")
            super().clear_target()

        def stop(self, *a, **kw):
            super().stop(*a, **kw)
            env.order.append("gripper stopped")

    class Cameras:
        def __init__(self, serials, **kw):
            env.camera_kw.append(kw)
            self.serials, self.frame = serials, 0

        def start(self):
            return {c: np.eye(3) for c in self.serials}

        def bind(self, K):
            if env.problem == "before_enter":
                trigger_stop()

        def drain(self, cam):
            self.frame += 1
            return [{"frame_id": self.frame, "capture_ns": time.monotonic_ns(),
                     "native_rgb": np.zeros((4, 6, 3), np.uint8), "metadata": {"color_frame_number": self.frame}}]

        def close(self):
            env.order.append("camera close")
            if env.problem == "cleanup_baseexception":
                raise KeyboardInterrupt("fake camera close")

    class Recorder:
        session_dir = tmp_path

        def __init__(self, root, *, meta, cameras):
            assert root == tmp_path
            self.meta, self.cameras, self.events = meta, cameras, []
            self.executions, self.commands, self.robot, self.grip, self.frames = [], [], [], [], []
            self.status = self.error = None
            env.recorders.append(self)

        def event(self, t, text):
            self.events.append(text)
            if text.startswith("start "):
                env.started = time.monotonic_ns()

        def executed(self, *a):
            self.executions.append(a)

        def command(self, cmd):
            self.commands.append(cmd)

        def robot_state(self, *a):
            self.robot.append(a)

        def gripper(self, *a):
            self.grip.append(a)

        def frame(self, *a):
            self.frames.append(a)

        observation = step = forbidden

        def close(self, status, error):
            assert all(not s._exec_thread or not s._exec_thread.is_alive() for s in env.sessions)
            assert all(not l.socks for l in env.links)
            self.status, self.error = status, error
            env.order.append("recorder close")
            if env.problem == "cleanup_signals":
                for sig in (signal.SIGINT, signal.SIGTERM, signal.SIGINT):
                    env.handlers[sig](sig, None)
                env.order.append("recorder finalized")
            return {}

    def make_link(auth):
        assert auth.hardware_test == "hold" and not env.workers[0].sent
        z, msg = FakeZmq(), fake_robot_msg()
        msg.ControllerCommand = lambda ctrl_mode: {"ctrl_mode": ctrl_mode}
        msg.encode = lambda m: json.dumps(m if isinstance(m, dict) else dataclasses.asdict(m)).encode()
        link = real_link(auth, zmq_module=z, robot_msg=msg)
        if env.problem == "hold_release_failure":
            def fail_hold(*a):
                raise OSError("fake release hold failure")
            link._send_measured_ee_hold = fail_hold
        if env.problem == "idle_release_failure":
            request = link._request_mode
            def fail_idle(mode):
                if mode == 1:
                    raise OSError("fake release IDLE failure")
                request(mode)
            link._request_mode = fail_idle
        link.update_peers = lambda: {"command": 0 if env.problem == "disconnect" and link.actions_sent > 10 else 1,
                                    "action": 1}
        engage_original, release_original, send_original = link.engage, link.release, link.send_flange_pose

        def wait(timeout_s, *, cancel):
            assert timeout_s == 3.0 and cancel is env.sessions[0].stop_event
            env.order.append("wait")
            if env.problem == "after_wait":
                trigger_stop()
            return link.connected()

        def engage(*, cancel):
            assert env.sessions[0].shutdown.link is link  # release is already registered
            env.order.append("engage")
            engage_original(settle_s=0, cancel=cancel)
            if env.problem == "ctrlc_engage":
                raise KeyboardInterrupt("during engage")
            if env.problem == "after_engage":
                trigger_stop()

        def release(*a, **kw):
            assert all(not s._exec_thread or not s._exec_thread.is_alive() for s in env.sessions)
            assert env.workers[0]._target is None
            env.order.append("release")
            outcome = release_original(*a, wait_still_s=0.08, **kw)
            env.order.append("released")
            return outcome

        def send(pose):
            if env.problem == "send_failure" and link.actions_sent >= 2:
                raise OSError("fake send failure")
            send_original(pose)

        link.wait_subscribers, link.engage, link.release, link.send_flange_pose = wait, engage, release, send
        env.links.append(link)
        env.zs.append(z)
        if env.problem == "constructed":
            trigger_stop()
        return link

    class Test(real_test):
        def __init__(self, *a, **kw):
            super().__init__(*a, **kw)
            env.sessions.append(self)

        def run(self):
            if self.recorder is None:
                env.started = time.monotonic_ns()
            return super().run()

        def join_executor(self, *a):
            super().join_executor(*a)
            env.order.append("executor joined")

    def authorize(**kw):
        env.authorizations.append(kw)
        return real_auth(**kw)

    def operator_input():
        env.prompts.append(1)
        if env.problem in ("prompt_sigterm", "prompt_sigint"):
            sig = signal.SIGTERM if env.problem == "prompt_sigterm" else signal.SIGINT
            env.handlers[sig](sig, None)
            pytest.fail("signal did not interrupt START input")
        if env.problem == "cancel_enter":
            return "s"
        return ""

    def invoke(test, *args):
        return hwtest.main([test, *( ["--dwell-s", "0.5"] if test == "gripper" else ["--duration-s", "1"]), *args],
                           operator_input=operator_input)

    monkeypatch.setattr(sc, "LiveStateReader", Reader)
    monkeypatch.setattr(G, "RobotiqRTU", lambda: real_rtu(serial_module=SimpleNamespace(Serial=Serial),
                                                       lock_path=str(tmp_path / "rtu.lock"),
                                                       check_holders=lambda port: [], ioctl=lambda *a: None,
                                                       listen_s=0))
    monkeypatch.setattr(G, "GripperWorker", Worker)
    monkeypatch.setattr(cameras, "LiveCameras", Cameras)
    monkeypatch.setattr(recorder, "SessionRecorder", Recorder)
    monkeypatch.setattr(controller, "ControllerLink", make_link)
    monkeypatch.setattr(hwtest, "HardwareTest", Test)
    monkeypatch.setattr(safety, "authorize_hardware_test", authorize)
    original_sink = live.LiveSink

    def make_sink(auth, link, *, gripper):
        assert gripper is None  # hold may never command the measured gripper
        env.order.append("sink without gripper")
        return original_sink(auth, link, gripper=gripper)

    monkeypatch.setattr(live, "LiveSink", make_sink)
    env.invoke, env.forbid = invoke, forbidden
    return env


def _verdict(env):
    import json

    texts = [e for e in env.recorders[0].events if e.startswith("verdict ")]
    assert len(texts) == 1
    return json.loads(texts[0][len("verdict "):])


def test_gripper_rehearsal_reads_without_writes(hw_env, monkeypatch):
    from ops.robot_pc import controller, safety

    monkeypatch.setattr(controller, "ControllerLink", hw_env.forbid)
    monkeypatch.setattr(safety, "authorize_hardware_test", hw_env.forbid)
    assert hw_env.invoke("gripper", "--rehearse", "--camera", "scene") == 0
    assert not hw_env.serial_writes and not hw_env.targets and not hw_env.links and not hw_env.prompts
    assert hw_env.workers[0].read_only
    v = _verdict(hw_env)
    assert v["pass"] and v["writes"] == 0 and v["status_reads"] > 10
    assert v["verdict"] == f"rehearsal: {v['status_reads']} status reads, 0 writes"
    assert hw_env.recorders[0].status == "complete" and hw_env.recorders[0].frames
    assert hw_env.camera_kw[0]["keep_native"] and hw_env.camera_kw[0]["allow_controller"]
    assert all(r[1] == 3 for s in hw_env.serials for r in s.requests)


def test_gripper_real_targets_expire_and_are_written_in_order(hw_env, monkeypatch):
    from ops.robot_pc import controller
    from ops.robot_pc.gripper import closure_to_position

    monkeypatch.setattr(controller, "ControllerLink", hw_env.forbid)
    assert hw_env.invoke("gripper") == 0
    assert hw_env.serial_writes[:2] == [closure_to_position(0.3), closure_to_position(0.0)]
    assert hw_env.serial_writes[2:] and set(hw_env.serial_writes[2:]) == {3}
    assert [c for c, expiry, now in hw_env.targets] == [0.3, 0.0]
    assert all(0.99 <= (expiry - now) / 1e9 <= 1.0 for c, expiry, now in hw_env.targets)
    assert hw_env.authorizations == [{"test": "gripper", "cli_flag": True, "operator_start": True}]
    assert hw_env.workers[0]._target is None and not hw_env.workers[0].is_alive()
    v, rec = _verdict(hw_env), hw_env.recorders[0]
    assert v["pass"] and v["writes"] == 2 + len(hw_env.workers[0].holds) and all(t["reached"] for t in v["targets"])
    assert all(0 <= t["time_to_reach_s"] < 0.5 and t["faults"] == [0] for t in v["targets"])
    assert rec.status == "complete" and rec.robot and rec.grip
    assert rec.meta["mode"] == "hwtest-gripper" and rec.meta["config"]["executor"]["max_linear_speed"] == 0.05


@pytest.mark.parametrize("problem", ["contact", "fault"])
def test_gripper_aborts_and_clears_target(hw_env, problem):
    hw_env.problem = problem
    assert hw_env.invoke("gripper") == 1
    v = _verdict(hw_env)
    assert not v["pass"] and len(v["targets"]) == 1 and len(hw_env.targets) == 1
    assert ("contact" if problem == "contact" else "fault 0x05") in " ".join(v["abort_reasons"])
    assert hw_env.workers[0]._target is None and hw_env.recorders[0].status == "stopped"


def test_gripper_refuses_closed_precondition(hw_env):
    hw_env.problem = "not_open"
    assert hw_env.invoke("gripper") == 3
    assert not hw_env.targets and not hw_env.serial_writes and not hw_env.prompts
    assert "essentially open" in " ".join(_verdict(hw_env)["abort_reasons"])


@pytest.mark.parametrize("test", ["gripper", "hold"])
@pytest.mark.parametrize("problem", ["before_enter", "cancel_enter"])
def test_cancel_before_start_sends_nothing(hw_env, test, problem):
    hw_env.problem = problem
    assert hw_env.invoke(test, "--camera", "scene") == 130
    assert not hw_env.authorizations and not hw_env.links and not hw_env.serial_writes and not hw_env.targets
    # the scene camera starts before the gripper owner and the recorder: a cancel at its bind precedes the recording
    assert (problem == "before_enter" and not hw_env.recorders) or hw_env.recorders[0].status == "stopped"


def test_hold_rehearsal_never_builds_a_commander(hw_env, monkeypatch):
    from ops.robot_pc import controller, safety

    monkeypatch.setattr(controller, "ControllerLink", hw_env.forbid)
    monkeypatch.setattr(safety, "authorize_hardware_test", hw_env.forbid)
    assert hw_env.invoke("hold", "--rehearse") == 0
    v, rec = _verdict(hw_env), hw_env.recorders[0]
    assert not hw_env.links and not hw_env.targets and not hw_env.serial_writes and not hw_env.prompts
    assert hw_env.workers[0].read_only and v["pass"] and v["duration_held_s"] >= 1
    assert v["actions_sent"] == v["mode_requests_sent"] == 0
    assert len(rec.executions) > 100 and not any(r[4] for r in rec.executions)
    assert rec.meta["mode"] == "hwtest-hold-rehearsal" and rec.commands


def test_hold_real_streams_start_pose_then_releases_before_close(hw_env):
    import json

    import numpy as np

    from ops.robot_pc import switch_controller as sc

    assert hw_env.invoke("hold", "--camera", "scene") == 0
    v, rec, order = _verdict(hw_env), hw_env.recorders[0], hw_env.order
    assert v["pass"] and v["duration_held_s"] >= 1 and v["max_drift_mm"] == v["max_abs_dq"] == 0
    assert v["loop_stats"]["executor_nominal_hz"] == 500 and v["executor_counts"]["track"] > 100
    assert v["loop_stats"]["period_ms_p99"] > 0 and "overruns_gt_1p5x" in v["loop_stats"]
    assert hw_env.authorizations == [{"test": "hold", "cli_flag": True, "operator_start": True}]
    assert "sink without gripper" in order and not hw_env.targets and not hw_env.serial_writes
    assert order.index("executor joined") < order.index("gripper clear") < order.index("release")
    assert order.index("released") < order.index("gripper stopped") < order.index("recorder close")
    assert order.index("released") < order.index("recorder close") < order.index("camera close") < order.index("reader close")
    assert hw_env.workers[0].read_only and rec.status == "complete"
    assert all(r[4] and json.loads(r[5]) == {"arm": "sent", "gripper": "not queued"} for r in rec.executions)
    assert all(c["session_id"] == "hwtest-hold" and c["reference_id"].startswith("hold:") and
               c["expires_ns"] - c["observation_time_ns"] == 200_000_000 for c in rec.commands)
    assert all(c["goal_pose7"] == list(hw_env.pose) for c in rec.commands)
    for r in rec.executions:
        np.testing.assert_allclose(r[2], hw_env.pose, atol=1e-9, rtol=0)
    msgs = [json.loads(b) for b in hw_env.zs[0].sent]
    assert msgs[0] == {"ctrl_mode": 3} and msgs[-1] == {"ctrl_mode": 1}
    actions = [m["real_time_action"] for m in msgs if "real_time_action" in m]
    assert len(actions) == v["actions_sent"] == len(rec.executions) + 1
    assert v["mode_requests_sent"] == 2
    for action in actions:  # includes the final measured-ee hold
        p = action["ee_pose_rt"][0]
        got = [p["position"][k] for k in ("x", "y", "z")] + [p["orientation"][k] for k in ("x", "y", "z", "w")]
        np.testing.assert_allclose(got, sc.ee_from_flange(hw_env.pose), atol=1e-9, rtol=0)


@pytest.mark.parametrize("problem,reason", [("drift", "drift"), ("rotation", "drift"), ("dq", "dq"),
                                           ("stale", "stale robot state"), ("frozen", "frozen"),
                                           ("disconnect", "subscriber disconnected"), ("send_failure", "send failed")])
def test_hold_aborts_latches_and_releases(hw_env, problem, reason):
    import json

    hw_env.problem = problem
    assert hw_env.invoke("hold") == 1
    v, rec = _verdict(hw_env), hw_env.recorders[0]
    assert not v["pass"] and reason in " ".join(v["abort_reasons"])
    assert v["duration_held_s"] < 1 and rec.status == "stopped"
    assert hw_env.order.count("release") == 1
    assert hw_env.order.index("released") < hw_env.order.index("recorder close")
    assert json.loads(hw_env.zs[0].sent[-1]) == {"ctrl_mode": 1}
    assert not hw_env.targets and not hw_env.serial_writes
    if problem == "drift":
        assert v["max_drift_mm"] == pytest.approx(6)
    if problem == "dq":
        assert v["max_abs_dq"] == pytest.approx(0.06)
    if problem == "send_failure":
        assert rec.executions[-1][4] is False and "fake send failure" in rec.executions[-1][5]


@pytest.mark.parametrize("problem", ["constructed", "after_wait", "after_engage", "ctrlc_engage"])
def test_hold_startup_cancellation_always_releases(hw_env, problem):
    import json

    hw_env.problem = problem
    assert hw_env.invoke("hold") == 130
    assert hw_env.order.count("release") == 1
    assert hw_env.order.index("released") < hw_env.order.index("recorder close")
    assert not hw_env.recorders[0].executions and not hw_env.targets
    modes = [m["ctrl_mode"] for b in hw_env.zs[0].sent if "ctrl_mode" in (m := json.loads(b))]
    assert modes == ([3, 1] if problem in ("after_engage", "ctrlc_engage") else [])
    assert not _verdict(hw_env)["pass"]


@pytest.mark.parametrize("test", ["gripper", "hold"])
def test_hwtest_no_record_and_initial_gripper_fault_clears(hw_env, test):
    hw_env.problem = "preflight_fault_clears"
    assert hw_env.invoke(test, "--rehearse", "--no-record") == 0
    assert not hw_env.recorders and not hw_env.serial_writes


def test_hwtest_persistent_preflight_fault_refuses(hw_env):
    hw_env.problem = "preflight_fault"
    assert hw_env.invoke("gripper") == 3
    assert not hw_env.prompts and not hw_env.serial_writes and not hw_env.links
    assert not hw_env.workers[0].is_alive()


@pytest.mark.parametrize("args", [("gripper", "--closures", "0.6"), ("gripper", "--dwell-s", "nan"),
                                  ("gripper", "--dwell-s", "0.4"), ("hold", "--duration-s", "0.5"),
                                  ("hold", "--duration-s", "31"), ("hold", "--max-drift-mm", "nan"),
                                  ("hold", "--max-dq", "0")])
def test_hwtest_invalid_arguments_open_nothing(hw_env, args):
    from scripts.robot_pc import hwtest

    with pytest.raises(SystemExit) as exc:
        hwtest.main(args)
    assert exc.value.code == 2 and not hw_env.workers and not hw_env.links and not hw_env.free_calls


@pytest.mark.parametrize("mode", ["hwtest-gripper", "hwtest-gripper-rehearsal", "hwtest-hold", "hwtest-hold-rehearsal"])
@pytest.mark.parametrize("passed", [True, False])
def test_verify_hwtest_verdict_without_policy_steps(tmp_path, mode, passed):
    import json

    import numpy as np

    from ops.robot_pc import switch_controller as sc
    from ops.robot_pc.recorder import SessionRecorder
    from scripts.robot_pc.verify_session import verify

    test = "gripper" if "gripper" in mode else "hold"
    meta = {"mode": mode, "test": test, "config": BASE, "gripper_origin": "measured fake"}
    rec = SessionRecorder(tmp_path, meta=meta, cameras={}, video=False)
    pose = (0.42, 0.0, 0.55, 0, 0, 0, 1)
    for i in range(100):
        rec.robot_state(1_000_000_000 + i * 1_000_000, i, np.ones(7), np.zeros(7), np.zeros(7),
                        pose, sc.ee_from_flange(pose))
    for i in range(5):
        rec.gripper(1_000_000_000 + i * 20_000_000, 3, {"fault": 0, "status": 3})
    verdict = {"test": test, "pass": passed, "rehearse": mode.endswith("-rehearsal"),
               "abort_reasons": [] if passed else ["injected abort"], "writes": 0, "actions_sent": 0,
               "release": {"hold": "sent", "still": True, "idle": "sent"}}
    rec.event(1_100_000_000, "verdict " + json.dumps(verdict))
    rec.close("complete" if passed else "stopped")
    report = verify(rec.session_dir)
    assert report["ok"] is passed
    assert report["checks"]["hwtest_verdict"]["ok"] is passed
    assert "steps" not in report["checks"] and "action_transformations" not in report["checks"]
    assert "latency_ms" not in report


@pytest.mark.parametrize("test", ["gripper", "hold"])
def test_hwtest_real_recording_verifies_with_fake_hardware(hw_env, monkeypatch, test):
    import h5py

    from ops.robot_pc import recorder
    from scripts.robot_pc.verify_session import verify

    recorded = []

    def record(root, **kw):
        r = hw_env.real_recorder(root, **kw, video=False)
        recorded.append(r)
        return r

    monkeypatch.setattr(recorder, "SessionRecorder", record)
    assert hw_env.invoke(test) == 0
    r = recorded[0]
    report = verify(r.session_dir)
    assert report["ok"], report["failures"]
    with h5py.File(r.session_dir / "inference.h5", "r") as h:
        assert len(h["steps/json"]) == 0
        executed = len(h["executed/t_ns"])
        assert executed > 100 if test == "hold" else executed == 0


@pytest.mark.parametrize("test", ["gripper", "hold"])
@pytest.mark.parametrize("problem", ["prompt_sigterm", "prompt_sigint", "ready_sigterm"])
def test_hwtest_prompt_signals_cancel_immediately_without_engagement(hw_env, test, problem):
    hw_env.problem = problem
    assert hw_env.invoke(test) == 130
    assert bool(hw_env.prompts) is (problem != "ready_sigterm")
    assert not hw_env.authorizations and not hw_env.links and not hw_env.serial_writes
    assert hw_env.recorders[0].status == "stopped" and not hw_env.workers[0].is_alive()


@pytest.mark.parametrize("problem", ["cleanup_signals", "cleanup_baseexception"])
def test_hwtest_cleanup_finishes_recorder_and_resources_preserving_exit_code(hw_env, problem):
    hw_env.problem = problem
    assert hw_env.invoke("hold", "--rehearse", "--camera", "scene") == 0
    assert hw_env.recorders[0].status == "complete"
    assert hw_env.order[-1] == "reader close"
    if problem == "cleanup_signals":
        assert hw_env.messages.count("shutdown in progress") == 3 and "recorder finalized" in hw_env.order


@pytest.mark.parametrize("problem,phase", [("hold_release_failure", "hold"), ("idle_release_failure", "idle")])
def test_hold_release_failures_are_recorded_and_fail_verdict_and_exit(hw_env, problem, phase):
    import json

    hw_env.problem = problem
    assert hw_env.invoke("hold") == 1
    verdict = _verdict(hw_env)
    assert not verdict["pass"] and not verdict["abort_reasons"] and verdict["duration_held_s"] >= 1
    assert verdict["release"][phase].startswith("error:")
    assert hw_env.recorders[0].status == "stopped"
    events = [json.loads(e[len("release "):]) for e in hw_env.recorders[0].events if e.startswith("release ")]
    assert events == [verdict["release"]]


def test_policy_authorization_cannot_run_hwtest_hold(hw_env, monkeypatch):
    from ops.robot_pc import safety

    auth = safety.authorize_execution(cli_execute=True, operator_start=True, policy_info={"execution_eligible": True})
    monkeypatch.setattr(safety, "authorize_hardware_test", lambda **kw: auth)
    assert hw_env.invoke("hold") == 3
    assert not hw_env.links and not hw_env.serial_writes
    assert "requires its hardware_test authorization" in " ".join(_verdict(hw_env)["abort_reasons"])
    # The hold runner itself also checks scope, even if a caller bypasses the launcher.
    session = hw_env.sessions[0]
    session.stop_event.clear()
    session.sink = type("PolicySink", (), {"authorization": auth})()
    with pytest.raises(safety.ExecutionRefused, match="hardware_test == 'hold'"):
        session._hold_test()


@pytest.mark.parametrize("release", [None, {"hold": "sent", "still": True, "idle": "error: injected"},
                                    {"hold": "sent", "still": True, "idle": "sent"}])
@pytest.mark.parametrize("mode", ["execute", "hwtest-hold"])
def test_verifier_requires_recorded_idle_after_engagement(tmp_path, release, mode):
    import json
    import numpy as np

    from ops.robot_pc.recorder import SessionRecorder
    from ops.robot_pc import switch_controller as sc
    from scripts.robot_pc.verify_session import verify

    pose = (0.42, 0, 0.55, 0, 0, 0, 1)
    rec = SessionRecorder(tmp_path, meta={"mode": mode, "test": "hold", "config": BASE}, cameras={}, video=False)
    for i in range(100):
        rec.robot_state(10**9 + i * 10**6, i, np.ones(7), np.zeros(7), np.zeros(7), pose, sc.ee_from_flange(pose))
    rec.event(10**9, "engage")
    if release is not None:
        rec.event(10**9 + 10**8, "release " + json.dumps(release))
    rec.close("complete")
    result = verify(rec.session_dir)
    assert result["checks"]["release"]["ok"] is (release is not None and release["idle"] == "sent")


@pytest.mark.parametrize("boundary", ["first", "last", "after_last_state", "before_first_state", "fresh"])
def test_verifier_includes_overlapping_gaps_and_ages_every_setpoint(tmp_path, boundary):
    import numpy as np

    from ops.robot_pc.recorder import SessionRecorder
    from ops.robot_pc import switch_controller as sc
    from scripts.robot_pc.verify_session import verify

    pose = (0.42, 0, 0.55, 0, 0, 0, 1)
    rt = np.arange(10**9, 3 * 10**9, 10**6)
    start, end = 1_500_000_000, 2_500_000_000
    if boundary == "first":
        rt = rt[(rt < start - 60_000_000) | (rt > start + 10_000_000)]
    if boundary == "last":
        rt = rt[(rt < end - 60_000_000) | (rt > end + 10_000_000)]
    if boundary == "after_last_state":
        end = int(rt[-1] + 60_000_000)
    if boundary == "before_first_state":
        start = int(rt[0] - 1)
    rec = SessionRecorder(tmp_path, meta={"mode": "hwtest-hold-rehearsal", "test": "hold", "config": BASE},
                          cameras={}, video=False)
    for i, t in enumerate(rt):
        rec.robot_state(int(t), i, np.ones(7), np.zeros(7), np.zeros(7), pose, sc.ee_from_flange(pose))
    # Every setpoint is checked even when message-byte verification is subsampled.
    for t in (start, (start + end) // 2, end):
        rec.executed(t, "track", pose, None, False, "rehearsal")
    rec.close("complete")
    result = verify(rec.session_dir, max_setpoints=1)
    check = result["checks"]["executed"]
    assert check["ok"] is (boundary == "fresh")
    if boundary in ("first", "last"):
        assert result["checks"]["robot_state"]["streaming_dt_ms_max"] > 50
        assert check["max_robot_state_age_ms"] > 50
    if boundary == "after_last_state":
        assert check["max_robot_state_age_ms"] == 60
    if boundary == "before_first_state":
        assert check["setpoints_without_preceding_state"] == 1
    if boundary == "fresh":
        assert check["max_robot_state_age_ms"] == 0
