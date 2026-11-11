"""infer.py (directive K): camera names, contract checks, --list, per-run output folders (CPU, fakes; nothing real)."""

from __future__ import annotations

import io
import json
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import yaml

from ops.robot_pc import config as C
from ops.robot_pc.safety import ExecutionRefused

REPO = Path(__file__).resolve().parents[1]
NAMES = {"left": "135122070988", "middle": "135122076054", "right": "135122073201"}
ALL3 = [["135122070988", "135122073201", "135122076054"]]


def _infer():
    from scripts.robot_pc import infer

    return infer


def test_camera_names_file_is_the_rig_geometry_mapping(tmp_path):
    m = yaml.safe_load((REPO / "configs/robot_pc/camera_names.yaml").read_text())
    assert m == NAMES
    cfg = C.load(root=tmp_path)  # no copy under root: the repo template is used
    assert _infer().camera_names(cfg) == NAMES
    (tmp_path / "config").mkdir()
    (tmp_path / "config/camera_names.yaml").write_text("front: '135122076054'\n")
    assert _infer().camera_names(cfg) == {"front": "135122076054"}  # the user's edited copy wins
    (tmp_path / "config/camera_names.yaml").write_text("a: '1'\nb: '1'\n")
    with pytest.raises(SystemExit, match="every serial used once"):
        _infer().camera_names(cfg)


def test_rig_geometry_puts_left_at_plus_y_middle_at_zero_right_at_minus_y():
    """Camera positions in the robot base frame from the released rig documents (calrel-6c3b9ec drawer-open; the
    documents ship with the robot-PC bundle, the robot-PC selfcheck re-checks them there): base +y = robot's left."""
    released = {"135122070988": (1.210, 0.421, 0.839), "135122076054": (1.193, 0.000, 0.837),
                "135122073201": (1.186, -0.431, 0.841)}
    cams = {}
    for serial, p in released.items():
        T_base_cam = np.eye(4)
        T_base_cam[:3, 3] = p
        cams[f"{serial}/color"] = {"T_cam_base": np.linalg.inv(T_base_cam).reshape(-1).tolist()}
    y = {k.split("/")[0]: np.linalg.inv(np.array(v["T_cam_base"]).reshape(4, 4))[1, 3] for k, v in cams.items()}
    assert y[NAMES["left"]] > 0.3 and abs(y[NAMES["middle"]]) < 0.05 and y[NAMES["right"]] < -0.3


def test_resolve_cameras_names_count_and_training_contract():
    resolve = _infer().resolve_cameras
    single = {"views": 1, "trained_camera_orders": ALL3}
    assert resolve(single, NAMES, None) == ["135122070988"]  # default: the contract's first trained camera
    assert resolve(single, NAMES, "right") == ["135122073201"]
    assert resolve(single, NAMES, None, "135122076054") == ["135122076054"]
    with pytest.raises(ExecutionRefused, match="single-view: give exactly ONE camera"):
        resolve(single, NAMES, "left,right")
    with pytest.raises(ExecutionRefused, match="unknown camera name"):
        resolve(single, NAMES, "front")
    only_left = {"views": 1, "trained_camera_orders": [["135122070988"]]}
    with pytest.raises(ExecutionRefused, match="not a trained camera"):
        resolve(only_left, NAMES, "middle")
    multi = {"views": 3, "trained_camera_orders": [["135122070988", "135122076054", "135122073201"]]}
    with pytest.raises(ExecutionRefused, match="needs 3 cameras in training order: left,middle,right"):
        resolve(multi, NAMES, "left,right")
    with pytest.raises(ExecutionRefused, match="multi-view .* not implemented"):
        resolve(multi, NAMES, "left,middle,right")


def test_list_shows_task_method_seed_eligibility(monkeypatch, capsys, tmp_path):
    from ops.robot_pc import cli

    offered = {
        "e2e_cnn-drawer-open-s0": {"info": {"task": "drawer-open", "condition": "e2e_cnn", "seed": 0,
                                            "status": "deployable", "execution_eligible": True},
                                   "input_spec": {"views": 1, "modalities": ["rgb"], "trained_camera_orders": ALL3}},
        "dynamo-drawer-open-s0": {"info": {"task": "drawer-open", "condition": "dynamo", "seed": 0,
                                           "status": "restricted", "execution_eligible": False},
                                  "input_spec": {"views": 1, "modalities": ["rgb"], "trained_camera_orders": ALL3}},
    }
    offered["e2e_cnn-drawer-open"] = {"info": {**offered["e2e_cnn-drawer-open-s0"]["info"],
                                               "alias_of": "e2e_cnn-drawer-open-s0"},
                                      "input_spec": offered["e2e_cnn-drawer-open-s0"]["input_spec"]}
    from ops.robot_pc import selection

    assert selection.select_policy(offered, condition="e2e_cnn", task="drawer-open", seed=0) == "e2e_cnn-drawer-open-s0"
    monkeypatch.setattr(cli, "welcome", lambda cfg, args: {"policies": offered})
    monkeypatch.setattr(cli, "load_config", lambda args: C.load(root=tmp_path))
    assert _infer().main(["--list"]) == 0
    out = capsys.readouterr().out.splitlines()
    rows = {r.split()[3]: r.split() for r in out[1:] if r.strip() and not r.startswith("camera names")}
    assert rows["e2e_cnn-drawer-open-s0"][:5] == ["drawer-open", "e2e_cnn", "0", "e2e_cnn-drawer-open-s0", "yes"]
    assert rows["dynamo-drawer-open-s0"][4] == "no" and rows["dynamo-drawer-open-s0"][-1] == "left,right,middle"
    assert "e2e_cnn-drawer-open" not in rows and "1 legacy alias ids not shown" in out[-1]


def test_run_folders_are_named_numbered_and_never_reused(tmp_path):
    from ops.robot_pc.recorder import SessionRecorder

    cams = {"135122073201": {"native_hw": [4, 6], "K_native": np.eye(3).tolist()}}
    dirs = []
    for _ in range(3):
        r = SessionRecorder(tmp_path, meta={"x": 1}, cameras=cams, run_prefix="drawer-open/e2e_cnn/seed0/right",
                            camera_names={"135122073201": "right"}, files={"config.json": {"policy": "p"}})
        r.frame("135122073201", 1, 10, np.zeros((4, 6, 3), np.uint8), None, None, {})
        summary = r.close()
        dirs.append(Path(summary["session_dir"]).name)
        d = Path(summary["session_dir"])
        assert json.loads((d / "config.json").read_text()) == {"policy": "p"}
        assert (d / "inference.h5").is_file() and (d / "summary.json").is_file()
        video = summary["videos"]["right"]
        assert video["status"] != "available" or Path(video["file"]).name == "right.mp4"
    assert dirs == [f"run{i:03d}" for i in (1, 2, 3)]
    base = tmp_path / "inference_outputs/drawer-open/e2e_cnn/seed0/right"
    assert all(d.parent == base for d in base.iterdir())
    (base / "run007").mkdir()  # a gap: next is 1 + max
    r = SessionRecorder(tmp_path, meta={}, cameras=cams, run_prefix="drawer-open/e2e_cnn/seed0/right")
    assert r.session_dir == base / "run008"
    r.close()
    with pytest.raises(ValueError):
        SessionRecorder(tmp_path, meta={}, cameras=cams, run_prefix="../escape")
    with pytest.raises(ValueError):
        SessionRecorder(tmp_path, meta={}, cameras=cams, run_prefix="drawer-open//right")


def test_infer_cli_refuses_conflicting_camera_flags(capsys):
    with pytest.raises(SystemExit) as exc:
        _infer().main(["--cameras", "left", "--camera", "135122070988", "--dry-run"])
    assert exc.value.code == 2 and "not both" in capsys.readouterr().err


@pytest.fixture
def warmup_launcher(monkeypatch, tmp_path):
    """Exercise the real gate through main(); all surrounding resources are inert fakes."""
    from ops.robot_pc import (
        camera_warmup,
        cameras,
        cli,
        controller,
        executor,
        gripper,
        live,
        recorder,
        selection,
        sinks,
        switch_controller,
        tunnel,
    )

    run = _infer()
    order, events, configs, results, messages = [], [], [], [], []
    now, cancel = [0], []
    scenario = {"status": "stable"}
    cfg = {"root": str(tmp_path), "camera_warmup": {"timeout_s": 10.0}}

    def clock():
        return now[0]

    def sleep(seconds):
        now[0] += round(seconds * 1e9)
        if cancel and scenario["status"] == "cancelled":
            cancel[0].set()

    class Cameras:
        def __init__(self, *a, **kw):
            self.errors, self.index = Counter(), 0

        def start(self, corrections):
            return {"a": np.eye(3)}

        def bind(self, K):
            self.start_ns = clock()
            order.append("bind")

        def drain(self, serial):
            if scenario["status"] == "no_frames":
                return []
            out = []
            while self.start_ns + round(self.index * 1e9 / 15) <= clock():
                r = 115 if scenario["status"] != "timeout" else 70 + 60 * (self.index % 2)
                out.append({"capture_ns": self.start_ns + round(self.index * 1e9 / 15),
                            "rgb": np.full((128, 128, 3), [r, 100, 100], np.uint8)})
                self.index += 1
            return out

        def close(self):
            order.append("cameras close")

    class Recorder:
        session_dir = tmp_path

        def __init__(self, *a, **kw):
            configs.append(json.loads(json.dumps(kw["files"]["config.json"])))
            order.append("record")

        def event(self, t_ns, text):
            events.append(text)
            if text.startswith("camera_warmup "):
                order.append("warmup event")

        def close(self, *a):
            order.append("recorder close")
            return {}

    class Shutdown:
        _thread = None
        release_outcome = {"hold": "sent", "idle": "sent"}

        def __init__(self, *a, **kw):
            pass

        def finish(self):
            order.append("cleanup")

    class Link:
        mode_requests_sent = 0

        def __init__(self, *a):
            order.append("link")

        def wait_subscribers(self, *a, **kw):
            return True

        def engage(self, **kw):
            order.append("engage")

    class Session:
        stop_reason = "budget reached"

        def __init__(self, **kw):
            order.append("session")

        def run(self, **kw):
            order.append("stream")
            return []

        def loop_stats(self):
            return {}

    real_wait = camera_warmup.wait_until_stable

    def wait(cams, serials, options, **kw):
        assert order == ["bind", "record"] or order == ["bind"]
        order.append("warmup")
        cancel.append(kw["cancel"])
        result = real_wait(cams, serials, options, clock=clock, sleep=sleep, **kw)
        results.append(result)
        order.append("warmup done")
        return result

    monkeypatch.setattr(run.time, "sleep", sleep)
    monkeypatch.setattr(run.sys, "stdin", io.StringIO(""))
    monkeypatch.setattr(run, "say", messages.append)
    monkeypatch.setattr(cli, "load_config", lambda args: cfg)
    monkeypatch.setattr(run, "camera_names", lambda cfg: {"front": "a"})
    monkeypatch.setattr(tunnel, "welcome", lambda *a: {"fake": True})
    from ops.robot_pc import devices

    monkeypatch.setattr(devices, "require_free", lambda **kw: None)
    monkeypatch.setattr(devices, "scan", lambda **kw: {"controllers": [1]})
    monkeypatch.setattr(switch_controller, "LiveStateReader", lambda **kw:
                        SimpleNamespace(samples=[], close=lambda: order.append("reader close")))
    monkeypatch.setattr(switch_controller, "state_blockers", lambda *a: [])
    monkeypatch.setattr(gripper, "GripperWorker", lambda *a, **kw:
                        SimpleNamespace(start=lambda: None, healthy=lambda: True, error=None, unverifiable=0,
                                        latest=SimpleNamespace(position=3, status=3, fault=0)))
    monkeypatch.setattr(live, "LiveState", lambda *a: object())
    monkeypatch.setattr(live, "MotionShutdown", Shutdown)
    monkeypatch.setattr(live, "LiveSink", lambda *a: object())
    monkeypatch.setattr(live, "LiveSession", Session)
    monkeypatch.setattr(sinks, "DryRunSink", lambda: object())
    monkeypatch.setattr(controller, "ControllerLink", Link)
    monkeypatch.setattr(executor, "HighRateExecutor", lambda *a, **kw: SimpleNamespace(counts={}))
    monkeypatch.setattr(C, "limits", lambda *a: object())
    monkeypatch.setattr(cli, "choose_policy", lambda *a:
                        ("p", "t", {"views": 1, "modalities": ["rgb"], "trained_camera_orders": [["a"]]},
                         {"execution_eligible": True}))
    monkeypatch.setattr(cli, "rig_document", lambda *a: None)
    monkeypatch.setattr(selection, "depth_corrections", lambda *a, **kw: {})
    monkeypatch.setattr(selection, "session_choices", lambda *a, **kw: {})
    monkeypatch.setattr(cli, "connect", lambda *a: SimpleNamespace(close=lambda: order.append("client close")))
    monkeypatch.setattr(cameras, "LiveCameras", Cameras)
    monkeypatch.setattr(recorder, "SessionRecorder", Recorder)
    monkeypatch.setattr(camera_warmup, "wait_until_stable", wait)

    def launch(args=()):
        def operator():
            assert results and "warmup event" in order
            order.append("prompt")
            return ""
        return run.main(["--policy", "p", *args], operator_input=operator)

    return SimpleNamespace(launch=launch, order=order, events=events, configs=configs, cfg=cfg,
                           results=results, scenario=scenario, messages=messages)


@pytest.mark.parametrize("dry_run", [False, True])
def test_warmup_precedes_prompt_engagement_and_streaming_and_records_event(warmup_launcher, dry_run):
    f = warmup_launcher
    assert f.launch(["--dry-run"] if dry_run else []) == 0
    r = f.results[0]
    assert r["status"] == "stable" and 1.5 <= r["waited_s"] < 1.6
    assert json.loads(next(e.removeprefix("camera_warmup ") for e in f.events
                           if e.startswith("camera_warmup "))) == r
    assert f.order.index("bind") < f.order.index("record") < f.order.index("warmup")
    assert f.order.index("warmup done") < f.order.index("warmup event") < f.order.index("stream")
    if dry_run:
        assert "prompt" not in f.order and "link" not in f.order
    else:
        assert f.order.index("warmup event") < f.order.index("prompt") < f.order.index("engage")


@pytest.mark.parametrize("dry_run", [False, True])
@pytest.mark.parametrize("on_timeout,code", [("proceed", 0), ("refuse", 3)])
def test_warmup_timeout_policy_is_enforced_before_start(warmup_launcher, dry_run, on_timeout, code):
    f = warmup_launcher
    f.scenario["status"] = "timeout"
    f.cfg["camera_warmup"].update(timeout_s=2, on_timeout=on_timeout)
    assert f.launch(["--dry-run"] if dry_run else []) == code
    assert f.results[0]["status"] == "timeout"
    assert ("stream" in f.order) == (code == 0)
    if code == 3:
        assert "prompt" not in f.order and "link" not in f.order
    else:
        assert any("WARNING" in m and "proceeding" in m for m in f.messages)


@pytest.mark.parametrize("status,code", [("no_frames", 3), ("cancelled", 130)])
def test_warmup_failure_refuses_without_prompt_or_engagement(warmup_launcher, status, code):
    f = warmup_launcher
    f.scenario["status"] = status
    assert f.launch() == code and f.results[0]["status"] == status
    assert "prompt" not in f.order and "link" not in f.order and "stream" not in f.order
    assert "warmup event" in f.order and "recorder close" in f.order


@pytest.mark.parametrize("timeout", [0, 0.2])
def test_warmup_cli_override_is_effective_and_recorded(warmup_launcher, timeout):
    f = warmup_launcher
    assert f.launch(["--dry-run", "--camera-warmup-timeout-s", str(timeout)]) == 0
    assert f.results[0]["status"] == ("disabled" if timeout == 0 else "timeout")
    assert f.configs[0]["config"]["camera_warmup"]["timeout_s"] == timeout


@pytest.mark.parametrize("timeout", ["-1", "nan", "inf"])
def test_warmup_cli_rejects_invalid_timeout(warmup_launcher, timeout):
    with pytest.raises(SystemExit) as exc:
        warmup_launcher.launch([f"--camera-warmup-timeout-s={timeout}"])
    assert exc.value.code == 2 and not warmup_launcher.order


def test_warmup_without_recording(warmup_launcher):
    f = warmup_launcher
    assert f.launch(["--dry-run", "--no-record"]) == 0
    assert f.results[0]["status"] == "stable" and not f.events and not f.configs


# ------------------------------------------------------------------------------------- review (robk-review) fixes
def _cmd(goal, t_obs=0, age_ms=1e6):
    from ops.robot_pc.validation import Command

    return Command("r", "s", int(t_obs), int(t_obs + age_ms * 1e6), tuple(POSE), tuple(goal), 0.0, 3, 0.0, 0.0)


POSE = (0.45, 0.0, 0.55, 1.0, 0.0, 0.0, 0.0)


def test_reengagement_after_a_pause_starts_from_the_measured_pose_not_an_old_setpoint():
    from ops.robot_pc import executor, sinks

    ex = executor.HighRateExecutor(sinks.MockSink())
    ex.accept(_cmd([POSE[0] + 0.15, *POSE[1:]]))
    t = 0
    for k in range(25):  # 50 ms of interpolation toward +0.15 m (last setpoint ~ +0.11 m)
        t += 2_000_000
        ex.tick(t, POSE, state_t_ns=t, signature=bytes([k]))
    t += 100_000_000  # no send for 100 ms (> target_dt): the controller now holds the MEASURED pose
    assert ex.accept(_cmd([POSE[0] + 0.02, *POSE[1:]], t_obs=1))
    sp = ex.tick(t, POSE, state_t_ns=t, signature=b"new")
    assert sp.pose7[0] == POSE[0]  # starts at the measured pose, never re-sends the obsolete +0.11 m target


def test_log6_small_angle_is_finite_and_the_guard_still_holds():
    from ops.robot_pc import executor, sinks
    from ops.robot_pc.poses import rotate_left

    far = [POSE[0] + 0.4, *POSE[1:3], *rotate_left(np.array(POSE[3:]), np.array([0.0, 0.0, 2e-9]))]
    m, r = executor.log6_norms(far, POSE)
    assert np.isfinite(m) and abs(m - 0.4) < 1e-9 and r < 1e-8
    ex = executor.HighRateExecutor(sinks.MockSink(), interpolation_s=1e-9)  # jump straight to the goal
    ex.accept(_cmd(far))
    assert ex.tick(2_000_000, POSE, state_t_ns=2_000_000, signature=b"a").pose7 == POSE  # segment start = measured
    assert ex.tick(4_000_000, POSE, state_t_ns=4_000_000, signature=b"b") is None  # 0.4 m: never sent
    assert ex.counts["controller_guard"] == 1


def test_expired_setpoint_is_dropped_under_the_transport_lock():
    import time

    from ops.robot_pc import controller, live, safety

    sent = []

    class Sock:
        def send(self, data, flags=0):
            sent.append(data)

    link = object.__new__(controller.ControllerLink)
    import threading

    link._send_lock, link._streaming, link.socks, link.actions_sent, link.expired_at_send = \
        threading.Lock(), True, {"action": Sock()}, 0, 0
    link.msg, link.zmq, link.target_dt = __import__("types").SimpleNamespace(encode=lambda m: b"x"), \
        __import__("types").SimpleNamespace(NOBLOCK=1), 0.05
    controller.sc.robot_action = lambda *a: "msg"
    assert link.send_flange_pose(POSE, time.monotonic_ns() - 1) is False and not sent and link.expired_at_send == 1
    assert link.send_flange_pose(POSE, time.monotonic_ns() + 10**9) is True and len(sent) == 1
    auth = safety.authorize_execution(cli_execute=True, operator_start=True, policy_info={"execution_eligible": True})
    sink = live.LiveSink(auth, link, None)
    sp = live.LiveSetpoint(0, "track", POSE, 0.0, "r", time.monotonic_ns() - 1)
    assert sink.send(sp)["arm"].startswith("expired at send") and len(sent) == 1


def test_validation_executes_the_raw_policy_goal_not_the_lead_clamped_one():
    import inspect

    from ops.robot_pc import validation

    src = inspect.getsource(validation.validate_action)
    assert 'o.get("flange_pose", o.get("execution_flange_pose"))' in src


def test_dry_run_selection_reaches_restricted_policies():
    from ops.robot_pc import selection

    offered = {"dynamo-drawer-open-s0": {"info": {"task": "drawer-open", "condition": "dynamo", "seed": 0,
                                                  "status": "restricted"}, "input_spec": {}}}
    with pytest.raises(selection.SelectionError):
        selection.select_policy(offered, condition="dynamo", task="drawer-open", seed=0)
    assert selection.select_policy(offered, condition="dynamo", task="drawer-open", seed=0,
                                   allow_restricted=True) == "dynamo-drawer-open-s0"
