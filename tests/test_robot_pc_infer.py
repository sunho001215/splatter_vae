"""infer.py (directive K): camera names, contract checks, --list, per-run output folders (CPU, fakes; nothing real)."""

from __future__ import annotations

import json
from pathlib import Path

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
        r = SessionRecorder(tmp_path, meta={"x": 1}, cameras=cams, run_prefix="drawer-open_e2e_cnn_seed0_right",
                            camera_names={"135122073201": "right"}, files={"config.json": {"policy": "p"}})
        r.frame("135122073201", 1, 10, np.zeros((4, 6, 3), np.uint8), None, None, {})
        summary = r.close()
        dirs.append(Path(summary["session_dir"]).name)
        d = Path(summary["session_dir"])
        assert json.loads((d / "config.json").read_text()) == {"policy": "p"}
        assert (d / "inference.h5").is_file() and (d / "summary.json").is_file()
        video = summary["videos"]["right"]
        assert video["status"] != "available" or Path(video["file"]).name == "right.mp4"
    assert dirs == [f"drawer-open_e2e_cnn_seed0_right_run{i:03d}" for i in (1, 2, 3)]
    (tmp_path / "inference_outputs/drawer-open_e2e_cnn_seed0_right_run007").mkdir()  # a gap: next is 1 + max
    r = SessionRecorder(tmp_path, meta={}, cameras=cams, run_prefix="drawer-open_e2e_cnn_seed0_right")
    assert r.session_dir.name.endswith("_run008")
    r.close()
    with pytest.raises(ValueError):
        SessionRecorder(tmp_path, meta={}, cameras=cams, run_prefix="../escape")


def test_infer_cli_refuses_conflicting_camera_flags(capsys):
    with pytest.raises(SystemExit) as exc:
        _infer().main(["--cameras", "left", "--camera", "135122070988", "--dry-run"])
    assert exc.value.code == 2 and "not both" in capsys.readouterr().err


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
