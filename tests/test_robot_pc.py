"""Robot-PC package (ops/robot_pc, scripts/robot_pc, s4d/rwlearn/deploy/m5_live) on CPU: labelled FIXTURE policies
over loopback, a synthetic ``panda_teleop_episode`` v2 recording, a synthetic URDF for the M5 helper. No skips, no
robot, no camera.

Safety is tested, not assumed: the manual execution mode needs three explicit conditions and is still refused
(EXECUTION_PERMITTED False); LiveSink / LiveStateReader are never built by setup, replay, dry run or bench (runtime spies
+ a source scan); the only sockets opened during replay / dry run go to the policy server. The switch_controller
adapter (frames, RobotAction format) is checked against the robot PC's code conventions with a fake robot_msg.
"""

from __future__ import annotations

import ast
import collections
import contextlib
import dataclasses
import io
import json
import os
import socket
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import h5py
import numpy as np
import pytest
import torch
import yaml
from PIL import Image

from ops.policy_serving.client import PolicyClient, StepResult
from ops.policy_serving.server import PolicyServer
from ops.robot_pc import cli, devices, dryrun, executor, loop, poses, replay, robot_state, safety, selection, sinks, syncsets
from ops.robot_pc import config as C
from ops.robot_pc import switch_controller as sc
from ops.robot_pc.cameras import LiveCameras, RecordedCameras
from ops.robot_pc.validation import ActionLimits, ActionRejected, BoundRequest, validate_action
from s4d.rwlearn.deploy import m5_live
from s4d.rwlearn.deploy.fixtures import make_fixture_policy
from s4d.rwlearn.deploy.realsense import native_correction
from s4d.rwlearn.deploy.replay import RecordedEpisode

REPO = Path(__file__).resolve().parents[1]
CAMS = ("cam_a", "cam_b", "cam_c")
PERIOD_NS = int(round(1e9 / 15))
POSE = (0.45, 0.0, 0.55, 1.0, 0.0, 0.0, 0.0)


# ------------------------------------------------------------------------------------------------ fixtures
@pytest.fixture(autouse=True)
def stop_fake_background_resources(monkeypatch):
    """No worker, acquisition or shutdown thread may escape a test, even when it asserts early."""
    from ops.robot_pc import gripper, live

    resources = {"workers": [], "readers": [], "cameras": [], "sessions": []}
    for cls, kind in ((gripper.GripperWorker, "workers"), (sc.LiveStateReader, "readers"),
                      (LiveCameras, "cameras"), (live.LiveSession, "sessions")):
        original = cls.__init__
        def tracked(self, *args, _original=original, _kind=kind, **kwargs):
            _original(self, *args, **kwargs)
            resources[_kind].append(self)
        monkeypatch.setattr(cls, "__init__", tracked)
    yield
    for session in resources["sessions"]:
        if session._exec_thread is not None or session.shutdown._thread is not None:
            session.stop_event.set()
            session.shutdown.finish()
            live.LiveSession.join_executor(session, 5)
            assert session._exec_thread is None or not session._exec_thread.is_alive()
            assert not session.shutdown._thread.is_alive()
    for worker in resources["workers"]:
        worker.stop(join_timeout=5)
        assert not worker.is_alive()
    for camera in resources["cameras"]:
        camera.close()
        assert all(not t.is_alive() for t in camera._threads)
    for reader in resources["readers"]:
        reader.close()
        reader._thread.join(5)
        assert not reader._thread.is_alive()


def _wait_until(predicate, timeout=3):
    """Wait for a completed worker state transition, rather than guessing its scheduler delay."""
    import threading
    import time

    deadline = time.monotonic() + timeout
    while not predicate():
        assert time.monotonic() < deadline, "worker did not complete the expected transition"
        threading.Event().wait(0.002)


def _png(a) -> np.ndarray:
    buf = io.BytesIO()
    Image.fromarray(a).save(buf, format="PNG")
    return np.frombuffer(buf.getvalue(), np.uint8)


def _jpeg(a) -> np.ndarray:
    buf = io.BytesIO()
    Image.fromarray(a).save(buf, format="JPEG", quality=90)
    return np.frombuffer(buf.getvalue(), np.uint8)


def write_recording(path: Path, *, frames: int = 14, phases_ms=(0.0, 9.0, -11.0), hw=(48, 64), seed: int = 0):
    """A small recorder v2 file: 3 cameras at 15 Hz with constant phase offsets, robot rows at 100 Hz (slow joint
    motion), gripper register at 60 Hz, sent/engaged commands near the measured pose."""
    rng = np.random.default_rng(seed)
    t0 = 1000.0
    robot_t = t0 - 0.2 + np.arange(int(1.6 * 100)) / 100
    n = len(robot_t)
    q = 0.02 * np.sin(robot_t[:, None] - t0 + np.arange(7)[None]) + np.array([0, -0.4, 0, -2.2, 0, 1.8, 0.1])
    pose = np.tile(POSE, (n, 1))
    pose[:, 0] += 0.002 * np.sin(robot_t - t0)
    cmd = pose.copy()
    cmd[:, 0] += 0.01
    grip_t = t0 - 0.2 + np.arange(int(1.6 * 60)) / 60
    teleop = {"max_position_lead": 0.04, "max_rotation_lead_deg": 20, "max_linear_speed": 0.3,
              "max_angular_speed": 1.5, "rate_hz": 500}
    with h5py.File(path, "w") as f:
        f.attrs.update(format="panda_teleop_episode", format_version=2, episode_index=0, task_name="drawer-open",
                       conventions_json=json.dumps({"time": "host CLOCK_MONOTONIC"}), T_flange_controller_ee=np.eye(4),
                       T_flange_tcp=np.eye(4), camera_calibration="", hostname="synthetic", software_version="test",
                       status="complete", config_yaml=yaml.safe_dump({"teleop": teleop}))
        f["segments/t_start"], f["segments/t_end"] = [t0 - 0.2], [t0 + 1.4]
        for k, a in {"t": robot_t, "q": q, "dq": np.zeros((n, 7)), "tau_ext": np.zeros((n, 7)),
                     "joint_frames": np.tile([0, 0, 0, 0, 0, 0, 1.0], (n, 7, 1)), "flange_pose": pose}.items():
            f[f"robot/{k}"] = a
        for k, a in {"flange_pose": cmd, "flange_target": cmd, "gripper": np.zeros(n), "engaged": np.ones(n, np.uint8),
                     "sent": np.ones(n, np.uint8), "segment": np.zeros(n, np.int32),
                     "lead_limited": np.zeros(n, np.uint8)}.items():
            f[f"action/{k}"] = a
        f["gripper/t"], f["gripper/position"] = grip_t, np.full(len(grip_t), 3.0)
        f["gripper/segment"] = np.zeros(len(grip_t), np.int32)
        h, w = hw
        for v, serial in enumerate(CAMS):
            g = f.create_group(f"cameras/cam_{serial}")
            ct = t0 + np.arange(frames) / 15 + phases_ms[v] / 1e3
            dt = ct + 0.00004
            cal = {"streams": {}, "depth_scale": 0.001, "calibration_id": f"cal-{serial}",
                   "device": {"serial_number": serial, "name": "Synthetic D435I", "sdk_version": "2.59.1",
                              "firmware_version": "test"},
                   "emitter": {"emitter_enabled": True, "laser_power": 150},
                   "extrinsics_convention": "p_target = R_target_from_source @ p_source + t_target_from_source_m",
                   "extrinsics": {"color_from_depth": {"source": "depth", "target": "color",
                                                       "R_target_from_source": np.eye(3).tolist(),
                                                       "t_target_from_source_m": [0.015, 0.0, 0.0]}}}
            for kind in ("color", "depth"):
                cal["streams"][kind] = {"width": w, "height": h, "fps": 15, "format": "bgr8" if kind == "color" else "z16",
                                        "intrinsics": {"width": w, "height": h, "fx": 60.0 + v, "fy": 60.0,
                                                       "ppx": w / 2 - 0.5, "ppy": h / 2 - 0.5,
                                                       "distortion_model": "none", "distortion_coeffs": [0] * 5}}
            g.attrs["calibration_json"] = json.dumps(cal)
            g["t"], g["stereo/t"] = ct, dt
            for k in ("t_valid", "stereo/t_valid", "stereo/depth_ok"):
                g[k] = np.ones(frames, np.uint8)
            g["segment"] = np.zeros(frames, np.int32)
            g["stereo/color_index"] = np.arange(frames)
            g["stereo/color_offset_ms"] = (ct - dt) * 1e3
            g["frame_number"] = 100 + np.arange(frames)
            g["hw_timestamp_ms"], g["t_recv"] = ct * 1e3, ct + 0.065
            g["stereo/hw_timestamp_ms"], g["stereo/t_recv"] = dt * 1e3, dt + 0.065
            for name in ("color", "stereo/depth"):
                g.create_dataset(name, (frames,), dtype=h5py.vlen_dtype(np.dtype("uint8")))
            for k in range(frames):
                g["color"][k] = _jpeg(rng.integers(0, 256, (h, w, 3), dtype=np.uint8))
                z = rng.integers(400, 1500, (h, w)).astype(np.uint16)
                z[::7, ::5] = 0
                g["stereo/depth"][k] = _png(z)
    return path


URDF = """<robot name="toy">
  <link name="panda_link0"><collision><geometry><box size="0.2 0.2 0.1"/></geometry></collision></link>
  {links}
  <link name="finger"><collision><origin xyz="0 0.02 0.03"/>
    <geometry><box size="0.02 0.01 0.05"/></geometry></collision></link>
  <link name="finger2"><collision><origin xyz="0 -0.02 0.03"/>
    <geometry><box size="0.02 0.01 0.05"/></geometry></collision></link>
  <joint name="finger_joint" type="revolute"><parent link="panda_link7"/><child link="finger"/>
    <origin xyz="0 0 0.1"/><axis xyz="1 0 0"/><limit lower="0" upper="0.8"/></joint>
  <joint name="finger2_joint" type="revolute"><parent link="panda_link7"/><child link="finger2"/>
    <origin xyz="0 0 0.1"/><axis xyz="1 0 0"/><mimic joint="finger_joint" multiplier="-1"/></joint>
</robot>"""


@pytest.fixture(scope="module")
def toy_robot(tmp_path_factory):
    from s4d.realworld.contract import RobotDescription

    root = tmp_path_factory.mktemp("toy-robot")
    links = "\n".join(
        f'<link name="panda_link{i}"><collision><origin xyz="0 0 0.05"/><geometry><cylinder radius="0.04" '
        f'length="0.1"/></geometry></collision></link><joint name="panda_joint{i}" type="revolute"><parent '
        f'link="panda_link{i - 1}"/><child link="panda_link{i}"/><origin xyz="0 0 0.12" rpy="0.1 0 0.3"/>'
        f'<axis xyz="0 {1 if i % 2 else 0} {0 if i % 2 else 1}"/><limit lower="-3" upper="3"/></joint>'
        for i in range(1, 8))
    urdf = root / "toy.urdf"
    urdf.write_text(URDF.format(links=links))
    desc = RobotDescription(arm_model="franka_panda", gripper_model="robotiq_2f_85", urdf_path=str(urdf),
                            mesh_root=None, joint_names=tuple(f"panda_joint{i}" for i in range(1, 8)),
                            gripper_state_semantics="robotiq_position_register", gripper_joint="finger_joint")
    asset = root / "toy.m5.npz"
    header = m5_live.export_surface(desc, asset)
    return SimpleNamespace(urdf=urdf, desc=desc, asset=asset, header=header,
                           motion=m5_live.SurfaceMotion(asset, urdf))


@pytest.fixture(scope="module")
def policies(tmp_path_factory):
    old = torch.get_num_threads()
    torch.set_num_threads(4)
    root = tmp_path_factory.mktemp("rob-fixtures")
    out = {c: make_fixture_policy(root / c, c) for c in ("reviwo", "mvp3d_single", "mvp3d_multiview")}
    yield out
    torch.set_num_threads(old)


@contextlib.contextmanager
def serve(pols: dict):
    srv = PolicyServer(pols, warmup_iterations=0)
    srv.start()
    try:
        yield srv
    finally:
        srv.shutdown()


@pytest.fixture
def recording(tmp_path):
    (tmp_path / "drawer-open").mkdir()
    return write_recording(tmp_path / "drawer-open" / "episode_000000.h5")


@pytest.fixture
def spies(monkeypatch):
    """Records every LiveSink / LiveStateReader construction and every outgoing TCP connection."""
    seen = {"live_sink": 0, "live_state": 0, "connect": []}
    real_sink, real_state = sinks.LiveSink.__init__, sc.LiveStateReader.__init__
    real_create = socket.create_connection

    def sink_init(self, *a, **k):
        seen["live_sink"] += 1
        real_sink(self, *a, **k)

    def state_init(self, *a, **k):
        seen["live_state"] += 1
        real_state(self, *a, **k)

    def create(addr, *a, **k):
        seen["connect"].append(tuple(addr))
        return real_create(addr, *a, **k)

    monkeypatch.setattr(sinks.LiveSink, "__init__", sink_init)
    monkeypatch.setattr(sc.LiveStateReader, "__init__", state_init)
    monkeypatch.setattr(socket, "create_connection", create)
    return seen


def multiview_choices(rec, cams):
    return {"native_hw": {c: list(rec.native_hw(c)) for c in cams},
            "calibration": {c: {"K_native": rec.K_native(c).tolist(), "c2w": np.eye(4).tolist()} for c in cams},
            "native_depth_correction": {c: native_correction() for c in cams}}


def test_config_file_and_limits(tmp_path):
    cfg = C.load(root=tmp_path)
    assert "execution" not in cfg
    lim = C.limits(cfg, "mug-hanging")
    assert lim.max_action_age_ms == 200.0 and not hasattr(lim, "workspace_max_m")  # directive K: no box / step cap
    assert cfg["gripper"] == {"speed": 146, "force": 255} and "lead_m" not in cfg["executor"]
    assert C.path(cfg, "assets/x") == tmp_path / "assets/x"
    bad = tmp_path / "bad.yaml"
    bad.write_text(yaml.safe_dump({**yaml.safe_load(C.DEFAULT_PATH.read_text()), "extra": 1}))
    with pytest.raises(ValueError, match="unknown robot-PC config sections"):
        C.load(bad)
    old = {**yaml.safe_load(C.DEFAULT_PATH.read_text()), "execution": {"armed": True}}  # the retired gate is refused
    bad.write_text(yaml.safe_dump(old))
    with pytest.raises(ValueError, match="unknown robot-PC config sections"):
        C.load(bad)
    for sec, key in (("limits", "max_translation_m"), ("executor", "lead_m")):  # removed caps are refused loudly
        old = yaml.safe_load(C.DEFAULT_PATH.read_text())
        old[sec][key] = 0.04
        bad.write_text(yaml.safe_dump(old))
        with pytest.raises(ValueError, match="were removed"):
            C.load(bad)


# ------------------------------------------------------------------------------------------------ imports
def test_robot_pc_stack_imports_with_torch_blocked():
    code = ("import sys; sys.modules['torch'] = None\n"
            "import ops.policy_serving.client, ops.policy_serving.protocol, ops.policy_serving.bench\n"
            "import ops.robot_pc, ops.robot_pc.safety, ops.robot_pc.sinks, ops.robot_pc.validation, ops.robot_pc.poses\n"
            "import ops.robot_pc.executor, ops.robot_pc.robot_state, ops.robot_pc.cameras, ops.robot_pc.devices\n"
            "import ops.robot_pc.selection, ops.robot_pc.loop, ops.robot_pc.replay, ops.robot_pc.dryrun\n"
            "import ops.robot_pc.cli, ops.robot_pc.config, ops.robot_pc.live, ops.robot_pc.gripper\n"
            "import ops.robot_pc.controller, ops.robot_pc.recorder\n"
            "import s4d.rwlearn.deploy.m5_live, s4d.rwlearn.deploy.replay, s4d.rwlearn.deploy.realsense\n"
            "from s4d.realworld.train_store import mutual_sync, set_surface_motion\n"
            "from s4d.realworld.robotics.model import URDFModel, RobotSurface\n"
            "from s4d.rwlearn.deploy import preprocess_frame, InputSpec, HistoryWindow\n"
            "assert sys.modules['torch'] is None\n")
    subprocess.run([sys.executable, "-c", code], check=True, cwd=REPO,
                   env={**os.environ, "CUDA_VISIBLE_DEVICES": "", "PYTHONPATH": str(REPO)})


# ------------------------------------------------------------------------------------------------ safety
def test_execution_mode_is_one_start_action_for_an_eligible_policy():
    full = {"cli_execute": True, "operator_start": True, "policy_info": {"execution_eligible": True}}
    for missing in full:
        with pytest.raises(safety.ExecutionRefused, match="NOT selected"):
            safety.authorize_execution(**{**full, missing: {"cli_execute": False, "operator_start": False,
                                                            "policy_info": None}[missing]})
    for info in ({}, {"execution_eligible": False}, {"execution_eligible": "true"}, {"execution_eligible": 1}):
        with pytest.raises(safety.ExecutionRefused, match="execution_eligible"):  # DynaMo / missing: fail closed
            safety.authorize_execution(**{**full, "policy_info": info})
    with pytest.raises(safety.ExecutionRefused, match="only issued by"):
        safety.ExecutionAuthorization(safety.Mode.EXECUTE, 0)
    assert not hasattr(safety, "EXECUTION_PERMITTED") and not hasattr(safety, "CONFIRMATION")
    with pytest.raises(safety.ExecutionRefused):
        sinks.LiveSink(None, object())
    with pytest.raises(safety.Unbound, match="not approved"):
        sc.LiveStateReader()
    assert isinstance(sinks.make_sink(safety.Mode.REPLAY), sinks.MockSink)
    assert isinstance(sinks.make_sink(safety.Mode.DRY_RUN), sinks.DryRunSink)
    with pytest.raises(safety.ExecutionRefused):
        sinks.make_sink(safety.Mode.EXECUTE)


def test_no_robot_io_in_the_robot_pc_sources():
    """Static guard: no robot/controller transport is imported anywhere in the package or its scripts except the
    read-only RobotState SUB of switch_controller.py (which never binds, never creates a PUB, never sends), and only
    execute.py (refusing) and the sink factory refer to the live sink."""
    banned = {"zmq", "serial", "rclpy", "rospy", "franky", "panda_py", "frankx", "pymodbus", "minimalmodbus"}
    allowed = {"switch_controller.py": {"zmq"}, "controller.py": {"zmq"}, "gripper.py": {"serial"}}
    files = sorted((REPO / "ops/robot_pc").glob("*.py")) + sorted((REPO / "scripts/robot_pc").glob("*.py"))
    for path in files:
        tree = ast.parse(path.read_text())
        for node in ast.walk(tree):
            names = [a.name for a in node.names] if isinstance(node, ast.Import) else \
                [node.module or ""] if isinstance(node, ast.ImportFrom) else []
            found = {n.split(".")[0] for n in names} & banned
            assert found <= allowed.get(path.name, set()), (path, names)
        text = path.read_text()
        if path.name not in ("sinks.py", "live.py", "safety.py", "infer.py"):
            assert "LiveSink(" not in text and "ControllerLink(" not in text.replace("class ControllerLink(", ""), path
    text = (REPO / "ops/robot_pc/switch_controller.py").read_text()  # the state reader never publishes
    assert ".bind(" not in text and ".PUB" not in text and ".send(" not in text and "request_mode" not in text
    for name in ("replay.py", "dry_run.py", "bench.py", "policies.py", "selfcheck.py", "setup.sh"):
        text = (REPO / "scripts/robot_pc" / name).read_text()
        assert "LiveSink" not in text and "ControllerLink" not in text and "--execute" not in text, name


# ------------------------------------------------------------------------------------------------ validation
def _result(outputs: dict, *, ref="r1", session="s1", ok=True):
    return StepResult("ok" if ok else "error", 1, session, None if ok else "timeout", None, ref, outputs)


def _request(t_obs=10_000_000_000, ref="r1"):
    return BoundRequest(ref, {"cam_a": 7}, t_obs, POSE, t_obs - 3_000_000, 3.0, t_obs - 20_000_000)


def _outputs(req, goal=None, c=0.25, **kw):
    goal = list(POSE) if goal is None else goal
    out = {"obs_ids": {"reference_id": req.reference_id, "session_id": "s1", "frame_ids": {"cam_a": [5, 6, 7]},
                       "observation_time_ns": req.observation_time_ns, "pose_time_ns": req.pose_time_ns,
                       "gripper_time_ns": req.gripper_time_ns},
           "measured_pose7": np.asarray(req.pose7), "measured_gripper_register": 3.0,
           "action_norm": np.full(7, 0.5, np.float32), "flange_pose": np.asarray(goal),
           "execution_flange_pose": np.asarray(goal), "gripper_closure": c,
           "gripper_register": np.int64(round(3 + 227 * c))}
    out.update(kw)
    return out


def test_action_validation_accepts_bound_actions_and_names_every_refusal():
    lim, req = ActionLimits(), _request()
    now = req.observation_time_ns + 80_000_000
    goal = [POSE[0] + 0.03, POSE[1], POSE[2], *POSE[3:]]
    cmd = validate_action(_result(_outputs(req, goal)), req, lim, session_id="s1", now_ns=now)
    assert cmd.goal_pose7 == tuple(goal) and cmd.expires_ns == req.observation_time_ns + 200_000_000
    assert cmd.measured_pose7 == POSE and abs(cmd.translation_m - 0.03) < 1e-12 and cmd.gripper_register == 60
    far = [POSE[0] + 0.05, *POSE[1:]]
    rot = [*POSE[:3], *Rquat(np.deg2rad(25))]
    cases = {
        "no_action": (_result({}, ok=False), now),
        "binding": (_result(_outputs(req), ref="other"), now),
        "non_finite": (_result(_outputs(req, gripper_closure=float("nan"))), now),
        "range": (_result(_outputs(req, action_norm=np.full(7, 1.5, np.float32))), now),
        "gripper": (_result(_outputs(req, gripper_register=np.int64(99))), now),
        "expired": (_result(_outputs(req)), req.observation_time_ns + 250_000_000),
        "future": (_result(_outputs(req)), req.observation_time_ns - 1),
    }
    for code, (res, t) in cases.items():
        with pytest.raises(ActionRejected) as e:
            validate_action(res, req, lim, session_id="s1", now_ns=t)
        assert e.value.code == code, (code, e.value)
    moved = _outputs(req, measured_pose7=np.asarray([POSE[0] + 1e-6, *POSE[1:]]))
    with pytest.raises(ActionRejected, match="binding"):
        validate_action(_result(moved), req, lim, session_id="s1", now_ns=now)
    with pytest.raises(ActionRejected, match="binding"):
        validate_action(_result(_outputs(req)), req, lim, session_id="other", now_ns=now)
    # directive K: no per-step distance / rotation cap and no workspace box - far goals are the policy's to make
    for g in (far, rot, [POSE[0] + 0.5, *POSE[1:]]):
        validate_action(_result(_outputs(req, g)), req, lim, session_id="s1", now_ns=now)
    with pytest.raises(ValueError, match="unknown action limit keys"):
        ActionLimits.from_config({"workspace_min_m": [1, 0, 0]})


def Rquat(angle: float):
    """POSE's orientation (x 180 deg) rotated by ``angle`` about the base z axis, xyzw."""
    qz = np.array([0, 0, np.sin(angle / 2), np.cos(angle / 2)])
    x, y, z, w = qz
    a = np.array(POSE[3:])
    return [w * a[0] + x * a[3] + y * a[2] - z * a[1], w * a[1] - x * a[2] + y * a[3] + z * a[0],
            w * a[2] + x * a[1] - y * a[0] + z * a[3], w * a[3] - x * a[0] - y * a[1] - z * a[2]]


def test_pose_limits_and_quaternion_helpers():
    start = np.array(POSE)
    goal = np.array([POSE[0] + 0.1, *POSE[1:3], *Rquat(np.deg2rad(40))])
    out = poses.toward(start, goal, 0.01, np.deg2rad(5))
    assert abs(np.linalg.norm(out[:3] - start[:3]) - 0.01) < 1e-12
    assert abs(poses.quat_angle(out[3:], start[3:]) - np.deg2rad(5)) < 1e-9
    np.testing.assert_allclose(poses.toward(start, start, 0.01, 0.1), start)
    r = poses.rotvec_between(start[3:], goal[3:])
    assert abs(np.linalg.norm(r) - np.deg2rad(40)) < 1e-9
    assert poses.quat_angle(poses.rotate_left(start[3:], r), goal[3:]) < 1e-9
    with pytest.raises(ValueError, match="degenerate"):
        poses.quat_angle([0, 0, 0, 0], [0, 0, 0, 1])


def _cmd(goal, t_obs=0, age_ms=200.0, closure=0.5):
    from ops.robot_pc.validation import Command

    return Command("r1", "s1", int(t_obs), int(t_obs + age_ms * 1e6), POSE, tuple(np.asarray(goal, float).tolist()),
                   closure, int(round(3 + 227 * closure)), 0.0, 0.0)


def test_executor_interpolates_without_caps_and_never_sends_what_the_controller_rejects():
    """Directive K: previous setpoint -> newest goal over one policy period (no speed / acceleration / lead caps); a
    setpoint beyond switch_controller's log6 guard (0.2 m / 0.5 rad from the measured pose) is never sent."""
    sink = sinks.MockSink()
    ex = executor.HighRateExecutor(sink)
    goal = [POSE[0] + 0.15, POSE[1] - 0.01, POSE[2], *Rquat(np.deg2rad(15))]  # far + fast: no cap applies
    assert ex.tick(0, POSE, state_t_ns=0, signature=b"0") is None and not sink.items  # nothing before a command
    assert ex.accept(_cmd(goal, age_ms=1e6))
    dt, t, out = 2_000_000, 0, []
    for k in range(60):
        t += dt
        out.append(ex.tick(t, POSE, state_t_ns=t, signature=bytes([k])))
    p = np.array([o.pose7[:3] for o in out])
    np.testing.assert_allclose(p[0], POSE[:3], atol=1e-12)  # starts at the measured pose
    a = np.clip((np.arange(60) * dt) / ex.interp_ns, 0, 1)  # linear in time over one policy period
    np.testing.assert_allclose(p, np.array(POSE[:3]) + a[:, None] * (np.array(goal[:3]) - POSE[:3]), atol=1e-12)
    assert poses.quat_angle(out[-1].pose7[3:], goal[3:]) < 1e-9 and ex.counts["controller_guard"] == 0
    peak = np.max(np.linalg.norm(np.diff(p, axis=0), axis=1)) / (dt / 1e9)
    assert peak > 2.0  # 0.15 m in 66.7 ms: ~2.3 m/s - the old 0.3 m/s cap is gone
    nxt = [goal[0] + 0.02, *goal[1:]]  # a newer goal continues from the last SENT setpoint (no jump back)
    assert ex.accept(_cmd(nxt, t_obs=1, age_ms=1e6))
    t += dt
    np.testing.assert_allclose(ex.tick(t, POSE, state_t_ns=t, signature=b"n").pose7[:3], goal[:3], atol=1e-12)
    far = executor.HighRateExecutor(sinks.MockSink())
    far.accept(_cmd([POSE[0] + 0.25, *POSE[1:]], age_ms=1e6))  # beyond the controller's 0.2 m guard
    t, sent = 0, []
    for k in range(60):
        t += dt
        o = far.tick(t, POSE, state_t_ns=t, signature=bytes([k]))
        if o is not None:
            sent.append(o.pose7[0] - POSE[0])
    assert sent and max(sent) <= 0.95 * 0.2 + 1e-12 and far.counts["controller_guard"] > 0 and far.stopped is None
    m, r = executor.log6_norms([*POSE[:3], *Rquat(0.3)], POSE)
    assert abs(m) < 1e-12 and abs(r - 0.3) < 1e-9


def test_executor_holds_on_expiry_and_latches_a_stop_on_stale_or_frozen_state():
    sink = sinks.MockSink()
    ex = executor.HighRateExecutor(sink)
    ex.accept(_cmd([POSE[0] + 0.02, *POSE[1:]], t_obs=0, age_ms=100, closure=0.3))
    a = ex.tick(10_000_000, POSE, state_t_ns=10_000_000, signature=b"a")
    assert a.kind == "track" and a.gripper_closure == 0.3
    moved = [POSE[0] + 0.001, *POSE[1:]]
    n_sent = len(sink.items)
    assert ex.tick(150_000_000, moved, state_t_ns=150_000_000, signature=b"b") is None  # expired: nothing sent
    assert ex.tick(150_000_000, moved, state_t_ns=150_000_000, signature=b"b") is None  # not even a re-send
    assert len(sink.items) == n_sent and ex.counts["expired"] == 2  # every expired tick sends nothing
    assert not ex.accept(_cmd([POSE[0] + 0.01, *POSE[1:]], t_obs=-1, closure=1.0))  # older observation refused
    assert ex.accept(_cmd([POSE[0] + 0.01, *POSE[1:]], t_obs=140_000_000, age_ms=100))  # re-engage from measured
    sp = ex.tick(152_000_000, moved, state_t_ns=152_000_000, signature=b"c")
    assert sp.kind == "track" and sp.pose7[0] == moved[0]  # a new engagement starts at the measured pose
    late = executor.HighRateExecutor(sinks.MockSink())
    late.accept(_cmd(POSE, t_obs=0, age_ms=10, closure=1.0))  # first tick after expiry: nothing of it applied
    assert late.tick(20_000_000, POSE, state_t_ns=20_000_000, signature=b"x") is None
    stale = executor.HighRateExecutor(sinks.MockSink())
    stale.accept(_cmd(POSE, age_ms=1e6))
    assert stale.tick(100_000_000, POSE, state_t_ns=40_000_000, signature=b"s") is None
    assert stale.stopped == "stale robot state"
    assert stale.tick(101_000_000, POSE, state_t_ns=101_000_000, signature=b"z") is None  # latched
    stale.rearm()
    assert stale.tick(102_000_000, POSE, state_t_ns=102_000_000) is None and stale.command is None
    frozen = executor.HighRateExecutor(sinks.MockSink())
    frozen.accept(_cmd(POSE, age_ms=1e6))
    t = 0
    for _ in range(140):  # the publisher keeps sending (fresh times) a bit-identical state
        t += 2_000_000
        out = frozen.tick(t, POSE, state_t_ns=t, signature=b"same")
    assert out is None and frozen.stopped.startswith("robot state frozen")


# ------------------------------------------------------------------------------------------------ devices
def test_device_busy_guard(tmp_path):
    def process(root, pid, cmd, fds=()):
        d = root / str(pid)
        (d / "fd").mkdir(parents=True)
        (d / "cmdline").write_bytes(cmd.encode().replace(b" ", b"\0"))
        for i, target in enumerate(fds):
            os.symlink(target, d / "fd" / str(i))
        return root

    clean = process(tmp_path / "p1", 10, "/usr/bin/bash", ["/dev/null"])
    assert devices.require_free(proc_root=str(clean), ipc_paths=())["holders"] == []
    held = process(process(tmp_path / "p2", 10, "/usr/bin/bash"), 11, "python3 viewer.py", ["/dev/video4"])
    with pytest.raises(devices.DevicesBusy, match="holds /dev/video4"):
        devices.require_free(proc_root=str(held), ipc_paths=())
    rec = process(tmp_path / "p3", 12, "python3 -m panda_teleop.collector --config x")
    with pytest.raises(devices.DevicesBusy, match="recorder/teleop"):
        devices.require_free(proc_root=str(rec), ipc_paths=())
    ctl = process(tmp_path / "p4", 13, "docker run --name switch-controller-panda -v /home/irsl/workspace/panda-teleop/"
                                        "switch_controller/config/panda_real.yaml:/x switch-controller:panda --backend real")
    assert devices.require_free(proc_root=str(ctl), ipc_paths=())["controllers"]  # the controller is not a recorder
    with pytest.raises(devices.DevicesBusy, match="is switch_controller"):
        devices.require_free(proc_root=str(ctl), ipc_paths=(), allow_controller=False)
    ipc = tmp_path / "teleop_collector_requests.ipc"
    ipc.write_text("")
    with pytest.raises(devices.DevicesBusy, match="collector IPC"):
        devices.require_free(proc_root=str(clean), ipc_paths=(str(ipc),))


# ------------------------------------------------------------------------------------------------ M5 helper
def test_surface_motion_equals_the_training_geometry(toy_robot):
    from s4d.realworld.pipeline import RobotGeometry

    geom = RobotGeometry(SimpleNamespace(robot=toy_robot.desc))
    rng = np.random.default_rng(1)
    for _ in range(5):
        q, g = rng.uniform(-1, 1, 7), float(rng.uniform(0, 255))
        np.testing.assert_array_equal(toy_robot.motion.points((q, g)), geom.surface.points(geom.q(q, g))[0])
    assert toy_robot.header["points"] == sum(len(v) for v in geom.surface.link_points.values())
    bad = toy_robot.urdf.parent / "changed.urdf"
    bad.write_text(toy_robot.urdf.read_text().replace("0.12", "0.13"))
    with pytest.raises(ValueError, match="sha256"):
        m5_live.SurfaceMotion(toy_robot.asset, bad)
    m5_live.export_surface(toy_robot.desc, toy_robot.urdf.parent / "again.npz")
    assert (toy_robot.urdf.parent / "again.npz").read_bytes() == toy_robot.asset.read_bytes()  # reproducible


def _timelines(rng, phases_ms, frames=40, drops=()):
    out = []
    for v, ph in enumerate(phases_ms):
        t = (5_000_000_000 + np.arange(frames) * PERIOD_NS + int(ph * 1e6)
             + rng.integers(-50_000, 50_000, frames)).astype(np.int64)
        keep = np.ones(frames, bool)
        for cam, k in drops:
            if cam == v:
                keep[k] = False
        out.append(t[keep])
    return out


@pytest.mark.parametrize("phases, drops", [((0, 9, -11), ()), ((0, 32.5, -32.5), ()), ((0, -30, 31), ((1, 7),)),
                                           ((0, 20, 25), ((0, 12), (2, 3)))])
def test_live_sets_reproduce_mutual_sync_in_arrival_order(phases, drops):
    from s4d.realworld.train_store import mutual_sync

    stamps = _timelines(np.random.default_rng(len(drops)), phases, drops=drops)
    ref, _ = mutual_sync(stamps)
    sets = m5_live.LiveSets(CAMS, 1 / 15)
    events = sorted((int(t) + 65_000_000, v, k, int(t)) for v, s in enumerate(stamps) for k, t in enumerate(s))
    got = {}
    for _, v, k, t in events:
        sets.add(CAMS[v], k, t)
        for s in sets.pop_final():
            got[s["frame_ids"]["cam_a"]] = s
    for s in sets.pop_final(end_of_stream=True):
        got[s["frame_ids"]["cam_a"]] = s
    assert sorted(got) == list(range(len(stamps[0])))
    for k, s in got.items():
        assert [s["frame_ids"].get(c, -1) for c in CAMS] == ref[k].tolist(), k


def test_live_m5_equals_the_training_rule_incl_rejections_and_missing_state(toy_robot):
    from s4d.realworld.pipeline import robot_state_at
    from s4d.realworld.train_store import mutual_sync, set_surface_motion

    rng = np.random.default_rng(3)
    stamps = _timelines(rng, (0, 12, -20), frames=30)
    rt = np.arange(5_000_000_000 + 300_000_000, 5_000_000_000 + 30 * PERIOD_NS, 10_000_000, dtype=np.int64)
    i = np.arange(len(rt))[:, None]
    q = np.where(i < 0.4 * len(rt), 0.0, 0.3 * np.sin(i / 7 + np.arange(7)))  # still, then fast (rejected sets)
    gt = np.arange(rt[0] - 5_000_000, rt[-1] + 20_000_000, 16_000_000, dtype=np.int64)
    gp = np.clip(3 + 4 * np.arange(len(gt)), 3, 230).astype(float)
    traj = SimpleNamespace(timestamps_ns=rt, joint_positions=q, gripper_state=np.interp(rt / 1e9, gt / 1e9, gp))
    states, ok = [], []
    for s in stamps:
        qq, gg, inside, _ = robot_state_at(SimpleNamespace(trajectory=traj), s, 20_000_000)
        states.append(list(zip(qq, gg)))
        ok.append(inside)
    frames, _ = mutual_sync(stamps)
    want = set_surface_motion(toy_robot.motion.points, frames, states, ok)
    assert np.isinf(want).any() and (want[np.isfinite(want)] > 0.005).any() and (want <= 0.005).any()
    live = m5_live.LiveM5(toy_robot.motion, CAMS, 1 / 15, 20_000_000)
    events = [(int(t) + 100_000, 0, ("robot", int(t), qq)) for t, qq in zip(rt, q)]
    events += [(int(t) + 1_000_000, 1, ("gripper", int(t), float(p))) for t, p in zip(gt, gp)]
    events += [(int(t) + 65_000_000, 2, ("frame", CAMS[v], k, int(t))) for v, s in enumerate(stamps)
               for k, t in enumerate(s)]
    got = {}
    for _, _, e in sorted(events, key=lambda x: (x[0], x[1])):
        getattr(live, f"add_{e[0]}")(*e[1:])
        for s in live.poll():
            got[s["frame_ids"]["cam_a"]] = s["surface_disp_m"]
    for s in live.poll(end_of_stream=True):
        got[s["frame_ids"]["cam_a"]] = s["surface_disp_m"]
    complete = (frames >= 0).all(1)
    assert sorted(got) == np.flatnonzero(complete).tolist()
    np.testing.assert_array_equal([got[k] for k in sorted(got)], want[complete])
    assert live.counts["rejected_motion"] == int((want[complete] > 0.005).sum())


class _DequeTimeline:
    """The previous deque-based ``StateTimeline`` (until 2026-11-12): the reference of the array-ring one."""

    def __init__(self, max_sync_ns, capacity):
        self.max_sync_ns = int(max_sync_ns)
        self._rows, self._grip = collections.deque(maxlen=capacity), collections.deque(maxlen=capacity)

    def add_robot(self, t_ns, q7):
        self._rows.append((int(t_ns), np.asarray(q7, np.float64).reshape(7)))

    def add_gripper(self, t_ns, register):
        self._grip.append((int(t_ns), float(register)))

    def ready(self, t_ns):
        if not self._rows or not self._grip or self._rows[-1][0] < t_ns:
            return False
        return self._grip[-1][0] >= next(t for t, _ in self._rows if t >= t_ns)

    def states(self, stamps_ns):
        from s4d.realworld.pipeline import robot_state_at

        stamps = np.asarray(stamps_ns, np.int64)
        if not self._rows or not self._grip:
            return [None] * len(stamps), np.zeros(len(stamps), bool)
        t = np.array([r[0] for r in self._rows], np.int64)
        q = np.stack([r[1] for r in self._rows])
        gt = np.array([g[0] for g in self._grip], np.float64) / 1e9
        gp = np.array([g[1] for g in self._grip], np.float64)
        trajectory = SimpleNamespace(timestamps_ns=t, joint_positions=q, gripper_state=np.interp(t / 1e9, gt, gp))
        qs, gs, inside, _ = robot_state_at(SimpleNamespace(trajectory=trajectory), stamps, self.max_sync_ns)
        return list(zip(qs, gs)), np.asarray(inside, bool)


def test_state_timeline_ring_equals_the_deque_timeline_through_wraparound():
    """The array-ring StateTimeline holds exactly the last ``capacity`` rows / samples: ``ready`` and ``states`` equal
    the deque implementation's bit for bit while both buffers wrap many times (NaN joints and a gap included)."""
    rng = np.random.default_rng(5)
    new, old = m5_live.StateTimeline(20_000_000, capacity=37), _DequeTimeline(20_000_000, 37)
    t, g = 7 * 10**14, 7 * 10**14 + 3_000_000
    with pytest.raises(ValueError, match="increasing"):
        bad = m5_live.StateTimeline(20_000_000, capacity=4)
        bad.add_robot(5, np.zeros(7))
        bad.add_robot(5, np.zeros(7))
    for step in range(400):
        for _ in range(int(rng.integers(1, 6))):
            t += int(rng.integers(500_000, 2_000_000)) + (300_000_000 if step == 200 else 0)
            q = rng.normal(size=7) if step != 120 else np.full(7, np.nan)
            new.add_robot(t, q)
            old.add_robot(t, q)
        if rng.random() < 0.4:
            g += int(rng.integers(5_000_000, 25_000_000))
            register = float(rng.uniform(3, 230))
            new.add_gripper(g, register)
            old.add_gripper(g, register)
        stamps = t - rng.integers(0, 40_000_000, 3)
        assert [new.ready(int(x)) for x in stamps] == [old.ready(int(x)) for x in stamps]
        (sn, okn), (so, oko) = new.states(stamps), old.states(stamps)
        np.testing.assert_array_equal(okn, oko)
        for a, b in zip(sn, so):
            np.testing.assert_array_equal(a[0], b[0])
            np.testing.assert_array_equal(a[1], b[1])
    assert len(new._t) == len(old._rows) == 37 and len(new._gt) == len(old._grip)


# ------------------------------------------------------------------------------------------------ replay / dry run
def _loop(client, state, sink, clock=None, max_age_ms=200.0):
    return loop.PolicyLoop(client, state, ActionLimits(max_action_age_ms=max_age_ms), sink,
                           **({} if clock is None else {"clock": clock}))


def test_single_view_replay_end_to_end(policies, recording, spies, tmp_path):
    rec = RecordedEpisode(recording)
    with serve({"reviwo-x": policies["reviwo"]}) as srv:
        cams = ["cam_b"]
        choices = {"native_hw": {"cam_b": list(rec.native_hw("cam_b"))}, "camera_id": "cam_b"}
        client = PolicyClient("127.0.0.1", srv.port, policy_id="reviwo-x", choices=choices, step_timeout_s=60)
        client.connect()
        cameras = RecordedCameras(rec, cams, depth=False)
        clock = replay.ReplayClock(int(cameras.timestamps_ns("cam_b")[0]), "max")
        state = robot_state.RecordedRobotState(rec, clock.offset_ns)
        sink = sinks.MockSink()
        lp = _loop(client, state, sink, clock.now)
        rows = replay.run(client, lp, cameras, state, clock, cams=cams, noise_seed=5)
        client.close()
    assert len(rows) == 14 and all(r["status"] == "ok" for r in rows), [r.get("code") for r in rows]
    assert len(sink.items) == 14 and all(not r["sent_to_robot"] for r in rows)
    assert [r["padded"] for r in rows[:3]] == [True, True, False]
    for r in rows:  # bound to the measured pose of its own observation (recorded rule, shifted clock)
        st = state.bound_state(r["observation_time_ns"])
        assert r["command"]["measured_pose7"] == list(st.pose7) and r["command"]["reference_id"] == r["reference_id"]
    s = loop.summarize(rows, route="loopback", mode="replay-max")
    assert s["ok"] == 14 and s["client_ms"]["capture_to_command_ms"]["n"] == 14 and s["payload_bytes"]["first_request"]
    cmp = replay.recorded_comparison(rec, cams, rows, clock.offset_ns, client.spec)
    assert cmp["n"] >= 10 and cmp["position_error_mm_median"] >= 0
    assert spies["live_sink"] == spies["live_state"] == 0 and set(spies["connect"]) == {("127.0.0.1", srv.port)}


def test_multiview_rgbd_replay_with_the_live_m5_helper(policies, recording, toy_robot, spies):
    rec = RecordedEpisode(recording)
    with serve({"mvp3d-mv": policies["mvp3d_multiview"]}) as srv:
        cams = list(CAMS)
        client = PolicyClient("127.0.0.1", srv.port, policy_id="mvp3d-mv", choices=multiview_choices(rec, cams),
                              step_timeout_s=60)
        client.connect()
        cameras = RecordedCameras(rec, cams, depth=True)
        clock = replay.ReplayClock(int(cameras.timestamps_ns(CAMS[0])[0]), "max")
        state = robot_state.RecordedRobotState(rec, clock.offset_ns)
        live = m5_live.LiveM5(toy_robot.motion, cams, 1 / 15, 20_000_000)
        sink = sinks.MockSink()
        rows = replay.run(client, _loop(client, state, sink, clock.now, max_age_ms=1e6), cameras, state, clock,
                          cams=cams, live_m5=live, noise_seed=1,  # CPU fixture inference may exceed 200 ms here
                          sync_mode="training-sets")
        client.close()
        ok = [r for r in rows if r["status"] == "ok"]
        assert len(ok) >= 10 and len(sink.items) == len(ok), [r.get("code") for r in rows]
        assert all(r["surface_disp_m"] is not None and r["surface_disp_m"] <= 0.005 for r in rows)
        assert live.counts["sets"] == len(rows)
        training_sets = {json.dumps(r["end_ids"], sort_keys=True) for r in rows}
        # the live rule (default): one step per reference frame on every camera's latest released frame
        client = PolicyClient("127.0.0.1", srv.port, policy_id="mvp3d-mv", choices=multiview_choices(rec, cams),
                              step_timeout_s=60)
        client.connect()
        clock = replay.ReplayClock(int(cameras.timestamps_ns(CAMS[0])[0]), "max")
        state = robot_state.RecordedRobotState(rec, clock.offset_ns)
        live = m5_live.LiveM5(toy_robot.motion, cams, 1 / 15, 20_000_000)
        sink = sinks.MockSink()
        loop = _loop(client, state, sink, clock.now, max_age_ms=1e6)
        rows = replay.run(client, loop, cameras, state, clock, cams=cams, live_m5=live, noise_seed=1)
        client.close()
    ok = [r for r in rows if r["status"] == "ok"]
    assert len(ok) >= 10 and len(sink.items) == len(ok), [r.get("code") for r in rows]
    assert loop.counts["sync_steps"] == len(rows) and loop.counts["sync_mismatch"] == 0
    for r in rows:
        sy = r["sync"]
        assert sy["rule"] == "latest" and sy["frame_ids"] == r["end_ids"] and set(sy["capture_ns"]) == set(cams)
        assert r["observation_time_ns"] == max(sy["capture_ns"].values()) and min(sy["age_rel_ms"].values()) == 0
        assert all(a >= 0 for a in sy["age_ms"].values()) and len(sy["offsets_ms"]) == 3
        assert sy["surface_disp_m"] is not None and sy["surface_disp_m"] <= 0.005 and not sy["motion_over_bound"]
    assert all(r["server_sync"]["mismatch"] is False for r in ok)
    # on the recorded (mutual-nearest) timelines the latest-frame steps are the training sets
    assert {json.dumps(r["end_ids"], sort_keys=True) for r in rows} <= training_sets
    assert live.counts["sets"] == len(rows) and live.counts["within_bound"] == len(rows)
    assert spies["live_sink"] == 0 and set(spies["connect"]) == {("127.0.0.1", srv.port)}


def test_replay_sync_injection_is_served_with_counted_warnings(policies, recording, toy_robot, spies):
    """Synthetic misalignment on the recorded streams (user directive 2026-11-12): a camera's capture timeline shifted
    by 45 ms (its own spacing intact; the latest-frame rule then picks its frame 21.7 ms early), a camera delivered
    50 ms late (stale frames: mismatches), seeded delivery jitter. Every step is SERVED (no unsynchronized /
    incomplete_history refusal), each row carries per-camera capture times, ages and pairwise offsets, every client
    mismatch is also the server's, and the shifted camera's logged offset is exactly the injected one."""
    rec = RecordedEpisode(recording)
    cams = list(CAMS)
    results = {}
    with serve({"mvp3d-mv": policies["mvp3d_multiview"]}) as srv:
        for name, inject in (("offset", {"offset_ms": {"cam_b": 45.0}}),
                             ("lag", {"delay_ms": {"cam_b": 50.0}}),
                             # equal phases (the fixture's +9 / -11 ms removed) + cam_b delivered 50 ms late: the
                             # partner wait (33.3 ms) runs out BETWEEN releases; the step is taken at its deadline
                             ("equal-lag", {"offset_ms": {"cam_b": -9.0, "cam_c": 11.0}, "delay_ms": {"cam_b": 50.0}}),
                             ("jitter", {"arrival_jitter_ms": 30.0, "seed": 7}),
                             ("both", {"offset_ms": {"cam_c": -40.0}, "arrival_jitter_ms": 20.0, "seed": 3})):
            client = PolicyClient("127.0.0.1", srv.port, policy_id="mvp3d-mv", choices=multiview_choices(rec, cams),
                                  step_timeout_s=60)
            client.connect()
            cameras = RecordedCameras(rec, cams, depth=True)
            clock = replay.ReplayClock(int(cameras.timestamps_ns(CAMS[0])[0]), "max")
            state = robot_state.RecordedRobotState(rec, clock.offset_ns)
            lp = _loop(client, state, sinks.MockSink(), clock.now, max_age_ms=1e6)
            rows = replay.run(client, lp, cameras, state, clock, cams=cams, noise_seed=1, inject=inject,
                              live_m5=m5_live.LiveM5(toy_robot.motion, cams, 1 / 15, 20_000_000))
            client.close()
            results[name] = (rows, lp.counts)
        with pytest.raises(ValueError, match="not in the session"):
            replay.injection_shifts(RecordedCameras(rec, cams, depth=True), cams, {"offset_ms": {"cam_x": 1.0}})
        with pytest.raises(ValueError, match=">= 0"):
            replay.injection_shifts(RecordedCameras(rec, cams, depth=True), cams, {"delay_ms": {"cam_a": -1.0}})
    for name, (rows, counts) in results.items():
        ok = [r for r in rows if r["status"] == "ok"]
        assert len(rows) >= 10 and len(ok) == len(rows), (name, [r.get("code") for r in rows])
        assert counts["sync_steps"] == len(rows) and not counts["sync_refused"]
        for r in ok:
            sy = r["sync"]
            assert set(sy["capture_ns"]) == set(cams) and set(sy["age_ms"]) == set(cams) and len(sy["offsets_ms"]) == 3
            # the server measures the whole history (older steps too): never less than the end step's offset
            assert r["server_sync"]["max_offset_ms"] >= sy["max_offset_ms"] and (r["server_sync"]["mismatch"] or
                                                                                  not sy["mismatch"])
            assert r["observation_time_ns"] == max(sy["capture_ns"].values())
    rows, counts = results["offset"]
    assert all(abs(r["sync"]["offsets_ms"]["cam_a|cam_b"]) < 33.4 for r in rows[1:])  # nearest frame picked
    native = {c: np.asarray(rec.timestamps_ns(c), np.int64) for c in cams}
    for r in rows:  # the injected shift is visible exactly in the logged capture times
        sy, ids = r["sync"], r["end_ids"]
        idx = {c: int(np.flatnonzero(np.asarray(rec.frame_numbers(c)) == ids[c])[0]) for c in cams}
        rel = {c: sy["capture_ns"][c] - int(native[c][idx[c]]) for c in cams}
        assert rel["cam_b"] - rel["cam_a"] == 45_000_000 and rel["cam_c"] == rel["cam_a"]
    rows, counts = results["lag"]  # cam_b's partner arrives after the half-period wait: its previous frame is used
    assert counts["sync_mismatch"] >= len(rows) // 2 and counts["sync_partner_wait_expired"] >= len(rows) // 2
    assert all(r["server_sync"]["mismatch"] for r in rows if r["sync"]["mismatch"])
    for r in rows:
        if r["sync"]["mismatch"]:  # only the lagging camera is off; the others stay at the reference instant
            assert r["sync"]["age_ms"]["cam_b"] > r["sync"]["age_ms"]["cam_a"]
            assert abs(r["sync"]["offsets_ms"]["cam_a|cam_c"]) < 33.4
    rows, counts = results["equal-lag"]
    late = [r for r in rows[1:] if r["sync"]["mismatch"]]
    assert len(late) == len(rows) - 1 and counts["sync_partner_wait_expired"] >= len(rows) - 1
    for r in late:  # stepped at the deadline: cam_b's previous frame (one period old), cam_c at the reference instant
        assert r["sync"]["offsets_ms"]["cam_a|cam_b"] == pytest.approx(66.667, abs=0.01)
        assert r["sync"]["offsets_ms"]["cam_a|cam_c"] == pytest.approx(0.0, abs=0.01)
        assert r["sync"]["age_ms"]["cam_a"] == pytest.approx(33.333, abs=2.0)  # frame clock at the deadline
    assert spies["live_sink"] == 0


def test_latest_frames_rule_waits_bounded_and_counts_warnings():
    """syncsets.LatestFrames on synthetic timelines: no step until every camera has a frame; a step on every camera's
    latest frame as soon as its partners are within half a period before the reference; a late camera is waited for
    at most wait_s, then the step goes ahead on its stale frame (mismatch + partner_wait_expired counted); a newer
    reference frame supersedes an unstepped one; M5 (informational) is declared to the server when final."""
    P = PERIOD_NS
    declared, counts = [], collections.Counter()
    client = SimpleNamespace(add_sync_set=lambda ids, d: declared.append((dict(ids), d)))
    cams = ["a", "b", "c"]
    lf = syncsets.LatestFrames(client, None, None, cams, period_s=1 / 15, counts=counts)
    T = 10**12
    lf.add_frame("a", 0, T, now_ns=T + 65_000_000)
    assert lf.newest_step(T + 65_000_000) is None  # b, c without any frame: never a partial set
    lf.add_frame("b", 0, T - 10_000_000, now_ns=T + 66_000_000)
    lf.add_frame("c", 0, T + 9_000_000, now_ns=T + 66_000_000)
    end_ids, t_obs, ref, rec = lf.newest_step(T + 70_000_000)
    assert end_ids == {"a": 0, "b": 0, "c": 0} and t_obs == T + 9_000_000 and ref == "latest:0"
    assert rec["offsets_ms"] == {"a|b": 10.0, "a|c": -9.0, "b|c": -19.0} and rec["max_offset_ms"] == 10.0
    assert rec["age_ms"] == {"a": 70.0, "b": 80.0, "c": 61.0} and rec["age_rel_ms"] == {"a": 9.0, "b": 19.0, "c": 0.0}
    assert not rec["mismatch"] and rec["surface_disp_m"] is None and not declared
    assert lf.newest_step(T + 71_000_000) is None  # no new reference frame
    lf.add_frame("a", 1, T + P, now_ns=T + P + 65_000_000)  # b lags: its newest is 76.7 ms older
    lf.add_frame("c", 1, T + P + 9_000_000, now_ns=T + P + 66_000_000)
    assert lf.newest_step(T + P + 80_000_000) is None  # waiting (< half a period since a's frame arrived)
    end_ids, _, _, rec = lf.newest_step(T + P + 65_000_000 + P // 2 + 1)
    assert end_ids == {"a": 1, "b": 0, "c": 1} and rec["mismatch"] and rec["max_offset_ms"] == pytest.approx(76.667, 1e-3)
    assert counts["sync_partner_wait_expired"] == 1 and counts["sync_mismatch"] == 1 and counts["sync_steps"] == 2
    lf.add_frame("b", 1, T + P - 5_000_000, now_ns=T + 2 * P)
    lf.add_frame("a", 2, T + 2 * P, now_ns=T + 2 * P + 65_000_000)
    lf.add_frame("a", 3, T + 3 * P, now_ns=T + 3 * P + 65_000_000)  # 2 superseded before it could step
    assert counts["sync_superseded"] == 1
    end_ids, _, ref, rec = lf.newest_step(T + 3 * P + 65_000_000, flush=True)  # flush: no waiting
    assert ref == "latest:3" and end_ids == {"a": 3, "b": 1, "c": 1} and rec["mismatch"]
    for c in "bc":  # partners at the reference instant; c's frame of the NEXT instant is never mixed in
        lf.add_frame(c, 4, T + 4 * P - 4_000_000, now_ns=T + 4 * P + 60_000_000)
    lf.add_frame("c", 5, T + 5 * P - 4_000_000, now_ns=T + 5 * P + 60_000_000)
    lf.add_frame("a", 4, T + 4 * P, now_ns=T + 5 * P + 61_000_000)  # the reference camera delivers late
    end_ids, _, _, rec = lf.newest_step(T + 5 * P + 61_000_000)
    assert end_ids == {"a": 4, "b": 4, "c": 4} and not rec["mismatch"] and rec["age_ms"]["a"] == pytest.approx(127.7, 1e-3)
    assert lf.deadline_ns() is None  # nothing pending
    lf.add_frame("a", 5, T + 5 * P + 40_000_000, now_ns=T + 6 * P)  # b's newest (T + 4P - 4 ms) is late for it
    assert lf.deadline_ns() == T + 6 * P + lf.wait_ns and lf.wait_ns == int(P / 2)
    assert counts["sync_cap_fallback"] == 0
    cap = syncsets.LatestFrames(client, None, None, cams, period_s=1 / 15, counts=counts)
    for k, c in enumerate("bc"):  # partners only from LATER instants than the (very late) reference frame
        cap.add_frame(c, 7, T + 2 * P + k, now_ns=T + 2 * P + 70_000_000)
        cap.add_frame(c, 8, T + 3 * P + k, now_ns=T + 3 * P + 70_000_000)
    cap.add_frame("a", 1, T, now_ns=T + 4 * P)
    end_ids, _, _, rec = cap.newest_step(T + 4 * P)
    assert end_ids == {"a": 1, "b": 8, "c": 8} and rec["mismatch"] and counts["sync_cap_fallback"] == 2

    class M5:  # informational M5: final states -> displacement declared with the step
        sets = SimpleNamespace(cameras=cams)
        counts = collections.Counter()
        disp = 0.007
        timeline = SimpleNamespace(ready=lambda t: True, states=lambda ts: ([None] * len(ts), np.ones(len(ts), bool)))
        motion = SimpleNamespace(displacement=lambda states, ok: M5.disp)

        def add_robot(self, *a):
            pass

        def add_gripper(self, *a):
            pass

    state = SimpleNamespace(rows=lambda after, up_to: ([], []))
    counts2 = collections.Counter()
    m5 = M5()
    lf = syncsets.LatestFrames(client, m5, state, cams, period_s=1 / 15, counts=counts2)
    for k, disp in enumerate((0.007, np.inf, 0.001)):
        M5.disp = disp
        for c in cams:
            lf.add_frame(c, k, T + k * P, now_ns=T + k * P + 65_000_000)
        rec = lf.newest_step(T + k * P + 66_000_000)[3]
        assert rec["surface_disp_m"] == (disp if np.isfinite(disp) else "no_state")
        assert rec["motion_over_bound"] == (not disp <= 0.005)
    assert declared == [({"a": k, "b": k, "c": k}, d) for k, d in enumerate((0.007, None, 0.001))]
    assert counts2["sync_motion_over_bound"] == 2 and counts2["sync_mismatch"] == 0
    assert m5.counts == {"sets": 3, "over_bound": 2, "within_bound": 1}
    with pytest.raises(ValueError, match="training order"):
        syncsets.LatestFrames(client, m5, state, cams[::-1], period_s=1 / 15, counts=counts2)


def test_replay_episode_composition_and_cli_with_a_config(policies, recording, tmp_path, spies):
    cfgd = yaml.safe_load(C.DEFAULT_PATH.read_text())
    cfgd["root"] = str(tmp_path)
    cfgd["tasks"]["drawer-open"]["rig_document"] = None
    with serve({"reviwo-drawer-open-s0": policies["reviwo"]}) as srv:
        cfgd["server"]["port"] = srv.port
        cfgd["server"]["step_timeout_s"] = 60
        path = tmp_path / "robot_pc.yaml"
        path.write_text(yaml.safe_dump(cfgd))
        rows_path, summary_path = tmp_path / "rows.jsonl", tmp_path / "summary.json"
        import importlib.util

        spec = importlib.util.spec_from_file_location("rob_replay", REPO / "scripts/robot_pc/replay.py")
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        assert mod.main(["--config", str(path), "--episode", str(recording), "--policy", "reviwo-drawer-open-s0",
                         "--task", "drawer-open", "--camera", "cam_c", "--pace", "realtime", "--steps", "6",
                         "--out", str(rows_path), "--summary", str(summary_path)]) == 0
    summary = json.loads(summary_path.read_text())
    assert summary["robot_commands_sent"] == 0 and summary["sink"] == "MockSink" and summary["ok"] == 6
    assert summary["commands"] == 6 and summary["achieved_hz"] == pytest.approx(15, rel=0.1)
    assert len(rows_path.read_text().splitlines()) == 6 and spies["live_sink"] == 0


class FakeCapture:
    """A RealSenseCapture stand-in: synthetic 15 Hz frames stamped on the client clock (no device)."""

    def __init__(self, serial, *, native_hw, fps, depth, correction, rs_module, keep_native=False):
        self.serial, self.hw, self.k = serial, tuple(native_hw), 0
        self.profile = SimpleNamespace(get_stream=lambda s: SimpleNamespace(
            as_video_stream_profile=lambda: SimpleNamespace(get_intrinsics=lambda: SimpleNamespace(
                width=self.hw[1], height=self.hw[0], fx=60.0, fy=60.0, ppx=self.hw[1] / 2, ppy=self.hw[0] / 2,
                coeffs=[0.0] * 5))))
        self.t0 = None

    def read(self, timeout_ms=1000):
        import time

        from s4d.rwlearn.deploy import preprocess_frame

        now = time.monotonic_ns()
        self.t0 = self.t0 or now
        target = self.t0 + self.k * PERIOD_NS
        time.sleep(max(0.0, (target - now) / 1e9))
        rgb = np.random.default_rng(self.k).integers(0, 256, (*self.hw, 3), dtype=np.uint8)
        out = preprocess_frame(rgb, K_native=np.array([[60.0, 0, self.hw[1] / 2 + 0.5], [0, 60.0, self.hw[0] / 2 + 0.5],
                                                       [0, 0, 1]]), out_size=32)
        t = time.monotonic_ns()
        out.update(frame_id=10 + self.k, capture_ns=target, preprocess_ns=(t, t), native_hw=list(self.hw), metadata={})
        out["K"] = np.asarray(out["K"]).tolist()
        self.k += 1
        return out

    def stop(self):
        pass


def test_live_dry_run_logs_only_and_refuses_busy_devices(policies, tmp_path, spies, monkeypatch):
    import ops.policy_serving.client as client_mod

    monkeypatch.setattr(client_mod, "RealSenseCapture", FakeCapture)
    monkeypatch.setattr(devices, "require_free", lambda **k: (_ for _ in ()).throw(devices.DevicesBusy("busy")))
    with pytest.raises(devices.DevicesBusy):
        LiveCameras(["cam_a"], native_hw=(48, 64), rs_module=SimpleNamespace())
    monkeypatch.setattr(devices, "require_free", lambda **k: {"holders": []})
    cams = LiveCameras(["cam_a"], native_hw=(48, 64), rs_module=SimpleNamespace(stream=SimpleNamespace(color=0)))
    K = cams.start()
    with serve({"reviwo-x": policies["reviwo"]}) as srv:
        client = PolicyClient("127.0.0.1", srv.port, policy_id="reviwo-x", step_timeout_s=60,
                              choices={"native_hw": {"cam_a": [48, 64]}, "camera_id": "cam_a"})
        client.connect()
        cams.bind(K)
        log = tmp_path / "dryrun.jsonl"
        sink = sinks.DryRunSink(log)
        state = robot_state.StaticLatencyState(POSE, np.zeros(7), 3.0)
        try:
            rows = dryrun.run(client, _loop(client, state, sink), cams, state, cams=["cam_a"], steps=8)
        finally:
            cams.close()
            client.close()
            sink.close()
    assert len(rows) == 8 and all(r["state_source"] == "static_latency_only" for r in rows)
    logged = [json.loads(x) for x in log.read_text().splitlines()]
    assert logged and all(x["dry_run"] and not x["sent_to_robot"] for x in logged)
    assert spies["live_sink"] == 0 and set(spies["connect"]) == {("127.0.0.1", srv.port)}


# ------------------------------------------------------------------------------------------------ selection
def test_selection_from_the_server_contracts(policies):
    with serve({"reviwo-a": policies["reviwo"], "mvp3d-mv": policies["mvp3d_multiview"]}) as srv:
        w = cli.welcome({"server": {"port": srv.port}}, SimpleNamespace(host=None, port=None))
    rows = {r["policy_id"]: r for r in selection.describe(w["policies"])}
    assert rows["mvp3d-mv"]["views"] == 3 and rows["mvp3d-mv"]["needs_calibration"]
    spec = w["policies"]["reviwo-a"]["input_spec"]
    assert selection.session_cameras(spec) == ["cam_a"] and selection.session_cameras(spec, "cam_c") == ["cam_c"]
    with pytest.raises(selection.SelectionError):
        selection.session_cameras(spec, "cam_x")
    mv = w["policies"]["mvp3d-mv"]["input_spec"]
    assert selection.session_cameras(mv) == list(CAMS)
    with pytest.raises(selection.SelectionError, match="rig"):
        selection.session_choices(mv, list(CAMS), native_hw={c: (48, 64) for c in CAMS},
                                  K_native={c: np.eye(3) for c in CAMS}, rig_document=None)
    assert selection.depth_corrections(mv, list(CAMS), None, identity=True)["cam_a"]["factor"] == 1.0
    with pytest.raises(selection.SelectionError, match="does not offer"):
        selection.select_policy(w["policies"], "nope")
    with pytest.raises(selection.SelectionError, match="0 selectable"):  # FIXTURE policies are never auto-selected
        selection.select_policy(w["policies"], condition="reviwo")
    assert selection.select_policy(w["policies"], "reviwo-a") == "reviwo-a"


def test_bundle_is_reproducible_and_lists_every_file(tmp_path, monkeypatch):
    import importlib.util
    import tarfile

    spec = importlib.util.spec_from_file_location("rob_bundle", REPO / "scripts/robot_pc/make_bundle.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    wheels = tmp_path / "wheels"
    wheels.mkdir()
    for name in ("numpy-1.26.4", "pillow-11.3.0", "pyyaml-6.0.3", "scipy-1.15.3", "h5py-3.16.0", "pip-24.3.1",
                 "pyrealsense2-2.59.1.11053", "pyzmq-27.2.0", "robot_msg_irsl-0.4.0", "pyserial-3.5",
                 "opencv_python_headless-4.10.0.84"):
        (wheels / f"{name}-cp312-cp312-manylinux_2_17_x86_64.whl").write_bytes(name.encode())
    urdf = tmp_path / "toy.urdf"
    urdf.write_text("<robot/>")
    urdf.with_suffix(".manifest.json").write_text("{}")
    rig = tmp_path / "rig"
    rig.mkdir()
    for doc in ("a", "b"):
        (rig / f"{doc}.json").write_text(doc)
        (rig / f"{doc}.reference.npz").write_bytes(doc.encode())
    asset = tmp_path / "x.m5.npz"
    asset.write_bytes(b"asset")
    validation = tmp_path / "v.json"
    validation.write_text(json.dumps({"pass": True, "asset_sha256": mod.sha256(asset),
                                      "header": {"format": "x", "urdf_sha256": mod.sha256(urdf)},
                                      "episodes": [{"episode": "e", "pass": True}]}))
    monkeypatch.setattr(mod, "SRC", ["ops/__init__.py", "scripts/_bootstrap.py"])
    monkeypatch.setattr(mod, "SETUP", "scripts/_bootstrap.py")
    monkeypatch.setattr(mod, "URDF", urdf)
    monkeypatch.setattr(mod, "RIG", rig)
    monkeypatch.setattr(mod, "RIG_DOCS", ("a", "b"))
    head = mod.git("rev-parse", "HEAD")
    args = SimpleNamespace(commit="HEAD", wheels=wheels, m5_asset=asset, m5_validation=validation)
    a = mod.build(SimpleNamespace(**vars(args), out_dir=tmp_path / "o1"))
    b = mod.build(SimpleNamespace(**vars(args), out_dir=tmp_path / "o2"))
    assert a["sha256"] == b["sha256"] and a["commit"] == head
    with tarfile.open(a["bundle"]) as t:
        files = {m.name.split("/", 1)[1] for m in t.getmembers() if m.isfile()}
        manifest = json.loads(t.extractfile(f"s4d-robot-pc-{head[:7]}/BUNDLE.json").read())
        req = t.extractfile(f"s4d-robot-pc-{head[:7]}/wheels/requirements.txt").read().decode()
    assert set(manifest["files"]) | {"BUNDLE.json"} == files and "setup.sh" in files
    assert "numpy==1.26.4 --hash=sha256:" in req and "pip" not in req
    assert manifest["pins"]["numpy"] == "numpy==1.26.4" and manifest["safety"].startswith("execution only via infer.py")
    validation.write_text(json.dumps({"pass": False, "header": {}, "episodes": []}))
    with pytest.raises(SystemExit, match="did not pass"):
        mod.build(SimpleNamespace(**vars(args), out_dir=tmp_path / "o3"))



def test_first_selection_waits_hello_timeout_not_connect_timeout(policies, monkeypatch):
    """A first HELLO may load the policy for seconds (lazy server load): the SPEC wait is hello_timeout_s, the TCP /
    WELCOME wait stays connect_timeout_s."""
    import time

    pol = policies["reviwo"]
    real = pol.reset_session

    def slow(*a, **k):
        time.sleep(0.8)
        return real(*a, **k)

    monkeypatch.setattr(pol, "reset_session", slow)
    choices = {"native_hw": {"cam_a": [48, 64]}, "camera_id": "cam_a"}
    with serve({"reviwo-x": pol}) as srv:
        quick = PolicyClient("127.0.0.1", srv.port, policy_id="reviwo-x", choices=choices, connect_timeout_s=0.2,
                             hello_timeout_s=0.3)
        with pytest.raises((TimeoutError, OSError)):
            quick.connect()
        quick.close()
        time.sleep(1.0)  # the server finishes the abandoned HELLO and ends that session with the connection
        patient = PolicyClient("127.0.0.1", srv.port, policy_id="reviwo-x", choices=choices, connect_timeout_s=0.2,
                               hello_timeout_s=5.0, takeover=True)
        assert patient.connect()["camera_ids"] == ["cam_a"]
        patient.close()


# ------------------------------------------------------------------------------------------------ switch_controller
class FakePublisher:
    def __init__(self):
        self.sent = []

    def send(self, data):
        self.sent.append(data)


def fake_robot_msg():
    """Stand-in of robot_msg 0.4.0's dataclasses (positional Vec3/Quaternion, keyword RobotAction) + a JSON codec."""
    from dataclasses import asdict, dataclass, field

    @dataclass
    class Vec3:
        x: float
        y: float
        z: float

    @dataclass
    class Quaternion:
        x: float
        y: float
        z: float
        w: float

    @dataclass
    class Pose:
        position: Vec3
        orientation: Quaternion

    @dataclass
    class RobotActionRealTimeAction:
        ee_pose_rt: list = None
        q_rt: list = None
        target_dt: float = 0.01

    @dataclass
    class RobotAction:
        frame_id: str
        real_time_action: RobotActionRealTimeAction = None
        metadata: dict = field(default_factory=dict)

    return SimpleNamespace(Vec3=Vec3, Quaternion=Quaternion, Pose=Pose, RobotAction=RobotAction,
                           RobotActionRealTimeAction=RobotActionRealTimeAction,
                           encode=lambda m: json.dumps(asdict(m)).encode())


def test_switch_controller_frames_and_action_format():
    from scipy.spatial.transform import Rotation

    rng = np.random.default_rng(4)
    for _ in range(5):
        flange = np.concatenate([rng.uniform(-1, 1, 3), Rotation.random(random_state=rng).as_quat()])
        ee = sc.ee_from_flange(flange)
        # the collector's rule: ee = flange @ make_T(rot_z(-90), 0) (teleop.T_FLANGE_EE), flange = ee @ inv(...)
        Rz = Rotation.from_euler("z", -90, degrees=True).as_matrix()
        want = Rotation.from_quat(flange[3:]).as_matrix() @ Rz
        np.testing.assert_allclose(Rotation.from_quat(ee[3:]).as_matrix(), want, atol=1e-12)
        np.testing.assert_allclose(ee[:3], flange[:3], atol=0)
        back = sc.flange_from_ee(ee)
        assert poses.quat_angle(back[3:], flange[3:]) < 1e-9 and np.allclose(back[:3], flange[:3])
        assert back[6] >= 0 and ee[6] >= 0  # the collector's mat_to_quat sign convention (w >= 0)
    rm = fake_robot_msg()
    msg = json.loads(rm.encode(sc.robot_action(rm, POSE, 0.05)))
    assert msg["frame_id"] == "world" and msg["real_time_action"]["target_dt"] == 0.05
    p = msg["real_time_action"]["ee_pose_rt"][0]
    ee = sc.ee_from_flange(POSE)
    assert [p["position"][k] for k in "xyz"] == list(ee[:3]) and [p["orientation"][k] for k in "xyzw"] == list(ee[3:])
    state = SimpleNamespace(sequence=7, q=list(range(7)), metadata=SimpleNamespace(send_time=SimpleNamespace(seconds=1.5)),
                            ee_frames=[SimpleNamespace(pose=SimpleNamespace(
                                position=SimpleNamespace(x=ee[0], y=ee[1], z=ee[2]),
                                orientation=SimpleNamespace(x=ee[3], y=ee[4], z=ee[5], w=ee[6])))])
    row = sc.state_sample(state, 123)
    assert row["seq"] == 7 and row["t_ns"] == 123 and np.allclose(row["flange_pose7"], POSE, atol=1e-12)


def test_live_sink_sends_setpoints_through_the_link_and_the_gripper():
    auth = safety.authorize_execution(cli_execute=True, operator_start=True, policy_info={"execution_eligible": True})
    sent, closures = [], []
    link = SimpleNamespace(send_flange_pose=lambda p, *a: sent.append(p))
    grip = SimpleNamespace(set_closure=lambda c: closures.append(c))
    live_sink = sinks.LiveSink(auth, link, grip)
    with pytest.raises(safety.ExecutionRefused, match="setpoints"):
        live_sink.send({"goal": POSE})
    live_sink.send(executor.Setpoint(1, "track", POSE, 0.4, "r1"))
    live_sink.send(executor.Setpoint(2, "track", POSE, None, "r1"))
    assert sent == [POSE, POSE] and closures == [0.4] and live_sink.sent == 2


class FakeZmq:
    PUB, SUB, LINGER, SNDHWM, NOBLOCK, CONFLATE, SUBSCRIBE = "PUB", "SUB", 1, 2, 4, 5, 6
    EVENT_ACCEPTED, EVENT_DISCONNECTED = 32, 512

    def __init__(self, fail_bind=False):
        self.fail_bind, self.sent, self.bound = fail_bind, [], []
        outer = self

        class Sock:
            def __init__(self, kind):
                self.kind = kind

            def setsockopt(self, *a):
                pass

            def get_monitor_socket(self, ev):
                return SimpleNamespace(poll=lambda t: False, close=lambda *a: None)

            def bind(self, ep):
                if outer.fail_bind:
                    raise OSError("Address already in use")
                outer.bound.append(ep)

            def send(self, data, flags=0):
                outer.sent.append(data)

            def close(self, *a):
                pass

        self.Context = lambda: SimpleNamespace(socket=Sock)


def test_controller_link_is_the_collectors_commander(monkeypatch):
    from ops.robot_pc import controller

    auth = safety.authorize_execution(cli_execute=True, operator_start=True, policy_info={"execution_eligible": True})
    with pytest.raises(safety.ExecutionRefused):
        controller.ControllerLink(None, zmq_module=FakeZmq(), robot_msg=fake_robot_msg())
    with pytest.raises(safety.ExecutionRefused, match="commander"):
        controller.ControllerLink(auth, zmq_module=FakeZmq(fail_bind=True), robot_msg=fake_robot_msg())
    z, rm = FakeZmq(), fake_robot_msg()
    rm.ControllerCommand = lambda ctrl_mode: {"ctrl_mode": ctrl_mode}
    rm.encode = lambda m: json.dumps(m if isinstance(m, dict) else __import__("dataclasses").asdict(m)).encode()
    link = controller.ControllerLink(auth, zmq_module=z, robot_msg=rm)
    assert z.bound == [sc.COMMAND_BIND, sc.ACTION_BIND]
    monkeypatch.setattr(link, "update_peers", lambda: {"command": 0, "action": 0})
    with pytest.raises(safety.ExecutionRefused, match="not connected"):
        link.engage(settle_s=0)
    monkeypatch.setattr(link, "update_peers", lambda: {"command": 1, "action": 1})
    link.engage(settle_s=0)
    link.send_flange_pose(POSE)
    import time
    link.release(lambda: {"t_ns": time.monotonic_ns(), "ee_pose7": POSE}, still=lambda: True, wait_still_s=0.1)
    msgs = [json.loads(m) for m in z.sent]
    assert msgs[0] == {"ctrl_mode": 3} and msgs[-1] == {"ctrl_mode": 1}  # engage mode 3 ... IDLE on release
    assert [m["real_time_action"]["target_dt"] for m in msgs[1:3]] == [0.05, 0.05]  # action + final hold


class FakeSerial:
    """A Robotiq answering FC03 status reads (open, activated) and FC16 writes; records every request."""

    def __init__(self, *a, **k):
        assert k.get("exclusive") is True  # single Modbus master
        assert k["write_timeout"] == 0.5
        self.requests, self.reply = [], b""

    def reset_input_buffer(self):
        pass

    def write(self, req):
        from ops.robot_pc import gripper as G

        self.requests.append(bytes(req))
        if req[1] == 0x03:
            body = bytes([9, 0x03, 6, 0x39, 0, 0, 3, 3, 0])
        else:
            body = bytes(req[:6])
        self.reply = body + G.crc16_modbus(body)

    def read(self, n):
        return self.reply[:n]

    def fileno(self):
        return 12345  # never passed to a real ioctl; tests inject the syscall

    def close(self):
        pass


def test_gripper_single_owner_read_only_status_and_command_path(tmp_path, monkeypatch):
    from ops.robot_pc import gripper as G

    with pytest.raises(G.PortBusy, match="second Modbus master"):
        G.RobotiqRTU(serial_module=SimpleNamespace(Serial=FakeSerial), lock_path=str(tmp_path / "l"),
                     check_holders=lambda port: [4242])
    made = []

    def factory():
        dev = G.RobotiqRTU(serial_module=SimpleNamespace(Serial=lambda *a, **k: made.append(FakeSerial(*a, **k)) or
                                                         made[-1]),
                           lock_path=str(tmp_path / "lock"), check_holders=lambda port: [], ioctl=lambda *a: None,
                           listen_s=0)
        return dev

    with pytest.raises(G.PortBusy, match="another s4d gripper owner"):
        first = factory()
        try:
            factory()  # the flock: one owner per host
        finally:
            first.close()
    ro = G.GripperWorker(factory, read_only=True)
    with pytest.raises(PermissionError):
        ro.set_closure(1.0, 10**30)
    ro.start()
    import time as _t

    deadline = _t.monotonic() + 3
    while len(ro.samples) < 5 and _t.monotonic() < deadline:
        _t.sleep(0.02)
    ro.stop()
    ro.join(2)
    assert len(ro.samples) >= 5 and ro.latest.position == 3 and ro.latest.status == 3
    assert all(r[1] == 0x03 for r in made[-1].requests)  # status reads ONLY: no activation, no command
    t, reg = ro.at_or_before(_t.monotonic_ns())
    assert reg == 3.0 and ro.origin.startswith("measured")
    rw = G.GripperWorker(factory, read_only=False)
    rw.set_closure(1.0, _t.monotonic_ns() + 1_000_000_000)
    rw.start()
    deadline = _t.monotonic() + 3
    while not rw.sent and _t.monotonic() < deadline:
        _t.sleep(0.02)
    rw.stop()
    rw.join(2)
    writes = [r for r in made[-1].requests if r[1] == 0x10]
    assert writes and writes[0][10] == 230  # closure 1 -> register 230; never an activation sequence
    assert all(w[7] == 0x09 for w in writes)  # rACT + rGTO set, no rACT=0 reset


def test_live_state_reader_is_a_read_only_subscriber(monkeypatch):
    calls = []

    class Sock:
        def __init__(self, kind):
            calls.append(("socket", kind))

        def setsockopt(self, *a):
            calls.append(("opt", a))

        def connect(self, ep):
            calls.append(("connect", ep))

        def poll(self, ms):
            import threading

            return threading.Event().wait(ms / 1000)

        def close(self, *a):
            calls.append(("close",))

    class Ctx:
        def socket(self, kind):
            return Sock(kind)

        def term(self):
            pass

    zmq = SimpleNamespace(Context=Ctx, SUB="SUB", LINGER=1, CONFLATE=2, SUBSCRIBE=3)
    reader = sc.LiveStateReader(approved=True, zmq_module=zmq, robot_msg=SimpleNamespace())
    try:
        assert ("socket", "SUB") in calls and ("connect", sc.STATE_ENDPOINT) in calls
        assert not any(c[0] in ("bind", "send") for c in calls)
        with pytest.raises(robot_state.StateUnavailable, match="no robot state"):
            reader.bound_state(10**18)
        reader.samples.append({"t_ns": 5, "seq": 1, "q": np.ones(7), "flange_pose7": np.array(POSE),
                               "ee_pose7": sc.ee_from_flange(POSE), "signature": b"s"})
        with pytest.raises(robot_state.StateUnavailable, match="gripper register"):
            reader.bound_state(10)
        with pytest.raises(robot_state.StateUnavailable, match="no fresh robot state"):
            reader.bound_state(10 + sc.STATE_TIMEOUT_NS)
    finally:
        reader.close()


def test_missing_token_fails_closed_with_the_path(policies, tmp_path):
    with serve({"reviwo-x": policies["reviwo"]}) as srv:
        srv.token = b"secret-token"  # an authenticated server (as on the 5090)
        cfg = C.load(root=tmp_path)
        cfg["server"]["port"] = srv.port
        args = SimpleNamespace(token_file=None, host=None, port=None, step_timeout_s=None)
        with pytest.raises(SystemExit, match=r"secrets/token is MISSING"):
            cli.connect(cfg, args, "reviwo-x", {"native_hw": {"cam_a": [48, 64]}, "camera_id": "cam_a"}, "t")


def test_live_sets_fall_back_to_exact_finality_after_an_irregular_interval():
    from s4d.realworld.train_store import mutual_sync

    rng = np.random.default_rng(9)
    stamps = _timelines(rng, (0, 20, -25), frames=30)
    stamps[1] = np.sort(np.append(stamps[1], stamps[1][10] + 15_000_000))  # an early (irregular) cam_b frame
    ref, _ = mutual_sync(stamps)
    sets = m5_live.LiveSets(CAMS, 1 / 15)
    events = sorted((int(t) + 65_000_000, v, k, int(t)) for v, s in enumerate(stamps) for k, t in enumerate(s))
    got = {}
    for _, v, k, t in events:
        sets.add(CAMS[v], k, t)
        got.update({s["frame_ids"]["cam_a"]: s for s in sets.pop_final()})
    got.update({s["frame_ids"]["cam_a"]: s for s in sets.pop_final(end_of_stream=True)})
    assert sets.irregular_intervals == 2 and not sets.early  # the inserted frame shortens two intervals
    after = [k for k in got if stamps[0][k] > stamps[1][11]]  # decided after the irregular frame arrived
    assert after and all([got[k]["frame_ids"].get(c, -1) for c in CAMS] == ref[k].tolist() for k in after)


def test_realtime_replay_steps_on_the_newest_observation_when_behind(policies, recording):
    """A path slower than the frame period lowers the step rate (newest observation) instead of a backlog."""
    import time

    rec = RecordedEpisode(recording)
    with serve({"reviwo-x": policies["reviwo"]}) as srv:
        client = PolicyClient("127.0.0.1", srv.port, policy_id="reviwo-x", step_timeout_s=60,
                              choices={"native_hw": {"cam_a": list(rec.native_hw("cam_a"))}, "camera_id": "cam_a"})
        client.connect()
        cameras = RecordedCameras(rec, ["cam_a"], depth=False)
        clock = replay.ReplayClock(int(cameras.timestamps_ns("cam_a")[0]), "realtime")
        state = robot_state.RecordedRobotState(rec, clock.offset_ns)
        lp = _loop(client, state, sinks.MockSink(), clock.now)
        real_step = lp.step

        def slow_step(*a, **k):
            time.sleep(0.12)  # ~2 frame periods
            return real_step(*a, **k)

        lp.step = slow_step
        rows = replay.run(client, lp, cameras, state, clock, cams=["cam_a"])
        client.close()
    assert lp.counts["skipped_behind"] >= 4 and len(rows) + lp.counts["skipped_behind"] == 14
    ages = [r["client_ms"]["capture_to_command_ms"] for r in rows if r["status"] == "ok"]
    assert max(ages) < 400  # no backlog: every step is on a fresh observation


def test_live_state_health_follows_the_collector_rules():
    ee = sc.ee_from_flange(POSE)

    def sample(t, sig=b"s", q=0.1):
        return {"t_ns": t, "seq": t, "q": np.full(7, q), "ee_pose7": ee, "flange_pose7": np.array(POSE),
                "signature": sig}

    assert sc.state_blockers([], 0) == ["no fresh robot state"]
    assert sc.state_blockers([sample(0)], 60_000_000) == ["no fresh robot state"]  # older than 50 ms
    assert sc.state_blockers([sample(0, q=0.0)], 1) == ["robot state not initialized (all joints zero)"]
    frozen = [sample(t) for t in range(0, 300_000_001, 10_000_000)]
    assert sc.state_blockers(frozen, 300_000_000)[0].startswith("robot state frozen")
    live = [sample(t, sig=bytes([t // 10_000_000])) for t in range(0, 300_000_001, 10_000_000)]
    assert sc.state_blockers(live, 300_000_000) == []
    bad = dict(sample(0), ee_pose7=np.array([0, 0, 0, 0, 0, 0, 2.0]))
    assert "invalid" in sc.state_blockers([bad], 1)[0]


def test_step_deadline_bounds_a_session_renewal(policies, monkeypatch):
    """A session lost on the server during a step renews with a HELLO bounded by the STEP deadline, never by the
    20 s first-selection wait (the step returns in time; the robot loop keeps control)."""
    import time

    pol = policies["reviwo"]
    srv = PolicyServer({"reviwo-x": pol}, warmup_iterations=0, session_idle_timeout_s=0.3)
    srv.start()
    try:
        client = PolicyClient("127.0.0.1", srv.port, policy_id="reviwo-x", hello_timeout_s=20.0, step_timeout_s=0.6,
                              choices={"native_hw": {"cam_a": [48, 64]}, "camera_id": "cam_a"})
        client.connect()
        time.sleep(1.0)  # the server expires the idle session
        real = pol.reset_session
        monkeypatch.setattr(pol, "reset_session", lambda *a, **k: (time.sleep(3.0), real(*a, **k))[1])
        rgb = np.zeros((48, 64, 3), np.uint8)
        t0 = time.monotonic_ns()
        for k in range(3):
            client.add_frame_native("cam_a", k, t0 + k * PERIOD_NS, rgb, pose7=POSE, gripper_register=3.0)
        start = time.monotonic()
        r = client.step({"cam_a": 2}, "r", list(POSE), 3.0, pose_time_ns=t0 + 2 * PERIOD_NS - 1_000_000,
                        gripper_time_ns=t0 + 2 * PERIOD_NS - 1_000_000, pad_run_start=True)
        elapsed = time.monotonic() - start
        assert not r.ok and r.code == "session_unknown" and elapsed < 1.5, (r.status, r.code, elapsed)
        assert client.stats["renew_failed"] == 1
        client.close()
    finally:
        srv.shutdown()


def test_outputs_never_open_devices_or_follow_symlinks(tmp_path):
    for bad in ("/dev/null", "/dev/ttyUSB0", "/proc/self/status", "/sys/kernel/x"):
        with pytest.raises(safety.ExecutionRefused, match="not a regular file"):
            safety.safe_output(bad)
    link = tmp_path / "log.jsonl"
    link.symlink_to("/dev/null")
    with pytest.raises(safety.ExecutionRefused):
        sinks.DryRunSink(link)
    with pytest.raises(safety.ExecutionRefused):
        cli.JsonlRows(link)
    with pytest.raises(safety.ExecutionRefused):
        cli.write_json(link, {"x": 1})
    (tmp_path / "dir").mkdir()
    with pytest.raises(safety.ExecutionRefused):
        safety.safe_output(tmp_path / "dir")
    ok = sinks.DryRunSink(tmp_path / "fine.jsonl")
    ok.send({"a": 1})
    ok.close()
    assert json.loads((tmp_path / "fine.jsonl").read_text())["sent_to_robot"] is False
    for port in (28400, 28401, 28402):
        with pytest.raises(safety.ExecutionRefused, match="switch_controller channel"):
            cli.welcome({"server": {"port": port}}, SimpleNamespace(host=None, port=None))
        with pytest.raises(safety.ExecutionRefused):
            C.client({"root": str(tmp_path), "server": {"port": port}}, "p", {}, name="t")


def test_route_bench_selects_policies_and_refuses_a_busy_robot_pc(monkeypatch):
    import importlib.util

    spec = importlib.util.spec_from_file_location("rob_route_bench", REPO / "scripts/robot_pc/route_bench.py")
    rb = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(rb)
    offered = {f"{c}-{t}-s{s}": {"info": {"condition": c, "task": t, "seed": s, "status": st}}
               for c, st in (("reviwo", "deployable"), ("dynamo", "restricted"), ("mvp3d_single", "quarantined"))
               for t in ("drawer-open", "mug-hanging") for s in (1, 0)}
    offered["reviwo-drawer-open"] = {"info": {"condition": "reviwo", "task": "drawer-open", "seed": 0,
                                              "status": "deployable", "alias_of": "reviwo-drawer-open-s0"}}
    assert rb.default_policies(offered) == ["dynamo-drawer-open-s0", "dynamo-mug-hanging-s0", "reviwo-drawer-open-s0",
                                            "reviwo-mug-hanging-s0"]
    busy = SimpleNamespace(returncode=1, stdout="", stderr="DevicesBusy: pid 7 looks like a recorder/teleop process")
    monkeypatch.setattr(rb.subprocess, "run", lambda *a, **k: busy)
    with pytest.raises(SystemExit, match="busy"):
        rb.robot_pc_idle()


@pytest.mark.parametrize("case", ["no_signature", "future", "backwards", "invalid_pose"])
def test_executor_fails_closed_on_state_edge_cases(case):
    ex = executor.HighRateExecutor(sinks.MockSink())
    ex.accept(_cmd([POSE[0] + 0.01, *POSE[1:]], age_ms=1e6))
    assert ex.tick(10_000_000, POSE, state_t_ns=10_000_000, signature=b"a").kind == "track"
    args = {"no_signature": (20_000_000, POSE, 20_000_000, None),
            "future": (20_000_000, POSE, 30_000_000, b"b"),
            "backwards": (20_000_000, POSE, 5_000_000, b"b"),
            "invalid_pose": (20_000_000, [*POSE[:3], 0, 0, 0, 2.0], 20_000_000, b"b")}[case]
    assert ex.tick(args[0], args[1], state_t_ns=args[2], signature=args[3]) is None and ex.stopped
    assert ex.tick(30_000_000, POSE, state_t_ns=30_000_000, signature=b"c") is None  # latched


def test_policy_gripper_input_is_the_measured_register_never_the_command(recording, policies):
    with h5py.File(recording, "r+") as f:  # the operator commanded "closed" while the gripper stayed open
        f["action/gripper"][...] = 1.0
    rec = RecordedEpisode(recording)
    state = robot_state.RecordedRobotState(rec)
    st = state.bound_state(int(rec.timestamps_ns("cam_a")[5]))
    assert st.gripper_register == 3.0 and st.gripper_origin.startswith("measured")  # gripper/position, not 3 + 227
    robot_state.require_measured_gripper(st)

    class CommandedGripper(robot_state.RobotStateSource):
        kind = "test"

        def bound_state(self, t):
            return robot_state.BoundState(POSE, t - 1_000_000, 230.0, t - 1_000_000, "commanded: action/gripper")

    with serve({"reviwo-x": policies["reviwo"]}) as srv:
        client = PolicyClient("127.0.0.1", srv.port, policy_id="reviwo-x", step_timeout_s=60,
                              choices={"native_hw": {"cam_a": [48, 64]}, "camera_id": "cam_a"})
        client.connect()
        row = _loop(client, CommandedGripper(), sinks.MockSink()).step({"cam_a": 0}, 10**9, "r")
        client.close()
    assert row["status"] == "skipped" and "not a measured gripper state" in row["detail"]
    for origin in ("action/gripper", "unlabelled", ""):
        with pytest.raises(robot_state.StateUnavailable):
            robot_state.require_measured_gripper(robot_state.BoundState(POSE, 0, 3.0, 0, origin))


def test_direct_tunnel_helper_command_and_fail_closed_checks(tmp_path):
    from ops.robot_pc import tunnel

    verified = ("ssh -N -i secrets/ssh_5090_ed25519 -o IdentitiesOnly=yes -o IdentityAgent=none -o "
                "PasswordAuthentication=no -o KbdInteractiveAuthentication=no -o UserKnownHostsFile=secrets/"
                "known_hosts_5090 -o StrictHostKeyChecking=yes -o ExitOnForwardFailure=yes -o ServerAliveInterval=15 "
                "-o ServerAliveCountMax=3 -o ControlMaster=no -o ControlPath=none -L 127.0.0.1:47100:172.17.0.2:47100 "
                "irsl@143.248.142.98")
    assert " ".join(tunnel.command(tmp_path)) == verified  # the coordinator-verified command, verbatim
    with pytest.raises(tunnel.TunnelError, match="missing"):
        tunnel.preflight(tmp_path, 47100)
    (tmp_path / "secrets").mkdir()
    key, kh = tmp_path / tunnel.KEY, tmp_path / tunnel.KNOWN_HOSTS
    key.write_text("not a real key")
    kh.write_text("x")
    key.chmod(0o644)
    with pytest.raises(tunnel.TunnelError, match="0600"):
        tunnel.preflight(tmp_path, 47100)
    key.chmod(0o600)
    with pytest.raises(safety.ExecutionRefused):
        tunnel.preflight(tmp_path, 28400)  # never a switch_controller channel
    busy = socket.socket()
    busy.bind(("127.0.0.1", 0))
    busy.listen(1)
    try:
        with pytest.raises(tunnel.TunnelError, match="already bound"):
            tunnel.preflight(tmp_path, busy.getsockname()[1])
    finally:
        busy.close()
    assert tunnel.stop(tmp_path, 47100) == {"stopped": False, "reason": "no pid file"}
    (tmp_path / "logs").mkdir()
    tunnel.paths(tmp_path, 47100)["pid"].write_text(f"{os.getpid()}\n")  # a pid that is NOT the tunnel's ssh
    with pytest.raises(safety.ExecutionRefused, match="not this tunnel"):
        tunnel.stop(tmp_path, 47100)
    st = tunnel.status(tmp_path, 1)
    assert st["ok"] is False and st["pid_alive_ssh_tunnel"] is False


def test_a_refused_session_renewal_is_an_error_result_not_an_exception(policies, monkeypatch):
    import time

    pol = policies["reviwo"]
    srv = PolicyServer({"reviwo-x": pol}, warmup_iterations=0, session_idle_timeout_s=0.3)
    srv.start()
    try:
        client = PolicyClient("127.0.0.1", srv.port, policy_id="reviwo-x", step_timeout_s=2.0,
                              choices={"native_hw": {"cam_a": [48, 64]}, "camera_id": "cam_a"})
        client.connect()
        time.sleep(1.0)  # the session expires on the server
        monkeypatch.setattr(pol, "reset_session", lambda *a, **k: (_ for _ in ()).throw(ValueError("withdrawn")))
        t0 = time.monotonic_ns()
        for k in range(3):
            client.add_frame_native("cam_a", k, t0 + k * PERIOD_NS, np.zeros((48, 64, 3), np.uint8), pose7=POSE,
                                    gripper_register=3.0)
        r = client.step({"cam_a": 2}, "r", list(POSE), 3.0, pose_time_ns=t0 + 2 * PERIOD_NS - 1_000_000,
                        gripper_time_ns=t0 + 2 * PERIOD_NS - 1_000_000, pad_run_start=True)
        assert not r.ok and r.code not in (None, "session_taken_over") and client.stats["renew_failed"] == 1
        client.close()
    finally:
        srv.shutdown()


class _SubOnly:
    """zmq stand-in for the read-only reader: a SUB that never receives (samples are injected by the test)."""
    SUB, LINGER, CONFLATE, SUBSCRIBE = "SUB", 1, 2, 3

    class Context:
        @staticmethod
        def poll(ms):
            import threading

            return threading.Event().wait(ms / 1000)

        def socket(self, kind):
            return SimpleNamespace(setsockopt=lambda *a: None, connect=lambda ep: None, poll=self.poll,
                                   close=lambda *a: None)

        def term(self):
            pass


class _Captures:
    """Synthetic 15 Hz camera frames (client-clock stamps), drained like LiveCameras."""

    def __init__(self, cam, hw=(48, 64)):
        self.cam, self.hw, self.k, self.t0, self.errors = cam, hw, 0, None, {}

    def drain(self, cam):
        import time as _t

        from s4d.rwlearn.deploy import preprocess_frame

        now = _t.monotonic_ns()
        self.t0 = self.t0 or now
        out = []
        while self.t0 + self.k * PERIOD_NS <= now:
            t = self.t0 + self.k * PERIOD_NS
            rgb = np.random.default_rng(self.k).integers(0, 256, (*self.hw, 3), dtype=np.uint8)
            cap = preprocess_frame(rgb, K_native=np.array([[60.0, 0, 32.5], [0, 60.0, 24.5], [0, 0, 1]]), out_size=32)
            cap.update(frame_id=10 + self.k, capture_ns=t, arrival_ns=now, preprocess_ns=(now, now),
                       native_hw=list(self.hw), metadata={}, native_rgb=rgb)
            cap["K"] = np.asarray(cap["K"]).tolist()
            out.append(cap)
            self.k += 1
        return out


def test_live_session_dry_run_full_chain_with_recording_calls(policies):
    """Live loop on fakes: state reader samples at 1 kHz, 15 Hz frames, fixture policy over loopback, validation,
    500 Hz executor -> DryRunSink; every frame / observation / state row / step / setpoint goes to the recorder with
    sent=False; nothing is sent to a controller."""
    import threading as _th
    import time as _t

    from ops.robot_pc import live

    reader = sc.LiveStateReader(approved=True, zmq_module=_SubOnly, robot_msg=SimpleNamespace())
    stop_feed = _th.Event()

    robot = {"pose": np.array(POSE)}  # a simulated robot that follows the setpoints (as switch_controller would)
    feed_lock = _th.Lock()
    next_sample, sequence = _t.monotonic_ns(), 0

    def feed_up_to(now):
        nonlocal next_sample, sequence
        with feed_lock:
            sp = getattr(ex, "_setpoint", None) if "ex" in box else None
            pose = robot["pose"] if sp is None else np.array(sp)
            ee = sc.ee_from_flange(pose)
            while next_sample <= now:
                sequence += 1
                reader.samples.append({"t_ns": next_sample, "seq": sequence, "q": np.full(7, 0.1),
                                       "dq": np.zeros(7), "tau": np.zeros(7), "flange_pose7": pose,
                                       "ee_pose7": ee, "signature": sequence.to_bytes(8, "little")})
                next_sample += 1_000_000

    def feed():
        while not stop_feed.is_set():
            feed_up_to(_t.monotonic_ns())
            _t.sleep(0.001)

    class Captures(_Captures):
        def drain(self, cam):
            caps = super().drain(cam)
            # Synthetic state is sampled on a 1 kHz clock even when the GIL delays the feeder thread. Preserve
            # the actual binding limits: don't turn host scheduling gaps into missing fake measurements.
            feed_up_to(_t.monotonic_ns())
            return caps

    box = {}

    feeder = _th.Thread(target=feed, name="fake-state-feeder", daemon=True)
    feeder.start()
    calls = collections.Counter()
    rec = SimpleNamespace(**{n: (lambda *a, n=n, **k: calls.update([n])) for n in
                             ("frame", "observation", "robot_state", "gripper", "step", "gripper_command", "executed", "event")})
    sink = sinks.DryRunSink()
    try:
        with serve({"reviwo-x": policies["reviwo"]}) as srv:
            client = PolicyClient("127.0.0.1", srv.port, policy_id="reviwo-x", step_timeout_s=60,
                                  choices={"native_hw": {"cam_a": [48, 64]}, "camera_id": "cam_a"})
            client.connect()
            state = live.LiveState(reader, live.StaticGripper())
            ex = executor.HighRateExecutor(sinks.MockSink(), state_timeout_s=2.0)  # in-process server holds the GIL
            box["ex"] = ex
            session = live.LiveSession(client=client, cameras=Captures("cam_a"), cams=["cam_a"], state=state,
                                       limits=ActionLimits(max_action_age_ms=1e5),  # in-process server: GIL-bound
                                       executor=ex, sink=sink, recorder=rec, sent=False)
            rows = session.run(steps=12)
            client.close()
    finally:
        stop_feed.set()
        feeder.join(5)
        assert not feeder.is_alive()
        reader.close()
    assert session.stop_reason == "budget reached" and len(rows) == 12
    ok = [r for r in rows if r["status"] == "ok"]  # the first frame may predate the state stream (binding refused)
    assert len(ok) >= 10 and all(r["gripper_origin"].startswith("static") for r in ok)
    assert calls["frame"] >= 12 and calls["observation"] >= 12 and calls["step"] == 12 and calls["robot_state"] > 100
    assert calls["executed"] > 0 and ex.counts["track"] > 0 and len(sink.items) == calls["executed"], (
        ex.counts, ex.stopped, [r.get("rejected") for r in rows], calls)


# ----------------------------------------------------------------------------- LIVE safety regressions (fakes only)
def _fake_link():
    from ops.robot_pc.controller import ControllerLink

    z, msg = FakeZmq(), fake_robot_msg()
    msg.ControllerCommand = lambda ctrl_mode: {"ctrl_mode": ctrl_mode}
    msg.encode = lambda m: json.dumps(m if isinstance(m, dict) else dataclasses.asdict(m)).encode()
    auth = safety.authorize_execution(cli_execute=True, operator_start=True, policy_info={"execution_eligible": True})
    return ControllerLink(auth, zmq_module=z, robot_msg=msg), z


@pytest.mark.parametrize("failure", ["stale", "nonfinite", "zero_quaternion", "nonunit_quaternion",
                                     "hold", "still", "sample", "idle"])
def test_release_independent_phases_always_idle_and_close(failure, monkeypatch, caplog):
    import time

    link, z = _fake_link()
    link.request_mode(3)
    sample = {"t_ns": time.monotonic_ns(), "ee_pose7": POSE}
    if failure == "stale":
        sample["t_ns"] -= 60_000_000
    if failure == "nonfinite":
        sample["ee_pose7"] = (float("nan"), *POSE[1:])
    if failure in ("zero_quaternion", "nonunit_quaternion"):
        sample["ee_pose7"] = (*POSE[:3], 0, 0, 0, 0 if failure == "zero_quaternion" else 1.002)

    def boom(*a):
        raise OSError("injected " + failure)

    if failure == "hold":
        monkeypatch.setattr(link, "_send_measured_ee_hold", boom)
    if failure == "idle":
        monkeypatch.setattr(link, "_request_mode", boom)
    outcome = link.release(boom if failure == "sample" else lambda: sample,
                           still=boom if failure == "still" else lambda: True, wait_still_s=0.1)
    msgs = [json.loads(m) for m in z.sent]
    assert not link.socks and outcome is link.release_outcome
    assert outcome["idle"].startswith("error:") if failure == "idle" else outcome["idle"] == "sent"
    assert msgs[-1] == {"ctrl_mode": 1} if failure != "idle" else "real_time_action" in msgs[-1]
    holds = [m for m in msgs if "real_time_action" in m]
    assert len(holds) == (1 if failure in ("still", "idle") else 0)
    assert outcome["hold"] == "sent" if holds else outcome["hold"].startswith(("skipped:", "error:"))
    assert outcome["still"].startswith("error:") if failure == "still" else outcome["still"] is True
    assert caplog.records  # the skipped/failed phase is visible


def test_release_hold_preserves_measured_pose_bytes_and_still_requires_freshness():
    import time

    from ops.robot_pc import live

    link, z = _fake_link()
    ee = tuple(-x for x in POSE)  # including the quaternion sign; no flange round trip
    link.request_mode(3)
    link.release(lambda: {"t_ns": time.monotonic_ns(), "ee_pose7": ee}, still=lambda: True)
    expected = link.msg.encode(sc.robot_action_ee(link.msg, ee, link.target_dt))
    assert z.sent[1] == expected
    state = SimpleNamespace(latest=lambda: {"t_ns": time.monotonic_ns() - 50_000_001, "dq": np.zeros(7)})
    assert not live.still(state)


def test_engage_retries_and_disconnect_is_latched(monkeypatch):
    from ops.robot_pc import controller

    link, z = _fake_link()
    link.peers = {"command": 1, "action": 1}
    events = collections.deque()
    monitor = SimpleNamespace(recv_monitor_message=lambda mon: events.popleft())
    monkeypatch.setitem(sys.modules, "zmq", SimpleNamespace(utils=SimpleNamespace(monitor=monitor)))
    monkeypatch.setitem(sys.modules, "zmq.utils", SimpleNamespace(monitor=monitor))
    monkeypatch.setitem(sys.modules, "zmq.utils.monitor", monitor)
    now = [0.0]
    monkeypatch.setattr(controller.time, "monotonic", lambda: now[0])
    monkeypatch.setattr(controller.time, "sleep", lambda dt: now.__setitem__(0, now[0] + dt))
    link.engage()
    assert 5 <= link.mode_requests_sent <= 6
    assert all(json.loads(m) == {"ctrl_mode": 3} for m in z.sent)
    events.extend([{"event": z.EVENT_DISCONNECTED}, {"event": z.EVENT_ACCEPTED}])
    link._monitors["action"] = SimpleNamespace(poll=lambda t: bool(events), close=lambda *a: None)
    assert not link.peers_ok() and link.peers["action"] == 1
    link.close()


def test_gripper_tiocexcl_rechecks_holders_and_reports_unverifiable(tmp_path, monkeypatch, caplog):
    import termios

    from ops.robot_pc import gripper as G

    monkeypatch.setattr(devices, "scan", lambda **kw: {"recorders": []})
    events = []

    class Serial(FakeSerial):
        def close(self):
            events.append("close")

    scans = iter([{"holders": [], "unverifiable": 2}, {"holders": [77], "unverifiable": 3}])

    def check(port):
        events.append("scan")
        return next(scans)

    def ioctl(fd, op):
        assert fd == 12345 and op == termios.TIOCEXCL
        events.append("TIOCEXCL")

    with pytest.raises(G.PortBusy, match="after TIOCEXCL"):
        G.RobotiqRTU(serial_module=SimpleNamespace(Serial=Serial), lock_path=str(tmp_path / "l"),
                     check_holders=check, ioctl=ioctl)
    assert events == ["scan", "TIOCEXCL", "scan", "close"]
    dev = G.RobotiqRTU(serial_module=SimpleNamespace(Serial=Serial), lock_path=str(tmp_path / "l"),
                       check_holders=lambda p: {"holders": [], "unverifiable": 4}, ioctl=ioctl)
    assert dev.unverifiable == 4 and "unverifiable" in caplog.text
    dev.close()


def test_port_holders_counts_unreadable_proc_fds(tmp_path, monkeypatch):
    from ops.robot_pc import gripper as G

    for pid in (1, 2, 3):
        (tmp_path / str(pid) / "fd").mkdir(parents=True)
    (tmp_path / "2/fd/7").symlink_to("/fake/ttyUSB0")
    original = Path.iterdir

    def iterdir(path):
        if path == tmp_path / "1/fd":
            raise PermissionError("root fds")
        return original(path)

    monkeypatch.setattr(Path, "iterdir", iterdir)
    assert G.port_holders("/fake/ttyUSB0", str(tmp_path), self_pid=3) == {"holders": [2], "unverifiable": 1}


@pytest.mark.parametrize("case", ["expired", "cleared", "fault", "inactive", "reconnect"])
def test_gripper_inhibits_invalid_targets_and_reconnect_replay(case):
    import threading
    import time

    from ops.robot_pc import gripper as G

    writes, reads, connections = [], [], []
    enough = threading.Event()
    worker = None

    class Device:
        def read_status(self):
            reads.append(1)
            if case == "reconnect" and len(connections) == 1:
                raise OSError("bus lost")
            if len(reads) >= 3:
                enough.set()
            return G.GripperStatus(time.monotonic_ns(), 1, 1, 0 if case == "inactive" else 3, 3,
                                   9 if case == "fault" else 0, 3, 3, 0)

        def write(self, *args):
            writes.append(args)

        def close(self):
            pass

    def factory():
        connections.append(1)
        return Device()

    worker = G.GripperWorker(factory, read_only=False)
    worker.set_closure(1.0, time.monotonic_ns() + (-1 if case == "expired" else 5_000_000_000))
    if case == "cleared":
        worker.clear_target()
    worker.start()
    try:
        assert enough.wait(2.0)
        if case in ("fault", "inactive"):
            assert not worker.healthy()
        if case == "reconnect":
            assert len(connections) == 2 and worker.healthy()
    finally:
        worker.stop(join_timeout=1.0)
    assert not worker.is_alive() and not writes


def test_gripper_expiry_inhibits_keepalive_and_stop_joins():
    import threading
    import time

    from ops.robot_pc import gripper as G

    first_write = threading.Event()
    writes = []
    dev = SimpleNamespace(
        read_status=lambda: G.GripperStatus(time.monotonic_ns(), 1, 1, 3, 0, 0, 3, 3, 0),  # gOBJ 0: moving
        write=lambda *a: (writes.append(a), first_write.set()), close=lambda: None)
    worker = G.GripperWorker(lambda: dev, read_only=False)
    worker.set_closure(1, time.monotonic_ns() + 5_000_000_000)
    worker.start()
    try:
        assert first_write.wait(1)
        with worker._target_lock:
            worker._expires_ns = time.monotonic_ns() + 50_000_000
        _wait_until(lambda: len(worker.holds) == 1)
        worker._last_write = 0  # make the next cycle a would-be keepalive
        samples = len(worker.samples)
        _wait_until(lambda: len(worker.samples) >= samples + 3)
        assert len(writes) == 2 and writes[-1][0] == 3
        assert len(worker.holds) == 1 and worker.holds[0][2] == "expiry"
        worker.stop()  # the reader must not overwrite the explicitly stale status during this assertion
        worker.latest = dataclasses.replace(worker.latest, t_ns=time.monotonic_ns() - 100_000_001)
        assert not worker.healthy()
    finally:
        worker.stop()
    assert not worker.is_alive() and worker._target is None


def _unit_session(*, sent=False, recorder=None, gripper=None, link=None, sink=None, **kw):
    import time

    from ops.robot_pc import live

    sample = {"t_ns": time.monotonic_ns(), "seq": 1, "q": np.ones(7), "dq": np.zeros(7), "tau": np.zeros(7),
              "flange_pose7": POSE, "ee_pose7": sc.ee_from_flange(POSE), "signature": b"fake"}
    state = live.LiveState(SimpleNamespace(samples=[sample]), gripper or live.StaticGripper())
    ex = executor.HighRateExecutor(sinks.MockSink())
    if sink is None:
        auth = safety.authorize_execution(cli_execute=True, operator_start=True, policy_info={"execution_eligible": True})
        sink = live.LiveSink(auth, link, gripper) if sent else sinks.DryRunSink()
    return live.LiveSession(client=SimpleNamespace(add_capture=lambda *a, **kw: None),
                            cameras=_Captures("cam_a"), cams=["cam_a"], state=state,
                            limits=ActionLimits(), executor=ex, sink=sink, recorder=recorder, sent=sent, **kw)


def _session_recorder(**overrides):
    return SimpleNamespace(**{**{name: lambda *a, **kw: None for name in
                                 ("frame", "observation", "robot_state", "gripper", "step", "gripper_command",
                                  "executed", "event")}, **overrides})


@pytest.mark.parametrize("failure", ["tick", "executed", "send_arm", "send_gripper"])
def test_executor_thread_exceptions_stop_and_send_outcomes_are_recorded(failure):
    import threading
    import time

    executed, events = [], []

    def boom(*a, **kw):
        raise OSError("injected " + failure)

    link = SimpleNamespace(peers_ok=lambda: True, release=lambda *a, **kw: events.append("release"),
                           send_flange_pose=boom if failure == "send_arm" else lambda *a: None)
    grip = SimpleNamespace(set_closure=boom if failure == "send_gripper" else lambda *a: None)
    rec = _session_recorder(executed=boom if failure == "executed" else lambda *a: executed.append(a),
                            event=lambda t, msg: events.append(msg))
    session = _unit_session(sent=True, recorder=rec, link=link, gripper=grip)
    session.executor.command = SimpleNamespace(reference_id="r", expires_ns=time.monotonic_ns() + 1_000_000_000)
    session.executor.tick = boom if failure == "tick" else lambda *a, **kw: \
        executor.Setpoint(time.monotonic_ns(), "track", POSE, 0.5, "r")
    session._exec_thread = threading.Thread(target=session._executor_loop)
    session._exec_thread.start()
    assert session.stop_event.wait(1)
    session.shutdown.finish()
    assert not session._exec_thread.is_alive() and "injected " + failure in session.stop_reason
    assert "release" in events and any(str(e).startswith("stop:") for e in events)
    if failure.startswith("send_"):
        assert len(executed) == 1 and executed[0][4] is False
        outcome = json.loads(executed[0][5])
        assert "injected" in outcome["arm" if failure == "send_arm" else "gripper"]
        if failure == "send_gripper":
            assert outcome["arm"] == "sent"


def test_live_sink_propagates_command_expiry_and_json_is_additive():
    import time

    from ops.robot_pc import live

    targets = []
    auth = safety.authorize_execution(cli_execute=True, operator_start=True, policy_info={"execution_eligible": True})
    sink = live.LiveSink(auth, SimpleNamespace(send_flange_pose=lambda p, *a: None),
                         SimpleNamespace(set_closure=lambda *a: targets.append(a)))
    base = executor.Setpoint(1, "track", POSE, 0.4, "r")
    sp = live.LiveSetpoint(1, "track", POSE, 0.4, "r", time.monotonic_ns() + 1_000_000)
    sink.send(sp)
    assert targets == [(0.4, sp.expires_ns)]
    assert sp.to_json() == {**base.to_json(), "expires_ns": sp.expires_ns}
    with pytest.raises(safety.ExecutionRefused, match="expiry"):
        sink.send(base)


@pytest.mark.parametrize("problem", ["disconnect", "fault", "comm_loss"])
def test_execution_peer_and_gripper_health_stops_are_latched(problem):
    import threading
    import time

    from ops.robot_pc import gripper as G

    released = threading.Event()
    link = SimpleNamespace(peers_ok=lambda: problem != "disconnect", release=lambda *a, **kw: released.set(),
                           send_flange_pose=lambda p, *a: pytest.fail("must not send"))
    grip = G.GripperWorker(lambda: pytest.fail("no hardware"), read_only=False)
    grip.latest = G.GripperStatus(time.monotonic_ns() - (210_000_000 if problem == "comm_loss" else 0),
                                  1, 1, 3, 3, 9 if problem == "fault" else 0, 3, 3, 0)
    grip.error = "bus lost" if problem == "comm_loss" else None
    session = _unit_session(sent=True, link=link, gripper=grip)
    session._exec_thread = threading.Thread(target=session._executor_loop)
    session._exec_thread.start()
    assert session.stop_event.wait(1) and released.wait(1)
    reason = session.stop_reason
    session.stop("operator later")
    session.shutdown.finish()
    assert session.stop_reason == reason
    assert ("disconnected" if problem == "disconnect" else "gripper unhealthy") in reason


def test_dry_run_unhealthy_gripper_is_recorded_without_stop():
    import time

    from ops.robot_pc import gripper as G

    events = []
    grip = G.GripperWorker(lambda: None)
    grip.latest = G.GripperStatus(time.monotonic_ns(), 1, 1, 3, 3, 9, 3, 3, 0)
    grip.error = "fault"
    session = _unit_session(gripper=grip, recorder=_session_recorder(event=lambda t, e: events.append(e)))
    assert session._check_gripper() and not session.stop_event.is_set()
    assert any("healthy=False" in e for e in events)


def test_server_failures_stop_but_rejections_do_not_and_main_exceptions_propagate():
    session = _unit_session(max_failed_steps=2)
    session.loop.step = lambda *a, **kw: {"status": "ok", "command": None, "rejected": "non_finite"}
    rows = session.run(steps=6)  # directive K: a rejected action sends nothing, it never ends the run
    assert len(rows) == 6 and session.stop_reason == "budget reached"
    down = _unit_session(max_failed_steps=2)
    down.loop.step = lambda *a, **kw: {"status": "timeout", "code": "timeout", "command": None}
    rows = down.run(steps=10)  # the policy server / route does not answer: communication loss stops the run
    assert len(rows) == 2 and "policy server: 2 consecutive steps without an answer" in down.stop_reason
    failed = _unit_session()

    def boom(*a, **kw):
        raise RuntimeError("policy main loop")

    failed.loop.step = boom
    with pytest.raises(RuntimeError, match="policy main loop"):
        failed.run()
    assert failed.stop_event.is_set() and "main loop failed" in failed.stop_reason
    assert not failed._exec_thread.is_alive()


def test_release_proceeds_after_executor_join_timeout_when_recorder_is_blocked(caplog):
    import threading
    import time

    recording, unblock_recording, released = threading.Event(), threading.Event(), threading.Event()

    def executed(*a):
        recording.set()
        assert unblock_recording.wait(3)

    link = SimpleNamespace(peers_ok=lambda: True, send_flange_pose=lambda p, *a: None,
                           release=lambda *a, **kw: released.set())
    session = _unit_session(sent=True, link=link, recorder=_session_recorder(executed=executed))
    session.executor.command = SimpleNamespace(reference_id="r", expires_ns=time.monotonic_ns() + 2_000_000_000)
    session.executor.tick = lambda *a, **kw: executor.Setpoint(time.monotonic_ns(), "track", POSE, None, "r")
    session._exec_thread = threading.Thread(target=session._executor_loop)
    session._exec_thread.start()
    try:
        assert recording.wait(1)
        session.stop("operator stop while recorder blocked")
        assert released.wait(1.5) and session._exec_thread.is_alive()
        assert "executor thread did not exit" in caplog.text
    finally:
        unblock_recording.set()
        session._exec_thread.join(1)
        session.shutdown.finish()


def test_release_wait_is_bounded_to_collector_timeout(monkeypatch):
    from ops.robot_pc import controller

    link, z = _fake_link()
    now = [10.0]
    monkeypatch.setattr(controller.time, "monotonic", lambda: now[0])
    monkeypatch.setattr(controller.time, "monotonic_ns", lambda: int(now[0] * 1e9))
    monkeypatch.setattr(controller.time, "sleep", lambda dt: now.__setitem__(0, now[0] + dt))
    link.request_mode(3)
    link.release(lambda: {"t_ns": int(now[0] * 1e9), "ee_pose7": POSE}, still=lambda: False, wait_still_s=9)
    assert now[0] == pytest.approx(11.05)  # <=1 s stillness wait, then the collector's 50 ms IDLE settle
    assert json.loads(z.sent[-1]) == {"ctrl_mode": 1}


def test_live_session_refuses_multiview_even_if_cams_are_incomplete():
    from ops.robot_pc import live

    for cams, spec in [(["a", "b"], {}), (["a"], {"views": 2}), (["a"], {"camera_ids": ["a", "b"]})]:
        with pytest.raises(safety.ExecutionRefused, match="multiview sync sets"):
            live.LiveSession(client=SimpleNamespace(spec=spec), cameras=None, cams=cams, state=None, limits=None,
                             executor=None, sink=None)


def test_stop_releases_while_policy_step_is_blocked_in_required_order():
    import threading

    order = []
    policy_entered, unblock_policy, released = threading.Event(), threading.Event(), threading.Event()
    grip = SimpleNamespace(clear_target=lambda: order.append("gripper clear"),
                           stop=lambda: order.append("gripper joined"))
    session = None

    def release(*a, **kw):
        assert session._exec_thread is not None and not session._exec_thread.is_alive()
        order.append("release")
        released.set()

    link = SimpleNamespace(peers_ok=lambda: True, release=release)
    session = _unit_session(sent=True, link=link, gripper=grip)
    original_join = session.join_executor

    def join():
        original_join()
        order.append("executor joined")

    session.join_executor = join

    def step(*a, **kw):
        policy_entered.set()
        assert unblock_policy.wait(3)
        return {"status": "error"}

    session.loop.step = step
    runner = threading.Thread(target=session.run)
    runner.start()
    try:
        assert policy_entered.wait(1)
        session.stop("operator key")
        assert released.wait(1) and runner.is_alive()  # release didn't need the policy call to return
        session.shutdown.finish()
        assert order == ["executor joined", "gripper clear", "release", "gripper joined"]
    finally:
        unblock_policy.set()
        runner.join(2)
        session.shutdown.finish()
    order.append("recorder close")
    session.stop("again")
    session.shutdown.finish()
    assert order.count("release") == 1 and order.index("release") < order.index("recorder close")


@pytest.mark.parametrize("checkpoint", ["before_engagement", "after_enter", "constructed", "after_wait",
                                        "during_engage", "after_engage", "before_streaming", "ctrlc", "normal",
                                        "multiview", "preflight_fault_clears", "preflight_fault_persists",
                                        "prompt_sigterm", "prompt_sigint", "ready_sigterm", "cleanup_signals", "cleanup_baseexception",
                                        "reposition_at_enter", "moving_at_enter", "unhealthy_at_enter",
                                        "moving_before_engage", "hold_authorization"])
def test_launcher_cancellation_checkpoints_and_resource_order(checkpoint, monkeypatch, tmp_path):
    """All launcher dependencies are fakes: no controller connection, serial port, camera, policy server or disk."""
    import signal
    import threading
    import time

    from ops.robot_pc import camera_warmup, controller, live, tunnel
    from ops.robot_pc import cameras as camera_mod
    from ops.robot_pc import gripper as G
    from ops.robot_pc import recorder as recorder_mod
    from scripts.robot_pc import infer as run

    order, links, zs, sessions, prompts = [], [], [], [], []
    handlers = {}
    engaging = [False]
    entered = [False]
    recorded_events, messages = [], []
    LinkClass, SessionClass = controller.ControllerLink, live.LiveSession

    def cancel():
        handlers[signal.SIGTERM](signal.SIGTERM, None)

    def fake_signal(sig, handler):
        previous = handlers.get(sig, signal.SIG_DFL)
        handlers[sig] = handler
        return previous

    def sleep(dt):
        if checkpoint == "ctrlc" and engaging[0]:
            raise KeyboardInterrupt("during settle")

    monkeypatch.setattr(run.signal, "signal", fake_signal)
    monkeypatch.setattr(run.time, "sleep", sleep)
    monkeypatch.setattr(run.sys, "stdin", io.StringIO(""))
    def say(message):
        messages.append(message)
        if checkpoint == "ready_sigterm" and message.startswith("READY"):
            cancel()

    monkeypatch.setattr(run, "say", say)
    monkeypatch.setattr(cli, "load_config", lambda args: {"root": str(tmp_path)})
    monkeypatch.setattr(tunnel, "welcome", lambda port: {"fake": True})
    monkeypatch.setattr(devices, "require_free", lambda **kw: None)
    monkeypatch.setattr(devices, "scan", lambda **kw: {"controllers": [1]})

    class Reader:
        def __init__(self, **kw):
            pass

        @property
        def samples(self):
            pose = list(POSE)
            if checkpoint == "reposition_at_enter" and entered[0]:
                pose[0] += 0.1
            moving = checkpoint == "moving_at_enter" and entered[0] or \
                checkpoint == "moving_before_engage" and "wait" in order
            return [{"t_ns": time.monotonic_ns(), "ee_pose7": pose, "flange_pose7": pose,
                     "dq": np.ones(7) if moving else np.zeros(7)}]

        def close(self):
            order.append("reader close")

    class Grip:
        origin, error, unverifiable = "measured fake", None, 2

        def __init__(self, *a, **kw):
            self.latest = G.GripperStatus(time.monotonic_ns(), 1, 1, 3, 3,
                                          9 if checkpoint.startswith("preflight_fault") else 0, 3, 3, 0)
            self.reads = 0

        def start(self):
            pass

        def healthy(self):
            self.reads += 1
            if checkpoint == "preflight_fault_clears" and self.reads > 2:
                self.latest = dataclasses.replace(self.latest, fault=0, t_ns=time.monotonic_ns())
            return self.latest.fault == 0

        def clear_target(self):
            order.append("gripper clear")

        def stop(self, join_timeout=1):
            order.append("gripper joined")

    class Cameras:
        def __init__(self, *a, **kw):
            pass

        def start(self, corrections):
            return {"cam_a": np.eye(3)}

        def bind(self, K):
            pass

        def close(self):
            order.append("cameras close")
            if checkpoint == "cleanup_baseexception":
                raise KeyboardInterrupt("fake resource close")

    class Recorder:
        session_dir = tmp_path

        def __init__(self, *a, **kw):
            self.closed = False

        def close(self, *a):
            if not self.closed:
                order.append("recorder close")
                if checkpoint == "cleanup_signals":
                    for sig in (signal.SIGINT, signal.SIGTERM, signal.SIGINT):
                        handlers[sig](sig, None)
                    order.append("recorder finalized")
                self.closed = True
            return {}

        def event(self, t, text):
            recorded_events.append(text)

    def make_link(auth):
        z, msg = FakeZmq(), fake_robot_msg()
        msg.ControllerCommand = lambda ctrl_mode: {"ctrl_mode": ctrl_mode}
        msg.encode = lambda m: json.dumps(m if isinstance(m, dict) else dataclasses.asdict(m)).encode()
        link = LinkClass(auth, zmq_module=z, robot_msg=msg)
        links.append(link)
        zs.append(z)
        link.update_peers = lambda: {"command": 1, "action": 1}

        def wait(*a, **kw):
            order.append("wait")
            if checkpoint == "after_wait":
                cancel()
            return True

        engage_original, release_original = link.engage, link.release

        def engage(**kw):
            order.append("engage")
            engaging[0] = True
            try:
                if checkpoint == "during_engage":
                    link.request_mode(3)
                    cancel()
                else:
                    if checkpoint == "ctrlc":
                        def interrupt(dt):
                            raise KeyboardInterrupt("during settle")
                        kw["cancel"] = SimpleNamespace(is_set=kw["cancel"].is_set, wait=interrupt)
                    engage_original(settle_s=0.3 if checkpoint == "ctrlc" else 0, **kw)
                if checkpoint == "after_engage":
                    cancel()
            finally:
                engaging[0] = False

        def release(*a, **kw):
            if sessions and sessions[0]._exec_thread is not None:
                assert not sessions[0]._exec_thread.is_alive()
            order.append("release")
            return release_original(*a, **kw)

        link.wait_subscribers, link.engage, link.release = wait, engage, release
        if checkpoint == "constructed":
            cancel()
        return link

    class Session(SessionClass):
        def __init__(self, **kw):
            super().__init__(**kw)
            sessions.append(self)
            if checkpoint == "before_streaming":
                cancel()

        def join_executor(self, timeout=1):
            super().join_executor(timeout)
            order.append("executor joined")

        def run(self, **kw):
            order.append("stream")
            assert not hasattr(self.loop.limits, "workspace_min_m")  # directive K: no box around the start pose
            self._exec_thread = threading.Thread(target=self.stop_event.wait)
            self._exec_thread.start()
            self.stop("operator key")
            self.shutdown.finish()
            return []

    def operator_input():
        prompts.append(1)
        entered[0] = True
        if checkpoint in ("prompt_sigterm", "prompt_sigint"):
            sig = signal.SIGTERM if checkpoint == "prompt_sigterm" else signal.SIGINT
            handlers[sig](sig, None)
            pytest.fail("signal did not interrupt START input")
        if checkpoint == "after_enter":
            cancel()
        return ""

    monkeypatch.setattr(sc, "LiveStateReader", Reader)
    monkeypatch.setattr(sc, "state_blockers", lambda *a: ["unhealthy"] if
                        checkpoint == "unhealthy_at_enter" and entered[0] else [])
    if checkpoint == "hold_authorization":
        monkeypatch.setattr(safety, "authorize_execution", lambda **kw: safety.authorize_hardware_test(
            test="hold", cli_flag=True, operator_start=True))
    monkeypatch.setattr(G, "GripperWorker", Grip)
    spec = {"views": 2 if checkpoint == "multiview" else 1, "modalities": ["rgb"],
            "trained_camera_orders": [["cam_a", "cam_b"]] if checkpoint == "multiview" else [["cam_a"]]}
    info = {"execution_eligible": True, "checkpoint_sha256": "ab" * 32}  # SPEC + START re-check: the same labels
    monkeypatch.setattr(cli, "choose_policy", lambda *a: ("p", "t", spec, dict(info)))
    monkeypatch.setattr(cli, "welcome", lambda *a: {"policies": {"p": {"info": dict(info)}}})
    monkeypatch.setattr(selection, "session_cameras", lambda *a: ["cam_a", "cam_b"] if spec["views"] > 1 else ["cam_a"])
    monkeypatch.setattr(cli, "rig_document", lambda *a: None)
    monkeypatch.setattr(selection, "depth_corrections", lambda *a, **kw: None)
    monkeypatch.setattr(selection, "session_choices", lambda *a, **kw: {})
    monkeypatch.setattr(camera_mod, "LiveCameras", Cameras)
    monkeypatch.setattr(camera_warmup, "wait_until_stable", lambda *a, **kw:
                        {"status": "stable", "waited_s": 1.5, "cameras": {}})
    monkeypatch.setattr(cli, "connect", lambda *a: SimpleNamespace(close=lambda: order.append("client close"),
                                                                    spec_info=dict(info)))
    monkeypatch.setattr(recorder_mod, "SessionRecorder", Recorder)
    OriginalExecutor = executor.HighRateExecutor

    def executor_factory(*a, **kw):
        if checkpoint == "before_engagement":
            cancel()
        return OriginalExecutor(*a, **kw)

    monkeypatch.setattr(executor, "HighRateExecutor", executor_factory)
    monkeypatch.setattr(C, "limits", lambda *a: ActionLimits())
    monkeypatch.setattr(controller, "ControllerLink", make_link)
    monkeypatch.setattr(live, "LiveSession", Session)
    code = run.main(["--policy", "p"], operator_input=operator_input)
    # directive K: no start box and no "robot still" requirement - repositioning / moving at ENTER runs normally
    if checkpoint in ("normal", "preflight_fault_clears", "cleanup_signals", "cleanup_baseexception",
                       "reposition_at_enter", "moving_at_enter", "moving_before_engage"):
        assert code == 0 and "stream" in order
        assert order.index("executor joined") < order.index("gripper clear") < order.index("gripper joined")
    elif checkpoint in ("multiview", "preflight_fault_persists"):
        assert code == 3 and not links and not prompts
    elif checkpoint in ("unhealthy_at_enter", "hold_authorization"):
        assert code == 3 and not links and prompts
    else:
        assert code == 130 and "stream" not in order
    if links:
        assert order.count("release") == 1 and not links[0].socks
        assert order.index("gripper clear") < order.index("release") < order.index("gripper joined")
        assert order.index("gripper joined") < order.index("recorder close")
        modes = [m["ctrl_mode"] for data in zs[0].sent if "ctrl_mode" in (m := json.loads(data))]
        if "engage" in order:
            assert modes[0] == 3 and modes[-1] == 1
        else:
            assert not modes
        closes = [order.index(name) for name in ("recorder close", "client close", "cameras close", "reader close")]
        assert closes == sorted(closes)
        release_events = [e for e in recorded_events if e.startswith("release ")]
        assert len(release_events) == 1 and release_events[0] in messages
    if checkpoint == "cleanup_signals":
        assert "recorder finalized" in order and messages.count("shutdown in progress") == 3
    assert not [e for e in recorded_events if e.startswith('{"profile"')]  # no profile / start box any more
    if checkpoint in ("before_engagement", "after_enter", "constructed", "after_wait"):
        assert "engage" not in order


def test_release_barrier_finishes_inflight_send_before_hold_and_refuses_late_send():
    import threading
    import time

    entered, resume, releasing = threading.Event(), threading.Event(), threading.Event()
    link, z = _fake_link()
    link.request_mode(3)
    send = link.socks["action"].send

    def delayed(data, flags):
        if not entered.is_set():
            entered.set()
            assert resume.wait(2)
        send(data, flags)

    link.socks["action"].send = delayed
    sender = threading.Thread(target=lambda: link.send_flange_pose(POSE))

    def latest():
        releasing.set()  # release has acquired the barrier and disabled streaming
        return {"t_ns": time.monotonic_ns(), "ee_pose7": POSE}

    releaser = threading.Thread(target=lambda: link.release(latest, still=lambda: True))
    sender.start()
    try:
        assert entered.wait(1)
        releaser.start()
        assert not releasing.wait(0.05) and len(z.sent) == 1
        resume.set()
        assert releasing.wait(1)
        with pytest.raises(safety.ExecutionRefused, match="released"):
            link.send_flange_pose(POSE)
    finally:
        resume.set()
        sender.join(2)
        releaser.join(2)
    assert not sender.is_alive() and not releaser.is_alive() and not link.socks
    msgs = [json.loads(m) for m in z.sent]
    assert len(msgs) == 4 and msgs[-1] == {"ctrl_mode": 1}
    assert link.release_outcome == {"hold": "sent", "still": True, "idle": "sent"}
    assert z.sent[2] == link.msg.encode(sc.robot_action_ee(link.msg, POSE, link.target_dt))
    with pytest.raises(safety.ExecutionRefused, match="released"):
        link.send_flange_pose(POSE)


@pytest.mark.parametrize("block", ["read", "write"])
def test_blocked_serial_never_blocks_executor_target_update_or_arm_release(block):
    import threading
    import time

    from ops.robot_pc import gripper as G
    from ops.robot_pc import live

    blocked, resume, sent, released = (threading.Event() for _ in range(4))
    writes, order = [], []

    class Device:
        def read_status(self):
            if block == "read" and writes:
                blocked.set()
                assert resume.wait(3)
            return G.GripperStatus(time.monotonic_ns(), 1, 1, 3, 3, 0, 230, 71, 0)

        def write(self, position, speed, force):
            writes.append(position)
            if block == "write" and len(writes) == 1:
                blocked.set()
                assert resume.wait(3)

        def close(self):
            pass

    worker = G.GripperWorker(Device, read_only=False)
    worker.set_closure(1, time.monotonic_ns() + 5_000_000_000)
    worker.start()
    assert blocked.wait(1)
    auth = safety.authorize_execution(cli_execute=True, operator_start=True, policy_info={"execution_eligible": True})
    link = SimpleNamespace(send_flange_pose=lambda p, *a: None,
                           release=lambda *a, **kw: (order.append("release"), released.set()) and
                           {"hold": "sent", "still": True, "idle": "sent"})
    sink = live.LiveSink(auth, link, worker)
    sp = live.LiveSetpoint(time.monotonic_ns(), "track", POSE, 0.5, "r", time.monotonic_ns() + 1_000_000_000)
    executor_thread = threading.Thread(target=lambda: (sink.send(sp), sent.set()))
    shutdown = live.MotionShutdown(threading.Event())
    shutdown.session = SimpleNamespace(join_executor=lambda: (executor_thread.join(0.1), order.append("joined")))
    shutdown.gripper, shutdown.link = worker, link
    shutdown.state = SimpleNamespace(latest=lambda: {"t_ns": time.monotonic_ns(), "dq": np.zeros(7)})
    try:
        executor_thread.start()
        assert sent.wait(0.1), "set_closure waited for serial I/O"
        shutdown.request()
        assert released.wait(0.2), "arm release waited for serial I/O or gripper.stop"
        assert worker._target is None and worker.is_alive()
        assert order[:2] == ["joined", "release"]
    finally:
        resume.set()
        executor_thread.join(2)
        shutdown.finish()
        worker.stop()
    assert not worker.is_alive()
    assert all(p == 71 for p in writes[1:])  # no mailbox target replays after cancellation


@pytest.mark.parametrize("event", ["stop", "clear", "expiry"])
@pytest.mark.parametrize("status", ["healthy", "stale", "fault", "inactive", "read_only", "never_written"])
def test_gripper_physical_hold_once_per_event_at_latest_measured_position(event, status):
    import threading
    import time

    from ops.robot_pc import gripper as G

    wrote, held = threading.Event(), threading.Event()
    writes = []
    worker = None

    class Device:
        def read_status(self):
            return G.GripperStatus(time.monotonic_ns() - (101_000_000 if wrote.is_set() and status == "stale" else 0),
                                   1, 1, 0 if wrote.is_set() and status == "inactive" else 3, 0,  # gOBJ 0: moving
                                   5 if wrote.is_set() and status == "fault" else 0, 230, 83, 0)

        def write(self, position, speed, force):
            writes.append((position, speed, force))
            (held if position == 83 else wrote).set()

        def close(self):
            pass

    worker = G.GripperWorker(Device, read_only=status == "read_only", speed=64, force=0)
    if status not in ("read_only", "never_written"):
        worker.set_closure(1, time.monotonic_ns() + 5_000_000_000)
    worker.start()
    try:
        if status not in ("read_only", "never_written"):
            assert wrote.wait(1)
        deadline = time.monotonic() + 1
        while (worker.latest is None or len(worker.samples) < 2) and time.monotonic() < deadline:
            time.sleep(0.002)
        if event == "stop":
            worker.stop()
            worker.stop()  # idempotent; a second call is the same stop event
        elif event == "clear":
            worker.clear_target()
        else:
            with worker._target_lock:
                if worker._target is not None:
                    worker._expires_ns = time.monotonic_ns() - 1
        if status == "healthy":
            assert held.wait(3)
            _wait_until(lambda: len(worker.holds) == 1)
        if event != "stop":
            samples = len(worker.samples)
            _wait_until(lambda: len(worker.samples) >= samples + 3)
        assert len(worker.holds) == (1 if status == "healthy" else 0)
        if worker.holds:
            t, register, reason = worker.holds[0]
            assert register == 83 and (t, register) in worker.sent
            assert ("stop" if event == "stop" else "clear_target" if event == "clear" else "expiry") in reason
            assert writes == [(230, 64, 0), (83, 64, 0)]
        elif status in ("read_only", "never_written"):
            assert not writes
    finally:
        worker.stop()


def test_rtu_listens_after_exclusivity_before_any_transmission(tmp_path, monkeypatch):
    from ops.robot_pc import gripper as G

    monkeypatch.setattr(devices, "scan", lambda **kw: {"recorders": []})
    events, made = [], []

    class Serial(FakeSerial):
        def __init__(self, *a, **kw):
            super().__init__(*a, **kw)
            made.append(self)

        def reset_input_buffer(self):
            events.append("reset")

        def close(self):
            events.append("close")

    def holders(port):
        events.append("holders")
        return {"holders": [], "unverifiable": 7}

    def read(n):
        assert n == 1 and events == ["holders", "TIOCEXCL", "holders", "reset"]
        events.append("read")
        return b"\x09"

    with pytest.raises(G.PortBusy, match="bus traffic without our request: another Modbus master"):
        G.RobotiqRTU(serial_module=SimpleNamespace(Serial=Serial), lock_path=str(tmp_path / "listen"),
                     check_holders=holders, ioctl=lambda *a: events.append("TIOCEXCL"), read_source=read)
    assert events[-2:] == ["read", "close"] and not made[0].requests


@pytest.mark.parametrize("event", ["stop", "expiry"])
def test_gripper_failed_physical_hold_is_attempted_only_once(event):
    import threading
    import time

    from ops.robot_pc import gripper as G

    first, hold = threading.Event(), threading.Event()
    writes = []

    def write(position, speed, force):
        writes.append(position)
        if position == 81:
            hold.set()
            raise OSError("lost hold acknowledgement")
        first.set()

    worker = G.GripperWorker(lambda: SimpleNamespace(write=write, close=lambda: None, read_status=lambda:
        G.GripperStatus(time.monotonic_ns(), 1, 1, 3, 0, 0, 230, 81, 0)), read_only=False)  # gOBJ 0: moving
    worker.set_closure(1, time.monotonic_ns() + 5_000_000_000)
    worker.start()
    try:
        assert first.wait(1)
        if event == "stop":
            worker.stop()
        else:
            with worker._target_lock:
                worker._expires_ns = time.monotonic_ns() - 1
        assert hold.wait(3)
        _wait_until(lambda: len(worker.hold_failures) == 1)
        if event != "stop":
            samples = len(worker.samples)
            _wait_until(lambda: len(worker.samples) >= samples + 3)
        assert writes == [230, 81] and not worker.holds
    finally:
        worker.stop()


@pytest.mark.parametrize("bad", ["crc", "length", "echo", "crc_reconnect"])
def test_rtu_three_consecutive_bad_replies_latch_worker_error_and_stop_execution(tmp_path, monkeypatch, bad):
    import threading
    import time

    from ops.robot_pc import gripper as G

    monkeypatch.setattr(devices, "scan", lambda **kw: {"recorders": []})
    serials, replies = [], []

    class Serial(FakeSerial):
        def __init__(self, *a, **kw):
            super().__init__(*a, **kw)
            serials.append(self)

        def read(self, n):
            replies.append(1)
            if bad == "crc_reconnect" and len(replies) == 3:
                raise OSError("transient serial failure, without a valid reply")
            reply = self.reply
            if bad in ("crc", "crc_reconnect"):
                return reply[:-1] + bytes([reply[-1] ^ 1])
            if bad == "length":
                return reply[:-1]
            body = reply[:2] + b"\x05" + reply[3:-2]  # wrong FC03 byte count, valid CRC
            return body + G.crc16_modbus(body)

    worker = G.GripperWorker(lambda: G.RobotiqRTU(serial_module=SimpleNamespace(Serial=Serial),
                            lock_path=str(tmp_path / "bad"), check_holders=lambda p: [],
                            ioctl=lambda *a: None, listen_s=0), read_only=True)
    worker.start()
    worker.join(2)
    assert not worker.is_alive() and worker.error_latched and "3 consecutive bad replies" in worker.error
    reconnect = bad == "crc_reconnect"
    assert len(serials) == (2 if reconnect else 1)
    assert sum(len(s.requests) for s in serials) == (4 if reconnect else 3) and not worker.healthy()
    # Even a fresh last-good status must not hide the latched protocol error from a policy session.
    worker.latest = G.GripperStatus(time.monotonic_ns(), 1, 1, 3, 3, 0, 3, 3, 0)
    released = threading.Event()
    session = _unit_session(sent=True, gripper=worker, link=SimpleNamespace(
        release=lambda *a, **kw: released.set(), peers_ok=lambda: True))
    assert not session._check_gripper()
    session.shutdown.finish()
    assert released.is_set() and "gripper unhealthy" in session.stop_reason


def test_rtu_default_quiet_listen_is_point_three_seconds_without_requests(tmp_path, monkeypatch):
    from ops.robot_pc import gripper as G

    monkeypatch.setattr(devices, "scan", lambda **kw: {"recorders": []})
    now = [0.0]
    serial = FakeSerial(exclusive=True, write_timeout=0.5)
    monkeypatch.setattr(G.time, "monotonic", lambda: now[0])

    def read(n):
        assert not serial.requests
        now[0] += 0.05
        return b""

    dev = G.RobotiqRTU(serial_module=SimpleNamespace(Serial=lambda *a, **kw: serial),
                       lock_path=str(tmp_path / "quiet"), check_holders=lambda p: [], ioctl=lambda *a: None,
                       read_source=read)
    try:
        assert now[0] >= 0.3 and not serial.requests
    finally:
        dev.close()


def test_rtu_bad_reply_streak_resets_on_valid_reply_and_checks_fc16_echo(tmp_path, monkeypatch):
    from ops.robot_pc import gripper as G

    monkeypatch.setattr(devices, "scan", lambda **kw: {"recorders": []})
    serial = FakeSerial(exclusive=True, write_timeout=0.5)
    corrupt = [True]

    def read(n):
        if not corrupt[0]:
            return serial.reply
        body = serial.reply[:3] + bytes([serial.reply[3] ^ 1]) + serial.reply[4:-2]
        return body + G.crc16_modbus(body)

    dev = G.RobotiqRTU(serial_module=SimpleNamespace(Serial=lambda *a, **kw: serial),
                       lock_path=str(tmp_path / "reset"), check_holders=lambda p: [], ioctl=lambda *a: None,
                       listen_s=0, read_source=read)
    try:
        for _ in range(2):
            with pytest.raises(G.BadReply):
                dev.write(230, 64, 0)
        assert dev.consecutive_bad_replies == 2
        corrupt[0] = False
        dev.read_status()
        assert dev.consecutive_bad_replies == 0
        corrupt[0] = True
        with pytest.raises(G.BadReply):
            dev.write(230, 64, 0)
        assert dev.consecutive_bad_replies == 1
    finally:
        dev.close()


def test_hardware_test_authorizations_cannot_run_a_policy_or_gripper_token_command_arm():
    from ops.robot_pc import controller, live

    hold = safety.authorize_hardware_test(test="hold", cli_flag=True, operator_start=True)
    grip = safety.authorize_hardware_test(test="gripper", cli_flag=True, operator_start=True)
    link = SimpleNamespace(send_flange_pose=lambda *a: pytest.fail("unauthorized arm send"))
    sink = live.LiveSink(hold, link)
    assert sink.authorization is hold
    with pytest.raises(safety.ExecutionRefused, match="policy authorization"):
        _unit_session(sent=True, sink=sink)
    with pytest.raises(safety.ExecutionRefused, match="gripper authorization"):
        sinks.LiveSink(grip, link)
    with pytest.raises(safety.ExecutionRefused, match="gripper authorization"):
        controller.ControllerLink(grip, zmq_module=FakeZmq(), robot_msg=fake_robot_msg())


def test_gripper_expiry_hold_skipped_when_stationary_or_in_contact():
    import threading
    import time

    from ops.robot_pc import gripper as G

    first_write = threading.Event()
    writes = []
    dev = SimpleNamespace(
        read_status=lambda: G.GripperStatus(time.monotonic_ns(), 1, 1, 3, 2, 0, 120, 110, 0),  # gOBJ 2: contact
        write=lambda *a: (writes.append(a), first_write.set()), close=lambda: None)
    worker = G.GripperWorker(lambda: dev, read_only=False)
    worker.set_closure(0.5, time.monotonic_ns() + 5_000_000_000)
    worker.start()
    try:
        assert first_write.wait(1)
        with worker._target_lock:
            worker._expires_ns = time.monotonic_ns() - 1
        _wait_until(lambda: worker._target is None)
        samples = len(worker.samples)
        _wait_until(lambda: len(worker.samples) >= samples + 3)
        assert len(writes) == 1 and not worker.holds  # expiry in contact: never relax the grasp
    finally:
        worker.stop()


def test_gripper_freeze_refuses_late_targets():
    from ops.robot_pc import gripper as G

    worker = G.GripperWorker(lambda: None, read_only=False)
    worker.freeze()
    worker.set_closure(0.4, 10**19)
    assert worker._target is None


# ----------------------------------------------------------------------------- LIVE multi-view (fakes only, no device)
def test_live_multiview_session_accepts_only_the_trained_camera_order_with_an_m5_helper():
    from ops.robot_pc import live

    cams = list(CAMS)
    spec = {"views": 3, "camera_ids": cams, "trained_camera_orders": [cams], "frame_period_s": 1 / 15}
    helper = SimpleNamespace(sets=SimpleNamespace(cameras=cams), pending=False)

    def session(cams, spec, m5):
        return live.LiveSession(client=SimpleNamespace(spec=spec), cameras=None, cams=cams, limits=None,
                                state=live.LiveState(SimpleNamespace(samples=[]), live.StaticGripper()),
                                executor=None, sink=sinks.DryRunSink(), live_m5=m5)

    refused = [(cams[::-1], spec, helper, "training order"),  # wrong order
               (cams[:2], spec, helper, "training order"),  # wrong count
               (cams, {**spec, "camera_ids": cams[::-1]}, helper, "training order"),  # bound order differs
               (cams, spec, None, "M5 helper"),  # no validated helper
               (cams, spec, SimpleNamespace(sets=SimpleNamespace(cameras=cams[::-1])), "M5 helper's cameras"),
               (["cam_a"], {"views": 1, "camera_ids": ["cam_a"]}, helper, "single-view")]
    for c, s, m, why in refused:
        with pytest.raises(safety.ExecutionRefused, match=why):
            session(c, s, m)
    s = session(cams, spec, helper)
    assert s.sync is not None and s.sync.cams == cams
    assert session(["cam_a"], {"views": 1, "camera_ids": ["cam_a"]}, None).sync is None  # single view unchanged


def test_sync_set_feed_is_exactly_once_in_order_and_steps_on_the_newest_set():
    from ops.robot_pc import syncsets

    got = []
    m5 = SimpleNamespace(add_robot=lambda t, q: got.append(("r", t)), add_gripper=lambda t, g: got.append(("g", t)))
    samples, grips = [], []

    class State:
        def rows(self, after, up_to):
            return ([(t, q) for t, q in samples if after < t <= up_to], [(t, g) for t, g in grips if after < t <= up_to])

    feed = syncsets.StateFeed(m5, State())
    feed.feed(10**12)  # nothing before the first frame
    samples += [(t, np.zeros(7)) for t in (100, 200)]
    grips += [(150, 3.0)]
    feed.start(1_000_000_050)  # rows from 1 s before the first frame (t > 50)
    feed.feed(250)
    samples += [(240, np.zeros(7)), (300, np.zeros(7))]  # 240: stamped before the last poll, appended after it
    grips.append((260, 3.0))
    feed.feed(400)
    feed.feed(400)
    assert got == [("r", 100), ("r", 200), ("g", 150), ("r", 240), ("r", 300), ("g", 260)]

    calls, steps, counts = [], [], collections.Counter()
    client = SimpleNamespace(add_sync_set=lambda ids, d: calls.append((dict(ids), d)))
    sets = [{"frame_ids": {"cam_a": k, "cam_b": k + 5}, "surface_disp_m": d, "observation_time_ns": 10 * k}
            for k, d in enumerate((0.001, np.inf, 0.004))]
    syncsets.step_sets(client, sets, ["cam_a", "cam_b"], lambda *a: steps.append(a), newest_only=True, counts=counts)
    assert calls == [({"cam_a": 0, "cam_b": 5}, 0.001), ({"cam_a": 1, "cam_b": 6}, None), ({"cam_a": 2, "cam_b": 7}, 0.004)]
    assert steps == [({"cam_a": 2, "cam_b": 7}, 20, "set:2", 0.004)] and counts["skipped_behind"] == 2
    steps.clear()
    syncsets.step_sets(client, sets, ["cam_a", "cam_b"], lambda *a: steps.append(a), newest_only=False, counts=counts)
    assert [s[2] for s in steps] == ["set:0", "set:1", "set:2"] and steps[1][3] is None
    assert counts["skipped_behind"] == 2


def test_live_sync_sets_poll_only_on_news_and_step_on_the_newest_final_set():
    from ops.robot_pc import syncsets

    cams = ["cam_a", "cam_b"]
    final = []
    polls, fed, sent = [], [], []

    class M5:
        sets = SimpleNamespace(cameras=cams)
        pending = False

        def add_frame(self, *a):
            fed.append(("frame", *a))

        def add_robot(self, t, q):
            fed.append(("robot", t))

        def add_gripper(self, t, g):
            fed.append(("gripper", t))

        def poll(self):
            polls.append(1)
            out, final[:] = list(final), []
            return out

    state = SimpleNamespace(rows=lambda a, b: ([(t, None) for t in (5, 10) if a < t <= b], [(7, 3.0)] if a < 7 <= b else []))
    m5 = M5()
    sync = syncsets.LiveSyncSets(SimpleNamespace(add_sync_set=lambda ids, d: sent.append((ids["cam_a"], d))), m5,
                                 state, cams)
    counts = collections.Counter()
    assert sync.newest_step(100, counts) is None and not polls  # nothing new: no poll
    sync.add_frame("cam_b", 3, 8)
    final += [{"frame_ids": {"cam_a": k, "cam_b": k}, "surface_disp_m": 0.001 * k, "observation_time_ns": k}
              for k in (1, 2, 3)]
    assert sync.newest_step(100, counts) == ({"cam_a": 3, "cam_b": 3}, 3, "set:3", 0.003)
    assert sent == [(1, 0.001), (2, 0.002), (3, 0.003)] and counts["skipped_behind"] == 2
    assert fed == [("frame", "cam_b", 3, 8), ("robot", 5), ("robot", 10), ("gripper", 7)]
    assert sync.newest_step(200, counts) is None and len(polls) == 1  # no new frame, nothing pending
    m5.pending = True  # a final set waits for member states: poll on every iteration
    assert sync.newest_step(300, counts) is None and len(polls) == 2
    with pytest.raises(ValueError, match="training order"):
        syncsets.LiveSyncSets(None, m5, state, cams[::-1])


class _RecordedAsLive:
    """The synthetic recording as LIVE captures: ONE frame per loop iteration in the replay's arrival order
    (``replay.frame_events``), each preprocessed exactly as ``PolicyClient.add_frame_native`` does in replay; the
    state stream reaches the frame's arrival (capture + 65 ms) when it is released."""

    def __init__(self, rec, cams, events, offset_ns, client, state):
        from s4d.rwlearn.deploy import preprocess_frame

        corr = client.choices["native_depth_correction"]
        cameras = RecordedCameras(rec, cams, depth=True)
        self.cams, self.state, self.queue = list(cams), state, []
        for t_rec, _, cam, idx in events:
            n = cameras.native(cam, idx)
            out = preprocess_frame(n["rgb"], depth=n["depth"], valid=n["valid"], K_native=cameras.K_native(cam),
                                   out_size=int(client.spec["image_hw"][0]), native_correction=corr[cam])
            t = int(t_rec) + offset_ns
            out.update(frame_id=int(n["seq"]), capture_ns=t, arrival_ns=t + replay.FRAME_ARRIVAL_NS,
                       preprocess_ns=(0, 0), native_hw=list(n["rgb"].shape[:2]),
                       metadata={"alignment": f"{n['alignment']} (recorded replay)",
                                 "depth_timestamp_ns": int(n["depth_timestamp_ns"]) + offset_ns,
                                 "depth_source": "native_realsense", "native_correction": corr[cam]})
            self.queue.append((cam, out))
        self.current = None

    def drain(self, cam):
        import time

        if cam == self.cams[0]:  # the first drain of a loop iteration releases the next frame
            self.current = self.queue.pop(0) if self.queue else None
            if self.current is not None:
                self.state.horizon = self.current[1]["capture_ns"] + replay.FRAME_ARRIVAL_NS
                self.current[1]["preprocess_ns"] = (time.monotonic_ns(),) * 2
        return [self.current[1]] if self.current is not None and self.current[0] == cam else []


class _RecordedLiveState:
    """``LiveState`` stand-in on the recorded robot / gripper stream (rows only up to the released frame's arrival)."""

    kind, label = "live", "recorded stream as a live source (test)"

    def __init__(self, rec, offset_ns):
        self.recorded, self.horizon = robot_state.RecordedRobotState(rec, offset_ns), -1
        self.gripper = SimpleNamespace(origin="measured (recorded)")
        self.reader = SimpleNamespace(samples=[])

    def bound_state(self, t_ns):
        return self.recorded.bound_state(t_ns)

    def rows(self, after_ns, up_to_ns):
        return self.recorded.rows(after_ns, min(up_to_ns, self.horizon))

    def latest(self):
        return None  # the 500 Hz executor idles: this test compares policy steps, not setpoints


def test_live_multiview_session_steps_equal_the_recorded_replay(policies, recording, toy_robot):
    """The live session's multi-view path (LiveSession + LatestFrames) on the recorded frames / state in arrival
    order == the recorded replay (realtime pace, the live rule): the same steps on the same latest end frames with the
    same M5 evidence declared, the same measured state, the same sync records (offsets, relative ages, displacement)
    and bitwise-equal fixture actions (same validation noise seeds)."""
    import time

    from ops.robot_pc import live

    rec = RecordedEpisode(recording)
    cams = list(CAMS)
    stop_at = len(rec.timestamps_ns(cams[0])) - 1  # no end-of-recording flush (a live stream has no end)
    ref0 = int(rec.timestamps_ns(cams[0])[0])
    with serve({"mvp3d-mv": policies["mvp3d_multiview"]}) as srv:
        def connect(sets):
            client = PolicyClient("127.0.0.1", srv.port, policy_id="mvp3d-mv", choices=multiview_choices(rec, cams),
                                  step_timeout_s=60, ttl_ms=60_000)
            client.connect()
            real = client.add_sync_set
            client.add_sync_set = lambda ids, d: (sets.append((dict(ids), d)), real(ids, d))[1]
            return client

        replay_sets, live_sets = [], []
        client = connect(replay_sets)
        cameras = RecordedCameras(rec, cams, depth=True)
        clock = replay.ReplayClock(ref0, "realtime")
        state = robot_state.RecordedRobotState(rec, clock.offset_ns)
        m5 = m5_live.LiveM5(toy_robot.motion, cams, 1 / 15, 20_000_000)
        lp = _loop(client, state, sinks.MockSink(), clock.now, max_age_ms=1e9)
        replay_rows = replay.run(client, lp, cameras, state, clock, cams=cams, count=stop_at, live_m5=m5,
                                 noise_seed=1)
        client.close()

        client = connect(live_sets)
        offset = time.monotonic_ns() - int(rec.timestamps_ns(cams[0])[-1]) - 5_000_000_000  # all in the past
        lstate = _RecordedLiveState(rec, offset)
        frames = _RecordedAsLive(rec, cams, [e[:4] for e in replay.frame_events(cameras, cams, 0, stop_at)], offset,
                                 client, lstate)
        lm5 = m5_live.LiveM5(toy_robot.motion, cams, 1 / 15, 20_000_000)
        session = live.LiveSession(client=client, cameras=frames, cams=cams, state=lstate,
                                   limits=ActionLimits(max_action_age_ms=1e9),
                                   executor=executor.HighRateExecutor(sinks.MockSink()), sink=sinks.DryRunSink(),
                                   live_m5=lm5)
        real_step, seeds = session.loop.step, iter(range(1, 10**6))
        session.loop.step = lambda *a, **k: real_step(*a, noise_seed=next(seeds), **k)  # replay's seed base + step
        live_rows = session.run(steps=len(replay_rows))
        client.close()
    assert session.stop_reason == "budget reached"
    assert replay_sets == live_sets and len(live_sets) >= 10
    assert {k: lp.counts[k] for k in syncsets.SYNC_COUNTS} == {k: session.loop.counts[k] for k in syncsets.SYNC_COUNTS}
    assert lp.counts["sync_steps"] == len(replay_rows) and lp.counts["sync_mismatch"] == 0
    assert dict(m5.counts) == dict(lm5.counts)
    ok = [r for r in replay_rows if r["status"] == "ok"]
    assert len(ok) >= 8 and len(live_rows) == len(replay_rows)
    keys = ("reference_id", "end_ids", "status", "code", "padded", "frames_sent", "measured_pose7",
            "action_norm", "flange_pose", "execution_flange_pose", "gripper_closure", "gripper_origin", "server_sync")
    sync_keys = ("rule", "frame_ids", "age_rel_ms", "offsets_ms", "max_offset_ms", "bound_ms", "mismatch",
                 "surface_disp_m", "motion_over_bound")
    for a, b in zip(replay_rows, live_rows):
        assert {k: a.get(k) for k in keys} == {k: b.get(k) for k in keys}, a["reference_id"]
        assert {k: a["sync"][k] for k in sync_keys} == {k: b["sync"][k] for k in sync_keys}, a["reference_id"]
        assert a["observation_time_ns"] - clock.offset_ns == b["observation_time_ns"] - offset
    assert all(r["state_source"] == "live" for r in live_rows)


class _RGBDCaptures:
    """Synthetic 15 Hz RGB-D captures of several free-running cameras (constant phase offsets, client-clock stamps),
    preprocessed like ``RealSenseRGBD.process`` (identity native correction), drained like LiveCameras."""

    def __init__(self, cams, phases_ms=(0.0, 9.0, -11.0), hw=(48, 64), out_size=32):
        from s4d.rwlearn.deploy.realsense import native_correction as nc

        self.cams, self.phase = list(cams), {c: int(p * 1e6) for c, p in zip(cams, phases_ms)}
        self.hw, self.out_size, self.k, self.t0, self.errors = hw, out_size, dict.fromkeys(cams, 0), None, {}
        self.K = np.array([[60.0, 0, hw[1] / 2], [0, 60.0, hw[0] / 2], [0, 0, 1]])
        self.corr = nc()

    def drain(self, cam):
        import time as _t

        from s4d.rwlearn.deploy import preprocess_frame

        now = _t.monotonic_ns()
        self.t0 = self.t0 or now + 50_000_000
        out = []
        while self.t0 + self.k[cam] * PERIOD_NS + self.phase[cam] <= now:
            k = self.k[cam]
            t = self.t0 + k * PERIOD_NS + self.phase[cam]
            rng = np.random.default_rng(1000 * self.cams.index(cam) + k)
            rgb = rng.integers(0, 256, (*self.hw, 3), dtype=np.uint8)
            depth = rng.uniform(0.4, 1.5, self.hw).astype(np.float32)
            depth[::7, ::5] = 0
            cap = preprocess_frame(rgb, depth=depth, valid=depth > 0, K_native=self.K, out_size=self.out_size,
                                   native_correction=self.corr)
            cap.update(frame_id=50 + k, capture_ns=t, arrival_ns=now, preprocess_ns=(now, now), native_hw=list(self.hw),
                       metadata={"depth_source": "native_realsense", "native_correction": self.corr})
            cap["K"] = np.asarray(cap["K"]).tolist()
            out.append(cap)
            self.k[cam] += 1
        return out


def _rgbd_choices(captures):
    return {"native_hw": {c: list(captures.hw) for c in captures.cams},
            "calibration": {c: {"K_native": captures.K.tolist(), "c2w": np.eye(4).tolist()} for c in captures.cams},
            "native_depth_correction": {c: captures.corr for c in captures.cams}}


def test_live_multiview_session_dry_run_full_chain(policies, toy_robot):
    """LiveSession on 3 free-running fake RGB-D cameras, the read-only state reader at 1 kHz + the labelled static
    gripper (dry run), the live M5 helper, fixture multi-view 3D-MVP over loopback, validation, the 500 Hz executor ->
    DryRunSink: one step per new reference frame on every camera's latest frame (sync record per row; M5 informational
    when the member states are final), commands executed (never sent)."""
    import threading as _th
    import time as _t

    from ops.robot_pc import live

    reader = sc.LiveStateReader(approved=True, zmq_module=_SubOnly, robot_msg=SimpleNamespace())
    stop_feed, seq, next_t = _th.Event(), [0], [_t.monotonic_ns() - 1_500_000_000]

    def feed():
        while not stop_feed.is_set():
            now = _t.monotonic_ns()
            while next_t[0] <= now:
                seq[0] += 1
                reader.samples.append({"t_ns": next_t[0], "seq": seq[0], "q": np.full(7, 0.1), "dq": np.zeros(7),
                                       "tau": np.zeros(7), "flange_pose7": np.array(POSE),
                                       "ee_pose7": sc.ee_from_flange(POSE), "signature": seq[0].to_bytes(8, "little")})
                next_t[0] += 1_000_000
            _t.sleep(0.001)

    feeder = _th.Thread(target=feed, name="fake-state-feeder", daemon=True)
    feeder.start()
    calls = collections.Counter()
    events = []
    rec = SimpleNamespace(**{n: (lambda *a, n=n, **k: calls.update([n])) for n in
                             ("frame", "observation", "robot_state", "gripper", "step", "gripper_command", "executed")},
                          event=lambda t, text: events.append(text))
    sink = sinks.DryRunSink()
    captures = _RGBDCaptures(CAMS)
    try:
        with serve({"mvp3d-mv": policies["mvp3d_multiview"]}) as srv:
            client = PolicyClient("127.0.0.1", srv.port, policy_id="mvp3d-mv", step_timeout_s=60, ttl_ms=60_000,
                                  choices=_rgbd_choices(captures))
            client.connect()
            sets = []
            real = client.add_sync_set
            client.add_sync_set = lambda ids, d: (sets.append((dict(ids), d)), real(ids, d))[1]
            m5 = m5_live.LiveM5(toy_robot.motion, CAMS, 1 / 15, 20_000_000)
            ex = executor.HighRateExecutor(sinks.MockSink(), state_timeout_s=2.0)  # in-process server holds the GIL
            session = live.LiveSession(client=client, cameras=captures, cams=list(CAMS),
                                       state=live.LiveState(reader, live.StaticGripper()),
                                       limits=ActionLimits(max_action_age_ms=1e5), executor=ex, sink=sink,
                                       recorder=rec, sent=False, live_m5=m5)
            rows = session.run(steps=10)
            client.close()
    finally:
        stop_feed.set()
        feeder.join(5)
        reader.close()
    assert session.stop_reason == "budget reached" and len(rows) == 10, session.stop_reason
    assert all(r["reference_id"].startswith("latest:") and set(r["end_ids"]) == set(CAMS) for r in rows)
    assert all(r["sync"]["frame_ids"] == r["end_ids"] and r["sync"]["max_offset_ms"] < 33.4 for r in rows)
    assert all(r["sync"]["surface_disp_m"] in (None, 0.0) for r in rows)  # the robot does not move
    ok = [r for r in rows if r["status"] == "ok"]
    assert len(ok) >= 8 and all(r["gripper_origin"].startswith("static") for r in ok), [r.get("code") for r in rows]
    stepped = [r["end_ids"] for r in rows]
    declared = [r["end_ids"] for r in rows if r["sync"]["surface_disp_m"] is not None]
    assert [ids for ids, _ in sets] == declared and m5.counts["within_bound"] == len(declared)
    assert m5.counts["sets"] + m5.counts["not_ready"] == len(rows)
    assert all(a["cam_a"] < b["cam_a"] for a, b in zip(stepped, stepped[1:]))  # newest frame, never an older one
    assert calls["frame"] == 0 and calls["observation"] >= 30 and calls["step"] == 10  # no native_rgb: no frame rows
    assert ex.counts["accepted"] > 0 and calls["executed"] > 0 and len(sink.items) == calls["executed"]
    sync_event = [json.loads(e[5:]) for e in events if e.startswith("sync ")]
    assert sync_event and sync_event[0]["sync_steps"] == 10 and sync_event[0]["sync_mismatch"] == 0
    assert sync_event[0]["m5"] == dict(m5.counts) and sync_event[0]["asset_sha256"]


def test_live_multiview_sync_refusals_are_answers_not_route_failures():
    """A multi-view answer without an action (an older server's sync refusal, a camera's own incomplete history)
    never stops the session as a server failure; other errors still do (single-view rule unchanged)."""
    from ops.robot_pc import live

    cams = list(CAMS)
    spec = {"views": 3, "camera_ids": cams, "trained_camera_orders": [cams], "frame_period_s": 1 / 15}
    codes = iter(["timeout"] * 4 + ["unsynchronized"] * 6 + ["incomplete_history"] * 3 + ["timeout"] * 5)

    class Sync:
        def __init__(self):
            self.k = 0

        def add_frame(self, *a):
            pass

        def newest_step(self, now, flush=False):
            self.k += 1
            return ({c: self.k for c in cams}, now, f"latest:{self.k}",
                    {"mismatch": False, "motion_over_bound": True, "offsets_ms": {}, "surface_disp_m": 0.009})

    s = live.LiveSession(client=SimpleNamespace(spec=spec, add_capture=lambda *a, **k: None),
                         cameras=SimpleNamespace(drain=lambda cam: []), cams=cams, state=live.LiveState(
                             SimpleNamespace(samples=[]), live.StaticGripper()),
                         limits=ActionLimits(), executor=executor.HighRateExecutor(sinks.MockSink()),
                         sink=sinks.DryRunSink(), live_m5=SimpleNamespace(sets=SimpleNamespace(cameras=cams)))
    s.sync = Sync()
    s.loop.step = lambda end_ids, t, ref, acquire_ms=None: {"reference_id": ref, "status": "error",
                                                            "code": next(codes), "t_step_ns": t}
    rows = s.run(steps=30)  # 4 timeouts, then answers (the route works again), then 5 timeouts in a row
    assert [r["code"] for r in rows] == ["timeout"] * 4 + ["unsynchronized"] * 6 + ["incomplete_history"] * 3 + \
        ["timeout"] * 5
    assert s.stop_reason.startswith("policy server: 5 consecutive steps") and s.loop.counts["sync_refused"] == 9
    assert all(r["sync"]["surface_disp_m"] == 0.009 for r in rows)


def test_live_multiview_session_refuses_a_missing_camera_by_its_idle_timeout():
    """Two cameras keep delivering, the third never does: no step is ever taken on a partial camera set and the
    unchanged idle timeout stops the session (missing-camera refusal kept by the 2026-11-12 relaxation)."""
    from ops.robot_pc import live

    cams = list(CAMS)
    spec = {"views": 3, "camera_ids": cams, "trained_camera_orders": [cams], "frame_period_s": 1 / 15}
    frames = iter(range(10**6))

    def drain(cam):
        if cam == cams[-1]:
            return []
        k = next(frames)
        return [{"frame_id": k, "capture_ns": 10**9 + k * PERIOD_NS, "arrival_ns": 10**9 + k, "rgb": None}]

    steps = []
    s = live.LiveSession(client=SimpleNamespace(spec=spec, add_capture=lambda *a, **k: None),
                         cameras=SimpleNamespace(drain=drain), cams=cams,
                         state=live.LiveState(SimpleNamespace(samples=[]), live.StaticGripper()),
                         limits=ActionLimits(), executor=executor.HighRateExecutor(sinks.MockSink()),
                         sink=sinks.DryRunSink(), idle_timeout_s=0.05,
                         live_m5=SimpleNamespace(sets=SimpleNamespace(cameras=cams), counts=collections.Counter()))
    s.loop.step = lambda *a, **k: steps.append(a) or {}
    assert s.run(duration_s=5) == [] and not steps
    assert s.stop_reason == f"cameras {[cams[-1]]} sent no frame for 0.05 s"
    assert s.sync.latest[cams[-1]] is None and s.sync.latest[cams[0]] is not None


def test_infer_refuses_a_multiview_policy_without_m5_before_touching_any_device(monkeypatch, tmp_path):
    """infer.py resolves the policy contract (cameras, live M5 helper) before the device guard, the state reader or
    the gripper: a multi-view policy without its M5 asset / with a wrong camera order is refused with nothing opened."""
    from ops.robot_pc import gripper as G
    from ops.robot_pc import tunnel
    from scripts.robot_pc import infer

    touched = []
    cams = ["135122070988", "135122076054", "135122073201"]
    spec = {"views": 3, "frame_period_s": 1 / 15, "trained_camera_orders": [cams], "modalities": ["rgb", "depth"]}
    monkeypatch.setattr(infer, "say", lambda m: None)
    monkeypatch.setattr(cli, "load_config", lambda args: {"root": str(tmp_path), "tasks": {"t": {}}})
    monkeypatch.setattr(tunnel, "welcome", lambda port: {"fake": True})
    monkeypatch.setattr(cli, "choose_policy", lambda *a: ("mv", "t", spec, {"execution_eligible": True}))
    monkeypatch.setattr(devices, "require_free", lambda **k: touched.append("devices"))
    monkeypatch.setattr(sc, "LiveStateReader", lambda **k: touched.append("reader"))
    monkeypatch.setattr(G, "GripperWorker", lambda *a, **k: touched.append("gripper"))
    for extra in ([], ["--cameras", "right,middle,left"]):
        assert infer.main(["--policy", "mv", "--dry-run", *extra]) == 3
    assert not touched


def test_live_multiview_dry_run_with_static_state(policies, toy_robot, tmp_path):
    """dry_run.py's loop (dryrun.run, --state static) on 3 fake RGB-D cameras: one step per reference frame on every
    camera's latest frame with its sync record (static state: M5 0 m where final), nothing sent."""
    captures = _RGBDCaptures(CAMS)
    with serve({"mvp3d-mv": policies["mvp3d_multiview"]}) as srv:
        client = PolicyClient("127.0.0.1", srv.port, policy_id="mvp3d-mv", step_timeout_s=60, ttl_ms=60_000,
                              choices=_rgbd_choices(captures))
        client.connect()
        log = tmp_path / "dryrun.jsonl"
        sink = sinks.DryRunSink(log)
        state = robot_state.StaticLatencyState(POSE, np.zeros(7), 3.0)
        m5 = m5_live.LiveM5(toy_robot.motion, CAMS, 1 / 15, 20_000_000)
        lp = _loop(client, state, sink, max_age_ms=1e5)
        try:
            rows = dryrun.run(client, lp, captures, state, cams=list(CAMS), steps=8, live_m5=m5)
        finally:
            client.close()
            sink.close()
    assert len(rows) == 8 and all(r["sync"]["surface_disp_m"] in (None, 0.0) and
                                  r["state_source"] == "static_latency_only" for r in rows)
    assert sum(r["status"] == "ok" for r in rows) >= 6 and m5.counts["sets"] + m5.counts["not_ready"] == 8
    assert lp.counts["sync_steps"] == 8 and lp.counts["sync_mismatch"] == 0
    assert all(set(r["sync"]["age_ms"]) == set(CAMS) and min(r["sync"]["age_ms"].values()) >= 0 for r in rows)
    assert all(not x["sent_to_robot"] for x in map(json.loads, log.read_text().splitlines()))


def test_multiview_m5_is_built_before_devices_or_refused(toy_robot, tmp_path, monkeypatch):
    """config.session_m5: the helper on all trained cameras in training order, refused when an asset or the task
    entry is missing / mismatching; the launchers refuse a multi-view policy before any device opens."""
    import importlib.util

    from scripts.robot_pc import infer

    cams = list(CAMS)
    spec = {"views": 3, "frame_period_s": 1 / 15, "trained_camera_orders": [cams], "modalities": ["rgb"]}
    cfg = {"root": str(tmp_path), "assets": {"m5_surface": str(toy_robot.asset), "urdf": str(toy_robot.urdf)},
           "tasks": {"t": {"m5_max_sync_ns": 30_000_000}}}
    m5 = C.session_m5(cfg, "t", spec, cams)
    assert m5.sets.cameras == cams and m5.timeline.max_sync_ns == 30_000_000
    assert C.session_m5(cfg, "t", {**spec, "views": 1}, cams[:1]) is None
    bad = [(cfg, cams[::-1], "training order"), ({**cfg, "assets": {}}, cams, "KeyError"),
           ({**cfg, "assets": {**cfg["assets"], "urdf": str(tmp_path / "missing.urdf")}}, cams, "no validated"),
           ({**cfg, "tasks": {"t": {}}}, cams, "m5_max_sync_ns")]
    for c, order, why in bad:
        with pytest.raises(C.M5Unavailable, match=why):
            C.session_m5(c, "t", spec, order)
    with pytest.raises(safety.ExecutionRefused, match="multi-view policy"):
        infer.multiview_m5({**cfg, "assets": {}}, "t", spec, cams)
    names = {"left": "cam_a", "middle": "cam_b", "right": "cam_c"}
    assert infer.resolve_cameras(spec, names, None) == cams
    assert infer.resolve_cameras(spec, names, "left,middle,right") == cams
    with pytest.raises(safety.ExecutionRefused, match="training order"):
        infer.resolve_cameras(spec, names, "right,middle,left")
    with pytest.raises(safety.ExecutionRefused, match="training order"):
        infer.resolve_cameras(spec, names, "left,middle")

    sp = importlib.util.spec_from_file_location("rob_dry_run", REPO / "scripts/robot_pc/dry_run.py")
    mod = importlib.util.module_from_spec(sp)
    sp.loader.exec_module(mod)
    opened = []
    monkeypatch.setattr(cli, "load_config", lambda args: {**cfg, "assets": {},
                                                          "static_latency_state": {"pose7": POSE, "q7": [0] * 7,
                                                                                   "gripper_register": 3.0}})
    monkeypatch.setattr(cli, "choose_policy", lambda *a: ("mv", "t", spec, {}))
    monkeypatch.setattr("ops.robot_pc.cameras.LiveCameras", lambda *a, **k: opened.append(1))
    assert mod.main(["--state", "static", "--steps", "3", "--log", str(tmp_path / "x.jsonl")]) == 3
    assert not opened
