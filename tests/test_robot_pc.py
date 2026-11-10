"""Robot-PC package (ops/robot_pc, scripts/robot_pc, s4d/rwlearn/deploy/m5_live) on CPU: labelled FIXTURE policies
over loopback, a synthetic ``panda_teleop_episode`` v2 recording, a synthetic URDF for the M5 helper. No skips, no
robot, no camera.

Safety is tested, not assumed: the manual execution mode needs three explicit conditions and still raises Unbound;
LiveSink / LiveRobotState can never be built by setup, replay, dry run or bench (runtime spies + a source scan); the
only sockets opened during replay / dry run go to the policy server.
"""

from __future__ import annotations

import ast
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
from ops.robot_pc import cli, devices, dryrun, executor, loop, poses, replay, robot_state, safety, selection, sinks
from ops.robot_pc import config as C
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
  <link name="finger"><collision><origin xyz="0 0.02 0.03"/><geometry><box size="0.02 0.01 0.05"/></geometry></collision></link>
  <link name="finger2"><collision><origin xyz="0 -0.02 0.03"/><geometry><box size="0.02 0.01 0.05"/></geometry></collision></link>
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
    """Records every LiveSink / LiveRobotState construction and every outgoing TCP connection."""
    seen = {"live_sink": 0, "live_state": 0, "connect": []}
    real_sink, real_state = sinks.LiveSink.__init__, robot_state.LiveRobotState.__init__
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
    monkeypatch.setattr(robot_state.LiveRobotState, "__init__", state_init)
    monkeypatch.setattr(socket, "create_connection", create)
    return seen


def multiview_choices(rec, cams):
    return {"native_hw": {c: list(rec.native_hw(c)) for c in cams},
            "calibration": {c: {"K_native": rec.K_native(c).tolist(), "c2w": np.eye(4).tolist()} for c in cams},
            "native_depth_correction": {c: native_correction() for c in cams}}


def test_config_file_and_limits(tmp_path):
    cfg = C.load(root=tmp_path)
    assert cfg["execution"] == {"armed": False, "controller_binding": "UNBOUND"}
    lim = C.limits(cfg, "mug-hanging")
    assert lim.max_translation_m == 0.04 and lim.workspace_max_m == (0.699, 0.243, 0.831)
    assert C.path(cfg, "assets/x") == tmp_path / "assets/x"
    bad = tmp_path / "bad.yaml"
    bad.write_text(yaml.safe_dump({**yaml.safe_load(C.DEFAULT_PATH.read_text()), "extra": 1}))
    with pytest.raises(ValueError, match="unknown robot-PC config sections"):
        C.load(bad)
    armed = yaml.safe_load(C.DEFAULT_PATH.read_text())
    armed["execution"]["armed"] = "yes"
    bad.write_text(yaml.safe_dump(armed))
    with pytest.raises(ValueError, match="boolean"):
        C.load(bad)


# ------------------------------------------------------------------------------------------------ imports
def test_robot_pc_stack_imports_with_torch_blocked():
    code = ("import sys; sys.modules['torch'] = None\n"
            "import ops.policy_serving.client, ops.policy_serving.protocol, ops.policy_serving.bench\n"
            "import ops.robot_pc, ops.robot_pc.safety, ops.robot_pc.sinks, ops.robot_pc.validation, ops.robot_pc.poses\n"
            "import ops.robot_pc.executor, ops.robot_pc.robot_state, ops.robot_pc.cameras, ops.robot_pc.devices\n"
            "import ops.robot_pc.selection, ops.robot_pc.loop, ops.robot_pc.replay, ops.robot_pc.dryrun\n"
            "import ops.robot_pc.cli, ops.robot_pc.config\n"
            "import s4d.rwlearn.deploy.m5_live, s4d.rwlearn.deploy.replay, s4d.rwlearn.deploy.realsense\n"
            "from s4d.realworld.train_store import mutual_sync, set_surface_motion\n"
            "from s4d.realworld.robotics.model import URDFModel, RobotSurface\n"
            "from s4d.rwlearn.deploy import preprocess_frame, InputSpec, HistoryWindow\n"
            "assert sys.modules['torch'] is None\n")
    subprocess.run([sys.executable, "-c", code], check=True, cwd=REPO,
                   env={**os.environ, "CUDA_VISIBLE_DEVICES": "", "PYTHONPATH": str(REPO)})


# ------------------------------------------------------------------------------------------------ safety
def test_execution_mode_needs_all_three_explicit_conditions():
    full = {"cli_execute": True, "typed_confirmation": safety.CONFIRMATION, "config_armed": True}
    for missing in full:
        with pytest.raises(safety.ExecutionRefused, match="NOT selected"):
            safety.authorize_execution(**{**full, missing: {"cli_execute": False, "typed_confirmation": "yes",
                                                            "config_armed": False}[missing]})
    with pytest.raises(safety.ExecutionRefused, match="only issued by"):
        safety.ExecutionAuthorization(safety.Mode.EXECUTE, 0)
    auth = safety.authorize_execution(**full)
    with pytest.raises(safety.Unbound, match="UNBOUND"):
        sinks.LiveSink(auth)
    with pytest.raises(safety.ExecutionRefused):
        sinks.LiveSink(None)
    with pytest.raises(safety.Unbound, match="UNBOUND"):
        robot_state.LiveRobotState()
    assert isinstance(sinks.make_sink(safety.Mode.REPLAY), sinks.MockSink)
    assert isinstance(sinks.make_sink(safety.Mode.DRY_RUN), sinks.DryRunSink)


def test_execute_script_refuses_even_fully_authorized(tmp_path, spies, capsys, monkeypatch):
    import importlib.util

    spec = importlib.util.spec_from_file_location("rob_execute", REPO / "scripts/robot_pc/execute.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    armed = yaml.safe_load(C.DEFAULT_PATH.read_text())
    armed["execution"]["armed"] = True
    cfg = tmp_path / "armed.yaml"
    cfg.write_text(yaml.safe_dump(armed))
    monkeypatch.setattr(socket.socket, "connect", lambda *a: (_ for _ in ()).throw(AssertionError("socket used")))
    assert mod.main(["--config", str(cfg), "--execute"], typed=safety.CONFIRMATION) == 3
    err = capsys.readouterr().err
    assert "REFUSED" in err and "UNBOUND" in err and spies["live_sink"] == 1 and not spies["connect"]
    assert mod.main(["--config", str(C.DEFAULT_PATH)], typed=None) == 3 and spies["live_sink"] == 1
    assert "missing: the --execute flag" in capsys.readouterr().err


def test_no_robot_io_in_the_robot_pc_sources():
    """Static guard: no robot/controller transport is imported anywhere in the package or its scripts, and only
    execute.py (refusing) and the sink factory refer to the live sink."""
    banned = {"zmq", "serial", "rclpy", "rospy", "franky", "panda_py", "frankx", "pymodbus", "minimalmodbus"}
    files = sorted((REPO / "ops/robot_pc").glob("*.py")) + sorted((REPO / "scripts/robot_pc").glob("*.py"))
    for path in files:
        tree = ast.parse(path.read_text())
        for node in ast.walk(tree):
            names = [a.name for a in node.names] if isinstance(node, ast.Import) else \
                [node.module or ""] if isinstance(node, ast.ImportFrom) else []
            assert not {n.split(".")[0] for n in names} & banned, (path, names)
        text = path.read_text()
        if path.name not in ("sinks.py", "execute.py", "safety.py"):
            assert "LiveSink(" not in text and "authorize_execution(cli_execute=True" not in text, path
    for name in ("replay.py", "dry_run.py", "bench.py", "policies.py", "selfcheck.py", "setup.sh"):
        text = (REPO / "scripts/robot_pc" / name).read_text()
        assert "LiveSink" not in text and "--execute" not in text, name


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
        "translation": (_result(_outputs(req, far)), now),
        "rotation": (_result(_outputs(req, rot)), now),
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
    box = ActionLimits(workspace_min_m=(0.5, -1, 0), workspace_max_m=(1, 1, 1))
    with pytest.raises(ActionRejected, match="workspace"):
        validate_action(_result(_outputs(req)), req, box, session_id="s1", now_ns=now)
    with pytest.raises(ValueError, match="increasing"):
        ActionLimits.from_config({"workspace_min_m": [1, 0, 0], "workspace_max_m": [0, 1, 1]})


def Rquat(angle: float):
    """POSE's orientation (x 180 deg) rotated by ``angle`` about the base z axis, xyzw."""
    qz = np.array([0, 0, np.sin(angle / 2), np.cos(angle / 2)])
    x, y, z, w = qz
    a = np.array(POSE[3:])
    return [w * a[0] + x * a[3] + y * a[2] - z * a[1], w * a[1] - x * a[2] + y * a[3] + z * a[0],
            w * a[2] + x * a[1] - y * a[0] + z * a[3], w * a[3] - x * a[0] - y * a[1] - z * a[2]]


def test_pose_limits_and_high_rate_executor():
    start = np.array(POSE)
    goal = np.array([POSE[0] + 0.1, *POSE[1:3], *Rquat(np.deg2rad(40))])
    out = poses.toward(start, goal, 0.01, np.deg2rad(5))
    assert abs(np.linalg.norm(out[:3] - start[:3]) - 0.01) < 1e-12
    assert abs(poses.quat_angle(out[3:], start[3:]) - np.deg2rad(5)) < 1e-9
    np.testing.assert_allclose(poses.toward(start, start, 0.01, 0.1), start)
    sink = sinks.MockSink()
    ex = executor.HighRateExecutor(sink, rate_hz=500, max_linear_speed=0.3, max_angular_speed=1.5)
    assert ex.tick(0, POSE) is None and not sink.items  # nothing before the first command
    from ops.robot_pc.validation import Command

    cmd = Command("r1", "s1", 1_000, 1_000 + 200_000_000, POSE, tuple(goal.tolist()), 0.5, 117, 0.1, 40.0)
    assert ex.accept(cmd) and not ex.accept(dataclasses.replace(cmd, reference_id="r0", observation_time_ns=999))
    sp = ex.tick(2_000, POSE)
    assert sp.kind == "track" and abs(np.linalg.norm(np.array(sp.pose7[:3]) - start[:3]) - 0.3 / 500) < 1e-12
    for k in range(200):  # the setpoint never leaves the lead envelope around the (fixed) measured pose
        sp = ex.tick(3_000 + k, POSE)
    assert np.linalg.norm(np.array(sp.pose7[:3]) - start[:3]) <= 0.04 + 1e-12
    assert poses.quat_angle(sp.pose7[3:], start[3:]) <= np.deg2rad(20) + 1e-9
    held = ex.tick(cmd.expires_ns + 1, [POSE[0] + 0.001, *POSE[1:]])
    assert held.kind == "hold" and held.pose7[0] == POSE[0] + 0.001 and ex.counts["refused_older"] == 1


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


# ------------------------------------------------------------------------------------------------ replay / dry run
def _loop(client, state, sink, clock=None):
    return loop.PolicyLoop(client, state, ActionLimits(), sink, **({} if clock is None else {"clock": clock}))


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
        rows = replay.run(client, _loop(client, state, sink, clock.now), cameras, state, clock, cams=cams,
                          live_m5=live, noise_seed=1)
        client.close()
    ok = [r for r in rows if r["status"] == "ok"]
    assert len(ok) >= 10 and len(sink.items) == len(ok), [r.get("code") for r in rows]
    assert all(r["surface_disp_m"] is not None and r["surface_disp_m"] <= 0.005 for r in rows)
    assert live.counts["sets"] == len(rows)
    assert spies["live_sink"] == 0 and set(spies["connect"]) == {("127.0.0.1", srv.port)}


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

    def __init__(self, serial, *, native_hw, fps, depth, correction, rs_module):
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
    with pytest.raises(selection.SelectionError, match="0 deployable"):  # FIXTURE policies are never auto-selected
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
                 "pyrealsense2-2.59.1.11053"):
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
    validation.write_text(json.dumps({"pass": True, "header": {"format": "x"}, "episodes": [{"episode": "e", "pass": True}]}))
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
    assert manifest["pins"]["numpy"] == "numpy==1.26.4" and manifest["safety"].startswith("dry run")
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
