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
from ops.robot_pc import switch_controller as sc
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
    assert cfg["execution"] == {"armed": False, "controller_binding": "message-format-only"}
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
    full = {"cli_execute": True, "typed_confirmation": safety.CONFIRMATION, "config_armed": True,
            "policy_info": {"execution_eligible": True}}
    for missing in full:
        with pytest.raises(safety.ExecutionRefused, match="NOT selected"):
            safety.authorize_execution(**{**full, missing: {"cli_execute": False, "typed_confirmation": "yes",
                                                            "config_armed": False, "policy_info": None}[missing]})
    for info in ({}, {"execution_eligible": False}, {"execution_eligible": "true"}, {"execution_eligible": 1}):
        with pytest.raises(safety.ExecutionRefused, match="execution_eligible"):  # DynaMo / missing: fail closed
            safety.authorize_execution(**{**full, "policy_info": info})
    with pytest.raises(safety.ExecutionRefused, match="only issued by"):
        safety.ExecutionAuthorization(safety.Mode.EXECUTE, 0)
    auth = safety.authorize_execution(**full)
    assert safety.EXECUTION_PERMITTED is False
    with pytest.raises(safety.ExecutionRefused, match="never operated"):
        sinks.LiveSink(auth, publisher=FakePublisher(), robot_msg=fake_robot_msg())
    with pytest.raises(safety.ExecutionRefused):
        sinks.LiveSink(None)
    with pytest.raises(safety.Unbound, match="not approved"):
        sc.LiveStateReader()
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
    catalog = {"policies": {"reviwo-drawer-open-s0": {"info": {"execution_eligible": True}},
                            "dynamo-drawer-open-s0": {"info": {"execution_eligible": False}}}}
    asked = []
    monkeypatch.setattr(cli, "welcome", lambda cfg, args: asked.append(1) or catalog)
    # this revision: refused before any network activity, even fully authorized and eligible
    assert mod.main(["--config", str(cfg), "--execute", "--policy", "reviwo-drawer-open-s0"],
                    typed=safety.CONFIRMATION) == 3
    err = capsys.readouterr().err
    assert "REFUSED" in err and "never operated" in err and not asked and spies["live_sink"] == 0
    # a later directive's switch (test only): eligibility is then enforced, and still no transport exists
    monkeypatch.setattr(safety, "EXECUTION_PERMITTED", True)
    assert mod.main(["--config", str(cfg), "--execute", "--policy", "dynamo-drawer-open-s0"],
                    typed=safety.CONFIRMATION) == 3
    assert "execution_eligible" in capsys.readouterr().err and spies["live_sink"] == 0
    assert mod.main(["--config", str(cfg), "--execute", "--policy", "reviwo-drawer-open-s0"],
                    typed=safety.CONFIRMATION) == 3
    assert "no" in capsys.readouterr().err and spies["live_sink"] == 1 and not spies["connect"]
    assert mod.main(["--config", str(C.DEFAULT_PATH)], typed=None) == 3 and spies["live_sink"] == 1
    assert "missing: the --execute flag" in capsys.readouterr().err


def test_no_robot_io_in_the_robot_pc_sources():
    """Static guard: no robot/controller transport is imported anywhere in the package or its scripts except the
    read-only RobotState SUB of switch_controller.py (which never binds, never creates a PUB, never sends), and only
    execute.py (refusing) and the sink factory refer to the live sink."""
    banned = {"zmq", "serial", "rclpy", "rospy", "franky", "panda_py", "frankx", "pymodbus", "minimalmodbus"}
    files = sorted((REPO / "ops/robot_pc").glob("*.py")) + sorted((REPO / "scripts/robot_pc").glob("*.py"))
    for path in files:
        tree = ast.parse(path.read_text())
        for node in ast.walk(tree):
            names = [a.name for a in node.names] if isinstance(node, ast.Import) else \
                [node.module or ""] if isinstance(node, ast.ImportFrom) else []
            found = {n.split(".")[0] for n in names} & banned
            assert not found or (path.name == "switch_controller.py" and found == {"zmq"}), (path, names)
        text = path.read_text()
        if path.name not in ("sinks.py", "execute.py", "safety.py"):
            assert "LiveSink(" not in text and "authorize_execution(cli_execute=True" not in text, path
    text = (REPO / "ops/robot_pc/switch_controller.py").read_text()
    assert ".bind(" not in text and ".PUB" not in text and ".send(" not in text and "request_mode" not in text
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


def test_executor_enforces_the_collector_speed_acceleration_and_lead_limits():
    """RateLimiter port (teleop.py:154-190): |v| <= 0.3 m/s, |dv| <= 4 m/s^2 dt, |w| <= 1.5 rad/s, |dw| <= 15 rad/s^2 dt;
    the command stays within the lead envelope of the measured pose; no overshoot at the goal."""
    sink = sinks.MockSink()
    ex = executor.HighRateExecutor(sink)
    goal = [POSE[0] + 0.03, POSE[1] - 0.01, POSE[2], *Rquat(np.deg2rad(15))]
    assert ex.tick(0, POSE, state_t_ns=0) is None and not sink.items  # nothing before the first command
    assert ex.accept(_cmd(goal, age_ms=1e6))
    dt_ns, t, prev_p, prev_v, prev_q, prev_w = 2_000_000, 1_000_000, np.array(POSE[:3]), np.zeros(3), \
        np.array(POSE[3:]), 0.0
    for k in range(600):
        t += dt_ns
        sp = ex.tick(t, POSE, state_t_ns=t - 500_000, signature=bytes([k % 256, k // 256]))
        p, q = np.array(sp.pose7[:3]), np.array(sp.pose7[3:])
        v = (p - prev_p) / (dt_ns / 1e9)
        w = poses.quat_angle(prev_q, q) / (dt_ns / 1e9)
        assert np.linalg.norm(v) <= 0.3 + 1e-9 and np.linalg.norm(v - prev_v) <= 4.0 * dt_ns / 1e9 + 1e-9, k
        assert w <= 1.5 + 1e-6 and abs(w - prev_w) <= 15.0 * dt_ns / 1e9 + 1e-6, k
        assert np.linalg.norm(p - np.array(POSE[:3])) <= 0.04 + 1e-12
        assert poses.quat_angle(q, POSE[3:]) <= np.deg2rad(20) + 1e-9
        prev_p, prev_v, prev_q, prev_w = p, v, q, w
    np.testing.assert_allclose(prev_p, goal[:3], atol=1e-6)  # reached, braking without overshoot
    assert poses.quat_angle(prev_q, goal[3:]) < 1e-5
    far = executor.HighRateExecutor(sinks.MockSink())
    far.accept(_cmd([POSE[0] + 0.15, *POSE[1:]], age_ms=1e6))
    t = 0
    for k in range(500):
        t += dt_ns
        sp = far.tick(t, POSE, state_t_ns=t, signature=bytes([k % 256, k // 256]))
    assert abs(np.linalg.norm(np.array(sp.pose7[:3]) - np.array(POSE[:3])) - 0.04) < 1e-9  # leashed at the lead


def test_executor_holds_on_expiry_and_latches_a_stop_on_stale_or_frozen_state():
    sink = sinks.MockSink()
    ex = executor.HighRateExecutor(sink)
    ex.accept(_cmd([POSE[0] + 0.02, *POSE[1:]], t_obs=0, age_ms=100, closure=0.3))
    a = ex.tick(10_000_000, POSE, state_t_ns=10_000_000, signature=b"a")
    assert a.kind == "track" and a.gripper_closure == 0.3
    again = ex.tick(10_000_000, POSE, state_t_ns=10_000_000, signature=b"a")
    assert again.pose7 == a.pose7 and ex.counts["resent"] == 1  # repeated timestamp: no motion
    moved = [POSE[0] + 0.001, *POSE[1:]]
    held = ex.tick(150_000_000, moved, state_t_ns=150_000_000, signature=b"b")
    assert held.kind == "hold" and held.pose7 == tuple(moved) and held.gripper_closure == 0.3
    assert not ex.accept(_cmd([POSE[0] + 0.01, *POSE[1:]], t_obs=-1, closure=1.0))  # older observation refused
    late = executor.HighRateExecutor(sinks.MockSink())
    late.accept(_cmd(POSE, t_obs=0, age_ms=10, closure=1.0))  # first tick after expiry: nothing of it applied
    assert late.tick(20_000_000, POSE, state_t_ns=20_000_000, signature=b"x").gripper_closure is None
    stale = executor.HighRateExecutor(sinks.MockSink())
    stale.accept(_cmd(POSE, age_ms=1e6))
    assert stale.tick(100_000_000, POSE, state_t_ns=40_000_000) is None and stale.stopped == "stale robot state"
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
                          cams=cams, live_m5=live, noise_seed=1)  # CPU fixture inference may exceed 200 ms here
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
                 "pyrealsense2-2.59.1.11053", "pyzmq-27.2.0", "robot_msg_irsl-0.4.0"):
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


def test_live_sink_is_bound_to_the_action_format_but_refused_in_this_revision(monkeypatch):
    auth = safety.authorize_execution(cli_execute=True, typed_confirmation=safety.CONFIRMATION, config_armed=True,
                                      policy_info={"execution_eligible": True})
    pub, rm = FakePublisher(), fake_robot_msg()
    with pytest.raises(safety.ExecutionRefused, match="never operated"):
        sinks.LiveSink(auth, publisher=pub, robot_msg=rm)
    monkeypatch.setattr(safety, "EXECUTION_PERMITTED", True)  # what a later directive would change (test only)
    with pytest.raises(safety.Unbound, match="no"):
        sinks.LiveSink(auth)  # still no transport: this package never creates the publisher
    live = sinks.LiveSink(auth, publisher=pub, robot_msg=rm)
    with pytest.raises(safety.ExecutionRefused, match="setpoints"):
        live.send(validate_action(_result(_outputs(_request())), _request(), ActionLimits(), session_id="s1",
                                  now_ns=_request().observation_time_ns + 1))
    live.send(executor.Setpoint(1, "track", POSE, 0.0, "r1"))
    sent = json.loads(pub.sent[0])
    assert len(pub.sent) == 1 and sent["real_time_action"]["target_dt"] == sc.TARGET_DT_S


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
            return 0

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
