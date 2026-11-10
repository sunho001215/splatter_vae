"""Adapter to the robot PC's real controller interface, written from its code (read-only inspection 2026-11-10 of
/home/irsl/workspace/panda-teleop: switch_controller/docs/communication.md, teleop_collector/teleop_collector/
{robot,teleop,app}.py, config/{default,site}.yaml; evidence in docs/robot_pc.md). Nothing here opens a socket at import;
the conversions are pure functions.

Interface (robot_msg protocol 2, ``robot-msg-irsl==0.4.0`` over ZeroMQ, the collector's pins):
* RobotState: switch_controller BINDS a PUB on ``tcp://*:28400`` (~1 kHz); consumers CONNECT a SUB with
  ``ZMQ_CONFLATE=1``. ``ee_frames[0].pose`` = switch_controller's end effector = flange (panda_link8) rotated -90 deg
  about z: ``flange = ee @ inv(T_FLANGE_EE)`` (teleop.flange_from_ee). A SUB socket cannot send application messages
  to a PUB, so reading it cannot change the controller (``LiveStateReader``; constructed only with explicit approval).
* ControllerCommand (``ctrl_mode``; 3 = task impedance) and RobotAction (``frame_id="world"``,
  ``real_time_action.ee_pose_rt=[ee pose]``, ``target_dt``) go out on PUB sockets that the COMMANDER binds
  (:28401 / :28402) and the controller's SUBs connect to. The controller holds the measured pose when no action
  arrives within ``target_dt`` and rejects targets > 0.2 m / 0.5 rad from the measured pose. This module only builds
  those messages (``robot_action``); it never binds a command or action socket and never requests a controller mode.
* The gripper (Robotiq 2F-85, Modbus RTU on /dev/ttyUSB0) is driven and read by the teleop collector itself; there is
  no gripper state publisher. A live gripper register source stays UNBOUND (reading it would mean sending Modbus
  requests to the gripper).
"""

from __future__ import annotations

import collections
import threading
import time

import numpy as np

from ops.robot_pc.robot_state import BoundState, RobotStateSource, StateUnavailable
from ops.robot_pc.safety import Unbound

PROTOCOL = "robot-msg-irsl==0.4.0 (robot_msg protocol 2) over ZeroMQ (pyzmq 27.2.0)"
STATE_ENDPOINT = "tcp://127.0.0.1:28400"  # site.yaml robot.state_endpoint (switch_controller binds tcp://*:28400)
COMMAND_BIND, ACTION_BIND = "tcp://127.0.0.1:28401", "tcp://127.0.0.1:28402"  # owned by the commander (collector)
CARTESIAN_MODES = (3, 5)  # task_impedance, task_nric (robot.py)
REJECT_M, REJECT_RAD = 0.2, 0.5  # switch_controller rejects pose targets farther from the measured pose
TARGET_DT_S = 0.05  # robot.command_timeout_s: the controller holds the measured pose without a fresher target


def _rot_z(deg: float) -> np.ndarray:
    a = np.deg2rad(deg)
    return np.array([[np.cos(a), -np.sin(a), 0.0], [np.sin(a), np.cos(a), 0.0], [0.0, 0.0, 1.0]])


T_FLANGE_EE = np.eye(4)
T_FLANGE_EE[:3, :3] = _rot_z(-90.0)  # teleop.T_FLANGE_EE = make_T(rot_z(-90), 0); recorded T_flange_controller_ee


def _quat_to_mat(q) -> np.ndarray:
    x, y, z, w = np.asarray(q, np.float64) / np.linalg.norm(q)
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def _mat_to_quat(R) -> np.ndarray:
    """The collector's ``teleop.mat_to_quat`` (Shepperd, unit, w >= 0), so converted poses equal its recorded ones."""
    R = np.asarray(R, np.float64)
    tr = R[0, 0] + R[1, 1] + R[2, 2]
    if tr > 0.0:
        s = 2.0 * np.sqrt(tr + 1.0)
        q = [(R[2, 1] - R[1, 2]) / s, (R[0, 2] - R[2, 0]) / s, (R[1, 0] - R[0, 1]) / s, 0.25 * s]
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
        q = [0.25 * s, (R[0, 1] + R[1, 0]) / s, (R[0, 2] + R[2, 0]) / s, (R[2, 1] - R[1, 2]) / s]
    elif R[1, 1] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
        q = [(R[0, 1] + R[1, 0]) / s, 0.25 * s, (R[1, 2] + R[2, 1]) / s, (R[0, 2] - R[2, 0]) / s]
    else:
        s = 2.0 * np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
        q = [(R[0, 2] + R[2, 0]) / s, (R[1, 2] + R[2, 1]) / s, 0.25 * s, (R[1, 0] - R[0, 1]) / s]
    q = np.asarray(q)
    q /= np.linalg.norm(q)
    return -q if q[3] < 0.0 else q


def _compose(pose7, T: np.ndarray) -> np.ndarray:
    p = np.asarray(pose7, np.float64)
    R = _quat_to_mat(p[3:]) @ T[:3, :3]
    return np.concatenate([p[:3] + _quat_to_mat(p[3:]) @ T[:3, 3], _mat_to_quat(R)])


def flange_from_ee(ee_pose7) -> np.ndarray:
    """switch_controller ee pose -> flange (panda_link8) pose, both in the base frame [x y z qx qy qz qw]."""
    return _compose(ee_pose7, np.linalg.inv(T_FLANGE_EE))


def ee_from_flange(flange_pose7) -> np.ndarray:
    return _compose(flange_pose7, T_FLANGE_EE)


def state_sample(msg, t_recv_ns: int) -> dict:
    """A decoded ``robot_msg.RobotState`` -> the recorder's row fields (flange pose, joints, sequence, times)."""
    p = msg.ee_frames[0].pose
    ee = [p.position.x, p.position.y, p.position.z, p.orientation.x, p.orientation.y, p.orientation.z,
          p.orientation.w]
    send = getattr(getattr(msg.metadata, "send_time", None), "seconds", None)
    return {"t_ns": int(t_recv_ns), "seq": int(msg.sequence), "q": np.asarray(msg.q, np.float64),
            "flange_pose7": flange_from_ee(ee), "t_send_unix": send}


def robot_action(robot_msg, flange_goal7, target_dt: float = TARGET_DT_S):
    """The RobotAction the collector publishes for a flange goal (robot.SwitchControllerClient.send_ee_pose): ee pose
    in the base ("world") frame, ``target_dt`` hold timeout. Message construction only; nothing is sent here."""
    ee = ee_from_flange(flange_goal7)
    pose = robot_msg.Pose(robot_msg.Vec3(*map(float, ee[:3])), robot_msg.Quaternion(*map(float, ee[3:])))
    return robot_msg.RobotAction(frame_id="world", real_time_action=robot_msg.RobotActionRealTimeAction(
        ee_pose_rt=[pose], target_dt=float(target_dt)))


class LiveStateReader(RobotStateSource):
    """Read-only live robot state: a CONFLATE SUB connected to switch_controller's RobotState PUB (never binds, never
    publishes). Constructing it requires ``approved=True`` (the coordinator's explicit go-ahead in this directive).
    Gripper: no read-only source exists -> ``bound_state`` refuses until one is given (``gripper_source``).
    Sample time = receipt on this host (the recorder's ``robot/t`` convention); a non-increasing ``sequence`` is
    dropped. A frozen publisher (switch_controller STOP republishes one state with new sequence numbers) is NOT
    detectable from the stream: a future execute mode must also check that q/dq evolve."""

    kind = "live_state_reader"
    label = "live switch_controller RobotState (read-only SUB)"

    def __init__(self, endpoint: str = STATE_ENDPOINT, *, approved: bool = False, zmq_module=None, robot_msg=None,
                 gripper_source=None, capacity: int = 4096):
        if approved is not True:
            raise Unbound("live robot state reader not approved: connecting to the controller's state publisher "
                          "needs the coordinator's explicit go-ahead (--live-state-approved)")
        if zmq_module is None:
            import zmq as zmq_module
        if robot_msg is None:
            import robot_msg
        self.zmq, self.robot_msg, self.gripper = zmq_module, robot_msg, gripper_source
        self.ctx = zmq_module.Context()
        self.sock = self.ctx.socket(zmq_module.SUB)
        self.sock.setsockopt(zmq_module.LINGER, 0)
        self.sock.setsockopt(zmq_module.CONFLATE, 1)
        self.sock.setsockopt(zmq_module.SUBSCRIBE, b"")
        self.sock.connect(endpoint)
        self.samples = collections.deque(maxlen=int(capacity))
        self.decode_errors = self.stale = 0
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._read, name="robot-state", daemon=True)
        self._thread.start()

    def _read(self) -> None:
        while not self._stop.is_set():
            if not self.sock.poll(50):
                continue
            buf = self.sock.recv()
            t = time.monotonic_ns()
            try:
                msg = self.robot_msg.decode(buf)
                if not isinstance(msg, self.robot_msg.RobotState) or not msg.ee_frames:
                    raise ValueError("not a RobotState with an end effector")
                row = state_sample(msg, t)
                if self.samples and row["seq"] <= self.samples[-1]["seq"]:
                    self.stale += 1  # repeated / reordered publication: never re-stamped as fresh
                    continue
                self.samples.append(row)
            except Exception:  # noqa: BLE001 - counted (malformed state is never used)
                self.decode_errors += 1

    def bound_state(self, t_obs_ns: int) -> BoundState:
        rows = [s for s in list(self.samples) if s["t_ns"] <= t_obs_ns]
        if not rows:
            raise StateUnavailable("no robot state at or before the observation time")
        if self.gripper is None:
            raise StateUnavailable("gripper register: no read-only source (the teleop collector owns the Robotiq "
                                   "serial port and publishes no gripper state)")
        g = self.gripper.at_or_before(t_obs_ns)
        return BoundState(tuple(rows[-1]["flange_pose7"].tolist()), rows[-1]["t_ns"], g[1], g[0])

    def rows(self, after_ns: int, up_to_ns: int) -> tuple[list, list]:
        robot = [(s["t_ns"], s["q"]) for s in list(self.samples) if after_ns < s["t_ns"] <= up_to_ns]
        grip = [] if self.gripper is None else self.gripper.rows(after_ns, up_to_ns)
        return robot, grip

    def close(self) -> None:
        self._stop.set()
        self._thread.join(timeout=1.0)
        self.sock.close(0)
        self.ctx.term()
