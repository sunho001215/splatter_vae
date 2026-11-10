"""Robot-state acquisition of the robot PC, READ-ONLY by construction (no source here can send anything).

The training binding (``s4d.rwlearn.actions.step_table``; enforced again by the client and the server): the measured
flange pose bound to an observation is the latest robot sample at or before the observation time (<= 15 ms old), the
gripper register the latest register sample at or before it (<= 100 ms old). Units and frames exactly as recorded:
pose7 = ``robot/flange_pose`` = panda_link8 (flange; T_flange_tcp = identity, the TCP IS the flange) in panda_link0,
[x y z qx qy qz qw] metres; gripper = ``gripper/position``, the raw Robotiq 2F-85 position register 0..255 (open 3,
command convention 3 + 227 x closure); joints = ``robot/q`` (rad, panda_joint1..7) for the M5 helper; all times on the
robot PC's CLOCK_MONOTONIC.

* ``RecordedRobotState``: a recording (replay), optionally shifted onto the current monotonic clock;
* ``StaticLatencyState``: a fixed, labelled state for LATENCY measurements with live cameras while the live source is
  unbound (its actions are meaningless and only ever reach the DryRunSink);
* ``LiveRobotState``: UNBOUND skeleton of the read-only live source (raises): the robot PC's state publisher
  (recorded ``robot.state_endpoint`` tcp://127.0.0.1:28400) and the gripper register (serial /dev/ttyUSB0, owned by
  the teleop collector) have not been inspected; docs/robot_pc.md lists what is inferred from the recordings.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ops.robot_pc.safety import CONTROLLER_BINDING, Unbound


class StateUnavailable(ValueError):
    """No robot or gripper sample at or before the observation time."""


@dataclass(frozen=True)
class BoundState:
    pose7: tuple
    pose_time_ns: int
    gripper_register: float
    gripper_time_ns: int


class RobotStateSource:
    kind = "abstract"
    label = ""

    def bound_state(self, t_obs_ns: int) -> BoundState:
        """Latest robot / gripper samples at or before the observation time (the client checks the training ages)."""
        raise NotImplementedError

    def rows(self, after_ns: int, up_to_ns: int) -> tuple[list, list]:
        """Robot rows [(t_ns, q7)] and gripper samples [(t_ns, register)] with after < t <= up_to (M5 helper input)."""
        raise NotImplementedError

    def close(self) -> None:
        pass


def _latest(times: np.ndarray, t: int) -> int:
    return int(np.searchsorted(times, t, side="right")) - 1  # actions.latest_at_or_before


class RecordedRobotState(RobotStateSource):
    kind = "recorded"

    def __init__(self, recording, offset_ns: int = 0):
        """``recording``: ``s4d.rwlearn.deploy.replay.RecordedEpisode``; ``offset_ns`` moves every sample time onto the
        replay clock (frame times get the same offset, so ages and the binding are unchanged)."""
        import h5py

        raw = recording.raw
        self.offset_ns = int(offset_ns)
        self.t = np.asarray(raw.robot_t, np.int64) + self.offset_ns
        self.pose = np.asarray(raw.measured, np.float64)
        self.gt = np.asarray(raw.gripper_t, np.int64) + self.offset_ns
        self.gp = np.asarray(raw.gripper_position, np.float64)
        with h5py.File(recording.path, "r") as f:
            self.q = np.asarray(f["robot/q"][()], np.float64)
        self.label = f"recorded {recording.path}"

    def bound_state(self, t_obs_ns: int) -> BoundState:
        i, j = _latest(self.t, t_obs_ns), _latest(self.gt, t_obs_ns)
        if i < 0 or j < 0:
            raise StateUnavailable("no robot or gripper sample at or before the observation time")
        return BoundState(tuple(self.pose[i].tolist()), int(self.t[i]), float(self.gp[j]), int(self.gt[j]))

    def rows(self, after_ns: int, up_to_ns: int) -> tuple[list, list]:
        r = slice(_latest(self.t, after_ns) + 1, _latest(self.t, up_to_ns) + 1)
        g = slice(_latest(self.gt, after_ns) + 1, _latest(self.gt, up_to_ns) + 1)
        return (list(zip(self.t[r].tolist(), self.q[r])), list(zip(self.gt[g].tolist(), self.gp[g].tolist())))


class StaticLatencyState(RobotStateSource):
    """LATENCY ONLY: a fixed pose / joint vector / register, sampled 1 ms (robot, 1 kHz) and 10 ms (gripper) before
    every observation. Its actions are meaningless; the dry run labels every row with it."""

    kind = "static_latency_only"
    label = "STATIC LATENCY-ONLY STATE: fixed pose, actions meaningless (live robot state source UNBOUND)"

    def __init__(self, pose7, q7, gripper_register: float):
        self.pose7 = tuple(float(x) for x in np.asarray(pose7, np.float64).reshape(7))
        self.q7 = np.asarray(q7, np.float64).reshape(7)
        self.register = float(gripper_register)

    def bound_state(self, t_obs_ns: int) -> BoundState:
        return BoundState(self.pose7, int(t_obs_ns) - 1_000_000, self.register, int(t_obs_ns) - 10_000_000)

    def rows(self, after_ns: int, up_to_ns: int) -> tuple[list, list]:
        robot = [(t, self.q7) for t in range((int(after_ns) // 1_000_000 + 1) * 1_000_000, int(up_to_ns) + 1,
                                             1_000_000)]
        grip = [(t, self.register) for t in range((int(after_ns) // 10_000_000 + 1) * 10_000_000, int(up_to_ns) + 1,
                                                  10_000_000)]
        return robot, grip


class LiveRobotState(RobotStateSource):
    """UNBOUND read-only live source. Constructing it raises: nothing connects to the robot PC's state endpoint until
    its publisher (switch_controller / teleop collector) has been inspected read-only."""

    kind = "live"

    def __init__(self, *_, **__):
        raise Unbound(f"live robot state {CONTROLLER_BINDING}: inferred read-only sources (not connected): "
                      "robot.state_endpoint tcp://127.0.0.1:28400 (switch_controller state, ~1 kHz seq), gripper "
                      "register via the teleop collector (it owns /dev/ttyUSB0; a second serial reader would "
                      "interfere); bind only after reading the robot PC's code (docs/robot_pc.md)")
