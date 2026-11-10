"""Controller-integration interface of the robot PC: where validated commands (and the local loop's setpoints) go.

* ``MockSink``: records in memory (recorded replay, tests);
* ``DryRunSink``: appends one JSON line per command to a log (or keeps them in memory); it opens no socket, device or
  port and never sends anything;
* ``LiveSink``: execution only (``ExecutionAuthorization`` required): setpoints -> ``controller.ControllerLink``,
  gripper closure -> the gripper worker.

Nothing in setup, replay, dry run or the benchmark constructs a ``LiveSink`` (tests/test_robot_pc.py).
"""

from __future__ import annotations

import json
import time

from ops.robot_pc.safety import CONTROLLER_BINDING, ExecutionAuthorization, ExecutionRefused, Mode, Unbound, safe_output


class CommandSink:
    mode: Mode
    sends_to_robot = False

    def send(self, item) -> None:
        raise NotImplementedError

    def close(self) -> None:
        pass


def _record(item) -> dict:
    return item.to_json() if hasattr(item, "to_json") else dict(item)


class MockSink(CommandSink):
    mode = Mode.REPLAY

    def __init__(self):
        self.items: list = []

    def send(self, item) -> None:
        self.items.append(item)


class DryRunSink(CommandSink):
    mode = Mode.DRY_RUN

    def __init__(self, log_path: str | Path | None = None):
        self.items: list = []
        self.log_path = None if log_path is None else safe_output(log_path)
        self._f = None if self.log_path is None else self.log_path.open("a")
        self.count = 0

    def send(self, item) -> None:
        self.count += 1
        if self._f is None:
            self.items.append(item)
            return
        self._f.write(json.dumps({"dry_run": True, "sent_to_robot": False, "t_log_ns": time.monotonic_ns(),
                                  **_record(item)}, allow_nan=False) + "\n")
        self._f.flush()

    def close(self) -> None:
        if self._f is not None:
            self._f.close()
            self._f = None


class LiveSink(CommandSink):
    """Execution: executor setpoints -> ``ControllerLink.send_flange_pose`` (switch_controller RobotAction) and the
    gripper closure -> the gripper worker's command (when given). Requires an ``ExecutionAuthorization``."""

    mode = Mode.EXECUTE
    sends_to_robot = True

    def __init__(self, authorization: ExecutionAuthorization, link, gripper=None, *, log=None):
        if not isinstance(authorization, ExecutionAuthorization):
            raise ExecutionRefused("LiveSink requires an ExecutionAuthorization (execution mode)")
        if link is None:
            raise Unbound("LiveSink needs the ControllerLink")
        if authorization.hardware_test not in (None, "hold"):
            raise ExecutionRefused("gripper authorization cannot command the arm")
        self.authorization = authorization
        self.link, self.gripper, self.log, self.sent = link, gripper, log, 0

    def send(self, item) -> None:
        if getattr(item, "kind", None) != "track":
            raise ExecutionRefused("LiveSink sends only executor setpoints, never raw policy commands")
        self.link.send_flange_pose(item.pose7)
        if self.gripper is not None and item.gripper_closure is not None:
            self.gripper.set_closure(item.gripper_closure)
        self.sent += 1
        if self.log is not None:
            self.log(item, True)


def make_sink(mode: Mode, *, log_path=None, authorization: ExecutionAuthorization | None = None) -> CommandSink:
    if mode is Mode.REPLAY:
        return MockSink()
    if mode is Mode.DRY_RUN:
        return DryRunSink(log_path)
    raise ExecutionRefused("execution sinks are built by the launcher with the ControllerLink (scripts/robot_pc/run.py)")
