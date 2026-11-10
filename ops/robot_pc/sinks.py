"""Controller-integration interface of the robot PC: where validated commands (and the local loop's setpoints) go.

* ``MockSink``: records in memory (recorded replay, tests);
* ``DryRunSink``: appends one JSON line per command to a log (or keeps them in memory); it opens no socket, device or
  port and never sends anything;
* ``LiveSink``: the future manual execution path. It refuses to exist without an ``ExecutionAuthorization`` (explicit
  manual mode: CLI flag + typed confirmation + config ``execution.armed``) and, even then, raises ``Unbound``: the robot
  PC's teleop / switch_controller interface has not been inspected, so no binding to it exists in this revision.

Nothing in setup, tests, replay, dry run or the benchmark constructs a ``LiveSink`` (tests/test_robot_pc.py).
"""

from __future__ import annotations

import json
import time
from pathlib import Path

from ops.robot_pc.safety import CONTROLLER_BINDING, ExecutionAuthorization, ExecutionRefused, Mode, Unbound


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
        self.log_path = None if log_path is None else Path(log_path)
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
    """UNBOUND. Constructing it requires an ExecutionAuthorization and still raises ``Unbound``."""

    mode = Mode.EXECUTE
    sends_to_robot = True

    def __init__(self, authorization: ExecutionAuthorization):
        if not isinstance(authorization, ExecutionAuthorization):
            raise ExecutionRefused("LiveSink requires an ExecutionAuthorization (manual execution mode)")
        raise Unbound(f"controller binding {CONTROLLER_BINDING}: the robot PC's teleop / switch_controller interface "
                      "(recorded endpoints tcp://127.0.0.1:28401 command, :28402 action, request "
                      "ipc:///tmp/teleop_collector_requests.ipc; gripper /dev/ttyUSB0) has not been inspected read-only; "
                      "no live command path exists in this revision (docs/robot_pc.md)")


def make_sink(mode: Mode, *, log_path=None, authorization: ExecutionAuthorization | None = None) -> CommandSink:
    if mode is Mode.REPLAY:
        return MockSink()
    if mode is Mode.DRY_RUN:
        return DryRunSink(log_path)
    return LiveSink(authorization)
