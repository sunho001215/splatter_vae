"""Controller-integration interface of the robot PC: where validated commands (and the local loop's setpoints) go.

* ``MockSink``: records in memory (recorded replay, tests);
* ``DryRunSink``: appends one JSON line per command to a log (or keeps them in memory); it opens no socket, device or
  port and never sends anything;
* ``LiveSink``: the future manual execution path, bound to switch_controller's RobotAction format. It refuses to exist
  without an ``ExecutionAuthorization`` (explicit manual mode: CLI flag + typed confirmation + config
  ``execution.armed``) and, in this revision, even with one (``safety.EXECUTION_PERMITTED`` is False); it never
  creates a socket (the publisher is supplied by a future, separately approved caller).

Nothing in setup, tests, replay, dry run or the benchmark constructs a ``LiveSink`` (tests/test_robot_pc.py).
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
    """The future manual execution path, bound to switch_controller's message format (``switch_controller.robot_action``:
    the collector's RobotAction for the flange goal of each executor setpoint) and to a caller-supplied ``publisher``
    (``.send(bytes)``; this package creates no command socket). Refused without an ``ExecutionAuthorization`` and,
    in this revision, refused even with one (``safety.EXECUTION_PERMITTED`` is False). Gripper commands have no path
    (the teleop collector owns the Robotiq port)."""

    mode = Mode.EXECUTE
    sends_to_robot = True

    def __init__(self, authorization: ExecutionAuthorization, publisher=None, *, robot_msg=None,
                 target_dt: float | None = None):
        from ops.robot_pc import safety

        if not isinstance(authorization, ExecutionAuthorization):
            raise ExecutionRefused("LiveSink requires an ExecutionAuthorization (manual execution mode)")
        if not safety.EXECUTION_PERMITTED:
            raise ExecutionRefused("Directive I (2026-11-10): the robot is never operated; EXECUTION_PERMITTED is "
                                   f"False (controller binding: {CONTROLLER_BINDING}, no command transport)")
        if publisher is None or robot_msg is None:
            raise Unbound("LiveSink needs the command publisher and robot_msg (the commander-owned :28402 PUB); "
                          "none is created by this package")
        from ops.robot_pc import switch_controller as sc

        self.publisher, self.robot_msg = publisher, robot_msg
        self.target_dt = sc.TARGET_DT_S if target_dt is None else float(target_dt)
        self.sent = 0

    def send(self, item) -> None:
        from ops.robot_pc import switch_controller as sc

        from ops.robot_pc import safety

        if not safety.EXECUTION_PERMITTED:  # re-checked on every send, not only at construction
            raise ExecutionRefused("execution is not permitted in this revision")
        if getattr(item, "kind", None) != "track":
            raise ExecutionRefused("LiveSink sends only executor setpoints, never raw policy commands")
        self.publisher.send(self.robot_msg.encode(sc.robot_action(self.robot_msg, item.pose7, self.target_dt)))
        self.sent += 1


def make_sink(mode: Mode, *, log_path=None, authorization: ExecutionAuthorization | None = None) -> CommandSink:
    if mode is Mode.REPLAY:
        return MockSink()
    if mode is Mode.DRY_RUN:
        return DryRunSink(log_path)
    return LiveSink(authorization)  # refused (no authorization / EXECUTION_PERMITTED False / no publisher)
