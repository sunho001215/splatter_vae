"""Execution mode of the robot PC (Directive J, 2026-11-10: controlled live execution with the operator at the
hardware e-stop). ONE arm/start action and an immediate stop:

* ``authorize_execution(cli_execute=True, operator_start=True, policy_info=...)``: the launcher's ``--execute`` flag,
  the operator's start key (ENTER at the console after the preflight checks) and a policy whose catalog entry is
  ``execution_eligible: true`` (missing = NOT eligible: fail closed). The ``ExecutionAuthorization`` object is only
  issued by that function; ``sinks.LiveSink`` and ``controller.ControllerLink`` refuse to exist without it.
* stop: Ctrl-C / ``s`` / SIGTERM / any failure (stale / frozen state, communication, controller) -> the loop stops
  streaming, sends a hold at the measured pose and requests IDLE, as the teleop collector does (docs/robot_pc.md).

Setup, tests, recorded replay, dry run and the benchmark never call ``authorize_execution``. Standard library only.
"""

from __future__ import annotations

import enum
import os
import time
from dataclasses import dataclass, field

CONTROLLER_BINDING = "switch_controller robot_msg 2 (state SUB :28400, command/action PUBs :28401/:28402)"
_TOKEN = object()  # only authorize_execution can construct an ExecutionAuthorization


class Mode(str, enum.Enum):
    REPLAY = "replay"  # recorded episode -> server -> MockSink (no camera, no robot)
    DRY_RUN = "dry_run"  # live cameras + read-only state -> server -> validation -> DryRunSink (never sends)
    EXECUTE = "execute"  # operator-started live execution: LiveSink -> ControllerLink (+ gripper command)


class ExecutionRefused(RuntimeError):
    """The execution mode was not selected (flag, operator start, eligible policy) or an action is refused."""


class Unbound(RuntimeError):
    """A robot-side interface that is not available (not approved, or no read-only source exists); nothing connects."""


@dataclass(frozen=True)
class ExecutionAuthorization:
    mode: Mode
    granted_monotonic_ns: int
    reasons: tuple = ()
    _token: object = field(default=None, repr=False, compare=False)

    def __post_init__(self):
        if self._token is not _TOKEN or self.mode is not Mode.EXECUTE:
            raise ExecutionRefused("an ExecutionAuthorization is only issued by authorize_execution()")


FORBIDDEN_PORTS = frozenset(range(28400, 28405))  # switch_controller state/command/action/contact/hand channels
DEVICE_PREFIXES = ("/dev/", "/proc/", "/sys/")


def admit_endpoint(host: str, port: int) -> None:
    """Every connection of this package goes to the POLICY SERVER only: never to a controller channel."""
    if int(port) in FORBIDDEN_PORTS:
        raise ExecutionRefused(f"refusing to connect to {host}:{port}: a switch_controller channel, not the policy "
                               "server")


def safe_output(path) -> "Path":
    """An output file this package may write: a regular file (or a new one) outside /dev, /proc, /sys, never through a
    symlink, so no log/row/summary path can open a device (e.g. the gripper's serial port)."""
    from pathlib import Path

    p = Path(path).expanduser()
    absolute, resolved = os.path.abspath(p), os.path.realpath(p)  # realpath: parent symlinks too
    if absolute.startswith(DEVICE_PREFIXES) or resolved.startswith(DEVICE_PREFIXES) or p.is_symlink() or \
            (p.exists() and not p.is_file()):
        raise ExecutionRefused(f"refusing to write {path}: not a regular file outside /dev, /proc, /sys")
    return p


def execution_eligible(policy_info: dict | None) -> bool:
    """The catalog's explicit per-policy execution flag; anything but ``True`` (missing included) is not eligible."""
    return isinstance(policy_info, dict) and policy_info.get("execution_eligible") is True


def authorize_execution(*, cli_execute: bool, operator_start: bool, policy_info: dict | None) -> \
        ExecutionAuthorization:
    """The execution mode: ``--execute`` + the operator's start key + an execution-eligible policy."""
    missing = []
    if cli_execute is not True:
        missing.append("the --execute flag")
    if operator_start is not True:
        missing.append("the operator's start key")
    if not execution_eligible(policy_info):
        missing.append(f"a policy whose catalog entry is execution_eligible: true (got "
                       f"{(policy_info or {}).get('execution_eligible', 'missing')!r})")
    if missing:
        raise ExecutionRefused("execution mode NOT selected; missing: " + "; ".join(missing))
    return ExecutionAuthorization(Mode.EXECUTE, time.monotonic_ns(),
                                  ("cli --execute", "operator start", "policy execution_eligible"), _TOKEN)
