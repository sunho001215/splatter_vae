"""Execution-mode guard of the robot PC: nothing in this package can command a robot unless a human explicitly selects
the manual execution mode, and in this revision not even then.

Four independent conditions are required to obtain an ``ExecutionAuthorization`` (``authorize_execution``):
the ``--execute`` command-line flag, the typed confirmation phrase ``CONFIRMATION``, ``execution.armed: true`` in
the robot-PC config file, and a policy whose server catalog entry says ``execution_eligible: true`` (a missing field is
NOT eligible: fail closed; e.g. DynaMo is dry-run / replay only by user decision 2026-11-10). The authorization object cannot be built any other way (private construction token), and
``sinks.LiveSink`` refuses to exist without one. On top, ``EXECUTION_PERMITTED`` is False for Directive I (the robot is
never operated): ``LiveSink`` refuses even an authorized construction. The controller binding is the MESSAGE FORMAT
only (``switch_controller.robot_action``, written from the robot PC's code); no command transport (socket) exists in
this package. Setup, tests, recorded replay, dry run and the benchmark never call ``authorize_execution``.
Standard library only.
"""

from __future__ import annotations

import enum
import os
import time
from dataclasses import dataclass, field

CONTROLLER_BINDING = "message-format-only"  # switch_controller RobotAction built from its code; no command transport
EXECUTION_PERMITTED = False  # Directive I (2026-11-10): the robot is never operated; a later directive may lift it
CONFIRMATION = "I am at the robot, the workspace is clear and the emergency stop is in my hand"
_TOKEN = object()  # only authorize_execution can construct an ExecutionAuthorization


class Mode(str, enum.Enum):
    REPLAY = "replay"  # recorded episode -> server -> MockSink (no camera, no robot)
    DRY_RUN = "dry_run"  # live cameras + read-only state -> server -> validation -> DryRunSink (never sends)
    EXECUTE = "execute"  # future manual mode: LiveSink (refused in this revision: EXECUTION_PERMITTED is False)


class ExecutionRefused(RuntimeError):
    """The manual execution mode was not selected with all three explicit conditions."""


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
    absolute = Path(os.path.abspath(p))
    if str(absolute).startswith(DEVICE_PREFIXES) or p.is_symlink() or (p.exists() and not p.is_file()):
        raise ExecutionRefused(f"refusing to write {path}: not a regular file outside /dev, /proc, /sys")
    return p


def execution_eligible(policy_info: dict | None) -> bool:
    """The catalog's explicit per-policy execution flag; anything but ``True`` (missing included) is not eligible."""
    return isinstance(policy_info, dict) and policy_info.get("execution_eligible") is True


def authorize_execution(*, cli_execute: bool, typed_confirmation: str | None, config_armed: bool,
                        policy_info: dict | None = None) -> ExecutionAuthorization:
    """All four explicit conditions, or ExecutionRefused naming every missing one."""
    missing = []
    if cli_execute is not True:
        missing.append("the --execute flag")
    if typed_confirmation != CONFIRMATION:
        missing.append(f"the typed confirmation {CONFIRMATION!r}")
    if config_armed is not True:
        missing.append("execution.armed: true in the robot-PC config")
    if not execution_eligible(policy_info):
        missing.append(f"a policy whose catalog entry is execution_eligible: true (got "
                       f"{(policy_info or {}).get('execution_eligible', 'missing')!r})")
    if missing:
        raise ExecutionRefused("manual execution mode NOT selected; missing: " + "; ".join(missing))
    return ExecutionAuthorization(Mode.EXECUTE, time.monotonic_ns(),
                                  ("cli --execute", "typed confirmation", "config execution.armed",
                                   "policy execution_eligible"), _TOKEN)
