"""Execution-mode guard of the robot PC: nothing in this package can command a robot unless a human explicitly selects
the manual execution mode, and even then the live controller binding is UNBOUND (raises) until the robot PC's own
teleop / switch_controller code has been inspected read-only and the binding written against it.

Three independent conditions are required to obtain an ``ExecutionAuthorization`` (``authorize_execution``):
the ``--execute`` command-line flag, the typed confirmation phrase ``CONFIRMATION`` and ``execution.armed: true`` in
the robot-PC config file. The authorization object cannot be built any other way (private construction token), and
``sinks.LiveSink`` refuses to exist without one. Setup, tests, recorded replay, dry run and the benchmark never call
``authorize_execution``. Standard library only.
"""

from __future__ import annotations

import enum
import time
from dataclasses import dataclass, field

CONTROLLER_BINDING = "UNBOUND"  # the robot PC's teleop/switch_controller interface has not been inspected yet
CONFIRMATION = "I am at the robot, the workspace is clear and the emergency stop is in my hand"
_TOKEN = object()  # only authorize_execution can construct an ExecutionAuthorization


class Mode(str, enum.Enum):
    REPLAY = "replay"  # recorded episode -> server -> MockSink (no camera, no robot)
    DRY_RUN = "dry_run"  # live cameras + read-only state -> server -> validation -> DryRunSink (never sends)
    EXECUTE = "execute"  # future manual mode: LiveSink (UNBOUND in this revision: always refuses)


class ExecutionRefused(RuntimeError):
    """The manual execution mode was not selected with all three explicit conditions."""


class Unbound(RuntimeError):
    """A robot-side interface whose real implementation has not been inspected; it never connects to anything."""


@dataclass(frozen=True)
class ExecutionAuthorization:
    mode: Mode
    granted_monotonic_ns: int
    reasons: tuple = ()
    _token: object = field(default=None, repr=False, compare=False)

    def __post_init__(self):
        if self._token is not _TOKEN or self.mode is not Mode.EXECUTE:
            raise ExecutionRefused("an ExecutionAuthorization is only issued by authorize_execution()")


def authorize_execution(*, cli_execute: bool, typed_confirmation: str | None, config_armed: bool) -> \
        ExecutionAuthorization:
    """All three explicit conditions, or ExecutionRefused naming every missing one."""
    missing = []
    if cli_execute is not True:
        missing.append("the --execute flag")
    if typed_confirmation != CONFIRMATION:
        missing.append(f"the typed confirmation {CONFIRMATION!r}")
    if config_armed is not True:
        missing.append("execution.armed: true in the robot-PC config")
    if missing:
        raise ExecutionRefused("manual execution mode NOT selected; missing: " + "; ".join(missing))
    return ExecutionAuthorization(Mode.EXECUTE, time.monotonic_ns(),
                                  ("cli --execute", "typed confirmation", "config execution.armed"), _TOKEN)
