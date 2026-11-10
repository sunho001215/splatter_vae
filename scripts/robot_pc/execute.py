"""ROBOT PC, FUTURE manual execution mode. REFUSES in this revision: Directive I (2026-11-10) forbids operating the
robot, and the controller binding is UNBOUND until the robot PC's teleop / switch_controller code has been inspected
read-only. Even with all three explicit conditions (``--execute``, the typed confirmation, ``execution.armed: true``)
``sinks.LiveSink`` raises ``Unbound``; nothing is opened, connected or sent.

    venv/bin/python -I src/scripts/robot_pc/execute.py --execute    # -> refused (exit 3)
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from ops.robot_pc import cli  # noqa: E402
from ops.robot_pc.safety import CONFIRMATION, ExecutionRefused, Unbound, authorize_execution  # noqa: E402
from ops.robot_pc.sinks import LiveSink  # noqa: E402


def main(argv=None, typed: str | None = None) -> int:
    ap = argparse.ArgumentParser(prog="execute.py", description=__doc__.split("\n")[0])
    ap.add_argument("--config", type=Path, default=None)
    ap.add_argument("--root", default=None)
    ap.add_argument("--execute", action="store_true", help="select the manual execution mode (still refused)")
    args = ap.parse_args(argv)
    cfg = cli.load_config(args)
    try:
        if typed is None and args.execute and sys.stdin.isatty():
            typed = input(f"Type exactly: {CONFIRMATION}\n> ")
        auth = authorize_execution(cli_execute=args.execute, typed_confirmation=typed,
                                   config_armed=(cfg.get("execution") or {}).get("armed") is True)
        LiveSink(auth)  # always raises Unbound in this revision
    except (ExecutionRefused, Unbound) as exc:
        print(f"execution REFUSED: {exc}", file=sys.stderr)
        return 3
    return 3  # unreachable: LiveSink never returns


if __name__ == "__main__":
    raise SystemExit(main())
