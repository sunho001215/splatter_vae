"""ROBOT PC, FUTURE manual execution mode. REFUSES in this revision: Directive I (2026-11-10) forbids operating the
robot (``safety.EXECUTION_PERMITTED`` is False) and this package has no command transport (the controller binding is
the RobotAction message format only). Even with all three explicit conditions (``--execute``, the typed confirmation,
``execution.armed: true``, a policy whose catalog entry is ``execution_eligible: true``) ``sinks.LiveSink`` refuses;
nothing is opened, connected or sent to a robot (``--execute`` reads the server's WELCOME for the policy's catalog entry).

    venv/bin/python -I src/scripts/robot_pc/execute.py --execute    # -> refused (exit 3)
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from ops.robot_pc import cli  # noqa: E402
from ops.robot_pc import safety  # noqa: E402
from ops.robot_pc.safety import CONFIRMATION, ExecutionRefused, Unbound, authorize_execution  # noqa: E402
from ops.robot_pc.sinks import LiveSink  # noqa: E402


def main(argv=None, typed: str | None = None) -> int:
    ap = argparse.ArgumentParser(prog="execute.py", description=__doc__.split("\n")[0])
    ap.add_argument("--config", type=Path, default=None)
    ap.add_argument("--root", default=None)
    ap.add_argument("--execute", action="store_true", help="select the manual execution mode (still refused)")
    ap.add_argument("--policy", default=None, help="policy id (its catalog entry must be execution_eligible)")
    ap.add_argument("--host", default=None)
    ap.add_argument("--port", type=int, default=None)
    args = ap.parse_args(argv)
    cfg = cli.load_config(args)
    try:
        if not safety.EXECUTION_PERMITTED:  # refused before any network activity in this revision
            raise ExecutionRefused("Directive I (2026-11-10): the robot is never operated (EXECUTION_PERMITTED False)")
        info = None
        if args.execute and args.policy:  # the catalog entry of the policy (WELCOME; no session, no robot)
            info = (cli.welcome(cfg, args)["policies"].get(args.policy) or {}).get("info")
        if typed is None and args.execute and sys.stdin.isatty():
            typed = input(f"Type exactly: {CONFIRMATION}\n> ")
        auth = authorize_execution(cli_execute=args.execute, typed_confirmation=typed,
                                   config_armed=(cfg.get("execution") or {}).get("armed") is True, policy_info=info)
        LiveSink(auth)  # refused in this revision (EXECUTION_PERMITTED False; no publisher is ever created here)
    except (ExecutionRefused, Unbound) as exc:
        print(f"execution REFUSED: {exc}", file=sys.stderr)
        return 3
    return 3  # unreachable: LiveSink never returns


if __name__ == "__main__":
    raise SystemExit(main())
