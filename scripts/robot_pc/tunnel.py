"""ROBOT PC: the direct ssh -L route to the 5090 policy server (start / stop / status), managed by pid file; never
touches the key or host-key files beyond checking presence and mode (ops/robot_pc/tunnel.py, docs/robot_pc.md).

    venv/bin/python -I src/scripts/robot_pc/tunnel.py status      # pid alive + listener + server WELCOME
    venv/bin/python -I src/scripts/robot_pc/tunnel.py start       # refuses if 127.0.0.1:47100 is already bound
    venv/bin/python -I src/scripts/robot_pc/tunnel.py stop        # SIGTERM to the pid-file pid after checking it
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from ops.robot_pc import tunnel  # noqa: E402


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="tunnel.py", description=__doc__.split("\n")[0])
    ap.add_argument("action", choices=("start", "stop", "status"))
    ap.add_argument("--root", default="~/workspace/s4d-robot-pc")
    ap.add_argument("--port", type=int, default=47100)
    ap.add_argument("--container", default=tunnel.CONTAINER, help="shkim_rw IP (the key's permitopen must match)")
    args = ap.parse_args(argv)
    try:
        if args.action == "start":
            out = tunnel.start(args.root, port=args.port, container=args.container)
        elif args.action == "stop":
            out = tunnel.stop(args.root, args.port)
        else:
            out = tunnel.status(args.root, args.port)
    except (tunnel.TunnelError, tunnel.ExecutionRefused) as exc:
        print(f"tunnel {args.action} REFUSED: {exc}", file=sys.stderr)
        return 3
    print(json.dumps(out))
    return 0 if args.action != "status" or out["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
