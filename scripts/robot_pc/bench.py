"""ROBOT PC, latency benchmark on a recorded episode (never commands a robot): the replay path at the trained 15 Hz
(``realtime``: frames released at their recorded times) and at the maximum rate (``max``), one persistent session,
summaries only (rows with --rows-dir).

    venv/bin/python -I src/scripts/robot_pc/bench.py --episode /data/drawer-open/episode_000000.h5 \
        --policy dynamo-drawer-open --steps 150 --route "robotpc -> 5090 (ssh -L)" --out runs/bench-dynamo.json

Reported per mode (client monotonic clock): p50/p95/p99/max of preprocessing->action (e2e_ms), capture->action and
capture->validated command (replay: capture = release time, no camera latency), replay acquisition (JPEG decode +
rs.align emulation), client preprocessing, serialisation, transport (round trip - server time), server prep / queue /
encoder / FQL / total, action validation; achieved Hz, payload bytes (first = history fill / steady / response),
missed 66.7 ms deadlines. Live camera latency (device stamp -> host, ~65 ms in the recordings) and native rs.align
are measured only by dry_run.py on the robot PC.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from ops.robot_pc import cli  # noqa: E402
from ops.robot_pc.replay import replay_episode  # noqa: E402


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="bench.py", description=__doc__.split("\n")[0])
    cli.add_common(ap)
    ap.add_argument("--episode", type=Path, required=True)
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--steps", type=int, default=150, help="reference frames per mode")
    ap.add_argument("--modes", default="realtime,max")
    ap.add_argument("--route", required=True, help="label of the measured route, e.g. 'robotpc -> 5090 (ssh -L)'")
    ap.add_argument("--rows-dir", type=Path, default=None)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args(argv)
    cfg = cli.load_config(args)
    import platform
    import socket

    out = {**cli.SAFETY, "route": args.route, "client_host": socket.gethostname(),
           "client_platform": platform.platform(), "python": platform.python_version(),
           "started": time.strftime("%Y-%m-%d %H:%M:%S %z"), "runs": []}
    for mode in args.modes.split(","):
        rows_out = cli.JsonlRows(None if args.rows_dir is None else args.rows_dir / f"bench-{mode}.jsonl")
        try:
            _, summary = replay_episode(cfg, args, episode=args.episode, pace=mode, start=args.start,
                                        count=args.steps, name=f"robot-pc-bench-{mode}", on_row=rows_out,
                                        route=args.route)
        finally:
            rows_out.close()
        out["runs"].append(summary)
    cli.write_json(args.out, out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
