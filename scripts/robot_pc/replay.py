"""ROBOT PC, recorded replay (never commands a robot): recorded episode H5 -> policy server -> validation -> MockSink.

    venv/bin/python -I src/scripts/robot_pc/replay.py --episode /data/drawer-open/episode_000000.h5 \
        --policy dynamo-drawer-open --steps 60 --out runs/replay-dynamo.jsonl --summary runs/replay-dynamo.json
    # multi-view (3D-MVP multi-view): the live M5 helper runs on the recorded joint/gripper stream;
    #   --m5-store <released training-store episode.h5> cross-checks its displacements

Rows (JSONL) carry the action, the validated command (or the rejection reason), the binding and every timing; the
summary adds p50/p95/p99, achieved Hz, payload bytes, missed 66.7 ms deadlines and the recorded-command comparison.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from ops.robot_pc import cli  # noqa: E402
from ops.robot_pc.replay import replay_episode  # noqa: E402


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="replay.py", description=__doc__.split("\n")[0])
    cli.add_common(ap)
    ap.add_argument("--episode", type=Path, required=True)
    ap.add_argument("--start", type=int, default=0, help="first reference-camera frame index")
    ap.add_argument("--steps", type=int, default=None, help="reference frames to replay (default: all)")
    ap.add_argument("--pace", choices=("realtime", "max"), default="realtime")
    ap.add_argument("--m5-store", type=Path, default=None)
    ap.add_argument("--route", default="", help="label of the network route (e.g. robotpc->5090 ssh -L)")
    ap.add_argument("--out", type=Path, default=None, help="rows JSONL")
    ap.add_argument("--summary", type=Path, default=None)
    args = ap.parse_args(argv)
    cfg = cli.load_config(args)
    rows_out = cli.JsonlRows(args.out)
    try:
        _, summary = replay_episode(cfg, args, episode=args.episode, pace=args.pace, start=args.start,
                                    count=args.steps, m5_store=args.m5_store, on_row=rows_out, route=args.route)
    finally:
        rows_out.close()
    cli.write_json(args.summary, {**cli.SAFETY, **summary})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
