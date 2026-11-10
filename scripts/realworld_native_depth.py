"""Build CPU-only deployment-native RealSense depth overlays for a compact 3D-MVP store."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from s4d.rwlearn.native_depth import build_native_depth  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="command", required=True)
    b = sub.add_parser("build")
    for flag in ("store", "rig", "out-dir", "out-store"):
        b.add_argument(f"--{flag}", required=True)
    b.add_argument("--workers", type=int, default=1)
    b.add_argument("--episodes", nargs="+", help="task/episode_id keys; descriptor published once all overlays verify")
    args = ap.parse_args()
    print(json.dumps(build_native_depth(args.store, args.rig, args.out_dir, args.out_store,
                                        workers=args.workers, episodes=args.episodes), indent=2))


if __name__ == "__main__":
    main()
