"""Build CPU-only deployment-native RealSense depth overlays for a compact 3D-MVP store."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _bootstrap import guard_gpus  # noqa: E402

# every scheduler job script validates its GPU allocation (this is CPU work registered with gpu: any)
GPU_MAPPING = guard_gpus()

from s4d.rwlearn.native_depth import build_native_depth  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="command", required=True)
    b = sub.add_parser("build")
    for flag in ("store", "rig", "out-dir", "out-store"):
        b.add_argument(f"--{flag}", required=True)
    b.add_argument("--workers", type=int, default=1)
    b.add_argument("--episodes", nargs="+", help="task/episode_id keys; descriptor published once all overlays verify")
    b.add_argument("--raw-root", metavar="RECORDED_ROOT=LOCAL_ROOT",
                   help="read each raw recording from a sha256-verified local copy (local placement; "
                        "s4d/rwlearn/native_depth_mapped.py)")
    args = ap.parse_args()
    if args.raw_root:
        from s4d.rwlearn.native_depth_mapped import build_native_depth_mapped

        result = build_native_depth_mapped(args.store, args.rig, args.out_dir, args.out_store, raw_root=args.raw_root,
                                           workers=args.workers, episodes=args.episodes)
    else:
        result = build_native_depth(args.store, args.rig, args.out_dir, args.out_store,
                                    workers=args.workers, episodes=args.episodes)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
