"""Native-depth entrypoint for a rig accepted for OFFLINE use by its task's acceptance override.

Same CLI as scripts/realworld_native_depth.py (``build --store --rig --out-dir --out-store [--workers] [--episodes]
[--raw-root]``), via s4d.rwlearn.native_depth_override. Existing native-depth rows keep their original scripts and
imports; a rig without a valid acceptance override is refused here (use the ordinary entrypoint).
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))  # python -I omits the script directory
from _bootstrap import guard_gpus  # noqa: E402

GPU_MAPPING = guard_gpus()
from s4d.rwlearn.dsreset_runtime import install  # noqa: E402

install()
from s4d.rwlearn.native_depth_override import build_native_depth_override  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="command", required=True)
    b = sub.add_parser("build")
    for flag in ("store", "rig", "out-dir", "out-store"):
        b.add_argument(f"--{flag}", required=True)
    b.add_argument("--workers", type=int, default=1)
    b.add_argument("--episodes", nargs="+")
    b.add_argument("--raw-root", metavar="RECORDED_ROOT=LOCAL_ROOT")
    args = ap.parse_args()
    result = build_native_depth_override(args.store, args.rig, args.out_dir, args.out_store, workers=args.workers,
                                         episodes=args.episodes, raw_root=args.raw_root)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
