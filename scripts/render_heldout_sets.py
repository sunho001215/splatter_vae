"""Render the near/trajectory held-out camera sets for one task's validation episodes (one GPU UUID for MuJoCo EGL).

    CUDA_VISIBLE_DEVICES=$GPU5 python scripts/render_heldout_sets.py --task hammer [--episodes 1 --output ...]

Replays the stored simulator states; writes /home/ws/data/metaworld/splatter4d_v1/heldout_sets/<task>.hdf5.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import _bootstrap  # noqa: F401
from _bootstrap import guard_gpus, guard_mujoco

GPU_MAPPING = guard_gpus()
EGL_DEVICE = guard_mujoco()

from s4d.data.metaworld.heldout import DATA_ROOT, heldout_path  # noqa: E402
from s4d.data.metaworld.heldout_render import render_heldout_sets  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True)
    ap.add_argument("--output", default=None, help="default: heldout_sets/<task>.hdf5 under the data root")
    ap.add_argument("--episodes", type=int, default=None, help="render only the first N validation episodes")
    args = ap.parse_args()
    output = Path(args.output) if args.output else heldout_path(args.task)
    source = DATA_ROOT / f"{args.task}.hdf5"
    manifest = DATA_ROOT / "splits" / f"{args.task}_seed0.json"
    print(f"GPU mapping {json.dumps(GPU_MAPPING)} EGL device {EGL_DEVICE}", flush=True)
    out = render_heldout_sets(
        args.task, source, manifest, output, max_episodes=args.episodes, log=lambda m: print(m, flush=True)
    )
    print(f"wrote {out} ({out.stat().st_size / 2**30:.2f} GiB)", flush=True)


if __name__ == "__main__":
    main()
