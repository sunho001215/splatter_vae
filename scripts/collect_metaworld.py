"""Collect one Meta-World task (one worker process = exactly one GPU UUID for MuJoCo EGL).

    CUDA_VISIBLE_DEVICES=$GPU4 python scripts/collect_metaworld.py --task hammer \
        --output /home/ws/data/metaworld/splatter4d_v1/hammer.hdf5 [--episodes 5]
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import _bootstrap  # noqa: F401
from _bootstrap import REPO, guard_gpus, guard_mujoco, require_passed_tests

GPU_MAPPING = guard_gpus()
EGL_DEVICE = guard_mujoco()

from s4d.data.metaworld.collect import CollectConfig, collect_task  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--episodes", type=int, default=None, help="override the 250-episode mixture (size estimate runs)")
    ap.add_argument("--max-steps", type=int, default=450)
    args = ap.parse_args()
    output = Path(args.output).resolve()
    data_root = Path("/home/ws/data/metaworld/splatter4d_v1")
    if data_root not in output.parents:
        raise ValueError("collection output must stay inside the authorized Meta-World data root")
    gate = json.loads((REPO / "docs/task_verification/summary.json").read_text())
    if not gate.get(args.task, {}).get("passed"):
        raise RuntimeError("task success/visibility gate has not passed")
    free_bytes = shutil.disk_usage(data_root).free
    if args.episodes is None or args.episodes > 5:
        require_passed_tests()
        pilot = data_root / f"{args.task}_pilot.hdf5"
        estimate = pilot.stat().st_size * 250 / 5
        if free_bytes < 2 * estimate:
            raise RuntimeError("insufficient disk margin for full collection")
    elif free_bytes < 2 * 1024**3:
        raise RuntimeError("insufficient free disk for a five-episode pilot")
    cfg = CollectConfig(task=args.task, output=str(output), seed=args.seed, max_steps=args.max_steps)
    start = time.time()
    log_path = Path(args.output).with_suffix(".log")
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "a") as log_file:

        def log(msg: str) -> None:
            print(msg, flush=True)
            log_file.write(msg + "\n")
            log_file.flush()

        log(f"GPU mapping {json.dumps(GPU_MAPPING)} EGL device {EGL_DEVICE}; free disk bytes {free_bytes}")
        out = collect_task(cfg, log=log, max_episodes=args.episodes)
        size = out.stat().st_size
        log(
            f"wrote {out} ({size / 2**30:.2f} GiB) in {time.time() - start:.0f}s; "
            f"episodes={args.episodes or cfg.num_episodes}"
        )


if __name__ == "__main__":
    main()
