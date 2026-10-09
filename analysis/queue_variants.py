"""Append the directive variant runs (two pretraining seeds x two tasks, 100k of the 200k schedule) and their full-split
evaluation jobs to ``experiments/queue.yaml``. Evaluation run directories need the pretraining config copied once the
run has started (``analysis/eval_config.py``).

    python analysis/queue_variants.py item1 --base aug.synth_views=2 aug.synth_render=true [--hold]
    python analysis/queue_variants.py item2 --base <item-1 winner's overrides> [--hold]
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parents[1]
QUEUE = REPO / "experiments" / "queue.yaml"
TASKS = ("hammer", "pick-place")
SEEDS = (0, 1)
SCHEDULE = ["train.steps=200000", "train.stop_step=100000"]

# directive item 1 (EXPERIMENT_LOG 2026-10-09): every variant has the far-plane validity fix
FAR = ["loss.depth_valid_far=true"]
OCC = ["loss.occlusion=1.0", "loss.occlusion_margin=0.02"]
ITEM1 = {
    "d0": FAR,
    "d1": FAR + OCC,
    "d2": FAR + OCC + ["loss.depth_grad=0.0", "loss.depth_hard=0.0"],
    "d3": FAR + OCC + ["loss.depth_grad=0.0"],
    "d4": FAR + OCC + ["loss.depth_hard=0.0"],
}
ITEM2 = {"m3d": ["loss.motion_space=gaussian"]}


def jobs_for(item: str, variants: dict, base: list[str], hold: bool) -> list[dict]:
    out = []
    for task in TASKS:
        for name, sets in variants.items():
            for seed in SEEDS:
                run = f"{item}-{name}-seed{seed}-{task}"
                out.append({
                    "id": run, "script": "scripts/train.py",
                    "args": ["--config", "configs/metaworld/base.yaml", f"configs/metaworld/tasks/{task}.yaml", "--name", run,
                             "--output-root", "runs/pretrain", "--resume", "auto", "--set", *SCHEDULE,
                             f"train.seed={seed}", *base, *sets],
                    "gpu": "any", "mem_gb": 10, "priority": 0, "deps": ["hold-memory"] if hold else [],
                    "max_restarts": 6, "stage": 1, "ram_gb": 13, "oom_score_adj": 0,
                })
                ev = f"fulleval-{run}-100k"
                out.append({
                    "id": ev, "script": "scripts/evaluate.py",
                    "args": ["--run", f"runs/{ev}", "--checkpoint", f"runs/pretrain/{run}/checkpoints/step_0100000.pt"],
                    "gpu": "any", "mem_gb": 5, "priority": 2, "deps": [run], "max_restarts": 1, "stage": 1,
                    "ram_gb": 12, "oom_score_adj": 300,
                })
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("item", choices=("item1", "item2"))
    ap.add_argument("--base", nargs="*", default=[], help="overrides of the configuration the variants build on")
    ap.add_argument("--hold", action="store_true", help="queue behind the hold-memory dependency")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    variants = ITEM1 if args.item == "item1" else ITEM2
    text = QUEUE.read_text()
    queue = yaml.safe_load(text)
    known = {j["id"] for j in queue["jobs"]}
    new = jobs_for("i1" if args.item == "item1" else "i2", variants, args.base, args.hold)
    clash = [j["id"] for j in new if j["id"] in known]
    if clash:
        raise SystemExit(f"already queued: {clash}")
    if args.dry_run:
        print(yaml.safe_dump(new[:2], sort_keys=False))
        print(f"{len(new)} jobs")
        return
    i = text.index("\nhost_ram_reserve_gb:") + 1
    out = text[:i] + yaml.safe_dump(new, sort_keys=False) + text[i:]
    assert len(yaml.safe_load(out)["jobs"]) == len(queue["jobs"]) + len(new)
    tmp = QUEUE.with_suffix(".yaml.tmp")
    tmp.write_text(out)
    os.replace(tmp, QUEUE)
    print(f"queued {len(new)} jobs")


if __name__ == "__main__":
    main()
