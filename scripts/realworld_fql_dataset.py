"""Build an offline-FQL dataset: frozen baseline features once per observation time + transitions (actions, rewards,
masks) from the read-only raw recordings (s4d/rwlearn/fql_data.py, docs/realworld_fql.md). No training, no robot.

    CUDA_VISIBLE_DEVICES=<uuid> python -I scripts/realworld_fql_dataset.py build --store <store.yaml> \
        --condition dynamo --export runs/pretrain/rw-dynamo-drawer-open-pre/encoder.pt \
        --out runs/realworld_fql/data/dynamo-drawer-open-encdrawer-open.h5 --gate-report <gate_report.json>

Every store episode is used (train and validation; the split manifest selects at training time). Without a passed gate
report only ``--smoke`` builds of at most 3 episodes are allowed (tests / plumbing).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _bootstrap import REPO, guard_gpus  # noqa: E402

GPU_MAPPING = guard_gpus()

from ops.resources.gpu_guard import validate_runtime_path  # noqa: E402
from s4d.rwlearn.baseline_data import check_gate  # noqa: E402
from s4d.rwlearn.fql_data import CONDITIONS, build_fql_dataset  # noqa: E402
from s4d.rwlearn.store import open_store  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build")
    b.add_argument("--store", required=True)
    b.add_argument("--condition", required=True, choices=sorted(CONDITIONS))
    b.add_argument("--export", required=True)
    b.add_argument("--out", required=True)
    b.add_argument("--gate-report")
    b.add_argument("--reward", default="ogbench_final_success")
    b.add_argument("--device", default="cuda")
    b.add_argument("--smoke", action="store_true")
    b.add_argument("--episodes", nargs="*")
    args = ap.parse_args()

    store = open_store(args.store)
    episodes = args.episodes or store.episodes()
    if args.smoke:
        if len(episodes) > 3:
            raise SystemExit("--smoke builds use at most 3 episodes")
    else:
        if args.episodes:
            raise SystemExit("a full build uses every store episode (the gate covers the whole store)")
        check_gate(store, json.loads(Path(args.gate_report).read_text()) if args.gate_report else None, steps=None)
    out = validate_runtime_path(Path(args.out) if Path(args.out).is_absolute() else REPO / args.out, repository=REPO)
    out.parent.mkdir(parents=True, exist_ok=True)
    print(json.dumps(build_fql_dataset(store, args.condition, args.export, out, episodes=episodes,
                                       reward=args.reward, device=args.device), indent=1))


if __name__ == "__main__":
    main()
