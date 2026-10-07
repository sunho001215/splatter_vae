"""Write the deterministic 96/4 episode split manifest of one collected task (``<root>/splits/<task>_seed<s>.json``)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _bootstrap import guard_gpus  # noqa: E402

GPU_MAPPING = guard_gpus()

from s4d.data.metaworld.dataset import split_episodes  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--path", type=Path, required=True)
    ap.add_argument("--train-ratio", type=float, default=0.96)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    train, val = split_episodes(args.path, args.train_ratio, args.seed)
    print(json.dumps({"path": str(args.path), "train": len(train), "validation": len(val), "validation_ids": val}))


if __name__ == "__main__":
    main()
