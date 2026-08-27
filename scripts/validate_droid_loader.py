from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import yaml
from torch.utils.data import DataLoader, Subset

from dataset.droid.dataset import DROIDLogicalDataset, droid_collate
from scripts.train_droid import _dataset_config, _validate_fixed_pipeline_contract


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate deterministic real-DROID decode and batch collation."
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--split", choices=("train", "validation"), default="train")
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--samples", type=int, default=4)
    parser.add_argument(
        "--multiprocessing-context",
        choices=("spawn", "forkserver", "fork"),
        default="spawn",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8")) or {}
    _validate_fixed_pipeline_contract(config)
    dataset = DROIDLogicalDataset(_dataset_config(config, args.split))
    if int(args.samples) <= 0:
        raise ValueError("--samples must be positive.")
    sample_count = min(int(args.samples), len(dataset))
    loader_kwargs = {
        # Exhausting a finite subset gives multiprocessing workers a clean
        # end-of-iteration signal instead of relying on interpreter teardown.
        "dataset": Subset(dataset, range(sample_count)),
        "batch_size": 1,
        "shuffle": False,
        "num_workers": int(args.workers),
        "collate_fn": droid_collate,
        "persistent_workers": False,
    }
    if args.workers:
        loader_kwargs["multiprocessing_context"] = args.multiprocessing_context
    loader = DataLoader(**loader_kwargs)
    started = time.perf_counter()
    rows = []
    for index, batch in enumerate(loader):
        rows.append(
            {
                "index": index,
                "episode_id": batch["episode_id"][0],
                "raw_histories_shape": list(batch["raw_histories"].shape),
                "raw_K_shape": list(batch["raw_K"].shape),
                "history_indices": batch["history_indices"][0].tolist(),
                "camera_serials": list(batch["camera_serials"][0]),
            }
        )
    elapsed = time.perf_counter() - started
    print(
        json.dumps(
            {
                "split": args.split,
                "workers": int(args.workers),
                "samples": len(rows),
                "seconds": elapsed,
                "samples_per_second": len(rows) / elapsed,
                "rows": rows,
            },
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
