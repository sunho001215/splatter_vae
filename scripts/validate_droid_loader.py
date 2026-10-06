from __future__ import annotations

import argparse
import json
import time
from collections import deque
from pathlib import Path

import yaml
from torch.utils.data import DataLoader, Subset

from dataset.droid.dataset import DROIDPreprocessedDataset, droid_collate
from scripts.train_droid import (
    _dataset_config,
    _validate_cached_manifest,
    _validate_fixed_pipeline_contract,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate deterministic cached Stage-0 decode and collation."
    )
    parser.add_argument("--config", required=True)
    parser.add_argument(
        "--preprocessed-root",
        default=None,
        help="Override dataset.preprocessed_root (for example, for the pilot cache).",
    )
    parser.add_argument("--split", choices=("train", "validation"), default="train")
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--samples", type=int, default=4)
    parser.add_argument(
        "--report-rows",
        type=int,
        default=8,
        help="Maximum first/last decoded-window examples included in JSON output.",
    )
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
    if args.preprocessed_root is not None:
        config["dataset"]["preprocessed_root"] = str(
            Path(args.preprocessed_root).expanduser().resolve()
        )
    _validate_cached_manifest(config)
    dataset = DROIDPreprocessedDataset(_dataset_config(config, args.split))
    if int(args.samples) <= 0:
        raise ValueError("--samples must be positive.")
    if int(args.report_rows) < 0:
        raise ValueError("--report-rows must be nonnegative.")
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
    first_limit = (int(args.report_rows) + 1) // 2
    last_limit = int(args.report_rows) // 2
    first_rows = []
    last_rows: deque[dict[str, object]] = deque(maxlen=last_limit or None)
    processed = 0
    for index, batch in enumerate(loader):
        row = {
            "index": index,
            "episode_id": batch["episode_id"][0],
            "representation_histories_shape": list(
                batch["representation_histories"].shape
            ),
            "target_depth_shape": list(batch["target_depth"].shape),
            "target_flow_shape": list(batch["target_flow"].shape),
            "novel_rgb_shape": list(batch["novel_rgb"].shape),
            "history_raw_timesteps": batch["history_raw_timesteps"][0].tolist(),
            "camera_serials": list(batch["camera_serials"][0]),
        }
        if len(first_rows) < first_limit:
            first_rows.append(row)
        elif last_limit:
            last_rows.append(row)
        processed += 1
    elapsed = time.perf_counter() - started
    rows = [*first_rows, *last_rows]
    print(
        json.dumps(
            {
                "split": args.split,
                "workers": int(args.workers),
                "samples": processed,
                "seconds": elapsed,
                "samples_per_second": processed / elapsed,
                "reported_rows": len(rows),
                "rows": rows,
            },
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
