"""Validate real sample batches and measure CPU loader throughput at both sizes."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _bootstrap import guard_gpus  # noqa: E402

GPU_MAPPING = guard_gpus()

import torch  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402

from s4d.data import DROID_CACHE_ROOT, writable_path  # noqa: E402
from s4d.data.contract import collate, validate_batch  # noqa: E402
from s4d.data.droid.convert import RESOLUTIONS  # noqa: E402
from s4d.data.droid.dataset import DroidCacheDataset  # noqa: E402


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=2)
    args = parser.parse_args()
    args.root = writable_path(args.root, DROID_CACHE_ROOT)
    torch.set_num_threads(1)
    report = {
        "workers": args.workers,
        "batch_size": args.batch_size,
        "device": "CPU",
        "gpu_mapping": GPU_MAPPING,
        "variants": {},
    }
    for name, (height, width) in RESOLUTIONS.items():
        row = {}
        for split in ("train", "validation"):
            dataset = DroidCacheDataset(
                args.root, split=split, with_eval=split == "validation", image_height=height, image_width=width
            )
            loader = DataLoader(dataset, batch_size=args.batch_size, num_workers=args.workers, collate_fn=collate)
            start, count, batches = time.perf_counter(), 0, 0
            target_pixels = nonzero_motion = 0
            for batch in loader:
                shape = validate_batch(batch)
                assert "eval_images" not in batch
                count += shape["B"]
                batches += 1
                target_pixels += int(batch["motion_weight"].sum())
                nonzero_motion += int(
                    ((batch["motion3d"].square().sum(3, keepdim=True) > 1e-8) & (batch["motion_weight"] > 0)).sum()
                )
            elapsed = time.perf_counter() - start
            row[split] = {
                "samples": count,
                "batches": batches,
                "height": height,
                "width": width,
                "elapsed_s": elapsed,
                "samples_per_s": count / elapsed,
                "target_pixels": target_pixels,
                "nonzero_motion_pixels": nonzero_motion,
            }
            dataset.close()
        report["variants"][name] = row
    (args.root / f"loader_validation_workers{args.workers}.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
