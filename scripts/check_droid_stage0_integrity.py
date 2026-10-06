#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from dataset.droid.integrity import (
    IntegrityConfig,
    verify_stage0_dataset,
    write_final_manifest,
)
from dataset.droid.shards import write_json_atomic


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Independently verify an indexed DROID Stage-0 dataset."
    )
    parser.add_argument(
        "--root", default="/home/ws/data/droid_stage0_preprocessed"
    )
    parser.add_argument("--random-samples", type=int, default=512)
    parser.add_argument("--loader-windows", type=int, default=32)
    parser.add_argument("--seed", type=int, default=20260828)
    parser.add_argument("--no-shard-checksums", action="store_true")
    parser.add_argument("--full-payload-scan", action="store_true")
    parser.add_argument("--decode-all-jpegs", action="store_true")
    parser.add_argument("--finalize", action="store_true")
    parser.add_argument("--output", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = Path(args.root).expanduser().resolve()
    report = verify_stage0_dataset(
        IntegrityConfig(
            root=str(root),
            random_samples=int(args.random_samples),
            seed=int(args.seed),
            verify_shard_checksums=not bool(args.no_shard_checksums),
            full_payload_scan=bool(args.full_payload_scan),
            decode_all_jpegs=bool(args.decode_all_jpegs),
            loader_windows=int(args.loader_windows),
        )
    )
    output = Path(
        args.output or root / "reports" / "integrity.json"
    ).expanduser().resolve()
    write_json_atomic(output, report)
    if args.finalize:
        report["final_manifest"] = str(write_final_manifest(root, report))
        write_json_atomic(output, report)
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
