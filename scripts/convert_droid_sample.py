"""Run with python -I; only trusted repository code is inserted into sys.path."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _bootstrap import guard_gpus  # noqa: E402

guard_gpus()

from s4d.data.droid.convert import convert_sample  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description="Convert the inspected real PointWorld-DROID sample")
    parser.add_argument("--sample-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True, help="existing empty approved output directory")
    parser.add_argument(
        "--rlds-root", type=Path, required=True, help="independently reread matched raw episode; no old teachers"
    )
    args = parser.parse_args()
    manifest = convert_sample(args.sample_root, args.output, rlds_root=args.rlds_root)
    print(
        json.dumps(
            {
                "windows": len(manifest["windows"]),
                "alignment": manifest["raw_state_alignment"],
                "gripper": {k: v for k, v in manifest["gripper_calibration"].items() if k != "candidates"},
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
