"""Read untrusted downloaded data only under python -I from this trusted script."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _bootstrap import guard_gpus  # noqa: E402

guard_gpus()

from s4d.data import DROID_CACHE_ROOT, writable_path  # noqa: E402
from s4d.data.droid.inspect import alignment_overlays, cached_coordinate_overlays, inspection_report  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description="Inspect actual schema, recompute alignment, and render timestamp overlays")
    parser.add_argument("--sample-root", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path, required=True)
    args = parser.parse_args()
    args.cache_root = writable_path(args.cache_root, DROID_CACHE_ROOT)
    report = inspection_report(args.sample_root, args.cache_root)
    report["overlays"] = [
        str(path)
        for path in alignment_overlays(args.sample_root, args.cache_root) + cached_coordinate_overlays(args.cache_root)
    ]
    (args.cache_root / "alignment_report.json").write_text(json.dumps(report, indent=2))
    print(
        json.dumps(
            {
                key: report[key]
                for key in ("state_alignment", "scene_depth", "cross_camera", "gripper", "acceptance", "overlays")
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
