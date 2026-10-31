"""Join PointWorld-DROID episodes with the indexed DROID RLDS release and DROID-3D calibration into a manifest.

Every accepted row passed the identity (uuid, scene suffix, camera serials), robot-state frame-mapping and
per-camera timestamp checks; every rejection is written with its reason to ``<out>.report.json``. Sources are read
only; outputs go to ``--out``.

    python scripts/droid_ls_build_manifest.py --pointworld-root <extracted droid/> --rlds-index <index dir> \
        --rlds-root <droid/1.0.1> --droid3d-extrinsics extrinsics.jsonl --droid3d-intrinsics camera_intrinsics.jsonl \
        --hand-eye gripper2wrist_transforms.json --calibration droid3d --out manifest.jsonl
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from s4d.data.droid.largescale import calibration as C  # noqa: E402
from s4d.data.droid.largescale.join import build_manifest  # noqa: E402
from s4d.data.droid.largescale.pointworld_source import DEFAULT_LAYOUT  # noqa: E402
from s4d.data.droid.largescale.rlds_index import RldsStepReader, load_episode_rows  # noqa: E402


def episodes_from(pointworld_root: Path, listing: str | None) -> list[str]:
    if listing:
        return [line.strip() for line in Path(listing).read_text().splitlines() if line.strip()]
    flows = pointworld_root / Path(DEFAULT_LAYOUT["flows"]).parent
    return sorted(p.name[: -len("_flows.h5")] for p in flows.glob("*_flows.h5"))


def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pointworld-root", required=True)
    ap.add_argument("--rlds-index", required=True)
    ap.add_argument("--rlds-root", required=True)
    ap.add_argument("--droid3d-extrinsics", default=None)
    ap.add_argument("--droid3d-intrinsics", default=None)
    ap.add_argument("--hand-eye", default=None, help="per-robot wrist<-flange (DepthWorld) or wrist<-EE (PointWorld) asset")
    ap.add_argument("--calibration", choices=("droid3d", "pointworld"), default="droid3d")
    ap.add_argument("--episodes", default=None, help="file with one PointWorld episode uuid per line")
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)
    root = Path(args.pointworld_root)
    reader = RldsStepReader(args.rlds_index, args.rlds_root)
    try:
        report = build_manifest(
            root,
            episodes_from(root, args.episodes),
            load_episode_rows(args.rlds_index),
            args.out,
            calibration=args.calibration,
            droid3d_extrinsics=C.JsonlIndex(args.droid3d_extrinsics) if args.droid3d_extrinsics else None,
            droid3d_intrinsics=C.JsonlIndex(args.droid3d_intrinsics) if args.droid3d_intrinsics else None,
            hand_eye=C.load_hand_eye(args.hand_eye) if args.hand_eye else None,
            rlds_reader=reader,
        )
    finally:
        reader.close()
    print(json.dumps({k: report[k] for k in ("accepted", "candidates")} | {"rejected": len(report["rejected"])}))


if __name__ == "__main__":
    main()
