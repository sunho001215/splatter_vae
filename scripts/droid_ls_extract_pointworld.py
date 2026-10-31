"""Extract PointWorld-DROID per-episode files from a decompressed tar stream on stdin (see s4d/.../extract.py).

zstd -dc droid/depth_320x180/package.tar.zst.part-* | python scripts/droid_ls_extract_pointworld.py --dest <root>
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from s4d.data.droid.largescale.extract import SUFFIX, extract_members  # noqa: E402


def main(argv=None, stdin=None) -> dict:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dest", required=True)
    ap.add_argument("--kinds", default=",".join(SUFFIX), help="comma list of flows-fs-optimized,depth_320x180,cameras")
    ap.add_argument("--episodes", default=None, help="optional file with the episode uuids to keep")
    args = ap.parse_args(argv)
    keep = None
    if args.episodes:
        keep = {line.strip() for line in Path(args.episodes).read_text().splitlines() if line.strip()}
    report = extract_members(stdin or sys.stdin.buffer, args.dest, kinds=tuple(args.kinds.split(",")), episodes=keep)
    print(json.dumps(report))
    return report


if __name__ == "__main__":
    main()
