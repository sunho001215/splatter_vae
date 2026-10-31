"""Extract PointWorld-DROID per-episode files from a decompressed tar stream on stdin (see s4d/.../extract.py).

zstd -dc droid/depth_320x180/package.tar.zst.part-* | python scripts/droid_ls_extract_pointworld.py --dest <root>

A stream cut inside the tar (for example only ``part-0000`` of the 252-part depth package) keeps every completed member
and drops the interrupted one; it exits 3 after printing the report unless ``--allow-truncated``.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from s4d.data.droid.largescale.extract import SUFFIX, TruncatedStreamError, extract_members  # noqa: E402


def main(argv=None, stdin=None) -> dict:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dest", required=True)
    ap.add_argument("--kinds", default=",".join(SUFFIX), help="comma list of flows-fs-optimized,depth_320x180,cameras")
    ap.add_argument("--episodes", default=None, help="optional file with the episode uuids to keep")
    ap.add_argument("--allow-truncated", action="store_true", help="accept a stream that ends inside the tar")
    args = ap.parse_args(argv)
    keep = None
    if args.episodes:
        keep = {line.strip() for line in Path(args.episodes).read_text().splitlines() if line.strip()}
    try:
        report = extract_members(
            stdin or sys.stdin.buffer,
            args.dest,
            kinds=tuple(args.kinds.split(",")),
            episodes=keep,
            allow_truncated=args.allow_truncated,
        )
    except TruncatedStreamError as error:
        print(json.dumps(error.report))
        raise SystemExit(3) from error
    print(json.dumps(report))
    return report


if __name__ == "__main__":
    main()
