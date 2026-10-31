"""Stream-extract PointWorld-DROID per-episode members from a released ``tar`` stream (after ``zstd -dc``).

The depth package is one 1.23 TB ``tar.zst`` stream without an index, so random access needs a one-time extraction of
the PointWorld annotations (never of DROID itself). Only regular files whose names match the released member layout are
written, flat into ``<dest>/<kind dir>/<basename>``; links, devices, path tricks and unexpected names are refused and
reported. Writes are atomic and resumable (an existing file of the same size is skipped).

    zstd -dc package.tar.zst.part-* | python scripts/droid_ls_extract_pointworld.py --dest <pw root>
"""

from __future__ import annotations

import os
import re
import tarfile
from pathlib import Path

MEMBER = re.compile(
    r"^(?:\./)?droid/(?P<kind>flows-fs-optimized|depth_320x180|cameras)/"
    r"(?P<episode>[A-Za-z0-9_\-]+\+[0-9a-f]+\+\d{4}-\d{2}-\d{2}-\d{2}h-\d{2}m-\d{2}s)_(?P<suffix>flows\.h5|depth\.h5|cameras\.json)$"
)
SUFFIX = {"flows-fs-optimized": "flows.h5", "depth_320x180": "depth.h5", "cameras": "cameras.json"}


def extract_members(fileobj, dest: str | Path, *, kinds=tuple(SUFFIX), episodes: set[str] | None = None) -> dict:
    dest = Path(dest)
    report = {"written": 0, "skipped_existing": 0, "ignored": 0, "refused": []}
    with tarfile.open(fileobj=fileobj, mode="r|") as archive:
        for member in archive:
            match = MEMBER.match(member.name)
            if member.isdir() and not match:
                continue
            if not match:
                report["ignored"] += 1
                continue
            kind, episode = match["kind"], match["episode"]
            if SUFFIX[kind] != match["suffix"]:
                report["refused"].append(f"{member.name}: suffix does not match its directory")
                continue
            if not member.isreg():
                report["refused"].append(f"{member.name}: not a regular file")
                continue
            if kind not in kinds or (episodes is not None and episode not in episodes):
                report["ignored"] += 1
                continue
            target = dest / kind / f"{episode}_{match['suffix']}"
            if target.is_file() and target.stat().st_size == member.size:
                report["skipped_existing"] += 1
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            source = archive.extractfile(member)
            tmp = target.with_name(target.name + ".partial")
            with open(tmp, "wb") as out:
                remaining = member.size
                while remaining:
                    block = source.read(min(remaining, 1 << 20))
                    if not block:
                        raise OSError(f"truncated member {member.name}")
                    out.write(block)
                    remaining -= len(block)
            os.replace(tmp, target)
            report["written"] += 1
    return report
