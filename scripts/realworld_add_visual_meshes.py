#!/usr/bin/env python3
"""Add the robot VISUAL meshes to an asset copy that has only the collision meshes (calibration release 2026-11-08).

The remote asset copy ``/home/compu/kaist/sunho/realworld/assets/validation-20261031`` held only the collision STLs, so
every remote robot-grounded run fell back to collision meshes silently (now refused: ``robot_visual.
require_visual_meshes``). This tool copies the COLLADA visual meshes listed in a sha256 manifest from a source copy into
the target copy. ADD ONLY: it never overwrites, truncates or deletes anything. Every source file must match the
manifest; a target file that already exists must match too (then it is left alone) - any mismatch aborts before the
first write. Each file is written to a temporary name in its target directory, fsync'd, hard-linked to its final name
(``os.link`` fails if the name exists, so a concurrent writer can never be overwritten), re-hashed and the temporary
name removed. Stdlib only (runs with the host's python3). Dry run by default.

    python3 realworld_add_visual_meshes.py --manifest visual_meshes.sha256 --source SRC_ASSET_ROOT \
        --target DST_ASSET_ROOT --report report.json [--apply]

Manifest lines: ``<sha256>  <path relative to the asset root>`` (sha256sum format).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import time
from pathlib import Path

LINE = re.compile(r"^([0-9a-f]{64})\s+\*?(\S.*)$")


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def read_manifest(path: Path) -> list[tuple[str, str]]:
    rows = []
    for n, line in enumerate(Path(path).read_text().splitlines(), 1):
        if not line.strip():
            continue
        m = LINE.match(line.strip())
        if not m:
            raise SystemExit(f"{path}:{n}: not a sha256sum line")
        rel = m.group(2)
        if rel.startswith("/") or ".." in Path(rel).parts:
            raise SystemExit(f"{path}:{n}: path must be relative and inside the asset root: {rel}")
        if not rel.endswith(".dae"):
            raise SystemExit(f"{path}:{n}: only visual (.dae) meshes are added: {rel}")
        rows.append((m.group(1), rel))
    if not rows:
        raise SystemExit(f"{path}: empty manifest")
    return rows


def plan(rows, source: Path, target: Path) -> list[dict]:
    out = []
    for sha, rel in rows:
        src, dst = source / rel, target / rel
        if not src.is_file():
            raise SystemExit(f"source missing: {src}")
        got = sha256_file(src)
        if got != sha:
            raise SystemExit(f"source {src} sha256 {got} != manifest {sha}")
        if dst.is_symlink():
            raise SystemExit(f"target {dst} is a symlink; refusing")
        if dst.exists():
            have = sha256_file(dst)
            if have != sha:
                raise SystemExit(f"target {dst} exists with sha256 {have} != {sha}; never overwritten")
            out.append({"path": rel, "sha256": sha, "action": "present"})
        else:
            out.append({"path": rel, "sha256": sha, "action": "add", "bytes": src.stat().st_size})
    return out


def add_file(src: Path, dst: Path, sha: str) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)  # adds directories only (exist_ok never alters an existing one)
    tmp = dst.with_name(f".{dst.name}.adding.{os.getpid()}")
    fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    try:
        with os.fdopen(fd, "wb") as f, open(src, "rb") as s:
            for block in iter(lambda: s.read(1 << 20), b""):
                f.write(block)
            f.flush()
            os.fsync(f.fileno())
        if sha256_file(tmp) != sha:
            raise SystemExit(f"copy of {src} does not match its sha256")
        os.link(tmp, dst)  # FileExistsError if the name appeared meanwhile: never overwritten
    finally:
        if tmp.exists():
            os.unlink(tmp)
    if sha256_file(dst) != sha:
        raise SystemExit(f"{dst} does not match its sha256 after the add")
    dfd = os.open(dst.parent, os.O_RDONLY)
    try:
        os.fsync(dfd)
    finally:
        os.close(dfd)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--source", required=True)
    ap.add_argument("--target", required=True)
    ap.add_argument("--report", required=True)
    ap.add_argument("--apply", action="store_true")
    a = ap.parse_args(argv)
    source, target = Path(a.source).resolve(), Path(a.target).resolve()
    if source == target:
        raise SystemExit("source and target are the same")
    report = Path(a.report)
    if report.exists():
        raise SystemExit(f"{report} exists")
    rows = read_manifest(Path(a.manifest))
    actions = plan(rows, source, target)  # every check passes before the first write
    if a.apply:
        for act in actions:
            if act["action"] == "add":
                add_file(source / act["path"], target / act["path"], act["sha256"])
                act["action"] = "added"
        for act in actions:  # final verification of every listed file
            act["verified_sha256"] = sha256_file(target / act["path"]) == act["sha256"]
    out = {"tool": "realworld_add_visual_meshes", "applied": bool(a.apply), "at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
           "source": str(source), "target": str(target), "manifest": str(a.manifest),
           "manifest_sha256": sha256_file(Path(a.manifest)), "files": actions,
           "counts": {k: sum(x["action"] == k for x in actions) for k in ("add", "added", "present")}}
    report.write_text(json.dumps(out, indent=1))
    print(json.dumps(out["counts"]))
    return 0 if not a.apply or all(x.get("verified_sha256") for x in actions) else 1


if __name__ == "__main__":
    sys.exit(main())
