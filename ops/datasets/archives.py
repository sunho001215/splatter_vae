"""Extraction of the PointWorld-DROID ``tar.zst`` packages: staging, verification, atomic publication.

Members are stream-extracted by ``s4d.data.droid.largescale.extract.extract_members`` (strict released-layout filter,
regular files only, per-member atomic writes, resumable: an existing staged file of the member's size is skipped) from
``zstd -dc -T1`` into a staging directory on the same file system, then verified and renamed into the published tree.

- A flow shard (one self-contained archive): the staged member names must equal the shard's members in
  ``_shards_manifest.json`` (or the selected subset) and, without a selection, their bytes must equal the shard's
  ``raw_member_bytes``; every file must carry the HDF5 signature.
- The cameras package: member count and bytes equal the values measured on the released bytes; every file is JSON
  naming its own uuid.
- The depth package: ONE stream over all parts, fed in order (``iter_stream``); every member must carry the HDF5
  signature; the member set is compared with the flow episodes and the difference is reported.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import threading
from pathlib import Path

from s4d.data.droid.largescale.extract import extract_members

HDF5_SIGNATURE = b"\x89HDF\r\n\x1a\n"
FEED_CHUNK = 4 << 20


class ExtractionError(RuntimeError):
    pass


class ExtractionStopped(RuntimeError):
    """The owner is stopping; staged members are kept and the extraction resumes later."""


def episode_of(name: str) -> str:
    for suffix in ("_flows.h5", "_depth.h5", "_cameras.json"):
        if name.endswith(suffix):
            return name[: -len(suffix)]
    raise ExtractionError(f"unexpected member file {name!r}")


def extract_stream(chunks, staging: Path, kinds, episodes=None, should_stop=lambda: False) -> dict:
    """Feed byte chunks (an iterable; the compressed stream in order) through ``zstd -dc -T1`` into
    ``extract_members``; returns its report. Raises ExtractionError on a zstd failure or a stopped feed."""
    staging.mkdir(parents=True, exist_ok=True)
    zstd = subprocess.Popen(["zstd", "-dc", "-T1", "-q"], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE)
    feed_error = []

    def feed():
        try:
            for chunk in chunks:
                if should_stop():
                    break
                zstd.stdin.write(chunk)
        except Exception as error:  # noqa: BLE001  (reported through the main thread)
            feed_error.append(f"{type(error).__name__}: {error}")
        finally:
            if should_stop():
                feed_error.append("stopped")
            try:
                zstd.stdin.close()
            except OSError:
                pass

    feeder = threading.Thread(target=feed, daemon=True)
    feeder.start()
    failed = True
    try:
        report = extract_members(zstd.stdout, staging, kinds=tuple(kinds), episodes=episodes, allow_truncated=True)
        failed = False
    finally:
        if failed or feed_error:
            zstd.kill()
            zstd.stdout.read()  # unblock a feeder that is still writing into the dead pipe
        feeder.join()
        stderr = zstd.stderr.read().decode(errors="replace")
        code = zstd.wait()
        zstd.stdout.close()
        zstd.stderr.close()
    if "stopped" in feed_error:
        raise ExtractionStopped("stop requested")
    if feed_error:
        raise ExtractionError(f"feed failed: {feed_error[0]}")
    if code != 0:
        raise ExtractionError(f"zstd exited {code}: {stderr.strip()[-500:]}")
    if report["stream_truncated"]:
        raise ExtractionError(f"tar stream truncated: {report['stream_truncated']}")
    if report["refused"]:
        raise ExtractionError(f"refused members: {report['refused'][:5]}")
    return report


def file_chunks(paths, should_stop=lambda: False):
    """The bytes of ``paths`` in order; ``paths`` may be a generator that waits for the next part to arrive."""
    for path in paths:
        with open(path, "rb") as handle:
            for block in iter(lambda: handle.read(FEED_CHUNK), b""):
                if should_stop():
                    return
                yield block


def staged_files(staging: Path, kind: str) -> dict[str, int]:
    folder = staging / kind
    if not folder.is_dir():
        return {}
    return {p.name: p.stat().st_size for p in folder.iterdir() if p.is_file() and not p.name.endswith(".partial")}


def verify_hdf5(staging: Path, kind: str, names) -> list[str]:
    bad = []
    for name in names:
        with open(staging / kind / name, "rb") as handle:
            if handle.read(8) != HDF5_SIGNATURE:
                bad.append(name)
    return bad


def verify_flow_shard(staging: Path, members: list[str], expected_bytes: int | None) -> dict:
    """``members``: the shard's (selected) episode names; ``expected_bytes``: ``raw_member_bytes`` (None with a
    selection, whose per-member sizes are not published)."""
    files = staged_files(staging, "flows-fs-optimized")
    names = {episode_of(n) for n in files}
    if names != set(members):
        raise ExtractionError(f"flow members differ from the manifest: missing {sorted(set(members) - names)[:3]}, "
                              f"extra {sorted(names - set(members))[:3]}")
    total = sum(files.values())
    if expected_bytes is not None and total != expected_bytes:
        raise ExtractionError(f"flow member bytes {total} != raw_member_bytes {expected_bytes}")
    bad = verify_hdf5(staging, "flows-fs-optimized", files)
    if bad:
        raise ExtractionError(f"not HDF5: {bad[:3]}")
    return {"members": len(files), "bytes": total}


def verify_cameras(staging: Path, members: int, expected_bytes: int) -> dict:
    files = staged_files(staging, "cameras")
    total = sum(files.values())
    if len(files) != members or total != expected_bytes:
        raise ExtractionError(f"cameras: {len(files)} members / {total} bytes != {members} / {expected_bytes}")
    for name in files:
        record = json.loads((staging / "cameras" / name).read_text())
        if str(record.get("uuid", episode_of(name))) != episode_of(name):
            raise ExtractionError(f"{name}: uuid {record.get('uuid')!r} does not name its file")
    return {"members": len(files), "bytes": total}


def verify_depth(staging: Path, flow_episodes: set[str] | None) -> dict:
    files = staged_files(staging, "depth_320x180")
    bad = verify_hdf5(staging, "depth_320x180", files)
    if bad:
        raise ExtractionError(f"not HDF5: {bad[:3]}")
    names = {episode_of(n) for n in files}
    out = {"members": len(files), "bytes": sum(files.values())}
    if flow_episodes is not None:
        out["flow_episodes_without_depth"] = len(flow_episodes - names)
        out["depth_without_flow_episode"] = len(names - flow_episodes)
    return out


def publish(staging: Path, kind: str, final_root: Path) -> int:
    """Rename every staged member of ``kind`` into ``final_root/kind`` (same file system: each rename is atomic);
    idempotent after an interruption. Returns the number moved."""
    source, target = staging / kind, final_root / kind
    target.mkdir(parents=True, exist_ok=True)
    moved = 0
    if source.is_dir():
        for path in sorted(source.iterdir()):
            if path.name.endswith(".partial"):
                continue
            os.replace(path, target / path.name)
            moved += 1
        fd = os.open(target, os.O_RDONLY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)
    return moved


def clear_staging(staging: Path) -> None:
    if staging.exists():
        shutil.rmtree(staging)
