"""Atomic writes, completion records and the on-disk layout (originals are never written to).

Layout under an output root (``root``)::

    root/episodes/<episode_id>/normalized/episode.json      contract document (no large arrays)
    root/episodes/<episode_id>/normalized/arrays.npz        trajectory + per-stream timestamps
    root/episodes/<episode_id>/normalized/frames/<stream>.h5  frames as recorded (uint8/uint16, never converted)
    root/episodes/<episode_id>/derived/<stage>/...          stage outputs
    root/episodes/<episode_id>/derived/<stage>/_COMPLETE.json  completion record (written last, atomically)

The source recording is only read (paths + sha256 in the provenance); derived data never lives next to it.
A stage is complete iff its record exists and its fingerprint (stage version + config + input records) matches;
otherwise the stage directory is rebuilt from a fresh temporary directory and swapped in atomically.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
import time
from collections.abc import Callable
from pathlib import Path

import numpy as np

RECORD = "_COMPLETE.json"


def safe_name(stream_id: str) -> str:
    return stream_id.replace("/", "__")


def episode_dir(root: Path, episode_id: str) -> Path:
    return Path(root) / "episodes" / episode_id


def stage_dir(root: Path, episode_id: str, stage: str) -> Path:
    return episode_dir(root, episode_id) / "derived" / stage


def atomic_write_bytes(path: Path, data: bytes) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(data)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    except BaseException:
        if os.path.exists(tmp):
            os.unlink(tmp)
        raise


def atomic_write_json(path: Path, doc) -> None:
    atomic_write_bytes(path, (json.dumps(doc, indent=2, sort_keys=True, allow_nan=False) + "\n").encode())


def atomic_write_npz(path: Path, **arrays) -> None:
    import io

    buf = io.BytesIO()
    np.savez_compressed(buf, **arrays)
    atomic_write_bytes(path, buf.getvalue())


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def fingerprint(obj) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True, default=str).encode()).hexdigest()


def read_record(directory: Path) -> dict | None:
    path = Path(directory) / RECORD
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text())
    except json.JSONDecodeError:
        return None


def run_stage(directory: Path, *, stage: str, version: str, config: dict, inputs: dict,
              build: Callable[[Path], dict], force: bool = False) -> dict:
    """Idempotent, restartable stage execution.

    ``inputs`` maps names to upstream completion-record fingerprints (or file hashes). ``build(tmpdir)`` writes all
    outputs into ``tmpdir`` and returns a summary dict. The directory is swapped in only after ``build`` returns,
    and the record is written last, so an interrupted run leaves either the old complete stage or nothing.
    """
    directory = Path(directory)
    fp = fingerprint({"stage": stage, "version": version, "config": config, "inputs": inputs})
    rec = read_record(directory)
    if rec is not None and rec.get("fingerprint") == fp and not force:
        return {**rec, "skipped": True}
    directory.parent.mkdir(parents=True, exist_ok=True)
    tmp = Path(tempfile.mkdtemp(dir=directory.parent, prefix=f".{directory.name}.building."))
    try:
        started = time.time()
        summary = build(tmp)
        record = {"stage": stage, "version": version, "fingerprint": fp, "config": config, "inputs": inputs,
                  "summary": summary, "started_unix": started, "finished_unix": time.time(),
                  "files": sorted(str(p.relative_to(tmp)) for p in tmp.rglob("*") if p.is_file())}
        atomic_write_json(tmp / RECORD, record)
        old = None
        if directory.exists():
            old = directory.with_name(f".{directory.name}.old.{os.getpid()}")
            os.replace(directory, old)
        os.replace(tmp, directory)
        if old is not None:
            shutil.rmtree(old, ignore_errors=True)
        return {**record, "skipped": False}
    except BaseException:
        shutil.rmtree(tmp, ignore_errors=True)
        raise


def clean_stale_tmp(root: Path) -> list[str]:
    """Remove leftovers of interrupted builds (``.<stage>.building.*``) under ``root``."""
    removed = []
    for p in Path(root).rglob(".*.building.*"):
        if p.is_dir():
            shutil.rmtree(p, ignore_errors=True)
            removed.append(str(p))
    return removed
