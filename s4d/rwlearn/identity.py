"""Join episode records by task and recorded raw provenance, never by processed episode names."""

from __future__ import annotations

import re
from pathlib import Path


def raw_identity(task, recording, sha256=None) -> dict:
    if not isinstance(task, str) or not task or not isinstance(recording, str) or not recording.strip():
        raise RuntimeError("missing raw identity: task and raw_recording are required")
    name = Path(recording).name
    if name in ("", ".", ".."):
        raise RuntimeError(f"missing raw identity: {recording!r}")
    sha256 = sha256 or ""
    if not isinstance(sha256, str) or (sha256 and re.fullmatch(r"[0-9a-f]{64}", sha256) is None):
        raise RuntimeError(f"invalid raw identity sha256 for {task}/{name}")
    return {"task": task, "raw_recording": recording, "raw_sha256": sha256}


def match_recordings(expected: dict, actual: dict, *, complete: bool = True) -> dict[str, str]:
    """Return actual key -> expected key. Relocated paths may match by basename within a task; every recorded
    hash must agree. Duplicate names/content and absent identities fail closed, even when episode keys agree."""
    def index(records):
        names, hashes, identities = {}, {}, {}
        for key, rec in records.items():
            r = raw_identity(rec.get("task"), rec.get("raw_recording"), rec.get("raw_sha256"))
            name = (r["task"], Path(r["raw_recording"]).name)
            sha = (r["task"], r["raw_sha256"])
            if name in names or (sha[1] and sha in hashes):
                raise RuntimeError(f"duplicate raw identity for {key}")
            names[name], identities[key] = key, r
            if sha[1]:
                hashes[sha] = key
        return names, identities

    expected_names, expected_ids = index(expected)
    _, actual_ids = index(actual)
    matched = {}
    for key, rec in actual_ids.items():
        name = (rec["task"], Path(rec["raw_recording"]).name)
        other = expected_names.get(name)
        if other is None:
            raise RuntimeError(f"{key}: raw identity not found in the validated episode set: {name}")
        wanted_sha, got_sha = expected_ids[other]["raw_sha256"], rec["raw_sha256"]
        if wanted_sha and got_sha and wanted_sha != got_sha:
            raise RuntimeError(f"{key}: raw identity sha256 differs from {other}")
        matched[key] = other
    if complete and set(matched.values()) != set(expected):
        raise RuntimeError("store episodes differ from the validated set: missing raw identity")
    return matched


def store_identities(store, episodes=None) -> dict:
    """Read provenance through the frame-store interface; the hash is optional on older stores."""
    sha256 = getattr(store, "raw_sha256", None)
    episodes = list(store.episodes() if episodes is None else episodes)
    if len(episodes) != len(set(episodes)):
        raise RuntimeError("duplicate store episode key")
    return {ep: raw_identity(store.task(ep), store.raw_recording(ep), sha256(ep) if sha256 else None)
            for ep in episodes}
