"""Native-depth overlay build reading the RAW recordings through a sha256-verified local mapping.

``build_native_depth_mapped`` is ``s4d.rwlearn.native_depth.build_native_depth`` unchanged (same overlays, same
descriptor, same verification), except that each episode's raw recording is read from a local read-only copy:
``RECORDED_ROOT=LOCAL_ROOT`` (local placement of the 3D-MVP chain, ops/jobs/realworld_local.py; the remote copies live
under RECORDED_ROOT, the ingestion-verified originals under LOCAL_ROOT). Every mapped file must be a regular file
(no symlink) inside LOCAL_ROOT whose sha256 equals the episode's recorded raw identity; the builder re-checks it.

This lives in its own module on purpose: ``native_depth.py`` is in the import closure of already placed real-world
rows, whose launch compares that closure byte for byte against their pinned release. The mapping is installed only for
the duration of one build: ``CompactStore`` inside ``native_depth`` is replaced by a subclass whose ``raw_recording``
returns the verified local path, in this process and in every spawned episode worker (the worker entry point re-installs
it from the ``S4D_NATIVE_DEPTH_RAW_ROOT`` environment variable the parent sets before the pool starts). The overlays'
``raw_recording`` attribute therefore names the local file that was read; their identity is ``raw_sha256``.
"""
from __future__ import annotations

import os
from pathlib import Path

from s4d.realworld.io import sha256_file
from s4d.rwlearn import native_depth as _nd

ENV = "S4D_NATIVE_DEPTH_RAW_ROOT"
_ORIGINAL_STORE = _nd.CompactStore
_ORIGINAL_BUILD = _nd._build_episode


def parse_mapping(mapping: str) -> tuple[Path, Path]:
    recorded, sep, local = str(mapping).partition("=")
    if not sep or not recorded.startswith("/") or not local.startswith("/"):
        raise ValueError("raw root mapping must be RECORDED_ROOT=LOCAL_ROOT (absolute paths)")
    local_root = Path(local)
    if any(p.is_symlink() for p in (local_root, *local_root.parents)) or local_root.resolve(strict=True) != local_root:
        raise ValueError(f"raw root {local_root} must be a real directory path without symlinks")
    if not local_root.is_dir():
        raise ValueError(f"raw root {local_root} is not a directory")
    return Path(recorded), local_root


def mapped_raw(recorded_path: str, want_sha256: str, mapping: str) -> str:
    """The verified local copy of one recorded raw file (never read unverified)."""
    recorded_root, local_root = parse_mapping(mapping)
    source = Path(recorded_path)
    try:
        relative = source.relative_to(recorded_root)
    except ValueError:
        raise ValueError(f"raw recording {source} is not below {recorded_root}") from None
    if ".." in relative.parts or not relative.parts:
        raise ValueError(f"unsafe raw recording path {source}")
    path = local_root / relative
    if path.is_symlink() or not path.is_file() or local_root not in path.resolve(strict=True).parents:
        raise ValueError(f"mapped raw recording {path} is not a regular file inside {local_root}")
    if not want_sha256 or sha256_file(path) != want_sha256:
        raise ValueError(f"mapped raw recording {path} sha256 differs from the recorded {want_sha256 or None}")
    return str(path)


class _MappedCompactStore(_ORIGINAL_STORE):
    def raw_recording(self, ep):
        return mapped_raw(super().raw_recording(ep), self.raw_sha256(ep), os.environ[ENV])


def _build_episode_mapped(job):
    """Episode worker entry point (spawned process): install the mapping, then the unchanged episode build."""
    if not os.environ.get(ENV):
        raise RuntimeError(f"{ENV} is not set in the native depth worker")
    _nd.CompactStore = _MappedCompactStore
    return _ORIGINAL_BUILD(job)


def build_native_depth_mapped(store_path, rig, out_dir, out_store, *, raw_root: str, workers=1, episodes=None) -> dict:
    parse_mapping(raw_root)
    saved = (_nd.CompactStore, _nd._build_episode, os.environ.get(ENV))
    os.environ[ENV] = raw_root
    _nd.CompactStore, _nd._build_episode = _MappedCompactStore, _build_episode_mapped
    try:
        return _nd.build_native_depth(store_path, rig, out_dir, out_store, workers=workers, episodes=episodes)
    finally:
        _nd.CompactStore, _nd._build_episode = saved[0], saved[1]
        if saved[2] is None:
            os.environ.pop(ENV, None)
        else:
            os.environ[ENV] = saved[2]
