"""Native-depth overlay build for a rig accepted for OFFLINE use by its task's acceptance override.

``build_native_depth_override`` is ``s4d.rwlearn.native_depth.build_native_depth`` unchanged (same overlays,
descriptor and verification), except that the per-episode native correction is read with
``calib.acceptance_override.load_rig_session_offline`` instead of the strict deployment reader
``rwlearn.deploy.realsense.load_rig_session``: a camera "validated_by_override" under the document's committed
acceptance override is accepted, and every correction's evidence names the override. Robot deployment keeps the strict
reader.

Like ``native_depth_mapped`` this lives in its own module: ``native_depth.py`` and ``deploy/realsense.py`` are in the
import closure of placed rows. The reader is installed only for the duration of one build, in this process and in
every spawned episode worker (the worker entry point re-installs it; ``S4D_NATIVE_DEPTH_ACCEPTANCE_OVERRIDE`` marks the
build). An optional ``raw_root`` composes with ``native_depth_mapped``'s sha256-verified local raw mapping.
"""
from __future__ import annotations

import os

from s4d.rwlearn import native_depth as _nd
from s4d.rwlearn import native_depth_mapped as _mapped

ENV = "S4D_NATIVE_DEPTH_ACCEPTANCE_OVERRIDE"
_ORIGINAL_BUILD = _nd._build_episode


def _build_episode_override(job):
    """Episode worker entry point (spawned process): install the offline reader (and the raw mapping when the parent
    set one), then the unchanged episode build."""
    from s4d.realworld.calib.acceptance_override import load_rig_session_offline

    if os.environ.get(ENV) != "1":
        raise RuntimeError(f"{ENV} is not set in the native depth worker")
    previous = _nd.load_rig_session
    _nd.load_rig_session = load_rig_session_offline
    try:
        return _mapped._build_episode_mapped(job) if os.environ.get(_mapped.ENV) else _ORIGINAL_BUILD(job)
    finally:  # a pooled worker process may be reused: leave its module state as found
        _nd.load_rig_session = previous


def build_native_depth_override(store_path, rig, out_dir, out_store, *, workers=1, episodes=None,
                                raw_root: str | None = None) -> dict:
    from s4d.realworld.calib.acceptance_override import load_rig_session_offline, offline_acceptance
    from s4d.realworld.calib.rig import load_rig

    doc, _, _ = load_rig(rig)
    state = offline_acceptance(doc)
    if not state["accepted"] or state["acceptance_override"] is None:
        raise ValueError(f"{rig}: not a rig accepted by a valid acceptance override ({state['problems']}); "
                         "use the ordinary native depth build")
    if raw_root:
        _mapped.parse_mapping(raw_root)
    saved = (_nd.load_rig_session, _nd._build_episode, _nd.CompactStore, os.environ.get(ENV), os.environ.get(_mapped.ENV))
    os.environ[ENV] = "1"
    _nd.load_rig_session, _nd._build_episode = load_rig_session_offline, _build_episode_override
    if raw_root:
        os.environ[_mapped.ENV] = raw_root
        _nd.CompactStore = _mapped._MappedCompactStore
    try:
        return _nd.build_native_depth(store_path, rig, out_dir, out_store, workers=workers, episodes=episodes)
    finally:
        _nd.load_rig_session, _nd._build_episode, _nd.CompactStore = saved[0], saved[1], saved[2]
        for name, value in ((ENV, saved[3]), (_mapped.ENV, saved[4])):
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value
