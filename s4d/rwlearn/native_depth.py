"""Opt-in native RealSense depth overlays for compact 3D-MVP training stores.

Only depth changes: alignment, colour-camera z, native scale and crop/resampling are
the deployment replay implementation. Inputs are read-only; publication is atomic.
"""
from __future__ import annotations

import hashlib
import json
import multiprocessing
import os
import tempfile
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import h5py
import numpy as np
import yaml

from s4d.realworld.io import atomic_write_bytes, sha256_file
from s4d.realworld.train_store import CompactStore
from s4d.rwlearn.deploy.contracts import NATIVE_DEPTH, validate_native_correction
from s4d.rwlearn.deploy.realsense import load_rig_session
from s4d.rwlearn.deploy.replay import RecordedEpisode
from s4d.rwlearn.store import TRAIN_HW

FORMAT = "rwlearn_native_depth"
VERSION = 1


def _verify(f, store, ep, rig_sha256, corrections=None) -> dict:
    """Verify overlay identity against its compact episode, without needing raw files at training time."""
    expected = {"format": FORMAT, "version": VERSION, "episode_id": ep,
                "raw_sha256": store.raw_sha256(ep), "rig_document_sha256": rig_sha256,
                "alignment": "rs_align", "depth_source": NATIVE_DEPTH}
    for key, value in expected.items():
        if not value or f.attrs.get(key) != value:
            raise ValueError(f"{ep}: native depth overlay {key} mismatch")
    counts = json.loads(f.attrs["frame_counts"])
    corr = json.loads(f.attrs["native_depth_correction"])
    cams = store.cameras(ep)
    if set(f["cameras"]) != set(cams) or set(counts) != set(cams) or set(corr) != set(cams):
        raise ValueError(f"{ep}: native depth overlay cameras mismatch")
    if corrections is not None and corr != corrections:
        raise ValueError(f"{ep}: native depth overlay correction mismatch")
    for cam in cams:
        validate_native_correction(corr[cam])
        if (corr[cam].get("evidence") or {}).get("rig_calibration") != rig_sha256:
            raise ValueError(f"{ep}/{cam}: native depth overlay correction rig mismatch")
        g, t = f[f"cameras/{cam}"], store.timestamps(ep, cam)
        if counts[cam] != len(t) or g["depth"].shape != (len(t), *store.image_hw):
            raise ValueError(f"{ep}/{cam}: native depth overlay frame count mismatch")
        if g["depth"].dtype != np.dtype("float32"):
            raise ValueError(f"{ep}/{cam}: native depth overlay must contain float32 metres")
        if not np.array_equal(g["timestamps_ns"][:], t):
            raise ValueError(f"{ep}/{cam}: native depth overlay timestamps mismatch")
    return {**expected, "frame_counts": counts, "native_depth_correction": corr}


class NativeDepthStore(CompactStore):
    """CompactStore with only ``depth`` overridden. Identity is checked once per episode per process.

    Overlay handles are short-lived; the base store keeps its existing bounded, fork-safe LRU.
    """
    depth_source = NATIVE_DEPTH

    def __init__(self, spec):
        if spec["depth_source"] != NATIVE_DEPTH:
            raise ValueError(f"unsupported compact depth_source {spec['depth_source']!r}")
        paths, overlays = spec["paths"], spec["depth_overlays"]
        if not isinstance(paths, list) or len(paths) != len(overlays):
            raise ValueError("native depth needs one overlay per compact episode path, in the same order")
        super().__init__(paths)
        if self.image_hw != TRAIN_HW:
            raise ValueError("native depth overlays require the 128x128 training grid")
        self.rig_document_sha256 = spec["rig_document_sha256"]
        self.depth_overlays = dict(zip(self._files, map(str, overlays)))
        for path in self.depth_overlays.values():
            if not Path(path).is_file():
                raise ValueError(f"missing native depth overlay: {path}")
        self._verified = {}

    def __getstate__(self):
        state = super().__getstate__()
        state["_verified"] = {}
        return state

    def _verify_episode(self, f, ep):
        key = (os.getpid(), ep)
        if key not in self._verified:
            self._verified[key] = _verify(f, self, ep, self.rig_document_sha256)
        return self._verified[key]

    def depth(self, ep, cam, idx):
        with h5py.File(self.depth_overlays[ep], "r") as f:
            self._verify_episode(f, ep)
            ds = f[f"cameras/{cam}/depth"]
            idx = np.atleast_1d(np.asarray(idx, np.int64))
            if len(idx) and (idx.min() < 0 or idx.max() >= len(ds)):
                raise IndexError(f"{ep}/{cam}: native depth frame index out of range")
            # Preserve duplicates and arbitrary order, as EpisodeFile._frames does.
            out = np.empty((len(idx), *self.image_hw), np.float32)
            for k, i in enumerate(idx):
                ds.read_direct(out, np.s_[int(i)], np.s_[k])
            return out

    def depth_provenance(self, episodes):
        overlays = {}
        for ep in episodes:
            with h5py.File(self.depth_overlays[ep], "r") as f:
                overlays[ep] = {"path": self.depth_overlays[ep], **self._verify_episode(f, ep)}
        return {"rig_document_sha256": self.rig_document_sha256, "overlays": overlays}


def _build_episode(job):
    path, overlay, rig, rig_sha256 = job
    store = CompactStore([path])
    ep, = store.episodes()
    try:
        raw = store.raw_recording(ep)
        if not store.raw_sha256(ep) or sha256_file(raw) != store.raw_sha256(ep):
            raise ValueError(f"{ep}: raw_sha256 mismatch against recomputed raw file sha256")
        rec = RecordedEpisode(raw)
        cams = store.cameras(ep)
        if set(cams) != set(rec.cameras()):
            raise ValueError(f"{ep}: store and recording cameras differ")
        for cam in cams:
            if not np.array_equal(store.timestamps(ep, cam), rec.timestamps_ns(cam)):
                raise ValueError(f"{ep}/{cam}: store and recording frame count/timestamps differ")
        corr = load_rig_session(rig, cams, {c: rec.K_native(c) for c in cams})["native_depth_correction"]
        if any(c["evidence"]["rig_calibration"] != rig_sha256 for c in corr.values()):
            raise ValueError("rig document changed during native depth build")
        overlay = Path(overlay)
        if overlay.exists():
            with h5py.File(overlay, "r") as f:
                _verify(f, store, ep, rig_sha256, corr)
            return "skipped"
        counts = {c: len(store.timestamps(ep, c)) for c in cams}
        fd, tmp = tempfile.mkstemp(prefix=f".{overlay.name}.", suffix=".partial", dir=overlay.parent)
        os.close(fd)
        try:
            with h5py.File(tmp, "w") as f:
                f.attrs.update(format=FORMAT, version=VERSION, episode_id=ep, raw_sha256=store.raw_sha256(ep),
                               raw_recording=str(raw), rig_document_sha256=rig_sha256, alignment="rs_align",
                               depth_source=NATIVE_DEPTH, frame_counts=json.dumps(counts),
                               native_depth_correction=json.dumps(corr))
                for cam in cams:
                    g = f.create_group(f"cameras/{cam}")
                    g.create_dataset("timestamps_ns", data=store.timestamps(ep, cam))
                    ds = g.create_dataset("depth", (counts[cam], *TRAIN_HW), dtype="float32",
                                          chunks=(1, *TRAIN_HW), compression="lzf")
                    for i in range(counts[cam]):
                        ds[i] = rec.frame(cam, i, depth=True, correction=corr[cam], alignment="rs_align").depth
                _verify(f, store, ep, rig_sha256, corr)
            Path(tmp).replace(overlay)
        finally:
            Path(tmp).unlink(missing_ok=True)
        return "built"
    finally:
        store.close()


def build_native_depth(store_path, rig, out_dir, out_store, *, workers=1, episodes=None) -> dict:
    """Build/resume overlays. A subset stages episodes; publish a descriptor only when ALL overlays verify.

    Existing invalid overlays refuse (remove/rebuild them explicitly). Raw hashes are recomputed even on resume.
    Spawned episode workers inherit single-thread BLAS/OpenMP settings before importing numpy/scipy.
    """
    store_path, rig, out_dir, out_store = map(Path, (store_path, rig, out_dir, out_store))
    spec = yaml.safe_load(store_path.read_text())
    if spec.get("kind") != "compact" or "depth_source" in spec or not isinstance(spec.get("paths"), list):
        raise ValueError("native depth build requires an ordinary compact descriptor with an episode paths list")
    if workers < 1:
        raise ValueError("workers must be >= 1")
    protected = {store_path.resolve(), rig.resolve(), rig.with_suffix(".reference.npz").resolve()}
    store = CompactStore(spec["paths"])
    try:
        if store.image_hw != TRAIN_HW:
            raise ValueError("native depth overlays require the 128x128 training grid")
        eps = list(store._files)  # descriptor order, not the sorted public episode order
        selected = eps if episodes is None else list(episodes)
        if len(set(selected)) != len(selected) or set(selected) - set(eps):
            raise ValueError("episodes must be unique compact-store episode keys (task/episode_id)")
        protected.update(Path(ef.path).resolve() for ef in store._files.values())
        protected.update(Path(store.raw_recording(ep)).resolve() for ep in eps)
        overlays = [str((out_dir / f"{hashlib.sha256(ep.encode()).hexdigest()[:24]}.h5").resolve()) for ep in eps]
    finally:
        store.close()
    if out_store.resolve() in protected or any(Path(p) in protected or Path(p) == out_store.resolve()
                                              for p in overlays):
        raise ValueError("native depth outputs must not overwrite inputs or each other")
    rig_sha256 = sha256_file(rig)
    out_dir.mkdir(parents=True, exist_ok=True)
    jobs = [(p, o, str(rig.resolve()), rig_sha256) for ep, p, o in zip(eps, spec["paths"], overlays) if ep in selected]
    thread_vars = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS",
                   "VECLIB_MAXIMUM_THREADS")
    previous = {k: os.environ.get(k) for k in thread_vars}
    try:
        os.environ.update(dict.fromkeys(thread_vars, "1"))
        with ProcessPoolExecutor(max_workers=min(workers, max(1, len(jobs))),
                                 mp_context=multiprocessing.get_context("spawn")) as pool:
            results = list(pool.map(_build_episode, jobs))
    finally:
        for k, v in previous.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
    output_spec = {**spec, "depth_source": NATIVE_DEPTH, "depth_overlays": overlays,
                   "rig_document_sha256": rig_sha256}
    complete = all(Path(p).is_file() for p in overlays)
    if complete:
        native = NativeDepthStore(output_spec)
        try:
            native.depth_provenance(eps)
        finally:
            native.close()
        atomic_write_bytes(out_store, yaml.safe_dump(output_spec, sort_keys=False).encode())
    return {"built": results.count("built"), "skipped": results.count("skipped"),
            "descriptor_written": complete, "out_store": str(out_store)}
