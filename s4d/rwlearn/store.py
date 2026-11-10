"""Abstract per-camera frame store that every real-world learning adapter reads (baselines + FQL), and the loader-side
indexes built from it. Data-format policy (user, 2026-11-06 03:00): training frames are stored ONCE per camera and
timestamp (centre square crop -> 128 x 128 RGB uint8; depth / validity / semantic masks at the same resolution,
lossless; calibration, timestamps and the crop affine); temporal windows, cross-camera synchronisation and viewpoints
are assembled in the loader. The compact store itself is designed elsewhere; it plugs in by implementing
``FrameStore`` (and registering an opener with ``register_store``).

Contract of a ``FrameStore`` (all arrays numpy):

* ``episodes()`` -> episode ids; ``task(ep)`` -> task name; ``cameras(ep)`` -> camera ids (sorted serials);
* ``timestamps(ep, cam)`` -> (F,) int64 ns (recorder host CLOCK_MONOTONIC, the raw ``cameras/cam_<serial>/t``);
  adapter normalization can omit colour frames without stereo records, so store indices need not equal raw indices;
  ``segments(ep, cam)`` -> (F,) int recording segment of each frame (pauses between segments);
* ``rgb(ep, cam, idx)`` -> (n, H, W, 3) uint8 RGB; ``depth(ep, cam, idx)`` -> (n, H, W) float32 metres, 0 invalid
  (camera z on the colour grid); ``semantic(ep, cam, idx)`` -> (n, H, W) uint8 SNeRL classes (0 background,
  1 robot, 2 object, 3 fixture), only if ``has_semantic``;
* ``intrinsics(ep, cam)`` -> (3, 3) K of the STORED grid in continuous pixel coordinates (pixel i covers [i, i+1));
  ``w2c(ep, cam)`` -> (4, 4) OpenCV, world = robot base (``panda_link0``), the episode's accepted calibration;
* ``raw_recording(ep)`` -> path of the read-only ``panda_teleop_episode`` file (actions for FQL);
* optional ``raw_sha256(ep)`` -> the recorded source hash (checked against the manifest when present);
* ``image_hw`` -> (H, W) of every stored image;
* optional ``upstream(ep)`` -> {stage: build id} of the preprocessing outputs the episode was built from (at least
  ``calibrate``); geometry-dependent consumers need it to prove the store carries the VALIDATED calibration
  (``s4d.rwlearn.gate.check_store_calibration``); a store without it is refused for them;
* optional ``sync_sets(ep)`` -> ``{"cameras", "frames" (F_ref,V), "surface_disp_m" (F_ref,)}``: the cross-camera sets
  with their robot motion (criterion M5, ``sync_index``); the compact store requires schema >= 1.1.0 for it.

Loader-side indexes (``sync_index``, ``camera_runs``) are pure functions of timestamps and segments: nothing derived
from them is written back.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, runtime_checkable

import numpy as np

SEMANTIC_CLASSES = ("background", "robot", "object", "fixture")
TRAIN_HW = (128, 128)


@runtime_checkable
class FrameStore(Protocol):
    image_hw: tuple[int, int]
    has_semantic: bool

    def episodes(self) -> list[str]: ...
    def task(self, ep: str) -> str: ...
    def cameras(self, ep: str) -> list[str]: ...
    def timestamps(self, ep: str, cam: str) -> np.ndarray: ...
    def segments(self, ep: str, cam: str) -> np.ndarray: ...
    def rgb(self, ep: str, cam: str, idx) -> np.ndarray: ...
    def depth(self, ep: str, cam: str, idx) -> np.ndarray: ...
    def semantic(self, ep: str, cam: str, idx) -> np.ndarray: ...
    def intrinsics(self, ep: str, cam: str) -> np.ndarray: ...
    def w2c(self, ep: str, cam: str) -> np.ndarray: ...
    def raw_recording(self, ep: str) -> str: ...


# ----------------------------------------------------------------------------------------------- loader indexes
def runs_from(times_ns: np.ndarray, segment: np.ndarray, max_gap_factor: float = 1.5,
              period_ns: float | None = None) -> np.ndarray:
    """(F,) int32 run id: a new run starts at a recording-segment change (pause) or where consecutive frames are more
    than ``max_gap_factor`` nominal periods apart (dropped frame). Every temporal window stays inside one run."""
    times_ns = np.asarray(times_ns, np.int64)
    if len(times_ns) < 2:
        return np.zeros(len(times_ns), np.int32)
    d = np.diff(times_ns)
    same = np.diff(np.asarray(segment)) == 0
    period = period_ns if period_ns is not None else float(np.median(d[same] if same.any() else d))
    brk = (~same) | (d > max_gap_factor * period)
    return np.concatenate([[0], np.cumsum(brk)]).astype(np.int32)


MAX_SET_MOTION_M = 0.005  # criterion M5 (s4d.realworld.train_store.MAX_SET_MOTION_M)


@dataclass
class SyncIndex:
    """Synchronised multi-camera time steps of one episode: step s uses frame ``frames[s, v]`` of camera v; camera 0
    (sorted serials) is the reference whose timestamps define the step times. Built in the loader."""

    cameras: list[str]
    frames: np.ndarray  # (S, V) native frame indices, strictly increasing per camera
    times: np.ndarray  # (S,) reference timestamps (ns)
    latest_times: np.ndarray  # (S,) the LATEST constituent frame time of each step (causal observation time)
    segment: np.ndarray  # (S,)
    run: np.ndarray  # (S,) contiguous runs (no pause, no dropped / unsynchronised step inside a run)
    max_sync_error_ns: dict
    criterion: str = "M5"  # "M5": robot-motion sets of the store; "time": synthetic stores without robot motion
    counts: dict | None = None


def sync_index(store: FrameStore, ep: str, max_sync_ns: int = 30_000_000, *,
               max_set_motion_m: float = MAX_SET_MOTION_M) -> SyncIndex:
    """Synchronised multi-view steps, criterion M5 (coordinator decision 2026-11-06, PREP2's sync review): the MUTUAL
    nearest frame of every camera to each reference frame, without a time cap (the cameras are free-running with
    constant phase offsets up to half a period), accepted iff the robot surface moved at most ``max_set_motion_m``
    across the set's timestamps. Stores that hold the per-set robot motion (``sync_sets``: the compact training store,
    schema >= 1.1.0) are REQUIRED to use it; a compact store without it refuses. Only stores that cannot know the robot
    motion (``MemoryStore`` / ``ProcessedEpisodeStore``: tests, synthetic data) fall back to the former time bound
    ``max_sync_ns`` on the mutual nearest pairs (``criterion`` "time"). Steps whose frames lie in different
    recording segments are dropped. Single-view consumers never use this (``transitions.timelines(multiview=False)``,
    DynaMo windows): they need only each camera's own robot <-> camera sync."""
    from s4d.realworld.train_store import mutual_sync

    cams = list(store.cameras(ep))
    stamps = {c: np.asarray(store.timestamps(ep, c), np.int64) for c in cams}
    ref = stamps[cams[0]]
    sets = store.sync_sets(ep) if hasattr(store, "sync_sets") else None
    if sets is not None:
        order = [sets["cameras"].index(c) for c in cams]
        matched = np.asarray(sets["frames"], np.int64)[:, order]
        disp = np.asarray(sets["surface_disp_m"], np.float64)
        complete = (matched >= 0).all(1)
        keep = complete & (disp <= max_set_motion_m)
        criterion = "M5"
        counts = {"complete_sets": int(complete.sum()), "accepted": int(keep.sum()),
                  "max_set_motion_m": float(max_set_motion_m)}
    else:
        matched, _ = mutual_sync([stamps[c] for c in cams], None)
        matched = matched.astype(np.int64)
        complete = (matched >= 0).all(1)
        err = np.stack([np.where(matched[:, v] >= 0, np.abs(stamps[c][np.maximum(matched[:, v], 0)] - ref), 0)
                        for v, c in enumerate(cams)], 1)
        keep = complete & (err <= max_sync_ns).all(1)
        criterion = "time"
        counts = {"complete_sets": int(complete.sum()), "accepted": int(keep.sum()), "max_sync_ns": int(max_sync_ns)}
    safe = np.maximum(matched, 0)
    err = np.stack([np.abs(stamps[c][safe[:, v]] - ref) for v, c in enumerate(cams)], 1)
    seg_all = [np.asarray(store.segments(ep, c)) for c in cams]
    keep &= np.all([seg_all[v][safe[:, v]] == seg_all[0][safe[:, 0]] for v in range(len(cams))], 0)
    k = np.flatnonzero(keep)
    seg = seg_all[0][matched[k, 0]]
    period = float(np.median(np.diff(ref))) if len(ref) > 1 else None
    latest = np.max([stamps[c][matched[k, v]] for v, c in enumerate(cams)], axis=0) if len(k) else ref[k]
    return SyncIndex(cameras=cams, frames=matched[k], times=ref[k], latest_times=latest, segment=seg,
                     run=runs_from(ref[k], seg, period_ns=period),
                     max_sync_error_ns={c: int(err[k, v].max()) if len(k) else None for v, c in enumerate(cams)},
                     criterion=criterion, counts=counts)


def camera_runs(store: FrameStore, ep: str, cam: str) -> np.ndarray:
    return runs_from(store.timestamps(ep, cam), store.segments(ep, cam))


def history(run: np.ndarray, k: int, length: int, spacing: int = 1) -> list[int]:
    """Indices of a ``length``-frame history ending at k (oldest first), ``spacing`` apart, padded with the first
    frame of k's run (frame-stack reset at the start of a recording run, as at an episode start)."""
    start = int(np.flatnonzero(run == run[k])[0])
    return [max(k - j * spacing, start) for j in range(length - 1, -1, -1)]


# ----------------------------------------------------------------------------------------------- registry
_OPENERS: dict[str, Callable[[dict], FrameStore]] = {}


def register_store(kind: str, opener: Callable[[dict], FrameStore]) -> None:
    _OPENERS[kind] = opener


def open_store(locator) -> FrameStore:
    """``locator``: a FrameStore, a dict ``{"kind": ..., ...}``, or a path to a JSON/YAML file holding that dict."""
    if isinstance(locator, FrameStore):
        return locator
    spec = locator
    if not isinstance(spec, dict):
        text = Path(locator).read_text()
        spec = json.loads(text) if str(locator).endswith(".json") else __import__("yaml").safe_load(text)
    kind = spec.get("kind")
    if kind not in _OPENERS:
        raise KeyError(f"no FrameStore opener registered for kind {kind!r} (known: {sorted(_OPENERS)})")
    return _OPENERS[kind](spec)


# ----------------------------------------------------------------------------------------------- implementations
class MemoryStore:
    """In-memory store (tests and synthetic fixtures)."""

    def __init__(self, episodes: dict, image_hw=TRAIN_HW):
        """``episodes[ep] = {"task", "raw", "cameras": {cam: {"t", "segment", "rgb", "depth", "semantic"?, "K",
        "w2c"}}}``."""
        self._eps = episodes
        self.image_hw = tuple(image_hw)
        self.has_semantic = all("semantic" in c for e in episodes.values() for c in e["cameras"].values())

    def episodes(self):
        return sorted(self._eps)

    def task(self, ep):
        return self._eps[ep]["task"]

    def cameras(self, ep):
        return sorted(self._eps[ep]["cameras"])

    def _c(self, ep, cam):
        return self._eps[ep]["cameras"][cam]

    def timestamps(self, ep, cam):
        return np.asarray(self._c(ep, cam)["t"], np.int64)

    def segments(self, ep, cam):
        return np.asarray(self._c(ep, cam)["segment"])

    def rgb(self, ep, cam, idx):
        return np.asarray(self._c(ep, cam)["rgb"])[np.asarray(idx)]

    def depth(self, ep, cam, idx):
        return np.asarray(self._c(ep, cam)["depth"], np.float32)[np.asarray(idx)]

    def semantic(self, ep, cam, idx):
        return np.asarray(self._c(ep, cam)["semantic"])[np.asarray(idx)]

    def intrinsics(self, ep, cam):
        return np.asarray(self._c(ep, cam)["K"], np.float64)

    def w2c(self, ep, cam):
        return np.asarray(self._c(ep, cam)["w2c"], np.float64)

    def raw_recording(self, ep):
        return self._eps[ep].get("raw")

    def raw_sha256(self, ep):
        return self._eps[ep].get("raw_sha256", "")

    def upstream(self, ep):
        return dict(self._eps[ep].get("upstream") or {})

    def sync_sets(self, ep):
        """Optional stored M5 sets (``{"cameras", "frames", "surface_disp_m"}``) of a synthetic episode; None when the
        episode carries no robot motion (``sync_index`` then uses the time fallback)."""
        sets = self._eps[ep].get("sync_sets")
        return None if sets is None else dict(sets)


class ProcessedEpisodeStore:
    """Reference implementation over the PREPROCESSING outputs (``s4d.realworld`` roots: normalized colour frames,
    a metric depth stage on the colour grid, ``sam3_masks``, accepted ``calibrate``), cropped and resized on access
    with EXACTLY the compact store's rule (``s4d.realworld.train_store``; data-format policy 2026-11-06): centred square
    crop (``s4d.data.droid.largescale.crop.center_box``, 480 x 480 of 640 x 480) -> 128 x 128 with
    ``crop_rgb`` (PIL bilinear, antialiased) for RGB, ``crop_depth`` (nearest native pixel at each output pixel centre)
    for depth and semantics, depth quantised to the store's 0.1 mm uint16 (0 invalid), K through ``crop_intrinsics``.
    For tests on real slices and as an executable specification; it writes nothing."""

    DEPTH_UNIT_M = 1e-4

    def __init__(self, roots: dict, *, depth_stage: str = "foundation_stereo_depth", image_hw=TRAIN_HW,
                 require_semantic: bool = True):
        """``roots``: {task: (processed root, [episode ids])}; episode keys are ``<task>/<episode id>``."""
        from s4d.realworld.pipeline import load_meta

        if image_hw[0] != image_hw[1]:
            raise ValueError("the training store is square")
        self.image_hw, self.depth_stage = tuple(image_hw), depth_stage
        self._eps, self._meta = {}, {}
        for task, (root, eps) in roots.items():
            for ep in eps:
                key = f"{task}/{ep}"
                self._eps[key] = (task, Path(root), ep)
                self._meta[key] = load_meta(Path(root), ep)
        from s4d.realworld import io as rio

        self.has_semantic = all(rio.read_record(rio.stage_dir(r, e, "sam3_masks")) is not None
                                for _, r, e in self._eps.values())
        if require_semantic and not self.has_semantic:
            raise RuntimeError("sam3_masks missing for some episode")
        self._K, self._w2c = {}, {}

    def episodes(self):
        return sorted(self._eps)

    def task(self, ep):
        return self._eps[ep][0]

    def _sid(self, ep, cam):
        return next(g.color_stream for g in self._meta[ep].rgbd if g.rig_id == cam)

    def cameras(self, ep):
        return sorted(g.rig_id for g in self._meta[ep].rgbd)

    def timestamps(self, ep, cam):
        return np.asarray(self._meta[ep].stream_timestamps_ns[self._sid(ep, cam)], np.int64)

    def segments(self, ep, cam):
        notes = json.loads(self._meta[ep].provenance.notes or "{}")
        starts = np.rint(np.asarray(notes.get("segments", {}).get("t_start", []), np.float64) * 1e9).astype(np.int64)
        t = self.timestamps(ep, cam)
        return (np.searchsorted(starts, t, side="right") - 1) if len(starts) else np.zeros(len(t), np.int64)

    def _box(self, ep, cam):
        from s4d.data.droid.largescale.crop import center_box

        s = self._meta[ep].stream(self._sid(ep, cam))
        return center_box(s.intrinsics.height, s.intrinsics.width)

    def rgb(self, ep, cam, idx):
        from s4d.data.droid.largescale.crop import crop_rgb
        from s4d.realworld.pipeline import read_frames

        _, root, e = self._eps[ep]
        frames = read_frames(root, e, self._sid(ep, cam), np.atleast_1d(idx))
        box = self._box(ep, cam)
        return np.stack([crop_rgb(f, box, self.image_hw[0]) for f in frames]).astype(np.uint8)

    def depth(self, ep, cam, idx):
        from s4d.data.droid.largescale.crop import crop_depth
        from s4d.realworld.pipeline import calibrated_view_depth

        _, root, e = self._eps[ep]  # depth x the accepted calibration's per-camera scale (consistent with w2c)
        d, K = calibrated_view_depth(root, e, self._meta[ep], self._sid(ep, cam), np.atleast_1d(idx), self.depth_stage)
        self._remember_K(ep, cam, K)
        box = self._box(ep, cam)
        out = np.stack([crop_depth(x, box, self.image_hw[0]) for x in d]).astype(np.float64)
        q = np.rint(out / self.DEPTH_UNIT_M)  # the compact store's quantize_depth: float64 rint, valid never 0
        q = np.where(out > 0, np.maximum(q, 1), 0)
        if q.max(initial=0) > 65535:
            raise ValueError("depth beyond the uint16 0.1 mm range")
        return q.astype(np.uint16).astype(np.float32) * np.float32(self.DEPTH_UNIT_M)

    def semantic(self, ep, cam, idx):
        import h5py

        from s4d.data.droid.largescale.crop import crop_depth
        from s4d.realworld import io as rio

        _, root, e = self._eps[ep]
        sid = self._sid(ep, cam)
        with h5py.File(rio.stage_dir(root, e, "sam3_masks") / f"{rio.safe_name(sid)}.h5", "r") as f:
            sem = np.stack([f["semantic"][int(i)] for i in np.atleast_1d(idx)])
        box = self._box(ep, cam)  # nearest native pixel at each output pixel centre (exact for class ids)
        return np.stack([crop_depth(s.astype(np.float32), box, self.image_hw[0]) for s in sem]).astype(np.uint8)

    def _remember_K(self, ep, cam, K_native):
        from s4d.data.droid.largescale.crop import crop_intrinsics

        self._K.setdefault((ep, cam), crop_intrinsics(np.asarray(K_native, np.float64), self._box(ep, cam),
                                                      self.image_hw[0]))

    def intrinsics(self, ep, cam):
        if (ep, cam) not in self._K:
            self.depth(ep, cam, [0])
        return self._K[(ep, cam)]

    def w2c(self, ep, cam):
        if (ep, cam) not in self._w2c:
            from s4d.realworld.pipeline import accepted_calibration

            _, root, e = self._eps[ep]
            T = accepted_calibration(root, e, self._sid(ep, cam))
            if T is None:
                raise RuntimeError(f"{ep}/{cam}: no ACCEPTED calibration")
            self._w2c[(ep, cam)] = np.asarray(T, np.float64)
        return self._w2c[(ep, cam)]

    def raw_recording(self, ep):
        prov = self._meta[ep].provenance
        return str(Path(prov.source_root) / prov.source_files[0].path) if prov.source_files else ""

    def raw_sha256(self, ep):
        prov = self._meta[ep].provenance
        return str(prov.source_files[0].sha256 or "") if prov.source_files else ""

    def upstream(self, ep):
        from s4d.realworld import io as rio

        _, root, e = self._eps[ep]
        rec = rio.read_record(rio.stage_dir(root, e, "calibrate"))
        return {"calibrate": rio.build_id(rec)} if rec else {}


register_store("processed", lambda spec: ProcessedEpisodeStore(
    {t: (r, list(eps)) for t, (r, eps) in spec["roots"].items()},
    depth_stage=spec.get("depth_stage", "foundation_stereo_depth")))


def _open_compact(spec: dict) -> FrameStore:
    """The compact training store (``s4d.realworld.train_store.CompactStore``, designed by the format work; implements
    ``FrameStore`` as-is). Imported lazily: it lives on its own branch until the coordinator merges it."""
    import importlib

    try:
        module = importlib.import_module("s4d.realworld.train_store")
    except ImportError as exc:
        raise RuntimeError("store kind 'compact' needs s4d.realworld.train_store (format branch not merged)") from exc
    if "depth_source" in spec:
        from s4d.rwlearn.native_depth import NativeDepthStore

        return NativeDepthStore(spec)
    return module.CompactStore(spec["paths"])


register_store("compact", _open_compact)


def split_episodes(store: FrameStore, validation_fraction: float = 0.1, seed: int = 0) -> dict:
    """Per task: a seeded permutation of its sorted episodes, the first ceil(fraction * n) are validation (none when a
    task has a single episode); a store with several tasks gets the union of the per-task splits. Preserve actual
    store keys and refuse missing/duplicate raw provenance so one recording cannot enter both partitions."""
    import hashlib

    from s4d.rwlearn.identity import match_recordings, store_identities

    identities = store_identities(store)
    match_recordings(identities, identities)

    out = {"train": [], "validation": [], "seed": int(seed), "validation_fraction": float(validation_fraction)}
    by_task: dict[str, list[str]] = {}
    for ep in store.episodes():
        by_task.setdefault(store.task(ep), []).append(ep)
    for task, eps in sorted(by_task.items()):
        eps = sorted(eps)
        rng = np.random.default_rng([int(seed), int(hashlib.sha256(task.encode()).hexdigest()[:8], 16)])
        perm = [eps[i] for i in rng.permutation(len(eps))]
        n_val = int(np.ceil(validation_fraction * len(eps))) if len(eps) >= 2 else 0
        out["validation"] += sorted(perm[:n_val])
        out["train"] += sorted(perm[n_val:])
    return out


def scene_statistics(store: FrameStore, episodes: list[str] | None = None, *, frames_per_camera: int = 8,
                     near_far_m: tuple[float, float] = (0.1, 2.5), bounds=None) -> dict:
    """Measurements the baselines' scene parameters are chosen from (docs/realworld_baselines.md): least-squares
    intersection of the optical axes (SNeRL origin) and camera distances; depth percentiles and the fraction inside
    SinCro's near/far and SNeRL's official near/far after normalisation; with ``bounds`` the fraction of each semantic
    class's lifted points inside the 3D-MVP scene bounds."""
    eps = episodes or store.episodes()
    C, D, depths, inside = [], [], [], {c: [0, 0] for c in range(len(SEMANTIC_CLASSES))}
    for ep in eps:
        for cam in store.cameras(ep):
            c2w = np.linalg.inv(store.w2c(ep, cam))
            C.append(c2w[:3, 3])
            D.append(c2w[:3, 2])
            n = len(store.timestamps(ep, cam))
            pick = np.linspace(0, n - 1, min(frames_per_camera, n)).round().astype(int)
            dep = store.depth(ep, cam, pick)
            K = store.intrinsics(ep, cam)
            sem = store.semantic(ep, cam, pick) if store.has_semantic else None
            for j, d in enumerate(dep):
                valid = d > 0
                depths.append(d[valid])
                if bounds is not None and sem is not None:
                    ys, xs = np.nonzero(valid)
                    z = d[valid]
                    p = np.stack([(xs + 0.5 - K[0, 2]) / K[0, 0] * z, (ys + 0.5 - K[1, 2]) / K[1, 1] * z, z,
                                  np.ones_like(z)], 1) @ c2w.T
                    b = np.asarray(bounds, np.float64)
                    ok = ((p[:, :3] >= b[:3]) & (p[:, :3] <= b[3:])).all(1)
                    cls = sem[j][ys, xs]
                    for c in inside:
                        inside[c][0] += int(ok[cls == c].sum())
                        inside[c][1] += int((cls == c).sum())
    C, D = np.array(C), np.array(D)
    A, bvec = np.zeros((3, 3)), np.zeros(3)
    for c, d in zip(C, D):
        P = np.eye(3) - np.outer(d, d)
        A += P
        bvec += P @ c
    lookat = np.linalg.solve(A, bvec)
    dist = np.linalg.norm(C - lookat, axis=1)
    depth = np.concatenate(depths) if depths else np.zeros(0)
    scale = 4.0 / float(np.median(dist))
    out = {"episodes": len(eps), "lookat_m": lookat.round(4).tolist(),
           "camera_distance_m": {"median": float(np.median(dist)), "min": float(dist.min()), "max": float(dist.max())},
           "snerl_pose_scale_for_distance_4": scale, "snerl_official_near_far_m": [2.0 / scale, 6.0 / scale],
           "depth_m_percentiles": dict(zip(("p1", "p5", "p50", "p95", "p99"),
                                           np.percentile(depth, [1, 5, 50, 95, 99]).round(3).tolist())),
           "fraction_in_snerl_near_far": float(((depth >= 2 / scale) & (depth <= 6 / scale)).mean()),
           "fraction_in_sincro_near_far": float(((depth >= near_far_m[0]) & (depth <= near_far_m[1])).mean())}
    if bounds is not None:
        out["fraction_inside_bounds_by_class"] = {SEMANTIC_CLASSES[c]: round(a / max(n, 1), 4)
                                                  for c, (a, n) in inside.items()}
    return out
