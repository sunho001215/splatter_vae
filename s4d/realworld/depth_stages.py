"""Depth stages and depth access on each camera view's pinhole grid (the colour grid of an RGB-D rig).

``sensor_depth``  RGB-D rigs: the device's own depth registered onto the colour grid (z-buffer forward warp with the
                  recorded depth->colour extrinsic). Raw, noisy, metric; holes stay invalid. The robot-grounded
                  calibration's depth evidence.

The lifting/export depth of the own rig is ``foundation_stereo_depth`` (``s4d.realworld.ir_stereo_depth``), written in
the same layout.

Output layout (all depth stages): ``<stage>/<view stream>.h5`` with ``depth`` (N,H,W) float32 metric camera-z (0 invalid),
``valid`` (N,H,W) bool and attrs ``K`` (continuous convention of that grid), ``stream_id``, ``rig_id``, ``grid``.

Multi-view consumers use ``episode_frame_sets``: the training store's M5 sets when present. Legacy synchronized
sets (``frame_sets``): the reference view's frames define set times;
every view contributes its nearest frame within ``max_sync_ns``, and every frame of every view must belong to exactly
one set (otherwise it refuses - views of different instants are never merged silently), or, with ``allow_unmatched``,
mutual nearest frames form sets and the rest is reported unmatched.
"""

from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np

from s4d.realworld import io as rio

STAGE_VERSIONS = {"sensor_depth": "1"}


def configured_sets_policy(root, episode_id, cfg: dict | None = None) -> dict:
    """Multi-view frame-set policy of an episode root for its non-training consumers (viewer, contract checks).

    Criterion M5 (2026-11-06; one source of truth with the training store and its loaders): when the root has a
    training store (``train_data/<episode>/episode.h5``) the sets are the store's mutual-nearest cross-camera sets
    accepted by robot motion -> ``{"criterion": "M5", "max_set_motion_m": train_store.MAX_SET_MOTION_M}``
    (``store_sets``; the loaders' default threshold, so viewer, contract checks and training select the same sets; no
    per-config threshold). ``stages.train_data.max_sync_ns`` is ONLY the per-camera
    robot <-> camera tolerance and is never used as a cross-camera cap. Roots without a training store: the former
    window export's explicit cross-camera tolerance ``stages.export.max_sync_ns`` with mutual pairing (legacy roots);
    otherwise strict 1:1 pairing within a third of a period ({}). ``cfg`` defaults to ``<root>/pipeline.yaml``."""
    from s4d.realworld.train_store import FILE, MAX_SET_MOTION_M

    if cfg is None:
        path = Path(root) / "pipeline.yaml"
        cfg = {}
        if path.is_file():
            import yaml

            cfg = yaml.safe_load(path.read_text()) or {}
    stages = cfg.get("stages") or {}
    if (Path(root) / "train_data" / episode_id / FILE).is_file():
        return {"criterion": "M5", "max_set_motion_m": float(MAX_SET_MOTION_M)}
    export_sync = (stages.get("export") or {}).get("max_sync_ns")
    return {} if export_sync is None else {"max_sync_ns": int(export_sync), "allow_unmatched": True}


def store_sets(root, episode_id, streams: list[str] | None = None, max_set_motion_m: float | None = None) -> dict:
    """The training store's M5 sets in ``frame_sets`` form: a set is a reference frame whose mutual nearest frame of
    EVERY store camera exists (no time cap) and whose robot surface moved <= ``max_set_motion_m`` across its members
    (``sync/surface_disp_m``; default ``train_store.MAX_SET_MOTION_M``). Columns follow ``streams`` (stream ids, e.g.
    ``<serial>/color``; default: the store's camera order); frame indices are the normalized stream frames (the store
    keeps every frame). Frames in no accepted set are listed in ``unmatched``. Refuses stores without per-set motion
    (schema < 1.1.0: re-export ``train_data``), episodes with held camera geometry, and streams the store does not
    hold. Calibration-optional stores retain all RGB cameras but cannot supply multi-view sets to consumers."""
    from s4d.realworld.train_store import FILE, MAX_SET_MOTION_M, CompactStore, sync_counts

    path = Path(root) / "train_data" / episode_id / FILE
    store = CompactStore([path])
    try:
        ep = store.episodes()[0]
        sync = store.sync_sets(ep)  # includes the canonical refusal of stores without per-set robot motion
        ef = store.episode(ep)
        stream_of = {c: str(ef.file[f"cameras/{c}"].attrs["stream_id"]) for c in ef.cameras}
        stamps_of = {stream_of[c]: np.asarray(ef.timestamps(c), np.int64) for c in ef.cameras}
    finally:
        store.close()
    col = {stream_of[c]: v for v, c in enumerate(ef.cameras)}
    order = list(streams) if streams is not None else [stream_of[c] for c in ef.cameras]
    missing = [s for s in order if s not in col]
    if missing:
        raise ValueError(f"{episode_id}: streams {missing} are not in the training store (its cameras: "
                         f"{list(col)})")
    thr = MAX_SET_MOTION_M if max_set_motion_m is None else float(max_set_motion_m)
    frames, disp = np.asarray(sync["frames"]), np.asarray(sync["surface_disp_m"], np.float64)
    accepted = (frames >= 0).all(1) & (disp <= thr)
    index = frames[accepted][:, [col[s] for s in order]].astype(np.int64)
    stamps = np.stack([stamps_of[s][index[:, v]] for v, s in enumerate(order)], 1).reshape(len(index), len(order))
    return {"streams": order, "index": index, "stamps_ns": stamps,
            "spread_ns": stamps.max(1) - stamps.min(1) if len(index) else np.zeros(0, np.int64),
            "criterion": "M5", "max_set_motion_m": thr, "max_sync_ns": None, "surface_disp_m": disp[accepted],
            "reference_rows": np.flatnonzero(accepted), "counts": sync_counts(frames, disp, thr), "store": str(path),
            "unmatched": {s: sorted(set(range(len(stamps_of[s]))) - set(index[:, v].tolist()))
                          for v, s in enumerate(order)}}


def episode_frame_sets(meta, root, episode_id, streams: list[str], cfg: dict | None = None) -> dict:
    """The multi-view sets every non-training consumer of an episode root uses: the store's M5 sets when the root has
    a training store (``configured_sets_policy``), else ``frame_sets`` under the legacy/strict policy."""
    policy = configured_sets_policy(root, episode_id, cfg)
    if policy.get("criterion") == "M5":
        return store_sets(root, episode_id, streams, policy["max_set_motion_m"])
    return frame_sets(meta, streams, **policy)


def depth_file(root: Path, episode_id: str, stage: str, stream_id: str) -> Path:
    return rio.stage_dir(root, episode_id, stage) / f"{rio.safe_name(stream_id)}.h5"


def load_view_depth(root, episode_id, stream_id, indices, stage: str):
    """(depth (n,H,W) float32, valid (n,H,W) bool, K continuous) of one view from a depth stage."""
    with h5py.File(depth_file(root, episode_id, stage, stream_id), "r") as f:
        idx = np.asarray(indices)
        depth = np.stack([f["depth"][i] for i in idx]) if len(idx) else f["depth"][:0]
        valid = np.stack([f["valid"][i] for i in idx]) if len(idx) else f["valid"][:0]
        return depth, valid, np.asarray(f.attrs["K"], float)


def view_K(meta, stream_id: str) -> np.ndarray:
    """Continuous intrinsics of a stream's recorded (zero-distortion) pinhole grid. Streams of stereo camera rigs are
    refused (no rectified grid exists in this pipeline); device IR pairs of RGB-D rigs are not stereo views."""
    if any(stream_id in (g.left_stream, g.right_stream) for g in meta.stereo):
        raise NotImplementedError(f"{stream_id}: stereo camera streams are not processed by this pipeline "
                                  "(RGB-D rigs only)")
    intr = meta.stream(stream_id).intrinsics
    if intr.distortion_model != "none" and any(abs(c) > 0 for c in intr.distortion):
        raise NotImplementedError(f"{stream_id}: distorted stream ({intr.distortion_model}); undistort in the adapter")
    return intr.K()


def _record_build_id(root, episode_id, stage):
    from s4d.realworld.pipeline import _record_fp

    return _record_fp(root, episode_id, stage)


# ---------------------------------------------------------------- sensor depth (RGB-D rigs)


def sensor_depth(root: Path, episode_id: str, runner, *, chunk: int = 32, min_depth_m: float = 0.1,
                 max_depth_m: float = 6.0, splat: int = 2, force: bool = False) -> dict:
    from s4d.realworld.pipeline import load_meta, read_frames

    meta = load_meta(root, episode_id)
    directory = rio.stage_dir(root, episode_id, "sensor_depth")
    config = {"chunk": chunk, "min_depth_m": min_depth_m, "max_depth_m": max_depth_m, "splat": splat,
              "kernel": "register_depth"}
    inputs = {"ingest": _record_build_id(root, episode_id, "ingest")}
    store = rio.ChunkStore(directory, rio.stage_fingerprint("sensor_depth", STAGE_VERSIONS["sensor_depth"], config,
                                                            inputs))
    if force:
        store.clear()

    def build(tmp: Path) -> dict:
        summary = {}
        for g in meta.rgbd:
            if g.status != "metric":
                summary[g.color_stream] = {"skipped": f"RGB-D geometry {g.status}"}
                continue
            cs, ds = meta.stream(g.color_stream), meta.stream(g.depth_stream)
            K_c, K_d = view_K(meta, g.color_stream), view_K(meta, g.depth_stream)
            hw = (cs.intrinsics.height, cs.intrinsics.width)
            n = len(meta.stream_timestamps_ns[g.color_stream])
            stats = []
            with h5py.File(tmp / f"{rio.safe_name(g.color_stream)}.h5", "w") as f:
                f.attrs.update(K=K_c, stream_id=g.color_stream, rig_id=g.rig_id, depth_stream=g.depth_stream,
                               grid="colour sensor pinhole grid (zero distortion)", T_color_depth=g.T(),
                               depth_convention="metric camera-z of the colour camera, 0 invalid",
                               source="sensor depth registered by z-buffer forward warp (holes not filled)")
                dd = f.create_dataset("depth", (n, *hw), np.float32, chunks=(1, *hw), compression="lzf")
                vv = f.create_dataset("valid", (n, *hw), bool, chunks=(1, *hw), compression="lzf")
                for start in range(0, n, chunk):
                    idx = list(range(start, min(n, start + chunk)))
                    cid = f"{g.color_stream}/{start:05d}-{idx[-1]:05d}"
                    got = store.get(cid)
                    if got is None:
                        out, kmeta, handle = runner.kernel(
                            "register_depth",
                            {"color_hw": list(hw), "min_depth_m": min_depth_m, "max_depth_m": max_depth_m,
                             "depth_scale_m": ds.depth_scale_m, "splat": splat},
                            {"depth_raw": read_frames(root, episode_id, g.depth_stream, idx), "K_depth": K_d,
                             "K_color": K_c, "T_color_depth": g.T()}, chunk_id=cid)
                        store.put(cid, out, kmeta)
                        runner.ack(handle)
                        got = (out, kmeta)
                    out, kmeta = got
                    dd[idx[0]:idx[-1] + 1] = out["depth"]
                    vv[idx[0]:idx[-1] + 1] = out["valid"]
                    stats += kmeta["frames"]
            summary[g.color_stream] = {
                "frames": n, "sensor_valid_fraction_mean": float(np.mean([s["sensor_valid_fraction"] for s in stats])),
                "registered_valid_fraction_mean": float(np.mean([s["registered_valid_fraction"] for s in stats])),
                "rejected_above_max_total": int(sum(s["rejected_above_max"] for s in stats)),
                "rejected_below_min_total": int(sum(s["rejected_below_min"] for s in stats))}
        summary["runner"] = runner.describe()
        return summary

    rec = rio.run_stage(directory, stage="sensor_depth", version=STAGE_VERSIONS["sensor_depth"], config=config,
                        inputs=inputs, build=build, force=force)
    store.cleanup()
    return rec


# ---------------------------------------------------------------- synchronized frame sets


def frame_sets(meta, streams: list[str], max_sync_ns: int | None = None, allow_unmatched: bool = False) -> dict:
    """Group the views' frames into synchronized sets (reference = first stream). Raises unless every frame of every
    view belongs to exactly one set within ``max_sync_ns`` (default: a third of the reference frame period).

    ``allow_unmatched`` (recordings whose cameras are not hardware-synchronized or dropped a frame): a set is a
    reference frame whose MUTUAL nearest frame in every other view lies within ``max_sync_ns`` (must stay below half
    a frame period so matches are unique); every other frame is listed in ``unmatched[stream]`` and never merged."""
    from s4d.data.droid.pointworld import nearest_timestamps

    ref = np.asarray(meta.stream_timestamps_ns[streams[0]])
    if max_sync_ns is None:
        max_sync_ns = int(np.median(np.diff(ref)) // 3) if len(ref) > 1 else 0
    if allow_unmatched:
        period = float(np.median(np.diff(ref))) if len(ref) > 1 else float("inf")
        if max_sync_ns >= period / 2:
            raise ValueError(f"max_sync_ns {max_sync_ns} >= half the frame period: matches would not be unique")
        keep = np.ones(len(ref), bool)
        idx = {}
        for s in streams:
            stamps = np.asarray(meta.stream_timestamps_ns[s])
            j = nearest_timestamps(stamps, ref)  # nearest stream frame for each reference frame
            back = nearest_timestamps(ref, stamps)  # nearest reference frame for each stream frame
            keep &= (np.abs(stamps[j] - ref) <= max_sync_ns) & (back[j] == np.arange(len(ref)))
            idx[s] = j
        index = np.stack([idx[s][keep] for s in streams], 1)
        stamps = np.stack([np.asarray(meta.stream_timestamps_ns[s])[index[:, v]] for v, s in enumerate(streams)], 1)
        unmatched = {s: sorted(set(range(len(meta.stream_timestamps_ns[s]))) - set(index[:, v].tolist()))
                     for v, s in enumerate(streams)}
        return {"streams": streams, "index": index, "stamps_ns": stamps, "spread_ns": stamps.max(1) - stamps.min(1),
                "max_sync_ns": max_sync_ns, "unmatched": unmatched}
    index, offset = {}, {}
    for s in streams:
        stamps = np.asarray(meta.stream_timestamps_ns[s])
        idx = nearest_timestamps(stamps, ref)
        err = np.abs(stamps[idx] - ref)
        if (err > max_sync_ns).any() or len(np.unique(idx)) != len(idx) or len(idx) != len(stamps):
            raise ValueError(f"{s}: frames do not pair 1:1 with {streams[0]} within {max_sync_ns} ns "
                             f"(max offset {int(err.max())} ns, {len(np.unique(idx))} unique of {len(stamps)} frames); "
                             "multi-view sets would merge different instants")
        index[s], offset[s] = idx, stamps[idx] - ref
    stamps = np.stack([np.asarray(meta.stream_timestamps_ns[s])[index[s]] for s in streams], 1)
    return {"streams": streams, "index": np.stack([index[s] for s in streams], 1), "stamps_ns": stamps,
            "spread_ns": stamps.max(1) - stamps.min(1), "max_sync_ns": max_sync_ns}
