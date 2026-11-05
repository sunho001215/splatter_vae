"""Depth stages on each camera view's pinhole grid (the colour grid of an RGB-D rig, the left eye of a stereo rig).

``sensor_depth``      RGB-D rigs: the device's own depth registered onto the colour grid (z-buffer forward warp with
                      the recorded depth->colour extrinsic). Raw, noisy, metric; holes stay invalid. Used by the
                      robot-grounded calibration and as MapAnything's metric depth input.
``mapanything_rig_depth`` Stereo rigs: one rectified pair with known metric rig-frame poses, before world calibration.
``mapanything_depth`` MapAnything (facebookresearch/map-anything) on synchronized multi-view frame sets: per set,
                      every view's image, intrinsics, calibrated pose (``calibrate`` stage, never MapAnything's own
                      pose estimate) and, when available, the sensor depth as metric input. Its predicted camera-z
                      depth (plus validity and confidence) is the depth every later stage lifts with.

Output layout (all depth stages): ``<stage>/<view stream>.h5`` with ``depth`` (N,H,W) float32 metric camera-z (0 invalid),
``valid`` (N,H,W) bool and attrs ``K`` (continuous convention of that grid), ``stream_id``, ``rig_id``, ``grid``;
MapAnything adds ``conf``, ``non_ambiguous``, ``set_index``, ``joint`` and per-frame pose diagnostics.

Synchronized sets: the reference view's frames define set times; every view contributes its nearest frame within
``max_sync_ns``, and every frame of every view must belong to exactly one set (otherwise the stage refuses - views
of different robot-motion instants are never merged silently). A set whose robot moved more than
``max_set_motion_m`` (largest displacement of the URDF surface between the set's earliest and latest timestamp) is
processed view by view instead of as one static scene, and recorded as such.
"""

from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np

from s4d.realworld import io as rio

STAGE_VERSIONS = {"sensor_depth": "1", "mapanything_depth": "1", "mapanything_rig_depth": "1"}
DEPTH_STAGES = ("mapanything_depth", "mapanything_rig_depth", "sensor_depth")  # + configured variants (same layout)


def sets_policy(root, episode_id, stage: str = "mapanything_depth") -> dict:
    """``frame_sets`` keyword arguments the episode's MapAnything stage used (single source of truth for reports and
    viewers): ``max_sync_ns`` and ``allow_unmatched`` from its record, defaults when absent."""
    rec = rio.read_record(rio.stage_dir(root, episode_id, stage))
    cfg = (rec or {}).get("config", {})
    out = {}
    if cfg.get("allow_unmatched"):
        out = {"max_sync_ns": int(cfg["max_sync_ns"]), "allow_unmatched": True}
    return out


def depth_file(root: Path, episode_id: str, stage: str, stream_id: str) -> Path:
    return rio.stage_dir(root, episode_id, stage) / f"{rio.safe_name(stream_id)}.h5"


def load_view_depth(root, episode_id, stream_id, indices, stage: str = "mapanything_depth"):
    """(depth (n,H,W) float32, valid (n,H,W) bool, K continuous) of one view from a depth stage."""
    with h5py.File(depth_file(root, episode_id, stage, stream_id), "r") as f:
        idx = np.asarray(indices)
        depth = np.stack([f["depth"][i] for i in idx]) if len(idx) else f["depth"][:0]
        valid = np.stack([f["valid"][i] for i in idx]) if len(idx) else f["valid"][:0]
        return depth, valid, np.asarray(f.attrs["K"], float)


def view_K(meta, stream_id: str, *, root=None, episode_id=None) -> np.ndarray:
    """Intrinsics of the shared grid; stereo intrinsics come from the completed rectification stage."""
    rig = next((g for g in meta.stereo if stream_id in (g.left_stream, g.right_stream)), None)
    if rig is not None:
        if root is None or episode_id is None:
            raise ValueError(f"{stream_id}: stereo grid intrinsics need root and episode_id")
        path = rio.stage_dir(root, episode_id, "stereo_rectify") / f"{rio.safe_name(rig.rig_id)}.h5"
        with h5py.File(path, "r") as f:
            return np.asarray(f.attrs["K_left" if stream_id == rig.left_stream else "K_right"], float)
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


def set_robot_motion(meta, stamps_ns: np.ndarray, geom) -> np.ndarray:
    """Largest URDF-surface displacement (m) of the robot between each set's earliest and latest timestamp; ``inf``
    (unknown, never merged) when either end lies outside the robot trajectory or beyond its sync tolerance."""
    from s4d.realworld.pipeline import robot_state_at

    lo, hi = stamps_ns.min(1), stamps_ns.max(1)
    q0, g0, ok0, _ = robot_state_at(meta, lo.astype(np.float64))
    q1, g1, ok1, _ = robot_state_at(meta, hi.astype(np.float64))
    motion = np.zeros(len(lo))
    for k in range(len(lo)):
        if not (ok0[k] and ok1[k]):
            motion[k] = np.inf
            continue
        if hi[k] == lo[k]:
            continue
        a, _ = geom.surface.points(geom.q(q0[k], g0[k]))
        b, _ = geom.surface.points(geom.q(q1[k], g1[k]))
        motion[k] = float(np.linalg.norm(a - b, axis=1).max())
    return motion


# ---------------------------------------------------------------- MapAnything


class MapAnythingBackend:
    """MapAnything through an isolated worker (``mapanything_worker.py``) reached by a runner."""

    def __init__(self, spec, runner, params: dict, allow_mock: bool = False):
        self.spec, self.runner, self.params, self.allow_mock = spec, runner, dict(params), allow_mock
        self.identity = {"worker": spec.name}

    def config(self) -> dict:
        from dataclasses import asdict

        spec = asdict(self.spec)
        spec.pop("timeout_s", None)
        spec.pop("env", None)
        return {"class": type(self).__name__, "spec": spec, "params": self.params, "allow_mock": self.allow_mock}


def mapanything_depth(root: Path, episode_id: str, backend: MapAnythingBackend, *, chunk: int = 4,
                      input_depth_stage: str | None = "auto", max_sync_ns: int | None = None,
                      max_set_motion_m: float = 0.005, include_right_eyes: bool = True, grouping: str = "all",
                      allow_unmatched: bool = False, output_stage: str = "mapanything_depth",
                      force: bool = False) -> dict:
    """``grouping="all"``: every view of a synchronized set in one MapAnything scene (requires all cameras to be
    synchronized within ``max_sync_ns``). ``"per_rig"``: each rig's views (a stereo pair, or one RGB-D colour view)
    form their own posed scenes, for recordings whose rigs are not synchronized with each other (e.g. DROID's ZED
    cameras); views of different instants are never merged in either mode."""
    from s4d.realworld.contract import CameraView
    from s4d.realworld.pipeline import RobotGeometry, accepted_calibration, load_meta

    meta = load_meta(root, episode_id)
    if meta.stereo and not backend.params.get("use_pose", True):
        raise ValueError("stereo mapanything_depth requires use_pose=True for calibrated geometry")
    views, poses = [], {}
    for view in meta.views():
        if view.kind == "stereo" and meta.rig_stereo(view.rig_id).status != "metric":
            continue
        views.append(view)
        poses[view.stream_id] = accepted_calibration(root, episode_id, view.stream_id)
        if view.kind == "stereo" and include_right_eyes:
            g = meta.rig_stereo(view.rig_id)
            views.append(CameraView(g.rig_id, g.right_stream, "stereo"))
            T = poses[g.left_stream]
            poses[g.right_stream] = None if T is None else rectified_rig_pose(root, episode_id, g.rig_id) @ T
    streams = [v.stream_id for v in views]
    if not streams:
        raise ValueError("mapanything_depth needs at least one metric view")
    if input_depth_stage == "auto":
        input_depth_stage = "sensor_depth" if meta.rgbd and not meta.stereo else None
    missing = [s for s, T in poses.items() if T is None]
    if missing:
        raise RuntimeError(f"mapanything_depth needs an accepted calibration for every view; missing/rejected: {missing}")
    if grouping not in ("all", "per_rig"):
        raise ValueError(f"grouping {grouping!r} not in ('all', 'per_rig')")
    if grouping == "all":
        groups = [("", list(range(len(streams))))]
    else:
        rigs = list(dict.fromkeys(v.rig_id for v in views))
        groups = [(rig, [i for i, v in enumerate(views) if v.rig_id == rig]) for rig in rigs]
    geom = RobotGeometry(meta)
    plans = []
    for name, members in groups:
        sets = frame_sets(meta, [streams[i] for i in members], max_sync_ns, allow_unmatched=allow_unmatched)
        motion = set_robot_motion(meta, sets["stamps_ns"], geom)
        plans.append({"name": name, "members": members, "sets": sets, "motion": motion,
                      "joint": motion <= max_set_motion_m})
        for m in members:  # frames without a synchronized partner: single-view, never merged
            extra = sets.get("unmatched", {}).get(streams[m], [])
            if extra:
                st = np.asarray(meta.stream_timestamps_ns[streams[m]])[extra][:, None]
                plans.append({"name": f"unmatched_{rio.safe_name(streams[m])}", "members": [m],
                              "sets": {"streams": [streams[m]], "index": np.asarray(extra)[:, None], "stamps_ns": st,
                                       "spread_ns": np.zeros(len(extra), np.int64), "max_sync_ns": 0},
                              "motion": np.zeros(len(extra)), "joint": np.zeros(len(extra), bool)})
    Ks = np.stack([view_K(meta, s, root=root, episode_id=episode_id) for s in streams])
    c2w = np.stack([np.linalg.inv(poses[s]) for s in streams])
    hw = {(meta.stream(s).intrinsics.height, meta.stream(s).intrinsics.width) for s in streams}
    if len(hw) != 1:
        raise NotImplementedError(f"views have different grids {hw}")
    (H, W), = hw
    n_frames = {s: len(meta.stream_timestamps_ns[s]) for s in streams}
    directory = rio.stage_dir(root, episode_id, output_stage)
    inputs = {"ingest": _record_build_id(root, episode_id, "ingest"),
              "calibrate": _record_build_id(root, episode_id, "calibrate")}
    if meta.stereo:
        inputs["stereo_rectify"] = _record_build_id(root, episode_id, "stereo_rectify")
    if input_depth_stage:
        inputs[input_depth_stage] = _record_build_id(root, episode_id, input_depth_stage)
    config = {"chunk": chunk, "input_depth_stage": input_depth_stage, "max_sync_ns": plans[0]["sets"]["max_sync_ns"],
              "max_set_motion_m": max_set_motion_m, "streams": streams, "backend": backend.config()}
    if meta.stereo:
        config["include_right_eyes"] = include_right_eyes
    if grouping != "all":  # (absent in the default mode: earlier fingerprints stay valid)
        config["grouping"] = grouping
    if allow_unmatched:  # likewise only recorded when used
        config["allow_unmatched"] = True
    if output_stage != "mapanything_depth":
        config["output_stage"] = output_stage
    store = rio.ChunkStore(directory, rio.stage_fingerprint("mapanything_depth", STAGE_VERSIONS["mapanything_depth"],
                                                            config, inputs))
    if force:
        store.clear()

    def build(tmp: Path) -> dict:
        files = {}
        for v, s in enumerate(streams):
            n = n_frames[s]
            plan = next(p for p in plans if v in p["members"])
            f = h5py.File(tmp / f"{rio.safe_name(s)}.h5", "w")
            f.attrs.update(K=Ks[v], stream_id=s, rig_id=views[v].rig_id,
                           grid=("stereo rectified pinhole grid" if views[v].kind == "stereo"
                                 else "view pinhole grid (as recorded)"),
                           depth_convention="metric camera-z, 0 invalid", T_cam_base=poses[s],
                           source="MapAnything predicted depth_z with calibrated poses as input")
            _create_prediction_planes(f, n, (H, W))
            for name, shape, dtype in (("input_depth_valid", (n, H, W), bool),):
                f.create_dataset(name, shape, dtype, chunks=(1, H, W), compression="lzf")
            _create_prediction_pose(f, n)
            f.create_dataset("set_index", data=np.arange(len(plan["sets"]["index"])))
            f.create_dataset("set_joint", data=plan["joint"])
            files[s] = f
        identity, metas = None, []
        try:
            for plan in plans:
                sets, joint, members = plan["sets"], plan["joint"], plan["members"]
                n = len(sets["index"])
                for start in range(0, n, chunk):
                    ks = list(range(start, min(n, start + chunk)))
                    cid = (f"{plan['name']}/" if plan["name"] else "") + f"sets/{start:05d}-{ks[-1]:05d}"
                    got = store.get(cid)
                    if got is None:
                        images = np.stack([np.stack([
                            model_view_frames(root, episode_id, meta, streams[m], [int(sets["index"][k, j])])[0]
                            for j, m in enumerate(members)]) for k in ks])
                        arrays = {"images": images, "K": Ks[members], "cam2world": c2w[members], "joint": joint[ks]}
                        if input_depth_stage:
                            arrays["depth"] = np.stack([np.stack([
                                load_view_depth(root, episode_id, streams[m], [int(sets["index"][k, j])],
                                                input_depth_stage)[0][0]
                                for j, m in enumerate(members)]) for k in ks])
                        resp, out, handle = backend.runner.worker(backend.spec, "infer_views", backend.params, arrays,
                                                                  chunk_id=cid, allow_mock=backend.allow_mock)
                        rmeta = {"model": resp["model"], "meta": resp["meta"], "transport": resp.get("transport")}
                        if input_depth_stage:
                            out = {**out, "input_depth_valid": arrays["depth"] > 0}
                        store.put(cid, out, rmeta)
                        backend.runner.ack(handle)
                        got = (out, rmeta)
                    out, rmeta = got
                    identity = rmeta["model"]
                    metas.append(rmeta["meta"])
                    for jk, k in enumerate(ks):
                        for j, m in enumerate(members):
                            i = int(sets["index"][k, j])
                            f = files[streams[m]]
                            _write_prediction(f, i, out, jk, j, ("depth", "valid", "conf", "non_ambiguous"))
                            if "input_depth_valid" in out:
                                f["input_depth_valid"][i] = out["input_depth_valid"][jk, j]
                            _write_prediction(f, i, out, jk, j, ("cam2world_pred", "metric_scale"))
        finally:
            for f in files.values():
                f.close()
        backend.identity = identity or backend.identity
        peak = max((m.get("max_memory_allocated", 0) for m in metas), default=0)

        def describe(plan):
            sets, joint, motion = plan["sets"], plan["joint"], plan["motion"]
            known = motion[np.isfinite(motion)]  # inf = robot state unknown at a set's time (never merged)
            out = {"sets": int(len(sets["index"])), "joint_sets": int(joint.sum()),
                   "single_view_sets": int((~joint).sum()), "max_set_spread_ns": int(sets["spread_ns"].max()),
                   "set_robot_motion_m": {"max": float(known.max()) if len(known) else None,
                                          "median": float(np.median(known)) if len(known) else None},
                   "sets_detail": {"stamps_ns": sets["stamps_ns"].tolist(),
                                   "robot_motion_m": [float(m) if np.isfinite(m) else None for m in motion]}}
            if len(known) < len(motion):
                out["unknown_robot_motion_sets"] = int(len(motion) - len(known))
            return out

        common = {"model": identity, "peak_cuda_allocated_bytes": int(peak),
                  "preprocess_checks": metas[-1].get("checks") if metas else None}
        if grouping == "all":
            d = describe(plans[0])
            return {"views": streams, **{k: d[k] for k in ("sets", "joint_sets", "single_view_sets",
                                                           "max_set_spread_ns", "set_robot_motion_m")},
                    **common, "sets_detail": d["sets_detail"]}
        return {"views": streams, "grouping": grouping, "groups": {p["name"]: describe(p) for p in plans}, **common}

    rec = rio.run_stage(directory, stage="mapanything_depth", version=STAGE_VERSIONS["mapanything_depth"],
                        config=config, inputs=inputs, build=build, force=force)
    store.cleanup()
    return rec


def rectified_rig_pose(root, episode_id, rig_id: str) -> np.ndarray:
    """Right-from-left transform of the actual shared image grids, in metres."""
    path = rio.stage_dir(root, episode_id, "stereo_rectify") / f"{rio.safe_name(rig_id)}.h5"
    with h5py.File(path, "r") as f:
        return np.asarray(f.attrs["T_rr_rl"], float)


def model_view_frames(root, episode_id, meta, stream_id, indices):
    """MapAnything RGB input; stereo eyes use the shared grid (IR replication is model input only)."""
    from s4d.realworld.pipeline import _model_input_rgb, grid_frames, read_frames, to_model_rgb

    g = next((g for g in meta.stereo if stream_id in (g.left_stream, g.right_stream)), None)
    if g is None:  # RGB-D colour stream: RGB order for the model (a BGR stream is flipped, as for the tracker)
        return to_model_rgb(read_frames(root, episode_id, stream_id, indices), meta.stream(stream_id))
    eye = "left" if stream_id == g.left_stream else "right"
    return _model_input_rgb(to_model_rgb(grid_frames(root, episode_id, meta, g, indices, eye), meta.stream(stream_id)))


def mapanything_rig_depth(root: Path, episode_id: str, backend: MapAnythingBackend, *,
                          force: bool = False) -> dict:
    """Metric stereo depth before world calibration: one two-view scene per rig and time, in the left rig frame.

    Both eyes are saved in the common depth layout; calibration reads only the primary (left) view.
    Known rig poses and intrinsics are required inputs, never replaced by model estimates.
    """
    from contextlib import ExitStack

    from s4d.realworld.pipeline import load_meta

    meta = load_meta(root, episode_id)
    directory = rio.stage_dir(root, episode_id, "mapanything_rig_depth")
    if not backend.params.get("use_pose", True):
        raise ValueError("mapanything_rig_depth requires use_pose=True for metric rig geometry")
    params = {**backend.params, "use_pose": True, "use_depth": False}
    config = {"backend": backend.config(), "params": params}
    inputs = {s: _record_build_id(root, episode_id, s) for s in ("ingest", "stereo_rectify")}
    store = rio.ChunkStore(directory, rio.stage_fingerprint("mapanything_rig_depth",
                                                            STAGE_VERSIONS["mapanything_rig_depth"], config, inputs))
    if force:
        store.clear()

    def build(tmp):
        summary = {}
        for g in meta.stereo:
            streams = [g.left_stream, g.right_stream]
            if g.status != "metric":
                for stream in streams:
                    summary[stream] = {"skipped": f"stereo geometry {g.status}: no metric depth (scale never invented)"}
                continue
            sets = frame_sets(meta, streams, max_sync_ns=1_000_000)
            n = len(sets["index"])
            Ks = np.stack([view_K(meta, stream, root=root, episode_id=episode_id) for stream in streams])
            poses = np.stack([np.eye(4), rectified_rig_pose(root, episode_id, g.rig_id)])
            c2w = np.linalg.inv(poses)
            intr = meta.stream(g.left_stream).intrinsics
            hw = (intr.height, intr.width)
            identity = backend.identity
            with ExitStack() as stack:
                files = [stack.enter_context(h5py.File(tmp / f"{rio.safe_name(stream)}.h5", "w")) for stream in streams]
                for v, (stream, f) in enumerate(zip(streams, files)):
                    f.attrs.update(K=Ks[v], stream_id=stream, rig_id=g.rig_id, grid="stereo rectified pinhole grid",
                                   depth_convention="metric camera-z, 0 invalid", T_cam_rig=poses[v],
                                   source="MapAnything predicted depth_z with metric rig poses as input")
                    _create_prediction_planes(f, n, hw)
                    _create_prediction_pose(f, n)
                for k, indices in enumerate(sets["index"]):
                    cid = f"rig/{g.rig_id}/{k:05d}"
                    got = store.get(cid)
                    if got is None:
                        images = np.stack([model_view_frames(root, episode_id, meta, stream, [int(i)])[0]
                                           for stream, i in zip(streams, indices)])[None]
                        resp, out, handle = backend.runner.worker(
                            backend.spec, "infer_views", params,
                            {"images": images, "K": Ks, "cam2world": c2w, "joint": np.ones(1, bool)},
                            chunk_id=cid, allow_mock=backend.allow_mock)
                        rmeta = {"model": resp["model"], "meta": resp["meta"]}
                        store.put(cid, out, rmeta)
                        backend.runner.ack(handle)
                        got = (out, rmeta)
                    out, rmeta = got
                    identity = rmeta["model"]
                    for v, f in enumerate(files):
                        i = int(indices[v])
                        _write_prediction(f, i, out, 0, v,
                                          ("depth", "valid", "conf", "non_ambiguous", "cam2world_pred", "metric_scale"))
            backend.identity = identity
            for stream in streams:
                summary[stream] = {"frames": n, "model": identity, "rig_id": g.rig_id,
                                   "model_input": "infrared replicated to 3 channels (model input only)"
                                   if meta.stream(stream).modality == "infrared" else "rgb"}
        return summary

    rec = rio.run_stage(directory, stage="mapanything_rig_depth", version=STAGE_VERSIONS["mapanything_rig_depth"],
                        config=config, inputs=inputs, build=build, force=force)
    store.cleanup()
    return rec


def _create_prediction_planes(f, n, hw):
    for name, dtype in (("depth", np.float32), ("valid", bool), ("conf", np.float16), ("non_ambiguous", bool)):
        f.create_dataset(name, (n, *hw), dtype, chunks=(1, *hw), compression="lzf")


def _create_prediction_pose(f, n):
    f.create_dataset("cam2world_pred", (n, 4, 4), np.float32)
    f.create_dataset("metric_scale", (n,), np.float32)


def _write_prediction(f, i, out, set_index, view_index, names):
    for name in names:
        f[name][i] = out[name][set_index, view_index]
