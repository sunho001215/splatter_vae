"""TrackEverything single-view tracks in the existing track2d/lift3d chunk contract.

This opt-in stage replaces both tracking and lifting in the supplied output root. It
never invokes the multiview demo, interleaves views, or estimates geometry. Existing
AllTracker stages and all downstream consumers keep their current behavior.
"""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
from PIL import Image

from s4d.realworld import io as rio
from s4d.realworld.geometry import project, transform_points
from s4d.realworld.pipeline import (
    _record_fp,
    accepted_calibration,
    chunk_queries,
    load_meta,
    to_model_rgb,
    view_depth,
    view_frames,
)
from s4d.realworld.tracking import NOT_TRACKED, OCCLUDED, OUT, STATUS, VALID, query_grid

VERSION = "1-trackeverything"


def recording_segments(timestamps_ns):
    """Half-open frame intervals; no clip crosses a gap >1.5 median camera periods."""
    stamps = np.asarray(timestamps_ns, np.int64)
    if not len(stamps):
        return []
    dt = np.diff(stamps)
    if (dt <= 0).any():
        raise ValueError("camera timestamps must be strictly increasing")
    cuts = np.flatnonzero(dt > 1.5 * np.median(dt)) + 1 if len(dt) else np.empty(0, int)
    edges = [0, *cuts.tolist(), len(stamps)]
    return list(zip(edges[:-1], edges[1:]))


def clip_plan(timestamps_ns, window, step):
    """Assign each *global* AllTracker query to one <=64-frame single-segment clip.

    Small segments use one clip. Longer ones overlap starting at the first query
    whose clipped window would not fit in the previous clip. Query frames never
    restart at a pause; chunk_queries remains the single source of query identities.
    """
    windows = chunk_queries(len(timestamps_ns), window, step) if len(timestamps_ns) else []
    clips = []
    for lo, hi in recording_segments(timestamps_ns):
        pending = [(q, idx) for q, idx in windows if lo <= q < hi]
        start = max(lo, min(pending[0][0] + window, hi) - 64) if pending else lo
        while pending:
            end = min(start + 64, hi)
            take = [(q, idx) for q, idx in pending if min(q + window, hi) <= end]
            if not take:
                raise ValueError("query window does not fit in a 64-frame clip")
            clips.append((start, end, hi, take))
            pending = pending[len(take):]
            if pending:
                start = pending[0][0]
    return clips


def resize_inputs(video, depth, K, model_hw):
    """One continuous affine: x'=sx*x, y'=sy*y; RGB Lanczos, depth exact nearest.

    For 480x640 -> 384x512, both first rows of K scale by exactly 0.8. K has
    continuous pixel centres already: upstream utils/common_utils.py:214-218 uses
    arange+0.5. No half-pixel correction is needed. use_input_resolution=True
    avoids demo_hf.py:583-585's (res-1)/(orig-1) scaling. Original query centres
    undergo this same affine, without rounding them to a different resized ray.
    """
    h, w = model_hw
    H, W = video.shape[1:3]
    scale = np.array([w / W, h / H])
    images = np.stack([np.asarray(Image.fromarray(frame).resize((w, h), Image.Resampling.LANCZOS)) for frame in video])
    rows = np.minimum(((np.arange(h) + 0.5) * H / h).astype(int), H - 1)
    cols = np.minimum(((np.arange(w) + 0.5) * W / w).astype(int), W - 1)
    depths = depth[:, rows[:, None], cols[None, :]].astype(np.float32)
    return images, depths, np.diag([*scale, 1.0]) @ K, scale


def query_chunks(out, *, q, indices, clip_start, query_slice, timestamps_ns, K, w2c, hw, source):
    """Project TE's metric 3D output; match Tracks3D's NaN/status invariant.

    OUT takes precedence over OCCLUDED as in tracking.lift_tracks. Nonfinite
    predictions/logits and prebirth/out-of-clip samples are NOT_TRACKED; points
    behind the camera are OUT. Visibility uses sigmoid(logit), confidence is 1.
    """
    n = query_slice.stop - query_slice.start
    xyz = np.full((len(indices), n, 3), np.nan, np.float32)
    logit = np.full((len(indices), n), np.nan, np.float32)
    frames = np.asarray(indices, np.int64)
    rel = frames - clip_start
    keep = (rel >= 0) & (rel < len(out["xyz_base"]))
    xyz[keep] = out["xyz_base"][rel[keep], query_slice]
    logit[keep] = out["vis_logit"][rel[keep], query_slice]
    birth = np.asarray(out["birth_t"])[query_slice] + clip_start
    tracked = (frames[:, None] >= birth[None]) & keep[:, None]
    xyz[~tracked] = np.nan
    logit[~tracked] = np.nan
    cam = transform_points(w2c, xyz).astype(np.float32)
    uv, _ = project(cam, K)
    uv = uv.astype(np.float32)
    visibility = (1 / (1 + np.exp(-np.clip(logit, -80, 80)))).astype(np.float32)
    confidence = np.ones(logit.shape, np.float32)
    finite = tracked & np.isfinite(xyz).all(-1) & np.isfinite(logit)
    h, w = hw
    inside = (np.isfinite(uv).all(-1) & (cam[..., 2] > 0) & (uv[..., 0] >= 0) & (uv[..., 0] < w)
              & (uv[..., 1] >= 0) & (uv[..., 1] < h))
    status = np.full(logit.shape, NOT_TRACKED, np.uint8)
    status[finite & ~inside] = OUT
    status[finite & inside & (visibility < 0.5)] = OCCLUDED
    status[finite & inside & (visibility >= 0.5)] = VALID
    common = {"frame_indices": frames, "timestamps_ns": np.asarray(timestamps_ns)[frames],
              "point_ids": np.arange(n, dtype=np.int64), "query_frame": np.asarray(q),
              "visibility": visibility, "confidence": confidence, "source": np.asarray(json.dumps(source))}
    t2d = {**common, "uv": uv, "image_hw": np.asarray(hw)}
    xyz[status != VALID] = np.nan
    cam[status != VALID] = np.nan
    t3d = {**common, "xyz_base": xyz, "xyz_cam": cam, "status": status,
           "camera_id": np.asarray(source["camera_id"])}
    return t2d, t3d


def track_everything(root: Path, episode_id: str, spec, runner, *, depth_stage,
                     window=16, step=8, grid_stride=16, model_hw=(384, 512), num_iters=3, force=False):
    """Write ordinary ``track2d`` and ``lift3d`` stages using single-camera TE clips.

    ``spec`` is a WorkerSpec with an existing local checkpoint and a pinned checkout
    named trackeverything. Synthetic mock specs must declare ``env={"S4D_TE_MOCK": "1"}``;
    production specs always refuse mock responses. Only RGB views with metric rig
    geometry and accepted calibration participate. depth_stage is explicit and must
    contain our metric camera-z depth on the original colour grid.

    Worker clips are fsync'd to ChunkStore before ack, retained until BOTH stage
    swaps finish. A failure during either stage reuses those clips on retry.
    Returns {"track2d": record, "lift3d": record}.
    """
    if not 1 <= window <= 64 or step < 1 or grid_stride < 1 or num_iters < 1:
        raise ValueError("require window in 1..64 and positive step, grid_stride, num_iters")
    model_hw = tuple(int(x) for x in model_hw)
    if len(model_hw) != 2 or any(x <= 0 or x % 16 for x in model_hw):
        raise ValueError("model_hw must contain two positive multiples of 16")
    allow_mock = spec.env.get("S4D_TE_MOCK") == "1"
    meta = load_meta(root, episode_id)
    worker = asdict(spec)
    worker.pop("timeout_s", None)
    config = {"tracker": "TrackEverything", "worker": worker, "depth_stage": depth_stage,
              "window": window, "step": step, "grid_stride": grid_stride, "model_hw": list(model_hw),
              "num_iters": num_iters, "mode": "sparse", "single_view": True, "max_clip_frames": 64,
              "max_gap_factor": 1.5, "vis_threshold": 0.5, "allow_mock": allow_mock}
    inputs = {k: _record_fp(root, episode_id, k) for k in ("ingest", depth_stage, "calibrate")}
    if meta.stereo:
        inputs["stereo_rectify"] = _record_fp(root, episode_id, "stereo_rectify")
    directory = rio.stage_dir(root, episode_id, "track2d")
    store = rio.ChunkStore(directory.with_name("trackeverything_clips"),
                           rio.stage_fingerprint("track2d", VERSION, config, inputs))
    if force:
        store.clear()

    def build(tmp, stage):
        summary = {}
        for view in meta.views():
            cam = view.stream_id
            stream = meta.stream(cam)
            T = accepted_calibration(root, episode_id, cam)
            rig = meta.rig_rgbd(view.rig_id) if view.kind == "rgbd" else meta.rig_stereo(view.rig_id)
            if stream.modality != "rgb":
                summary[cam] = {"skipped": "TrackEverything colour view required"}
                continue
            if T is None or rig.status != "metric":
                summary[cam] = {"skipped": "accepted calibration and metric rig geometry required"}
                continue
            stamps = meta.stream_timestamps_ns[cam]
            hw = (stream.intrinsics.height, stream.intrinsics.width)
            pixels = query_grid(hw, grid_stride, grid_stride // 2)
            if not len(pixels):
                raise ValueError(f"{cam}: query grid is empty")
            files, counts, models = [], dict.fromkeys(STATUS.values(), 0), []
            for start, end, segment_end, windows in clip_plan(stamps, window, step):
                cid = f"{cam}/{start:05d}-{end:05d}"
                # Read just this camera and clip. K is from the chosen depth grid, not a model estimate.
                got = store.get(cid)
                if got is None:
                    idx = list(range(start, end))
                    video = to_model_rgb(view_frames(root, episode_id, meta, cam, idx), stream)
                    depth, K = view_depth(root, episode_id, meta, cam, idx, depth_stage)
                    if video.shape[1:3] != hw or depth.shape != (len(idx), *hw):
                        raise ValueError(f"{cam}: depth and colour must share the original view grid")
                    video, depth, Km, scale = resize_inputs(video, depth, K, model_hw)
                    queries = np.concatenate([np.column_stack([
                        np.full(len(pixels), q - start), (pixels[:, ::-1] + 0.5) * scale]) for q, _ in windows])
                    response, out, handle = runner.worker(
                        spec, "track_queries", {"mode": "sparse", "num_iters": num_iters},
                        {"video": video, "depth": depth, "K": np.broadcast_to(Km, (len(idx), 3, 3)).copy(),
                         "w2c": np.broadcast_to(T, (len(idx), 4, 4)).copy(), "queries": queries.astype(np.float32)},
                        chunk_id=f"trackeverything/{cid}", allow_mock=allow_mock)
                    expected = (len(idx), len(queries))
                    if (out["xyz_base"].shape != (*expected, 3) or out["vis_logit"].shape != expected
                            or not np.array_equal(out["birth_t"], queries[:, 0])):
                        raise ValueError("worker output shape/birth_t does not match requested queries")
                    cmeta = {"model": response["model"], "meta": response["meta"], "K": K.tolist()}
                    store.put(cid, out, cmeta)
                    runner.ack(handle)
                    got = out, cmeta
                out, cmeta = got
                model = {**cmeta["model"], "meta": cmeta["meta"]}
                models.append(model)
                for j, (q, idx) in enumerate(windows):
                    source = {**model, "camera_id": cam, "clip_frames": [start, end], "segment_end": segment_end,
                              "model_hw": list(model_hw), "num_iters": num_iters, "mode": "sparse",
                              "query_grid": "original view grid centres, continuous resize without snapping",
                              "geometry": "supplied metric depth/K/T_cam_base; predictions restored to base"}
                    t2d, t3d = query_chunks(out, q=q, indices=idx, clip_start=start,
                                          query_slice=slice(j * len(pixels), (j + 1) * len(pixels)),
                                          timestamps_ns=stamps, K=np.asarray(cmeta["K"]), w2c=T, hw=hw, source=source)
                    name = f"{rio.safe_name(cam)}_q{q:05d}.npz"
                    rio.atomic_write_npz(tmp / name, **(t2d if stage == "track2d" else t3d))
                    files.append(name)
                    for code, label in STATUS.items():
                        counts[label] += int((t3d["status"] == code).sum())
            summary[cam] = {"chunks": files, "model": models[0] if models else {}, "clips": models,
                            "status_counts": counts, "T_cam_base": T.tolist()}
        return summary

    t2d = rio.run_stage(directory, stage="track2d", version=VERSION, config=config, inputs=inputs,
                        build=lambda tmp: build(tmp, "track2d"), force=force)
    t3d = rio.run_stage(rio.stage_dir(root, episode_id, "lift3d"), stage="lift3d", version=VERSION,
                        config=config, inputs={**inputs, "track2d": rio.build_id(t2d)},
                        build=lambda tmp: build(tmp, "lift3d"), force=force)
    store.cleanup()
    return {"track2d": t2d, "lift3d": t3d}
