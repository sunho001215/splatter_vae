"""TrackEverything single-view tracks in the track2d/lift3d chunk contract (the own-rig tracker).

This stage writes both the ``track2d`` and the ``lift3d`` records of an output root. It never invokes the multiview
demo, interleaves views, or estimates geometry; downstream consumers (track_motion, export, viewer) read the same
chunk contract.
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
    calibrated_geometry_config,
    calibrated_version,
    calibrated_view_depth,
    camera_views,
    chunk_queries,
    load_meta,
    to_model_rgb,
    view_frames,
)
from s4d.realworld.tracking import (
    EDGE,
    NO_DEPTH,
    NOT_TRACKED,
    OCCLUDED,
    OUT,
    STATUS,
    VALID,
    depth_support_status,
    query_grid,
)

VERSION = "2-trackeverything"


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

    OUT takes precedence over OCCLUDED. Nonfinite
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
    if "birth_status" in out:
        support = out["birth_status"][query_slice]
        unsupported = support != VALID
        status[:, unsupported] = NOT_TRACKED
        at_birth = frames[:, None] == birth[None]
        status[at_birth & unsupported[None]] = np.broadcast_to(support, status.shape)[at_birth & unsupported[None]]
        uv[:, unsupported] = np.nan
        visibility[:, unsupported] = np.nan
    common = {"frame_indices": frames, "timestamps_ns": np.asarray(timestamps_ns)[frames],
              "point_ids": np.arange(n, dtype=np.int64), "query_frame": np.asarray(q),
              "visibility": visibility, "confidence": confidence, "source": np.asarray(json.dumps(source))}
    t2d = {**common, "uv": uv, "image_hw": np.asarray(hw)}
    xyz[status != VALID] = np.nan
    cam[status != VALID] = np.nan
    t3d = {**common, "xyz_base": xyz, "xyz_cam": cam, "status": status,
           "camera_id": np.asarray(source["camera_id"])}
    return t2d, t3d


def te_provenance(models: list[dict], runner, spec) -> dict:
    """What pins one camera's TrackEverything outputs (2026-11-06, user decision item 3): the s4d source (runner),
    the upstream commit and checkpoint sha256 of every clip, the torch.hub checkpoints, the seed and the distinct
    hardware/runtime identities (GPU, driver, CUDA, cuDNN, torch, flags) reported by the worker. The stage config (also
    in the record and fingerprinted) holds the inference parameters. Clips served from a chunk store written before the
    worker reported ``runtime`` are counted, never back-filled."""
    called = [m for m in models if "skipped" not in (m.get("meta") or {})]  # no supported birth: TE never called
    runtimes, seen, missing = [], set(), 0
    for m in called:
        rt = (m.get("meta") or {}).get("runtime")
        if rt is None:
            missing += 1
            continue
        key = json.dumps(rt, sort_keys=True, default=str)
        if key not in seen:
            seen.add(key)
            runtimes.append(rt)
    return {"source": runner.describe() if hasattr(runner, "describe") else None,
            "expected_commit": spec.expected_commit, "commits": sorted({m.get("commit") for m in called} - {None}),
            "checkpoint": spec.checkpoint,
            "checkpoint_sha256": sorted({m.get("checkpoint_sha256") for m in called} - {None}),
            "seeds": sorted({rt.get("seed") for rt in runtimes} - {None}), "runtime": runtimes,
            "clips_called": len(called), "clips_without_runtime": missing,
            "inference": "normal fast GPU inference (no deterministic mode); identical inputs reproduce up to GPU "
                         "kernel nondeterminism (accepted <= 0.12 px / 0.17 mm, 2026-11-06)"}


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
    if not 1 <= window <= 16 or step < 1 or grid_stride < 1 or num_iters < 1:
        raise ValueError("require window in 1..16 and positive step, grid_stride, num_iters")
    model_hw = tuple(int(x) for x in model_hw)
    if len(model_hw) != 2 or any(x <= 0 or x % 16 for x in model_hw):
        raise ValueError("model_hw must contain two positive multiples of 16")
    allow_mock = spec.env.get("S4D_TE_MOCK") == "1"
    meta = load_meta(root, episode_id)
    worker = asdict(spec)
    worker.pop("timeout_s", None)
    config = {"tracker": "TrackEverything", "worker": worker, "depth_stage": depth_stage,
              "window": window, "step": step, "grid_stride": grid_stride, "model_hw": list(model_hw),
              "num_iters": num_iters, "mode": "sparse", "single_view": True, "max_clip_frames": 16,
              "inference_frames": "exactly pipeline.chunk_queries, including recording pauses",
              "birth_depth_edge_rel": 0.05, "query_depth_lookup": "floor on resized grid; no depth invention",
              "vis_threshold": 0.5, "allow_mock": allow_mock}
    # per-camera depth scale of the accepted calibration (TE predicts 3D in the input depth's metric scale and restores
    # it to base with the pose): only recorded - and the version bumped - when a factor differs from 1
    cal_extra = calibrated_geometry_config(root, episode_id, meta, depth_stage)
    cal_extra.pop("robot_model", None)  # TrackEverything never poses the robot
    config.update(cal_extra)
    version = calibrated_version(VERSION, cal_extra)
    inputs = {k: _record_fp(root, episode_id, k) for k in ("ingest", depth_stage, "calibrate")}
    directory = rio.stage_dir(root, episode_id, "track2d")
    store = rio.ChunkStore(directory.with_name("trackeverything_clips"),
                           rio.stage_fingerprint("track2d", version, config, inputs))
    if force:
        store.clear()

    def build(tmp, stage):
        summary = {}
        for view in camera_views(meta):
            cam = view.stream_id
            stream = meta.stream(cam)
            T = accepted_calibration(root, episode_id, cam)
            rig = meta.rig_rgbd(view.rig_id)
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
            birth_counts = {"queries": 0, "unsupported_birth_depth": 0, "unsupported_birth_edge": 0}
            for q, idx in chunk_queries(len(stamps), window, step):
                start, end = idx[0], idx[-1] + 1
                cid = f"{cam}/{start:05d}-{end:05d}"
                # Read just this camera and clip. K is from the chosen depth grid, not a model estimate.
                got = store.get(cid)
                if got is None:
                    video = to_model_rgb(view_frames(root, episode_id, meta, cam, idx), stream)
                    depth, K = calibrated_view_depth(root, episode_id, meta, cam, idx, depth_stage)
                    if video.shape[1:3] != hw or depth.shape != (len(idx), *hw):
                        raise ValueError(f"{cam}: depth and colour must share the original view grid")
                    xy = pixels[:, ::-1] + 0.5
                    support = depth_support_status(depth[:1], xy[None])[0]
                    video, depth, Km, scale = resize_inputs(video, depth, K, model_hw)
                    resized_support = depth_support_status(depth[:1], (xy * scale)[None])[0]
                    # Original-grid support matches lift3d; also guard the exact resized cells TE will sample.
                    support[support == VALID] = resized_support[support == VALID]
                    valid = support == VALID
                    queries = np.column_stack([np.zeros(valid.sum()), (xy * scale)[valid]]).astype(np.float32)
                    out = {"xyz_base": np.full((len(idx), len(pixels), 3), np.nan, np.float32),
                           "vis_logit": np.full((len(idx), len(pixels)), np.nan, np.float32),
                           "birth_t": np.zeros(len(pixels), np.int64), "birth_status": support}
                    cmeta = {"model": {}, "meta": {"skipped": "no supported birth depths"}, "K": K.tolist()}
                    handle = None
                    if valid.any():
                        response, predicted, handle = runner.worker(
                            spec, "track_queries", {"mode": "sparse", "num_iters": num_iters},
                            {"video": video, "depth": depth, "K": np.broadcast_to(Km, (len(idx), 3, 3)).copy(),
                             "w2c": np.broadcast_to(T, (len(idx), 4, 4)).copy(), "queries": queries},
                            chunk_id=f"trackeverything/{cid}", allow_mock=allow_mock)
                        expected = (len(idx), len(queries))
                        if (predicted["xyz_base"].shape != (*expected, 3) or predicted["vis_logit"].shape != expected
                                or not np.array_equal(predicted["birth_t"], queries[:, 0])):
                            raise ValueError("worker output shape/birth_t does not match requested queries")
                        for key in ("xyz_base", "vis_logit", "dyn_logit"):
                            if key in predicted:
                                if key not in out:
                                    out[key] = np.full((len(idx), len(pixels)), np.nan, np.float32)
                                out[key][:, valid] = predicted[key]
                        cmeta.update(model=response["model"], meta=response["meta"])
                    store.put(cid, out, cmeta)
                    if handle is not None:
                        runner.ack(handle)
                    got = out, cmeta
                out, cmeta = got
                model = {**cmeta["model"], "meta": cmeta["meta"]}
                models.append(model)
                birth_counts["queries"] += len(pixels)
                birth_counts["unsupported_birth_depth"] += int((out["birth_status"] == NO_DEPTH).sum())
                birth_counts["unsupported_birth_edge"] += int((out["birth_status"] == EDGE).sum())
                source = {**model, "camera_id": cam, "clip_frames": [start, end],
                          "model_hw": list(model_hw), "num_iters": num_iters, "mode": "sparse",
                          "query_grid": "original view grid centres, continuous resize without snapping",
                          "geometry": "supplied metric depth/K/T_cam_base; predictions restored to base"}
                t2d, t3d = query_chunks(out, q=q, indices=idx, clip_start=start, query_slice=slice(0, len(pixels)),
                                      timestamps_ns=stamps, K=np.asarray(cmeta["K"]), w2c=T, hw=hw, source=source)
                name = f"{rio.safe_name(cam)}_q{q:05d}.npz"
                rio.atomic_write_npz(tmp / name, **(t2d if stage == "track2d" else t3d))
                files.append(name)
                for code, label in STATUS.items():
                    counts[label] += int((t3d["status"] == code).sum())
            summary[cam] = {"chunks": files, "model": models[0] if models else {}, "clips": models,
                            "status_counts": counts, **birth_counts, "T_cam_base": T.tolist(),
                            "provenance": te_provenance(models, runner, spec)}
        return summary

    t2d = rio.run_stage(directory, stage="track2d", version=version, config=config, inputs=inputs,
                        build=lambda tmp: build(tmp, "track2d"), force=force)
    t3d = rio.run_stage(rio.stage_dir(root, episode_id, "lift3d"), stage="lift3d", version=version,
                        config=config, inputs={**inputs, "track2d": rio.build_id(t2d)},
                        build=lambda tmp: build(tmp, "lift3d"), force=force)
    store.cleanup()
    return {"track2d": t2d, "lift3d": t3d}
