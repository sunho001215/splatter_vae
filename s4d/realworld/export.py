"""Export training windows in the repository's accepted track contract (the DROID cache layout).

Output ``<out>/<backbone>.h5`` with ``windows/{i:05d}/`` groups holding exactly the DROID cache tensors -
images (T,V,3,H,W) uint8, depth (T,V,1,H,W) metric camera-z (0 invalid), K (V,3,3) continuous convention, w2c/c2w
(V,4,4) OpenCV with world = robot base, motion_score (T,V,1,H,W), track_src/track_disp (P,V,N,3), track_weight
(P,V,N), track_count (V,), gripper_points (T,G,3), gripper_weight (P,V,G), probe_state (T,7) - plus
``manifest.json``. The target helpers are reused verbatim from ``s4d.data.droid.pointworld`` (track_targets,
splat_score, depth_test) so losses and methods are unchanged. Loaded by ``s4d.realworld.dataset``.

``track_source="track_motion"`` (intentional preprocessing change, 2026-11-05) exports only the tracks the
``track_motion`` stage classified as moving - each with its full valid trajectory - and zeroes the weight of every
pair whose two samples lie in different motion segments (occlusion, recording pause, implausible jump), so no
displacement is supervised across a break. Static scene geometry stays in the exported RGB-D; track tensors keep the
same names, shapes and padding (``track_count`` = moving tracks of that camera window). The default
``"lift3d"`` keeps every valid lifted track.
"""

from __future__ import annotations

import json
from pathlib import Path

import h5py
import numpy as np
from scipy.spatial.transform import Rotation

from s4d.data.droid.pointworld import (
    PAIR_TIMES,
    depth_test,
    nearest_timestamps,
    nearest_z_winners,
    project,
    splat_score,
    track_targets,
)
from s4d.realworld import io as rio
from s4d.realworld.contract import CONVENTIONS, SCHEMA_VERSION
from s4d.realworld.geometry import crop_affine, invert_T, resize_affine, warp_affine
from s4d.realworld.pipeline import (
    RobotGeometry,
    accepted_calibration,
    camera_views,
    default_depth_stage,
    load_meta,
    robot_state_at,
    view_depth,
    view_frames,
)
from s4d.realworld.track_motion import MOVING, load_motion
from s4d.realworld.tracking import VALID

FORMAT = "splatter4d_realworld_v1_tracks"
RESOLUTIONS = {"scratch": (144, 256), "dinov2": (140, 252)}


def export_affine(src_hw: tuple, dst_hw: tuple, aspect: str = "stretch"):
    """Pixel affine from a view grid to an export resolution: full-frame resize, or a centred crop to the target
    aspect ratio (integer crop offsets) followed by the resize."""
    if aspect == "stretch":
        return resize_affine(src_hw, dst_hw)
    H, W = src_hw
    target = dst_hw[1] / dst_hw[0]
    ch, cw = (H, int(round(H * target))) if W / H > target else (int(round(W / target)), W)
    top, left = (H - ch) // 2, (W - cw) // 2
    return crop_affine(top, left).then(resize_affine((ch, cw), dst_hw))


def motion_target_source(track2d_record: dict, depth_stage: str) -> str:
    """How the exported 3D tracks were produced, from the tracker that wrote the ``track2d`` record (TrackEverything,
    the only tracker of this pipeline; a record of any other tracker is refused)."""
    tracker = track2d_record.get("config", {}).get("tracker")
    if tracker != "TrackEverything":
        raise ValueError(f"track2d record written by {tracker!r}, not TrackEverything (the only tracker of this pipeline)")
    return (f"per-camera single-view TrackEverything 3D tracks (inputs: the camera's RGB clip, {depth_stage} "
            "depth, its intrinsics and the accepted calibration)")


def segment_motion(points: np.ndarray, seg: np.ndarray) -> np.ndarray:
    """(N,) largest displacement over ``PAIR_TIMES`` counting only pairs whose samples lie in one motion segment
    (others contribute 0); ``points`` (3,N,3), ``seg`` (3,N) segment ids (-1 invalid)."""
    out = np.zeros(points.shape[1], np.float64)
    for a, b in PAIR_TIMES:
        same = (seg[a] >= 0) & (seg[a] == seg[b])
        d = np.linalg.norm(points[b] - points[a], axis=-1)
        out = np.maximum(out, np.where(same & np.isfinite(d), d, 0.0))
    return out


def splat_score_given_motion(points, visibility, motion, K, w2c, depths) -> np.ndarray:
    """``pointworld.splat_score`` (same visibility, depth test and nearest-z winners) with the per-point displacement
    supplied instead of recomputed from all pairs."""
    height, width = depths.shape[-2:]
    score = np.zeros((3, 1, height, width), np.float32)
    for t in range(3):
        valid, pix, _ = depth_test(points[t], K, w2c, depths[t], continuous=True)
        valid &= visibility[t] & np.isfinite(points).all(axis=(0, 2))
        _, z = project(points[t], K, w2c)
        winners = nearest_z_winners(pix, z, valid, width)
        score[t, 0, pix[winners, 1], pix[winners, 0]] = np.minimum(motion[winners] / 0.03, 1)
    return score


def _chunks(root, episode_id, cam):
    rec = rio.read_record(rio.stage_dir(root, episode_id, "lift3d"))
    out = []
    for name in rec["summary"][cam]["chunks"]:
        with np.load(rio.stage_dir(root, episode_id, "lift3d") / name) as z:
            out.append({"name": name, "frames": z["frame_indices"].tolist(), "query": int(z["query_frame"])})
    return out


def _load_chunk(root, episode_id, name):
    with np.load(rio.stage_dir(root, episode_id, "lift3d") / name) as z:
        return {k: z[k] for k in z.files}


def export(root: Path, episode_id: str, out: Path, *, strides=(1, 2), resolutions: dict | None = None,
           cameras: list[str] | None = None, allow_infrared_input: bool = False, allow_mock: bool = False,
           max_sync_ns: int | None = None, gripper_points: int = 32, validation_tail_frames: int = 0,
           force: bool = False, depth_stage: str | None = None, max_gap_factor: float | None = None,
           aspect: str = "stretch", track_source: str = "lift3d") -> dict:
    """``max_sync_ns`` defaults to a third of the reference camera's median frame period (so two canonical times can
    never match the same frame of another camera). ``validation_tail_frames`` puts windows lying entirely in the
    last frames of the reference timeline into the validation split; windows straddling the boundary are dropped,
    so no frame is shared between splits (same episode: not generalisation evidence). ``depth_stage`` defaults to
    ``default_depth_stage(meta, "lift")`` (the FoundationStereo IR depth). ``max_gap_factor`` drops windows whose
    canonical frames are further apart than ``max_gap_factor * stride * median reference period`` (recording pauses
    between segments would otherwise enter as one uniform time step). ``aspect="center_crop"`` first crops the largest
    centred region with the target aspect ratio (square pixels, e.g. 4:3 RealSense -> 16:9 targets); the default
    ``"stretch"`` resizes the full frame (exact K, but anisotropic pixels when the aspect ratios differ)."""
    if aspect not in ("stretch", "center_crop"):
        raise ValueError(f"aspect {aspect!r} not in ('stretch', 'center_crop')")
    if track_source not in ("lift3d", "track_motion"):
        raise ValueError(f"track_source {track_source!r} not in ('lift3d', 'track_motion')")
    meta = load_meta(root, episode_id)
    depth_stage = depth_stage or default_depth_stage(meta, "lift")
    resolutions = dict(resolutions or RESOLUTIONS)
    records = {k: rio.read_record(rio.stage_dir(root, episode_id, k)) for k in
               (depth_stage, "calibrate", "track2d", "lift3d") + (("track_motion",) if track_source != "lift3d" else ())}
    if any(v is None for v in records.values()):
        raise RuntimeError(f"export needs complete upstream stages: {[k for k, v in records.items() if v is None]}")
    motion_target = motion_target_source(records["track2d"], depth_stage)  # refuses other trackers, also when cached
    # any model or landmark source flagged as mock anywhere in an upstream record (config or summary)
    mock = any('"is_mock": true' in json.dumps(r, default=str) for r in records.values())
    if mock and not allow_mock:
        raise RuntimeError("upstream stages used MOCK model outputs; refusing to export (allow_mock=False)")
    excluded, chosen = {}, []
    for view in camera_views(meta):
        cam = view.stream_id
        status = meta.rig_rgbd(view.rig_id).status
        if cameras is not None and cam not in cameras:
            excluded[cam] = "not selected"
        elif accepted_calibration(root, episode_id, cam) is None:
            excluded[cam] = "calibration missing or rejected"
        elif status != "metric":
            excluded[cam] = f"{view.kind} geometry {status}: no metric depth"
        elif "skipped" in records["lift3d"]["summary"].get(cam, {"skipped": 1}):
            excluded[cam] = "no lifted tracks"
        else:
            chosen.append((cam, view.rig_id))
    if not chosen:
        raise RuntimeError(f"no exportable camera: {excluded}")
    modalities = {cam: meta.stream(cam).modality for cam, _ in chosen}
    if any(m == "infrared" for m in modalities.values()) and not allow_infrared_input:
        raise RuntimeError("cameras are infrared; the model input contract expects RGB. Pass allow_infrared_input "
                           "to export channel-replicated infrared (recorded as such in the manifest)")
    ref_cam = chosen[0][0]
    ref_stamps = meta.stream_timestamps_ns[ref_cam]
    if max_sync_ns is None:
        max_sync_ns = int(np.median(np.diff(ref_stamps)) // 3) if len(ref_stamps) > 1 else 0
    geom = RobotGeometry(meta)
    rng = np.random.default_rng(0)
    grip_links = geom.gripper_links()
    pool = np.concatenate([geom.surface.link_points[n] for n in grip_links])
    pool_link = np.concatenate([[n] * len(geom.surface.link_points[n]) for n in grip_links])
    pick = rng.choice(len(pool), min(gripper_points, len(pool)), replace=False)

    def gripper_at(q, grip):
        fk = geom.model.fk(geom.q(q, grip))
        return np.stack([fk[pool_link[i]][:3, :3] @ pool[i] + fk[pool_link[i]][:3, 3] for i in pick]).astype(np.float32)

    # canonical timeline = reference camera frames; other cameras matched by nearest timestamp
    matched, sync_err = {}, {}
    for cam, _ in chosen:
        stamps = meta.stream_timestamps_ns[cam]
        idx = nearest_timestamps(stamps, ref_stamps)
        matched[cam] = idx
        sync_err[cam] = np.abs(stamps[idx] - ref_stamps)
    q_all, g_all, robot_ok, robot_err = robot_state_at(meta, ref_stamps, max_sync_ns)
    chunks = {cam: _chunks(root, episode_id, cam) for cam, _ in chosen}

    windows, gap_dropped = [], 0
    period = float(np.median(np.diff(ref_stamps))) if len(ref_stamps) > 1 else 0.0
    for s in strides:
        for t0 in range(len(ref_stamps) - 2 * s):
            canon = [t0, t0 + s, t0 + 2 * s]
            if not robot_ok[canon].all() or any((sync_err[c][canon] > max_sync_ns).any() for c, _ in chosen):
                continue
            if max_gap_factor is not None and np.diff(ref_stamps[canon]).max() > max_gap_factor * s * period:
                gap_dropped += 1
                continue
            per_cam = {}
            for cam, _ in chosen:
                frames = [int(matched[cam][t]) for t in canon]
                if not (frames[0] < frames[1] < frames[2]):
                    break  # two canonical times matched one frame of this camera: never duplicate frames
                hits = [ch for ch in chunks[cam] if all(f in ch["frames"] for f in frames)]
                if not hits:
                    break
                per_cam[cam] = {"chunk": max(hits, key=lambda ch: ch["query"])["name"], "frames": frames}
            else:
                windows.append({"canonical_indices": canon, "stride": s, "cameras": per_cam, "split": "train"})
    if validation_tail_frames:
        boundary = len(ref_stamps) - int(validation_tail_frames)
        kept = []
        for w in windows:
            if min(w["canonical_indices"]) >= boundary:
                w["split"] = "validation"
            elif max(w["canonical_indices"]) >= boundary:
                continue  # straddles the split: dropped so no frame is shared
            kept.append(w)
        windows = kept
    if not windows:
        raise RuntimeError("no window fits inside the tracking chunks of every camera")
    def keep_index(name):
        """Exported track columns of a lift3d chunk and their motion segments (None: unfiltered)."""
        if track_source == "lift3d":
            return None, None
        mo = load_motion(root, episode_id, name)
        keep = np.flatnonzero(mo["class"] == MOVING)
        return keep, mo["segment"][:, keep]

    if track_source == "lift3d":
        capacity = max(len(_load_chunk(root, episode_id, ch["name"])["point_ids"])
                       for cam, _ in chosen for ch in chunks[cam])
    else:  # at least one slot so empty windows keep valid shapes
        capacity = max(1, max(len(keep_index(ch["name"])[0]) for cam, _ in chosen for ch in chunks[cam]))

    def build(tmp: Path) -> dict:
        handles = {}
        try:
            for name, (H, W) in resolutions.items():
                handles[name] = h5py.File(tmp / f"{name}.h5", "x")
                handles[name].attrs.update(height=H, width=W, format=FORMAT)
            for wi, w in enumerate(windows):
                canon = w["canonical_indices"]
                grip = np.stack([gripper_at(q_all[t], g_all[t]) for t in canon])
                flange = [geom.model.fk(geom.q(q_all[t], g_all[t]))[meta.robot.flange_frame] for t in canon]
                probe = np.stack([np.concatenate([T[:3, 3], Rotation.from_matrix(T[:3, :3]).as_euler("xyz"),
                                                  [geom.closure(g_all[t])]]) for T, t in zip(flange, canon)]
                                 ).astype(np.float32)
                cams = []
                for cam, _rig in chosen:
                    info = w["cameras"][cam]
                    ch = _load_chunk(root, episode_id, info["chunk"])
                    pos = [list(ch["frame_indices"]).index(f) for f in info["frames"]]
                    pts = ch["xyz_base"][pos].astype(np.float32)
                    vis = ch["status"][pos] == VALID
                    keep, seg = keep_index(info["chunk"])
                    if keep is not None:
                        pts, vis, seg = pts[:, keep], vis[:, keep], seg[pos]
                    depth, K = view_depth(root, episode_id, meta, cam, info["frames"], depth_stage)
                    imgs = view_frames(root, episode_id, meta, cam, info["frames"])
                    w2c = accepted_calibration(root, episode_id, cam).astype(np.float32)
                    cams.append((cam, pts, vis, seg if keep is not None else None, depth, K, imgs, w2c))
                for name, (H, W) in resolutions.items():
                    tensors = {k: [] for k in ("images", "depth", "K", "w2c", "motion_score", "track_src", "track_disp",
                                               "track_weight", "track_count", "gripper_weight")}
                    for cam, pts, vis, seg, depth, K, imgs, w2c in cams:
                        hw0 = depth.shape[1:]
                        aff = export_affine(hw0, (H, W), aspect)
                        Kr = aff.apply_K(K).astype(np.float32)
                        rd = np.stack([warp_affine(d, aff, (H, W), order=0) for d in depth])
                        ri = np.stack([warp_affine(im.astype(np.float32), aff, (H, W), order=1) for im in imgs])
                        ri = np.clip(np.rint(ri), 0, 255).astype(np.uint8)
                        if ri.ndim == 3:  # infrared, replicated for the RGB model input (manifest says so)
                            ri = np.repeat(ri[..., None], 3, -1)
                        elif meta.stream(cam).channel_order == "BGR":
                            ri = ri[..., ::-1]
                        gvis = np.ones((3, len(grip[0])), bool)
                        for t in range(3):
                            gvis[t] = depth_test(grip[t], K.astype(np.float32), w2c, depth[t], continuous=True)[0]
                        if seg is None:  # default route: unchanged helper
                            tensors["motion_score"].append(splat_score(np.concatenate([pts, grip], 1),
                                                                       np.concatenate([vis, gvis], 1), Kr, w2c, rd))
                        else:  # moving tracks: no displacement across a motion-segment break
                            gmotion = segment_motion(grip, np.zeros(grip.shape[:2], np.int16))
                            tensors["motion_score"].append(splat_score_given_motion(
                                np.concatenate([pts, grip], 1), np.concatenate([vis, gvis], 1),
                                np.concatenate([segment_motion(pts, seg), gmotion]), Kr, w2c, rd))
                        src, disp, weight = track_targets(pts, vis)
                        if seg is not None:  # never supervise a displacement across a motion-segment break
                            for pair, (a, b) in enumerate(PAIR_TIMES):
                                cross = (seg[a] < 0) | (seg[a] != seg[b])
                                src[pair, cross], disp[pair, cross], weight[pair, cross] = 0, 0, 0
                        pad = ((0, 0), (0, capacity - pts.shape[1]))
                        tensors["track_src"].append(np.pad(src, (*pad, (0, 0))))
                        tensors["track_disp"].append(np.pad(disp, (*pad, (0, 0))))
                        tensors["track_weight"].append(np.pad(weight, pad))
                        tensors["track_count"].append(pts.shape[1])
                        tensors["gripper_weight"].append(gvis[[a for a, _ in PAIR_TIMES]].astype(np.float32))
                        tensors["images"].append(ri.transpose(0, 3, 1, 2))
                        tensors["depth"].append(rd[:, None])
                        tensors["K"].append(Kr)
                        tensors["w2c"].append(w2c)
                    out_t = {
                        "images": np.stack(tensors["images"], 1),
                        "depth": np.stack(tensors["depth"], 1).astype(np.float32),
                        "K": np.stack(tensors["K"]),
                        "w2c": np.stack(tensors["w2c"]),
                        "c2w": np.stack([invert_T(T) for T in tensors["w2c"]]).astype(np.float32),
                        "motion_score": np.stack(tensors["motion_score"], 1),
                        "track_src": np.stack(tensors["track_src"], 1),
                        "track_disp": np.stack(tensors["track_disp"], 1),
                        "track_weight": np.stack(tensors["track_weight"], 1),
                        "track_count": np.array(tensors["track_count"], np.int64),
                        "gripper_points": grip,
                        "gripper_weight": np.stack(tensors["gripper_weight"], 1),
                        "probe_state": probe,
                    }
                    group = handles[name].create_group(f"windows/{wi:05d}")
                    for key, arr in out_t.items():
                        group.create_dataset(key, data=arr, compression="lzf" if arr.ndim >= 3 else None)
        finally:
            for h in handles.values():
                h.close()
        manifest = {
            "format": FORMAT,
            "episode_schema": SCHEMA_VERSION,
            "episode": episode_id,
            "cameras": [cam for cam, _ in chosen],
            "excluded_cameras": excluded,
            "image_modality": {cam: ("infrared, 1 channel replicated to 3 for the model input (NOT rgb)"
                                     if m == "infrared" else "rgb") for cam, m in modalities.items()},
            "windows": windows,
            "resolutions": {k: list(v) for k, v in resolutions.items()},
            "track_capacity": capacity,
            "motion_pairs": [[0, 1, 0], [1, 2, 1], [0, 2, 0]],
            "conventions": CONVENTIONS,
            "intrinsics_convention": "continuous pixel coordinates; K tracked through resize as a pixel affine",
            "aspect": aspect,
            "depth_resize": f"nearest neighbour, zero invalid, metric camera-z of the colour grid ({depth_stage})",
            "motion_target": (motion_target
                              + "; weight 1 where the source-time status is valid and both positions finite"
                              + ("" if track_source == "lift3d" else
                                 "; ONLY tracks classified moving by track_motion (full valid trajectories), pairs "
                                 "across motion-segment breaks weighted 0; static geometry is in images/depth only")),
            "track_source": track_source,
            **({} if track_source == "lift3d" else {"track_motion": records["track_motion"]["summary"]["counts"]}),
            "depth_stage": depth_stage,
            "windows_dropped_for_time_gaps": gap_dropped,
            "gripper_target": f"{len(pick)} surface points of the URDF gripper links ({grip_links}) via FK at the "
                              "recorded gripper state; weight from the 2 cm source-time depth test",
            "probe_state": "panda flange xyz (m), flange euler xyz (rad, scipy 'xyz'), gripper closure in [0,1] "
                           "(0 open; from the recorded state via the URDF joint range). DROID uses its EE pose instead "
                           "of the flange",
            "timeline": {"reference_camera": ref_cam, "max_sync_ns": max_sync_ns,
                         "camera_sync_max_ns": {c: int(e.max()) for c, e in sync_err.items()},
                         "robot_sync_max_ns": int(robot_err.max())},
            "upstream_records": {k: rio.build_id(v) for k, v in records.items()},
            "models": {k: records[k]["summary"] for k in (depth_stage, "track2d")},
            "calibration": records["calibrate"]["summary"],
            "MOCK_MODEL_OUTPUTS": bool(mock),
            "split_policy": f"windows inside the last {validation_tail_frames} reference frames are validation, "
                            "straddling windows dropped (same episode; not generalisation)",
        }
        rio.atomic_write_json(tmp / "manifest.json", manifest)
        return {"windows": len(windows), "cameras": [c for c, _ in chosen], "mock": bool(mock)}

    return rio.run_stage(Path(out), stage="export", version="1",
                         config={"strides": list(strides), "resolutions": resolutions, "cameras": cameras,
                                 "allow_infrared_input": allow_infrared_input, "allow_mock": allow_mock,
                                 "max_sync_ns": max_sync_ns, "gripper_points": gripper_points,
                                 "validation_tail_frames": validation_tail_frames,
                                 "depth_stage": depth_stage,
                                 **({} if max_gap_factor is None else {"max_gap_factor": max_gap_factor}),
                                 **({} if aspect == "stretch" else {"aspect": aspect}),
                                 **({} if track_source == "lift3d" else {"track_source": track_source})},
                         inputs={k: rio.build_id(v) for k, v in records.items()}, build=build, force=force)
