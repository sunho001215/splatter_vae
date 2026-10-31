"""Export training windows in the repository's accepted track contract (the DROID cache layout).

Output ``<out>/<backbone>.h5`` with ``windows/{i:05d}/`` groups holding exactly the DROID cache tensors -
images (T,V,3,H,W) uint8, depth (T,V,1,H,W) metric camera-z (0 invalid), K (V,3,3) continuous convention, w2c/c2w
(V,4,4) OpenCV with world = robot base, motion_score (T,V,1,H,W), track_src/track_disp (P,V,N,3), track_weight
(P,V,N), track_count (V,), gripper_points (T,G,3), gripper_weight (P,V,G), probe_state (T,7) - plus
``manifest.json``. The target helpers are reused verbatim from ``s4d.data.droid.pointworld`` (track_targets,
splat_score, depth_test) so losses and methods are unchanged. Loaded by ``s4d.realworld.dataset``.
"""

from __future__ import annotations

import json
from pathlib import Path

import h5py
import numpy as np
from scipy.spatial.transform import Rotation

from s4d.data.droid.pointworld import PAIR_TIMES, depth_test, nearest_timestamps, splat_score, track_targets
from s4d.realworld import io as rio
from s4d.realworld.contract import CONVENTIONS, SCHEMA_VERSION
from s4d.realworld.geometry import invert_T, resize_affine, warp_affine
from s4d.realworld.pipeline import RobotGeometry, accepted_calibration, load_depth, load_meta, read_frames, robot_state_at
from s4d.realworld.tracking import VALID

FORMAT = "splatter4d_realworld_v1_tracks"
RESOLUTIONS = {"scratch": (144, 256), "dinov2": (140, 252)}


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
           max_sync_ns: int = 20_000_000, gripper_points: int = 32, validation_tail_windows: int = 0,
           force: bool = False) -> dict:
    meta = load_meta(root, episode_id)
    resolutions = dict(resolutions or RESOLUTIONS)
    records = {k: rio.read_record(rio.stage_dir(root, episode_id, k)) for k in
               ("stereo_depth", "calibrate", "track2d", "lift3d")}
    if any(v is None for v in records.values()):
        raise RuntimeError(f"export needs complete upstream stages: {[k for k, v in records.items() if v is None]}")
    mock = any("is_mock" in json.dumps(r["summary"]) and '"is_mock": true' in json.dumps(r["summary"])
               for r in records.values())
    if mock and not allow_mock:
        raise RuntimeError("upstream stages used MOCK model outputs; refusing to export (allow_mock=False)")
    excluded, chosen = {}, []
    for g in meta.stereo:
        cam = g.left_stream
        if cameras is not None and cam not in cameras:
            excluded[cam] = "not selected"
        elif accepted_calibration(root, episode_id, cam) is None:
            excluded[cam] = "calibration missing or rejected"
        elif g.status != "metric":
            excluded[cam] = f"stereo geometry {g.status}: no metric depth"
        elif "skipped" in records["lift3d"]["summary"].get(cam, {"skipped": 1}):
            excluded[cam] = "no lifted tracks"
        else:
            chosen.append((cam, g.rig_id))
    if not chosen:
        raise RuntimeError(f"no exportable camera: {excluded}")
    modalities = {cam: meta.stream(cam).modality for cam, _ in chosen}
    if any(m == "infrared" for m in modalities.values()) and not allow_infrared_input:
        raise RuntimeError("cameras are infrared; the model input contract expects RGB. Pass allow_infrared_input "
                           "to export channel-replicated infrared (recorded as such in the manifest)")
    ref_cam = chosen[0][0]
    ref_stamps = meta.stream_timestamps_ns[ref_cam]
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

    windows = []
    for s in strides:
        for t0 in range(len(ref_stamps) - 2 * s):
            canon = [t0, t0 + s, t0 + 2 * s]
            if not robot_ok[canon].all() or any((sync_err[c][canon] > max_sync_ns).any() for c, _ in chosen):
                continue
            per_cam = {}
            for cam, _ in chosen:
                frames = [int(matched[cam][t]) for t in canon]
                hits = [ch for ch in chunks[cam] if all(f in ch["frames"] for f in frames)]
                if not hits:
                    break
                per_cam[cam] = {"chunk": max(hits, key=lambda ch: ch["query"])["name"], "frames": frames}
            else:
                windows.append({"canonical_indices": canon, "stride": s, "cameras": per_cam, "split": "train"})
    if not windows:
        raise RuntimeError("no window fits inside the tracking chunks of every camera")
    for w in windows[len(windows) - validation_tail_windows:] if validation_tail_windows else []:
        w["split"] = "validation"
    capacity = max(len(_load_chunk(root, episode_id, ch["name"])["point_ids"]) for cam, _ in chosen for ch in chunks[cam])

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
                                                  [g_all[t]]]) for T, t in zip(flange, canon)]).astype(np.float32)
                cams = []
                for cam, rig in chosen:
                    info = w["cameras"][cam]
                    ch = _load_chunk(root, episode_id, info["chunk"])
                    pos = [list(ch["frame_indices"]).index(f) for f in info["frames"]]
                    pts = ch["xyz_base"][pos].astype(np.float32)
                    vis = ch["status"][pos] == VALID
                    depth, K, rectified = load_depth(root, episode_id, rig, info["frames"])
                    if rectified:
                        with h5py.File(rio.stage_dir(root, episode_id, "stereo_depth") / f"{rig}.h5", "r") as f:
                            imgs = np.stack([f["left"]["rectified_image"][i] for i in info["frames"]])
                    else:
                        imgs = read_frames(root, episode_id, cam, info["frames"])
                    w2c = accepted_calibration(root, episode_id, cam).astype(np.float32)
                    cams.append((cam, pts, vis, depth, K, imgs, w2c))
                for name, (H, W) in resolutions.items():
                    tensors = {k: [] for k in ("images", "depth", "K", "w2c", "motion_score", "track_src", "track_disp",
                                               "track_weight", "track_count", "gripper_weight")}
                    for cam, pts, vis, depth, K, imgs, w2c in cams:
                        hw0 = depth.shape[1:]
                        aff = resize_affine(hw0, (H, W))
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
                        tensors["motion_score"].append(splat_score(np.concatenate([pts, grip], 1),
                                                                   np.concatenate([vis, gvis], 1), Kr, w2c, rd))
                        src, disp, weight = track_targets(pts, vis)
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
            "depth_resize": "nearest neighbour, zero invalid, metric camera-z of the rectified left image",
            "motion_target": "per-camera AllTracker tracks lifted with same-frame stereo depth and the accepted "
                             "calibration; weight 1 where the source-time status is valid and both positions finite",
            "gripper_target": f"{len(pick)} surface points of the URDF gripper links ({grip_links}) via FK at the "
                              "recorded gripper state; weight from the 2 cm source-time depth test",
            "probe_state": "panda flange xyz (m), flange euler xyz (rad), recorded gripper state",
            "timeline": {"reference_camera": ref_cam, "max_sync_ns": max_sync_ns,
                         "camera_sync_max_ns": {c: int(e.max()) for c, e in sync_err.items()},
                         "robot_sync_max_ns": int(robot_err.max())},
            "upstream_records": {k: v["fingerprint"] for k, v in records.items()},
            "models": {k: records[k]["summary"] for k in ("stereo_depth", "track2d")},
            "calibration": records["calibrate"]["summary"],
            "MOCK_MODEL_OUTPUTS": bool(mock),
            "split_policy": f"last {validation_tail_windows} windows validation (same episode; not generalisation)",
        }
        rio.atomic_write_json(tmp / "manifest.json", manifest)
        return {"windows": len(windows), "cameras": [c for c, _ in chosen], "mock": bool(mock)}

    return rio.run_stage(Path(out), stage="export", version="1",
                         config={"strides": list(strides), "resolutions": resolutions, "cameras": cameras,
                                 "allow_infrared_input": allow_infrared_input, "allow_mock": allow_mock,
                                 "max_sync_ns": max_sync_ns, "gripper_points": gripper_points,
                                 "validation_tail_windows": validation_tail_windows},
                         inputs={k: v["fingerprint"] for k, v in records.items()}, build=build, force=force)
