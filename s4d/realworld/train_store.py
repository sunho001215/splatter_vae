"""Compact real-world TRAINING store (schema ``s4d.realworld.train/1.1.0``; data-format policy of 2026-11-06).

One HDF5 file per episode (``<out>/<episode_id>/episode.h5``, written through ``io.run_stage`` as the ``train_data``
stage: fingerprinted completion record, atomic swap-in). Frames are stored ONCE per camera and timestamp; temporal
windows, cross-camera sets, viewpoints and every motion target are assembled in the loader
(``s4d.realworld.train_windows`` for our method; ``CompactStore`` below is the per-episode per-camera frame access the
baselines and offline FQL read).

Image geometry (every camera): the centred square of the colour grid (side ``min(H, W)``, integer offsets: 480x480 at
x0 = 80 for 640x480) resized to ``IMAGE_SIZE`` (128). The exact pixel affine ``u' = a (u - x0)``, ``v' = a (v - y0)``
with ``a = 128 / side`` (continuous pixel coordinates) is stored with the native and the 128-pixel intrinsics
(``K = A @ K_native``), so 2D tracks, projections and masks map exactly. Resampling (shared with the DROID square-crop
profile, ``s4d.data.droid.largescale.crop``, and the SNeRL contract check): RGB by PIL BILINEAR (antialiased
area-weighted downsampling); depth, validity and semantic masks by the nearest native pixel of each output pixel
centre (no value mixing across depth edges or class boundaries). World positions are never rescaled.

Layout of ``episode.h5`` (frame arrays chunked one frame per chunk, gzip level 4, shuffle for multi-byte types; every
array is stored LOSSLESSLY: the values are exactly the resampled preprocessing outputs)::

    attrs   format, schema_version, writer_version, code_commit, episode_id, task, raw_recording(+sha256), cameras,
            reference_camera, image_size, conventions, resampling, upstream build ids, models, ...
    cameras/<serial>/
        timestamps_ns (F,) int64          the stream's timestamps (host CLOCK_MONOTONIC, ns)
        segment       (F,) int16          recording segment of each frame (pauses separate segments)
        rgb           (F,S,S,3) uint8     RGB channel order
        depth         (F,S,S) float32     metric camera-z of the depth stage, bitwise; 0 = invalid (validity mask)
        semantic      (F,S,S) uint8       SNeRL classes (background, robot, object, fixture)   [only with SAM 3]
        class_present (F,4) bool                                                                [only with SAM 3]
        K (3,3) / K_native (3,3) float64  continuous convention;  w2c (4,4) float64 accepted T_cam_base
        affine (3,3) float64              native -> stored pixel affine;  crop_box (3,) int64 [x0, y0, side]
        robot/  ok (F,) bool, sync_err_ns (F,) int64, joint_positions (F,7) f64, gripper_state (F,) f64,
                probe_state (F,7) f32, gripper_points (F,G,3) f32     (robot state at THIS camera's frame times)
        tracks/ (dynamic tracks only: ``track_motion`` class MOVING, full valid trajectories, base frame metres)
                chunk_frames (C,L) int32 (-1 pad), chunk_query (C,), chunk_offset (C+1,) rows of each chunk,
                point_id (J,) int32 (camera- and chunk-local), xyz (J,L,3) float32 (NaN unless valid),
                status (J,L) uint8 (``tracking.STATUS``), segment (J,L) int16 (-1 invalid), visibility/confidence f16
    sync/   frames (F_ref,V) int32: the MUTUAL nearest frame of every camera to each reference frame, no time cap
            (-1: none; unique for any constant phase offset below half a period: the cameras are free-running),
            error_ns (F_ref,V) int64 (|time offset| to the reference frame),
            surface_disp_m (F_ref,) float64: the largest displacement of the robot's URDF surface between any two members
            of the set (robot state interpolated to each member's timestamp; inf when a member has no valid robot
            state); the loaders accept a set iff it is <= a configurable threshold (criterion M5, default 5 mm,
            ``MAX_SET_MOTION_M``),
            gripper_visible (F_ref,V,G) bool: reference-time gripper FK points depth-tested (2 cm) against each
            camera's native-resolution depth of its synchronised frame (the export's gripper visibility, exactly);
            attrs: criterion, pairing, surface sampling, default threshold, per-episode set counts
    robot/  timestamps_ns, joint_positions, gripper_state[, ee_pose_base]   (the recorded trajectory, as normalized)

Dense XYZ is never stored (depth + K + w2c reconstruct it); no window-level copies, no DINO features (SNeRL DINO
targets are computed online by the trainer from the stored RGB: ``s4d.baselines.snerl.model.dino_target``).

Episodes with a rejected/held/missing camera calibration carry ``layout=calibration_optional/1`` in the file,
stage config and summary. Every RGB camera remains, with explicit ``calibration`` / ``calibration_status`` and
``geometry_available`` attributes; cameras without geometry omit ``w2c``, ``depth`` and ``tracks``. The episode omits
``sync/gripper_visible``. Robot FK points and M5 surface motion depend on robot kinematics only and remain valid.
Readers refuse multi-view/geometry access to such episodes. Fully accepted episodes retain their existing bytes,
config and fingerprint, so neither the writer version nor schema is bumped and no full re-export is needed.
"""

from __future__ import annotations

import functools
import json
import os
import time
from collections import OrderedDict
from pathlib import Path

import h5py
import numpy as np
from scipy.spatial.transform import Rotation

from s4d.data.droid.largescale import crop as CR
from s4d.realworld import io as rio
from s4d.realworld.contract import CONVENTIONS
from s4d.realworld.contract import SCHEMA_VERSION as EPISODE_SCHEMA
from s4d.realworld.frame_preprocessing import IMAGE_SIZE, nearest_resample, training_depth
from s4d.realworld.geometry import PixelAffine
from s4d.realworld.tracking import NOT_TRACKED, VALID

FORMAT = "s4d.realworld.train"
SCHEMA_VERSION = "1.1.0"  # 1.1.0: cross-camera sets by mutual nearest WITHOUT a time cap + per-set robot motion (M5)
WRITER_VERSION = "2"
CALIBRATION_OPTIONAL_LAYOUT = "calibration_optional/1"  # only affected episodes; accepted exports stay byte-identical
SYNC_CRITERION = "M5"
MAX_SET_MOTION_M = 0.005  # M5 default: a set is accepted iff the robot surface moved <= 5 mm across its timestamps
STAGE = "train_data"
FILE = "episode.h5"
SEMANTIC_CLASSES = ("background", "robot", "object", "fixture")
GZIP = {"compression": "gzip", "compression_opts": 4}
RGB_RESAMPLE = ("centred square (side min(H,W), integer offsets) resized with PIL BILINEAR (antialiased); "
                "s4d.data.droid.largescale.crop.crop_rgb")
NEAREST_RESAMPLE = ("nearest native pixel of each output pixel centre: row/col = floor(offset + (i + 0.5) * side / S); "
                    "s4d.data.droid.largescale.crop.crop_depth rule (depth, validity, semantic masks)")
REPO = Path(__file__).resolve().parents[2]
_INHERITED_HANDLES: list = []  # HDF5 handles a forked worker inherited from its parent (kept alive, never closed there)


# ------------------------------------------------------------------------------------------------ geometry
def square_box(height: int, width: int) -> CR.CropBox:
    """The centred square of a ``height`` x ``width`` grid (integer offsets)."""
    return CR.center_box(int(height), int(width))


def box_affine(box: CR.CropBox, size: int = IMAGE_SIZE) -> PixelAffine:
    """Native continuous pixel coordinates -> stored ones: ``u' = a (u - x0)``, ``v' = a (v - y0)``, ``a = size/side``."""
    a = size / box.side
    return PixelAffine(a, a, -a * box.x0, -a * box.y0)


def recording_segments(meta, stamps_ns: np.ndarray) -> np.ndarray:
    """(F,) recording segment of each frame from the recorder's segment start times (provenance notes; 0 when the
    recording has one segment or does not report segments)."""
    try:
        notes = json.loads(meta.provenance.notes or "{}")
    except json.JSONDecodeError:  # adapters other than panda_teleop_h5 keep free-text notes
        notes = {}
    notes = notes if isinstance(notes, dict) else {}
    starts = np.rint(np.asarray((notes.get("segments") or {}).get("t_start", []), np.float64) * 1e9).astype(np.int64)
    if not len(starts):
        return np.zeros(len(stamps_ns), np.int16)
    return (np.searchsorted(starts, np.asarray(stamps_ns, np.int64), side="right") - 1).astype(np.int16)


def mutual_sync(stamps: list[np.ndarray], max_sync_ns: int | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Frame sets over the reference (first) camera: ``frames`` (F_ref, V), the MUTUAL nearest frame of every camera
    (else -1), and ``error_ns`` (F_ref, V) of the nearest frame. Mutual nearest pairing is one-to-one by construction
    and finds every pair of two free-running cameras whose constant phase offset is below half a period. Without
    ``max_sync_ns`` there is no time cap (criterion M5: sets are then filtered by robot motion, ``set_surface_motion``);
    a given cap must stay below half the reference period (``depth_stages.frame_sets(allow_unmatched=True)`` rule)."""
    from s4d.data.droid.pointworld import nearest_timestamps

    ref = np.asarray(stamps[0], np.int64)
    if max_sync_ns is None:
        max_sync_ns = np.iinfo(np.int64).max
    elif len(ref) > 1 and max_sync_ns >= float(np.median(np.diff(ref))) / 2:
        raise ValueError(f"max_sync_ns {max_sync_ns} >= half the frame period: matches would not be unique")
    frames = np.full((len(ref), len(stamps)), -1, np.int32)
    error = np.zeros((len(ref), len(stamps)), np.int64)
    for v, s in enumerate(stamps):
        s = np.asarray(s, np.int64)
        j = nearest_timestamps(s, ref)
        back = nearest_timestamps(ref, s)
        error[:, v] = np.abs(s[j] - ref)
        ok = (error[:, v] <= max_sync_ns) & (back[j] == np.arange(len(ref)))
        frames[ok, v] = j[ok]
    return frames, error


def set_surface_motion(surface_points, frames: np.ndarray, states: list, ok: list) -> np.ndarray:
    """(F_ref,) largest displacement (m) of the robot surface between any two members of each set (``frames`` (F,V),
    -1 = no member of camera v). ``states[v][f]`` is the robot state of camera v's frame f (joint positions, gripper
    state; interpolated to that frame's timestamp), ``surface_points(state)`` -> (P, 3) base-frame URDF surface points,
    ``ok[v][f]`` its validity. A set with an invalid member state is ``inf`` (never accepted); a single member is 0.
    The quantity of the retired MapAnything static-scene rule (``max_set_motion_m``) and of PREP2's sync review."""
    out = np.zeros(len(frames))
    for k, row in enumerate(frames):
        members = [(v, int(f)) for v, f in enumerate(row) if f >= 0]
        if not all(ok[v][f] for v, f in members):
            out[k] = np.inf
            continue
        pts = [np.asarray(surface_points(states[v][f]), np.float64) for v, f in members]
        if not all(np.isfinite(x).all() for x in pts):  # a non-finite interpolated state is no valid state
            out[k] = np.inf
            continue
        out[k] = max((float(np.linalg.norm(pts[a] - pts[b], axis=1).max())
                      for a in range(len(pts)) for b in range(a + 1, len(pts))), default=0.0)
    return out


def sync_counts(frames: np.ndarray, disp: np.ndarray, max_set_motion_m: float = MAX_SET_MOTION_M) -> dict:
    complete = (frames >= 0).all(1)
    return {"reference_frames": int(len(frames)), "complete_sets": int(complete.sum()),
            "accepted": int((complete & (disp <= max_set_motion_m)).sum()),
            "rejected_motion": int((complete & np.isfinite(disp) & (disp > max_set_motion_m)).sum()),
            "rejected_no_robot_state": int((complete & ~np.isfinite(disp)).sum()),
            "max_set_motion_m": float(max_set_motion_m)}


# ------------------------------------------------------------------------------------------------ writer
def _code_commit() -> str | None:
    """HEAD of the writing checkout, suffixed ``+dirty`` when ``s4d/`` has uncommitted or untracked changes."""
    import subprocess

    from s4d.realworld.workers._protocol import git_commit

    commit = git_commit(str(REPO))
    try:
        dirty = subprocess.run(["git", "-C", str(REPO), "status", "--porcelain", "--", "s4d"], capture_output=True,
                               text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        dirty = ""
    return f"{commit}+dirty" if commit and dirty else commit


def _chunk_names(root, episode_id, cam) -> list[str]:
    rec = rio.read_record(rio.stage_dir(root, episode_id, "lift3d"))
    return list(rec["summary"][cam]["chunks"])


def _camera_tracks(root, episode_id, cam, stamps) -> dict:
    """Moving tracks of every lift3d chunk of ``cam`` (``track_motion`` class MOVING), padded to the longest chunk."""
    from s4d.realworld.stages.motion import MOVING, load_motion

    names = _chunk_names(root, episode_id, cam)
    rows = []
    for name in names:
        with np.load(rio.stage_dir(root, episode_id, "lift3d") / name) as z:
            lift = {k: z[k] for k in ("frame_indices", "timestamps_ns", "point_ids", "query_frame", "xyz_base",
                                      "status", "visibility", "confidence")}
        mo = load_motion(root, episode_id, name)
        if not (np.array_equal(mo["point_ids"], lift["point_ids"]) and
                np.array_equal(mo["frame_indices"], lift["frame_indices"])):
            raise RuntimeError(f"{name}: track_motion and lift3d disagree on points/frames")
        if not np.array_equal(np.asarray(stamps)[lift["frame_indices"]], lift["timestamps_ns"]):
            raise RuntimeError(f"{name}: track timestamps disagree with the camera's frame timestamps")
        keep = np.flatnonzero(mo["class"] == MOVING)
        rows.append((name, lift, mo["segment"], keep))
    L = max(len(r[1]["frame_indices"]) for r in rows)
    J = sum(len(r[3]) for r in rows)
    out = {"chunk_frames": np.full((len(rows), L), -1, np.int32), "chunk_query": np.zeros(len(rows), np.int32),
           "chunk_offset": np.zeros(len(rows) + 1, np.int64), "point_id": np.zeros(J, np.int32),
           "xyz": np.full((J, L, 3), np.nan, np.float32), "status": np.full((J, L), NOT_TRACKED, np.uint8),
           "segment": np.full((J, L), -1, np.int16), "visibility": np.zeros((J, L), np.float16),
           "confidence": np.zeros((J, L), np.float16)}
    j = 0
    for c, (_name, lift, seg, keep) in enumerate(rows):
        n = len(lift["frame_indices"])
        out["chunk_frames"][c, :n] = lift["frame_indices"]
        out["chunk_query"][c] = int(lift["query_frame"])
        k = len(keep)
        out["point_id"][j:j + k] = lift["point_ids"][keep]
        out["xyz"][j:j + k, :n] = lift["xyz_base"][:, keep].transpose(1, 0, 2)
        out["status"][j:j + k, :n] = lift["status"][:, keep].T
        out["segment"][j:j + k, :n] = seg[:, keep].T
        out["visibility"][j:j + k, :n] = lift["visibility"][:, keep].T
        out["confidence"][j:j + k, :n] = lift["confidence"][:, keep].T
        j += k
        out["chunk_offset"][c + 1] = j
    valid = out["status"] == VALID
    if not (np.isfinite(out["xyz"]).all(-1) == valid).all():
        raise RuntimeError(f"{cam}: track positions must be finite exactly where the status is valid")
    out["_names"] = [r[0] for r in rows]
    return out


def write_episode(root: Path, episode_id: str, out_dir: Path, *, task: str | None = None, depth_stage: str | None = None,
                  semantic: bool | None = None, max_sync_ns: int | None = None, gripper_points: int = 32,
                  image_size: int = IMAGE_SIZE, chunk: int = 16, allow_mock: bool = False, force: bool = False,
                  cross_camera_cap_ns: int | None = None, max_set_motion_m: float = MAX_SET_MOTION_M) -> dict:
    """Write the compact training store of one processed episode root into ``out_dir`` (stage ``train_data``).

    Inputs (complete, validated stage records of ``root``): ingest (normalized frames, timestamps, robot trajectory),
    ``depth_stage`` (default: the FoundationStereo IR depth on the colour grid), ``calibrate`` (accepted T_cam_base),
    ``track2d``/``lift3d`` (TrackEverything) and ``track_motion``; ``sam3_masks`` when ``semantic`` (default: when its
    record exists). ``max_sync_ns`` is the per-camera ROBOT <-> camera sync tolerance (default: a third of the reference
    period, as the window export); it no longer bounds the cross-camera sets: those pair mutual nearest frames without a
    time cap (``cross_camera_cap_ns`` None) and store each set's robot-surface motion (criterion M5; the loaders accept
    a set iff it is <= their threshold, default ``max_set_motion_m`` 5 mm, which also sets the recorded counts).
    ``task`` defaults to the dataset folder of the recording (the adapter's task identity). Cameras without an accepted
    calibration retain RGB and robot sync under ``CALIBRATION_OPTIONAL_LAYOUT``. Geometry fields are omitted for
    those cameras; multi-view and geometric consumers must refuse them. Fully accepted episodes keep the existing
    layout and fingerprint (including the existing metric-depth/track eligibility rules)."""
    from s4d.data.droid.pointworld import depth_test
    from s4d.realworld.export import motion_target_source
    from s4d.realworld.pipeline import (
        RobotGeometry,
        accepted_calibration,
        calibrated_geometry_config,
        calibrated_version,
        calibration_depth_scale,
        calibration_robot_model,
        camera_views,
        default_depth_stage,
        load_meta,
        read_frames,
        robot_state_at,
        to_model_rgb,
    )
    from s4d.realworld.stages.depth import load_view_depth

    root, out_dir = Path(root), Path(out_dir)
    meta = load_meta(root, episode_id)
    depth_stage = depth_stage or default_depth_stage(meta, "lift")
    task = task or Path(meta.provenance.source_root or "").name
    if not task:
        raise ValueError(f"{episode_id}: no task given and the recording has no dataset folder")
    norm = rio.read_record(rio.episode_dir(root, episode_id) / "normalized")
    names = [depth_stage, "calibrate", "track2d", "lift3d", "track_motion"]
    if semantic is None:
        semantic = rio.read_record(rio.stage_dir(root, episode_id, "sam3_masks")) is not None
    if semantic:
        names.append("sam3_masks")
    records = {k: rio.read_record(rio.stage_dir(root, episode_id, k)) for k in names}
    views = camera_views(meta)
    calibration = {v.stream_id: accepted_calibration(root, episode_id, v.stream_id) for v in views}
    partial = any(calibration[v.stream_id] is None for v in views)
    required = ["calibrate", *(["sam3_masks"] if semantic else [])] if partial else names
    missing = [k for k in required if records[k] is None] + ([] if norm else ["ingest"])
    if missing:
        raise RuntimeError(f"{STAGE} needs complete upstream stages: {missing}")
    mock = any('"is_mock": true' in json.dumps(r, default=str) for r in records.values())
    if mock and not allow_mock:
        raise RuntimeError("upstream stages used MOCK model outputs; refusing to write training data (allow_mock=False)")
    excluded, chosen, geometry = {}, [], set()
    for view in views:
        cam = view.stream_id
        reason = None
        if calibration[cam] is None:
            reason = "calibration missing or rejected"
        elif meta.rig_rgbd(view.rig_id).status != "metric":
            reason = "no metric depth"
        elif "chunks" not in ((records["lift3d"] or {}).get("summary") or {}).get(cam, {}):
            reason = "no lifted tracks"
        elif meta.stream(cam).modality != "rgb":
            reason = f"modality {meta.stream(cam).modality} (training images are RGB)"
        if reason is None:
            missing = [k for k in names if records[k] is None]
            if missing:
                raise RuntimeError(f"{STAGE} needs complete upstream stages: {missing}")
            geometry.add(cam)
        if reason is None or partial and meta.stream(cam).modality == "rgb":
            chosen.append(view)
        else:
            excluded[view.rig_id] = reason
    if not chosen:
        raise RuntimeError(f"no camera for the training store: {excluded}")
    chosen.sort(key=lambda v: v.rig_id)
    ref_stamps = np.asarray(meta.stream_timestamps_ns[chosen[0].stream_id], np.int64)
    if max_sync_ns is None:
        max_sync_ns = int(np.median(np.diff(ref_stamps)) // 3) if len(ref_stamps) > 1 else 0
    config = {"task": task, "depth_stage": depth_stage, "semantic": bool(semantic), "max_sync_ns": int(max_sync_ns),
              "gripper_points": int(gripper_points), "image_size": int(image_size), "schema": SCHEMA_VERSION,
              "writer_version": WRITER_VERSION, "allow_mock": bool(allow_mock), "sync_criterion": SYNC_CRITERION,
              "cross_camera_cap_ns": None if cross_camera_cap_ns is None else int(cross_camera_cap_ns),
              "max_set_motion_m": float(max_set_motion_m)}
    if partial:
        config["layout"] = CALIBRATION_OPTIONAL_LAYOUT
    # calibration-derived settings of the GEOMETRY fields (stored depth and gripper-visibility test x the accepted
    # per-camera depth scale; gripper target points posed with the accepted robot model). Robot proprioception
    # (probe_state closure) and the M5 set motion keep the legacy robot geometry, so consumers that use no geometry
    # see identical data. Recorded - and the writer version bumped - only when they differ from the legacy calibration.
    cal_extra = calibrated_geometry_config(root, episode_id, meta, depth_stage) if geometry else {}
    config.update(cal_extra)
    writer_version = calibrated_version(WRITER_VERSION, cal_extra)
    scales = {v.stream_id: (calibration_depth_scale(root, episode_id, v.stream_id, depth_stage)
                            if v.stream_id in geometry else 1.0) for v in chosen}
    inputs = {"ingest": rio.build_id(norm), **{k: rio.build_id(v) for k, v in records.items() if v is not None}}

    def build(tmp: Path) -> dict:
        t_start = time.perf_counter()
        geom = RobotGeometry(meta)  # legacy: probe closure + M5 set motion (unchanged for non-geometric consumers)
        model = cal_extra.get("robot_model")
        ggeom = geom if model is None else RobotGeometry(meta, gripper_map=model["gripper_map"],
                                                         surface=model["surface"])
        if model is not None and calibration_robot_model(root, episode_id, meta) != model:
            raise RuntimeError("robot model changed while writing")
        rng = np.random.default_rng(0)  # the window export's gripper sample (same points)
        links = ggeom.gripper_links()
        pool = np.concatenate([ggeom.surface.link_points[n] for n in links])
        pool_link = np.concatenate([[n] * len(ggeom.surface.link_points[n]) for n in links])
        pick = rng.choice(len(pool), min(gripper_points, len(pool)), replace=False)
        source = meta.provenance.source_files[0] if meta.provenance.source_files else None
        timing, calibration_status = {}, {}
        with h5py.File(tmp / FILE, "x") as f:
            f.attrs.update(
                format=FORMAT, schema_version=SCHEMA_VERSION, writer_version=writer_version,
                code_commit=_code_commit() or "unknown", episode_id=episode_id, task=task,
                episode_schema=EPISODE_SCHEMA, created_unix=time.time(),
                raw_recording=str(source.path) if source else "", raw_sha256=str(source.sha256) if source else "",
                cameras=json.dumps([v.rig_id for v in chosen]), reference_camera=chosen[0].rig_id,
                excluded_cameras=json.dumps(excluded), image_size=int(image_size),
                depth_encoding="float32 metres (metric camera-z of the depth stage, bitwise), 0 = invalid",
                conventions=json.dumps(CONVENTIONS), rgb_resample=RGB_RESAMPLE, nearest_resample=NEAREST_RESAMPLE,
                semantic_classes=json.dumps(SEMANTIC_CLASSES if semantic else []),
                upstream=json.dumps(inputs), depth_stage=depth_stage,
                models=json.dumps({depth_stage: (records[depth_stage] or {}).get("summary", {}).get("model"),
                                   "track2d": {"tracker": (records["track2d"] or {}).get("config", {}).get("tracker"),
                                               "worker": (records["track2d"] or {}).get("config", {}).get("worker")},
                                   **({"sam3_masks": {c: s.get("model") for c, s in records["sam3_masks"]["summary"]
                                                      .items() if isinstance(s, dict)}} if semantic else {})},
                                  default=str),
                track_source=((motion_target_source(records["track2d"], depth_stage) if geometry else
                               "unavailable") + "; ONLY tracks classified "
                              "moving by track_motion, full valid trajectories with their motion segments"),
                gripper_target=f"{len(pick)} surface points of the URDF gripper links {links} via FK at the "
                               "camera frame's interpolated robot state (seed-0 sample, as the window export)",
                probe_state="panda flange xyz (m), flange euler xyz (rad, scipy 'xyz'), gripper closure in [0,1] "
                            "(0 open; recorded state through the URDF joint range)",
                max_sync_ns=int(max_sync_ns), MOCK_MODEL_OUTPUTS=bool(mock))
            if partial:
                f.attrs["layout"] = CALIBRATION_OPTIONAL_LAYOUT
            if cal_extra:
                f.attrs["calibration_derived"] = json.dumps(cal_extra)
            tr = meta.trajectory
            g = f.create_group("robot")
            g.create_dataset("timestamps_ns", data=tr.timestamps_ns.astype(np.int64))
            g.create_dataset("joint_positions", data=np.asarray(tr.joint_positions, np.float64), **GZIP)
            g.create_dataset("gripper_state", data=np.asarray(tr.gripper_state, np.float64), **GZIP)
            if tr.ee_pose_base is not None:
                g.create_dataset("ee_pose_base", data=np.asarray(tr.ee_pose_base, np.float64), **GZIP)
            r = meta.robot
            g.attrs.update(joint_names=json.dumps(r.joint_names), flange_frame=r.flange_frame,
                           ee_frame=str(tr.ee_frame), gripper_model=r.gripper_model,
                           gripper_state_semantics=r.gripper_state_semantics, urdf_sha256=json.dumps(r.asset_sha256),
                           robot_sync_max_ns=int(max_sync_ns))
            stamps_all, states_all, ok_all = [], [], []
            for view in chosen:
                cam, rig = view.stream_id, view.rig_id
                t0 = time.perf_counter()
                s = meta.stream(cam)
                stamps = np.asarray(meta.stream_timestamps_ns[cam], np.int64)
                stamps_all.append(stamps)
                n, H, W = len(stamps), s.intrinsics.height, s.intrinsics.width
                box = square_box(H, W)
                aff = box_affine(box, image_size)
                has_geometry = cam in geometry
                if has_geometry:
                    _, _, K_nat = load_view_depth(root, episode_id, cam, [0], depth_stage)
                else:
                    K_nat = s.intrinsics.K()
                w2c = accepted_calibration(root, episode_id, cam)
                cg = f.create_group(f"cameras/{rig}")
                cal_path = rio.stage_dir(root, episode_id, "calibrate") / f"{rio.safe_name(cam)}.json"
                cal = json.loads(cal_path.read_text()) if cal_path.is_file() else {"status": "missing"}
                calibration_status[rig] = cal["status"]
                cg.attrs.update(stream_id=cam, rig_id=rig, modality=s.modality, channel_order="RGB",
                                native_hw=[H, W], depth_stage=depth_stage,
                                calibration=json.dumps({"status": cal["status"], "reasons": cal.get("reasons"),
                                                        "init": cal.get("init_method")}, default=str))
                if partial:
                    cg.attrs.update(calibration_status=cal["status"], geometry_available=has_geometry)
                cg.create_dataset("timestamps_ns", data=stamps)
                cg.create_dataset("segment", data=recording_segments(meta, stamps))
                cg.create_dataset("K_native", data=np.asarray(K_nat, np.float64))
                cg.create_dataset("K", data=aff.apply_K(K_nat))
                cg.create_dataset("affine", data=aff.matrix())
                cg.create_dataset("crop_box", data=np.array([box.x0, box.y0, box.side], np.int64))
                if has_geometry:
                    cg.create_dataset("w2c", data=np.asarray(w2c, np.float64))
                S = int(image_size)
                rgb_ds = cg.create_dataset("rgb", (n, S, S, 3), np.uint8, chunks=(1, S, S, 3), **GZIP)
                if has_geometry:
                    dep_ds = cg.create_dataset("depth", (n, S, S), np.float32, chunks=(1, S, S), shuffle=True, **GZIP)
                sem_path = rio.stage_dir(root, episode_id, "sam3_masks") / f"{rio.safe_name(cam)}.h5"
                camera_semantic = bool(semantic and sem_path.is_file())
                if semantic and not camera_semantic and has_geometry:
                    raise RuntimeError(f"{cam}: missing semantic masks")
                if camera_semantic:
                    sem_ds = cg.create_dataset("semantic", (n, S, S), np.uint8, chunks=(1, S, S), **GZIP)
                    sem_file = h5py.File(sem_path, "r")
                    if json.loads(sem_file.attrs.get("class_names", json.dumps(SEMANTIC_CLASSES))) != list(SEMANTIC_CLASSES):
                        raise RuntimeError(f"{cam}: unexpected semantic classes")
                    cg.create_dataset("class_present", data=sem_file["class_present"][:].astype(bool))
                try:
                    for start in range(0, n, chunk):
                        idx = list(range(start, min(n, start + chunk)))
                        frames = to_model_rgb(read_frames(root, episode_id, cam, idx), s)
                        if frames.shape[1:] != (H, W, 3):
                            raise RuntimeError(f"{cam}: frames {frames.shape} are not RGB {H}x{W}")
                        rgb_ds[start:start + len(idx)] = np.stack([CR.crop_rgb(im, box, S) for im in frames])
                        if has_geometry:
                            depth, valid, _ = load_view_depth(root, episode_id, cam, idx, depth_stage)
                            if scales[cam] != 1.0:  # the accepted calibration's depth scale (consistent with w2c)
                                depth = depth * np.float32(scales[cam])
                            depth = training_depth(depth, valid)
                            dep_ds[start:start + len(idx)] = np.stack([nearest_resample(d, box, S) for d in depth])
                        if camera_semantic:
                            sem_ds[start:start + len(idx)] = np.stack([nearest_resample(sem_file["semantic"][i], box, S)
                                                                       for i in idx])
                finally:
                    if camera_semantic:
                        sem_file.close()
                # robot state at this camera's frame times (the window export's rule at the reference camera)
                q, gs, ok, err = robot_state_at(meta, stamps, max_sync_ns)
                states_all.append(list(zip(q, gs)))
                ok_all.append(np.asarray(ok, bool))
                grip = np.zeros((n, len(pick), 3), np.float32)
                probe = np.zeros((n, 7), np.float32)
                for k in range(n):
                    fk = geom.model.fk(geom.q(q[k], gs[k]))
                    gfk = fk if ggeom is geom else ggeom.model.fk(ggeom.q(q[k], gs[k]))
                    grip[k] = np.stack([gfk[pool_link[i]][:3, :3] @ pool[i] + gfk[pool_link[i]][:3, 3] for i in pick])
                    T = fk[meta.robot.flange_frame]
                    probe[k] = np.concatenate([T[:3, 3], Rotation.from_matrix(T[:3, :3]).as_euler("xyz"),
                                               [geom.closure(gs[k])]])
                rg = cg.create_group("robot")
                rg.create_dataset("ok", data=ok.astype(bool))
                rg.create_dataset("sync_err_ns", data=np.asarray(err, np.int64))
                rg.create_dataset("joint_positions", data=np.asarray(q, np.float64), **GZIP)
                rg.create_dataset("gripper_state", data=np.asarray(gs, np.float64), **GZIP)
                rg.create_dataset("probe_state", data=probe, **GZIP)
                rg.create_dataset("gripper_points", data=grip, shuffle=True, **GZIP)
                if has_geometry:
                    tracks = _camera_tracks(root, episode_id, cam, stamps)
                    tg = cg.create_group("tracks")
                    tg.attrs["chunk_names"] = json.dumps(tracks.pop("_names"))
                    for key, arr in tracks.items():
                        big = arr.ndim >= 2 and arr.size > 64
                        tg.create_dataset(key, data=arr, **({**GZIP, "shuffle": True} if big else {}))
                timing[rig] = round(time.perf_counter() - t0, 2)
            frames_sync, err_sync = mutual_sync(stamps_all, cross_camera_cap_ns)
            disp = set_surface_motion(lambda st: geom.surface.points(geom.q(st[0], st[1]))[0], frames_sync,
                                      states_all, ok_all)
            counts = sync_counts(frames_sync, disp, max_set_motion_m)
            sg = f.create_group("sync")
            sg.create_dataset("frames", data=frames_sync)
            sg.create_dataset("error_ns", data=err_sync)
            sg.create_dataset("surface_disp_m", data=disp.astype(np.float64))  # float64: counts == loaders exactly
            # gripper FK points at the reference frame times, depth-tested (2 cm) against each camera's NATIVE-resolution
            # depth of its synchronised frame: the window export's gripper visibility, kept exactly (G bits per set)
            ref_grip = f[f"cameras/{chosen[0].rig_id}/robot/gripper_points"][:]
            gvis = np.zeros((len(frames_sync), len(chosen), len(pick)), bool)
            for v, view in enumerate(chosen):
                if view.stream_id not in geometry:
                    continue
                K_nat = f[f"cameras/{view.rig_id}/K_native"][:].astype(np.float32)
                w2c = f[f"cameras/{view.rig_id}/w2c"][:].astype(np.float32)
                rows = np.flatnonzero(frames_sync[:, v] >= 0)
                for start in range(0, len(rows), chunk):
                    sel = rows[start:start + chunk]
                    depth, valid, _ = load_view_depth(root, episode_id, view.stream_id, frames_sync[sel, v], depth_stage)
                    depth = np.where(valid, depth * np.float32(scales[view.stream_id]), 0.0).astype(np.float32)
                    for k, t in enumerate(sel):
                        gvis[t, v] = depth_test(ref_grip[t], K_nat, w2c, depth[k], continuous=True)[0]
            if len(geometry) == len(chosen):
                sg.create_dataset("gripper_visible", data=gvis, **GZIP)
            n_points = int(sum(len(x) for x in geom.surface.link_points.values()))
            sg.attrs.update(reference_camera=chosen[0].rig_id, cameras=json.dumps([v.rig_id for v in chosen]),
                            robot_sync_max_ns=int(max_sync_ns), criterion=SYNC_CRITERION,
                            cross_camera_cap_ns=-1 if cross_camera_cap_ns is None else int(cross_camera_cap_ns),
                            default_max_set_motion_m=float(max_set_motion_m), counts=json.dumps(counts),
                            policy=("M5: mutual nearest frame of every camera to the reference frame, no time cap; a set "
                                    "is accepted iff surface_disp_m (largest URDF-surface displacement between any two "
                                    "members, robot state interpolated to each member's timestamp) <= the loader's "
                                    f"threshold (default {max_set_motion_m} m)"),
                            surface=f"RobotGeometry URDF surface, {n_points} points over "
                                    f"{len(geom.surface.link_points)} links (seed 0)")
        sizes = component_sizes(tmp / FILE)
        return {"format": FORMAT, "schema_version": SCHEMA_VERSION, "writer_version": writer_version,
                "code_commit": _code_commit(), "task": task, "cameras": [v.rig_id for v in chosen],
                "excluded_cameras": excluded, "frames": {v.rig_id: len(t) for v, t in zip(chosen, stamps_all)},
                "sync_sets": int((frames_sync >= 0).all(1).sum()), "sync": {"criterion": SYNC_CRITERION, **counts},
                "semantic": bool(semantic),
                "moving_tracks": sizes.pop("_tracks"), "bytes": sizes, "file_bytes": (tmp / FILE).stat().st_size,
                "file_sha256": rio.sha256_file(tmp / FILE), "write_seconds": round(time.perf_counter() - t_start, 2),
                "camera_seconds": timing, "mock": bool(mock),
                **({"layout": CALIBRATION_OPTIONAL_LAYOUT, "calibration_status": calibration_status}
                   if partial else {})}

    return rio.run_stage(out_dir, stage=STAGE, version=writer_version, config=config, inputs=inputs, build=build,
                         force=force)


def component_sizes(path: Path) -> dict:
    """Allocated (compressed) and raw bytes per component of a store file: rgb, depth, semantic, tracks, robot, sync,
    camera metadata; plus the moving-track count per camera (``_tracks``)."""
    out = {"stored": {}, "raw": {}}
    tracks = {}

    def visit(name, obj):
        if not isinstance(obj, h5py.Dataset):
            return
        parts = name.split("/")
        if parts[0] == "cameras":
            key = parts[2] if parts[2] in ("rgb", "depth", "semantic", "tracks", "robot") else "camera_meta"
            if parts[2] == "class_present":
                key = "semantic"
            if parts[2] == "tracks" and parts[-1] == "point_id":
                tracks[parts[1]] = int(obj.shape[0])
        else:
            key = parts[0]
        out["stored"][key] = out["stored"].get(key, 0) + int(obj.id.get_storage_size())
        out["raw"][key] = out["raw"].get(key, 0) + int(obj.size * obj.dtype.itemsize)

    with h5py.File(path, "r") as f:
        f.visititems(visit)
    out["_tracks"] = tracks
    return out


def manifest_fields(record: dict) -> dict:
    """The fields an ingestion manifest records for a completed ``train_data`` stage."""
    s = record["summary"]
    return {"stage": STAGE, "format": s["format"], "schema_version": s["schema_version"],
            "writer_version": s["writer_version"], "code_commit": s["code_commit"], "fingerprint": record["fingerprint"],
            "build_id": record["build_id"], "file": FILE, "file_bytes": s["file_bytes"], "file_sha256": s["file_sha256"],
            "cameras": s["cameras"], "frames": s["frames"], "semantic": s["semantic"], "inputs": record["inputs"],
            "sync": s.get("sync"),
            **({k: s[k] for k in ("layout", "calibration_status")} if "layout" in s else {})}


# ------------------------------------------------------------------------------------------------ reader
class EpisodeFile:
    """Read access to one ``episode.h5`` (lazy, fork/pickle safe: the handle is reopened per process)."""

    def __init__(self, path):
        path = Path(path)
        self.path = path / FILE if path.is_dir() else path
        self._f = None
        self._pid = None
        self._cache: dict = {}
        self._on_open = None
        f = self.file
        if f.attrs.get("format") != FORMAT:
            raise ValueError(f"{self.path}: not a {FORMAT} file")
        major = str(f.attrs["schema_version"]).split(".")[0]
        if major != SCHEMA_VERSION.split(".")[0]:
            raise ValueError(f"{self.path}: schema {f.attrs['schema_version']} (reader {SCHEMA_VERSION})")
        self.episode_id = str(f.attrs["episode_id"])
        self.task = str(f.attrs["task"])
        self.cameras = json.loads(f.attrs["cameras"])
        self.reference_camera = str(f.attrs["reference_camera"])
        self.image_size = int(f.attrs["image_size"])
        self.max_sync_ns = int(f.attrs["max_sync_ns"])
        self.has_semantic = all("semantic" in f[f"cameras/{c}"] for c in self.cameras)
        self.raw_recording = str(f.attrs["raw_recording"])
        self.raw_sha256 = str(f.attrs.get("raw_sha256") or "")
        self.mock = bool(f.attrs.get("MOCK_MODEL_OUTPUTS", False))
        self.close()

    @property
    def file(self) -> h5py.File:
        if self._f is None or self._pid != os.getpid():
            if self._f is not None:  # inherited through fork: never closed (or garbage-collected) in the child
                _INHERITED_HANDLES.append(self._f)
            self._f = h5py.File(self.path, "r", rdcc_nbytes=1 << 20)
            self._pid = os.getpid()
            if self._on_open is not None:  # the owning store's LRU sees every (re)open, also of a retained reader
                self._on_open()
        return self._f

    def close(self) -> None:
        if self._f is not None:
            if self._pid == os.getpid():
                self._f.close()
            else:  # inherited through fork: left to the parent
                _INHERITED_HANDLES.append(self._f)
        self._f = None

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_f"] = state["_pid"] = None
        state["_cache"] = {}  # a new process starts with no cached arrays (its store's LRU starts empty too)
        return state

    def _cam(self, cam: str) -> h5py.Group:
        return self.file[f"cameras/{cam}"]

    def _small(self, key: str, read):
        if key not in self._cache:
            self._cache[key] = read()
        return self._cache[key]

    # frames ---------------------------------------------------------------------------------------------
    def _frames(self, cam: str, name: str, idx) -> np.ndarray:
        ds = self._cam(cam)[name]
        idx = np.atleast_1d(np.asarray(idx, np.int64))
        if len(idx) and (idx.min() < 0 or idx.max() >= ds.shape[0]):
            raise IndexError(f"{cam}/{name}: frame index outside [0, {ds.shape[0]})")
        out = np.empty((len(idx), *ds.shape[1:]), ds.dtype)
        for k, i in enumerate(idx):
            ds.read_direct(out, np.s_[int(i)], np.s_[k])
        return out

    def rgb(self, cam: str, idx) -> np.ndarray:
        return self._frames(cam, "rgb", idx)

    def depth(self, cam: str, idx) -> np.ndarray:
        """(n, S, S) float32 metres, 0 = invalid."""
        return self._frames(cam, "depth", idx)

    def valid(self, cam: str, idx) -> np.ndarray:
        return self._frames(cam, "depth", idx) > 0

    def semantic(self, cam: str, idx) -> np.ndarray:
        if "semantic" not in self._cam(cam):
            raise KeyError(f"{self.path}: no semantic masks (sam3_masks was not part of this store)")
        return self._frames(cam, "semantic", idx)

    # metadata -------------------------------------------------------------------------------------------
    def timestamps(self, cam: str) -> np.ndarray:
        return self._small(("t", cam), lambda: self._cam(cam)["timestamps_ns"][:])

    def segments(self, cam: str) -> np.ndarray:
        return self._small(("seg", cam), lambda: self._cam(cam)["segment"][:].astype(np.int64))

    def num_frames(self, cam: str) -> int:
        return len(self.timestamps(cam))

    def intrinsics(self, cam: str) -> np.ndarray:
        return self._small(("K", cam), lambda: self._cam(cam)["K"][:])

    def intrinsics_native(self, cam: str) -> np.ndarray:
        return self._small(("Kn", cam), lambda: self._cam(cam)["K_native"][:])

    def affine(self, cam: str) -> PixelAffine:
        m = self._small(("A", cam), lambda: self._cam(cam)["affine"][:])
        return PixelAffine(float(m[0, 0]), float(m[1, 1]), float(m[0, 2]), float(m[1, 2]))

    def geometry_problems(self) -> list[str]:
        """Explicit calibration/geometry holds; legacy accepted stores need no new marker."""
        problems = []
        for cam in self.cameras:
            g = self._cam(cam)
            cal = json.loads(g.attrs.get("calibration", "{}"))
            status = g.attrs.get("calibration_status", cal.get("status"))
            partial = self.file.attrs.get("layout") == CALIBRATION_OPTIONAL_LAYOUT
            if (status is not None or partial) and status != "accepted":
                problems.append(f"{cam}: calibration {status}")
            elif not bool(g.attrs.get("geometry_available", True)):
                problems.append(f"{cam}: geometry unavailable")
        return problems

    def require_geometry(self) -> None:
        problems = self.geometry_problems()
        if problems:
            raise ValueError(f"{self.path}: geometry held: {'; '.join(problems)}")

    def w2c(self, cam: str) -> np.ndarray:
        self.require_geometry()
        return self._small(("w2c", cam), lambda: self._cam(cam)["w2c"][:])

    def robot(self, cam: str) -> dict:
        """Robot state at this camera's frame times: ok, sync_err_ns, joint_positions, gripper_state, probe_state,
        gripper_points."""
        return self._small(("robot", cam), lambda: {k: v[:] for k, v in self._cam(cam)["robot"].items()})

    def trajectory(self) -> dict:
        return self._small("traj", lambda: {k: v[:] for k, v in self.file["robot"].items()})

    def tracks(self, cam: str) -> dict:
        self.require_geometry()
        return self._small(("tracks", cam), lambda: {k: v[:] for k, v in self._cam(cam)["tracks"].items()})

    def track_index(self, cam: str) -> dict:
        """The chunk tables only (frames, query frame, row offsets of each TrackEverything query window)."""
        self.require_geometry()
        return self._small(("tindex", cam), lambda: {k: self._cam(cam)["tracks"][k][:]
                                                     for k in ("chunk_frames", "chunk_query", "chunk_offset")})

    def clear_cache(self) -> None:
        self._cache.clear()

    def sync(self) -> dict:
        return self._small("sync", lambda: {k: v[:] for k, v in self.file["sync"].items()})

    def attrs(self) -> dict:
        return dict(self.file.attrs)


class CompactStore:
    """The per-episode per-camera frame store over ``train_data`` outputs (``FrameStore`` protocol of the real-world
    learning adapters: ``episodes``, ``task``, ``cameras``, ``timestamps``, ``segments``, ``rgb``, ``depth``,
    ``semantic``, ``intrinsics``, ``w2c``, ``raw_recording``, ``image_hw``, ``has_semantic``). Episode keys are
    ``<task>/<episode_id>``; cameras are the rig serials. At most ``max_open`` files are open per process."""

    def __init__(self, paths, max_open: int = 16):
        if isinstance(paths, (str, Path)):
            p = Path(paths)
            paths = sorted(p.glob(f"*/{FILE}")) if p.is_dir() and not (p / FILE).exists() else [p]
        self._files = {}
        for path in paths:
            ef = EpisodeFile(path)
            key = f"{ef.task}/{ef.episode_id}"
            if key in self._files:
                raise ValueError(f"episode {key} given twice")
            self._files[key] = ef
        if not self._files:
            raise ValueError("no training-store episodes")
        sizes = {ef.image_size for ef in self._files.values()}
        if len(sizes) != 1:
            raise ValueError(f"mixed image sizes {sizes}")
        size = sizes.pop()
        self.image_hw = (size, size)
        self.has_semantic = all(ef.has_semantic for ef in self._files.values())
        if any(ef.mock for ef in self._files.values()):
            import warnings

            warnings.warn("training store built from MOCK model outputs (tests only)", stacklevel=2)
        self.max_open = int(max_open)
        if self.max_open < 1:
            raise ValueError("max_open must be >= 1")
        self._open: OrderedDict[str, None] = OrderedDict()
        for key, ef in self._files.items():
            ef._on_open = functools.partial(self._touch, key)

    def _touch(self, ep: str) -> None:
        """LRU bookkeeping: bounded open handles AND bounded cached arrays per process."""
        self._open[ep] = None
        self._open.move_to_end(ep)
        while len(self._open) > self.max_open:
            old, _ = self._open.popitem(last=False)
            self._files[old].close()
            self._files[old].clear_cache()

    def episode(self, ep: str) -> EpisodeFile:
        self._touch(ep)
        return self._files[ep]

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_open"] = OrderedDict()
        return state

    def close(self) -> None:
        for ef in self._files.values():
            ef.close()
            ef.clear_cache()
        self._open.clear()

    def episodes(self) -> list[str]:
        return sorted(self._files)

    def task(self, ep):
        return self._files[ep].task

    def cameras(self, ep):
        return list(self._files[ep].cameras)

    def timestamps(self, ep, cam):
        return self.episode(ep).timestamps(cam)

    def segments(self, ep, cam):
        return self.episode(ep).segments(cam)

    def rgb(self, ep, cam, idx):
        return self.episode(ep).rgb(cam, idx)

    def depth(self, ep, cam, idx):
        return self.episode(ep).depth(cam, idx)

    def semantic(self, ep, cam, idx):
        return self.episode(ep).semantic(cam, idx)

    def intrinsics(self, ep, cam):
        return self.episode(ep).intrinsics(cam)

    def w2c(self, ep, cam):
        return self.episode(ep).w2c(cam)

    def raw_recording(self, ep):
        return self._files[ep].raw_recording

    def raw_sha256(self, ep):
        return self._files[ep].raw_sha256

    def sync_sets(self, ep, *, geometry: bool = True) -> dict:
        """The stored cross-camera sets of an episode: ``cameras`` (store order), ``frames`` (F_ref,V), ``times`` of the
        reference frames, ``surface_disp_m`` (F_ref,). Refuses stores written before schema 1.1.0 (no per-set robot
        motion: re-export train_data) and, by default, episodes with held camera calibration/geometry.

        ``geometry=False`` is for consumers that read only the RGB frames of each set (ReViWo pretraining): the sets
        (mutual-nearest frames of the timestamps) and their M5 robot-surface motion (legacy robot kinematics) do not
        depend on the camera calibration, so ``calibration_optional/1`` episodes return the same values as an accepted
        episode would; nothing geometric is returned or opened."""
        ef = self.episode(ep)
        if geometry:
            ef.require_geometry()
        sync = ef.sync()
        if "surface_disp_m" not in sync:
            raise ValueError(f"{ep}: training store schema {ef.attrs().get('schema_version')} has no per-set robot "
                             "motion (sync/surface_disp_m); re-export train_data (schema >= 1.1.0)")
        return {"cameras": list(ef.cameras), "frames": sync["frames"], "surface_disp_m": sync["surface_disp_m"]}

    def upstream(self, ep) -> dict:
        """{stage: build id} of the preprocessing outputs this episode file was built from (attribute ``upstream``;
        geometry-dependent consumers compare its ``calibrate`` with the validated calibration)."""
        raw = self.episode(ep).attrs().get("upstream") or "{}"
        return json.loads(raw.decode() if isinstance(raw, bytes) else raw)


def verify_snerl_teacher(device: str = "cpu", image: np.ndarray | None = None) -> dict:
    """The online SNeRL DINO target path is available and pinned: the official weights (sha256), the pinned
    ``third_party/dino`` checkout, and ``dino_target`` on a stored 128x128 RGB frame equal to the cached-teacher
    token path (``dino_snerl_worker``: the first lines of ``dino_target`` + the consumer's upsampling). Nothing is
    written; this replaces the retired ``dino_snerl`` cache stage."""
    import torch
    import torch.nn.functional as F

    from s4d.baselines.snerl.model import dino_target, to_device_scene
    from s4d.baselines.snerl.upstream import DINO_WEIGHTS_SHA256, dino_extractor, dino_weights_path, sha256_file
    from s4d.third_party import source_tree

    if image is None:
        image = (np.random.default_rng(0).random((IMAGE_SIZE, IMAGE_SIZE, 3)) * 255).astype(np.uint8)
    tree = source_tree("dino")
    digest = sha256_file(dino_weights_path())
    extractor = dino_extractor(device)
    target = to_device_scene({"target_rgb": image}, torch.device(device))["target_rgb"]  # uint8 / 255, as SNeRLScenes
    online = dino_target(extractor, target)
    with torch.no_grad():
        tok = extractor.extract_descriptors(((target - 0.45) / 0.225).unsqueeze(0).permute(0, 3, 1, 2), 11, "key", False)
        tok = tok.view(1, *extractor.num_patches, -1)
        cached = F.interpolate(tok.permute(0, 3, 1, 2), tuple(target.shape[:2]), mode="bilinear").permute(0, 2, 3, 1)[0]
    return {"weights_sha256_ok": digest == DINO_WEIGHTS_SHA256, "dino_tree": str(tree),
            "token_grid": list(tok.shape[1:3]), "dim": int(tok.shape[-1]), "target_shape": list(online.shape),
            "online_equals_cached_path": bool(torch.equal(online, cached)),
            "max_abs_diff": float((online - cached).abs().max())}
