"""The normalized episode and its shared access helpers on each RGB-D camera's colour grid: the ``ingest`` stage,
normalized frames and metadata, camera views, view depth, robot state at camera time, robot geometry, accepted
calibrations and track-chunk helpers. The stages live in ``calibration_stage`` (calibrate; ``calibrate`` and
``fk_vs_recorded_ee`` are also importable from here), ``depth_stages`` (sensor depth), ``ir_stereo_depth``
(FoundationStereo on the device IR pair), ``trackeverything_stage``, ``track_motion``, ``export`` and
``baseline_stages`` (sam3_masks, dino_snerl).

Every stage runs through ``s4d.realworld.io.run_stage`` (fingerprinted completion record, atomic swap-in) and
processes frames in bounded chunks. Model calls go through isolated workers; test doubles are flagged ``is_mock`` and
the exporter refuses mock-derived data unless explicitly allowed (and then stamps the manifest).
"""

from __future__ import annotations

import json
from pathlib import Path

import h5py
import numpy as np

from s4d.realworld import io as rio
from s4d.realworld.contract import CameraView, EpisodeMeta, meta_from_json, meta_to_json, validate_episode
from s4d.realworld.robotics.model import (
    GRIPPER_MAP_LEGACY,
    GRIPPER_MAPS,
    ROBOTIQ_FINGER_JOINT_MAX,
    RobotSurface,
    URDFModel,
    gripper_joint_value,
    joint_dict,
)
from s4d.realworld.tracking import Tracks2D

STAGE_VERSIONS = {"ingest": "1", "calibrate": "2", "export": "1"}
# calibrate runs on the registered sensor depth; lifting/export use FoundationStereo depth from the device IR pair (B1)
RGBD_CALIBRATION_DEPTH, RGBD_LIFT_DEPTH = "sensor_depth", "foundation_stereo_depth"


# ---------------------------------------------------------------- model input


def to_model_rgb(frames: np.ndarray, stream) -> np.ndarray:
    """Frames as models expect them: RGB channel order (BGR flipped). Infrared stays 1-channel."""
    if frames.ndim == 4 and stream.channel_order == "BGR":
        return np.ascontiguousarray(frames[..., ::-1])
    return frames


# ---------------------------------------------------------------- normalized episode access


def load_meta(root: Path, episode_id: str) -> EpisodeMeta:
    d = rio.episode_dir(root, episode_id) / "normalized"
    doc = json.loads((d / "episode.json").read_text())
    with np.load(d / "arrays.npz") as z:
        arrays = {k: z[k] for k in z.files}
    return meta_from_json(doc, arrays)


def read_frames(root: Path, episode_id: str, stream_id: str, indices) -> np.ndarray:
    path = rio.episode_dir(root, episode_id) / "normalized" / "frames" / f"{rio.safe_name(stream_id)}.h5"
    with h5py.File(path, "r") as f:
        idx = np.asarray(indices)
        order = np.argsort(idx)
        data = f["frames"][idx[order]] if len(idx) else f["frames"][:0]
        out = np.empty_like(data)
        out[order] = data
        return out


def _record_fp(root, episode_id, stage) -> str:
    rec = rio.read_record(rio.stage_dir(root, episode_id, stage)) if stage != "ingest" else rio.read_record(
        rio.episode_dir(root, episode_id) / "normalized")
    if rec is None:
        raise RuntimeError(f"stage {stage} of {episode_id} is not complete")
    return rio.build_id(rec)


# ---------------------------------------------------------------- stage 1: ingest


def ingest(adapter, episode_id: str, root: Path, *, chunk: int = 64, force: bool = False) -> dict:
    meta = adapter.read_meta(episode_id)
    report = validate_episode(meta)
    report.raise_if_errors()
    files = adapter.source_files(episode_id)

    def build(tmp: Path) -> dict:
        doc, arrays = meta_to_json(meta)
        doc["provenance"]["source_files"] = [
            {"path": str(p), "sha256": rio.sha256_file(p), "bytes": p.stat().st_size} for p in files]
        rio.atomic_write_json(tmp / "episode.json", doc)
        rio.atomic_write_npz(tmp / "arrays.npz", **arrays)
        counts = {}
        for s in meta.streams:
            n = len(meta.stream_timestamps_ns[s.stream_id])
            path = tmp / "frames" / f"{rio.safe_name(s.stream_id)}.h5"
            path.parent.mkdir(parents=True, exist_ok=True)
            shape = (s.intrinsics.height, s.intrinsics.width) + ((s.channels,) if s.channels > 1 else ())
            with h5py.File(path, "w") as f:
                ds = f.create_dataset("frames", (n,) + shape, dtype=s.dtype, chunks=(1,) + shape, compression="lzf")
                ts = f.create_dataset("timestamps_ns", (n,), dtype=np.int64)
                f.attrs.update(stream_id=s.stream_id, modality=s.modality, channel_order=s.channel_order)
                count = 0
                for start in range(0, n, chunk):
                    for i, stamp, arr in adapter.iter_frames(episode_id, s.stream_id, start, min(n, start + chunk)):
                        if arr.shape != shape or str(arr.dtype) != s.dtype:
                            raise ValueError(f"{s.stream_id}[{i}]: {arr.shape}/{arr.dtype} != {shape}/{s.dtype}")
                        if stamp != meta.stream_timestamps_ns[s.stream_id][i]:
                            raise ValueError(f"{s.stream_id}[{i}]: frame timestamp disagrees with metadata")
                        ds[i], ts[i] = arr, stamp
                        count += 1
                if count != n:
                    raise ValueError(f"{s.stream_id}: {count} frames read, {n} timestamps")
            counts[s.stream_id] = n
        return {"warnings": report.warnings, "frames": counts, "assumptions": list(meta.assumptions)}

    config = {"adapter": adapter.name, "adapter_version": adapter.version, "adapter_config": adapter.config}
    source_identity = getattr(adapter, "ingest_identity", lambda _: {})(episode_id)
    if source_identity:  # (absent otherwise: earlier fingerprints stay valid)
        config["source_identity"] = source_identity
    return rio.run_stage(rio.episode_dir(root, episode_id) / "normalized", stage="ingest",
                         version=STAGE_VERSIONS["ingest"], config=config, inputs={}, build=build, force=force)


# ---------------------------------------------------------------- camera views


def camera_views(meta: EpisodeMeta) -> list[CameraView]:
    """The camera views every stage processes: the colour streams of the RGB-D rigs. Stereo camera views (a stereo rig's
    left eye) have no rectification, depth or tracking stage here and are refused rather than processed on a raw grid."""
    views = meta.views()
    stereo = [v.stream_id for v in views if v.kind != "rgbd"]
    if stereo:
        raise NotImplementedError(f"{meta.episode_id}: stereo camera views {stereo} are not processed by this pipeline "
                                  "(RGB-D rigs only)")
    return views


def view_frames(root, episode_id, meta: EpisodeMeta, stream_id: str, indices) -> np.ndarray:
    """Frames of a camera view on the grid its depth, tracks and masks use: the recorded (zero-distortion) colour grid
    of an RGB-D rig."""
    next(v for v in camera_views(meta) if v.stream_id == stream_id)
    return read_frames(root, episode_id, stream_id, indices)


def default_depth_stage(meta: EpisodeMeta, purpose: str) -> str:
    """RGB-D calibrates on sensor depth and lifts with the device-IR FoundationStereo depth."""
    if purpose not in ("calibrate", "lift"):
        raise ValueError(f"unknown depth purpose {purpose!r}")
    camera_views(meta)
    return RGBD_CALIBRATION_DEPTH if purpose == "calibrate" else RGBD_LIFT_DEPTH


def view_depth(root, episode_id, meta: EpisodeMeta, stream_id: str, indices, stage: str):
    """(depth (n,H,W), K continuous) of a view from ``stage`` on that view's grid."""
    from s4d.realworld.stages.depth import load_view_depth

    depth, valid, K = load_view_depth(root, episode_id, stream_id, indices, stage)
    return np.where(valid, depth, 0.0).astype(np.float32), K


# ---------------------------------------------------------------- robot state at camera time


def robot_state_at(meta: EpisodeMeta, stamps_ns: np.ndarray, max_sync_ns: int = 20_000_000):
    tr = meta.trajectory
    ts = tr.timestamps_ns.astype(np.float64)
    q = np.stack([np.interp(stamps_ns, ts, tr.joint_positions[:, j]) for j in range(tr.joint_positions.shape[1])], 1)
    g = np.interp(stamps_ns, ts, tr.gripper_state)
    nearest = np.abs(ts[np.clip(np.searchsorted(ts, stamps_ns), 0, len(ts) - 1)] - stamps_ns)
    nearest = np.minimum(nearest, np.abs(ts[np.clip(np.searchsorted(ts, stamps_ns) - 1, 0, len(ts) - 1)] - stamps_ns))
    inside = (stamps_ns >= ts[0]) & (stamps_ns <= ts[-1]) & (nearest <= max_sync_ns)
    return q, g, inside, nearest


ROBOT_SURFACES = ("collision", "visual")
LEGACY_ROBOT_MODEL = {"gripper_map": GRIPPER_MAP_LEGACY, "surface": "collision"}


class RobotGeometry:
    """Composed URDF + per-link surface samples + the recorded-gripper -> joint map.

    ``gripper_map`` (``robot.GRIPPER_MAPS``; default the legacy register map) is used where the geometry is POSED
    (``q``); ``closure`` (the proprioceptive closure in [0, 1] stored for learners) always keeps the legacy map, so
    consumers that do not pose the robot never change. ``surface``: ``collision`` (default; the URDF's STL hulls) or
    ``visual`` (COLLADA visual meshes; fails closed when any is missing, ``RobotSurface.build_visual``)."""

    def __init__(self, meta: EpisodeMeta, points_per_link: int = 300, closed_value: float | None = None, *,
                 gripper_map: str | None = None, surface: str = "collision"):
        r = meta.robot
        if r.urdf_path is None or r.gripper_model in ("", "unknown"):
            raise ValueError("robot-grounded stages need the composed Panda+Robotiq URDF and the exact gripper model")
        if surface not in ROBOT_SURFACES:
            raise ValueError(f"robot surface {surface!r} not in {ROBOT_SURFACES}")
        self.gripper_map = gripper_map or GRIPPER_MAP_LEGACY
        if self.gripper_map not in GRIPPER_MAPS:
            raise ValueError(f"unknown gripper map {self.gripper_map!r}")
        self.model = URDFModel.from_file(r.urdf_path)
        build = RobotSurface.build_visual if surface == "visual" else RobotSurface.build
        self.surface = build(self.model, points_per_link, seed=0, mesh_root=r.mesh_root)
        self.surface_kind = surface
        self.desc = r
        self.closed_value = closed_value
        if r.gripper_model not in ("robotiq_2f_85", "robotiq_2f_140") and closed_value is None:
            lim = self.model.joints[r.gripper_joint].limits if r.gripper_joint in self.model.joints else None
            self.closed_value = lim[1] if lim else None
        if self.gripper_map != GRIPPER_MAP_LEGACY:  # refuse a map the recorded gripper does not have, at construction
            gripper_joint_value(0.0, r.gripper_state_semantics, r.gripper_model, closed_value=self.closed_value,
                                mapping=self.gripper_map)

    @property
    def robot_model(self) -> dict:
        """What a fingerprint must record when it differs from ``LEGACY_ROBOT_MODEL``."""
        return {"gripper_map": self.gripper_map, "surface": self.surface_kind}

    def closure(self, grip) -> float:
        """Recorded gripper state -> closure in [0, 1] (0 open) through the URDF joint range (legacy map, always)."""
        top = self.closed_value if self.closed_value is not None else ROBOTIQ_FINGER_JOINT_MAX.get(self.desc.gripper_model)
        value = gripper_joint_value(grip, self.desc.gripper_state_semantics, self.desc.gripper_model,
                                    closed_value=self.closed_value)
        return float(np.clip(value / top, 0, 1)) if top else float("nan")

    def q(self, q7, grip) -> dict:
        value = gripper_joint_value(grip, self.desc.gripper_state_semantics, self.desc.gripper_model,
                                    closed_value=self.closed_value, mapping=self.gripper_map)
        return joint_dict(self.desc.joint_names, q7, self.desc.gripper_joint, value)

    def gripper_links(self) -> list[str]:
        """Links with geometry in the gripper subtree (below ``gripper_base_frame``). Without that frame, the old rule
        (everything but ``panda_link0..8``) is kept; it misclassifies arms with other link names (e.g. FR3)."""
        root = self.desc.gripper_base_frame
        if root is None:
            arm = {f"panda_link{i}" for i in range(9)}
            return [n for n in sorted(self.surface.link_points) if n not in arm]
        if root not in self.model.links:
            raise ValueError(f"gripper_base_frame {root!r} is not a link of the URDF")
        subtree, stack = set(), [root]
        while stack:
            link = stack.pop()
            subtree.add(link)
            stack += [j.child for j in self.model._children.get(link, [])]
        return [n for n in sorted(self.surface.link_points) if n in subtree]


# ---------------------------------------------------------------- calibration access


def accepted_calibration_doc(root, episode_id, camera_id) -> dict | None:
    """The episode's ACCEPTED calibrate record of a camera (``calibrate/<camera>.json``), None when missing or
    rejected. The single source of truth for everything a geometry consumer takes from the calibration: ``T_cam_base``
    (``accepted_calibration``), the per-depth-stage ``depth_scale`` (``calibration_depth_scale``) and the robot model
    the calibration was validated with (``calibration_robot_model``)."""
    path = rio.stage_dir(root, episode_id, "calibrate") / f"{rio.safe_name(camera_id)}.json"
    if not path.is_file():
        return None
    doc = json.loads(path.read_text())
    return doc if doc["status"] == "accepted" else None


def accepted_calibration(root, episode_id, camera_id) -> np.ndarray | None:
    doc = accepted_calibration_doc(root, episode_id, camera_id)
    return None if doc is None else np.asarray(doc["T_cam_base"])


# per-camera depth scale (calibration release 2026-11-08): the rig solve estimates each camera's metric depth scale
# jointly with its pose (log z_true = log d + alpha); the pose is only consistent with depth multiplied by exp(alpha).
DEPTH_SCALE_RULE = "calibrate-record depth_scale/1: depth_used = depth[stage] * calibrate[camera].depth_scale[stage]"
DEPTH_SCALE_BOUNDS = (0.9, 1.1)  # sanity bounds of a factor (measured 0.99-1.05); outside: refuse


def calibration_depth_scale(root, episode_id, camera_id, stage: str) -> float:
    """Factor that makes ``stage``'s depth of ``camera_id`` consistent with its accepted pose.

    1.0 when the camera has no accepted calibration (no pose to be consistent with) or its accepted record carries no
    ``depth_scale`` map (per-episode calibrate@1 and rig documents without depth scales: unchanged behaviour). A record
    WITH a map that lacks ``stage`` raises (the pose was fitted against scaled depth of other sources; a scale must never
    be borrowed from another depth source or silently dropped). Non-finite or implausible factors raise."""
    doc = accepted_calibration_doc(root, episode_id, camera_id)
    if doc is None or doc.get("depth_scale") is None:
        return 1.0
    scales = doc["depth_scale"]
    if stage not in scales:
        raise ValueError(f"{episode_id} {camera_id}: the accepted calibration has depth scales for {sorted(scales)} "
                         f"but not for {stage!r}; refusing to use unscaled {stage} depth with its pose")
    f = float(scales[stage])
    if not (np.isfinite(f) and DEPTH_SCALE_BOUNDS[0] <= f <= DEPTH_SCALE_BOUNDS[1]):
        raise ValueError(f"{episode_id} {camera_id}: depth scale {f} of {stage} outside {DEPTH_SCALE_BOUNDS}")
    return f


def episode_depth_scales(root, episode_id, meta: EpisodeMeta, stage: str) -> dict:
    """{camera stream: factor} of every camera view (``calibration_depth_scale``)."""
    return {v.stream_id: calibration_depth_scale(root, episode_id, v.stream_id, stage) for v in camera_views(meta)}


def calibrated_view_depth(root, episode_id, meta: EpisodeMeta, stream_id: str, indices, stage: str):
    """``view_depth`` multiplied by the camera's accepted ``depth_scale`` of ``stage`` (invalid stays 0). Every
    consumer that combines depth with the accepted pose uses this (TrackEverything/lift3d, training store, window
    export, viewer store, rwlearn processed store); ``view_depth`` stays the raw stage depth."""
    depth, K = view_depth(root, episode_id, meta, stream_id, indices, stage)
    f = calibration_depth_scale(root, episode_id, stream_id, stage)
    return (depth if f == 1.0 else (depth * np.float32(f)).astype(np.float32)), K


def calibration_robot_model(root, episode_id, meta: EpisodeMeta) -> dict:
    """The robot model (gripper map, surface) the episode's accepted calibrations were validated with: the
    ``robot_model`` field of the accepted calibrate records (rig release documents set it), else the legacy model.
    Accepted cameras must agree (raises otherwise)."""
    models = []
    for v in camera_views(meta):
        doc = accepted_calibration_doc(root, episode_id, v.stream_id)
        if doc is not None:
            models.append({**LEGACY_ROBOT_MODEL, **(doc.get("robot_model") or {})})
    keyed = {json.dumps(m, sort_keys=True) for m in models}
    if len(keyed) > 1:
        raise ValueError(f"{episode_id}: accepted calibrations disagree on the robot model: {sorted(keyed)}")
    return models[0] if models else dict(LEGACY_ROBOT_MODEL)


def calibrated_geometry_config(root, episode_id, meta: EpisodeMeta, depth_stage: str | None = None) -> dict:
    """The calibration-derived settings a geometry stage must add to its fingerprint config: ``depth_scale`` (when any
    camera's factor differs from 1) and ``robot_model`` (when it differs from the legacy model). Empty for legacy
    calibrations, so existing fingerprints, build ids and outputs stay valid; non-empty -> new build ids."""
    out = {}
    if depth_stage is not None:
        scales = episode_depth_scales(root, episode_id, meta, depth_stage)
        if any(f != 1.0 for f in scales.values()):
            out["depth_scale"] = {"stage": depth_stage, "factors": scales, "rule": DEPTH_SCALE_RULE}
    model = calibration_robot_model(root, episode_id, meta)
    if model != LEGACY_ROBOT_MODEL:
        out["robot_model"] = model
    return out


def calibrated_version(version: str, extra: dict) -> str:
    """Stage version of an output built with calibration-derived settings (``calibrated_geometry_config``): bumped
    (``+cal1``) only when they apply, so legacy outputs keep their version and fingerprint."""
    return f"{version}+cal1" if extra else version


def calibrated_robot_geometry(root, episode_id, meta: EpisodeMeta, **kw) -> RobotGeometry:
    """RobotGeometry posed with the gripper map (and surface) the accepted calibration was validated with."""
    model = calibration_robot_model(root, episode_id, meta)
    return RobotGeometry(meta, gripper_map=model["gripper_map"], surface=model["surface"], **kw)


# ---------------------------------------------------------------- tracking chunks


def chunk_queries(n: int, window: int, step: int) -> list[tuple[int, list[int]]]:
    return [(s, list(range(s, min(n, s + window)))) for s in range(0, max(1, n - 2), step)]


def load_tracks2d(path: Path, camera_id: str) -> Tracks2D:
    with np.load(path) as z:
        return Tracks2D(camera_id, int(z["query_frame"]), z["frame_indices"], z["timestamps_ns"], z["point_ids"],
                        z["uv"], z["visibility"], z["confidence"], tuple(z["image_hw"]), json.loads(str(z["source"])))


def camera_poses(root, episode_id, meta) -> dict:
    return {v.stream_id: accepted_calibration(root, episode_id, v.stream_id) for v in camera_views(meta)}


# the calibrate stage lives in calibration_stage (imported last: it reads this module's helpers at call time)
from s4d.realworld.stages.calibrate import calibrate, fk_vs_recorded_ee  # noqa: E402,F401
