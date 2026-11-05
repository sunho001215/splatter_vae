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
from s4d.realworld.robot import (
    ROBOTIQ_FINGER_JOINT_MAX,
    RobotSurface,
    URDFModel,
    gripper_joint_value,
    joint_dict,
)
from s4d.realworld.tracking import Tracks2D

STAGE_VERSIONS = {"ingest": "1", "calibrate": "1", "export": "1"}
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

    return rio.run_stage(rio.episode_dir(root, episode_id) / "normalized", stage="ingest",
                         version=STAGE_VERSIONS["ingest"], config={"adapter": adapter.name,
                                                                   "adapter_version": adapter.version,
                                                                   "adapter_config": adapter.config},
                         inputs={}, build=build, force=force)


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
    from s4d.realworld.depth_stages import load_view_depth

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


class RobotGeometry:
    def __init__(self, meta: EpisodeMeta, points_per_link: int = 300, closed_value: float | None = None):
        r = meta.robot
        if r.urdf_path is None or r.gripper_model in ("", "unknown"):
            raise ValueError("robot-grounded stages need the composed Panda+Robotiq URDF and the exact gripper model")
        self.model = URDFModel.from_file(r.urdf_path)
        self.surface = RobotSurface.build(self.model, points_per_link, seed=0, mesh_root=r.mesh_root)
        self.desc = r
        self.closed_value = closed_value
        if r.gripper_model not in ("robotiq_2f_85", "robotiq_2f_140") and closed_value is None:
            lim = self.model.joints[r.gripper_joint].limits if r.gripper_joint in self.model.joints else None
            self.closed_value = lim[1] if lim else None

    def closure(self, grip) -> float:
        """Recorded gripper state -> closure in [0, 1] (0 open) through the URDF joint range."""
        top = self.closed_value if self.closed_value is not None else ROBOTIQ_FINGER_JOINT_MAX.get(self.desc.gripper_model)
        value = gripper_joint_value(grip, self.desc.gripper_state_semantics, self.desc.gripper_model,
                                    closed_value=self.closed_value)
        return float(np.clip(value / top, 0, 1)) if top else float("nan")

    def q(self, q7, grip) -> dict:
        value = gripper_joint_value(grip, self.desc.gripper_state_semantics, self.desc.gripper_model,
                                    closed_value=self.closed_value)
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


def accepted_calibration(root, episode_id, camera_id) -> np.ndarray | None:
    path = rio.stage_dir(root, episode_id, "calibrate") / f"{rio.safe_name(camera_id)}.json"
    if not path.is_file():
        return None
    doc = json.loads(path.read_text())
    return np.asarray(doc["T_cam_base"]) if doc["status"] == "accepted" else None


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
from s4d.realworld.calibration_stage import calibrate, fk_vs_recorded_ee  # noqa: E402,F401
