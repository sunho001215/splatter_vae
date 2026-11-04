"""Restartable per-episode stages: ingest -> stereo_depth -> calibrate -> track2d -> lift3d -> export
(+ optional dino_snerl, sam3_masks in s4d.realworld.baseline_stages).

Every stage runs through ``s4d.realworld.io.run_stage`` (fingerprinted completion record, atomic swap-in) and
processes frames in bounded chunks. Model calls go through backends; the real ones wrap isolated workers
(``WorkerStereoBackend`` / ``WorkerTrackerBackend``), test ones are flagged ``is_mock`` and the exporter refuses
mock-derived data unless explicitly allowed (and then stamps the manifest).
"""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import h5py
import numpy as np

from s4d.realworld import io as rio
from s4d.realworld.calibration import CalibrationConfig
from s4d.realworld.contract import EpisodeMeta, meta_from_json, meta_to_json, validate_episode
from s4d.realworld.geometry import (
    rectify_map,
    remap,
    resize_affine,
    stereo_rectify,
    undistort_pixels,
    warp_affine,
)
from s4d.realworld.robot import (
    ROBOTIQ_FINGER_JOINT_MAX,
    RobotSurface,
    URDFModel,
    gripper_joint_value,
    joint_dict,
    landmarks,
)
from s4d.realworld.runners import LocalRunner
from s4d.realworld.stereo import depth_from_pair
from s4d.realworld.tracking import Tracks2D, alltracker_to_tracks2d, query_grid

STAGE_VERSIONS = {"ingest": "1", "stereo_depth": "1", "calibrate": "1", "track2d": "1", "lift3d": "1", "export": "1"}
# RGB-D episodes: calibrate runs on the registered sensor depth, lifting/export use MapAnything depth
RGBD_CALIBRATION_DEPTH, RGBD_LIFT_DEPTH = "sensor_depth", "mapanything_depth"


# ---------------------------------------------------------------- backends


def backend_config(backend) -> dict:
    """Everything that determines a backend's output (worker spec incl. commit/checkpoint, params, mock flag)."""
    if hasattr(backend, "spec"):
        from dataclasses import asdict as _asdict

        spec = _asdict(backend.spec)
        spec.pop("timeout_s", None)
        return {"class": type(backend).__name__, "spec": spec, "params": backend.params,
                "allow_mock": backend.allow_mock}
    return {"class": type(backend).__name__, "identity": getattr(backend, "identity", {})}


def to_model_rgb(frames: np.ndarray, stream) -> np.ndarray:
    """Frames as models expect them: RGB channel order (BGR flipped). Infrared stays 1-channel here; workers'
    inputs replicate it (``_model_input_rgb``) and the stage records say so."""
    if frames.ndim == 4 and stream.channel_order == "BGR":
        return np.ascontiguousarray(frames[..., ::-1])
    return frames


class StereoBackend:
    identity: dict = {}

    def disparity(self, left: np.ndarray, right: np.ndarray) -> np.ndarray:
        """(N,H,W[,C]) rectified pairs -> (N,H,W) left-reference disparity."""
        raise NotImplementedError


class TrackerBackend:
    identity: dict = {}

    def track(self, camera_id: str, frames: np.ndarray, frame_indices: list, query: int) -> tuple:
        """frames (T,H,W[,C]) -> (flows (T',2,H,W), visconf (T',2,H,W)) for frames >= query (AllTracker layout)."""
        raise NotImplementedError


def _model_input_rgb(x: np.ndarray) -> np.ndarray:
    """Infrared (N,H,W) -> (N,H,W,3) by channel replication, as MODEL INPUT ONLY (recorded in stage records)."""
    return np.repeat(x[..., None], 3, -1) if x.ndim == 3 else x


class WorkerStereoBackend(StereoBackend):
    def __init__(self, spec, workdir: Path, params: dict | None = None, allow_mock: bool = False):
        self.spec, self.workdir, self.params, self.allow_mock = spec, Path(workdir), dict(params or {}), allow_mock
        self.identity = {"worker": spec.name}

    def disparity(self, left, right):
        from s4d.realworld.workers import call_worker

        resp, out = call_worker(self.spec, "stereo_disparity", self.params,
                                {"left": _model_input_rgb(left), "right": _model_input_rgb(right)}, self.workdir,
                                allow_mock=self.allow_mock)
        self.identity = {**resp["model"], "meta": resp["meta"]}
        return out["disparity"]


class WorkerTrackerBackend(TrackerBackend):
    """AllTracker worker. With a ``runner`` (``s4d.realworld.runners``) the call goes through it and the caller
    acknowledges ``last_handle`` after saving; without one, the file protocol runs locally as before."""

    def __init__(self, spec, workdir: Path, params: dict | None = None, allow_mock: bool = False, runner=None):
        self.spec, self.workdir, self.params, self.allow_mock = spec, Path(workdir), dict(params or {}), allow_mock
        self.identity = {"worker": spec.name}
        self.runner, self.last_handle = runner, None

    def track(self, camera_id, frames, frame_indices, query, chunk_id: str | None = None):
        params, arrays = {**self.params, "query_frame": int(query)}, {"rgbs": _model_input_rgb(frames)}
        if self.runner is not None:
            resp, out, self.last_handle = self.runner.worker(self.spec, "track", params, arrays,
                                                             chunk_id=chunk_id or f"{camera_id}/{frame_indices[0]}",
                                                             allow_mock=self.allow_mock)
        else:
            from s4d.realworld.workers import call_worker

            resp, out = call_worker(self.spec, "track", params, arrays, self.workdir, allow_mock=self.allow_mock)
        self.identity = {**resp["model"], "meta": resp["meta"]}
        return out["flows"], out["visconf"]


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


# ---------------------------------------------------------------- stage 2: stereo depth


def rig_rectification(meta: EpisodeMeta, rig: str):
    g = meta.rig_stereo(rig)
    L, R = meta.stream(g.left_stream), meta.stream(g.right_stream)
    KL, KR = L.intrinsics.K(), R.intrinsics.K()
    hw = (L.intrinsics.height, L.intrinsics.width)
    if L.rectified and R.rectified:
        if L.intrinsics.distortion_model != "none" or R.intrinsics.distortion_model != "none":
            raise ValueError(f"{rig}: streams flagged rectified but carry distortion")
        return None, KL, KR, hw
    if L.rectified is None or R.rectified is None:
        raise ValueError(f"{rig}: rectification state unknown; the adapter must decide")
    rect = stereo_rectify(KL, KR, g.T(), hw)
    return rect, rect.K_rect, rect.K_rect, hw


def stereo_depth(root: Path, episode_id: str, backend: StereoBackend, *, chunk: int = 16,
                 references: tuple = ("left",), pair_sync_ns: int = 1_000_000, force: bool = False) -> dict:
    meta = load_meta(root, episode_id)

    def build(tmp: Path) -> dict:
        summary = {}
        for g in meta.stereo:
            if g.status != "metric":
                summary[g.rig_id] = {"skipped": f"stereo geometry {g.status}: no metric depth (scale never invented)"}
                continue
            rect, KL, KR, hw = rig_rectification(meta, g.rig_id)
            L, R = meta.stream(g.left_stream), meta.stream(g.right_stream)
            n = len(meta.stream_timestamps_ns[g.left_stream])
            if len(meta.stream_timestamps_ns[g.right_stream]) != n:
                raise ValueError(f"{g.rig_id}: left/right frame counts differ; pair by timestamp in the adapter")
            pair_err = np.abs(meta.stream_timestamps_ns[g.left_stream] - meta.stream_timestamps_ns[g.right_stream])
            if pair_err.max() > pair_sync_ns:
                raise ValueError(f"{g.rig_id}: left/right frames up to {pair_err.max()} ns apart (> {pair_sync_ns})")
            maps = None
            if rect is not None:
                maps = (rectify_map(L.intrinsics.K(), L.intrinsics.distortion_model, L.intrinsics.distortion, rect.R1,
                                    rect.K_rect, hw),
                        rectify_map(R.intrinsics.K(), R.intrinsics.distortion_model, R.intrinsics.distortion, rect.R2,
                                    rect.K_rect, hw))
            with h5py.File(tmp / f"{g.rig_id}.h5", "w") as f:
                f.attrs.update(rig_id=g.rig_id, left_stream=g.left_stream, right_stream=g.right_stream,
                               baseline_m=g.baseline_m, K_left=KL, K_right=KR, modality=L.modality,
                               rectification=json.dumps(rect.to_json() if rect else "input already rectified"),
                               depth_convention="metric camera-z, 0 invalid", disparity="x_left - x_right")
                for ref in references:
                    grp = f.create_group(ref)
                    grp.create_dataset("depth", (n, *hw), np.float32, chunks=(1, *hw), compression="lzf")
                    grp.create_dataset("valid", (n, *hw), bool, chunks=(1, *hw), compression="lzf")
                    grp.create_dataset("disparity", (n, *hw), np.float32, chunks=(1, *hw), compression="lzf")
                    if rect is not None:
                        grp.create_dataset("rectified_image", (n, *hw) + ((3,) if L.channels == 3 else ()),
                                           L.dtype, compression="lzf")
                for start in range(0, n, chunk):
                    idx = list(range(start, min(n, start + chunk)))
                    left = to_model_rgb(read_frames(root, episode_id, g.left_stream, idx), L)
                    right = to_model_rgb(read_frames(root, episode_id, g.right_stream, idx), R)
                    if maps is not None:
                        left = np.stack([remap(x, maps[0], order=1) for x in left])
                        right = np.stack([remap(x, maps[1], order=1) for x in right])
                    batch_left = backend.disparity(left, right)
                    # right reference: flip-and-swap inputs for the whole chunk in ONE backend call (a worker call
                    # loads the model once); depth_from_pair flips the returned disparity back.
                    batch_flipped = (backend.disparity(np.ascontiguousarray(right[:, :, ::-1]),
                                                       np.ascontiguousarray(left[:, :, ::-1]))
                                     if "right" in references else None)
                    for k, i in enumerate(idx):
                        for ref in references:
                            if ref == "left":
                                matcher = (lambda _a, _b, d=batch_left[k]: d)
                            else:
                                matcher = (lambda _a, _b, d=batch_flipped[k]: d)
                            res = depth_from_pair(left[k], right[k], KL, KR, g.baseline_m, matcher, ref)
                            f[ref]["depth"][i], f[ref]["valid"][i], f[ref]["disparity"][i] = (res.depth, res.valid,
                                                                                               res.disparity)
                            if maps is not None:
                                f[ref]["rectified_image"][i] = (left if ref == "left" else right)[k]
            summary[g.rig_id] = {"frames": n, "references": list(references), "rectified_here": rect is not None,
                                 "model": backend.identity,
                                 "model_input": "infrared replicated to 3 channels (model input only)"
                                 if L.modality == "infrared" else "rgb"}
        return summary

    return rio.run_stage(rio.stage_dir(root, episode_id, "stereo_depth"), stage="stereo_depth",
                         version=STAGE_VERSIONS["stereo_depth"],
                         config={"references": list(references), "chunk": chunk, "pair_sync_ns": pair_sync_ns,
                                 "backend": backend_config(backend)},
                         inputs={"ingest": _record_fp(root, episode_id, "ingest")}, build=build, force=force)


def grid_frames(root, episode_id, meta: EpisodeMeta, g, indices, eye: str = "left") -> np.ndarray:
    """Frames on the grid every later stage uses: rectified images when the depth stage rectified this rig. A rig
    that needs rectification but has no rectified images yet is an error (never mix raw tracks with rectified
    depth)."""
    stream = g.left_stream if eye == "left" else g.right_stream
    path = rio.stage_dir(root, episode_id, "stereo_depth") / f"{g.rig_id}.h5"
    if path.is_file():
        with h5py.File(path, "r") as f:
            if eye in f and "rectified_image" in f[eye]:
                return np.stack([f[eye]["rectified_image"][i] for i in indices])
    s = meta.stream(stream)
    if g.status == "metric" and not s.rectified:
        raise RuntimeError(f"{stream}: needs rectification; run stereo_depth first")
    return read_frames(root, episode_id, stream, indices)


def load_depth(root, episode_id, rig, indices, reference="left"):
    with h5py.File(rio.stage_dir(root, episode_id, "stereo_depth") / f"{rig}.h5", "r") as f:
        depth = np.stack([f[reference]["depth"][i] for i in indices])
        K = f.attrs["K_left" if reference == "left" else "K_right"]
        rectified = "rectified_image" in f[reference]
    return depth, np.asarray(K), rectified


def view_frames(root, episode_id, meta: EpisodeMeta, stream_id: str, indices) -> np.ndarray:
    """Frames of a camera view on the grid its depth, tracks and masks use: the stereo grid of ``grid_frames`` for a
    stereo rig's left eye, the recorded (zero-distortion) colour grid of an RGB-D rig."""
    view = next(v for v in meta.views() if v.stream_id == stream_id)
    if view.kind == "stereo":
        return grid_frames(root, episode_id, meta, meta.rig_stereo(view.rig_id), indices)
    return read_frames(root, episode_id, stream_id, indices)


def default_depth_stage(meta: EpisodeMeta, purpose: str) -> str:
    """Depth source per purpose ("calibrate" or "lift"): stereo episodes use ``stereo_depth``; RGB-D episodes calibrate
    on the registered sensor depth and lift with MapAnything depth."""
    if meta.rgbd and not meta.stereo:
        return RGBD_CALIBRATION_DEPTH if purpose == "calibrate" else RGBD_LIFT_DEPTH
    if meta.rgbd:
        raise NotImplementedError("episodes mixing stereo and RGB-D rigs need an explicit depth stage")
    return "stereo_depth"


def view_depth(root, episode_id, meta: EpisodeMeta, stream_id: str, indices, stage: str):
    """(depth (n,H,W), K continuous) of a view from ``stage`` on that view's grid."""
    if stage == "stereo_depth":
        view = next(v for v in meta.views() if v.stream_id == stream_id)
        depth, K, _ = load_depth(root, episode_id, view.rig_id, indices)
        return depth, K
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


# ---------------------------------------------------------------- stage 3: calibration


def calibration_image(root, episode_id, rig, cam, index):
    """The image on the grid calibration works in: the rectified left image when the depth stage rectified."""
    path = rio.stage_dir(root, episode_id, "stereo_depth") / f"{rig}.h5"
    if path.is_file():
        with h5py.File(path, "r") as f:
            if "rectified_image" in f["left"]:
                return f["left"]["rectified_image"][index]
    return read_frames(root, episode_id, cam, [index])[0]


def fk_vs_recorded_ee(meta: EpisodeMeta, geom) -> dict | None:
    """Cross-check URDF FK against the recorded EE pose (only when its frame exists in the URDF). A large mismatch
    means wrong joint order/units, a wrong URDF or a wrong EE-frame declaration; it is reported, not used."""
    tr = meta.trajectory
    if tr.ee_pose_base is None or tr.ee_frame not in geom.model.links:
        return None
    errs_t, errs_r = [], []
    for k in np.linspace(0, len(tr.timestamps_ns) - 1, min(50, len(tr.timestamps_ns))).astype(int):
        T = geom.model.fk(geom.q(tr.joint_positions[k], tr.gripper_state[k]))[tr.ee_frame]
        errs_t.append(float(np.linalg.norm(T[:3, 3] - tr.ee_pose_base[k][:3, 3])))
        cos = (np.trace(T[:3, :3].T @ tr.ee_pose_base[k][:3, :3]) - 1) / 2
        errs_r.append(float(np.degrees(np.arccos(np.clip(cos, -1, 1)))))
    return {"ee_frame": tr.ee_frame, "max_trans_m": max(errs_t), "max_rot_deg": max(errs_r)}


def applied_joint_correction(meta: EpisodeMeta) -> dict | None:
    """The adapter's joint-correction provenance (``provenance.notes["joint_correction"]``) when it was applied, i.e.
    when ``trajectory.joint_positions`` are recorded joints plus offsets that belong to one specific calibration."""
    try:
        notes = json.loads(meta.provenance.notes or "{}")
    except (TypeError, ValueError):
        return None
    jc = notes.get("joint_correction") if isinstance(notes, dict) else None
    return jc if isinstance(jc, dict) and jc.get("applied") else None


def calibrate(root: Path, episode_id: str, landmark_source, *, landmark_links: list[str], every: int = 1,
              cfg: CalibrationConfig | None = None, overlays: int = 3, force: bool = False, runner=None,
              depth_stage: str | None = None) -> dict:
    """landmark_source(camera_id, frame_index, landmark_links) -> (uv (M,2) continuous on the view grid with NaN for
    missing, conf (M,)) - visual evidence (detector, annotation or render-and-match worker). With
    ``cfg.evidence == "rgbd"`` there is no landmark source (``None``): the metric depth of ``depth_stage`` (default:
    the registered sensor depth of RGB-D rigs) is the evidence. The solve runs as the ``calibrate_camera`` kernel
    through ``runner`` (in-process by default)."""
    meta = load_meta(root, episode_id)
    if applied_joint_correction(meta):
        raise ValueError(f"{episode_id}: joints carry corrections tied to an external calibration "
                         f"({applied_joint_correction(meta)['kind']}); robot-grounded calibration needs raw joints - "
                         "re-ingest with a calibration_source other than adapter_provided")
    if landmark_source is None and (cfg is None or cfg.evidence != "rgbd"):
        raise ValueError("a landmark source is required unless cfg.evidence == 'rgbd'")
    depth_stage = depth_stage or default_depth_stage(meta, "calibrate")
    depth_rec = rio.read_record(rio.stage_dir(root, episode_id, depth_stage))
    geom = RobotGeometry(meta)
    runner = runner or LocalRunner(rio.episode_dir(root, episode_id) / "_work" / "calibrate")
    inputs = {"ingest": _record_fp(root, episode_id, "ingest")}
    if depth_rec is not None:
        inputs[depth_stage] = rio.build_id(depth_rec)
    config = {"landmark_links": landmark_links, "every": every, "cfg": asdict(cfg or CalibrationConfig()),
              "landmark_source": getattr(landmark_source, "identity", repr(landmark_source))}
    if depth_stage != "stereo_depth":
        config["depth_stage"] = depth_stage  # (absent for stereo episodes: their fingerprints stay unchanged)
    directory = rio.stage_dir(root, episode_id, "calibrate")
    store = rio.ChunkStore(directory, rio.stage_fingerprint("calibrate", STAGE_VERSIONS["calibrate"], config, inputs))
    if force:
        store.clear()

    def build(tmp: Path) -> dict:
        from PIL import Image

        summary = {"fk_vs_recorded_ee": fk_vs_recorded_ee(meta, geom)}
        for view in meta.views():
            cam = view.stream_id
            stamps = meta.stream_timestamps_ns[cam]
            idx = list(range(0, len(stamps), every))
            q, grip, ok, sync = robot_state_at(meta, stamps[idx])
            intr = meta.stream(cam).intrinsics
            if view.kind == "stereo":
                g = meta.rig_stereo(view.rig_id)
                has_depth = g.status == "metric" and depth_rec is not None and g.rig_id in depth_rec["summary"] and \
                    "skipped" not in depth_rec["summary"][g.rig_id]
            else:
                has_depth = depth_rec is not None and cam in depth_rec["summary"] and \
                    "skipped" not in depth_rec["summary"][cam]
            if has_depth:
                depth, K = view_depth(root, episode_id, meta, cam, idx, depth_stage)
            else:
                depth, K = None, intr.K()
            local_cfg = CalibrationConfig(**{**asdict(cfg or CalibrationConfig()),
                                             "image_hw": (intr.height, intr.width)})
            rows = {k: [] for k in ("t_index", "timestamp_ns", "landmarks_base", "landmarks_uv", "landmarks_conf",
                                    "surface_base", "flange_base", "depth")}
            for k, i in enumerate(idx):
                if not ok[k]:
                    continue
                qd = geom.q(q[k], grip[k])
                if landmark_source is not None:
                    uv, conf = landmark_source(cam, i, landmark_links)
                else:
                    uv, conf = np.full((len(landmark_links), 2), np.nan), np.zeros(len(landmark_links))
                uv = np.asarray(uv, float)
                if not has_depth and intr.distortion_model != "none":
                    # raw (unrectified) grid: map observations to the ideal pinhole image of the same K
                    uv = undistort_pixels(uv, intr.K(), intr.distortion_model, intr.distortion)
                surf, _ = geom.surface.points(qd)
                fk = geom.model.fk(qd)
                rows["t_index"].append(i)
                rows["timestamp_ns"].append(int(stamps[i]))
                rows["landmarks_base"].append(landmarks(geom.model, qd, landmark_links).reshape(-1, 3))
                rows["landmarks_uv"].append(uv.reshape(-1, 2))
                rows["landmarks_conf"].append(np.asarray(conf, float).reshape(-1))
                rows["surface_base"].append(surf)
                rows["flange_base"].append(fk[meta.robot.flange_frame][:3, 3])
                if depth is not None:
                    rows["depth"].append(depth[k])
            arrays = {k: np.asarray(v) for k, v in rows.items() if k != "depth"}
            arrays["t_index"] = arrays["t_index"].astype(np.int64)
            arrays["timestamp_ns"] = arrays["timestamp_ns"].astype(np.int64)
            if depth is not None:
                arrays["depth"] = np.stack(rows["depth"]).astype(np.float32)
            arrays["K"] = K
            n = len(rows["t_index"])
            pick = np.linspace(0, n - 1, min(overlays, n)).astype(int) if n else np.zeros(0, int)
            if len(pick):
                arrays["overlay_frames"] = pick
                arrays["overlay_images"] = np.stack([
                    (calibration_image(root, episode_id, view.rig_id, cam, rows["t_index"][k]) if view.kind == "stereo"
                     else to_model_rgb(read_frames(root, episode_id, cam, [rows["t_index"][k]]), meta.stream(cam))[0])
                    for k in pick])
            got = store.get(cam)
            if got is None:
                out, kmeta, handle = runner.kernel("calibrate_camera", {"camera_id": cam, "cfg": asdict(local_cfg)},
                                                   arrays, chunk_id=f"calibrate/{cam}")
                store.put(cam, out, kmeta)
                runner.ack(handle)
                got = (out, kmeta)
            out, kmeta = got
            doc = kmeta["result"]
            doc["diagnostics"]["robot_sync_max_ns"] = int(sync.max()) if len(sync) else None
            doc["diagnostics"]["frames_dropped_for_sync"] = int((~ok).sum())
            doc["diagnostics"]["depth_available"] = bool(has_depth)
            if depth_stage != "stereo_depth":
                doc["diagnostics"]["depth_stage"] = depth_stage
            rio.atomic_write_json(tmp / f"{rio.safe_name(cam)}.json", doc)
            for img, k in zip(out["overlays"], arrays.get("overlay_frames", [])):
                (tmp / "overlays").mkdir(exist_ok=True)
                Image.fromarray(img).save(tmp / "overlays" / f"{rio.safe_name(cam)}_{rows['t_index'][int(k)]:05d}.png")
            summary[cam] = {"status": doc["status"], "reasons": doc["reasons"], "init": doc["init_method"],
                            "reproj_median_px": doc["diagnostics"].get("reproj_median_px"),
                            "depth_abs_median_m": doc["diagnostics"].get("depth_abs_median_m")}
        if runner.describe().get("kind") != "local":
            summary["runner"] = runner.describe()
        return summary

    rec = rio.run_stage(directory, stage="calibrate", version=STAGE_VERSIONS["calibrate"], config=config,
                        inputs=inputs, build=build, force=force)
    store.cleanup()
    return rec


def calibrate_provided(root: Path, episode_id: str, provided: dict, *, force: bool = False) -> dict:
    """Camera poses supplied by an external, documented calibration (e.g. DROID-3D extrinsics) instead of the
    robot-grounded optimisation. ``provided`` maps camera id -> {"T_cam_base": 4x4, "source": {...}}. The poses are
    stored in the calibrate stage's format (status ``accepted``, ``init_method`` ``provided``) only after a rigidity
    check; cameras without a provided pose get status ``rejected`` (lift3d/export then skip them). If the ingested joints
    carry corrections tied to a calibration (``applied_joint_correction``), every provided pose must come from exactly
    that calibration record; anything else is refused rather than mixing geometries."""
    meta = load_meta(root, episode_id)
    jc = applied_joint_correction(meta)
    if jc:
        need = jc["requires_calibration"]
        for cam, entry in provided.items():
            src = entry.get("source", {}) or {}
            bad = {k: src.get(k) for k, v in need.items() if src.get(k) != v}
            if bad:
                raise ValueError(f"{episode_id} {cam}: provided calibration {bad} is not the record the ingested joint "
                                 f"correction belongs to ({need})")

    def build(tmp: Path) -> dict:
        summary = {}
        for g in meta.stereo:
            cam = g.left_stream
            entry = provided.get(cam)
            doc = {"camera_id": cam, "status": "rejected", "reasons": [], "init_method": "provided",
                   "T_cam_base": None, "T_cam_base_init": None, "diagnostics": {}}
            if entry is None:
                doc["reasons"].append("no provided pose for this camera")
            else:
                T = np.asarray(entry["T_cam_base"], float)
                R = T[:3, :3]
                if T.shape != (4, 4) or not np.allclose(R @ R.T, np.eye(3), atol=1e-6) or \
                        abs(np.linalg.det(R) - 1) > 1e-6 or not np.allclose(T[3], [0, 0, 0, 1]):
                    doc["reasons"].append("provided pose is not a rigid transform")
                else:
                    doc.update(status="accepted", T_cam_base=T.tolist(), T_cam_base_init=T.tolist())
                    doc["diagnostics"] = {"source": entry.get("source", {}),
                                          "camera_position_base_m": (-R.T @ T[:3, 3]).tolist(),
                                          "robot_joint_correction_dq": jc["dq"] if jc else None}
            rio.atomic_write_json(tmp / f"{rio.safe_name(cam)}.json", doc)
            summary[cam] = {"status": doc["status"], "reasons": doc["reasons"], "init": "provided",
                            "source": (entry or {}).get("source", {}).get("kind")}
        return summary

    return rio.run_stage(rio.stage_dir(root, episode_id, "calibrate"), stage="calibrate",
                         version=STAGE_VERSIONS["calibrate"] + "-provided",
                         config={"provided": {k: {"T_cam_base": np.asarray(v["T_cam_base"], float).tolist(),
                                                  "source": v.get("source", {})} for k, v in provided.items()}},
                         inputs={"ingest": _record_fp(root, episode_id, "ingest")}, build=build, force=force)


def accepted_calibration(root, episode_id, camera_id) -> np.ndarray | None:
    path = rio.stage_dir(root, episode_id, "calibrate") / f"{rio.safe_name(camera_id)}.json"
    if not path.is_file():
        return None
    doc = json.loads(path.read_text())
    return np.asarray(doc["T_cam_base"]) if doc["status"] == "accepted" else None


# ---------------------------------------------------------------- stage 4/5: tracking and lifting


def chunk_queries(n: int, window: int, step: int) -> list[tuple[int, list[int]]]:
    return [(s, list(range(s, min(n, s + window)))) for s in range(0, max(1, n - 2), step)]


def track2d(root: Path, episode_id: str, backend: TrackerBackend, *, window: int = 16, step: int = 8,
            grid_stride: int = 8, tracker_hw: tuple | None = None, force: bool = False,
            depth_stage: str | None = None) -> dict:
    """``tracker_hw`` runs the tracker on resized frames (H, W multiples of 8 for AllTracker); the query grid lives on
    the tracker grid and the returned positions are mapped back to the full view grid with the inverse pixel affine,
    so depth lookup and lifting stay on the depth grid. Each query window is one resumable chunk."""
    meta = load_meta(root, episode_id)
    depth_stage = depth_stage or default_depth_stage(meta, "lift")
    depth_rec = rio.read_record(rio.stage_dir(root, episode_id, depth_stage))
    inputs = {"ingest": _record_fp(root, episode_id, "ingest")}
    if depth_rec is not None:
        inputs[depth_stage] = rio.build_id(depth_rec)  # tracks live on the grid the depth stage produced
    config = {"window": window, "step": step, "grid_stride": grid_stride,
              "tracker_hw": list(tracker_hw) if tracker_hw else None, "backend": backend_config(backend)}
    directory = rio.stage_dir(root, episode_id, "track2d")
    store = rio.ChunkStore(directory, rio.stage_fingerprint("track2d", STAGE_VERSIONS["track2d"], config, inputs))
    if force:
        store.clear()

    def build(tmp: Path) -> dict:
        summary = {}
        for view in meta.views():
            cam = view.stream_id
            n = len(meta.stream_timestamps_ns[cam])
            files = []
            for q, idx in chunk_queries(n, window, step):
                name = f"{rio.safe_name(cam)}_q{q:05d}.npz"
                got = store.get(name)
                if got is None:
                    frames = to_model_rgb(view_frames(root, episode_id, meta, cam, idx), meta.stream(cam))
                    hw = frames.shape[1:3]
                    aff = None
                    if tracker_hw is not None and tuple(tracker_hw) != tuple(hw):
                        aff = resize_affine(hw, tuple(tracker_hw))
                        top = np.iinfo(frames.dtype).max
                        frames = np.stack([np.clip(np.rint(warp_affine(x.astype(np.float32), aff, tuple(tracker_hw),
                                                                       order=1)), 0, top)
                                           for x in frames]).astype(frames.dtype)
                    if isinstance(backend, WorkerTrackerBackend):
                        flows, visconf = backend.track(cam, frames, idx, 0, chunk_id=f"track2d/{name}")
                    else:
                        flows, visconf = backend.track(cam, frames, idx, 0)
                    thw = frames.shape[1:3]
                    source = dict(backend.identity)
                    tr = alltracker_to_tracks2d(flows, visconf, query_grid(thw, grid_stride, grid_stride // 2),
                                                camera_id=cam, query_frame=q, frame_indices=idx,
                                                timestamps_ns=meta.stream_timestamps_ns[cam][idx], source=source)
                    if aff is not None:
                        tr.uv = aff.inverse().apply_uv(tr.uv).astype(np.float32)
                        tr.image_hw = tuple(hw)
                        tr.source["tracker_affine_full_to_tracker"] = aff.to_json()
                    arrays = dict(frame_indices=tr.frame_indices, timestamps_ns=tr.timestamps_ns,
                                  point_ids=tr.point_ids, uv=tr.uv, visibility=tr.visibility, confidence=tr.confidence,
                                  query_frame=q, image_hw=np.array(hw), source=json.dumps(tr.source))
                    store.put(name, {k: np.asarray(v) for k, v in arrays.items()}, {"model": backend.identity})
                    if getattr(backend, "runner", None) is not None:
                        backend.runner.ack(backend.last_handle)
                    got = store.get(name)
                arrays, cmeta = got
                rio.atomic_write_npz(tmp / name, **arrays)
                backend.identity = cmeta.get("model", backend.identity)
                files.append(name)
            summary[cam] = {"chunks": files, "model": backend.identity}
        return summary

    rec = rio.run_stage(directory, stage="track2d", version=STAGE_VERSIONS["track2d"], config=config, inputs=inputs,
                        build=build, force=force)
    store.cleanup()
    return rec


def load_tracks2d(path: Path, camera_id: str) -> Tracks2D:
    with np.load(path) as z:
        return Tracks2D(camera_id, int(z["query_frame"]), z["frame_indices"], z["timestamps_ns"], z["point_ids"],
                        z["uv"], z["visibility"], z["confidence"], tuple(z["image_hw"]), json.loads(str(z["source"])))


def lift3d(root: Path, episode_id: str, *, vis_threshold: float = 0.5, conf_threshold: float = 0.5,
           force: bool = False, depth_stage: str | None = None, runner=None) -> dict:
    """Same-frame lifting of every 2D track chunk with the depth of ``depth_stage`` (default: stereo depth for stereo
    episodes, MapAnything depth for RGB-D episodes) and the accepted calibration (``lift_tracks`` kernel)."""
    meta = load_meta(root, episode_id)
    depth_stage = depth_stage or default_depth_stage(meta, "lift")
    t2d = rio.read_record(rio.stage_dir(root, episode_id, "track2d"))
    runner = runner or LocalRunner(rio.episode_dir(root, episode_id) / "_work" / "lift3d")
    config = {"vis_threshold": vis_threshold, "conf_threshold": conf_threshold}
    if depth_stage != "stereo_depth":
        config["depth_stage"] = depth_stage
    inputs = {k: _record_fp(root, episode_id, k) for k in (depth_stage, "calibrate", "track2d")}

    def build(tmp: Path) -> dict:
        summary = {}
        for view in meta.views():
            cam = view.stream_id
            T = accepted_calibration(root, episode_id, cam)
            if T is None:
                summary[cam] = {"skipped": "no accepted calibration for this episode/camera"}
                continue
            if view.kind == "stereo" and meta.rig_stereo(view.rig_id).status != "metric":
                summary[cam] = {"skipped": f"stereo {meta.rig_stereo(view.rig_id).status}: no metric depth"}
                continue
            counts = {}
            for name in t2d["summary"][cam]["chunks"]:
                tr = load_tracks2d(rio.stage_dir(root, episode_id, "track2d") / name, cam)
                depth, K = view_depth(root, episode_id, meta, cam, tr.frame_indices, depth_stage)
                out, kmeta, handle = runner.kernel(
                    "lift_tracks", {"camera_id": cam, "query_frame": tr.query_frame, "image_hw": list(tr.image_hw),
                                    "source": tr.source, "vis_threshold": vis_threshold, "conf_threshold": conf_threshold},
                    {"frame_indices": tr.frame_indices, "timestamps_ns": tr.timestamps_ns, "point_ids": tr.point_ids,
                     "uv": tr.uv, "visibility": tr.visibility, "confidence": tr.confidence, "depth": depth, "K": K,
                     "T_cam_base": T}, chunk_id=f"lift3d/{name}")
                rio.atomic_write_npz(tmp / name, **out, query_frame=tr.query_frame, camera_id=cam,
                                     source=json.dumps(kmeta["source"]))
                runner.ack(handle)
                for k, v in kmeta["status_counts"].items():
                    counts[k] = counts.get(k, 0) + v
            summary[cam] = {"chunks": t2d["summary"][cam]["chunks"], "status_counts": counts,
                            "T_cam_base": T.tolist()}
        return summary

    return rio.run_stage(rio.stage_dir(root, episode_id, "lift3d"), stage="lift3d", version=STAGE_VERSIONS["lift3d"],
                         config=config, inputs=inputs, build=build, force=force)


def camera_poses(root, episode_id, meta) -> dict:
    return {v.stream_id: accepted_calibration(root, episode_id, v.stream_id) for v in meta.views()}

