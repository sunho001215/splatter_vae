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
from s4d.realworld.calibration import CalibrationConfig, CalibrationFrame, calibrate_camera, overlay_image
from s4d.realworld.contract import EpisodeMeta, meta_from_json, meta_to_json, validate_episode
from s4d.realworld.geometry import rectify_map, remap, stereo_rectify, undistort_pixels
from s4d.realworld.robot import RobotSurface, URDFModel, gripper_joint_value, joint_dict, landmarks
from s4d.realworld.stereo import depth_from_pair
from s4d.realworld.tracking import Tracks2D, alltracker_to_tracks2d, lift_tracks, query_grid, status_counts

STAGE_VERSIONS = {"ingest": "1", "stereo_depth": "1", "calibrate": "1", "track2d": "1", "lift3d": "1", "export": "1"}


# ---------------------------------------------------------------- backends


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
    def __init__(self, spec, workdir: Path, params: dict | None = None, allow_mock: bool = False):
        self.spec, self.workdir, self.params, self.allow_mock = spec, Path(workdir), dict(params or {}), allow_mock
        self.identity = {"worker": spec.name}

    def track(self, camera_id, frames, frame_indices, query):
        from s4d.realworld.workers import call_worker

        resp, out = call_worker(self.spec, "track", {**self.params, "query_frame": int(query)},
                                {"rgbs": _model_input_rgb(frames)}, self.workdir, allow_mock=self.allow_mock)
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
    return rec["fingerprint"]


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
                    left = read_frames(root, episode_id, g.left_stream, idx)
                    right = read_frames(root, episode_id, g.right_stream, idx)
                    if maps is not None:
                        left = np.stack([remap(x, maps[0], order=1) for x in left])
                        right = np.stack([remap(x, maps[1], order=1) for x in right])
                    batch_left = backend.disparity(left, right)
                    for k, i in enumerate(idx):
                        for ref in references:
                            if ref == "left":
                                matcher = (lambda _a, _b, d=batch_left[k]: d)
                            else:
                                matcher = (lambda a, b: backend.disparity(a[None], b[None])[0])
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
                                 "backend": type(backend).__name__},
                         inputs={"ingest": _record_fp(root, episode_id, "ingest")}, build=build, force=force)


def load_depth(root, episode_id, rig, indices, reference="left"):
    with h5py.File(rio.stage_dir(root, episode_id, "stereo_depth") / f"{rig}.h5", "r") as f:
        depth = np.stack([f[reference]["depth"][i] for i in indices])
        K = f.attrs["K_left" if reference == "left" else "K_right"]
        rectified = "rectified_image" in f[reference]
    return depth, np.asarray(K), rectified


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

    def q(self, q7, grip) -> dict:
        value = gripper_joint_value(grip, self.desc.gripper_state_semantics, self.desc.gripper_model,
                                    closed_value=self.closed_value)
        return joint_dict(self.desc.joint_names, q7, self.desc.gripper_joint, value)

    def gripper_links(self) -> list[str]:
        arm = {f"panda_link{i}" for i in range(9)}
        return [n for n in sorted(self.surface.link_points) if n not in arm]


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


def calibrate(root: Path, episode_id: str, landmark_source, *, landmark_links: list[str], every: int = 1,
              cfg: CalibrationConfig | None = None, overlays: int = 3, force: bool = False) -> dict:
    """landmark_source(camera_id, frame_index, landmark_links) -> (uv (M,2) continuous on the rectified left grid
    with NaN for missing, conf (M,)) - visual evidence (detector, annotation or render-and-match worker)."""
    meta = load_meta(root, episode_id)
    depth_rec = rio.read_record(rio.stage_dir(root, episode_id, "stereo_depth"))
    geom = RobotGeometry(meta)

    def build(tmp: Path) -> dict:
        from PIL import Image

        summary = {"fk_vs_recorded_ee": fk_vs_recorded_ee(meta, geom)}
        for g in meta.stereo:
            cam = g.left_stream
            stamps = meta.stream_timestamps_ns[cam]
            idx = list(range(0, len(stamps), every))
            q, grip, ok, sync = robot_state_at(meta, stamps[idx])
            has_depth = g.status == "metric" and depth_rec is not None and g.rig_id in depth_rec["summary"] and \
                "skipped" not in depth_rec["summary"][g.rig_id]
            if has_depth:
                depth, K, _ = load_depth(root, episode_id, g.rig_id, idx)
            else:
                depth, K = None, meta.stream(cam).intrinsics.K()
            local_cfg = CalibrationConfig(**{**asdict(cfg or CalibrationConfig()),
                                             "image_hw": (meta.stream(cam).intrinsics.height,
                                                          meta.stream(cam).intrinsics.width)})
            frames = []
            for k, i in enumerate(idx):
                if not ok[k]:
                    continue
                qd = geom.q(q[k], grip[k])
                uv, conf = landmark_source(cam, i, landmark_links)
                uv = np.asarray(uv, float)
                intr = meta.stream(cam).intrinsics
                if not has_depth and intr.distortion_model != "none":
                    # raw (unrectified) grid: map observations to the ideal pinhole image of the same K
                    uv = undistort_pixels(uv, intr.K(), intr.distortion_model, intr.distortion)
                surf, _ = geom.surface.points(qd)
                fk = geom.model.fk(qd)
                frames.append(CalibrationFrame(i, int(stamps[i]), landmarks(geom.model, qd, landmark_links),
                                               np.asarray(uv, float), np.asarray(conf, float), surf,
                                               fk[meta.robot.flange_frame][:3, 3],
                                               depth[k] if depth is not None else None))
            res = calibrate_camera(cam, frames, K, local_cfg)
            res.diagnostics["robot_sync_max_ns"] = int(sync.max()) if len(sync) else None
            res.diagnostics["frames_dropped_for_sync"] = int((~ok).sum())
            res.diagnostics["depth_available"] = bool(has_depth)
            rio.atomic_write_json(tmp / f"{rio.safe_name(cam)}.json", res.to_json())
            if res.T_cam_base is not None:
                for k in np.linspace(0, len(frames) - 1, min(overlays, len(frames))).astype(int):
                    image = calibration_image(root, episode_id, g.rig_id, cam, frames[k].t_index)
                    ov = overlay_image(image, res.T_cam_base, frames[k], K, local_cfg.image_hw)
                    (tmp / "overlays").mkdir(exist_ok=True)
                    Image.fromarray(ov).save(tmp / "overlays" / f"{rio.safe_name(cam)}_{frames[k].t_index:05d}.png")
            summary[cam] = {"status": res.status, "reasons": res.reasons, "init": res.init_method,
                            "reproj_median_px": res.diagnostics.get("reproj_median_px"),
                            "depth_abs_median_m": res.diagnostics.get("depth_abs_median_m")}
        return summary

    inputs = {"ingest": _record_fp(root, episode_id, "ingest")}
    if depth_rec is not None:
        inputs["stereo_depth"] = depth_rec["fingerprint"]
    return rio.run_stage(rio.stage_dir(root, episode_id, "calibrate"), stage="calibrate",
                         version=STAGE_VERSIONS["calibrate"],
                         config={"landmark_links": landmark_links, "every": every, "cfg": asdict(cfg or CalibrationConfig()),
                                 "landmark_source": getattr(landmark_source, "identity", repr(landmark_source))},
                         inputs=inputs, build=build, force=force)


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
            grid_stride: int = 8, force: bool = False) -> dict:
    meta = load_meta(root, episode_id)

    def build(tmp: Path) -> dict:
        summary = {}
        for g in meta.stereo:
            cam = g.left_stream
            n = len(meta.stream_timestamps_ns[cam])
            depth_file = rio.stage_dir(root, episode_id, "stereo_depth") / f"{g.rig_id}.h5"
            files = []
            for q, idx in chunk_queries(n, window, step):
                frames = None
                if depth_file.is_file():
                    with h5py.File(depth_file, "r") as f:
                        if "rectified_image" in f["left"]:
                            frames = np.stack([f["left"]["rectified_image"][i] for i in idx])
                if frames is None:
                    frames = read_frames(root, episode_id, cam, idx)
                flows, visconf = backend.track(cam, frames, idx, 0)
                hw = frames.shape[1:3]
                tr = alltracker_to_tracks2d(flows, visconf, query_grid(hw, grid_stride, grid_stride // 2),
                                            camera_id=cam, query_frame=q, frame_indices=idx,
                                            timestamps_ns=meta.stream_timestamps_ns[cam][idx], source=backend.identity)
                name = f"{rio.safe_name(cam)}_q{q:05d}.npz"
                rio.atomic_write_npz(tmp / name, frame_indices=tr.frame_indices, timestamps_ns=tr.timestamps_ns,
                                     point_ids=tr.point_ids, uv=tr.uv, visibility=tr.visibility,
                                     confidence=tr.confidence, query_frame=q, image_hw=np.array(hw),
                                     source=json.dumps(tr.source))
                files.append(name)
            summary[cam] = {"chunks": files, "model": backend.identity}
        return summary

    return rio.run_stage(rio.stage_dir(root, episode_id, "track2d"), stage="track2d", version=STAGE_VERSIONS["track2d"],
                         config={"window": window, "step": step, "grid_stride": grid_stride,
                                 "backend": type(backend).__name__},
                         inputs={"ingest": _record_fp(root, episode_id, "ingest")}, build=build, force=force)


def load_tracks2d(path: Path, camera_id: str) -> Tracks2D:
    with np.load(path) as z:
        return Tracks2D(camera_id, int(z["query_frame"]), z["frame_indices"], z["timestamps_ns"], z["point_ids"],
                        z["uv"], z["visibility"], z["confidence"], tuple(z["image_hw"]), json.loads(str(z["source"])))


def lift3d(root: Path, episode_id: str, *, vis_threshold: float = 0.5, conf_threshold: float = 0.5,
           force: bool = False) -> dict:
    meta = load_meta(root, episode_id)
    t2d = rio.read_record(rio.stage_dir(root, episode_id, "track2d"))

    def build(tmp: Path) -> dict:
        summary = {}
        for g in meta.stereo:
            cam = g.left_stream
            T = accepted_calibration(root, episode_id, cam)
            if T is None:
                summary[cam] = {"skipped": "no accepted calibration for this episode/camera"}
                continue
            if g.status != "metric":
                summary[cam] = {"skipped": f"stereo {g.status}: no metric depth"}
                continue
            counts = {}
            for name in t2d["summary"][cam]["chunks"]:
                tr = load_tracks2d(rio.stage_dir(root, episode_id, "track2d") / name, cam)
                depth, K, _ = load_depth(root, episode_id, g.rig_id, tr.frame_indices)
                t3 = lift_tracks(tr, depth, K, T, vis_threshold=vis_threshold, conf_threshold=conf_threshold)
                rio.atomic_write_npz(tmp / name, frame_indices=t3.frame_indices, timestamps_ns=t3.timestamps_ns,
                                     point_ids=t3.point_ids, xyz_base=t3.xyz_base, xyz_cam=t3.xyz_cam,
                                     status=t3.status, visibility=t3.visibility, confidence=t3.confidence,
                                     query_frame=t3.query_frame, camera_id=cam, source=json.dumps(t3.source))
                for k, v in status_counts(t3).items():
                    counts[k] = counts.get(k, 0) + v
            summary[cam] = {"chunks": t2d["summary"][cam]["chunks"], "status_counts": counts,
                            "T_cam_base": T.tolist()}
        return summary

    return rio.run_stage(rio.stage_dir(root, episode_id, "lift3d"), stage="lift3d", version=STAGE_VERSIONS["lift3d"],
                         config={"vis_threshold": vis_threshold, "conf_threshold": conf_threshold},
                         inputs={k: _record_fp(root, episode_id, k) for k in ("stereo_depth", "calibrate", "track2d")},
                         build=build, force=force)


def camera_poses(root, episode_id, meta) -> dict:
    return {g.left_stream: accepted_calibration(root, episode_id, g.left_stream) for g in meta.stereo}

