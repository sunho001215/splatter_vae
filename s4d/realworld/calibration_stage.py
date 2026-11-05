"""The ``calibrate`` stage: per-episode, robot-grounded camera extrinsics (``T_cam_base``) on each RGB-D camera view.

The stage assembles the evidence of every camera - robot state at camera time, the URDF surface at the recorded joints
and gripper state, the metric depth of ``depth_stage`` (default: the registered sensor depth), optional robot masks
(``robot_mask_stage``, e.g. SAM 3) and optional 2D robot landmarks (a landmark source, e.g.
``cli.AnnotationLandmarkSource``) - and solves it with the ``calibrate_camera`` kernel (``s4d.realworld.calibration``)
through a runner. Results are stored per camera (``<camera>.json`` + overlays) in the stage's record;
``pipeline.accepted_calibration`` reads them. ``pipeline.calibrate`` and ``pipeline.fk_vs_recorded_ee`` are the same
functions.
"""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import h5py
import numpy as np

from s4d.realworld import io as rio
from s4d.realworld.calibration import CalibrationConfig
from s4d.realworld.contract import EpisodeMeta
from s4d.realworld.geometry import undistort_pixels
from s4d.realworld.robot import landmarks
from s4d.realworld.runners import LocalRunner


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
              cfg: CalibrationConfig | None = None, overlays: int = 3, force: bool = False, runner=None,
              depth_stage: str | None = None, robot_mask_stage: str | None = None,
              max_frames: int | None = None) -> dict:
    """landmark_source(camera_id, frame_index, landmark_links) -> (uv (M,2) continuous on the view grid with NaN for
    missing, conf (M,)) - visual evidence (detector, annotation or render-and-match worker). With
    ``cfg.evidence == "rgbd"`` there is no landmark source (``None``): the metric depth of ``depth_stage`` (default:
    the registered sensor depth of RGB-D rigs) is the evidence. ``robot_mask_stage`` (e.g. ``sam3_masks``) supplies
    per-frame robot masks (semantic class ``robot``) that gate the depth term and, in RGB-D mode, select the observed
    robot points of the initialisation. The solve runs as the ``calibrate_camera`` kernel through ``runner``
    (in-process by default); ``max_frames`` caps the frames per camera (evenly spaced after ``every``) so one request
    stays bounded."""
    from s4d.realworld.pipeline import (
        STAGE_VERSIONS,
        RobotGeometry,
        _record_fp,
        camera_views,
        default_depth_stage,
        load_meta,
        read_frames,
        robot_state_at,
        to_model_rgb,
        view_depth,
    )

    meta = load_meta(root, episode_id)
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
    config["depth_stage"] = depth_stage
    if robot_mask_stage:
        config["robot_mask_stage"] = robot_mask_stage
        inputs[robot_mask_stage] = _record_fp(root, episode_id, robot_mask_stage)
    if max_frames is not None:
        config["max_frames"] = int(max_frames)
    directory = rio.stage_dir(root, episode_id, "calibrate")
    store = rio.ChunkStore(directory, rio.stage_fingerprint("calibrate", STAGE_VERSIONS["calibrate"], config, inputs))
    if force:
        store.clear()

    def build(tmp: Path) -> dict:
        import time

        from PIL import Image

        summary = {"fk_vs_recorded_ee": fk_vs_recorded_ee(meta, geom)}
        for view in camera_views(meta):
            cam = view.stream_id
            clock = {"t": time.perf_counter()}
            timing = {}

            def lap(name, timing=timing, clock=clock):
                now = time.perf_counter()
                timing[name] = round(timing.get(name, 0.0) + now - clock["t"], 3)
                clock["t"] = now

            stamps = meta.stream_timestamps_ns[cam]
            idx = list(range(0, len(stamps), every))
            if max_frames is not None and len(idx) > max_frames:
                idx = [idx[k] for k in np.linspace(0, len(idx) - 1, int(max_frames)).round().astype(int)]
            q, grip, ok, sync = robot_state_at(meta, stamps[idx])
            lap("robot_state")
            intr = meta.stream(cam).intrinsics
            has_depth = depth_rec is not None and cam in depth_rec["summary"] and \
                "skipped" not in depth_rec["summary"][cam]
            if has_depth:
                depth, K = view_depth(root, episode_id, meta, cam, idx, depth_stage)
            else:
                depth, K = None, intr.K()
            lap("depth_load")
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
                if not has_depth and intr.distortion_model != "none":  # landmarks on a distorted recorded grid
                    uv = undistort_pixels(uv, intr.K(), intr.distortion_model, intr.distortion)
                surf, _ = geom.surface.points(qd)
                fk = geom.model.fk(qd)
                rows["t_index"].append(i)
                rows["timestamp_ns"].append(int(stamps[i]))
                rows["landmarks_base"].append(landmarks(geom.model, qd, landmark_links).reshape(-1, 3)
                                              if landmark_links else np.zeros((0, 3)))
                rows["landmarks_uv"].append(uv.reshape(-1, 2))
                rows["landmarks_conf"].append(np.asarray(conf, float).reshape(-1))
                rows["surface_base"].append(surf)
                rows["flange_base"].append(fk[meta.robot.flange_frame][:3, 3])
                if depth is not None:
                    rows["depth"].append(depth[k])
            lap("fk_and_surface_sampling")
            arrays = {k: np.asarray(v) for k, v in rows.items() if k != "depth"}
            arrays["t_index"] = arrays["t_index"].astype(np.int64)
            arrays["timestamp_ns"] = arrays["timestamp_ns"].astype(np.int64)
            if depth is not None:
                arrays["depth"] = np.stack(rows["depth"]).astype(np.float32)
            arrays["K"] = K
            if robot_mask_stage:
                with h5py.File(rio.stage_dir(root, episode_id, robot_mask_stage) / f"{rio.safe_name(cam)}.h5", "r") as f:
                    robot_class = json.loads(f.attrs.get("class_names", '["background", "robot", "object", "fixture"]'))
                    arrays["robot_mask"] = np.stack([f["semantic"][i] == robot_class.index("robot")
                                                     for i in rows["t_index"]])
            lap("mask_load")
            n = len(rows["t_index"])
            pick = np.linspace(0, n - 1, min(overlays, n)).astype(int) if n else np.zeros(0, int)
            if len(pick):
                arrays["overlay_frames"] = pick
                arrays["overlay_images"] = np.stack([
                    to_model_rgb(read_frames(root, episode_id, cam, [rows["t_index"][k]]), meta.stream(cam))[0]
                    for k in pick])
            lap("overlay_images")
            got = store.get(cam)
            cached = got is not None
            if got is None:
                out, kmeta, handle = runner.kernel("calibrate_camera", {"camera_id": cam, "cfg": asdict(local_cfg)},
                                                   arrays, chunk_id=f"calibrate/{cam}")
                lap("kernel_call_transfer_and_compute")
                store.put(cam, out, kmeta)
                runner.ack(handle)
                lap("durable_save")
                got = (out, kmeta)
            out, kmeta = got
            doc = kmeta["result"]
            doc["diagnostics"]["stage_timing_s"] = {**timing, "kernel_result_from_chunk_store": cached,
                                                    "frames": len(rows["t_index"]),
                                                    "input_bytes": int(sum(np.asarray(v).nbytes for v in arrays.values()))}
            doc["diagnostics"]["robot_sync_max_ns"] = int(sync.max()) if len(sync) else None
            doc["diagnostics"]["frames_dropped_for_sync"] = int((~ok).sum())
            doc["diagnostics"]["depth_available"] = bool(has_depth)
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
