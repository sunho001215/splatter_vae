"""Evidence extraction for the board-free per-task rig calibration (``rig_dw``).

Per episode and camera, a bounded set of robot frames is selected (robot state at the camera timestamp, robot visible
in the SAM 3 mask, flange speed below ``max_speed_m_s`` so that camera/robot clock offsets and inter-camera timestamp
offsets move the arm by at most ~2 mm) and stored compactly:

* robot state: joint vector, gripper state, FK link poses of every link with geometry (base frame), flange speed;
* image evidence on a crop around the robot (robot mask dilated by ``crop_margin_px``): SAM 3 semantic classes, FS
  depth and sensor depth (uint16, 0.1 mm units, 0 = invalid).

Everything is read from the processed release (read-only); outputs go to the study directory. One ``.npz`` per
episode: ``<out>/<episode>.npz`` (+ ``meta`` JSON entry)."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np


@dataclass
class EvidenceConfig:
    every: int = 2
    frames_per_camera: int = 12
    min_robot_pixels: int = 3000
    max_speed_m_s: float = 0.10
    crop_margin_px: int = 40
    max_sync_ns: int = 20_000_000
    velocity_dt_ns: int = 10_000_000
    depth_stages: tuple = ("foundation_stereo_depth", "sensor_depth")
    robot_mask_stage: str = "sam3_masks"


def flange_speed(meta, geom, stamps_ns: np.ndarray, flange: str) -> np.ndarray:
    """Flange speed (m/s) at ``stamps_ns`` from the recorded joint trajectory (FK, central difference +-10 ms)."""
    from s4d.realworld.pipeline import robot_state_at

    out = []
    for t in stamps_ns:
        q, g, ok, _ = robot_state_at(meta, np.array([t - 10_000_000, t + 10_000_000]))
        p = [geom.model.fk(geom.q(q[k], g[k]))[flange][:3, 3] for k in range(2)]
        out.append(float(np.linalg.norm(p[1] - p[0]) / 0.02) if ok.all() else np.inf)
    return np.asarray(out)


def _to_u16(depth: np.ndarray) -> np.ndarray:
    return np.clip(np.round(np.where(depth > 0, depth, 0) * 1e4), 0, 65535).astype(np.uint16)


def extract_episode(root: Path, episode_id: str, out_path: Path, cfg: EvidenceConfig | None = None) -> dict:
    import h5py

    from s4d.realworld import io as rio
    from s4d.realworld.pipeline import RobotGeometry, camera_views, load_meta, robot_state_at
    from s4d.realworld.stages.depth import load_view_depth

    cfg = cfg or EvidenceConfig()
    meta = load_meta(root, episode_id)
    geom = RobotGeometry(meta, points_per_link=1)
    links = sorted(k for k, v in geom.model.links.items() if v)
    flange = meta.robot.flange_frame
    arrays, doc = {}, {"episode_id": episode_id, "config": asdict(cfg), "links": links, "cameras": {}}
    for view in camera_views(meta):
        cam = view.stream_id
        key = rio.safe_name(cam)
        stamps = meta.stream_timestamps_ns[cam]
        idx = np.arange(0, len(stamps), cfg.every)
        q, grip, ok, sync = robot_state_at(meta, stamps[idx], cfg.max_sync_ns)
        with h5py.File(rio.stage_dir(root, episode_id, cfg.robot_mask_stage) / f"{key}.h5", "r") as f:
            names = json.loads(f.attrs.get("class_names", '["background", "robot", "object", "fixture"]'))
            sem_all = f["semantic"]
            robot_px = np.array([(sem_all[int(i)] == names.index("robot")).sum() for i in idx])
        cand = np.flatnonzero(ok & (robot_px >= cfg.min_robot_pixels))
        speed = np.full(len(idx), np.inf)
        speed[cand] = flange_speed(meta, geom, stamps[idx[cand]], flange)
        cand = cand[speed[cand] <= cfg.max_speed_m_s]
        if len(cand) > cfg.frames_per_camera:
            cand = cand[np.linspace(0, len(cand) - 1, cfg.frames_per_camera).round().astype(int)]
        sel = idx[cand]
        with h5py.File(rio.stage_dir(root, episode_id, cfg.robot_mask_stage) / f"{key}.h5", "r") as f:
            sem = np.stack([f["semantic"][int(i)] for i in sel]) if len(sel) else np.zeros((0, 1, 1), np.uint8)
        depth = {}
        K = None
        for stage in cfg.depth_stages:
            if len(sel):
                d, v, K_s = load_view_depth(root, episode_id, cam, sel, stage)
                depth[stage] = np.where(v, d, 0.0)
                K = K_s if K is None else K
        H, W = sem.shape[1:] if len(sel) else (0, 0)
        crops = []
        for k in range(len(sel)):
            m = sem[k] == names.index("robot")
            ys, xs = np.nonzero(m)
            y0, y1 = max(0, ys.min() - cfg.crop_margin_px), min(H, ys.max() + 1 + cfg.crop_margin_px)
            x0, x1 = max(0, xs.min() - cfg.crop_margin_px), min(W, xs.max() + 1 + cfg.crop_margin_px)
            crops.append((y0, y1, x0, x1))
            arrays[f"{key}/f{k}/sem"] = sem[k, y0:y1, x0:x1]
            for stage in cfg.depth_stages:
                arrays[f"{key}/f{k}/{stage}"] = _to_u16(depth[stage][k, y0:y1, x0:x1])
        fk = np.zeros((3, len(sel), len(links), 4, 4))  # at t - dt, t, t + dt (surface velocity, timing offsets)
        for j, off in enumerate((-cfg.velocity_dt_ns, 0, cfg.velocity_dt_ns)):
            qj, gj, _, _ = robot_state_at(meta, stamps[sel] + off, cfg.max_sync_ns)
            for k in range(len(sel)):
                T = geom.model.fk(geom.q(qj[k], gj[k]))
                fk[j, k] = np.stack([T[l] for l in links])
        arrays[f"{key}/fk"] = fk[1]
        arrays[f"{key}/fk_minus"] = fk[0]
        arrays[f"{key}/fk_plus"] = fk[2]
        arrays[f"{key}/t_index"] = sel.astype(np.int64)
        arrays[f"{key}/timestamp_ns"] = stamps[sel].astype(np.int64)
        arrays[f"{key}/q"] = q[cand]
        arrays[f"{key}/grip"] = grip[cand]
        arrays[f"{key}/speed"] = speed[cand]
        arrays[f"{key}/crop"] = np.asarray(crops, np.int32).reshape(-1, 4)
        doc["velocity_dt_ns"] = cfg.velocity_dt_ns
        doc["cameras"][cam] = {"key": key, "K": None if K is None else np.asarray(K).tolist(), "hw": [int(H), int(W)],
                               "class_names": names, "frames": int(len(sel)),
                               "candidates": int(len(idx)), "robot_visible": int((robot_px >= cfg.min_robot_pixels).sum())}
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_path.with_suffix(".tmp.npz")
    np.savez_compressed(tmp, meta=json.dumps(doc), **arrays)
    tmp.replace(out_path)
    return doc


def load_episode(path: Path) -> tuple[dict, dict]:
    with np.load(path) as z:
        doc = json.loads(str(z["meta"]))
        arrays = {k: z[k] for k in z.files if k != "meta"}
    return doc, arrays


def main(argv=None):
    import argparse
    import time

    ap = argparse.ArgumentParser(description="extract robot evidence for rig_dw (one or more episodes)")
    ap.add_argument("--task-root", required=True, help="processed/<release>/<task>")
    ap.add_argument("--out", required=True)
    ap.add_argument("--episodes", nargs="*", default=None)
    a = ap.parse_args(argv)
    task_root = Path(a.task_root)
    eps = a.episodes or sorted(p.name for p in task_root.iterdir() if p.is_dir())
    for ep in eps:
        out = Path(a.out) / f"{ep}.npz"
        if out.exists():
            continue
        t0 = time.time()
        doc = extract_episode(task_root / ep, ep, out)
        print(json.dumps({"episode": ep, "seconds": round(time.time() - t0, 1),
                          "frames": {k: v["frames"] for k, v in doc["cameras"].items()}}), flush=True)


if __name__ == "__main__":
    main()
