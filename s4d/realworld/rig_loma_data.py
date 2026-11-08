"""SAM-free robot frame selection and coarse initialisation for the render-to-image rig calibration (``rig_loma``).

Frames (``extract_episode``): per camera, every ``every``-th frame with the robot state synchronised to the camera
timestamp (``pipeline.robot_state_at``, <= 20 ms gap) and flange speed <= ``max_speed_m_s`` (FK central difference
+-10 ms); ``frames_per_camera`` of them evenly spaced in time. No mask, image or depth is read: whether the robot is
visible is decided later by rendering it at the current pose estimate (``workers/loma_render_match_worker.py``).
Stored per frame: t_index, timestamp, joints, gripper state, flange speed and the FK link poses (base frame) of every
link with geometry at t and t +- ``velocity_dt_ns`` (surface velocities / time offsets). The frames of the RoMa v2
static-match time points (``matches-v1`` meta) are stored as a second group ``romat`` (no speed gate; cross-checks).

Coarse initialisation (``mask_free_init``): the existing per-episode RGB-D robot registration
(``calibration.calibrate_camera``, evidence ``rgbd``) WITHOUT robot masks: its initialisation then uses moving-depth
evidence (depth pixels that change against the temporal median) instead of SAM masks. Only a starting point for the
render-and-match iterations; never a constraint."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np


@dataclass
class FrameConfig:
    every: int = 2
    frames_per_camera: int = 16
    max_speed_m_s: float = 0.10
    max_sync_ns: int = 20_000_000
    velocity_dt_ns: int = 10_000_000


def _fk_links(geom, q, g, links):
    T = geom.model.fk(geom.q(q, g))
    return np.stack([T[l] for l in links])


def extract_episode(root: Path, episode_id: str, out_path: Path, cfg: FrameConfig | None = None,
                    romat_frames: dict | None = None) -> dict:
    from s4d.realworld import io as rio
    from s4d.realworld.pipeline import RobotGeometry, camera_views, load_meta, robot_state_at
    from s4d.realworld.rig_dw_data import flange_speed

    cfg = cfg or FrameConfig()
    meta = load_meta(root, episode_id)
    geom = RobotGeometry(meta, points_per_link=1)
    links = sorted(k for k, v in geom.model.links.items() if v)
    flange = meta.robot.flange_frame
    arrays, doc = {}, {"episode_id": episode_id, "config": asdict(cfg), "links": links, "cameras": {},
                       "velocity_dt_ns": cfg.velocity_dt_ns}

    def store(key, sel, stamps, speed):
        fk = np.zeros((3, len(sel), len(links), 4, 4))
        qs, gs = None, None
        for j, off in enumerate((-cfg.velocity_dt_ns, 0, cfg.velocity_dt_ns)):
            qj, gj, okj, _ = robot_state_at(meta, stamps[sel] + off, cfg.max_sync_ns)
            if j == 1:
                qs, gs = qj, gj
            for k in range(len(sel)):
                fk[j, k] = _fk_links(geom, qj[k], gj[k], links)
        arrays[f"{key}/fk"] = fk[1]
        arrays[f"{key}/fk_minus"] = fk[0]
        arrays[f"{key}/fk_plus"] = fk[2]
        arrays[f"{key}/t_index"] = np.asarray(sel, np.int64)
        arrays[f"{key}/timestamp_ns"] = stamps[sel].astype(np.int64)
        arrays[f"{key}/q"] = qs
        arrays[f"{key}/grip"] = gs
        arrays[f"{key}/speed"] = np.asarray(speed, float)

    for view in camera_views(meta):
        cam = view.stream_id
        key = rio.safe_name(cam)
        stamps = meta.stream_timestamps_ns[cam]
        idx = np.arange(0, len(stamps), cfg.every)
        _, _, ok, _ = robot_state_at(meta, stamps[idx], cfg.max_sync_ns)
        cand = np.flatnonzero(ok)
        speed = np.full(len(idx), np.inf)
        speed[cand] = flange_speed(meta, geom, stamps[idx[cand]], flange)
        cand = cand[speed[cand] <= cfg.max_speed_m_s]
        if len(cand) > cfg.frames_per_camera:
            cand = cand[np.linspace(0, len(cand) - 1, cfg.frames_per_camera).round().astype(int)]
        sel = idx[cand]
        store(key, sel, stamps, speed[cand])
        intr = meta.stream(cam).intrinsics
        doc["cameras"][cam] = {"key": key, "K": intr.K().tolist(), "hw": [int(intr.height), int(intr.width)],
                               "frames": int(len(sel)), "candidates": int(len(idx))}
        if romat_frames:
            rs = np.array(sorted({int(v[cam]) for v in romat_frames.values() if cam in v}), np.int64)
            rs = rs[rs < len(stamps)]
            _, _, okr, _ = robot_state_at(meta, stamps[rs], cfg.max_sync_ns)
            rs = rs[okr]
            sp = flange_speed(meta, geom, stamps[rs], flange) if len(rs) else np.zeros(0)
            store(f"romat/{key}", rs, stamps, sp)
            doc["cameras"][cam]["romat_frames"] = int(len(rs))
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


def mask_free_init(root: Path, episode_id: str, depth_stage: str = "foundation_stereo_depth",
                   max_frames: int = 60, every: int = 2) -> dict:
    """Per camera T_cam_base of one episode from ``calibrate_camera`` (evidence rgbd) WITHOUT robot masks."""
    from s4d.realworld import io as rio
    from s4d.realworld.calibration import CalibrationConfig
    from s4d.realworld.calibration_stage import camera_arrays
    from s4d.realworld.kernels import calibrate_camera_kernel
    from s4d.realworld.pipeline import RobotGeometry, camera_views, load_meta

    meta = load_meta(root, episode_id)
    geom = RobotGeometry(meta)
    depth_rec = rio.read_record(rio.stage_dir(root, episode_id, depth_stage))
    out = {}
    for view in camera_views(meta):
        cam = view.stream_id
        stamps = meta.stream_timestamps_ns[cam]
        idx = list(range(0, len(stamps), every))
        if len(idx) > max_frames:
            idx = [idx[k] for k in np.linspace(0, len(idx) - 1, max_frames).round().astype(int)]
        arrays, _, _, _, has_depth, local_cfg = camera_arrays(
            root, episode_id, meta, geom, cam, idx, depth_rec=depth_rec, depth_stage=depth_stage,
            robot_mask_stage=None, cfg=CalibrationConfig(evidence="rgbd"))
        if not has_depth:
            out[cam] = {"T": None, "status": "no_depth"}
            continue
        _, kmeta = calibrate_camera_kernel({"camera_id": cam, "cfg": asdict(local_cfg)}, arrays)
        res = kmeta["result"]
        out[cam] = {"T": res.get("T_cam_base"), "status": res.get("status"), "init": res.get("init_method"),
                    "diag": {k: res.get("diagnostics", {}).get(k) for k in
                             ("reproj_median_px", "depth_abs_median_m", "explained_fraction", "init")}}
    return out


def main(argv=None):
    import argparse
    import time

    ap = argparse.ArgumentParser(description="SAM-free robot frames (+ optional mask-free init) for rig_loma")
    ap.add_argument("--task-root", required=True, help="processed/<release>/<task>")
    ap.add_argument("--out", required=True)
    ap.add_argument("--episodes", nargs="*", default=None)
    ap.add_argument("--matches", default=None, help="matches-v1/<task> dir (RoMa time points -> romat frames)")
    ap.add_argument("--frames-per-camera", type=int, default=16)
    ap.add_argument("--init", action="store_true", help="mask-free per-episode init instead of frames")
    ap.add_argument("--depth-stage", default="foundation_stereo_depth")
    a = ap.parse_args(argv)
    task_root = Path(a.task_root)
    eps = a.episodes or sorted(p.name for p in task_root.iterdir() if p.is_dir())
    for ep in eps:
        t0 = time.time()
        if a.init:
            out = Path(a.out) / f"{ep}.json"
            if out.exists():
                continue
            res = mask_free_init(task_root / ep, ep, a.depth_stage)
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text(json.dumps(res, indent=1, default=lambda x: x.tolist() if hasattr(x, "tolist") else str(x)))
            print(json.dumps({"episode": ep, "seconds": round(time.time() - t0, 1),
                              "status": {k: v["status"] for k, v in res.items()}}), flush=True)
            continue
        out = Path(a.out) / f"{ep}.npz"
        if out.exists():
            continue
        romat = None
        if a.matches and (Path(a.matches) / f"{ep}.npz").exists():
            with np.load(Path(a.matches) / f"{ep}.npz") as z:
                romat = json.loads(str(z["meta"]))["frames"]
        doc = extract_episode(task_root / ep, ep, out, FrameConfig(frames_per_camera=a.frames_per_camera), romat)
        print(json.dumps({"episode": ep, "seconds": round(time.time() - t0, 1),
                          "frames": {k: v["frames"] for k, v in doc["cameras"].items()}}), flush=True)


if __name__ == "__main__":
    main()
