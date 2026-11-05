"""Dynamic-track filter (stage ``track_motion``): which lifted tracks show reliable motion.

Before this stage nothing filtered tracks by motion: ``track2d`` queries a regular grid and ``lift3d`` only marks
per-sample validity (tracker visibility/confidence, image bounds, depth present, depth edge), so every valid grid
track - mostly static background - reached the export. This stage classifies each lifted track (one camera, one
query window; ids are never matched across cameras or windows) and the export can keep only the moving ones. The
dense AllTracker inference is unchanged: filtering stored outputs does not reduce model inference cost.

Per track, in this order:

1. **Segments.** The valid 3D samples are cut into segments that are never connected: a new segment starts after a
   sample whose 2D identity is not observed (tracker ``occluded``, ``low_confidence``, ``out_of_image``,
   ``not_tracked``), after more than ``max_depth_hole_frames`` consecutive samples with the 2D point still tracked but
   no usable depth (``no_depth``/``depth_edge``), at a recording pause (consecutive frames further apart than
   ``max_gap_factor`` x the camera's median frame period) and at a physically implausible 3D jump (speed above
   ``max_speed_m_s``: an identity switch or a depth flip, not motion).
2. **Statistics** per segment, on samples smoothed by a median of 3 consecutive valid samples (one noisy jump cannot
   move it; positions in the outputs are never smoothed): the tracker's image-plane extent ``extent_px`` (largest
   distance between smoothed 2D positions), its metric size ``lateral_m = extent_px * z / f`` and the extent of the
   smoothed camera depth along the ray ``extent_z_m``.
3. **Thresholds from observed noise**, per camera: the RealSense depth noise is large, grows with distance and is
   temporally correlated (static background: median smoothed depth extent ~1 cm at 1 m, several cm beyond 1.5 m),
   while the tracker's static 2D jitter is a fraction of a pixel. Thresholds are quantiles of the statistics over a
   REFERENCE population of tracks that stay at least ``reference_robot_distance_m`` from the calibrated, articulated
   robot surface during the whole window (the only actuated body; anything the robot moves is near it); the reference
   still contains tracker failures and any unrelated motion, which makes the thresholds conservative. Move:
   ``extent_px`` above the ``reference_quantile_move`` quantile (at least ``min_move_px``) with ``lateral_m`` at
   least ``min_move_m``, or ``extent_z_m`` above the same quantile of its depth bin (at least ``min_move_m``). Static:
   both statistics at most the ``reference_quantile_static`` quantile.
4. **Temporal support.** A move also needs at least ``min_support`` smoothed samples beyond the threshold from the
   segment's reference sample (a persistent displacement, not one excursion).
5. **Class** = moving if any segment of ``min_segment_samples`` valid samples moves, else uncertain if any such segment
   is neither moving nor static, else static if any was judged, else invalid (not enough samples to judge). A moving
   track keeps its full valid trajectory, stationary intervals included; segment ids are stored so trails and
   supervision never cross a break.

The per-sample noise of both sensors (robust second differences, camera frame) is reported alongside, for reference.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from s4d.realworld import io as rio
from s4d.realworld.tracking import EDGE, LOW_CONF, NO_DEPTH, NOT_TRACKED, OCCLUDED, OUT, VALID

CLASSES = {0: "invalid", 1: "static", 2: "moving", 3: "uncertain"}
INVALID, STATIC, MOVING, UNCERTAIN = range(4)
IDENTITY_BREAK = (OCCLUDED, LOW_CONF, OUT, NOT_TRACKED)
DEPTH_HOLE = (NO_DEPTH, EDGE)
BREAKS = ("identity", "depth_hole", "time_gap", "speed")
STAGE_VERSION = "1"


@dataclass(frozen=True)
class MotionFilterConfig:
    min_segment_samples: int = 5
    max_depth_hole_frames: int = 2
    max_gap_factor: float = 1.5
    max_speed_m_s: float = 1.5
    reference_robot_distance_m: float = 0.15
    reference_quantile_move: float = 0.995
    reference_quantile_static: float = 0.9
    min_move_px: float = 1.5
    min_move_m: float = 0.005
    min_support: int = 2
    depth_bins_m: tuple = (0.0, 0.4, 0.6, 0.8, 1.0, 1.2, 1.5, 2.0, 3.0, 10.0)
    min_bin_samples: int = 200
    robot_points_per_link: int = 150


# ------------------------------------------------------------------------------------------------ segmentation


def segment_tracks(status: np.ndarray, xyz_cam: np.ndarray, times_s: np.ndarray, cfg: MotionFilterConfig,
                   period_s: float) -> tuple[np.ndarray, dict]:
    """status (T,N) lift3d codes, xyz_cam (T,N,3) (NaN unless valid), times_s (T,) -> segment ids (T,N) int16
    (-1 where the sample is not valid; ids restart at 0 per track) and counts of each break reason."""
    T, N = status.shape
    seg = np.full((T, N), -1, np.int16)
    cur = np.full(N, -1, np.int32)
    broken = np.ones(N, bool)
    hole = np.zeros(N, np.int32)
    last_t = np.full(N, -1, np.int32)
    counts = dict.fromkeys(BREAKS, 0)
    for t in range(T):
        if t > 0 and times_s[t] - times_s[t - 1] > cfg.max_gap_factor * period_s:
            counts["time_gap"] += int((cur >= 0).sum())
            broken[:] = True
        st = status[t]
        ident = np.isin(st, IDENTITY_BREAK)
        counts["identity"] += int((ident & ~broken & (cur >= 0)).sum())
        broken |= ident
        hole[ident] = 0
        h = np.isin(st, DEPTH_HOLE)
        hole[h] += 1
        long_hole = h & (hole > cfg.max_depth_hole_frames) & ~broken & (cur >= 0)
        counts["depth_hole"] += int(long_hole.sum())
        broken |= long_hole
        v = st == VALID
        check = v & ~broken & (last_t >= 0)
        if check.any():
            idx = np.flatnonzero(check)
            dt = times_s[t] - times_s[last_t[idx]]
            step = np.linalg.norm(xyz_cam[t, idx] - xyz_cam[last_t[idx], idx], axis=-1)
            jump = step > cfg.max_speed_m_s * dt
            counts["speed"] += int(jump.sum())
            broken[idx[jump]] = True
        new = v & broken
        cur[new] += 1
        seg[t, v] = cur[v]
        broken[v] = False
        hole[v] = 0
        last_t[v] = t
    return seg, counts


# ------------------------------------------------------------------------------------------------ noise model


def second_differences(xyz_cam: np.ndarray, seg: np.ndarray, frame_steps: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(z (K,), e (K,3)) for every triplet of consecutive frames (equal steps) valid in one segment."""
    T = xyz_cam.shape[0]
    zs, es = [], []
    for t in range(1, T - 1):
        if frame_steps[t - 1] != frame_steps[t]:
            continue
        ok = (seg[t - 1] >= 0) & (seg[t - 1] == seg[t]) & (seg[t] == seg[t + 1])
        if ok.any():
            es.append(xyz_cam[t + 1, ok] - 2 * xyz_cam[t, ok] + xyz_cam[t - 1, ok])
            zs.append(xyz_cam[t, ok, 2])
    if not zs:
        return np.zeros(0), np.zeros((0, 3))
    return np.concatenate(zs).astype(np.float64), np.concatenate(es).astype(np.float64)


def fit_noise(z: np.ndarray, e: np.ndarray, cfg: MotionFilterConfig) -> dict:
    """Per depth bin: sigma_lat, sigma_ray = 1.4826 * median|e| / sqrt(6) (iid per-sample noise; MAD -> sigma)."""
    edges = np.asarray(cfg.depth_bins_m, float)
    rows = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (z >= lo) & (z < hi)
        n = int(m.sum())
        if n >= cfg.min_bin_samples:
            lat = 1.4826 * float(np.median(np.abs(e[m, :2]))) / np.sqrt(6.0)
            ray = 1.4826 * float(np.median(np.abs(e[m, 2]))) / np.sqrt(6.0)
        else:
            lat = ray = None
        rows.append({"z_lo": float(lo), "z_hi": float(hi), "triplets": n, "sigma_lat_m": lat, "sigma_ray_m": ray,
                     "median_z_m": float(np.median(z[m])) if n else None})
    filled = [r for r in rows if r["sigma_lat_m"] is not None]
    if not filled:
        raise ValueError("no depth bin has enough second-difference triplets to estimate the noise")
    for r in rows:  # borrow from the nearest populated bin (recorded)
        if r["sigma_lat_m"] is None:
            mid = 0.5 * (r["z_lo"] + min(r["z_hi"], 5.0))
            near = min(filled, key=lambda f: abs(0.5 * (f["z_lo"] + min(f["z_hi"], 5.0)) - mid))
            r.update(sigma_lat_m=near["sigma_lat_m"], sigma_ray_m=near["sigma_ray_m"],
                     borrowed_from=[near["z_lo"], near["z_hi"]])
    return {"bins": rows, "estimator": "1.4826*median|second difference|/sqrt(6) per axis group, camera frame",
            "triplets": int(len(z))}


# ------------------------------------------------------------------------------------------------ statistics


def median3(x: np.ndarray) -> np.ndarray:
    """(M,D) -> (M-2,D) component-wise median of each 3 consecutive rows."""
    return np.median(np.stack([x[:-2], x[1:-1], x[2:]]), 0)


def segment_stats(uv: np.ndarray, xyz_cam: np.ndarray, seg: np.ndarray, focal_px: float,
                  min_samples: int) -> list[list[dict]]:
    """Per track, per judged segment (>= ``min_samples`` valid samples): smoothed 2D/depth samples and extents."""
    T, N = seg.shape
    out = []
    for n in range(N):
        segs = []
        for s_id in np.unique(seg[:, n][seg[:, n] >= 0]):
            t = np.flatnonzero(seg[:, n] == s_id)
            if len(t) < min_samples:
                continue
            su = median3(uv[t, n].astype(np.float64))
            sz = median3(xyz_cam[t, n, 2:3].astype(np.float64))[:, 0]
            du = np.linalg.norm(su[:, None] - su[None], axis=-1)
            i, j = np.unravel_index(int(np.argmax(du)), du.shape)
            z_med = float(np.median(sz))
            segs.append({"su": su, "sz": sz, "ref": min(i, j), "extent_px": float(du[i, j]),
                         "lateral_m": float(du[i, j]) * z_med / focal_px, "extent_z_m": float(sz.max() - sz.min()),
                         "z_m": z_med, "samples": len(t)})
        out.append(segs)
    return out


def depth_bin(z, edges) -> np.ndarray:
    return np.clip(np.searchsorted(edges, z, side="right") - 1, 0, len(edges) - 2)


def reference_thresholds(ref_px: np.ndarray, ref_z: np.ndarray, ref_depth: np.ndarray,
                         cfg: MotionFilterConfig) -> dict:
    """Quantile thresholds from the reference population (see module docstring); depth-axis thresholds per depth bin,
    bins with fewer than ``min_bin_samples`` reference segments borrow the nearest populated bin (recorded)."""
    if len(ref_px) < cfg.min_bin_samples:
        raise ValueError(f"only {len(ref_px)} reference segments (far from the robot); cannot set noise thresholds")
    edges = np.asarray(cfg.depth_bins_m, float)
    qm, qs = cfg.reference_quantile_move, cfg.reference_quantile_static
    out = {"reference_segments": int(len(ref_px)),
           "px_move": max(cfg.min_move_px, float(np.quantile(ref_px, qm))), "px_static": float(np.quantile(ref_px, qs)),
           "px_quantiles": {str(q): float(np.quantile(ref_px, q)) for q in (0.5, 0.9, 0.99, 0.995, 0.999)},
           "z_bins": []}
    b = depth_bin(ref_depth, edges)
    for k in range(len(edges) - 1):
        m = b == k
        n = int(m.sum())
        out["z_bins"].append({"z_lo": float(edges[k]), "z_hi": float(edges[k + 1]), "segments": n,
                              "z_move_m": max(cfg.min_move_m, float(np.quantile(ref_z[m], qm))) if n >= cfg.min_bin_samples else None,
                              "z_static_m": float(np.quantile(ref_z[m], qs)) if n >= cfg.min_bin_samples else None,
                              "z_median_m": float(np.median(ref_z[m])) if n else None})
    filled = [r for r in out["z_bins"] if r["z_move_m"] is not None]
    if not filled:
        raise ValueError("no depth bin has enough reference segments")
    for r in out["z_bins"]:
        if r["z_move_m"] is None:
            mid = 0.5 * (r["z_lo"] + min(r["z_hi"], 5.0))
            near = min(filled, key=lambda f: abs(0.5 * (f["z_lo"] + min(f["z_hi"], 5.0)) - mid))
            r.update(z_move_m=near["z_move_m"], z_static_m=near["z_static_m"], borrowed_from=[near["z_lo"], near["z_hi"]])
    return out


def decide(segs: list[dict], thr: dict, cfg: MotionFilterConfig) -> tuple[int, dict | None]:
    """Track class from its judged segments (module docstring, steps 3-5) and the decisive segment."""
    edges = np.asarray(cfg.depth_bins_m, float)
    best_rank, best = 0, None
    for s in segs:
        zb = thr["z_bins"][int(depth_bin(np.array([s["z_m"]]), edges)[0])]
        d_px = np.linalg.norm(s["su"] - s["su"][s["ref"]], axis=-1)
        d_z = np.abs(s["sz"] - s["sz"][s["ref"]])
        move_px = s["extent_px"] > thr["px_move"] and s["lateral_m"] >= cfg.min_move_m and \
            int((d_px > thr["px_move"]).sum()) >= cfg.min_support
        move_z = s["extent_z_m"] > zb["z_move_m"] and int((d_z > zb["z_move_m"]).sum()) >= cfg.min_support
        if move_px or move_z:
            rank = 3
        elif s["extent_px"] <= thr["px_static"] and s["extent_z_m"] <= zb["z_static_m"]:
            rank = 1
        else:
            rank = 2
        if rank > best_rank:
            best_rank, best = rank, {**s, "by": ("px" if move_px else "") + ("z" if move_z else "")}
    return {3: MOVING, 2: UNCERTAIN, 1: STATIC, 0: INVALID}[best_rank], best


def class_counts(cls: np.ndarray) -> dict:
    out = {"candidate": int(cls.size)}
    out.update({CLASSES[k]: int((cls == k).sum()) for k in CLASSES})
    return out


# ------------------------------------------------------------------------------------------------ stage


def track_motion(root: Path, episode_id: str, *, cfg: MotionFilterConfig | None = None, force: bool = False) -> dict:
    """Classify every lift3d chunk (local CPU stage: lift3d + track2d outputs, calibration, robot FK)."""
    from scipy.spatial import cKDTree

    from s4d.realworld.geometry import transform_points
    from s4d.realworld.pipeline import RobotGeometry, _record_fp, accepted_calibration, load_meta, robot_state_at

    cfg = cfg or MotionFilterConfig()
    meta = load_meta(root, episode_id)
    lift = rio.read_record(rio.stage_dir(root, episode_id, "lift3d"))
    if lift is None:
        raise RuntimeError("track_motion needs a complete lift3d stage")
    config = {"cfg": asdict(cfg)}
    inputs = {k: _record_fp(root, episode_id, k) for k in ("lift3d", "track2d", "calibrate", "ingest")}
    directory = rio.stage_dir(root, episode_id, "track_motion")
    lift_dir, t2d_dir = rio.stage_dir(root, episode_id, "lift3d"), rio.stage_dir(root, episode_id, "track2d")
    geom = RobotGeometry(meta, points_per_link=cfg.robot_points_per_link)

    def build(tmp: Path) -> dict:
        summary, models = {}, {}
        for cam, info in lift["summary"].items():
            if "chunks" not in info:
                summary[cam] = {"skipped": info.get("skipped", "no lifted tracks")}
                continue
            stamps = np.asarray(meta.stream_timestamps_ns[cam], np.int64)
            period = float(np.median(np.diff(stamps))) * 1e-9
            intr = meta.stream(cam).intrinsics
            focal = 0.5 * (intr.fx + intr.fy)
            T = accepted_calibration(root, episode_id, cam)
            q_all, g_all, ok_all, _ = robot_state_at(meta, stamps.astype(np.float64))
            chunks, zs, es = {}, [], []
            breaks = dict.fromkeys(BREAKS, 0)
            ref_px, ref_z, ref_depth = [], [], []
            for name in info["chunks"]:
                with np.load(lift_dir / name) as z:
                    ch = {k: z[k] for k in ("frame_indices", "xyz_cam", "status", "point_ids")}
                with np.load(t2d_dir / name) as z:
                    uv = z["uv"]
                fi = ch["frame_indices"].astype(np.int64)
                xyz = ch["xyz_cam"].astype(np.float64)
                seg, cnt = segment_tracks(ch["status"], xyz, (stamps[fi] - stamps[0]) * 1e-9, cfg, period)
                for k, v in cnt.items():
                    breaks[k] += v
                zz, ee = second_differences(xyz, seg, np.diff(fi))
                zs.append(zz)
                es.append(ee)
                stats = segment_stats(uv, xyz, seg, focal, cfg.min_segment_samples)
                # distance of every valid sample to the articulated robot surface at that frame (camera frame)
                far = np.ones(xyz.shape[1], bool)
                for ti, f in enumerate(fi):
                    v = seg[ti] >= 0
                    if not v.any():
                        continue
                    if not ok_all[f]:
                        far[v] = False  # robot state unknown: never a reference
                        continue
                    surf = transform_points(T, geom.surface.points(geom.q(q_all[f], g_all[f]))[0])
                    d, _ = cKDTree(surf).query(xyz[ti, v])
                    far[np.flatnonzero(v)[d < cfg.reference_robot_distance_m]] = False
                for n, segs in enumerate(stats):
                    if far[n]:
                        for s in segs:
                            ref_px.append(s["extent_px"])
                            ref_z.append(s["extent_z_m"])
                            ref_depth.append(s["z_m"])
                chunks[name] = (ch, seg, stats, far)
            thr = reference_thresholds(np.array(ref_px), np.array(ref_z), np.array(ref_depth), cfg)
            noise = fit_noise(np.concatenate(zs), np.concatenate(es), cfg)
            totals = dict.fromkeys(["candidate", *CLASSES.values()], 0)
            by = {"px": 0, "z": 0, "pxz": 0}
            moving_samples = 0
            for name, (ch, seg, stats, far) in chunks.items():
                N = seg.shape[1]
                res = {"class": np.zeros(N, np.uint8), "extent_px": np.full(N, np.nan, np.float32),
                       "lateral_m": np.full(N, np.nan, np.float32), "extent_z_m": np.full(N, np.nan, np.float32),
                       "decisive_segment_samples": np.zeros(N, np.int16), "reference": far,
                       "n_valid": (seg >= 0).sum(0).astype(np.int16)}
                for n, segs in enumerate(stats):
                    c, best = decide(segs, thr, cfg)
                    res["class"][n] = c
                    if best is not None:
                        res["extent_px"][n], res["lateral_m"][n] = best["extent_px"], best["lateral_m"]
                        res["extent_z_m"][n], res["decisive_segment_samples"][n] = best["extent_z_m"], best["samples"]
                        if c == MOVING:
                            by[best["by"]] += 1
                for k, v in class_counts(res["class"]).items():
                    totals[k] += v
                moving_samples += int((seg[:, res["class"] == MOVING] >= 0).sum())
                rio.atomic_write_npz(tmp / name, segment=seg, frame_indices=ch["frame_indices"],
                                     point_ids=ch["point_ids"], **res)
            models[cam] = {"thresholds": thr, "per_sample_noise": noise, "median_period_s": period, "focal_px": focal}
            summary[cam] = {"chunks": info["chunks"], "counts": totals, "moving_by": by, "breaks": breaks,
                            "moving_valid_samples": moving_samples,
                            "thresholds": {"px_move": thr["px_move"], "px_static": thr["px_static"],
                                           "reference_segments": thr["reference_segments"],
                                           "z_move_m_by_bin": [[b["z_lo"], b["z_hi"], b["z_move_m"], b["segments"]]
                                                               for b in thr["z_bins"]]}}
        rio.atomic_write_json(tmp / "noise_models.json", {"config": asdict(cfg), "cameras": models,
                                                          "classes": CLASSES})
        totals = {}
        for v in summary.values():
            for k, c in v.get("counts", {}).items():
                totals[k] = totals.get(k, 0) + c
        return {"cameras": summary, "counts": totals,
                "note": "dense AllTracker inference is unchanged; only the stored/exported tracks are filtered"}

    return rio.run_stage(directory, stage="track_motion", version=STAGE_VERSION, config=config, inputs=inputs,
                         build=build, force=force)


def load_motion(root: Path, episode_id: str, chunk_name: str) -> dict:
    with np.load(rio.stage_dir(root, episode_id, "track_motion") / chunk_name) as z:
        return {k: z[k] for k in z.files}


def describe(cfg: MotionFilterConfig) -> str:
    return json.dumps(asdict(cfg), sort_keys=True)
