"""Compute kernels of the real-world pipeline: pure functions ``(params, arrays) -> (outputs, meta)``.

Stages read their inputs, call a kernel through a runner (``s4d.realworld.runners``), durably save the outputs and only
then acknowledge. The same kernel runs in-process (tests, ``LocalRunner``) or inside the inference container
(``s4d/realworld/workers/s4d_kernels_worker.py`` in the repository environment), so where the compute happens never
changes what it computes. Kernels only wrap the existing, tested functions of this package; they hold no I/O.
"""

from __future__ import annotations

import json

import numpy as np

KERNELS: dict = {}


def kernel(name: str):
    def register(fn):
        KERNELS[name] = fn
        return fn

    return register


def run_kernel(task: str, params: dict, arrays: dict) -> tuple[dict, dict]:
    if task not in KERNELS:
        raise KeyError(f"unknown kernel {task!r}; known: {sorted(KERNELS)}")
    return KERNELS[task](params, arrays)


@kernel("register_depth")
def register_depth(params: dict, arrays: dict) -> tuple[dict, dict]:
    """Raw sensor depth (N,Hd,Wd) uint16 -> metric camera-z on the colour grid (N,H,W) float32 + validity.

    Units: ``depth_scale_m`` per stored unit; 0 is invalid; values outside [min_depth_m, max_depth_m] are rejected
    (counted) before the z-buffer forward warp ``geometry.reproject_depth`` (holes stay invalid, never filled)."""
    from s4d.realworld.geometry import reproject_depth

    raw = arrays["depth_raw"]
    K_d, K_c = np.asarray(arrays["K_depth"], float), np.asarray(arrays["K_color"], float)
    T = np.asarray(arrays["T_color_depth"], float)
    hw = tuple(int(x) for x in params["color_hw"])
    lo, hi, scale = float(params["min_depth_m"]), float(params["max_depth_m"]), float(params["depth_scale_m"])
    out = np.zeros((len(raw),) + hw, np.float32)
    stats = []
    for k, d in enumerate(raw):
        m = d.astype(np.float64) * scale
        sensor_valid = d > 0
        keep = sensor_valid & (m >= lo) & (m <= hi)
        out[k] = reproject_depth(np.where(keep, m, 0.0), K_d, T, K_c, hw, splat=int(params.get("splat", 2)))
        stats.append({"sensor_valid_fraction": float(sensor_valid.mean()),
                      "rejected_below_min": int((sensor_valid & (m < lo)).sum()),
                      "rejected_above_max": int((sensor_valid & (m > hi)).sum()),
                      "registered_valid_fraction": float((out[k] > 0).mean())})
    return {"depth": out, "valid": out > 0}, {"frames": stats}


@kernel("ir_disparity_to_color")
def ir_disparity_to_color(params: dict, arrays: dict) -> tuple[dict, dict]:
    """Left-referenced IR disparity (N,H,W) + flip-and-swap right disparity -> metric left-IR depth -> colour grid
    (``s4d.realworld.ir_stereo_depth`` conventions). Returns colour ``depth``, ``source_index``, IR ``lr_error`` and
    ``ir_valid``; values outside [min_depth_m, max_depth_m] are rejected before the same z-buffer forward warp."""
    from s4d.realworld.geometry import reproject_depth
    from s4d.realworld.stereo import disparity_to_depth, lr_consistency, visible_in_other

    dl, dr = np.asarray(arrays["disparity_left"], np.float32), np.asarray(arrays["disparity_right"], np.float32)
    K_ir, K_c = np.asarray(arrays["K_ir"], float), np.asarray(arrays["K_color"], float)
    T = np.asarray(arrays["T_color_left"], float)
    hw = tuple(int(x) for x in params["color_hw"])
    lo, hi = float(params["min_depth_m"]), float(params["max_depth_m"])
    out = {"depth": np.zeros((len(dl),) + hw, np.float32), "source_index": np.zeros((len(dl),) + hw, np.int32),
           "lr_error": np.zeros(dl.shape, np.float32), "ir_valid": np.zeros(dl.shape, bool)}
    stats = []
    for k in range(len(dl)):
        z, ok = disparity_to_depth(dl[k], K_ir[0, 0], float(params["baseline_m"]), 0.0, float(params["min_disparity"]))
        ok &= visible_in_other(dl[k], "left") & (z >= lo) & (z <= hi)
        out["ir_valid"][k] = ok
        out["lr_error"][k] = lr_consistency(dl[k], dr[k], float(params["min_disparity"]))
        out["depth"][k], out["source_index"][k] = reproject_depth(np.where(ok, z, 0.0), K_ir, T, K_c, hw,
                                                                  splat=int(params["splat"]), return_source=True)
        e = out["lr_error"][k][ok & np.isfinite(out["lr_error"][k])]
        stats.append({"ir_valid_fraction": float(ok.mean()),
                      "registered_valid_fraction": float((out["depth"][k] > 0).mean()),
                      "lr_error_median_px": float(np.median(e)) if len(e) else None})
    return out, {"frames": stats}


@kernel("calibrate_camera")
def calibrate_camera_kernel(params: dict, arrays: dict) -> tuple[dict, dict]:
    """``calibration.calibrate_camera`` on stacked frames; overlays for the requested frames."""
    from s4d.realworld.calibration import CalibrationConfig, CalibrationFrame, calibrate_camera, overlay_image

    cfg = CalibrationConfig(**{**params["cfg"], "image_hw": tuple(params["cfg"]["image_hw"])})
    n = len(arrays["t_index"])
    depth = arrays.get("depth")
    masks = arrays.get("robot_mask")
    frames = [CalibrationFrame(int(arrays["t_index"][k]), int(arrays["timestamp_ns"][k]), arrays["landmarks_base"][k],
                               arrays["landmarks_uv"][k], arrays["landmarks_conf"][k], arrays["surface_base"][k],
                               arrays["flange_base"][k], None if depth is None else depth[k],
                               None if masks is None else masks[k]) for k in range(n)]
    K = np.asarray(arrays["K"], float)
    res = calibrate_camera(params["camera_id"], frames, K, cfg,
                           T_init=None if "T_init" not in arrays else arrays["T_init"])
    overlays = []
    if res.T_cam_base is not None and "overlay_images" in arrays:
        for img, k in zip(arrays["overlay_images"], arrays["overlay_frames"]):
            overlays.append(overlay_image(img, res.T_cam_base, frames[int(k)], K, cfg.image_hw))
    out = {"overlays": np.stack(overlays) if overlays else np.zeros((0, *cfg.image_hw, 3), np.uint8)}
    return out, {"result": _finite(json.loads(json.dumps(res.to_json(), default=_json_default)))}


@kernel("lift_tracks")
def lift_tracks_kernel(params: dict, arrays: dict) -> tuple[dict, dict]:
    """``tracking.lift_tracks`` for one track chunk with its frames' depth on the same grid."""
    from s4d.realworld.tracking import Tracks2D, lift_tracks, status_counts

    tr = Tracks2D(params["camera_id"], int(params["query_frame"]), arrays["frame_indices"], arrays["timestamps_ns"],
                  arrays["point_ids"], arrays["uv"], arrays["visibility"], arrays["confidence"],
                  tuple(int(x) for x in params["image_hw"]), dict(params.get("source", {})))
    t3 = lift_tracks(tr, arrays["depth"], np.asarray(arrays["K"], float), np.asarray(arrays["T_cam_base"], float),
                     vis_threshold=float(params["vis_threshold"]), conf_threshold=float(params["conf_threshold"]))
    out = {"frame_indices": t3.frame_indices, "timestamps_ns": t3.timestamps_ns, "point_ids": t3.point_ids,
           "xyz_base": t3.xyz_base, "xyz_cam": t3.xyz_cam, "status": t3.status, "visibility": t3.visibility,
           "confidence": t3.confidence}
    return out, {"status_counts": status_counts(t3), "source": json.loads(json.dumps(t3.source, default=_json_default))}


def _finite(value):
    """Non-finite floats -> "nan"/"inf"/"-inf" strings so results survive strict JSON (records use allow_nan=False)."""
    import math

    if isinstance(value, dict):
        return {k: _finite(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_finite(v) for v in value]
    if isinstance(value, float) and not math.isfinite(value):
        return "nan" if math.isnan(value) else ("inf" if value > 0 else "-inf")
    return value


def _json_default(x):
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, (np.floating, np.integer, np.bool_)):
        return x.item()
    return str(x)


@kernel("depth_set_diagnostics")
def depth_set_diagnostics(params: dict, arrays: dict) -> tuple[dict, dict]:
    """Raw-versus-MapAnything consistency metrics for a chunk of synchronized sets (``s4d.realworld.diagnostics``).

    arrays: ``images`` (S,V,H,W,3) uint8; ``depth_raw``/``valid_raw``/``depth_ma``/``valid_ma`` (S,V,H,W); ``K`` (V,3,3);
    ``T_cam_base`` (V,4,4); ``surface_base`` (S,V,N,3) URDF surface at each view's frame time; ``semantic`` (S,V,H,W)
    uint8 SNeRL classes (optional); ``calib_used`` (S,V) bool. Returns per-set metrics (meta) and, on a fixed pixel grid,
    the per-frame depth of both sources plus a static-pixel flag (outputs) for the temporal-stability statistic."""
    from scipy.ndimage import binary_dilation

    from s4d.realworld import diagnostics as dg

    imgs = arrays["images"]
    S, V, H, W = imgs.shape[:4]
    K, T = np.asarray(arrays["K"], float), np.asarray(arrays["T_cam_base"], float)
    occ, step = float(params.get("occlusion_m", 0.05)), int(params.get("grid_step", 8))
    sem = arrays.get("semantic")
    gi, gj = (x.ravel() for x in np.meshgrid(np.arange(step // 2, H, step), np.arange(step // 2, W, step),
                                             indexing="ij"))
    grid = {"raw": np.zeros((S, V, gi.size), np.float32), "ma": np.zeros((S, V, gi.size), np.float32),
            "static": np.zeros((S, V, gi.size), bool)}
    sets = []
    rng = np.random.default_rng(int(params.get("seed", 0)))
    for s in range(S):
        row = {"views": []}
        zbufs = [dg.robot_zbuffer(transform_points_(T[v], arrays["surface_base"][s, v]), K[v], (H, W)) for v in range(V)]
        for v in range(V):
            robot_px = binary_dilation(np.isfinite(zbufs[v]), iterations=int(params.get("dilate_px", 6)))
            obj = (sem[s, v] == 2) if sem is not None else np.zeros((H, W), bool)
            fix = (sem[s, v] == 3) if sem is not None else np.zeros((H, W), bool)
            static = ~robot_px & ~binary_dilation(obj, iterations=4)
            vrow = {"calib_used": bool(arrays["calib_used"][s, v])}
            for src in ("raw", "ma"):
                d, ok = arrays[f"depth_{src}"][s, v], arrays[f"valid_{src}"][s, v]
                grid[src][s, v] = np.where(ok, d, np.nan)[gi, gj]
                ra = dg.robot_alignment(d, ok, zbufs[v])
                vrow[src] = {"valid_fraction": float(ok.mean()),
                             "robot_alignment": dg.summarize(ra),
                             "flying_pixel_fraction": dg.flying_pixel_fraction(d, ok),
                             "edge_to_rgb_px": dg.summarize(dg.edge_to_rgb_distance(d, ok, imgs[s, v]), tols=(1, 2, 4),
                                                            unit="px", scale=1.0),
                             "object_valid_fraction": float(ok[obj].mean()) if obj.any() else None,
                             "fixture_valid_fraction": float(ok[fix].mean()) if fix.any() else None,
                             "static_valid_fraction": float(ok[static].mean()) if static.any() else None}
            grid["static"][s, v] = static[gi, gj]
            vrow["raw_invalid_filled_by_ma"] = float((~arrays["valid_raw"][s, v] & arrays["valid_ma"][s, v]).sum() /
                                                     max(1, (~arrays["valid_raw"][s, v]).sum()))
            row["views"].append(vrow)
        pairs = {}
        for a in range(V):
            for b in range(V):
                if a == b:
                    continue
                entry = {}
                for src in ("raw", "ma"):
                    da, va = arrays[f"depth_{src}"][s, a], arrays[f"valid_{src}"][s, a]
                    db, vb = arrays[f"depth_{src}"][s, b], arrays[f"valid_{src}"][s, b]
                    entry[src] = dg.summarize(dg.reproject_compare(da, va, K[a], T[a], db, vb, K[b], T[b],
                                                                   occlusion_m=occ, rng=rng))
                    if sem is not None:
                        for name, cls in (("object", 2), ("fixture", 3)):
                            m = sem[s, a] == cls
                            if m.any():
                                entry[f"{src}_{name}"] = dg.summarize(dg.reproject_compare(
                                    da, va, K[a], T[a], db, vb, K[b], T[b], occlusion_m=occ, rng=rng, mask_a=m))
                pairs[f"{a}->{b}"] = entry
        row["pairs"] = pairs
        sets.append(row)
    return {**{f"grid_{k}": v for k, v in grid.items()}, "grid_ij": np.stack([gi.ravel(), gj.ravel()], 1)}, \
        {"sets": _finite(sets)}


REGIONS_V2 = ("scene", "mug", "fixture", "robot", "boundary")


@kernel("depth_compare_v2")
def depth_compare_v2(params: dict, arrays: dict) -> tuple[dict, dict]:
    """Raw RealSense vs MapAnything depth, v2 (2026-11-05): every direct comparison on COMMON valid support (pixels
    and reprojected points valid in both sources), coverage and hole filling reported separately, per region.

    arrays as ``depth_set_diagnostics`` plus ``conf_ma`` (S,V,H,W). Regions of a view: ``scene`` (all pixels), ``mug``
    / ``fixture`` (SAM 3 object / fixture masks), ``robot`` (calibrated URDF z-buffer at the recorded joints),
    ``boundary`` (within ``boundary_px`` of an RGB edge; source independent). Per set: per-view coverage, MA - raw
    on common support, robot agreement on common support, edge alignment (+ shift scan on every ``scan_every``-th
    set), confidence split; per ordered view pair and region: cross-view residuals of both sources on the SAME
    points (common support in both views) and of MapAnything-filled points (raw invalid) against the other view's raw
    and MapAnything depth. Returns per-set rows (meta) and the static-pixel grid (outputs) as v1."""
    from scipy.ndimage import binary_dilation

    from s4d.realworld import diagnostics as dg
    from s4d.realworld.geometry import depth_to_points

    imgs = arrays["images"]
    S, V, H, W = imgs.shape[:4]
    K, T = np.asarray(arrays["K"], float), np.asarray(arrays["T_cam_base"], float)
    occ, step = float(params.get("occlusion_m", 0.05)), int(params.get("grid_step", 8))
    bpx, front = int(params.get("boundary_px", 3)), float(params.get("front_tol_m", 0.05))
    max_pts, scan_every = int(params.get("max_points", 20000)), int(params.get("scan_every", 8))
    first_set = int(params.get("first_set", 0))
    sem = arrays.get("semantic")
    gi, gj = (x.ravel() for x in np.meshgrid(np.arange(step // 2, H, step), np.arange(step // 2, W, step),
                                             indexing="ij"))
    grid = {"raw": np.zeros((S, V, gi.size), np.float32), "ma": np.zeros((S, V, gi.size), np.float32),
            "static": np.zeros((S, V, gi.size), bool)}
    rng = np.random.default_rng(int(params.get("seed", 0)) + first_set)
    sets = []
    for s in range(S):
        row = {"set": first_set + s, "views": [], "pairs": {}}
        dr, vr = arrays["depth_raw"][s].astype(np.float64), arrays["valid_raw"][s].astype(bool)
        dm, vm = arrays["depth_ma"][s].astype(np.float64), arrays["valid_ma"][s].astype(bool)
        both = vr & vm
        regions = []
        for v in range(V):
            zb = dg.robot_zbuffer(transform_points_(T[v], arrays["surface_base"][s, v]), K[v], (H, W))
            robot = np.isfinite(zb)
            reg = {"scene": np.ones((H, W), bool), "robot": robot, "boundary": dg.rgb_boundary(imgs[s, v], bpx),
                   "mug": (sem[s, v] == 2) if sem is not None else np.zeros((H, W), bool),
                   "fixture": (sem[s, v] == 3) if sem is not None else np.zeros((H, W), bool)}
            regions.append(reg)
            vrow = {"calib_used": bool(arrays["calib_used"][s, v]), "coverage": {}, "ma_minus_raw": {}}
            for name in REGIONS_V2:
                m = reg[name]
                n = int(m.sum())
                vrow["coverage"][name] = None if not n else {
                    "pixels": n, "raw": float(vr[v][m].mean()), "ma": float(vm[v][m].mean()),
                    "both": float(both[v][m].mean()), "ma_only_filled": float((vm[v] & ~vr[v])[m].mean()),
                    "raw_only_dropped_by_ma": float((vr[v] & ~vm[v])[m].mean())}
                c = both[v] & m
                vrow["ma_minus_raw"][name] = dg.summarize(dm[v][c] - dr[v][c]) if c.any() else {"n": 0}
            c = both[v] & (dr[v] > 0)
            vrow["ma_over_raw_median"] = float(np.median(dm[v][c] / dr[v][c])) if c.any() else None
            # robot geometry on common support: same pixels for both sources (robot front-most in both)
            m = robot & both[v] & (dr[v] - zb > -front) & (dm[v] - zb > -front)
            vrow["robot_common"] = {"raw": dg.summarize(dr[v][m] - zb[m]), "ma": dg.summarize(dm[v][m] - zb[m])}
            conf = arrays["conf_ma"][s, v].astype(np.float32)
            if both[v].any():
                cmed = float(np.median(conf[both[v]]))
                hi, lo = both[v] & (conf >= cmed), both[v] & (conf < cmed)
                vrow["conf"] = {"median_on_common": cmed,
                                "ma_minus_raw_high_conf": dg.summarize(dm[v][hi] - dr[v][hi]),
                                "ma_minus_raw_low_conf": dg.summarize(dm[v][lo] - dr[v][lo]),
                                "filled_median_conf": float(np.median(conf[vm[v] & ~vr[v]]))
                                if (vm[v] & ~vr[v]).any() else None}
            for src, d, ok in (("raw", dr[v], vr[v]), ("ma", dm[v], vm[v])):
                vrow[src] = {"edge_to_rgb_px": dg.summarize(dg.edge_to_rgb_distance(d, ok, imgs[s, v]), tols=(1, 2, 4),
                                                            unit="px", scale=1.0),
                             # common support: whole neighbourhoods valid in BOTH sources (a hole in one source must
                             # not change the other's score); own-support value kept as a separate diagnostic
                             "flying_pixel_fraction_common": dg.flying_pixel_fraction(d, both[v]),
                             "flying_pixel_fraction_own_support": dg.flying_pixel_fraction(d, ok)}
                if (first_set + s) % scan_every == 0:
                    vrow[src]["edge_shift_scan"] = dg.edge_shift_scan(d, ok, imgs[s, v])
            static = ~binary_dilation(robot, iterations=int(params.get("dilate_px", 6))) & \
                ~binary_dilation(reg["mug"], iterations=4)
            grid["raw"][s, v] = np.where(vr[v], dr[v], np.nan)[gi, gj]
            grid["ma"][s, v] = np.where(vm[v], dm[v], np.nan)[gi, gj]
            grid["static"][s, v] = static[gi, gj]
            row["views"].append(vrow)
        pts = {src: [depth_to_points(np.where(ok[v], d[v], 0.0), K[v])[0] for v in range(V)]
               for src, d, ok in (("raw", dr, vr), ("ma", dm, vm))}
        for a in range(V):
            for b in range(V):
                if a == b:
                    continue
                entry = {}
                for name in REGIONS_V2:
                    idx = np.flatnonzero((both[a] & regions[a][name]).ravel())
                    if len(idx) > max_pts:
                        idx = rng.choice(idx, max_pts, replace=False)
                    if not len(idx):
                        continue
                    res = {}
                    for src, d_b, ok_b in (("raw", dr[b], both[b]), ("ma", dm[b], both[b])):
                        res[src] = dg.reproject_pair(pts[src][a].reshape(-1, 3)[idx], T[a], T[b], K[b], d_b, ok_b,
                                                     occlusion_m=occ)
                    keep = res["raw"][1] & res["ma"][1]  # identical point set for both sources
                    entry[name] = {"points": int(len(idx)), "kept": int(keep.sum()),
                                   "occluded_raw": float(res["raw"][2].mean()),
                                   "occluded_ma": float(res["ma"][2].mean()),
                                   "raw": dg.summarize(res["raw"][0][keep]), "ma": dg.summarize(res["ma"][0][keep])}
                filled = np.flatnonzero((vm[a] & ~vr[a]).ravel())
                if len(filled) > max_pts:
                    filled = rng.choice(filled, max_pts, replace=False)
                if len(filled):
                    P = pts["ma"][a].reshape(-1, 3)[filled]
                    d1, k1, _ = dg.reproject_pair(P, T[a], T[b], K[b], dr[b], vr[b], occlusion_m=occ)
                    d2, k2, _ = dg.reproject_pair(P, T[a], T[b], K[b], dm[b], vm[b], occlusion_m=occ)
                    entry["filled"] = {"points": int(len(filled)), "vs_other_raw": dg.summarize(d1[k1]),
                                       "vs_other_ma": dg.summarize(d2[k2])}
                ref = np.flatnonzero(vr[a].ravel())
                if len(ref) > max_pts:
                    ref = rng.choice(ref, max_pts, replace=False)
                if len(ref):  # reference: measured raw points vs the other camera's raw depth (all raw support)
                    d3, k3, _ = dg.reproject_pair(pts["raw"][a].reshape(-1, 3)[ref], T[a], T[b], K[b], dr[b], vr[b],
                                                  occlusion_m=occ)
                    entry["raw_all_vs_other_raw"] = dg.summarize(d3[k3])
                row["pairs"][f"{a}->{b}"] = entry
        sets.append(row)
    return {**{f"grid_{k}": v for k, v in grid.items()}, "grid_ij": np.stack([gi.ravel(), gj.ravel()], 1)}, \
        {"sets": _finite(sets)}


REGIONS_MULTI = ("scene", "mug", "fixture", "robot", "gripper", "boundary")


@kernel("depth_compare_multi")
def depth_compare_multi(params: dict, arrays: dict) -> tuple[dict, dict]:
    """K depth sources of the same views (controlled depth comparison, 2026-11-05).

    arrays: ``images`` (S,V,H,W,3); ``depth_<name>``/``valid_<name>`` (S,V,H,W) for every name in
    ``params["sources"]``; ``K`` (V,3,3); ``T_cam_base`` (V,4,4); ``surface_base`` (S,V,N,3) URDF surface at each view's
    frame time with ``gripper_points`` (N,) bool marking gripper-link samples; ``semantic`` (S,V,H,W) optional;
    ``calib_used`` (S,V). Per set and view: coverage per source and region; per ordered view pair, region and
    ``params["pairs"]`` (a, b): cross-view residuals of both sources on the SAME points (valid in a and b in both views);
    the same for all sources on the joint common support; robot/gripper agreement on pairwise common support; for each
    reference-free source its pixels missing in ``params["reference_source"]`` checked against the other views'
    reference measurement; edge alignment (own support) and flying pixels (joint common support)."""
    from s4d.realworld import diagnostics as dg
    from s4d.realworld.geometry import depth_to_points

    imgs = arrays["images"]
    S, V, H, W = imgs.shape[:4]
    names = list(params["sources"])
    pairs = [tuple(p) for p in params["pairs"]]
    ref = params.get("reference_source")
    K, T = np.asarray(arrays["K"], float), np.asarray(arrays["T_cam_base"], float)
    occ, bpx, front = float(params.get("occlusion_m", 0.05)), int(params.get("boundary_px", 3)), \
        float(params.get("front_tol_m", 0.05))
    max_pts, first_set = int(params.get("max_points", 15000)), int(params.get("first_set", 0))
    sem = arrays.get("semantic")
    gmask = np.asarray(arrays["gripper_points"], bool)
    step = int(params.get("grid_step", 8))
    gi, gj = (x.ravel() for x in np.meshgrid(np.arange(step // 2, H, step), np.arange(step // 2, W, step),
                                             indexing="ij"))
    grid = {n: np.zeros((S, V, gi.size), np.float32) for n in names}
    grid["static"] = np.zeros((S, V, gi.size), bool)
    rng = np.random.default_rng(int(params.get("seed", 0)) + first_set)
    sets = []
    for s in range(S):
        D = {n: arrays[f"depth_{n}"][s].astype(np.float64) for n in names}
        Vd = {n: arrays[f"valid_{n}"][s].astype(bool) for n in names}
        joint = np.logical_and.reduce([Vd[n] for n in names])
        row = {"set": first_set + s, "views": [], "pairs": {}}
        regions = []
        for v in range(V):
            surf = transform_points_(T[v], arrays["surface_base"][s, v])
            zb = dg.robot_zbuffer(surf, K[v], (H, W))
            zg = dg.robot_zbuffer(surf[gmask], K[v], (H, W))
            reg = {"scene": np.ones((H, W), bool), "robot": np.isfinite(zb),
                   "gripper": np.isfinite(zg) & (zg <= zb + 1e-6), "boundary": dg.rgb_boundary(imgs[s, v], bpx),
                   "mug": (sem[s, v] == 2) if sem is not None else np.zeros((H, W), bool),
                   "fixture": (sem[s, v] == 3) if sem is not None else np.zeros((H, W), bool)}
            regions.append((reg, zb))
            vrow = {"calib_used": bool(arrays["calib_used"][s, v]), "coverage": {}, "robot": {}, "edges": {},
                    "flying_common": {}}
            for name in REGIONS_MULTI:
                m = reg[name]
                if m.any():
                    vrow["coverage"][name] = {"pixels": int(m.sum()), "joint": float(joint[v][m].mean()),
                                              **{n: float(Vd[n][v][m].mean()) for n in names}}
            for a, b in pairs:  # robot and gripper geometry on the pair's common support
                both = Vd[a][v] & Vd[b][v]
                for rname, rmask in (("robot", reg["robot"]), ("gripper", reg["gripper"])):
                    m = rmask & both & (D[a][v] - zb > -front) & (D[b][v] - zb > -front)
                    vrow["robot"][f"{a}|{b}|{rname}"] = {a: dg.summarize(D[a][v][m] - zb[m]),
                                                         b: dg.summarize(D[b][v][m] - zb[m])}
            for n in names:
                vrow["edges"][n] = dg.summarize(dg.edge_to_rgb_distance(D[n][v], Vd[n][v], imgs[s, v]), tols=(1, 2, 4),
                                                unit="px", scale=1.0)
                vrow["flying_common"][n] = dg.flying_pixel_fraction(D[n][v], joint[v])
                grid[n][s, v] = np.where(Vd[n][v], D[n][v], np.nan)[gi, gj]
            from scipy.ndimage import binary_dilation

            grid["static"][s, v] = (~binary_dilation(reg["robot"], iterations=6) &
                                    ~binary_dilation(reg["mug"], iterations=4))[gi, gj]
            row["views"].append(vrow)
        pts = {n: [depth_to_points(np.where(Vd[n][v], D[n][v], 0.0), K[v])[0].reshape(-1, 3) for v in range(V)]
               for n in names}
        for a_v in range(V):
            for b_v in range(V):
                if a_v == b_v:
                    continue
                entry = {}
                reg = regions[a_v][0]
                for name in REGIONS_MULTI:
                    for key, members in [(f"{p[0]}|{p[1]}", list(p)) for p in pairs] + [("joint", names)]:
                        sup = np.logical_and.reduce([Vd[n][a_v] for n in members]) & reg[name]
                        idx = np.flatnonzero(sup.ravel())
                        if len(idx) > max_pts:
                            idx = rng.choice(idx, max_pts, replace=False)
                        if not len(idx):
                            continue
                        land = np.logical_and.reduce([Vd[n][b_v] for n in members])
                        res = {n: dg.reproject_pair(pts[n][a_v][idx], T[a_v], T[b_v], K[b_v], D[n][b_v], land,
                                                    occlusion_m=occ) for n in members}
                        keep = np.logical_and.reduce([res[n][1] for n in members])
                        entry[f"{key}|{name}"] = {"kept": int(keep.sum()),
                                                  **{n: dg.summarize(res[n][0][keep]) for n in members}}
                if ref is not None:  # pixels a source fills where the reference is missing, vs the other view's reference
                    for n in names:
                        if n == ref:
                            continue
                        idx = np.flatnonzero((Vd[n][a_v] & ~Vd[ref][a_v]).ravel())
                        if len(idx) > max_pts:
                            idx = rng.choice(idx, max_pts, replace=False)
                        if len(idx):
                            d1, k1, _ = dg.reproject_pair(pts[n][a_v][idx], T[a_v], T[b_v], K[b_v], D[ref][b_v],
                                                          Vd[ref][b_v], occlusion_m=occ)
                            entry[f"filled|{n}"] = {"points": int(len(idx)), "vs_other_reference": dg.summarize(d1[k1])}
                    idx = np.flatnonzero(Vd[ref][a_v].ravel())
                    if len(idx) > max_pts:
                        idx = rng.choice(idx, max_pts, replace=False)
                    if len(idx):
                        d3, k3, _ = dg.reproject_pair(pts[ref][a_v][idx], T[a_v], T[b_v], K[b_v], D[ref][b_v],
                                                      Vd[ref][b_v], occlusion_m=occ)
                        entry[f"reference_self|{ref}"] = dg.summarize(d3[k3])
                row["pairs"][f"{a_v}->{b_v}"] = entry
        sets.append(row)
    return {**{f"grid_{k}": v for k, v in grid.items()}}, {"sets": _finite(sets)}


def transform_points_(T, points):
    from s4d.realworld.geometry import transform_points

    return transform_points(T, points)


@kernel("forward_check")
def forward_check(params: dict, arrays: dict) -> tuple[dict, dict]:
    """Inference-only pass of the training objective on one collated real-episode batch: ``build_model(cfg)`` and
    ``forward_losses`` under ``torch.no_grad()`` (no backward, no optimiser, nothing saved). ``arrays`` are the batch
    tensors from ``s4d.data.contract.collate`` as numpy; ``params["meta"]`` the batch ``meta``; ``params["config"]`` a
    repository config path (+ ``overrides``). Returns every loss/metric value, finiteness and the input shapes."""
    import torch

    from s4d.config import load_config
    from s4d.data.contract import validate_batch
    from s4d.train.loop import build_model, forward_losses, move_batch, seed_everything

    repo = __import__("pathlib").Path(__file__).resolve().parents[2]
    cfg = load_config([str(repo / params["config"])], list(params.get("overrides", [])))
    seed_everything(int(params.get("seed", 0)))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch = {k: torch.from_numpy(np.asarray(v)) for k, v in arrays.items()}
    meta = dict(params["meta"])
    meta["t_indices"] = [tuple(x) for x in meta["t_indices"]]
    batch["meta"] = meta
    validate_batch(batch)
    model = build_model(cfg).to(device)
    model.train(bool(params.get("train_mode", True)))
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    with torch.no_grad():
        out = forward_losses(model, move_batch(batch, device), cfg, int(params.get("step", 0)))

    def scalar(x):
        return float(x.detach().float().mean()) if torch.is_tensor(x) else float(x)

    losses = {k: scalar(v) for k, v in out["losses"].items()}
    metrics = {k: scalar(v) for k, v in out.get("metrics", {}).items() if torch.is_tensor(v) or
               isinstance(v, (int, float))}
    total = scalar(out["total"])
    finite = bool(np.isfinite([total, *losses.values(), *metrics.values()]).all())
    meta_out = {"total": total, "losses": losses, "metrics": metrics, "finite": finite,
                "shapes": {k: list(np.asarray(v).shape) for k, v in arrays.items()},
                "dtypes": {k: str(np.asarray(v).dtype) for k, v in arrays.items()},
                "params_million": sum(p.numel() for p in model.parameters()) / 1e6, "device": str(device),
                "train_mode": bool(params.get("train_mode", True)),
                "max_memory_allocated": int(torch.cuda.max_memory_allocated()) if torch.cuda.is_available() else 0}
    return {}, _finite(meta_out)


@kernel("snerl_check")
def snerl_check(params: dict, arrays: dict) -> tuple[dict, dict]:
    """SNeRL's own batch/loss interface on real scenes, inference only (``training_step`` under ``torch.no_grad``).

    Scenes: every view of a synchronized set as encoder input, one view as target (cycled). Images: the largest
    centred square, resized to ``size`` with PIL bilinear (antialiased; SNeRL renders are 128x128 squares), K by the
    exact crop+resize pixel affine (``official_K`` keeps the rig K, as for Meta-World). Semantics: SAM 3 semantic map
    (``SEMANTIC_CLASSES`` ids) resized by exact nearest lookup. Poses: OpenCV c2w -> ``normalise_c2w`` about ``lookat``
    (metres, base frame) with ``pose_scale``. DINO: the ``dino_snerl`` worker's tokens on the SAME 128x128 image,
    upsampled exactly as ``dino_target`` does, compared with the consumer's ``dino_target`` (equality expected)."""
    import importlib.util
    from pathlib import Path

    import torch
    from PIL import Image

    from s4d.baselines.snerl.data import SEMANTIC_CLASSES, normalise_c2w, official_K
    from s4d.baselines.snerl.model import create, dino_target, to_device_scene, training_step
    from s4d.baselines.snerl.upstream import dino_extractor
    from s4d.config import load_config
    from s4d.realworld.geometry import crop_affine, resize_affine

    repo = Path(__file__).resolve().parents[2]
    cfg = load_config([str(repo / params["config"])], [])
    size = int(params.get("size", 128))
    device = torch.device("cuda")
    torch.manual_seed(int(params.get("seed", 0)))
    model = create(dict(cfg["nerf"]), device)
    extractor = dino_extractor(device)
    spec = importlib.util.spec_from_file_location("dino_snerl_worker", repo / "s4d/realworld/workers/dino_snerl_worker.py")
    worker = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(worker)
    frames, sem = arrays["images"], arrays["semantic"]
    S, V, H, W = frames.shape[:4]
    side = min(H, W)
    top, left = (H - side) // 2, (W - side) // 2
    aff = crop_affine(top, left).then(resize_affine((side, side), (size, size)))
    K128 = np.stack([aff.apply_K(np.asarray(k, float)) for k in arrays["K"]])
    rows = (np.floor((np.arange(size) + 0.5) * side / size)).astype(int) + top
    cols = (np.floor((np.arange(size) + 0.5) * side / size)).astype(int) + left
    c2w = normalise_c2w(torch.from_numpy(np.asarray(arrays["cam2world"], np.float64)),
                        tuple(float(x) for x in params["lookat"]), float(params.get("pose_scale", 4.0))).float()
    rng = np.random.default_rng(int(params.get("seed", 0)))
    weights = {k: float(v) for k, v in dict(cfg.get("loss") or {}).items() if k in ("semantic_weight", "dino_weight")}
    out_rows, imgs_out = [], []
    for s in range(S):
        imgs = np.stack([np.asarray(Image.fromarray(frames[s, v][top:top + side, left:left + side]).resize(
            (size, size), Image.BILINEAR)) for v in range(V)])
        sem128 = sem[s][:, rows][:, :, cols].astype(np.int64)
        tgt = s % V
        scene = {"obs_rgb": imgs, "obs_c2w": c2w.clone(), "target_rgb": imgs[tgt], "target_semantic": sem128[tgt],
                 "target_c2w": c2w[tgt].clone(), "target_K": torch.from_numpy(official_K(K128[tgt])),
                 "episode": "real", "t": int(s), "target": int(tgt)}
        dev = to_device_scene(scene, device)
        with torch.no_grad():
            loss, terms = training_step(model, extractor, dev, int(params.get("global_step", 1000)), rng, **weights)
            target = dev["target_rgb"]
            consumer = dino_target(extractor, target)
            _, wout, wmeta = worker.handle({"params": {"output": "tokens"}}, {"images": target[None].cpu().numpy()})
            tokens = torch.from_numpy(wout["tokens"]).to(device)
            up = torch.nn.functional.interpolate(tokens.permute(0, 3, 1, 2), tuple(target.shape[:2]),
                                                 mode="bilinear").permute(0, 2, 3, 1)[0]
        frac = np.bincount(sem128[tgt].ravel(), minlength=len(SEMANTIC_CLASSES))[:len(SEMANTIC_CLASSES)] / sem128[tgt].size
        out_rows.append({"set": int(s), "target_view": int(tgt), "loss": float(loss),
                         "terms": {k: float(v) for k, v in terms.items()},
                         "finite": bool(torch.isfinite(loss)) and all(np.isfinite(float(v)) for v in terms.values()),
                         "dino_tokens_shape": list(wout["tokens"].shape),
                         "dino_worker_equals_consumer": bool(torch.equal(up, consumer)),
                         "dino_max_abs_diff": float((up - consumer).abs().max()),
                         "target_class_fractions": dict(zip(SEMANTIC_CLASSES, map(float, frac)))})
        imgs_out.append(imgs)
    meta = {"scenes": out_rows, "size": size, "crop": [int(top), int(left), int(side)], "K128": K128.tolist(),
            "lookat": list(map(float, params["lookat"])), "pose_scale": float(params.get("pose_scale", 4.0)),
            "near_far_units": [float(cfg["nerf"]["near"]), float(cfg["nerf"]["far"])],
            "dino_spec": getattr(worker, "SNERL_DINO", None)}
    return {"images128": np.stack(imgs_out)}, _finite(meta)


@kernel("rectify_stereo")
def rectify_stereo_kernel(params: dict, arrays: dict) -> tuple[dict, dict]:
    """Bouguet maps and bilinear resampling for both eyes, preserving recorded channel order."""
    from s4d.realworld.geometry import rectify_map, remap

    rect = params["rectification"]
    out = {}
    for eye, rotation, spec in zip(("left", "right"), (rect["R1"], rect["R2"]), params["eyes"]):
        coords = rectify_map(arrays[f"K_{eye}"], spec["distortion_model"], spec["distortion"],
                             np.asarray(rotation), np.asarray(rect["K_rect"]), tuple(rect["size_hw"]))
        out[eye] = np.stack([remap(im, coords, order=1) for im in arrays[eye]])
    return out, {"interpolation_order": 1}
