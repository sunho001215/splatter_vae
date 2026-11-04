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
    gi, gj = np.meshgrid(np.arange(step // 2, H, step), np.arange(step // 2, W, step), indexing="ij")
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
                             "edge_to_rgb_px": dg.summarize(dg.edge_to_rgb_distance(d, ok, imgs[s, v]), tols=(1, 2, 4)),
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
    cfg = load_config(str(repo / params["config"]), list(params.get("overrides", [])))
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
