"""Static cross-camera correspondences for ``rig_dw``: loading, filtering and stratified subsampling of the RoMa v2
matches written by ``workers/romav2_match_worker.py`` (one ``<episode>.npz`` per episode).

Filters (decided a priori, recorded in ``MatchConfig``):
* static ends: both ends farther than ``min_dyn_dist_px`` from SAM robot/object pixels and MAGSAC++ fundamental inlier
  of their frame pair (the worker ran MAGSAC on the static matches only);
* bidirectional cycle consistency <= ``max_cycle_px``; RoMa overlap >= ``min_overlap``;
* a depth end is used only when FS depth is valid, locally smooth (5x5 range <= ``max_range_rel`` x depth) and agrees
  with the sensor depth within ``max_fs_sensor_rel``; otherwise that end is image-only (the factor keeps its 2D part).
Subsampling: per (episode, time, pair) at most ``per_frame_pair`` matches, stratified on a ``grid`` over image a (round
robin over cells, deterministic RNG) so that no image region dominates by count."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np


@dataclass
class MatchConfig:
    min_dyn_dist_px: float = 9.0
    max_cycle_px: float = 1.5
    min_overlap: float = 0.3
    max_range_rel: float = 0.02
    max_fs_sensor_rel: float = 0.05
    per_frame_pair: int = 300
    grid: tuple = (8, 6)
    seed: int = 0
    depth_source: str = "fs"  # "fs" or "sensor"


def _stratified(uv: np.ndarray, n: int, grid, hw, rng) -> np.ndarray:
    if len(uv) <= n:
        return np.arange(len(uv))
    gx, gy = grid
    cell = (np.clip((uv[:, 0] / hw[1] * gx).astype(int), 0, gx - 1) * gy
            + np.clip((uv[:, 1] / hw[0] * gy).astype(int), 0, gy - 1))
    order = rng.permutation(len(uv))
    cell_o = cell[order]
    # rank of each match within its cell (in random order) -> take by rank (round robin)
    srt = np.argsort(cell_o, kind="stable")
    rank = np.empty(len(uv), int)
    c_sorted = cell_o[srt]
    start = np.r_[0, np.flatnonzero(np.diff(c_sorted)) + 1]
    counts = np.diff(np.r_[start, len(srt)])
    rank[srt] = np.arange(len(srt)) - np.repeat(start, counts)
    pick = np.lexsort((cell_o, rank))[:n]
    return np.sort(order[pick])


def load_matches(paths: list[Path], cameras: list[str], cfg: MatchConfig | None = None, hw=(480, 640)) -> dict:
    """Concatenated, filtered and subsampled matches of ``paths``. Returns a dict of arrays: cam_a, cam_b (indices into
    ``cameras``), uv_a, uv_b, d_a, d_b (metres, NaN = image-only end), episode (index into ``episodes``), time, pair,
    plus ``episodes`` (list) and ``config``."""
    cfg = cfg or MatchConfig()
    rng = np.random.default_rng(cfg.seed)
    cols = {k: [] for k in ("cam_a", "cam_b", "uv_a", "uv_b", "d_a", "d_b", "episode", "time", "overlap", "cycle")}
    episodes, stats = [], {"raw": 0, "static_inlier": 0, "kept": 0, "depth_ends": 0}
    for p in paths:
        with np.load(p) as z:
            meta = json.loads(str(z["meta"]))
            arr = {k: z[k] for k in z.files if k != "meta"}
        cams = meta["cameras"]
        e = len(episodes)
        episodes.append(meta["episode_id"])
        stats["raw"] += len(arr["pair"])
        ok = (arr["magsac_inlier"].astype(bool) & (arr["dyn_dist_a"] > cfg.min_dyn_dist_px)
              & (arr["dyn_dist_b"] > cfg.min_dyn_dist_px) & (arr["cycle_px"] <= cfg.max_cycle_px)
              & (arr["overlap"] >= cfg.min_overlap))
        stats["static_inlier"] += int(ok.sum())
        src = ("fs", "sd") if cfg.depth_source == "fs" else ("sd", "fs")
        dep = {}
        for end in ("a", "b"):
            d, other = arr[f"{src[0]}_{end}"].astype(float), arr[f"{src[1]}_{end}"].astype(float)
            good = (d > 0) & (arr[f"fs_range_{end}"] <= cfg.max_range_rel * np.maximum(d, 1e-6)) & (other > 0) \
                & (np.abs(d - other) <= cfg.max_fs_sensor_rel * d)
            dep[end] = np.where(good, d, np.nan)
        for key in np.unique(arr["time"].astype(int) * 100 + arr["pair"].astype(int)):
            sel = np.flatnonzero(ok & (arr["time"].astype(int) * 100 + arr["pair"].astype(int) == key))
            if not len(sel):
                continue
            sel = sel[_stratified(arr["uv_a"][sel].astype(float), cfg.per_frame_pair, cfg.grid, hw, rng)]
            pa, pb = divmod(int(key % 100), 10)
            ca, cb = cameras.index(cams[pa]), cameras.index(cams[pb])
            cols["cam_a"].append(np.full(len(sel), ca, np.int8))
            cols["cam_b"].append(np.full(len(sel), cb, np.int8))
            cols["uv_a"].append(arr["uv_a"][sel].astype(float))
            cols["uv_b"].append(arr["uv_b"][sel].astype(float))
            cols["d_a"].append(dep["a"][sel])
            cols["d_b"].append(dep["b"][sel])
            cols["episode"].append(np.full(len(sel), e, np.int32))
            cols["time"].append(np.full(len(sel), key // 100, np.int16))
            cols["overlap"].append(arr["overlap"][sel])
            cols["cycle"].append(arr["cycle_px"][sel])
    out = {k: np.concatenate(v) if v else np.zeros(0) for k, v in cols.items()}
    stats["kept"] = int(len(out["cam_a"]))
    stats["depth_ends"] = int(np.isfinite(out["d_a"]).sum() + np.isfinite(out["d_b"]).sum())
    out["episodes"] = episodes
    out["config"] = asdict(cfg)
    out["stats"] = stats
    return out


def subset(m: dict, mask: np.ndarray) -> dict:
    out = {k: (v[mask] if isinstance(v, np.ndarray) and len(v) == len(mask) else v) for k, v in m.items()}
    return out


