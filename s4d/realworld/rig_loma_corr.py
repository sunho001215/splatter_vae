"""Loading + geometric filtering of the evidence of ``rig_loma`` (no SAM anywhere):

* robot 2D-3D correspondences written by ``workers/loma_render_match_worker.py``: structural filters (a render
  triangle was hit, the render neighbourhood has no depth discontinuity > ``max_z_range_m``, the ray/plane intersection
  lies inside its triangle), the per-(episode, camera) pooled MAGSAC++ PnP inlier flag, then a per-frame cap
  (round robin over robot links, deterministic RNG) so that no frame / link dominates by count;
* static cross-camera RoMa v2 matches (``matches-v1``) with the SAM-free static flags of
  ``workers/static_flags_worker.py`` (temporal-change distance + MAGSAC++ on those static matches) replacing the SAM
  distance / SAM-static MAGSAC fields; otherwise the ``rig_dw_matches`` rules (cycle, overlap, depth-end validity,
  stratified subsampling);
* depth by stage NAME: ``foundation_stereo_depth`` / ``sensor_depth`` use the columns stored in the match files;
  any other stage (e.g. S2M2) reads ``<depth-samples>/<episode>.npz`` written by ``rig_loma_depth`` (same per-match
  layout: d_a, d_b, range_a, range_b for ext-ext; d, range for the robot correspondences)."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from s4d.realworld.rig_dw_matches import MatchConfig, _stratified
from s4d.realworld.rig_loma_solve import CorrSet


@dataclass
class CorrConfig:
    max_z_range_m: float = 0.02
    max_bary_out: float = 0.05
    require_magsac_cam: bool = True
    min_score: float = 0.0
    per_frame: int = 100
    seed: int = 0
    # Robotiq finger links (names containing "finger"/"knuckle") are dropped when the recorded gripper state exceeds
    # this value: the composed URDF maps reg/255*0.8 rad while a real 2F-85 closes at reg ~227-230 (robot model check
    # 2026-11-08, loma/notes/robot_model_check.md): at the mug-hanging grasp (reg 207-211) each pad renders 4.0-4.4 mm
    # too far out; for reg <= 50 the error is <= 1.5 mm per pad.
    finger_max_grip: float = 50.0


def load_corr(paths: list[Path], cameras: list[str], cfg: CorrConfig | None = None, depth_stage: str | None = None,
              depth_dir: Path | None = None) -> tuple[dict, dict]:
    """{camera: CorrSet} of the correspondence files ``paths`` + statistics. ``depth_stage``/``depth_dir``: observed
    depth at the matched pixels (secondary term / validation) from ``rig_loma_depth`` sample files."""
    cfg = cfg or CorrConfig()
    rng = np.random.default_rng(cfg.seed)
    acc = {c: {k: [] for k in ("X", "V", "uv", "episode", "frame", "link", "d", "T7")} for c in cameras}
    stats = {"raw": 0, "structural": 0, "magsac": 0, "kept": 0, "frames": 0}
    fid = 0
    for e, p in enumerate(paths):
        with np.load(p) as z:
            meta = json.loads(str(z["meta"]))
            a = {k: z[k] for k in z.files if k != "meta"}
        if not len(a.get("cam", [])):
            continue
        dep = None
        if depth_stage and depth_dir is not None:
            q = Path(depth_dir) / p.name
            if q.exists():
                with np.load(q) as zd:
                    dep = np.where(zd["valid"], zd["d"], np.nan)
        stats["raw"] += len(a["cam"])
        ok = (np.isfinite(a["X_base"]).all(1) & (a["z_range"] <= cfg.max_z_range_m)
              & (a["bary_out"] <= cfg.max_bary_out) & (a["score"] >= cfg.min_score))
        if "grip" in a and len(a["grip"]):
            fingers = np.array([("finger" in l or "knuckle" in l) for l in meta["links"]])
            drop = fingers[a["link"].astype(int)] & (a["grip"] > cfg.finger_max_grip)
            stats["finger_dropped"] = stats.get("finger_dropped", 0) + int((ok & drop).sum())
            ok &= ~drop
        stats["structural"] += int(ok.sum())
        if cfg.require_magsac_cam:
            ok &= a["magsac_cam"].astype(bool)
        stats["magsac"] += int(ok.sum())
        cams_f = meta["cameras"]
        stats.setdefault("links", meta["links"])
        for f in np.unique(a["frame"]):
            sel = np.flatnonzero(ok & (a["frame"] == f))
            if not len(sel):
                continue
            if len(sel) > cfg.per_frame:
                lk = a["link"][sel]
                order = rng.permutation(len(sel))
                lo = lk[order]
                srt = np.argsort(lo, kind="stable")
                rank = np.empty(len(sel), int)
                start = np.r_[0, np.flatnonzero(np.diff(lo[srt])) + 1]
                counts = np.diff(np.r_[start, len(srt)])
                rank[srt] = np.arange(len(srt)) - np.repeat(start, counts)
                pick = np.lexsort((lo, rank))[:cfg.per_frame]
                sel = np.sort(sel[order[pick]])
            cam = cams_f[int(a["cam"][sel[0]])]
            if cam not in acc:
                continue
            d = acc[cam]
            d["X"].append(a["X_base"][sel])
            d["V"].append(a["V_base"][sel].astype(float))
            d["uv"].append(a["uv_obs"][sel].astype(float))
            d["episode"].append(np.full(len(sel), e, np.int32))
            d["frame"].append(np.full(len(sel), fid, np.int32))
            d["link"].append(a["link"][sel].astype(np.int32))
            d["d"].append(dep[sel] if dep is not None else np.full(len(sel), np.nan))
            t7 = meta["frames"][int(f)].get("T7")
            d["T7"].append(np.repeat(np.asarray(t7 if t7 is not None else np.full((4, 4), np.nan))[None], len(sel), 0))
            fid += 1
            stats["frames"] += 1
            stats["kept"] += len(sel)
    out = {}
    for c, d in acc.items():
        if not d["X"]:
            continue
        out[c] = CorrSet(np.concatenate(d["X"]), np.concatenate(d["V"]), np.concatenate(d["uv"]),
                         np.concatenate(d["episode"]), np.concatenate(d["frame"]), np.concatenate(d["link"]),
                         np.concatenate(d["d"]) if depth_stage else None, T7=np.concatenate(d["T7"]))
    stats["config"] = asdict(cfg)
    stats["per_camera"] = {c: int(len(v)) for c, v in out.items()}
    return out, stats


def load_matches_samfree(paths: list[Path], flag_dir: Path, cameras: list[str], cfg: MatchConfig | None = None,
                         hw=(480, 640), depth_stage: str = "foundation_stereo_depth",
                         depth_dir: Path | None = None) -> dict:
    """``rig_dw_matches.load_matches`` with the SAM-free static flags (``flag_dir/<episode>.npz``) and the depth of
    ``depth_stage`` (see module doc). The depth-end validity rule is unchanged: valid, 5x5 range <= max_range_rel x d
    and agreement with an independent source (sensor depth; FS for the sensor stage) within max_fs_sensor_rel."""
    cfg = cfg or MatchConfig()
    rng = np.random.default_rng(cfg.seed)
    cols = {k: [] for k in ("cam_a", "cam_b", "uv_a", "uv_b", "d_a", "d_b", "episode", "time", "overlap", "cycle")}
    episodes, stats = [], {"raw": 0, "static_inlier": 0, "kept": 0, "depth_ends": 0}
    for p in paths:
        with np.load(p) as z:
            meta = json.loads(str(z["meta"]))
            arr = {k: z[k] for k in z.files if k != "meta"}
        with np.load(Path(flag_dir) / p.name) as zf:
            arr["dyn_dist_a"], arr["dyn_dist_b"] = zf["dyn_dist_t_a"], zf["dyn_dist_t_b"]
            arr["magsac_inlier"] = zf["magsac_sf"]
        if depth_stage not in ("foundation_stereo_depth", "sensor_depth"):
            with np.load(Path(depth_dir) / p.name) as zd:
                for end in "ab":
                    arr[f"x_{end}"] = np.where(zd[f"valid_{end}"], zd[f"d_{end}"], 0.0)
                    arr[f"x_range_{end}"] = zd[f"range_{end}"]
        cams = meta["cameras"]
        e = len(episodes)
        episodes.append(meta["episode_id"])
        stats["raw"] += len(arr["pair"])
        ok = (arr["magsac_inlier"].astype(bool) & (arr["dyn_dist_a"] > cfg.min_dyn_dist_px)
              & (arr["dyn_dist_b"] > cfg.min_dyn_dist_px) & (arr["cycle_px"] <= cfg.max_cycle_px)
              & (arr["overlap"] >= cfg.min_overlap))
        stats["static_inlier"] += int(ok.sum())
        dep = {}
        for end in ("a", "b"):
            if depth_stage == "foundation_stereo_depth":
                d, other, rng_ = arr[f"fs_{end}"], arr[f"sd_{end}"], arr[f"fs_range_{end}"]
            elif depth_stage == "sensor_depth":
                d, other, rng_ = arr[f"sd_{end}"], arr[f"fs_{end}"], arr[f"fs_range_{end}"]
            else:
                d, other, rng_ = arr[f"x_{end}"], arr[f"sd_{end}"], arr[f"x_range_{end}"]
            d, other = d.astype(float), other.astype(float)
            good = (d > 0) & (rng_ <= cfg.max_range_rel * np.maximum(d, 1e-6)) & (other > 0) \
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
    out["config"] = asdict(cfg) | {"static_flags": str(flag_dir), "depth_stage": depth_stage}
    out["stats"] = stats
    return out
