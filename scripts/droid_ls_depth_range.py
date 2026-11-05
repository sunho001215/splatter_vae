"""Released PointWorld-DROID depth and tracks against candidate far planes (read-only; justifies ``render.far``).

For every manifest episode (``scripts/droid_ls_build_manifest.py`` output) and the extracted PointWorld tree, measures:

* depth: the uint16-millimetre histogram of every stored frame per camera kind (ext / wrist), and of the external
  frames the training windows read, weighted by how many windows read each frame (the window rule of
  ``DroidLargeScaleDataset``: ``(i, i+s, i+2s)`` inside one clip, both external cameras valid). 0 and the saturation
  code 65535 are invalid (``depth_mm_to_m``). Fractions of valid depth in depth bands, percentiles, the largest valid
  value, saturation counts, and the share of camera-episodes with depth in (3, 4] m.
* temporal stability (noise proxy, external cameras are static): |d(t) - d(t+1)| / d(t) between consecutive canonical
  frames (2 raw steps, about 0.13 s) on pixels valid in both, per depth bin of d(t). Scene motion also enters; beyond
  the robot's reach (about 2 m from the cameras) it is mostly static background.
* cross-camera consistency per source-depth bin: valid ext1 pixels lifted with the training K and calibration, projected
  into ext2 (and back), compared with the other camera's depth at the nearest pixel; |z / d - 1| over all overlapping
  points and over the one-sided selection z <= d + 2 cm (occluded points removed, as the dense validation). Both the
  training calibration (``w2c``, DROID-3D) and PointWorld's (``w2c_pointworld``).
* tracks: camera z of visible and depth-valid PointWorld track points (``scene_visibility & scene_depth_valid_mask``)
  in their own camera (PointWorld calibration), and the largest value of each clip's ``initial_depth``.

    python scripts/droid_ls_depth_range.py --manifest manifest.jsonl --pointworld-root <extracted droid/> --out out.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict
from multiprocessing import get_context
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from s4d.data.droid.largescale.pointworld_source import DEPTH_SATURATED_MM, clip_bounds  # noqa: E402

EXT_ROLES = ("ext1", "ext2")
BIN_EDGES_M = (0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 6.0, 8.0, np.inf)
BANDS_M = ((0.0, 3.0), (3.0, 4.0), (4.0, np.inf))  # (lo, hi]: valid depth at or below 3 m, in (3, 4] m, above 4 m
THRESHOLDS_M = (1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 5.0, 6.0)
REL_STEP, REL_BINS = 0.0005, 1000  # |relative error| histograms: 0.05 % bins up to 50 %, then one overflow bin
TRACK_STEP_M, TRACK_BINS = 0.01, 1000  # track camera z: 1 cm bins up to 10 m, then one overflow bin
SELECT_TOL_M = 0.02


def _bin(depth_m: np.ndarray) -> np.ndarray:
    return np.searchsorted(np.asarray(BIN_EDGES_M), depth_m, side="right") - 1


def _rel_index(rel: np.ndarray) -> np.ndarray:
    return np.minimum((np.abs(rel) / REL_STEP).astype(np.int64), REL_BINS)


def _add(table: np.ndarray, rows: np.ndarray, cols: np.ndarray, weights: np.ndarray | None = None) -> None:
    """table[rows, cols] += 1 (or += weights) for repeated index pairs, in place."""
    flat = np.bincount(rows * table.shape[1] + cols, weights=weights, minlength=table.size)
    table += flat.reshape(table.shape).astype(table.dtype)


def window_multiplicity(row: dict, stride: int) -> np.ndarray:
    """How many training windows read each canonical frame (the dataset's window rule, split ignored)."""
    n = len(row["timeline"]["canonical_timestamps"])
    valid = np.ones(n, bool)
    for role in EXT_ROLES:
        valid &= np.asarray(row["frames"]["valid"][role], bool)
    count = np.zeros(n, np.int64)
    for clip in row["clips"]:
        start, end = clip_bounds(clip)
        for i in range(start, end - 2 * stride):
            frames = [i, i + stride, i + 2 * stride]
            if valid[frames].all():
                count[frames] += 1
    return count


def lift(depth_m: np.ndarray, K: np.ndarray, w2c: np.ndarray, keep: np.ndarray) -> np.ndarray:
    """World points (N,3) of the pixels ``keep`` (continuous pixel centres, OpenCV, w2c extrinsics)."""
    i, j = np.nonzero(keep)
    z = depth_m[i, j]
    cam = np.stack([(j + 0.5 - K[0, 2]) / K[0, 0] * z, (i + 0.5 - K[1, 2]) / K[1, 1] * z, z], 1)
    c2w = np.linalg.inv(w2c)
    return cam @ c2w[:3, :3].T + c2w[:3, 3]


def cross_camera(src_m, src_K, src_w2c, tgt_m, tgt_K, tgt_w2c):
    """Source depth, projected z and target depth of the source pixels landing on valid target depth."""
    keep = src_m > 0
    world = lift(src_m, src_K, src_w2c, keep)
    cam = world @ tgt_w2c[:3, :3].T + tgt_w2c[:3, 3]
    z = cam[:, 2]
    with np.errstate(divide="ignore", invalid="ignore"):
        u, v = tgt_K[0, 0] * cam[:, 0] / z + tgt_K[0, 2], tgt_K[1, 1] * cam[:, 1] / z + tgt_K[1, 2]
    H, W = tgt_m.shape
    inside = (z > 0) & np.isfinite(u) & np.isfinite(v) & (u >= 0) & (u < W) & (v >= 0) & (v < H)
    d = np.zeros_like(z)
    d[inside] = tgt_m[v[inside].astype(np.int64), u[inside].astype(np.int64)]
    hit = inside & (d > 0)
    return src_m[keep][hit], z[hit], d[hit]


def _metres(raw: np.ndarray) -> np.ndarray:
    out = raw.astype(np.float64) / 1000.0
    out[raw == DEPTH_SATURATED_MM] = 0.0
    return out


def episode_stats(row: dict, root: str | Path, stride: int = 3, cross_every: int = 4) -> dict:
    """Histograms of one manifest episode (module docstring); all arrays are int64 counts or float sums."""
    import h5py  # noqa: PLC0415

    root = Path(root)
    nb = len(BIN_EDGES_M) - 1
    out = {
        "hist_all": {"ext": np.zeros(65536, np.int64), "wrist": np.zeros(65536, np.int64)},
        "hist_windows": np.zeros(65536, np.int64),
        "temporal": np.zeros((nb, REL_BINS + 1), np.int64),
        "cross": {c: {"all": np.zeros((nb, REL_BINS + 1), np.int64), "selected": np.zeros((nb, REL_BINS + 1), np.int64),
                      "signed_sum": np.zeros(nb), "signed_n": np.zeros(nb, np.int64)} for c in ("droid3d", "pointworld")},
        "track_z": np.zeros(TRACK_BINS + 1, np.int64),
        "track_valid_above_4_01m": 0,
        "track_points": 0,
        "initial_depth_max_mm": 0,
        "cameras": [],
    }
    mult = window_multiplicity(row, stride)
    canon_ext = {}
    with h5py.File(root / row["provenance"]["pointworld"]["depth"], "r") as f:
        for role, group in row["depth_groups"].items():
            raw = f[group]["depth"][:]
            kind = "wrist" if role == "wrist" else "ext"
            hist = np.bincount(raw.ravel(), minlength=65536)
            out["hist_all"][kind] += hist
            valid = hist[1:DEPTH_SATURATED_MM].sum()
            mm = np.arange(65536)
            out["cameras"].append({
                "episode": row["episode"], "role": role, "frames": int(raw.shape[0]), "pixels": int(raw.size),
                "valid": int(valid), "saturated": int(hist[DEPTH_SATURATED_MM]),
                "valid_3_4": int(hist[(mm > 3000) & (mm <= 4000)].sum()),
                "valid_above_4": int(hist[4001:DEPTH_SATURATED_MM].sum()),
                "max_valid_mm": int(np.flatnonzero(hist[:DEPTH_SATURATED_MM])[-1]) if valid else 0,
            })
            if kind != "ext":
                continue
            index = np.asarray(row["frames"]["depth_index"][role], np.int64)
            for c in np.flatnonzero(mult):
                out["hist_windows"] += mult[c] * np.bincount(raw[index[c]].ravel(), minlength=65536)
            ok = np.asarray(row["frames"]["valid"][role], bool)
            for c in range(len(index) - 1):
                if not (ok[c] and ok[c + 1]):
                    continue
                a, b = _metres(raw[index[c]]), _metres(raw[index[c + 1]])
                both = (a > 0) & (b > 0)
                _add(out["temporal"], _bin(a[both]), _rel_index(b[both] / a[both] - 1.0))
            canon_ext[role] = (raw, index, ok)
    cams = {r: (np.asarray(row["cameras"][r]["K"], np.float64), np.asarray(row["cameras"][r]["w2c"], np.float64),
                np.asarray(row["cameras"][r]["w2c_pointworld"], np.float64)) for r in EXT_ROLES}
    if all(r in canon_ext for r in EXT_ROLES):
        n = len(canon_ext["ext1"][1])
        for c in range(0, n, cross_every):
            if not all(canon_ext[r][2][c] for r in EXT_ROLES):
                continue
            depth = {r: _metres(canon_ext[r][0][canon_ext[r][1][c]]) for r in EXT_ROLES}
            for src, tgt in (("ext1", "ext2"), ("ext2", "ext1")):
                for calib, k in (("droid3d", 1), ("pointworld", 2)):
                    s, z, d = cross_camera(depth[src], cams[src][0], cams[src][k], depth[tgt], cams[tgt][0], cams[tgt][k])
                    rel, b = z / d - 1.0, _bin(s)
                    acc = out["cross"][calib]
                    _add(acc["all"], b, _rel_index(rel))
                    sel = z <= d + SELECT_TOL_M
                    _add(acc["selected"], b[sel], _rel_index(rel[sel]))
                    nb = len(acc["signed_n"])
                    acc["signed_sum"] += np.bincount(b[sel], weights=rel[sel], minlength=nb)
                    acc["signed_n"] += np.bincount(b[sel], minlength=nb)
    with h5py.File(root / row["provenance"]["pointworld"]["flows"], "r") as f:
        for clip in row["clips"]:
            for role in EXT_ROLES:
                g = f[clip][f"camera_{row['cameras'][role]['serial']}_ext"]
                w2c = np.asarray(g["extrinsic"], np.float64)
                mask = g["scene_visibility"][:] & g["scene_depth_valid_mask"][:]
                pts = g["scene_flows"][:].astype(np.float64)[mask]
                z = pts @ w2c[2, :3] + w2c[2, 3]
                out["track_points"] += int(z.size)
                out["track_valid_above_4_01m"] += int((z > 4.01).sum())
                zi = np.minimum((np.maximum(z, 0) / TRACK_STEP_M).astype(np.int64), TRACK_BINS)
                out["track_z"] += np.bincount(zi, minlength=TRACK_BINS + 1)
                out["initial_depth_max_mm"] = max(out["initial_depth_max_mm"], int(g["initial_depth"][:].max()))
    return out


def _merge(total: dict | None, part: dict) -> dict:
    if total is None:
        return part
    for key, value in part.items():
        if isinstance(value, dict):
            total[key] = _merge(total[key], value)
        elif key == "cameras":
            total[key].extend(value)
        elif key == "initial_depth_max_mm":
            total[key] = max(total[key], value)
        else:
            total[key] = total[key] + value
    return total


def depth_summary(hist: np.ndarray) -> dict:
    """Valid-depth statistics of a uint16-millimetre histogram (0 and 65535 invalid)."""
    mm = np.arange(65536)
    valid = hist.copy()
    valid[0] = valid[DEPTH_SATURATED_MM] = 0
    n = int(valid.sum())
    cdf = np.cumsum(valid) / max(n, 1)
    band = {f"({lo:g},{hi:g}]": float(valid[(mm > 1000 * lo) & (mm <= 1000 * hi)].sum() / max(n, 1))
            for lo, hi in BANDS_M}
    return {
        "pixels": int(hist.sum()), "valid": n, "valid_fraction_of_pixels": n / max(int(hist.sum()), 1),
        "saturated_65535": int(hist[DEPTH_SATURATED_MM]),
        "max_valid_m": float(np.flatnonzero(valid)[-1] / 1000.0) if n else None,
        "percentile_m": {str(q): float(np.searchsorted(cdf, q / 100.0) / 1000.0) for q in (1, 5, 50, 95, 99, 99.5, 99.9)},
        "fraction_valid_above_m": {str(t): float(valid[mm > 1000 * t].sum() / max(n, 1)) for t in THRESHOLDS_M},
        "fraction_valid_in_band_m": band,
        "fraction_of_all_pixels_in_(3,4]": float(valid[(mm > 3000) & (mm <= 4000)].sum() / max(int(hist.sum()), 1)),
        "exactly_4000_mm": int(hist[4000]), "exactly_3000_mm": int(hist[3000]),
    }


def _quantiles(hist: np.ndarray, qs=(50, 90)) -> dict:
    n = int(hist.sum())
    if n == 0:
        return {"n": 0}
    cdf = np.cumsum(hist) / n
    out = {"n": n}
    for q in qs:
        k = int(np.searchsorted(cdf, q / 100.0))
        out[f"p{q}"] = None if k >= REL_BINS else round((k + 0.5) * REL_STEP, 6)  # None: in the overflow bin (> 50 %)
    out["within_3pct"] = float(hist[: int(0.03 / REL_STEP)].sum() / n)
    out["within_10pct"] = float(hist[: int(0.10 / REL_STEP)].sum() / n)
    return out


def _per_bin(table: np.ndarray) -> dict:
    names = [f"[{BIN_EDGES_M[i]:g},{BIN_EDGES_M[i + 1]:g})" for i in range(len(BIN_EDGES_M) - 1)]
    return {name: _quantiles(table[i]) for i, name in enumerate(names)}


def summarize(total: dict, episodes: int) -> dict:
    cams = total["cameras"]
    ext = [c for c in cams if c["role"] != "wrist" and c["valid"]]
    frac34 = np.array([c["valid_3_4"] / c["valid"] for c in ext])
    frac4 = np.array([c["valid_above_4"] / c["valid"] for c in ext])
    sessions = defaultdict(lambda: np.zeros(3, np.int64))
    for c in ext:
        sessions[c["episode"].split("+")[2][:10]] += (c["valid"], c["valid_3_4"], c["valid_above_4"])
    names = [f"[{BIN_EDGES_M[i]:g},{BIN_EDGES_M[i + 1]:g})" for i in range(len(BIN_EDGES_M) - 1)]
    cross = {}
    for calib, acc in total["cross"].items():
        cross[calib] = {
            "all": _per_bin(acc["all"]), "selected": _per_bin(acc["selected"]),
            "selected_mean_signed": {names[i]: (float(acc["signed_sum"][i] / acc["signed_n"][i]) if acc["signed_n"][i]
                                                else None) for i in range(len(names))},
        }
    tz = total["track_z"]
    tn = int(tz.sum())
    tcdf = np.cumsum(tz) / max(tn, 1)
    return {
        "episodes": episodes,
        "depth_ext_all_frames": depth_summary(total["hist_all"]["ext"]),
        "depth_ext_training_windows_weighted": depth_summary(total["hist_windows"]),
        "depth_wrist_all_frames": depth_summary(total["hist_all"]["wrist"]),
        "ext_camera_episodes": {
            "n": len(ext),
            "fraction_valid_in_(3,4]_percentiles": {str(q): float(np.percentile(frac34, q)) for q in (50, 90, 99, 100)},
            "share_with_more_than_1pct_in_(3,4]": float((frac34 > 0.01).mean()),
            "share_with_more_than_5pct_in_(3,4]": float((frac34 > 0.05).mean()),
            "fraction_valid_above_4_percentiles": {str(q): float(np.percentile(frac4, q)) for q in (50, 90, 99, 100)},
            "max_valid_m_percentiles": {str(q): float(np.percentile([c["max_valid_mm"] / 1000 for c in ext], q))
                                        for q in (50, 90, 100)},
            "saturated_camera_episodes": int(sum(c["saturated"] > 0 for c in ext)),
            "by_date": {k: {"valid": int(v[0]), "fraction_(3,4]": float(v[1] / v[0]),
                            "fraction_above_4": float(v[2] / v[0])} for k, v in sorted(sessions.items())},
        },
        "temporal_relative_change_by_depth_bin": _per_bin(total["temporal"]),
        "cross_camera_relative_error_by_source_depth_bin": cross,
        "tracks": {
            "visible_depth_valid_points": tn,
            "camera_z_percentile_m": {str(q): float(np.searchsorted(tcdf, q / 100.0) * TRACK_STEP_M)
                                      for q in (50, 95, 99, 99.9, 100)},
            "fraction_z_above_m": {str(t): float(tz[int(round(t / TRACK_STEP_M)):].sum() / max(tn, 1))
                                   for t in (2.0, 3.0, 3.5, 4.0)},
            "points_z_above_4.01m": int(total["track_valid_above_4_01m"]),
            "initial_depth_max_m": total["initial_depth_max_mm"] / 1000.0,
        },
        "settings": {"bin_edges_m": [str(e) for e in BIN_EDGES_M], "relative_bin": REL_STEP,
                     "selection_tolerance_m": SELECT_TOL_M},
    }


def _job(args):
    line, root, stride, cross_every = args
    return episode_stats(json.loads(line), root, stride, cross_every)


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--pointworld-root", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--stride", type=int, default=3, help="data.stride of the profile (canonical steps)")
    ap.add_argument("--cross-every", type=int, default=4, help="every n-th canonical frame for the cross-camera check")
    ap.add_argument("--workers", type=int, default=min(8, os.cpu_count() or 1))
    ap.add_argument("--limit", type=int, default=0, help="first n episodes only (0: all)")
    args = ap.parse_args(argv)
    lines = [ln for ln in Path(args.manifest).read_text().splitlines() if ln.strip()]
    if args.limit:
        lines = lines[: args.limit]
    jobs = [(ln, args.pointworld_root, args.stride, args.cross_every) for ln in lines]
    total = None
    if args.workers > 1:
        with get_context("spawn").Pool(args.workers) as pool:
            for part in pool.imap_unordered(_job, jobs):
                total = _merge(total, part)
    else:
        for job in jobs:
            total = _merge(total, _job(job))
    result = summarize(total, len(lines))
    result["sources"] = {"manifest": str(Path(args.manifest).resolve()), "pointworld_root": str(args.pointworld_root)}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(result, indent=1))
    print(json.dumps(result, indent=1))


if __name__ == "__main__":
    main()
