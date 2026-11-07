"""Dense cross-camera static correspondences with RoMa v2 for the board-free rig calibration (``rig_dw``).

Runs in an environment with torch + einops + h5py + OpenCV (rwprep image: /opt/envs/trackeverything) and the pinned
RoMa v2 source on ``PYTHONPATH`` (github.com/Parskatt/romav2 @ 95c9968, weights romav2.0.1.pt from the torch-hub cache,
sha256 1557dec0...). It does not import ``s4d``: it reads the processed episode directly (normalized RGB frames,
foundation_stereo_depth, sensor_depth, sam3_masks) so the RoMa environment needs no repository dependencies.

    python romav2_match_worker.py --task-root R [--episodes E ...] --cameras c0 c1 c2 --out OUTDIR \
        [--times 6] [--num 5000] [--setting precise] [--seed 0]

For K time points (evenly spaced over the reference camera's frames) and every camera pair (a < b), the frames of a
and b nearest in time (|dt| <= 20 ms) are matched (bidirectional). Every sampled match is stored with the per-end
evidence needed downstream (no filtering decisions beyond NaN removal are taken here except what is recorded as
flags): pixel coordinates (continuous convention: pixel (i, j) centre at (j + .5, i + .5)), RoMa overlap, cycle error
(px), SAM 3 class at both ends + distance to the nearest robot/object pixel, FS and sensor depth (0 = invalid) and the
local 5x5 FS depth range, MAGSAC++ fundamental inlier flag (computed on the matches whose ends are static)."""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import h5py
import numpy as np


def safe(stream):
    return stream.replace("/", "__")


def load_inputs(root: Path, ep: str, cam: str, idx):
    d = root / "episodes" / ep
    s = safe(cam)
    idx = np.asarray(idx)
    with h5py.File(d / "normalized" / "frames" / f"{s}.h5", "r") as f:
        rgb = np.stack([f["frames"][int(i)] for i in idx])
    out = {"rgb": rgb}
    for stage in ("foundation_stereo_depth", "sensor_depth"):
        p = d / "derived" / stage / f"{s}.h5"
        if p.exists():
            with h5py.File(p, "r") as f:
                dep = np.stack([f["depth"][int(i)] for i in idx]).astype(np.float32)
                val = np.stack([f["valid"][int(i)] for i in idx]).astype(bool)
                out[stage] = np.where(val, dep, 0.0).astype(np.float32)
                out[stage + "_K"] = np.asarray(f.attrs["K"], float)
    with h5py.File(d / "derived" / "sam3_masks" / f"{s}.h5", "r") as f:
        names = json.loads(f.attrs.get("class_names", '["background", "robot", "object", "fixture"]'))
        out["sem"] = np.stack([f["semantic"][int(i)] for i in idx])
        out["class_names"] = names
    return out


def dist_to(mask: np.ndarray) -> np.ndarray:
    from scipy.ndimage import distance_transform_edt
    if not mask.any():
        return np.full(mask.shape, 1e4, np.float32)
    return distance_transform_edt(~mask).astype(np.float32)


def local_range(depth: np.ndarray, r: int = 2) -> np.ndarray:
    from scipy.ndimage import maximum_filter, minimum_filter
    d = np.where(depth > 0, depth, np.nan)
    big = np.where(np.isnan(d), np.inf, d)
    small = np.where(np.isnan(d), -np.inf, d)
    rng = maximum_filter(np.where(np.isnan(d), -np.inf, d), size=2 * r + 1) - minimum_filter(big, size=2 * r + 1)
    del small
    return np.where(np.isfinite(rng), rng, np.inf).astype(np.float32)


def sample_at(img: np.ndarray, uv: np.ndarray) -> np.ndarray:
    """nearest-pixel lookup at continuous uv."""
    h, w = img.shape[:2]
    j = np.clip(np.floor(uv[:, 0]).astype(int), 0, w - 1)
    i = np.clip(np.floor(uv[:, 1]).astype(int), 0, h - 1)
    return img[i, j]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task-root", required=True, help="processed/<release>/<task>: episodes <task-root>/<id>/")
    ap.add_argument("--episodes", nargs="*", default=None, help="default: every episode under the task root")
    ap.add_argument("--cameras", nargs="+", required=True)
    ap.add_argument("--out", required=True, help="output directory: <out>/<episode>.npz")
    ap.add_argument("--times", type=int, default=6)
    ap.add_argument("--num", type=int, default=5000)
    ap.add_argument("--setting", default="precise")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-dt-ms", type=float, default=20.0)
    ap.add_argument("--magsac-px", type=float, default=1.0)
    a = ap.parse_args()

    import torch
    torch.set_float32_matmul_precision("highest")
    from romav2 import RoMaV2

    model = RoMaV2()
    model.apply_setting(a.setting)
    task_root = Path(a.task_root)
    eps = a.episodes or sorted(p.name for p in task_root.iterdir() if p.is_dir())
    for ep in eps:
        out = Path(a.out) / f"{ep}.npz"
        if out.exists():
            continue
        out.parent.mkdir(parents=True, exist_ok=True)
        torch.manual_seed(a.seed)
        run_episode(model, a, task_root / ep, ep, str(out))
        print(json.dumps({"max_mem_gib": round(torch.cuda.max_memory_allocated() / 2**30, 2)
                          if torch.cuda.is_available() else None}), flush=True)


def run_episode(model, a, root, episode_id, out_path):
    import cv2
    import torch
    from romav2.geometry import bhwc_grid_sample

    t0 = time.time()
    root = Path(root)
    with np.load(root / "episodes" / episode_id / "normalized" / "arrays.npz") as z:
        stamps = {c: z[f"timestamps/{c}"].astype(np.int64) for c in a.cameras}
    ref = a.cameras[0]
    n0 = len(stamps[ref])
    t_ref = stamps[ref][np.linspace(0, n0 - 1, a.times + 2).round().astype(int)[1:-1]]
    rows = {k: [] for k in ("pair", "time", "frame_a", "frame_b", "dt_ns", "uv_a", "uv_b", "overlap", "cycle_px",
                            "cls_a", "cls_b", "dyn_dist_a", "dyn_dist_b", "fs_a", "fs_b", "sd_a", "sd_b",
                            "fs_range_a", "fs_range_b", "magsac_inlier")}
    frames_used = {}
    pairs = [(i, j) for i in range(len(a.cameras)) for j in range(i + 1, len(a.cameras))]
    cache = {}

    def get(cam, i):
        key = (cam, int(i))
        if key not in cache:
            inp = load_inputs(root, episode_id, cam, [i])
            names = inp["class_names"]
            sem = inp["sem"][0]
            dyn = np.isin(sem, [names.index(c) for c in ("robot", "object") if c in names])
            fs = inp.get("foundation_stereo_depth", np.zeros((1, *sem.shape), np.float32))[0]
            sd = inp.get("sensor_depth", np.zeros((1, *sem.shape), np.float32))[0]
            cache[key] = {"rgb": inp["rgb"][0], "sem": sem, "dyn_dist": dist_to(dyn), "fs": fs, "sd": sd,
                          "fs_range": local_range(fs), "names": names,
                          "K": inp.get("foundation_stereo_depth_K")}
        return cache[key]

    for k, t in enumerate(t_ref):
        idx = {c: int(np.argmin(np.abs(stamps[c] - t))) for c in a.cameras}
        frames_used[int(k)] = {c: idx[c] for c in a.cameras}
        for (pa, pb) in pairs:
            ca, cb = a.cameras[pa], a.cameras[pb]
            dt = int(stamps[cb][idx[cb]] - stamps[ca][idx[ca]])
            if abs(dt) > a.max_dt_ms * 1e6:
                continue
            A, B = get(ca, idx[ca]), get(cb, idx[cb])
            H, W = A["rgb"].shape[:2]
            preds = model.match(A["rgb"], B["rgb"])
            m, ov, _, _ = model.sample(preds, a.num)
            # cycle consistency: warp_AB at uv_a vs uv_b and warp_BA at uv_b vs uv_a
            m_t = m[None, None]  # (1,1,N,4)
            fwd = bhwc_grid_sample(preds["warp_AB"], m_t[..., :2], mode="bilinear", align_corners=False)[0, 0]
            if preds.get("warp_BA") is None:
                raise RuntimeError("cycle consistency needs a bidirectional RoMa setting (e.g. 'precise')")
            bwd = bhwc_grid_sample(preds["warp_BA"], m_t[..., 2:], mode="bilinear", align_corners=False)[0, 0]
            uv_a, uv_b = model.to_pixel_coordinates(m, H, W, H, W)
            fa, fb = model.to_pixel_coordinates(torch.cat([bwd, fwd], -1), H, W, H, W)
            cyc = torch.maximum((fa - uv_a).norm(dim=-1), (fb - uv_b).norm(dim=-1))
            uv_a, uv_b = uv_a.double().cpu().numpy(), uv_b.double().cpu().numpy()
            ov, cyc = ov.float().cpu().numpy(), cyc.float().cpu().numpy()
            ok = np.isfinite(uv_a).all(1) & np.isfinite(uv_b).all(1)
            uv_a, uv_b, ov, cyc = uv_a[ok], uv_b[ok], ov[ok], cyc[ok]
            n = len(uv_a)
            cls_a, cls_b = sample_at(A["sem"], uv_a), sample_at(B["sem"], uv_b)
            dd_a, dd_b = sample_at(A["dyn_dist"], uv_a), sample_at(B["dyn_dist"], uv_b)
            static = (dd_a > 9) & (dd_b > 9)
            inl = np.zeros(n, bool)
            if static.sum() >= 30:
                F, mask = cv2.findFundamentalMat(uv_a[static], uv_b[static], cv2.USAC_MAGSAC, a.magsac_px, 0.99999,
                                                 10000)
                if F is not None and mask is not None:
                    inl[np.flatnonzero(static)] = mask.ravel().astype(bool)
            vals = {"pair": np.full(n, pa * 10 + pb, np.int16), "time": np.full(n, k, np.int16),
                    "frame_a": np.full(n, idx[ca], np.int32), "frame_b": np.full(n, idx[cb], np.int32),
                    "dt_ns": np.full(n, dt, np.int64), "uv_a": uv_a.astype(np.float32), "uv_b": uv_b.astype(np.float32),
                    "overlap": ov, "cycle_px": cyc, "cls_a": cls_a.astype(np.uint8), "cls_b": cls_b.astype(np.uint8),
                    "dyn_dist_a": dd_a, "dyn_dist_b": dd_b, "fs_a": sample_at(A["fs"], uv_a),
                    "fs_b": sample_at(B["fs"], uv_b), "sd_a": sample_at(A["sd"], uv_a), "sd_b": sample_at(B["sd"], uv_b),
                    "fs_range_a": sample_at(A["fs_range"], uv_a), "fs_range_b": sample_at(B["fs_range"], uv_b),
                    "magsac_inlier": inl}
            for key, v in vals.items():
                rows[key].append(v)
        for key in list(cache):  # frames of this time point are not reused
            del cache[key]
    out = {k: np.concatenate(v) if v else np.zeros(0) for k, v in rows.items()}
    meta = {"episode_id": episode_id, "cameras": a.cameras, "times": a.times, "num": a.num, "setting": a.setting,
            "seed": a.seed, "frames": frames_used, "magsac_px": a.magsac_px, "max_dt_ms": a.max_dt_ms,
            "seconds": round(time.time() - t0, 1), "torch": torch.__version__, "cv2": cv2.__version__,
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}
    tmp = out_path + ".tmp.npz"
    np.savez_compressed(tmp, meta=json.dumps(meta), **out)
    Path(tmp).replace(out_path)
    print(json.dumps({k: meta[k] for k in ("episode_id", "seconds")} | {"matches": int(len(out["pair"])),
                     "inliers": int(out["magsac_inlier"].sum())}), flush=True)


if __name__ == "__main__":
    main()
