"""SAM-free static-scene flags for the RoMa v2 cross-camera matches (``rig_loma``; replaces the SAM robot/object
distance and the MAGSAC flag computed on SAM-static matches in ``matches-v1``).

Environment: rwprep /opt/envs/trackeverything (numpy, scipy, h5py, OpenCV); no repository imports.

Per episode and camera: a temporal median image (grey) and a temporal median sensor depth over ``--median-frames``
frames spread over the episode. At every RoMa time point a pixel is DYNAMIC when its (Gaussian-smoothed) grey value
differs from the median by more than ``--grey-thr`` levels or its sensor depth (valid in both) differs by more than
max(``--depth-abs`` m, ``--depth-rel`` x depth); the dynamic mask is opened (3x3) to drop sensor noise. Per match end:
distance (px) to the nearest dynamic pixel. MAGSAC++ fundamental matrix per (time point, camera pair) on the matches
whose both ends are >= ``--margin`` px from dynamic pixels (threshold 1 px, confidence 0.99999, 10k iterations, as
rig_dw). Pose-free, mask-free and depth-source-free (only the native sensor depth enters, identically for every depth
stage of the calibration).

    python static_flags_worker.py --task-root R --matches DIR --out DIR [--episodes ...]
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np


def safe(stream):
    return stream.replace("/", "__")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--task-root", required=True)
    ap.add_argument("--matches", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--episodes", nargs="*", default=None)
    ap.add_argument("--median-frames", type=int, default=15)
    ap.add_argument("--grey-thr", type=float, default=20.0)
    ap.add_argument("--depth-abs", type=float, default=0.03)
    ap.add_argument("--depth-rel", type=float, default=0.03)
    ap.add_argument("--margin", type=float, default=9.0)
    ap.add_argument("--magsac-px", type=float, default=1.0)
    a = ap.parse_args(argv)
    import cv2
    import h5py
    from scipy import ndimage

    files = sorted(Path(a.matches).glob("*.npz"))
    if a.episodes:
        files = [f for f in files if f.stem in set(a.episodes)]
    for fp in files:
        ep = fp.stem
        out = Path(a.out) / f"{ep}.npz"
        if out.exists():
            continue
        t0 = time.time()
        with np.load(fp) as z:
            meta = json.loads(str(z["meta"]))
            m = {k: z[k] for k in ("pair", "time", "frame_a", "frame_b", "uv_a", "uv_b", "cycle_px", "overlap",
                                   "dyn_dist_a", "dyn_dist_b", "magsac_inlier")}
        cams = meta["cameras"]
        root = Path(a.task_root) / ep / "episodes" / ep
        dist = {}  # (cam, frame) -> distance map
        stats = {}
        for cam in cams:
            s = safe(cam)
            with h5py.File(root / "normalized" / "frames" / f"{s}.h5", "r") as f:
                n = f["frames"].shape[0]
                ref = np.linspace(0, n - 1, a.median_frames).round().astype(int)
                grey = np.stack([cv2.cvtColor(f["frames"][int(i)], cv2.COLOR_RGB2GRAY) for i in ref]).astype(np.float32)
                need = sorted({int(v[cam]) for v in meta["frames"].values()})
                g_t = {i: cv2.cvtColor(f["frames"][i], cv2.COLOR_RGB2GRAY).astype(np.float32) for i in need}
            dpath = root / "derived" / "sensor_depth" / f"{s}.h5"
            with h5py.File(dpath, "r") as f:
                dep = np.stack([np.where(f["valid"][int(i)], f["depth"][int(i)], np.nan) for i in ref]).astype(np.float32)
                d_t = {i: np.where(f["valid"][i], f["depth"][i], np.nan).astype(np.float32) for i in need}
            med_g = ndimage.gaussian_filter(np.median(grey, 0), 1.0)
            with np.errstate(all="ignore"):
                med_d = np.nanmedian(dep, 0)
            for i in need:
                gd = np.abs(ndimage.gaussian_filter(g_t[i], 1.0) - med_g) > a.grey_thr
                with np.errstate(invalid="ignore"):
                    dd = np.abs(d_t[i] - med_d) > np.maximum(a.depth_abs, a.depth_rel * med_d)
                dyn = ndimage.binary_opening(gd | (dd & np.isfinite(d_t[i]) & np.isfinite(med_d)), iterations=1)
                dist[(cam, i)] = ndimage.distance_transform_edt(~dyn).astype(np.float32) if dyn.any() else \
                    np.full(dyn.shape, 1e4, np.float32)
                stats[f"{cam}/{i}"] = float(dyn.mean())
        N = len(m["pair"])
        dda, ddb = np.zeros(N, np.float32), np.zeros(N, np.float32)
        inl = np.zeros(N, bool)
        for key in np.unique(m["time"].astype(int) * 100 + m["pair"].astype(int)):
            sel = np.flatnonzero(m["time"].astype(int) * 100 + m["pair"].astype(int) == key)
            pa, pb = divmod(int(key % 100), 10)
            for end, cam_i, uv, fr, dst in (("a", pa, m["uv_a"], m["frame_a"], dda), ("b", pb, m["uv_b"], m["frame_b"], ddb)):
                D = dist[(cams[cam_i], int(fr[sel[0]]))]
                h, w = D.shape
                j = np.clip(np.floor(uv[sel, 0]).astype(int), 0, w - 1)
                i = np.clip(np.floor(uv[sel, 1]).astype(int), 0, h - 1)
                dst[sel] = D[i, j]
            st = sel[(dda[sel] > a.margin) & (ddb[sel] > a.margin)]
            if len(st) >= 30:
                F, mk = cv2.findFundamentalMat(m["uv_a"][st].astype(np.float64), m["uv_b"][st].astype(np.float64),
                                               cv2.USAC_MAGSAC, a.magsac_px, 0.99999, 10000)
                if F is not None and mk is not None:
                    inl[st] = mk.ravel().astype(bool)
        sam_static = (m["dyn_dist_a"] > 9) & (m["dyn_dist_b"] > 9)
        sf_static = (dda > a.margin) & (ddb > a.margin)
        diag = {"n": int(N), "samfree_static": int(sf_static.sum()), "sam_static": int(sam_static.sum()),
                "samfree_static_and_sam_dynamic": int((sf_static & ~sam_static).sum()),
                "sam_static_and_samfree_dynamic": int((~sf_static & sam_static).sum()),
                "magsac_samfree": int(inl.sum()), "magsac_sam": int(m["magsac_inlier"].sum()),
                "both_magsac": int((inl & m["magsac_inlier"].astype(bool)).sum())}
        doc = {"episode_id": ep, "config": vars(a), "dynamic_fraction": stats, "diagnostic_vs_sam": diag,
               "seconds": round(time.time() - t0, 1), "cv2": cv2.__version__}
        out.parent.mkdir(parents=True, exist_ok=True)
        tmp = str(out) + ".tmp.npz"
        np.savez_compressed(tmp, meta=json.dumps(doc), dyn_dist_t_a=dda, dyn_dist_t_b=ddb, magsac_sf=inl)
        Path(tmp).replace(out)
        print(json.dumps({"episode": ep, "seconds": doc["seconds"], **diag}), flush=True)


if __name__ == "__main__":
    main()
