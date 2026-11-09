"""Render-to-image robot 2D-3D correspondences with LoMa-G for the rig calibration (``rig_loma``; DepthWorld App. B.1).

Environment: rwprep image, /opt/envs/trackeverything (torch, torchvision, einops, PIL, OpenCV, h5py) + the pinned LoMa
source on ``PYTHONPATH`` (github.com/davnords/LoMa @ 415caed; weights loma_G.pth, dedode_descriptor_G.pth,
dinov2_vitl14_pretrain.pth, dad.pth from ``$TORCH_HOME/hub/checkpoints``) + this repository on ``PYTHONPATH`` for the
renderer only (``s4d.realworld.rig_loma_render``: numpy + torch).

Per frame of the SAM-free frame files (``rig_loma_data``) and camera pose T_cam_base of ``--state``:
1. render the posed robot (visual meshes, non-PBR lit shading) - frames whose rendered robot covers fewer than
   ``--min-render-px`` pixels are skipped; tight window = rendered-arm bounding box + margin;
2. crop the real RGB frame to the same window; LoMa-G (template = render, query = real; the official image-path
   preprocessing: DaD on the aspect-preserving 1024 resize, DeDoDe-G descriptors on the 784 x 784 resize, mutual
   nearest neighbour + ``filter_threshold``);
3. every match -> (robot surface point X in its link frame, link id, base-frame point at the camera timestamp + FK
   surface velocity, real pixel u, render pixel, LoMa score, render depth-discontinuity range);
4. geometric filters only (no masks): per-frame MAGSAC++ PnP inliers and per-(episode, camera) pooled MAGSAC++ PnP
   inliers (the camera is fixed, all frames share one pose).

``--synthetic ROT_DEG,TRANS_MM,SEED``: accuracy test with known geometry. The "real" image is the robot rendered at
the state pose with a DIFFERENT appearance (other light directions, albedo jitter, blur, noise) composited over the
real frame; the template is rendered at a perturbed pose (random camera-frame rotation / translation of the given
size). Ground truth uv_true = projection of each matched surface point at the state pose.

    python loma_render_match_worker.py --task-root R --frames DIR --mesh robot_mesh.npz --state state.json \
        --out OUT [--episodes ...] [--group main|romat] [--synthetic 2,20,0]
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np


def load_state(path: str) -> dict:
    d = json.loads(Path(path).read_text())
    d = d.get("state", d)
    return {k: np.asarray(v, float) for k, v in d["T_cam_base"].items()}


def se3_exp(xi):
    w, v = np.asarray(xi[:3], float), np.asarray(xi[3:], float)
    th = np.linalg.norm(w)
    Wx = np.array([[0, -w[2], w[1]], [w[2], 0, -w[0]], [-w[1], w[0], 0]])
    if th < 1e-12:
        R, Vm = np.eye(3) + Wx, np.eye(3) + 0.5 * Wx
    else:
        R = np.eye(3) + np.sin(th) / th * Wx + (1 - np.cos(th)) / th ** 2 * Wx @ Wx
        Vm = np.eye(3) + (1 - np.cos(th)) / th ** 2 * Wx + (th - np.sin(th)) / th ** 3 * Wx @ Wx
    T = np.eye(4)
    T[:3, :3], T[:3, 3] = R, Vm @ v
    return T


def magsac_pnp(X, uv, K, thr_px, T0=None):
    """MAGSAC++ PnP (OpenCV USAC); returns (inlier mask, T or None, method)."""
    import cv2

    n = len(X)
    if n < 6:
        return np.zeros(n, bool), None, "too_few"
    X = np.ascontiguousarray(X, np.float64)
    uv = np.ascontiguousarray(uv, np.float64)
    method = "usac_magsac"
    try:
        ok, rvec, tvec, inl = cv2.solvePnPRansac(X, uv, K, None, flags=cv2.USAC_MAGSAC, reprojectionError=thr_px,
                                                 confidence=0.9999, iterationsCount=10000)
    except cv2.error:
        method = "ransac_epnp"
        ok, rvec, tvec, inl = cv2.solvePnPRansac(X, uv, K, None, flags=cv2.SOLVEPNP_EPNP, reprojectionError=thr_px,
                                                 confidence=0.9999, iterationsCount=10000)
    mask = np.zeros(n, bool)
    if not ok or inl is None:
        return mask, None, method
    mask[np.asarray(inl).ravel()] = True
    R, _ = cv2.Rodrigues(rvec)
    T = np.eye(4)
    T[:3, :3], T[:3, 3] = R, np.asarray(tvec).ravel()
    return mask, T, method


def reproject(T, X, K):
    P = X @ T[:3, :3].T + T[:3, 3]
    return np.stack([K[0, 0] * P[:, 0] / P[:, 2] + K[0, 2], K[1, 1] * P[:, 1] / P[:, 2] + K[1, 2]], 1)


class Matcher:
    def __init__(self, name: str, num_keypoints: int, filter_threshold: float | None, tmpdir: str):
        from loma import LoMa, LoMaB, LoMaG

        cfg = {"loma_g": LoMaG, "loma_b": LoMaB}[name]()
        self.model = LoMa(cfg)
        self.name = name
        self.num_keypoints = num_keypoints
        self.th = filter_threshold if filter_threshold is not None else self.model.cfg.filter_threshold
        self.tmp = Path(tmpdir)
        self.tmp.mkdir(parents=True, exist_ok=True)

    def match(self, img_a: np.ndarray, img_b: np.ndarray):
        """LoMa matches between two uint8 RGB arrays through the official path preprocessing. Returns (uv_a, uv_b,
        score, n_kp_a, n_kp_b), coordinates continuous in each crop (pixel centres at +.5)."""
        import torch
        from loma.loma import filter_matches, to_pixel_coords
        from PIL import Image

        pa, pb = self.tmp / "a.png", self.tmp / "b.png"
        Image.fromarray(img_a).save(pa)
        Image.fromarray(img_b).save(pb)
        ka, da, h1, w1 = self.model.detect_and_describe(str(pa), self.num_keypoints)
        kb, db, h2, w2 = self.model.detect_and_describe(str(pb), self.num_keypoints)
        with torch.inference_mode():
            scores = self.model(ka, kb, da, db)["scores"]
        m0, _, ms0, _ = filter_matches(scores, self.th)
        valid = m0[0] > -1
        ia = torch.where(valid)[0]
        ua = to_pixel_coords(ka[0][ia], h1, w1).float().cpu().numpy()
        ub = to_pixel_coords(kb[0][m0[0][valid]], h2, w2).float().cpu().numpy()
        sc = ms0[0][valid].float().cpu().numpy()
        return ua.astype(np.float64), ub.astype(np.float64), sc, int(ka.shape[1]), int(kb.shape[1])


def pseudo_real(rgb_real, raster, Vc, Nc, mesh, cfg_alt, rng):
    """Robot rendered with an alternative appearance composited over the real frame (synthetic test)."""
    from scipy import ndimage

    from s4d.realworld.calib.loma.render import shade

    col = np.clip(mesh.color.astype(np.float64) * rng.uniform(0.85, 1.1, 3)[None], 0, 1)
    rgb, _ = shade(Vc, Nc, col, raster.fid, raster.bary, raster.ss, cfg_alt)
    x0, y0, w, h = raster.window
    hit = (raster.fid >= 0).reshape(h, raster.ss, w, raster.ss).mean((1, 3))
    out = rgb_real.astype(np.float64).copy()
    crop = out[y0:y0 + h, x0:x0 + w]
    a = hit[..., None]
    crop[:] = a * rgb + (1 - a) * crop
    crop[:] = ndimage.gaussian_filter(crop, sigma=(0.6, 0.6, 0))
    out = np.clip(out + rng.normal(0, 2.0, out.shape), 0, 255)
    return out.astype(np.uint8)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--task-root", required=True)
    ap.add_argument("--frames", required=True)
    ap.add_argument("--mesh", required=True)
    ap.add_argument("--state", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--episodes", nargs="*", default=None)
    ap.add_argument("--group", default="main", choices=["main", "romat"])
    ap.add_argument("--matcher", default="loma_g")
    ap.add_argument("--num-keypoints", type=int, default=2048)
    ap.add_argument("--filter-threshold", type=float, default=None)
    ap.add_argument("--min-render-px", type=int, default=3000)
    ap.add_argument("--margin", type=int, default=16)
    ap.add_argument("--ss", type=int, default=2)
    ap.add_argument("--background", type=float, default=0.0)
    ap.add_argument("--max-frames-per-camera", type=int, default=None)
    ap.add_argument("--magsac-px", type=float, default=4.0)
    ap.add_argument("--synthetic", default=None)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--tmp", default=None, help="scratch dir for the PNG crops (default: per process)")
    a = ap.parse_args(argv)

    import h5py
    import torch

    from s4d.realworld.calib.loma.render import RenderConfig, RobotMesh, render_view, surface_points

    torch.set_float32_matmul_precision("highest")
    mesh = RobotMesh.load(a.mesh)
    Ts = load_state(a.state)
    rcfg = RenderConfig(ss=a.ss, margin_px=a.margin, background=(a.background,) * 3)
    alt = RenderConfig(ss=a.ss, ambient=0.25, key=0.45, fill=0.55, background=(0.0,) * 3, albedo_scale=0.8)
    synth = None
    if a.synthetic:
        r_deg, t_mm, sd = a.synthetic.split(",")
        synth = (float(r_deg), float(t_mm), int(sd))
    import os

    matcher = Matcher(a.matcher, a.num_keypoints, a.filter_threshold, a.tmp or f"/ram/tmp/loma-{os.getpid()}")
    task_root = Path(a.task_root)
    files = sorted(Path(a.frames).glob("*.npz"))
    if a.episodes:
        files = [f for f in files if f.stem in set(a.episodes)]
    for fpath in files:
        ep = fpath.stem
        out = Path(a.out) / f"{ep}.npz"
        if out.exists():
            continue
        out.parent.mkdir(parents=True, exist_ok=True)
        t0 = time.time()
        with np.load(fpath) as z:
            doc = json.loads(str(z["meta"]))
            arr = {k: z[k] for k in z.files if k != "meta"}
        links = doc["links"]
        dt = 2 * doc.get("velocity_dt_ns", 10_000_000) * 1e-9
        rows = {k: [] for k in ("cam", "frame", "t_index", "uv_obs", "uv_ren", "score", "link", "X_link", "X_base",
                                "V_base", "z_range", "bary_out", "z_ren", "magsac_frame", "magsac_cam", "uv_true",
                                "grip")}
        frames_doc = []
        i7 = links.index("panda_link7") if "panda_link7" in links else None
        cams = sorted(doc["cameras"])
        for ci, cam in enumerate(cams):
            c = doc["cameras"][cam]
            key = c["key"]
            pre = f"romat/{key}" if a.group == "romat" else key
            if f"{pre}/t_index" not in arr or cam not in Ts:
                continue
            K = np.asarray(c["K"], float)
            hw = tuple(c["hw"])
            tis = arr[f"{pre}/t_index"]
            ks = range(len(tis))
            if a.max_frames_per_camera and len(tis) > a.max_frames_per_camera:
                ks = np.linspace(0, len(tis) - 1, a.max_frames_per_camera).round().astype(int)
            rgb_path = task_root / ep / "episodes" / ep / "normalized" / "frames" / f"{key}.h5"
            cam_rows_start = len(rows["cam"])
            with h5py.File(rgb_path, "r") as fh:
                for k in ks:
                    ti = int(tis[k])
                    fk, fkm, fkp = arr[f"{pre}/fk"][k], arr[f"{pre}/fk_minus"][k], arr[f"{pre}/fk_plus"][k]
                    T_true = Ts[cam]
                    rng = np.random.default_rng((synth[2] if synth else 0) * 100003 + ci * 1009 + ti)
                    T_ren = T_true
                    if synth:
                        ax = rng.normal(size=3)
                        ax /= np.linalg.norm(ax)
                        dv = rng.normal(size=3)
                        dv *= synth[1] * 1e-3 / np.linalg.norm(dv)
                        T_ren = se3_exp(np.r_[np.radians(synth[0]) * ax, dv]) @ T_true
                    r, Vc, Nc = render_view(mesh, fk, links, T_ren, K, hw, rcfg, device=a.device)
                    fdoc = {"cam": cam, "t_index": ti, "k": int(k), "speed": float(arr[f"{pre}/speed"][k])}
                    if r is None or int(r.mask.sum()) < a.min_render_px:
                        fdoc["skipped"] = "render_px" if r is not None else "not_in_view"
                        fdoc["render_px"] = 0 if r is None else int(r.mask.sum())
                        frames_doc.append(fdoc)
                        continue
                    real = fh["frames"][ti]
                    if synth:
                        rt, Vt, Nt = render_view(mesh, fk, links, T_true, K, hw, alt, window=(0, 0, hw[1], hw[0]),
                                                 device=a.device)
                        real = pseudo_real(real, rt, Vt, Nt, mesh, alt, rng)
                    x0, y0, w, h = r.window
                    crop = np.ascontiguousarray(real[y0:y0 + h, x0:x0 + w])
                    ua, ub, sc, nka, nkb = matcher.match(r.rgb, crop)
                    uv_ren = ua + np.array([x0, y0])
                    uv_obs = ub + np.array([x0, y0])
                    sp = surface_points(r, Vc, mesh, uv_ren, K)
                    ok = sp["valid"]
                    fdoc.update(window=[int(v) for v in r.window], render_px=int(r.mask.sum()), kp_render=nka,
                                kp_real=nkb, matches=int(len(ua)), surface_hits=int(ok.sum()))
                    if not ok.any():
                        frames_doc.append(fdoc)
                        continue
                    lk = sp["link"][ok]
                    XL = sp["X_link"][ok]
                    li = [links.index(mesh.links[j]) for j in lk]
                    Tl, Tm, Tp = fk[li], fkm[li], fkp[li]
                    Xb = np.einsum("nij,nj->ni", Tl[:, :3, :3], XL) + Tl[:, :3, 3]
                    Xp = np.einsum("nij,nj->ni", Tp[:, :3, :3], XL) + Tp[:, :3, 3]
                    Xm = np.einsum("nij,nj->ni", Tm[:, :3, :3], XL) + Tm[:, :3, 3]
                    m = int(ok.sum())
                    fr_idx = len(frames_doc)
                    inl, Tf, meth = magsac_pnp(Xb, uv_obs[ok], K, a.magsac_px)
                    fdoc.update(magsac_frame_inliers=int(inl.sum()), pnp_method=meth)
                    rows["cam"].append(np.full(m, ci, np.int8))
                    rows["frame"].append(np.full(m, fr_idx, np.int32))
                    rows["t_index"].append(np.full(m, ti, np.int32))
                    rows["uv_obs"].append(uv_obs[ok].astype(np.float32))
                    rows["uv_ren"].append(uv_ren[ok].astype(np.float32))
                    rows["score"].append(sc[ok].astype(np.float32))
                    rows["link"].append(np.asarray(li, np.int16))
                    rows["X_link"].append(XL.astype(np.float32))
                    rows["X_base"].append(Xb)
                    rows["V_base"].append(((Xp - Xm) / dt).astype(np.float32))
                    rows["z_range"].append(sp["z_range"][ok].astype(np.float32))
                    rows["bary_out"].append(sp["bary_out"][ok].astype(np.float32))
                    rows["z_ren"].append(sp["X_cam"][ok, 2].astype(np.float32))
                    rows["magsac_frame"].append(inl)
                    rows["magsac_cam"].append(np.zeros(m, bool))
                    rows["grip"].append(np.full(m, float(arr[f"{pre}/grip"][k]), np.float32))
                    rows["uv_true"].append(reproject(T_true, Xb, K).astype(np.float32) if synth
                                           else np.full((m, 2), np.nan, np.float32))
                    if Tf is not None:
                        fdoc["pnp_T"] = Tf.tolist()
                    if i7 is not None:
                        fdoc["T7"] = fk[i7].tolist()  # panda_link7 pose (gripper-mount identifiability analysis)
                    frames_doc.append(fdoc)
            # pooled per (episode, camera): one fixed camera pose for all its frames
            if len(rows["cam"]) > cam_rows_start:
                sl = slice(cam_rows_start, len(rows["cam"]))
                Xb = np.concatenate(rows["X_base"][sl])
                uv = np.concatenate(rows["uv_obs"][sl]).astype(np.float64)
                inl, Tc, meth = magsac_pnp(Xb, uv, K, a.magsac_px)
                off = 0
                for j in range(sl.start, sl.stop):
                    n = len(rows["X_base"][j])
                    rows["magsac_cam"][j] = inl[off:off + n]
                    off += n
                frames_doc.append({"cam": cam, "pooled": True, "inliers": int(inl.sum()), "n": int(len(inl)),
                                   "pnp_T": None if Tc is None else Tc.tolist(), "method": meth})
        res = {k: (np.concatenate(v) if v else np.zeros(0)) for k, v in rows.items()}
        meta = {"episode_id": ep, "cameras": cams, "links": links, "group": a.group, "matcher": a.matcher,
                "K": {c: doc["cameras"][c]["K"] for c in cams}, "hw": {c: doc["cameras"][c]["hw"] for c in cams},
                "num_keypoints": a.num_keypoints, "filter_threshold": matcher.th, "render": vars(rcfg),
                "min_render_px": a.min_render_px, "magsac_px": a.magsac_px, "synthetic": a.synthetic,
                "state": a.state, "frames": frames_doc, "seconds": round(time.time() - t0, 1),
                "mesh": a.mesh, "torch": torch.__version__,
                "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}
        tmp = str(out) + ".tmp.npz"
        np.savez_compressed(tmp, meta=json.dumps(meta, default=lambda x: x.tolist() if hasattr(x, "tolist") else str(x)),
                            **res)
        Path(tmp).replace(out)
        print(json.dumps({"episode": ep, "seconds": meta["seconds"], "matches": int(len(res["cam"])),
                          "magsac_cam": int(res["magsac_cam"].sum()) if len(res["cam"]) else 0,
                          "frames": sum(1 for f in frames_doc if "matches" in f)}), flush=True)


if __name__ == "__main__":
    main()
