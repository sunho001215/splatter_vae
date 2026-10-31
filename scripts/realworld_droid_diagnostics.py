"""Real-data diagnostics for the real-world pipeline on a raw DROID episode (docs/realworld_droid_validation.md).

Reads only the pipeline's outputs (``--root``), the raw episode's MP4 left views (SDK-rectified, for the
rectification check) and optional reference depth (PointWorld-DROID ``*_depth.h5``). Needs OpenCV (SIFT) and
matplotlib, so it runs in an isolated environment, e.g.

    <media env>/bin/python -I scripts/realworld_droid_diagnostics.py rectify --config CONFIG --episode ID --out DIR
    ... stereo | tracks  (after the stereo_depth / track2d+lift3d stages)

Every subcommand writes ``<out>/<name>.json`` (numbers) and PNGs next to it.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

try:
    import cv2  # noqa: E402  (media env: SIFT, PNG)
except ImportError:  # repository venv (export check): no OpenCV
    cv2 = None
import h5py  # noqa: E402
import numpy as np  # noqa: E402
import yaml  # noqa: E402

from s4d.realworld import geometry as G  # noqa: E402
from s4d.realworld import io as rio  # noqa: E402
from s4d.realworld import pipeline as P  # noqa: E402

SIFT = cv2.SIFT_create(4000) if cv2 else None
BF = cv2.BFMatcher() if cv2 else None


def sift_matches(a: np.ndarray, b: np.ndarray, ratio: float = 0.7) -> tuple[np.ndarray, np.ndarray]:
    """Matched keypoints (OpenCV integer-centre pixels) of two RGB uint8 images."""
    ka, da = SIFT.detectAndCompute(cv2.cvtColor(a, cv2.COLOR_RGB2GRAY), None)
    kb, db = SIFT.detectAndCompute(cv2.cvtColor(b, cv2.COLOR_RGB2GRAY), None)
    if da is None or db is None:
        return np.zeros((0, 2), np.float32), np.zeros((0, 2), np.float32)
    good = [m for m, n in BF.knnMatch(da, db, k=2) if m.distance < ratio * n.distance]
    return (np.float32([ka[m.queryIdx].pt for m in good]), np.float32([kb[m.trainIdx].pt for m in good]))


def stats(x: np.ndarray) -> dict:
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    if not len(x):
        return {"n": 0}
    return {"n": int(len(x)), "median": float(np.median(x)), "mean": float(x.mean()),
            "p90": float(np.percentile(x, 90)), "median_abs": float(np.median(np.abs(x)))}


def load(cfg_path: str):
    cfg = yaml.safe_load(Path(cfg_path).read_text())
    return cfg, Path(cfg["output_root"])


def mp4_frames(episode_dir: Path, serial: str, indices) -> np.ndarray:
    import av

    with av.open(str(episode_dir / "recordings" / "MP4" / f"{serial}.mp4")) as c:
        frames = [f.to_ndarray(format="rgb24") for f in c.decode(video=0)]
    return np.stack([frames[i] for i in indices])


def sdk_left_K(intrinsics_jsonl: str, episode_id: str, role: str) -> np.ndarray:
    for line in open(intrinsics_jsonl):
        if episode_id in line:
            rec = json.loads(line)
            if rec["episode_uuid"] == episode_id:
                L = rec["cameras"][role]["left"]
                return np.array([[L["fx"], 0, L["cx"]], [0, L["fy"], L["cy"]], [0, 0, 1.0]])
    raise KeyError(episode_id)


def save_png(path: Path, image: np.ndarray) -> None:
    from PIL import Image

    Image.fromarray(np.ascontiguousarray(image)).save(path)


def colorize(x: np.ndarray, lo: float, hi: float, valid: np.ndarray | None = None) -> np.ndarray:
    import matplotlib

    v = np.clip((np.nan_to_num(x) - lo) / max(hi - lo, 1e-9), 0, 1)
    rgb = (matplotlib.colormaps["turbo"](v)[..., :3] * 255).astype(np.uint8)
    if valid is not None:
        rgb[~valid] = 0
    return rgb


# ---------------------------------------------------------------- rectification


def cmd_rectify(args) -> dict:
    cfg, root = load(args.config)
    ac = cfg["adapter"]["config"]
    episode_dir = Path(ac["episodes"][args.episode])
    meta = P.load_meta(root, args.episode)
    notes = json.loads(meta.provenance.notes)["cameras"]
    out = {}
    for g in meta.stereo:
        L, R = meta.stream(g.left_stream), meta.stream(g.right_stream)
        hw = (L.intrinsics.height, L.intrinsics.width)
        idx = list(range(0, len(meta.stream_timestamps_ns[g.left_stream]), args.every))
        raw_l = P.read_frames(root, args.episode, g.left_stream, idx)
        raw_r = P.read_frames(root, args.episode, g.right_stream, idx)
        # (a) the pipeline's own rectification (stereo_depth stage path): left/right row alignment
        rect, KL, KR, _ = P.rig_rectification(meta, g.rig_id)
        maps = (G.rectify_map(L.intrinsics.K(), L.intrinsics.distortion_model, L.intrinsics.distortion, rect.R1, KL, hw),
                G.rectify_map(R.intrinsics.K(), R.intrinsics.distortion_model, R.intrinsics.distortion, rect.R2, KR, hw))
        dy, disp = [], []
        for k in range(len(idx)):
            a, b = G.remap(raw_l[k], maps[0], order=1), G.remap(raw_r[k], maps[1], order=1)
            pa, pb = sift_matches(a, b)
            ok = (pa[:, 0] > pb[:, 0]) & (np.abs(pa[:, 1] - pb[:, 1]) < 8)
            dy.append(pa[ok, 1] - pb[ok, 1])
            disp.append(pa[ok, 0] - pb[ok, 0])
            if k == 0:
                pair = np.concatenate([a, b], 1)
                for y in range(40, hw[0], 60):
                    pair[y] = (255, 0, 255)
                save_png(Path(args.out) / f"rectified_pair_{g.rig_id}_{idx[k]:04d}.png", pair)
        # (b) the same rotation R1 with the SDK's rectified K reproduces the SDK-rectified left view (DROID MP4)?
        K_sdk_cont = sdk_left_K(ac["droid3d_intrinsics"], args.episode, g.rig_id)
        K_sdk_cont[:2, 2] += 0.5  # ZED SDK / OpenCV integer-centre -> continuous convention
        sdk_map = G.rectify_map(L.intrinsics.K(), L.intrinsics.distortion_model, L.intrinsics.distortion, rect.R1,
                                K_sdk_cont, hw)
        mp4 = mp4_frames(episode_dir, notes[g.rig_id]["serial"], idx)
        dm, photo = [], []
        for k in range(len(idx)):
            ours = G.remap(raw_l[k], sdk_map, order=1)
            pa, pb = sift_matches(ours, mp4[k])
            ok = np.linalg.norm(pa - pb, axis=1) < 8
            dm.append(pa[ok] - pb[ok])
            inner = (slice(40, hw[0] - 40), slice(40, hw[1] - 40))
            photo.append(np.abs(ours[inner].astype(float) - mp4[k][inner].astype(float)).mean())
            if k == 0:
                diff = np.clip(np.abs(ours.astype(int) - mp4[k].astype(int)) * 4, 0, 255).astype(np.uint8)
                save_png(Path(args.out) / f"ours_vs_sdk_{g.rig_id}_{idx[k]:04d}.png",
                         np.concatenate([ours, mp4[k], diff], 1))
        dm = np.concatenate(dm)
        # (c) cross-check against OpenCV's Bouguet implementation with the same calibration
        D1 = np.array(L.intrinsics.distortion)
        D2 = np.array(R.intrinsics.distortion)
        T = g.T()
        R1cv, R2cv = cv2.stereoRectify(L.intrinsics.K("integer_centers"), D1, R.intrinsics.K("integer_centers"), D2,
                                       (hw[1], hw[0]), T[:3, :3], T[:3, 3].reshape(3, 1),
                                       flags=cv2.CALIB_ZERO_DISPARITY, alpha=0)[:2]
        out[g.rig_id] = {
            "frames": idx, "K_rect_pipeline": np.asarray(KL).tolist(), "baseline_m": rect.baseline_m,
            "lr_vertical_residual_px": stats(np.concatenate(dy)), "lr_disparity_px": stats(np.concatenate(disp)),
            "vs_sdk_mp4_dx_px": stats(dm[:, 0]), "vs_sdk_mp4_dy_px": stats(dm[:, 1]),
            "vs_sdk_mp4_mean_abs_rgb": float(np.mean(photo)),
            "R1_vs_opencv_deg": G.rotation_angle_deg(rect.R1, R1cv), "R2_vs_opencv_deg": G.rotation_angle_deg(rect.R2, R2cv),
            "sdk_baseline_m": notes[g.rig_id].get("sdk_baseline_m"),
        }
    return out


# ---------------------------------------------------------------- stereo depth


def cmd_stereo(args) -> dict:
    cfg, root = load(args.config)
    meta = P.load_meta(root, args.episode)
    rec = rio.read_record(rio.stage_dir(root, args.episode, "stereo_depth"))
    out = {"model": {k: v.get("model") for k, v in rec["summary"].items()}}
    poses = {g.rig_id: P.accepted_calibration(root, args.episode, g.left_stream) for g in meta.stereo}
    rigs = {}
    for g in meta.stereo:
        with h5py.File(rio.stage_dir(root, args.episode, "stereo_depth") / f"{g.rig_id}.h5", "r") as f:
            n = f["left"]["depth"].shape[0]
            idx = list(range(0, n, args.every))
            dl = np.stack([f["left"]["disparity"][i] for i in idx])
            vl = np.stack([f["left"]["valid"][i] for i in idx])
            zl = np.stack([f["left"]["depth"][i] for i in idx])
            img_l = np.stack([f["left"]["rectified_image"][i] for i in idx])
            has_right = "right" in f
            if has_right:
                dr = np.stack([f["right"]["disparity"][i] for i in idx])
                vr = np.stack([f["right"]["valid"][i] for i in idx])
                img_r = np.stack([f["right"]["rectified_image"][i] for i in idx])
            K = np.asarray(f.attrs["K_left"])
            B = float(f.attrs["baseline_m"])
        H, W = dl.shape[1:]
        res = {"frames": idx, "valid_fraction_left": float(vl.mean()), "depth_left_m": stats(zl[vl]),
               "disparity_left_px": stats(dl[vl]), "baseline_m": B, "K": K.tolist()}
        # sparse, independent check: SIFT left/right correspondences on the rectified pair vs model disparity
        sparse = []
        for k in range(len(idx)):
            if not has_right:
                break
            pa, pb = sift_matches(img_l[k], img_r[k])
            ok = (pa[:, 0] > pb[:, 0]) & (np.abs(pa[:, 1] - pb[:, 1]) < 1.5)
            u, v = pa[ok, 0], pa[ok, 1]  # integer-centre keypoints == array indices
            iu, iv = np.clip(np.rint(u).astype(int), 0, W - 1), np.clip(np.rint(v).astype(int), 0, H - 1)
            sparse.append(dl[k, iv, iu] - (pa[ok, 0] - pb[ok, 0]))
        if sparse:
            res["model_minus_sift_disparity_px"] = stats(np.concatenate(sparse))
        if has_right:  # left/right consistency: d_R(x - d_L) == d_L
            xs = np.arange(W)[None, None, :] + 0.5 - dl
            xi = np.clip(np.floor(xs).astype(int), 0, W - 1)
            dr_at = np.take_along_axis(dr, xi, axis=2)
            both = vl & np.take_along_axis(vr, xi, axis=2)
            lr = np.abs(dr_at - dl)[both]
            res["lr_consistency_abs_px"] = stats(lr)
            res["lr_consistent_fraction_1px"] = float((lr < 1.0).mean()) if len(lr) else None
            res["valid_fraction_right"] = float(vr.mean())
        k = 0
        save_png(Path(args.out) / f"depth_{g.rig_id}_{idx[k]:04d}.png", np.concatenate([
            img_l[k], colorize(dl[k], 0, np.percentile(dl[vl], 99), vl[k]),
            colorize(zl[k], 0.2, 2.0, vl[k])], 1))
        rigs[g.rig_id] = res
    out["rigs"] = rigs
    # cross-camera metric consistency: lift rig A's left depth with its provided pose, project into rig B
    names = [g.rig_id for g in meta.stereo]
    cross = {}
    for a in names:
        for b in names:
            if a == b or poses[a] is None or poses[b] is None:
                continue
            with h5py.File(rio.stage_dir(root, args.episode, "stereo_depth") / f"{a}.h5", "r") as fa, \
                    h5py.File(rio.stage_dir(root, args.episode, "stereo_depth") / f"{b}.h5", "r") as fb:
                Ka, Kb = np.asarray(fa.attrs["K_left"]), np.asarray(fb.attrs["K_left"])
                errs, rel = [], []
                for i in range(0, fa["left"]["depth"].shape[0], args.every):
                    za, zb = fa["left"]["depth"][i], fb["left"]["depth"][i]
                    T_ba = poses[b] @ G.invert_T(poses[a])
                    pred = G.reproject_depth(za, Ka, T_ba, Kb, zb.shape)
                    both = (pred > 0) & (zb > 0)
                    d = (pred - zb)[both]
                    errs.append(d)
                    rel.append(np.abs(d) / zb[both])
                    if i == 0:
                        img = fb["left"]["rectified_image"][i]
                        err = np.zeros(zb.shape)
                        err[both] = np.abs(pred - zb)[both]
                        save_png(Path(args.out) / f"cross_{a}_to_{b}_{i:04d}.png", np.concatenate([
                            img, colorize(pred, 0.2, 2.0, pred > 0), colorize(err, 0, 0.05, both)], 1))
            e, r = np.concatenate(errs), np.concatenate(rel)
            cross[f"{a}->{b}"] = {"signed_m": stats(e), "relative": stats(r), "rel_below_2pct": float((r < 0.02).mean()),
                                  "note": "z-buffered reprojection; occlusions and non-overlapping regions included"}
    out["cross_camera"] = cross
    # reference depth (PointWorld-DROID full depth package, uint16 mm, 320x180 on the SDK-rectified grid K/4)
    if args.reference_depth:
        ac = cfg["adapter"]["config"]
        notes = json.loads(meta.provenance.notes)["cameras"]
        ref = {}
        with h5py.File(args.reference_depth, "r") as f:
            for g in meta.stereo:
                key = f"{notes[g.rig_id]['serial']}+ext"
                if key not in f:
                    continue
                K_sdk = sdk_left_K(ac["droid3d_intrinsics"], args.episode, g.rig_id)
                K_ref = K_sdk.copy()
                K_ref[:2, 2] += 0.5  # SDK integer-centre -> continuous
                K_ref[:2] *= 0.25  # 1280x720 -> 320x180 is an exact x0.25 map in continuous coordinates
                rect, KL, _, _ = P.rig_rectification(meta, g.rig_id)
                diffs, rels = [], []
                with h5py.File(rio.stage_dir(root, args.episode, "stereo_depth") / f"{g.rig_id}.h5", "r") as fd:
                    for i in range(0, fd["left"]["depth"].shape[0], args.every):
                        ours = G.reproject_depth(fd["left"]["depth"][i], np.asarray(KL), np.eye(4), K_ref, (180, 320))
                        theirs = f[key]["depth"][i].astype(np.float32) / 1000.0
                        both = (ours > 0) & (theirs > 0)
                        diffs.append((ours - theirs)[both])
                        rels.append(np.abs(ours - theirs)[both] / theirs[both])
                d, r = np.concatenate(diffs), np.concatenate(rels)
                ref[g.rig_id] = {"signed_m": stats(d), "relative": stats(r), "rel_below_2pct": float((r < 0.02).mean()),
                                 "reference": str(args.reference_depth), "reference_K_continuous": K_ref.tolist()}
        out["reference_depth"] = ref
    return out


# ---------------------------------------------------------------- tracks


def cmd_tracks(args) -> dict:
    cfg, root = load(args.config)
    meta = P.load_meta(root, args.episode)
    t2d = rio.read_record(rio.stage_dir(root, args.episode, "track2d"))
    l3d = rio.read_record(rio.stage_dir(root, args.episode, "lift3d"))
    out = {"model": {k: v.get("model") for k, v in t2d["summary"].items()}, "cameras": {}}
    for g in meta.stereo:
        cam = g.left_stream
        if cam not in t2d["summary"]:
            continue
        stamps = meta.stream_timestamps_ns[cam]
        chunks = t2d["summary"][cam]["chunks"]
        vis_all, conf_all, idx_ok, drift_static, drift_moving, n_static = [], [], True, [], [], 0
        for c, name in enumerate(chunks):
            tr = P.load_tracks2d(rio.stage_dir(root, args.episode, "track2d") / name, cam)
            idx_ok &= bool(np.array_equal(tr.timestamps_ns, stamps[tr.frame_indices]))
            idx_ok &= int(tr.frame_indices[0]) == tr.query_frame
            vis_all.append(tr.visibility.ravel())
            conf_all.append(tr.confidence.ravel())
            if name in (l3d or {}).get("summary", {}).get(cam, {}).get("chunks", []):
                with np.load(rio.stage_dir(root, args.episode, "lift3d") / name) as z:
                    xyz, status = z["xyz_base"], z["status"]
                ok = np.isfinite(xyz).all(-1)
                full = ok.all(0)
                if full.any():
                    motion2d = np.linalg.norm(tr.uv[-1] - tr.uv[0], axis=-1)
                    spread = np.linalg.norm(xyz[:, full] - np.median(xyz[:, full], 0), axis=-1).max(0)
                    static = motion2d[full] < 1.0
                    n_static += int(static.sum())
                    drift_static.append(spread[static])
                    drift_moving.append(spread[~static])
            if c == len(chunks) // 2:
                frames = P.grid_frames(root, args.episode, meta, g, list(tr.frame_indices))
                img = frames[-1].copy()
                sel = np.arange(0, tr.uv.shape[1], max(1, tr.uv.shape[1] // 600))
                for p in sel:
                    color = tuple(int(x) for x in (np.array(cv2.applyColorMap(np.uint8([[p * 255 // max(1, tr.uv.shape[1])]]),
                                                                               cv2.COLORMAP_TURBO)[0, 0])[::-1]))
                    pts = tr.uv[:, p]
                    vis = tr.visibility[:, p] > 0.5
                    for t in range(1, len(pts)):
                        if vis[t] and vis[t - 1] and np.isfinite(pts[t]).all() and np.isfinite(pts[t - 1]).all():
                            cv2.line(img, tuple(np.int32(pts[t - 1] - 0.5)), tuple(np.int32(pts[t] - 0.5)), color, 1)
                save_png(Path(args.out) / f"tracks_{g.rig_id}_q{tr.query_frame:05d}.png", np.concatenate([frames[0], img], 1))
        # lifted points of this camera seen by the other rigs: projected depth vs that rig's own stereo depth
        cross = {}
        for other in meta.stereo:
            if other.rig_id == g.rig_id:
                continue
            T_o = P.accepted_calibration(root, args.episode, other.left_stream)
            if T_o is None:
                continue
            rel = []
            with h5py.File(rio.stage_dir(root, args.episode, "stereo_depth") / f"{other.rig_id}.h5", "r") as fo:
                K_o = np.asarray(fo.attrs["K_left"])
                for name in chunks:
                    p3 = rio.stage_dir(root, args.episode, "lift3d") / name
                    if not p3.is_file():
                        continue
                    with np.load(p3) as z:
                        xyz, fidx = z["xyz_base"], z["frame_indices"]
                    for t in range(0, len(fidx), 5):
                        pts = xyz[t][np.isfinite(xyz[t]).all(-1)]
                        if not len(pts):
                            continue
                        uv, zc = G.project(G.transform_points(T_o, pts), K_o)
                        d_o = fo["left"]["depth"][int(fidx[t])]
                        H, W = d_o.shape
                        ok = np.isfinite(uv).all(1) & (zc > 0) & (uv[:, 0] >= 0) & (uv[:, 0] < W) & (uv[:, 1] >= 0) & (uv[:, 1] < H)
                        ii, jj = np.floor(uv[ok, 1]).astype(int), np.floor(uv[ok, 0]).astype(int)
                        target = d_o[ii, jj]
                        vis = (target > 0) & (zc[ok] <= target + 0.02)  # not occluded in the other view
                        rel.append(np.abs(zc[ok][vis] - target[vis]) / target[vis])
            r = np.concatenate(rel) if rel else np.zeros(0)
            cross[other.rig_id] = {"relative_depth_error": stats(r), "below_2pct": float((r < 0.02).mean()) if len(r) else None,
                                   "selection": "in image, positive depth, projected z <= target + 2 cm (occlusion test "
                                                "uses the target depth: not an independent calibration test)"}
        out["cameras"][cam] = {
            "lifted_points_in_other_rigs": cross,
            "chunks": len(chunks), "temporal_indexing_ok": idx_ok,
            "visibility": stats(np.concatenate(vis_all)), "confidence": stats(np.concatenate(conf_all)),
            "lift_status_counts": (l3d or {}).get("summary", {}).get(cam, {}).get("status_counts"),
            "static_tracks": n_static,
            "static_track_3d_max_deviation_m": stats(np.concatenate(drift_static)) if drift_static else None,
            "moving_track_3d_max_deviation_m": stats(np.concatenate(drift_moving)) if drift_moving else None,
        }
    return out


REPO_PY = "/home/ws/ws/splatter4d/.venv/bin/python"  # the SNeRL consumer's own environment (torch)


def run_snerl_consumer(semantic: np.ndarray | None) -> dict:
    """In the consumer's environment: SNeRL's class order and its cross-entropy (s4d.baselines.snerl.model.snerl_loss
    uses nn.CrossEntropyLoss over the official 5-way head) on int64 class-id targets."""
    import subprocess
    import tempfile

    code = (
        "import json,sys; sys.path.insert(0, sys.argv[1]); import numpy as np, torch\n"
        "from s4d.baselines.snerl.data import SEMANTIC_CLASSES\n"
        "out={'SEMANTIC_CLASSES': list(SEMANTIC_CLASSES)}\n"
        "if sys.argv[2] != '-':\n"
        "    t=torch.from_numpy(np.load(sys.argv[2])).long().reshape(-1); torch.manual_seed(0)\n"
        "    out['cross_entropy']=float(torch.nn.CrossEntropyLoss()(torch.randn(t.numel(), 5), t))\n"
        "print(json.dumps(out))\n")
    with tempfile.TemporaryDirectory() as tmp:
        arg = "-"
        if semantic is not None:
            arg = str(Path(tmp) / "semantic.npy")
            np.save(arg, semantic)
        res = subprocess.run([REPO_PY, "-I", "-c", code, str(ROOT), arg], capture_output=True, text=True, check=True,
                             env={"CUDA_VISIBLE_DEVICES": "", "PATH": "/usr/bin:/bin"})
    return json.loads(res.stdout.strip().splitlines()[-1])


def cmd_sam3(args) -> dict:
    """SAM 3 stage outputs: shapes, grid alignment, mask/score values, explicit absences, temporal identities, and
    the SNeRL consumer contract (class ids, nearest resize to 128x128, the consumer's cross-entropy)."""
    from s4d.realworld.baseline_stages import SNERL_CLASSES

    cfg, root = load(args.config)
    meta = P.load_meta(root, args.episode)
    sdir = rio.stage_dir(root, args.episode, "sam3_masks")
    rec = rio.read_record(sdir)
    out = {"record_config": rec["config"], "cameras": {}}
    consumer = run_snerl_consumer(None)
    out["class_order_matches_snerl"] = tuple(SNERL_CLASSES) == tuple(consumer["SEMANTIC_CLASSES"])
    for cam in rec["config"]["cameras"]:
        g = [x for x in meta.stereo if cam in (x.left_stream, x.right_stream)][0]
        with h5py.File(sdir / f"{rio.safe_name(cam)}.h5", "r") as f:
            prompts = json.loads(f.attrs["prompts"])
            sem = f["semantic"][:]
            present = f["class_present"][:]
            chunks = [f["chunks"][k] for k in sorted(f["chunks"])]
            masks = np.concatenate([c["masks"][:] for c in chunks]) if chunks else np.zeros((0,) + sem.shape[1:], bool)
            scores = np.concatenate([c["scores"][:] for c in chunks])
            frame = np.concatenate([c["frame"][:] + int(k) for k, c in zip(sorted(f["chunks"]), chunks)])
            pidx = np.concatenate([c["prompt_index"][:] for c in chunks])
            iid = np.concatenate([c["instance_id"][:] for c in chunks])
            count = np.concatenate([c["count"][:] for c in chunks])
            model = json.loads(chunks[0].attrs["model"])
            boxes = np.concatenate([c["boxes_xyxy"][:] for c in chunks])
        images = P.grid_frames(root, args.episode, meta, g, list(range(sem.shape[0])))
        per_prompt = {}
        for p, text in enumerate(prompts):
            sel = pidx == p
            ids = sorted(set(iid[sel].tolist()))
            ious = []
            for i in ids:  # mask IoU of the same object id in consecutive frames (temporal identity)
                m = sel & (iid == i)
                fr = frame[m]
                order = np.argsort(fr)
                mk, fr = masks[m][order], fr[order]
                for a in range(len(fr) - 1):
                    if fr[a + 1] == fr[a] + 1:
                        inter = np.logical_and(mk[a], mk[a + 1]).sum()
                        union = np.logical_or(mk[a], mk[a + 1]).sum()
                        if union:
                            ious.append(inter / union)
            per_prompt[text] = {"frames_detected": int((count[:, p] > 0).sum()), "frames_absent": int((count[:, p] == 0).sum()),
                                "object_ids": ids, "consecutive_iou": stats(np.array(ious)),
                                "scores": stats(scores[sel]), "mask_area_px": stats(masks[sel].sum((1, 2)))}
        box_iou = []  # reported box vs the tight box of the mask (pixel edges; checks the box convention/scaling)
        for m, b in zip(masks, boxes):
            ys, xs = np.nonzero(m)
            if not len(xs):
                continue
            mb = np.array([xs.min(), ys.min(), xs.max() + 1, ys.max() + 1], float)
            inter = max(0, min(mb[2], b[2]) - max(mb[0], b[0])) * max(0, min(mb[3], b[3]) - max(mb[1], b[1]))
            union = (mb[2] - mb[0]) * (mb[3] - mb[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
            box_iou.append(inter / union if union > 0 else 0.0)
        # consumer contract: nearest resize to SNeRL's 128x128, int64 ids, the consumer's cross-entropy is finite
        small = np.stack([cv2.resize(s, (128, 128), interpolation=cv2.INTER_NEAREST) for s in sem]).astype(np.int64)
        ce = run_snerl_consumer(small)["cross_entropy"]
        out["cameras"][cam] = {
            "model": model, "prompts": prompts, "semantic_shape": list(sem.shape), "image_shape": list(images.shape),
            "grid_matches_images": list(sem.shape[1:]) == list(images.shape[1:3]),
            "mask_dtype": str(masks.dtype), "mask_shape": list(masks.shape[1:]),
            "scores_in_unit_interval": bool(((scores >= 0) & (scores <= 1)).all()),
            "box_vs_mask_iou": stats(np.array(box_iou)), "empty_masks": int((masks.sum((1, 2)) == 0).sum()),
            "semantic_values": sorted(np.unique(sem).tolist()), "class_pixel_fraction":
                {c: float((sem == k).mean()) for k, c in enumerate(SNERL_CLASSES)},
            "class_present_frames": {c: int(present[:, k].sum()) for k, c in enumerate(SNERL_CLASSES)},
            "per_prompt": per_prompt,
            "consumer": {"resized_hw": [128, 128], "dtype": str(small.dtype), "max_id": int(small.max()),
                         "cross_entropy_finite": bool(np.isfinite(ce)), "cross_entropy": ce},
        }
        colors = np.array([[0, 0, 0], [255, 60, 60], [60, 220, 60], [60, 120, 255]], np.uint8)
        for t in (0, sem.shape[0] // 2, sem.shape[0] - 1):
            overlay = (0.55 * images[t] + 0.45 * colors[sem[t]]).astype(np.uint8)
            save_png(Path(args.out) / f"sam3_{g.rig_id}_{t:04d}.png", np.concatenate([images[t], overlay], 1))
    return out


def cmd_export(args) -> dict:
    """Exported windows through the real-world dataset class (it calls s4d.data.contract.validate_batch on every
    sample), field statistics, and the exported gripper points projected into the exported images (mount check).
    Runs in the repository venv (torch)."""
    from s4d.realworld.dataset import RealworldCacheDataset

    cfg, root = load(args.config)
    cache = root / "export" / args.episode
    manifest = json.loads((cache / "manifest.json").read_text())
    out = {"manifest_keys": sorted(manifest), "windows": len(manifest["windows"]), "cameras": manifest["cameras"],
           "resolutions": manifest["resolutions"], "splits": {}}
    sam3_dir = rio.stage_dir(root, args.episode, "sam3_masks")
    for split in ("train", "validation"):
        try:
            ds = RealworldCacheDataset(cache, split=split)
        except ValueError as exc:
            out["splits"][split] = {"error": str(exc)}
            continue
        fields, inside, in_robot = {}, [], []
        for i in range(len(ds)):
            sample = ds[i]  # validate_batch runs here
            for k, v in sample.items():
                if k == "meta":
                    continue
                a = v.numpy()
                f = fields.setdefault(k, {"shape": list(a.shape), "dtype": str(a.dtype), "finite": True,
                                          "min": float("inf"), "max": float("-inf")})
                if a.dtype.kind == "f":
                    f["finite"] &= bool(np.isfinite(a).all())
                f["min"], f["max"] = min(f["min"], float(np.nanmin(a))), max(f["max"], float(np.nanmax(a)))
            # gripper points (T,G,3) in the base frame -> every view of every window time
            K, w2c, grip = sample["K"].numpy(), sample["w2c"].numpy(), sample["gripper_points"].numpy()
            H, W = sample["images"].shape[-2:]
            w = manifest["windows"][ds.indices[i]]
            for t in range(grip.shape[0]):
                for v in range(K.shape[0]):
                    uv, z = G.project(G.transform_points(w2c[v], grip[t]), K[v])
                    ok = (z > 0) & (uv[:, 0] >= 0) & (uv[:, 0] < W) & (uv[:, 1] >= 0) & (uv[:, 1] < H)
                    inside.append(ok.mean())
                    cam = manifest["cameras"][v]
                    frame = w["cameras"][cam]["frames"][t] if "cameras" in w and cam in w["cameras"] and \
                        "frames" in w["cameras"][cam] else w["canonical_indices"][t]
                    p = sam3_dir / f"{rio.safe_name(cam)}.h5"
                    if p.is_file() and ok.any():
                        with h5py.File(p, "r") as fs:
                            sem = fs["semantic"][int(frame)]
                        sy, sx = sem.shape[0] / H, sem.shape[1] / W
                        u = np.clip((uv[ok, 0] * sx).astype(int), 0, sem.shape[1] - 1)
                        vv = np.clip((uv[ok, 1] * sy).astype(int), 0, sem.shape[0] - 1)
                        in_robot.append(float((sem[vv, u] == 1).mean()))
            if i == 0:
                img = sample["images"].numpy()[0].transpose(0, 2, 3, 1).copy()  # (V,H,W,3) at window time 0
                for v in range(img.shape[0]):
                    uv, z = G.project(G.transform_points(w2c[v], grip[0]), K[v])
                    for x, y in uv[(z > 0)]:
                        xi, yi = int(x), int(y)
                        if 1 <= xi < W - 1 and 1 <= yi < H - 1:
                            img[v, yi - 1:yi + 2, xi - 1:xi + 2] = (255, 0, 255)
                save_png(Path(args.out) / f"export_gripper_points_{split}.png", np.concatenate(list(img), 1))
        out["splits"][split] = {"samples": len(ds), "validate_batch": "passed for every sample", "fields": fields,
                                "gripper_points_in_image": stats(np.array(inside)),
                                "gripper_points_on_sam3_robot_mask": stats(np.array(in_robot)) if in_robot else None}
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=("rectify", "stereo", "tracks", "sam3", "export"))
    ap.add_argument("--config", required=True)
    ap.add_argument("--episode", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--every", type=int, default=10)
    ap.add_argument("--reference-depth", default=None)
    args = ap.parse_args()
    Path(args.out).mkdir(parents=True, exist_ok=True)
    result = {"rectify": cmd_rectify, "stereo": cmd_stereo, "tracks": cmd_tracks, "sam3": cmd_sam3,
              "export": cmd_export}[args.cmd](args)
    (Path(args.out) / f"{args.cmd}.json").write_text(json.dumps(result, indent=1, default=str))
    print(json.dumps(result, indent=1, default=str)[:6000])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
