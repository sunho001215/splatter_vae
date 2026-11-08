"""POST-FREEZE independent board cross-validation of the board-free per-task rigs (``rig_dw``).

Run ONLY after the rig_dw algorithm, its parameters and both per-task solutions are frozen and committed; nothing in
this module feeds back into a solution. Inputs: the board recording (raw HDF5 of the recorder, read through the
panda_teleop adapter), the frozen per-task results, the task fixedness references and the static-match FS/sensor ratios.

Checks (per task rig):
1. fixedness of every camera in the board recording vs the task reference (episode-median registration);
2. board-only relative camera poses (ZVDA-CB01 detections, planar PnP + multi-camera bundle adjustment,
   leave-one-placement-out spread) vs the rig's relative poses: rotation angle, baseline direction, baseline length
   ratio (metric scale; the board pitch is the only metric input);
3. board corners triangulated with the rig (>= 2 cameras): reprojection error in every camera (px) and the
   triangulated pitch vs the nominal pitch (scale);
4. depth: sensor depth at the board corners vs the board-pose depth (per camera scale), and the implied FS scale
   (sensor scale x FS/sensor ratio of the task's static matches) vs the rig's exp(alpha);
5. absolute base-frame accuracy is NOT testable with an unattached board (reported as such).

    python -m s4d.realworld.rig_dw_boardcv --board H5 --rig-result TASK=RESULT.json [...] --out OUT.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from s4d.realworld.calibration_board import BoardRigConfig, BoardSpec, detect_board, leave_one_out_spread, solve_board
from s4d.realworld.rig_dw_geom import camera_centre, pose_distance

CAMS = ("135122070988/color", "135122073201/color", "135122076054/color")


def load_board(path: Path, every: int = 5, max_dt_ms: float = 20.0):
    """Adapter-normalised meta of the board recording and, for every ``every``-th reference-camera colour frame, the
    synchronised colour frame of every camera (nearest timestamp within ``max_dt_ms``) and its paired depth frame
    (nearest depth timestamp). Returns (meta, sync = {time key: {camera: (colour index, depth index, dt_ns)}},
    colour = {camera: {index: image}}, depth = {camera: {index: image}})."""
    from s4d.realworld.adapters.panda_teleop import PandaTeleopH5Adapter

    ad = PandaTeleopH5Adapter({"episodes": {"board": str(path)}, "robot": {
        "urdf_path": "/home/ws/ws/validation-20261031/panda_assets/panda_robotiq_2f85_droid_iso50.urdf",
        "gripper_model": "robotiq_2f_85", "gripper_joint": "robotiq_85_left_knuckle_joint", "base_frame": "panda_link0",
        "flange_frame": "panda_link8", "joint_names": [f"panda_joint{i}" for i in range(1, 8)]}})
    meta = ad.read_meta("board")
    ref = CAMS[0]
    st = {c: np.asarray(meta.stream_timestamps_ns[c]) for c in CAMS}
    sd = {c: np.asarray(meta.stream_timestamps_ns[c.replace("/color", "/depth")]) for c in CAMS}
    sync = {}
    for i in range(0, len(st[ref]), every):
        row = {}
        for c in CAMS:
            j = int(np.argmin(np.abs(st[c] - st[ref][i])))
            dt = int(st[c][j] - st[ref][i])
            if abs(dt) > max_dt_ms * 1e6:
                continue
            k = int(np.argmin(np.abs(sd[c] - st[c][j])))
            row[c] = (j, k, dt)
        if len(row) == len(CAMS):
            sync[f"p{i:05d}"] = row
    need_c = {c: {r[c][0] for r in sync.values()} for c in CAMS}
    need_d = {c: {r[c][1] for r in sync.values()} for c in CAMS}
    colour = {c: {i: img for i, _, img in ad.iter_frames("board", c) if i in need_c[c]} for c in CAMS}
    depth = {c: {i: img for i, _, img in ad.iter_frames("board", c.replace("/color", "/depth")) if i in need_d[c]}
             for c in CAMS}
    return meta, sync, colour, depth


def placements(sync, colour, spec: BoardSpec, min_move_px: float = 2.0):
    """Detections at synchronised times; consecutive times whose reference-camera corners moved < ``min_move_px``
    (median) are the same stationary placement and are not counted again."""
    det = {c: {} for c in CAMS}
    last = None
    for p in sorted(sync):
        views = {c: detect_board(colour[c][sync[p][c][0]], spec) for c in CAMS}
        ids0, uv0 = views[CAMS[0]]
        if last is not None and len(ids0) and len(last[0]):
            common = np.intersect1d(ids0, last[0])
            if len(common) >= 8:
                a = uv0[np.searchsorted(ids0, common)] if np.all(np.diff(ids0) > 0) else \
                    np.stack([uv0[list(ids0).index(i)] for i in common])
                b = np.stack([last[1][list(last[0]).index(i)] for i in common])
                if np.median(np.linalg.norm(a - b, axis=1)) < min_move_px:
                    continue
        if len(ids0) >= 8:
            last = (ids0, uv0)
        for c, (ids, uv) in views.items():
            if len(ids) >= 8:
                det[c][p] = (ids, uv)
    return det


def triangulate(uvs: dict, Ts: dict, Ks: dict) -> np.ndarray:
    """Linear (DLT) multi-view triangulation of one point from {camera: uv}."""
    A = []
    for c, uv in uvs.items():
        P = Ks[c] @ Ts[c][:3]
        A.append(uv[0] * P[2] - P[0])
        A.append(uv[1] * P[2] - P[1])
    X = np.linalg.svd(np.asarray(A))[2][-1]
    return X[:3] / X[3]


def rig_checks(rig: dict, det: dict, board: dict, K: dict, spec: BoardSpec) -> dict:
    T = {c: np.asarray(rig["state"]["T_cam_base"][c]) for c in CAMS}
    ref = CAMS[0]
    out = {"relative": {}, "triangulation": {}}
    for c in CAMS[1:]:
        rig_rel = T[c] @ np.linalg.inv(T[ref])
        b_rel = board["T_cam_ref"][c]
        rot, _ = pose_distance(rig_rel, b_rel)
        tb, tr = b_rel[:3, 3], rig_rel[:3, 3]
        out["relative"][c] = {"rotation_diff_deg": float(rot),
                              "baseline_dir_diff_deg": float(np.degrees(np.arccos(np.clip(
                                  tb @ tr / np.linalg.norm(tb) / np.linalg.norm(tr), -1, 1)))),
                              "baseline_rig_m": float(np.linalg.norm(tr)), "baseline_board_m": float(np.linalg.norm(tb)),
                              "baseline_ratio_rig_over_board": float(np.linalg.norm(tr) / np.linalg.norm(tb)),
                              "translation_diff_m": float(np.linalg.norm(tr - tb))}
    obj = spec.object_points()
    err = {c: [] for c in CAMS}
    pitch = []
    for p in sorted({q for c in CAMS for q in det[c]}):
        views = {c: det[c][p] for c in CAMS if p in det[c]}
        if len(views) < 2:
            continue
        ids_all = sorted(set.intersection(*[set(v[0].tolist()) for v in views.values()]))
        if len(ids_all) < 8:
            continue
        X = {}
        for i in ids_all:
            uvs = {c: v[1][list(v[0]).index(i)] for c, v in views.items()}
            # continuous-pixel K (pixel centres at +.5) is the repository convention for both detections and K
            X[i] = triangulate(uvs, T, K)
            if any((T[c][:3, :3] @ X[i] + T[c][:3, 3])[2] <= 0.05 for c in uvs):
                del X[i]
                continue
            for c, uv in uvs.items():
                pc = T[c][:3, :3] @ X[i] + T[c][:3, 3]
                pr = (K[c] @ pc)[:2] / pc[2]
                err[c].append(float(np.linalg.norm(pr - uv)))
        nx = spec.squares_x - 1
        for i in ids_all:  # neighbouring inner corners along x and y
            for j in (i + 1, i + nx):
                if j in X and (j != i + 1 or (i % nx) != nx - 1):
                    pitch.append(float(np.linalg.norm(X[i] - X[j])))
    out["triangulation"] = {"reproj_median_px": {c: float(np.median(v)) if v else None for c, v in err.items()},
                            "reproj_p90_px": {c: float(np.percentile(v, 90)) if v else None for c, v in err.items()},
                            "pitch_median_m": float(np.median(pitch)) if pitch else None,
                            "pitch_nominal_m": spec.square_m,
                            "pitch_ratio": float(np.median(pitch) / spec.square_m) if pitch else None,
                            "n_pitch": len(pitch)}
    return out


def depth_scale(meta, det, sync, depth, board, K, spec) -> dict:
    """Sensor depth at the board corners (depth-sensor grid, factory T_color_depth) vs the board-pose depth, using the
    depth frame paired with each camera's own synchronised colour frame."""
    from s4d.realworld.geometry import invert_T

    obj = spec.object_points()
    out = {}
    for c in CAMS:
        did = c.replace("/color", "/depth")
        Kd = meta.stream(did).intrinsics.K()
        scale = meta.stream(did).depth_scale_m
        T_d_c = invert_T(np.asarray(meta.rig_rgbd(c.split("/")[0]).T_color_depth))
        ratios = []
        for p, (ids, _) in det[c].items():
            if p not in board["T_ref_board"]:
                continue
            T_cb = board["T_cam_ref"][c] @ board["T_ref_board"][p]
            k = sync[p][c][1]
            if k not in depth[c]:
                continue
            Pd = (T_d_c @ T_cb @ np.c_[obj[ids], np.ones(len(ids))].T).T[:, :3]
            uv = (Kd @ Pd.T).T
            uv = uv[:, :2] / uv[:, 2:]
            D = depth[c][k].astype(float) * scale
            jj, ii = np.floor(uv[:, 0]).astype(int), np.floor(uv[:, 1]).astype(int)
            ok = (ii >= 0) & (jj >= 0) & (ii < D.shape[0]) & (jj < D.shape[1])
            meas = D[ii[ok], jj[ok]]
            good = meas > 0
            ratios += list(meas[good] / Pd[ok][good, 2])
        out[c] = {"sensor_over_board_median": float(np.median(ratios)) if ratios else None, "n": len(ratios)}
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--board", required=True)
    ap.add_argument("--rig-result", action="append", required=True, help="TASK=frozen result json")
    ap.add_argument("--fs-sensor", default=None, help="json {task: {camera: FS/sensor median ratio}}")
    ap.add_argument("--reference", action="append", default=[], help="TASK=fixedness reference npz")
    ap.add_argument("--square-m", type=float, default=0.030)
    ap.add_argument("--every", type=int, default=5)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    spec = BoardSpec.zivid_cb01(a.square_m)
    meta, sync, colour, depth = load_board(Path(a.board), a.every)
    K = {c: meta.stream(c).intrinsics.K() for c in CAMS}
    det = placements(sync, colour, spec)
    cfg = BoardRigConfig()
    board = solve_board(det, K, spec, cfg, CAMS[0])
    out = {"detections": {c: len(v) for c, v in det.items()}, "board_reproj_rms_px": board["reproj_rms_px"],
           "board_views": len(board["views"]), "placements": len(board["T_ref_board"]),
           "loo_spread": leave_one_out_spread(det, K, spec, cfg, CAMS[0], board), "square_m": a.square_m,
           "board_T_cam_ref": {c: np.asarray(T).tolist() for c, T in board["T_cam_ref"].items()},
           "depth": depth_scale(meta, det, sync, depth, board, K, spec), "tasks": {},
           "absolute_base_frame": "not testable: the board is not attached to the robot"}
    # fixedness of the board recording vs each task reference
    from s4d.realworld.calibration_rig import FixednessConfig, image_motion

    fcfg = FixednessConfig()
    refs = dict(x.split("=", 1) for x in a.reference)
    for task, path in refs.items():
        z = np.load(path)
        res = {}
        for c in CAMS:
            key = c.replace("/", "__")
            grey = np.median(np.stack([f.astype(np.float32).mean(-1) for f in colour[c].values()]), 0)
            res[c] = image_motion(z[f"{key}__grey"], z[f"{key}__dynamic"], grey, np.zeros_like(z[f"{key}__dynamic"]),
                                  K[c], fcfg)
        out.setdefault("fixedness", {})[task] = res
    fs_sensor = json.loads(Path(a.fs_sensor).read_text()) if a.fs_sensor else {}
    for spec_s in a.rig_result:
        task, path = spec_s.split("=", 1)
        rig = json.loads(Path(path).read_text())
        chk = rig_checks(rig, det, board, K, spec)
        alpha = rig["state"]["alpha"]
        dep = {}
        for c in CAMS:
            s = out["depth"][c]["sensor_over_board_median"]
            r = (fs_sensor.get(task) or {}).get(c)
            dep[c] = {"rig_depth_scale_exp_alpha": float(np.exp(alpha[c])),
                      "board_implied_fs_correction": (1.0 / (s * r)) if (s and r) else None,
                      "board_implied_sensor_correction": (1.0 / s) if s else None}
        chk["depth_scale"] = dep
        out["tasks"][task] = chk
    Path(a.out).write_text(json.dumps(out, indent=1, default=float))
    print(json.dumps({t: {"relative": v["relative"], "pitch_ratio": v["triangulation"]["pitch_ratio"]}
                      for t, v in out["tasks"].items()}, indent=1, default=float))


if __name__ == "__main__":
    main()
