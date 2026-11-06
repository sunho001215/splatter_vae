"""Board-based shared calibration of the fixed RGB-D cameras (calibration study 2026-11-06, board path).

The robot-only evidence leaves each camera's orientation ambiguous (near-equivalent rotations about axes 0.2-0.9 m in
front of the camera) and biased by per-link model/depth errors; the static scene is too texture-poor for cross-camera
correspondences. A planar ChArUco board seen by all cameras at several placements gives 2D reprojection constraints of
known geometry that fix the RELATIVE camera poses independently of the robot model and of depth scale. The robot then
only has to place the rigid camera triplet in the robot base frame (6 DoF, all three views jointly).

Pipeline (``build_board_rig``):
1. corner detections per camera and placement (``detect_charuco``, OpenCV with aruco - run in a worker env that has it);
2. per-camera planar PnP, then a joint bundle adjustment of the cameras relative to a reference camera + one board pose
   per placement (reprojection residuals); leave-one-placement-out replicates give the empirical relative-pose spread;
3. registration of the rigid triplet to the robot base with the stage's robot-depth residuals of all cameras jointly
   (multi-start; episode-half replicates and configuration groups give the empirical spread);
4. held-out episode validation and the camera-fixedness check of the capture against the rig reference are done by
   ``calibration_rig`` (``validate_episode`` / ``episode_motion``); the result is published as a rig document.
Per-camera depth scale against the board (``board_depth_scale``) is reported, not applied.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
from scipy.optimize import least_squares

from s4d.realworld.calibration import CalibrationConfig, _depth_gates, _frame_stats, _residuals, pose_distance
from s4d.realworld.geometry import invert_T, pose_from_vec, project, transform_points, unproject, vec_from_pose


@dataclass
class BoardSpec:
    """ChArUco board: ``squares_x`` x ``squares_y`` chessboard squares of ``square_m``, ArUco markers of ``marker_m``
    in the white squares (OpenCV convention). Inner-corner ids are row-major over the (squares_x-1) x (squares_y-1)
    grid; the board frame has x along squares_x, y along squares_y, z out of the board (OpenCV)."""

    squares_x: int = 8  # realworld-ops/calib/board-capture-protocol.md: 8 x 6 squares of 45 mm on A3, 34 mm markers
    squares_y: int = 6
    square_m: float = 0.045  # replace by the MEASURED printed square size
    marker_m: float = 0.034
    dictionary: str = "DICT_5X5_100"

    def object_points(self) -> np.ndarray:
        nx, ny = self.squares_x - 1, self.squares_y - 1
        j, i = np.mgrid[0:ny, 0:nx]
        return np.stack([(i.ravel() + 1) * self.square_m, (j.ravel() + 1) * self.square_m, np.zeros(nx * ny)], 1)


@dataclass
class BoardRigConfig:
    """Acceptance (same evidence-based tolerances as ``calibration_rig.RigConfig``: 1 training px at 128x128 for a scene
    at 1.3 m = 0.35 deg / 8 mm) plus board-fit gates."""

    min_placements: int = 3
    min_corners: int = 8  # per camera and placement to use that view
    max_reproj_rms_px: float = 0.5
    max_board_spread_rot_deg: float = 0.1  # leave-one-placement-out relative poses
    max_board_spread_centre_m: float = 0.002
    max_spread_rot_deg: float = 0.35  # robot registration: multistart / episode halves / configuration groups
    max_spread_centre_m: float = 0.008
    registration_starts: int = 7
    start_rot_deg: float = 3.0
    start_trans_m: float = 0.03
    outer_iters: int = 3


def detect_charuco(image: np.ndarray, spec: BoardSpec) -> tuple[np.ndarray, np.ndarray]:
    """(corner ids (M,), pixels (M,2) in the s4d continuous convention) of one image; OpenCV >= 4.7 with aruco."""
    import cv2

    aruco = cv2.aruco
    dictionary = aruco.getPredefinedDictionary(getattr(aruco, spec.dictionary))
    board = aruco.CharucoBoard((spec.squares_x, spec.squares_y), spec.square_m, spec.marker_m, dictionary)
    grey = image if image.ndim == 2 else cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    corners, ids, _, _ = aruco.CharucoDetector(board).detectBoard(grey)
    if ids is None or corners is None:
        return np.zeros(0, int), np.zeros((0, 2))
    return ids.ravel().astype(int), corners.reshape(-1, 2).astype(np.float64) + 0.5


def planar_pnp(obj: np.ndarray, uv: np.ndarray, K: np.ndarray) -> np.ndarray:
    """T_cam_board of a planar target (z=0 points) from >= 4 correspondences: homography (DLT on normalised
    coordinates) -> rotation/translation, refined by reprojection least squares."""
    xy = unproject(uv, np.ones(len(uv)), K)[:, :2]
    A = []
    for (X, Y, _), (x, y) in zip(obj, xy):
        A.append([X, Y, 1, 0, 0, 0, -x * X, -x * Y, -x])
        A.append([0, 0, 0, X, Y, 1, -y * X, -y * Y, -y])
    H0 = np.linalg.svd(np.asarray(A))[2][-1].reshape(3, 3)
    best = None
    for H in (H0, -H0):  # the homography sign is free: keep the one that puts the OBSERVED corners in front
        s = 2.0 / (np.linalg.norm(H[:, 0]) + np.linalg.norm(H[:, 1]))
        r1, r2, t = H[:, 0] * s, H[:, 1] * s, H[:, 2] * s
        R = np.stack([r1, r2, np.cross(r1, r2)], 1)
        U, _, Vt = np.linalg.svd(R)
        R = U @ np.diag([1, 1, np.linalg.det(U @ Vt)]) @ Vt
        T = np.eye(4)
        T[:3, :3], T[:3, 3] = R, t
        front = (transform_points(T, obj)[:, 2] > 0).sum()
        if best is None or front > best[0]:
            best = (front, T)
    T0 = best[1]

    def fun(v):
        p, _ = project(transform_points(pose_from_vec(v), obj), K)
        r = (p - uv).ravel()
        return np.where(np.isfinite(r), r, 1e3)  # a corner behind the camera is maximally wrong

    T = pose_from_vec(least_squares(fun, vec_from_pose(T0), method="lm").x)
    if not np.isfinite(T).all() or (transform_points(T, obj)[:, 2] <= 0).any():
        raise ValueError("planar PnP failed (corners behind the camera)")
    return T


def solve_board(detections: dict, K: dict, spec: BoardSpec, cfg: BoardRigConfig, reference: str,
                placements: list | None = None) -> dict:
    """Joint bundle adjustment. ``detections[camera][placement] = (ids, uv)``, ``K[camera]``. Unknowns: T_cam_ref of
    every non-reference camera, T_ref_board of every placement. Returns poses, per-camera reprojection RMS and the
    number of views used."""
    obj = spec.object_points()
    cams = sorted(detections)
    pl = sorted(placements if placements is not None else {p for c in cams for p in detections[c]})
    views = [(c, p) for c in cams for p in pl if p in detections[c] and len(detections[c][p][0]) >= cfg.min_corners]
    if reference not in cams:
        raise ValueError(f"reference camera {reference} has no detections")
    T_cb = {}
    for c, p in views:
        try:
            T_cb[(c, p)] = planar_pnp(obj[detections[c][p][0]], detections[c][p][1], K[c])
        except ValueError:
            continue
    views = [v for v in views if v in T_cb]
    ref_pl = [p for p in pl if (reference, p) in T_cb]
    if not ref_pl:
        raise ValueError("the reference camera sees no placement")
    others = [c for c in cams if c != reference]
    init_rel = {}
    for c in others:
        shared = [p for p in ref_pl if (c, p) in T_cb]
        if not shared:
            raise ValueError(f"camera {c} shares no placement with the reference camera")
        init_rel[c] = T_cb[(c, shared[0])] @ invert_T(T_cb[(reference, shared[0])])
    x0 = np.concatenate([vec_from_pose(init_rel[c]) for c in others] + [vec_from_pose(T_cb[(reference, p)]) for p in ref_pl])

    def unpack(x):
        rel = {reference: np.eye(4), **{c: pose_from_vec(x[6 * i:6 * i + 6]) for i, c in enumerate(others)}}
        board = {p: pose_from_vec(x[6 * (len(others) + k):6 * (len(others) + k) + 6]) for k, p in enumerate(ref_pl)}
        return rel, board

    used = [(c, p) for c, p in views if p in ref_pl]

    def fun(x):
        rel, board = unpack(x)
        out = []
        for c, p in used:
            ids, uv = detections[c][p]
            pr, _ = project(transform_points(rel[c] @ board[p], obj[ids]), K[c])
            r = (pr - uv).ravel()
            out.append(np.where(np.isfinite(r), r, 1e3))
        return np.concatenate(out)

    res = least_squares(fun, x0, loss="soft_l1", f_scale=1.0)
    rel, board = unpack(res.x)
    r = res.fun.reshape(-1, 2)
    rms, k = {}, 0
    for c, p in used:
        n = len(detections[c][p][0])
        rms.setdefault(c, []).append(r[k:k + n])
        k += n
    return {"T_cam_ref": rel, "T_ref_board": board, "views": used,
            "reproj_rms_px": {c: float(np.sqrt((np.concatenate(v) ** 2).sum(1).mean())) for c, v in rms.items()}}


def leave_one_out_spread(detections, K, spec, cfg, reference, full) -> dict:
    """Max pose change of every camera relative to the reference when one placement is left out."""
    pl = sorted({p for c in detections for p in detections[c]})
    worst = {c: [0.0, 0.0] for c in full["T_cam_ref"]}
    for drop in pl:
        try:
            sub = solve_board(detections, K, spec, cfg, reference, [p for p in pl if p != drop])
        except ValueError as exc:
            return {c: {"max_rot_deg": None, "max_centre_m": None, "failed": f"without {drop}: {exc}"} for c in worst}
        for c, T in sub["T_cam_ref"].items():
            d = pose_distance(T, full["T_cam_ref"][c])
            worst[c] = [max(worst[c][0], d[0]), max(worst[c][1], d[1])]
    return {c: {"max_rot_deg": v[0], "max_centre_m": v[1]} for c, v in worst.items()}


def board_depth_scale(detections, K, spec, T_cam_board: dict, depth: dict) -> dict:
    """Per camera: median ratio of the measured depth at the board corners to the board-pose depth (1 = unbiased)."""
    obj = spec.object_points()
    out = {}
    for (c, p), T in T_cam_board.items():
        if p not in depth.get(c, {}):
            continue
        ids, uv = detections[c][p]
        z = transform_points(T, obj[ids])[:, 2]
        d = depth[c][p]
        px = np.floor(uv).astype(int)
        ok = (px[:, 0] >= 0) & (px[:, 0] < d.shape[1]) & (px[:, 1] >= 0) & (px[:, 1] < d.shape[0])
        zm = np.zeros(len(uv))
        zm[ok] = d[px[ok, 1], px[ok, 0]]
        ok &= zm > 0
        out.setdefault(c, []).extend((zm[ok] / z[ok]).tolist())
    return {c: {"median_ratio": float(np.median(v)), "n": len(v)} for c, v in out.items() if v}


def register_to_robot(frames: dict, K: dict, rel: dict, reference: str, cfg: CalibrationConfig,
                      T_ref_base0: np.ndarray, outer_iters: int = 3) -> tuple[np.ndarray, float]:
    """6-DoF pose of the rigid camera set in the robot base: T_c_base = T_c_ref @ T_ref_base, robot-depth residuals of
    every camera's frames together (the stage's gates and residuals, rgbd evidence). Returns (T_ref_base, fit score =
    median over cameras of the median per-frame |depth residual|)."""
    rng_seed = cfg.seed
    T = np.asarray(T_ref_base0, float)
    gates = {}
    res = None
    for _ in range(max(1, outer_iters)):
        for c in frames:
            gates[c] = _depth_gates(rel[c] @ T, frames[c], K[c], cfg, np.random.default_rng(rng_seed))

        def fun(v):
            Tb = pose_from_vec(v)
            return np.concatenate([_residuals(vec_from_pose(rel[c] @ Tb), frames[c], K[c], cfg, gates[c], True)
                                   for c in frames])

        res = least_squares(fun, vec_from_pose(T), loss="cauchy", f_scale=1.0, method="trf", x_scale=np.full(6, 0.01))
        T = pose_from_vec(res.x)
    scores = []
    for c in frames:
        vals = [r["depth_abs_median_m"] for r in _frame_stats(rel[c] @ T, frames[c], K[c], cfg)
                if r.get("depth_abs_median_m") is not None]
        scores.append(float(np.median(vals)) if vals else float("inf"))
    return T, float(np.median(scores)) if np.isfinite(res.x).all() else float("inf")


def registration_starts(T0: np.ndarray, cfg: BoardRigConfig, seed: int = 0) -> list:
    from scipy.spatial.transform import Rotation

    rng = np.random.default_rng(seed)
    out = [np.asarray(T0, float)]
    for _ in range(cfg.registration_starts - 1):
        ax = rng.normal(size=3)
        tr = rng.normal(size=3)
        D = np.eye(4)
        angle = np.radians(rng.uniform(0, cfg.start_rot_deg))
        D[:3, :3] = Rotation.from_rotvec(ax / np.linalg.norm(ax) * angle).as_matrix()
        D[:3, 3] = tr / np.linalg.norm(tr) * rng.uniform(0, cfg.start_trans_m)
        out.append(D @ out[0])
    return out


def build_board_rig(detections: dict, K: dict, spec: BoardSpec, robot_frames: dict, cfg: CalibrationConfig,
                    T_ref_base0: np.ndarray, *, reference: str, board_cfg: BoardRigConfig | None = None,
                    replicate_frames: dict | None = None, group_frames: dict | None = None,
                    heldout_frames: dict | None = None, capture_motion: dict | None = None,
                    episode_motion: dict | None = None, name: str = "board-rig") -> dict:
    """Shared calibration from a board capture + robot evidence. ``robot_frames[camera]`` = CalibrationFrames pooled over
    episodes; ``replicate_frames`` / ``group_frames``: {label: {camera: frames}} for episode-half replicates and
    configuration groups (e.g. tasks); ``capture_motion[camera]``: ``calibration_rig.episode_motion`` of the capture
    against the rig reference (the rig did not move since); ``heldout_frames``: {episode: {camera: frames}} of episodes
    not used anywhere above, each judged by ``calibration_rig.validate_episode``; ``episode_motion``: {episode:
    {camera: episode_motion result}} for EVERY robot-evidence episode (pool and held out). Returns a rig document body
    (``calibration_rig.publish_rig`` adds the reference); a camera is "validated" only if every gate passes, else
    "held"."""
    bc = board_cfg or BoardRigConfig()
    reasons_all = []
    board = solve_board(detections, K, spec, bc, reference)
    used = board["views"]
    # placements that count: seen (with enough corners) by the reference and at least one other camera
    usable = sorted({p for _, p in used if sum(1 for c2, p2 in used if p2 == p) >= 2})
    if len(usable) < bc.min_placements:
        reasons_all.append(f"{len(usable)} usable board placements < {bc.min_placements}")
    loo = leave_one_out_spread(detections, K, spec, bc, reference, board)
    rel = board["T_cam_ref"]

    def registration(fr):
        """all starts on one frame set; None if any camera lacks frames or no solve is finite."""
        if not fr or any(not fr.get(c) for c in rel):
            return None
        runs = [register_to_robot(fr, K, rel, reference, cfg, T0, bc.outer_iters)
                for T0 in registration_starts(T_ref_base0, bc)]
        runs = [(T, sc) for T, sc in runs if np.isfinite(T).all() and np.isfinite(sc)]
        if not runs:
            return None
        best = min(runs, key=lambda r: r[1])
        return best, [T for T, sc in runs if sc <= best[1] + cfg.near_best_tol_m]

    full = registration(robot_frames)
    if full is None:
        raise ValueError("robot registration failed: no finite solve on the pooled robot frames")
    (T_rb, score), near = full
    subsets, sub_fail = {}, []
    for kind, coll in (("replicate", replicate_frames or {}), ("group", group_frames or {})):
        if len(coll) < 2:
            reasons_all.append(f"need >= 2 {kind} frame sets for the empirical spread, got {len(coll)}")
        for label, fr in coll.items():
            r = registration(fr)
            if r is None:
                sub_fail.append(f"{kind}:{label}")
            else:
                subsets[f"{kind}:{label}"] = r
    if sub_fail:
        reasons_all.append(f"registration subsets without a finite solve: {sub_fail}")
    doc = {"name": name, "method": "board relative poses + joint robot registration", "board": asdict(spec),
           "board_config": asdict(bc), "placements_usable": usable, "reference_camera": reference,
           "board_reproj_rms_px": board["reproj_rms_px"], "board_leave_one_out": loo,
           "registration_fit_m": score, "cameras": {}}
    for c in rel:
        T_c = rel[c] @ T_rb
        reasons = list(reasons_all)
        rms = board["reproj_rms_px"].get(c)
        if rms is None or not rms <= bc.max_reproj_rms_px:
            reasons.append(f"board reprojection RMS {rms} px > {bc.max_reproj_rms_px}")
        n_pl = sum(1 for c2, p in used if c2 == c and p in usable)
        if n_pl < bc.min_placements:
            reasons.append(f"camera sees {n_pl} usable placements < {bc.min_placements}")
        lo = loo.get(c, {})
        if lo.get("failed") or lo.get("max_rot_deg") is None or not (
                lo["max_rot_deg"] <= bc.max_board_spread_rot_deg and lo["max_centre_m"] <= bc.max_board_spread_centre_m):
            reasons.append(f"leave-one-placement-out spread {lo}")
        # every solution that is equally supported (near-best starts of the full set AND of every subset) and every
        # subset optimum must agree with the chosen pose
        cands = [rel[c] @ T for T in near] + [rel[c] @ T for _, ns in subsets.values() for T in ns] + \
            [rel[c] @ b[0] for b, _ in subsets.values()]
        allT = [T_c] + cands
        d = [pose_distance(a, b) for i, a in enumerate(allT) for b in allT[i + 1:]]
        spread = {"solves": len(allT), "near_best_starts": len(near),
                  "max_pair_rot_deg": max((x[0] for x in d), default=0.0),
                  "max_pair_centre_m": max((x[1] for x in d), default=0.0)}
        if not (spread["max_pair_rot_deg"] <= bc.max_spread_rot_deg
                and spread["max_pair_centre_m"] <= bc.max_spread_centre_m):
            reasons.append(f"robot registration spread {spread['max_pair_rot_deg']:.3f} deg / "
                           f"{spread['max_pair_centre_m'] * 1000:.1f} mm > {bc.max_spread_rot_deg} deg / "
                           f"{bc.max_spread_centre_m * 1000:.0f} mm")
        val = []
        if not heldout_frames:
            reasons.append("no held-out episodes")
        for ep, fr in (heldout_frames or {}).items():
            from s4d.realworld.calibration_rig import validate_episode

            m = (episode_motion or {}).get(ep, {}).get(c, {"fixed": False, "reason": "fixedness not measured"})
            r = validate_episode(c, fr.get(c, []), K[c], cfg, {"T_cam_base": T_c.tolist(), "status": "validated"}, m)
            val.append({"episode": ep, "status": r.status, "reasons": r.reasons,
                        "depth_abs_median_m": r.diagnostics.get("heldout_depth_abs_median_m"),
                        "explained_fraction": r.diagnostics.get("explained_fraction_heldout")})
        failed = [v["episode"] for v in val if v["status"] != "accepted"]
        if failed:
            reasons.append(f"{len(failed)}/{len(val)} held-out episodes reject the pose: {failed[:5]}")
        unfixed = sorted(ep for ep, mm in (episode_motion or {}).items() if not mm.get(c, {}).get("fixed"))
        if episode_motion is None:
            reasons.append("fixedness of the robot-evidence episodes not measured")
        elif unfixed:
            reasons.append(f"robot-evidence episodes not verified fixed: {unfixed[:5]}")
        motion = (capture_motion or {}).get(c)
        if motion is None or not motion.get("fixed"):
            why = None if motion is None else motion.get("reason")
            reasons.append(f"capture not verified fixed against the rig reference: {why}")
        doc["cameras"][c] = {"T_cam_base": T_c.tolist(), "T_cam_ref": rel[c].tolist(),
                             "status": "held" if reasons else "validated", "reasons": reasons,
                             "empirical_spread": spread, "provenance": {"rig": name, "method": "board"},
                             "fixedness_capture": motion, "validation": {"held_out": val, "failed": failed}}
    return doc


# ---------------------------------------------------------------- capture processing (board-capture-protocol.md)


def detect_capture(root, episode_id: str, spec: BoardSpec, frames: int = 7) -> dict:
    """Corner detections of one placement recording (board and robot static): ``frames`` evenly spaced colour frames per
    camera, per-corner median over the frames where it was detected (needs OpenCV with aruco)."""
    from s4d.realworld.pipeline import camera_views, load_meta, read_frames, to_model_rgb

    meta = load_meta(root, episode_id)
    out = {}
    for view in camera_views(meta):
        cam = view.stream_id
        n = len(meta.stream_timestamps_ns[cam])
        pick = np.linspace(0, n - 1, min(frames, n)).round().astype(int)
        per = {}
        for img in to_model_rgb(read_frames(root, episode_id, cam, pick), meta.stream(cam)):
            ids, uv = detect_charuco(np.asarray(img), spec)
            for i, p in zip(ids, uv):
                per.setdefault(int(i), []).append(p)
        keep = [i for i, v in per.items() if len(v) >= max(1, len(pick) // 2)]
        if keep:
            out[cam] = (np.asarray(sorted(keep)), np.stack([np.median(per[i], 0) for i in sorted(keep)]))
    return out


def board_rig_from_capture(captures: list, robot_episodes: list, reference_episode: tuple, out_path, *, spec: BoardSpec,
                           cfg: CalibrationConfig, reference_camera: str, T_ref_base0: np.ndarray,
                           detections: dict | None = None, board_cfg: BoardRigConfig | None = None,
                           frames_per_episode: int = 6, every: int = 2, max_frames: int | None = 200,
                           depth_stage: str = "sensor_depth", robot_mask_stage: str = "sam3_masks",
                           name: str = "board-rig", seed: int = 0) -> dict:
    """End to end: ``captures`` [(root, episode_id)] one recording per placement; ``robot_episodes`` [(root, id)] in
    recording order, already through sensor depth + SAM 3 (every other one is held out; the task prefix of the id is
    the configuration group); ``reference_episode`` (root, id) = the rig reference of the fixedness test (e.g. the first
    11-05 episode); ``T_ref_base0`` initial pose of the reference camera in the robot base (e.g. its calibrate@1 pose).
    ``detections`` {camera: {placement: (ids, uv)}} may be passed precomputed (``detect_capture`` in an OpenCV env).
    Publishes the rig document + reference via ``calibration_rig.publish_rig``."""
    import h5py

    from s4d.realworld import io as rio
    from s4d.realworld.calibration import CalibrationFrame
    from s4d.realworld.calibration_rig import FixednessConfig, episode_motion, publish_rig, static_image
    from s4d.realworld.calibration_stage import camera_arrays
    from s4d.realworld.pipeline import RobotGeometry, camera_views, load_meta, read_frames, to_model_rgb

    fix = FixednessConfig()
    for label, eps in (("captures", captures), ("robot_episodes", robot_episodes)):
        if len({e for _, e in eps}) != len(eps):
            raise ValueError(f"{label}: episode ids must be unique")
    if {e for _, e in captures} & {e for _, e in robot_episodes}:
        raise ValueError("a capture episode cannot also be robot evidence")
    if len(robot_episodes[0::2]) < 4 or not robot_episodes[1::2]:
        raise ValueError("need >= 8 robot-evidence episodes (>= 4 pool, the rest held out)")
    if detections is None:
        detections = {}
        for k, (r, e) in enumerate(captures):
            for cam, d in detect_capture(r, e, spec).items():
                detections.setdefault(cam, {})[f"p{k}"] = d
    r0, e0 = reference_episode
    meta0 = load_meta(r0, e0)
    cams = [v.stream_id for v in camera_views(meta0)]
    K = {c: meta0.stream(c).intrinsics.K() for c in cams}
    ref_arrays, ref = {}, {}
    for cam in cams:
        n = len(meta0.stream_timestamps_ns[cam])
        pick = np.linspace(0, n - 1, min(fix.frames, n)).round().astype(int)
        rgb = to_model_rgb(read_frames(r0, e0, cam, pick), meta0.stream(cam))
        with h5py.File(rio.stage_dir(r0, e0, robot_mask_stage) / f"{rio.safe_name(cam)}.h5", "r") as f:
            sem = f["semantic"][()]
        ref[cam] = static_image(rgb, np.isin(sem, [1, 2]), fix)
        key = cam.replace("/", "__")
        ref_arrays[f"{key}__grey"], ref_arrays[f"{key}__dynamic"] = ref[cam]
    # fixedness of the capture: every placement recording against the 11-05 reference; the board (convex hull of its
    # detected corners, dilated) is a new object and is masked as dynamic
    motion = {}
    for k, (r, e) in enumerate(captures):
        meta = load_meta(r, e)
        for cam in cams:
            n = len(meta.stream_timestamps_ns[cam])
            pick = np.linspace(0, n - 1, min(fix.frames, n)).round().astype(int)
            rgb = to_model_rgb(read_frames(r, e, cam, pick), meta.stream(cam))
            dyn = np.zeros(rgb.shape[:3], bool)
            d = detections.get(cam, {}).get(f"p{k}")
            if d is not None and len(d[1]) >= 3:
                lo, hi = np.floor(d[1].min(0) - 60).astype(int), np.ceil(d[1].max(0) + 60).astype(int)
                dyn[:, max(lo[1], 0):hi[1], max(lo[0], 0):hi[0]] = True
            m = episode_motion(ref[cam][0], ref[cam][1], rgb, dyn, K[cam], fix)
            if not m["fixed"] or cam not in motion:
                motion[cam] = m if not m["fixed"] else motion.get(cam, m)
    # robot evidence: same intrinsics as the reference, and every episode verified unmoved against it
    metas = {e: load_meta(r, e) for r, e in robot_episodes}
    for e, m in list(metas.items()) + [(e, load_meta(r, e)) for r, e in captures]:
        for c in cams:
            if not np.allclose(m.stream(c).intrinsics.K(), K[c]) or \
                    (m.stream(c).intrinsics.height, m.stream(c).intrinsics.width) != \
                    (meta0.stream(c).intrinsics.height, meta0.stream(c).intrinsics.width):
                raise ValueError(f"{e} {c}: intrinsics/image size differ from the reference episode")
    geoms = {e: RobotGeometry(metas[e]) for e in metas}
    ep_motion = {}
    for r, e in robot_episodes:
        for c in cams:
            n = len(metas[e].stream_timestamps_ns[c])
            pick = np.linspace(0, n - 1, min(fix.frames, n)).round().astype(int)
            rgb = to_model_rgb(read_frames(r, e, c, pick), metas[e].stream(c))
            with h5py.File(rio.stage_dir(r, e, robot_mask_stage) / f"{rio.safe_name(c)}.h5", "r") as f:
                sem = f["semantic"][()]
            ep_motion.setdefault(e, {})[c] = episode_motion(ref[c][0], ref[c][1], rgb, np.isin(sem, [1, 2]), K[c], fix)

    def stage_pick(e, c, per):
        n = len(metas[e].stream_timestamps_ns[c])
        idx = list(range(0, n, every))
        if max_frames is not None and len(idx) > max_frames:
            idx = [idx[k] for k in np.linspace(0, len(idx) - 1, int(max_frames)).round().astype(int)]
        if per is not None:
            idx = [idx[k] for k in np.linspace(0, len(idx) - 1, min(per, len(idx))).round().astype(int)]
        return idx

    def frames_of(subset, per):
        out = {c: [] for c in cams}
        for r, e in subset:
            rec = rio.read_record(rio.stage_dir(r, e, depth_stage))
            for c in cams:
                idx = stage_pick(e, c, per)
                a, *_ = camera_arrays(r, e, metas[e], geoms[e], c, idx, depth_rec=rec, depth_stage=depth_stage,
                                      robot_mask_stage=robot_mask_stage, cfg=cfg)
                out[c] += [CalibrationFrame(int(a["t_index"][k]), int(a["timestamp_ns"][k]), a["landmarks_base"][k],
                                            a["landmarks_uv"][k], a["landmarks_conf"][k], a["surface_base"][k],
                                            a["flange_base"][k], a["depth"][k], a["robot_mask"][k])
                           for k in range(len(a["t_index"]))]
        return out

    pool, held = robot_episodes[0::2], robot_episodes[1::2]
    rng = np.random.default_rng(seed)
    halves = [sorted(rng.choice(len(pool), max(2, len(pool) // 2), replace=False).tolist()) for _ in range(2)]
    groups = sorted({e.rsplit("-", 1)[0] for _, e in pool})
    body = build_board_rig(
        detections, K, spec, frames_of(pool, frames_per_episode), cfg, T_ref_base0, reference=reference_camera,
        board_cfg=board_cfg,
        replicate_frames={f"half{i}": frames_of([pool[j] for j in h], frames_per_episode) for i, h in enumerate(halves)},
        group_frames={g: frames_of([x for x in pool if x[1].startswith(g)], frames_per_episode) for g in groups}
        if len(groups) > 1 else None,
        heldout_frames={e: frames_of([(r, e)], None) for r, e in held}, capture_motion=motion,
        episode_motion=ep_motion, name=name)
    body.update({"format": "s4d.realworld.rig_calibration/1", "reference_episode": e0,
                 "pool_episodes": [e for _, e in pool], "held_out_episodes": [e for _, e in held],
                 "captures": [e for _, e in captures],
                 # frames that entered the robot registration: the stage reports them as fit, not held-out, evidence
                 "pool_frames": {e: {c: [int(i) for i in stage_pick(e, c, frames_per_episode)] for c in cams}
                                 for _, e in pool}})
    return publish_rig(out_path, body, ref_arrays)
