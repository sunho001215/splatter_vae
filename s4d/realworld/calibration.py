"""Episode-level robot-grounded extrinsic calibration (simplified from DepthWorld, arXiv 2610.08780).

One constant ``T_cam_base`` per camera per episode (cameras may move between episodes). Two steps:

1. **Visual initialisation** from 2D observations of robot landmarks (URDF link origins / marker corners / matched
   render points) whose 3D positions follow from forward kinematics: RANSAC PnP (DLT + Levenberg-Marquardt) and,
   when metric stereo depth exists, a RANSAC Kabsch fit of depth-lifted landmarks; the lower-error one wins.
2. **Robot-grounded refinement**: a Cauchy-robust least-squares over the 6-DoF extrinsic with
   (a) landmark reprojection residuals (scale ``cauchy_px``) and (b) robot-surface depth residuals: articulated
   URDF surface points (arm + gripper at the recorded gripper state) projected into the camera and compared with
   the bootstrap stereo depth (scale ``cauchy_depth_m``), gated for occlusion and re-gated in an outer loop.

What is NOT reproduced from DepthWorld (see docs): the multi-episode factor graph per physical robot, the shared
hand-eye and joint-offset (dq) corrections, the wrist camera, ext-ext RoMa correspondences and the featuremetric
term, LoMa-G render-to-image matching (here an injectable landmark source), and the GPU Schur solver.

Unreliable results are never silently accepted: every result carries ``status`` ("accepted" / "rejected"), the
reasons, and diagnostics (visibility, motion coverage, residuals on fit and held-out frames, covariance-based
uncertainty, conditioning, init-to-final change, optimiser status).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

import numpy as np
from scipy.optimize import least_squares

from s4d.realworld.geometry import invert_T, make_T, pose_from_vec, project, transform_points, unproject, vec_from_pose


@dataclass
class CalibrationFrame:
    t_index: int
    timestamp_ns: int
    landmarks_base: np.ndarray  # (M,3) FK landmark positions in the base frame
    landmarks_uv: np.ndarray  # (M,2) observed pixels (continuous convention), NaN where not observed
    landmarks_conf: np.ndarray  # (M,) in [0,1]
    surface_base: np.ndarray  # (N,3) articulated robot surface points (arm + gripper) in the base frame
    flange_base: np.ndarray  # (3,) panda_link8 origin, for motion coverage
    depth: np.ndarray | None = None  # (H,W) metric camera-z on this camera's grid, 0 invalid
    robot_mask: np.ndarray | None = None  # (H,W) bool, optional (e.g. SAM 3); used for gating + IoU diagnostics


@dataclass
class CalibrationConfig:
    image_hw: tuple = (480, 640)
    ransac_iters: int = 300
    ransac_px: float = 8.0
    cauchy_px: float = 5.0
    cauchy_depth_m: float = 0.01
    depth_term: bool = True
    depth_points_per_frame: int = 300
    occlusion_tol_m: float = 0.03
    self_occlusion_block_px: int = 4
    self_occlusion_tol_m: float = 0.01
    outer_iters: int = 3
    holdout_every: int = 5
    seed: int = 0
    # acceptance gates
    min_visible_frames: int = 5
    min_visible_fraction: float = 0.25
    min_landmark_obs: int = 12
    min_motion_extent_m: float = 0.08  # second-largest principal extent of flange positions
    min_distinct_poses: int = 4
    max_reproj_median_px: float = 3.0
    max_depth_median_m: float = 0.02
    max_heldout_reproj_median_px: float = 5.0
    max_rot_std_deg: float = 0.5
    max_trans_std_m: float = 0.01
    max_condition: float = 1e10
    min_inlier_fraction: float = 0.5


@dataclass
class CalibrationResult:
    camera_id: str
    status: str
    reasons: list
    T_cam_base: np.ndarray | None
    T_cam_base_init: np.ndarray | None
    init_method: str | None
    diagnostics: dict = field(default_factory=dict)

    def to_json(self) -> dict:
        d = asdict(self)
        for key in ("T_cam_base", "T_cam_base_init"):
            d[key] = None if d[key] is None else np.asarray(d[key]).tolist()
        return d


# ---------------------------------------------------------------- visual initialisation


def pnp_dlt(X: np.ndarray, uv: np.ndarray, K: np.ndarray) -> np.ndarray:
    """Direct linear transform PnP (>= 6 non-degenerate points) -> T_cam_base."""
    xy = unproject(uv, np.ones(len(uv)), K)[:, :2]
    A = []
    for (Xw, Yw, Zw), (x, y) in zip(X, xy):
        A.append([Xw, Yw, Zw, 1, 0, 0, 0, 0, -x * Xw, -x * Yw, -x * Zw, -x])
        A.append([0, 0, 0, 0, Xw, Yw, Zw, 1, -y * Xw, -y * Yw, -y * Zw, -y])
    _, _, Vt = np.linalg.svd(np.asarray(A))
    P = Vt[-1].reshape(3, 4)
    U, S, Vt2 = np.linalg.svd(P[:, :3])
    R = U @ Vt2
    scale = S.mean()
    t = P[:, 3] / scale
    if np.linalg.det(R) < 0:
        R, t = -R, -t
    T = make_T(R, t)
    return T  # the det(R) sign fix above also fixes cheirality (P is only known up to a signed scale)


def refine_reprojection(T: np.ndarray, X: np.ndarray, uv: np.ndarray, K: np.ndarray, scale_px: float = 5.0):
    def fun(v):
        pred, z = project(transform_points(pose_from_vec(v), X), K)
        r = pred - uv
        r[~np.isfinite(r)] = 1e3
        return r.ravel()

    res = least_squares(fun, vec_from_pose(T), loss="cauchy", f_scale=scale_px, method="trf")
    return pose_from_vec(res.x), res


def pnp_ransac(X, uv, K, iters=300, thresh_px=8.0, seed=0):
    rng = np.random.default_rng(seed)
    best_inl, best_T = None, None
    n = len(X)
    if n < 6:
        return None, np.zeros(n, bool)
    for _ in range(iters):
        idx = rng.choice(n, 6, replace=False)
        try:
            T = pnp_dlt(X[idx], uv[idx], K)
        except np.linalg.LinAlgError:
            continue
        pred, z = project(transform_points(T, X), K)
        err = np.linalg.norm(pred - uv, axis=1)
        inl = np.isfinite(err) & (err < thresh_px) & (z > 0)
        if best_inl is None or inl.sum() > best_inl.sum():
            best_inl, best_T = inl, T
    if best_inl is None or best_inl.sum() < 6:
        return None, np.zeros(n, bool)
    T = pnp_dlt(X[best_inl], uv[best_inl], K)
    T, _ = refine_reprojection(T, X[best_inl], uv[best_inl], K)
    pred, z = project(transform_points(T, X), K)
    inl = (np.linalg.norm(pred - uv, axis=1) < thresh_px) & (z > 0)
    return T, inl


def kabsch(P_src: np.ndarray, P_dst: np.ndarray) -> np.ndarray:
    """Rigid T with P_dst ~ T P_src (no scale: stereo depth is metric)."""
    cs, cd = P_src.mean(0), P_dst.mean(0)
    H = (P_src - cs).T @ (P_dst - cd)
    U, _, Vt = np.linalg.svd(H)
    D = np.diag([1, 1, np.sign(np.linalg.det(Vt.T @ U.T))])
    R = Vt.T @ D @ U.T
    return make_T(R, cd - R @ cs)


def kabsch_ransac(P_base, P_cam, iters=300, thresh_m=0.02, seed=0):
    rng = np.random.default_rng(seed)
    n = len(P_base)
    if n < 3:
        return None, np.zeros(n, bool)
    best = None
    for _ in range(iters):
        idx = rng.choice(n, 3, replace=False)
        T = kabsch(P_base[idx], P_cam[idx])
        inl = np.linalg.norm(transform_points(T, P_base) - P_cam, axis=1) < thresh_m
        if best is None or inl.sum() > best[1].sum():
            best = (T, inl)
    if best[1].sum() < 3:
        return None, np.zeros(n, bool)
    T = kabsch(P_base[best[1]], P_cam[best[1]])
    return T, np.linalg.norm(transform_points(T, P_base) - P_cam, axis=1) < thresh_m


def bilinear_depth(depth: np.ndarray, uv: np.ndarray, max_spread_m: float = 0.02) -> tuple[np.ndarray, np.ndarray]:
    """Bilinear depth at continuous pixels; invalid unless all four neighbours are valid and within ``max_spread_m``."""
    H, W = depth.shape
    x, y = uv[:, 0] - 0.5, uv[:, 1] - 0.5
    finite = np.isfinite(x) & np.isfinite(y)
    x0 = np.floor(np.where(finite, x, -10)).astype(int)
    y0 = np.floor(np.where(finite, y, -10)).astype(int)
    inside = finite & (x0 >= 0) & (y0 >= 0) & (x0 + 1 < W) & (y0 + 1 < H)
    out = np.zeros(len(uv))
    ok = np.zeros(len(uv), bool)
    i = np.flatnonzero(inside)
    if len(i):
        xa, ya = x0[i], y0[i]
        q = np.stack([depth[ya, xa], depth[ya, xa + 1], depth[ya + 1, xa], depth[ya + 1, xa + 1]], 1)
        good = (q > 0).all(1) & ((q.max(1) - q.min(1)) < max_spread_m)
        fx, fy = x[i] - xa, y[i] - ya
        val = (q[:, 0] * (1 - fx) * (1 - fy) + q[:, 1] * fx * (1 - fy) + q[:, 2] * (1 - fx) * fy + q[:, 3] * fx * fy)
        out[i] = val
        ok[i] = good
    return out, ok


def _visual_init(frames, K, cfg):
    X, uv, conf = [], [], []
    for f in frames:
        m = np.isfinite(f.landmarks_uv).all(1) & (f.landmarks_conf > 0)
        X.append(f.landmarks_base[m])
        uv.append(f.landmarks_uv[m])
    X = np.concatenate(X) if X else np.zeros((0, 3))
    uv = np.concatenate(uv) if uv else np.zeros((0, 2))
    candidates = []
    T, inl = pnp_ransac(X, uv, K, cfg.ransac_iters, cfg.ransac_px, cfg.seed)
    if T is not None:
        candidates.append(("pnp_ransac", T, inl))
    P_base, P_cam = [], []
    for f in frames:
        if f.depth is None:
            continue
        m = np.isfinite(f.landmarks_uv).all(1) & (f.landmarks_conf > 0)
        z, ok = bilinear_depth(f.depth, np.where(m[:, None], f.landmarks_uv, np.nan))
        ok &= m
        P_base.append(f.landmarks_base[ok])
        P_cam.append(unproject(f.landmarks_uv[ok], z[ok], K))
    if P_base and sum(len(p) for p in P_base) >= 3:
        T2, inl2 = kabsch_ransac(np.concatenate(P_base), np.concatenate(P_cam), cfg.ransac_iters, 0.03, cfg.seed)
        if T2 is not None:
            pred, z = project(transform_points(T2, X), K)
            candidates.append(("depth_kabsch_ransac", T2, (np.linalg.norm(pred - uv, axis=1) < cfg.ransac_px) & (z > 0)))
    if not candidates:
        return None, None, X, uv

    def score(c):
        pred, _ = project(transform_points(c[1], X), K)
        e = np.linalg.norm(pred - uv, axis=1)
        return (-c[2].sum(), float(np.nanmedian(np.where(np.isfinite(e), e, 1e3))))

    name, T, _ = min(candidates, key=score)
    return name, T, X, uv


# ---------------------------------------------------------------- refinement


def _residuals(v, frames, K, cfg, gates, use_depth):
    T = pose_from_vec(v)
    out = []
    for f, gate in zip(frames, gates):
        m = np.isfinite(f.landmarks_uv).all(1) & (f.landmarks_conf > 0)
        pred, _ = project(transform_points(T, f.landmarks_base), K)
        r = (pred - np.nan_to_num(f.landmarks_uv)) / cfg.cauchy_px * np.sqrt(f.landmarks_conf)[:, None]
        r[~m] = 0
        r[~np.isfinite(r)] = 1e2
        out.append(r.ravel())
        if use_depth and f.depth is not None:
            pts = f.surface_base[gate["idx"]]
            uvp, z = project(transform_points(T, pts), K)
            zo, ok = bilinear_depth(f.depth, uvp)
            rd = np.where(ok & gate["use"], (z - zo) / cfg.cauchy_depth_m, 0.0)
            out.append(np.nan_to_num(rd))
    return np.concatenate(out)


def self_visible(points_cam: np.ndarray, K: np.ndarray, hw, block: int = 4, tol_m: float = 0.01) -> np.ndarray:
    """Points on the camera-facing robot surface: within ``tol_m`` of the robot's own z-buffer at ``block``-pixel
    resolution. Without this, back/side surface points sit behind the observed surface and bias the depth term."""
    H, W = hw
    uv, z = project(points_cam, K)
    ok = np.isfinite(uv).all(1) & (z > 0)
    bj = np.floor(np.where(ok, uv[:, 0], -1) / block).astype(int)
    bi = np.floor(np.where(ok, uv[:, 1], -1) / block).astype(int)
    hb, wb = -(-H // block), -(-W // block)
    inside = ok & (bi >= 0) & (bi < hb) & (bj >= 0) & (bj < wb)
    zbuf = np.full((hb, wb), np.inf)
    np.minimum.at(zbuf, (bi[inside], bj[inside]), z[inside])
    vis = np.zeros(len(points_cam), bool)
    vis[inside] = z[inside] <= zbuf[bi[inside], bj[inside]] + tol_m
    return vis


def _depth_gates(T, frames, K, cfg, rng):
    gates = []
    for f in frames:
        n = len(f.surface_base)
        idx = rng.choice(n, min(n, cfg.depth_points_per_frame), replace=False)
        if f.depth is None:
            gates.append({"idx": idx, "use": np.zeros(len(idx), bool)})
            continue
        cam = transform_points(T, f.surface_base)
        front = self_visible(cam, K, cfg.image_hw, cfg.self_occlusion_block_px, cfg.self_occlusion_tol_m)[idx]
        uvp, z = project(cam[idx], K)
        zo, ok = bilinear_depth(f.depth, uvp)
        # self-occlusion from the robot's own z-buffer, then a one-sided occlusion test against the observed depth:
        # points hidden behind something closer are excluded, in-front residuals are kept
        use = ok & front & (z > 0) & (z <= zo + cfg.occlusion_tol_m)
        if f.robot_mask is not None:
            px = np.clip(np.floor(np.nan_to_num(uvp, nan=-1)).astype(int), -1, None)
            H, W = f.robot_mask.shape
            inside = (px[:, 0] >= 0) & (px[:, 0] < W) & (px[:, 1] >= 0) & (px[:, 1] < H)
            on = np.zeros(len(idx), bool)
            on[inside] = f.robot_mask[px[inside, 1], px[inside, 0]]
            use &= on
        gates.append({"idx": idx, "use": use})
    return gates


def _frame_stats(T, frames, K, cfg):
    H, W = cfg.image_hw
    rows = []
    for f in frames:
        uvp, z = project(transform_points(T, f.surface_base), K)
        inside = np.isfinite(uvp).all(1) & (z > 0) & (uvp[:, 0] >= 0) & (uvp[:, 0] < W) & (uvp[:, 1] >= 0) & (uvp[:, 1] < H)
        m = np.isfinite(f.landmarks_uv).all(1) & (f.landmarks_conf > 0)
        pred, _ = project(transform_points(T, f.landmarks_base), K)
        e = np.linalg.norm(pred - f.landmarks_uv, axis=1)[m]
        row = {"t_index": f.t_index, "timestamp_ns": int(f.timestamp_ns), "robot_in_view_fraction": float(inside.mean()),
               "landmarks_observed": int(m.sum()), "reproj_median_px": float(np.median(e)) if len(e) else None}
        if f.depth is not None:
            zo, ok = bilinear_depth(f.depth, uvp)
            front = self_visible(transform_points(T, f.surface_base), K, cfg.image_hw, cfg.self_occlusion_block_px,
                                 cfg.self_occlusion_tol_m)
            vis = ok & inside & front & (z <= zo + cfg.occlusion_tol_m)
            row["depth_points"] = int(vis.sum())
            row["depth_abs_median_m"] = float(np.median(np.abs(z[vis] - zo[vis]))) if vis.any() else None
        if f.robot_mask is not None:
            pred_mask = np.zeros_like(f.robot_mask, bool)
            px = np.floor(uvp[inside]).astype(int)
            pred_mask[px[:, 1], px[:, 0]] = True
            union = (pred_mask | f.robot_mask).sum()
            row["mask_point_hit_rate"] = float(f.robot_mask[px[:, 1], px[:, 0]].mean()) if len(px) else None
            row["mask_union_px"] = int(union)
        rows.append(row)
    return rows


def motion_coverage(frames) -> dict:
    P = np.stack([f.flange_base for f in frames]) if frames else np.zeros((0, 3))
    if len(P) < 2:
        return {"extents_m": [0.0, 0.0, 0.0], "distinct_poses": len(P)}
    sv = np.linalg.svd(P - P.mean(0), compute_uv=False)
    extents = (2 * sv / np.sqrt(len(P))).tolist()  # ~ +-1 std span along principal axes
    distinct = [P[0]]
    for p in P[1:]:
        if min(np.linalg.norm(p - d) for d in distinct) > 0.02:
            distinct.append(p)
    return {"extents_m": [float(e) for e in extents] + [0.0] * (3 - len(extents)), "distinct_poses": len(distinct)}


def calibrate_camera(camera_id: str, frames: list[CalibrationFrame], K: np.ndarray, cfg: CalibrationConfig,
                     T_init: np.ndarray | None = None) -> CalibrationResult:
    """Calibrate one camera of one episode. ``T_init`` (e.g. a previous episode) is used only if visual init fails."""
    reasons: list[str] = []
    diag: dict = {"config": asdict(cfg), "camera_id": camera_id, "frames": len(frames)}
    method, T0, X, uv = _visual_init(frames, K, cfg)
    if T0 is None and T_init is not None:
        method, T0 = "external_prior", np.asarray(T_init, float)
    diag["init_method"] = method
    total_obs = int(sum((np.isfinite(f.landmarks_uv).all(1) & (f.landmarks_conf > 0)).sum() for f in frames))
    diag["landmark_observations"] = total_obs
    if T0 is None:
        return CalibrationResult(camera_id, "rejected", ["visual initialisation failed (too few landmark observations)"],
                                 None, None, None, diag)
    # visibility under the initial estimate
    rows0 = _frame_stats(T0, frames, K, cfg)
    visible = [r["robot_in_view_fraction"] >= cfg.min_visible_fraction and r["landmarks_observed"] > 0 for r in rows0]
    diag["visible_frames"] = int(sum(visible))
    if sum(visible) < cfg.min_visible_frames:
        reasons.append(f"insufficient robot visibility: {sum(visible)} frames < {cfg.min_visible_frames}")
    if total_obs < cfg.min_landmark_obs:
        reasons.append(f"too few landmark observations: {total_obs} < {cfg.min_landmark_obs}")
    use = [f for f, v in zip(frames, visible) if v]
    holdout = [f for i, f in enumerate(use) if cfg.holdout_every and i % cfg.holdout_every == cfg.holdout_every - 1]
    fit = [f for f in use if not any(f is h for h in holdout)] or use
    cov = motion_coverage(fit)
    diag["motion_coverage"] = cov
    if cov["extents_m"][1] < cfg.min_motion_extent_m or cov["distinct_poses"] < cfg.min_distinct_poses:
        reasons.append(f"poor motion coverage: extents {np.round(cov['extents_m'], 3).tolist()} m, "
                       f"{cov['distinct_poses']} distinct poses")
    if not fit:
        return CalibrationResult(camera_id, "rejected", reasons or ["no usable frames"], None, T0, method, diag)
    rng = np.random.default_rng(cfg.seed)
    use_depth = cfg.depth_term and any(f.depth is not None for f in fit)
    T = T0
    res = None
    for _ in range(max(1, cfg.outer_iters)):
        gates = _depth_gates(T, fit, K, cfg, rng) if use_depth else [None] * len(fit)
        res = least_squares(_residuals, vec_from_pose(T), args=(fit, K, cfg, gates, use_depth), loss="cauchy",
                            f_scale=1.0, method="trf", x_scale=np.array([0.01, 0.01, 0.01, 0.01, 0.01, 0.01]))
        T = pose_from_vec(res.x)
    diag["optimizer"] = {"status": int(res.status), "message": str(res.message), "nfev": int(res.nfev),
                         "cost": float(res.cost), "success": bool(res.success)}
    if not res.success or not np.isfinite(res.x).all():
        reasons.append(f"optimisation failed: {res.message}")
    J = res.jac
    JTJ = J.T @ J
    cond = float(np.linalg.cond(JTJ))
    diag["condition_number"] = cond
    dof = max(1, len(res.fun) - 6)
    try:
        covm = np.linalg.inv(JTJ) * (2 * res.cost / dof)
        std = np.sqrt(np.clip(np.diag(covm), 0, None))
        diag["std_rot_deg"] = float(np.degrees(np.linalg.norm(std[:3])))
        diag["std_trans_m"] = float(np.linalg.norm(std[3:]))
    except np.linalg.LinAlgError:
        diag["std_rot_deg"] = diag["std_trans_m"] = float("inf")
    if cond > cfg.max_condition:
        reasons.append(f"ill-conditioned problem (cond {cond:.3g})")
    if diag["std_rot_deg"] > cfg.max_rot_std_deg or diag["std_trans_m"] > cfg.max_trans_std_m:
        reasons.append(f"uncertain extrinsic (std {diag['std_rot_deg']:.3f} deg, {diag['std_trans_m'] * 1000:.1f} mm)")
    fit_rows = _frame_stats(T, fit, K, cfg)
    hold_rows = _frame_stats(T, holdout, K, cfg)
    diag["per_frame_fit"] = fit_rows
    diag["per_frame_heldout"] = hold_rows

    def med(rows, key):
        vals = [r[key] for r in rows if r.get(key) is not None]
        return float(np.median(vals)) if vals else None

    diag["reproj_median_px"] = med(fit_rows, "reproj_median_px")
    diag["heldout_reproj_median_px"] = med(hold_rows, "reproj_median_px")
    diag["depth_abs_median_m"] = med(fit_rows, "depth_abs_median_m") if use_depth else None
    diag["heldout_depth_abs_median_m"] = med(hold_rows, "depth_abs_median_m") if use_depth else None
    # landmark inliers at the solution
    errs = []
    for f in fit:
        m = np.isfinite(f.landmarks_uv).all(1) & (f.landmarks_conf > 0)
        pred, _ = project(transform_points(T, f.landmarks_base[m]), K)
        errs.append(np.linalg.norm(pred - f.landmarks_uv[m], axis=1))
    errs = np.concatenate(errs) if errs else np.zeros(0)
    diag["landmark_inlier_fraction"] = float((errs < cfg.ransac_px).mean()) if len(errs) else 0.0
    delta = invert_T(T0) @ T
    diag["init_to_final"] = {"rot_deg": float(np.degrees(np.linalg.norm(vec_from_pose(delta)[:3]))),
                             "trans_m": float(np.linalg.norm(delta[:3, 3]))}
    if diag["reproj_median_px"] is None or diag["reproj_median_px"] > cfg.max_reproj_median_px:
        reasons.append(f"reprojection median {diag['reproj_median_px']} px > {cfg.max_reproj_median_px}")
    if diag["heldout_reproj_median_px"] is not None and diag["heldout_reproj_median_px"] > cfg.max_heldout_reproj_median_px:
        reasons.append(f"held-out reprojection median {diag['heldout_reproj_median_px']:.2f} px too high")
    if use_depth and (diag["depth_abs_median_m"] is None or diag["depth_abs_median_m"] > cfg.max_depth_median_m):
        reasons.append(f"robot-depth residual median {diag['depth_abs_median_m']} m > {cfg.max_depth_median_m}")
    if diag["landmark_inlier_fraction"] < cfg.min_inlier_fraction:
        reasons.append(f"landmark inlier fraction {diag['landmark_inlier_fraction']:.2f} too low")
    if not use_depth:
        diag["note"] = "no metric depth: reprojection-only refinement"
    return CalibrationResult(camera_id, "rejected" if reasons else "accepted", reasons, T, T0, method, diag)


# ---------------------------------------------------------------- overlays


def overlay_image(image: np.ndarray | None, T: np.ndarray, frame: CalibrationFrame, K: np.ndarray, hw) -> np.ndarray:
    """RGB uint8 overlay: projected robot surface (green), predicted landmarks (blue), observed landmarks (red)."""
    from PIL import Image, ImageDraw

    H, W = hw
    if image is None:
        base = np.zeros((H, W, 3), np.uint8)
    else:
        base = np.asarray(image)
        if base.ndim == 2:
            base = np.repeat(base[..., None], 3, -1)  # display only; the overlay is not training data
        base = base.astype(np.uint8)
    canvas = Image.fromarray(base.copy())
    draw = ImageDraw.Draw(canvas)
    uvp, z = project(transform_points(T, frame.surface_base), K)
    for (u, v), zz in zip(uvp, z):
        if zz > 0 and 0 <= u < W and 0 <= v < H:
            draw.point((u - 0.5, v - 0.5), fill=(0, 255, 0))
    pred, _ = project(transform_points(T, frame.landmarks_base), K)
    for (u, v), (uo, vo) in zip(pred, frame.landmarks_uv):
        if np.isfinite([u, v]).all():
            draw.ellipse((u - 2.5, v - 2.5, u + 1.5, v + 1.5), outline=(0, 128, 255))
        if np.isfinite([uo, vo]).all():
            draw.line((uo - 3.5, vo - 0.5, uo + 2.5, vo - 0.5), fill=(255, 0, 0))
            draw.line((uo - 0.5, vo - 3.5, uo - 0.5, vo + 2.5), fill=(255, 0, 0))
    return np.asarray(canvas)
