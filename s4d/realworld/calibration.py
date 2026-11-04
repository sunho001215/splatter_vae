"""Episode-level robot-grounded extrinsic calibration (simplified from DepthWorld, arXiv 2610.08780).

One constant ``T_cam_base`` per camera per episode (cameras may move between episodes). Two steps:

1. **Visual initialisation** from 2D observations of robot landmarks (URDF link origins / marker corners / matched
   render points) whose 3D positions follow from forward kinematics: RANSAC PnP (DLT + Levenberg-Marquardt) and,
   when metric stereo depth exists, a RANSAC Kabsch fit of depth-lifted landmarks; the lower-error one wins.
2. **Robot-grounded refinement**: a Cauchy-robust least-squares over the 6-DoF extrinsic with
   (a) landmark reprojection residuals (scale ``cauchy_px``) and (b) robot-surface depth residuals: articulated
   URDF surface points (arm + gripper at the recorded gripper state) projected into the camera and compared with
   the bootstrap stereo depth (scale ``cauchy_depth_m``), gated for occlusion and re-gated in an outer loop.

RGB-D evidence (``CalibrationConfig.evidence = "rgbd"``): when no 2D landmark source exists for the images but the
camera measures metric depth (RealSense), the visual initialisation is replaced by a registration of the moving
articulated URDF surface (recorded joints + gripper state) to the moving part of the measured depth (pixels that
differ from the per-pixel temporal median): a global search over camera orientations (optical axis on a sphere x
roll) with centroid-aligned translations, scored by the trimmed mean observed-to-model distance, followed by trimmed
ICP. The refinement is the same robot-grounded depth term; the landmark gates do not apply and are replaced by depth
gates on fit AND held-out frames plus an explained-fraction gate (moving measured points within ``explained_tol_m`` of
the posed robot surface). With per-frame robot masks (e.g. SAM 3 "robot arm"), the observed points are the masked
measured depth and the whole articulated surface is registered (the static base and links constrain the pose even when
the arm moves little); without masks only the moving links can be separated from the scene. No pose ever comes from a
learned reconstruction model.

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
    max_nfev: int | None = None  # optimiser budget per outer iteration (None = scipy default)
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
    # evidence: "landmarks" (2D robot landmarks; optional depth term) or "rgbd" (metric depth only, see module doc)
    evidence: str = "landmarks"
    init_frames: int = 10
    init_obs_points: int = 400
    init_axes: int = 400
    init_rolls: int = 24
    init_top: int = 30
    icp_iters: int = 30
    init_inlier_m: float = 0.03
    init_translation_steps: int = 3
    motion_threshold_m: float = 0.03
    moving_link_motion_m: float = 0.05
    init_trim: float = 0.8
    explained_tol_m: float = 0.02
    min_explained_fraction: float = 0.5
    max_heldout_depth_median_m: float = 0.02
    # RGB-D: Jacobian rows above this multiple of the median non-zero row norm are finite-difference jumps (a projected
    # surface point crossing into invalid depth switches its residual on/off); they are excluded from the conditioning
    # and covariance estimates, which they would otherwise distort (inflated largest eigenvalue, shrunken variance)
    jac_discontinuity_factor: float = 100.0


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
    best_inl = None
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
            best_inl = inl
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
    X, uv = [], []
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


# ---------------------------------------------------------------- RGB-D initialisation (no landmarks)


def moving_masks(depths: np.ndarray, threshold_m: float) -> np.ndarray:
    """(F,H,W) bool: valid measured depth IN FRONT of the per-pixel temporal median by more than ``threshold_m`` (or
    where the median is invalid), cleaned by a 3x3 opening - the moving robot (and whatever it carries) occluding the
    static scene. Pixels behind the median (where the robot was) and isolated depth noise are excluded."""
    from scipy.ndimage import binary_opening

    d = np.where(depths > 0, depths, np.nan)
    bg = np.nanmedian(d, axis=0)
    front = (depths > 0) & (~np.isfinite(bg) | (np.nan_to_num(bg - d, nan=0.0) > threshold_m))
    return np.stack([binary_opening(m, structure=np.ones((3, 3), bool)) for m in front])


def fibonacci_sphere(n: int) -> np.ndarray:
    i = np.arange(n) + 0.5
    phi = np.arccos(1 - 2 * i / n)
    theta = np.pi * (1 + 5 ** 0.5) * i
    return np.stack([np.cos(theta) * np.sin(phi), np.sin(theta) * np.sin(phi), np.cos(phi)], 1)


def _rotations(n_axes: int, n_rolls: int) -> np.ndarray:
    """Camera rotations R_cam_base whose optical axis (+z, in base coordinates) covers the sphere, times rolls."""
    out = []
    for z in fibonacci_sphere(n_axes):
        a = np.array([1.0, 0, 0]) if abs(z[0]) < 0.9 else np.array([0, 1.0, 0])
        x = np.cross(a, z)
        x /= np.linalg.norm(x)
        y = np.cross(z, x)
        for r in np.linspace(0, 2 * np.pi, n_rolls, endpoint=False):
            c, s_ = np.cos(r), np.sin(r)
            xr, yr = c * x + s_ * y, -s_ * x + c * y
            out.append(np.stack([xr, yr, z]))  # rows = camera axes in base coordinates -> R_cam_base
    return np.asarray(out)


def _trimmed(d: np.ndarray, keep: float) -> float:
    k = max(1, int(len(d) * keep))
    return float(np.partition(d, k - 1)[:k].mean())


def _depth_evidence_frames(frames):
    usable = [f for f in frames if f.depth is not None]
    masked = bool(usable) and all(f.robot_mask is not None for f in usable)
    return usable, masked


def _depth_evidence_masks(usable, masked, cfg):
    if masked:
        return np.stack([f.robot_mask & (f.depth > 0) for f in usable])
    return moving_masks(np.stack([f.depth for f in usable]), cfg.motion_threshold_m)


def rgbd_initialisation(frames, K, cfg, rng) -> tuple[np.ndarray | None, dict]:
    """Global registration of the moving URDF surface to the moving measured depth (see module doc)."""
    from scipy.spatial import cKDTree

    usable, masked = _depth_evidence_frames(frames)
    diag = {"method": "rgbd_robot_registration"}
    if len(usable) < (1 if masked else 3):  # motion evidence needs a temporal median; masks work per frame
        return None, {**diag, "error": f"too few frames with depth ({len(usable)})"}
    masks = _depth_evidence_masks(usable, masked, cfg)
    diag["evidence"] = "robot_mask" if masked else "moving_depth"
    pick = np.linspace(0, len(usable) - 1, min(cfg.init_frames, len(usable))).round().astype(int)
    surf = np.stack([usable[k].surface_base for k in range(len(usable))])  # same sample order every frame
    if diag["evidence"] == "robot_mask":  # the whole robot is observed: register the whole articulated surface
        moving = np.ones(surf.shape[1], bool)
    else:  # only what moves is separable from the static scene: register the moving links
        moving = np.linalg.norm(surf - surf.mean(0), axis=2).max(0) > cfg.moving_link_motion_m
    diag["model_points"] = int(moving.sum())
    obs, trees, cents_obs, cents_model = [], [], [], []
    hw = usable[0].depth.shape
    from s4d.realworld.geometry import pixel_centers

    centers = pixel_centers(*hw)
    for k in pick:
        f, m = usable[k], masks[k]
        uv = centers[m]
        if len(uv) < 50:
            continue
        sel = rng.choice(len(uv), min(cfg.init_obs_points, len(uv)), replace=False)
        P = unproject(uv[sel], f.depth[m][sel], K)
        model = f.surface_base[moving]
        obs.append(P)
        trees.append(cKDTree(model))
        cents_obs.append(P.mean(0))
        cents_model.append(model.mean(0))
    diag["init_frames"] = len(obs)
    diag["moving_pixels_median"] = int(np.median(masks.reshape(len(masks), -1).sum(1)))
    if len(obs) < (1 if masked else 3) or moving.sum() < 20:
        return None, {**diag, "error": "too little robot motion visible in the measured depth"}

    def nearest(R, t):
        """Per frame: observed points mapped to base coordinates and their nearest model points."""
        out = []
        for j in range(len(obs)):
            pb = (obs[j] - t) @ R
            d, idx = trees[j].query(pb, k=1, distance_upper_bound=0.5, workers=-1)
            out.append((pb, d, idx))
        return out

    def cost(R, t, keep=cfg.init_trim):
        d = np.minimum(np.concatenate([x[1] for x in nearest(R, t)]), 0.5)
        return _trimmed(d, keep)

    def score_all(Rs, ts, batch=512):
        """For every candidate (R, t): ``init_translation_steps`` translation-only updates (median residual of the
        observed points within 15 cm of the model), then the inlier fraction within ``init_inlier_m`` (higher is
        better). Vectorised over candidates: one KD-tree query per frame and candidate batch."""
        ts = ts.copy()
        inl = np.zeros(len(Rs))
        for b0 in range(0, len(Rs), batch):
            R, t = Rs[b0:b0 + batch], ts[b0:b0 + batch]
            for step in range(cfg.init_translation_steps + 1):
                res, hits, total = [], np.zeros(len(R)), 0
                for j in range(len(obs)):
                    pb = np.einsum("cpk,ckl->cpl", obs[j][None] - t[:, None], R)  # base coordinates (C,P,3)
                    d, idx = trees[j].query(pb.reshape(-1, 3), k=1, distance_upper_bound=0.5, workers=-1)
                    d = d.reshape(len(R), -1)
                    total += d.shape[1]
                    if step == cfg.init_translation_steps:
                        hits += (d <= cfg.init_inlier_m).sum(1)
                        continue
                    idx = np.minimum(idx, len(trees[j].data) - 1).reshape(len(R), -1)
                    pred = np.einsum("cpl,ckl->cpk", trees[j].data[idx], R) + t[:, None]
                    r = obs[j][None] - pred
                    r[~(d < 0.15)] = np.nan
                    res.append(r)
                if step == cfg.init_translation_steps:
                    inl[b0:b0 + batch] = hits / max(1, total)
                else:
                    med = np.nanmedian(np.concatenate(res, 1), axis=1)
                    t = t + np.nan_to_num(med)
            ts[b0:b0 + batch] = t
        return inl, ts

    co, cm = np.mean(cents_obs, 0), np.mean(cents_model, 0)
    Rs = _rotations(cfg.init_axes, cfg.init_rolls)
    inl, ts = score_all(Rs, co[None] - np.einsum("ckl,l->ck", Rs, cm))
    scores = [(-inl[c], Rs[c], ts[c]) for c in range(len(Rs))]
    scores.sort(key=lambda x: x[0])
    refined = []
    for _, R, t in scores[:cfg.init_top]:
        for _ in range(cfg.icp_iters):  # trimmed point-to-point ICP over all init frames (observed -> model)
            src, dst = [], []
            for j in range(len(obs)):
                pb = (obs[j] - t) @ R  # observed points in base coordinates
                d, idx = trees[j].query(pb, k=1, distance_upper_bound=0.5, workers=-1)
                ok = np.isfinite(d)
                if ok.sum() < 3:
                    continue
                thr = np.quantile(d[ok], cfg.init_trim)
                ok &= d <= thr
                src.append(trees[j].data[idx[ok]])
                dst.append(obs[j][ok])
            if not src:
                break
            T_new = kabsch(np.concatenate(src), np.concatenate(dst))
            R, t = T_new[:3, :3], T_new[:3, 3]
        refined.append((cost(R, t), R, t))
    refined.sort(key=lambda x: x[0])
    best = refined[0]
    T0 = make_T(best[1], best[2])
    distinct = [r for r in refined[1:] if np.degrees(np.arccos(np.clip((np.trace(r[1] @ best[1].T) - 1) / 2, -1, 1))) > 10
                or np.linalg.norm(r[2] - best[2]) > 0.05]
    d_best = np.concatenate([x[1] for x in nearest(best[1], best[2])])
    diag.update({"candidates": len(scores), "best_trimmed_mean_m": best[0],
                 "best_inlier_fraction": float((d_best <= cfg.init_inlier_m).mean()),
                 "second_distinct_trimmed_mean_m": distinct[0][0] if distinct else None,
                 "coarse_best_inlier_fraction": -scores[0][0]})
    return T0, diag


def explained_fraction(T, frames, K, cfg) -> float | None:
    """Fraction of observed robot points (masked depth when masks exist, else the moving measured points of the
    temporal-median test) within ``explained_tol_m`` of the posed URDF surface."""
    from scipy.spatial import cKDTree

    usable, masked = _depth_evidence_frames(frames)
    if len(usable) < (1 if masked else 3):
        return None
    masks = _depth_evidence_masks(usable, masked, cfg)
    from s4d.realworld.geometry import pixel_centers

    centers = pixel_centers(*usable[0].depth.shape)
    hits, total = 0, 0
    for f, m in zip(usable, masks):
        if not m.any():
            continue
        P = unproject(centers[m], f.depth[m], K)
        d, _ = cKDTree(transform_points(T, f.surface_base)).query(P, k=1)
        hits += int((d <= cfg.explained_tol_m).sum())
        total += len(d)
    return hits / total if total else None


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
        e = np.where(np.isfinite(e), e, np.inf)  # observed landmark behind the camera = infinitely wrong
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
    rgbd = cfg.evidence == "rgbd"
    if cfg.evidence not in ("landmarks", "rgbd"):
        raise ValueError(f"unknown calibration evidence {cfg.evidence!r}")
    if rgbd:
        T0, diag["rgbd_init"] = rgbd_initialisation(frames, K, cfg, np.random.default_rng(cfg.seed))
        method = "rgbd_robot_registration" if T0 is not None else None
    else:
        method, T0, X, uv = _visual_init(frames, K, cfg)
    if T0 is None and T_init is not None:
        method, T0 = "external_prior", np.asarray(T_init, float)
    diag["init_method"] = method
    total_obs = int(sum((np.isfinite(f.landmarks_uv).all(1) & (f.landmarks_conf > 0)).sum() for f in frames))
    diag["landmark_observations"] = total_obs
    if T0 is None:
        why = ("RGB-D initialisation failed: " + diag["rgbd_init"].get("error", "no solution")) if rgbd else \
            "visual initialisation failed (too few landmark observations)"
        return CalibrationResult(camera_id, "rejected", [why], None, None, None, diag)
    # visibility under the initial estimate
    rows0 = _frame_stats(T0, frames, K, cfg)
    if rgbd:  # no landmarks: visibility = robot in view and measured depth present
        visible = [r["robot_in_view_fraction"] >= cfg.min_visible_fraction and f.depth is not None
                   for r, f in zip(rows0, frames)]
    else:
        visible = [r["robot_in_view_fraction"] >= cfg.min_visible_fraction and r["landmarks_observed"] > 0
                   for r in rows0]
    diag["visible_frames"] = int(sum(visible))
    if sum(visible) < cfg.min_visible_frames:
        reasons.append(f"insufficient robot visibility: {sum(visible)} frames < {cfg.min_visible_frames}")
    if not rgbd and total_obs < cfg.min_landmark_obs:
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
    if rgbd and not use_depth:
        return CalibrationResult(camera_id, "rejected", reasons + ["RGB-D evidence needs the depth term"], None, T0,
                                 method, diag)
    T = T0
    res = None
    for _ in range(max(1, cfg.outer_iters)):
        gates = _depth_gates(T, fit, K, cfg, rng) if use_depth else [None] * len(fit)
        res = least_squares(_residuals, vec_from_pose(T), args=(fit, K, cfg, gates, use_depth), loss="cauchy",
                            f_scale=1.0, method="trf", x_scale=np.array([0.01, 0.01, 0.01, 0.01, 0.01, 0.01]),
                            max_nfev=cfg.max_nfev)
        T = pose_from_vec(res.x)
    diag["optimizer"] = {"status": int(res.status), "message": str(res.message), "nfev": int(res.nfev),
                         "cost": float(res.cost), "success": bool(res.success)}
    if not res.success or not np.isfinite(res.x).all():
        reasons.append(f"optimisation failed: {res.message}")
    J = res.jac
    jumps = 0
    if rgbd:
        rn = np.linalg.norm(J, axis=1)
        nz = rn > 0
        jump = rn > cfg.jac_discontinuity_factor * np.median(rn[nz]) if nz.any() else np.zeros(len(rn), bool)
        jumps = int(jump.sum())
        diag["condition_number_all_rows"] = float(np.linalg.cond(J.T @ J))
        diag["jacobian_rows_excluded_discontinuous"] = jumps
        J = J[~jump]
    JTJ = J.T @ J
    cond = float(np.linalg.cond(JTJ))
    diag["condition_number"] = cond
    if rgbd:
        w, V = np.linalg.eigh(JTJ)
        diag["jtj_eigenvalues"] = w.tolist()
        diag["jtj_weakest_direction"] = V[:, 0].tolist()  # [rotation vector (rad), translation (m)] components
    # only active residuals count (masked landmarks and gated depth samples are zero-filled entries)
    active = sum(2 * int((np.isfinite(f.landmarks_uv).all(1) & (f.landmarks_conf > 0)).sum()) for f in fit)
    if use_depth:
        active += sum(int(g["use"].sum()) for g in gates if g is not None)
    active -= jumps
    dof = max(1, active - 6)
    diag["active_residuals"] = active
    try:
        covm = np.linalg.inv(JTJ) * (2 * res.cost / dof)
        std = np.sqrt(np.clip(np.diag(covm), 0, None))
        diag["std_rot_deg"] = float(np.degrees(np.linalg.norm(std[:3])))
        diag["std_trans_m"] = float(np.linalg.norm(std[3:]))
    except np.linalg.LinAlgError:
        diag["std_rot_deg"] = diag["std_trans_m"] = float("inf")
    def bad(x, thr):  # NaN/inf never passes a gate
        return x is None or not np.isfinite(x) or x > thr

    if bad(cond, cfg.max_condition):
        reasons.append(f"ill-conditioned problem (cond {cond:.3g})")
    if bad(diag["std_rot_deg"], cfg.max_rot_std_deg) or bad(diag["std_trans_m"], cfg.max_trans_std_m):
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
    errs = np.where(np.isfinite(errs), errs, np.inf)
    diag["landmark_inlier_fraction"] = float((errs < cfg.ransac_px).mean()) if len(errs) else 0.0
    delta = invert_T(T0) @ T
    diag["init_to_final"] = {"rot_deg": float(np.degrees(np.linalg.norm(vec_from_pose(delta)[:3]))),
                             "trans_m": float(np.linalg.norm(delta[:3, 3]))}
    if rgbd:
        diag["landmark_gates"] = "not applicable (RGB-D evidence: no 2D landmark source)"
        diag["explained_fraction_fit"] = explained_fraction(T, fit, K, cfg)
        diag["explained_fraction_heldout"] = explained_fraction(T, holdout, K, cfg) if holdout else None
        if holdout and bad(diag["heldout_depth_abs_median_m"], cfg.max_heldout_depth_median_m):
            reasons.append(f"held-out robot-depth residual median {diag['heldout_depth_abs_median_m']} m > "
                           f"{cfg.max_heldout_depth_median_m}")
        ef = diag["explained_fraction_heldout"] if holdout else diag["explained_fraction_fit"]
        if ef is None or not np.isfinite(ef) or ef < cfg.min_explained_fraction:
            reasons.append(f"moving measured depth explained by the posed robot: {ef} < {cfg.min_explained_fraction}")
        if bad(diag["depth_abs_median_m"], cfg.max_depth_median_m):
            reasons.append(f"robot-depth residual median {diag['depth_abs_median_m']} m > {cfg.max_depth_median_m}")
    else:
        if bad(diag["reproj_median_px"], cfg.max_reproj_median_px):
            reasons.append(f"reprojection median {diag['reproj_median_px']} px > {cfg.max_reproj_median_px}")
        if holdout and bad(diag["heldout_reproj_median_px"], cfg.max_heldout_reproj_median_px):
            reasons.append(f"held-out reprojection median {diag['heldout_reproj_median_px']} px too high")
        if use_depth and bad(diag["depth_abs_median_m"], cfg.max_depth_median_m):
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
