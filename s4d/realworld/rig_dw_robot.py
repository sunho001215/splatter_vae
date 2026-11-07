"""Robot-grounded factors of the board-free rig calibration (``rig_dw``): the rig's cameras against the robot seen in
their images (URDF forward kinematics at the camera timestamp, Franka/Robotiq VISUAL meshes, SAM 3 robot masks, metric
depth). Three residual families per (frame, camera), all with analytic Jacobians w.r.t. the camera's left perturbation
xi_c = (omega, v) (T_c <- exp(xi_c) T_c, camera frame), its log depth scale alpha_c and its effective time offset tau_c
(robot geometry at t + tau_c, linearised with the FK surface velocity from t +- 10 ms):

* ``sil_m2o``  model -> observed silhouette: visible model contour points X_j (front samples on the boundary of the
  rendered model silhouette) -> signed distance (px, + outside) to the SAM robot-mask boundary at pi(T_c X_j).
* ``sil_o2m``  observed -> model silhouette: SAM boundary pixels b -> point-to-line distance n_j . (pi(T_c X_j) - b)
  to the nearest rendered model contour point X_j along the rendered silhouette's outward normal n_j.
* ``depth``    point-to-plane: observed robot pixels (eroded SAM mask, valid smooth depth) lifted with
  exp(alpha_c) d; projective association to the rendered front sample X_m with normal n_m;
  r = n_c . (X_obs - T_c X_m), n_c = R_c n_m.

Support (contour points, associations, exclusions) is re-selected by ``build_frame_factors`` at the current estimate
(an outer iteration) and held fixed inside the inner Gauss-Newton/LM solve (``FrameFactors`` stores only what the
linearisation needs). Exclusions (independent of the candidate pose's fit): image/crop border, boundaries adjacent to
SAM object/fixture pixels, and occlusion boundaries where a non-robot pixel is >= ``occluder_m`` in front of the robot
(the counter hides the lower arm).

Rendering is a dense point z-buffer of the visual-mesh surface (``VisualSurface`` at ~1.5 mm spacing, >= 2 samples
per pixel footprint at 1-2 m) with a 3x3 closing for residual holes; the contour is the rendered mask boundary.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import ndimage

from s4d.realworld.rig_dw_geom import project as _project


@dataclass
class RobotTermConfig:
    render_spacing_m: float = 0.0015
    closing: int = 1  # iterations of a 3x3 binary closing of the rendered mask
    border_px: int = 4
    adjacent_px: int = 3  # boundary pixels within this distance of SAM object/fixture pixels are excluded
    occluder_m: float = 0.10
    depth_points: int = 300
    depth_erode_px: int = 3
    depth_smooth_rel: float = 0.02
    depth_stage: str = "foundation_stereo_depth"
    max_surface_speed_m_s: float = 0.10
    sigma_sil_px: float = 1.0
    sigma_depth_rel: float = 0.01  # observation noise of depth (relative), along the ray
    sigma_model_m: float = 0.002
    contour_stride: int = 1  # keep every k-th boundary pixel (arc-length subsampling)
    seed: int = 0


def sdf_of_mask(mask: np.ndarray) -> np.ndarray:
    """Signed distance (px) to the boundary of a binary mask, + outside, - inside; the boundary lies half a pixel
    from the centres of the boundary pixels (continuous convention: pixel (i, j) centre at (j + .5, i + .5))."""
    if not mask.any():
        return np.full(mask.shape, 1e3, np.float32)
    if mask.all():
        return np.full(mask.shape, -1e3, np.float32)
    out = ndimage.distance_transform_edt(~mask) - 0.5
    inn = ndimage.distance_transform_edt(mask) - 0.5
    return np.where(mask, -inn, out).astype(np.float32)


def bilinear(img: np.ndarray, uv: np.ndarray, fill: float = np.nan):
    """Bilinear value and gradient (d/du, d/dv) of an image sampled at continuous coordinates uv (pixel centres at
    +.5); outside the image -> ``fill`` and zero gradient."""
    h, w = img.shape
    x, y = uv[:, 0] - 0.5, uv[:, 1] - 0.5
    x0, y0 = np.floor(x).astype(int), np.floor(y).astype(int)
    ok = (x0 >= 0) & (y0 >= 0) & (x0 < w - 1) & (y0 < h - 1)
    x0c, y0c = np.clip(x0, 0, w - 2), np.clip(y0, 0, h - 2)
    fx, fy = x - x0c, y - y0c
    a, b = img[y0c, x0c], img[y0c, x0c + 1]
    c, d = img[y0c + 1, x0c], img[y0c + 1, x0c + 1]
    val = a * (1 - fx) * (1 - fy) + b * fx * (1 - fy) + c * (1 - fx) * fy + d * fx * fy
    gu = (b - a) * (1 - fy) + (d - c) * fy
    gv = (c - a) * (1 - fx) + (d - b) * fx
    val = np.where(ok, val, fill)
    return val, np.where(ok, gu, 0.0), np.where(ok, gv, 0.0)


def boundary(mask: np.ndarray) -> np.ndarray:
    """Pixels of ``mask`` with a 4-neighbour outside it."""
    er = ndimage.binary_erosion(mask, structure=ndimage.generate_binary_structure(2, 1), border_value=1)
    return mask & ~er


@dataclass
class Render:
    mask: np.ndarray  # (h, w) bool on the crop
    zbuf: np.ndarray  # (h, w) camera-z of the front sample (inf where empty)
    front: np.ndarray  # (h, w) int index into the posed sample arrays (-1 where empty)
    idx: np.ndarray | None = None  # in-crop sample indices
    pix: np.ndarray | None = None  # their crop pixel (flat)
    uv: np.ndarray | None = None  # their continuous full-image pixel coordinates


def outward_normals(mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Unit outward normal field (gx, gy) of a binary mask: gradient of its (slightly smoothed) signed distance."""
    sd = ndimage.gaussian_filter(sdf_of_mask(mask).astype(np.float64), 0.7)
    gy, gx = np.gradient(sd)
    n = np.hypot(gx, gy)
    n = np.where(n > 1e-9, n, 1.0)
    return gx / n, gy / n


def extreme_contour_samples(rend: "Render", pix_sel: np.ndarray, nx: np.ndarray, ny: np.ndarray) -> tuple:
    """For each selected boundary pixel (flat crop index), the sample projecting into it that lies furthest along the
    outward normal there (the best estimate of the silhouette edge inside that pixel; every sample counts, the outline
    is the union of all projections). Returns (pixels that have a sample, their sample indices)."""
    want = np.zeros(len(rend.front.ravel()), bool)
    want[pix_sel] = True
    m = want[rend.pix]
    if not m.any():
        return np.zeros(0, np.int64), np.zeros(0, np.int64)
    pix, ids, uv = rend.pix[m], rend.idx[m], rend.uv[m]
    score = uv[:, 0] * nx.ravel()[pix] + uv[:, 1] * ny.ravel()[pix]
    order = np.lexsort((-score, pix))
    ps = pix[order]
    first = np.ones(len(ps), bool)
    first[1:] = ps[1:] != ps[:-1]
    return ps[first], ids[order[first]]


def render(Pc: np.ndarray, K: np.ndarray, crop, closing: int = 1) -> Render:
    """Point z-buffer of camera-frame samples Pc (N,3) on the crop window (y0, y1, x0, x1) of the full image."""
    y0, y1, x0, x1 = (int(v) for v in crop)
    h, w = y1 - y0, x1 - x0
    z = Pc[:, 2]
    good = z > 0.05
    uv = np.full((len(Pc), 2), -1.0)
    uv[good] = (Pc[good, :2] / z[good, None]) @ K[:2, :2].T + K[:2, 2]
    j = np.floor(uv[:, 0]).astype(np.int64) - x0
    i = np.floor(uv[:, 1]).astype(np.int64) - y0
    inside = good & (i >= 0) & (j >= 0) & (i < h) & (j < w)
    idx = np.flatnonzero(inside)
    pix = i[idx] * w + j[idx]
    zq = np.minimum((z[idx] * 1e5).astype(np.int64), 2**31 - 1)
    order = np.argsort(pix * 2**31 + zq, kind="stable")
    pix_s = pix[order]
    first = np.ones(len(order), bool)
    first[1:] = pix_s[1:] != pix_s[:-1]
    win = idx[order[first]]
    front = np.full(h * w, -1, np.int64)
    front[pix_s[first]] = win
    zbuf = np.full(h * w, np.inf)
    zbuf[pix_s[first]] = z[win]
    front, zbuf = front.reshape(h, w), zbuf.reshape(h, w)
    mask = front >= 0
    extra = dict(idx=idx, pix=pix, uv=uv[idx])
    if closing:
        closed = ndimage.binary_closing(mask, iterations=closing, border_value=0)
        fill = closed & ~mask
        if fill.any():  # filled holes take the nearest rendered sample (for the 3D point / depth)
            _, (ii, jj) = ndimage.distance_transform_edt(~mask, return_indices=True)
            front[fill] = front[ii[fill], jj[fill]]
            zbuf[fill] = zbuf[ii[fill], jj[fill]]
        mask = closed
    return Render(mask, zbuf, front, **extra)


@dataclass
class FrameFactors:
    """Linearisation data of one (frame, camera) at fixed support. Points are in the BASE frame (X) with their
    base-frame velocities (V) for the time-offset derivative."""

    cam: str
    kind: np.ndarray  # (M,) 0 = sil_m2o, 1 = sil_o2m, 2 = depth
    X: np.ndarray  # (M,3) model point (base frame)
    V: np.ndarray  # (M,3) model point velocity (base frame, m/s)
    sdf: np.ndarray | None  # (h,w) observed SDF crop (sil_m2o)
    crop: tuple
    b: np.ndarray  # (M,2) observed boundary pixel (sil_o2m) [full-image continuous coords]
    n2: np.ndarray  # (M,2) image normal (sil_o2m)
    obs_ray: np.ndarray  # (M,3) d * K^-1 [u,1] (depth; camera frame, before the scale exp(alpha))
    nrm: np.ndarray  # (M,3) model normal (base frame) (depth)
    sig: np.ndarray | None = None  # (M,) depth-term whitening (m), frozen at support selection
    episode: str = ""
    t_index: int = -1


def frame_factors(cam, X_posed, N_posed, V_posed, T, K, ev: dict, cfg: RobotTermConfig, rng, alpha=0.0,
                  tau=0.0) -> tuple[FrameFactors | None, dict]:
    """Select the support of one (frame, camera) at the current estimate T. ``X_posed`` (N,3) base-frame samples at
    the camera timestamp; ``N_posed(ids)`` / ``V_posed(ids)`` return normals / velocities of selected samples (arrays
    are accepted too). The support is rendered at the timestamp (|tau| <= tens of ms moves it < 1 px); the factors
    carry the velocity so that tau enters the residuals."""
    if not callable(N_posed):
        N_arr, V_arr = N_posed, V_posed
        N_posed, V_posed = (lambda ids: N_arr[ids]), (lambda ids: V_arr[ids])
    crop = tuple(int(v) for v in ev["crop"])
    y0, y1, x0, x1 = crop
    sem = ev["sem"]
    names = ev["class_names"]
    robot = sem == names.index("robot")
    other = np.isin(sem, [names.index(c) for c in ("object", "fixture") if c in names])
    depth = ev["depth"]  # metres, 0 invalid, crop
    H, W = ev["hw"]
    Xt = X_posed
    Pc = Xt @ T[:3, :3].T + T[:3, 3]
    rend = render(Pc, K, crop, cfg.closing)
    stats = {"rendered_px": int(rend.mask.sum()), "observed_px": int(robot.sum())}
    if not rend.mask.any():
        return None, stats
    h, w = robot.shape
    ii, jj = np.mgrid[0:h, 0:w]
    gi, gj = ii + y0, jj + x0
    near_border = (gi < cfg.border_px) | (gj < cfg.border_px) | (gi >= H - cfg.border_px) | (gj >= W - cfg.border_px)
    crop_border = (ii < cfg.border_px) | (jj < cfg.border_px) | (ii >= h - cfg.border_px) | (jj >= w - cfg.border_px)
    near_other = ndimage.binary_dilation(other, iterations=cfg.adjacent_px) if other.any() else np.zeros_like(other)
    bad = near_border | crop_border | near_other
    zmod = rend.zbuf
    # occluders: non-robot pixels whose observed depth is clearly in front of the robot surface there
    occl = (~robot) & (depth > 0) & np.isfinite(zmod) & (depth < zmod - cfg.occluder_m)
    sdf_obs = sdf_of_mask(robot)

    kinds, Xs, Vs, bs, n2s, rays, nrms = [], [], [], [], [], [], []
    # --- model -> observed contour
    bm = boundary(rend.mask) & ~bad & ~ndimage.binary_dilation(occl, iterations=1)
    bi, bj = np.nonzero(bm)
    if cfg.contour_stride > 1:
        bi, bj = bi[::cfg.contour_stride], bj[::cfg.contour_stride]
    rnx, rny = outward_normals(rend.mask)
    _, fid = extreme_contour_samples(rend, bi * w + bj, rnx, rny)
    Xs.append(Xt[fid])
    Vs.append(V_posed(fid))
    kinds.append(np.zeros(len(fid), np.int8))
    bs.append(np.zeros((len(fid), 2)))
    n2s.append(np.zeros((len(fid), 2)))
    rays.append(np.zeros((len(fid), 3)))
    nrms.append(np.zeros((len(fid), 3)))
    stats["m2o"] = int(len(fid))
    # --- observed -> model contour (point-to-line to the nearest rendered boundary pixel)
    bo = boundary(robot)
    outside = ~robot
    # an observed boundary pixel is an occlusion boundary if a neighbouring non-robot pixel is an occluder
    occl_edge = ndimage.binary_dilation(occl & outside, iterations=2)
    # pose-independent part (observed depth only): a non-robot pixel within 2 px that is clearly in front of the
    # robot's observed depth nearby
    rob_d = np.where(robot & (depth > 0), depth, 0.0) if depth is not None else np.zeros_like(sem, float)
    near_rob_d = ndimage.maximum_filter(rob_d, size=7)
    front_occ = outside & (depth > 0) & (near_rob_d > 0) & (depth < near_rob_d - cfg.occluder_m)
    occl_obs = ndimage.binary_dilation(front_occ, iterations=2)
    bo &= ~bad & ~occl_edge & ~occl_obs
    oi, oj = np.nonzero(bo)
    if cfg.contour_stride > 1:
        oi, oj = oi[::cfg.contour_stride], oj[::cfg.contour_stride]
    rb = boundary(rend.mask)
    if len(oi) and rb.any():
        _, (ni, nj) = ndimage.distance_transform_edt(~rb, return_indices=True)
        mi, mj = ni[oi, oj], nj[oi, oj]
        # model contour point: the extreme sample of the nearest rendered boundary pixel; observed contour point:
        # the observed boundary pixel's edge (centre + half a pixel along the observed outward normal)
        pix_m, fid_m = extreme_contour_samples(rend, mi * w + mj, rnx, rny)
        lut = np.full(h * w, -1, np.int64)
        lut[pix_m] = fid_m
        fid_all = lut[mi * w + mj]
        onx, ony = outward_normals(robot)
        ok = fid_all >= 0
        fid = fid_all[ok]
        oi, oj, mi, mj = oi[ok], oj[ok], mi[ok], mj[ok]
        n2 = np.stack([rnx[mi, mj], rny[mi, mj]], 1)
        step = -sdf_obs[oi, oj]  # pixel centre -> zero level of the observed SDF along its normal (same contour as m2o)
        ob = np.stack([oj + x0 + 0.5 + step * onx[oi, oj], oi + y0 + 0.5 + step * ony[oi, oj]], 1)
        ok = np.ones(len(fid), bool)
        nn = np.ones(len(fid))
        Xs.append(Xt[fid])
        Vs.append(V_posed(fid))
        kinds.append(np.ones(len(fid), np.int8))
        bs.append(ob)
        n2s.append(n2[ok] / nn[ok, None])
        rays.append(np.zeros((len(fid), 3)))
        nrms.append(np.zeros((len(fid), 3)))
        stats["o2m"] = int(len(fid))
    # --- depth point-to-plane with projective association
    if depth is not None:
        er = ndimage.binary_erosion(robot, iterations=cfg.depth_erode_px)
        dmax = ndimage.maximum_filter(np.where(depth > 0, depth, -np.inf), size=5)
        dmin = ndimage.minimum_filter(np.where(depth > 0, depth, np.inf), size=5)
        smooth = (depth > 0) & (dmax - dmin <= cfg.depth_smooth_rel * depth)
        cand = er & smooth & rend.mask & ~bad
        ci, cj = np.nonzero(cand)
        stats["depth_candidates"] = int(len(ci))
        if len(ci) > cfg.depth_points:
            pick = rng.choice(len(ci), cfg.depth_points, replace=False)
            ci, cj = ci[pick], cj[pick]
        if len(ci):
            fid = rend.front[ci, cj]
            u = np.stack([cj + x0 + 0.5, ci + y0 + 0.5], 1)
            d = depth[ci, cj]
            ray = np.stack([(u[:, 0] - K[0, 2]) / K[0, 0], (u[:, 1] - K[1, 2]) / K[1, 1], np.ones(len(u))], 1) * d[:, None]
            Xs.append(Xt[fid])
            Vs.append(V_posed(fid))
            kinds.append(np.full(len(fid), 2, np.int8))
            bs.append(np.zeros((len(fid), 2)))
            n2s.append(np.zeros((len(fid), 2)))
            rays.append(ray)
            nrms.append(N_posed(fid))
            stats["depth"] = int(len(fid))
    if not Xs:
        return None, stats
    ff = FrameFactors(cam, np.concatenate(kinds), np.concatenate(Xs), np.concatenate(Vs), sdf_obs, crop,
                      np.concatenate(bs), np.concatenate(n2s), np.concatenate(rays), np.concatenate(nrms))
    # depth whitening frozen at this estimate (a weight, not part of the residual's derivative)
    nc = ff.nrm @ T[:3, :3].T
    proj = np.einsum("mi,mi->m", nc, np.exp(alpha) * ff.obs_ray)
    ff.sig = np.sqrt((cfg.sigma_depth_rel * proj) ** 2 + cfg.sigma_model_m ** 2)
    # fixed-denominator coverage statistics at this estimate
    stats["iou"] = float((rend.mask & robot).sum() / max(1, (rend.mask | robot).sum()))
    return ff, stats


def residuals_and_jacobians(ff: FrameFactors, T: np.ndarray, K: np.ndarray, alpha: float, tau: float,
                            cfg: RobotTermConfig):
    """Whitened residuals r (M,) and Jacobians (M, 8) w.r.t. [xi_c (6), alpha_c, tau_c] at fixed support."""
    R, t = T[:3, :3], T[:3, 3]
    Xb = ff.X + tau * ff.V
    Pc = Xb @ R.T + t
    Vc = ff.V @ R.T
    M = len(Pc)
    r = np.zeros(M)
    J = np.zeros((M, 8))
    uv, Dpi = _project(Pc, K)
    # d Pc / d xi = [-[Pc]x, I]
    dP = np.zeros((M, 3, 6))
    dP[:, 0, 1], dP[:, 0, 2] = Pc[:, 2], -Pc[:, 1]
    dP[:, 1, 0], dP[:, 1, 2] = -Pc[:, 2], Pc[:, 0]
    dP[:, 2, 0], dP[:, 2, 1] = Pc[:, 1], -Pc[:, 0]
    dP[:, :, 3:] = np.eye(3)
    duv_dxi = np.einsum("mij,mjk->mik", Dpi, dP)  # (M,2,6)
    duv_dtau = np.einsum("mij,mj->mi", Dpi, Vc)  # (M,2)
    s = ff.kind == 0
    if s.any():
        y0, y1, x0, x1 = ff.crop
        loc = uv[s] - np.array([x0, y0])
        val, gu, gv = bilinear(ff.sdf, loc, fill=np.nan)
        g = np.stack([gu, gv], 1)
        far = ~np.isfinite(val)  # projected outside the crop: a large constant (rejected by the robust loss)
        val = np.where(far, 50.0, val)
        r[s] = val / cfg.sigma_sil_px
        J[s, :6] = np.einsum("mi,mik->mk", g, duv_dxi[s]) / cfg.sigma_sil_px
        J[s, 7] = np.einsum("mi,mi->m", g, duv_dtau[s]) / cfg.sigma_sil_px
    s = ff.kind == 1
    if s.any():
        n2 = ff.n2[s]
        r[s] = np.einsum("mi,mi->m", n2, uv[s] - ff.b[s]) / cfg.sigma_sil_px
        J[s, :6] = np.einsum("mi,mik->mk", n2, duv_dxi[s]) / cfg.sigma_sil_px
        J[s, 7] = np.einsum("mi,mi->m", n2, duv_dtau[s]) / cfg.sigma_sil_px
    s = ff.kind == 2
    if s.any():
        nc = ff.nrm[s] @ R.T
        Xo = np.exp(alpha) * ff.obs_ray[s]
        raw = np.einsum("mi,mi->m", nc, Xo - Pc[s])
        if ff.sig is None:
            raise ValueError("depth factors need the whitening frozen at support selection (FrameFactors.sig)")
        sig = ff.sig[s]
        r[s] = raw / sig
        J[s, :3] = np.cross(nc, Xo) / sig[:, None]
        J[s, 3:6] = -nc / sig[:, None]
        J[s, 6] = np.einsum("mi,mi->m", nc, Xo) / sig
        J[s, 7] = -np.einsum("mi,mi->m", nc, Vc[s]) / sig
    return r, J


@dataclass
class Packed:
    """All frame factors of ONE camera concatenated (vectorised residuals); ``sdf`` crops packed in a flat atlas."""

    cam: str
    kind: np.ndarray
    X: np.ndarray
    V: np.ndarray
    b: np.ndarray
    n2: np.ndarray
    obs_ray: np.ndarray
    nrm: np.ndarray
    sig: np.ndarray
    frame: np.ndarray  # (M,) index into ``frames``
    atlas: np.ndarray  # flat float32
    geo: np.ndarray  # (F, 5) int: atlas offset, h, w, y0, x0
    frames: list  # (episode, t_index)


def pack(factors: list) -> dict:
    """{camera: Packed} from a list of FrameFactors (None entries skipped)."""
    by = {}
    for ff in factors:
        if ff is not None:
            by.setdefault(ff.cam, []).append(ff)
    out = {}
    for cam, lst in by.items():
        offs, geo, atl = 0, [], []
        for ff in lst:
            y0, y1, x0, x1 = ff.crop
            h, w = ff.sdf.shape
            geo.append((offs, h, w, y0, x0))
            atl.append(ff.sdf.astype(np.float32).ravel())
            offs += h * w
        frame = np.concatenate([np.full(len(ff.kind), k, np.int32) for k, ff in enumerate(lst)])
        out[cam] = Packed(cam, np.concatenate([ff.kind for ff in lst]), np.concatenate([ff.X for ff in lst]),
                          np.concatenate([ff.V for ff in lst]), np.concatenate([ff.b for ff in lst]),
                          np.concatenate([ff.n2 for ff in lst]), np.concatenate([ff.obs_ray for ff in lst]),
                          np.concatenate([ff.nrm for ff in lst]), np.concatenate([ff.sig for ff in lst]),
                          frame, np.concatenate(atl),
                          np.asarray(geo, np.int64), [(ff.episode, ff.t_index) for ff in lst])
    return out


def atlas_bilinear(pk: Packed, uv: np.ndarray, frame: np.ndarray, fill=np.nan):
    """Bilinear SDF value + gradient at full-image continuous uv of the frames ``frame`` (crop-local lookup)."""
    g = pk.geo[frame]
    off, h, w, y0, x0 = g[:, 0], g[:, 1], g[:, 2], g[:, 3], g[:, 4]
    x, y = uv[:, 0] - x0 - 0.5, uv[:, 1] - y0 - 0.5
    xf, yf = np.floor(x), np.floor(y)
    ok = np.isfinite(x) & np.isfinite(y) & (xf >= 0) & (yf >= 0) & (xf < w - 1) & (yf < h - 1)
    xi = np.where(ok, xf, 0).astype(np.int64)
    yi = np.where(ok, yf, 0).astype(np.int64)
    fx, fy = np.where(ok, x - xf, 0), np.where(ok, y - yf, 0)
    base = off + yi * w + xi
    a, b = pk.atlas[base], pk.atlas[base + 1]
    c, d = pk.atlas[base + w], pk.atlas[base + w + 1]
    val = a * (1 - fx) * (1 - fy) + b * fx * (1 - fy) + c * (1 - fx) * fy + d * fx * fy
    gu = (b - a) * (1 - fy) + (d - c) * fy
    gv = (c - a) * (1 - fx) + (d - b) * fx
    return np.where(ok, val, fill), np.where(ok, gu, 0.0), np.where(ok, gv, 0.0)


def packed_residuals(pk: Packed, T: np.ndarray, K: np.ndarray, alpha: float, tau: float, cfg: RobotTermConfig,
                     jac: bool = True):
    """Whitened residuals (M,) and Jacobians (M, 8) [xi(6), alpha, tau] of one camera's packed factors."""
    R, t = T[:3, :3], T[:3, 3]
    Pc = (pk.X + tau * pk.V) @ R.T + t
    M = len(Pc)
    r = np.zeros(M)
    J = np.zeros((M, 8)) if jac else None
    uv, Dpi = _project(Pc, K)
    if jac:
        Vc = pk.V @ R.T
        dP = np.zeros((M, 3, 6))
        dP[:, 0, 1], dP[:, 0, 2] = Pc[:, 2], -Pc[:, 1]
        dP[:, 1, 0], dP[:, 1, 2] = -Pc[:, 2], Pc[:, 0]
        dP[:, 2, 0], dP[:, 2, 1] = Pc[:, 1], -Pc[:, 0]
        dP[:, :, 3:] = np.eye(3)
        duv_dxi = np.einsum("mij,mjk->mik", Dpi, dP)
        duv_dtau = np.einsum("mij,mj->mi", Dpi, Vc)
    s = np.flatnonzero(pk.kind == 0)
    if len(s):
        val, gu, gv = atlas_bilinear(pk, uv[s], pk.frame[s], fill=np.nan)
        val = np.where(np.isfinite(val), val, 50.0)
        r[s] = val / cfg.sigma_sil_px
        if jac:
            gr = np.stack([gu, gv], 1)
            J[s, :6] = np.einsum("mi,mik->mk", gr, duv_dxi[s]) / cfg.sigma_sil_px
            J[s, 7] = np.einsum("mi,mi->m", gr, duv_dtau[s]) / cfg.sigma_sil_px
    s = np.flatnonzero(pk.kind == 1)
    if len(s):
        n2 = pk.n2[s]
        r[s] = np.einsum("mi,mi->m", n2, uv[s] - pk.b[s]) / cfg.sigma_sil_px
        if jac:
            J[s, :6] = np.einsum("mi,mik->mk", n2, duv_dxi[s]) / cfg.sigma_sil_px
            J[s, 7] = np.einsum("mi,mi->m", n2, duv_dtau[s]) / cfg.sigma_sil_px
    s = np.flatnonzero(pk.kind == 2)
    if len(s):
        nc = pk.nrm[s] @ R.T
        Xo = np.exp(alpha) * pk.obs_ray[s]
        proj = np.einsum("mi,mi->m", nc, Xo)
        sig = pk.sig[s]
        r[s] = np.einsum("mi,mi->m", nc, Xo - Pc[s]) / sig
        if jac:
            J[s, :3] = np.cross(nc, Xo) / sig[:, None]
            J[s, 3:6] = -nc / sig[:, None]
            J[s, 6] = proj / sig
            J[s, 7] = -np.einsum("mi,mi->m", nc, Vc[s]) / sig
    return r, J
