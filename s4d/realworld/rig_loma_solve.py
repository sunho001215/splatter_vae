"""Joint per-task rig solve with render-to-image robot 2D-3D correspondences (``rig_loma``).

DepthWorld's robot reprojection term ("l", App. B.2) replaces the SAM-silhouette / masked robot-depth factors of
``rig_dw``: every LoMa-G match between the robot rendered at the render pose and the real image gives a robot surface
point X (base frame, FK at the camera timestamp, surface velocity V) and an observed pixel u, and

    r_l = (pi(K_c, T_c (X + tau_c V)) - u) / sigma_px            (2 rows, Cauchy on the whitened norm)

The 3D point is fixed data (it comes from the render geometry, not from the current estimate), so the term needs no
support re-selection; new correspondences are extracted by re-rendering at an updated pose (outer loop, GPU worker).
Optional secondary RGB-D term on the same matches (DepthWorld "3d"-like, validation only by default):

    r_d = (log z_c(X) - log d_obs(u) - alpha_c) / sigma_logd    (1 row, Cauchy)

The static cross-camera RGB-D factors (``rig_dw_extext``, RoMa v2 matches with SAM-free static flags) are kept
unchanged; stages and parametrisation are those of ``rig_dw_solve.RigProblem`` (shape from ext-ext -> common Sim(3) +
tau -> full). Optional per-task gripper-mount correction (6 DoF, rigid transform of every gripper link about the
panda_link7 frame) for the identifiability analysis (``mount_information``); it is NOT estimated by default."""

from __future__ import annotations

from dataclasses import dataclass, field, replace

import numpy as np

from s4d.realworld.rig_dw_geom import project
from s4d.realworld.rig_dw_robot import RobotTermConfig
from s4d.realworld.rig_dw_solve import NPC, RigProblem, SolveConfig, State


@dataclass
class CorrSet:
    """Robot 2D-3D correspondences of ONE camera (packed)."""

    X: np.ndarray  # (M,3) base frame at the camera timestamp
    V: np.ndarray  # (M,3) base-frame surface velocity (m/s)
    uv: np.ndarray  # (M,2) observed pixel (continuous, pixel centres at +.5)
    episode: np.ndarray  # (M,) int
    frame: np.ndarray  # (M,) int global frame id
    link: np.ndarray  # (M,) int
    d: np.ndarray | None = None  # (M,) observed depth at uv (metres, NaN invalid) of the selected depth stage
    w: np.ndarray | None = None  # (M,) per-match weight (default 1)
    meta: dict = field(default_factory=dict)
    T7: np.ndarray | None = None  # (M,4,4) panda_link7 pose of each match's frame (mount analysis)

    def __len__(self):
        return len(self.X)

    def subset(self, m):
        return CorrSet(self.X[m], self.V[m], self.uv[m], self.episode[m], self.frame[m], self.link[m],
                       None if self.d is None else self.d[m], None if self.w is None else self.w[m], dict(self.meta),
                       None if self.T7 is None else self.T7[m])


@dataclass
class LomaConfig:
    sigma_px: float = 1.0
    cauchy: float = 3.0
    use_corr: bool = True
    corr_cameras: tuple | None = None  # leave-camera-out: 2D-3D term only from these cameras
    depth_weight: float = 0.0  # secondary match-depth term (0 = validation only)
    sigma_logd: float = 0.01
    cauchy_depth: float = 3.0


def _dP_dxi(Pc):
    M = len(Pc)
    dP = np.zeros((M, 3, 6))
    dP[:, 0, 1], dP[:, 0, 2] = Pc[:, 2], -Pc[:, 1]
    dP[:, 1, 0], dP[:, 1, 2] = -Pc[:, 2], Pc[:, 0]
    dP[:, 2, 0], dP[:, 2, 1] = Pc[:, 1], -Pc[:, 0]
    dP[:, :, 3:] = np.eye(3)
    return dP


def corr_residuals(T, K, cs: CorrSet, tau: float, sigma: float, jac: bool = True):
    """Whitened 2D-3D residuals (M, 2) and Jacobians (M, 2, NPC) w.r.t. [xi (6), alpha, tau, focal]; K effective."""
    R, t = T[:3, :3], T[:3, 3]
    Pc = (cs.X + tau * cs.V) @ R.T + t
    uv, Dpi = project(Pc, K)
    r = (uv - cs.uv) / sigma
    if not jac:
        return r, None
    J = np.zeros((len(Pc), 2, NPC))
    J[:, :, :6] = np.einsum("mij,mjk->mik", Dpi, _dP_dxi(Pc)) / sigma
    J[:, :, 7] = np.einsum("mij,mj->mi", Dpi, cs.V @ R.T) / sigma
    J[:, :, 8] = (uv - K[:2, 2]) / sigma
    return r, J


def corr_depth_residuals(T, cs: CorrSet, alpha: float, tau: float, sigma: float, jac: bool = True):
    """(log z_c - log d - alpha) / sigma at matches with valid depth; (r (M,), J (M, NPC), valid mask)."""
    ok = np.isfinite(cs.d) & (cs.d > 0) if cs.d is not None else np.zeros(len(cs), bool)
    R, t = T[:3, :3], T[:3, 3]
    X = cs.X[ok] + tau * cs.V[ok]
    Pc = X @ R.T + t
    z = Pc[:, 2]
    r = (np.log(np.maximum(z, 1e-6)) - np.log(cs.d[ok]) - alpha) / sigma
    if not jac:
        return r, None, ok
    J = np.zeros((len(z), NPC))
    J[:, :6] = _dP_dxi(Pc)[:, 2, :] / z[:, None] / sigma
    J[:, 6] = -1.0 / sigma
    J[:, 7] = (cs.V[ok] @ R.T)[:, 2] / z / sigma
    return r, J, ok


def _cauchy(rn2, c):
    return c ** 2 * np.log1p(rn2 / c ** 2), 1.0 / (1.0 + rn2 / c ** 2)


class LomaProblem(RigProblem):
    """``RigProblem`` with the robot evidence replaced by 2D-3D correspondences (no SAM, no support selection)."""

    def __init__(self, cams, K, matches, corr: dict, lcfg: LomaConfig, scfg: SolveConfig, pivot=None):
        super().__init__(cams, K, matches, [], None, RobotTermConfig(), replace(scfg, use_robot=False), pivot=pivot)
        self.corr = {c: v for c, v in corr.items() if c in self.cams and v is not None and len(v)}
        self.lcfg = lcfg
        self.support = []  # enables the robot / combined blocks of ``information``

    def _corr_cams(self):
        cams = [c for c in self.cams if c in self.corr]
        if self.lcfg.corr_cameras is not None:
            cams = [c for c in cams if c in self.lcfg.corr_cameras]
        return cams

    def corr_normal(self, s: State, jac: bool = True):
        n = self.n
        H, g, cost, parts = np.zeros((n, n)), np.zeros(n), 0.0, {"corr": 0.0, "corr_depth": 0.0}
        Ke = self.Keff(s)
        for c in self._corr_cams():
            cs = self.corr[c]
            k = self.cams.index(c)
            cols = slice(k * NPC, (k + 1) * NPC)
            w0 = np.ones(len(cs)) if cs.w is None else cs.w
            if self.lcfg.use_corr:
                r, J = corr_residuals(s.T[c], Ke[c], cs, s.tau[c], self.lcfg.sigma_px, jac)
                rho, w = _cauchy((r ** 2).sum(1), self.lcfg.cauchy)
                cost += float((w0 * rho).sum())
                parts["corr"] += float((w0 * rho).sum())
                if jac:
                    ww = w0 * w
                    H[cols, cols] += np.einsum("m,mia,mib->ab", ww, J, J)
                    g[cols] += np.einsum("m,mia,mi->a", ww, J, r)
            if self.lcfg.depth_weight > 0 and cs.d is not None:
                r, J, ok = corr_depth_residuals(s.T[c], cs, s.alpha[c], s.tau[c], self.lcfg.sigma_logd, jac)
                rho, w = _cauchy(r ** 2, self.lcfg.cauchy_depth)
                lw = self.lcfg.depth_weight * w0[ok]
                cost += float((lw * rho).sum())
                parts["corr_depth"] += float((lw * rho).sum())
                if jac:
                    H[cols, cols] += (J * (lw * w)[:, None]).T @ J
                    g[cols] += (J * (lw * w)[:, None]).T @ r
        return H, g, cost, parts

    def normal(self, s: State, robot=True, extext=True, priors=True):
        H, g, cost, parts = super().normal(s, robot=False, extext=extext, priors=priors)
        if robot:
            Hc, gc, cc, pc = self.corr_normal(s)
            H += Hc
            g += gc
            cost += cc
            parts.update(pc)
        return H, g, cost, parts

    def cost(self, s: State, robot=True, extext=True):
        c = super().cost(s, robot=False, extext=extext)
        if robot:
            c += self.corr_normal(s, jac=False)[2]
        return c

    def solve_stage(self, s: State, stage: str, log=None, robot=True, extext=True):
        return self.lm(s, stage, self.cfg.inner_iters * 4, robot=robot, extext=extext, log=log)

    # ------------------------------------------------------------------ statistics
    def corr_stats(self, s: State, corr: dict | None = None) -> dict:
        """Unwhitened 2D-3D reprojection statistics per camera (px) and match-depth log residual (%)."""
        corr = corr if corr is not None else self.corr
        Ke = self.Keff(s)
        out = {}
        for c, cs in corr.items():
            if cs is None or not len(cs):
                continue
            r, _ = corr_residuals(s.T[c], Ke[c], cs, s.tau[c], 1.0, jac=False)
            e = np.linalg.norm(r, axis=1)
            d = {"n": int(len(e)), "median_px": float(np.median(e)), "p90_px": float(np.percentile(e, 90)),
                 "mean_du_dv_px": np.round(r.mean(0), 3).tolist(), "le3px": float((e <= 3).mean())}
            if cs.d is not None:
                rd, _, ok = corr_depth_residuals(s.T[c], cs, s.alpha[c], s.tau[c], 1.0, jac=False)
                if ok.any():
                    d["depth_log_median_pct"] = float(100 * np.median(rd))
                    d["depth_abs_median_mm"] = float(1e3 * np.median(np.abs(np.exp(rd) - 1) * cs.d[ok]))
                    d["depth_n"] = int(ok.sum())
            # per-link mean residual (systematic model / kinematic errors)
            links = {}
            for l in np.unique(cs.link):
                m = cs.link == l
                if m.sum() >= 30:
                    links[int(l)] = {"n": int(m.sum()), "median_px": float(np.median(e[m])),
                                     "mean_du_dv_px": np.round(r[m].mean(0), 2).tolist()}
            d["links"] = links
            out[c] = d
        return out


# ---------------------------------------------------------------------------------------------- gripper mount block
def mount_information(s: State, K: dict, corr: dict, gripper_links: set, T7: dict, sigma_px: float = 1.0,
                      cauchy: float = 3.0) -> dict:
    """Identifiability of a per-task gripper-mount correction dM (6 DoF twist in the panda_link7 frame applied to every
    gripper link: X' = T7 exp(dM) T7^-1 X) given the rig: Fisher information of the 2D-3D term (robust weights at the
    solution) for [dM] alone and with the 18 camera-pose columns marginalised (Schur complement). ``T7[cam]`` (M,4,4)
    is the panda_link7 pose of each correspondence. Returns eigenvalues in units of (1 mm, 0.1 deg) and the marginal
    standard deviations."""
    cams = [c for c in corr if c in s.T]
    nC = 6 * len(cams)
    H = np.zeros((nC + 6, nC + 6))
    for k, c in enumerate(cams):
        cs = corr[c]
        g = np.isin(cs.link, list(gripper_links))
        if not g.any():
            continue
        R, t = s.T[c][:3, :3], s.T[c][:3, 3]
        X = cs.X[g] + s.tau[c] * cs.V[g]
        Pc = X @ R.T + t
        uv, Dpi = project(Pc, K[c])
        r = (uv - cs.uv[g]) / sigma_px
        w = 1.0 / (1.0 + (r ** 2).sum(1) / cauchy ** 2)
        Jc = np.einsum("mij,mjk->mik", Dpi, _dP_dxi(Pc)) / sigma_px  # (m,2,6) camera twist
        # dX/d(dM): dM twist (w, v) in the link-7 frame: X7 = T7^-1 X; dX = R7 (w x X7 + v)
        T7c = T7[c][g]
        R7 = T7c[:, :3, :3]
        X7 = np.einsum("mji,mj->mi", R7, X - T7c[:, :3, 3])
        dX = np.zeros((len(X), 3, 6))
        sk = np.zeros((len(X), 3, 3))
        sk[:, 0, 1], sk[:, 0, 2] = X7[:, 2], -X7[:, 1]
        sk[:, 1, 0], sk[:, 1, 2] = -X7[:, 2], X7[:, 0]
        sk[:, 2, 0], sk[:, 2, 1] = X7[:, 1], -X7[:, 0]
        dX[:, :, :3] = np.einsum("mij,mjk->mik", R7, sk)  # sk = -[X7]x = d(w x X7)/dw
        dX[:, :, 3:] = R7
        Jm = np.einsum("mij,jk,mkl->mil", Dpi, R, dX) / sigma_px  # (m,2,6)
        J = np.concatenate([np.zeros((len(X), 2, nC)), Jm], 2)
        J[:, :, 6 * k:6 * k + 6] = Jc
        H += np.einsum("m,mia,mib->ab", w, J, J)
    D = np.diag([np.radians(0.1)] * 3 + [0.001] * 3)  # units: 0.1 deg, 1 mm
    Hmm = D @ H[nC:, nC:] @ D
    Hcc, Hcm = H[:nC, :nC], H[:nC, nC:]
    Hs = H[nC:, nC:] - Hcm.T @ np.linalg.pinv(Hcc) @ Hcm
    Hs = D @ Hs @ D
    out = {}
    for name, M in (("alone", Hmm), ("marginal_over_cameras", Hs)):
        w_, V = np.linalg.eigh((M + M.T) / 2)
        cov = np.linalg.pinv(M)
        out[name] = {"eig": w_.tolist(), "sd_units": np.sqrt(np.clip(np.diag(cov), 0, None)).tolist(),
                     "weakest": V[:, 0].tolist(), "cond": float(w_[-1] / max(w_[0], 1e-300))}
    out["units"] = "rotation sd in 0.1 deg, translation sd in 1 mm (link-7 frame, x y z); naive (independent matches)"
    return out


def _mount_jac(s: State, K: dict, cs: CorrSet, c: str, g: np.ndarray, sigma_px: float):
    R, t = s.T[c][:3, :3], s.T[c][:3, 3]
    X = cs.X[g] + s.tau[c] * cs.V[g]
    Pc = X @ R.T + t
    uv, Dpi = project(Pc, K[c])
    r = (uv - cs.uv[g]) / sigma_px
    T7c = cs.T7[g]
    R7 = T7c[:, :3, :3]
    X7 = np.einsum("mji,mj->mi", R7, X - T7c[:, :3, 3])
    sk = np.zeros((len(X), 3, 3))
    sk[:, 0, 1], sk[:, 0, 2] = X7[:, 2], -X7[:, 1]
    sk[:, 1, 0], sk[:, 1, 2] = -X7[:, 2], X7[:, 0]
    sk[:, 2, 0], sk[:, 2, 1] = X7[:, 1], -X7[:, 0]
    dX = np.concatenate([np.einsum("mij,mjk->mik", R7, sk), R7], 2)
    return r, np.einsum("mij,jk,mkl->mil", Dpi, R, dX) / sigma_px


def apply_mount(corr: dict, gripper_links: set, dM: np.ndarray) -> dict:
    """Corrected correspondences: gripper-link points X' = T7 exp(dM) T7^-1 X (dM twist in the panda_link7 frame)."""
    from s4d.realworld.rig_dw_geom import se3_exp

    E = se3_exp(np.asarray(dM, float))
    out = {}
    for c, cs in corr.items():
        g = np.isin(cs.link, list(gripper_links))
        X = cs.X.copy()
        if g.any():
            T7 = cs.T7[g]
            X7 = np.einsum("mji,mj->mi", T7[:, :3, :3], X[g] - T7[:, :3, 3])
            X7 = X7 @ E[:3, :3].T + E[:3, 3]
            X[g] = np.einsum("mij,mj->mi", T7[:, :3, :3], X7) + T7[:, :3, 3]
        n = CorrSet(X, cs.V, cs.uv, cs.episode, cs.frame, cs.link, cs.d, cs.w, dict(cs.meta), cs.T7)
        out[c] = n
    return out


def estimate_mount(s: State, K: dict, corr: dict, gripper_links: set, sigma_px: float = 1.0, cauchy: float = 3.0,
                   iters: int = 8, prior_m: float = 0.02, prior_rad: float = 0.05) -> dict:
    """Gauss-Newton (Cauchy IRLS) estimate of the gripper-mount twist dM at FIXED camera poses (diagnostic), with a weak
    prior (2 cm / ~3 deg). Returns dM (rad, m) and the robust cost before/after on gripper-link matches."""
    dM = np.zeros(6)
    Pinv = np.diag([1 / prior_rad ** 2] * 3 + [1 / prior_m ** 2] * 3)
    hist = []
    for _ in range(iters):
        cc = apply_mount(corr, gripper_links, dM)
        H, gvec, cost = Pinv.copy(), Pinv @ dM, 0.0
        for c, cs in cc.items():
            g = np.isin(cs.link, list(gripper_links))
            if not g.any() or cs.T7 is None:
                continue
            r, J = _mount_jac(s, K, cs, c, g, sigma_px)
            n2 = (r ** 2).sum(1)
            w = 1.0 / (1.0 + n2 / cauchy ** 2)
            cost += float((cauchy ** 2 * np.log1p(n2 / cauchy ** 2)).sum())
            H += np.einsum("m,mia,mib->ab", w, J, J)
            gvec += np.einsum("m,mia,mi->a", w, J, r)
        hist.append(cost)
        step = -np.linalg.solve(H, gvec)
        dM = dM + step  # left-composition approx (small steps)
        if np.abs(step).max() < 1e-7:
            break
    return {"dM_rad_m": dM.tolist(), "dM_deg_mm": (np.r_[np.degrees(dM[:3]), 1e3 * dM[3:]]).tolist(), "cost_hist": hist}
