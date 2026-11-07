"""Joint per-task rig solve of ``rig_dw`` (board-free, DepthWorld-style): static cross-camera RGB-D factors
(``rig_dw_extext``) + robot factors (``rig_dw_robot``/``rig_dw_frames``) over the three cameras of ONE task.

Global parameter vector (per camera c in ``cams`` order, 8 columns): [xi_c (6, camera-frame left twist), alpha_c (log
depth scale: true z = exp(alpha_c) d_obs), tau_c (s, effective robot-time offset of the camera)].

Stages (``solve``):
 B  rig shape from ext-ext only (gauge: camera 0 pose and alpha_0 fixed);
 C1 common 7 (rigid motion of the whole rig in the base frame + common similarity scale about a workspace pivot)
    + tau, relative rig fixed: robot factors ground the rig;
 C2 everything free (ext-ext factors retained; weak priors only on alpha (0.05) and tau (0.05 s)).
Each robot stage alternates support selection (outer) and LM at fixed support (inner).
"""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass, field

import numpy as np

from s4d.realworld import rig_dw_extext as ee
from s4d.realworld import rig_dw_frames as rf
from s4d.realworld.rig_dw_geom import camera_centre, pose_distance, se3_exp, skew
from s4d.realworld.rig_dw_robot import RobotTermConfig

NPC = 8  # columns per camera


@dataclass
class SolveConfig:
    cauchy_ee: float = 3.0
    cauchy_robot: float = 3.0
    alpha_prior: float = 0.05
    tau_prior_s: float = 0.05
    estimate_tau: bool = True
    outer_iters: int = 6
    inner_iters: int = 8
    lm_lambda: float = 1e-3
    point_iters: int = 3
    converge_rot_deg: float = 0.002
    converge_centre_m: float = 0.0002
    processes: int = 8
    robot_kinds: tuple = (0, 1, 2)
    use_extext: bool = True
    use_robot: bool = True
    robot_cameras: tuple | None = None  # leave-camera-out: robot factors only from these cameras


@dataclass
class State:
    T: dict
    alpha: dict
    tau: dict
    P: np.ndarray | None = None

    def copy(self):
        return State({k: v.copy() for k, v in self.T.items()}, dict(self.alpha), dict(self.tau),
                     None if self.P is None else self.P.copy())

    def as_dict(self):
        return {"T": self.T, "alpha": self.alpha, "tau": self.tau}

    def to_json(self):
        return {"T_cam_base": {k: np.asarray(v).tolist() for k, v in self.T.items()},
                "alpha": {k: float(v) for k, v in self.alpha.items()},
                "tau_s": {k: float(v) for k, v in self.tau.items()}}


def adjoint(T):
    R, t = T[:3, :3], T[:3, 3]
    A = np.zeros((6, 6))
    A[:3, :3] = R
    A[3:, :3] = skew(t) @ R
    A[3:, 3:] = R
    return A


class RigProblem:
    def __init__(self, cams, K, matches: ee.MatchSet | None, frames, surf, rcfg: RobotTermConfig,
                 scfg: SolveConfig, pivot=None, episode_of_match=None):
        self.cams = list(cams)
        self.K = {c: np.asarray(K[c], float) for c in self.cams}
        self.M = matches
        self.frames = frames or []
        self.surf = surf
        self.rcfg = rcfg
        self.cfg = scfg
        self.n = NPC * len(self.cams)
        self.pivot = np.zeros(3) if pivot is None else np.asarray(pivot, float)
        self.idx_ee = ee.ParamIndex({c: (True, True) for c in self.cams}, order=self.cams)
        self.map_ee = np.concatenate([[k * NPC + j for j in range(7)] for k in range(len(self.cams))])
        self.support = None
        self.support_stats = None
        self._pack_cache = None
        self.episode_of_match = episode_of_match

    # ------------------------------------------------------------------ parametrisation
    def cols(self, cam):
        k = self.cams.index(cam)
        return list(range(k * NPC, k * NPC + NPC))

    def apply(self, s: State, delta) -> State:
        out = s.copy()
        for k, c in enumerate(self.cams):
            d = delta[k * NPC:(k + 1) * NPC]
            out.T[c] = se3_exp(d[:6]) @ out.T[c]
            out.alpha[c] = out.alpha[c] + d[6]
            out.tau[c] = out.tau[c] + d[7]
        return out

    def basis(self, s: State, stage: str) -> np.ndarray:
        n = self.n
        if stage == "full":
            cols = [j for k in range(len(self.cams)) for j in range(k * NPC, k * NPC + (8 if self.cfg.estimate_tau else 7))]
            return np.eye(n)[:, cols]
        if stage == "shape":  # camera 0 pose + alpha fixed, no tau
            cols = [j for k in range(1, len(self.cams)) for j in range(k * NPC, k * NPC + 7)]
            return np.eye(n)[:, cols]
        if stage == "common":
            B = np.zeros((n, 7 + (len(self.cams) if self.cfg.estimate_tau else 0)))
            for k, c in enumerate(self.cams):
                B[k * NPC:k * NPC + 6, :6] = -adjoint(s.T[c])
                C = camera_centre(s.T[c])
                B[k * NPC + 3:k * NPC + 6, 6] = -s.T[c][:3, :3] @ (C - self.pivot)
                B[k * NPC + 6, 6] = 1.0
                if self.cfg.estimate_tau:
                    B[k * NPC + 7, 7 + k] = 1.0
            return B
        raise ValueError(stage)

    # ------------------------------------------------------------------ costs
    def _ee_points(self, s: State, iters=None):
        if self.M is None or not self.cfg.use_extext:
            return None
        P = ee.init_points(s.T, self.K, self.M) if s.P is None else s.P
        return ee.refine_points(s.T, s.alpha, self.K, self.M, P, iters=iters or self.cfg.point_iters)

    def normal(self, s: State, robot=True, extext=True, priors=True):
        H, g, cost, parts = np.zeros((self.n, self.n)), np.zeros(self.n), 0.0, {}
        if extext and self.M is not None and self.cfg.use_extext:
            ne = ee.normal_equations(s.T, s.alpha, self.K, self.M, s.P, self.idx_ee, self.cfg.cauchy_ee)
            H[np.ix_(self.map_ee, self.map_ee)] += ne["H"]
            g[self.map_ee] += ne["g"]
            cost += ne["cost"]
            parts["extext"] = ne["cost"]
        if robot and self.support is not None and self.cfg.use_robot:
            Hr, gr, cr, nr = rf.robot_normal_equations(self._packed(), s.as_dict(), self.K, self.rcfg,
                                                       self.cols, self.n, self.cfg.cauchy_robot,
                                                       kinds=self.cfg.robot_kinds)
            H += Hr
            g += gr
            cost += cr
            parts["robot"] = cr
        if priors:
            for k, c in enumerate(self.cams):
                ia, it = k * NPC + 6, k * NPC + 7
                H[ia, ia] += 1 / self.cfg.alpha_prior ** 2
                g[ia] += s.alpha[c] / self.cfg.alpha_prior ** 2
                H[it, it] += 1 / self.cfg.tau_prior_s ** 2
                g[it] += s.tau[c] / self.cfg.tau_prior_s ** 2
                cost += (s.alpha[c] / self.cfg.alpha_prior) ** 2 + (s.tau[c] / self.cfg.tau_prior_s) ** 2
            parts["prior"] = cost - sum(v for k, v in parts.items() if k != "prior")
        return H, g, cost, parts

    def cost(self, s: State, robot=True, extext=True):
        c = 0.0
        if extext and self.M is not None and self.cfg.use_extext:
            c += ee.normal_equations(s.T, s.alpha, self.K, self.M, s.P, self.idx_ee, self.cfg.cauchy_ee)["cost"]
        if robot and self.support is not None and self.cfg.use_robot:
            c += rf.robot_cost_packed(self._packed(), s.as_dict(), self.K, self.rcfg, self.cfg.cauchy_robot,
                                      kinds=self.cfg.robot_kinds)
        for c_ in self.cams:
            c += (s.alpha[c_] / self.cfg.alpha_prior) ** 2 + (s.tau[c_] / self.cfg.tau_prior_s) ** 2
        return c

    def _robot_support(self):
        sup = self.support
        if self.cfg.robot_cameras is not None:
            sup = [f for f in sup if f is not None and f.cam in self.cfg.robot_cameras]
        return sup

    def _packed(self):
        if self._pack_cache is None:
            from s4d.realworld.rig_dw_robot import pack

            self._pack_cache = pack(self._robot_support())
        return self._pack_cache

    # ------------------------------------------------------------------ support
    def select(self, s: State):
        res = rf.select_support(self.frames, self.surf, s.as_dict(), self.rcfg, self.cfg.processes)
        keep, stats = [], []
        for ff, st in res:
            if ff is not None and st.get("max_surface_speed", 0.0) > self.rcfg.max_surface_speed_m_s:
                st["dropped"] = "speed"
                ff = None
            keep.append(ff)
            stats.append(st)
        self.support, self.support_stats = keep, stats
        self._pack_cache = None
        return stats

    # ------------------------------------------------------------------ LM at fixed support
    def lm(self, s: State, stage: str, iters: int, robot=True, extext=True, log=None):
        lam = self.cfg.lm_lambda
        if s.P is None:
            s = s.copy()
            s.P = self._ee_points(s, iters=5)
        H, g, cost, parts = self.normal(s, robot=robot, extext=extext)
        hist = [cost]
        for _ in range(iters):
            B = self.basis(s, stage)
            Hb, gb = B.T @ H @ B, B.T @ g
            accepted = False
            for _ in range(10):
                dz = ee.solve_step(Hb, gb, lam)
                trial = self.apply(s, B @ dz)
                try:
                    trial.P = self._ee_points(trial) if (extext and self.M is not None and self.cfg.use_extext) else s.P
                    Ht, gt, ct, pt = self.normal(trial, robot=robot, extext=extext)
                except (ValueError, np.linalg.LinAlgError):
                    ct = np.inf
                if ct < cost:
                    s, H, g, cost, parts = trial, Ht, gt, ct, pt
                    lam = max(lam * 0.3, 1e-9)
                    accepted = True
                    break
                lam *= 10
            hist.append(cost)
            if not accepted or (hist[-2] - hist[-1]) < 1e-9 * max(1.0, abs(hist[-1])):
                break
        if log is not None:
            log.append({"stage": stage, "cost": cost, "parts": parts, "iters": len(hist) - 1, "lambda": lam})
        return s

    def solve_stage(self, s: State, stage: str, log=None, robot=True, extext=True):
        if not robot or not self.cfg.use_robot or not self.frames:
            return self.lm(s, stage, self.cfg.inner_iters * 4, robot=False, extext=extext, log=log)
        for outer in range(self.cfg.outer_iters):
            self.select(s)
            prev = s.copy()
            s = self.lm(s, stage, self.cfg.inner_iters, robot=True, extext=extext, log=log)
            moves = [pose_distance(prev.T[c], s.T[c]) for c in self.cams]
            if log is not None:
                log[-1]["outer"] = outer
                log[-1]["move"] = [[float(a), float(b)] for a, b in moves]
            if max(m[0] for m in moves) < self.cfg.converge_rot_deg and \
                    max(m[1] for m in moves) < self.cfg.converge_centre_m:
                break
        self.select(s)
        return s

    def solve(self, s0: State, stages=("shape", "common", "full"), log=None) -> State:
        s = s0.copy()
        for st in stages:
            t0 = time.time()
            if st == "shape":
                s = self.solve_stage(s, "shape", log=log, robot=False)
            else:
                s = self.solve_stage(s, st, log=log)
            if log is not None:
                log.append({"stage_done": st, "seconds": round(time.time() - t0, 1)})
        return s

    # ------------------------------------------------------------------ information / uncertainty
    def scales(self) -> np.ndarray:
        """Physical units of one tolerance step per parameter: 0.35 deg, 8 mm, 0.01 log-scale, 10 ms."""
        d = []
        for _ in self.cams:
            d += [np.radians(0.35)] * 3 + [0.008] * 3 + [0.01, 0.010]
        return np.asarray(d)

    def information(self, s: State) -> dict:
        """Data-only (no priors, no damping) information in tolerance units, separately for ext-ext, robot and
        combined, with the weakest directions; marginal per-camera rotation / centre standard deviations."""
        D = np.diag(self.scales())
        out = {}
        for name, kw in (("extext", dict(robot=False, extext=True)), ("robot", dict(robot=True, extext=False)),
                         ("combined", dict(robot=True, extext=True))):
            H, _, _, _ = self.normal(s, priors=False, **kw)
            if not self.cfg.estimate_tau:
                for k in range(len(self.cams)):
                    H[k * NPC + 7, k * NPC + 7] += 1e12
            Hs = D @ H @ D
            w, V = np.linalg.eigh((Hs + Hs.T) / 2)
            out[name] = {"eig": w.tolist(), "weakest": [V[:, i].tolist() for i in range(4)]}
            if name == "combined":
                cov = np.linalg.pinv(H)
                out["marginal"] = self.marginal(s, cov)
                out["cov"] = cov
        return out

    def marginal(self, s: State, cov) -> dict:
        res = {}
        for k, c in enumerate(self.cams):
            C = cov[k * NPC:k * NPC + 8, k * NPC:k * NPC + 8]
            R = s.T[c][:3, :3]
            rot_sd = np.degrees(np.sqrt(max(np.trace(C[:3, :3]), 0)))
            centre_cov = R.T @ C[3:6, 3:6] @ R  # dC = -R^T v
            res[c] = {"rot_sd_deg": float(rot_sd), "centre_sd_m": float(np.sqrt(max(np.trace(centre_cov), 0))),
                      "alpha_sd": float(np.sqrt(max(C[6, 6], 0))), "tau_sd_s": float(np.sqrt(max(C[7, 7], 0)))}
        return res


def initial_from_reference(cams, rig: State, ref_T: dict) -> State:
    """Rigidly place the relative rig ``rig`` (any gauge) onto reference poses ``ref_T`` (e.g. the median of the
    current per-episode calibration over the FIT episodes): least-squares rotation over the cameras' orientations and
    centres (Kabsch on centres + chordal rotation mean), scale kept."""
    Cr = np.stack([camera_centre(rig.T[c]) for c in cams])
    Cf = np.stack([camera_centre(ref_T[c]) for c in cams])
    # rotation: chordal mean of R_ref^T R_rig (base-frame rotation Q with T_ref ~ T_rig Q^-1)
    M = sum(ref_T[c][:3, :3].T @ rig.T[c][:3, :3] for c in cams)
    U, _, Vt = np.linalg.svd(M)
    Q = U @ np.diag([1, 1, np.sign(np.linalg.det(U @ Vt))]) @ Vt
    tq = (Cf - Cr @ Q.T).mean(0)
    G = np.eye(4)
    G[:3, :3], G[:3, 3] = Q, tq
    out = rig.copy()
    for c in cams:
        out.T[c] = rig.T[c] @ np.linalg.inv(G)
    out.P = None
    return out


def median_pose(Ts: list) -> np.ndarray:
    """Robust central pose: chordal L1-ish (iteratively reweighted) rotation mean and coordinate-wise median centre."""
    Rs = np.stack([T[:3, :3] for T in Ts])
    Cs = np.stack([camera_centre(T) for T in Ts])
    R = Rs.mean(0)
    for _ in range(20):
        U, _, Vt = np.linalg.svd(R)
        R = U @ np.diag([1, 1, np.sign(np.linalg.det(U @ Vt))]) @ Vt
        d = np.linalg.norm(Rs - R, axis=(1, 2)) + 1e-9
        R = (Rs / d[:, None, None]).sum(0)
    U, _, Vt = np.linalg.svd(R)
    R = U @ np.diag([1, 1, np.sign(np.linalg.det(U @ Vt))]) @ Vt
    C = np.median(Cs, 0)
    T = np.eye(4)
    T[:3, :3], T[:3, 3] = R, -R @ C
    return T
