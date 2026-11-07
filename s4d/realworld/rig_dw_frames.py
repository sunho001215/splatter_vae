"""Robot frames of ``rig_dw``: evidence records (``rig_dw_data`` output) -> per (episode, camera, frame) inputs of the
robot factors (posed visual-mesh samples, velocities), parallel support selection (``rig_dw_robot.frame_factors``)
and the robot part of the normal equations."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from s4d.realworld import rig_dw_robot as rr


@dataclass
class RobotFrame:
    episode: str
    cam: str
    t_index: int
    K: np.ndarray
    hw: tuple
    class_names: list
    crop: np.ndarray
    sem: np.ndarray
    depth: np.ndarray  # metres, 0 invalid (crop)
    fk: np.ndarray  # (L,4,4)
    fk_minus: np.ndarray
    fk_plus: np.ndarray
    dt_s: float
    links: list
    meta: dict = field(default_factory=dict)

    def ev(self) -> dict:
        return {"crop": self.crop, "sem": self.sem, "class_names": self.class_names, "depth": self.depth, "hw": self.hw}


def load_frames(paths: list[Path], cameras: list[str], depth_stage: str = "foundation_stereo_depth",
                max_frames_per_camera: int | None = None) -> list[RobotFrame]:
    from s4d.realworld.rig_dw_data import load_episode

    out = []
    for p in paths:
        doc, a = load_episode(p)
        dt = doc.get("velocity_dt_ns", 10_000_000) * 1e-9
        for cam in cameras:
            c = doc["cameras"].get(cam)
            if c is None or not c["frames"]:
                continue
            key = c["key"]
            n = c["frames"]
            ks = range(n) if max_frames_per_camera is None or n <= max_frames_per_camera else \
                np.linspace(0, n - 1, max_frames_per_camera).round().astype(int)
            for k in ks:
                out.append(RobotFrame(
                    doc["episode_id"], cam, int(a[f"{key}/t_index"][k]), np.asarray(c["K"], float), tuple(c["hw"]),
                    c["class_names"], a[f"{key}/crop"][k], a[f"{key}/f{k}/sem"],
                    a[f"{key}/f{k}/{depth_stage}"].astype(np.float64) * 1e-4, a[f"{key}/fk"][k],
                    a[f"{key}/fk_minus"][k] if f"{key}/fk_minus" in a else a[f"{key}/fk"][k],
                    a[f"{key}/fk_plus"][k] if f"{key}/fk_plus" in a else a[f"{key}/fk"][k],
                    2 * dt, doc["links"], {"speed_flange": float(a[f"{key}/speed"][k])}))
    return out


class Surface:
    """Dense visual-mesh samples (link frames, float32) used for rendering + normals; posed per frame. ``posed``
    returns the points and LAZY normal/velocity providers (only the selected samples are transformed)."""

    def __init__(self, model, spacing_m: float, seed: int = 0):
        from s4d.realworld.robot_visual import VisualSurface

        self.vs = VisualSurface.build(model, spacing_m=spacing_m, seed=seed)
        self.links = sorted(self.vs.link_points)
        self.pts = np.concatenate([self.vs.link_points[l] for l in self.links]).astype(np.float32)
        self.nrm = np.concatenate([self.vs.link_normals[l] for l in self.links]).astype(np.float32)
        self.lid = np.concatenate([np.full(len(self.vs.link_points[l]), i, np.int16)
                                   for i, l in enumerate(self.links)])
        self.bounds = np.cumsum([0] + [len(self.vs.link_points[l]) for l in self.links])

    def _mats(self, fr, fk):
        li = {l: i for i, l in enumerate(fr.links)}
        return np.stack([fk[li[l]] for l in self.links])

    def posed(self, fr: RobotFrame):
        Ts = self._mats(fr, fr.fk)
        X = np.empty((len(self.pts), 3), np.float64)
        for i in range(len(self.links)):
            a, b = self.bounds[i], self.bounds[i + 1]
            X[a:b] = self.pts[a:b] @ Ts[i, :3, :3].T.astype(np.float32) + Ts[i, :3, 3]
        Tm, Tp = self._mats(fr, fr.fk_minus), self._mats(fr, fr.fk_plus)

        def normals(ids):
            R = Ts[self.lid[ids], :3, :3]
            return np.einsum("nij,nj->ni", R, self.nrm[ids].astype(np.float64))

        def velocity(ids):
            p = self.pts[ids].astype(np.float64)
            l = self.lid[ids]
            xp = np.einsum("nij,nj->ni", Tp[l, :3, :3], p) + Tp[l, :3, 3]
            xm = np.einsum("nij,nj->ni", Tm[l, :3, :3], p) + Tm[l, :3, 3]
            return (xp - xm) / fr.dt_s

        return X, normals, velocity


_POOL = {}


def _select_one(k):
    frames, surf, state, cfg = _POOL["frames"], _POOL["surf"], _POOL["state"], _POOL["cfg"]
    fr = frames[k]
    T, alpha, tau = state["T"][fr.cam], state["alpha"][fr.cam], state["tau"][fr.cam]
    X, N, V = surf.posed(fr)
    rng = np.random.default_rng(cfg.seed + 7919 * k)
    ff, st = rr.frame_factors(fr.cam, X, N, V, T, fr.K, fr.ev(), cfg, rng, alpha=alpha, tau=tau)
    # visible-surface speed at this frame: max over the model points that carry factors
    if ff is not None:
        sp = np.linalg.norm(ff.V, axis=1)
        st["max_surface_speed"] = float(sp.max()) if len(sp) else 0.0
        ff.episode, ff.t_index = fr.episode, fr.t_index
        ff.sdf = ff.sdf.astype(np.float32)
    return ff, st


def select_support(frames, surf, state, cfg, processes: int | None = None):
    """Frame factors of every frame at ``state`` (dict T/alpha/tau per camera); forked worker pool."""
    import multiprocessing as mp

    _POOL.update(frames=frames, surf=surf, state=state, cfg=cfg)
    procs = processes or min(8, os.cpu_count() or 1)
    try:
        if procs > 1 and "fork" in mp.get_all_start_methods():
            with mp.get_context("fork").Pool(procs) as pool:
                res = pool.map(_select_one, range(len(frames)), chunksize=4)
        else:
            res = [_select_one(k) for k in range(len(frames))]
    finally:
        _POOL.clear()
    return res


def robot_normal_equations(packed, state, Kmap, cfg, col_of, n, cauchy_c: float = 3.0, kinds=(0, 1, 2),
                           robust: bool = True):
    """Robot normal equations over ``rig_dw_robot.pack`` output (vectorised per camera)."""
    H = np.zeros((n, n))
    g = np.zeros(n)
    cost, nres = 0.0, 0
    for cam, pk in packed.items():
        r, J = rr.packed_residuals(pk, state["T"][cam], Kmap[cam], state["alpha"][cam], state["tau"][cam], cfg)
        use = np.isin(pk.kind, kinds)
        r, J = r[use], J[use]
        w = 1.0 / (1.0 + (r / cauchy_c) ** 2) if robust else np.ones_like(r)
        cost += float(np.sum(cauchy_c ** 2 * np.log1p((r / cauchy_c) ** 2))) if robust else float(np.sum(r ** 2))
        nres += len(r)
        cols = np.asarray(col_of(cam))
        free = cols >= 0
        Jf, cf = J[:, free], cols[free]
        H[np.ix_(cf, cf)] += (Jf * w[:, None]).T @ Jf
        g[cf] += (Jf * w[:, None]).T @ r
    return H, g, cost, nres


def robot_cost_packed(packed, state, Kmap, cfg, cauchy_c: float = 3.0, kinds=(0, 1, 2)) -> float:
    cost = 0.0
    for cam, pk in packed.items():
        r, _ = rr.packed_residuals(pk, state["T"][cam], Kmap[cam], state["alpha"][cam], state["tau"][cam], cfg,
                                   jac=False)
        r = r[np.isin(pk.kind, kinds)]
        cost += float(np.sum(cauchy_c ** 2 * np.log1p((r / cauchy_c) ** 2)))
    return cost


def _robot_normal_equations_unpacked(factors, state, Kmap, cfg, col_of, n, cauchy_c: float = 3.0, kinds=(0, 1, 2),
                                     robust: bool = True):
    """Robot part of the normal equations: H (n,n), g (n,), cost = sum c^2 log(1 + r^2/c^2) (robust) or sum r^2.
    ``col_of(cam)`` -> 8 global column indices [xi(6), alpha, tau] (entries < 0 are fixed)."""
    H = np.zeros((n, n))
    g = np.zeros(n)
    cost = 0.0
    nres = 0
    for ff in factors:
        if ff is None:
            continue
        cam = ff.cam
        r, J = rr.residuals_and_jacobians(ff, state["T"][cam], Kmap[cam], state["alpha"][cam], state["tau"][cam], cfg)
        use = np.isin(ff.kind, kinds)
        r, J = r[use], J[use]
        if robust:
            w = 1.0 / (1.0 + (r / cauchy_c) ** 2)
            cost += float(np.sum(cauchy_c ** 2 * np.log1p((r / cauchy_c) ** 2)))
        else:
            w = np.ones_like(r)
            cost += float(np.sum(r ** 2))
        nres += len(r)
        cols = np.asarray(col_of(cam))
        free = cols >= 0
        Jf = J[:, free]
        cf = cols[free]
        H[np.ix_(cf, cf)] += (Jf * w[:, None]).T @ Jf
        g[cf] += (Jf * w[:, None]).T @ r
    return H, g, cost, nres


def robot_cost(factors, state, Kmap, cfg, cauchy_c: float = 3.0, kinds=(0, 1, 2)) -> float:
    cost = 0.0
    for ff in factors:
        if ff is None:
            continue
        r, _ = rr.residuals_and_jacobians(ff, state["T"][ff.cam], Kmap[ff.cam], state["alpha"][ff.cam],
                                          state["tau"][ff.cam], cfg)
        r = r[np.isin(ff.kind, kinds)]
        cost += float(np.sum(cauchy_c ** 2 * np.log1p((r / cauchy_c) ** 2)))
    return cost


def robot_stats(factors, state, Kmap, cfg) -> dict:
    """Unwhitened residual statistics at fixed support: silhouette distances (px) per direction, point-to-plane depth
    (mm), per camera."""
    out = {}
    for ff in factors:
        if ff is None:
            continue
        r, _ = rr.residuals_and_jacobians(ff, state["T"][ff.cam], Kmap[ff.cam], state["alpha"][ff.cam],
                                          state["tau"][ff.cam], cfg)
        d = out.setdefault(ff.cam, {"m2o": [], "o2m": [], "depth_mm": []})
        d["m2o"].append(np.abs(r[ff.kind == 0]) * cfg.sigma_sil_px)
        d["o2m"].append(np.abs(r[ff.kind == 1]) * cfg.sigma_sil_px)
        s = ff.kind == 2
        if s.any():
            R = state["T"][ff.cam][:3, :3]
            nc = ff.nrm[s] @ R.T
            Xo = np.exp(state["alpha"][ff.cam]) * ff.obs_ray[s]
            Pc = (ff.X[s] + state["tau"][ff.cam] * ff.V[s]) @ R.T + state["T"][ff.cam][:3, 3]
            d["depth_mm"].append(1e3 * np.abs(np.einsum("mi,mi->m", nc, Xo - Pc)))
    res = {}
    for cam, d in out.items():
        res[cam] = {k: (float(np.median(np.concatenate(v))) if v and len(np.concatenate(v)) else None)
                    for k, v in d.items()}
        res[cam].update({f"n_{k}": int(sum(len(x) for x in v)) for k, v in d.items()})
    return res
