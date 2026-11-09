"""Shared camera constants, episode splits and calibration validation helpers for DW and LoMa."""

from __future__ import annotations

import numpy as np

from s4d.realworld.calib.dw import extext as ee
from s4d.realworld.calib.dw.solve import State

CAMS = ("135122070988/color", "135122073201/color", "135122076054/color")
URDF = "/home/ws/ws/validation-20261031/panda_assets/panda_robotiq_2f85_droid_iso50.urdf"


def episode_split(episodes: list[str], holdout_mod: int = 5, holdout_rem: int = 2):
    eps = sorted(episodes, key=lambda e: int(e.rsplit("-", 1)[1]))
    held = [e for k, e in enumerate(eps) if k % holdout_mod == holdout_rem]
    fit = [e for e in eps if e not in held]
    return fit, held


def matchset(m: dict, sigma_px: float, sigma_logd: float) -> ee.MatchSet:
    cams = np.array(CAMS)
    return ee.MatchSet(cams[m["cam_a"].astype(int)], cams[m["cam_b"].astype(int)], m["uv_a"], m["uv_b"],
                       m["d_a"], m["d_b"], sigma_px, sigma_logd)


def perturb_common(s: State, k: int, rot_deg: float = 4.0, trans_m: float = 0.04, pivot=None) -> State:
    """Multistart k: a deterministic common rigid perturbation of the whole rig about the workspace pivot."""
    if k == 0:
        return s.copy()
    rng = np.random.default_rng(1000 + k)
    ax = rng.normal(size=3)
    ax /= np.linalg.norm(ax)
    dt = rng.normal(size=3)
    dt *= trans_m / np.linalg.norm(dt)
    from s4d.realworld.calib.dw.geom import so3_exp

    Q = so3_exp(np.radians(rot_deg) * ax)
    p = np.zeros(3) if pivot is None else pivot
    G = np.eye(4)
    G[:3, :3] = Q
    G[:3, 3] = p - Q @ p + dt
    out = s.copy()
    for c in out.T:
        out.T[c] = s.T[c] @ np.linalg.inv(G)
    out.P = None
    return out


def heldout_extext(state: State, K, Mh: ee.MatchSet) -> dict:
    """Held-out static correspondences: symmetric point-to-epipolar-line distance (px, depth-free; mean of the
    distances in both images), Sampson distance, depth-lifted transfer a->b and b->a (px, alpha-corrected depths) and
    3D consistency of both lifted ends (mm)."""
    from s4d.realworld.calib.dw.geom import project, skew

    st = ee.reduced_residual_stats(state.T, state.alpha, K, Mh)
    out, allv = {}, {"line": [], "tba": [], "d3": []}
    for a, b, ids in Mh._groups:
        R = state.T[b][:3, :3] @ state.T[a][:3, :3].T
        t = state.T[b][:3, 3] - R @ state.T[a][:3, 3]
        F = np.linalg.inv(K[b]).T @ skew(t) @ R @ np.linalg.inv(K[a])
        xa = np.column_stack((Mh.uv_a[ids], np.ones(len(ids))))
        xb = np.column_stack((Mh.uv_b[ids], np.ones(len(ids))))
        la, lb = xb @ F, xa @ F.T  # line in a (F^T x_b), line in b (F x_a)
        num = np.abs(np.sum(xb * lb, axis=1))
        line = 0.5 * (num / np.hypot(lb[:, 0], lb[:, 1]) + num / np.hypot(la[:, 0], la[:, 1]))
        ok = np.isfinite(Mh.d_b[ids]) & (Mh.d_b[ids] > 0)
        Pb = (np.column_stack((Mh.uv_b[ids][ok], np.ones(ok.sum()))) @ np.linalg.inv(K[b]).T) \
            * (np.exp(state.alpha[b]) * Mh.d_b[ids][ok])[:, None]
        uva, _ = project((Pb - t) @ R, K[a])
        tba = np.linalg.norm(uva - Mh.uv_a[ids][ok], axis=1)
        e, tr = st["epipolar_px"][ids], st["transfer_px"][ids]
        key = f"{a[:12]}-{b[:12]}"
        out[key] = {"line_median_px": float(np.nanmedian(line)), "line_p90_px": float(np.nanpercentile(line, 90)),
                    "sampson_median_px": float(np.nanmedian(e)), "transfer_ab_median_px": float(np.nanmedian(tr)),
                    "transfer_ba_median_px": float(np.nanmedian(tba)) if len(tba) else None, "n": int(len(ids))}
        allv["line"].append(line)
        allv["tba"].append(tba)
        both = np.isfinite(Mh.d_a[ids]) & np.isfinite(Mh.d_b[ids]) & (Mh.d_a[ids] > 0) & (Mh.d_b[ids] > 0)
        i = ids[both]
        if len(i):
            def lift(uv, d, Kc, T, al):
                ray = np.column_stack((uv, np.ones(len(uv)))) @ np.linalg.inv(Kc).T
                return (ray * (np.exp(al) * d)[:, None] - T[:3, 3]) @ T[:3, :3]
            Xa = lift(Mh.uv_a[i], Mh.d_a[i], K[a], state.T[a], state.alpha[a])
            Xb = lift(Mh.uv_b[i], Mh.d_b[i], K[b], state.T[b], state.alpha[b])
            d3 = np.linalg.norm(Xa - Xb, axis=1)
            allv["d3"].append(d3)
            out[key]["consistency3d_median_mm"] = float(1e3 * np.median(d3))
    out["all"] = {"line_median_px": float(np.nanmedian(np.concatenate(allv["line"]))),
                  "sampson_median_px": float(np.nanmedian(st["epipolar_px"])),
                  "transfer_ab_median_px": float(np.nanmedian(st["transfer_px"])),
                  "transfer_ba_median_px": float(np.nanmedian(np.concatenate(allv["tba"]))),
                  "consistency3d_median_mm": float(1e3 * np.median(np.concatenate(allv["d3"]))) if allv["d3"] else None}
    return out
