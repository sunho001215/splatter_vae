"""Driver of the board-free per-task rig calibration (``rig_dw``): one solve variant on one task + task-internal
held-out validation. Inputs are the RoMa match files (``workers/romav2_match_worker.py``), the robot evidence files
(``rig_dw_data``) and the current per-episode calibration (only as an INITIALISATION and for comparison). Board data
are never read here.

    python -m s4d.realworld.rig_dw_run --task drawer-open --matches DIR --evidence DIR --cal1 cal1_poses.json \
        --out OUT.json [--variant full] [--frames-per-camera 6] [--seed 0] ...

Variants (``--variant``): ``full`` (all fit episodes), ``half:<seed>`` (random half of the fit episodes),
``group:<g>`` (one of two workspace groups: 2-means of the episodes' mean flange position), ``block:<b>`` (first /
second half of the fit episodes in recording order), ``loco:<camera>`` (robot factors without that camera),
``start:<k>`` (common-pose perturbation k of the multistart), ``mask:<erode|dilate>`` (robot masks eroded/dilated
2 px), ``depth:sensor`` (sensor depth instead of FS), ``all`` (fit on all 100 episodes; no held-out),
``matchseed:<s>`` (match subsampling seed), ``extonly`` / ``robotonly`` (ablations), ``alpha0`` (depth scale fixed:
FS metric trusted), ``kinds:<digits>`` (robot factor kinds: 0 m2o, 1 o2m, 2 depth), ``spacing:<m>`` (visual-mesh sample
spacing), ``notau`` (no time offsets), ``focal`` (diagnostic per-camera focal multiplier)."""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np

from s4d.realworld import rig_dw_extext as ee
from s4d.realworld.rig_dw_frames import load_frames, Surface
from s4d.realworld.rig_dw_geom import camera_centre, pose_distance, se3_exp
from s4d.realworld.rig_dw_matches import MatchConfig, load_matches
from s4d.realworld.rig_dw_robot import RobotTermConfig, pack, packed_residuals, render
from s4d.realworld.rig_dw_solve import RigProblem, SolveConfig, State, median_pose

CAMS = ("135122070988/color", "135122073201/color", "135122076054/color")
URDF = "/home/ws/ws/validation-20261031/panda_assets/panda_robotiq_2f85_droid_iso50.urdf"


def episode_split(episodes: list[str], holdout_mod: int = 5, holdout_rem: int = 2):
    eps = sorted(episodes, key=lambda e: int(e.rsplit("-", 1)[1]))
    held = [e for k, e in enumerate(eps) if k % holdout_mod == holdout_rem]
    fit = [e for e in eps if e not in held]
    return fit, held


def workspace_groups(evidence_dir: Path, eps: list[str], seed: int = 0) -> dict:
    """2-means of the episodes' mean flange (panda_link8) position over their evidence frames."""
    from s4d.realworld.rig_dw_data import load_episode

    pts = []
    for e in eps:
        doc, a = load_episode(evidence_dir / f"{e}.npz")
        li = doc["links"].index("panda_link7") if "panda_link7" in doc["links"] else 0
        p = [a[f"{c['key']}/fk"][:, li, :3, 3] for c in doc["cameras"].values() if c["frames"]]
        pts.append(np.concatenate(p).mean(0))
    X = np.asarray(pts)
    rng = np.random.default_rng(seed)
    cen = X[rng.choice(len(X), 2, replace=False)]
    for _ in range(50):
        lab = np.argmin(((X[:, None] - cen[None]) ** 2).sum(-1), 1)
        cen = np.stack([X[lab == k].mean(0) if (lab == k).any() else cen[k] for k in range(2)])
    return {e: int(l) for e, l in zip(eps, lab)}


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
    from s4d.realworld.rig_dw_geom import so3_exp

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


def explained_and_depth(frames, surf, state: State, cfg: RobotTermConfig, use_alpha: bool):
    """Held-out robot statistics per camera on full frames with FIXED (pose-independent) denominators:
    * along-ray depth |s d_obs - z_model| at observed robot pixels (valid depth) covered by the rendered model
      (median / p90, mm) and the coverage of those pixels by the render;
    * explained fraction: observed robot pixels with valid depth whose lifted point lies within 2 cm of the posed
      visible model surface (pixels with no rendered model count as unexplained);
    * o2m_fixed: observed robot-mask boundary pixels (pose-independent exclusions: image border, object/fixture
      adjacency, observed occluding edges) -> distance (px) to the rendered silhouette boundary;
    * silhouette IoU. Robot geometry is posed at t + tau."""
    from scipy import ndimage
    from scipy.spatial import cKDTree

    from s4d.realworld.rig_dw_robot import boundary

    out = {}
    for fr in frames:
        T = state.T[fr.cam]
        X, _, Vf = surf.posed(fr)
        tau = state.tau[fr.cam]
        Kf = fr.K.copy()
        Kf[0, 0] *= np.exp(state.focal[fr.cam])
        Kf[1, 1] *= np.exp(state.focal[fr.cam])
        if abs(tau) > 1e-4:
            X = X + tau * Vf(np.arange(len(X)))
        Pc = X @ T[:3, :3].T + T[:3, 3]
        rend = render(Pc, Kf, fr.crop, cfg.closing)
        names = fr.class_names
        robot = fr.sem == names.index("robot")
        other = np.isin(fr.sem, [names.index(c) for c in ("object", "fixture") if c in names])
        d = fr.depth * (np.exp(state.alpha[fr.cam]) if use_alpha else 1.0)
        ok = robot & (d > 0)
        both = ok & rend.mask
        res = np.abs(d[both] - rend.zbuf[both])
        y0, y1, x0, x1 = fr.crop
        ii, jj = np.nonzero(ok)
        u = np.stack([jj + x0 + 0.5, ii + y0 + 0.5], 1)
        K = Kf
        P = np.stack([(u[:, 0] - K[0, 2]) / K[0, 0], (u[:, 1] - K[1, 2]) / K[1, 1], np.ones(len(u))], 1) * d[ok][:, None]
        front = np.unique(rend.front[rend.front >= 0])
        if len(P) and len(front):
            dist = cKDTree(Pc[front]).query(P, k=1, workers=1)[0]
            expl = float((dist <= 0.02).mean())
        else:
            expl = 0.0 if len(P) else None
        # pose-independent observed boundary set
        h, w = robot.shape
        gi, gj = np.mgrid[0:h, 0:w]
        gi, gj = gi + y0, gj + x0
        H, W = fr.hw
        border = (gi < cfg.border_px) | (gj < cfg.border_px) | (gi >= H - cfg.border_px) | (gj >= W - cfg.border_px)
        near_other = ndimage.binary_dilation(other, iterations=cfg.adjacent_px) if other.any() else np.zeros_like(other)
        rob_d = np.where(robot & (fr.depth > 0), fr.depth, 0.0)
        near_rob_d = ndimage.maximum_filter(rob_d, size=7)
        front_occ = (~robot) & (fr.depth > 0) & (near_rob_d > 0) & (fr.depth < near_rob_d - cfg.occluder_m)
        bo = boundary(robot) & ~border & ~near_other & ~ndimage.binary_dilation(front_occ, iterations=2)
        # continuous: observed boundary point on the zero level of the observed SDF -> |rendered-mask SDF| there
        from s4d.realworld.rig_dw_robot import bilinear, outward_normals, sdf_of_mask, zero_crossing

        sdf_o = sdf_of_mask(robot).astype(np.float64)
        onx, ony = outward_normals(robot)
        oi, oj = np.nonzero(bo)
        ob = zero_crossing(sdf_o, np.stack([oj + 0.5, oi + 0.5], 1), np.stack([onx[oi, oj], ony[oi, oj]], 1))
        if rend.mask.any():
            val, _, _ = bilinear(sdf_of_mask(rend.mask).astype(np.float64), ob, fill=np.nan)
            o2m = np.where(np.isfinite(val), np.abs(val), 1e3)
        else:
            o2m = np.full(len(oi), 1e3)
        st = out.setdefault(fr.cam, {"depth_res": [], "explained": [], "iou": [], "coverage": [], "o2m": []})
        st["depth_res"].append(res)
        if expl is not None:
            st["explained"].append(expl)
        st["coverage"].append(float(both.sum() / max(1, ok.sum())))
        st["o2m"].append(o2m)
        st["iou"].append(float((rend.mask & robot).sum() / max(1, (rend.mask | robot).sum())))
    res = {}
    for cam, st in out.items():
        allres = np.concatenate(st["depth_res"]) if st["depth_res"] else np.zeros(0)
        o2m = np.concatenate(st["o2m"]) if st["o2m"] else np.zeros(0)
        res[cam] = {"depth_median_mm": float(1e3 * np.median(allres)) if len(allres) else None,
                    "depth_p90_mm": float(1e3 * np.percentile(allres, 90)) if len(allres) else None,
                    "depth_coverage": float(np.median(st["coverage"])),
                    "explained_fraction": float(np.median(st["explained"])) if st["explained"] else None,
                    "explained_fraction_mean": float(np.mean(st["explained"])) if st["explained"] else None,
                    "o2m_fixed_median_px": float(np.median(o2m)) if len(o2m) else None,
                    "o2m_fixed_le2px": float((o2m <= 2.0).mean()) if len(o2m) else None,
                    "iou_median": float(np.median(st["iou"])), "frames": len(st["iou"])}
    return res


def heldout_robot(prob: RigProblem, frames, state: State, surf, cfg: RobotTermConfig) -> dict:
    """Silhouette / point-to-plane statistics at the given state on frames NOT used for fitting."""
    saved = prob.frames, prob.support, prob.support_stats, prob._pack_cache
    prob.frames = frames
    prob.select(state)
    pk = pack([f for f in prob.support if f is not None])  # every camera (also the one a loco fit withheld)
    stats = {}
    for cam, p in pk.items():
        r, _ = packed_residuals(p, state.T[cam], prob.Keff(state)[cam], state.alpha[cam], state.tau[cam], cfg,
                                jac=False, focal=state.focal[cam])
        sil0, sil1 = np.abs(r[p.kind == 0]) * cfg.sigma_sil_px, np.abs(r[p.kind == 1]) * cfg.sigma_sil_px
        s = p.kind == 2
        R = state.T[cam][:3, :3]
        nc = p.nrm[s] @ R.T
        Xo = np.exp(state.alpha[cam]) * p.obs_ray[s] * np.array([np.exp(-state.focal[cam])] * 2 + [1.0])
        Pc = (p.X[s] + state.tau[cam] * p.V[s]) @ R.T + state.T[cam][:3, 3]
        ptp = 1e3 * np.abs(np.einsum("mi,mi->m", nc, Xo - Pc))
        signed_m2o = np.clip(r[p.kind == 0] * cfg.sigma_sil_px, -20, 20)
        signed_o2m = r[p.kind == 1] * cfg.sigma_sil_px
        stats[cam] = {"m2o_median_px": float(np.median(sil0)), "o2m_median_px": float(np.median(sil1)),
                      "contour_median_px": float(np.median(np.concatenate([sil0, sil1]))),
                      "m2o_signed_mean_px": float(np.mean(signed_m2o)),
                      "m2o_signed_median_px": float(np.median(signed_m2o)),
                      "o2m_signed_median_px": float(np.median(signed_o2m)),
                      "ptp_median_mm": float(np.median(ptp)) if len(ptp) else None,
                      "n": {"m2o": int(len(sil0)), "o2m": int(len(sil1)), "ptp": int(len(ptp))}}
    stats["robust_cost_per_residual"] = None
    prob.frames, prob.support, prob.support_stats, prob._pack_cache = saved
    return stats


def common_challenge(prob: RigProblem, frames, state: State, cfg: RobotTermConfig, rot_deg: float = 0.35,
                     trans_m: float = 0.008) -> dict:
    """G6: do the held-out robot data reject a common rig perturbation of the tolerance size? Support fixed at the
    solution; per held-out frame robust cost at the solution vs each of 12 perturbations (+-rot about the base axes
    through the workspace pivot, +-trans along them); one-sided sign test over frames."""
    from scipy.stats import binomtest

    from s4d.realworld.rig_dw_geom import so3_exp

    saved = prob.frames, prob.support, prob.support_stats, prob._pack_cache
    prob.frames = frames
    prob.select(state)
    factors = [f for f in prob.support if f is not None]
    pk = pack(factors)
    prob.frames, prob.support, prob.support_stats, prob._pack_cache = saved
    c = 3.0

    episodes = sorted({f.episode for f in factors})
    ep_of = np.concatenate([[episodes.index(e) for e, _ in pk[k].frames] for k in sorted(pk)])

    def per_frame(st: State):
        out = {}
        for cam, p in pk.items():
            r, _ = packed_residuals(p, st.T[cam], prob.Keff(st)[cam], st.alpha[cam], st.tau[cam], cfg, jac=False,
                                    focal=st.focal[cam])
            rho = c ** 2 * np.log1p((r / c) ** 2)
            out[cam] = np.bincount(p.frame, weights=rho, minlength=len(p.frames))
        return np.concatenate([out[k] for k in sorted(out)])

    base = per_frame(state)
    res = {}
    for kind in ("rot", "trans"):
        for ax in range(3):
            for sgn in (1, -1):
                G = np.eye(4)
                if kind == "rot":
                    e = np.zeros(3)
                    e[ax] = sgn * np.radians(rot_deg)
                    Q = so3_exp(e)
                    G[:3, :3] = Q
                    G[:3, 3] = prob.pivot - Q @ prob.pivot
                else:
                    G[ax, 3] = sgn * trans_m
                st = state.copy()
                for cam in st.T:
                    st.T[cam] = state.T[cam] @ np.linalg.inv(G)
                d = per_frame(st) - base
                k = int((d > 0).sum())
                pval = float(binomtest(k, len(d), 0.5, alternative="greater").pvalue)
                de = np.bincount(ep_of, weights=d, minlength=len(episodes))  # episode clusters
                ke = int((de > 0).sum())
                pe = float(binomtest(ke, len(de), 0.5, alternative="greater").pvalue)
                res[f"{kind}{'xyz'[ax]}{'+' if sgn > 0 else '-'}"] = {
                    "frames_worse": k, "frames": int(len(d)), "fraction_worse": float(k / max(1, len(d))),
                    "mean_cost_increase": float(d.mean()), "p_value": pval,
                    "episodes_worse": ke, "episodes": int(len(de)), "p_value_episode": pe,
                    "rejected": bool(pval < 0.01 and k > len(d) / 2 and pe < 0.01)}
    res["all_rejected"] = all(v["rejected"] for v in res.values() if isinstance(v, dict))
    return res


def fs_over_sensor(mdir: Path, episodes: list) -> dict:
    """Median FS / sensor depth ratio per camera over the static MAGSAC-inlier matches of ``episodes``."""
    acc = {c: [] for c in CAMS}
    for e in episodes:
        with np.load(mdir / f"{e}.npz") as z:
            mc = json.loads(str(z["meta"]))["cameras"]
            ok = z["magsac_inlier"].astype(bool) & (z["dyn_dist_a"] > 9) & (z["dyn_dist_b"] > 9)
            for end in "ab":
                for p in np.unique(z["pair"]):
                    pa, pb = divmod(int(p), 10)
                    cam = mc[pa if end == "a" else pb]
                    s = ok & (z["pair"] == p) & (z["fs_" + end] > 0) & (z["sd_" + end] > 0) \
                        & (z["fs_range_" + end] < 0.02 * z["fs_" + end])
                    acc[cam].append(z["fs_" + end][s] / z["sd_" + end][s])
    return {c: float(np.median(np.concatenate(v))) for c, v in acc.items()}


def per_episode_extext(state: State, K, mdir: Path, episodes: list, per_frame_pair: int) -> dict:
    """Fixedness / consistency of every episode with the shared rig: per episode and camera pair the median symmetric
    epipolar line distance (px) of its static matches (protected subsampling seed 1). A camera that moved in an
    episode raises both pairs containing it (0.1 deg ~ 1 px)."""
    from s4d.realworld.rig_dw_geom import skew

    out = {}
    for e in episodes:
        m = load_matches([mdir / f"{e}.npz"], list(CAMS), MatchConfig(per_frame_pair=per_frame_pair, seed=1))
        if not len(m["cam_a"]):
            continue
        M = matchset(m, 0.5, 0.01)
        res = {}
        for a, b, ids in M._groups:
            R = state.T[b][:3, :3] @ state.T[a][:3, :3].T
            t = state.T[b][:3, 3] - R @ state.T[a][:3, 3]
            F = np.linalg.inv(K[b]).T @ skew(t) @ R @ np.linalg.inv(K[a])
            xa = np.column_stack((M.uv_a[ids], np.ones(len(ids))))
            xb = np.column_stack((M.uv_b[ids], np.ones(len(ids))))
            la, lb = xb @ F, xa @ F.T
            num = np.abs(np.sum(xb * lb, axis=1))
            line = 0.5 * (num / np.hypot(lb[:, 0], lb[:, 1]) + num / np.hypot(la[:, 0], la[:, 1]))
            med = float(np.median(line))
            se = float(1.2533 * 1.4826 * np.median(np.abs(line - med)) / np.sqrt(max(1, len(line))))
            res[f"{a[:12]}-{b[:12]}"] = {"median_px": med, "se_px": se, "n": int(len(line))}
        out[e] = res
    return out


def heldout_extext(state: State, K, Mh: ee.MatchSet) -> dict:
    """Held-out static correspondences: symmetric point-to-epipolar-line distance (px, depth-free; mean of the
    distances in both images), Sampson distance, depth-lifted transfer a->b and b->a (px, alpha-corrected depths) and
    3D consistency of both lifted ends (mm)."""
    from s4d.realworld.rig_dw_geom import project, skew

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


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True)
    ap.add_argument("--matches", required=True)
    ap.add_argument("--evidence", required=True)
    ap.add_argument("--cal1", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--variant", default="full")
    ap.add_argument("--frames-per-camera", type=int, default=6)
    ap.add_argument("--heldout-frames-per-camera", type=int, default=12)
    ap.add_argument("--per-frame-pair", type=int, default=200)
    ap.add_argument("--processes", type=int, default=8)
    ap.add_argument("--sigma-px", type=float, default=0.5)
    ap.add_argument("--sigma-logd", type=float, default=0.01)
    ap.add_argument("--spacing-m", type=float, default=0.0015)
    ap.add_argument("--no-heldout", action="store_true")
    ap.add_argument("--challenge", action="store_true", help="run the G6 common-transform challenge for any variant")
    ap.add_argument("--ref-result", default=None, help="result JSON whose state defines a COMMON support on which this "
                    "solution's and the reference's fit costs are compared (multistart near-best rule)")
    ap.add_argument("--init", default=None, help="JSON with a State (T_cam_base, alpha, tau_s) to start from")
    a = ap.parse_args(argv)
    t0 = time.time()
    mdir, edir = Path(a.matches), Path(a.evidence)
    all_eps = sorted(p.stem for p in edir.glob(f"{a.task}-*.npz"))
    fit, held = episode_split(all_eps)
    v, _, arg = a.variant.partition(":")
    mcfg = MatchConfig(per_frame_pair=a.per_frame_pair)
    rcfg = RobotTermConfig()
    scfg = SolveConfig(processes=a.processes)
    depth_stage = "foundation_stereo_depth"
    fit_eps = list(fit)
    start = 0
    if v == "half":
        rng = np.random.default_rng(int(arg))
        fit_eps = sorted(rng.choice(fit, len(fit) // 2, replace=False).tolist())
    elif v == "group":
        groups = workspace_groups(edir, fit)
        fit_eps = [e for e in fit if groups[e] == int(arg)]
    elif v == "block":
        fit_eps = fit[: len(fit) // 2] if int(arg) == 0 else fit[len(fit) // 2:]
    elif v == "loco":
        scfg = replace(scfg, robot_cameras=tuple(c for c in CAMS if not c.startswith(arg)))
    elif v == "start":
        start = int(arg)
    elif v == "depth":
        mcfg = replace(mcfg, depth_source="sensor")
        depth_stage = "sensor_depth"
    elif v == "all":
        fit_eps, held = sorted(all_eps), []
    elif v == "matchseed":
        mcfg = replace(mcfg, seed=int(arg))
    elif v == "extonly":
        scfg = replace(scfg, use_robot=False)
    elif v == "alpha0":
        scfg = replace(scfg, fix_alpha=True)
    elif v == "kinds":
        scfg = replace(scfg, robot_kinds=tuple(int(ch) for ch in arg))
    elif v == "spacing":
        a.spacing_m = float(arg)
    elif v == "notau":
        scfg = replace(scfg, estimate_tau=False)
    elif v == "focal":
        scfg = replace(scfg, estimate_focal=True)
    elif v == "robotonly":
        scfg = replace(scfg, use_extext=False)
    elif v != "full" and v != "mask":
        raise SystemExit(f"unknown variant {a.variant}")

    m = load_matches([mdir / f"{e}.npz" for e in fit_eps], list(CAMS), mcfg)
    M = matchset(m, a.sigma_px, a.sigma_logd)
    frames = load_frames([edir / f"{e}.npz" for e in fit_eps], list(CAMS), depth_stage=depth_stage,
                         max_frames_per_camera=a.frames_per_camera)
    if v == "mask":
        from scipy import ndimage

        for fr in frames:
            rid = fr.class_names.index("robot")
            rob = fr.sem == rid
            new = ndimage.binary_erosion(rob, iterations=2) if arg == "erode" else ndimage.binary_dilation(rob, iterations=2)
            sem = fr.sem.copy()
            sem[rob & ~new] = 0
            sem[new & ~rob] = rid
            fr.sem = sem
    K = {}
    for fr in frames:
        K[fr.cam] = fr.K
    from s4d.realworld.robot import URDFModel

    model = URDFModel.from_file(URDF)
    surf = Surface(model, a.spacing_m)
    pivot = np.mean([fr.fk[fr.links.index("panda_link7")][:3, 3] for fr in frames], 0)
    cal = json.loads(Path(a.cal1).read_text())
    T0 = {c: median_pose([np.asarray(cal[e][c.split("/")[0]]["T"]) for e in fit_eps if e in cal]) for c in CAMS}
    s0 = State(T0, {c: 0.0 for c in CAMS}, {c: 0.0 for c in CAMS})
    if a.init:
        d = json.loads(Path(a.init).read_text())["state"]
        s0 = State({c: np.asarray(d["T_cam_base"][c]) for c in CAMS}, dict(d["alpha"]), dict(d["tau_s"]))
    if scfg.fix_alpha:
        s0.alpha = {c: 0.0 for c in CAMS}
    if not scfg.estimate_tau:
        s0.tau = {c: 0.0 for c in CAMS}
    prob = RigProblem(CAMS, K, M, frames, surf, rcfg, scfg, pivot=pivot)
    stages = ("shape", "common", "full")
    if v == "extonly":
        stages = ("shape",)
        prob.select(s0)  # robot support (diagnostics only; not used by the ext-ext solve)
    if v == "robotonly":
        stages = ("full",)
    log = []
    if start:
        # the relative rig is solved first; the multistart perturbs the COMMON pose of the whole rig
        s_shape = prob.solve(s0, stages=("shape",), log=log)
        s0 = perturb_common(s_shape, start, pivot=pivot)
        stages = ("common", "full")
    s = prob.solve(s0, stages=stages, log=log)
    s_fit = s
    result = {"task": a.task, "variant": a.variant, "fit_episodes": fit_eps, "heldout_episodes": held,
              "state": s.to_json(), "init_state": s0.to_json(), "log": log,
              "config": {"match": asdict(mcfg), "robot": asdict(rcfg), "solve": asdict(scfg),
                         "frames_per_camera": a.frames_per_camera, "sigma_px": a.sigma_px,
                         "sigma_logd": a.sigma_logd, "spacing_m": a.spacing_m, "depth_stage": depth_stage},
              "match_stats": m["stats"], "n_frames": len(frames), "pivot": pivot.tolist(),
              "fit_frames": {e: {c: [f.t_index for f in frames if f.episode == e and f.cam == c] for c in CAMS}
                             for e in fit_eps},
              "surface": surf.vs.provenance()}
    # fit statistics (fixed support at the solution)
    if a.ref_result:
        d = json.loads(Path(a.ref_result).read_text())["state"]
        sref = State({c: np.asarray(d["T_cam_base"][c]) for c in CAMS}, dict(d["alpha"]), dict(d["tau_s"]),
                     focal=dict(d.get("focal_log") or {c: 0.0 for c in CAMS}))
        sref.P = prob._ee_points(sref, iters=5)
        own_support = prob.support, prob.support_stats, prob._pack_cache
        prob.select(sref)
        s_eval = s.copy()
        s_eval.P = prob._ee_points(s_eval, iters=5)
        result["cost_on_ref_support"] = {"solution": prob.cost(s_eval), "reference": prob.cost(sref)}
        prob.support, prob.support_stats, prob._pack_cache = own_support
    pk = prob._packed() if prob.support is not None else {}
    fit_stats = {}
    for cam, p in pk.items():
        r, _ = packed_residuals(p, s.T[cam], prob.Keff(s)[cam], s.alpha[cam], s.tau[cam], rcfg, jac=False,
                                focal=s.focal[cam])
        fit_stats[cam] = {f"k{k}_median_whitened": float(np.median(np.abs(r[p.kind == k]))) for k in (0, 1, 2)}
    result["fit_robot"] = fit_stats
    result["fit_extext"] = heldout_extext(s, prob.Keff(s), M)
    info = prob.information(s)
    info.pop("cov", None)
    result["information"] = info
    result["moves_vs_init"] = {c: list(pose_distance(s0.T[c], s.T[c])) for c in CAMS}
    if held and not a.no_heldout:
        s_fit = s
        if v == "depth":  # evaluate on the protected FS evidence with the FS-equivalent depth scale
            r_fs = fs_over_sensor(mdir, fit_eps)
            s = s.copy()
            s.alpha = {c: s.alpha[c] - float(np.log(r_fs[c])) for c in CAMS}
            result["fs_over_sensor_fit"] = r_fs
        # protected evaluation configuration, identical for every variant: FS depth, match subsampling seed 1
        mh = load_matches([mdir / f"{e}.npz" for e in held], list(CAMS), MatchConfig(per_frame_pair=a.per_frame_pair,
                                                                                      seed=1))
        Mh = matchset(mh, a.sigma_px, a.sigma_logd)
        hframes = load_frames([edir / f"{e}.npz" for e in held], list(CAMS), depth_stage="foundation_stereo_depth",
                              max_frames_per_camera=a.heldout_frames_per_camera)
        result["heldout"] = {"extext": heldout_extext(s, prob.Keff(s), Mh),
                             "robot": heldout_robot(prob, hframes, s, surf, rcfg),
                             "robot_raw_depth": explained_and_depth(hframes, surf, s, rcfg, use_alpha=False),
                             "robot_alpha_depth": explained_and_depth(hframes, surf, s, rcfg, use_alpha=True)}
        if v == "full" or a.challenge:
            result["heldout"]["common_challenge"] = common_challenge(prob, hframes, s, rcfg)
        # the current calibration (per-episode calibrate@1 poses) on the same held-out evidence, for comparison
        if v == "full":
            cmp = {"cal1_median_fit": State(T0, {c: 0.0 for c in CAMS}, {c: 0.0 for c in CAMS})}
            result["heldout_cal1_median"] = {
                "extext": heldout_extext(cmp["cal1_median_fit"], prob.K, Mh),
                "robot": heldout_robot(prob, hframes, cmp["cal1_median_fit"], surf, rcfg),
                "robot_raw_depth": explained_and_depth(hframes, surf, cmp["cal1_median_fit"], rcfg, use_alpha=False)}
            per_ep = {}
            for e in held:
                if e not in cal:
                    continue
                se = State({c: np.asarray(cal[e][c.split("/")[0]]["T"]) for c in CAMS}, {c: 0.0 for c in CAMS},
                           {c: 0.0 for c in CAMS})
                fe = [f for f in hframes if f.episode == e]
                me = mh["episode"] == mh["episodes"].index(e) if e in mh["episodes"] else None
                per_ep[e] = {"robot": heldout_robot(prob, fe, se, surf, rcfg),
                             "robot_raw_depth": explained_and_depth(fe, surf, se, rcfg, use_alpha=False)}
                if me is not None and me.any():
                    from s4d.realworld.rig_dw_matches import subset

                    per_ep[e]["extext"] = heldout_extext(se, prob.K, matchset(subset(mh, me), a.sigma_px,
                                                                               a.sigma_logd))["all"]
            result["heldout_cal1_per_episode"] = per_ep
    if v == "depth":
        s = s_fit
    if v in ("full", "all"):
        result["per_episode_extext"] = per_episode_extext(s, prob.Keff(s), mdir, all_eps, a.per_frame_pair)
    result["seconds"] = round(time.time() - t0, 1)
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(".tmp")
    tmp.write_text(json.dumps(result, indent=1, default=lambda x: x.tolist() if hasattr(x, "tolist") else str(x)))
    tmp.replace(out)
    print(json.dumps({"out": str(out), "seconds": result["seconds"]}))


if __name__ == "__main__":
    main()
