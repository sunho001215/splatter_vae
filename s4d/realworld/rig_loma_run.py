"""Driver of the render-to-image robot 2D-3D rig calibration (``rig_loma``): one solve variant on one task + held-out
validation. No SAM mask enters initialisation, optimisation, correspondence filtering or solution selection; SAM-based
rig_dw checks (silhouette contour, masked robot depth, SAM-static held-out matches, SAM common challenge) are computed
only as independent validation (``heldout_sam``).

    python -m s4d.realworld.rig_loma_run --task drawer-open --corr DIR --matches DIR --flags DIR --init state.json \
        --out OUT.json [--variant full] [--depth-stage foundation_stereo_depth] [--evidence DIR (SAM validation)]

Variants: ``full``, ``half:<seed>``, ``group:<g>`` (2-means of the episodes' mean robot-point position), ``block:<b>``,
``loco:<camera prefix>`` (2D-3D term withheld for that camera), ``start:<k>`` (common perturbation 4 deg / 40 mm of
the rig after the shape stage), ``all`` (100 episodes, no held-out), ``matchseed:<s>``, ``corrseed:<s>``,
``extonly`` / ``corronly`` (ablations), ``alpha0``, ``notau``, ``depthterm`` (secondary match-depth term weight 1),
``ext2d`` (ext-ext without depth rows), ``focal`` (diagnostic focal multiplier). ``--depth-stage`` selects the depth
source by name (FS / sensor columns of the match files, or any other stage through ``--depth-dir``)."""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np

from s4d.realworld import rig_dw_extext as ee
from s4d.realworld.rig_dw_geom import camera_centre, pose_distance, so3_exp
from s4d.realworld.rig_dw_matches import MatchConfig
from s4d.realworld.rig_dw_run import CAMS, URDF, episode_split, heldout_extext, matchset, perturb_common
from s4d.realworld.rig_dw_solve import SolveConfig, State
from s4d.realworld.rig_loma_corr import CorrConfig, load_corr, load_matches_samfree
from s4d.realworld.rig_loma_solve import LomaConfig, LomaProblem, corr_residuals


def state_from_json(path: str) -> State:
    d = json.loads(Path(path).read_text())
    d = d.get("state", d)
    return State({c: np.asarray(d["T_cam_base"][c], float) for c in CAMS},
                 {c: float((d.get("alpha") or {}).get(c, 0.0)) for c in CAMS},
                 {c: float((d.get("tau_s") or {}).get(c, 0.0)) for c in CAMS},
                 focal={c: float((d.get("focal_log") or {}).get(c, 0.0)) for c in CAMS})


def corr_groups(corr: dict, episodes: list, seed: int = 0) -> dict:
    """2-means of the episodes' mean robot-point position (workspace groups, from the correspondences)."""
    pts = {}
    for cs in corr.values():
        for e in np.unique(cs.episode):
            pts.setdefault(int(e), []).append(cs.X[cs.episode == e].mean(0))
    keys = sorted(pts)
    X = np.array([np.mean(pts[k], 0) for k in keys])
    rng = np.random.default_rng(seed)
    cen = X[rng.choice(len(X), 2, replace=False)]
    for _ in range(50):
        lab = np.argmin(((X[:, None] - cen[None]) ** 2).sum(-1), 1)
        cen = np.stack([X[lab == k].mean(0) if (lab == k).any() else cen[k] for k in range(2)])
    return {episodes[k]: int(l) for k, l in zip(keys, lab)}


def corr_challenge(prob: LomaProblem, corr: dict, state: State, rot_deg=0.35, trans_m=0.008) -> dict:
    """SAM-free G6 analogue: do held-out robot 2D-3D correspondences reject a common rig perturbation of the tolerance
    size? Per held-out frame robust cost at the solution vs each of 12 perturbations about the workspace pivot; the
    frame and the episode-cluster one-sided sign tests must both give p < 0.01."""
    from scipy.stats import binomtest

    c = prob.lcfg.cauchy
    Ke = prob.Keff(state)

    def per_frame(st):
        fr, ep, val = [], [], []
        for cam, cs in corr.items():
            r, _ = corr_residuals(st.T[cam], Ke[cam], cs, st.tau[cam], prob.lcfg.sigma_px, jac=False)
            rho = c ** 2 * np.log1p((r ** 2).sum(1) / c ** 2)
            u, inv = np.unique(cs.frame, return_inverse=True)
            val.append(np.bincount(inv, weights=rho))
            fr.append(u)
            ep.append(np.array([cs.episode[cs.frame == f][0] for f in u]))
        return np.concatenate(val), np.concatenate(ep)

    base, ep_of = per_frame(state)
    eps = np.unique(ep_of)
    res = {}
    for kind in ("rot", "trans"):
        for ax in range(3):
            for sgn in (1, -1):
                G = np.eye(4)
                if kind == "rot":
                    e = np.zeros(3)
                    e[ax] = sgn * np.radians(rot_deg)
                    Q = so3_exp(e)
                    G[:3, :3], G[:3, 3] = Q, prob.pivot - Q @ prob.pivot
                else:
                    G[ax, 3] = sgn * trans_m
                st = state.copy()
                for cam in st.T:
                    st.T[cam] = state.T[cam] @ np.linalg.inv(G)
                d = per_frame(st)[0] - base
                k = int((d > 0).sum())
                p = float(binomtest(k, len(d), 0.5, alternative="greater").pvalue)
                de = np.array([d[ep_of == e].sum() for e in eps])
                ke = int((de > 0).sum())
                pe = float(binomtest(ke, len(de), 0.5, alternative="greater").pvalue)
                res[f"{kind}{'xyz'[ax]}{'+' if sgn > 0 else '-'}"] = {
                    "frames_worse": k, "frames": int(len(d)), "episodes_worse": ke, "episodes": int(len(de)),
                    "p_value": p, "p_value_episode": pe, "mean_cost_increase": float(d.mean()),
                    "rejected": bool(p < 0.01 and pe < 0.01 and k > len(d) / 2)}
    res["all_rejected"] = all(v["rejected"] for v in res.values() if isinstance(v, dict))
    return res


def sam_validation(task, edir: Path, held: list, mdir: Path, state: State, processes: int, per_frame_pair: int,
                   heldout_frames: int, spacing_m: float, challenge: bool = True) -> dict:
    """rig_dw's held-out checks (SAM-based where rig_dw used SAM) for a given state - VALIDATION ONLY."""
    from s4d.realworld.rig_dw_frames import Surface, load_frames
    from s4d.realworld.rig_dw_matches import load_matches
    from s4d.realworld.rig_dw_robot import RobotTermConfig
    from s4d.realworld.rig_dw_run import common_challenge, explained_and_depth, heldout_robot
    from s4d.realworld.rig_dw_solve import RigProblem
    from s4d.realworld.robot import URDFModel

    rcfg = RobotTermConfig()
    hframes = load_frames([edir / f"{e}.npz" for e in held], list(CAMS), depth_stage="foundation_stereo_depth",
                          max_frames_per_camera=heldout_frames)
    K = {fr.cam: fr.K for fr in hframes}
    surf = Surface(URDFModel.from_file(URDF), spacing_m)
    pivot = np.mean([fr.fk[fr.links.index("panda_link7")][:3, 3] for fr in hframes], 0)
    prob = RigProblem(CAMS, K, None, hframes, surf, rcfg, SolveConfig(processes=processes), pivot=pivot)
    mh = load_matches([mdir / f"{e}.npz" for e in held], list(CAMS), MatchConfig(per_frame_pair=per_frame_pair, seed=1))
    out = {"surface_fallback_links": surf.vs.provenance()["fallback_links"],
           "extext_sam_static": heldout_extext(state, prob.Keff(state), matchset(mh, 0.5, 0.01)),
           "robot": heldout_robot(prob, hframes, state, surf, rcfg),
           "robot_raw_depth": explained_and_depth(hframes, surf, state, rcfg, use_alpha=False),
           "robot_alpha_depth": explained_and_depth(hframes, surf, state, rcfg, use_alpha=True)}
    if challenge:
        out["common_challenge_sam"] = common_challenge(prob, hframes, state, rcfg)
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True)
    ap.add_argument("--corr", required=True, help="worker outputs (all episodes of the task)")
    ap.add_argument("--matches", required=True)
    ap.add_argument("--flags", required=True, help="SAM-free static flags (static_flags_worker)")
    ap.add_argument("--init", required=True, help="state JSON (SAM-free initialisation or a previous round)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--variant", default="full")
    ap.add_argument("--depth-stage", default="foundation_stereo_depth")
    ap.add_argument("--depth-dir", default=None, help="ext-ext depth samples for other stages (rig_loma_depth)")
    ap.add_argument("--corr-depth-dir", default=None, help="match-depth samples of the 2D-3D correspondences")
    ap.add_argument("--evidence", default=None, help="rig_dw evidence-v2 dir: SAM-based held-out VALIDATION only")
    ap.add_argument("--per-frame", type=int, default=100)
    ap.add_argument("--per-frame-pair", type=int, default=200)
    ap.add_argument("--sigma-px", type=float, default=1.0)
    ap.add_argument("--cauchy", type=float, default=3.0)
    ap.add_argument("--depth-weight", type=float, default=0.0)
    ap.add_argument("--ee-sigma-px", type=float, default=0.5)
    ap.add_argument("--ee-sigma-logd", type=float, default=0.01)
    ap.add_argument("--processes", type=int, default=8)
    ap.add_argument("--heldout-frames-per-camera", type=int, default=12)
    ap.add_argument("--spacing-m", type=float, default=0.0015)
    ap.add_argument("--no-heldout", action="store_true")
    ap.add_argument("--no-sam-validation", action="store_true")
    ap.add_argument("--challenge", action="store_true")
    ap.add_argument("--ref-result", default=None, help="multistart rescoring: cost of the reference state")
    a = ap.parse_args(argv)
    t0 = time.time()
    cdir, mdir, fdir = Path(a.corr), Path(a.matches), Path(a.flags)
    all_eps = sorted({p.stem for p in cdir.glob(f"{a.task}-*.npz")} & {p.stem for p in mdir.glob(f"{a.task}-*.npz")})
    fit, held = episode_split(all_eps)
    v, _, arg = a.variant.partition(":")
    mcfg = MatchConfig(per_frame_pair=a.per_frame_pair)
    ccfg = CorrConfig(per_frame=a.per_frame)
    lcfg = LomaConfig(sigma_px=a.sigma_px, cauchy=a.cauchy, depth_weight=a.depth_weight)
    scfg = SolveConfig(processes=a.processes)
    ee_sigma_logd = a.ee_sigma_logd
    fit_eps, start = list(fit), 0
    if v == "half":
        fit_eps = sorted(np.random.default_rng(int(arg)).choice(fit, len(fit) // 2, replace=False).tolist())
    elif v == "block":
        fit_eps = fit[: len(fit) // 2] if int(arg) == 0 else fit[len(fit) // 2:]
    elif v == "loco":
        lcfg = replace(lcfg, corr_cameras=tuple(c for c in CAMS if not c.startswith(arg)))
    elif v == "start":
        start = int(arg)
    elif v == "all":
        fit_eps, held = sorted(all_eps), []
    elif v == "matchseed":
        mcfg = replace(mcfg, seed=int(arg))
    elif v == "corrseed":
        ccfg = replace(ccfg, seed=int(arg))
    elif v == "extonly":
        lcfg = replace(lcfg, use_corr=False)
    elif v == "corronly":
        scfg = replace(scfg, use_extext=False)
    elif v == "alpha0":
        scfg = replace(scfg, fix_alpha=True)
    elif v == "notau":
        scfg = replace(scfg, estimate_tau=False)
    elif v == "depthterm":
        lcfg = replace(lcfg, depth_weight=1.0)
    elif v == "ext2d":
        ee_sigma_logd = 1e3
    elif v == "focal":
        scfg = replace(scfg, estimate_focal=True)
    elif v not in ("full", "group"):
        raise SystemExit(f"unknown variant {a.variant}")
    corr_depth = a.corr_depth_dir if a.corr_depth_dir else None
    corr_all, cstats = load_corr([cdir / f"{e}.npz" for e in fit], list(CAMS), ccfg,
                                 depth_stage=a.depth_stage if corr_depth else None, depth_dir=corr_depth)
    if v == "group":
        groups = corr_groups(corr_all, fit)
        fit_eps = [e for e in fit if groups.get(e) == int(arg)]
    corr, cstats = load_corr([cdir / f"{e}.npz" for e in fit_eps], list(CAMS), ccfg,
                             depth_stage=a.depth_stage if corr_depth else None, depth_dir=corr_depth)
    m = load_matches_samfree([mdir / f"{e}.npz" for e in fit_eps], fdir, list(CAMS), mcfg,
                             depth_stage=a.depth_stage, depth_dir=Path(a.depth_dir) if a.depth_dir else None)
    M = matchset(m, a.ee_sigma_px, ee_sigma_logd)
    s0 = state_from_json(a.init)
    s0.P = None
    if scfg.fix_alpha:
        s0.alpha = {c: 0.0 for c in CAMS}
    if not scfg.estimate_tau:
        s0.tau = {c: 0.0 for c in CAMS}
    with np.load(cdir / f"{fit_eps[0]}.npz") as z:
        K = {c: np.asarray(k, float) for c, k in json.loads(str(z["meta"]))["K"].items()}
    pivot = np.concatenate([cs.X for cs in corr.values()]).mean(0)
    prob = LomaProblem(CAMS, K, M, corr, lcfg, scfg, pivot=pivot)
    stages = ("shape", "common", "full")
    if v == "extonly":
        stages = ("shape",)
    if v == "corronly":
        stages = ("full",)
    log = []
    if start:
        s_shape = prob.solve(s0, stages=("shape",), log=log)
        s0 = perturb_common(s_shape, start, pivot=pivot)
        stages = ("common", "full")
    s = prob.solve(s0, stages=stages, log=log)
    if s.P is None:
        s.P = prob._ee_points(s, iters=5)
    result = {"task": a.task, "variant": a.variant, "method": "rig_loma", "fit_episodes": fit_eps,
              "heldout_episodes": held, "state": s.to_json(), "init_state": s0.to_json(), "init_file": a.init,
              "log": log, "pivot": pivot.tolist(),
              "config": {"match": asdict(mcfg), "corr": asdict(ccfg), "loma": asdict(lcfg), "solve": asdict(scfg),
                         "ee_sigma_px": a.ee_sigma_px, "ee_sigma_logd": ee_sigma_logd, "depth_stage": a.depth_stage,
                         "depth_dir": a.depth_dir, "corr_dir": str(cdir), "flags": str(fdir)},
              "match_stats": m["stats"], "corr_stats": cstats,
              "cost": prob.cost(s), "cost_parts": prob.normal(s)[3]}
    if a.ref_result:
        sref = state_from_json(a.ref_result)
        sref.P = prob._ee_points(sref, iters=5)
        result["cost_on_ref_support"] = {"solution": prob.cost(s), "reference": prob.cost(sref)}
    result["fit_corr"] = prob.corr_stats(s)
    result["fit_extext"] = heldout_extext(s, prob.Keff(s), M)
    info = prob.information(s)
    info.pop("cov", None)
    result["information"] = info
    result["moves_vs_init"] = {c: list(pose_distance(s0.T[c], s.T[c])) for c in CAMS}
    links = cstats.get("links") or []
    gl = {i for i, n in enumerate(links) if "robotiq" in n}
    if gl and all(cs.T7 is not None and np.isfinite(cs.T7).all() for cs in corr.values()):
        from s4d.realworld.rig_loma_solve import mount_information

        result["mount_information"] = mount_information(s, prob.Keff(s), corr, gl, {c: cs.T7 for c, cs in corr.items()},
                                                        a.sigma_px, a.cauchy)
    if held and not a.no_heldout:
        hcorr, hst = load_corr([cdir / f"{e}.npz" for e in held], list(CAMS), replace(ccfg, seed=1),
                               depth_stage=a.depth_stage if corr_depth else None, depth_dir=corr_depth)
        mh = load_matches_samfree([mdir / f"{e}.npz" for e in held], fdir, list(CAMS),
                                  MatchConfig(per_frame_pair=a.per_frame_pair, seed=1), depth_stage=a.depth_stage,
                                  depth_dir=Path(a.depth_dir) if a.depth_dir else None)
        result["heldout"] = {"extext_samfree": heldout_extext(s, prob.Keff(s), matchset(mh, 0.5, 0.01)),
                             "corr": prob.corr_stats(s, hcorr), "corr_load": hst}
        if v == "full" or a.challenge:
            result["heldout"]["corr_challenge"] = corr_challenge(prob, hcorr, s)
        if a.evidence and not a.no_sam_validation:
            result["heldout_sam"] = sam_validation(a.task, Path(a.evidence), held, mdir, s, a.processes,
                                                   a.per_frame_pair, a.heldout_frames_per_camera, a.spacing_m,
                                                   challenge=(v == "full" or a.challenge))
    result["seconds"] = round(time.time() - t0, 1)
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(".tmp")
    tmp.write_text(json.dumps(result, indent=1, default=lambda x: x.tolist() if hasattr(x, "tolist") else str(x)))
    tmp.replace(out)
    print(json.dumps({"out": str(out), "seconds": result["seconds"]}))


if __name__ == "__main__":
    main()
