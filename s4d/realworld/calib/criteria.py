"""Versioned acceptance criteria of the per-task rig calibration release.

``rig_loma-release/1`` (the frozen criteria, unchanged and still the default) is ``calib.loma.publish.release_verdict``:
G1-G5 of rig_dw DESIGN v1.1 (``calib.dw.publish.verdict``) + the SAM-free G6. ``rig_loma-release/2`` (agent CALSYNC,
user directive 2026-11-12 ~20:40 KST: re-evaluate the calibration / camera-synchronization criteria from evidence,
keep the original measurements and verdicts) changes exactly three things; every other gate, threshold, estimator and
solve is release/1's. Rationale and numbers: docs/realworld_calibration_release.md ("Criteria v2") and
realworld-ops/rwx/CALSYNC-REPORT-20261112.md.

1. G4 / G5b static-correspondence EVIDENCE with pairwise-nearest pairing. The release/1 RoMa matches pair every camera
   with the frame nearest the reference camera's sample time and drop a pair whose frames differ by > 20 ms. The three
   D435s run free at 15 Hz (66.7 ms) with constant phase offsets of 0-33 ms, so whole pairs were dropped for whole
   episodes (a camera was G5b-verifiable, i.e. had both of its pairs, in only 35-92 % of the episodes) and four
   pick-and-place episodes had no pair at all (-> "G5b: per-episode consistency incomplete" held every camera).
   release/2 uses the pairs' actual nearest frames (|dt| <= half a frame period; ``romav2_match_worker --pairing
   pairwise``; existing matches kept bitwise, missing pairs added by ``--supplement-of``). Static correspondences
   do not depend on dt (measured slope 0.00 +- 0.01 px per 10 ms over 935 episode-pairs); dynamic pixels are masked
   at each end's own frame time. The G5b rule itself is unchanged (median <= 1.0 px, excess + 2.58 SE <= 0.25 px,
   both pairs) and so is G4's (epipolar median <= 1.0 px, transfer <= 3.75 px); release/2 additionally REQUIRES
   G5b coverage: every episode has evidence and every camera is verifiable (both pairs) in >= 90 % of the episodes.
2. G1 as the stability RADIUS about the PUBLISHED solution (the ``all`` refit): every solve of the complete,
   unchanged stability set (near-best multistart, 5 halves, 2 workspace groups, 2 temporal blocks, 3 leave-one-camera-
   out groundings, full) must lie within 0.35 deg and 8 mm (thresholds unchanged) of the published pose, AND its
   image-space effect on the workspace must stay below one training pixel: p95 over a probe (8 x 8 grid on the
   480 x 480 training crop, lifted at 0.7 / 1.0 / 1.3 m) of the reprojection displacement <= 3.75 px. release/1's
   max-PAIRWISE spread (diameter, between two perturbed solves, neither of them published) is reported next to it.
   The 0.35 deg / 8 mm numbers were each defined as one training pixel; the radius measures how far the published
   estimate moves under the stability perturbations, the image check the downstream quantity itself (an uncompensated
   lateral translation of 7 mm passes release/1's 8 mm but fails the image check).
3. G5a (registration fixedness; detection thresholds 0.25 px / 0.03 deg / 1000 ppm unchanged) - a camera held by
   G5a ALONE is fixedness-EXCEPTION-ELIGIBLE under release/2 when the motion is immaterial: for every flagged episode
   the independent FIXREG estimate (pre-registered method, estimator B) is verifiable and its peak displacement
   D + 2.58 SE_D <= 0.375 px (= 1/10 training pixel = the existing ``MAX_PERIOD_ROT_DEG`` 0.035 deg), and FIXREG
   reports no step / drift / within-episode EVENT for the camera (a persistent change is never covered). The
   exception is still filled by the coordinator (``rig_loma_publish fill-exception``) citing this evidence.

    python -m s4d.realworld.calib.criteria static-eval --full full.json --corr CORR --evidence EVID \
        --matches M1 [--matches M2 ...] --out static_eval.json [--check-frozen]
    python -m s4d.realworld.calib.criteria verdict --results DIR --fixedness F.json --static-eval S.json \
        [--fixreg fixreg_results.json] --task T --out verdict-v2.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from s4d.realworld.calib.dw.publish import CAMS, EXPECTED, GATES, fixedness_epipolar, max_pair, spread_sets

V1 = "rig_loma-release/1"
V2 = "rig_loma-release/2"
GATE_VERSION_V2 = ("rig_loma-release/2 (CALSYNC 2026-11-12; release/1 kept): G1 = stability radius about the published "
                   "solution (<= 0.35 deg, <= 8 mm) + workspace image p95 <= 3.75 px; G2, G3, G6 = release/1; G4 / G5b "
                   "on pairwise-nearest static matches (|dt| <= half a frame period), G5b complete with >= 90 % "
                   "verifiable episodes per camera; G5a detection = release/1, a G5a-only camera is exception-eligible "
                   "iff FIXREG D + 2.58 SE <= 0.375 px on every flagged episode and no FIXREG event")
TRAINING_PX = 3.75  # full-resolution px per training px: 480 x 480 centre crop -> 128 x 128
G1_V2 = {"radius_rot_deg": GATES["spread_rot_deg"], "radius_centre_m": GATES["spread_centre_m"],
         "image_p95_px": TRAINING_PX, "reference": "all", "crop_px": 480, "grid": 8, "depths_m": (0.7, 1.0, 1.3),
         "image_hw": (480, 640)}
G5A_MATERIAL_PX = TRAINING_PX / 10  # 0.375 px ~ MAX_PERIOD_ROT_DEG (0.035 deg x fx 609 px = 0.372 px)
G5A_Z = 2.58
G5B_MIN_VERIFIABLE = 0.90
USER_DECISION_V2 = ("user directive 2026-11-12 ~20:40 KST (agent CALSYNC): evidence-based revision of the calibration "
                    "and camera-synchronization acceptance criteria (rig_loma-release/2, calib.criteria); the release/1 "
                    "verdict and evidence are preserved in the document's 'criteria' block")


def _sha256(path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _ref(path) -> dict:
    return {"path": str(path), "sha256": _sha256(path)}


# ------------------------------------------------------------------------------------------------------------- G1
def probe_points(K, *, crop_px=None, grid=None, depths=None, hw=None) -> np.ndarray:
    """Camera-frame workspace probe (N, 3): a grid x grid lattice over the centred crop_px x crop_px training crop
    (continuous pixel convention), lifted along each ray to every probe depth (metres, camera z)."""
    crop_px = G1_V2["crop_px"] if crop_px is None else crop_px
    grid = G1_V2["grid"] if grid is None else grid
    depths = G1_V2["depths_m"] if depths is None else depths
    h, w = G1_V2["image_hw"] if hw is None else hw
    K = np.asarray(K, float)
    x0, y0 = (w - crop_px) / 2.0, (h - crop_px) / 2.0
    step = crop_px / grid
    us = x0 + step * (np.arange(grid) + 0.5)
    vs = y0 + step * (np.arange(grid) + 0.5)
    uv = np.array([(u, v) for v in vs for u in us])
    rays = np.column_stack(((uv[:, 0] - K[0, 2]) / K[0, 0], (uv[:, 1] - K[1, 2]) / K[1, 1], np.ones(len(uv))))
    return np.concatenate([rays * float(d) for d in depths])


def pose_effect(T_ref, T_other, K, **probe) -> dict:
    """Downstream effect of using pose ``T_other`` instead of ``T_ref`` (both base -> camera): probe points seen by the
    camera at ``T_ref`` are reprojected with ``T_other`` (image displacement, px) and the same pixels/depths are lifted
    to the base frame with both poses (3D displacement, mm)."""
    TA, TB, K = np.asarray(T_ref, float), np.asarray(T_other, float), np.asarray(K, float)
    p = probe_points(K, **probe)
    X = (p - TA[:3, 3]) @ TA[:3, :3]            # base frame under the reference pose
    q = X @ TB[:3, :3].T + TB[:3, 3]            # camera frame under the other pose
    uA = np.column_stack((K[0, 0] * p[:, 0] / p[:, 2] + K[0, 2], K[1, 1] * p[:, 1] / p[:, 2] + K[1, 2]))
    uB = np.column_stack((K[0, 0] * q[:, 0] / q[:, 2] + K[0, 2], K[1, 1] * q[:, 1] / q[:, 2] + K[1, 2]))
    du = np.linalg.norm(uB - uA, axis=1)
    d3 = np.linalg.norm((p - TB[:3, 3]) @ TB[:3, :3] - X, axis=1) * 1e3
    return {"image_p50_px": float(np.median(du)), "image_p95_px": float(np.percentile(du, 95)),
            "image_max_px": float(du.max()), "lift_p95_mm": float(np.percentile(d3, 95))}


def g1_radius(R: dict, cam: str, K, ref: str | None = None) -> dict:
    """G1 (release/2): stability radius about the published solution ``ref`` over release/1's complete stability set,
    plus the image-space effect; release/1's diameter is reported. Incomplete evidence never passes."""
    from s4d.realworld.calib.dw.geom import pose_distance

    ref = ref or G1_V2["reference"]
    missing = [n for v in EXPECTED.values() for n in v if n not in R]
    names = sorted(set(sum(spread_sets(R, "full").values(), [])))
    out = {"reference": ref, "solves": len(names), "missing": missing, "reasons": [],
           "thresholds": {k: (list(v) if isinstance(v, tuple) else v) for k, v in G1_V2.items()},
           "diameter_release1": max_pair(R, names, cam)}
    if missing:
        out["reasons"].append(f"G1: stability evidence incomplete, missing {missing}")
    if ref not in R:
        out["reasons"].append(f"G1: published solution '{ref}' missing")
        return out
    Tref = np.asarray(R[ref]["state"]["T_cam_base"][cam], float)
    rows = {}
    for n in names:
        if n == ref:
            continue
        T = np.asarray(R[n]["state"]["T_cam_base"][cam], float)
        rot, cen = pose_distance(Tref, T)
        rows[n] = {"rot_deg": float(rot), "centre_m": float(cen), **pose_effect(Tref, T, K)}
    if not rows:
        out["reasons"].append("G1: no stability solves")
        return out
    worst = {k: max(rows, key=lambda n: rows[n][k]) for k in ("rot_deg", "centre_m", "image_p95_px")}
    out.update({"radius_rot_deg": rows[worst["rot_deg"]]["rot_deg"], "radius_centre_m": rows[worst["centre_m"]]["centre_m"],
                "image_p95_px": rows[worst["image_p95_px"]]["image_p95_px"],
                "image_p50_px_at_worst": rows[worst["image_p95_px"]]["image_p50_px"],
                "lift_p95_mm_max": max(r["lift_p95_mm"] for r in rows.values()), "worst": worst, "per_solve": rows})
    if out["radius_rot_deg"] > G1_V2["radius_rot_deg"]:
        out["reasons"].append(f"G1: rotation radius {out['radius_rot_deg']:.3f} deg > {G1_V2['radius_rot_deg']} "
                              f"({worst['rot_deg']})")
    if out["radius_centre_m"] > G1_V2["radius_centre_m"]:
        out["reasons"].append(f"G1: centre radius {1e3 * out['radius_centre_m']:.1f} mm > "
                              f"{1e3 * G1_V2['radius_centre_m']:.0f} ({worst['centre_m']})")
    if out["image_p95_px"] > G1_V2["image_p95_px"]:
        out["reasons"].append(f"G1: workspace image displacement p95 {out['image_p95_px']:.2f} px > "
                              f"{G1_V2['image_p95_px']} ({worst['image_p95_px']})")
    return out


# ------------------------------------------------------------------------------------------------------------ G5b
def g5b_v2(per_ep: dict, episodes: list, cam: str) -> dict:
    """G5b (release/2): release/1's per-episode rule (``fixedness_epipolar``, unchanged) on the pairwise evidence, plus
    completeness (every episode has evidence) and coverage (the camera is verifiable = both of its pairs present in
    >= ``G5B_MIN_VERIFIABLE`` of the episodes)."""
    epi = fixedness_epipolar(per_ep, cam) if per_ep else {"moved": [], "not_verifiable": list(episodes),
                                                          "single_pair_violations": [], "baseline_px": {}}
    missing = sorted(set(episodes) - set(per_ep))
    unver = set(epi["not_verifiable"]) | set(missing)
    coverage = 1.0 - len(unver & set(episodes)) / max(1, len(episodes))
    reasons = []
    if epi["moved"]:
        reasons.append(f"G5b: epipolar consistency flags {len(epi['moved'])} episodes {epi['moved'][:6]}")
    if missing:
        reasons.append(f"G5b: per-episode consistency incomplete ({len(missing)} episodes without evidence "
                       f"{missing[:6]})")
    if coverage < G5B_MIN_VERIFIABLE:
        reasons.append(f"G5b: camera verifiable in {coverage:.1%} of the episodes < {G5B_MIN_VERIFIABLE:.0%}")
    return {"epipolar": epi, "missing": missing, "verifiable_fraction": coverage, "reasons": reasons}


# ------------------------------------------------------------------------------------------------------------ G5a
def fixreg_series(fixreg: dict, cam: str, task: str) -> str | None:
    """The FIXREG series label of ``cam`` whose episodes are ``task``'s (the reused covering driver keeps its labels)."""
    key = cam[:12]
    for label, s in (fixreg.get("series") or {}).items():
        eps = s.get("episodes") or []
        if str(s.get("cam")) == key and eps and str(eps[0].get("episode", "")).startswith(task + "-"):
            return label
    return None


def g5a_materiality(fixreg: dict | None, cam: str, flagged: list, task: str) -> dict:
    """Exception eligibility of a camera held by G5a alone (release/2): every flagged episode verifiable in FIXREG with
    D + 2.58 SE_D <= ``G5A_MATERIAL_PX`` and no FIXREG event (T1-T4) for the camera's series."""
    out = {"flagged": list(flagged), "material_px": G5A_MATERIAL_PX, "z": G5A_Z, "episodes": {}, "problems": []}
    if not flagged:
        out["eligible"] = False
        out["problems"].append("no G5a flag")
        return out
    if not fixreg:
        out["eligible"] = False
        out["problems"].append("no FIXREG evidence")
        return out
    label = fixreg_series(fixreg, cam, task)
    events = ((fixreg.get("series") or {}).get(label) or {}).get("verdict", {}).get("events") if label else None
    out["series"], out["events"] = label, events
    if label is None:
        out["problems"].append(f"no FIXREG series of {cam} for task {task}")
    elif events:
        out["problems"].append(f"FIXREG events {events} (persistent / credible change is never covered)")
    rows = {r["episode"]: r for r in (fixreg.get("g5a_flagged") or {}).get(cam[:12], [])}
    for ep in flagged:
        r = rows.get(ep)
        if r is None or not r.get("verifiable"):
            out["episodes"][ep] = None
            out["problems"].append(f"{ep}: no verifiable FIXREG estimate")
            continue
        upper = float(r["D"]) + G5A_Z * float(r["se_D"])
        out["episodes"][ep] = {"D_px": float(r["D"]), "se_D_px": float(r["se_D"]), "upper_px": upper,
                               "T": r.get("T"), "material": upper > G5A_MATERIAL_PX}
        if upper > G5A_MATERIAL_PX:
            out["problems"].append(f"{ep}: D + {G5A_Z} SE = {upper:.3f} px > {G5A_MATERIAL_PX} px (material)")
    out["eligible"] = not out["problems"]
    return out


# -------------------------------------------------------------------------------------------- static evidence (G4/G5b)
def _per_episode(state, K, paths_by_ep: dict, per_frame_pair: int) -> dict:
    """``calib.dw.run.per_episode_extext`` over several match files per episode (in the given order: the release/1
    file first, so its subsample - protected seed 1 - is reproduced bitwise and the supplement rows follow)."""
    from s4d.realworld.calib.dw.common import matchset
    from s4d.realworld.calib.dw.geom import skew
    from s4d.realworld.calib.dw.matches import MatchConfig, load_matches

    out = {}
    for e, paths in paths_by_ep.items():
        m = load_matches(list(paths), list(CAMS), MatchConfig(per_frame_pair=per_frame_pair, seed=1))
        if not len(m["cam_a"]):
            continue
        M = matchset(m, 0.5, 0.01)
        res = {}
        for a, b, ids in M._groups:
            Rm = state.T[b][:3, :3] @ state.T[a][:3, :3].T
            t = state.T[b][:3, 3] - Rm @ state.T[a][:3, 3]
            F = np.linalg.inv(K[b]).T @ skew(t) @ Rm @ np.linalg.inv(K[a])
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


def _match_files(match_dirs: list, episodes: list) -> dict:
    return {e: [Path(d) / f"{e}.npz" for d in match_dirs if (Path(d) / f"{e}.npz").is_file()] for e in episodes}


def _dt_stats(paths_by_ep: dict) -> dict:
    """|dt| of every evaluated (episode, pair) (from the stored per-match dt_ns), split by match file."""
    rows = []
    for e, paths in paths_by_ep.items():
        for i, p in enumerate(paths):
            with np.load(p) as z:
                if not len(z["pair"]):
                    continue
                keys = np.unique(z["pair"].astype(int))
                for k in keys:
                    sel = z["pair"] == k
                    rows.append({"episode": e, "pair": int(k), "file": i,
                                 "abs_dt_ms": float(np.median(np.abs(z["dt_ns"][sel]))) / 1e6})
    return {"rows": rows}


def static_eval(full_path, corr_dir, evidence_dir, match_dirs: list, *, per_frame_pair: int = 200) -> dict:
    """G4 (held-out SAM-static correspondences) and G5b (per-episode epipolar) evidence at the full solve's state, from
    the union of ``match_dirs`` (release/1 matches first, then the pairwise supplement). Also returns the same
    quantities on the FIRST directory alone, which must reproduce the solve's stored values bitwise."""
    from s4d.realworld.calib.dw.common import heldout_extext, matchset
    from s4d.realworld.calib.dw.frames import load_frames
    from s4d.realworld.calib.dw.matches import MatchConfig, load_matches
    from s4d.realworld.calib.loma.run import state_from_json

    full = json.loads(Path(full_path).read_text())
    s = state_from_json(full_path)
    episodes = sorted(set(full["fit_episodes"]) | set(full["heldout_episodes"]))
    held = list(full["heldout_episodes"])

    def keff(K):
        out = {}
        for c in CAMS:
            k = np.asarray(K[c], float).copy()
            k[0, 0] *= np.exp(s.focal[c])
            k[1, 1] *= np.exp(s.focal[c])
            out[c] = k
        return out

    with np.load(Path(corr_dir) / f"{sorted(full['fit_episodes'])[0]}.npz") as z:  # = rig_loma_run's K (corr meta)
        K_corr = {c: np.asarray(k, float) for c, k in json.loads(str(z["meta"]))["K"].items()}
    hframes = load_frames([Path(evidence_dir) / f"{held[0]}.npz"], list(CAMS), depth_stage="foundation_stereo_depth",
                          max_frames_per_camera=1)  # = sam_validation's K (evidence frames)
    K_ev = {fr.cam: fr.K for fr in hframes}
    first = _match_files(match_dirs[:1], episodes)
    union = _match_files(match_dirs, episodes)

    def g4(files_by_ep):
        paths = [p for e in held for p in files_by_ep[e][:1]] + [p for e in held for p in files_by_ep[e][1:]]
        mh = load_matches(paths, list(CAMS), MatchConfig(per_frame_pair=per_frame_pair, seed=1))
        return heldout_extext(s, keff(K_ev), matchset(mh, 0.5, 0.01))

    return {"format": "s4d-calib-static-eval/1", "criteria": V2, "full": _ref(full_path),
            "match_dirs": [str(d) for d in match_dirs], "episodes": episodes, "heldout_episodes": held,
            "K_nominal": {c: np.asarray(K_corr[c]).tolist() for c in CAMS},
            "per_episode_extext": _per_episode(s, keff(K_corr), union, per_frame_pair),
            "heldout_extext_sam_static": g4(union),
            "release1_reproduction": {"per_episode_extext": _per_episode(s, keff(K_corr), first, per_frame_pair),
                                      "heldout_extext_sam_static": g4(first)},
            "dt": _dt_stats(union),
            "files": {e: [str(p) for p in v] for e, v in union.items()}}


def check_reproduction(se: dict, full: dict) -> list:
    """Problems if the release/1-only recomputation differs from the solve's stored G4 / G5b values (exact equality)."""
    rep = se["release1_reproduction"]
    probs = []
    if json.dumps(rep["per_episode_extext"], sort_keys=True) != json.dumps(full.get("per_episode_extext") or {},
                                                                           sort_keys=True):
        probs.append("per_episode_extext of the release/1 matches differs from full.json")
    stored = ((full.get("heldout_sam") or {}).get("extext_sam_static")) or {}
    if json.dumps(rep["heldout_extext_sam_static"], sort_keys=True) != json.dumps(stored, sort_keys=True):
        probs.append("held-out SAM-static extext of the release/1 matches differs from full.json")
    return probs


# ---------------------------------------------------------------------------------------------------------- verdict
def release_verdict_v2(R: dict, fixed: dict, se: dict, *, task: str, fixreg: dict | None = None) -> dict:
    """Per-camera release/2 verdict next to the recomputed release/1 verdict (``calib.loma.publish.release_verdict``,
    on the solve's stored evidence). ``R`` = ``calib.loma.publish.load_results`` of the batch (incl. ``all``)."""
    from s4d.realworld.calib.loma.publish import release_verdict

    v1 = release_verdict(R, fixed)
    full = R["full"]
    episodes = sorted(set(full.get("fit_episodes") or []) | set(full.get("heldout_episodes") or []))
    if sorted(se["episodes"]) != episodes:
        raise SystemExit("the static evaluation covers other episodes than the solve")
    e = se["heldout_extext_sam_static"]["all"]
    g4 = {"line_median_px": e["line_median_px"], "transfer_ab_median_px": e["transfer_ab_median_px"],
          "transfer_ba_median_px": e["transfer_ba_median_px"]}
    g4_ok = all(x is not None and np.isfinite(x) for x in g4.values()) and \
        g4["line_median_px"] <= GATES["epi_line_px"] and g4["transfer_ab_median_px"] <= GATES["transfer_px"] and \
        g4["transfer_ba_median_px"] <= GATES["transfer_px"]
    K = {c: np.asarray(k, float) for c, k in se["K_nominal"].items()}
    out = {}
    for cam in CAMS:
        r1 = v1[cam]
        kept = [r for r in r1["reasons"] if r.split(":", 1)[0] in ("G2", "G3", "G5a", "G6")]
        g1 = g1_radius(R, cam, K[cam])
        g5b = g5b_v2(se["per_episode_extext"], episodes, cam)
        reasons = list(g1["reasons"]) + [r for r in kept if r.startswith(("G2", "G3"))]
        if not g4_ok:
            reasons.append(f"G4: static correspondences (pairwise evidence) {g4}")
        reasons += [r for r in kept if r.startswith("G5a")] + g5b["reasons"] + [r for r in kept if r.startswith("G6")]
        held_by = sorted({r.split(":", 1)[0] for r in reasons})
        flagged = list(r1["base"]["fixedness"]["registration"]["moved"])
        g5a = g5a_materiality(fixreg, cam, flagged, task) if held_by == ["G5a"] else None
        checks = {g: not any(r.startswith(g + ":") for r in reasons) for g in ("G1", "G2", "G3", "G4", "G5a", "G5b",
                                                                              "G6")}
        out[cam] = {"status": "validated" if not reasons else "held", "reasons": reasons, "held_by": held_by,
                    "checks": checks, "exception_eligible_v2": bool(g5a and g5a["eligible"]),
                    "g1": {k: v for k, v in g1.items() if k != "per_solve"}, "g1_per_solve": g1.get("per_solve"),
                    "g4": g4, "g5b": {k: v for k, v in g5b.items() if k != "epipolar"} | {
                        "moved": g5b["epipolar"]["moved"], "single_pair_violations":
                        g5b["epipolar"]["single_pair_violations"], "baseline_px": g5b["epipolar"]["baseline_px"]},
                    "g5a_materiality": g5a,
                    "release1": {"status": r1["status"], "reasons": r1["reasons"], "held_by": r1["held_by"]}}
    return out


def verdict_document(R, fixed, se, *, task, fixreg=None, sources=None) -> dict:
    v = release_verdict_v2(R, fixed, se, task=task, fixreg=fixreg)
    return {"format": "s4d-calib-release-verdict/2", "criteria": V2, "gate_version": GATE_VERSION_V2, "task": task,
            "thresholds": {"g1": {k: (list(x) if isinstance(x, tuple) else x) for k, x in G1_V2.items()},
                           "g5a_material_px": G5A_MATERIAL_PX, "g5a_z": G5A_Z,
                           "g5b_min_verifiable": G5B_MIN_VERIFIABLE, "release1_gates": GATES},
            "sources": sources or {}, "cameras": v}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("static-eval")
    s.add_argument("--full", required=True)
    s.add_argument("--corr", required=True, help="corr dir of the solve (its meta K = rig_loma_run's K)")
    s.add_argument("--evidence", required=True, help="evidence dir (frames' K = the SAM validation's K)")
    s.add_argument("--matches", action="append", required=True, help="release/1 match dir first, then supplements")
    s.add_argument("--per-frame-pair", type=int, default=200)
    s.add_argument("--check-frozen", action="store_true", help="fail unless the release/1 recomputation is bitwise")
    s.add_argument("--out", required=True)
    v = sub.add_parser("verdict")
    v.add_argument("--results", required=True)
    v.add_argument("--fixedness", required=True)
    v.add_argument("--static-eval", required=True)
    v.add_argument("--fixreg", default=None)
    v.add_argument("--task", required=True)
    v.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    if a.cmd == "static-eval":
        se = static_eval(a.full, a.corr, a.evidence, a.matches, per_frame_pair=a.per_frame_pair)
        probs = check_reproduction(se, json.loads(Path(a.full).read_text()))
        se["release1_reproduction_problems"] = probs
        Path(a.out).write_text(json.dumps(se, indent=1))
        print(json.dumps({"episodes": len(se["episodes"]), "reproduction_problems": probs}))
        if probs and a.check_frozen:
            raise SystemExit(2)
        return
    from s4d.realworld.calib.loma.publish import load_results

    R = load_results(a.results)
    se = json.loads(Path(a.static_eval).read_text())
    if se["full"]["sha256"] != _sha256(Path(a.results) / "full.json"):
        raise SystemExit("the static evaluation was computed on another full.json")
    if se.get("release1_reproduction_problems") != []:
        raise SystemExit(f"static evaluation does not reproduce release/1: {se['release1_reproduction_problems']}")
    fixed = json.loads(Path(a.fixedness).read_text())
    fixreg = json.loads(Path(a.fixreg).read_text()) if a.fixreg else None
    sources = {"results_full": _ref(Path(a.results) / "full.json"), "results_all": _ref(Path(a.results) / "all.json"),
               "fixedness": _ref(a.fixedness), "static_eval": _ref(a.static_eval),
               **({"fixreg": _ref(a.fixreg)} if a.fixreg else {})}
    doc = verdict_document(R, fixed, se, task=a.task, fixreg=fixreg, sources=sources)
    Path(a.out).write_text(json.dumps(doc, indent=1))
    print(json.dumps({c: {"release1": d["release1"]["held_by"], "release2": d["status"], "held_by": d["held_by"],
                          "exception_eligible_v2": d["exception_eligible_v2"]} for c, d in doc["cameras"].items()},
                     indent=1))


if __name__ == "__main__":
    main()
