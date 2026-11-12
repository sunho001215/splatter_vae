"""Versioned acceptance criteria of the per-task rig calibration release.

``rig_loma-release/1`` (the frozen criteria, unchanged and still the default) is ``calib.loma.publish.release_verdict``:
G1-G5 of rig_dw DESIGN v1.1 (``calib.dw.publish.verdict``) + the SAM-free G6. ``rig_loma-release/2`` (agent CALSYNC,
user directive 2026-11-12 ~20:40 KST: re-evaluate the calibration / camera-synchronization criteria from evidence,
keep the original measurements and verdicts) changes exactly three things; every other gate, threshold, estimator and
solve is release/1's. Rationale and numbers: docs/realworld_calibration_release.md (section 3b) and
realworld-ops/rwx/CALSYNC-REPORT-20261112.md. Every check fails closed on missing, malformed or non-finite evidence.

1. G4 / G5b static-correspondence EVIDENCE with pairwise-nearest pairing. The release/1 RoMa matches pair every camera
   with the frame nearest the reference camera's sample time and drop a pair whose frames differ by > 20 ms. The three
   D435s run free at 15 Hz (66.7 ms) with constant phase offsets of 0-33 ms, so whole pairs were dropped for whole
   episodes (a camera had both of its pairs in only 35-92 % of the episodes) and four pick-and-place episodes had no
   pair at all ("G5b: per-episode consistency incomplete" held every camera). release/2 uses each pair's actual
   nearest frames (|dt| <= half a frame period; ``romav2_match_worker --pairing pairwise --supplement-of``: release/1
   matches kept bitwise, only the missing (time, pair) combinations added, ancestry checked here). The G5b rule
   (median <= 1.0 px, excess + 2.58 SE <= 0.25 px, both pairs) and G4's thresholds are unchanged. NEW requirements:
   every episode has evidence; every camera is verifiable (BOTH of its pairs, named from the camera list, never
   discovered from the evidence) in >= 90 % of the episodes; and the dt extension must be validated on the data: the
   per-episode-pair medians of the supplement pairs (|dt| > 20 ms) may exceed those of the release/1 pairs by at most
   0.05 px (1/5 of G5b's excess tolerance).
2. G1 as the stability RADIUS about the PUBLISHED solution (the ``all`` refit) over release/1's complete, unchanged
   stability set, thresholds unchanged (rotation <= 0.35 deg, camera centre <= 8 mm) - release/1's max-PAIRWISE spread
   (diameter, between two perturbed solves, neither of them published) is reported next to it; keeping the numbers on
   the radius admits up to twice the former diameter, hence the added image-space constraint: over the scene volume
   (``SCENE_BOX`` = the measured 3D-MVP bounds, a 16^3 lattice, points in front of the camera that project into the
   480 x 480 training crop) the reprojection displacement of every stability solve vs the published pose must have
   p50 <= 1 training px (3.75 px) and p95 <= 2 training px (7.5 px; release/1's separate rotation and translation
   tolerances of one training px each jointly admit two). An uncompensated 7.5 mm lateral translation passes the 8 mm
   centre radius but fails the image p50.
3. G5a (registration fixedness; detection thresholds 0.25 px / 0.03 deg / 1000 ppm unchanged): a camera held by G5a
   ALONE is fixedness-EXCEPTION-ELIGIBLE under release/2 iff the independent pre-registered FIXREG series of the camera
   bounds the motion of EVERY episode of the document (not only the flagged ones) below ``G5A_MATERIAL_PX`` = 1/4
   training px (0.94 px): each episode verifiable, finite D and SE_D, (D + 2.58 SE_D) / r <= 0.94 px with r = the
   camera's lowest median recovered/injected ratio of FIXREG's own synthetic injections (estimator under-recovery),
   and FIXREG lists no T1-T4 event for the camera. 1/10 training px (``MAX_PERIOD_ROT_DEG``) is below the
   instruments' resolution (per-episode 99 % bounds reach 0.65 px on camera 070988, which release/1 itself validates);
   1/4 training px is below the 128-px resampling resolution and ~1/3.5 of the measured multi-view lifting noise floor
   (held-out static-point 3D consistency 5.7 mm ~ 3.2 px). The coordinator still fills the exception
   (``rig_loma_publish fill-exception``) citing the evidence.

    python -m s4d.realworld.calib.criteria static-eval --full full.json --corr CORR --evidence EVID \
        --matches RELEASE1_DIR [--matches SUPPLEMENT_DIR] --out static_eval.json [--check-frozen]
    python -m s4d.realworld.calib.criteria verdict --results DIR --fixedness F.json --static-eval S.json \
        [--fixreg fixreg_results.json] --task T --out verdict-v2.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
from itertools import combinations
from pathlib import Path

import numpy as np

from s4d.realworld.calib.dw.publish import CAMS, EXPECTED, GATES, max_pair, spread_sets

V1 = "rig_loma-release/1"
V2 = "rig_loma-release/2"
GATE_VERSION_V2 = ("rig_loma-release/2 (CALSYNC 2026-11-12; release/1 kept): G1 = stability radius about the published "
                   "solution (<= 0.35 deg, <= 8 mm) + scene-volume image displacement p50 <= 3.75 px, p95 <= 7.5 px; "
                   "G2, G3, G6 = release/1; G4 / G5b on pairwise-nearest static matches (|dt| <= half a frame period, "
                   "dt extension validated), G5b complete, >= 90 % verifiable episodes per camera; G5a detection = "
                   "release/1, a G5a-only camera is exception-eligible iff FIXREG bounds every episode's motion below "
                   "1/4 training px (injection-corrected 99 % bound) with no FIXREG event")
TRAINING_PX = 3.75  # full-resolution px per training px: 480 x 480 centre crop -> 128 x 128
SCENE_BOX = (-0.15, -0.7, 0.15, 1.05, 0.6, 1.0)  # = s4d.rwlearn.configs.MVP3D_BOUNDS (base frame, m; test-checked)
G1_V2 = {"radius_rot_deg": GATES["spread_rot_deg"], "radius_centre_m": GATES["spread_centre_m"],
         "image_p50_px": TRAINING_PX, "image_p95_px": 2 * TRAINING_PX, "reference": "all", "box": SCENE_BOX,
         "lattice": 16, "crop_px": 480, "image_hw": (480, 640), "min_depth_m": 0.2, "min_probe_points": 100}
G5A_MATERIAL_PX = TRAINING_PX / 4  # 0.9375 px
G5A_Z = 2.58
G5B_MIN_VERIFIABLE = 0.90
G5B_DT_MAX_DIFF_PX = 0.05  # supplement (|dt| > 20 ms) vs release/1 pairs: median excess allowed (1/5 of 0.25 px)
RELEASE1_MAX_DT_MS = 20.0
USER_DECISION_V2 = ("user directive 2026-11-12 ~20:40 KST (agent CALSYNC): evidence-based revision of the calibration "
                    "and camera-synchronization acceptance criteria (rig_loma-release/2, calib.criteria); the release/1 "
                    "verdict and evidence are preserved in the document's 'criteria' block")


def _sha256(path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _ref(path) -> dict:
    return {"path": str(path), "sha256": _sha256(path)}


def _finite(*xs) -> bool:
    try:
        return all(x is not None and np.isfinite(float(x)) for x in xs)
    except (TypeError, ValueError):
        return False


def pair_name(a: str, b: str) -> str:
    return f"{a[:12]}-{b[:12]}"


def required_pairs(cam: str) -> list:
    """The two camera pairs containing ``cam`` (named from CAMS, never discovered from the evidence)."""
    return sorted(pair_name(a, b) for a, b in combinations(CAMS, 2) if cam in (a, b))


# ------------------------------------------------------------------------------------------------------------- G1
def probe_points(T_ref, K, *, box=None, lattice=None, crop_px=None, hw=None, min_depth=None) -> np.ndarray:
    """Base-frame probe (N, 3): a lattice^3 grid over the scene box, kept where it lies >= min_depth in front of the
    camera at ``T_ref`` and projects into the centred crop_px x crop_px training crop."""
    box = G1_V2["box"] if box is None else box
    n = G1_V2["lattice"] if lattice is None else lattice
    crop_px = G1_V2["crop_px"] if crop_px is None else crop_px
    h, w = G1_V2["image_hw"] if hw is None else hw
    min_depth = G1_V2["min_depth_m"] if min_depth is None else min_depth
    T, K = np.asarray(T_ref, float), np.asarray(K, float)
    g = [np.linspace(box[i], box[i + 3], n) for i in range(3)]
    X = np.stack(np.meshgrid(*g, indexing="ij"), -1).reshape(-1, 3)
    p = X @ T[:3, :3].T + T[:3, 3]
    with np.errstate(divide="ignore", invalid="ignore"):
        u = K[0, 0] * p[:, 0] / p[:, 2] + K[0, 2]
        v = K[1, 1] * p[:, 1] / p[:, 2] + K[1, 2]
    x0, y0 = (w - crop_px) / 2.0, (h - crop_px) / 2.0
    ok = (p[:, 2] >= min_depth) & (u >= x0) & (u < x0 + crop_px) & (v >= y0) & (v < y0 + crop_px)
    return X[ok]


def pose_effect(T_ref, T_other, K, X=None) -> dict:
    """Downstream effect of using pose ``T_other`` instead of ``T_ref`` (both base -> camera) on the base-frame probe X
    (default ``probe_points``): reprojection displacement (px) and the 3D displacement of the same pixels lifted at the
    same depths with the other pose (mm)."""
    TA, TB, K = np.asarray(T_ref, float), np.asarray(T_other, float), np.asarray(K, float)
    X = probe_points(TA, K) if X is None else X
    pa = X @ TA[:3, :3].T + TA[:3, 3]
    pb = X @ TB[:3, :3].T + TB[:3, 3]
    uA = np.column_stack((K[0, 0] * pa[:, 0] / pa[:, 2] + K[0, 2], K[1, 1] * pa[:, 1] / pa[:, 2] + K[1, 2]))
    uB = np.column_stack((K[0, 0] * pb[:, 0] / pb[:, 2] + K[0, 2], K[1, 1] * pb[:, 1] / pb[:, 2] + K[1, 2]))
    du = np.linalg.norm(uB - uA, axis=1)
    d3 = np.linalg.norm((pa - TB[:3, 3]) @ TB[:3, :3] - X, axis=1) * 1e3
    if not len(du):
        return {"points": 0, "image_p50_px": float("nan"), "image_p95_px": float("nan"), "image_max_px": float("nan"),
                "lift_p95_mm": float("nan")}
    return {"points": int(len(du)), "image_p50_px": float(np.median(du)), "image_p95_px": float(np.percentile(du, 95)),
            "image_max_px": float(du.max()), "lift_p95_mm": float(np.percentile(d3, 95))}


def g1_radius(R: dict, cam: str, K, ref: str | None = None) -> dict:
    """G1 (release/2): stability radius about the published solution ``ref`` over release/1's complete stability set,
    plus the scene-volume image check; release/1's diameter is reported. Incomplete or non-finite evidence never
    passes."""
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
    K = np.asarray(K, float)
    Tref = np.asarray(R[ref]["state"]["T_cam_base"][cam], float)
    if Tref.shape != (4, 4) or not np.isfinite(Tref).all() or K.shape != (3, 3) or not np.isfinite(K).all():
        out["reasons"].append("G1: published pose or intrinsics not finite")
        return out
    X = probe_points(Tref, K)
    out["probe_points"] = int(len(X))
    if len(X) < G1_V2["min_probe_points"]:
        out["reasons"].append(f"G1: scene probe has {len(X)} points < {G1_V2['min_probe_points']}")
        return out
    rows = {}
    for n in names:
        if n == ref:
            continue
        T = np.asarray(R[n]["state"]["T_cam_base"][cam], float)
        if T.shape != (4, 4) or not np.isfinite(T).all():
            out["reasons"].append(f"G1: solve {n} pose not finite")
            continue
        rot, cen = pose_distance(Tref, T)
        rows[n] = {"rot_deg": float(rot), "centre_m": float(cen), **pose_effect(Tref, T, K, X)}
    if not rows:
        out["reasons"].append("G1: no stability solves")
        return out
    keys = ("rot_deg", "centre_m", "image_p50_px", "image_p95_px", "lift_p95_mm")
    if not all(_finite(r[k]) for r in rows.values() for k in keys):
        out["reasons"].append("G1: non-finite stability metrics")
        return out
    worst = {k: max(rows, key=lambda n: rows[n][k]) for k in keys}
    out.update({"radius_rot_deg": rows[worst["rot_deg"]]["rot_deg"], "radius_centre_m": rows[worst["centre_m"]]["centre_m"],
                "image_p50_px": rows[worst["image_p50_px"]]["image_p50_px"],
                "image_p95_px": rows[worst["image_p95_px"]]["image_p95_px"],
                "lift_p95_mm": rows[worst["lift_p95_mm"]]["lift_p95_mm"], "worst": worst, "per_solve": rows})
    if out["radius_rot_deg"] > G1_V2["radius_rot_deg"]:
        out["reasons"].append(f"G1: rotation radius {out['radius_rot_deg']:.3f} deg > {G1_V2['radius_rot_deg']} "
                              f"({worst['rot_deg']})")
    if out["radius_centre_m"] > G1_V2["radius_centre_m"]:
        out["reasons"].append(f"G1: centre radius {1e3 * out['radius_centre_m']:.1f} mm > "
                              f"{1e3 * G1_V2['radius_centre_m']:.0f} ({worst['centre_m']})")
    for k in ("image_p50_px", "image_p95_px"):
        if out[k] > G1_V2[k]:
            out["reasons"].append(f"G1: scene image displacement {k[6:9]} {out[k]:.2f} px > {G1_V2[k]} ({worst[k]})")
    return out


# ------------------------------------------------------------------------------------------------------------ G5b
def epipolar_v2(per_ep: dict, cam: str, episodes: list, max_px: float = 1.0, max_excess_px: float = 0.25) -> dict:
    """release/1's per-episode rule (``calib.dw.publish.fixedness_epipolar``: median <= max_px and excess over the
    pair's baseline + 2.58 SE <= max_excess_px; moved iff BOTH pairs violate) with the camera's two pairs REQUIRED:
    an episode lacking a pair (or with a non-finite measurement) is not verifiable for the camera; a pair absent from
    every episode leaves the camera unverifiable everywhere."""
    pairs = required_pairs(cam)

    def val(v, p):
        r = (v or {}).get(p)
        return r if isinstance(r, dict) and _finite(r.get("median_px"), r.get("se_px")) else None

    base = {}
    for p in pairs:
        xs = [val(per_ep.get(e), p)["median_px"] for e in episodes if val(per_ep.get(e), p) is not None]
        base[p] = float(np.median(xs)) if xs else None
    moved, single, unver = [], [], []
    for ep in episodes:
        v = per_ep.get(ep)
        rows = {p: val(v, p) for p in pairs}
        if any(r is None or base[p] is None for p, r in rows.items()):
            unver.append(ep)
        bad = [p for p, r in rows.items() if r is not None and base[p] is not None and
               (r["median_px"] > max_px or (r["median_px"] - base[p]) + 2.58 * r["se_px"] > max_excess_px)]
        if len(bad) == len(pairs):
            moved.append(ep)
        elif bad:
            single.append((ep, bad))
    return {"episodes": len(episodes), "pairs": pairs, "baseline_px": base, "moved": moved,
            "single_pair_violations": single, "not_verifiable": sorted(set(unver))}


def dt_extension(se: dict) -> dict:
    """Validation of the dt extension on the data: per-episode-pair G5b medians of pairs evaluated at |dt| <= 20 ms
    (release/1) vs > 20 ms (pairwise supplement). Fails closed when the dt table is missing or inconsistent."""
    pe = se.get("per_episode_extext") or {}
    rows = (se.get("dt") or {}).get("rows")
    if not isinstance(rows, list):
        return {"ok": False, "problem": "no dt table in the static evaluation"}
    lo, hi = [], []
    for r in rows:
        try:
            pa, pb = divmod(int(r["pair"]), 10)
            name = pair_name(CAMS[pa], CAMS[pb])
            m = pe[r["episode"]][name]["median_px"]
            dt = float(r["abs_dt_ms"])
        except (KeyError, IndexError, TypeError, ValueError):
            return {"ok": False, "problem": f"dt row {r} has no matching per-episode measurement"}
        if not _finite(m, dt):
            return {"ok": False, "problem": f"dt row {r}: non-finite"}
        (lo if dt <= RELEASE1_MAX_DT_MS else hi).append(m)
    out = {"n_le20": len(lo), "n_gt20": len(hi), "median_le20_px": float(np.median(lo)) if lo else None,
           "median_gt20_px": float(np.median(hi)) if hi else None, "max_diff_px": G5B_DT_MAX_DIFF_PX}
    if not hi:
        return {**out, "ok": True, "problem": None}
    if not lo:
        return {**out, "ok": False, "problem": "no release/1 (|dt| <= 20 ms) pairs to compare with"}
    diff = out["median_gt20_px"] - out["median_le20_px"]
    out["diff_px"] = float(diff)
    return {**out, "ok": diff <= G5B_DT_MAX_DIFF_PX,
            "problem": None if diff <= G5B_DT_MAX_DIFF_PX else
            f"pairs at |dt| > 20 ms are {diff:.3f} px less consistent than at <= 20 ms (> {G5B_DT_MAX_DIFF_PX})"}


def g5b_v2(per_ep: dict, episodes: list, cam: str, dt_check: dict | None = None) -> dict:
    """G5b (release/2): ``epipolar_v2`` on the pairwise evidence + completeness (every episode has evidence) + coverage
    (both pairs in >= ``G5B_MIN_VERIFIABLE`` of the episodes) + the validated dt extension (``dt_extension``)."""
    epi = epipolar_v2(per_ep, cam, episodes)
    missing = sorted(e for e in episodes if not per_ep.get(e))
    coverage = 1.0 - len(set(epi["not_verifiable"]) & set(episodes)) / max(1, len(episodes))
    reasons = []
    if epi["moved"]:
        reasons.append(f"G5b: epipolar consistency flags {len(epi['moved'])} episodes {epi['moved'][:6]}")
    if missing:
        reasons.append(f"G5b: per-episode consistency incomplete ({len(missing)} episodes without evidence "
                       f"{missing[:6]})")
    if coverage < G5B_MIN_VERIFIABLE:
        reasons.append(f"G5b: camera verifiable (both pairs) in {coverage:.1%} of the episodes < "
                       f"{G5B_MIN_VERIFIABLE:.0%}")
    if dt_check is not None and not dt_check.get("ok"):
        reasons.append(f"G5b: dt extension not validated ({dt_check.get('problem')})")
    return {"epipolar": epi, "missing": missing, "verifiable_fraction": coverage, "reasons": reasons}


# ------------------------------------------------------------------------------------------------------------ G5a
def fixreg_series(fixreg: dict, cam: str, task: str) -> str | None:
    """The FIXREG series label of ``cam`` whose episodes are ``task``'s (the reused covering driver keeps its labels)."""
    key = cam[:12]
    found = [label for label, s in (fixreg.get("series") or {}).items()
             if str(s.get("cam")) == key and (s.get("episodes") or [])
             and all(str(e.get("episode", "")).startswith(task + "-") for e in s["episodes"])]
    return found[0] if len(found) == 1 else None


def g5a_materiality(fixreg: dict | None, cam: str, flagged: list, task: str, episodes: list) -> dict:
    """Exception eligibility of a camera held by G5a alone (release/2): the camera's FIXREG series covers every episode
    of the document, each verifiable with finite D / SE_D, the injection-corrected 99 % bound (D + 2.58 SE_D) / r <=
    ``G5A_MATERIAL_PX`` (r = lowest median recovered/injected D ratio of the camera's synthetic injections), and the
    series verdict lists no event."""
    out = {"flagged": list(flagged), "material_px": G5A_MATERIAL_PX, "z": G5A_Z, "problems": []}
    if not flagged:
        out["problems"].append("no G5a flag")
    if not isinstance(fixreg, dict):
        out["problems"].append("no FIXREG evidence")
        return {**out, "eligible": False}
    label = fixreg_series(fixreg, cam, task)
    s = (fixreg.get("series") or {}).get(label) if label else None
    out["series"] = label
    if not isinstance(s, dict):
        out["problems"].append(f"no unique FIXREG series of {cam} for task {task}")
        return {**out, "eligible": False}
    verdict = s.get("verdict")
    events = verdict.get("events") if isinstance(verdict, dict) else None
    out["events"], out["fixreg_verdict"] = events, verdict.get("verdict") if isinstance(verdict, dict) else None
    if not isinstance(events, list):
        out["problems"].append("FIXREG verdict has no explicit event list")
    elif events:
        out["problems"].append(f"FIXREG events {events} (persistent / credible change is never covered)")
    ratios = {}
    for row in (s.get("sensitivity") or {}).get("rows") or []:
        if row.get("kind") == "zero" or not row.get("verifiable"):
            continue
        if _finite(row.get("recovered_D"), row.get("truth_D")) and float(row["truth_D"]) > 0:
            ratios.setdefault(f"{row['kind']}{row.get('val')}", []).append(float(row["recovered_D"]) / float(row["truth_D"]))
    med = {k: float(np.median(v)) for k, v in ratios.items()}
    r = min(med.values()) if med else None
    out["recovery"] = {"per_injection": med, "r": r}
    if r is None or not 0 < r <= 1.5:
        out["problems"].append(f"no valid injection recovery ratio ({r})")
        return {**out, "eligible": False}
    by_ep = {e.get("episode"): e for e in s.get("episodes") or []}
    rows, worst = {}, None
    for ep in episodes:
        e = by_ep.get(ep)
        if e is None or not e.get("verifiable_B") or not _finite(e.get("D_B"), e.get("se_D_B")) \
                or float(e["D_B"]) < 0 or float(e["se_D_B"]) < 0:
            out["problems"].append(f"{ep}: no verifiable finite FIXREG estimate")
            continue
        upper = (float(e["D_B"]) + G5A_Z * float(e["se_D_B"])) / r
        rows[ep] = {"D_px": float(e["D_B"]), "se_D_px": float(e["se_D_B"]), "upper_corrected_px": upper}
        if worst is None or upper > rows[worst]["upper_corrected_px"]:
            worst = ep
    out["episodes"], out["worst"] = rows, worst
    over = [ep for ep, v in rows.items() if v["upper_corrected_px"] > G5A_MATERIAL_PX]
    if over:
        out["problems"].append(f"{len(over)} episodes with a corrected 99 % bound > {G5A_MATERIAL_PX} px {over[:6]}")
    out["flagged_bounds"] = {ep: rows.get(ep) for ep in flagged}
    return {**out, "eligible": not out["problems"]}


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


def match_inputs(match_dirs: list, episodes: list) -> tuple[dict, dict]:
    """({episode: [release/1 file, supplement files...]}, manifest) after validating the inputs: per episode exactly
    one release/1 (reference-pairing) file, every further file a pairwise supplement whose ``supplement_of`` names that
    release/1 file by sha256, no (time, pair) combination in two files, and every file's rows consistent with its
    meta (pair frames, |dt| within its own tolerance). Raises ValueError otherwise. The manifest hashes every file."""
    files, manifest = {}, {}
    for e in episodes:
        paths = [Path(d) / f"{e}.npz" for d in match_dirs if (Path(d) / f"{e}.npz").is_file()]
        if not paths or paths[0].parent != Path(match_dirs[0]):
            raise ValueError(f"{e}: no release/1 match file in {match_dirs[0]}")
        entries, seen = [], set()
        for i, p in enumerate(paths):
            sha = _sha256(p)
            with np.load(p) as z:
                meta = json.loads(str(z["meta"]))
                keys = {(int(t), int(q)) for t, q in zip(z["time"].astype(int), z["pair"].astype(int))} \
                    if len(z["pair"]) else set()
                dts = {(int(t), int(q)): int(d) for t, q, d in zip(z["time"].astype(int), z["pair"].astype(int),
                                                                   z["dt_ns"].astype(np.int64))} if len(z["pair"]) else {}
            if meta.get("episode_id") != e:
                raise ValueError(f"{p}: episode_id {meta.get('episode_id')!r} != {e}")
            pairing = meta.get("pairing", "reference")
            if i == 0:
                if pairing != "reference":
                    raise ValueError(f"{p}: the first match file must be the release/1 (reference-pairing) file")
                tol = float(meta.get("max_dt_ms", RELEASE1_MAX_DT_MS)) * 1e6
                if any(abs(d) > tol for d in dts.values()):
                    raise ValueError(f"{p}: rows beyond its own max_dt_ms")
            else:
                sup = meta.get("supplement_of") or {}
                if pairing != "pairwise" or sup.get("sha256") != entries[0]["sha256"]:
                    raise ValueError(f"{p}: not a pairwise supplement of {paths[0]} (supplement_of sha256 mismatch)")
                planned = {(int(r[0]), int(r[1]) * 10 + int(r[2])): int(r[5]) for r in meta.get("pair_frames") or []}
                tol = meta.get("max_dt_ns") or {}
                for (k, q), d in dts.items():
                    if planned.get((k, q)) != d or abs(d) > int(tol.get(f"{q // 10}{q % 10}", -1)):
                        raise ValueError(f"{p}: row (time {k}, pair {q}) not in its plan or beyond its tolerance")
            dup = seen & keys
            if dup:
                raise ValueError(f"{e}: (time, pair) combinations {sorted(dup)[:4]} in two match files")
            seen |= keys
            entries.append({"path": str(p), "sha256": sha, "pairing": pairing, "combinations": len(keys),
                            "abs_dt_ms_max": max((abs(d) / 1e6 for d in dts.values()), default=None)})
        files[e], manifest[e] = paths, entries
    return files, manifest


def _dt_rows(paths_by_ep: dict) -> list:
    """|dt| of every evaluated (episode, pair) and the file it came from (index 0 = release/1)."""
    rows = []
    for e, paths in paths_by_ep.items():
        for i, p in enumerate(paths):
            with np.load(p) as z:
                if not len(z["pair"]):
                    continue
                for k in np.unique(z["pair"].astype(int)):
                    sel = z["pair"] == k
                    rows.append({"episode": e, "pair": int(k), "file": i,
                                 "abs_dt_ms": float(np.median(np.abs(z["dt_ns"][sel]))) / 1e6})
    return rows


def static_eval(full_path, corr_dir, evidence_dir, match_dirs: list, *, per_frame_pair: int = 200) -> dict:
    """G4 (held-out SAM-static correspondences) and G5b (per-episode epipolar) evidence at the full solve's state, from
    the validated union of ``match_dirs`` (release/1 matches first, then the pairwise supplement). Also returns the
    same quantities on the release/1 files alone, which must reproduce the solve's stored values bitwise."""
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

    corr_file = Path(corr_dir) / f"{sorted(full['fit_episodes'])[0]}.npz"
    with np.load(corr_file) as z:  # = rig_loma_run's K (corr meta)
        K_corr = {c: np.asarray(k, float) for c, k in json.loads(str(z["meta"]))["K"].items()}
    ev_file = Path(evidence_dir) / f"{held[0]}.npz"
    hframes = load_frames([ev_file], list(CAMS), depth_stage="foundation_stereo_depth",
                          max_frames_per_camera=1)  # = sam_validation's K (evidence frames)
    K_ev = {fr.cam: fr.K for fr in hframes}
    union, manifest = match_inputs(match_dirs, episodes)
    first = {e: v[:1] for e, v in union.items()}

    def g4(files_by_ep):
        paths = [p for e in held for p in files_by_ep[e][:1]] + [p for e in held for p in files_by_ep[e][1:]]
        mh = load_matches(paths, list(CAMS), MatchConfig(per_frame_pair=per_frame_pair, seed=1))
        return heldout_extext(s, keff(K_ev), matchset(mh, 0.5, 0.01))

    code = Path(__file__)
    return {"format": "s4d-calib-static-eval/2", "criteria": V2, "full": _ref(full_path),
            "match_dirs": [str(d) for d in match_dirs], "episodes": episodes, "heldout_episodes": held,
            "K_nominal": {c: np.asarray(K_corr[c]).tolist() for c in CAMS},
            "K_sources": {"per_episode": _ref(corr_file), "heldout": _ref(ev_file)},
            "code": {"criteria.py": _sha256(code), "dw/run.py": _sha256(code.parent / "dw" / "run.py"),
                     "dw/matches.py": _sha256(code.parent / "dw" / "matches.py"),
                     "dw/common.py": _sha256(code.parent / "dw" / "common.py")},
            "per_frame_pair": per_frame_pair,
            "per_episode_extext": _per_episode(s, keff(K_corr), union, per_frame_pair),
            "heldout_extext_sam_static": g4(union),
            "release1_reproduction": {"per_episode_extext": _per_episode(s, keff(K_corr), first, per_frame_pair),
                                      "heldout_extext_sam_static": g4(first)},
            "dt": {"rows": _dt_rows(union)}, "inputs": manifest}


def check_reproduction(se: dict, full: dict) -> list:
    """Problems if the release/1-only recomputation differs from the solve's stored G4 / G5b values (exact equality)."""
    rep = se.get("release1_reproduction") or {}
    probs = []
    if json.dumps(rep.get("per_episode_extext"), sort_keys=True) != json.dumps(full.get("per_episode_extext") or {},
                                                                               sort_keys=True):
        probs.append("per_episode_extext of the release/1 matches differs from full.json")
    stored = ((full.get("heldout_sam") or {}).get("extext_sam_static")) or {}
    if json.dumps(rep.get("heldout_extext_sam_static"), sort_keys=True) != json.dumps(stored, sort_keys=True):
        probs.append("held-out SAM-static extext of the release/1 matches differs from full.json")
    return probs


def check_static_eval(se: dict, full_path) -> None:
    """Refuse a static evaluation that is not of this full.json or does not reproduce release/1 (recomputed here from
    the embedded measurements, never from a cached flag)."""
    if (se.get("full") or {}).get("sha256") != _sha256(full_path):
        raise SystemExit("the static evaluation was computed on another full.json")
    probs = check_reproduction(se, json.loads(Path(full_path).read_text()))
    if probs:
        raise SystemExit(f"static evaluation does not reproduce release/1: {probs}")


# ---------------------------------------------------------------------------------------------------------- verdict
def release_verdict_v2(R: dict, fixed: dict, se: dict, *, task: str, fixreg: dict | None = None) -> dict:
    """Per-camera release/2 verdict next to the recomputed release/1 verdict (``calib.loma.publish.release_verdict``,
    on the solve's stored evidence). ``R`` = ``calib.loma.publish.load_results`` of the batch (incl. ``all``)."""
    from s4d.realworld.calib.loma.publish import release_verdict

    v1 = release_verdict(R, fixed)
    full = R["full"]
    episodes = sorted(set(full.get("fit_episodes") or []) | set(full.get("heldout_episodes") or []))
    if sorted(se.get("episodes") or []) != episodes:
        raise SystemExit("the static evaluation covers other episodes than the solve")
    e = (se.get("heldout_extext_sam_static") or {}).get("all") or {}
    g4 = {k: e.get(k) for k in ("line_median_px", "transfer_ab_median_px", "transfer_ba_median_px")}
    g4_ok = _finite(*g4.values()) and g4["line_median_px"] <= GATES["epi_line_px"] and \
        g4["transfer_ab_median_px"] <= GATES["transfer_px"] and g4["transfer_ba_median_px"] <= GATES["transfer_px"]
    K = {c: np.asarray(k, float) for c, k in (se.get("K_nominal") or {}).items()}
    dtc = dt_extension(se)
    out = {}
    for cam in CAMS:
        r1 = v1[cam]
        kept = [r for r in r1["reasons"] if r.split(":", 1)[0] in ("G2", "G3", "G5a", "G6")]
        g1 = g1_radius(R, cam, K.get(cam, np.full((3, 3), np.nan)))
        g5b = g5b_v2(se.get("per_episode_extext") or {}, episodes, cam, dtc)
        reasons = list(g1["reasons"]) + [r for r in kept if r.startswith(("G2", "G3"))]
        if not g4_ok:
            reasons.append(f"G4: static correspondences (pairwise evidence) {g4}")
        reasons += [r for r in kept if r.startswith("G5a")] + g5b["reasons"] + [r for r in kept if r.startswith("G6")]
        held_by = sorted({r.split(":", 1)[0] for r in reasons})
        flagged = list(r1["base"]["fixedness"]["registration"]["moved"])
        g5a = g5a_materiality(fixreg, cam, flagged, task, episodes) if held_by == ["G5a"] else None
        checks = {g: not any(r.startswith(g + ":") for r in reasons) for g in ("G1", "G2", "G3", "G4", "G5a", "G5b",
                                                                              "G6")}
        out[cam] = {"status": "validated" if not reasons else "held", "reasons": reasons, "held_by": held_by,
                    "checks": checks, "exception_eligible_v2": bool(g5a and g5a["eligible"]),
                    "g1": {k: v for k, v in g1.items() if k != "per_solve"}, "g1_per_solve": g1.get("per_solve"),
                    "g4": g4, "g5b": {k: v for k, v in g5b.items() if k != "epipolar"} | {
                        "moved": g5b["epipolar"]["moved"], "single_pair_violations":
                        g5b["epipolar"]["single_pair_violations"], "baseline_px": g5b["epipolar"]["baseline_px"],
                        "not_verifiable": g5b["epipolar"]["not_verifiable"], "dt_extension": dtc},
                    "g5a_materiality": g5a,
                    "release1": {"status": r1["status"], "reasons": r1["reasons"], "held_by": r1["held_by"]}}
    return out


def verdict_document(R, fixed, se, *, task, fixreg=None, sources=None) -> dict:
    v = release_verdict_v2(R, fixed, se, task=task, fixreg=fixreg)
    return {"format": "s4d-calib-release-verdict/2", "criteria": V2, "gate_version": GATE_VERSION_V2, "task": task,
            "thresholds": {"g1": {k: (list(x) if isinstance(x, tuple) else x) for k, x in G1_V2.items()},
                           "g5a_material_px": G5A_MATERIAL_PX, "g5a_z": G5A_Z,
                           "g5b_min_verifiable": G5B_MIN_VERIFIABLE, "g5b_dt_max_diff_px": G5B_DT_MAX_DIFF_PX,
                           "release1_gates": GATES},
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
        Path(a.out).write_text(json.dumps(se, indent=1))
        print(json.dumps({"episodes": len(se["episodes"]), "reproduction_problems": probs,
                          "dt_extension": dt_extension(se)}))
        if probs and a.check_frozen:
            raise SystemExit(2)
        return
    from s4d.realworld.calib.loma.publish import load_results

    R = load_results(a.results)
    se = json.loads(Path(a.static_eval).read_text())
    check_static_eval(se, Path(a.results) / "full.json")
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
