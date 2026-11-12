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
   discovered from the evidence; finite, non-negative, >= 30 matches) in >= 90 % of the episodes; and the dt extension
   is validated on the data (``dt_extension``): the dt table reconciles exactly with the evaluated measurements, per
   camera pair the long-offset (|dt| > 20 ms) per-episode medians exceed the short-offset ones by <= 0.05 px at the
   median and <= 0.10 px at p90, and NO long-offset episode-pair violates the per-episode G5b rule (release/1 evidence
   of pick-and-place and covering: 0 violations). Match files are validated against the episode timestamps (frames,
   camera order, dt, half-period limit) before use.
2. G1 as the stability RADIUS about the PUBLISHED solution (the ``all`` refit) over release/1's complete, unchanged
   stability set, thresholds unchanged (rotation <= 0.35 deg, camera centre <= 8 mm) - release/1's max-PAIRWISE spread
   (diameter, between two perturbed solves, neither of them published) is reported next to it; keeping the numbers on
   the radius admits up to twice the former diameter, hence the added image-space constraint: over the scene volume
   (``SCENE_BOX`` = the measured 3D-MVP bounds, a 16^3 lattice, points in front of the camera that project into the
   480 x 480 training crop) the reprojection displacement of every stability solve vs the published pose must have
   p50 <= 1 training px (3.75 px) and p95 <= 2 training px (7.5 px; release/1's separate rotation and translation
   tolerances of one training px each jointly admit two). An uncompensated 7.5 mm lateral translation passes the 8 mm
   centre radius but fails the image p50.
3. G5a is NOT changed (detection thresholds, the fixedness exception procedure and its FIXREG rule are release/1's).
   ``g5a_materiality`` only REPORTS decision support for a camera held by G5a alone: FIXREG's per-episode pooled
   displacement bounds (D + 2.58 SE_D, corrected by the median and by the lowest observed injection recovery), the
   within-episode group maxima and FIXREG's unresolved candidates, against 1/10 and 1/4 training px. Codex review 2
   showed that these bounds lack validated coverage near such budgets (recovery uncertainty, within-episode motion),
   so no materiality-based eligibility is derived from them; a G5a-only camera still needs the release/1 decision.

    python -m s4d.realworld.calib.criteria static-eval --full full.json --corr CORR --evidence EVID --task-root VIEW \
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
                   "dt extension validated per pair), G5b complete, >= 90 % verifiable episodes per camera; G5a and "
                   "its exception procedure = release/1 (materiality reported as decision support only)")
TRAINING_PX = 3.75  # full-resolution px per training px: 480 x 480 centre crop -> 128 x 128
SCENE_BOX = (-0.15, -0.7, 0.15, 1.05, 0.6, 1.0)  # = s4d.rwlearn.configs.MVP3D_BOUNDS (base frame, m; test-checked)
G1_V2 = {"radius_rot_deg": GATES["spread_rot_deg"], "radius_centre_m": GATES["spread_centre_m"],
         "image_p50_px": TRAINING_PX, "image_p95_px": 2 * TRAINING_PX, "reference": "all", "box": SCENE_BOX,
         "lattice": 16, "crop_px": 480, "image_hw": (480, 640), "min_depth_m": 0.2, "min_probe_points": 100}
G5A_BUDGETS_PX = (TRAINING_PX / 10, TRAINING_PX / 4)  # decision-support reference levels only (not gates)
G5A_Z = 2.58
G5B_MIN_MATCHES = 30
G5B_MIN_VERIFIABLE = 0.90
G5B_DT_MAX_DIFF_PX = 0.05  # long (|dt| > 20 ms) vs short-offset pairs: median excess allowed (1/5 of 0.25 px)
G5B_DT_MAX_P90_DIFF_PX = 0.10
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
        ok = isinstance(r, dict) and _finite(r.get("median_px"), r.get("se_px"), r.get("n")) and \
            float(r["median_px"]) >= 0 and float(r["se_px"]) >= 0 and int(r["n"]) >= G5B_MIN_MATCHES
        return r if ok else None

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


def _violates(m: float, se: float, base: float, max_px: float = 1.0, max_excess_px: float = 0.25) -> bool:
    return m > max_px or (m - base) + 2.58 * se > max_excess_px


def dt_extension(se: dict) -> dict:
    """Validation of the dt extension on the data (fail closed). The dt table must reconcile exactly with the evaluated
    per-episode measurements (one row per (episode, pair) and vice versa). Per camera pair with long-offset rows
    (|dt| > 20 ms, i.e. pairs only the pairwise supplement evaluates): the long-offset per-episode medians may exceed the
    short-offset ones (same pair if it has >= 10, else all pairs pooled) by <= ``G5B_DT_MAX_DIFF_PX`` at the median and
    <= ``G5B_DT_MAX_P90_DIFF_PX`` at p90, and no long-offset episode-pair may violate the per-episode G5b rule against
    its pair's baseline."""
    pe = se.get("per_episode_extext") or {}
    rows = (se.get("dt") or {}).get("rows")
    if not isinstance(rows, list):
        return {"ok": False, "problem": "no dt table in the static evaluation"}
    measured = {(e, p) for e, v in pe.items() for p in (v or {})}
    table = {}
    for r in rows:
        try:
            pa, pb = divmod(int(r["pair"]), 10)
            key = (r["episode"], pair_name(CAMS[pa], CAMS[pb]))
            dt = float(r["abs_dt_ms"])
        except (KeyError, IndexError, TypeError, ValueError):
            return {"ok": False, "problem": f"malformed dt row {r}"}
        if key in table or not _finite(dt):
            return {"ok": False, "problem": f"dt row {r}: duplicate or non-finite (a pair must come from one file)"}
        table[key] = dt
    if set(table) != measured:
        return {"ok": False, "problem": f"dt table does not reconcile with the measurements "
                                        f"({len(set(table) ^ measured)} differences)"}
    vals = {}
    for (e, p), dt in table.items():
        r = pe[e][p]
        if not _finite(r.get("median_px"), r.get("se_px")):
            return {"ok": False, "problem": f"{e} {p}: non-finite measurement"}
        vals.setdefault(p, {"short": [], "long": []})["long" if dt > RELEASE1_MAX_DT_MS else "short"].append(
            (e, float(r["median_px"]), float(r["se_px"])))
    pooled_short = [m for v in vals.values() for _, m, _ in v["short"]]
    out = {"pairs": {}, "max_diff_px": G5B_DT_MAX_DIFF_PX, "max_p90_diff_px": G5B_DT_MAX_P90_DIFF_PX, "problems": []}
    for p, v in sorted(vals.items()):
        if not v["long"]:
            continue
        short = [m for _, m, _ in v["short"]] if len(v["short"]) >= 10 else pooled_short
        if len(short) < 10:
            out["problems"].append(f"{p}: fewer than 10 short-offset measurements to compare with")
            continue
        long = [m for _, m, _ in v["long"]]
        base = float(np.median([m for _, m, _ in v["short"] + v["long"]]))
        viol = [e for e, m, s_ in v["long"] if _violates(m, s_, base)]
        row = {"n_short": len(short), "n_long": len(long), "short_from": "pair" if short is not pooled_short else "pooled",
               "median_diff_px": float(np.median(long) - np.median(short)),
               "p90_diff_px": float(np.percentile(long, 90) - np.percentile(short, 90)), "long_violations": viol}
        out["pairs"][p] = row
        if row["median_diff_px"] > G5B_DT_MAX_DIFF_PX:
            out["problems"].append(f"{p}: long-offset median {row['median_diff_px']:+.3f} px > {G5B_DT_MAX_DIFF_PX}")
        if row["p90_diff_px"] > G5B_DT_MAX_P90_DIFF_PX:
            out["problems"].append(f"{p}: long-offset p90 {row['p90_diff_px']:+.3f} px > {G5B_DT_MAX_P90_DIFF_PX}")
        if viol:
            out["problems"].append(f"{p}: {len(viol)} long-offset episode-pairs violate the G5b rule {viol[:4]}")
    out["n_long"] = sum(len(v["long"]) for v in vals.values())
    out["ok"] = not out["problems"]
    out["problem"] = "; ".join(out["problems"]) or None
    return out


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
    """DECISION SUPPORT ONLY (no gate, no eligibility): what FIXREG says about the size of a G5a-only camera's motion.
    Per document episode: pooled D_B, SE_D_B, the 99 % bound corrected by the camera's median and by its lowest
    observed injection recovery ratio, the largest within-episode group displacement; plus FIXREG's verdict, events,
    unresolved candidates and missing within-episode estimates, counted against 1/10 and 1/4 training px."""
    out = {"role": "decision support only (Codex review 2: coverage of these bounds is not validated)",
           "flagged": list(flagged), "budgets_px": list(G5A_BUDGETS_PX), "z": G5A_Z, "problems": []}
    if not isinstance(fixreg, dict):
        return {**out, "problems": ["no FIXREG evidence"]}
    label = fixreg_series(fixreg, cam, task)
    s = (fixreg.get("series") or {}).get(label) if label else None
    out["series"] = label
    if not isinstance(s, dict):
        return {**out, "problems": [f"no unique FIXREG series of {cam} for task {task}"]}
    verdict = s.get("verdict") if isinstance(s.get("verdict"), dict) else {}
    out.update({"fixreg_verdict": verdict.get("verdict"), "events": verdict.get("events"),
                "candidates_not_verifiable": verdict.get("candidates_not_verifiable"),
                "coherent_significant_inside_envelope": verdict.get("coherent_significant_inside_envelope"),
                "fixreg_reasons": verdict.get("reasons")})
    ratios = {}
    for row in (s.get("sensitivity") or {}).get("rows") or []:
        if row.get("kind") == "zero" or not row.get("verifiable"):
            continue
        if _finite(row.get("recovered_D"), row.get("truth_D")) and float(row["truth_D"]) > 0:
            ratios.setdefault(f"{row['kind']}{row.get('val')}", []).append(float(row["recovered_D"]) / float(row["truth_D"]))
    r_med = min((float(np.median(v)) for v in ratios.values()), default=None)
    r_min = min((min(v) for v in ratios.values()), default=None)
    out["recovery"] = {"min_of_medians": r_med, "min_observed": r_min,
                       "per_injection_median": {k: float(np.median(v)) for k, v in ratios.items()}}
    by_ep = {e.get("episode"): e for e in s.get("episodes") or []}
    rows, no_within = {}, []
    for ep in episodes:
        e = by_ep.get(ep)
        if e is None or not e.get("verifiable_B") or not _finite(e.get("D_B"), e.get("se_D_B")):
            out["problems"].append(f"{ep}: no verifiable finite FIXREG estimate")
            continue
        up = float(e["D_B"]) + G5A_Z * float(e["se_D_B"])
        groups = [float(g) for g in (e.get("group_D") or []) if _finite(g)]
        if not isinstance(e.get("within"), dict):
            no_within.append(ep)
        rows[ep] = {"D_px": float(e["D_B"]), "se_D_px": float(e["se_D_B"]), "upper_px": up,
                    "upper_corr_median_px": up / r_med if r_med else None,
                    "upper_corr_min_px": up / r_min if r_min else None,
                    "group_max_px": max(groups) if groups else None}
    out["episodes"], out["no_within_episode_estimate"] = rows, no_within

    def worst(k):
        xs = [(v[k], ep) for ep, v in rows.items() if v.get(k) is not None]
        return max(xs) if xs else None
    out["worst"] = {k: worst(k) for k in ("upper_px", "upper_corr_median_px", "upper_corr_min_px", "group_max_px")}
    out["over_budget"] = {f"{b:.4f}": {k: sum(1 for v in rows.values() if v.get(k) is not None and v[k] > b)
                                       for k in ("upper_px", "upper_corr_median_px", "upper_corr_min_px")}
                          for b in G5A_BUDGETS_PX}
    out["flagged_rows"] = {ep: rows.get(ep) for ep in flagged}
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


def _episode_stamps(task_root, episode: str) -> dict:
    """Per-camera colour timestamps (ns) of a processed episode - exactly what the RoMa worker paired on."""
    with np.load(Path(task_root) / episode / "episodes" / episode / "normalized" / "arrays.npz") as z:
        return {c: z[f"timestamps/{c}"].astype(np.int64) for c in CAMS}


def _check_rows(p, z, plan_rows: list) -> set:
    """Every (time, pair) group of match file ``p`` must be ONE planned combination: its frame_a / frame_b / dt_ns
    equal the plan's (recomputed from the timestamps). Returns the set of (time, pair code) present."""
    planned = {(int(k), int(pa) * 10 + int(pb)): (int(ia), int(ib), int(dt)) for k, pa, pb, ia, ib, dt in plan_rows}
    keys = set()
    if not len(z["pair"]):
        return keys
    t, q = z["time"].astype(int), z["pair"].astype(int)
    for key in sorted(set(zip(t.tolist(), q.tolist()))):
        sel = (t == key[0]) & (q == key[1])
        got = {(int(a), int(b), int(d)) for a, b, d in zip(z["frame_a"][sel], z["frame_b"][sel], z["dt_ns"][sel])}
        if key not in planned or got != {planned[key]}:
            raise ValueError(f"{p}: rows of (time {key[0]}, pair {key[1]}) {sorted(got)[:2]} are not the planned "
                             f"combination {planned.get(key)} recomputed from the timestamps")
        keys.add(key)
    return keys


def match_inputs(match_dirs: list, episodes: list, task_root=None) -> tuple[dict, dict]:
    """({episode: [release/1 file, supplement files...]}, manifest) after validating the inputs: per episode exactly
    one release/1 (reference-pairing) file first and at most pairwise supplements whose ``supplement_of`` names it by
    sha256; no (time, pair) combination in two files. With ``task_root`` (required by ``static_eval``) every file is
    also checked against the episode's own timestamps: camera order, times, the release/1 frames (reference rule at its
    max_dt_ms) and the supplement plan (pairwise rule, half the measured frame period - never a self-declared
    tolerance), frame by frame. Raises ValueError otherwise. The manifest hashes every file."""
    from s4d.realworld.workers import romav2_match_worker as W

    files, manifest = {}, {}
    for e in episodes:
        paths = [Path(d) / f"{e}.npz" for d in match_dirs if (Path(d) / f"{e}.npz").is_file()]
        if not paths or paths[0].parent != Path(match_dirs[0]):
            raise ValueError(f"{e}: no release/1 match file in {match_dirs[0]}")
        stamps = _episode_stamps(task_root, e) if task_root is not None else None
        entries, seen, meta0 = [], set(), None
        for i, p in enumerate(paths):
            sha = _sha256(p)
            with np.load(p) as z:
                meta = json.loads(str(z["meta"]))
                dts = {(int(t), int(q)): int(d) for t, q, d in zip(z["time"].astype(int), z["pair"].astype(int),
                                                                   z["dt_ns"].astype(np.int64))} if len(z["pair"]) else {}
                if meta.get("episode_id") != e:
                    raise ValueError(f"{p}: episode_id {meta.get('episode_id')!r} != {e}")
                if list(meta.get("cameras") or []) != list(CAMS):
                    raise ValueError(f"{p}: camera order {meta.get('cameras')} != {list(CAMS)}")
                pairing = meta.get("pairing", "reference")
                if i == 0:
                    if pairing != "reference":
                        raise ValueError(f"{p}: the first match file must be the release/1 (reference-pairing) file")
                    meta0 = meta
                    tol = float(meta.get("max_dt_ms", RELEASE1_MAX_DT_MS)) * 1e6
                    if any(abs(d) > tol for d in dts.values()):
                        raise ValueError(f"{p}: rows beyond its own max_dt_ms")
                    if stamps is not None:
                        plan = W.plan_pairs(stamps, list(CAMS), int(meta["times"]), "reference", meta["max_dt_ms"])
                        if {int(k): v for k, v in meta["frames"].items()} != plan["frames_used"]:
                            raise ValueError(f"{p}: meta frames differ from the reference rule on the timestamps")
                        keys = _check_rows(p, z, plan["pairs"])
                    else:
                        keys = set(dts)
                else:
                    sup = meta.get("supplement_of") or {}
                    if pairing != "pairwise" or sup.get("sha256") != entries[0]["sha256"]:
                        raise ValueError(f"{p}: not a pairwise supplement of {paths[0]} (supplement_of sha256 mismatch)")
                    planned = [list(r) for r in meta.get("pair_frames") or []]
                    if stamps is not None:
                        plan = W.supplement_plan(stamps, list(CAMS), int(meta0["times"]), meta0)
                        if sorted(map(tuple, planned)) != sorted(map(tuple, plan["pairs"])):
                            raise ValueError(f"{p}: meta pair_frames differ from the pairwise supplement plan")
                        keys = _check_rows(p, z, plan["pairs"])
                    else:
                        tol = meta.get("max_dt_ns") or {}
                        byk = {(int(r[0]), int(r[1]) * 10 + int(r[2])): int(r[5]) for r in planned}
                        for (k, q), d in dts.items():
                            if byk.get((k, q)) != d or abs(d) > int(tol.get(f"{q // 10}{q % 10}", -1)):
                                raise ValueError(f"{p}: row (time {k}, pair {q}) not in its plan or beyond its tolerance")
                        keys = set(dts)
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


def static_eval(full_path, corr_dir, evidence_dir, match_dirs: list, task_root, *, per_frame_pair: int = 200) -> dict:
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
    union, manifest = match_inputs(match_dirs, episodes, task_root)
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
            "dt": {"rows": _dt_rows(union)}, "inputs": manifest, "task_root": None if task_root is None else str(task_root),
            "frames_validated": task_root is not None}


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
    """Refuse a static evaluation that is not a complete, timestamp-validated release/2 evaluation of this full.json,
    or that does not reproduce release/1 (recomputed here from the embedded measurements, never a cached flag)."""
    if se.get("format") != "s4d-calib-static-eval/2" or se.get("frames_validated") is not True:
        raise SystemExit("not a timestamp-validated s4d-calib-static-eval/2 evaluation")
    if (se.get("full") or {}).get("sha256") != _sha256(full_path):
        raise SystemExit("the static evaluation was computed on another full.json")
    full = json.loads(Path(full_path).read_text())
    episodes = sorted(set(full["fit_episodes"]) | set(full["heldout_episodes"]))
    inputs = se.get("inputs") or {}
    if sorted(inputs) != episodes or not all(v and v[0].get("pairing") == "reference" and all(
            x.get("sha256") for x in v) for v in inputs.values()):
        raise SystemExit("the static evaluation's input manifest does not cover every episode with hashed files")
    if not all(((se.get("K_sources") or {}).get(k) or {}).get("sha256") for k in ("per_episode", "heldout")) or \
            not se.get("code"):
        raise SystemExit("the static evaluation lacks K-source / code identity")
    probs = check_reproduction(se, full)
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
        g5a = g5a_materiality(fixreg, cam, flagged, task, episodes) if "G5a" in held_by else None
        checks = {g: not any(r.startswith(g + ":") for r in reasons) for g in ("G1", "G2", "G3", "G4", "G5a", "G5b",
                                                                              "G6")}
        out[cam] = {"status": "validated" if not reasons else "held", "reasons": reasons, "held_by": held_by,
                    "checks": checks, "exception_eligible": held_by == ["G5a"],
                    "g1": {k: v for k, v in g1.items() if k != "per_solve"}, "g1_per_solve": g1.get("per_solve"),
                    "g4": g4, "g5b": {k: v for k, v in g5b.items() if k != "epipolar"} | {
                        "moved": g5b["epipolar"]["moved"], "single_pair_violations":
                        g5b["epipolar"]["single_pair_violations"], "baseline_px": g5b["epipolar"]["baseline_px"],
                        "not_verifiable": g5b["epipolar"]["not_verifiable"], "dt_extension": dtc},
                    "g5a_decision_support": g5a,
                    "release1": {"status": r1["status"], "reasons": r1["reasons"], "held_by": r1["held_by"]}}
    return out


def verdict_document(R, fixed, se, *, task, fixreg=None, sources=None) -> dict:
    v = release_verdict_v2(R, fixed, se, task=task, fixreg=fixreg)
    return {"format": "s4d-calib-release-verdict/2", "criteria": V2, "gate_version": GATE_VERSION_V2, "task": task,
            "thresholds": {"g1": {k: (list(x) if isinstance(x, tuple) else x) for k, x in G1_V2.items()},
                           "g5a": "release/1 (decision-support levels " + str(list(G5A_BUDGETS_PX)) + " px, not gates)",
                           "g5b_min_verifiable": G5B_MIN_VERIFIABLE, "g5b_min_matches": G5B_MIN_MATCHES,
                           "g5b_dt_max_diff_px": G5B_DT_MAX_DIFF_PX, "g5b_dt_max_p90_diff_px": G5B_DT_MAX_P90_DIFF_PX,
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
    s.add_argument("--task-root", required=True, help="the episodes' processed roots (normalized timestamps)")
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
        se = static_eval(a.full, a.corr, a.evidence, a.matches, a.task_root, per_frame_pair=a.per_frame_pair)
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
                          "exception_eligible": d["exception_eligible"]} for c, d in doc["cameras"].items()},
                     indent=1))


if __name__ == "__main__":
    main()
