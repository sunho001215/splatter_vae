"""Publication of a ``rig_dw`` per-task rig calibration as a ``calibration_rig`` document (format
``s4d.realworld.rig_calibration/1``, consumed by the calibrate@2 rig mode of ``calibration_stage``), plus the per-camera
acceptance verdict of DESIGN v1 (gates G1-G6). The verdict is computed from the stored results; nothing is refit here.

    python -m s4d.realworld.rig_dw_publish --task T --results DIR --fixedness F.json --reference F.reference.npz \
        --final all.json --out rig-dw-<task>.json [--commit SHA]
"""

from __future__ import annotations

import argparse
import json
from itertools import combinations
from pathlib import Path

import numpy as np

from s4d.realworld.calib.dw.common import CAMS
from s4d.realworld.calib.dw.geom import pose_distance

GATES = {"spread_rot_deg": 0.35, "spread_centre_m": 0.008, "depth_median_mm": 12.0, "explained": 0.70,
         "contour_px": 2.0, "o2m_fixed_px": 2.0, "epi_line_px": 1.0, "transfer_px": 3.75}


def _cost(r):
    c = [l["cost"] for l in r["log"] if "cost" in l]
    return c[-1] if c else None


EXPECTED = {"multistart": [f"start-{k}" for k in range(1, 9)], "halves": [f"half-{k}" for k in range(5)],
            "groups": ["group-0", "group-1"], "blocks": ["block-0", "block-1"],
            "loco": ["loco-135122070988", "loco-135122073201", "loco-135122076054"]}


def near_best_starts(R: dict, ref: str = "full", rel_tol: float = 1e-3) -> list:
    """Multistart solutions whose fit cost, re-scored on the COMMON support of the reference solution
    (``cost_on_ref_support``), is within ``rel_tol`` of the best (the reference included)."""
    costs = {ref: None}
    for n in EXPECTED["multistart"]:
        c = (R.get(n) or {}).get("cost_on_ref_support")
        if c:
            costs[n] = c["solution"]
            costs[ref] = c["reference"] if costs[ref] is None else min(costs[ref], c["reference"])
    valid = {k: v for k, v in costs.items() if v is not None and np.isfinite(v)}
    if not valid:
        return [ref]
    best = min(valid.values())
    return [k for k, v in valid.items() if v <= best * (1 + rel_tol)]


def spread_sets(R: dict, ref: str = "full") -> dict:
    return {"multistart_near_best": near_best_starts(R, ref),
            "halves": [n for n in EXPECTED["halves"] if n in R], "groups": [n for n in EXPECTED["groups"] if n in R],
            "blocks": [n for n in EXPECTED["blocks"] if n in R],
            "loco": [n for n in EXPECTED["loco"] if n in R] + [ref]}


def max_pair(R, names, cam):
    Ts = [np.asarray(R[n]["state"]["T_cam_base"][cam]) for n in names if n in R]
    if len(Ts) < 2:
        return None
    d = [pose_distance(a, b) for a, b in combinations(Ts, 2)]
    return {"solves": len(Ts), "max_pair_rot_deg": float(max(x[0] for x in d)),
            "max_pair_centre_m": float(max(x[1] for x in d))}


def fixedness_registration(fixed: dict, cam: str) -> dict:
    """G5(a): whole-episode-median registration (1 segment). Detected motion = a registration that is VERIFIABLE
    (>= 6 and >= 60 % consistent patches) and exceeds a threshold; otherwise not verifiable."""
    moved, unver, n = [], [], 0
    for ep, v in (fixed.get("episodes") or {}).items():
        if cam not in v:
            unver.append(ep)
            continue
        n += 1
        for sgm in v[cam].get("segments", []):
            tot = sgm.get("patches", 0)
            inl = sgm.get("inlier_patches", 0)
            verifiable = "shift_px" in sgm and inl >= 6 and inl >= 0.6 * max(1, tot)
            if not verifiable:
                unver.append(ep)
            elif not sgm.get("fixed", False):
                moved.append(ep)
    ref = fixed.get("reference_episode")
    ref_ok = ref in (fixed.get("episodes") or {}) and cam in fixed["episodes"][ref] and \
        all(sg.get("fixed") for sg in fixed["episodes"][ref][cam].get("segments", []))
    return {"episodes": n, "moved": sorted(set(moved)), "not_verifiable": sorted(set(unver)),
            "reference_self_registration_ok": bool(ref_ok)}


def fixedness_epipolar(per_ep: dict, cam: str, max_px: float = 1.0, max_excess_px: float = 0.25) -> dict:
    """G5(b) (pre-registered 2026-11-07 before inspecting values): per episode and camera pair, the median symmetric
    epipolar line distance under the shared rig must be <= ``max_px`` and its excess over the pair's baseline (median
    over all episodes) plus 2.58 standard errors <= ``max_excess_px``. A camera is flagged as moved in an episode when
    BOTH pairs containing it violate; one violating pair is reported. Missing pairs: not verifiable."""
    key = cam[:12]
    pairs = sorted({p for v in per_ep.values() for p in v if key in p})
    base = {p: float(np.median([v[p]["median_px"] for v in per_ep.values() if p in v])) for p in pairs}
    moved, single, unver = [], [], []
    for ep, v in per_ep.items():
        bad = []
        for p in pairs:
            if p not in v:
                unver.append(ep)
                continue
            m, se = v[p]["median_px"], v[p]["se_px"]
            if m > max_px or (m - base[p]) + 2.58 * se > max_excess_px:
                bad.append(p)
        if len(bad) == len(pairs) and pairs:
            moved.append(ep)
        elif bad:
            single.append((ep, bad))
    return {"episodes": len(per_ep), "baseline_px": base, "moved": moved, "single_pair_violations": single,
            "not_verifiable": sorted(set(unver))}


def verdict(R: dict, fixed: dict, ref: str = "full") -> dict:
    """Per-camera gate evaluation (DESIGN v1 + the G5 amendment); incomplete evidence never passes."""
    full = R[ref]
    h = full["heldout"]
    sets = spread_sets(R, ref)
    missing = [n for k, v in EXPECTED.items() for n in v if n not in R]
    allnames = sorted(set(sum(sets.values(), [])))
    out = {}
    e = h["extext"]["all"]
    g4 = {"line_median_px": e["line_median_px"], "transfer_ab_median_px": e["transfer_ab_median_px"],
          "transfer_ba_median_px": e["transfer_ba_median_px"]}
    g4_ok = all(np.isfinite(x) for x in g4.values()) and e["line_median_px"] <= GATES["epi_line_px"] and \
        e["transfer_ab_median_px"] <= GATES["transfer_px"] and e["transfer_ba_median_px"] <= GATES["transfer_px"]
    cc = h.get("common_challenge") or {}
    per_ep = full.get("per_episode_extext") or {}
    for cam in CAMS:
        reasons = []
        sp = {k: max_pair(R, v, cam) for k, v in sets.items()}
        allsp = max_pair(R, allnames, cam)
        if missing:
            reasons.append(f"G1: stability evidence incomplete, missing {missing}")
        if allsp is None:
            reasons.append("G1: no stability solves")
        else:
            if allsp["max_pair_rot_deg"] > GATES["spread_rot_deg"]:
                reasons.append(f"G1: rotation spread {allsp['max_pair_rot_deg']:.3f} deg > {GATES['spread_rot_deg']}")
            if allsp["max_pair_centre_m"] > GATES["spread_centre_m"]:
                reasons.append(f"G1: centre spread {1e3 * allsp['max_pair_centre_m']:.1f} mm > "
                               f"{1e3 * GATES['spread_centre_m']:.0f}")
        ra, rr, rb = h["robot_alpha_depth"][cam], h["robot_raw_depth"][cam], h["robot"][cam]
        dm, ex = ra.get("depth_median_mm"), ra.get("explained_fraction")
        if dm is None or not np.isfinite(dm) or dm > GATES["depth_median_mm"]:
            reasons.append(f"G2: held-out robot depth median {dm} mm > {GATES['depth_median_mm']}")
        if ex is None or not np.isfinite(ex) or ex < GATES["explained"]:
            reasons.append(f"G2: explained {ex} < {GATES['explained']}")
        if not np.isfinite(rb["contour_median_px"]) or rb["contour_median_px"] > GATES["contour_px"]:
            reasons.append(f"G3: contour median {rb['contour_median_px']:.3f} px > {GATES['contour_px']}")
        o2m = ra.get("o2m_fixed_median_px")
        if o2m is None or not np.isfinite(o2m) or o2m > GATES["o2m_fixed_px"]:
            reasons.append(f"G3: fixed o2m median {o2m} px > {GATES['o2m_fixed_px']}")
        if not g4_ok:
            reasons.append(f"G4: static correspondences {g4}")
        f_reg = fixedness_registration(fixed, cam)
        f_epi = fixedness_epipolar(per_ep, cam) if per_ep else {"moved": [], "not_verifiable": ["all"]}
        if f_reg["moved"]:
            reasons.append(f"G5a: registration detects motion in {len(f_reg['moved'])} episodes {f_reg['moved'][:6]}")
        if f_epi["moved"]:
            reasons.append(f"G5b: epipolar consistency flags {len(f_epi['moved'])} episodes {f_epi['moved'][:6]}")
        if not per_ep or len(per_ep) < len(full.get("fit_episodes", [])) + len(full.get("heldout_episodes", [])):
            reasons.append("G5b: per-episode consistency incomplete")
        entries = [k for k, v in cc.items() if isinstance(v, dict)]
        if len(entries) != 12:
            reasons.append("G6: common-transform challenge missing or incomplete")
        elif not cc.get("all_rejected"):
            bad = [k for k in entries if not cc[k]["rejected"]]
            reasons.append(f"G6: held-out robot evidence does not reject common perturbations {bad}")
        alpha = full["state"]["alpha"][cam]
        out[cam] = {"status": "validated" if not reasons else "held", "reasons": reasons,
                    "empirical_spread": {**(allsp or {}), "sets": sp},
                    "heldout": {"robot_alpha_depth": ra, "robot_raw_depth": rr, "silhouette": rb, "static": g4},
                    "fixedness": {"registration": f_reg, "epipolar": f_epi},
                    "alpha": alpha, "raw_depth_flag": abs(alpha) > 0.01}
    return out


def main(argv=None):
    from s4d.realworld.calib.rig import FORMAT, publish_rig

    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True)
    ap.add_argument("--results", required=True, help="directory of rig_dw_run results of the task (incl. full.json)")
    ap.add_argument("--final", required=True, help="the result whose poses are published (e.g. all.json)")
    ap.add_argument("--fixedness", required=True)
    ap.add_argument("--reference", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--commit", default="")
    ap.add_argument("--sensor-result", default=None, help="depth:sensor result for the sensor depth scale")
    a = ap.parse_args(argv)
    R = {p.stem: json.loads(p.read_text()) for p in Path(a.results).glob("*.json")}
    fixed = json.loads(Path(a.fixedness).read_text())
    final = json.loads(Path(a.final).read_text())
    ver = verdict(R, fixed)
    sensor = json.loads(Path(a.sensor_result).read_text()) if a.sensor_result else None
    name = f"rig-dw-{a.task}-{a.commit[:12]}" if a.commit else f"rig-dw-{a.task}"
    doc = {"format": FORMAT, "name": name, "method": "rig_dw: board-free per-task joint 3-camera rig calibration "
           "(RoMa v2 static RGB-D correspondences + robot silhouettes/depth with visual meshes)",
           "task": a.task, "reference_episode": fixed["reference_episode"],
           "episodes": sorted(final["fit_episodes"]), "pool_frames": final.get("fit_frames", {}),
           "provenance": {"commit": a.commit, "final_variant": final["variant"], "config": final["config"],
                          "surface": final.get("surface"), "gates": GATES}, "cameras": {}}
    for cam in CAMS:
        v = ver[cam]
        ds = {"foundation_stereo_depth": float(np.exp(final["state"]["alpha"][cam]))}
        if sensor:
            ds["sensor_depth"] = float(np.exp(sensor["state"]["alpha"][cam]))
        doc["cameras"][cam] = {"status": v["status"], "reasons": v["reasons"],
                               "T_cam_base": final["state"]["T_cam_base"][cam], "depth_scale": ds,
                               "tau_s": final["state"]["tau_s"][cam], "empirical_spread": v["empirical_spread"],
                               "validation": {k: v[k] for k in ("heldout", "fixedness", "alpha", "raw_depth_flag")},
                               "final_vs_evaluated_fit": {"rot_deg": float(pose_distance(
                                   np.asarray(final["state"]["T_cam_base"][cam]),
                                   np.asarray(R["full"]["state"]["T_cam_base"][cam]))[0]), "centre_m": float(pose_distance(
                                   np.asarray(final["state"]["T_cam_base"][cam]),
                                   np.asarray(R["full"]["state"]["T_cam_base"][cam]))[1])},
                               "provenance": {"rig": name}}
    ref = np.load(a.reference)
    publish_rig(a.out, doc, {k: ref[k] for k in ref.files})
    print(json.dumps({cam: {"status": doc["cameras"][cam]["status"], "reasons": doc["cameras"][cam]["reasons"]}
                      for cam in CAMS}, indent=1))


if __name__ == "__main__":
    main()
