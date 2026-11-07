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

from s4d.realworld.rig_dw_geom import pose_distance

CAMS = ("135122070988/color", "135122073201/color", "135122076054/color")
GATES = {"spread_rot_deg": 0.35, "spread_centre_m": 0.008, "depth_median_mm": 12.0, "explained": 0.70,
         "contour_px": 2.0, "o2m_fixed_px": 2.0, "epi_line_px": 1.0, "transfer_px": 3.75}


def _cost(r):
    c = [l["cost"] for l in r["log"] if "cost" in l]
    return c[-1] if c else None


def spread_sets(R: dict, ref: str = "full") -> dict:
    starts = [n for n in R if n.startswith("start-") or n == ref]
    costs = {n: _cost(R[n]) for n in starts}
    best = min(c for c in costs.values() if c is not None)
    return {"multistart_near_best": [n for n in starts if costs[n] is not None and costs[n] <= best * 1.001],
            "halves": [n for n in R if n.startswith("half-")], "groups": [n for n in R if n.startswith("group-")],
            "blocks": [n for n in R if n.startswith("block-")],
            "loco": [n for n in R if n.startswith("loco-")] + [ref]}


def max_pair(R, names, cam):
    Ts = [np.asarray(R[n]["state"]["T_cam_base"][cam]) for n in names if n in R]
    if len(Ts) < 2:
        return None
    d = [pose_distance(a, b) for a, b in combinations(Ts, 2)]
    return {"solves": len(Ts), "max_pair_rot_deg": float(max(x[0] for x in d)),
            "max_pair_centre_m": float(max(x[1] for x in d))}


def verdict(R: dict, fixedness: dict, ref: str = "full") -> dict:
    """Per-camera gate evaluation (DESIGN v1)."""
    full = R[ref]
    h = full["heldout"]
    sets = spread_sets(R, ref)
    allnames = sorted(set(sum(sets.values(), [])))
    out = {}
    e = h["extext"]["all"]
    g4 = {"line_median_px": e["line_median_px"], "transfer_ab_median_px": e["transfer_ab_median_px"],
          "transfer_ba_median_px": e["transfer_ba_median_px"]}
    g4_ok = (e["line_median_px"] <= GATES["epi_line_px"] and e["transfer_ab_median_px"] <= GATES["transfer_px"]
             and e["transfer_ba_median_px"] <= GATES["transfer_px"])
    cc = h.get("common_challenge") or {}
    for cam in CAMS:
        reasons = []
        sp = {k: max_pair(R, v, cam) for k, v in sets.items()}
        allsp = max_pair(R, allnames, cam)
        if allsp is None:
            reasons.append("G1: no stability solves")
        else:
            if allsp["max_pair_rot_deg"] > GATES["spread_rot_deg"]:
                reasons.append(f"G1: rotation spread {allsp['max_pair_rot_deg']:.3f} deg > {GATES['spread_rot_deg']}")
            if allsp["max_pair_centre_m"] > GATES["spread_centre_m"]:
                reasons.append(f"G1: centre spread {1e3 * allsp['max_pair_centre_m']:.1f} mm > "
                               f"{1e3 * GATES['spread_centre_m']:.0f}")
        ra, rr, rb = h["robot_alpha_depth"][cam], h["robot_raw_depth"][cam], h["robot"][cam]
        if ra["depth_median_mm"] is None or ra["depth_median_mm"] > GATES["depth_median_mm"]:
            reasons.append(f"G2: held-out robot depth median {ra['depth_median_mm']} mm > {GATES['depth_median_mm']}")
        if ra["explained_fraction"] is None or ra["explained_fraction"] < GATES["explained"]:
            reasons.append(f"G2: explained {ra['explained_fraction']} < {GATES['explained']}")
        if rb["contour_median_px"] > GATES["contour_px"]:
            reasons.append(f"G3: contour median {rb['contour_median_px']:.2f} px > {GATES['contour_px']}")
        if ra.get("o2m_fixed_median_px") is None or ra["o2m_fixed_median_px"] > GATES["o2m_fixed_px"]:
            reasons.append(f"G3: fixed o2m median {ra.get('o2m_fixed_median_px')} px > {GATES['o2m_fixed_px']}")
        if not g4_ok:
            reasons.append(f"G4: static correspondences {g4}")
        moved = [ep for ep, v in (fixedness.get("episodes") or {}).items()
                 if cam in v and not v[cam]["fixed"] and _detected_motion(v[cam])]
        unverifiable = [ep for ep, v in (fixedness.get("episodes") or {}).items()
                        if cam in v and not v[cam]["fixed"] and not _detected_motion(v[cam])]
        if moved:
            reasons.append(f"G5: motion detected in {len(moved)} episodes: {moved[:8]}")
        if not cc:
            reasons.append("G6: common-transform challenge missing")
        elif not cc.get("all_rejected"):
            bad = [k for k, v in cc.items() if isinstance(v, dict) and not v["rejected"]]
            reasons.append(f"G6: held-out robot evidence does not reject common perturbations {bad}")
        alpha = full["state"]["alpha"][cam]
        out[cam] = {"status": "validated" if not reasons else "held", "reasons": reasons,
                    "empirical_spread": {**(allsp or {}), "sets": sp},
                    "heldout": {"robot_alpha_depth": ra, "robot_raw_depth": rr, "silhouette": rb, "static": g4},
                    "fixedness": {"moved": moved, "not_verifiable": unverifiable},
                    "alpha": alpha, "raw_depth_flag": abs(alpha) > 0.01}
    return out


def _detected_motion(m: dict) -> bool:
    """A segment failed because of a measured shift/rotation/scale (not merely too few registrable patches)."""
    for s in m.get("segments", []):
        reason = s.get("reason", "") or ""
        if s.get("fixed") is False and any(k in reason for k in ("image shift", "in-plane rotation", "scale")):
            return True
    return False


def main(argv=None):
    from s4d.realworld.calibration_rig import FORMAT, publish_rig

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
                               "provenance": {"rig": name}}
    ref = np.load(a.reference)
    publish_rig(a.out, doc, {k: ref[k] for k in ref.files})
    print(json.dumps({cam: {"status": doc["cameras"][cam]["status"], "reasons": doc["cameras"][cam]["reasons"]}
                      for cam in CAMS}, indent=1))


if __name__ == "__main__":
    main()
