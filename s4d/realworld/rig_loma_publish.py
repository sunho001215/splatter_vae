"""Release publication of an accepted ``rig_loma`` per-task rig calibration (calibration release 2026-11-08).

Writes a ``calibration_rig`` document (format ``s4d.realworld.rig_calibration/1``, consumed by the calibrate rig mode of
``calibration_stage``) from the frozen held-out batch (commit a935aef; FREEZE.md): poses of the ``all`` refit on all
100 episodes, the per-camera verdict recomputed from the stored results under the RELEASE gate, nothing refit.

Release gate (``GATE_VERSION``; user decision update 2026-11-08 ~22:00 KST): G1-G5 exactly as the frozen verdict
(rig_dw DESIGN v1.1 incl. the G5 amendment, thresholds unchanged; ``rig_dw_publish.verdict`` on the SAM validation);
G6 = the SAM-free common-perturbation challenge ``corr_challenge`` of frozen commit a935aef (12 rig perturbations of the
G6 tolerance size scored on held-out LoMa-G robot 2D-3D correspondences, each rejected by frame AND episode one-sided
sign tests p < 0.01 with a frame majority) is PRIMARY; the SAM-based G6 is kept as a DIAGNOSTIC. Cameras held only by
the solution-independent registration-fixedness check G5a may be named ``--exception-eligible``: they stay "held"
(``held_by == ["G5a"]``) and the document's ``fixedness_exception`` stays EMPTY until the coordinator fills it from the
independent FIXREG evidence (``fill-exception``; writes a NEW document). The rig mode then accepts such a camera only
with a valid exception (``calibration_rig.rig_camera_status``) and still reports the registration metrics.

Also published per camera: ``depth_scale`` {depth stage: exp(alpha)} of the fs solution and of the independent sensor /
s2m2 solutions (their poses must agree with fs), the per-episode G5a / G5b evidence (``fixedness_evidence``), the robot
model the per-episode checks use (corrected Robotiq 2F-85 register map, visual meshes) and the per-episode check
thresholds (``episode_checks`` = RigConfig: robot depth 12 mm, explained 0.70). One fixed rig per task: no per-episode
pose refinement exists anywhere in the release path.

    python -m s4d.realworld.rig_loma_publish publish --task T --results fz/fs/T --final fz/fs/T/all.json \
        --fixedness fixedness-1seg-T.json --reference fixedness-1seg-T.reference.npz --verdict verdict/fs_T.json \
        --freeze FREEZE.md --commit <a935aef sha> [--depth-result sensor_depth=fz/sensor/T/all.json ...] \
        [--exception-eligible 135122070988/color ...] --out rig-loma-T.json
    python -m s4d.realworld.rig_loma_publish fill-exception --rig rig-loma-T.json --cameras C ... \
        --evidence FILE ... --decision accepted --decided-by NAME --reference TEXT --out rig-loma-T-fx.json
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import time
from pathlib import Path

import numpy as np

from s4d.realworld.rig_dw_geom import pose_distance
from s4d.realworld.rig_dw_publish import CAMS, GATES, fixedness_epipolar, fixedness_registration, verdict
from s4d.realworld.robot import GRIPPER_MAP_2F85

GATE_VERSION = ("rig_loma-release/1: G1-G5 = rig_dw DESIGN v1.1 (G5 amended), thresholds unchanged; "
                "G6 = SAM-free corr_challenge primary; SAM G6 diagnostic")
USER_DECISION = ("FREEZE.md 'USER decision update (2026-11-08 ~22:00 KST)': drawer-open G6 = SAM-free corr_challenge "
                 "(frozen a935aef, untuned) primary, SAM G6 diagnostic; mug-hanging 070988/073201 fixedness accepted "
                 "only as a documented exception after the independent FIXREG verification; one fixed rig per task, "
                 "no per-episode pose refinement")
CHALLENGE = tuple(f"{k}{s}" for k in ("rotx", "roty", "rotz", "transx", "transy", "transz") for s in "+-")
P_MAX = 0.01
POSE_AGREEMENT = (0.05, 0.001)  # deg, m: a depth source's scale is published only if its pose agrees with fs
EPISODE_CHECKS = {"max_heldout_depth_median_m": GATES["depth_median_mm"] / 1000.0,
                  "min_explained_fraction": GATES["explained"]}
ROBOT_MODEL = {"gripper_map": GRIPPER_MAP_2F85, "surface": "visual"}
EVIDENCE_FIELDS = ("path", "sha256")


def sha256_file(path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _ref(path) -> dict:
    return {"path": str(path), "sha256": sha256_file(path)}


def load_results(results_dir) -> dict:
    """{solve name: result} of one task's rig_loma results, with the SAM validation mapped onto rig_dw's held-out
    layout (exactly as the frozen verdict tool did: SAM checks are validation only)."""
    R = {p.stem: json.loads(p.read_text()) for p in sorted(Path(results_dir).glob("*.json"))}
    if "full" not in R:
        raise SystemExit(f"{results_dir}: no full.json")
    full = R["full"]
    hs = full.get("heldout_sam") or {}
    full["heldout"] = dict(full.get("heldout") or {})
    full["heldout"].update({"extext": hs.get("extext_sam_static"), "robot": hs.get("robot"),
                            "robot_alpha_depth": hs.get("robot_alpha_depth"),
                            "robot_raw_depth": hs.get("robot_raw_depth"),
                            "common_challenge": hs.get("common_challenge_sam")})
    return R


def g6_samfree(full: dict) -> dict:
    """The primary G6: frozen ``corr_challenge`` (held-out LoMa-G 2D-3D evidence). Every one of the 12 perturbations must
    be present and rejected; the stored ``rejected`` flags are re-derived from the stored counts / p-values and must
    agree (an inconsistent record never passes)."""
    cc = (full.get("heldout") or {}).get("corr_challenge") or {}
    entries, problems = {}, []
    for name in CHALLENGE:
        e = cc.get(name)
        if not isinstance(e, dict):
            problems.append(f"{name} missing")
            continue
        rej = (e["frames_worse"] > e["frames"] / 2 and e["p_value"] < P_MAX and e["p_value_episode"] < P_MAX)
        if bool(rej) != bool(e.get("rejected")):
            raise SystemExit(f"corr_challenge {name}: stored rejected={e.get('rejected')} disagrees with its counts")
        entries[name] = {k: e[k] for k in ("frames_worse", "frames", "episodes_worse", "episodes", "p_value",
                                           "p_value_episode", "rejected")}
        if not rej:
            problems.append(f"{name} not rejected")
    return {"all_rejected": not problems and cc.get("all_rejected") is True, "problems": problems,
            "entries": entries, "criterion": "frame AND episode one-sided sign tests p < 0.01 with a frame majority, "
                                             "12 common rig perturbations (0.35 deg / 8 mm about the workspace pivot)"}


def release_verdict(R: dict, fixed: dict, eligible=()) -> dict:
    """Per-camera release verdict: G1-G5 of ``rig_dw_publish.verdict`` (unchanged), G6 = SAM-free challenge (primary),
    SAM G6 reported as diagnostic. ``eligible``: cameras that may be held by G5a alone pending a fixedness exception."""
    base = verdict(R, fixed, "full")
    g6 = g6_samfree(R["full"])
    out = {}
    for cam in CAMS:
        b = base[cam]
        sam_g6 = [r for r in b["reasons"] if r.startswith("G6")]
        reasons = [r for r in b["reasons"] if not r.startswith("G6")]
        if not g6["all_rejected"]:
            reasons.append(f"G6: SAM-free corr_challenge does not reject {g6['problems']}")
        held_by = sorted({r.split(":", 1)[0] for r in reasons})
        if cam in eligible and held_by != ["G5a"]:
            raise SystemExit(f"{cam}: --exception-eligible but held by {held_by} (only a camera held by G5a alone "
                             "can be covered by a fixedness exception)")
        checks = {g: not any(r.startswith(g + ":") for r in reasons) for g in ("G1", "G2", "G3", "G4", "G5a", "G5b")}
        checks["G6"] = g6["all_rejected"]
        out[cam] = {"status": "validated" if not reasons else "held", "reasons": reasons, "held_by": held_by,
                    "checks": checks, "base": b,
                    "diagnostics": {"g6_sam": {"passed": not sam_g6, "reasons": sam_g6,
                                               "role": "diagnostic (SAM evidence; user decision 2026-11-08)"},
                                    "g6_samfree": g6}}
    return out


def check_against_frozen(rel: dict, frozen: dict) -> None:
    """The recomputed G1-G6 (SAM) verdict must equal the stored frozen verdict JSON exactly (same tooling)."""
    for cam in CAMS:
        a, b = rel[cam]["base"], (frozen.get("rig_dw_gates") or {}).get(cam) or {}
        if a["status"] != b.get("status") or a["reasons"] != b.get("reasons"):
            raise SystemExit(f"{cam}: recomputed verdict {a['status']} {a['reasons']} != frozen {b.get('status')} "
                             f"{b.get('reasons')}")
        fz = ((frozen.get("samfree") or {}).get(cam) or {}).get("corr_challenge_all_rejected")
        if fz is not rel[cam]["diagnostics"]["g6_samfree"]["all_rejected"]:
            raise SystemExit(f"{cam}: SAM-free challenge {rel[cam]['diagnostics']['g6_samfree']['all_rejected']} != "
                             f"frozen {fz}")


def fixedness_evidence(R: dict, fixed: dict) -> dict:
    """{camera: {episode: {"registration": fixed|moved|not_verifiable, "epipolar": ok|moved|single_pair|
    not_verifiable}}} - the per-episode G5a / G5b classification the verdict used (rig mode reads ``epipolar``)."""
    full = R["full"]
    per_ep = full.get("per_episode_extext") or {}
    episodes = sorted(set(full.get("fit_episodes") or []) | set(full.get("heldout_episodes") or []))
    out = {}
    for cam in CAMS:
        reg = fixedness_registration(fixed, cam)
        epi = fixedness_epipolar(per_ep, cam)
        single = {e for e, _ in epi["single_pair_violations"]}
        rows = {}
        for ep in episodes:
            r = ("moved" if ep in reg["moved"] else "not_verifiable" if ep in reg["not_verifiable"] or
                 ep not in (fixed.get("episodes") or {}) else "fixed")
            e = ("moved" if ep in epi["moved"] else "not_verifiable" if ep not in per_ep or ep in epi["not_verifiable"]
                 else "single_pair" if ep in single else "ok")
            rows[ep] = {"registration": r, "epipolar": e}
        out[cam] = rows
    return out


def depth_scales(final: dict, others: dict) -> tuple[dict, dict]:
    """({camera: {stage: exp(alpha)}}, {stage: pose agreement}) - fs from the published solution, other stages from
    their independent solves when each camera's pose agrees with fs within ``POSE_AGREEMENT``."""
    ds = {cam: {"foundation_stereo_depth": float(np.exp(final["state"]["alpha"][cam]))} for cam in CAMS}
    agree = {}
    for stage, res in others.items():
        rows = {}
        for cam in CAMS:
            rot, cen = pose_distance(np.asarray(final["state"]["T_cam_base"][cam]),
                                     np.asarray(res["state"]["T_cam_base"][cam]))
            rows[cam] = {"rot_deg": float(rot), "centre_m": float(cen)}
            if rot > POSE_AGREEMENT[0] or cen > POSE_AGREEMENT[1]:
                raise SystemExit(f"{stage}: {cam} pose differs from fs by {rot:.3f} deg / {cen * 1e3:.2f} mm; its depth "
                                 "scale would not belong to the published pose")
            ds[cam][stage] = float(np.exp(res["state"]["alpha"][cam]))
        agree[stage] = rows
    return ds, agree


def build_document(*, task, R, final, fixed, frozen, ref_files, depth_results, eligible, commit, sources) -> dict:
    from s4d.realworld.calibration_rig import FIXEDNESS_RULE_G5AB, FORMAT

    if final.get("task") != task or R["full"].get("task") != task:
        raise SystemExit(f"results are not of task {task}")
    if final.get("variant") != "all" or final.get("heldout_episodes"):
        raise SystemExit("the published poses must be the 'all' refit (every episode, no held-out)")
    rel = release_verdict(R, fixed, eligible)
    check_against_frozen(rel, frozen)
    ds, agree = depth_scales(final, depth_results)
    evidence = fixedness_evidence(R, fixed)
    if sorted(final["fit_episodes"]) != sorted(next(iter(evidence.values()))):
        raise SystemExit("the published solve and the verdict cover different episodes")
    name = f"rig-loma-{task}-{commit[:7]}"
    doc = {"format": FORMAT, "name": name, "task": task,
           "method": "rig_loma: SAM-free LoMa-G render-to-image robot 2D-3D correspondences + RoMa v2 static RGB-D "
                     "two-view factors, per-task joint 3-camera rig (FS depth, sigma_logd 3 %, per-task gripper mount "
                     "estimate), one fixed rig per task",
           "reference_episode": fixed["reference_episode"], "episodes": sorted(final["fit_episodes"]),
           "pool_frames": {},
           "pool_frames_note": "rig_loma fitted LoMa-G 2D-3D correspondences and static RoMa matches; it used no robot "
                               "depth term, so every frame's SAM-masked robot-depth check in rig mode is independent "
                               "of the fit (held-out evidence type)",
           "gate_version": GATE_VERSION, "gates": {**GATES, "g6_primary": "SAM-free corr_challenge (frozen a935aef)",
                                                   "g6_sam": "diagnostic"},
           "fixedness_rule": FIXEDNESS_RULE_G5AB, "fixedness_config": fixed["config"],
           "fixedness_evidence": evidence,
           "fixedness_exception": {"eligible_cameras": sorted(eligible), "cameras": [], "decision": None,
                                   "evidence": [], "decided_by": None, "decided_at": None, "reference": None,
                                   "procedure": "docs/realworld_calibration_release.md (fixedness exception): the "
                                                "coordinator fills cameras, evidence files (path + sha256) and the "
                                                "decision from the FIXREG stream with "
                                                "`rig_loma_publish fill-exception` (writes a new document)"},
           "episode_checks": EPISODE_CHECKS, "robot_model": ROBOT_MODEL,
           "provenance": {"frozen_commit": commit, "gate_version": GATE_VERSION, "user_decision": USER_DECISION,
                          "final_variant": final["variant"], "config": final["config"],
                          "depth_pose_agreement": agree, **sources},
           "cameras": {}}
    for cam in CAMS:
        v, b = rel[cam], rel[cam]["base"]
        doc["cameras"][cam] = {
            "status": v["status"], "reasons": v["reasons"], "held_by": v["held_by"], "checks": v["checks"],
            "T_cam_base": final["state"]["T_cam_base"][cam], "depth_scale": ds[cam], "tau_s": final["state"]["tau_s"][cam],
            "empirical_spread": b["empirical_spread"],
            "validation": {k: b[k] for k in ("heldout", "fixedness", "alpha", "raw_depth_flag")},
            "diagnostics": {**v["diagnostics"], "gripper_mount_estimate_dM_deg_mm": final.get("mount_dM"),
                            "gripper_mount_note": "estimated jointly with the poses; NOT applied downstream (nominal "
                                                  "mount everywhere, as in the frozen SAM validation)"},
            "final_vs_evaluated_fit": dict(zip(("rot_deg", "centre_m"), map(float, pose_distance(
                np.asarray(final["state"]["T_cam_base"][cam]), np.asarray(R["full"]["state"]["T_cam_base"][cam]))))),
            "provenance": {"rig": name}}
    for cam in CAMS:
        for k in ("grey", "dynamic"):
            if f"{cam.replace('/', '__')}__{k}" not in ref_files:
                raise SystemExit(f"reference arrays lack {cam} {k}")
    return doc


def publish(a) -> dict:
    from s4d.realworld.calibration_rig import publish_rig

    out = Path(a.out)
    if out.exists() or out.with_suffix(".reference.npz").exists():
        raise SystemExit(f"{out} exists: published documents are never replaced (choose a new path)")
    R = load_results(a.results)
    fixed = json.loads(Path(a.fixedness).read_text())
    final = json.loads(Path(a.final).read_text())
    frozen = json.loads(Path(a.verdict).read_text())
    depth_results = {}
    for item in a.depth_result or []:
        stage, path = item.split("=", 1)
        depth_results[stage] = json.loads(Path(path).read_text())
    ref = np.load(a.reference)
    ref_files = {k: ref[k] for k in ref.files}
    sources = {"final": _ref(a.final), "full": _ref(Path(a.results) / "full.json"), "verdict": _ref(a.verdict),
               "fixedness": _ref(a.fixedness), "reference_source": _ref(a.reference), "freeze_record": _ref(a.freeze),
               "depth_results": {s.split("=", 1)[0]: _ref(s.split("=", 1)[1]) for s in a.depth_result or []}}
    doc = build_document(task=a.task, R=R, final=final, fixed=fixed, frozen=frozen, ref_files=ref_files,
                         depth_results=depth_results, eligible=tuple(a.exception_eligible or ()), commit=a.commit,
                         sources=sources)
    return publish_rig(out, doc, ref_files)


def fill_exception(doc: dict, *, cameras, evidence, decision, decided_by, reference, decided_at=None) -> dict:
    """A copy of a release document with its ``fixedness_exception`` filled (the document is new: its sha256 changes,
    so every calibrate record built with it gets a new fingerprint). ``evidence``: [(path, sha256)]."""
    ex = doc.get("fixedness_exception")
    if not isinstance(ex, dict):
        raise SystemExit("not a release document (no fixedness_exception)")
    if ex.get("decision") is not None:
        raise SystemExit("the exception is already filled; start from the originally published document")
    if decision not in ("accepted", "rejected"):
        raise SystemExit("decision must be 'accepted' or 'rejected'")
    bad = sorted(set(cameras) - set(ex.get("eligible_cameras") or []))
    if not cameras or bad:
        raise SystemExit(f"cameras {bad or cameras} are not eligible ({ex.get('eligible_cameras')})")
    if not evidence or not decided_by or not reference:
        raise SystemExit("evidence, decided_by and reference are required")
    new = copy.deepcopy(doc)
    new.pop("reference_sha256", None)
    new["fixedness_exception"] = {**ex, "cameras": sorted(cameras), "decision": decision, "decided_by": decided_by,
                                  "decided_at": decided_at or time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                                  "reference": reference,
                                  "evidence": [{"path": str(p), "sha256": h} for p, h in evidence]}
    new["name"] = f"{doc['name']}-fx-{decision}"
    new["supersedes"] = {"name": doc["name"]}
    return new


def main(argv=None):
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("publish")
    p.add_argument("--task", required=True)
    p.add_argument("--results", required=True, help="fz/<depth>/<task> directory incl. full.json")
    p.add_argument("--final", required=True, help="the 'all' refit whose poses are published")
    p.add_argument("--fixedness", required=True, help="G5a whole-episode registration JSON (fixedness-1seg)")
    p.add_argument("--reference", required=True, help="its reference arrays (.reference.npz)")
    p.add_argument("--verdict", required=True, help="frozen verdict JSON (recomputation must match it)")
    p.add_argument("--freeze", required=True, help="FREEZE.md (hashed into the provenance)")
    p.add_argument("--commit", required=True)
    p.add_argument("--depth-result", action="append", help="stage=all.json of an independent solve (depth scale)")
    p.add_argument("--exception-eligible", nargs="*", default=[])
    p.add_argument("--out", required=True)
    f = sub.add_parser("fill-exception")
    f.add_argument("--rig", required=True)
    f.add_argument("--cameras", nargs="+", required=True)
    f.add_argument("--evidence", nargs="+", required=True, help="evidence files (sha256 computed now)")
    f.add_argument("--decision", required=True, choices=("accepted", "rejected"))
    f.add_argument("--decided-by", required=True)
    f.add_argument("--reference", required=True, help="decision reference (e.g. HANDOFF line, FIXREG report)")
    f.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    if a.cmd == "publish":
        doc = publish(a)
        print(json.dumps({"name": doc["name"], "reference_sha256": doc["reference_sha256"],
                          "cameras": {c: {"status": v["status"], "held_by": v["held_by"]}
                                      for c, v in doc["cameras"].items()}}, indent=1))
        return
    from s4d.realworld.calibration_rig import load_rig, publish_rig

    out = Path(a.out)
    if out.exists() or out.with_suffix(".reference.npz").exists():
        raise SystemExit(f"{out} exists: published documents are never replaced")
    doc, ref, _ = load_rig(a.rig)
    new = fill_exception(doc, cameras=a.cameras, evidence=[(p, sha256_file(p)) for p in a.evidence],
                         decision=a.decision, decided_by=a.decided_by, reference=a.reference)
    new["supersedes"]["sha256"] = sha256_file(a.rig)
    arrays = {f"{c.replace('/', '__')}__{k}": v for c, d in ref.items() for k, v in d.items()}
    pub = publish_rig(out, new, arrays)
    print(json.dumps({"name": pub["name"], "fixedness_exception": pub["fixedness_exception"]}, indent=1))


if __name__ == "__main__":
    main()
