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
    python -m s4d.realworld.rig_loma_publish split-periods --rig rig-loma-T[-fx].json --periods period_steps.json \
        --cameras C ... --evidence FILE ... --decided-by NAME --reference TEXT --out-dir DIR

RECORDING PERIODS (user decision 2026-11-10 ~13:00, covering: "handle credible recording-period camera movement
explicitly rather than forcing one shared calibration"): ``split-periods`` writes one NEW document per recording period
(``<stem>.<period>.json``) from a published document and the pre-registered period analysis (``rwx-period-steps/1``,
realworld-ops/rwx/tools/period_steps.py on the FIXREG registrations). Each period document lists only its period's
episodes (episodes, per-episode G5b evidence, pool frames: the released rig mode therefore rejects every other
episode); a camera with a credible rigid step between a period and the BASE period (the one the shared solve is
dominated by) gets that period's pose ``exp([omega]x) @ T_cam_base`` - omega = the camera rotation whose image field
best matches the measured rigid image step - bounded by ``MAX_PERIOD_ROT_DEG``; everything else (verdict, depth
scales, exception, reference arrays) is the published document's. Format, rig mode and records are unchanged; the
records builder takes the period documents of a task together (``calib.release plan --rig T=A --rig T=B``).
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import time
from pathlib import Path

import numpy as np

from s4d.realworld.calib.dw.geom import pose_distance, so3_exp
from s4d.realworld.calib.dw.publish import CAMS, GATES, fixedness_epipolar, fixedness_registration, verdict
from s4d.realworld.robotics.model import GRIPPER_MAP_2F85

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
PERIOD_FORMAT = "rwx-period-steps/1"
MAX_PERIOD_ROT_DEG = 0.035  # 1/10 of the 0.35 deg rig tolerance (G1 spread, G6 perturbation); larger -> user decision
PERIOD_MIN_PX, PERIOD_P_MAX = 0.10, 0.001  # a 'step' of the analysis: D_rigid >= FIXREG's A, permutation p < 0.001
# FIXREG's evaluation grid (calib-loma/fixreg PREREG: 10 x 7 points spanning x 32..608, y 32..448 px)
PERIOD_GRID = np.array([(x, y) for y in np.linspace(32.0, 448.0, 7) for x in np.linspace(32.0, 608.0, 10)])


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


def fixedness_evidence(R: dict, fixed: dict, per_ep: dict | None = None, epipolar=None) -> dict:
    """{camera: {episode: {"registration": fixed|moved|not_verifiable, "epipolar": ok|moved|single_pair|
    not_verifiable}}} - the per-episode G5a / G5b classification the verdict used (rig mode reads ``epipolar``).
    ``per_ep`` / ``epipolar(per_ep, cam, episodes)``: the G5b evidence and rule (default: the solve's stored
    ``per_episode_extext`` and release/1's ``fixedness_epipolar``; criteria v2: the pairwise static evaluation and
    ``calib.criteria.epipolar_v2`` with both pairs required)."""
    full = R["full"]
    per_ep = (full.get("per_episode_extext") or {}) if per_ep is None else per_ep
    episodes = sorted(set(full.get("fit_episodes") or []) | set(full.get("heldout_episodes") or []))
    out = {}
    for cam in CAMS:
        reg = fixedness_registration(fixed, cam)
        epi = fixedness_epipolar(per_ep, cam) if epipolar is None else epipolar(per_ep, cam, episodes)
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


def build_document(*, task, R, final, fixed, frozen, ref_files, depth_results, eligible, commit, sources,
                   user_decision=USER_DECISION, criteria_v2: dict | None = None) -> dict:
    """``criteria_v2`` = {"static_eval": dict, "fixreg": dict | None} publishes under ``calib.criteria`` release/2
    (G1 radius + image check, G4 / G5b on the pairwise static evidence, G5a materiality for exception eligibility);
    the release/1 verdict is still recomputed, checked against the frozen verdict and stored in ``criteria``."""
    from s4d.realworld.calib.rig import FIXEDNESS_RULE_G5AB, FORMAT

    if final.get("task") != task or R["full"].get("task") != task:
        raise SystemExit(f"results are not of task {task}")
    if final.get("variant") != "all" or final.get("heldout_episodes"):
        raise SystemExit("the published poses must be the 'all' refit (every episode, no held-out)")
    v2 = None
    if criteria_v2 is None:
        rel = release_verdict(R, fixed, eligible)
    else:
        from s4d.realworld.calib import criteria as C

        if R.get("all", {}).get("state") != final.get("state"):
            raise SystemExit("criteria v2: the results' all.json is not the published solve")
        rel = release_verdict(R, fixed)
        v2 = C.release_verdict_v2(R, fixed, criteria_v2["static_eval"], task=task, fixreg=criteria_v2.get("fixreg"))
        for cam in eligible:
            if not v2[cam]["exception_eligible_v2"]:
                raise SystemExit(f"{cam}: --exception-eligible but under criteria v2 held by {v2[cam]['held_by']} "
                                 f"(eligible only when held by G5a alone with immaterial FIXREG motion: "
                                 f"{(v2[cam]['g5a_materiality'] or {}).get('problems')})")
    check_against_frozen(rel, frozen)
    ds, agree = depth_scales(final, depth_results)
    evidence = fixedness_evidence(R, fixed) if v2 is None else fixedness_evidence(
        R, fixed, criteria_v2["static_eval"]["per_episode_extext"], C.epipolar_v2)
    if sorted(final["fit_episodes"]) != sorted(next(iter(evidence.values()))):
        raise SystemExit("the published solve and the verdict cover different episodes")
    name = f"rig-loma-{task}-{commit[:7]}" + ("" if v2 is None else "-c2")
    doc = {"format": FORMAT, "name": name, "task": task,
           "method": "rig_loma: SAM-free LoMa-G render-to-image robot 2D-3D correspondences + RoMa v2 static RGB-D "
                     "two-view factors, per-task joint 3-camera rig (FS depth, sigma_logd 3 %, per-task gripper mount "
                     "estimate), one fixed rig per task",
           "reference_episode": fixed["reference_episode"], "episodes": sorted(final["fit_episodes"]),
           "pool_frames": {},
           "pool_frames_note": "rig_loma fitted LoMa-G 2D-3D correspondences and static RoMa matches; it used no robot "
                               "depth term, so every frame's SAM-masked robot-depth check in rig mode is independent "
                               "of the fit (held-out evidence type)",
           "gate_version": GATE_VERSION if v2 is None else C.GATE_VERSION_V2,
           "gates": {**GATES, "g6_primary": "SAM-free corr_challenge (frozen a935aef)", "g6_sam": "diagnostic",
                     **({} if v2 is None else {"criteria": C.V2, "g1_v2": {k: (list(x) if isinstance(x, tuple) else x)
                                                                           for k, x in C.G1_V2.items()},
                                               "g5a_material_px": C.G5A_MATERIAL_PX, "g5a_z": C.G5A_Z,
                                               "g5b_min_verifiable": C.G5B_MIN_VERIFIABLE})},
           "fixedness_rule": FIXEDNESS_RULE_G5AB, "fixedness_config": fixed["config"],
           "fixedness_evidence": evidence,
           "fixedness_exception": {"eligible_cameras": sorted(eligible), "cameras": [], "decision": None,
                                   "evidence": [], "decided_by": None, "decided_at": None, "reference": None,
                                   "procedure": "docs/realworld_calibration_release.md (fixedness exception): the "
                                                "coordinator fills cameras, evidence files (path + sha256) and the "
                                                "decision from the FIXREG stream with "
                                                "`rig_loma_publish fill-exception` (writes a new document)"},
           "episode_checks": EPISODE_CHECKS, "robot_model": ROBOT_MODEL,
           "provenance": {"frozen_commit": commit, "gate_version": GATE_VERSION if v2 is None else C.GATE_VERSION_V2,
                         "user_decision": user_decision,
                          "final_variant": final["variant"], "config": final["config"],
                          "depth_pose_agreement": agree, **sources},
           "cameras": {}}
    if v2 is not None:
        doc["criteria"] = {
            "version": C.V2, "gate_version_release1": GATE_VERSION,
            "release1_verdict": {cam: {k: rel[cam][k] for k in ("status", "reasons", "held_by")} for cam in CAMS},
            "release1_fixedness_evidence": fixedness_evidence(R, fixed),
            "g5b_evidence": "pairwise-nearest static matches (romav2_match_worker --pairing pairwise; release/1 "
                            "matches + supplement), calib.criteria.static_eval",
            "rationale": "docs/realworld_calibration_release.md (Criteria v2); realworld-ops/rwx/CALSYNC-REPORT-20261112.md"}
    for cam in CAMS:
        v, b = (rel[cam] if v2 is None else {**rel[cam], **{k: v2[cam][k] for k in ("status", "reasons", "held_by",
                                                                                  "checks")}}), rel[cam]["base"]
        doc["cameras"][cam] = {
            "status": v["status"], "reasons": v["reasons"], "held_by": v["held_by"], "checks": v["checks"],
            "T_cam_base": final["state"]["T_cam_base"][cam], "depth_scale": ds[cam], "tau_s": final["state"]["tau_s"][cam],
            "empirical_spread": b["empirical_spread"],
            "validation": {k: b[k] for k in ("heldout", "fixedness", "alpha", "raw_depth_flag")},
            "diagnostics": {**v["diagnostics"], **({} if v2 is None else {"criteria_v2": {
                                k: x for k, x in v2[cam].items() if k != "g1_per_solve"}}),
                            "gripper_mount_estimate_dM_deg_mm": final.get("mount_dM"),
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
    from s4d.realworld.calib.rig import publish_rig

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
    crit = None
    if getattr(a, "criteria", "v1") == "v2":
        from s4d.realworld.calib.criteria import USER_DECISION_V2, check_static_eval

        if not getattr(a, "static_eval", None):
            raise SystemExit("--criteria v2 needs --static-eval")
        se = json.loads(Path(a.static_eval).read_text())
        check_static_eval(se, Path(a.results) / "full.json")
        crit = {"static_eval": se, "fixreg": json.loads(Path(a.fixreg).read_text()) if getattr(a, "fixreg", None) else None}
        sources["static_eval"] = _ref(a.static_eval)
        if getattr(a, "fixreg", None):
            sources["fixreg"] = _ref(a.fixreg)
    doc = build_document(task=a.task, R=R, final=final, fixed=fixed, frozen=frozen, ref_files=ref_files,
                         depth_results=depth_results, eligible=tuple(a.exception_eligible or ()), commit=a.commit,
                         sources=sources, criteria_v2=crit,
                         user_decision=getattr(a, "user_decision", None) or (USER_DECISION if crit is None
                                                                             else USER_DECISION_V2))
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


def rotation_from_rigid(rigid, K, grid=PERIOD_GRID) -> np.ndarray:
    """Camera rotation omega (rad, camera frame) whose small-angle pure-rotation image field (K R K^-1, FIXREG
    ``design_rot``) best matches, in least squares over ``grid``, the rigid image step ``rigid`` = (tx px, ty px,
    in-plane rotation deg) about the principal point (FIXREG similarity without its scale, which is common-mode
    thermal). Image content displaced by that step <=> ``T_cam_base' = exp([omega]x) @ T_cam_base``."""
    fx, fy, cx, cy = map(float, K)
    tx, ty, rot = float(rigid[0]), float(rigid[1]), np.radians(float(rigid[2]))
    x, y = grid[:, 0], grid[:, 1]
    field = np.stack([tx - rot * (y - cy), ty + rot * (x - cx)], 1)
    xh, yh = (x - cx) / fx, (y - cy) / fy
    J = np.zeros((len(grid), 2, 3))
    J[:, 0] = np.stack([-fx * xh * yh, fx * (1 + xh ** 2), -fx * yh], 1)
    J[:, 1] = np.stack([-fy * (1 + yh ** 2), fy * xh * yh, fy * xh], 1)
    return np.linalg.lstsq(J.reshape(-1, 3), field.reshape(-1), rcond=None)[0]


def period_steps(doc: dict, ev: dict) -> tuple[dict, str, dict, list]:
    """(periods {name: sorted episodes}, base period, {camera: {period: omega rad}}, problems) of a pre-registered
    period analysis (``rwx-period-steps/1``) for ``doc``. Always refused (SystemExit, nothing is written): another
    format / task, periods not partitioning the document's episodes, a step camera whose omega does not reproduce from
    the recorded step, exceeds ``MAX_PERIOD_ROT_DEG`` or has a camera-specific scale flag. ``problems`` (USER
    territory of the pre-registered rule, refused unless a user override is recorded): within-period movement or an
    unverifiable candidate of any camera, an inconclusive step or an unresolved scale flag of any camera, and a
    decision table that is missing or does not allow the split."""
    from s4d.realworld.calib.rig import FORMAT

    if ev.get("format") != PERIOD_FORMAT or ev.get("task") != doc.get("task") or doc.get("format") != FORMAT:
        raise SystemExit(f"not a {PERIOD_FORMAT} analysis of task {doc.get('task')!r}")
    periods = {p: sorted(e) for p, e in (ev.get("periods") or {}).items()}
    base = ev.get("base_period")
    flat = [e for v in periods.values() for e in v]
    if len(periods) < 2 or base not in periods or len(flat) != len(set(flat)) or sorted(flat) != sorted(doc["episodes"]):
        raise SystemExit("the periods must partition the document's episodes exactly (>= 2 periods incl. the base)")
    steps, problems = {}, []
    for cam in doc["cameras"]:
        c = (ev.get("cameras") or {}).get(cam) or {}
        within = c.get("within") or {}
        if within.get("event") is not False:
            problems.append(f"{cam}: within-period movement {within.get('event')!r} {within.get('reasons')}")
        for p in periods:
            if p == base:
                continue
            s = (c.get("period_steps") or {}).get(p) or {}
            if s.get("class") != "step":
                if s.get("class") != "no_step" or s.get("scale_flag") is not False:
                    problems.append(f"{cam} {p}: period step {s.get('class')!r}, scale flag {s.get('scale_flag')!r}")
                continue
            if not (s.get("D_rigid", 0) >= PERIOD_MIN_PX and s.get("p", 1) < PERIOD_P_MAX and s.get("A_agrees") is True):
                raise SystemExit(f"{cam} {p}: class 'step' without its evidence (D {s.get('D_rigid')}, p {s.get('p')}, "
                                 f"A agrees {s.get('A_agrees')})")
            omega = rotation_from_rigid(s["rigid"], c["K"])
            if not np.allclose(np.degrees(omega), s.get("omega_deg"), rtol=0, atol=1e-9):
                raise SystemExit(f"{cam} {p}: recorded omega {s.get('omega_deg')} does not reproduce")
            if np.degrees(np.linalg.norm(omega)) > MAX_PERIOD_ROT_DEG:
                raise SystemExit(f"{cam} {p}: period step {np.degrees(np.linalg.norm(omega)):.4f} deg > "
                                 f"{MAX_PERIOD_ROT_DEG} deg -> user decision")
            if s.get("scale_flag") is not False:
                raise SystemExit(f"{cam} {p}: camera-specific scale flag {s.get('scale_flag')!r} -> user decision")
            steps.setdefault(cam, {})[p] = omega
    dec = ev.get("decision")
    if not isinstance(dec, dict):
        problems.append("the analysis carries no decision table (run with the release verdict)")
    else:
        if dec.get("split_periods") is not True or dec.get("user_decision_needed"):
            problems.append(f"decision table: split_periods {dec.get('split_periods')!r}, user decision needed for "
                            f"{dec.get('user_decision_needed')}")
        if sorted(dec.get("period_pose_cameras") or []) != sorted(steps):
            problems.append(f"decision period_pose_cameras {dec.get('period_pose_cameras')} != {sorted(steps)}")
    return periods, base, steps, problems


def split_periods(doc: dict, ev: dict, *, cameras, evidence, decided_by, reference, source_sha256,
                  decided_at=None, user_override=None) -> dict:
    """{period: NEW document} (see the module docstring). ``cameras`` must be exactly the cameras with a credible
    period step (the pre-registered rule applies to all of them); ``evidence``: [(path, sha256)]. ``user_override``:
    the text of a USER decision that resolves the analysis' open problems (recorded with them in every document);
    without it any open problem refuses."""
    periods, base, steps, problems = period_steps(doc, ev)
    if problems and not user_override:
        raise SystemExit(f"user decision needed (pass the user's decision as override): {problems}")
    if not steps:
        raise SystemExit("no camera has a credible period step: the published document applies to every period")
    if sorted(cameras) != sorted(steps):
        raise SystemExit(f"cameras {sorted(cameras)} != cameras with a credible period step {sorted(steps)}")
    if not evidence or not decided_by or not reference:
        raise SystemExit("evidence, decided_by and reference are required")
    names = {p: f"{doc['name']}-{p}" for p in periods}
    record = {"base_period": base, "periods": {p: len(v) for p, v in periods.items()}, "documents": sorted(names.values()),
              "derived_from": {"name": doc["name"], "sha256": source_sha256}, "cameras_with_period_pose": sorted(steps),
              "evidence": [{"path": str(p), "sha256": h} for p, h in evidence], "decided_by": decided_by,
              "decided_at": decided_at or time.strftime("%Y-%m-%dT%H:%M:%S%z"), "reference": reference,
              "user_override": {"decision": user_override, "problems": problems} if problems else None,
              "rule": f"{PERIOD_FORMAT}: T_period = exp([omega]x) @ T_cam_base for cameras with a credible rigid step "
                      f"vs the base period (|omega| <= {MAX_PERIOD_ROT_DEG} deg); every other field is the published "
                      "document's; within-period movement, an inconclusive step, a scale flag or a decision table that does "
                      "not allow the split refuses unless a user decision is recorded (user_override)"}
    out = {}
    for p, eps in periods.items():
        keep = set(eps)
        new = copy.deepcopy(doc)
        new.pop("reference_sha256", None)
        new["name"], new["episodes"] = names[p], eps
        new["fixedness_evidence"] = {c: {e: v for e, v in rows.items() if e in keep}
                                     for c, rows in doc["fixedness_evidence"].items()}
        if isinstance((new.get("criteria") or {}).get("release1_fixedness_evidence"), dict):  # criteria v2 documents
            new["criteria"]["release1_fixedness_evidence"] = {
                c: {e: v for e, v in rows.items() if e in keep}
                for c, rows in doc["criteria"]["release1_fixedness_evidence"].items()}
        new["pool_frames"] = {e: v for e, v in (doc.get("pool_frames") or {}).items() if e in keep}
        for cam, entry in new["cameras"].items():
            entry["provenance"] = {**(entry.get("provenance") or {}), "rig": names[p]}
            omega = steps.get(cam, {}).get(p)
            if omega is None:
                continue
            T = np.asarray(doc["cameras"][cam]["T_cam_base"], float)
            R = np.eye(4)
            R[:3, :3] = so3_exp(omega)
            entry["T_cam_base"] = (R @ T).tolist()
            entry["period_pose"] = {"period": p, "base_period": base, "omega_deg": np.degrees(omega).tolist(),
                                    "rot_deg": float(np.degrees(np.linalg.norm(omega))),
                                    "rigid_step": ev["cameras"][cam]["period_steps"][p]["rigid"],
                                    "base_T_cam_base": T.tolist()}
        new["period"] = {"name": p, "episodes": eps, **record}
        out[p] = new
    return out


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
    p.add_argument("--user-decision", help="the task's decision text for the provenance (default: the 2026-11-08 "
                                           "drawer-open / mug-hanging release text)")
    p.add_argument("--criteria", choices=("v1", "v2"), default="v1",
                   help="v1 = rig_loma-release/1 (default, frozen); v2 = calib.criteria release/2 (CALSYNC)")
    p.add_argument("--static-eval", help="v2: calib.criteria static-eval output (pairwise G4 / G5b evidence)")
    p.add_argument("--fixreg", help="v2: FIXREG results (G5a materiality of G5a-only cameras)")
    p.add_argument("--out", required=True)
    f = sub.add_parser("fill-exception")
    f.add_argument("--rig", required=True)
    f.add_argument("--cameras", nargs="+", required=True)
    f.add_argument("--evidence", nargs="+", required=True, help="evidence files (sha256 computed now)")
    f.add_argument("--decision", required=True, choices=("accepted", "rejected"))
    f.add_argument("--decided-by", required=True)
    f.add_argument("--reference", required=True, help="decision reference (e.g. HANDOFF line, FIXREG report)")
    f.add_argument("--out", required=True)
    s = sub.add_parser("split-periods")
    s.add_argument("--rig", required=True, help="the published (or exception-filled) document")
    s.add_argument("--periods", required=True, help=f"the pre-registered period analysis ({PERIOD_FORMAT})")
    s.add_argument("--cameras", nargs="+", required=True, help="exactly the cameras with a credible period step")
    s.add_argument("--evidence", nargs="*", default=[], help="further evidence files (sha256 computed now)")
    s.add_argument("--decided-by", required=True)
    s.add_argument("--reference", required=True)
    s.add_argument("--user-override", help="the user's decision text resolving open problems of the analysis")
    s.add_argument("--out-dir", required=True)
    a = ap.parse_args(argv)
    if a.cmd == "publish":
        doc = publish(a)
        print(json.dumps({"name": doc["name"], "reference_sha256": doc["reference_sha256"],
                          "cameras": {c: {"status": v["status"], "held_by": v["held_by"]}
                                      for c, v in doc["cameras"].items()}}, indent=1))
        return
    from s4d.realworld.calib.rig import load_rig, publish_rig

    if a.cmd == "split-periods":
        doc, ref, fp = load_rig(a.rig)
        raw = Path(a.periods).read_bytes()  # parsed and hashed from the same bytes
        docs = split_periods(doc, json.loads(raw), cameras=a.cameras,
                             evidence=[(a.periods, hashlib.sha256(raw).hexdigest())]
                             + [(p, sha256_file(p)) for p in a.evidence],
                             decided_by=a.decided_by, reference=a.reference, source_sha256=fp["rig_calibration"],
                             user_override=a.user_override)
        paths = {p: Path(a.out_dir) / f"{Path(a.rig).stem}.{p}.json" for p in docs}
        if any(x.exists() or x.with_suffix(".reference.npz").exists() for x in paths.values()):
            raise SystemExit(f"{a.out_dir}: a period document exists already (published documents are never replaced)")
        arrays = {f"{c.replace('/', '__')}__{k}": v for c, d in ref.items() for k, v in d.items()}
        Path(a.out_dir).mkdir(parents=True, exist_ok=True)
        for p, d in docs.items():
            pub = publish_rig(paths[p], d, arrays)
            print(json.dumps({"path": str(paths[p]), "name": pub["name"], "episodes": len(pub["episodes"]),
                              "period_poses": {c: v["period_pose"]["rot_deg"] for c, v in pub["cameras"].items()
                                               if "period_pose" in v}}))
        return
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
