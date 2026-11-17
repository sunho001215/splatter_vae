"""Task-specific, user-authorized acceptance override of a held per-task rig calibration (OFFLINE use only).

A per-task rig calibration is published as accepted only when the established automatic criteria
(``calib.criteria`` release/2; ``calib.rig.rig_camera_status``) validate every camera. This module adds ONE narrowly
scoped exception path for a task whose user decision accepts specific, recorded automatic failures for offline use:

* The decision is a committed record (``overrides/<file>`` named in ``RECORDS``; one explicit entry per task). It names
  the task, every covered camera with the gates the override covers for it, each exception (gate, check, metric, the
  recorded measured value and the UNCHANGED threshold), the automatic decision it overrides (accepted false, sha256),
  the user decisions verbatim, the sensitivity result and evidence hashes.
* A held camera is effective-accepted (status ``STATUS`` = "validated_by_override", never "validated") ONLY when its
  held_by gates are a subset of the gates named for that task / camera, every reason it is held for is one of the
  recorded exception reasons (any other failing sub-check, e.g. the G1 rotation radius, keeps it held), and every
  recomputed metric of its exceptions equals the recorded value within ``REL_TOL`` (integers / strings exactly). Any
  other task, camera, gate, pair, threshold or value drift keeps the camera held.
* Global criteria, thresholds and code paths are unchanged (the thresholds are re-checked against the record).
  ``automatic-decision.json`` keeps accepted:false and is never rewritten; the publisher writes a separate
  ``override-decision.json`` and embeds an ``acceptance_override`` block (record, sha256s, automatic decision) in the
  published rig document, whose camera entries keep their automatic "held" verdicts.
* Robot execution is NOT authorized: ``calib.rig.rig_camera_status`` and ``rwlearn.deploy.realsense.load_rig_session``
  (the robot / deployment path) are unchanged and keep refusing such a camera. Only the offline consumers below accept
  ``STATUS`` and propagate an ``acceptance_override`` reference into their records.

Execution-scope rule (tests/test_dsreset_registration.py placement-scope guards): ``calib.rig``, ``stages.calibrate``,
``rwlearn.gate`` / ``native_depth`` / ``deploy.realsense`` and ``scripts/realworld_prep_register.py`` are inside the
recorded execution scope of placed rows and stay byte-identical. Offline consumers that live there are served by
process-local installation, exactly as ``rwlearn.native_depth_mapped`` and ``rwlearn.dsreset_runtime`` do:

    python -m s4d.realworld.calib.acceptance_override rwprep -- episode --pipeline P ...   (phase B worker step)
    python -m s4d.realworld.calib.acceptance_override compile-phase-b <realworld_prep_register.py args, --phase B>
    python -m s4d.realworld.calib.acceptance_override gate --rig RIG --task T             (offline route gate)
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import runpy
import sys
from pathlib import Path

RECORD_FORMAT = "s4d-calib-acceptance-override/1"
BLOCK_FORMAT = "s4d-calib-acceptance-override-block/1"
DECISION_FORMAT = "s4d-calib-override-decision/1"
STATUS = "validated_by_override"
OFFLINE_ACCEPTED = frozenset({"validated", STATUS})
OFFLINE_USES = ("phase_b_preprocessing", "dataset_publication", "pretraining", "fql", "evaluation")
REL_TOL = 1e-9
RECORDS_DIR = Path(__file__).resolve().parent / "overrides"
# One explicit, reviewed entry per task. Nothing is discovered from the filesystem.
RECORDS = {"drawer-open-v2": "drawer-open-v2-20261117.json"}
CHECKS = ("g1_centre_radius", "g5b_dt_extension_median_upper")
RWPREP = "/opt/rwprep/rwprep_local.py"
_HEX64 = __import__("re").compile(r"^[0-9a-f]{64}$")


def _sha_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path) -> str:
    return _sha_bytes(Path(path).read_bytes())


def _close(a, b) -> bool:
    try:
        a, b = float(a), float(b)
    except (TypeError, ValueError):
        return False
    return math.isfinite(a) and math.isfinite(b) and math.isclose(a, b, rel_tol=REL_TOL, abs_tol=0.0)


# --------------------------------------------------------------------------------------------------------- record
def record_problem(rec: dict, task: str) -> str | None:
    """Why a committed override record is not usable for ``task`` (None: well-formed). Fails closed."""
    if not isinstance(rec, dict) or rec.get("format") != RECORD_FORMAT:
        return f"record format is not {RECORD_FORMAT}"
    if rec.get("task") != task:
        return f"record task {rec.get('task')!r} != {task!r}"
    if rec.get("decision") != "accepted" or rec.get("status") != STATUS:
        return "record decision is not an accepted override"
    scope = rec.get("scope") or {}
    if scope.get("robot_execution_authorized") is not False:
        return "an acceptance override never authorizes robot execution"
    if tuple(scope.get("offline_uses") or ()) != OFFLINE_USES:
        return "record offline uses differ from the supported offline uses"
    auto = rec.get("automatic_acceptance") or {}
    if auto.get("passed") is not False or (auto.get("decision") or {}).get("accepted") is not False:
        return "the record must preserve a FAILED automatic decision"
    if not _HEX64.match(str(auto.get("decision_sha256", ""))) or not _HEX64.match(
            str(auto.get("verdict_release1_sha256", ""))):
        return "the automatic decision and verdict sha256 are required"
    if auto.get("thresholds_changed") is not False:
        return "an acceptance override never changes thresholds"
    if (rec.get("value_tolerance") or {}).get("relative") != REL_TOL:
        return f"record value tolerance must be the fixed relative {REL_TOL}"
    cams = rec.get("cameras")
    if not isinstance(cams, dict) or not cams or not all(
            isinstance(v, dict) and v.get("gates") and all(isinstance(g, str) for g in v["gates"])
            for v in cams.values()):
        return "record names no cameras / gates"
    exc = rec.get("exceptions")
    if not isinstance(exc, list) or not exc:
        return "record names no exceptions"
    for e in exc:
        if not isinstance(e, dict) or e.get("check") not in CHECKS or not e.get("reason") or not e.get("cameras"):
            return f"malformed exception {e.get('id') if isinstance(e, dict) else e!r}"
        if not all(c in cams and e.get("gate") in cams[c]["gates"] for c in e["cameras"]):
            return f"exception {e.get('id')} covers a camera / gate the record does not name"
        if not (_close(e.get("measured"), e.get("measured")) and _close(e.get("threshold"), e.get("threshold"))):
            return f"exception {e.get('id')} lacks finite measured / threshold values"
        if not float(e["measured"]) > float(e["threshold"]):
            return f"exception {e.get('id')}: a measured value within its threshold needs no override"
    if not rec.get("user_decisions") or not all(isinstance(d, dict) and d.get("text") for d in rec["user_decisions"]):
        return "record lacks the verbatim user decisions"
    sens = rec.get("sensitivity_check") or {}
    if sens.get("accepted_by_user") is not True or not sens.get("artifacts") or not all(
            _HEX64.match(str(a.get("sha256", ""))) for a in sens["artifacts"]):
        return "record lacks the sensitivity result, its evidence hashes or the user's acceptance of it"
    prov = rec.get("provenance") or {}
    if not all(_HEX64.match(str(prov.get(k, ""))) for k in ("queue_sha256", "spec_sha256", "freeze_sha256")):
        return "record lacks queue / spec / FREEZE sha256"
    if not str(prov.get("criteria_commit", "")).startswith(str(auto.get("criteria_commit", "-"))):
        return "record criteria commit inconsistent"
    return None


def record_for(task: str | None) -> tuple[dict, str, str] | None:
    """(record, sha256 of the committed file bytes, repository-relative path) of ``task``'s override, or None when the
    task has none. A malformed committed record raises (fail closed)."""
    name = RECORDS.get(task or "")
    if name is None:
        return None
    path = RECORDS_DIR / name
    data = path.read_bytes()
    rec = json.loads(data)
    why = record_problem(rec, task)
    if why:
        raise ValueError(f"{path}: {why}")
    return rec, _sha_bytes(data), f"s4d/realworld/calib/overrides/{name}"


# ----------------------------------------------------------------------------------------------- camera evaluation
def _exception_problems(e: dict, v2: dict) -> list:
    """Recomputed evidence of one recorded exception (``v2`` = the camera's criteria v2 diagnostics) vs the record."""
    from s4d.realworld.calib import criteria as C

    out, tag = [], e.get("id")
    if e["check"] == "g1_centre_radius":
        g1 = (v2 or {}).get("g1") or {}
        if C.G1_V2["radius_centre_m"] != e["threshold"] or (g1.get("thresholds") or {}).get(
                "radius_centre_m") != e["threshold"]:
            out.append(f"{tag}: G1 centre-radius threshold differs from the record")
        if not _close(g1.get("radius_centre_m"), e["measured"]):
            out.append(f"{tag}: G1 centre radius {g1.get('radius_centre_m')} != recorded {e['measured']}")
        if (g1.get("worst") or {}).get("centre_m") != e.get("worst_solve"):
            out.append(f"{tag}: G1 worst centre solve {(g1.get('worst') or {}).get('centre_m')} != "
                       f"{e.get('worst_solve')}")
        if list(g1.get("reasons") or []) != [e["reason"]]:
            out.append(f"{tag}: G1 fails for other reasons {g1.get('reasons')}")
    elif e["check"] == "g5b_dt_extension_median_upper":
        g5b = (v2 or {}).get("g5b") or {}
        dt = g5b.get("dt_extension") or {}
        row = (dt.get("pairs") or {}).get(e.get("pair")) or {}
        ctx = e.get("context") or {}
        if C.G5B_DT_MAX_DIFF_PX != e["threshold"] or dt.get("max_diff_px") != e["threshold"]:
            out.append(f"{tag}: G5b dt-extension threshold differs from the record")
        if list(g5b.get("reasons") or []) != [e["reason"]]:
            out.append(f"{tag}: G5b fails for other reasons {g5b.get('reasons')}")
        if list(dt.get("problems") or []) != [e["problem"]]:
            out.append(f"{tag}: dt extension problems {dt.get('problems')} != the recorded single problem")
        if g5b.get("moved") or g5b.get("missing"):
            out.append(f"{tag}: per-episode G5b flags episodes")
        ci = row.get("median_diff_ci95_px") or [None, None]
        if not _close(ci[-1], e["measured"]):
            out.append(f"{tag}: {e.get('pair')} median upper bound {ci[-1]} != recorded {e['measured']}")
        for k in ("n_short", "n_long"):
            if row.get(k) != ctx.get(k):
                out.append(f"{tag}: {k} {row.get(k)} != recorded {ctx.get(k)}")
        for k in ("median_diff_px", "p90_diff_px"):
            if not _close(row.get(k), ctx.get(k)):
                out.append(f"{tag}: {k} {row.get(k)} != recorded {ctx.get(k)}")
        if row.get("long_violations") != ctx.get("long_violations", []):
            out.append(f"{tag}: long-offset violations {row.get('long_violations')}")
    else:
        out.append(f"{tag}: unknown check {e['check']!r}")
    return out


def camera_problems(rec: dict, cam: str, entry: dict) -> list:
    """Why ``rec`` does NOT make the published camera ``entry`` effective-accepted (empty list: it does)."""
    cover = (rec.get("cameras") or {}).get(cam)
    if not cover:
        return [f"{cam} is not named by the override"]
    if entry.get("status") != "held":
        return [f"{cam} published status {entry.get('status')!r} (only a held camera can be covered)"]
    named = set(cover["gates"])
    held_by = set(entry.get("held_by") or [])
    out = []
    if not held_by:
        out.append(f"{cam} is held without a held_by gate")
    if held_by - named:
        out.append(f"{cam} held by gate(s) {sorted(held_by - named)} the override does not name")
    failing = {g for g, ok in (entry.get("checks") or {}).items() if not ok}
    if failing - named:
        out.append(f"{cam} fails check(s) {sorted(failing - named)} the override does not name")
    exc = [e for e in rec["exceptions"] if cam in e["cameras"]]
    allowed = {e["reason"] for e in exc}
    extra = [r for r in entry.get("reasons") or [] if r not in allowed]
    if extra:
        out.append(f"{cam} held for unrecorded reasons {extra}")
    v2 = (entry.get("diagnostics") or {}).get("criteria_v2")
    if not isinstance(v2, dict):
        out.append(f"{cam} carries no criteria v2 evidence")
        return out
    for e in exc:
        out += [f"{cam}: {p}" for p in _exception_problems(e, v2)]
    return out


def long_offset_cohort(se: dict, pair: str, aliases: dict) -> list:
    """Canonical episodes of ``pair`` classified long-offset by ``calib.criteria.dt_extension``'s rule (largest |dt|
    over the contributing files > 20 ms), mapped through the calibration's alias map."""
    from s4d.realworld.calib.criteria import CAMS, RELEASE1_MAX_DT_MS, pair_name

    table = {}
    for r in (se.get("dt") or {}).get("rows") or []:
        pa, pb = divmod(int(r["pair"]), 10)
        key = (r["episode"], pair_name(CAMS[pa], CAMS[pb]))
        table[key] = max(float(r["abs_dt_ms"]), table.get(key, 0.0))
    return sorted(aliases.get(e, e) for (e, p), dt in table.items() if p == pair and dt > RELEASE1_MAX_DT_MS)


# -------------------------------------------------------------------------------------------- document consumers
def override_problem(doc: dict, cam: str) -> str | None:
    """Why the document's ``acceptance_override`` block does NOT make ``cam`` effective-accepted (None: it does).
    The embedded record must be byte-for-byte the committed record of the document's task."""
    block = doc.get("acceptance_override")
    if not isinstance(block, dict):
        return "the document carries no acceptance override"
    if block.get("format") != BLOCK_FORMAT:
        return f"acceptance override block format is not {BLOCK_FORMAT}"
    try:
        committed = record_for(doc.get("task"))
    except (OSError, ValueError) as exc:
        return f"committed override record unusable: {exc}"
    if committed is None:
        return f"no committed acceptance override for task {doc.get('task')!r}"
    rec, rec_sha, _ = committed
    if block.get("record_sha256") != rec_sha or block.get("record") != rec:
        return "the document's override record differs from the committed record"
    if block.get("robot_execution_authorized") is not False:
        return "an acceptance override never authorizes robot execution"
    auto = block.get("automatic_decision") or {}
    if auto.get("accepted") is not False or auto.get("sha256") != rec["automatic_acceptance"]["decision_sha256"] or \
            auto.get("cameras") != rec["automatic_acceptance"]["decision"]["cameras"]:
        return "the block does not carry the recorded failed automatic decision"
    if not _HEX64.match(str((block.get("override_decision") or {}).get("sha256", ""))):
        return "the block names no override decision file"
    if cam not in doc.get("cameras", {}):
        return f"{cam} is not a camera of the document"
    problems = camera_problems(rec, cam, doc["cameras"][cam])
    return "; ".join(problems) or None


def summary(doc: dict) -> dict:
    """Compact reference to a document's acceptance override (what downstream records carry)."""
    block = doc["acceptance_override"]
    return {"id": block["record"]["id"], "record_sha256": block["record_sha256"], "status": STATUS,
            "task": doc.get("task"), "rig": doc.get("name"),
            "cameras": sorted(block["record"]["cameras"]),
            "exceptions": [e["id"] for e in block["record"]["exceptions"]],
            "automatic_decision_sha256": block["automatic_decision"]["sha256"],
            "override_decision_sha256": block["override_decision"]["sha256"],
            "robot_execution_authorized": False, "offline_uses": list(OFFLINE_USES)}


def effective_status(doc: dict, cam: str) -> tuple[str, list, dict | None]:
    """OFFLINE effective status: ``calib.rig.rig_camera_status`` unchanged, except that a held camera validly covered
    by the document's acceptance override is ``STATUS``. Returns (status, reasons, override summary or None)."""
    status, why, _ = _base_status()(doc, cam)
    if status == "validated" or doc.get("acceptance_override") is None:
        return status, why, None
    problem = override_problem(doc, cam)
    if problem is None:
        return STATUS, [], summary(doc)
    return status, [*why, f"acceptance override does not apply: {problem}"], None


def offline_acceptance(doc: dict) -> dict:
    """Every camera's offline effective status; ``accepted`` iff all are in ``OFFLINE_ACCEPTED``."""
    rows = {c: effective_status(doc, c) for c in sorted(doc["cameras"])}
    statuses = {c: r[0] for c, r in rows.items()}
    override = next((r[2] for r in rows.values() if r[2] is not None), None)
    return {"accepted": all(s in OFFLINE_ACCEPTED for s in statuses.values()), "statuses": statuses,
            "acceptance_override": override, "robot_execution_authorized": False,
            "problems": {c: r[1] for c, r in rows.items() if r[0] not in OFFLINE_ACCEPTED}}


# ------------------------------------------------------------------------------------------------------ publisher
def write_once(path, doc) -> None:
    """Create ``path`` with ``ops.jobs.rwprep_mig.atomic``'s exact serialization, or require identical existing bytes
    (original decision / verdict records are never rewritten)."""
    from ops.jobs.rwprep_mig import atomic

    path = Path(path)
    if path.exists():
        if path.read_text() != json.dumps(doc, indent=2):
            raise ValueError(f"{path.name} exists with different content: original records are never rewritten")
        return
    atomic(path, doc)


def apply_publisher_override(doc: dict, *, task: str, work, specpath, static: dict, aliases: dict,
                             automatic: dict, commit: str) -> dict:
    """The publishable document of a held rig under the task's committed override, after writing
    ``override-decision.json``; raises ``ValueError`` (the rig stays held) on ANY mismatch with the record."""
    found = record_for(task)
    if found is None:
        raise ValueError("rig held by established automatic criteria; see automatic-decision.json")
    rec, rec_sha, rec_path = found
    work, specpath = Path(work), Path(specpath)
    auto_rec = rec["automatic_acceptance"]
    problems = []
    auto_path, verdict_path, freeze = work / "automatic-decision.json", work / "verdict-release1.json", \
        specpath.parent / "FREEZE.json"
    if sha256_file(auto_path) != auto_rec["decision_sha256"] or automatic != auto_rec["decision"]:
        problems.append("automatic-decision.json differs from the recorded failed decision")
    if sha256_file(verdict_path) != auto_rec["verdict_release1_sha256"]:
        problems.append("verdict-release1.json differs from the recorded verdict")
    if sha256_file(specpath) != rec["provenance"]["spec_sha256"]:
        problems.append("spec.json differs from the recorded spec")
    if sha256_file(freeze) != rec["provenance"]["freeze_sha256"]:
        problems.append("FREEZE.json differs from the recorded FREEZE")
    if commit != auto_rec["criteria_commit"]:
        problems.append(f"criteria commit {commit} != recorded {auto_rec['criteria_commit']}")
    if doc.get("task") != task or sorted(doc["cameras"]) != sorted(rec["cameras"]):
        problems.append("document task / cameras differ from the record")
    retained = rec["retained"]
    if len(doc.get("episodes") or []) != retained["episodes"] or len(doc["cameras"]) != retained["cameras"] or \
            len(set(doc.get("episodes") or [])) != retained["episodes"]:
        problems.append(f"document retains {len(doc.get('episodes') or [])} episodes / {len(doc['cameras'])} cameras")
    if any(set(rows) != set(doc["episodes"]) for rows in (doc.get("fixedness_evidence") or {}).values()):
        problems.append("per-episode G5b evidence does not cover every retained episode")
    for e in rec["exceptions"]:
        if "long_offset_cohort" in e and long_offset_cohort(static, e["pair"], aliases) != e["long_offset_cohort"]:
            problems.append(f"{e['id']}: recomputed long-offset cohort differs from the recorded cohort")
    for cam in sorted(doc["cameras"]):
        if automatic["cameras"].get(cam) != "validated":
            problems += camera_problems(rec, cam, doc["cameras"][cam])
    if problems:
        raise ValueError("rig held by established automatic criteria; acceptance override does not apply: "
                         + "; ".join(problems))
    effective = {c: "validated" if automatic["cameras"][c] == "validated" else STATUS for c in sorted(doc["cameras"])}
    decision = {"format": DECISION_FORMAT, "task": task, "status": STATUS,
                "record": {"id": rec["id"], "path": rec_path, "sha256": rec_sha},
                "automatic_decision": {"path": str(auto_path), "sha256": auto_rec["decision_sha256"], **automatic},
                "verdict_release1": {"path": str(verdict_path), "sha256": auto_rec["verdict_release1_sha256"]},
                "effective_cameras": effective, "effective_accepted_offline": True,
                "automatic_acceptance_passed": False, "robot_execution_authorized": False,
                "offline_uses": list(OFFLINE_USES), "exceptions": [e["id"] for e in rec["exceptions"]],
                "verified": ["automatic decision / verdict / spec / FREEZE sha256 equal the record",
                             "criteria commit equals the record", "98 episodes and 3 cameras retained",
                             "recomputed long-offset cohort equals the record",
                             "each held camera: held_by within the named gates, only recorded reasons, recomputed "
                             f"metrics equal the record (relative tolerance {REL_TOL})",
                             "thresholds equal the record and the unchanged criteria constants"],
                "user_decisions": rec["user_decisions"], "sensitivity_check": rec["sensitivity_check"]}
    write_once(work / "override-decision.json", decision)
    new = copy.deepcopy(doc)
    new["name"] = doc["name"] + rec["document_name_suffix"]
    for cam in new["cameras"]:
        new["cameras"][cam]["provenance"] = {**(new["cameras"][cam].get("provenance") or {}), "rig": new["name"]}
    new["acceptance_override"] = {
        "format": BLOCK_FORMAT, "status": STATUS, "record": rec, "record_sha256": rec_sha, "record_path": rec_path,
        "automatic_decision": {"path": str(auto_path), "sha256": auto_rec["decision_sha256"], **automatic},
        "override_decision": {"path": str(work / "override-decision.json"),
                              "sha256": sha256_file(work / "override-decision.json")},
        "effective_cameras": effective, "automatic_acceptance_passed": False, "robot_execution_authorized": False,
        "note": "camera entries keep their automatic verdicts; offline consumers read the effective status through "
                "calib.acceptance_override.effective_status"}
    new["provenance"] = {**new.get("provenance", {}), "acceptance_override": {
        "record": {"path": rec_path, "sha256": rec_sha},
        "override_decision": new["acceptance_override"]["override_decision"]}}
    got = {c: effective_status(new, c)[0] for c in sorted(new["cameras"])}
    if got != effective:
        raise ValueError(f"published override document does not verify through the consumer path: {got}")
    return new


def offline_gate(rig_path, task: str) -> dict:
    """Route gate for OFFLINE consumers (phase B, publication): the published rig and reference are present and
    hash-consistent, and either the automatic decision accepted every camera (unchanged) or the document carries a
    valid acceptance override bound to ``automatic-decision.json`` and ``override-decision.json``. Raises ValueError."""
    from s4d.realworld.calib.rig import load_rig, rig_camera_status

    rig_path = Path(rig_path)
    decision_path = rig_path.parent / "automatic-decision.json"
    if not (rig_path.is_file() and rig_path.with_suffix(".reference.npz").is_file() and decision_path.is_file()):
        raise ValueError("HELD: no published rig, reference NPZ and automatic decision")
    doc, _, identity = load_rig(rig_path)
    if doc.get("task") != task:
        raise ValueError(f"HELD: rig task {doc.get('task')!r} != {task!r}")
    dec = json.loads(decision_path.read_bytes())
    if set(dec.get("cameras") or {}) != set(doc["cameras"]):
        raise ValueError("HELD: automatic decision cameras differ from the rig")
    if dec.get("accepted") is True and all(v == "validated" for v in dec["cameras"].values()):
        if any(rig_camera_status(doc, c)[0] != "validated" for c in doc["cameras"]):
            raise ValueError("HELD: automatic decision and rig cameras disagree")
        return {"mode": "automatic", "rig": doc.get("name"), **identity, "acceptance_override": None,
                "robot_execution_authorized": None}
    state = offline_acceptance(doc)
    if not state["accepted"] or state["acceptance_override"] is None:
        raise ValueError(f"HELD: established automatic calibration criteria (no valid acceptance override: "
                         f"{state['problems']})")
    block = doc["acceptance_override"]
    if sha256_file(decision_path) != block["automatic_decision"]["sha256"] or dec.get("accepted") is not False:
        raise ValueError("HELD: automatic-decision.json differs from the one the override was decided on")
    override_path = rig_path.parent / "override-decision.json"
    if not override_path.is_file() or sha256_file(override_path) != block["override_decision"]["sha256"]:
        raise ValueError("HELD: override-decision.json missing or changed")
    return {"mode": "acceptance_override", "rig": doc.get("name"), **identity, "statuses": state["statuses"],
            "acceptance_override": state["acceptance_override"], "robot_execution_authorized": False}


# ------------------------------------------------------------------------------- offline native-depth session
def load_rig_session_offline(path, camera_ids, K_native: dict) -> dict:
    """OFFLINE variant of ``rwlearn.deploy.realsense.load_rig_session`` (dataset measurement / native-depth builds).
    A document without an acceptance override goes to the unchanged function; with one, a camera is accepted when its
    offline effective status is in ``OFFLINE_ACCEPTED`` and its correction evidence names the override. Robot
    deployment keeps calling ``load_rig_session`` (strict)."""
    import numpy as np

    from s4d.realworld.calib.rig import load_rig
    from s4d.rwlearn.deploy.contracts import validate_native_correction
    from s4d.rwlearn.deploy.realsense import load_rig_session

    doc, _, identity = load_rig(path)
    if doc.get("acceptance_override") is None:
        return load_rig_session(path, camera_ids, K_native)
    calibration, corrections = {}, {}
    for cam in camera_ids:
        record = doc["cameras"][f"{cam}/color"]
        status, reasons, override = effective_status(doc, f"{cam}/color")
        if status not in OFFLINE_ACCEPTED:
            raise ValueError(f"{cam}: published rig camera is {status}, not accepted for offline use ({reasons})")
        if "sensor_depth" not in (record.get("depth_scale") or {}):
            raise ValueError(f"{cam}: published rig lacks a validated sensor_depth scale")
        correction = {"stage": "sensor_depth", "factor": float(record["depth_scale"]["sensor_depth"]),
                      "validated": True, "evidence": {"document": str(path), "name": doc.get("name"), **identity,
                                                      "camera_id": cam, "fixedness_exception": False,
                                                      "effective_rig_status": status,
                                                      "acceptance_override": override}}
        validate_native_correction(correction)
        calibration[cam] = {"K_native": np.asarray(K_native[cam], np.float64).tolist(),
                            "c2w": np.linalg.inv(np.asarray(record["T_cam_base"], np.float64)).tolist()}
        corrections[cam] = correction
    return {"calibration": calibration, "native_depth_correction": corrections}


# ------------------------------------------------------------------- phase B: process-local rig-mode installation
_ORIGINAL = {}
_COVERED: dict = {}


def _base_status():
    """The unchanged ``calib.rig.rig_camera_status`` (also while this module has replaced it in the process)."""
    from s4d.realworld.calib import rig

    return _ORIGINAL.get("rig_camera_status") or rig.rig_camera_status


def _status_with_override(doc, camera_id):
    status, why, exception = _ORIGINAL["rig_camera_status"](doc, camera_id)
    if status == "validated" or doc.get("acceptance_override") is None:
        return status, why, exception
    problem = override_problem(doc, camera_id)
    if problem is not None:
        return status, [*why, f"acceptance override does not apply: {problem}"], exception
    _COVERED[(doc.get("name"), camera_id)] = summary(doc)
    return STATUS, [], None


def _validate_with_override(camera_id, frames, K, cfg, rig_camera, motion, *, pooled_frames=()):
    """The unchanged per-episode rig-mode validation. Only a camera whose ``STATUS`` this process verified from its
    document is let past the rig-status check; the result names the override."""
    if rig_camera.get("status") != STATUS:
        return _ORIGINAL["validate_episode"](camera_id, frames, K, cfg, rig_camera, motion, pooled_frames=pooled_frames)
    ref = _COVERED.get(((rig_camera.get("provenance") or {}).get("rig"), camera_id))
    if ref is None:  # not verified here: the unchanged check rejects it
        return _ORIGINAL["validate_episode"](camera_id, frames, K, cfg, rig_camera, motion, pooled_frames=pooled_frames)
    res = _ORIGINAL["validate_episode"](camera_id, frames, K, cfg, {**rig_camera, "status": "validated"}, motion,
                                        pooled_frames=pooled_frames)
    res.diagnostics["effective_rig_status"] = STATUS
    res.diagnostics["acceptance_override"] = ref
    return res


def install_rig_override():
    """Install the override-aware rig status / episode validation into ``calib.rig`` for THIS process (the rig-mode
    calibrate stage imports both at call time). Returns a restore function."""
    from s4d.realworld.calib import rig

    if _ORIGINAL:
        raise RuntimeError("acceptance override already installed in this process")
    _ORIGINAL.update(rig_camera_status=rig.rig_camera_status, validate_episode=rig.validate_episode)
    rig.rig_camera_status, rig.validate_episode = _status_with_override, _validate_with_override

    def restore():
        rig.rig_camera_status, rig.validate_episode = _ORIGINAL["rig_camera_status"], _ORIGINAL["validate_episode"]
        _ORIGINAL.clear()
        _COVERED.clear()
    return restore


def _pipeline_rig(argv: list) -> Path:
    import yaml

    if "--pipeline" not in argv:
        raise SystemExit("acceptance_override rwprep: the rwprep argv names no --pipeline")
    cfg = yaml.safe_load(Path(argv[argv.index("--pipeline") + 1]).read_text())
    rig = ((cfg.get("stages") or {}).get("calibrate") or {}).get("rig_calibration")
    if not rig:
        raise SystemExit("acceptance_override rwprep: the pipeline has no rig calibration")
    return Path(rig)


def run_rwprep(argv: list, rwprep=RWPREP) -> None:
    """Phase B worker step: verify the episode's rig document is offline-accepted (fail closed), install the override
    for this process, then run the established ``rwprep_local.py`` in-process with ``argv``."""
    from s4d.realworld.calib.rig import load_rig

    doc, _, _ = load_rig(_pipeline_rig(argv))
    state = offline_acceptance(doc)
    if not state["accepted"]:
        print(json.dumps({"refused": "rig not accepted for offline phase B", "problems": state["problems"]}))
        raise SystemExit(20)
    print(json.dumps({"acceptance_override": state["acceptance_override"], "statuses": state["statuses"]}),
          flush=True)
    restore = install_rig_override()
    saved = sys.argv
    try:
        sys.argv = [str(rwprep), *argv]
        runpy.run_path(str(rwprep), run_name="__main__")
    finally:
        sys.argv = saved
        restore()


def phase_b_binding(document: dict, rig_path) -> dict | None:
    """None for an automatically accepted rig (unchanged compile); the job's ``acceptance_override`` reference for a
    valid override; ValueError otherwise."""
    state = offline_acceptance(document)
    if not state["accepted"]:
        raise ValueError(f"phase B rig cameras must be accepted (automatically or by a valid task acceptance "
                         f"override): {state['problems']}")
    if state["acceptance_override"] is None:
        return None
    return {**state["acceptance_override"], "rig_sha256": sha256_file(rig_path),
            "effective_cameras": state["statuses"]}


def compile_phase_b(argv: list) -> int:
    """``scripts/realworld_prep_register.py --phase B`` (unchanged, execution-scoped) for an override rig: the
    compiler's automatic-acceptance check is answered by ``offline_acceptance`` for this one process, then every
    compiled job is bound to the override and its episode step runs through ``run_rwprep`` (same argv)."""
    import importlib.util

    from ops.jobs.rwprep_mig import atomic, validate
    from s4d.realworld.calib import rig
    from s4d.realworld.calib.rig import load_rig

    if argv[argv.index("--phase") + 1] != "B" or "--rig-map" not in argv:
        raise SystemExit("compile-phase-b compiles phase B with a --rig-map only")
    out = Path(argv[argv.index("--out") + 1]).resolve()
    rigmap = json.loads(Path(argv[argv.index("--rig-map") + 1]).read_text())
    bindings = {}
    for path in sorted(set(rigmap.values())):
        document, _, _ = load_rig(path)
        bindings[str(Path(path).resolve())] = phase_b_binding(document, path)
    if _ORIGINAL:
        raise RuntimeError("acceptance override already installed in this process")
    original = _ORIGINAL["rig_camera_status"] = rig.rig_camera_status

    def compiler_status(doc, cam):
        """The established compiler admits only "validated"; in THIS process an offline-accepted camera (verified
        above, and again here) passes it, and every job compiled for that rig is then marked with the override."""
        status, why, exception = effective_status(doc, cam)
        return ("validated", [], exception) if status in OFFLINE_ACCEPTED else (status, why, exception)

    script = Path(__file__).resolve().parents[3] / "scripts/realworld_prep_register.py"
    spec = importlib.util.spec_from_file_location("realworld_prep_register_ao", script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    saved = sys.argv
    rig.rig_camera_status = compiler_status
    try:
        sys.argv = [str(script), *argv]
        module.main()
    finally:
        sys.argv = saved
        rig.rig_camera_status = original
        _ORIGINAL.clear()
    jobs = json.loads((out / "jobs.json").read_text())
    for job in jobs:
        rig_input = next(p for p in job["inputs"] if p.endswith(".json") and Path(p).with_suffix(
            ".reference.npz").as_posix() in job["inputs"])
        ref = bindings.get(rig_input)
        if ref is None:
            continue
        step = job["steps"][0]
        if step["kind"] != "rwprep" or step["argv"][0] != "episode":
            raise ValueError(f"{job['id']}: unexpected phase B episode step")
        job["steps"][0] = {"kind": "python",
                           "argv": ["-m", "s4d.realworld.calib.acceptance_override", "rwprep", "--", *step["argv"]]}
        job["acceptance_override"] = ref
    config = json.loads((out / "dispatcher.json").read_text())
    validate(config, jobs)
    atomic(out / "jobs.json", jobs)
    print(json.dumps({"acceptance_override_jobs": sum("acceptance_override" in j for j in jobs), "jobs": len(jobs)}))
    return 0


# ------------------------------------------------------------------------------------------------------------ CLI
def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv[:1] == ["rwprep"]:
        rest = argv[1:]
        rwprep = RWPREP
        if rest[:1] == ["--rwprep"]:
            rwprep, rest = rest[1], rest[2:]
        if rest[:1] == ["--"]:
            rest = rest[1:]
        run_rwprep(rest, rwprep)
        return 0
    if argv[:1] == ["compile-phase-b"]:
        return compile_phase_b(argv[1:])
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    g = sub.add_parser("gate", help="offline route gate of a published rig (exit 20 = held)")
    g.add_argument("--rig", required=True)
    g.add_argument("--task", required=True)
    s = sub.add_parser("status", help="offline effective statuses of a rig document")
    s.add_argument("--rig", required=True)
    a = ap.parse_args(argv)
    if a.cmd == "gate":
        try:
            print(json.dumps(offline_gate(a.rig, a.task), indent=1, sort_keys=True))
        except ValueError as exc:
            print(str(exc))
            return 20
        return 0
    from s4d.realworld.calib.rig import load_rig

    doc, _, _ = load_rig(a.rig)
    print(json.dumps(offline_acceptance(doc), indent=1, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
