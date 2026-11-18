"""Process-local PNP acceptance extension; the placed drawer reader stays byte-identical.

Only PNP publishers, compilers, workers and offline readers enter installed(). It temporarily registers the explicit
committed PNP record and its three check kinds in the established AO module, verifies exact source/document evidence,
and restores every shared attribute on exit. Existing drawer/native and deployment entrypoints never install it.
"""
from __future__ import annotations

import argparse
import contextlib
import copy
import json
from pathlib import Path
import sys

from s4d.realworld.calib import acceptance_override as base

TASK = "pick-and-place-v2"
RECORDS = {**base.RECORDS, TASK: "pick-and-place-v2-20261118.json"}
CHECKS = (*base.CHECKS, "g1_rotation_spread", "g1_centre_spread", "g5a_registration_motion")
RECORDS_DIR = base.RECORDS_DIR
RECORD_FORMAT, STATUS, REL_TOL = base.RECORD_FORMAT, base.STATUS, base.REL_TOL
OFFLINE_USES, OFFLINE_ACCEPTED = base.OFFLINE_USES, base.OFFLINE_ACCEPTED
_close, _HEX64, _sha_bytes = base._close, base._HEX64, base._sha_bytes
sha256_file, summary, write_once = base.sha256_file, base.summary, base.write_once
_ORIGINALS = {name: getattr(base, name) for name in (
    "record_problem", "camera_problems", "override_problem", "effective_status", "apply_publisher_override",
    "write_once", "install_rig_override")}
_DEPTH = 0
_SAVED = {}




def _record_problem(rec: dict, task: str) -> str | None:
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
            isinstance(v, dict) and (v.get("gates") or
                (auto.get("decision", {}).get("cameras", {}).get(c) == "validated" and v.get("gates") == []))
            and all(isinstance(g, str) for g in v["gates"])
            for c, v in cams.items()):
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

def record_problem(rec, task):
    return _record_problem(rec, task) if task == TASK else _ORIGINALS['record_problem'](rec, task)


def document_evidence_sha256(doc: dict) -> str:
    """Stable identity of the PNP evidence, excluding publisher paths, document name and override metadata."""
    evidence = {k: doc.get(k) for k in ("task", "episodes", "reference_episode", "gates", "criteria",
                "fixedness_config", "fixedness_evidence", "fixedness_exception", "episode_checks", "robot_model")}
    evidence["cameras"] = {c: {k: v for k, v in row.items() if k != "provenance"}
                           for c, row in doc["cameras"].items()}
    return _sha_bytes(json.dumps(evidence, sort_keys=True, allow_nan=False).encode())

def document_problems(rec: dict, doc: dict) -> list:
    if rec.get("task") != "pick-and-place-v2":
        return []  # drawer's established evidence rule stays unchanged
    if doc.get("reference_sha256", rec.get("reference_sha256")) != rec.get("reference_sha256"):
        return ["reference arrays differ from the recorded pick-and-place-v2 evidence"]
    if document_evidence_sha256(doc) != rec.get("document_evidence_sha256"):
        return ["document automatic evidence differs from the recorded pick-and-place-v2 evidence"]
    return []

def _exception_problems(e, v2, entry):
    from s4d.realworld.calib import criteria as C
    out, tag = [], e.get('id')
    if e["check"] in ("g1_rotation_spread", "g1_centre_spread"):
        from s4d.realworld.calib.dw.publish import GATES

        metric = "max_pair_rot_deg" if e["check"] == "g1_rotation_spread" else "max_pair_centre_m"
        threshold = "spread_rot_deg" if e["check"] == "g1_rotation_spread" else "spread_centre_m"
        if e.get("metric") != metric or GATES[threshold] != e["threshold"]:
            out.append(f"{tag}: G1 spread threshold differs from the record")
        if not _close(((entry or {}).get("empirical_spread") or {}).get(metric), e["measured"]):
            out.append(f"{tag}: G1 spread differs from the record")
        if e["reason"] not in ((v2.get("release1") or {}).get("reasons") or []):
            out.append(f"{tag}: release/1 reason differs from the record")
    elif e["check"] == "g5a_registration_motion":
        from dataclasses import asdict
        from s4d.realworld.calib.rig import FixednessConfig

        reg = (((entry or {}).get("validation") or {}).get("fixedness") or {}).get("registration") or {}
        if reg != e.get("registration") or len(reg.get("moved", [])) != e["measured"]:
            out.append(f"{tag}: G5a registration episode sets differ from the record")
        if e["threshold"] != 0 or any(asdict(FixednessConfig()).get(k) != v
                                      for k, v in e.get("thresholds", {}).items()):
            out.append(f"{tag}: G5a thresholds differ from the unchanged criteria")
        if e["reason"] not in (v2.get("reasons") or []):
            out.append(f"{tag}: G5a reason differs from the record")
    else:
        out.append(f"{tag}: unknown check {e['check']!r}")
    return out

def _camera_problems(rec: dict, cam: str, entry: dict) -> list:
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
        out += [f"{cam}: {p}" for p in _exception_problems(e, v2, entry)]
    return out

def camera_problems(rec, cam, entry):
    if rec.get('task') != TASK:
        return _ORIGINALS['camera_problems'](rec, cam, entry)
    if rec['automatic_acceptance']['decision']['cameras'].get(cam) == 'validated':
        return [] if entry.get('status') == 'validated' else ['automatically validated camera drifted']
    return _camera_problems(rec, cam, entry)


def _override_problem(doc, cam):
    problem = _ORIGINALS['override_problem'](doc, cam)
    if problem is not None or doc.get('task') != TASK:
        return problem
    found = base.record_for(TASK)
    return '; '.join(document_problems(found[0], doc)) or None


def _effective_status(doc, cam):
    if doc.get('task') == TASK and doc.get('acceptance_override') is not None:
        problem = _override_problem(doc, cam)
        if problem is not None:
            return 'held', [f'acceptance override does not apply: {problem}'], None
    return _ORIGINALS['effective_status'](doc, cam)


def _publisher(doc, **kw):
    if kw['task'] == TASK:
        found = base.record_for(TASK)
        if found is not None:
            rec, _, _ = found
            problems = document_problems(rec, doc)
            work = Path(kw['work'])
            for name, key in (('fixedness.json', 'fixedness_sha256'), ('static.json', 'static_eval_sha256')):
                if sha256_file(work / name) != rec['provenance'][key]:
                    problems.append(f'{name} differs from the recorded evidence')
            if sha256_file(work / 'fixedness.reference.npz') != rec['reference_sha256']:
                problems.append('reference arrays differ from the recorded evidence')
            if problems:
                raise ValueError('rig held by established automatic criteria; acceptance override does not apply: '
                                 + '; '.join(problems))
    return _ORIGINALS['apply_publisher_override'](doc, **kw)


def _write_once(path, doc):
    if doc.get('format') == base.DECISION_FORMAT and doc.get('task') == TASK:
        doc = {**doc, 'verified': [
            'automatic decision / verdict / spec / FREEZE sha256 equal the record',
            'criteria commit equals the record', '100 episodes and 3 cameras retained',
            'complete automatic document evidence equals the recorded identity',
            'release/1 G1 diameters and exact G5a episode sets / fits equal the record',
            'release/2 G1 passes unchanged; only recorded G5a reasons are overridden',
            'thresholds equal the record and the unchanged criteria constants']}
    return _ORIGINALS['write_once'](path, doc)


def _fixedness_with_override(original, doc, cam, episode, motion, cfg):
    """Waive only an exactly recorded PNP G5a fit; keep G5b and all other episode checks."""
    result = original(doc, cam, episode, motion, cfg)
    if doc.get("task") != "pick-and-place-v2" or _override_problem(doc, cam) is not None:
        return result
    from dataclasses import asdict

    rec = doc["acceptance_override"]["record"]
    exc = next((e for e in rec["exceptions"] if e["check"] == "g5a_registration_motion"
                and cam in e["cameras"]), None)
    if exc is None:
        return result
    alias = rec["episode_aliases"].get(episode)
    recorded = exc["motions"].get(alias)
    if asdict(cfg) != rec["fixedness_config"]:
        return {**result, "fixed": False, "reason": "G5a config differs from the recorded override"}
    moved = result["registration"]["moved"]
    if recorded is not None and (not moved or recorded != motion):
        return {**result, "fixed": False, "reason": "recorded G5a fit drifted"}
    if moved and (recorded is None or asdict(cfg) != rec["fixedness_config"]):
        return {**result, "fixed": False, "reason": "G5a motion/episode/config differs from the recorded override"}
    ev = result.get("evidence")
    if ev is None or ev.get("epipolar") == "moved":
        return result
    return {**result, "fixed": True, "reason": None, "acceptance_override": summary(doc),
            "notes": [*result.get("notes", []), "Recorded G5a evidence accepted for offline use by task override"]}

def _install_rig_override():
    from s4d.realworld.calib import rig
    from functools import partial

    original = rig.release_fixedness
    restore = _ORIGINALS['install_rig_override']()
    rig.release_fixedness = partial(_fixedness_with_override, original)

    def undo():
        rig.release_fixedness = original
        restore()
    return undo


@contextlib.contextmanager
def installed():
    """Install for one PNP operation in this process; nested offline operations share the same installation."""
    global _DEPTH
    if _DEPTH == 0:
        patches = dict(RECORDS=dict(RECORDS), CHECKS=CHECKS, RECORDS_DIR=RECORDS_DIR,
                       record_problem=record_problem, camera_problems=camera_problems,
                       override_problem=_override_problem, effective_status=_effective_status,
                       apply_publisher_override=_publisher, write_once=_write_once,
                       install_rig_override=_install_rig_override)
        _SAVED.update({name: getattr(base, name) for name in patches})
        for name, value in patches.items():
            setattr(base, name, value)
    _DEPTH += 1
    try:
        yield base
    finally:
        _DEPTH -= 1
        if _DEPTH == 0:
            for name, value in _SAVED.items():
                setattr(base, name, value)
            _SAVED.clear()


def record_for(task):
    with installed():
        return base.record_for(task)


def effective_status(doc, cam):
    with installed():
        return base.effective_status(doc, cam)


def offline_acceptance(doc):
    with installed():
        return base.offline_acceptance(doc)


def apply_publisher_override(doc, **kw):
    with installed():
        return base.apply_publisher_override(doc, **kw)


def offline_gate(path, task):
    with installed():
        return base.offline_gate(path, task)


def load_rig_session_offline(path, camera_ids, K_native):
    with installed():
        return base.load_rig_session_offline(path, camera_ids, K_native)


def install_rig_override():
    context = installed()
    context.__enter__()
    try:
        restore = base.install_rig_override()
    except BaseException:
        context.__exit__(*sys.exc_info())
        raise

    def undo():
        try:
            restore()
        finally:
            context.__exit__(None, None, None)
    return undo


def phase_b_episode_step(step):
    out = base.phase_b_episode_step(step)
    out['argv'][1] = 's4d.realworld.calib.pnpv2_acceptance_override'
    return out


def main(argv=None):
    with installed():
        return base.main(argv)


if __name__ == '__main__':
    raise SystemExit(main())
