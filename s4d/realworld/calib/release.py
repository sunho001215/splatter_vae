"""``calibration_validated`` records of a released per-task rig calibration (calibration release 2026-11-08).

The rwlearn gate (``s4d.rwlearn.gate.calibration_record_problem``) releases geometry-dependent consumers of an episode
only with a manifest stage record ``calibration_validated`` (status done, ``input_sha256`` = the episode sha256,
``output_format`` ``rw-calibration-validation/<n>``, ``metrics.decision == "accepted"``, ``metrics.calibrate_build_id``
= the episode's CURRENT calibrate build id, ``metrics.cameras`` = the validated camera serials). This module builds
those records from the rig-mode calibrate outputs and writes them only through the ingest worker's ``set_stage``.

Two steps, both refusing on any failed check (fail closed, per task: one failing episode -> no record for the task):

* ``plan`` (run where the processed roots are readable, e.g. a remote container with read-only mounts): for every
  valid episode of the task in the manifest, the calibrate record named by the manifest (``stages.calibrate``:
  status done, same sha256, ``metrics.build_id``, ``outputs`` = episode dir) must be a rig-mode build with EXACTLY the
  published release document (its input hash), its build id must equal the manifest's, and every camera of the
  document must be ``accepted`` with the document's pose, depth scales and robot model; the document's cameras must
  be effectively validated (a G5a-held camera only with a valid fixedness exception). Output: a JSON plan.
* ``apply`` (where the manifest lives): DRY RUN by default (prints what would be written); ``--apply`` re-checks
  every episode's current calibrate build id and sha256 in the live manifest against the plan and then calls
  ``ingest_worker.set_stage(state_dir, key, "calibration_validated", record)`` per episode.

    python -m s4d.realworld.calibration_release plan --manifest M.json --rig drawer-open=rig-loma-drawer-open.json \
        [--path-map /remote/prefix=/local/prefix] --out plan.json

Recording-period documents (``calib.loma.publish split-periods``): ``--rig T=A --rig T=B`` gives a task several
documents; they must be the period documents of ONE published document, their periods partition exactly the task's
valid episodes, and each episode is checked against the document of its period (its calibrate record must have been
built with exactly that document). A single document keeps the previous behaviour.
    python -m s4d.realworld.calibration_release apply --plan plan.json --state-dir /home/ws/ws/realworld-ops/ingest \
        [--apply]
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import subprocess
from pathlib import Path

import numpy as np

OUTPUT_FORMAT = "rw-calibration-validation/1"
CODE_VERSION = "calibration_validated@1+rig_release"
STAGE = "calibration_validated"
INGEST_WORKER = Path(__file__).resolve().parents[3] / "runs/setup/tools/realworld_ingest_20261106/ingest_worker.py"


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _code_commit() -> str:
    try:
        return subprocess.run(["git", "-C", str(Path(__file__).resolve().parents[1]), "rev-parse", "HEAD"],
                              capture_output=True, text=True, timeout=30).stdout.strip() or "unknown"
    except Exception:  # noqa: BLE001
        return "unknown"


def _mapped(path: str, path_map) -> Path:
    for src, dst in path_map:
        if path.startswith(src):
            return Path(dst + path[len(src):])
    return Path(path)


def _valid_entries(manifest: dict, task: str) -> dict:
    """Manifest entries of ``task`` the gate counts (valid, not excluded by override)."""
    out = {}
    for key, e in sorted((manifest.get("episodes") or {}).items()):
        if e.get("task") != task:
            continue
        o = e.get("override")
        if isinstance(o, dict) and o.get("decision") == "exclude":
            continue
        if (e.get("validation") or {}).get("status") == "valid":
            out[key] = e
    return out


def check_episode(key: str, entry: dict, rig_doc: dict, rig_sha: str, path_map=()) -> tuple[dict | None, list]:
    """(record or None, problems) of one manifest episode."""
    from s4d.realworld import io as rio
    from s4d.realworld.calib.rig import rig_camera_status

    problems = []
    sha = entry.get("sha256")
    cal = ((entry.get("preprocessing") or {}).get("stages") or {}).get("calibrate") or {}
    bid = (cal.get("metrics") or {}).get("build_id")
    if cal.get("status") != "done" or cal.get("input_sha256") != sha or not bid or not cal.get("outputs"):
        return None, [f"{key}: manifest calibrate record not done for this file (status {cal.get('status')!r}, "
                      f"build id {bid!r})"]
    stage_dir = _mapped(str(cal["outputs"]), path_map) / "derived" / "calibrate"
    rec_path = stage_dir / rio.RECORD  # the stage's completion record as run_stage writes it (_COMPLETE.json)
    if not rec_path.is_file():
        return None, [f"{key}: no calibrate record at {rec_path}"]
    rec = json.loads(rec_path.read_text())
    if rec.get("build_id") != bid:
        problems.append(f"{key}: calibrate record build {rec.get('build_id')!r} != manifest {bid!r}")
    rc = (rec.get("config") or {}).get("rig_calibration") or {}
    if (rec.get("inputs") or {}).get("rig_calibration") != rig_sha or rc.get("name") != rig_doc.get("name"):
        problems.append(f"{key}: calibrate record was not built with the release document {rig_doc.get('name')!r} "
                        f"(input {str((rec.get('inputs') or {}).get('rig_calibration'))[:12]} != {rig_sha[:12]})")
    if (rec.get("inputs") or {}).get("rig_reference") != rig_doc.get("reference_sha256"):
        problems.append(f"{key}: calibrate record reference arrays differ from the document's")
    cams, per_cam = [], {}
    for cam, entry_cam in sorted(rig_doc["cameras"].items()):
        status, why, exception = rig_camera_status(rig_doc, cam)
        if status != "validated":
            problems.append(f"{key}: release document camera {cam} is {status}: {why}")
            continue
        p = stage_dir / f"{cam.replace('/', '__')}.json"
        if not p.is_file():
            problems.append(f"{key}: no calibrate output for {cam}")
            continue
        out = json.loads(p.read_text())
        diag = out.get("diagnostics") or {}
        if out.get("status") != "accepted":
            problems.append(f"{key}: {cam} {out.get('status')}: {out.get('reasons')}")
            continue
        if out.get("init_method") != "rig_shared" or (diag.get("release") or {}).get("rig") != rig_doc.get("name"):
            problems.append(f"{key}: {cam} is not a rig-mode result of {rig_doc.get('name')!r}")
            continue
        if not np.allclose(np.asarray(out["T_cam_base"], float), np.asarray(entry_cam["T_cam_base"], float),
                           atol=1e-12, rtol=0):
            problems.append(f"{key}: {cam} pose differs from the release document")
        if out.get("depth_scale") != entry_cam.get("depth_scale"):
            problems.append(f"{key}: {cam} depth scales differ from the release document")
        if out.get("robot_model") != rig_doc.get("robot_model"):
            problems.append(f"{key}: {cam} robot model differs from the release document")
        fx = diag.get("fixedness") or {}
        if fx.get("fixed") is not True:
            problems.append(f"{key}: {cam} fixedness not verified: {fx.get('reason')}")
        cams.append(cam.split("/")[0])
        per_cam[cam.split("/")[0]] = {
            "fixedness": {"rule": fx.get("rule"), "registration": fx.get("registration"),
                          "evidence": fx.get("evidence"), "exception_applied": fx.get("exception") is not None,
                          "notes": fx.get("notes")},
            "heldout_depth_abs_median_m": diag.get("heldout_depth_abs_median_m"),
            "explained_fraction_heldout": diag.get("explained_fraction_heldout"),
            "depth_scale_applied": diag.get("depth_scale_applied")}
    if problems:
        return None, problems
    if len(cams) != len(rig_doc["cameras"]):
        return None, [f"{key}: {len(cams)} of {len(rig_doc['cameras'])} cameras validated"]
    exception = (rig_doc.get("fixedness_exception") or {})
    metrics = {"decision": "accepted", "calibrate_build_id": bid, "cameras": sorted(cams),
               "gate_version": rig_doc.get("gate_version"), "method": rig_doc.get("method"),
               "shared_calibration": {"name": rig_doc.get("name"), "sha256": rig_sha,
                                      "reference_sha256": rig_doc.get("reference_sha256")},
               "evidence": {"frozen_commit": (rig_doc.get("provenance") or {}).get("frozen_commit"),
                            "verdict_sha256": ((rig_doc.get("provenance") or {}).get("verdict") or {}).get("sha256"),
                            "freeze_record_sha256": ((rig_doc.get("provenance") or {}).get("freeze_record") or {})
                            .get("sha256"),
                            "user_decision": (rig_doc.get("provenance") or {}).get("user_decision"),
                            "calibrate_fingerprint": rec.get("fingerprint"),
                            "fixedness_exception": ({k: exception.get(k) for k in ("cameras", "decision", "evidence",
                                                                                   "decided_by", "decided_at",
                                                                                   "reference")}
                                                    if exception.get("decision") else None)},
               "depth_scale": {c.split("/")[0]: v["depth_scale"] for c, v in rig_doc["cameras"].items()},
               "robot_model": rig_doc.get("robot_model"), "per_camera": per_cam}
    config_sha = _sha(json.dumps({"gate_version": rig_doc.get("gate_version"), "rig_sha256": rig_sha},
                                 sort_keys=True).encode())
    record = {"status": "done", "code_version": CODE_VERSION, "stage_schema_version": 1,
              "output_format": OUTPUT_FORMAT, "input_sha256": sha, "config_sha256": config_sha,
              "outputs": str(cal["outputs"]), "metrics": metrics,
              "notes": f"rig-mode calibrate with {rig_doc.get('name')} (one fixed rig per task, no per-episode pose "
                       "refinement); written by s4d.realworld.calibration_release"}
    return record, []


def period_problems(docs: list, task: str) -> list:
    """Problems of several documents given for one task: they must be the period documents (``period``) of one
    published document with disjoint episode sets."""
    periods = [d.get("period") for d, _ in docs]
    if not all(isinstance(p, dict) and p.get("name") and p.get("episodes") for p in periods):
        return [f"{task}: several documents but not all are recording-period documents"]
    problems = []
    if len({json.dumps(p.get("derived_from"), sort_keys=True) for p in periods}) != 1:
        problems.append(f"{task}: the period documents derive from different published documents")
    if sorted(p["name"] for p in periods) != sorted(periods[0].get("periods") or {}):
        problems.append(f"{task}: the period documents are not exactly the periods {sorted(periods[0].get('periods'))}")
    eps = [e for p in periods for e in p["episodes"]]
    if len(eps) != len(set(eps)):
        problems.append(f"{task}: an episode belongs to two period documents")
    return problems


def plan(manifest: dict, rigs: dict, *, path_map=(), code_commit: str | None = None) -> dict:
    """{"tasks": {task: {"refused", "problems", "records": {key: record}}}} - records only for tasks without any
    problem. ``rigs``: {task: path of the release document, or a list of a task's recording-period documents}."""
    from s4d.realworld.calib.rig import load_rig

    commit = code_commit or _code_commit()
    out = {"stage": STAGE, "output_format": OUTPUT_FORMAT, "code_commit": commit, "tasks": {}}
    for task, paths in rigs.items():
        paths = [paths] if isinstance(paths, (str, Path)) else list(paths)
        docs = [(doc, fp["rig_calibration"]) for doc, _, fp in map(load_rig, paths)]  # refuses unknown / mismatched
        problems, records = [], {}
        for doc, _ in docs:
            if doc.get("task") != task:
                problems.append(f"document task {doc.get('task')!r} != {task!r}")
            if not doc.get("gate_version") or not doc.get("fixedness_rule"):
                problems.append("not a release document (gate_version / fixedness_rule missing)")
        if len(docs) > 1:
            problems += period_problems(docs, task)
        entries = _valid_entries(manifest, task)
        if not entries:
            problems.append("no valid manifest episodes")
        for key, entry in entries.items():
            doc, sha = docs[0]
            if len(docs) > 1:  # the episode's period document (the calibrate output dir is named by the episode id)
                outputs = (((entry.get("preprocessing") or {}).get("stages") or {}).get("calibrate") or {}).get("outputs")
                eid = Path(str(outputs)).name if outputs else None
                mine = [(d, s) for d, s in docs if eid in d["period"]["episodes"]]
                if len(mine) != 1:
                    problems.append(f"{key}: episode {eid!r} is in {len(mine)} period documents (need exactly 1)")
                    continue
                doc, sha = mine[0]
            rec, why = check_episode(key, entry, doc, sha, path_map)
            problems += why
            if rec is not None:
                if doc.get("period"):
                    rec["metrics"]["period"] = {"name": doc["period"]["name"],
                                                "derived_from": doc["period"].get("derived_from")}
                records[key] = {**rec, "code_commit": commit}
        refused = bool(problems)
        rig = [{"path": str(p), "name": d.get("name"), "sha256": s} for p, (d, s) in zip(paths, docs)]
        out["tasks"][task] = {"refused": refused, "rig": rig[0] if len(rig) == 1 else rig,
                              "episodes": len(entries), "problems": problems[:200], "problem_count": len(problems),
                              "records": {} if refused else records}
    return out


def _load_set_stage():
    spec = importlib.util.spec_from_file_location("rw_ingest_worker", INGEST_WORKER)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.set_stage


def apply(plan_doc: dict, state_dir, *, write: bool = False, set_stage=None) -> dict:
    """Dry run unless ``write``. Refuses (writes nothing for the task) when the live manifest disagrees with the plan
    (sha256 or current calibrate build id) or the task was refused."""
    state_dir = Path(state_dir)
    manifest = json.loads((state_dir / "manifest.json").read_text())
    report = {"write": bool(write), "tasks": {}}
    for task, t in plan_doc["tasks"].items():
        if t["refused"] or not t["records"]:
            report["tasks"][task] = {"written": 0, "refused": f"plan refused: {t['problems'][:3]}"}
            continue
        stale = []
        for key, rec in t["records"].items():
            e = (manifest.get("episodes") or {}).get(key) or {}
            cur = ((((e.get("preprocessing") or {}).get("stages") or {}).get("calibrate") or {}).get("metrics")
                   or {}).get("build_id")
            if e.get("sha256") != rec["input_sha256"] or cur != rec["metrics"]["calibrate_build_id"]:
                stale.append(key)
        if stale:
            report["tasks"][task] = {"written": 0, "refused": f"manifest changed since the plan: {stale[:5]}"}
            continue
        written = []
        if write:
            set_stage = set_stage or _load_set_stage()
            for key, rec in t["records"].items():
                set_stage(state_dir, key, STAGE, rec)
                written.append(key)
        report["tasks"][task] = {"written": len(written), "would_write": len(t["records"]),
                                 "example": next(iter(t["records"].items()))}
    return report


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("plan")
    p.add_argument("--manifest", required=True)
    p.add_argument("--rig", action="append", required=True, help="task=path of the release document")
    p.add_argument("--path-map", action="append", default=[], help="prefix=replacement for manifest output paths")
    p.add_argument("--out", required=True)
    a_ = sub.add_parser("apply")
    a_.add_argument("--plan", required=True)
    a_.add_argument("--state-dir", required=True)
    a_.add_argument("--apply", action="store_true", help="write through ingest_worker.set_stage (default: dry run)")
    a = ap.parse_args(argv)
    if a.cmd == "plan":
        out = Path(a.out)
        if out.exists():
            raise SystemExit(f"{out} exists")
        rigs = {}
        for item in a.rig:  # repeated task=path: the task's recording-period documents
            task, path = item.split("=", 1)
            rigs.setdefault(task, []).append(path)
        doc = plan(json.loads(Path(a.manifest).read_text()), {t: p[0] if len(p) == 1 else p for t, p in rigs.items()},
                   path_map=[tuple(m.split("=", 1)) for m in a.path_map])
        out.write_text(json.dumps(doc, indent=1))
        for task, t in doc["tasks"].items():
            print(task, "REFUSED" if t["refused"] else f"{len(t['records'])} records", t["problems"][:5])
        return 1 if any(t["refused"] for t in doc["tasks"].values()) else 0
    rep = apply(json.loads(Path(a.plan).read_text()), a.state_dir, write=a.apply)
    print(json.dumps({t: {k: v for k, v in r.items() if k != "example"} for t, r in rep["tasks"].items()}, indent=1))
    return 0 if all("refused" not in r for r in rep["tasks"].values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
