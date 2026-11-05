"""The user's full-data gate (directive 2026-11-06, item 4): no representation pretraining - and no FQL - for a task
until ALL 100 episodes of that task passed validation + the required preprocessing; joint (two-task) runs wait for both
tasks. No partial data, no warm start.

Source of truth: the persistent ingestion manifest (``/home/ws/ws/realworld-ops/ingest/manifest.json``; per episode:
``validation.status``, blocking ``flags``, ``transfer.status``, and the preprocessing agent's ``preprocessing`` block,
schema ``rw-preprocess`` v1, with per-stage records whose ``input_sha256`` must equal the episode's sha256).
Manifest schema ``rw-ingest-manifest`` v2 (the worker migrates v1 files; this check accepts only v2). A task passes
iff

* the manifest reports the task's ingestion complete (``counts[task].complete``) with target 100,
* exactly ``target`` episodes of the task are valid (``validation.status == "valid"``), with unique episode indices,
  no blocking flag, and the recording verified on the remote (``transfer.status == "verified"``),
* no episode of the task is still pending / incomplete / retrying (the set is final), and
* every valid episode's ``preprocessing.status == "done"`` and each REQUIRED stage is ``done`` for that episode's sha256.

The report lists, per task, the episode keys and the training-store episode ids (manifest key without ``.h5``:
``<task>/episode_000123``), which the trainers compare with the store they read (``baseline_data.check_gate``,
``check_dataset_gate``). Everything fails closed: an unknown schema or missing field is a failure, not a pass.
"""

from __future__ import annotations

import hashlib
import json
import re
import time
from pathlib import Path

MANIFEST = Path("/home/ws/ws/realworld-ops/ingest/manifest.json")
SCHEMA, SCHEMA_VERSION = "rw-ingest-manifest", 2
STAGE_SCHEMA, STAGE_SCHEMA_VERSION = "rw-preprocess", 1
TARGET = 100
# Preprocessing stages the baselines and FQL consume, named as the preprocessing dispatcher records them
# (realworld-ops/pre/tools/rwprep_dispatch.py: the pipeline's stage records + its QA record) plus the compact training
# store (stage ``train_data`` of s4d.realworld.train_store, format branch). DINO is NOT required: SNeRL's DINO
# targets are computed online (data-format policy 2026-11-06). A name that never appears makes the gate fail
# closed; override with --required-stages.
REQUIRED_STAGES = ("normalized", "calibrate", "foundation_stereo_depth", "sam3_masks", "qa", "train_data")
NOT_FINAL = ("pending", "waiting", "incomplete", "retrying")
VALIDATION_STATUSES = NOT_FINAL + ("invalid", "valid")
# ING's blocking flags (runs/setup/tools/realworld_ingest_20261106/ingest_worker.py BLOCKING_FLAGS); an override
# {"decision": "accept"} clears all but source_missing, {"decision": "exclude"} removes the episode; any flag this
# check does not know blocks (fail closed)
BLOCKING_FLAGS = {"duplicate_content", "index_conflict", "source_missing", "source_changed_after_transfer",
                  "remote_conflict", "transfer_failed_repeatedly"}
_SHA = re.compile(r"^[0-9a-f]{64}$")


def store_episode_id(key: str) -> str:
    return key[:-3] if key.endswith(".h5") else key


def _decision(e: dict) -> str | None:
    o = e.get("override")
    return o.get("decision") if isinstance(o, dict) else None


def evaluate(manifest: dict, tasks, *, required_stages=REQUIRED_STAGES, target: int = TARGET) -> dict:
    report = {"passed": False, "checked_at_unix": time.time(), "schema": manifest.get("schema"),
              "required_stages": list(required_stages), "target": target, "tasks": {}}
    if (manifest.get("schema"), manifest.get("schema_version")) != (SCHEMA, SCHEMA_VERSION):
        report["error"] = (f"manifest schema {manifest.get('schema')!r} v{manifest.get('schema_version')!r} != "
                           f"{SCHEMA!r} v{SCHEMA_VERSION}")
        return report
    episodes = manifest.get("episodes") or {}
    all_ok = True
    for task in tasks:
        problems = []
        counts = (manifest.get("counts") or {}).get(task) or {}
        if counts.get("complete") is not True:
            problems.append("ingestion not complete for this task (counts.complete is not true)")
        if counts.get("target") != target:
            problems.append(f"manifest target {counts.get('target')} != {target}")
        mine = {k: e for k, e in episodes.items() if e.get("task") == task}
        valid, indices, excluded = [], set(), []
        for key, e in sorted(mine.items()):
            if _decision(e) == "exclude":
                excluded.append({"key": key, "blocked_by": ["excluded_by_override"]})
                continue  # excluded by the coordinator's override: not part of the task's episode set
            vstatus = (e.get("validation") or {}).get("status")
            if vstatus not in VALIDATION_STATUSES:
                problems.append(f"{key}: unknown validation status {vstatus!r}")
                continue
            if vstatus in NOT_FINAL:
                problems.append(f"{key}: validation {vstatus} (episode set not final)")
                continue
            if vstatus != "valid":
                continue  # invalid recordings are excluded by validation, not counted
            if not (e.get("source") or {}).get("present", True):
                excluded.append({"key": key, "blocked_by": ["source_not_present"]})  # ING counts present files only
                continue
            flags = set(e.get("flags") or {})
            unknown = flags - BLOCKING_FLAGS
            if unknown:  # flags this check does not know: fail closed
                problems.append(f"{key}: unknown flags {sorted(unknown)}")
                continue
            blocking = (flags & {"source_missing"}) if _decision(e) == "accept" else flags
            if blocking:  # ING does not count blocked recordings toward the 100 (counted() / blocked())
                excluded.append({"key": key, "blocked_by": sorted(blocking)})
                continue
            sha = e.get("sha256")
            transfer = e.get("transfer") or {}
            if not (isinstance(sha, str) and _SHA.match(sha)):
                problems.append(f"{key}: no valid sha256")
                continue
            if transfer.get("status") != "verified" or transfer.get("remote_sha256") != sha:
                problems.append(f"{key}: transfer {transfer.get('status')} / remote content not verified equal")
                continue
            idx = e.get("episode_index")
            if not isinstance(idx, int) or idx in indices:
                problems.append(f"{key}: missing or duplicate episode index {idx!r}")
                continue
            indices.add(idx)
            pre = e.get("preprocessing") or {}
            if (pre.get("schema"), pre.get("schema_version")) != (STAGE_SCHEMA, STAGE_SCHEMA_VERSION) \
                    or pre.get("status") != "done":
                problems.append(f"{key}: preprocessing {pre.get('status')!r} (schema {pre.get('schema')!r} "
                                f"v{pre.get('schema_version')!r})")
                continue
            stages = pre.get("stages") or {}
            bad = [s for s in required_stages if (stages.get(s) or {}).get("status") != "done"
                   or (stages.get(s) or {}).get("input_sha256") != sha]
            if bad:
                problems.append(f"{key}: required stages not done for this file: {bad}")
                continue
            valid.append(key)
        if len(valid) != target:
            problems.append(f"{len(valid)} episodes valid + preprocessed, need exactly {target}")
        passed = not problems
        all_ok &= passed
        report["tasks"][task] = {"passed": passed, "valid_and_preprocessed": len(valid), "excluded": excluded,
                                 "problems": problems[:50], "problem_count": len(problems),
                                 "episodes": valid if passed else [],
                                 "store_episodes": [store_episode_id(k) for k in valid] if passed else []}
    report["passed"] = bool(all_ok and tasks)
    return report


def evaluate_file(path: Path = MANIFEST, tasks=("drawer-open", "mug-hanging"), **kw) -> dict:
    raw = Path(path).read_bytes()
    report = evaluate(json.loads(raw), list(tasks), **kw)
    report["manifest"] = {"path": str(path), "sha256": hashlib.sha256(raw).hexdigest()}
    return report


def check_dataset_gate(dataset_path, gate_report, *, steps: int | None, split: dict | None = None,
                       short_run_steps: int = 10_000) -> dict:
    """FQL trainer check (beyond ``short_run_steps``): every task of the dataset passed the gate report, the dataset's
    episodes equal, per task, the report's store episodes, and ``split`` is a disjoint train / validation partition
    of exactly those episodes."""
    import h5py

    report = json.loads(Path(gate_report).read_text()) if isinstance(gate_report, (str, Path)) else gate_report
    with h5py.File(dataset_path, "r") as f:
        eps = json.loads(f.attrs["episodes"])
        tasks: dict[str, list[str]] = {}
        for ep in eps:
            any_tl = next(iter(f["episodes"][ep].values()))
            tasks.setdefault(str(any_tl.attrs["task"]), []).append(ep)
    passed = bool(report) and all((report.get("tasks") or {}).get(t, {}).get("passed") for t in tasks)
    if not passed:
        if steps is None or steps > short_run_steps:
            raise RuntimeError(f"no passed full-data gate report for {sorted(tasks)}; refusing an FQL run of {steps} "
                               "steps")
        return {"smoke": True}
    for task, task_eps in tasks.items():
        if sorted(report["tasks"][task]["store_episodes"]) != sorted(task_eps):
            raise RuntimeError(f"{task}: dataset episodes differ from the gate's validated + preprocessed set")
    if split is not None:
        train, val = set(split.get("train", [])), set(split.get("validation", []))
        if (train & val or (train | val) != set(eps) or len(split["train"]) != len(train)
                or len(split.get("validation", [])) != len(val)):
            raise RuntimeError("the split is not a disjoint train / validation partition of the dataset's episodes")
    return {"smoke": False, "tasks": sorted(tasks)}
