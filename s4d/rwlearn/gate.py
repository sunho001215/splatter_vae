"""The user's full-data gate (directive 2026-11-06, item 4): no representation pretraining - and no FQL - for a task
until ALL 100 episodes of that task passed validation + the required preprocessing; each task is gated alone (task-specific
encoders only). No partial data, no warm start.

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

CALIBRATION HOLD (user decision 2026-11-06 ~11:20, item 2: only geometry-dependent outputs and their training
eligibility are held until the calibration is validated). Consumers are classified in ``CONSUMERS``:

* GEOMETRY-DEPENDENT (inputs or targets use the camera-to-robot extrinsics, world-frame tracks or depth fused through
  them): our method, SNeRL, 3D-MVP, SinCro, and every FQL condition on their encoders. They need, per episode and in
  addition to the 100-episode gate above, the stage record ``calibration_validated`` (``CALIBRATION_STAGE``) with
  ``status == "done"``, ``input_sha256`` = the episode sha256, ``output_format`` ``rw-calibration-validation/<n>``,
  ``metrics.decision == "accepted"``, ``metrics.calibrate_build_id`` equal to the episode's CURRENT ``calibrate``
  record build id (``stages.calibrate.metrics.build_id``: the validated calibration is the one downstream stages were
  built from) and ``metrics.cameras`` (the validated camera serials). The report section ``calibration_hold`` carries
  the per-task verdict and, per store episode, the validated calibrate build id and cameras; the trainers then check
  that the training store was built from exactly that calibration (its ``upstream.calibrate`` attribute) and that every
  store camera is validated (``check_store_calibration``).
* NON-GEOMETRIC (RGB, timestamps, proprioception and recorded commands only): ReViWo, DynaMo, the end-to-end CNN + FQL
  and the FQL conditions on their encoders. They need only the gate above.

The hold stage never blocks non-geometric consumers: the aggregate "all recorded stages done" check skips
``HOLD_STAGES`` (``set_stage`` folds every recorded stage into ``preprocessing.status``, so a pending or failed
``calibration_validated`` record would otherwise hold every consumer). The 100-episode requirement is unchanged.
"""

from __future__ import annotations

import hashlib
import json
import re
import time
from dataclasses import dataclass
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

# ---------------------------------------------------------------------------------------------- calibration hold
CALIBRATION_STAGE = "calibration_validated"
CALIBRATION_OUTPUT_FORMAT = "rw-calibration-validation/"  # + layout version, e.g. rw-calibration-validation/1
HOLD_STAGES = (CALIBRATION_STAGE,)  # never part of the aggregate check of non-geometric consumers


@dataclass(frozen=True)
class Consumer:
    geometry: bool
    uses: str
    reason: str


CONSUMERS = {
    "ours": Consumer(True, "dynamic 3D tracks in the robot base frame, depth fused across cameras, camera poses",
                     "tracks are lifted to the base frame and depth is fused with the camera-to-robot extrinsics "
                     "(pretraining deferred until the architecture is final)"),
    "snerl": Consumer(True, "3 RGB views + normalised camera poses (encoder input); RGB / semantic / DINO targets "
                            "rendered along rays", "the encoder input poses and every rendered ray come from the "
                                               "extrinsics; a wrong calibration changes inputs and targets"),
    "mvp3d": Consumer(True, "RGB-D back-projected into the robot base frame, base-frame scene bounds",
                      "the fused point cloud and its bounds are built with the extrinsics"),
    "sincro": Consumer(True, "RGB windows + per-sample K / c2w rays for its rendering loss",
                       "the rendering targets are cast along rays from the extrinsics (sincro_forward_per_sample_"
                       "cameras); its encoder is therefore trained on calibration-dependent targets"),
    "reviwo": Consumer(False, "synchronised multi-view RGB (timestamps only)",
                       "no camera pose, depth or track enters inputs or targets"),
    "dynamo": Consumer(False, "single-camera RGB videos", "no camera pose, depth or track enters inputs or targets"),
    "e2e_cnn": Consumer(False, "single-view RGB history + proprioception + recorded commands",
                        "pixels, the robot's own kinematics and the sent commands only"),
}
# offline-FQL conditions inherit the class of the encoder they are trained on (s4d/rwlearn/fql_data.py CONDITIONS)
FQL_CONSUMER = {"dynamo": "dynamo", "reviwo": "reviwo", "sincro": "sincro", "snerl": "snerl",
                "mvp3d_multiview": "mvp3d", "mvp3d_single": "mvp3d", "e2e_cnn": "e2e_cnn"}


def consumer_info(name: str | None) -> Consumer:
    """An unknown or missing consumer is treated as geometry-dependent (fail closed)."""
    return CONSUMERS.get(name) or Consumer(True, "unknown", "unclassified consumer: held like a geometric one")


def calibration_record_problem(stages: dict, sha: str) -> str | None:
    """Why the episode's ``calibration_validated`` record does not release geometry-dependent consumers (None = it
    does)."""
    rec = stages.get(CALIBRATION_STAGE)
    if not rec:
        return "no calibration_validated record"
    if rec.get("status") != "done":
        return f"calibration_validated {rec.get('status')!r}"
    if rec.get("input_sha256") != sha:
        return "calibration_validated input_sha256 != episode sha256"
    if not str(rec.get("output_format") or "").startswith(CALIBRATION_OUTPUT_FORMAT):
        return f"calibration_validated output_format {rec.get('output_format')!r} is not {CALIBRATION_OUTPUT_FORMAT}<n>"
    m = rec.get("metrics") or {}
    if m.get("decision") != "accepted":
        return f"calibration_validated decision {m.get('decision')!r} != 'accepted'"
    current = ((stages.get("calibrate") or {}).get("metrics") or {}).get("build_id")
    if not current or m.get("calibrate_build_id") != current:
        return (f"validated calibrate build {m.get('calibrate_build_id')!r} is not the current calibrate build "
                f"{current!r}")
    cams = m.get("cameras")
    if not (isinstance(cams, list) and cams and all(isinstance(c, str) for c in cams)):
        return "calibration_validated metrics.cameras must list the validated camera serials"
    return None


def store_episode_id(key: str) -> str:
    return key[:-3] if key.endswith(".h5") else key


def _decision(e: dict) -> str | None:
    o = e.get("override")
    return o.get("decision") if isinstance(o, dict) else None


def evaluate(manifest: dict, tasks, *, required_stages=REQUIRED_STAGES, target: int = TARGET) -> dict:
    report = {"passed": False, "geometry_passed": False, "checked_at_unix": time.time(),
              "schema": manifest.get("schema"), "required_stages": list(required_stages),
              "calibration_stage": CALIBRATION_STAGE, "target": target, "tasks": {}, "calibration_hold": {},
              "consumers": {k: {"geometry_dependent": c.geometry, "uses": c.uses, "reason": c.reason}
                            for k, c in CONSUMERS.items()}}
    if (manifest.get("schema"), manifest.get("schema_version")) != (SCHEMA, SCHEMA_VERSION):
        report["error"] = (f"manifest schema {manifest.get('schema')!r} v{manifest.get('schema_version')!r} != "
                           f"{SCHEMA!r} v{SCHEMA_VERSION}")
        return report
    episodes = manifest.get("episodes") or {}
    all_ok = geo_ok = True
    for task in tasks:
        problems, hold_problems, hold_ok = [], [], {}
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
            stages = pre.get("stages") or {}
            # aggregate: every recorded stage done (or skipped) EXCEPT the hold stages, which only gate geometry-
            # dependent consumers (set_stage's preprocessing.status folds them in)
            open_stages = sorted(s for s, r in stages.items() if s not in HOLD_STAGES
                                 and (r or {}).get("status") not in ("done", "skipped"))
            # an aggregate other than "done" must be explained by an open hold stage (fail closed otherwise)
            explained = pre.get("status") == "done" or any(
                (stages.get(h) or {}).get("status") not in ("done", "skipped") for h in HOLD_STAGES if h in stages)
            if (pre.get("schema"), pre.get("schema_version")) != (STAGE_SCHEMA, STAGE_SCHEMA_VERSION) \
                    or not stages or open_stages or not explained:
                problems.append(f"{key}: preprocessing {pre.get('status')!r} (schema {pre.get('schema')!r} "
                                f"v{pre.get('schema_version')!r}; stages not done: {open_stages})")
                continue
            bad = [s for s in required_stages if (stages.get(s) or {}).get("status") != "done"
                   or (stages.get(s) or {}).get("input_sha256") != sha]
            if bad:
                problems.append(f"{key}: required stages not done for this file: {bad}")
                continue
            valid.append(key)
            why = calibration_record_problem(stages, sha)
            if why:
                hold_problems.append(f"{key}: {why}")
            else:
                m = stages[CALIBRATION_STAGE]["metrics"]
                hold_ok[store_episode_id(key)] = {"calibrate_build_id": m["calibrate_build_id"],
                                                  "cameras": sorted(m["cameras"])}
        if len(valid) != target:
            problems.append(f"{len(valid)} episodes valid + preprocessed, need exactly {target}")
        passed = not problems
        all_ok &= passed
        report["tasks"][task] = {"passed": passed, "valid_and_preprocessed": len(valid), "excluded": excluded,
                                 "problems": problems[:50], "problem_count": len(problems),
                                 "episodes": valid if passed else [],
                                 "store_episodes": [store_episode_id(k) for k in valid] if passed else []}
        hold_passed = passed and not hold_problems and len(hold_ok) == target
        geo_ok &= hold_passed
        report["calibration_hold"][task] = {
            "passed": hold_passed, "validated": len(hold_ok), "problems": hold_problems[:50],
            "problem_count": len(hold_problems), "episodes": hold_ok if hold_passed else {}}
    report["passed"] = bool(all_ok and tasks)
    report["geometry_passed"] = bool(report["passed"] and geo_ok)
    return report


def task_passed(report: dict, task: str, consumer: str | None) -> bool:
    """Whether ``consumer`` may train on ``task``: the 100-episode gate, plus the calibration hold for geometry-
    dependent consumers."""
    base = bool(((report or {}).get("tasks") or {}).get(task, {}).get("passed"))
    if not consumer_info(consumer).geometry:
        return base
    return base and bool(((report or {}).get("calibration_hold") or {}).get(task, {}).get("passed"))


def task_problems(report: dict, task: str, consumer: str | None, limit: int = 5) -> list[str]:
    t = ((report or {}).get("tasks") or {}).get(task)
    if t is None:
        return ["not evaluated"]
    out = list(t.get("problems", []))
    if consumer_info(consumer).geometry:
        out += ((report.get("calibration_hold") or {}).get(task) or {}).get("problems", ["calibration hold not "
                                                                                          "evaluated"])
    return out[:limit]


def check_store_calibration(report: dict, task_episodes: dict, upstream, cameras) -> None:
    """Geometry-dependent trainers: every store episode was built from the VALIDATED calibration (its
    ``upstream["calibrate"]`` build id equals the gate's) and every store camera is a validated camera.
    ``upstream(ep)`` / ``cameras(ep)`` read the store (``FrameStore.upstream`` is optional; a store without it cannot
    prove its calibration and is refused)."""
    hold = (report.get("calibration_hold") or {})
    for task, eps in task_episodes.items():
        h = hold.get(task) or {}
        if not h.get("passed"):
            raise RuntimeError(f"{task}: calibration not validated for every episode (calibration_hold); "
                               "geometry-dependent consumers are held")
        for ep in eps:
            v = (h.get("episodes") or {}).get(ep)
            if v is None:
                raise RuntimeError(f"{ep}: no validated calibration in the gate report")
            if upstream is None:
                raise RuntimeError("the store does not record its upstream calibrate build; cannot prove the "
                                   "validated calibration")
            built = (upstream(ep) or {}).get("calibrate")
            if built != v["calibrate_build_id"]:
                raise RuntimeError(f"{ep}: store built from calibrate {built!r}, validated is "
                                   f"{v['calibrate_build_id']!r}")
            extra = sorted(set(cameras(ep)) - set(v["cameras"]))
            if extra:
                raise RuntimeError(f"{ep}: store cameras {extra} have no validated calibration")


def evaluate_file(path: Path = MANIFEST, tasks=("drawer-open", "mug-hanging"), **kw) -> dict:
    raw = Path(path).read_bytes()
    report = evaluate(json.loads(raw), list(tasks), **kw)
    report["manifest"] = {"path": str(path), "sha256": hashlib.sha256(raw).hexdigest()}
    return report


def check_dataset_gate(dataset_path, gate_report, *, steps: int | None, split: dict | None = None,
                       short_run_steps: int = 10_000) -> dict:
    """FQL trainer check (beyond ``short_run_steps``): every task of the dataset passed the gate report AT THE LEVEL OF
    THE DATASET'S CONDITION (geometry-dependent conditions also need the calibration hold released, and the dataset
    must have been built from the validated calibration of every episode: its ``calibration`` attribute, recorded from
    the store at build time, equals the gate's), the dataset's episodes equal, per task, the report's store episodes,
    and ``split`` is a disjoint train / validation partition of exactly those episodes."""
    import h5py

    report = json.loads(Path(gate_report).read_text()) if isinstance(gate_report, (str, Path)) else gate_report
    with h5py.File(dataset_path, "r") as f:
        eps = json.loads(f.attrs["episodes"])
        condition = f.attrs.get("condition")
        condition = condition.decode() if isinstance(condition, bytes) else condition
        built_cal = json.loads(f.attrs.get("calibration", "{}"))
        tasks: dict[str, list[str]] = {}
        for ep in eps:
            any_tl = next(iter(f["episodes"][ep].values()))
            tasks.setdefault(str(any_tl.attrs["task"]), []).append(ep)
    consumer = FQL_CONSUMER.get(condition)
    passed = bool(report) and all(task_passed(report, t, consumer) for t in tasks)
    if not passed:
        if steps is None or steps > short_run_steps:
            level = "with the calibration hold released " if consumer_info(consumer).geometry else ""
            raise RuntimeError(f"no passed full-data gate report {level}for {sorted(tasks)} (condition "
                               f"{condition}); refusing an FQL run of {steps} steps")
        return {"smoke": True}
    for task, task_eps in tasks.items():
        if sorted(report["tasks"][task]["store_episodes"]) != sorted(task_eps):
            raise RuntimeError(f"{task}: dataset episodes differ from the gate's validated + preprocessed set")
    if consumer_info(consumer).geometry:
        check_store_calibration(report, tasks, lambda ep: {"calibrate": (built_cal.get(ep) or {}).get("calibrate")},
                                lambda ep: (built_cal.get(ep) or {}).get("cameras") or ["<unrecorded>"])
    if split is not None:
        train, val = set(split.get("train", [])), set(split.get("validation", []))
        if (train & val or (train | val) != set(eps) or len(split["train"]) != len(train)
                or len(split.get("validation", [])) != len(val)):
            raise RuntimeError("the split is not a disjoint train / validation partition of the dataset's episodes")
    return {"smoke": False, "tasks": sorted(tasks), "consumer": consumer,
            "geometry_dependent": consumer_info(consumer).geometry}
