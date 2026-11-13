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

TASK-SPECIFIC AMENDMENTS (user decision 2026-11-10 ~13:00, covering: "permit training on the 98 valid episodes
through a documented task-specific amendment ... never duplicate episodes or repair timestamps"): a versioned record
``configs/realworld/gate_amendments/<name>.json`` (format ``rw-gate-amendment/1``) fixes a task's episode set - the
listed episodes (manifest key -> sha256) and the excluded invalid ones - and its ``target`` (= the number of listed
episodes). For that task the ingest worker's ``counts.complete`` (its 100-rule) is replaced by: the manifest's task
episodes are EXACTLY the listed + excluded keys (no new arrival, none missing), every excluded key is still invalid,
every listed episode is valid with the recorded sha256, ``counts.target`` equals the record's ``base_target``; every
other check is unchanged and exactly ``target`` episodes must pass. Tasks without a record keep the default 100.

Every episode also needs ``train_data.output_format`` >= ``s4d.realworld.train/1.1.0`` and explicit family verdicts
from ``qa.output_format == "rwprep-qa/3"``. Stage-name overrides cannot waive these evidence requirements; old
train_data@1 / qa@2 records and reports without families never release a consumer.

The report lists manifest episode keys and their raw-recording identities. ``bind_store_episodes`` joins those
identities to the actual store keys (which need not resemble the manifest keys), records both keys and rekeys the
calibration evidence. Only bound reports list ``store_episodes``. Everything fails closed.

CONSUMER FAMILIES (PREP2, 2026-11-06): the per-episode QA is reported per consumer family (``QA_FAMILIES``; the qa
record's status is the base single-view family); each consumer needs its families (``CONSUMER_FAMILIES``) on every
episode, so an episode without an all-camera multi-view window or without a valid calibration never blocks a
single-view, non-geometric consumer, and never passes a consumer that needs the failed family. ReViWo (RGB of the
all-camera M5 sets, no camera geometry) needs the DERIVED family ``multiview_sets`` (``DERIVED_FAMILIES``) instead of
the full multi-view family, whose contract checks run our model on calibrated geometry (2026-11-12).

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
``calibration_validated`` record would otherwise hold every consumer); the recorded aggregate must still equal
``set_stage``'s aggregation (``aggregate_status``). The consumer is always the calling trainer's own identity
(``baseline_data.realworld_classes(consumer=...)``, the FQL dataset's condition), never a config value, and
geometry-dependent short runs need the calibration too unless explicitly opted out for synthetic plumbing
(``short_run_allowed``). The 100-episode requirement is unchanged.
"""

from __future__ import annotations

import copy
import hashlib
import json
import re
import time
from dataclasses import dataclass
from pathlib import Path

from s4d.rwlearn.identity import match_recordings, raw_identity
from s4d.rwlearn.tasks import TARGETS

MANIFEST = Path("/home/ws/ws/realworld-ops/ingest/manifest.json")
SCHEMA, SCHEMA_VERSION = "rw-ingest-manifest", 2
STAGE_SCHEMA, STAGE_SCHEMA_VERSION = "rw-preprocess", 1
TARGET = 100
AMENDMENTS = Path(__file__).resolve().parents[2] / "configs" / "realworld" / "gate_amendments"
AMENDMENT_FORMAT = "rw-gate-amendment/1"
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

# ---------------------------------------------------------------------------------------------- consumer families
# Per-episode QA verdicts per consumer family (realworld-ops/pre/tools/rwprep/rwprep_local.py ``qa --families``, schema
# rwprep-qa/3; the dispatcher copies them into the ``qa`` stage record as ``metrics.families[<family>].passed``):
# singleview = the recording + each camera's own frames and robot sync (every consumer); multiview = the cross-camera M5
# sets and the multi-view contract family; geometry = per-episode calibration, metric depth and tracks. The ``qa``
# record's status is the base (single-view) family; a consumer additionally needs ITS families on all ``target``
# episodes. Only rwprep-qa/3 with explicit family verdicts is current; older records never prove a pass.
QA_FAMILIES = ("singleview", "multiview", "geometry")
# DERIVED family (2026-11-12, calibration_optional/1 stores, e.g. pick-and-place with every camera held):
# ``multiview_sets`` = the episode's all-camera M5 sets exist with their recorded robot motion (QA gates
# ``train_data:m5_motion_recorded`` + ``train_data:m5_sets``) WITHOUT the multi-view contract family (our model's forward
# and the SNeRL scenes, which need the calibrated geometry). It is what an RGB-only multi-view consumer (ReViWo) reads.
# A qa record proves it by its own ``multiview_sets`` verdict or by a passed ``multiview`` family (which contains both
# M5 gates); a gate report written before this family (no such key) falls back to its ``multiview`` verdict.
DERIVED_FAMILIES = {"multiview_sets": "multiview"}
REPORT_FAMILIES = QA_FAMILIES + tuple(DERIVED_FAMILIES)
CONSUMER_FAMILIES = {"ours": QA_FAMILIES, "snerl": QA_FAMILIES, "mvp3d": QA_FAMILIES, "sincro": QA_FAMILIES,
                     "reviwo": ("singleview", "multiview_sets"), "dynamo": ("singleview",), "e2e_cnn": ("singleview",),
                     "fql_reviwo": ("singleview",), "fql_sincro": ("singleview", "geometry"),
                     "fql_ours": ("singleview", "geometry"), "mvp3d_single": ("singleview", "geometry")}
_TRAIN_FORMAT = re.compile(r"^s4d\.realworld\.train/(\d+)\.(\d+)\.(\d+)$")
MIN_TRAIN_VERSION = (1, 1, 0)


def consumer_families(name: str | None) -> tuple:
    """The QA families a consumer needs; an unknown consumer needs every family (fail closed)."""
    return CONSUMER_FAMILIES.get(name, QA_FAMILIES)


def _report_family(fams: dict, family: str) -> dict:
    """A task's report entry of ``family``; ONLY a derived family ABSENT from an older report is its source family's
    (an explicit null / malformed / failed entry is kept and fails closed)."""
    if family not in fams and family in DERIVED_FAMILIES:
        family = DERIVED_FAMILIES[family]
    entry = fams.get(family)
    return entry if isinstance(entry, dict) else {}


def qa_families(stages: dict) -> dict | None:
    """{family: passed} of an episode's ``qa`` record over ``REPORT_FAMILIES`` (None: a family-QA record without
    families, fail closed); a derived family is its own verdict when the record carries one (an explicit failure or a
    malformed entry fails), else - records written before it - its source family's verdict."""
    rec = stages.get("qa") or {}
    metrics = rec.get("metrics") or {}
    if not isinstance(metrics, dict):
        return None
    if rec.get("output_format") != "rwprep-qa/3":
        return None
    fam = metrics.get("families")
    if not isinstance(fam, dict):
        return None
    out = {f: isinstance(fam.get(f), dict) and fam[f].get("passed") is True for f in REPORT_FAMILIES}
    for derived, source in DERIVED_FAMILIES.items():
        if derived not in fam:  # a record written before the derived family: its source family decides
            out[derived] = out[source]
    return out


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
                     "(Original pretraining s4d/rwlearn/ours.py; offline FQL reads its export, consumer fql_ours)"),
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
# FQL's single-view conditions keep the encoder's geometry class but consume each camera's own timeline.
CONSUMERS.update({"fql_reviwo": CONSUMERS["reviwo"], "fql_sincro": CONSUMERS["sincro"],
                  "fql_ours": CONSUMERS["ours"], "mvp3d_single": CONSUMERS["mvp3d"]})
# offline-FQL conditions inherit the class of the encoder they are trained on (s4d/rwlearn/fql_data.py CONDITIONS)
FQL_CONSUMER = {"dynamo": "dynamo", "reviwo": "fql_reviwo", "sincro": "fql_sincro", "snerl": "snerl",
                "mvp3d_multiview": "mvp3d", "mvp3d_single": "mvp3d_single", "e2e_cnn": "e2e_cnn", "ours": "fql_ours"}


def aggregate_status(stages: dict) -> str:
    """``preprocessing.status`` exactly as ING's ``set_stage`` aggregates the recorded stage statuses."""
    statuses = [(r or {}).get("status") for r in stages.values()]
    return ("failed" if "failed" in statuses else "running" if "running" in statuses else
            "done" if statuses and all(x in ("done", "skipped") for x in statuses) else "pending")


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


def _decision(e: dict) -> str | None:
    o = e.get("override")
    return o.get("decision") if isinstance(o, dict) else None


# ---------------------------------------------------------------------------------------------- task amendments
def amendment_problem(rec: dict, base_target: int = TARGET) -> str | None:
    """Why a ``rw-gate-amendment/1`` record is malformed (None: well-formed). It may only shrink a task's set to its
    valid recordings among ``base_target`` received ones: listed + excluded = ``base_target``, distinct content."""
    if not isinstance(rec, dict) or rec.get("format") != AMENDMENT_FORMAT:
        return f"format is not {AMENDMENT_FORMAT}"
    task, eps, exc = rec.get("task"), rec.get("episodes"), rec.get("excluded")
    if not isinstance(task, str) or not task:
        return "task missing"
    if not (isinstance(eps, dict) and eps and isinstance(exc, dict) and exc):
        return "episodes and excluded must be non-empty maps"
    if not all(isinstance(k, str) and k.startswith(f"{task}/") for k in (*eps, *exc)):
        return "episode keys must be manifest keys of the task"
    if not all(isinstance(s, str) and _SHA.match(s) for s in eps.values()) or len(set(eps.values())) != len(eps):
        return "every listed episode needs its own sha256 (no duplicated content)"
    if set(eps) & set(exc):
        return "an episode is both listed and excluded"
    if not all(isinstance(v, dict) and v.get("reason") for v in exc.values()):
        return "every exclusion needs a reason"
    if rec.get("target") != len(eps):
        return f"target {rec.get('target')!r} != {len(eps)} listed episodes"
    if rec.get("base_target") != base_target or len(eps) + len(exc) != base_target:
        return f"listed + excluded must equal base_target {base_target}"
    doc = rec.get("document")
    if not (isinstance(doc, dict) and doc.get("path") and _SHA.match(str(doc.get("sha256", "")))):
        return "document path + sha256 missing"
    if not all(rec.get(k) for k in ("decision", "decided_by", "decided_at")):
        return "decision provenance missing"
    return None


def load_amendments(directory: Path = AMENDMENTS) -> dict:
    """{task: {"record", "path", "sha256"}} of the versioned amendment records; a malformed record or two records for
    one task raise (fail closed)."""
    out = {}
    for path in sorted(Path(directory).glob("*.json")):
        raw = path.read_bytes()
        rec = json.loads(raw)
        why = amendment_problem(rec)
        if why:
            raise RuntimeError(f"gate amendment {path}: {why}")
        if rec["task"] in out:
            raise RuntimeError(f"two gate amendments for {rec['task']}: {out[rec['task']]['path']}, {path}")
        out[rec["task"]] = {"record": rec, "path": str(path), "sha256": hashlib.sha256(raw).hexdigest()}
    return out


def amendment_mismatch(rec: dict, mine: dict) -> list[str]:
    """Problems of a task's manifest episodes against its amendment: the set must be exactly listed + excluded and
    every excluded episode still invalid."""
    listed, excluded = set(rec["episodes"]), set(rec["excluded"])
    out = [f"{k}: not in the gate amendment (new recording? the amendment needs a new decision)"
           for k in sorted(set(mine) - listed - excluded)]
    out += [f"{k}: amendment episode missing from the manifest" for k in sorted((listed | excluded) - set(mine))]
    out += [f"{k}: excluded by the amendment but validation is {(mine[k].get('validation') or {}).get('status')!r}"
            for k in sorted(excluded & set(mine)) if (mine[k].get("validation") or {}).get("status") != "invalid"]
    return out


def evaluate(manifest: dict, tasks, *, required_stages=REQUIRED_STAGES, target: int = TARGET,
             amendments: dict | None = None) -> dict:
    """``target``: the default per-task episode count; ``amendments`` ({task: load_amendments() entry}, default: the
    versioned records) replace it for their task."""
    amendments = load_amendments() if amendments is None else amendments
    report = {"passed": False, "geometry_passed": False, "checked_at_unix": time.time(),
              "schema": manifest.get("schema"), "required_stages": list(required_stages),
              "calibration_stage": CALIBRATION_STAGE, "target": target, "tasks": {}, "calibration_hold": {},
              "families": {}, "consumer_families": {k: list(v) for k, v in CONSUMER_FAMILIES.items()},
              "consumers": {k: {"geometry_dependent": c.geometry, "uses": c.uses, "reason": c.reason}
                            for k, c in CONSUMERS.items()}}
    if (manifest.get("schema"), manifest.get("schema_version")) != (SCHEMA, SCHEMA_VERSION):
        report["error"] = (f"manifest schema {manifest.get('schema')!r} v{manifest.get('schema_version')!r} != "
                           f"{SCHEMA!r} v{SCHEMA_VERSION}")
        return report
    episodes = manifest.get("episodes") or {}
    all_ok = geo_ok = True
    for task in tasks:
        problems, hold_problems, hold_ok, identities = [], [], {}, {}
        family_fail = {f: [] for f in REPORT_FAMILIES}
        counts = (manifest.get("counts") or {}).get(task) or {}
        amend = amendments.get(task)
        rec = amend["record"] if amend else None
        why = (amendment_problem(rec) or (None if rec["task"] == task else f"record task {rec['task']!r}")) if rec \
            else None
        if why:  # a supplied map is validated like the versioned records: a malformed one releases nothing
            report["tasks"][task] = {"passed": False, "target": None, "amendment": None, "valid_and_preprocessed": 0,
                                     "excluded": [], "problems": [f"gate amendment: {why}"], "problem_count": 1,
                                     "episodes": [], "episode_identities": {}, "store_episodes": []}
            report["families"][task] = {f: {"passed": False, "failed_count": 0, "failed_episodes": []}
                                        for f in REPORT_FAMILIES}
            report["calibration_hold"][task] = {"passed": False, "validated": 0, "problems": [], "problem_count": 0,
                                                "episodes": {}}
            all_ok = geo_ok = False
            continue
        task_target = rec["target"] if rec else target
        mine = {k: e for k, e in episodes.items() if e.get("task") == task}
        if rec is None:
            if counts.get("complete") is not True:
                problems.append("ingestion not complete for this task (counts.complete is not true)")
            if counts.get("target") != target:
                problems.append(f"manifest target {counts.get('target')} != {target}")
        else:  # the amendment's exact episode set replaces the ingest worker's 100-rule
            if counts.get("target") != rec["base_target"]:
                problems.append(f"manifest target {counts.get('target')} != amendment base_target {rec['base_target']}")
            problems += amendment_mismatch(rec, mine)
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
            if rec is not None and rec["episodes"].get(key) != sha:
                problems.append(f"{key}: sha256 differs from the gate amendment's")
                continue
            idx = e.get("episode_index")
            index_identity = (e.get("episode_label", "success"), idx)
            if (e.get("episode_label", "success") not in ("success", "failure")
                    or not isinstance(idx, int) or index_identity in indices):
                problems.append(f"{key}: missing or duplicate episode index {idx!r}")
                continue
            indices.add(index_identity)
            pre = e.get("preprocessing") or {}
            stages = pre.get("stages") or {}
            # aggregate: every recorded stage done (or skipped) EXCEPT the hold stages, which only gate geometry-
            # dependent consumers (set_stage's preprocessing.status folds them in). The recorded aggregate must be
            # exactly set_stage's aggregation of the recorded stages (fail closed on a missing / unknown / stale one)
            open_stages = sorted(s for s, r in stages.items() if s not in HOLD_STAGES
                                 and (r or {}).get("status") not in ("done", "skipped"))
            explained = pre.get("status") == aggregate_status(stages)
            if (pre.get("schema"), pre.get("schema_version")) != (STAGE_SCHEMA, STAGE_SCHEMA_VERSION) \
                    or not stages or open_stages or not explained:
                problems.append(f"{key}: preprocessing {pre.get('status')!r} (schema {pre.get('schema')!r} "
                                f"v{pre.get('schema_version')!r}; stages not done: {open_stages})")
                continue
            bad = [s for s in dict.fromkeys((*required_stages, "qa", "train_data"))
                   if (stages.get(s) or {}).get("status") != "done"
                   or (stages.get(s) or {}).get("input_sha256") != sha]
            if bad:
                problems.append(f"{key}: required stages not done for this file: {bad}")
                continue
            fmt = (stages.get("train_data") or {}).get("output_format")
            match = _TRAIN_FORMAT.fullmatch(str(fmt or ""))
            if match is None or tuple(map(int, match.groups())) < MIN_TRAIN_VERSION:
                problems.append(f"{key}: train_data output_format {fmt!r}; requires s4d.realworld.train/1.1.0 or later")
                continue
            try:
                recordings = [p for p in (e.get("filename"), (e.get("source") or {}).get("path"),
                                          key if key.endswith(".h5") else None) if p]
                if not recordings or len({Path(p).name for p in recordings}) != 1:
                    raise RuntimeError("missing or conflicting raw identity in manifest")
                recording = recordings[0]
                identity = raw_identity(task, recording, sha)
            except RuntimeError as exc:
                problems.append(f"{key}: {exc}")
                continue
            identities[key] = {**identity, "store_episode": None}
            valid.append(key)
            fams = qa_families(stages)
            for f in REPORT_FAMILIES:
                if fams is None or not fams[f]:
                    family_fail[f].append(key if fams is not None else f"{key} (requires rwprep-qa/3 with families)")
            why = calibration_record_problem(stages, sha)
            if why:
                hold_problems.append(f"{key}: {why}")
            else:
                m = stages[CALIBRATION_STAGE]["metrics"]
                hold_ok[key] = {"calibrate_build_id": m["calibrate_build_id"], "cameras": sorted(m["cameras"])}
        try:
            match_recordings(identities, identities)
        except RuntimeError as exc:
            problems.append(str(exc))
        if len(valid) != task_target:
            problems.append(f"{len(valid)} episodes valid + preprocessed, need exactly {task_target}"
                            + (f" (gate amendment {Path(amend['path']).name})" if rec else ""))
        passed = not problems
        all_ok &= passed
        report["tasks"][task] = {"passed": passed, "valid_and_preprocessed": len(valid), "excluded": excluded,
                                 "target": task_target,
                                 "amendment": {"path": amend["path"], "sha256": amend["sha256"], "target": rec["target"],
                                               "base_target": rec["base_target"], "excluded": sorted(rec["excluded"])}
                                 if rec else None,
                                 "problems": problems[:50], "problem_count": len(problems),
                                 "episodes": valid if passed else [],
                                 "episode_identities": identities if passed else {}, "store_episodes": []}
        report["families"][task] = {f: {"passed": passed and not family_fail[f], "failed_count": len(family_fail[f]),
                                        "failed_episodes": family_fail[f][:50]} for f in REPORT_FAMILIES}
        hold_passed = passed and not hold_problems and len(hold_ok) == task_target
        geo_ok &= hold_passed
        report["calibration_hold"][task] = {
            "passed": hold_passed, "validated": len(hold_ok), "problems": hold_problems[:50],
            "problem_count": len(hold_problems), "episodes": hold_ok if hold_passed else {}}
    report["passed"] = bool(all_ok and tasks)
    report["geometry_passed"] = bool(report["passed"] and geo_ok)
    return report


def bind_store_episodes(report: dict, task_episodes: dict, identities: dict) -> dict:
    """Return a report bound to actual store (or FQL dataset) keys using a bijection of recorded raw identities.
    An unbound gate can plan runs; every data consumer must bind it before comparing episode sets/calibration."""
    bound = copy.deepcopy(report)
    for task, eps in task_episodes.items():
        rec = bound["tasks"][task]
        expected = rec.get("episode_identities")
        if not expected or set(expected) != set(rec.get("episodes", [])):
            raise RuntimeError(f"{task}: gate report missing raw identity; regenerate the gate report")
        if len(eps) != len(set(eps)) or any(ep not in identities for ep in eps):
            raise RuntimeError(f"{task}: duplicate episode or missing raw identity")
        actual = {ep: identities[ep] for ep in eps}
        if any(r.get("task") != task for r in actual.values()) or any(r.get("task") != task for r in expected.values()):
            raise RuntimeError(f"{task}: raw identity task differs from the episode task")
        mapping = match_recordings(expected, actual)
        recorded = [r.get("store_episode") for r in expected.values()]
        if rec.get("store_episodes") or any(ep is not None for ep in recorded):
            if (sorted(rec.get("store_episodes", [])) != sorted(eps)
                    or any(expected[key].get("store_episode") != ep for ep, key in mapping.items())):
                raise RuntimeError(f"{task}: store episodes differ from the gate's recorded store keys")
        hold = (bound.get("calibration_hold") or {}).get(task) or {}
        if hold.get("passed"):
            previous = hold.get("episodes") or {}
            rekeyed = {}
            for ep, key in mapping.items():
                old = expected[key].get("store_episode") or key
                if old not in previous:
                    raise RuntimeError(f"{key}: no validated calibration in the gate report")
                rekeyed[ep] = previous[old]
            hold["episodes"] = rekeyed
        rec["store_episodes"] = sorted(eps)
        for ep, key in mapping.items():
            expected[key]["store_episode"] = ep
    return bound


def task_passed(report: dict, task: str, consumer: str | None) -> bool:
    """Whether ``consumer`` may train on ``task``: the 100-episode gate, plus the calibration hold for geometry-
    dependent consumers."""
    base = bool(((report or {}).get("tasks") or {}).get(task, {}).get("passed"))
    fams = ((report or {}).get("families") or {}).get(task) or {}
    base = base and all(_report_family(fams, f).get("passed") is True for f in consumer_families(consumer))
    if not consumer_info(consumer).geometry:
        return base
    return base and bool(((report or {}).get("calibration_hold") or {}).get(task, {}).get("passed"))


def task_problems(report: dict, task: str, consumer: str | None, limit: int = 5) -> list[str]:
    t = ((report or {}).get("tasks") or {}).get(task)
    if t is None:
        return ["not evaluated"]
    out = list(t.get("problems", []))
    fams = ((report or {}).get("families") or {}).get(task) or {}
    for f in consumer_families(consumer):
        x = _report_family(fams, f)
        if x.get("passed") is not True and t.get("passed"):
            out.append(f"QA family {f}: {x.get('failed_count', 'not evaluated')} episodes failed "
                       f"{x.get('failed_episodes', [])[:3]}")
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


def evaluate_file(path: Path = MANIFEST, tasks=TARGETS, **kw) -> dict:
    raw = Path(path).read_bytes()
    report = evaluate(json.loads(raw), list(tasks), **kw)
    report["manifest"] = {"path": str(path), "sha256": hashlib.sha256(raw).hexdigest()}
    return report


def short_run_allowed(consumer: str | None, steps, short_run_steps: int, smoke_without_calibration: bool) -> bool:
    """Short (plumbing) runs may skip the 100-episode gate; for a geometry-dependent consumer only with the explicit
    ``smoke_without_calibration`` opt-in (synthetic test stores; never set by generated configs or plan rows), so
    that an unvalidated calibration never reaches a geometric run by default."""
    if steps is None or steps > short_run_steps:
        return False
    return not consumer_info(consumer).geometry or smoke_without_calibration is True  # a literal True, never "false"


def check_dataset_gate(dataset_path, gate_report, *, steps: int | None, split: dict | None = None,
                       short_run_steps: int = 10_000, smoke_without_calibration: bool = False) -> dict:
    """FQL trainer check (beyond ``short_run_steps``): every task of the dataset passed the gate report AT THE LEVEL OF
    THE DATASET'S CONDITION (geometry-dependent conditions also need the calibration hold released, and the dataset
    must have been built from the validated calibration of every episode: its ``calibration`` attribute, recorded from
    the store at build time, equals the gate's), the dataset's episodes equal, per task, the report's store episodes,
    and ``split`` is a disjoint train / validation partition of exactly those episodes."""
    import h5py

    report = json.loads(Path(gate_report).read_text()) if isinstance(gate_report, (str, Path)) else gate_report
    with h5py.File(dataset_path, "r") as f:
        eps = json.loads(f.attrs["episodes"])
        if not eps or len(eps) != len(set(eps)):
            raise RuntimeError("FQL dataset has missing or duplicate episode keys")
        condition = f.attrs.get("condition")
        condition = condition.decode() if isinstance(condition, bytes) else condition
        built_cal = json.loads(f.attrs.get("calibration", "{}"))
        identities = json.loads(f.attrs.get("raw_identities", "{}"))
        tasks: dict[str, list[str]] = {}
        for ep in eps:
            any_tl = next(iter(f["episodes"][ep].values()))
            tasks.setdefault(str(any_tl.attrs["task"]), []).append(ep)
    consumer = FQL_CONSUMER.get(condition)
    passed = bool(report) and all(task_passed(report, t, consumer) for t in tasks)
    if not passed:
        if not short_run_allowed(consumer, steps, short_run_steps, smoke_without_calibration):
            level = "with the calibration hold released " if consumer_info(consumer).geometry else ""
            raise RuntimeError(f"no passed full-data gate report {level}for {sorted(tasks)} (condition "
                               f"{condition}); refusing an FQL run of {steps} steps")
        return {"smoke": True, "consumer": consumer, "geometry_dependent": consumer_info(consumer).geometry}
    report = bind_store_episodes(report, tasks, identities)
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
