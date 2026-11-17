# Pick-and-place-v2: collection-to-evaluation preparation

Prepared for user directive 2026-11-17 ~18:10, item 3, from main `6be7607`.
**No collection, robot/remote access, experiment, registration, placement, or launch
is authorized by this document.** This change prepares files and review proposals.
Existing pick-and-place (v1), drawer-open-v2, covering, running jobs, queue,
registry, and ownership records are preserved. S only; SF disabled; mug-hanging
is discontinued. Do not reuse v1 data, scene measurements, calibration, split,
checkpoint, or gate amendment merely because the task names are similar.

## Inspect the proposal now

From this worktree, the only workflow command intended for use now is:

```bash
/home/ws/ws/splatter4d/.venv/bin/python -I scripts/register_pick_and_place_v2.py
```

It prints all 88 candidate rows to stdout and writes nothing. `registered` and
`launch_authorized` remain false. The rows contain unresolved paths and null
release pins, and carry `preparation_only: true` plus a placement pause. They are
**not scheduler-ready queue rows**. Do not pipe the output into a live registrar.

`--register` without `--authorization-file` refuses. An authorization file alone
also refuses. The reserved authorization schema is
`pick-and-place-v2-registration-authorization/1`, with `task: pick-and-place-v2`,
`authorize_registration: true`, a named `authorized_by`, and exact proposal
`spec_sha256` / `rows_sha256`. Even a matching file refuses: this dry-run-only
version intentionally contains no queue writer. A later authorized change must
supply reviewed queue admission and publication; this file is not authorization.
No authorization file is checked in or created by this work.

## One task spec, explicit placeholders

[The task spec](../../configs/realworld/tasks/pick-and-place-v2.yaml) is the single
source for dataset roots, 100-success collection target, S, split seed 0,
validation fraction 0.1, FQL seeds 0/1/2, calibration policy, pending scene
measurements, SAM3 prompts, and future release/image pins. All roots initially
start with `/__TODO_COLLECT_PICK_AND_PLACE_V2__/`. They intentionally do not point
to any existing dataset. After collection, replace roots in that spec only;
`pnpv2_spec.values()` derives these paths:

| Artifact | Derived location |
|---|---|
| Complete raw success recordings | `<raw>/pick-and-place-v2/success/episode_NNNNNN.h5` |
| Per-episode prep | `<processed>/pick-and-place-v2/{success}/episode_NNNNNN` |
| Canonical membership, split, outcomes | `<publication>/pick-and-place-v2/S/{manifest.json,split_seed0.json,outcomes.json}` |
| Calibration-independent store/report | `<publication>/pick-and-place-v2/S/rgb/{store.yaml,gate_report.json}` |
| Accepted geometry store/report | `<publication>/pick-and-place-v2/S/geometry/{store.yaml,gate_report.json}` |
| Rig / reference | `<calibration>/pick-and-place-v2/rig-pick-and-place-v2-accepted.{json,reference.npz}` |
| Confirmed robot model | `<robot_assets>/pick-and-place-v2/confirmed-robot.urdf` |

The new YAML files are **templates**, not configs for direct `scripts/train.py`
use. `${pnpv2.*}` tokens resolve through `s4d.rwlearn.pnpv2_spec.config()` and the
new `scripts/pick_and_place_v2.py` entrypoint. Whole-value tokens preserve YAML
types (including deliberate null scene parameters). Unknown/embedded tokens
refuse. The entrypoint rejects unresolved TODO roots before importing any trainer.
It installs task support only inside its new Python process; existing task
registries and placed scripts retain their original bytes.

## 1. Ingestion and membership (future, separately authorized)

Collect into a new task namespace with all three cameras, robot timestamps,
recorded sent pose commands and continuous gripper commands. Confirm the robot,
gripper, URDF, camera serials and adapter contract against the actual collection;
the prep template retains the drawer workflow's Panda/Robotiq adapter recipe as
a starting point, not a claim about a newly collected rig. Review the task's SAM3
robot/object/fixture prompts and fill `sam3_prompts`; no drawer object names are
inherited. Record collection/session identities and camera relocations.

Ingest only finalized complete recordings. Preserve raw files and their SHA256,
validation results, transfer verification, and all rejection evidence. Never
repair timestamps, duplicate episodes to fill counts, admit partial transfers,
or mutate the live v1 inventory. Failure/discarded roots are never inspected for
S membership. The current spec requires all **100** successful recordings. It
copies neither drawer's 98 nor v1's 97 accepted set. Any future exclusion/count
amendment needs a new task-specific decision and reviewed implementation.

The future `--entry manifest --config configs/realworld/prep/pick-and-place-v2.yaml
-- --out <fresh-bundle>` path reads only the success root, rehashes it, requires
the `.VERIFIED` marker, validates raw timing/actions, freezes directory-derived
identities and outcomes, and saves a deterministic 90-train/10-validation
**episode** split with seed 0. It writes manifests and prep configs only; it
never schedules prep. Both RGB and geometry publications must keep exactly the
same raw identities, split and outcomes. No warm start or partial-data training.

## 2. Calibration gates and Phase A/B preparation

The reference is the local route at
`/home/ws/ws/realworld-ops/drawer-geo-20261116`: `lib.sh`, `stage1_upload.sh`,
`stage2_rolling.sh` / `stage2_sam3.sh`, `stage3_calib.sh`,
`stage4_phaseB.sh`, `stage5_publish.sh`, `RUNNERS.md`, and `VALIDATION.md`.
There is no README in that directory at inspection time; `RUNNERS.md` documents
the route. These files were read, never sourced, imported or executed.

Prepare a **fresh task-scoped** dispatcher/bundle namespace in a later authorized
session. Do not copy/source the drawer shell scripts: their guards, 98-member
assertions, aliases, pins, mounts, paths, override hooks and 39 IDs are drawer
specific. Use their reviewed sequence and admission rules:

1. Phase A normalizes finalized recordings and computes sensor depth,
   FoundationStereo depth, 2D TrackEverything products and SAM3 masks as required.
   Local Phase A upload uses immutable per-file hash verification and atomic
   no-overwrite publication. Uploaded batches may overlap SAM3 preparation; a log
   saying DONE is not transfer proof. Bootstrap bundles are metadata only and
   must not be attached/launched as full pipelines.
2. Preserve disjoint durable batch assignment and every original dispatcher,
   jobs file, source hash, completion record and external-state dependency.
   Existing STOP/hold/reservation/ownership rules remain in force; never synthesize
   all-done state or reset failed attempts. 2D RGB work may proceed while camera
   calibration remains held; do not interpret it as validated world geometry.
3. Fit and validate a new per-task/per-session rig using the standard
   `s4d/realworld/calib/loma` / `criteria.py` / `release.py` path. Freeze fitting
   and held-out aliases and preserve complete evidence for every selected episode
   and all three cameras, including cameras relocated between sessions. Retain
   original release/1 evidence and the release/2 verdict. A fitting subset is not
   evidence of full episode coverage.
4. Require `rig_loma-release/2` G1–G6 acceptance: stability about the published
   solution plus image displacement, consistent pairwise-nearest correspondence
   evidence and dt-extension checks, fixedness and complete per-episode evidence.
   Use code thresholds unchanged; missing/nonfinite evidence fails closed.
   The drawer-only `s4d/realworld/calib/overrides/drawer-open-v2-20261117.json`
   does **not** apply. This task spec and publication readers reject overrides.
5. Phase B compiles only after the accepted task/session rig and hashed reference
   are present and every camera is accepted. Bind each episode to its selected
   successful SAM3 job and actual calibrate build ID, then build world-frame
   tracks/motion, M5 sets, compact stores and per-family QA. The reference
   compiler is `scripts/realworld_prep_register.py` (A-sam3 vs B). Use the standard
   native-depth path for 3D-MVP; never the drawer override entrypoint.

`calibration_validated` must be done for the same raw SHA and current
`calibrate.metrics.build_id`; every store camera and its upstream calibration
must match. Neither an existing filename nor an operator note releases this hold.

## 3. Publication and early calibration-independent work

| Consumer | Earliest eligible input | Additional requirements |
|---|---|---|
| DynaMo pretrain/export/FQL | RGB publication after complete ingestion and required prep | singleview QA, exact raw hashes/split |
| ReViWo pretrain/export/FQL | RGB publication | singleview plus valid all-camera M5 `multiview_sets` for pretraining |
| E2E CNN and noaug FQL | RGB publication | singleview QA, proprioception and recorded commands; no representation pretrain |
| Ours orig/ch01, SinCro | Accepted geometry publication | calibration hold released, required multiview/geometry QA |
| SNeRL | Accepted geometry publication | measured task lookat and pose_scale |
| 3D-MVP single/multiview | Accepted geometry + native-depth overlay | measured task bounds and containment; shared fused pretrain |

Thus DynaMo, ReViWo and E2E work can start after ingestion **once their full-data
RGB/preprocessing gates are ready**, without waiting for calibration. Ingestion
alone does not permit training on missing preprocessing or a partial dataset.
Use the existing `calibration_optional/1` RGB-store route while geometry is held;
never invent camera extrinsics to satisfy geometry QA. RGB and geometry barriers
in the proposal are independent. Both use the same canonical split.

Build a fresh task-only ingest snapshot from immutable manifest + actual source
prep completion records and verified remote raw hashes. The future `--entry
snapshot -- --success-only ...` uses the established snapshot producer while
replacing its historical amendment-only membership reader with this task's
unamended 100-success validator. Pass every contributing dispatcher with repeated
`--prep-config`; preserve their source queues/states, not an aggregate state.

For each view, the future `--entry publish -- --view rgb|geometry --manifest
<bundle/S/manifest.json> --ingest-manifest <snapshot> --store <descriptor>` path
materializes the task spec's immutable view and common membership/split/outcomes.
It refuses conflicting RGB/geometry membership and any acceptance override.
The gate row rechecks exact store/raw membership and each view's consumers with
`baseline_data.check_gate`, including `rwprep-qa/3`, train format >=1.1.0,
full-data QA and the geometry calibration/build/camera checks. A publication is
usable only after this barrier passes. Partial publication output is never proof
of readiness; preserve it for inspection on failure, then use a new publication
root. Hash and verify the entire compact-store closure when mirroring/publicizing
it; changing the store descriptor alone is insufficient.

Measure scene geometry on the accepted task store with the established
`store.scene_statistics`/`dsreset_measure` route: optical-axis lookat, 4 / median
camera distance, task bounds with >=99% robot/object/fixture containment and
near/far coverage evidence. Fill the spec's `scene` only from that new report.
Keep SNeRL/3D-MVP held while their values are null. Ours workspace statistics come
from its accepted store and **training split only** via the stats row.

## 4. Pretraining, FQL and offline evaluation

The 88-row proposal consists of two readiness barriers, Ours workspace stats,
one native-depth overlay, seven pretrain/export pairs, ten FQL-dataset builds,
30 FQL policies and 30 offline policy evaluations. IDs have fresh
`pnpv2-20261117` identities and contain `pick-and-place-v2-S`. All dependencies
are inside the proposal. No legacy row or checkpoint is reused.

- Ours has independent `orig` and `ch01` scratch400k runs. Both use the current
  drawer S architecture/augmentation/loss schedule: 400k single cosine,
  warmup 10k, peak LR 5e-4, minimum 1e-5, no restart/200k continuation. ch01 adds
  only `loss.chamfer: 0.1`. Export only `step_0400000.pt`.
- DynaMo, ReViWo, SinCro, SNeRL and 3D-MVP retain drawer-open-v2's architecture,
  objective, optimizer and schedule. 3D-MVP single and multiview share the fused
  representation export, then use distinct FQL dataset conditions. E2E CNN is
  trained jointly with FQL; the retained noaug variant uses `pixel.p_aug=0.0`.
- Every new-task FQL config explicitly sets **alpha 10 and
  `normalize_q_loss: true`**, including E2E, as requested in this directive.
  Historical dsreset E2E used false; no historical config is changed. Frozen
  3D-MVP features retain `train_standardize`, fitted on the train split only.
  Each condition runs seeds 0/1/2 for 1M offline steps with the existing action,
  continuous gripper and reward contract. S outcomes are all success; no SF rows.
- Each eval config requests the established offline artifact checker: policy
  checkpoint hash, complete 1M budget, and final-step held-out validation metrics
  from `metrics.jsonl`. Training also retains its representation validation.
  These are offline diagnostics, not real-world task success rates, deployment
  clearance or robot evaluation. Robot eligibility/physical execution require a
  separate later directive and gates.

For future execution, the wrapper interface is `--entry <operation> --task-spec
<resolved-spec> [--config <template>] -- <original-operation-arguments>`. The
proposal contains exact per-row argv. Templates are resolved into a temporary
YAML before the existing operation is imported; its saved run config records the
resolved values. Existing full-suite/release/GPU/memory/admission guards still
apply. The focused checks for this preparation are not long-run release evidence.

## 5. Why the drawer stage-5 count is 39, and the later handoff

The committed `docs/drawer-identity-20261116/overlap.json` records a 71-row full
historical drawer graph partitioned into 32 already handled RGB rows and 39
remaining geometry/barrier rows. It is not a generic new-dataset row count.
External `stage5_publish.sh` waits for all Phase-B members and accepted rig,
publishes immutable portable inputs with original hashes, mirrors and verifies
the compact closure, stages S inputs, verifies the **exact** overlap IDs in its
dry-run, and executes that same locked registration record only after queue,
registry and input hash revalidation. Partial existing publication is held;
completed publication is reused only with the full original proof. Placement is
a separate preview/execute pair. Its later native-depth override hook is drawer
specific too. None of these actions was run for this task.

A future authorized PNP v2 handoff must review actual artifacts and recipe hashes,
fill immutable release/image pins with passing full release evidence, add
scheduler input-resolution/host-role support for the new wrapper, check duplicate
IDs and existing run paths, and implement/adopt locked append-only publication
of an exact reviewed proposal with the explicit authorization file. Keep existing
queue/registry/input hashes as drift guards, then separately review placement.
Do not pass this proposal to the old hardcoded dsreset registrar or remove its
placement pause. No such queue integration, launch or placement is enabled by
this preparation-only change.

## Focused validation

Run only `tests/test_pick_and_place_v2.py`, `tests/test_dsreset_registration.py`,
`tests/test_memory_admission.py`, `tests/test_run_tests_sharding.py`, and
`tests/test_ops_layout.py` with the existing guarded test interpreter and an
approved local device environment. The new file is classified `cpu` in
`scripts/test_shards.py`. Tests cover config parity/token resolution, all retained
conditions and seeds, S-only dependencies, independent RGB eligibility,
registration refusal/read-only behavior, process-local task support, override
rejection and placed fingerprint preservation. No full suite or experiment is
part of this work.

Preparation validation result (2026-11-17): **190 passed in 49.42 s** across
exactly those five focused test files. Ruff passes for all new Python files;
`git diff --cached --check` passes. The persisted 89-row placement-scope snapshot
reports zero new release changes and the same pre-existing differences as its
baseline. Only the test shard classification changes an existing file. The
standalone registrar dry run prints 88 rows with registration/launch false.
