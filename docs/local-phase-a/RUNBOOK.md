# Local phase A, 2026-11-15

Status: **IMPLEMENTED, NOT QUALIFIED FOR BULK**. The two-success CPU gate passed;
FoundationStereo/SAM3 inference equivalence and their full footprints remain unmeasured.
The requested MIG was occupied by a healthy FQL trainer throughout qualification.
No bulk job, queue registration, placement, adoption, upload, host operation or trainer
interruption was performed. Main remained `093e044d37d53313afee952d0435000f3eaee990`.

The implementation branch is `dsreset-local-phaseA-20261115`; source/evidence worktree
`/home/ws/ws/splatter4d/.worktrees/lpa`. Qualification bundle:
`/home/ws/ws/splatter4d/runs/local-phase-a-qualification-20261115`.
The final scope audit covered 97 ownership records (93 with persisted execution scopes)
with zero release changes.
Seventy-eight already differed from their historical pins in main; those baseline
differences are unchanged (`absolute_scope_identical=false` is not waived or repinned).
See [EQUIVALENCE.md](EQUIVALENCE.md), `equivalence.json`, `scope-audit.json`, and
`../../docs/tests.json` in this worktree. The bundle retains the immutable snapshot,
actual two-episode outputs, per-stage worker/run logs, checksums and resource measurements.

## Validation

The full suite passed on tested source `7a4f2e9148b0005f1e90c53ab301360419f5ef70`: **4039 passed, 0 failed, 0 errors, 0 skipped** (4012 collected tests plus subtests), with `full_suite_passed=true` and `source_changed_during_suite=false`. It ran in this short-path worktree with pinned submodules, existing DINO weights and GPU `MIG-2ebd9e11-f2e0-5e22-870a-7779d44ec10a`. The final commit adds only documentation and unedited suite evidence over the tested source. Focused integrity, gate, publication and scheduler checks passed 328 tests; the final test-layout check passed 63 tests, including all 18 new tool tests. See `focused-tests.log`, `layout-tests.log`, `cpu-gate.log` and `qualification-summary.json`.

## Envelope and layout

The runner preserves the established `rwprep_local.py` adapter, persistent stream
transport, model workers, native resolution, FS 32 iterations, SAM3 video mode and
full-video chunk lengths, durable chunk-save-before-ACK protocol, stage fingerprints
and completion records. It closes each stage's pool before starting the next; FS
and SAM3 models cannot coexist. No bulk input inventory is inferred from filenames:
the three canonical plans supply the episode's task, outcome, raw SHA and exact config.
The 195 accepted successes are reused; the preview contains 248 new episode rows.

`prepare` makes standalone shallow Git snapshots of the committed campaign code,
SAM3 `f1647e8`, FS `6e88068`, DINOv2 hub code `7764ea0`, the established helper, and
COPIES of the three audited SAM3/FS weight/config files. It verifies each weight
against `pipeline/local-weight-audit.json`, records every snapshot file's SHA256,
then removes write permissions from every snapshot file/directory. It never chmods
shared source, original weights, or existing environments. Execution checks the full
snapshot before and after **each episode**, including the failure path; raw SHA is
also checked before and after. There is no stat-only weight checksum cache in this
outer check. Shared interpreter bytes and installed package RECORD metadata are
checked, and upstream and campaign Git HEADs are checked.

Workers run in a private mode-0700 `/dev/shm/s4d-localA-*` directory. TMPDIR, Torch,
HF, Triton and CUDA caches point there; the established protocol still requires
RAM backing. Only one allowlisted local MIG UUID may be CUDA-visible. Worker Torch
allocation is capped at 12 GiB; the process tree has four-core CPU affinity and
four-thread numerical libraries. The runner records process-tree RSS, worker CUDA
peak reservations, output disk footprint and actual stage timings. It does not
install packages or download weights. It uses these existing interpreters:

- `/home/ws/ws/validation-20261031/envs/sam3/bin/python`
- `/home/ws/ws/validation-20261031/envs/foundation_stereo/bin/python`
- `/home/ws/ws/splatter4d/.venv/bin/python` for orchestration and CPU kernels

This is an integrity envelope, not a mount/security boundary: root can bypass
chmod; the campaign's root filesystem/network/cgroups and installed environments
remain shared. Installed dependency trees are not copied or mounted read-only;
interpreter/package metadata checks do not hash every installed library byte.
There is no separate per-episode kernel cgroup RAM limit. Scheduler RAM admission
is retained. These differences must not be described as the strict LocalServer.

Production output is ONLY the NEW root
`/home/ws/data/realworld-processed/dsreset-20261113`, with exactly the R1 suffix:

```
<task>/<success|failure>/episode_NNNNNN/
  pipeline.yaml
  episodes/<task>-<success|failure>-episode_NNNNNN/
    normalized/{episode.json,arrays.npz,frames/*.h5,_COMPLETE.json}
    derived/{sensor_depth,foundation_stereo_depth,sam3_masks}/...
  runs/<UTC>/run.json
  local-phase-a.json
```

Gate outputs intentionally stay under the qualification bundle's `gate/`, so
incomplete qualification artifacts cannot be mistaken for publishable production
outputs. The requested new processed root was created empty. Existing processed
roots, original recordings, STOP files and active executions were left unchanged.

## Equivalence and measurements

The local mirrors are the established outputs at
`/home/ws/ws/realworld-20261104/processed/tasks/{covering,pick-and-place}/base`:
covering `covering-000000` (228 frames/view), PNP `pick-and-place-000002` (215/view).
Their recorded raw SHA256s equal the current `episode_000000.h5` and
`episode_000002.h5`. The gate maps old episode IDs/raw filenames to label-inclusive
campaign IDs; only the contract's episode ID, raw path and `provenance.created_utc`
are normalized. Every other contract field, raw SHA/size, trajectory, timestamp and
normalized frame is checked exactly. Sensor depth and validity payloads/attrs are
bitwise checked. HDF5 wrapper bytes and stage UUID/time fields are not deterministic.

Predeclared FS thresholds apply to each frame: mean absolute depth error <= 0.002 m,
p99 absolute <= 0.01 m, p99 relative <= 0.01 on jointly valid pixels, valid-mask IoU
>= 0.995; nonfinite outputs fail. Semantic foreground class/frame IoU must have mean
>= 0.99 and minimum >= 0.95, excluding absent-in-both classes from the mean.
Raw instance masks have the same IoU limits; frame/prompt/instance IDs, detection
counts and class presence are exact; score absolute difference <= 0.005 and box
coordinate absolute difference <= 1 pixel. The gate records full numbers and
reference/candidate file SHA256s. It fails closed on incomplete coverage.

Observed CPU-only footprints (not full phase-A measurements): covering 1.6081 GiB
peak process-tree RSS / 1.1739 GiB output disk; PNP 1.6721 GiB / 1.0916 GiB; four
CPU cores each. GPU peak reservation was zero. The registrar's conservative
16 GiB GPU / 48 GiB RAM / 4 CPU / 32 GiB transient-disk declarations are **not**
claimed as measured GPU-phase footprints. `--execute` refuses until the FULL
four-stage, two-success report passes and measured inference footprints exist.

Full-video SAM3 reference chunks are 228 and 215 frames; truncating them changes
inference context. No truncated-video result is accepted as full equivalence.
The requested short GPU smoke was deferred because MIG
`MIG-c3511b41-5f78-52f1-9b93-9c6794cdbbdd` had active healthy trainer PID 993329,
`rwnt-20261110-fql-e2e_cnn-pick-and-place-s1`, using 7388 MiB (7.2148 GiB) plus
active CPU work. No FS/SAM depth/mask numbers are invented. The full gate below
needs an idle GPU and a longer qualification window; it was not run under the
short-smoke-only instruction. A read-only SSH probe still returned `RemoteUnreachable`.

## Exact coordinator commands

Commands are provided for review/resumption; the implementer did not adopt or
release jobs. Keep the legacy `/home/ws/ws/realworld-ops/pre/STOP` present.

```bash
M=/home/ws/ws/splatter4d
W=$M/.worktrees/lpa
PY=$M/.venv/bin/python
P=/home/ws/ws/realworld-ops/dsreset-20261113/pipeline
B=$M/runs/local-phase-a-qualification-20261115
C=$(git -C "$W" rev-parse HEAD)

# The adoption helper requires origin/main to equal the candidate.
# COORDINATOR ONLY, after review and passing suite; no push was performed here.
git -C "$W" push origin "$C:refs/heads/main"

# Passing full-suite evidence must match this candidate's sources.
# Recheck expected main if another release has been adopted meanwhile.
"$PY" -I "$M/runs/setup/tools/merge_adopt_local_mig_20261112.py" \
  "$C" 093e044d37d53313afee952d0435000f3eaee990 "$W" --dry-run
# Normal reviewed adoption, only after reviewing the incomplete equivalence status:
"$PY" -I "$M/runs/setup/tools/merge_adopt_local_mig_20261112.py" \
  "$C" 093e044d37d53313afee952d0435000f3eaee990 "$W"

# Read-only comparison of the already-generated ingest/sensor evidence (exit 3
# is EXPECTED while FS/SAM3 are absent). No episode processing in this command.
"$PY" -I -B -c 'import sys;sys.path.insert(0,sys.argv.pop(1));from ops.jobs.local_phase_a_gate import main;raise SystemExit(main())' \
  "$M" --bundle "$B" --cpu-only --compare-only

# FULL two-episode GPU qualification, when the designated MIG is idle and a
# longer gate window is available. This is NOT a bulk run. Existing CPU stages
# resume by fingerprint. Preserve full SAM3 video context.
CUDA_VISIBLE_DEVICES=MIG-c3511b41-5f78-52f1-9b93-9c6794cdbbdd \
OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 \
"$PY" -I -B -c 'import sys;sys.path.insert(0,sys.argv.pop(1));from ops.jobs.local_phase_a_gate import main;raise SystemExit(main())' \
  "$M" --bundle "$B"

# Preview: never writes queue, registry, placements, holds, or job inputs.
cd "$M"
"$PY" -I -B runs/setup/tools/register_local_phase_a_20261115.py \
  --bundle "$B" \
  --plan "$P/drawer-canonical-manifests/prep-plan.json" \
  --plan "$P/covering-canonical-manifests/prep-plan.json" \
  --plan "$P/pick-and-place-canonical-manifests/prep-plan.json" \
  > "$B/coordinator-preview.json"
```

Stop here unless `equivalence.json: qualified=true` and measured inference
footprints are present. The next commands select R2 and allow bulk preprocessing
through the existing scheduler. They were NOT executed by the implementer.
The R1 prepare-when-reachable helper can launch automatically; disabling only the
old whole-GPU dispatcher does not disable it. Creating the R1 queue directory with
STOP also makes that helper refuse new preparation. If R1 already launched any
job, reconcile its running/unknown states before selecting R2; never force-clear them.

```bash
Q=$M/runs/dsreset-phaseA-remote-20261114
mkdir -p "$Q/state"
touch "$Q/state/STOP"

# Materialize NEW immutable job inputs once, then append exact-ID HELD rows.
"$PY" -I -B runs/setup/tools/register_local_phase_a_20261115.py \
  --bundle "$B" \
  --plan "$P/drawer-canonical-manifests/prep-plan.json" \
  --plan "$P/covering-canonical-manifests/prep-plan.json" \
  --plan "$P/pick-and-place-canonical-manifests/prep-plan.json" \
  --stage --execute > "$B/coordinator-registration.json"

# Existing place helper: dry-run first. Only the requested new IDs are eligible;
# existing ownership, scopes, priorities, pins and started jobs are not rewritten.
"$PY" -I -B - "$B/coordinator-registration.json" <<'PYCODE'
import json,subprocess,sys
ids=[r['id'] for r in json.load(open(sys.argv[1]))['rows']]
subprocess.run([sys.executable,'-I','runs/setup/tools/place_local_phase_a_20261115.py',*ids],check=True)
PYCODE

# BULK RELEASE: established realworld_local.apply, then release ONLY moved IDs
# from the new exact-ID hold. Normal scheduler priority/capacity admission applies.
"$PY" -I -B - "$B/coordinator-registration.json" <<'PYCODE'
import json,subprocess,sys
ids=[r['id'] for r in json.load(open(sys.argv[1]))['rows']]
subprocess.run([sys.executable,'-I','runs/setup/tools/place_local_phase_a_20261115.py','--execute',*ids],check=True)
PYCODE

# Once reachable, publish each completed production episode. Preview first.
EP=covering-failure-episode_000076
"$PY" -I -B runs/setup/tools/upload_local_phase_a_20261115.py --episode "$EP"
"$PY" -I -B runs/setup/tools/upload_local_phase_a_20261115.py --episode "$EP" --execute
```

Publication uses RemoteClient's checked upload into remote `incoming/`, validates
the archive SHA and EVERY extracted file SHA, then atomically renames a sibling
staging directory to the canonical remote suffix. It rejects symlinks, archive
escapes, incomplete episodes and conflicting destinations. An identical existing
destination is accepted idempotently. It never overwrites a canonical result.
Pipeline/run files retain original local paths as provenance; phase B uses the
canonical regenerated remote pipeline configs, not these local paths as workers.

For a NEW qualification snapshot after any execution-source/weight change, use a
NEW bundle (never replace the existing frozen snapshot):

```bash
BNEW=$M/runs/local-phase-a-qualification-next
"$PY" -I -B -c 'import sys;sys.path.insert(0,sys.argv.pop(1));from ops.jobs.local_phase_a import main;raise SystemExit(main())' \
 "$M" prepare --source "$M" --bundle "$BNEW"
```

The existing bundle's immutable runner source is `d99fd98d7d9c21cbe48f73641c795753a5e746f7`.
Later candidate changes are comparator/publication guards, tests and documentation;
registration proves the frozen runner's complete executable import/config scope is
identical to adopted main. This permits documentation-only final commits without
rewriting a qualified snapshot or falsifying its worker source commit.
