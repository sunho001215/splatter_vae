# BH1 final implementation handoff — automatic geometry route ready

Adopt implementation commits **32ad47abd0f4d67434348f151148b9a2f29ccb08** and
**c70812e3212365b004e7c5e6bb3760941703cbfa** from branch `bh-20261120`, worktree
`/home/ws/ws/splatter4d/.worktrees/bh`, base `d7a1148`. Stage-1 delivery is `HANDOFF-BH1-stage1.md` (doc commit
`8be05ce`). `DAG-BH1.md` contains the full method/input/row analysis and streaming limits.

CONFIRMED
- Thin process-local BH task context reuses pnp-v2 modules. Original pnp-v2 defaults, full-data rules, scientific
  stage configs, losses, protocol and seeds are preserved. Only BH uses the independent normalization branch.
- Release `bh-20261120`; task `ball-hiding`; S only; exactly 100 successes, original seed-0 90/10 split.
- Task-specific automatic rig route: `rig_loma-release/2`, original G1–G6, frozen full collection, no exceptions or
  BH override. Calibration plan commands/resources/dependencies equal pnp-v2 after task/path substitution (tested).
  Accepted rig: `/home/compu/kaist/sunho/realworld/processed/bh-20261120/calibration/ball-hiding/rig-ball-hiding-accepted.json`
  and matching `.reference.npz`. The pnp-v2 rig is never reused.
- Phase B uses the original strict automatic compiler and measured rig, original calibrate/TrackEverything/motion/
  compact-store/contracts/QA pipeline; 4 GiB declared disk. A held or overridden rig fails before publication (tested).
- Geometry worker selects BH before importing shared modules; verifies original Phase-A/B queue/config hashes and
  states, all raw hashes and rig identity, creates a NEW snapshot, writes calibration_validated records through the
  original release writer, publishes the original geometry inputs, and runs the same TRAIN-only native scene estimator.
  All 100 episodes stay gated; native sampling uses only the 90 train episodes. No live ingest catalog edits.
- Geometry fetch verifies/localizes the complete compact closure and rig/reference/measurement. Upload stages verified
  copies under repository runs/, refuses conflicting remote bytes, and verifies the canonical mirror and receipts.
- Detached RGB and geometry branches have separate locks/logs; concurrent normalization-controller starts are
  serialized. Each branch starts its own required controller, propagates its own failed/unknown/STOP upstream state,
  retries only planner admission deferrals, and advances fetch/upload without registration, placement, role release,
  scheduler daemon changes, job deletion, repairs, or acceptance overrides.
- Focused geometry/calibration/worker selection: **120 passed**, `evidence/tests-stage2-final/` + log.
  Final BH-specific selection: **16 passed**, `evidence/tests-bh-final/` + log; includes synthetic snapshot/release/
  measurement/resume, full mirror/upload, held/override refusal, identical calibration plan, existing/conflicting raw
  file verification, uncapped partial-upload options/postpublish SHA proof, and split hash/dependency integration.
  Stage 1: **80 passed, 1 deselected**, `evidence/tests-stage1/` + log.
  No full suite. CPU harness records the in-process GPU-startup-check replacement; CUDA hidden/uninitialized,
  one Torch/BLAS thread, peak parent RSS ~0.6 GiB (`resources.json`). The one legacy GPU-guarded child startup test
  is deselected; product GPU guards are unchanged. `git diff --check` and syntax checks passed.
- Main working tree, other worktrees, datasets, live queue/registry/pauses/placement/releases and running jobs were
  not modified. No Robot-PC or remote access occurred. Only the requested worktree/git branch and /tmp were written.

DATA VERDICT: **CONFIRMED READY for PREP.** `success.VERIFIED` is present; canonical `check-data --full` exited 0,
`passed=true`, `problems=[]`, with both SHA lists present and exactly the expected 100 episodes (0..101 except 25/54).
Total 20,143,791,322 bytes; original seed-0 split: **90 train / 10 validation**, no exclusions.
Canonical manifest SHA256: `b58ea733f6f6ff2a80d4bca5f9a66c07e49de75cd50d4a2b1e0a951c3f0ef1ca`.
Evidence: `evidence/check-data-full.json`, `data-summary.json`, `data-metrics.json`, `anomaly-triage.json`,
`episode_*.json`; `sha256-index.json` identifies the evidence bytes. This final verdict supersedes Stage 1's partial snapshot.
All 100/100 canonical episode audits passed. All 100 pnp reference episodes share one field schema, matched by BH;
no missing/added/type/shape field differences, no nonmonotonic audited clocks, all numeric robot/action/gripper fields
finite, sensor depth fields in all three cameras; sampled first/mid/last RGB/IR/depth decode successfully.
RGB 14.988–14.990 Hz, 137–203 frames per camera;
robot ~100 Hz, gripper 62.402–62.476 Hz; robot spans 9.19–13.59 s.
11 episodes have camera pause gaps >100 ms, all between recorded segments; zero within-segment camera gaps >100 ms
(within-segment maximum 66.767 ms). 67 episodes have nearest-camera offsets >20 ms, maximum 62.741 ms; these remain
advisory under the unchanged motion-based M5 criterion. No timestamp repair or episode exclusion.
Validation episode suffixes: 000005, 000037, 000042, 000047, 000056, 000070, 000073, 000079, 000086, 000097.

Prompts: `arrival.json`, reviewed `evidence/contact-reviewed.jpg` (episodes 0/9/17, all cameras, first/mid/last);
the final `contact-sheet.jpg` also confirms episodes 0/51/101, all cameras, first/mid/last.
robot arm/end effector=robot; red ball/metal cup=object; wooden table=fixture. Same confirmed Panda/Robotiq URDF hash
`ca60bab9732bcad7b5cd0b5c12bd83f48a49fbb523ab5df23a3403f063080051`. The unchanged scene estimator needs all three
classes; SAM quality remains a worker QA condition.

PENDING RUNTIME EVIDENCE: numerical rig acceptance, Phase A/B products, SAM masks, native scene containment and actual
remote publication are **not claimed complete**. Hard no-GPU/no-launch/no-remote-write instructions require Claude to
execute the commands. A G1–G6 hold is a real hold; this handoff authorizes no override.

## Exact Claude-only commands (not executed here)

From main; if Stage-1 code was already adopted, cherry-pick only `c70812e`. The two implementation commits contain
only this task's named source/tests/tools/config/DAG files; they do not include live-state files or generated evidence.

```bash
M=/home/ws/ws/splatter4d
BH="$M/.worktrees/bh"
W=/home/ws/ws/realworld-ops/ballhiding-20261120
PY="$M/.venv/bin/python"
cd "$M"
git cherry-pick 32ad47abd0f4d67434348f151148b9a2f29ccb08 c70812e3212365b004e7c5e6bb3760941703cbfa
mkdir -p "$W/logs" "$W/evidence"
cp -n "$BH/arrival.json" "$W/arrival.json"
cmp "$BH/arrival.json" "$W/arrival.json"
cp -n "$BH/evidence/contact-reviewed.jpg" "$W/evidence/contact-reviewed.jpg"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=
```

Use Stage-1 commands for check-data, ingest, raw upload, raw-proof and initial normalize/Phase-A staging. Then choose
the detached advancer **or** these manual commands, in order at each real completion barrier:

```bash
# Stage calibration early: its view waits for every original Phase-A job, then the frozen automatic DAG runs.
nice -n 19 ionice -c 3 "$PY" -I scripts/ball_hiding.py --entry calib-remote
nice -n 19 ionice -c 3 "$PY" -I tools/ball_hiding/start_tick.py "$W/calib"
# After calibration publish is done (no held cameras):
nice -n 19 ionice -c 3 "$PY" -I scripts/ball_hiding.py --entry rig-fetch
nice -n 19 ionice -c 3 "$PY" -I scripts/ball_hiding.py status
# Phase B independently rechecks strict automatic rig acceptance before staging:
nice -n 19 ionice -c 3 "$PY" -I scripts/ball_hiding.py --entry phase-b-remote
nice -n 19 ionice -c 3 "$PY" -I tools/ball_hiding/start_tick.py "$W/phaseB"
# After all original Phase-B jobs are done:
nice -n 19 ionice -c 3 "$PY" -I scripts/ball_hiding.py --entry geometry-remote
nice -n 19 ionice -c 3 "$PY" -I tools/ball_hiding/start_tick.py "$W/geometry-remote"
# After the geometry publication job is done:
nice -n 19 ionice -c 3 "$PY" -I scripts/ball_hiding.py --entry geometry-fetch
nice -n 19 ionice -c 3 "$PY" -I scripts/ball_hiding.py --entry upload -- --view geometry
nice -n 19 ionice -c 3 "$PY" -I scripts/ball_hiding.py status

# Preferred detached independent branches, after Phase-A staging; use instead of manual downstream commands:
nice -n 19 ionice -c 3 "$PY" -I tools/ball_hiding/advance.py --detach
# Or only the geometry branch:
setsid nohup nice -n 19 ionice -c 3 "$PY" -I tools/ball_hiding/advance.py --branch geometry \
  > "$W/logs/advance-geometry.log" 2>&1 < /dev/null &
```

Do not rerun staging against an existing immutable attachment. Partial/conflicting bundles/fetches hold for inspection;
verified geometry fetch supports equality-only resume. No command above registers/places training or changes scheduler
state. Training task configs/wrappers and release/admission adoption are outside BH1 and must come from the coordinator's
other task; the informational row graph in BH status is not a registration implementation.
