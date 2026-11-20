# BH1 Stage 1 — ready for coordinator review/adoption

Implementation commit: **32ad47abd0f4d67434348f151148b9a2f29ccb08**, branch `bh-20261120`, base `d7a1148`.
Worktree: `/home/ws/ws/splatter4d/.worktrees/bh`. The handoff lives here because main's working tree must stay untouched.

CONFIRMED
- No Robot-PC access, GPU use, remote access/staging/launch, registration/placement, live-state edit, or dataset write.
- 59 locally complete episodes audited at this snapshot; **59/59 canonical lightweight validation passed**.
  All 100 pnp-v2 reference episodes have one identical field/type/shape schema; BH matches it, field by field.
  Evidence: `evidence/data-summary.json`, `data-metrics.json`, `pnp-all-reference-schemas.json`, `episode_*.json`.
  Transfer is still partial, `success.VERIFIED` absent. Full-data verdict is **PENDING**, never inferred from partial QA.
- Three RealSense D435I serials: 135122070988, 135122073201, 135122076054; RGB/stereo IR/sensor depth present.
  Observed RGB 14.988–14.990 Hz; robot ~100 Hz; gripper ~62.45 Hz; camera frames 137–181, recorded spans ~9–13.2 s.
  No nonmonotonic clocks. Some segment/poll gaps, max robot ~1.406 s / camera ~1.534 s; nearest camera offset reaches
  ~62.74 ms. These are flagged per episode; existing segment/M5/action gates decide eligibility without repair.
- `arrival.json` in this worktree confirms robot arm/end effector=robot, red ball/metal cup=object,
  wooden table=fixture. `evidence/contact-sheet.jpg` + `contact-sheet.json` identify exact all-camera first/mid/last
  samples. Table fixture supports the unchanged scene estimator's mandatory robot/object/fixture classes.
  Same Panda/Robotiq 2F-85; URDF SHA `ca60bab9732bcad7b5cd0b5c12bd83f48a49fbb523ab5df23a3403f063080051`
  rehashed locally (`evidence/urdf.json`) and rechecked again by ingest.
- S only, 100 successes, original deterministic seed-0 90/10 split; scientific settings and pnp-v2 defaults preserved.
  Release `bh-20261120`; all bundles/jobs/processed/publication roots are separate from pnp-v2/dsreset.
- `DAG-BH1.md`: actual inputs for all ten method labels and all gate/stats/native/pre/export/fqldata/fql/eval/taskeval
  row kinds, with file:line evidence. RGB reads normalized frames/robot/actions alone. Separate normalization jobs
  release RGB without waiting for FS/SAM/calibration. Phase A streams per corresponding normalized episode.
  Whole-task rig, Phase B and geometry barriers remain. BH1 exposes PREP only; training/registration belongs to
  the coordinator's separate work. Do not register the informational BH status row graph as training support.
- Focused existing `run_tests.py` worker selection: **80 passed, 1 deselected**, no full suite.
  `evidence/tests-stage1/{results.json,junit.xml,worker.json}`, `evidence/tests-stage1-final.log`.
  Includes existing exact RGB/M5/proprio/DynaMo/ReViWo equality test and fake two-bundle/hash/dependency integration.
  CPU harness hides CUDA and replaces only the in-process startup GPU enumeration check. One existing test launches
  a GPU-guarded child and is explicitly deselected to obey no-GPU. Its initial failure is retained in
  `evidence/tests-first.log`; BH CPU refusal path is covered. This is development evidence, not full-suite release acceptance.
- Four submodules initialized from local refs only. `git diff --check` and compile checks passed.

HYPOTHESIS / pending runtime evidence
- SAM quality and automatic rig acceptance require coordinator-run workers. No override exists for BH and none is allowed.
- No claim that HDF5 container timestamps/provenance from different runs are identical; scientific consumed arrays,
  selection and settings use the existing byte-equality-tested exporter unchanged.
- Partial transfer cannot establish final 100-episode integrity. Rerun the audit and check-data after VERIFIED.

## Exact Claude-only commands

These commands are prepared, **not executed by BH1**. Existing main operator edits are preserved; adopt just this
commit's named files. Do not run any pnp-v2 registration/release/placement scripts.

```bash
M=/home/ws/ws/splatter4d
BH="$M/.worktrees/bh"
W=/home/ws/ws/realworld-ops/ballhiding-20261120
PY="$M/.venv/bin/python"
cd "$M"
git show --stat 32ad47abd0f4d67434348f151148b9a2f29ccb08
git cherry-pick 32ad47abd0f4d67434348f151148b9a2f29ccb08
mkdir -p "$W/logs"
cp -n "$BH/arrival.json" "$W/arrival.json"
cmp "$BH/arrival.json" "$W/arrival.json"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=

# Read-only while arrival is partial; check-data correctly refuses until complete.
nice -n 19 ionice -c 3 "$PY" -I scripts/ball_hiding.py status
# After transfer marker is present (the tool checks both SHA lists and exact episode membership):
nice -n 19 ionice -c 3 "$PY" -I tools/ball_hiding/verify_data.py --out "$W/evidence"
nice -n 19 ionice -c 3 "$PY" -I scripts/ball_hiding.py check-data --full > "$W/check-data.json"
nice -n 19 ionice -c 3 "$PY" -I scripts/ball_hiding.py --entry manifest

# Uncapped only inside these upload rsync processes; no persistent budget is modified.
# Resumes partial remote files; refuses any conflicting canonical bytes; rehashes existing files too.
nice -n 19 ionice -c 3 "$PY" -I tools/ball_hiding/upload_raw.py > "$W/logs/upload-raw.log" 2>&1
# Requires upload/ball-hiding__success.REMOTE_VERIFIED and remote-sha256.json for evidence.
nice -n 19 ionice -c 3 "$PY" -I scripts/ball_hiding.py --entry raw-proof

# Stage only, no launch: separate normalize and Phase-A bundles, 4 GiB disk per episode job.
nice -n 19 ionice -c 3 "$PY" -I scripts/ball_hiding.py --entry phase-a-remote
# Claude starts each controller. Jobs remain subject to original remote admission/locks.
nice -n 19 ionice -c 3 "$PY" -I tools/ball_hiding/start_tick.py "$W/normalize"
nice -n 19 ionice -c 3 "$PY" -I tools/ball_hiding/start_tick.py "$W/phaseA"
nice -n 19 ionice -c 3 "$PY" -I scripts/ball_hiding.py --entry rgb-remote
nice -n 19 ionice -c 3 "$PY" -I tools/ball_hiding/start_tick.py "$W/rgb-remote"

# Once the RGB publication job is done, manual equivalent of the RGB advance branch:
nice -n 19 ionice -c 3 "$PY" -I scripts/ball_hiding.py --entry rgb-fetch
nice -n 19 ionice -c 3 "$PY" -I scripts/ball_hiding.py --entry upload -- --view rgb
nice -n 19 ionice -c 3 "$PY" -I scripts/ball_hiding.py status
```

Detached alternatives after initial Phase-A staging (choose these instead of manual downstream fetch/upload):

```bash
# RGB only, independent of all depth/SAM/calibration progress:
setsid nohup nice -n 19 ionice -c 3 "$PY" -I tools/ball_hiding/advance.py --branch rgb \
  > "$W/logs/advance-rgb.log" 2>&1 < /dev/null &
# Both branches, separate detached sessions/logs/branch locks; geometry route verification continues in Stage 2:
nice -n 19 ionice -c 3 "$PY" -I tools/ball_hiding/advance.py --detach
```

`advance.py` never registers or places. It propagates only each branch's upstream failed/unknown/STOP states;
planner admission exit 20 is retried. Existing dispatcher attachments/publications are immutable: a partial/conflicting
staging or fetch stays held for inspection. `STOP` in `$W` stops only these advance helpers; it does not kill jobs or
controllers. No cleanup/deletion/override command is proposed.
