# BH3 — exclude recorder-rejected episode; ready for Claude adoption

Implementation commit: **0310256b8a4ab8fb2443292ebd65516661f9a802**, branch `bh-20261120`,
worktree `/home/ws/ws/splatter4d/.worktrees/bh`. Main `931c8a6` was merged first.
Cherry-pick ONLY this implementation commit; the preceding merge need not be adopted.

Completed: hash-bound membership, canonical split, all prep/publication/runtime consumers,
new immutable RGB/calibration staging paths, selective advancer failure handling, focused tests.
Remote staging and controller actions below are **Claude-only and have NOT been executed**.
Local review-only bundles are concrete and inspectable at
`/home/ws/ws/splatter4d/.worktrees/bh/evidence/bh3-review-bundles/`:
`ingest99/`, `rgb2/jobs.json`, `calib2/{jobs.json,spec.json,FREEZE.json}`, `summary.json`.
These are planning evidence, not runnable remote attachments; do not copy their dispatcher into a live bundle.
One RGB job requires 99 normalize IDs; calibration has **253** jobs (was 255), view requires 99 Phase-A IDs.

No dataset, original ingest/raw proof, running normalize/Phase-A bundle/controller, main working file,
BH2 worktree, live queue/registry/pause/placement/role release or scheduler file was changed.
No remote connection, registration or launch was performed. Only BH worktree/git, /tmp and this requested
handoff were written. CPU runs were nice 19, one Torch/BLAS thread, CUDA hidden/uninitialized.

## Evidence and exact membership

`configs/realworld/gate_amendments/ball-hiding-99-20261121.json` is the standard
`rw-gate-amendment/1` decision, accepted by the shared amendment loader. It binds all 100 original hashes,
99 active and one excluded, and hashes `docs/ball-hiding/AMEND-BH3-99.md`.

Excluded `ball-hiding/success/episode_000008.h5`:
- camera `cam_135122073201`, recorder dataset `stereo/depth_ok`, value 0/False;
- **zero-based frame index 121**, 150 entries on that camera;
- H5 SHA256 `364c3eff063cd9ab495a2455a9bd7392dba8a8a3583de199e74b29858833c322`;
- its other validity arrays, and every validity array on the other two cameras, are nonzero.
- Strict `panda_teleop.read_meta` rejection is unchanged. No repair and no waiver.

Canonical read-only `check-data --full`: passed=true, problems=[], 99 eligible, **89 train / 10 validation**.
Canonical JSON manifest identity (`sha256(json.dumps(m, sort_keys=True).encode())`):
`6b70e16d5e8e7ea7006f6508b8704ac1f38218cfe71427d70cfbfd6f0e3159e9`.
Report: `.worktrees/bh/evidence/bh3-check-data.json`.
Arrival still verifies all 100 physical files and both original transfer SHA lists; the rejected file stays present.
A different hash/flag/frame, changed included identity or changed split fails closed.

The existing `s4d.rwlearn.dsreset.paired_manifests` and `s4d.rwlearn.store.split_episodes` agree:
sort the available episodes, `np.random.default_rng([0, int(sha256(task)[:8],16)])`, permutation,
first `ceil(0.1*n)` validation. Drawer precedent is explicit in
`docs/dsreset-20261113/AMEND-drawer-open-v2-98.md`; covering's amendment explicitly names `store.split_episodes`.
98 -> 88/10; **99 -> 89/10**. No legacy 100-episode split is preserved after removal.

New validation suffixes:
`000005, 000038, 000043, 000048, 000057, 000071, 000074, 000080, 000086, 000097`.
Versus the original 100 split:
- removed from train: `000008`;
- validation -> train: `000037, 000042, 000047, 000056, 000070, 000073, 000079`;
- train -> validation: `000038, 000043, 000048, 000057, 000071, 000074, 000080`;
- validation unchanged: `000005, 000086, 000097`; every other eligible episode stays train.

**BH2 rows MUST read `/home/ws/ws/splatter4d/configs/realworld/tasks/ball-hiding-split99.json` after
cherry-pick** (repo-relative `configs/realworld/tasks/ball-hiding-split99.json`). Its top-level `train` and
`validation` arrays are the original label-inclusive episode IDs; it binds the amendment SHA and records all
membership changes. Do not read the old `W/ingest/S/manifest.json` or infer 90/10 from raw count 100.
Runtime row inputs must use their view's published `manifest.json` and `split_seed0.json`, whose membership is
checked against this same decision. Spec `collection.expected_success=100` is intentionally the physical arrival
inventory, not training size. New canonical prep manifest: `W/bh3/ingest99/S/manifest.json`.

Normalize and Phase A keep their original 100-job immutable queues/states. Consumers select 99 exact identities;
no completion is fabricated for episode 8. The advancer ignores only its exact prepN/prepA job ID; missing/pending
eligible jobs keep waiting, any other failed/unknown/unexpected job and any STOP still holds. Advancers no longer
start or otherwise touch normalize/Phase-A controllers. Calibration specs/freezes bind the amendment; Phase B gets
a separate 99-row eligible plan, leaving the old Phase-A plan untouched. Geometry verifies original queue/state
hashes before selecting 99, retains one invalid exclusion evidence record in its private snapshot, and publishes
99 stores with train-only measurement over 89. Calibration acceptance and all scientific settings remain unchanged.

## Validation

Requested harness only, no full suite:
- `evidence/tests-bh3-verified/`: **93 passed** across BH exclusion/prep + PNP geometry/remote/pipeline files.
- After the last RGB-fetch count fix: `evidence/tests-bh3-fetch/`: **31 passed**, including the added 99-episode
  RGB-fetch regression and all PNP remote-route tests. These overlap; do not add the counts as unique tests.
- The focused tests cover standard amendment loading/document hash, deterministic store split, exact recorder
  rejection, split/hash tampering, excluded failure versus every other failure, STOP/missing/pending states,
  99 remote RGB roots, original queue hash checks, fresh source pin, 253-job calibration and 99-row Phase-B
  plans without rewriting Phase A, accepted/held/override rig behavior, snapshot/release/measurement and fetch/upload.
- Peak parent RSS 798720 KiB; Torch threads=1; cuda_initialized=false. Harness startup GPU enumeration is replaced
  only inside this CPU test process, recorded in resources.json. Product GPU guards remain intact.
- `git diff --check`, changed Python syntax and JSON parsing passed.

## Exact Claude-only commands, in order

Run from main only after reviewing this handoff. No commands below register/place training, edit the scheduler,
or restart normalize/Phase A. Staging uses existing CPU planner admission; controller starts invoke normal guarded
remote launches. Do not rerun immutable staging into a partially created attachment: inspect a failure first.

### 1. Cherry-pick

```bash
set -euo pipefail
M=/home/ws/ws/splatter4d
W=/home/ws/ws/realworld-ops/ballhiding-20261120
PY="$M/.venv/bin/python"
cd "$M"
git cherry-pick 0310256b8a4ab8fb2443292ebd65516661f9a802
export CUDA_VISIBLE_DEVICES= OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
```

### 2. Stop old idle RGB/calibration controllers; quiesce old advancers

This script verifies exact process arguments before sending SIGTERM. It refuses if an old bundle has any
attempt/start/running evidence. It writes only the explicitly authorized old idle bundles' local state/STOP
markers; old queue/config/input bytes stay untouched. It does not write a global STOP or touch normalize/Phase A.

```bash
nice -n 19 ionice -c 3 "$PY" -I - <<'PY'
import json, os, signal, time
from pathlib import Path
M=Path('/home/ws/ws/splatter4d')
W=Path('/home/ws/ws/realworld-ops/ballhiding-20261120')
def args(pid):
    try:return [v.decode() for v in Path(f'/proc/{pid}/cmdline').read_bytes().split(b'\0') if v]
    except FileNotFoundError:return []
def stop(pid, expected):
    current=args(pid)
    if not current:return
    assert all(x in current for x in expected), f'HELD: PID {pid} does not match expected command'
    os.kill(pid, signal.SIGTERM)
    for _ in range(100):
        if not args(pid):return
        time.sleep(.2)
    raise RuntimeError(f'HELD: controller/advancer PID {pid} did not exit')
# Old advancers may already have exited after episode 8 failed; do not kill unrelated reused PIDs.
for branch in ('rgb','geometry'):
    pidfile=W/'logs'/f'advance-{branch}.pid'
    if pidfile.exists():
        pid=int(pidfile.read_text());current=args(pid)
        if str(M/'tools/ball_hiding/advance.py') in current:
            stop(pid,[str(M/'tools/ball_hiding/advance.py'),'--branch',branch])
        elif current:
            print('stale advancer PID file; unrelated PID left untouched:',pid)
for alias in ('rgb-remote','calib'):
    bundle=(W/alias).resolve(strict=True)
    assert bundle.name in ('bh-20261120-rgb','bh-20261120-calib')
    state=bundle/'state/state.json'
    if state.exists():
        jobs=json.loads(state.read_text())['jobs']
        assert all(s.get('attempt',0)==0 and s.get('status')=='pending'
                   and not s.get('container') for s in jobs.values()), f'HELD: {alias} was used'
    config=str(bundle/'dispatcher.json')
    found=[]
    for p in Path('/proc').iterdir():
        if p.name.isdigit():
            current=args(int(p.name))
            if all(x in current for x in ('ops.jobs.rwprep_remote','tick','--apply','--loop',config)):
                found.append(int(p.name))
    assert len(found)<=1, f'HELD: multiple {alias} controllers'
    (bundle/'state/STOP').touch(exist_ok=True)
    for pid in found:stop(pid,['ops.jobs.rwprep_remote','tick','--apply','--loop',config])
    print(alias,'STOP retained; idle controller stopped',found)
PY
```

### 3. Validate/build NEW 99-episode manifest; stage NEW RGB then calibration bundles

The RGB stage publishes the current code snapshot (not normalize's older commit); calibration uses that new
snapshot and pins its controller to the same commit. This order is required. Names and aliases are new:
`runs/bh-20261120-rgb2` <-> `W/rgb2-remote`, `runs/bh-20261120-calib2` <-> `W/calib2`.

```bash
nice -n 19 ionice -c 3 "$PY" -I scripts/ball_hiding.py check-data --full > "$W/bh3/check-data99.json"
nice -n 19 ionice -c 3 "$PY" -I scripts/ball_hiding.py --entry manifest
nice -n 19 ionice -c 3 "$PY" -I scripts/ball_hiding.py --entry rgb-remote
nice -n 19 ionice -c 3 "$PY" -I scripts/ball_hiding.py --entry calib-remote
# Verify the attached jobs before any new controller starts.
nice -n 19 ionice -c 3 "$PY" -I - <<'PY'
import json
from pathlib import Path
W=Path('/home/ws/ws/realworld-ops/ballhiding-20261120')
m=json.loads((W/'bh3/ingest99/S/manifest.json').read_text())
assert len(m['episodes'])==99 and tuple(map(len,(m['split']['train'],m['split']['validation'])))==(89,10)
for alias,count,upstream in (('rgb2-remote',1,'prepN'),('calib2',253,'prepA')):
    b=(W/alias).resolve(strict=True)
    jobs=json.loads((b/'jobs.json').read_text());assert len(jobs)==count
    dependencies=jobs[0]['external_states'];ids=next(iter(dependencies.values()))
    assert len(ids)==99 and all(f'-{upstream}-' in i and 'episode_000008' not in i for i in ids)
    assert not (b/'state/STOP').exists()
    print(alias,b,'jobs',len(jobs),'eligible upstream',len(ids))
r=json.loads((W/'rgb2-remote/dispatcher.json').read_text())
c=json.loads((W/'calib2/dispatcher.json').read_text())
assert r['commit']==c['commit'] and r['code']==c['code']
assert r['remote'].endswith('/bh-20261120-rgb2') and c['remote'].endswith('/bh-20261120-calib2')
PY
```

### 4. Start only the NEW controllers

```bash
nice -n 19 ionice -c 3 "$PY" -I tools/ball_hiding/start_tick.py "$W/rgb2-remote"
nice -n 19 ionice -c 3 "$PY" -I tools/ball_hiding/start_tick.py "$W/calib2"
```

### 5. Restart advancers with the adopted code

```bash
nice -n 19 ionice -c 3 "$PY" -I tools/ball_hiding/advance.py --detach
nice -n 19 ionice -c 3 "$PY" -I scripts/ball_hiding.py status
```

Advancers start no normalize/Phase-A controllers. After eligible barriers they perform normal RGB fetch/upload,
accepted rig fetch, 99-job Phase B staging/controller, and 99-episode geometry staging/controller/fetch/upload.
If G1–G6 holds, or any eligible upstream fails, the hold remains a real hold; no override is authorized here.
BH2 training adoption/registration is separate and must consume the exact file contract above.

## Optional launch-speed audit — proposal ONLY, not applied

Static inspection identifies a serialized launch path, not an explicit 80-second sleep:
`rwprep_remote.tick` calls `launch_job` and then `parallel.snapshot` before the next job.
`launch_job` publishes one worker JSON through `rwprep_launch.push_bytes`; `_launch` publishes a second launcher
JSON, then executes the original locked guard and Docker launch. Each fresh `push_bytes` does remote existence/hash
checks, mkdir, a separate rsync and checksum/no-replace publication. The guard rechecks source/input identities,
STOP/deadline, duplicate episode/name, image, tenant/resource quotas and locks. The following snapshot performs
fresh MIG probes (cached per parent within one snapshot), Docker listings, serial per-container inspect/top,
cgroup and disk/RAM reads and a one-second CPU-idle sample. All are serialized per launched job.

The exact split of the observed ~80 seconds among transfers/locks/guard/source verification/snapshot is **unmeasured**;
no remote profiling or timing instrumentation was added. Do not claim that the one-second sample explains 80 seconds.
A safe candidate for a future controller: batch the two immutable small-file publications into one transport
transaction while preserving per-file SHA256 and atomic no-replace checks; batch Docker inspect queries inside
each fresh snapshot with equivalent per-container/error handling. Keep the fresh per-job snapshot and every original
locked admission/resource/source/input/STOP/exclusivity check. First add phase timings, then benchmark with synthetic
or new isolated controllers. No snapshot caching, optimistic resource reuse, admission relaxation or change to the
running normalize/Phase-A controllers is proposed for this urgent fix.
