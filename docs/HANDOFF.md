# splatter4d campaign handoff — planned local hardware shutdown

**Phase3 permitted controllers stopped by 2026-10-16 08:26:16 KST; final local verification08:28:31. All-local shutdown remains BLOCKED by the denied scheduler-daemon stop. All times are KST.**
Repository `/home/ws/ws/splatter4d`, branch `splatter4d`, origin
`https://github.com/sunho001215/splatter_vae.git`. Start with this document, `PROGRESS.md`, then the append-only
`experiments/registry.jsonl`. Do not infer an all-local shutdown: PID2819571 and its transport activity remain; remote containers were not stopped.

## 1. Goal, frozen method and evidence

Learn a camera-invariant, dynamically structured representation from offline Meta-World RGB-D videos, then evaluate
frozen encoders with DrM on eight tasks: hammer, pick-place, bin-picking, coffee-push, door-open, peg-insert-side,
peg-unplug-side and stick-push. Development tasks are hammer/pick-place; final non-development RL never selects the
method. No DROID experiments; the historical DROID adapter work is smoke-only.

**Frozen `method-frozen-v1`: C + D0 + M3D, 200,000 pretraining steps, seed 0.** Tag/source:
`226d8ae89b6f65097d372e597e0a7734d2eee3e0`. Manifest `runs/setup/method-frozen-v1.json` includes the full eight-task
resolved configs and seed plans. Overrides of `configs/metaworld/base.yaml` plus the task YAML:

```text
train.steps=200000 train.seed=0 aug.synth_views=2 aug.synth_render=true
loss.depth_valid_far=true loss.motion_space=gaussian
```

C: synthetic near views are invariance positives and rendering targets; D0: depth/far-mask choice from the registered
rule; M3D: Gaussian-space 3D motion. The A200/A300 length study selected 200k under its unchanged prospective rule.
Hammer/pick-place final encoders reuse `runs/exports/ls-a200-m3d-seed0-{hammer,pick-place}-200k.pt`; six other tasks
have fresh final pretrains. Do not silently substitute the earlier M2D/crop screen or speculative 300k checkpoints.

Detailed method/results/rules: `docs/RESULTS.md`, `EXPERIMENT_LOG.md`, `RL_PROTOCOL.md`, `BASELINES.md`,
`COMPUTE_PLAN.md`, `REMOTE.md`, and `runs/setup/*registration*.json` / `*margins*.json`.

## 2. Code map and job flow

- `s4d/config.py`: layered YAML/overrides. `s4d/model/{encoder,decoder,gaussians,render}.py`: ViT/state tokens,
  scene/dynamic Gaussian groups, native gsplat (FP32 render). `s4d/losses/`: RGB/depth, InfoNCE, M2D/M3D, regularizers.
- `s4d/train/`: loop, augmentation (including crop/synthetic views), checkpoints, validation, workers/DDP.
  `s4d/data/metaworld/`: collection, camera sets, dataset/splits. `s4d/diag/`: probes/retrieval, geometry/panels/logging.
- `s4d/rl/`: DrM agent, frozen/CNN encoders, Meta-World/EGL envs, vector workers, replay and evaluation. Baselines
  accepted on main: `s4d/baselines/{sincro,reviwo}`. Encoder sizes/original hyperparameters stay unchanged.
- `scripts/train.py`, `evaluate.py`, `export_encoder.py`: representation pretrain → full-split evaluation/export.
  `train_rl.py` → `eval_rl.py`: trainer publishes policy snapshots; companion evaluates every 10k through 1M.
  `train_sincro.py`, `train_reviwo.py`: accepted baseline pretraining.
- `configs/metaworld/{base,tasks,ablations}`, `configs/rl/{base,tasks,encoders}`, `configs/baselines`,
  `configs/hosts/{local,remote}.yaml`: scientific and guarded host configurations.
- `analysis/eval_config.py`: creates the evaluation run config from the pretrained config; `review_rules.py`:
  registered development comparisons. Never use development margins as evidence that effects do not exist.
- `scripts/jobs.py` is the **only scientific-job launcher**. `experiments/queue.yaml` defines jobs/dependencies;
  `experiments/registry.jsonl` is append-only; `jobs.read_registry()` reconstructs state. Detached local sessions
  survive this conversation. Remote launches are detached, owned `s4d-` Docker containers.
- `experiments/HOLD` blocks default-queue admissions, not reconciliation. **Scoped queues check HOLD beside their
  own queue**, so this does not by itself stop drivers using `runs/setup/` queues.
- `runs/setup/tools/stage2_local_driver.py`: verifies Stage 2 closeouts, prepares evaluation configs, measured
  local admission. Reads original registrations plus `stage2-placement-20261015.json` execution overlay, which
  preserves the scientific definitions. Remote Stage 2 jobs use the same scientific closeout verifier.
- `runs/setup/tools/remote_campaign_driver.py`: flocked remote reconcile/closeout, data/export readiness and
  job-by-job admission; loads the placement overlay. Runtime remains the accepted immutable source/image below.
- Both drivers run every five minutes. Driver locks: `runs/setup/{stage2,remote}-driver.lock`. Logs:
  `stage2-driver-20261013.log`, `remote-driver-20261012.log`. Restart only one copy each.
- `scripts/watch_events.py --once`: **one tracked background task**, absolute script path. Watch cursor
  `experiments/watch_state.json`; heartbeat `watcher.alive`. Re-arm only after its completion.
- `pause_pretrain_at_checkpoint.py`: verified owned-session pause after next 10k checkpoint + validation and before
  the next logged step; `--reason planned_local_shutdown_20261016` writes checkpoint/hash/identity evidence.
  RL equivalent `pause_rl_at_checkpoint.py` pauses at 50k. Resume restores optimizer/scheduler/RNG, not just weights.
- `evict_partial_cache.py`: clean-page `POSIX_FADV_DONTNEED` for live-sync partials and finished RL replay/snapshots;
  no file deletion. `sync_wandb_remote.py`: checksum pull + main-host W&B upload.
- Closeout proofs: `runs/setup/s2-<job>-completion.json`, `runs/remote/finished/<job>.json`, older CNN records
  `runs/remote/stage4-finished/`. Numerical results are decision-ready only after these checks, not merely exit 0.
- `s4d/{remote_access,remote_jobs,remote_launch,run_identity}.py` and `scripts/remote/`: SSH/rsync, admission,
  container identity/isolation and provenance. `gpu_guard.py`, `scripts/_bootstrap.py`: fail-closed GPU guard.
- `scripts/run_tests.py`: full native suite and source fingerprints; no native mocks/skips or JIT acceptance.
- New baseline ports/`third_party/` exist on WIP branches, **not main**. Pinned upstreams: SNeRL d01533b,
  DynaMo 6287403, RVT 367995a, DINO 7c446df, with explicit minimal patch files. Upstream execution remains blocked.

## 3. Campaign state and exact job families

Brace notation below denotes every listed Cartesian combination, not a new naming proposal. Registry/queue and
placement manifest give the individual records. Preserve IDs, seeds, schedules and historical failures.

| Stage | State before final shutdown snapshot | Exact IDs / evidence |
|---|---|---|
| Earlier development / Stage 1 | Complete; paired representation seeds and six-seed decision proxies where registered | `RESULTS.md`, registered margins and chronological log |
| Stage 2 original RL attribution | **30/30 trainers + 30/30 companions complete**, eight comparisons | `s2rl-{full,t1,nomotion,noinfonce,singlegroup}-{hammer,pick-place}-s{1000,1001,1002}[-eval]` |
| Stage 2 representation, RL arms | 8/8 ablation pretrains/evals/exports complete | `s2ab-{t1,nomotion,noinfonce,singlegroup}-seed0-{hammer,pick-place}` and `fulleval-<id>-200k`, `export-<id>-200k` |
| Stage 2 pretrain-only, already local | Both no-synth pretrains/full evaluations complete/reported; no-consistency hammer completed200k/driver closeout, pick-place intentionally paused180k | `s2ab-{nosynth,nocons}-seed0-{hammer,pick-place}`; no-consistency full evaluations pending |
| Stage 2 pretrain-only, newly remote | Eight registered jobs placed remotely, then full evaluations | `s2ab-{nolambdadyn,nocoverage,misscaled-none,misscaled-scale}-seed0-{hammer,pick-place}`; `fulleval-<id>-200k` |
| Stage 3 final method | **24/24 RL runs complete**; six fresh pretrains/exports complete | `s3-drm-splatter4d-<task>-s{2000,2001,2002}[-eval]`; fresh `s3-m3d-seed0-<task>-200k` except hammer/pick-place |
| Stage 4 SinCro/ReViWo | **16/16 pretrains; 48/48 RL runs complete** | `s4-{sincro,reviwo}-<task>`; `s4-drm-{sincro,reviwo}-<task>-s{2000,2001,2002}[-eval]` (queue is authoritative for pretrain suffixes) |
| Stage 4 CNN | **24/24 RL runs complete** | `s4-drm-cnn-<task>-s{2000,2001,2002}`; bin-picking seed2000 uses `s4-drm-cnn-bin-picking-s2000-r4`; pick-place2000 local eval is in trainer dir |
| Crop step 1 | **4/4 read-only 100k re-evaluations complete**, reported | `creval-{base,crop}-{hammer,pick-place}-100k`; historical M2D-era single seed, not frozen-method evidence |
| Crop step 2 (10-16 06:24) | **4/4 pretrains, full evaluations and exports complete/driver-verified**; representation report done;3/12 DrM pairs running remotely,9 pending | `s2ab-{crop,cropnosynth}-seed0-{hammer,pick-place}`, eval/export `-200k`; `s2rl-{crop,cropnosynth}-{hammer,pick-place}-s{1000,1001,1002}[-eval]` |
| New baselines SNeRL/DynaMo/3D-MVP | Review-only, unaccepted, upstream execution permission-blocked | WIP branch table below; no final campaign results |

Stage 2 original registration: `runs/setup/stage2-registration-20261013.json` (108 jobs).
Crop registration: `stage2-crop-registration-20261015.json` (36), report-only registration
`crop-reeval-registration-20261015.json` (4). Placement overlay: `stage2-placement-20261015.json` (40 effective remote
job definitions: eight pretrains, eight evaluations, 12 crop RL trainer/companion pairs). Original configs/margins
are immutable; path/host/resource changes are recorded separately. v1 is preserved; v2 removes `config.yaml` from
required numerical artifacts (config is still synced/verified), after a launch was correctly refused before Docker.

**10-16 06:24 downstream update:** all four full evaluations/exports have driver closeout proofs. Full counts,
console-summary equality, unchanged source checkpoints, native export metadata and proof hashes independently
verified. Report `runs/setup/stage2-report-20261016-062420.json`, SHA-256
`cad9fafe3bec90d968e7d568497ec6081f2faa17512cded07f4f0aacbe744421`; all128 crop metric calculations rechecked.
Standard crop has0 gains beyond the registered representation margins and6/3 degradations (hammer/pick-place);
cropnosynth has24/22 degradations with3/2 isolated gains. These are single-seed descriptive margins, not statistical
significance/equivalence tests, and cropnosynth bundles multiple changes. See RESULTS tables; frozen method unchanged.
**08:27:43–08:27:44 post-controller-stop refresh:** all14 actual remote containers remain running, with advancing
complete records and finite training losses/gradients. The no-coverage pick-place log has an undefined
`metric/relepe_bin3_12` at190900 with bin count0; the source deliberately emits NaN for an empty diagnostic region,
not loss divergence. Earlier08:19 checks found the same empty-bin case in two other pick-place logs; no restart.
Proof `runs/setup/local-shutdown-remote-survival-20261016.json` records exact container IDs/start times/metrics.
Cached `unknown` flags may lag direct observations: reconcile after restart, never reset/relaunch from an outage.
Three crop hammer DrM pairs: `s2rl-crop-hammer-s{1000,1001,1002}[-eval]`, attempt1; accepted source/image unchanged.

| Seed | Trainer GPU | Evaluator GPU | Trainer / evaluator progress (08:27) |
|---|---|---|---|
| 1000 | remote0 /4391 | remote1 /6122 | 493000 /480000 |
| 1001 | remote2 /9e9f | remote3 /ca65 | 491000 /480000 |
| 1002 | remote1 /6122 | remote3 /ca65 | 487000 /480000 |

| Remote pretrain variant | Hammer step | Pick-place step | GPUs hammer / pick |
|---|---:|---:|---|
| no-lambda-dyn | 189700 | 196000 | 0 /1 |
| no-coverage | 187450 | 191500 | 2 /3 |
| misscaled-none | 189400 | 187250 | 0 /1 |
| misscaled-scale | 190000 | 181800 | 2 /3 |

All eight pretrains are still incomplete; full evaluations pending. Container start times remain10-15. A reconciliation
`launched` re-announcement or watcher restart label is not proof of a new container/training restart. Remote numerical
mirrors remain unverified under the failed07:30 batch below. Remote SSD3,359,475,376,128bytes free of7,528,923,684,864
(~44.6%), above the15% floor; no fresh remote RAM reading is claimed here.

Full authorized UUIDs are in section5. Last per-GPU measured load is2.5/2.8/2.5/2.6; no GPU had the0.5 headroom
needed for another trainer, despite fragmented aggregate headroom. Nine crop pairs remain pending. Initial trainer
rates suggested11:00–11:25 completion, training-only; evaluator closeout/sharing may delay this, and no final
success/return is available. Remote containers continue across local downtime. **Both local driver loops are now
stopped and scoped admissions held; pending jobs require a safely restarted dispatcher after reconnect.**

**Analysis already done; do not recompute to select methods:**
- Original Stage 2: `runs/setup/stage2-report-20261015-102432.json`. Pure no-InfoNCE degrades both tasks; no-motion,
  T=1 and single-group show no detected RL harm beyond the inherited margin. This is not proof of no effect.
  Representation degradation is much broader (32 prospective metrics). The Stage 2 RL threshold was inherited
  from development, not separately pre-registered; retain this caveat.
- Fairness + RL success/return standard: `rl-report-20261015-203425.json`, tool `rl_reporting.py`; all 30 original
  Stage 2 RL configs differ only by encoder path, same original local host/commit, 240 episodes at every point,
  intended T=1 encoder-frame difference only. Three full references were losslessly paused/resumed at 50k.
- Final comparison: `final-comparison-20261015-084143.json`; final training-camera success IQM ours .76, SinCro .44,
  ReViWo .31, CNN .72; held-out ours .13 vs <=.01, trajectories ours .72 vs CNN .76. Low held-out absolute scores.
  Keep genuine nonlearning seeds (ReViWo stick2001, SinCro stick2002); do not reseed after seeing results.
- Crop re-evaluation: `crop-reeval-report-20261015.json`, read-only source hashes verified. Trajectory probes improve
  in places, retrieval tradeoffs and pick-place CD-render degradation; it is a historical M2D-era single seed.
- Original no-synth representation: `stage2-report-20261016-070832.json`, SHA-256
  `3a438a17eeebedd525c9777cd316fe373589d578b08a128d7d8f148983033699`. Both full evaluations have driver proofs;
  all64 metric calculations independently rechecked. Beyond-margin better/worse/within counts1/27/4 hammer and
  1/24/7 pick-place, broad retrieval/trajectory-probe/appearance/CD-render damage. Single-seed attribution only;
  bundled synthetic-view/rendering removal, no crop, no registered RL arm or method-selection consequence.

## 4. Local shutdown and resume ledger

**Hard deadline: 2026-10-16 09:00; target final stop 08:45. Phase 3 begins 08:25. Never stop remote containers.**
No new long local work. `runs/setup/stage2-admission-hold.json` admits only crop full evaluations/exports and the two
already-registered no-synth full evaluations before07:15 (short-job whitelist extended06:41; comparable evaluations
~10minutes). All other local admissions withheld. Keep this file and `experiments/HOLD` until restart checks are complete.
Phase3 also created **`runs/setup/HOLD`**, which blocks all scoped local/remote admissions; explicitly clear it only
after reconciling survivors and checking resources/provenance/checkpoints on restart.
Controller identity inventory: `runs/setup/local-shutdown-controller-inventory-20261015.json` (PID/start_ticks/session/
boot ID and command hash; inventory only, **not an automatic shutdown watchdog**). All one-shot deadline prompts have
fired; hourly health cron `2ba15fda` was cancelled during Phase3 and CronList is empty. Do not re-arm the watcher or
controllers during this shutdown. This one-time shutdown is not a permanent cancellation of the research campaign.

The following eight jobs were running at20:54 on10-15; all logs fresh and finite, full cgroup118.8GB.
Exact PID/start-ticks/session, steps, GPU placement and recent45-minute ETAs: `runs/setup/servers-filled-20261015.json`.
Crop uninterrupted training ETAs05:54/06:50 (hammer/pick), cropnosynth07:30/08:24; four other runs09:22–10:55.
These are not completed closeout or permission to overrun the deadline. Reassess06:40; pause runs without time for
training AND closeout before~08:00. **These are not final shutdown states.** Final checkpoint step/hash, actual
finish/pause, process identity and resume instructions will replace this table.

| Job | Latest verified state / handling before 08:00 |
|---|---|
| `s2ab-crop-seed0-hammer` | **Completed200k**, full evaluation/export verified; DrM seeds1000–1002 running remotely |
| `s2ab-crop-seed0-pick-place` | **Completed200k**, full evaluation independently verified/export driver-verified; DrM pending |
| `s2ab-cropnosynth-seed0-hammer` | **Completed200k**, full evaluation/export driver-verified; DrM pending |
| `s2ab-cropnosynth-seed0-pick-place` | **Completed200k**, full evaluation/export driver-verified; DrM pending |
| `s2ab-nosynth-seed0-hammer` | **Completed200k**, config/checkpoint verified; SHA-256 `0079c165cbc97f0453b5a849f845977430d42880200524ff921823bf0a14cc11`. Pretrain/full-evaluation closeouts done; representation report07:08 complete |
| `s2ab-nosynth-seed0-pick-place` | **Completed200k**, config/checkpoint verified; SHA-256 `75642a46be60a6f993d62318d911f9714e1e7e7241630c8c9c119b23cd6c485f`. Pretrain and full-evaluation driver closeouts done; independent full evaluation verified |
| `s2ab-nocons-seed0-hammer` | **Completed200k normally**, config/checkpoint ZIP/hash/finite final loss verified07:43; driver closeout07:43:48. SHA-256 `23b15625c6d050f63417599ca2cb0d607fe740b50e67a69834f854dde4774ef5`; loss0.2051568180322647. Original session fully exited, zero failures. Full evaluation/config prepared but admission held past07:15 |
| `s2ab-nocons-seed0-pick-place` | **Intentionally paused180k**, proof/session exit/checkpoint payloads independently verified07:02; SHA-256 `b4e3a3328927afdf2945e432f0671ed8c1941259b689ebea6eb7bdc471e4e357`. No logged steps lost, zero charged failures. Reducer status `crashed` with matching `intentional_stops` proof, not an ordinary crash. Remains held; resume via scheduler `--resume auto` |

**10-16 06:24 completion update:** all four crop/cropnosynth pretrains finished200k normally; registered configs,
checkpoint ZIP integrity/hashes and finite final losses verified independently. All four pretrain driver closeout
proofs are present: `runs/setup/s2-s2ab-{crop,cropnosynth}-seed0-{hammer,pick-place}-completion.json`.
All four full evaluations and exports are independently verified with driver closeouts. Representation endpoint
report is complete (section3/RESULTS); three hammer crop RL pairs run remotely, nine pending. These actual completions
supersede all earlier crop training ETA/pause plans, without changing any deadline.

| Completed local job | Checkpoint step | SHA-256 of `checkpoints/step_0200000.pt` |
|---|---|---|
| `s2ab-crop-seed0-hammer` | 200000 | `58d9ff4808739902c0039eb668fbb518783e4470a84ae4a7da7132e7eab90b00` |
| `s2ab-cropnosynth-seed0-hammer` | 200000 | `33b7fda352240b52d1599ac0d7b2907927fdb105e7806f7b1f4bcd33789012ae` |
| `s2ab-crop-seed0-pick-place` | 200000 | `0bc4f8e5da640426a477891fbb892644fce04815961bc2c4d4691866543b2620` |
| `s2ab-cropnosynth-seed0-pick-place` | 200000 | `11426bf2152b52ebed438a05830b6ac32b8807bfa585dee5834bc5331d1934bc` |

**07:49 local scientific closeout check:** all seven completed local pretrains have driver proofs; no-consistency
pick-place is intentionally paused180k. Registry has no active local scientific jobs; process inventory found no
train/eval/export/RL/baseline workers, and original hammer session1358017 is empty. Full cgroup51.04GB at07:43.
Controllers/monitoring still run and final all-process shutdown remains pending, including the denied daemon blocker.
No-consistency hammer full-evaluation config is prepared, but both no-consistency evaluations remain held after07:15.

Memory alert task `b00cd94fv` replaced normally expired `bgv08dcyq` at22:58 on10-15, without an alert. Updated
controller inventory identifies outer PID2082985 and monitor PID2082986, deadline08:45. Reverify identities before
stopping; this monitoring replacement does not clear the separate scheduler-daemon stop denial below.

Pause evidence: `runs/<id>/checkpoint_pause-planned_local_shutdown_20261016.json` and `checkpoint_paused` registry
records. Checkpoints are in `runs/pretrain/<id>/checkpoints/step_<7-digit-step>.pt` and `latest.pt`. Verify SHA-256
against the ledger before resuming. Existing queue commands already contain `--resume auto`; **resume through the
scheduler/driver**, never launch `train.py` manually, never reset an active/unknown remote job.

Accounting fix `7a14773` is **integrated and pushed on main** (only `scripts/jobs.py` + `tests/test_jobs.py`).
Full native suite **763 passed, 0 failed/errors/skips**,293.05s; all214 source fingerprints verified against the tested
commit. Evidence `runs/nv/i/runs/u/tests.json`, preserved as `runs/setup/native-tests-intentional-stop-20261015.json`.
Intentional stops do not consume a failure/restart budget, including exit-before-hash races. Refused launches release
only their job-definition fingerprint after two confirmed absence checks; ordinary resets retain it. No baseline WIP
was merged. The registry remains append-only.

Main subsequently advanced to **c5a7571**, integrated/pushed after **764 passed,0 failed/errors/skips**,310.65s,
all214 source hashes verified against that commit. `runs/setup/native-tests-launch-diagnostics-20261015.json`
preserves the report. This additional diagnostic-only change saves bounded redacted launcher stdout/stderr as
`<attempt-spec>.launch-error.json` after a refused launch; outcome still requires reconciliation, never a blind retry.
Remote training release/image and frozen scientific configuration are unchanged.

**Permission blocker (20:33):** the permission layer denied stopping/restarting existing scheduler daemon PID2819571
(start_ticks415518078, owned repo daemon, started10-10). Do not retry through another mechanism or include that PID in
a shutdown backstop without renewed permission. Scoped drivers/new registry readers use the tested reducer; this
long-lived daemon still holds its old imported reducer. HOLD prevents its admissions. The user must authorize its
stop for the final all-local-processes shutdown; the prior denied action did not stop it.

## 5. Remote access, runtime, reconciliation and sync

- Host `compu@192.168.10.22`, authorized write root **`/home/compu/kaist/sunho` only**.
  Credentials inside this local container: `~/.config/s4d_remote/credentials` (SSH_HOST, SSH_USER, SSH_PASSWORD,
  REMOTE_ROOT). Never print/copy secrets to repo/logs/docs/remote/W&B/process arguments. `sshpass -e`/`-f`, not `-p`;
  use existing multiplexed helpers, no SSH key installation outside the root.
- VPN runs on the **local host**, user-managed, outside this container. Do not start/stop/reconfigure it or touch
  host networking. If unreachable >10 minutes, notify the user; remote jobs are **unknown**, not failed/relaunched.
- Helpers: `scripts/remote/check_tunnel.sh`; `scripts/remote/ssh.sh '<read-only command>'`; `sync_results.sh`.
  `RemoteClient` supplies checked SSH and `rsync --partial --append-verify` with before/after checksum proofs.
  Control sockets/known hosts: local `runs/remote/access`, ControlPersist 600s.
- Accepted source **226d8ae89b6f65097d372e597e0a7734d2eee3e0**, image
  **`sha256:c3ffea7f7c114dd12755f9d537f760a6d5b31c96c923233185b775e0324799b7`**
  (`s4d-runtime-03cca5218096dc88` is only a mutable label). Clean pushed release at
  `/home/compu/kaist/sunho/releases/<commit>/code`, mounted read-only `/workspace/splatter4d`.
  Run data `/home/compu/kaist/sunho/runs/<id>`, datasets `data/metaworld/splatter4d_v1`, runtime caches `runtime/`.
  Accepted full native suite 760/0/0/0 and four GPU isolation checks:
  `runs/remote/acceptance/226d8ae/manifest.json`. Historical cross-host artifacts are context, **not an active
  scientific placement criterion** under the 20:09 user decision.
- Remote code executes only in our owned `s4d-` containers, explicit immutable image, guarded UUID/CDI, no named
  volumes or mounts outside the allowed root. Never edit remote source or repair the remote host/NVML.
- Remote GPU UUIDs (all authorized, GPUs 4-7 foreign and untouchable):
  - GPU0 `GPU-4391fcee-537f-d408-d138-10d8a3866eea`
  - GPU1 `GPU-6122c7a9-eaa6-b539-6005-f20e339da4ea`
  - GPU2 `GPU-9e9f1e97-b2ca-04e0-eb2a-8035398daa79`
  - GPU3 `GPU-ca65e1b7-c0ac-6757-7d3f-816c3119a4c1`
- **Verified20:52 inventory:** no-lambda-dyn + misscaled-none hammer on GPU0, their pick-place runs GPU1; no-coverage +
  misscaled-scale hammer GPU2, their pick-place runs GPU3. All eight actual containers running with finite fresh metrics,
  ~60.4GiB summed container RAM, SSD3.397TB free (~45.1%). `runs/setup/servers-filled-20261015.json` records exact
  container/attempt/UUID/step and resource readings. Early training-only ETAs08:12–09:33 on10-16;25% planning overhead
  allowance gives11:03–12:43 (not a confidence bound, no first10k validation yet observed; later sharing may delay).
  **Refresh before shutdown.** Dispatch is local: containers continue during downtime; no pending job starts then.
  Two pretrains/GPU leave0.8 of2.8 load: at most one crop RL trainer/companion pair perGPU fits alongside, only after
  export proof. Do not promise all12 crop pairs launch before08:45; some may remain pending during downtime.
- Reconnect: snapshot/reconcile first (`remote_campaign_driver.py`, or `jobs.tick` under HOLD). Final exit0 is not
  done until final checksum-verified sync plus numerical artifact verification. Preserve replay; disk floor applies.
- Refused-launch special case: `runs/setup/tools/reconcile_unlaunched_remote.py <id>`, only after >=600s and two
  confirmed remote checks prove deterministic container absent in every state AND output absent. Never use outage
  as evidence. Tool supports evaluator output+trainer positive control and pretraining output+accepted-release positive
  control. Three proved-refused attempts were reconciled (no-lambda-dyn hammer, no-coverage pick, misscaled-scale pick),
  then actual attempt2 containers started; no lost training or charged failures. The latter two refusal causes were
  not established. A brief~2-minute SSH outage recovered; it did not authorize resets or relaunches.
- W&B **offline on remote**, never disabled; never send W&B credentials there. Main-host procedure:
  `.venv/bin/python -I runs/setup/tools/sync_wandb_remote.py [--run ID ...]` (or `--no-upload` for pull only).
  Checksummed W&B directories are pulled into `runs/<id>/wandb/`, status `runs/remote/wandb/status.json`; then
  `wandb sync` runs here. Last pre-shutdown baseline:243 uploaded; failures never block numerics.
- **07:30 late-sync phase:** W&B task `ba1a6vvlo` retries `s3-m3d-seed0-peg-unplug-side-200k` first, then
  checksum-pulls all14 active remote jobs; only finished runs upload. **Batch failed:2400s TimeoutExpired**, observed08:11.
  Atomic status was not updated (mtime10-15 08:41);243 persisted uploads remain the old baseline and the retry target's
  persisted `uploaded` map is empty. Target upload outcome is unconfirmed, not a proved upload failure/success.
  Report `runs/setup/late-remote-wandb-sync-20261016-0730.json`. Transient rsync descendants exited naturally on followup;
  original session2762546 empty, no cleanup signals or duplicate retry. Inspect sync markers before retry after restart.
  Numerical result-sync task `b82ry3ven` **completed07:48:39 with0/14 verified live snapshots**: nine checksum/change
  verification errors and five RemoteUnreachable errors. Report `runs/setup/late-remote-result-sync-20261016-0730.json`.
  All requests were `final=False`, checkpoints/replay excluded. Live files may change mid-sync; exact cause of every
  checksum failure is unestablished. Partial mirrors are unverified, not final/decision-ready evidence; previous
  completed accepted results remain valid. No restart/reset/threshold relaxation. Watcher recorded brief9.4s/3.6s
  reconnects, not a proven continuous10-minute outage. W&B batch/session has now exited; retry is deferred until restart.
- 07:30 pause checkpoint hash/session exit and matching intentional-stop accounting reverified, zero charged failures.
  Completed analyses are preserved: final/Stage2 success+return last5/final/AUC with CIs, historical crop report,
  frozen-method crop report and no-synth report. No-consistency/remote representation and crop RL remain incomplete;
  do not finalize them from partial training or retune the method.

## 6. Rules and latest decisions (precedence matters)

1. **2026-10-15 20:09 user decision supersedes every earlier host-specific experiment rule.** The remote is simply
   a second equivalent machine. Any job may run/resume on either host; compared arms may span hosts; existing
   references remain valid. No same-host requirement, cross-host equivalence check, checkpoint-resume comparison
   check, or experiment-type-to-host assignment remains. Record every run's host. Earlier contrary text is history.
   Job selection/placement/registration/launch is delegated without asking; record each new set's purpose/reporting
   in EXPERIMENT_LOG before results, and avoid low-value or duplicate work. This does **not** lift permission denials
   for executing the pinned new-baseline upstream code.
2. Local GPUs only GPU4 `GPU-76871c16-ff1d-f7b1-cdf5-fabbaf9df8ce`, GPU5
   `GPU-d09f0338-71b9-d915-3c7f-e99754a3b639`, selected by UUID through the guard. EGL gets exactly one UUID.
3. Local full cgroup charge <=150 decimal GB, steady target130GB, admission ceiling140GB, alert147GB;
   host-available reserve60GiB, disk floor300GB. Account page cache, PSS growth and nonresident dataset reserve.
   Measured reservations: pretrain10.77GB PSS, RL trainer3.5GB, RL evaluator4.5GB, full representation eval10GB.
   Admit job by job, no group requirement, smaller fitting work may pass larger unfitting candidates.
4. Remote campaign RAM <=1TB, SSD >=15% free, host reserve120GiB, GPU memory <=90GB/UUID, measured per-GPU load
   limit2.8; per-container CPU threads8. No foreign processes, host actions, VPN changes or system-wide prune.
5. Frozen method/hyperparameters do not change. Prospective comparison margins before any result; pretrain seeds
   paired for items1/2, later seed0 unless residual collapse; six RL seeds for method-selection proxies unchanged.
   Stage2 attribution RL seeds1000-1002, final seeds2000-2002. No post-result reseeding/protocol tweaks.
6. Baseline sizes: ours/SinCro/ReViWo unchanged; SNeRL/DynaMo official size; **only 3D-MVP resized**, <=SinCro
   9.06M parameters and1.79GFLOPs/frame. Width first, resolution/patch second, layers last; algorithm unchanged;
   no original-size side runs. 3D-MVP pretraining fuses **all six training cameras every sample**, one encoder/task;
   RL A: three training RGB-D+extrinsic cameras/episode, fused, replace/perturb one retain two; B: one camera/episode,
   sparser-cloud condition, separate rows. Both eight tasks x seeds2000-2002. Hammer sanity A pretrained-vs-random,
   plus one seed B. Review-only copied trunk needs direct pinned RVT correction, acceptance and permission first.
7. Report **success and return, last-5 / final / AUC**, separately for training(6)/held-out(4)/moving(2) cameras.
   Last5 = last five evals, final =1M, AUC = mean100 points10k..1M. Per-task seeds/mean/95%bootstrap; success aggregate
   IQM/probability improvement,10k stratified within-task resamples RNG0. No raw-return cross-task IQM.
8. Source edits in isolated worktrees, integrate after full native suite. Do not drain healthy GPUs for non-GPU
   releases; re-run GPU isolation only when selection/guard changes. Runtime identity/checksum/security checks remain.
9. Health entries only on change. Completed replays retained; no automatic runtime-computed deletion. Never touch
   `/home/ws/data/droid` (read-only), foreign work, or credentials. No foreground sleep-poll loops.
10. Local shutdown order remains in force until verified complete; remote containers must not be stopped.

## 7. Preserved work, open requests and next actions

Unaccepted work is pushed, not merged:

| WIP branch | Commit at initial preservation | Meaning |
|---|---|---|
| `wip/bx-mvp3d-input-conditions-20261016` | `ff30ad7` | fused-six-camera pretraining and A/B RL adapters/config/tests, review-only |
| `wip/bx-snerl-20261016` | `5ad913f` | SNeRL port/render script and historical sanity evidence, review-only |
| `wip/review-items-20261016` | `bd719fd` | third-party/release infrastructure + baseline docs, full suite needed |
| `wip/bx-dynamo-20261016` | `119ffb6` | clean already-committed DynaMo/review bundle preserved |
| `wip/intentional-stop-reducer-20261016` | `c5a7571` | scheduler accounting/race/refused-launch + redacted diagnostics, tested/integrated/pushed |

At08:00 all five worktrees were rechecked: no tracked changes or untracked non-run source work. All five WIP refs
above match their published origin tips exactly; no empty preservation commits or unaccepted merges are needed.
Worktrees `.worktrees/{bx-mvp3d,bx-snerl,bx-dynamo,review,intentional-stop}` remain on disk. WIP smoke figures/logs
are historical evidence, not accepted results or proof that the current pinned upstream adapter has passed tests.
Do not execute permission-blocked upstream code to fill idle GPUs. The old `.claude/worktrees/agent-*` branch is an
ancestor of main, not another outstanding change. Do not blindly add `runs/` (checkpoints/replays/access state).

Priority after safe restart:
1. Reconcile/sync surviving remote runs before any launch. Check final shutdown ledger and intentional accounting.
2. Finish registered Stage2 representation attribution (12 pretrain-only variants/evaluations) and crop follow-up
   (four frozen-method pretrains/full evaluations/exports,12 DrM pairs), on either machine as headroom permits.
3. Re-run reporting only as newly complete groups become available; add crop separated camera success/return beside
   representation metrics. Keep the frozen method; results are attribution, not permission to retune it.
4. Finish integration of host-neutral execution support where historical hard-coded checks remain; no scientific
   cross-host numerical/migration comparison is needed. Preserve checkpoint integrity and host provenance.
5. New baselines: obtain explicit permission for pinned upstream execution if still blocked, audit direct RVT trunk,
   run full tests and hammer pretrained-vs-random sanity; report once after sanity, except blockers/deviations.
6. Complete final plots/report with fair limitations, per-task return scales and retained genuine failing seeds.

## 8. Restart checklist and commands

Run from `/home/ws/ws/splatter4d`. Inspect before changing/stopping anything. Shutdown is one-time, not a permanent
campaign cancellation. Do not relaunch the old deadline watchdog after it completes.

```bash
# No secrets; read state before admitting anything.
git status --short
git log -5 --oneline
.venv/bin/python -I scripts/jobs.py status
df -h /home/ws
cat /sys/fs/cgroup/memory.current
scripts/remote/check_tunnel.sh
```

If `experiments/HOLD` is absent on a fresh clone, create it **before** starting any daemon or driver. Also create or
retain `runs/setup/HOLD` until scoped queues are safe to admit; a restored scoped HOLD is intentional, not a scheduler
fault. Check authorized GPU UUID visibility/NVML and any foreign usage, host RAM/disk floors, dataset hashes/paths and
`.venv`. Check `docs/tests.json` fingerprints against actual source; new source needs a full native suite, not a
mock or a threshold relaxation. Full suite (use short output paths inside its checkout):

```bash
CUDA_VISIBLE_DEVICES=GPU-76871c16-ff1d-f7b1-cdf5-fabbaf9df8ce MUJOCO_GL=egl \
 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 \
 .venv/bin/python -I scripts/run_tests.py --output-dir runs/restart-tests
```

Do not run the suite with a long nested TMPDIR (see below). Do not unnecessarily rerun isolation for a non-GPU
change. Inspect `experiments/daemon.pid` and process identity before starting one daemon. Keep global HOLD while
reconciling. One pass of each driver (first dry-run); **the remote pass may admit remote jobs** if its admission hold
is absent. Stop obsolete deadline loops and check paused checkpoint hashes first.

```bash
.venv/bin/python -I runs/setup/tools/stage2_local_driver.py --dry-run
.venv/bin/python -I runs/setup/tools/remote_campaign_driver.py --dry-run
# After resource/provenance checks and after updating the expired local admission hold:
.venv/bin/python -I runs/setup/tools/stage2_local_driver.py
.venv/bin/python -I runs/setup/tools/remote_campaign_driver.py
```

For continuous operation, launch driver loops and daemon as **tracked background Bash tasks**, one each; each driver
call is serialized internally, use timeout3000(local)/7000(remote), append logs named above, sleep300 **inside the
background task only**. Default daemon command `.venv/bin/python -I scripts/jobs.py daemon --interval 30`, log
`experiments/daemon.log`; retain HOLD unless explicitly replacing scoped admission. Watcher command as one tracked
background task: `python3 -I /home/ws/ws/splatter4d/scripts/watch_events.py --once`. Cache loop: run the eviction tool
every900s as one tracked background task. Memory alert reads cgroup every30s and alerts147GB. Schedule health checks
hourly, but no-change checks do not create repetitive docs/artifacts. Re-arm watcher only on completion.

For paused local jobs, existing driver normalizes restartable nonzero exits to pending. Correct intentional-stop
accounting must be active before resuming; never erase failure history with blanket resets. If migrating a checkpoint,
checksum-transfer the checkpoint/config and retain original provenance plus migration event; do not overwrite a remote
run's identity with a local identity. No same-next-step cross-host diagnostic is required by the user.

## 9. Known issues and pitfalls

- **NVIDIA after host `systemd daemon-reload`:** device access was removed for new processes. The user applied
  host-side `systemctl set-property --runtime DeviceAllow=...`; rules survive reloads until container restart. If it
  recurs after the hardware/container change, report to the user, do not repair the host. Verify NVML/CUDA/EGL before
  work. CUDA/NVML initialization outages are infrastructure failures with evidence, not charged scientific retries.
- **AF_UNIX path length:** DataLoader `resource_sharer` sockets must fit107 usable bytes. Long candidate TMPDIR once
  stalled a suite14h. Use short test worktree/evidence paths (`runs/nv/i/runs/u`, `runs/rel/<shortsha>`).
- **CPU oversubscription:** remote ReViWo once used~250cores from two jobs. Keep8thread/container OMP/MKL/OpenBLAS;
  do not change scientific settings to compensate for host contention.
- **Page cache counts:** resident datasets, live rsync partials, finished replay/snapshots all charge the local cgroup.
  Reserve nonresident dataset growth; clean-page fadvise is safe, memory.reclaim is unavailable. Do not exclude cache
  from the150GB cap or reintroduce automatic deletion.
- **QueryUnavailable:** shared bounded nvidia-smi lock can be busy; skip admission and retry next pass, not a failed
  trainer. Snapshot failures/outages never prove a remote job is gone.
- **Live-sync checksum churn:** files append during rsync; mirror may lag. Final sync verifies unchanged before/after
  manifests. Do not treat a changing live file as a final numerical failure.
- **Refused launch:** correct launcher validation may reject before Docker; unknown state requires the specific
  reconciliation evidence, not `jobs reset` on an active/unknown job. `config.yaml` is not a numerical completion
  artifact and must not appear in `required_results`; it is still transferred automatically.
- **Source/test identity:** docs/queue dirty does not imply scientific source drift, but changing source/config files
  invalidates acceptance fingerprints. Review branches are not accepted main or accepted remote runtime.
- **Git push credentials:** detached VS Code helper fails; use per-command override, never global config edits:
  `git -c credential.helper= -c credential.helper='!gh auth git-credential' push origin <branch>`.
- Do not stop foreign processes or remote containers at local shutdown. Use verified PID/start-ticks/session identity;
  never a broad `pkill python`, root-owned-process sweep or GPU-index kill.

## Final shutdown verification

**PARTIAL / BLOCKED — not an all-local-process shutdown.** Phase3 permitted controller stops were complete by
**2026-10-16T08:26:16.090524+09:00**; final local process/registry inventory was taken at
**2026-10-16T08:28:31.774707+09:00**. Identity checks before stopping matched boot ID, start ticks and command hashes.

| Local workload / monitor | Actual handling |
|---|---|
| Stage2 driver `b4dyxt2gb`, PID3600420 | Intentionally stopped; PID/session absent |
| Remote dispatcher `b0zf288b4`, PID1860610 | Local loop intentionally stopped; remote containers untouched |
| Cache loop `b8mmcvqt8`, PID326907 | Intentionally stopped; PID/session absent |
| Memory alert `b00cd94fv`, PID2082985/child2082986 | Intentionally stopped; both PIDs absent |
| Watcher `bqo04rfe4`, PID2807957 | Intentionally stopped; PID absent; do not re-arm |
| Hourly health cron `2ba15fda` | Cancelled; CronList empty, no remaining one-shots |
| Scheduler daemon PID2819571, start_ticks415518078 | **Still alive: prior stop permission denial; no retry/signal** |

No local scientific job remains active. Seven local pretrains completed200k; no-consistency pick-place remains a
lossless intentional180k pause with the checkpoint/hash ledger in section4, matching registry proof and zero charged
failures. At08:27:49.695172 both authorized local UUIDs were queried successfully (`nvidia-smi` return0, empty compute
process list): **GPUs4/5 had no compute processes, including ours**. Local cgroup52.71GB at08:28:31, below150GB.

`experiments/HOLD` is retained; Phase3 created `runs/setup/HOLD` to prevent scoped local/remote admissions, and the
expired short-job hold remains. All known tracked campaign background tasks are stopped; current publication tools
are transient and excluded from the workload inventory. No other reachable agent/session was listed. Unknown or
interactive shell/CLI candidates were not assumed owned and were not signalled.

**Remaining blocker:** the unchanged denied daemon and its local rsync/sshpass/SSH activity remain. The shared local
SSH multiplex master PID2573436 was also left untouched rather than disrupt the daemon indirectly. No foreign process
or remote container was stopped. Explicit renewed permission for the denied daemon is required to finish all-local
shutdown; this scheduled Phase3 prompt does not clear the earlier denial. Do not claim the08:45 all-process target met.

Controller intentional-stop evidence is durable in `runs/setup/local-shutdown-verification-20261016.json`, including
`pending_registry_event`. The existing daemon holds `experiments/registry.lock`; a nonblocking acquisition and one
bounded25s lock wait could not acquire it. **The controller event has not been appended to the registry**; do not bypass
the lock. After the daemon is safely stopped/the lock is released, append that exact operational event under the lock
once. It uses `__host_local__`, not a scientific job, and must not charge failures. The scientific180k pause event is
already recorded correctly; its accounting is not pending.

Surviving remote inventory at08:27:44: **all14 containers running and advancing**, exact IDs/start times/steps above;
eight pretrains and three crop DrM trainer/companion pairs. Finite training losses/gradients; one empty-bin diagnostic
NaN has zero supporting count and is not a training failure. Remote SSD3.359TB free (~44.6%); no new RAM measurement
or final numerical sync is claimed. Both local driver loops are stopped, so pending jobs wait for a safely restarted
dispatcher. Never stop the surviving containers as part of local cleanup.

Evidence published with this ledger:
- `runs/setup/local-shutdown-verification-20261016.json`: intentional control stops, remaining process identities,
  holds, scientific pause accounting and deferred operational registry event.
- `runs/setup/local-shutdown-gpu-verification-20261016.json`: successful UUID-only query, no compute processes.
- `runs/setup/local-shutdown-remote-survival-20261016.json`: read-only post-controller-stop container/metric snapshot.

Actual permitted-controller stop verified **08:26:16 KST**; final local inventory **08:28:31 KST**. **Full local shutdown
remains blocked by PID2819571 and transport activity; it has not been verified complete.**
