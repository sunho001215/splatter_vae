# Experiment log (append-only)

Each entry: hypothesis, change, evidence runs, result, decision. Never edit past entries; add corrections as new entries.

## 2026-10-07 — E0: campaign setup and runtime diagnosis

- **Hypothesis.** The earlier native-renderer failure came from gsplat binaries compiled against a different torch
  than the one at runtime; compiling gsplat and fused-ssim from source against the locked torch 2.10.0+cu129 with
  `TORCH_CUDA_ARCH_LIST=12.0` and CUDA 12.9 (matching torch's CUDA) will import and render on sm_120.
- **Change.** `pyproject.toml` now pins the build-time torch to the runtime torch (`match-runtime = true`) and passes
  the CUDA arch list and toolkit path as build variables. `reference_runtime.pth` was removed from the venv.
- **Evidence.** None yet. The `uv sync` that performs the build was denied by the Claude Code auto-mode classifier.
- **Result.** Blocked pending user permission.
- **Decision.** Prepare the RL port, scheduler and protocol, which do not need the runtime, then stop and ask.

## 2026-10-07 — E1: frame spacing aligned between pretraining and RL (user decision)

- **Hypothesis.** An encoder pretrained only on frames 3, 6 or 9 simulator steps apart is evaluated out of
  distribution when the RL observation (action repeat 2, frame stack 3) spaces frames 2 simulator steps apart.
- **Change.** Pretraining strides are now {2, 4, 6} simulator steps, exactly uniform per sample (start frames are
  kept only where every stride fits). Validation, probes, retrieval and panels run separately at stride 2 (the RL
  spacing) and stride 6 (the largest pretraining stride); every metric carries the suffix `@s2` or `@s6`.
  Workspace statistics use stride 6. RL keeps the reference observation for every method (3 consecutive agent-step
  frames); `train_rl.py` refuses a splatter4d export whose recorded strides exclude 2. The earlier plan to give our
  encoder a 6-step RL frame gap is withdrawn.
- **Evidence.** None yet (no data or training has run).
- **Decision.** Applied before any pretraining, so no result depends on the old strides.

## 2026-10-07 — E2: exploration schedule per task

- **Finding.** The reference configs use `linear(1.0,0.1,150000)` for every task and encoder, although their tasks span
  easy (door-open, peg-unplug-side), medium (hammer) and very hard (stick-push) in Seo et al. (MWM, Appendix F).
- **Decision.** The category-to-schedule mapping consistent with the reference assigns the same schedule to all
  categories, so the four new tasks also use it (`configs/rl/tasks.yaml`). No difficulty-dependent schedule is
  introduced, because the reference provides no evidence for one.

## 2026-10-07 — E3: difficulty-dependent exploration schedules (user decision; supersedes E2)

- **Change.** The stddev schedule now depends on the MWM Appendix F difficulty category: easy
  `linear(1.0,0.1,100000)`, medium `linear(1.0,0.1,250000)`, hard and very hard `linear(1.0,0.1,500000)`.
  Unit: agent steps, verified in the reference loop (`train_drqv2_metaworld.py` lines 504-524 pass the loop counter,
  one `env.step` = action repeat 2 simulator steps, to `schedule(stddev_schedule, step)`).
  Per task: door-open, peg-unplug-side easy; hammer, peg-insert-side, bin-picking medium; pick-place hard;
  stick-push, shelf-place very hard. `configs/rl/tasks.yaml`, resolved by `s4d/rl/protocol.py` for every method.
- **Deviation.** The reference used `linear(1.0,0.1,150000)` for every task; the four reference tasks now differ from
  it (door-open and peg-unplug-side decay faster, hammer and stick-push slower).
- **Evidence.** `tests/test_rl.py::test_every_task_resolves_to_its_difficulty_schedule_for_every_method`.

## 2026-10-07 — E4: DrM replaces DrQ-v2 for every RL run (user decision; supersedes E2, E3)

- **Change.** DrM ported from the official code (github.com/XuGW-Kevin/DrM @ `989732d6`, `agents/drm_mw.py`,
  `utils.py`, Meta-World configs as applied by `train_mw.py`): dormant ratio, shrink-and-perturb every 100k agent
  steps, awake exploration (`linear(1.0,0.1,500000)` after awakening), expectile-0.9 value network and lambda-0.5
  blended target, n-step 10 and discount 0.97 (the official loader's effective values), continuation 1.0 at the time
  limit, 2,000 uniform warm-up steps. The difficulty-dependent DrQ-v2 schedules of E3 are removed; no campaign task
  has an official per-task override. Frozen encoders: only actor, critic, critic target and value are perturbed.
  Environment, observation (with proprio) and evaluation stay the shared reference protocol.
- **Paper vs code.** The official code differs from the paper's Table 1 (n-step 3, discount 0.99,
  `linear(1.0,0.1,300000)`, max perturb 0.9, dormant-dependent lambda). The code is followed; both are recorded in
  `docs/RL_PROTOCOL.md`.
- **Evidence.** Tests in `tests/test_rl.py` compare dormant ratio, perturbation, target and expectile loss with
  verbatim copies of the official functions (`tests/_drm_official.py`); not yet run (environment still building).
- **Paper comparison available for Stage 0.** The paper's Meta-World figure covers assembly, stick-pull,
  pick-place-wall, disassemble (dense) and coffee-push, soccer, sweep-into, hammer ("sparse"); hammer is the only
  overlap. The released code has no sparse-reward wrapper (the vendored hammer env returns the dense v2 reward).
  Expected differences from the paper: 6 random training cameras instead of one fixed camera, 128x128 instead of
  84x84, 125- instead of 250-agent-step episodes, Meta-World v3 instead of v2.

## 2026-10-07 — E0 result: Phase A gates pass

- **Build.** `uv sync --extra dev` (authorized) built gsplat (d28ee0c) and fused-ssim (a7c48d6) from source for sm_120
  only, in a uv build env pinned to the runtime torch 2.10.0+cu129 (CUDA 12.9 = nvcc 12.9); 97 min, of which one
  translation unit took ~60 min. `reference_runtime.pth` is gone; the venv is standalone. Versions:
  `docs/runtime_versions.json`.
- **Native rasterization.** `scripts/check_native_render.py` on GPU 4 and GPU 5: finite, alpha max 0.9999, gradients
  for all five Gaussian attributes (`docs/native_render/*.json`). A first attempt reported a zero rotation gradient
  because the check used isotropic Gaussians; the check was fixed (anisotropic, random rotations), not the renderer.
- **Bugs found by the first native runs.** (1) `set_anchor_statistics` built mean/std on the CPU for a CUDA decoder;
  fixed, regression test added. (2) The motion splat has 7 channels, which the gsplat build does not compile
  (`GSPLAT_NUM_CHANNELS` has 1-6, 8, 9, ...); features are now zero-padded to the next compiled count and sliced back
  (exact; native test compares with per-part renders).
- **Tests.** Full suite on GPU 4: 187 passed, 0 failed, 0 errors, 0 skipped (`docs/tests.json`).
- **Isolation.** `scripts/check_gpu_isolation.py`: the GPU 4 child appears only on GPU 4 (EGL device 16), the GPU 5
  child only on GPU 5 (EGL device 17); `CUDA_VISIBLE_DEVICES=0` and unset are rejected (`docs/gpu_isolation.json`).
- **50-step rendered training** on `hammer_pilot.hdf5`, GPU 4 then GPU 5 (`outputs/smoke50-gpu{4,5}`): all losses
  finite, total loss 4.116 -> 4.083, encoder/decoder gradient norms 0.130/0.992, 0.12 s/step, 3.2 GB; the two GPUs
  give bit-identical metrics from the same seed. Validation reported at strides 2 and 6.

## 2026-10-07 — E5: Meta-World 3.0 reset semantics (finding; affects evaluation determinism)

- **Finding.** `SawyerXYZEnv.reset` ignores its `seed` argument, and the MT1 `RandomTaskSelectWrapper` calls
  `set_task` on every reset, which re-freezes the random vector (`_freeze_rand_vec = True`), so setting
  `_freeze_rand_vec = False` (reference code, ours) has no effect after the first reset. Each episode therefore uses
  one of the 50 MT1 object/goal configurations fixed by the construction seed, drawn from the env RNG in reset order,
  and the hand starts from wherever the previous episode left it. The reference collector and RL envs behave the
  same way, so the collected data (seed 0, 250 episodes over 50 configurations) and the RL training envs match the
  reference protocol.
- **Change (evaluation only).** For explicit evaluation episode seeds, `MetaWorldCameraEnv.reset` reseeds the env RNG
  (which selects the configuration) and resets the MuJoCo data to the model defaults first. Start states are now a
  function of (construction seed, episode seed) only: verified bit-identical between a fresh and a used env on
  hammer, pick-place, shelf-place and bin-picking, and between the worker pool and a single env (test). Evaluation
  envs use the reference evaluation construction seed, run seed + 1, whose 50 configurations differ from training's.

## 2026-10-07 — Phase B result: full Meta-World data

- **Collection.** Eight scheduler jobs (one UUID each, EGL device 16 on GPU 4, 17 on GPU 5), 250 episodes per task
  with the reference mixture, 26-60 min per task, 162 GB in total (`/home/ws/data/metaworld/splatter4d_v1/`).
- **Splits.** Deterministic seed-0 manifests, 240 training / 10 validation episodes per task (`splits/`).
- **Workspace statistics.** Recomputed for all eight tasks at stride 6 (largest training stride) (`workspace_stats/`).
- **Geometry (D2/D3), unchanged thresholds 3 mm / 5 mm.** All eight pass; D2 median 0.86-1.00 mm, D3 median
  1.91-2.04 mm, world-body motion exactly zero (`docs/data_checks/<task>/`). Sanity panels at strides 2: RGB, depth,
  body ids and motion score per training camera.

## 2026-10-07 — G1: M2 overfit gate, iteration 1 (fail) and diagnosis

- **Run.** `runs/pretrain/gate-m2-hammer` (hammer ep001, one episode, 3k steps, batch 16, warm-up shortened to 300
  for the gate only, original 20k temporal ramp, cosine LR to 1e-5 at 3k).
- **Result (thresholds unchanged).** Train-view PSNR 23.9 dB (s2) / 24.1 dB (s6), target >= 32: FAIL. Moving relative
  EPE 0->2 0.79 (s2) / 0.51 (s6), target <= 0.15: FAIL. Depth AbsRel 0.029; dynamic alpha share on moving pixels 0.72.
- **Diagnosis.** (1) The temporal ramp multiplies both the t1/t2 render terms and the motion loss
  (`loss.py`: `ramp * motion * m_loss`); at 3k steps the ramp is 0.15, so motion supervision was mostly off. Panels
  show predicted motion on the arm only, the hammer barely rendered. (2) PSNR was still rising ~0.45 dB per 300 steps
  when the cosine schedule reached its floor. (3) Capacity is not the limit: free Gaussians fitted directly to the
  same frame reach 39.6 dB with the method's 8,192 Gaussians, 44.5 dB with 32,768 (`runs/diag/capacity-*`;
  `scripts/diag_capacity.py`). Held-out cameras reach only ~15 dB even for these direct fits of one frame from six
  views, so held-out PSNR is not informative in a one-episode gate.
- **Decision.** Iteration 2 (gate only): ramp shortened to 300 steps (allowed by the user for the gate) and constant
  LR after warm-up (`train.min_lr = lr`). Loader workers raised to 32 (resource only: the window loader is CPU-bound,
  ~76 ms per window).

## 2026-10-07 — G2: M2 gate, iteration 2 (fail)

- **Run.** `runs/pretrain/gate-m2-hammer-it2` (ramp 300 steps, constant LR 5e-4 after warm-up; otherwise as G1).
- **Result.** PSNR 24.5 (s2) / 24.4 (s6); moving relative EPE 0->2 0.69 (s2) / 0.54 (s6); dynamic alpha share 0.76.
  FAIL on both thresholds; small gains over G1, so the ramp and the LR floor were not the main limitation.
- **Checked.** Negative displacements survive the feature splat (native tests with signed features), so motion is not
  clipped by the renderer; zero motion would give relative EPE 1.0, so motion is learned, slowly.
- **Decision.** Iteration 3 (last for M2) with two gate-only arms: (a) LR x4 (2e-3) for 3k steps — is it
  optimisation speed?; (b) LR 5e-4 for 12k steps — are the thresholds reachable at all, and when? Full 200k
  pretraining of the unchanged base config started in parallel on both development tasks
  (`runs/pretrain/s1-pretrain-{hammer,pick-place}-base`).

## 2026-10-07 — G3: M2 gate, iteration 3 (fail); M2 closed

- **Runs.** (a) LR 2e-3 constant, 3k steps (`runs/pretrain/gate-m2-hammer-it3a`): PSNR 25.5 / 25.1, rel. EPE 0.79 /
  0.60. (b) iteration-2 settings for 12k steps (`runs/pretrain/gate-m2-hammer-it3b`): at 3k 24.0 / 24.1 and 0.66 /
  0.45; at 12k 26.1 / 25.9 and 0.37 / 0.25; dynamic alpha share 0.88.
- **Conclusion.** Learning rate gives ~1 dB at 3k but no motion gain; four times the gate length still misses both
  thresholds, with PSNR rising ~0.3 dB per 1k steps and motion error flattening. Capacity is sufficient (G1), so
  the decoder fits slowly with the specified architecture and losses. M2 is recorded as FAIL in `docs/RESULTS.md`
  after three diagnosed iterations; iteration stops on M2 and the plan continues.
- **Pointers for the Stage 1 improvement loop** (development tasks only): motion supervision is weak relative to
  rendering (motion loss ~1e-3 vs render ~0.2; moving pixels are a few percent of valid pixels and the motion loss is
  averaged over all valid pixels); the scene renders blurry at 3k steps, so decoder learning speed (width/depth,
  per-group learning rates) and the motion weight are the first levers to test against full-training metrics.

## 2026-10-07 — Screens for the Stage 1 improvement loop (diagnostics, not counted as iterations)

- **Purpose.** Cheap single-change tests on the one-episode gate setup (iteration-2 settings, 6k steps), compared with
  `gate-m2-hammer-it3b` at 6k (PSNR 25.15 / 24.86, moving PSNR 23.70 / 22.07, rel. EPE 0.48 / 0.40). Only changes that
  help here are promoted to a counted improvement iteration (100k pretraining on both development tasks + DrM proxy).
- **S1** `loss.motion` 5 -> 20 (motion supervision weak in the gate). **S2** `loss.lambda_dyn` 1 -> 4 (moving object
  under-rendered). **S3** `model.decoder.dim` 128 -> 256 (decoder learning speed). Runs:
  `runs/pretrain/screen-{s1-motion20,s2-lambdadyn4,s3-decdim256}`.

## 2026-10-07 — Fix: forked DataLoader workers aborting during evaluation

- **Incident.** `screen-s1-motion20` failed in its step-6000 evaluation: three evaluation-loader workers aborted with
  `CUDA error: initialization error` thrown from `c10::TensorImpl::~TensorImpl -> c10::cuda::ExchangeDevice`, i.e. a
  CUDA tensor was destroyed inside a forked worker (CUDA cannot be used in a child forked after CUDA init). The
  scheduler restarted the job once; it resumed from the step-6000 checkpoint and completed.
- **Cause (from the stack trace).** Evaluation loaders fork new workers at every evaluation, after the model is on the
  GPU. A forked worker inherits the parent's uncollected cyclic garbage; when the worker's garbage collector runs, it
  can free CUDA tensors it inherited. A minimal reproduction (cyclic garbage holding CUDA and pinned tensors, workers
  calling `gc.collect()`) did not crash, so the exact object could not be pinned down.
- **Fix.** `s4d/train/workers.fork_safe_iter`: the parent runs `gc.collect()` and `gc.freeze()` while the workers are
  forked and `gc.unfreeze()` afterwards, so workers never collect inherited objects. Used for the training loader,
  validation loaders and probe loaders. Test: `tests/test_workers.py` (mechanics). Suite 200/200.
- **Exposure.** The two running base pretraining runs started with the old code; if they hit this, the scheduler
  resumes them from their last checkpoint (every 10k steps) with the fixed code.

## 2026-10-07 — Screen results and Stage 1 improvement iterations 1-2

- **Screens at 6k** (vs reference PSNR 25.15 / 24.86, moving PSNR 23.70 / 22.07, rel. EPE 0.48 / 0.40, dyn 0.84):
  S2 `lambda_dyn` 4: 25.97 / 25.78, 24.40 / 23.20, 0.41 / 0.32, dyn 0.91 / 0.93 — better on every metric.
  S3 decoder dim 256: 26.94 / 26.56, 23.99 / 22.60, 0.42 / 0.35, dyn 0.86 — largest PSNR gain, motion better.
  S1 motion weight 20 (5k; its 6k evaluation crashed, see fork fix): 25.14 / 25.43, rel. EPE 0.61 / 0.32 — mixed.
- **Hypotheses promoted (counted iterations).** Iteration 1 (H1): decoder dim 128 -> 256 speeds decoder learning and
  raises PSNR without hurting motion. Iteration 2 (H2): `lambda_dyn` 1 -> 4 concentrates the render loss on moving
  pixels and improves moving PSNR, motion and the dynamic share. Both run on hammer and pick-place with the base
  200k schedule (`runs/pretrain/s1-it1-decdim256-*`, `runs/pretrain/s1-it2-lambdadyn4-*`) and are compared with the
  base runs at the same step (100k), so the LR schedule is identical; then each 100k encoder gets the DrM proxy
  (2 seeds, 200k agent steps, both tasks). Iteration 3 combines them if both help.

## 2026-10-08 — Fix: pretraining loaders starved Stage 0 of page cache

- **Symptom.** Stage 0 DrM + CNN throughput fell from 29 to 5 agent steps/s between 23:00 and 23:50 (update time
  19 -> 107 s per 1k steps) while GPU 5 utilisation dropped to 37%; `vmstat` showed 1.7 GB/s of block reads.
- **Cause.** Six concurrent pretraining runs used 32 (base) or 16 (iterations) loader workers with prefetch factor 4:
  ~2.25 GB of resident memory per worker plus up to 128 prefetched 127 MB batches per run in shared memory
  (~118 GB system-wide shared memory from our loaders). The page cache left for files was smaller than the working
  set (the CNN replay memmaps plus the HDF5 data), so replay pages were evicted and re-read at every sample.
- **Fix (no code change).** Loader workers lowered to 12 (base runs) and 8 (iteration runs); the measured window cost
  (~1.2 core-seconds per batch) needs about 7 workers at 0.18 s/step. Each run is stopped right after its next
  checkpoint and evaluation (`.cache/tmp/restart_after_ckpt.sh`, log `runs/setup/memory_fix_restarts.log`) and the
  scheduler resumes it from that checkpoint; no training steps are lost and the data order is unchanged (the sampler
  runs in the main process). `max_restarts` raised to 2 for these jobs so the usual one restart after a genuine crash
  remains. After the first two restarts available RAM rose from 76 to 123 GB.
- **Rule for later launches.** At most ~12 loader workers per pretraining run when RL runs share the host.

## 2026-10-08 — External memory pressure: improvement iterations paused

- **Observation (02:08).** Available RAM fell to 25 GB (free 3 GB) and Stage 0 hammer dropped to 1.8 steps/s. Our own
  processes hold ~68 GB anonymous memory and <= 52 GB shared memory; host-wide anonymous memory is 362 GB, so ~290 GB
  belongs to other tenants of the shared host (not visible from this container; untouched). Load average 233.
- **Decision.** Pause the four iteration runs (lowest priority of the running work) right after their 30k checkpoints
  so no steps are lost: the queue holds them behind a placeholder dependency `hold-memory`; `max_restarts` raised to 3
  to keep one restart for a genuine crash after resuming. Base pretraining (needed for Stage 1) and Stage 0 (tier 1)
  keep running. Resume the iterations when available RAM stays above ~120 GB.

## 2026-10-08 — OOM kills of both base pretraining runs; scheduler now cleans up killed sessions

- **Incident (02:27).** Host memory ran out (available 9 GB; other tenants ~290 GB anonymous memory). The kernel
  OOM killer SIGKILLed both base pretraining mains (exit 137): hammer at 79.65k (last checkpoint 70k), pick-place at
  ~68k (last checkpoint 60k). Their DataLoader workers survived as orphans (17 processes, ~17 GB RAM) and kept the
  dead parents' GPU contexts (4.4 GB per GPU) alive; terminated with SIGTERM (they were ours).
- **Fix.** `scripts/jobs.py` now terminates the remaining process group (= session) of any job whose exit it records,
  so no child of a killed job survives it. Test: `test_exit_terminates_leftover_processes_of_the_job_session`.
  Suite 201/201. Daemon restarted on the new code.
- **Recovery.** Scheduler held; the four iteration runs pause at their 30k checkpoints (no steps lost); then the base
  runs resume from 70k / 60k with 8 loader workers each (`max_restarts` 3: this OOM kill, one deliberate restart, one
  for a genuine crash). Steps lost to the OOM kill: 9.65k (hammer), ~8k (pick-place).

## 2026-10-08 — Fix: resuming loaded the whole trained prefix of the epoch

- **Symptom.** After the 02:53 relaunches, hammer base logged its first step 14 min after resuming at 70k and
  pick-place base none in 20 min (loader workers at 100% CPU).
- **Cause.** `infinite()` resumed by iterating the DataLoader and discarding the already-trained batches of the
  current epoch (up to ~3,700 batches = ~60k windows of motion computation).
- **Fix.** `ResumableDistributedSampler` drops those indices before batching, so nothing is loaded for them and the
  batch stream is identical (`tests/test_checkpoint.py::test_resumable_sampler_matches_replay_without_loading_the_skipped_prefix`);
  suite 202/202. Pick-place base restarted onto it (first step 1 min after relaunch). Host memory recovered to ~300 GB
  available at 03:16, so the four iteration runs were released from `hold-memory` and resumed from 30k.

## 2026-10-08 — Stage 0 result: DrM + CNN sanity (seed 2000, 1M agent steps, full evaluation protocol)

Runs: `runs/stage0-drm-cnn-hammer-s2000`, `runs/stage0-drm-cnn-shelf-place-s2000` (W&B `splatter4d-rl`).

| Task | Train cameras final / last-5 mean / AUC | Held-out cameras last-5 | Trajectories last-5 | Dormant ratio (10k -> 1M) |
|---|---|---|---|---|
| hammer | 0.70 / 0.65 / 0.34 | 0.02 | 0.60 | 0.13 -> 0.01 (awake at 10k) |
| shelf-place | 0.00 / 0.00 / 0.00 | 0.00 | 0.00 | 0.68 -> 0.22 (awake at 103k) |

- **hammer learns.** Training-camera success rises steadily (0.02 at 100k, 0.20 at 200k, 0.45 at 600k, 0.70 at 1M) with
  returns 150 -> 1430; per camera (last 5): train0 1.00, train2 1.00, train1 0.83, train5 0.66, train3 0.41, train4 0.00
  (train4 is the lowest camera, theta 30 deg). Held-out cameras stay at 0.0-0.05: the end-to-end CNN does not transfer to
  unseen viewpoints, which is the gap the campaign targets. DrM behaves as designed: dormant ratio falls below 0.2 at 10k
  (awake exploration), stays ~0.01, perturbation factor 0.95 at every 100k.
- **Comparison with the DrM paper.** The paper's hammer (2M frames = 1M agent steps, one fixed corner camera, 84x84,
  250-agent-step episodes, a "sparse" variant not in the released code) reaches ~95%. Our 0.65-0.70 on six random
  training cameras with 125-agent-step episodes is lower, as expected from the harder protocol; on the best two cameras
  it reaches 1.00. The pipeline (env, replay, DrM updates, evaluation) works.
- **shelf-place never receives reward.** Return is exactly 0.0 for every training and evaluation episode over 1M steps.
  Checked directly: the Meta-World v3 shelf-place reward is 0 until the object is grasped and lifted (in-place term zeroed
  while the block is on the table); random actions give return 0.0 over a full episode, the scripted expert 1180
  (`hammer` random 266 for contrast). So shelf-place is a sparse-reward, hard-exploration task here; the DrM CNN did not
  discover a lift in 1M steps (dormant ratio stays 0.2-0.7, perturbation factor 0.2-0.33). This is a property of the
  task, not a pipeline bug; it will likely be 0 for every method unless the representation makes the object salient.
- **Use of these runs.** Code and protocol changed after launch only in evaluation-side robustness (reset seeding fixed
  before launch; renderer sharing, memory and resume fixes later), not in the training loop or protocol, so both remain
  valid Stage 4 CNN seed-2000 runs (re-check before Stage 4).

## 2026-10-08 — Second OOM cascade; RAM-aware admission in the scheduler

- **Incident (08:11-08:36).** Other tenants' anonymous memory reached ~324 GB (ours 32 GB); the kernel OOM killer
  killed five of our pretraining runs (exit 137): it1-pick-place (08:11), it1-hammer, both base runs and it2-pick-place
  (08:35). The scheduler relaunched the first two straight into the same pressure.
- **Fix.** Jobs declare `ram_gb` (pretraining 20, RL 4-6, evaluators 6, exports 1); `scripts/jobs.py` launches a job only
  if host `MemAvailable` minus its `ram_gb` stays above `host_ram_reserve_gb` (40), so relaunches wait for memory
  instead of being killed again. Base pretraining runs have priority 0 and come back first; `max_restarts` 6 for
  pretraining so host-pressure kills do not exhaust the crash budget. Test: `test_launches_wait_for_host_ram`;
  suite 204/204. The it1 runs (just resumed from 70k) were stopped to give the base runs the memory.
- **State at 08:45.** Hammer base relaunched (from 110k); pick-place base, it1 x2 and it2-pick-place wait for RAM;
  it2-hammer kept running.

## 2026-10-08 — Base-encoder DrM proxy (100k pretraining, 200k agent steps, seeds 1000/1001): reference for iterations

| Task | Seed | Train cameras final / last-5 / AUC | Held-out last-5 | Trajectories last-5 | Peak train |
|---|---|---|---|---|---|
| hammer | 1000 | 0.13 / 0.13 / 0.15 | 0.03 | 0.24 | 0.44 |
| hammer | 1001 | 0.29 / 0.26 / 0.18 | 0.06 | 0.17 | 0.43 |
| pick-place | 1000 | 0.01 / ~0.01 / ~0.01 | ~0.01 | - | 0.03 |
| pick-place | 1001 | 0.00 / ~0.00 / ~0.00 | ~0.01 | - | 0.02 |

hammer learns quickly (0.4 near 80-100k agent steps vs 0.02 for the Stage 0 CNN at 100k; different seed, informal) but
then degrades; held-out success is low but above the CNN's (0.03-0.09 vs 0.00 at 200k). pick-place is not learned
within 200k agent steps. Runs: `runs/s1-proxy-base-{hammer,pick-place}-s{1000,1001}`.

## 2026-10-08 — Third OOM round; OOM priorities

- **Incident (09:06).** it1-pick-place and it2-hammer were OOM-killed shortly after being readmitted (available RAM
  fell from ~60 to ~30 GB within minutes as other tenants grew again).
- **Fix.** Jobs set their own `oom_score_adj` at launch (inherited by all children): improvement iterations 500,
  evaluators 400, RL runs 300, base pretraining and exports 0, so the kernel kills low-priority work first and the base
  encoders survive. The running it1-hammer session was raised to 500 by hand. Admission reserve raised to 60 GB.
  Liveness checks (`jobs.pid_alive`, `heartbeat.alive`) now treat ESRCH/ENOENT while reading `/proc` as dead (an
  intermittent scheduler test failure). Suite 204/204.
- **Plan.** Iterations are readmitted one at a time as memory allows; if host memory stays this tight, iterations 1
  and 2 run sequentially (both tasks each) instead of together.

## 2026-10-08 — Pretraining length: user update; pre-registered decision rule for our method

- **User decision (2026-10-08, ~09:40).** SinCro pretrains for exactly 300k steps on every task (configs:
  `max_global_steps: 300000`, i.e. exactly 300 000 updates; all other reference hyperparameters unchanged, including
  `lrate_decay: 500`; `docs/BASELINES.md` deviation 9). ReViWo keeps its reference length (100 001) and
  hyperparameters. Our method: 300k is the default (`configs/metaworld/base.yaml: train.steps: 300000`); the final
  value is chosen here from development-task evidence only (hammer, pick-place), fixed before `method-frozen-v1`, and
  used for all 8 tasks and every ablation of our method.
- **Running jobs keep their schedule.** `train.py` resumes with the command-line configs, so the six running
  200k-schedule runs (base and iterations 1-2) now pin `train.steps=200000` in `experiments/queue.yaml`; a restart
  cannot silently change their cosine schedule. `configs/metaworld/ablations/full_100k.yaml` (an old DROID-plan
  schedule) is marked unused.
- **Why annealed endpoints.** The LR follows warmup + cosine decay to `train.steps`, so an intermediate checkpoint of
  a 300k run is not a converged 200k encoder. The length study therefore compares the *annealed* endpoint of a 200k
  schedule (A200) with the annealed endpoint of a 300k schedule (A300) of the same configuration. The 300k run's
  intermediate checkpoints (100k, 200k) are reported as curve-shape context.
- **Calibration (from the base runs at 95k-120k, before any 300k data).** The in-training validation reads the first
  8 batches of an unshuffled loader, i.e. always the same 128 windows from the start of the validation split. Between
  consecutive evaluations moving-pixel PSNR changes by 0.2-1.0 dB, relative EPE by 0.03-0.05, retrieval by
  0.005-0.01 and held-out-camera probe R² by 0.1-0.8 (model fluctuation on a fixed subset), and the subset is not
  representative: on the full split hammer's relative EPE at 100k is 0.69, not the 0.47 of the in-training subset.
  Comparisons therefore use `scripts/evaluate.py` (full validation split, ~2000 windows per stride, probes; 4.5 min
  per checkpoint) on each compared checkpoint, not in-training evaluations. Also visible already: held-out-camera PSNR is flat at ~14.9 dB,
  held-out retrieval is near chance (0.04 vs 1/64) and held-out probe R² is negative, while training cameras give
  retrieval 0.83-0.90 and probe R² 0.93-0.97 (position) / 0.67-0.77 (velocity). Generalisation to held-out cameras is
  the weak point and will be reported in the Stage 1 diagnostics.
- **Plan.**
  1. Method configuration C = outcome of the iteration comparison at 100k (base, it1, it2 or a combination).
  2. A200: base and the winning iteration already run 200k schedules and continue to 200k; a new combination would
     also get a 200k-schedule run.
  3. A300: C with the default 300k schedule on hammer and pick-place, launched as soon as C is chosen.
  4. Exports at 100k/200k/300k of the 300k run and at A200; DrM proxies (seeds 1000/1001, 200k agent steps, full
     evaluation protocol) on each; full-split `scripts/evaluate.py` on each.
- **Pre-registered rule (A300 vs A200, same configuration).** Ten comparisons per task (stride 2 and 6 for each):
  moving-pixel PSNR (better if > 0.3 dB higher), held-out-camera PSNR (> 0.2 dB), relative EPE 0->2 (> 0.03 lower),
  training-camera retrieval top-1 (> 0.02), training-camera hand-velocity probe R² (> 0.03); held-out retrieval and
  held-out probes are reported but not counted (near chance / negative R², see above). RL: last-5 mean
  training-camera success averaged over the two proxy seeds; a difference counts only above 0.10.
  - **200k (shorter)** if A300 wins at most 4 of the 20 comparisons, or loses more than it wins, and A300's RL proxy
    is not better than A200's by more than 0.10 on either task.
  - **400k (longer)** if A300 wins at least 12 of 20 comparisons, at least 4 on each task, and its RL proxy is not
    worse than A200's by more than 0.10 on either task. The development encoders are then retrained at 400k.
  - **300k (default)** otherwise.
  - The decision, the 20 comparisons and the proxy table will be appended here before `method-frozen-v1`.

## 2026-10-08 — Mid-campaign review (user, ~11:40): constraints and launch order

- **Constraints restated by the user.** Six fixed training cameras, no camera-randomised data (viewpoint robustness
  must come from the training method); no dataset-derived scene initialisation in the decoder; dips after DrM
  perturbations are expected (no perturbation ablations); no frozen DINOv2 baseline in this regime. Hard rules
  unchanged. Running jobs continue; new work enters the queue in the order 1a-1c -> 3a/3b -> 2a-2c and 4a-4e screens
  (interleaved as RAM allows) -> item 5 -> counted iterations.
- **Open question raised with the user.** The decoder's anchors are initialised from per-task workspace statistics
  (`model.anchor_stats`: mean and std of all valid GT depth points of the task, `scripts/compute_workspace_stats.py`).
  This is a coarse Gaussian prior over the workspace, not per-scene geometry, but it is computed from the dataset.
  Kept unchanged until the user decides, because changing it would invalidate every running encoder.

## 2026-10-08 — Item 1 (review): near-view held-out sets, oracle sanity, Chamfer metrics

- **1a camera sets.** Each held-out camera perturbs one training camera (orbit about `LOOKAT`, radius 1.0):
  azimuth +/- 10 deg, elevation +/- 6 deg, radius +/- 5 %, then a lateral shift of the camera centre along its right
  vector of up to 0.12 m with the camera re-aimed at `LOOKAT` (exactly how `s4d/rl/env.trajectory_path("lateral")`
  moves the RL camera). All four offsets are drawn independently and uniformly. "trajectory" uses the full ranges
  (identical to the RL trajectory ranges); "near" uses half. Two cameras per training camera and set (12 per set),
  drawn once from fixed seeds and stored with the data. The 4 old far cameras are the "extrapolation" set (metric
  names `*_heldout` keep their meaning), reported for context only.
- **Ground truth.** `scripts/render_heldout_sets.py` replays the stored `qpos`/`qvel` of every validation episode
  (`splits/<task>_seed0.json`, 10 episodes; no new episodes) and renders RGB and metric depth for the 24 new cameras
  into `/home/ws/data/metaworld/splatter4d_v1/heldout_sets/<task>.hdf5`. Replay fidelity is checked by re-rendering
  the six training cameras and comparing with the stored frames; the file records the result and the script fails
  if they differ beyond renderer noise.
- **1b oracle.** Per validation window and time: the six training cameras' GT depth is lifted and fused (with GT
  colours) and z-buffer splatted (1 pixel) into every held-out camera. Reported per set: coverage (fraction of
  pixels hit) and PSNR on covered pixels. The model's held-out PSNR is reported unmasked and on the oracle-covered
  mask (`psnr_<set>`, `psnr_<set>_covered`).
- **1c Chamfer (metres, each direction separately; mean, p50, p90 per window, averaged over windows).** GT clouds are
  cropped to camera depth <= render far plane (3.0 m) and voxel-downsampled at 5 mm. Computed on the first window of
  every validation batch (about 125 windows per stride in a full evaluation).
  - CD-centers: Gaussian centres with opacity > 0.3 at t0 vs the fused GT cloud of all available cameras at t0
    (6 training + 4 extrapolation + 24 near/trajectory cameras for validation episodes); and dynamic-group centres
    vs GT points with motion score > 0.5 (training cameras, where GT motion exists).
  - CD-render: per trajectory-set camera, points lifted from the rendered expected depth (alpha > 0.5) vs points
    lifted from that camera's GT depth.
  - CD-motion: centres displaced by the predicted 0->2 motion vs GT points (training cameras, t0) displaced by the GT
    0->2 motion; all points and the dynamic subset (dynamic group vs motion score > 0.5).
  - "pred->gt" measures floaters, "gt->pred" missing surfaces; tails (p90) show floaters.
- **Retrieval and probes on near/trajectory.** The new sets exist for validation episodes only, so retrieval uses 64
  validation windows (evenly spaced within the 10 validation episodes) with the existing pairwise protocol
  (held-out camera query vs each training camera's states), reported for train, extrapolation, near and trajectory
  cameras on the same 64 windows (`retrieval_top1_<set>_val`). Probes (fitted on training-episode states as before)
  are scored on validation windows seen from each set (`r2_<target>_val_<set>`).
- **Where.** `scripts/evaluate.py` (full split) computes all of it; in-training evaluations stay unchanged
  (`eval.heldout_sets` off) to keep them cheap.
- **Pre-registered length rule, amendment A (before any A300-vs-A200 data exists).** The far-camera PSNR leaves the
  counted set (extrapolation is context only). Counted per task and stride (10, so 40 in total): training-camera
  moving-pixel PSNR (> 0.3 dB), trajectory-set PSNR unmasked (> 0.2 dB), relative EPE 0->2 (> 0.03 lower),
  training-camera retrieval (> 0.02), training-camera hand-velocity R² (> 0.03), CD-render on the trajectory set
  (mean of the two directions' p90, > 10 % lower), near-set retrieval (> 0.02), trajectory-set retrieval (> 0.02),
  trajectory-set hand-position R² (> 0.03), trajectory-set hand-velocity R² (> 0.03). Thresholds scale with the
  count: 200k if A300 wins at most 8 of 40 (or loses more than it wins); 400k if it wins at least 24 of 40 with at
  least 8 per task; otherwise 300k. The RL part is amended under item 5.

## 2026-10-08 — Item 2 (review): viewpoint robustness within the fixed six cameras — screens and rule

- **Screens** (single change each, hammer and pick-place, base configuration, base 200k LR schedule stopped at 100k
  with the new `train.stop_step`, so the comparison with the base run at 100k is at the same LR; full-split
  evaluation with the item-1 metrics). Precedents: depth-plus-reprojection augmentation (VISTA's baseline; Mirage, which
  notes the limit to small pose changes) and novel-view rendering from a feed-forward 3DGS reconstruction for
  augmentation (GenSplat, 2026). Ours use GT depth and stay inside the near/trajectory ranges of 1a.
  - **2a crop** (`aug.crop`): with probability 0.5 no crop, otherwise a random resized crop with scale [0.8, 1.0] and
    aspect ratio [0.95, 1.05] back to 128x128, the same crop for the three frames of a view; encoder input (and the
    motion-score map that only steers the tube mask) only; render targets untouched; no other augmentation; never
    at RL time.
  - **2b privileged near-view synthesis** (`aug.synth_views: 2`): per sample, two cameras drawn from the trajectory
    ranges around random training cameras; the six training cameras' GT depth+RGB at each time is fused and
    z-buffer splatted into them. Patches with < 90 % coverage in any of the three frames get the lowest tube-mask
    priority, so holes are masked tokens whenever enough covered patches exist. The views join InfoNCE (as positives)
    and slot consistency, and are extra render targets on covered pixels (RGB and depth). The decoder still decodes
    from a training camera. Regime-consistent: the same works with PointWorld depth in DROID.
  - **2c self-rendered view consistency** (`loss.self_render: 0.5`): the decoded scene is rendered at t0..t2 into two
    jittered cameras (same ranges); the renders are detached (stop-gradient on the decoder path) and encoded with
    the training tube mask; loss = 1 - cosine between their slot state and the (detached) source-view state.
- **2d decision rule (per screen, fixed before the runs).** Adopt if, on both development tasks (mean of strides 2
  and 6, full split, at 100k vs base at 100k): trajectory-set retrieval >= +0.10, trajectory-set hand-position R²
  >= +0.10 and trajectory-set hand-velocity R² >= +0.10; training-camera moving-pixel PSNR not lower by more than
  0.5 dB; CD-render (trajectory, symmetric p90) not higher by more than 5 % (evaluation tolerance). Winners are
  combined in one counted iteration (100k pretraining on both tasks + RL proxies of item 5).

## 2026-10-08 — Item 3 (review): replace shelf-place before Stages 3/4

- **3a screen** (`scripts/screen_tasks.py`; candidates sweep-into, coffee-push, lever-pull, assembly, push-back,
  drawer-open, plate-slide) under the campaign protocol: 125 agent steps, action repeat 2, Meta-World v3 reward summed
  per agent step, MT1 configurations seeded as in RL evaluation. Per task: 50 random-policy episodes (uniform actions:
  return distribution, success); 50 scripted-expert episodes (the Meta-World policy queried at every agent step, its
  action repeated, as an agent would act: return and success, required >= 80 %); object visibility in the six
  training cameras (object position from the observation projected into each camera every 5 agent steps of the expert
  episodes; visible when inside the image and within 2 cm of the rendered depth there; a camera counts if the object
  is visible in >= 50 % of checks; required >= 4 of 6).
- **Ranking for 3b (fixed now).** Among candidates passing 3a with random-policy mean return > 0: lowest
  random-policy success first (a task a random policy solves is too easy), ties broken by larger 3D object
  displacement in the expert episodes. The top 2 get DrM + CNN, seed 2000, 300k agent steps, light evaluation (every
  25k agent steps, 20 episodes over the training cameras).
- **3c choice (fixed now).** Requires (i) random-policy return > 0, (ii) CNN training-camera success at 300k above 0
  and below ~0.7, (iii) clear 3D/motion content. If both qualify: larger 3D object displacement, then success closer
  to 0.35. Then: collection (same recipe as the other tasks), split, workspace statistics, D2/D3, held-out sets;
  `docs/RL_PROTOCOL.md` records the replacement and the reason; shelf-place moves to an appendix note on sparse-reward
  tasks. If sweep-into is chosen, the official DrM per-task override (max_perturb_factor 0.9, target_lambda 0.6)
  applies to every method.

## 2026-10-08 — Item 4 (review): scene-general screens on the one-episode gate (diagnostics only)

- **Setup.** Gate configuration with iteration-2 settings (hammer ep001, ramp 300, constant LR after a 300-step
  warm-up, 6k steps, checkpoint at 6k), each compared at 6k with a fresh reference run of the same settings
  (`gate-it3b` only kept a 12k checkpoint, and the new metrics need the 6k model). S1 (motion weight 20) is re-evaluated
  from its existing 6k checkpoint. All evaluations: `scripts/evaluate.py` on the gate episode, strides 2 and 6.
- **Screens.** 4a decoder LR x3 (`train.decoder_lr_mult: 3`) with 4 decoder blocks at dim 256; 4b K = 4 slots with
  the full state concatenated to every parent token before the first block (`model.decoder.state_concat`), FiLM and
  per-block cross-attention kept; 4c AdaLN-zero conditioning (DiT-style shift/scale/gate per block, gates initialised
  at zero) instead of FiLM, plus Fourier features of each parent's anchor position added to its token; 4d hard-depth
  weight x3 for the first 20 % of training (`loss.depth_hard_boost`); 4e motion loss normalised over moving pixels:
  sum over pixels with motion_weight > 0 of w * Huber(pred - target) / max(sum w, 1 % of valid pixels) with
  w = 1 + |target| / 3 cm, plus a static term (weight 0.1) averaged over pixels with zero target motion
  (`loss.motion_norm: moving`), compared with the current all-pixel average and with S1.
- **Promotion rule (fixed now).** Metrics: PSNR, moving-pixel PSNR, relative EPE 0->2, CD-centers (symmetric p90),
  CD-motion (dynamic subset, symmetric p90), means of strides 2 and 6. Promote to a counted iteration if better than
  the reference on at least 3 of the 5 by margins of 0.5 dB / 0.5 dB / 0.03 / 10 % / 10 % and not worse on any by
  more than the same margin.

## 2026-10-08 — Item 5 (review): RL proxies to 400k on hammer; pick-place not counted

- **5a.** Every compared encoder gets hammer proxies of 400k agent steps (seeds 1000/1001, full evaluation protocol),
  so recovery after the perturbations at 100k and 200k is visible. Done for the queued iteration-1/2 and base-200k
  hammer proxies (edited before launch); the base-100k hammer proxies (200k steps, replays already deleted) are rerun
  at 400k as `s1-proxy400-base-hammer-s{1000,1001}`. Decision quantities: last-5 mean training-camera success at 400k
  (mean of two seeds) and recovery = success at 150k minus the best success at or before 100k, and success at 250k
  minus the best at or before 200k.
- **5b, amendment B to the pre-registered length rule (before any A300-vs-A200 data).** pick-place proxies are
  reported but not counted (no method learns it within 200k). The RL condition becomes: hammer only, last-5 mean at
  400k; 200k requires A300 not better than A200 by more than 0.10; 400k requires A300 not worse than A200 by more
  than 0.10. Recovery is reported alongside. With amendment A, the full rule is: 200k if A300 wins at most 8 of the 40
  representation comparisons (or loses more than it wins) and the hammer RL condition for 200k holds; 400k if it
  wins at least 24 of 40 (at least 8 per task) and the hammer RL condition for 400k holds; otherwise 300k.

## 2026-10-08 — Item 3a result: reserve-task screen (50 random + 50 expert episodes per task, campaign protocol)

- **Definition refined before the run.** Object visibility uses the simulator segmentation, as for the eight
  campaign tasks (`scripts/verify_metaworld_tasks.py`): the non-robot body that moves most in the expert episode
  (plus its descendants) is visible in a camera when at least one of its pixels is; it must be visible in >= 4 of the
  6 training cameras in > 50 % of sampled steps (every 5 agent steps). Script: `scripts/screen_tasks.py`; output
  `docs/task_screening/summary.json`, `ranking.json` and contact sheets.

| Task | 3a | Random return mean (fraction > 0) | Random success | Expert return | Expert success | Visible 4/6 | 3D object displacement |
|---|---|---|---|---|---|---|---|
| sweep-into | pass | 48.3 (1.00) | 0.02 | 1051 | 0.90 | 1.00 | 0.262 m |
| coffee-push | pass | 10.6 (1.00) | 0.00 | 1121 | 1.00 | 1.00 | 0.221 m |
| lever-pull | pass | 118.5 (1.00) | 0.00 | 577 | 1.00 | 1.00 | 0.000 m (rotation only) |
| assembly | pass | 111.8 (1.00) | 0.00 | 1797 | 1.00 | 1.00 | 0.268 m |
| push-back | **fail** (expert 0.64) | 2.7 (1.00) | 0.00 | 196 | 0.64 | 1.00 | 0.225 m |
| drawer-open | pass | 316.5 (1.00) | 0.00 | 1800 | 1.00 | 1.00 | 0.202 m |
| plate-slide | pass | 160.8 (1.00) | 0.00 | 2107 | 1.00 | 1.00 | 0.286 m |

- **Ranking by the pre-registered rule** (lowest random success, then larger object displacement): plate-slide,
  assembly, coffee-push, drawer-open, lever-pull, sweep-into. 3b runs DrM + CNN (seed 2000, 300k agent steps, light
  evaluation) on **plate-slide** and **assembly**. If neither meets 3c, the next two in the ranking follow.
- The displacement tie-break measures translation only, so lever-pull (a rotating lever) ranks low on it; it is
  recorded, not changed.

## 2026-10-08 — Item 1b/1c result: base encoders at 100k under the new held-out diagnostics

Full-split `scripts/evaluate.py` (merged code, ac02dd0); strides 2 / 6. Jobs `runs/fulleval2-s1-base-{hammer,pick-place}-100k`.

| Metric | hammer | pick-place |
|---|---|---|
| PSNR training cameras | 25.9 / 25.9 | 24.2 / 24.1 |
| PSNR near (unmasked / oracle-covered) | 22.1 / 25.5 | 21.5 / 24.8 |
| PSNR trajectory (unmasked / oracle-covered) | 20.7 / 24.9 | 20.3 / 24.6 |
| PSNR extrapolation (unmasked / covered) | 14.8 / 19.9 | 14.6 / 19.4 |
| Oracle on trajectory set: coverage, PSNR covered | 0.91, 26.1 | 0.91, 26.4 |
| Retrieval top-1, 64 validation windows: train / near / trajectory / extrapolation | 0.89 / 0.50 / 0.31 / 0.03 (s2) | 0.77 / 0.52 / 0.35 / 0.06 (s2) |
| Hand-position R²: train / near / trajectory | 0.96 / 0.86 / 0.51 | 0.96 / 0.92 / 0.69 |
| Hand-velocity R²: train / near / trajectory | 0.58 / 0.44 / 0.22 | 0.21 / 0.15 / -0.23 |
| CD-render trajectory (p2g p50 / p90; g2p p50 / p90), m | 0.007 / 0.074; 0.006 / 0.021 | 0.007 / 0.057; 0.007 / 0.018 |
| CD-centers (p2g p50 / p90; g2p p50 / p90), m | 0.145 / 0.311; 0.042 / 0.126 | 0.189 / 0.384; 0.054 / 0.115 |
| CD-centers dynamic p2g p90 / g2p p90, m | 0.43 / 0.08 | 0.48 / 0.07 |
| Relative EPE 0->2 | 0.69 / 0.65 | 0.98 / 0.88 |

- **Rendering generalises to near views; the state does not.** On the pixels the training cameras also observe
  (oracle-covered, 91 % of a trajectory view), held-out PSNR is within ~1 dB of the training cameras; the unmasked drop
  (4-5 dB) is the never-observed background. Retrieval falls from ~0.8-0.9 (training cameras) to ~0.5 (near) and
  ~0.3 (trajectory), and the velocity probe collapses on the trajectory set. This is what item 2 targets.
- **Where the CD-centers tail comes from (diagnostic on 16 windows per task).** 87-89 % of the opaque Gaussians
  (opacity > 0.3, ~7000 per scene) are more than 5 cm from every GT point, but 85-87 % of all opaque Gaussians lie
  behind the observed surface in every training camera that sees them (inside or under the table and objects), and
  only 1.5-1.7 % sit in free space in front of a surface. So the large p2g values measure hidden mass, not visible
  floaters; rendered geometry is accurate (CD-render p50 7 mm). A visible-only variant of CD-centers would separate the
  two; proposed to the user rather than changed silently.

## 2026-10-08 — Iterations 1-2 vs base at 100k (full split, new diagnostics); RL proxies pending

Means of strides 2 and 6; jobs `runs/fulleval2-s1-{base,it1-decdim256,it2-lambdadyn4}-{hammer,pick-place}-100k`.

| Metric | hammer base / it1 / it2 | pick-place base / it1 / it2 |
|---|---|---|
| PSNR training cameras | 25.86 / 24.80 / 25.43 | 24.15 / 24.19 / 24.42 |
| PSNR moving pixels | 23.93 / 24.27 / 24.30 | 17.02 / 16.92 / 17.37 |
| PSNR trajectory set (unmasked) | 20.76 / 20.66 / 21.02 | 20.34 / 20.29 / 20.34 |
| Relative EPE 0->2 | 0.67 / 0.89 / 0.70 | 0.93 / 0.97 / 0.92 |
| Retrieval trajectory set | 0.33 / 0.31 / 0.34 | 0.36 / 0.32 / 0.34 |
| Hand-position R² trajectory set | 0.49 / 0.53 / 0.71 | 0.68 / 0.73 / 0.72 |
| Hand-velocity R² trajectory set | 0.22 / -0.01 / 0.16 | -0.16 / -0.16 / -0.49 |
| CD-motion dynamic p2g p50 (m) | 0.28 / 0.21 / 0.10 | 0.37 / 0.31 / 0.17 |
| CD-centers dynamic g2p p90 (m) | 0.07 / 0.17 / 0.08 | 0.07 / 0.16 / 0.05 |

- **Iteration 1 (decoder dim 256): hypothesis H1 not supported.** PSNR is not higher (hammer -1.1 dB), motion is
  worse (hammer relative EPE 0.67 -> 0.89), trajectory-set retrieval and the hammer velocity probe drop, and the
  dynamic group covers the moving surfaces less (g2p p90 7 -> 16-17 cm). The gate-screen gain (S3) did not transfer.
- **Iteration 2 (lambda_dyn 4): H2 partly supported.** Moving-pixel PSNR +0.35 dB (both tasks) and the displaced
  dynamic centres sit much closer to the moved surfaces (CD-motion dynamic p50 halves), but relative EPE is unchanged,
  hammer training-camera PSNR drops 0.4 dB, and the pick-place velocity probe on the trajectory set is lower.
- **Decision pending** the hammer RL proxies at 400k (item 5: last-5 mean and recovery at 150k/250k), which arbitrate
  between the base configuration and iteration 2; iteration 1 is not carried into any combination.

## 2026-10-08 — Iteration 1 runs ended at their latest checkpoints; S1 not promoted

- **Iteration 1 stopped (13:54).** With iteration 1 rejected at 100k (entry above) it cannot become the method
  configuration, so its runs were ended at their latest checkpoints (hammer 110k, pick-place 100k) through
  `train.stop_step` in the queue and SIGTERM to their own sessions; both resumed and exited 0 at the stop step. This
  frees GPU time under heavy contention (the item 2 screens ran at 0.8 s/step). The user's "running jobs continue" was
  respected while integrating the review; this is a later decision on a rejected configuration. Iteration 2 continues
  (its decision waits for the 400k hammer proxies), as do the base runs.
- **S1 (motion weight 20) vs the fresh gate reference at 6k** (item 4 rule, means of strides 2/6): PSNR -0.99 dB,
  moving PSNR -0.62 dB (both worse beyond the margin), relative EPE 0.00, CD-centers -1 cm, CD-motion dynamic -1 cm:
  not promoted. Gate reference at 6k: PSNR 25.81 / 25.89, moving PSNR 23.11 / 21.99, relative EPE 0.51 / 0.32.

## 2026-10-08 — Item 4 result: no gate screen is promoted

Gate (hammer ep001, iteration-2 settings, 6k steps), full evaluation on the gate episode, means of strides 2 and 6;
jobs `runs/fulleval2-screen-*-6k`. Rule (fixed before the runs): better on >= 3 of 5 by 0.5 dB / 0.5 dB / 0.03 / 10 % /
10 % and worse on none beyond those margins.

| Screen | PSNR | Moving PSNR | Rel. EPE 0->2 | CD-centers sym p90 (m) | CD-motion dyn sym p90 (m) | CD-motion dyn p2g p50 (m) | Decision |
|---|---|---|---|---|---|---|---|
| reference | 25.85 | 22.55 | 0.413 | 0.222 | 0.171 | 0.083 | - |
| S1 motion x20 | 24.86 | 21.93 | 0.412 | 0.219 | 0.169 | 0.048 | no (PSNR, moving PSNR worse) |
| 4a decoder LR x3, 4 blocks x 256 | 7.38 | 9.38 | 1.000 | 4.375 | 4.363 | 4.187 | no (diverged near step 1000, never recovered) |
| 4b K = 4 + state concatenation | 24.81 | 20.92 | 0.504 | 0.217 | 0.203 | 0.203 | no (4 metrics worse) |
| 4c AdaLN-zero + Fourier anchors | 25.12 | 22.66 | 0.390 | 0.209 | 0.123 | 0.027 | no (PSNR -0.73 dB; CD-motion -28 %) |
| 4d hard depth x3 for 20 % | 24.51 | 22.08 | 0.422 | 0.225 | 0.161 | 0.133 | no (PSNR -1.34 dB) |
| 4e moving-pixel motion normalisation | 24.84 | 21.29 | 0.486 | 0.216 | 0.180 | 0.039 | no (PSNR, moving PSNR, EPE worse) |

- 4e vs the current all-pixel average and vs S1 (as the review asked): both re-weighted motion losses lower PSNR by
  ~1 dB; the normalised loss moves the dynamic centres closer to the moved surfaces (p50 3.9 cm vs 8.3 cm) but its
  relative EPE is worse (0.49 vs 0.41); S1 keeps the EPE and costs moving PSNR. Neither is promoted.
- Every change lowered PSNR by 0.7-1.3 dB against the fresh reference, so the run-to-run spread at 6k matters for
  reading these margins. A second-seed reference (`screen-g-ref-seed1`, `train.seed=1`) is queued to measure it; it is
  context only and does not change the decisions above.

## 2026-10-08 — Item 3b interim (16:07): next two candidates started early

- plate-slide (CNN, seed 2000): training-camera success 0.65 at 100k and 0.89 at 150k, above the ~0.7 ceiling of
  criterion (ii); assembly: 0.00 at 150k. If neither qualifies at 300k, the pre-registered ranking continues with
  coffee-push and drawer-open; both are started now (same protocol) so that the 3c decision is not delayed. The
  decision itself still uses the 300k results of the runs in ranking order.

## 2026-10-08 — Item 4 context: seed-to-seed spread of the gate at 6k

| Gate reference | PSNR | Moving PSNR | Rel. EPE 0->2 | CD-centers sym p90 | CD-motion dyn sym p90 | CD-motion dyn p2g p50 |
|---|---|---|---|---|---|---|
| seed 0 (`screen-g-ref`) | 25.85 | 22.55 | 0.413 | 0.222 | 0.171 | 0.083 |
| seed 1 (`screen-g-ref-seed1`) | 25.07 | 21.58 | 0.437 | 0.227 | 0.168 | 0.036 |

- Changing only the seed moves PSNR by 0.78 dB, moving PSNR by 0.97 dB and the dynamic CD-motion median by 4.7 cm.
  That is as large as the 0.5 dB / 10 % promotion margins and as the screens' effects: every screen's PSNR deficit
  (0.7-1.3 dB, except the diverged 4a) and 4c's CD-motion gain (2.7 cm) lie within one seed's spread.
- The item 4 decisions stand as pre-registered (nothing promoted). The honest reading is that the one-episode gate at
  6k cannot resolve effects of this size: none of 4b-4e is shown to be worse or better; only 4a (divergence) is a
  clear negative. Gate screens should use several seeds or be replaced by full-split 100k screens in future.

## 2026-10-08 — Item 5: base encoder (100k) hammer proxies at 400k agent steps

`analysis/review_rules.py proxies s1-proxy400-base-hammer` (seeds 1000 / 1001):

| Seed | Last-5 train cameras | Last-5 held-out | Last-5 trajectories | Peak train | Recovery 150k | Recovery 250k |
|---|---|---|---|---|---|---|
| 1000 | 0.23 | 0.10 | 0.24 | 0.49 | -0.02 | -0.20 |
| 1001 | 0.61 | 0.04 | 0.39 | 0.73 | +0.13 | +0.08 |
| mean | 0.42 | 0.07 | 0.31 | 0.61 | +0.06 | -0.06 |

The two seeds differ by 0.38 in last-5 success, so two-seed proxy comparisons can only resolve large effects; the
0.10 threshold of the amended length rule is below this spread. Iteration 1 and 2 proxies are still running.

## 2026-10-08 — Item 5: iteration 1 hammer proxies at 400k (consistent with its rejection)

| Encoder (100k) | Last-5 train cameras (s1000 / s1001 / mean) | Last-5 held-out | Last-5 trajectories | Peak | Recovery 150k / 250k |
|---|---|---|---|---|---|
| base | 0.23 / 0.61 / 0.42 | 0.07 | 0.31 | 0.61 | +0.06 / -0.06 |
| iteration 1 (decoder dim 256) | 0.20 / 0.21 / 0.21 | 0.04 | 0.16 | 0.42 | -0.06 / -0.28 |

Iteration 1 is below base on every RL quantity, in line with its pretraining metrics; it stays rejected. Iteration 2
proxies are at ~230k of 400k.

## 2026-10-08 — Item 3b: assembly fails criterion (ii)

assembly (DrM + CNN, seed 2000, 300k agent steps, light evaluation): training-camera success 0.00 at every evaluation
(25k-300k) while the return rises (125 -> 1250; scripted expert ~1800): no success above 0 at 300k, so it fails
criterion (ii). plate-slide is far above the ~0.7 ceiling (0.89-1.00 from 150k). The choice continues with
coffee-push and drawer-open (0.29 and 0.44 at 75k), in ranking order.
- plate-slide final: training-camera success 0.89 at 300k (0.85-1.00 from 125k on): fails criterion (ii) (too easy).

## 2026-10-08 — Item 5 / iteration 2: not adopted; base configuration remains the method configuration

| Encoder (100k) | Last-5 train cameras (s1000 / s1001 / mean) | Last-5 held-out | Last-5 trajectories | Peak | Recovery 150k / 250k |
|---|---|---|---|---|---|
| base | 0.23 / 0.61 / 0.42 | 0.07 | 0.31 | 0.61 | +0.06 / -0.06 |
| iteration 1 (decoder dim 256) | 0.20 / 0.21 / 0.21 | 0.04 | 0.16 | 0.42 | -0.06 / -0.28 |
| iteration 2 (lambda_dyn 4) | 0.23 / 0.31 / 0.27 | 0.06 | 0.34 | 0.53 | +0.16 / -0.19 |

- Iteration 2's RL proxy is not better than base (mean last-5 0.27 vs 0.42; within base's own 0.38 seed spread, so
  not shown worse either), and its pretraining gains were mixed (entry above). With no clear improvement it is not
  adopted. The base configuration remains the method configuration C, pending the item 2 viewpoint screens.
- The iteration-2 runs were ended at their latest checkpoints (hammer 130k, pick-place 140k) with `train.stop_step`,
  as for iteration 1; only the base runs continue to 200k (the A200 point of the length study for C = base).
- Counted iterations used so far: 2 of 8 (both rejected).

## 2026-10-08 — Length study: A300 for the base configuration started (19:50)

- Base pick-place reached 200k (A200; its full evaluation, export and proxies are queued); base hammer follows.
- With iterations 1-2 not adopted, the method configuration is base unless an item 2 viewpoint screen is adopted
  (decision ~15 h away; the 2d thresholds are strict). To keep the length study moving, A300 for base starts now:
  `s1-pretrain-{hammer,pick-place}-base300k` (configs/metaworld/base.yaml, `train.steps=300000`, everything else as the
  base runs). If item 2 changes the configuration, A200/A300 are redone for the new configuration and these runs are
  context only. Exports, full-split evaluations and hammer proxies at 100k/200k/300k are queued as the run progresses.
- drawer-open final: training-camera success 1.00 at 300k (0.76-1.00 from 150k on): fails criterion (ii) (too easy).

## 2026-10-08 — Item 3c decided: coffee-push replaces shelf-place

| Reserve task (DrM + CNN, seed 2000) | Training-camera success at 300k | Range over the run | (i) random return > 0 | (ii) 0 < success < ~0.7 | (iii) 3D/motion content |
|---|---|---|---|---|---|
| plate-slide | 0.89 | 0.19-1.00 | yes (160.8) | no (too easy) | puck slides 0.29 m |
| assembly | 0.00 | 0.00 | yes (111.8) | no (never succeeds) | peg lifted, 0.27 m |
| coffee-push | **0.18** | 0.10-0.42 | **yes (10.6)** | **yes** | **mug pushed 0.22 m, visible in 6/6 cameras** |
| drawer-open | 1.00 | 0.08-1.00 | yes (316.5) | no (too easy) | drawer slides 0.20 m |

- coffee-push is the only candidate meeting all three criteria. The verification gate passed (20/20 scripted
  successes, object visible in >= 4 of 6 cameras at every sampled step). Data pipeline queued with the same recipe as
  the other tasks: 5-episode pilot, 250-episode collection, split, workspace statistics, D2/D3 checks, held-out sets.
- Recorded in `docs/RL_PROTOCOL.md` ("Campaign tasks and the shelf-place replacement"), with shelf-place as an appendix
  note on sparse-reward tasks. Task, baseline and RL configs for coffee-push follow with the next code merge.

## 2026-10-08 — Length study: A200 (base, annealed 200k) vs the same run at 100k

Full split, strides 2 / 6; jobs `runs/fulleval2-s1-base-*-100k` and `runs/fulleval-s1-base-*-200k`.

| Metric | hammer 100k -> 200k | pick-place 100k -> 200k |
|---|---|---|
| PSNR training cameras | 25.88 / 25.85 -> 26.08 / 26.03 | 24.17 / 24.14 -> 24.63 / 24.55 |
| PSNR moving pixels | 25.10 / 22.75 -> 25.76 / 23.18 | 16.19 / 17.86 -> 16.42 / 18.21 |
| PSNR trajectory set | 20.75 / 20.77 -> 20.64 / 20.68 | 20.33 / 20.35 -> 20.49 / 20.49 |
| Relative EPE 0->2 | 0.69 / 0.65 -> 0.80 / 0.69 | 0.98 / 0.88 -> 1.00 / 0.83 |
| Retrieval train / near / trajectory (s2) | 0.89 / 0.50 / 0.31 -> 0.89 / 0.49 / 0.30 | 0.77 / 0.52 / 0.35 -> 0.76 / 0.52 / 0.33 |
| Hand-velocity R² train / trajectory (s2) | 0.58 / 0.22 -> 0.58 / 0.07 | 0.21 / -0.23 -> 0.33 / -0.03 |
| CD-render trajectory p2g p90 (m) | 0.074 -> 0.074 | 0.057 -> 0.058 |

- Doubling the training (to the end of the 200k schedule) adds 0.2-0.5 dB of training-camera PSNR and nothing on the
  viewpoint, retrieval, probe or geometry metrics; hammer motion error gets slightly worse. The representation metrics
  have largely plateaued by 100k under this configuration. A300 (300k schedule) will show whether a longer schedule
  changes that; the pre-registered rule compares A300 with A200.

## 2026-10-09 — coffee-push data collected and checked

- Collection: 250 episodes (same mixture and seed recipe as the other tasks), 16.9 GiB, 3.0 h
  (`/home/ws/data/metaworld/splatter4d_v1/coffee-push.hdf5`); split 240 / 10 (`splits/coffee-push_seed0.json`).
- D2 / D3 (`docs/data_checks/coffee-push/summary.json`, same script and unchanged thresholds): D2 median 1.00 mm, D3
  median 1.93 mm (p90 4.10 mm < 5 mm), world-body motion exactly zero: all pass, in line with the other tasks
  (hammer: 1.00 / 1.97 mm). Workspace statistics and the near/trajectory held-out sets are running.
- 2026-10-09 01:28: coffee-push held-out sets rendered after the target-site replay fix (merge db5f6c8): all 10
  validation episodes replay with max |rgb| difference 0 on the training cameras; 2.0 GiB. coffee-push is ready for
  Stage 3/4 (data, split, statistics, D2/D3, held-out sets, task and baseline configs).

## 2026-10-09 — Length study: hammer RL proxies of the base encoder at 100k vs A200

| Base encoder | Last-5 train cameras (s1000 / s1001 / mean) | Last-5 held-out | Last-5 trajectories | Peak | Recovery 150k / 250k |
|---|---|---|---|---|---|
| 100k (from the 200k schedule) | 0.23 / 0.61 / 0.42 | 0.07 | 0.31 | 0.61 | +0.06 / -0.06 |
| A200 (annealed 200k) | 0.30 / 0.50 / 0.40 | 0.06 | 0.29 | 0.58 | -0.10 / +0.06 |

The 200k encoder is indistinguishable from the 100k one on RL (mean 0.40 vs 0.42, both inside the 0.38 seed
spread), matching the flat representation metrics between 100k and 200k. A200's hammer RL value for the length rule
is 0.40 (last-5 mean at 400k). pick-place A200 proxies: 0.003 (not counted).

## 2026-10-09 — Item 2a result: crop augmentation rejected by the 2d rule

Full split at 100k (base 200k schedule, `train.stop_step=100000`) vs the base run at 100k; means of strides 2 / 6.

| Metric | hammer base -> crop | pick-place base -> crop | 2d threshold |
|---|---|---|---|
| Retrieval trajectory set | 0.33 -> 0.37 (+0.04) | 0.36 -> 0.35 (-0.01) | >= +0.10 |
| Hand-position R² trajectory | 0.49 -> 0.82 (+0.33) | 0.68 -> 0.78 (+0.09) | >= +0.10 |
| Hand-velocity R² trajectory | 0.22 -> 0.29 (+0.07) | -0.16 -> 0.01 (+0.17) | >= +0.10 |
| Moving-pixel PSNR | 23.93 -> 23.96 | 17.02 -> 17.44 | >= -0.5 dB |
| CD-render trajectory, symmetric p90 | 4.7 -> 4.5 cm | 3.8 -> 4.4 cm (+15 %) | <= +5 % |

- Rejected: the retrieval criterion fails on both tasks and pick-place also fails hand-position R² and CD-render.
- Context (not used for the decision): crop strongly improves position decoding from trajectory-set cameras on hammer
  (+0.33 R²) and velocity on pick-place, but lowers retrieval among the training cameras (hammer 0.89 -> 0.75, pick-
  place 0.78 -> 0.75): with crop, states of the same scene from different training cameras are less alike, while
  linear position/velocity content becomes more viewpoint-robust.

## 2026-10-09 — Item 2d decided: synthetic near views adopted; crop and self-render rejected

Full split at 100k (base 200k schedule stopped at 100k) vs the base run at 100k; means of strides 2 and 6. Decision
columns per the pre-registered rule (retrieval, hand-position R² and hand-velocity R² on the trajectory set each
>= +0.10; moving-pixel PSNR >= -0.5 dB; CD-render trajectory symmetric p90 <= +5 %; both tasks).

| Task / screen | Retrieval traj | Pos R² traj | Vel R² traj | Moving PSNR | CD-render sym p90 | 2d | Context: PSNR | rel. EPE | dyn. share | retrieval train |
|---|---|---|---|---|---|---|---|---|---|---|
| hammer base | 0.33 | 0.49 | 0.22 | 23.93 | 4.77 cm | - | 25.86 | 0.67 | 0.82 | 0.91 |
| hammer crop | 0.37 | 0.82 | 0.29 | 23.96 | 4.48 cm | fail | 25.53 | 0.69 | 0.82 | 0.77 |
| hammer synth | 0.74 | 0.89 | 0.51 | 24.85 | 2.54 cm | pass | 25.95 | 0.64 | 0.89 | 0.89 |
| hammer self-render | 0.73 | 0.83 | 0.50 | 25.11 | 4.14 cm | pass | 25.99 | 0.58 | 0.91 | 0.93 |
| pick-place base | 0.36 | 0.68 | -0.16 | 17.02 | 3.78 cm | - | 24.15 | 0.93 | 0.66 | 0.78 |
| pick-place crop | 0.35 | 0.78 | 0.01 | 17.44 | 4.36 cm | fail | 23.79 | 0.90 | 0.74 | 0.75 |
| pick-place synth | 0.77 | 0.91 | 0.08 | 18.18 | 2.82 cm | pass | 24.50 | **1.00** | **0.06** | 0.89 |
| pick-place self-render | 0.69 | 0.94 | 0.47 | 18.58 | 4.29 cm (+13.5 %) | **fail** | 25.11 | 0.63 | 0.88 | 0.90 |

- **Synthetic near views (2b): adopted** (passes on both tasks). **Crop (2a): rejected** (retrieval, and on pick-place
  position R² and CD-render). **Self-render (2c): rejected** by the CD-render tolerance on pick-place only (+13.5 % vs
  <= +5 %), while it passes every other criterion and is the only screen that improves motion on pick-place.
- **Problem not covered by the rule: synthetic views abandon the dynamic group on pick-place.** The dynamic group's
  share of moving pixels falls from 0.44 at step 0 to 0.05 by 10k and stays there (base: rises to 0.66-0.77); relative
  EPE is 1.00 at every evaluation (zero predicted motion), the moving object is drawn by scene Gaussians. On hammer the
  same change improves motion (EPE 0.67 -> 0.64, dynamic share 0.82 -> 0.89). Raised with the user.
- **Next, per the plan.** "Combine winners in one counted iteration": iteration 3 = base + synthetic views. Its 100k
  pretraining is the screen runs themselves (same configuration and schedule), so iteration 3 adds the exports and RL
  proxies (hammer 400k agent steps, pick-place 200k reported). The method configuration C becomes base + synthetic
  views, so the length study is redone for it: the synth screen runs continue to 200k on their schedule (A200) and
  300k-schedule synth runs start (A300). The base A300 runs continue as context and as the fallback if the user
  decides against synthetic views. Counted iterations used: 3 of 8.

## 2026-10-09 — Iteration 3 (synthetic near views) RL proxies

| Encoder (100k) | Hammer last-5 train cameras (s1000 / s1001 / mean) | Last-5 held-out | Last-5 trajectories | Peak | Recovery 150k / 250k |
|---|---|---|---|---|---|
| base | 0.23 / 0.61 / 0.42 | 0.07 | 0.31 | 0.61 | +0.06 / -0.06 |
| iteration 3 (synthetic views) | 0.56 / 0.22 / 0.39 | 0.09 | 0.15 | 0.50 | +0.04 / -0.05 |

pick-place (200k agent steps, reported only): base 0.003, iteration 3 0.024 (peak 0.067).

- The large representation gains on the trajectory set (retrieval 0.33 -> 0.74, hand-position R² 0.49 -> 0.89) do not
  show in hammer RL with two seeds: training-camera success is unchanged within the seed spread (0.39 vs 0.42),
  held-out-camera success 0.09 vs 0.07, and trajectory success is lower (0.15 vs 0.31; per-seed 0.23 / 0.07 vs
  0.24 / 0.39). Two seeds cannot resolve differences of this size; the decision of item 2d (rule-based, representation
  metrics) stands, and the user's call on the pick-place motion collapse is pending.

## 2026-10-09 — Fourth host-memory incident (11:10-11:20); synth 300k runs paused

- **Incident.** Host MemAvailable fell to ~6 GB with swap full (other tenants ~250 GB; our six pretraining runs had
  grown to 17-30 GB each including DataLoader shared memory, ~156 GB in total). Pretraining slowed to >200 s/step and
  the kernel OOM-killed (exit 137) the base300k@100k hammer proxies and their evaluators, base300k hammer pretraining
  and synth300k pick-place pretraining.
- **Action.** Launches held, the two synth300k runs (20-22k steps, latest checkpoints at 20k) stopped and held behind
  `hold-memory` (they lose 1-2k steps), HOLD released after checking the test evidence. Available memory recovered to
  ~36 GB and the remaining runs to 0.46-0.54 s/step. The killed jobs restart from their checkpoints through the
  scheduler as host RAM admits them (RAM admission: job need + 60 GB reserve).
- **Next.** The synth300k runs resume when memory allows and the user has decided on synthetic views; if the user
  rejects synthetic views they are not needed. Concurrency of pretraining runs is kept at <= 4 until then.
- **Second round (11:37).** Other tenants grew further and the remaining pretraining runs (base300k pick-place, both
  synth continuations to 200k) were OOM-killed as well. Their declared RAM (20 GB) was below the measured 17-30 GB per
  run including DataLoader shared memory, so admission had over-committed; pretraining jobs now declare 30 GB. With
  the 60 GB reserve this admits one pretraining run per ~90 GB of available host memory: synth hammer (continuation to
  200k) restarted first (priority 1); synth pick-place and both base300k runs wait for RAM and resume from their
  latest checkpoints (synth 130k, base300k 110k).

## 2026-10-09 — Context: hammer proxies of the base300k run at 100k

| Base encoder | Last-5 train cameras (s1000 / s1001 / mean) | Last-5 held-out | Last-5 trajectories | Peak |
|---|---|---|---|---|
| 200k schedule @100k | 0.23 / 0.61 / 0.42 | 0.07 | 0.31 | 0.61 |
| A200 (200k schedule, annealed) | 0.30 / 0.50 / 0.40 | 0.06 | 0.29 | 0.58 |
| 300k schedule @100k | 0.13 / 0.60 / 0.37 | 0.08 | 0.34 | 0.52 |

All three base encoders are indistinguishable on hammer RL within the two-seed spread (seed 1001 is consistently the
stronger seed, 0.50-0.61; seed 1000 0.13-0.30). Context only; the length decision compares A300 with A200.

## 2026-10-09 — Length study for C = base + synthetic views: A200 representation metrics

Full split, means of strides 2 / 6. Columns: PSNR, moving PSNR, retrieval trajectory set, hand-position R² trajectory,
hand-velocity R² trajectory, relative EPE 0->2, dynamic share of moving pixels, CD-render trajectory symmetric p90 (cm).

| Task / encoder | PSNR | Mov. PSNR | Retr. traj | Pos R² traj | Vel R² traj | Rel. EPE | Dyn. share | CD-render |
|---|---|---|---|---|---|---|---|---|
| hammer base 100k | 25.86 | 23.93 | 0.33 | 0.49 | 0.22 | 0.67 | 0.82 | 4.77 |
| hammer base A200 | 26.06 | 24.47 | 0.32 | 0.52 | 0.13 | 0.74 | 0.82 | 4.76 |
| hammer synth 100k | 25.95 | 24.85 | 0.74 | 0.89 | 0.51 | 0.64 | 0.89 | 2.54 |
| hammer synth A200 | 26.44 | 25.74 | 0.80 | 0.93 | 0.62 | 0.51 | 0.90 | 2.49 |
| pick-place base 100k | 24.15 | 17.02 | 0.36 | 0.68 | -0.16 | 0.93 | 0.66 | 3.78 |
| pick-place base A200 | 24.59 | 17.32 | 0.34 | 0.70 | -0.07 | 0.91 | 0.72 | 3.81 |
| pick-place synth 100k | 24.50 | 18.18 | 0.77 | 0.91 | 0.08 | 1.00 | 0.06 | 2.82 |
| pick-place synth A200 | 24.71 | 18.63 | 0.79 | 0.93 | 0.16 | 1.00 | 0.06 | 2.72 |

- With synthetic views, training from 100k to the end of the 200k schedule keeps improving on hammer (moving PSNR
  +0.9 dB, trajectory retrieval +0.06, velocity R² +0.11, relative EPE 0.64 -> 0.51) and slightly on pick-place, unlike
  the base configuration, whose representation metrics were flat between 100k and 200k.
- The pick-place motion collapse persists at 200k (relative EPE 1.00, dynamic share 0.06): a stable property of the
  configuration on this task, not a slow start.
- A200 hammer proxies (400k agent steps, seeds 1000/1001) and pick-place proxies (200k) are running; A300 synth runs at
  ~60k of 300k.

## 2026-10-09 — User directives (items 1-3, S1-S6): validation standard (pre-registered before any result)

- **Standard for items 1-2 and the S1 resolution.** Two pretraining seeds (`train.seed` 0 and 1) per variant on hammer
  and pick-place, base 200k schedule stopped at 100k (`train.stop_step=100000`), full-split `scripts/evaluate.py`.
  Existing runs are reused as seed 0 only where the configuration is byte-identical.
- **Margins (pre-registered formula).** For each metric and task, margin = max(|reference seed 0 - reference seed 1|,
  floor), computed from the two seeds of that comparison's reference once they exist and written into this log before
  any variant is compared. Floors: PSNR-type 0.2 dB; retrieval 0.03; probe R² 0.05; relative EPE 0.03; EPE 1 mm;
  dynamic share 0.05; utilisation / hidden / floater fractions 0.02 (absolute); Chamfer metrics 5 % (relative).
  "Better / worse beyond the margin" compares the two-seed means of variant and reference.
- **Motion metrics use stride 6 as primary** (stride 2 reported) in every comparison; other metrics use the mean of
  strides 2 and 6 as before.
- **RL proxies (S2 adopted).** Every compared encoder gets 6 hammer seeds (1000-1005), 400k agent steps, full
  evaluation protocol. Decision quantity: mean over seeds of the last-5 training-camera success; RL margin =
  max(0.10, 2 x standard error of the difference of means). Two-seed proxies stay context only.
- **Budget.** Items 1-2 and S1 are counted iterations beyond the original 8 (user authorisation, 2026-10-09). Counted
  so far: 3 (it1 decoder dim, it2 lambda_dyn, it3 synthetic views). `method-frozen-v1` only after items 1-2.

## 2026-10-09 — S1 decided first: does a synthetic-view variant keep the viewpoint gains without the pick-place motion collapse?

- **Why first.** S1 fixes the configuration C on which items 1-2 run; items 1-2 code is written meanwhile.
- **Runs** (counted iteration 4; all at 100k of the base 200k schedule, prefetch per item 3):
  - R = current C (synthetic views as invariance positives and render targets): seed 0 = `s2-screen-synth-*`
    (identical configuration), seed 1 = `s1v-synth-seed1-*` (new).
  - V1 = synthetic views as invariance positives only (`aug.synth_render=false`), seeds 0/1.
  - V2 = synthetic views + self-render (`loss.self_render=0.5`), seeds 0/1.
- **Rule (margins from R's two seeds).** A variant V qualifies if
  (a) on pick-place the motion recovers: dynamic share >= 0.5 and relative EPE (stride 6) below R by more than the
  margin;
  (b) on both tasks, trajectory-set retrieval, hand-position R² and hand-velocity R² are not below R beyond the
  margins;
  (c) on both tasks, training-camera PSNR, moving-pixel PSNR and CD-render (trajectory, symmetric p90) are not worse
  than R beyond the margins.
  If both qualify, V1 is chosen (fewer loss terms) unless V2 is better than V1 beyond the margins on a majority of
  {retrieval traj, position R² traj, velocity R² traj, moving PSNR, relative EPE s6, CD-render} pooled over both tasks
  (>= 7 of 12). If none qualifies, C stays as is and the collapse is reported as a limitation.

## 2026-10-09 — Item 1 (pre-registered): depth supervision redesign

- **1a occlusion / free-space loss** (`loss.occlusion`, `loss.occlusion_margin` m = 0.02 m: the measured D3 depth
  error has median 1.9-2.0 mm and p90 ~4 mm, the 2 cm margin is ~5x the p90 and also absorbs Gaussian extent; weight
  1.0, a first guess recorded before any run). Centres at each time (with predicted motion) are projected into all
  training cameras; GT depth is read at the nearest pixel (0 < D < far). Behind term min_v relu(z_v - D_v - m) for
  centres behind by more than m in every camera where they land on valid depth; front term mean_v relu(D_v - z_v - m)
  over cameras where they lie in observed free space; both averaged over Gaussians, gradient to positions only.
- **1b** depth validity in the losses becomes 0 < depth < `render.far` for every variant (bug fix; therefore D0 is
  re-run with it and the existing runs are not reused for D0).
- **1c diagnostics** in training logs and `scripts/evaluate.py`: utilisation (fraction of Gaussians whose accumulated
  blending weight sum_p T_i(p) alpha_i(p) exceeds 1e-3 in at least one training camera, computed exactly as the
  gradient of a summed scalar-feature render), hidden fraction (opaque Gaussians behind the surface by > m in every
  camera), floater fraction (opaque Gaussians in front of the surface by > m in at least one camera), visible-only
  CD-centers (centres passing the utilisation test), old CD-centers kept, step time and GPU time per step.
- **1d variants** (2 seeds, both tasks, current image-space motion loss, on C after S1): D0 current depth losses;
  D1 = D0 + occlusion; D2 = expected-depth L1 + coverage + occlusion (gradient and hard-depth pass removed; the hard
  pass is skipped entirely when its weight is 0); D3 = D2 + hard depth; D4 = D2 + gradient term.
- **Rule (margins from D0's two seeds).** D2 is adopted if (i) it is better than D0 beyond the margins on utilisation,
  hidden fraction and visible-only CD-centers (symmetric p90) on both tasks; (ii) it is not worse than D0 beyond the
  margins on training PSNR, moving PSNR, trajectory PSNR, CD-render, relative EPE (s6), trajectory retrieval,
  trajectory position R² and velocity R², on both tasks; (iii) neither D3 nor D4 beats D2, where "beats" means better
  beyond the margins on >= 3 of the 16 (ii) metric-task pairs and worse beyond them on none; a beating variant brings
  its term back (both beating: D1's term set). If D2 fails (i) or (ii), D1 is adopted if it passes (i) and (ii)
  against D0; otherwise D0 stays. Step-time saving of dropping the hard pass is reported.
- **1e** follows item 2 with the same leave-one-out protocol; candidates chosen then and pre-registered before running.

## 2026-10-09 — Item 2 (pre-registered): Gaussian-space 3D motion loss (M3D) vs image-space (M2D), exclusive

- **M3D** (`loss.motion_space: gaussian`; the image-space loss is then not computed into the objective, only its
  metrics). Per view and pair (0->1 at t0, 1->2 at t1, 0->2 at t0, pair weights 0.4/0.4/0.2): N = 256 track points
  sampled from pixels with motion_weight > 0, half from |target| > 5 mm (when available) and half uniform, lifted with
  GT depth at the source time. Displacement term: each point's k = 4 nearest dynamic Gaussians at the source time
  within r = 3 cm, Gaussian kernel (sigma = r/2), Huber(delta 1 cm) between the Gaussian's pair displacement and the
  target, weighted by kernel and motion_weight, averaged over points (static points pull nearby dynamic Gaussians to
  zero motion). Attraction term: for moving points, the distance to the nearest dynamic centre at the source time
  (Huber, delta 1 cm), gradient to positions, averaged over moving points. Total = displacement + 1.0 x attraction,
  times the existing `loss.motion` weight and temporal ramp. Tube masking and pixel weights unchanged.
- **New metrics** (both motion spaces): relative EPE in GT-magnitude bins 5-10 mm, 1-3 cm, > 3 cm, and EPE in mm, on
  the image-space evaluation used so far.
- **Comparison** on the item-1 winner (2 seeds, both tasks; M2D = the item-1 winner itself, reused).
- **Rule (margins from M2D's two seeds).** Motion metrics (stride 6): relative EPE in the three bins, EPE (mm),
  CD-motion dynamic (symmetric p90), dynamic share on moving pixels, hand-velocity R² on training cameras and on the
  trajectory set: 8 per task. Guards: moving-pixel PSNR and training PSNR. RL: hammer, 6 seeds each (standard above).
  M3D is adopted if it is better beyond the margins on >= 9 of the 16 motion metric-task pairs (>= 3 per task), worse
  beyond them on none of the 4 guard pairs, and its RL mean is not below M2D's by more than the RL margin. Otherwise
  M2D stays. Only the adopted loss remains in the training code path; if M3D is adopted, 2c (DROID track arrays)
  follows with smoke tests only.

## 2026-10-09 — Item 3 (pre-registered): DataLoader prefetch factor

- **Measurement.** `scripts/bench_loader.py` runs real training steps (C configuration, hammer, batch 16) for 400 steps
  after 100 warm-up steps per setting, under the usual host load, for prefetch factor {4, 2, 1} x workers {8, 6}, in
  the order ABCDEF then FEDCBA (load drift cancels); per setting: mean step time, mean data-wait per step, and the
  proportional set size (PSS, shared memory counted once) of the whole process tree (main + workers), sampled every
  25 steps.
- **Rule.** Choose the setting with the smallest peak PSS whose mean step time is <= 1.10 x the current setting's
  (prefetch 4, 8 workers). The default becomes that setting; `ram_gb` per pretraining job = measured peak PSS x 1.25,
  rounded up; running jobs are restarted at their next checkpoint only if the saving is >= 40 % of their memory.

## 2026-10-09 — Reviewer suggestions S2-S6: decisions

- **S2 (adopted):** 6-seed hammer RL proxies for every decision that uses RL (standard above).
- **S3 (adopted for base300k):** the base300k runs are context only and compete for host memory: base300k hammer is
  stopped at its next checkpoint and both stay held. The length study is redone for the final configuration after items
  1-2 (A200 and A300 of that configuration; the rule's amendments A/B stand, RL now with 6 seeds). The synth300k runs
  keep running for now ("running jobs continue"); they are stopped if S1 or items 1-2 change C.
- **S4 (adopted):** DrM + CNN on pick-place, seed 2000, full 1M agent steps and full evaluation protocol now, to learn
  whether pick-place is solvable under this protocol. It is a Stage 4 seed if code and protocol stay unchanged.
- **S5 (kept):** anchors stay initialised from the per-task workspace mean/std, treated as a coordinate-normalisation
  statistic (any dataset provides it; a generic workspace box is a drop-in replacement). A generic-box ablation on one
  task is deferred to Stage 2 if compute allows; no evidence against it so far.
- **S6 (adopted):** gate screens at 6k are retired for decisions; future screens use the 100k full-split two-seed
  standard.
- **Update (16:56): synth300k runs stopped too.** Every item-1 variant includes the 1b depth-validity fix, so the final
  configuration cannot equal the current C and the synth300k runs could only ever be context; with host memory at
  30-36 GB available they were blocking the S1 runs. Stopped at ~70k (checkpoints kept) and held.

## 2026-10-09 — Items 1-2: implementation details fixed before any variant run

Written into the code before any D- or M3D run exists (worktree commits df1df24, 648ff5e); none of these choices can
be informed by results.

- **Occlusion loss (1a).** Per state t in {0, 1, 2} with the predicted centres at t and the GT depth of time t,
  combined over states with the temporal ramp like the render and visibility terms. A camera counts for a centre when
  the centre is in front of the near plane, inside the image and its nearest pixel (pixel centres at +0.5) has
  0 < D < far. Behind = min over counted cameras of relu(z - D - m) (0 without counted cameras); front = relu(D - z - m)
  summed over counted cameras and divided by the number of cameras where it is > 0 (free-space cameras), so one camera
  that sees through a centre is enough. All Gaussians (scene and dynamic, any opacity) are averaged; gradient reaches
  the centres (and through the states at t1/t2 the displacements), nothing else.
- **Far-plane validity (1b, `loss.depth_valid_far`).** Measured on 5 episodes per task: 1.8 % (hammer) / 2.2 %
  (pick-place) of training-camera pixels have depth >= 3 m (max 5.85 m) and were treated as valid targets the renderer
  cannot reach. The fix applies to depth L1, gradient and hard depth, coverage, and the motion weights of each pair's
  source time (and therefore the M3D track points). The evaluator's training-camera depth AbsRel uses the same mask.
- **Diagnostics (1c)** at t0 for every sample (training: every log step; evaluation: every batch): utilisation over all
  Gaussians (blending weight > 1e-3 in at least one training camera, from the gradient of a per-camera unit-feature
  render); hidden and floater fractions over opaque Gaussians (opacity > 0.3, the CD-centers threshold), with the
  counted-camera rule above (hidden needs at least one counted camera); visible-only CD-centers = opaque and utilised
  centres vs the same GT cloud as CD-centers; `train/gpu_time` = mean CUDA-event time of forward + backward + update
  per step over each log interval. Sanity on the synth encoder at 100k (hammer, 3 training batches): utilisation
  0.62, hidden fraction 0.87 (the earlier 85-87 % estimate), floater fraction 0.006, occlusion loss 0.157; the
  diagnostic costs ~8 ms per log step.
- **M3D (2a).** Track points are sampled per (sample, view, pair) with replacement; the kernel weights
  exp(-d^2 / (2 sigma^2)) inside r are detached and not normalised over neighbours (a point whose neighbours are far
  pulls less); the displacement term sums kernel x Huber over the k neighbours, times the point's motion weight, and
  averages over all counted points (motion weight > 0); attraction is Huber(1 cm) of the distance to the nearest
  dynamic centre, averaged over moving points. The image-space render still runs for the metrics (`motion_image`).
  Sanity on the same synth encoder (trained with M2D): 54-59 % of track points have a dynamic Gaussian within 3 cm,
  moving points are 42-49 mm from the nearest dynamic centre, so attraction dominates the M3D loss at the start
  (0.037-0.044 vs displacement 0.002-0.006 per pair); no measurable change in step time or GPU memory.
- **Item 2 RL proxies: which encoders get the 6 seeds.** The comparison is between loss variants, so the 6 hammer
  seeds are split over the two pretraining seeds of each variant: RL seeds 1000-1002 on pretraining seed 0 and
  1003-1005 on pretraining seed 1 (same split for M2D and M3D). The RL margin and decision quantity are as
  pre-registered (mean of last-5 training-camera success over the 6 runs; max(0.10, 2 x SE of the difference)).

## 2026-10-09 — Reading of the directive rules, fixed before any variant result (`analysis/review_rules.py`)

- **Motion metrics** (read at stride 6): relative EPE overall and per magnitude bin, EPE in mm, CD-motion (dynamic,
  symmetric p90), dynamic share on moving pixels and the hand-velocity R² probes (training cameras and trajectory
  set), as item 2 lists them. Every other metric is the mean of strides 2 and 6. All motion quantities use the pair
  0->2 (as `rel_epe_02` in every earlier decision).
- **Chamfer margins are relative:** margin = max(|seed 0 - seed 1| / their mean, 5 %), compared with the relative
  difference of the two-seed means. All other margins are absolute: max(|seed 0 - seed 1|, floor).
- **S1 tie-break** uses R's margins for the V2-vs-V1 comparison; item-1 (iii) and item-2 use D0's / M2D's margins.
- **Evaluations.** Every S1 member is evaluated with the current evaluator, including R seed 0 (`s2-screen-synth-*`
  at 100k, re-evaluated so that all six runs per task share one evaluator version; its earlier numbers are not mixed
  in). Commands: `python analysis/review_rules.py s1 | item1 | item2m3d <m2d>`.

## 2026-10-09 — Two evaluation-protocol defects found before any S1 variant result; fixed

- **Seed-1 runs used a different episode split.** `scripts/train.py` drew the train/validation split from
  `train.seed`, so the six S1 seed-1 runs (`s1v-*-seed1-*`) trained on the seed-1 split: some seed-0 validation
  episodes were in their training data, and their own validation episodes mostly have no near/trajectory held-out
  views. The pre-registered "two pretraining seeds" means initialisation, masks and data order, not the split. Fix:
  the split comes from `data.split_seed` (default 0) for every run; the six runs were stopped at 2.3k-6.5k steps (no
  checkpoint) and rerun from step 0 as `s1v-*-seed1-split0-*`; their evaluation jobs keep their names. The stopped
  runs' directories and the written `splits/{hammer,pick-place}_seed1.json` are left in place and never used.
  Earlier campaign runs all used seed 0 (the 6k gate screen with seed 1 used an explicit episode list), so nothing
  else is affected.
- **Probe fits depended on an unseeded shuffle.** The probes' training windows came from a shuffled loader without a
  generator, so evaluating the same checkpoint twice gave different R²: R seed 0 on pick-place, hand-velocity R² on the
  trajectory set 0.125 vs 0.016 (stride 2) and 0.165 vs 0.138 (stride 6); hand-position R² differed by <= 0.03; every
  rendering, motion, Chamfer and retrieval metric was identical. Fix: a fixed generator (`eval.probe_seed`, 0), so
  every evaluation fits its probes on the same training windows. Consequence for decisions already taken: the 2d
  decision (synthetic views adopted) rests on gains far larger than this noise (trajectory retrieval +0.41 on both
  tasks, position R² +0.40 / +0.23, velocity R² +0.29 / +0.24 against a +0.10 threshold and <= 0.11 observed noise);
  the velocity-R² parts of other earlier comparisons carry this noise and are not re-litigated. The two R seed-0 evaluations made today are redone with the fix, so all S1 numbers share it.

## 2026-10-09 — Context: pick-place proxies of the synth encoder at 200k (A200), 200k agent steps

Two seeds (1000/1001), last-5 training-camera success 0.015 / 0.017 (held-out 0.005 / 0.000, trajectories 0.015 /
0.010), peak 0.025 / 0.042. Pick-place stays unsolved within 200k agent steps for every encoder tried so far; it is
context only (directive S2) and the S4 CNN run to 1M (80k at 18:07) tells whether it is solvable under this protocol.

## 2026-10-09 — Item 3 decided: loader default prefetch 2 x 6 workers; pretraining ram_gb from PSS

`runs/bench-loader-synth-hammer/bench.json` (C configuration, hammer, batch 16, GPU 4 shared with 6 pretraining and
4 RL jobs; 100 warm-up + 400 timed steps per setting, order ABCDEF then FEDCBA; PSS of the whole process tree sampled
every 25 steps). Means of the two passes, peak PSS over both:

| prefetch x workers | step time (s) | vs 4x8 | data wait (s) | GPU time (s) | peak PSS (GB) | mean PSS (GB) |
|---|---|---|---|---|---|---|
| 4 x 8 (current) | 0.717 | 1.000 | 0.010 | 0.703 | 11.29 | 10.65 |
| 2 x 8 | 0.678 | 0.945 | 0.003 | 0.671 | 11.33 | 10.48 |
| 1 x 8 | 0.750 | 1.045 | 0.016 | 0.729 | 10.85 | 10.23 |
| 4 x 6 | 0.768 | 1.071 | 0.207 | 0.557 | 9.70 | 9.28 |
| 2 x 6 | 0.481 | 0.671 | 0.002 | 0.475 | 9.61 | 9.19 |
| 1 x 6 | 0.498 | 0.694 | 0.025 | 0.469 | 9.62 | 9.14 |

- **Rule applied:** every setting is within 1.10 x of 4 x 8; the smallest peak PSS is 2 x 6 (9.61 GB; 1 x 6 9.62 and
  4 x 6 9.70 are within 0.1 GB). New default `train.prefetch_factor: 2`, `train.num_workers: 6` in `base.yaml`;
  `ram_gb` for 2 x 6 pretraining jobs = ceil(9.61 x 1.25) = 13; for jobs that keep 4 x 8 (all S1 runs, for a uniform
  S1 data pipeline) ceil(11.29 x 1.25) = 15. Saving for running jobs 15 % (< 40 %): no restarts.
- **What the numbers say.** The step time is set by GPU contention (GPU time ~= step time; the GPU-time drift between
  passes, 0.84 -> 0.47 -> 0.57 s, is the shared load changing, not the loader). Prefetch depth barely matters for
  memory (<= 0.5 GB); the worker count does (~0.8 GB per worker). The 4 x 8 measurement agrees with the running S1
  jobs measured at the same time: 10.5-12.4 GB PSS per job tree, versus 25-26 GB RSS, which counts the workers'
  shared pages once per process and is what the old `ram_gb: 30` was based on. With 6 workers the loader can become
  the bottleneck when a GPU is lightly loaded (first 4 x 6 pass: 0.41 s data wait); under the current sharing it is
  not.
- **Limitation.** The benchmark has no in-training evaluation (validation and probe loaders start their own workers
  for a few minutes every 5k steps); the 25 % margin and the scheduler's 60 GB host reserve cover these transients,
  and the first 2 x 6 runs will be measured during an evaluation to confirm.

## 2026-10-09 — Context: hammer proxies of the synth encoder at 200k (A200), 400k agent steps

Seeds 1000 / 1001: last-5 training-camera success 0.470 / 0.882 (mean 0.676), held-out cameras 0.092 / 0.025,
trajectories 0.285 / 0.745, peak 0.558 / 0.917. The same encoder at 100k gave 0.388 (it3), base at 100k 0.419,
base A200 0.398, base300k at 100k 0.366 (all two-seed means). Context only (directive S2): the two seeds differ by 0.41,
larger than any difference between encoders so far, which is why RL decisions now use 6 seeds. It is consistent with
the representation metrics, which kept improving from 100k to 200k for this configuration; the length study for the
final configuration (A200 vs A300, 6-seed proxies) decides the length.

## 2026-10-10 — Fifth host-memory incident (03:16): S4 CNN pick-place run OOM-killed at 693k; held until S1 ends

- From ~02:20 other tenants grew to ~370 GB (this container: 127 GB PSS, of which the 10 S1 runs ~115 GB), the page
  cache collapsed and available memory reached 2-6 GB with 32 % full memory pressure (60 s). The kernel killed the
  S4 CNN trainer and its evaluation companion (exit 137) at agent step 693k (resume checkpoint at 650k, `checkpoint_every_steps` 50k; training-camera
  success 0.42 at 690k). No S1 run was hit.
- Before the kill the CNN run had slowed from ~15 to 2.5 agent steps/s: its 33 GB frame memmap fell out of the page
  cache and each random ~49 KB frame fault read ~1.5 MB through the 2 MB device readahead (805 MB/s). Fix (merged
  25793e0, suite 241/241): MADV_RANDOM on the CNN frame memmap; an I/O hint only, the sampled bytes are unchanged (test),
  so the run stays a Stage 4 candidate seed when it resumes with it.
- **Decision:** the CNN trainer is held (`hold-memory`) until the S1 pretraining runs reach 100k (~14:00), then
  resumes from its 650k checkpoint (the replay is returned to that point; 43k agent steps are redone). S1 is the critical path to the freeze; the CNN run is context
  (S4) and cannot finish usefully under this pressure anyway.

## 2026-10-10 — Sixth host-memory incident (04:58-05:20): three S1 seed-1 runs OOM-killed; two more stopped

- Other tenants grew again (host 400 GB used with this container at ~87-115 GB PSS; swap full; memory pressure "full"
  82 % over 5 min). The kernel killed `s1v-synth-seed1-split0-pick-place` (at 59.3k), `s1v-synthinv-seed1-split0-
  pick-place` (59.3k), `s1v-synthinv-seed1-split0-hammer` (80.6k) and the CNN evaluation companion (exit 137); the
  remaining runs crawled at ~21 s/step.
- **Action:** stopped the two slowest S1 runs (`s1v-synthsr-seed1-split0-hammer` at 57.6k and `-pick-place` at
  54.2k), freeing ~22 GB; the five seed-0 / near-100k runs recovered to 0.32-0.53 s/step within two minutes. All five
  stopped or killed runs resume from their last checkpoints (80k, 50k, 50k, 50k, 50k; 0.6k-9.3k steps redone) once
  the scheduler sees 75 GB available. Resuming restores optimizer, scheduler, RNG and the sampler position, so the
  S1 comparison is unaffected apart from time; the S1 decision moves to ~tonight.
- The CNN evaluation companion exhausted its restarts (`failed`); it is reset when the CNN trainer resumes after S1.
- **06:58 update:** pressure continued (host 415 GB used, memory pressure 91-95 % over 5 min). Two more S1 runs were
  OOM-killed: `s1v-synthsr-seed0-pick-place` at 80.3k and `s1v-synth-seed1-split0-hammer` at 82.2k (both resume from
  80k). Three S1 runs keep running (`synthinv-seed0-hammer` at 98.0k, `synthsr-seed0-hammer` at 87.2k,
  `synthinv-seed0-pick-place` at 75.5k). The other tenants hold ~380 GB; this container holds ~35 GB. The scheduler
  relaunches the seven waiting runs only when 75 GB is available, so it does not feed the OOM churn. Changing the
  resumed S1 runs to the 2 x 6 loader would save only ~2 GB each (the window loader has no randomness, so it would not
  change the data); not done, to keep the S1 pipeline uniform.

## 2026-10-10 — S1 margins, hammer (from R's two seeds only; written before any variant is compared)

R = current C (synthetic views as positives and render targets), 100k of the 200k schedule, full split. Seed 0 =
`s2-screen-synth-hammer` (re-evaluated), seed 1 = `s1v-synth-seed1-split0-hammer`.

| metric (reading) | seed 0 | seed 1 | margin |
|---|---|---|---|
| dynamic share, moving pixels (s6) | 0.884 | 0.170 | 0.714 |
| relative EPE 0->2 (s6) | 0.627 | 0.940 | 0.313 |
| trajectory retrieval (mean s2/s6) | 0.742 | 0.752 | 0.030 (floor) |
| hand-position R², trajectory (mean) | 0.911 | 0.884 | 0.050 (floor) |
| hand-velocity R², trajectory (s6) | 0.625 | 0.571 | 0.055 |
| training-camera PSNR (mean) | 25.95 | 24.94 | 1.01 dB |
| moving-pixel PSNR (mean) | 24.85 | 23.45 | 1.40 dB |
| CD-render trajectory, sym. p90 (mean) | 2.54 cm | 2.94 cm | 14.7 % (relative) |

**Observation (reference only):** R seed 1 on hammer shows the same motion collapse that R seed 0 shows on
pick-place: its dynamic share falls to 0.06 by 10k and recovers only to 0.22 (relative EPE 0.88-1.00 throughout),
while seed 0 rises to 0.91 (relative EPE 0.36). The collapse is present from 10k, long before the resume at 80k
(incident 6), so it is not a resume artifact. The collapse of C is therefore seed-dependent on both tasks, not
pick-place-specific; the S1 question ("does V1/V2 remove the collapse?") is unchanged, but R's hammer margins on the
motion and PSNR metrics are wide, so on hammer the rule's (b)/(c) checks will only flag large degradations.
Pick-place margins follow when `s1v-synth-seed1-split0-pick-place` is evaluated.

## 2026-10-10 — S4 CNN pick-place run resumed at 10:08 (from 650k, on GPU 5)

- Host memory had recovered (188 GB available, no pressure) and GPU 5 carried a single S1 run, so the CNN trainer was
  released early instead of waiting for all of S1; pinned to GPU 5 with `ram_gb` 12 (measured RSS ~10-13 GB).
  Resumed from its 650k checkpoint (events.jsonl "resumed" at 650000; replay returned to that point; with the
  MADV_RANDOM hint). Its evaluation companion was reset and restarted.
- **Record caveat:** `eval.jsonl` already holds evaluations at 660k-690k from the branch that was killed at 693k; the
  companion skips steps it has evaluated, so those four points stay from the pre-incident branch, while every point
  from 700k on comes from the resumed branch. `train.jsonl` repeats steps 651k-694k. Final success (1M) is
  unaffected; success AUC uses the records as they are, and this is noted with the run's results.

## 2026-10-10 — S1 margins, pick-place (from R's two seeds only; written before any variant is compared)

Seed 0 = `s2-screen-synth-pick-place` (re-evaluated), seed 1 = `s1v-synth-seed1-split0-pick-place`.

| metric (reading) | seed 0 | seed 1 | margin |
|---|---|---|---|
| dynamic share, moving pixels (s6) | 0.062 | 0.074 | 0.050 (floor) |
| relative EPE 0->2 (s6) | 1.000 | 1.000 | 0.030 (floor) |
| trajectory retrieval (mean s2/s6) | 0.767 | 0.758 | 0.030 (floor) |
| hand-position R², trajectory (mean) | 0.916 | 0.910 | 0.050 (floor) |
| hand-velocity R², trajectory (s6) | 0.263 | 0.291 | 0.050 (floor) |
| training-camera PSNR (mean) | 24.50 | 24.47 | 0.20 dB (floor) |
| moving-pixel PSNR (mean) | 18.18 | 18.12 | 0.20 dB (floor) |
| CD-render trajectory, sym. p90 (mean) | 2.82 cm | 3.43 cm | 19.7 % (relative) |

Both R seeds collapse on pick-place (dynamic share 0.06-0.07, relative EPE 1.00), so the pick-place margins sit at
their floors except CD-render. All 12 inputs of the S1 rule exist once `synthsr-seed1` hammer and pick-place are
evaluated; the rule is applied then, unchanged.

## 2026-10-10 — S4 result: DrM + CNN solves pick-place under this protocol, slowly

`s4-drm-cnn-pick-place-s2000`, 1M agent steps, full evaluation protocol (seed 2000). Final (1M): training cameras
0.675, held-out cameras 0.000, trajectories 0.575 (lateral 0.65, circular 0.50). Last-5 mean: training 0.645,
held-out 0.015, trajectories 0.570. Success AUC over 100 evaluation points: training 0.241, trajectories 0.192.
Training-camera success by step: 300k 0.03, 500k 0.16, 650k 0.26, 700k 0.38, 800k 0.48, 900k 0.60, 1M 0.68 (still
rising). Caveats: the run was OOM-killed at 693k and resumed from 650k (evaluations at 660k-690k are from the
pre-incident branch; `eval.jsonl` holds one record per step); the resumed part used the MADV_RANDOM replay hint
(I/O only).

**Reading (interpretation corrected at 17:17).** Pick-place is solvable under the protocol and this end-to-end CNN
learned late. That does not establish how quickly any frozen encoder could learn or rule out success in a 200k/400k
proxy. Pick-place proxies remain context only under the pre-registered S2 rule, not because of a universal inference
from this CNN run. Pick-place stays one of the 8 campaign tasks. The run is a Stage 4 CNN seed-2000 candidate if code
and protocol stay unchanged until Stage 4; its OOM-resume and mixed-curve caveats must accompany any reuse.

## 2026-10-10 — S1 decided: neither variant qualifies; C stays (synthetic views as positives and render targets)

`python analysis/review_rules.py s1` (full output with margins:
`docs/decisions/s1_rule_2026-10-10.txt`). Two-seed means, margins from R (entries above):

| check | V1 = invariance positives only | V2 = C + self-render |
|---|---|---|
| (a) pick-place: dynamic share >= 0.5 and rel. EPE (s6) better than R | dyn 0.368 (seeds 0.718 / 0.017), rel EPE 0.902 vs 1.000 (better) -> **fail** (share < 0.5) | dyn 0.051, rel EPE 1.000 -> **fail** |
| (b) retrieval, position R², velocity R² (trajectory) not worse, both tasks | pass (pick-place velocity R² better: 0.366 vs 0.277) | pass (retrieval better on both tasks: 0.803 / 0.807 vs 0.747 / 0.762; hammer velocity R² better) |
| (c) PSNR, moving PSNR, CD-render not worse, both tasks | **fail**: CD-render worse on hammer (+47 %) and pick-place (+31 %), pick-place PSNR -0.42 dB | **fail**: pick-place moving PSNR -0.35 dB |

**Decision (pre-registered fallback):** neither qualifies, so C is unchanged and the motion collapse is reported as a
limitation. Counted iteration 4 is used.

**What the seeds show** (context, not a decision input). The collapse (dynamic Gaussians lose the moving pixels by
~10k steps and never recover) is a per-seed event in every configuration: R hammer 1 of 2 seeds collapsed (dynamic
share 0.88 / 0.17), R pick-place 2 of 2 (0.06 / 0.07), V1 hammer 1 of 2 (0.83 / 0.18), V1 pick-place 1 of 2 (0.72 /
0.02), V2 hammer 0 of 2 but one partial (0.92 / 0.44), V2 pick-place 2 of 2 (0.05 / 0.05). Removing the synthetic
render targets (V1) gave the only non-collapsed pick-place seed so far, at the cost of geometry (CD-render +31-47 %);
self-rendering (V2) improves retrieval (+0.045 to +0.056) but not motion. With two seeds per variant the motion
metrics of items 1-2 will be dominated by which seeds collapse; this is why item 2's attraction term (pulls dynamic
centres onto moving track points) is the most direct test of the failure. Items 1-2 proceed on C as pre-registered.

## 2026-10-10 — NVIDIA management access failure (18:07); scheduler recovery blocked by native isolation gate

- Scheduler pid 957531 exited when `nvidia-smi --query-compute-apps` returned 255. A direct query of the two allowed
  UUIDs also failed with `Failed to initialize NVML: Unknown Error`. The cause of the host/container access failure
  is not established; no GPU reset, permission change, driver reload, or foreign-process termination was attempted.
- All 20 item-1 training jobs remain alive and advancing (14.8k-21.5k at 18:07; D0 seed-1 pick-place completed its
  20k evaluation and reached 20.2k by 18:13). No new item-1 exits/crashes appear in the registry. 4.8 TB disk free,
  107 GiB RAM available, memory pressure 13-16% at the initial check.
- The watcher reported `GPU_QUERY_FAILING consecutive=5 reason=CalledProcessError`; re-armed once as tracked task
  `bo345vngk`. New launches are held with `experiments/HOLD`; existing jobs are not restarted.
- Recovery prepared only in `.worktrees/review`: bound the GPU process query to 60 s; on query error/timeout,
  reconcile exited jobs but launch nothing and keep the daemon running. Two regression cases require no launch
  during failure and admission on a later successful query. This is operational recovery, not a method change.
- **Acceptance blocked:** the full-suite attempt (`b12xr4j29`, GPU 5 UUID) exited 1 at the GPU guard before tests
  started: `torch sees 0 CUDA devices but CUDA_VISIBLE_DEVICES lists 1`. Syntax and whitespace checks alone do not
  satisfy acceptance. The recovery patch is unmerged/uncommitted; production sources and existing test evidence
  remain unchanged. Guard failure log: `runs/setup/nvml-20261010-1812-suite-attempt.log`.
- Resume only after GPU visibility recovers and the complete native suite passes; then merge under HOLD and restart
  the scheduler. Hourly checks and the existing watcher continue. Seeds, thresholds, margins and protocols unchanged.

## 2026-10-10 — Second-server extension: placement and verification mechanics registered before remote results

- **Authorization and continuity.** The user authorized a second execution server, not a campaign stop. The atomic
  scheduler recovery was committed review-only as `cd09c8a` (correcting the previous entry's "uncommitted" status),
  then existing main documentation was merged into review. It remains unaccepted and undeployed. The 20 healthy
  item-1 trainers stay local; no fresh CUDA process or GPU reset is used to replace them.
- **Connectivity.** Container-side TCP and password-authenticated SSH to the authorized host succeeded. The bridge
  route, pinned SSH host key, remote inventory and operational boundaries are recorded in `docs/REMOTE.md`. No host
  VPN action, remote SSH-key installation, foreign process termination or foreign Docker modification is permitted.
- **Single authority and failure semantics.** The local queue/append-only registry and all decision evidence remain
  authoritative. Remote execution uses owned, detached, deterministically named Docker containers and persisted local
  launch intents. Connectivity loss means unknown, never failure/restart eligibility. A success releases dependencies
  only after required results return with verified checksums. Each detected outage is recorded with its duration.
- **Data.** Transfer, rather than regenerate, only the scheduled tasks' original HDF5, held-out sets, fixed splits and
  workspace statistics, using resumable checksum-verified transfers. Initial scope is hammer and pick-place. Preserve
  numerical container data paths. No DROID transfer or additional experiment is authorized by this extension.
- **Code/runtime.** Campaign releases must be clean commits already pushed to `origin/splatter4d`; remote edits are
  forbidden. Candidate snapshots are only for native acceptance, never campaign provenance. Use the same lock and
  audited installed gsplat/fused-ssim native payloads: cached wheels differ from production binaries and cannot be
  substituted. Record locked Git revisions, payload hashes, immutable image ID, build/runtime versions and driver.
- **Resource admission.** Remote disk floor is at least 15% of measured capacity (initially 1,129,338,552,730 bytes),
  plus reservations for declared future growth, including 46 GB CNN replay. Reserve 120 GiB host RAM and account for
  recent-launch ramp. Four authorized UUIDs have separate admission; foreign work blocks a device. Main GPU/RAM/disk
  limits remain unchanged. All remote bind sources and host writes remain below `/home/compu/kaist/sunho`.
- **Placement.** Keep item-1's existing seeds/variants together on the main host. Following item-1's unchanged decision,
  prioritize item-2's two-seed/both-task validation and both six-seed hammer RL arms on the remote. Comparison arms
  must share commit, numerical configs, data and protocol; reusing an older local arm requires actual equivalence,
  not a retroactive provenance claim. Then proceed with the registered leave-one-out, length, freeze and Stage 3/4
  sequence. Baseline pretraining may fill spare capacity at its existing lower priority; no baseline RL before freeze.
- **Gates.** The full native suite, one-UUID CUDA/EGL isolation on all four devices, data checksum identity and
  cross-host identical initial losses plus short-run agreement are required before remote campaign admission. Initial
  SSH/Docker inventory is not acceptance. Fresh main-host equivalence remains blocked by the CUDA/NVML outage;
  historical first-forward logs are candidates only until commit/config/batch identity is demonstrated.
- **Decision rules.** No seeds, schedules, losses, thresholds, margins, task selection or method ordering change.
  Extra capacity changes elapsed time only. Setup is in progress; no remote campaign result exists yet.
