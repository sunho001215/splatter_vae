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
