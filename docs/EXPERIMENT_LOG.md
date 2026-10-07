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
