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
