# RL protocol: DrM on Meta-World

Every RL run in this campaign uses **DrM** (Xu et al., *DrM: Mastering Visual Reinforcement Learning through
Dormant Ratio Minimization*, ICLR 2024, arXiv:2310.19668). Algorithms are never mixed. DrQ-v2 is no longer used.

- **Algorithm source:** official code, github.com/XuGW-Kevin/DrM, pinned commit
  `989732d68d3eed986ddfc9809909a3c3171ca049`: `agents/drm_mw.py`, `utils.py`, `train_mw.py`, `replay_buffer.py`,
  `cfgs/config.yaml`, `cfgs/task/metaworld.yaml`, `cfgs/agent/drm_mw.yaml`. MIT license.
- **Environment, observation and evaluation source:** the reference repository at commit `c0abf56`
  (`origin/Dynamic3D`), so every encoder is compared on the same shared protocol.
- **Port:** `s4d/rl/agent.py` (DrM), `s4d/rl/{encoders,env,replay,evaluate,vecenv,protocol}.py`,
  `scripts/train_rl.py`, `scripts/eval_rl.py`, `configs/rl/`.

## DrM components (official Meta-World agent)

| Component | Implementation (official) |
|---|---|
| Networks | actor: trunk Linear-LayerNorm-Tanh, MLP 1024-1024, tanh mean, truncated normal; twin critic: trunk, then Linear-Dropout(0.01)-LayerNorm-ReLU x2, Linear; value network: trunk, MLP 1024-1024-1 |
| Dormant ratio | `utils.cal_dormant_ratio`: over all Linear layers of the actor (including the output layer), a unit is dormant if its batch-mean absolute output is below 0.025 x the layer's mean; computed every update on the encoded batch |
| Perturbation | every 100,000 agent steps (`step % dormant_perturb_interval == 0` inside `update`), `utils.perturb` with factor `min(max(0.2, 1 - 2 x dormant_ratio), 0.95)`: parameters of Linear layers become `factor x old + (1 - factor) x fresh orthogonal init`; other parameters kept; optimizer state reset. Applied to actor, critic, critic target (with a fresh init of its own) and value network, and to the encoder |
| Exploration | `stddev_type: awake`: noise `sigmoid(10 x (dormant_ratio - 0.2))` until the dormant ratio first falls below 0.2 (the awakening step), then `max(that, linear(1.0,0.1,500000)(step - awaken_step))`; uniform actions for the first 2,000 agent steps; target-policy noise clip 0.3 |
| Exploitation | value network trained by expectile regression (expectile 0.9) towards `min(Q1, Q2)(s, a)` on replay actions; critic target `r + gamma^n x [lambda x V(s') + (1 - lambda) x min(Q'1, Q'2)(s', a')]` with constant `lambda = 0.5` |
| Optimisation | Adam lr 1e-4 for encoder, actor, critic and value network; target tau 0.01; batch 256; one update every 2 agent steps after 2,000 seed agent steps; hidden 1024 |
| Returns | n-step 10, discount 0.97 (see below) |
| Augmentation | random shifts, pad 4, on every update (CNN only, see deviations) |

### Values actually applied by the official code, and where they differ from the paper

The official Meta-World configuration is `config.yaml` + `task/metaworld.yaml` + `agent/drm_mw.yaml`. Following
the user's instruction, the code is followed where it disagrees with the paper.

| Hyperparameter | Official code (used here) | Paper, Table 1 |
|---|---|---|
| n-step | 10: `train_mw.py` builds the loader with `floor(nstep 3 + nstep_alpha 7)`; the annealing in `update_buffer` is never called | 3 |
| Discount | 0.97 = `0.997 - discount_alpha 0.02 - discount_beta 0.007`, fixed for the same reason | 0.99 |
| Exploration schedule | `linear(1.0,0.1,500000)` (`task/metaworld.yaml`) | `linear(1.0,0.1,300000)` |
| Maximum perturb factor | 0.95 (`agent/drm_mw.yaml`) | 0.9 |
| Exploitation lambda | constant 0.5 (`agent/drm_mw.yaml`); the dormant-ratio-dependent lambda exists only in the DMC agent `agents/drm.py` | dormant-ratio dependent, target 0.6, temperature 0.02 |
| Continuation at the time limit | 1.0 (`metaworld_env.py` always emits discount 1.0) | not stated |
| Perturb interval | 100,000 agent steps | 200,000 frames (same) |

Per-task settings: the official code overrides only Meta-World `sweep-into` (max perturb factor 0.9, lambda 0.6),
which is not a campaign task (it was screened as a reserve task). All eight campaign tasks use the defaults (`configs/rl/tasks.yaml`,
`drm_overrides: {}`). Units: every step count is in agent steps (one `env.step` = action repeat 2 simulator steps),
the counter the official agent receives.

## Campaign tasks and the shelf-place replacement (review item 3, 2026-10-08)

The eight campaign tasks are door-open, peg-unplug-side, hammer, peg-insert-side, bin-picking, pick-place, stick-push
and **coffee-push** (MWM Appendix F: medium), which replaces shelf-place for Stages 3 and 4.

- **Why shelf-place is replaced.** Under this protocol its v3 reward is zero until the block is lifted: random-policy
  return 0, expert return ~1180, and DrM + CNN (Stage 0, 1M agent steps) never received a non-zero reward, so it
  measures exploration luck rather than representation quality.
- **How coffee-push was chosen** (pre-registered in `docs/EXPERIMENT_LOG.md`, items 3a-3c). Seven reserve tasks were
  screened (random and scripted-expert returns, expert success >= 80 %, object visible in >= 4 of 6 training cameras);
  push-back failed (expert 0.64). DrM + CNN (seed 2000, 300k agent steps, light evaluation) then ran in ranking order:
  plate-slide 0.89 and drawer-open 1.00 at 300k (too easy), assembly 0.00 (no success), coffee-push 0.18 (range
  0.10-0.42 over the run): dense reward (random return 10.6 > 0), success above 0 and below ~0.7, and the mug is pushed
  0.22 m across the table in view of all six training cameras.
- **Appendix note: sparse-reward tasks.** shelf-place is kept out of the comparison because a zero reward until the
  sub-goal is reached makes outcomes depend on whether exploration ever reaches it, not on the observation encoder.
  Its Stage 0 run (DrM + CNN, seed 2000) and its data remain available.

## Shared protocol (reference repository)

| Item | Value |
|---|---|
| Environment | `gym.make("Meta-World/MT1", env_name="<task>-v3", seed=seed)`, `_freeze_rand_vec=False`, training reset seed `seed + reset_count` |
| Observation | 128x128 RGB from one free camera, 3 frames one agent step (2 simulator steps) apart, for every method, plus proprio = state obs `[0:4]` (hand xyz, gripper) |
| Frame stack at episode start | earlier frames repeat the first frame |
| Training cameras | the six training cameras of the pretraining rig; each training episode draws one uniformly |
| Episode | 250 simulator steps (125 agent steps), action repeat 2 |
| Training length | 1,000,000 agent steps (= 2M frames, the x-axis of the paper's Meta-World figure; the official run length 2.1M frames only adds a margin past the last evaluation) |
| Replay capacity | 1,000,000 transitions |
| Evaluation | every 10,000 agent steps, deterministic policy (actor mean): 120 episodes on the six training cameras (20 each, reference), 20 on each of the four held-out cameras, 20 on each reference camera trajectory (lateral: base cam1, 72 looping poses, 0.12 m; circular: +/-10 deg azimuth, +/-6 deg elevation); 240 episodes per point |

## Deviations from the official DrM code, with reasons

1. **Shared environment and evaluation instead of the official DrM env.** The official code uses the
   `corner2` camera at 84x84, 250 *agent*-step episodes, a zero-padded frame stack at reset, no proprio, and 10
   evaluation episodes every 10k frames. The campaign compares encoders on the reference protocol above, so these
   environment-side choices follow the reference for every method.
2. **Proprio input.** Following the shared observation, the actor, critic and value trunks take
   `[representation, proprio]` (the reference repository's interface). The dormant ratio is computed on the
   actor with the same inputs.
3. **Frozen pretrained encoders (ours, SinCro, ReViWo).** The official agent trains its CNN end to end. Frozen
   encoders follow the reference repository: features are computed once per environment step and cached in RAM
   replay, a trainable projection head (`feature_dim` 256) maps them to the actor/critic/value trunks, and no image
   augmentation is applied (the reference applies augmentation only to trainable pixel encoders).
4. **Perturbation scope for frozen encoders.** Only the actor, critic, critic target and value network are
   perturbed; the frozen encoder is never touched and stays bit-identical
   (`tests/test_rl.py::test_end_to_end_updates_with_perturbation_for_every_encoder_type`). The official
   `perturb(encoder, encoder_opt)` changes no encoder weights, because the official CNN encoder has no Linear layer,
   and only resets the encoder optimizer state. The trainable projection head of a frozen encoder is treated the same
   way: its optimizer state is reset and its weights are kept.
5. **CNN encoder.** Follows the official code exactly, including `perturb(encoder, encoder_opt)` (optimizer reset,
   weights unchanged) and augmentation of every update batch. At 128x128 the representation is 32x57x57.
6. **Replay layout.** The official buffer stores whole episodes of stacked frames as `.npz` files. Here the CNN replay
   stores each frame once in a disk memmap and rebuilds stacks by index; frozen encoders store fp16 latents in RAM.
   Sampling is uniform over valid start states of the last 1M transitions (the official sampler draws an episode
   uniformly, then a step), and n-step windows never cross an episode boundary in either implementation.
7. **Uniform-random warm-up.** As in the official agent, actions are uniform for the first 2,000 agent steps
   (`num_expl_steps`) and updates start at 2,000 agent steps (`num_seed_frames` 4,000 / action repeat 2).
8. **Logging.** In addition to the official actor dormant ratio, the critic dormant ratio is computed (logging only),
   together with the current exploration noise, the exploitation coefficient lambda, the awakening flag and every
   perturbation event (`runs/<id>/events.jsonl`, W&B `train/perturb_*`).

## Replay design per method type

| Method type | Stored per environment step | Backing | Stacks |
|---|---|---|---|
| Frozen encoders (splatter4d, SinCro, ReViWo) | fp16 latent computed once, plus proprio, action, reward, continuation | in-process RAM arrays; snapshotted to `runs/<id>/replay/ram_snapshot.npz` at checkpoints | splatter4d and SinCro store one stack-level latent per state; per-frame encoders store one per-frame latent and rebuild the stack by index |
| Pixel encoder (CNN) | one uint8 128x128 RGB frame, plus proprio, action, reward, continuation | preallocated memmap under `runs/<id>/replay/`, read by DataLoader workers | rebuilt by index; no stacked duplicates |

Equivalence of latent replay with encoding the stored frames:
`tests/test_rl.py::test_frozen_latent_replay_update_equals_encoding_stored_frames`. Measured RAM, disk and
throughput: see "Measurements".

## Pretraining frame spacing

Encoders are pretrained with strides {2, 4, 6} simulator steps, uniform per sample, so the RL spacing of 2 is inside
the pretraining distribution. `train_rl.py` refuses a splatter4d export whose recorded strides exclude 2. The T=1
ablation encoder receives only the newest frame.

## Pretraining length of the frozen encoders

| Encoder | Pretraining steps | Notes |
|---|---|---|
| splatter4d (ours) | **200 000**, selected 2026-10-13 by the registered C + D0 + M3D A200/A300 rule (5/40 wins, hammer RL improvement below the six-seed margin) | warmup 10k + cosine decay to `train.steps=200000`; fixed for all 8 tasks and every ours ablation. Explicit override supersedes the historical 300k base default; evidence in `docs/RESULTS.md` and `runs/setup/length-verification-20261013.json`. |
| SinCro | exactly 300 000 on every task | user decision; reference 500 001; other reference hyperparameters unchanged (`docs/BASELINES.md`, deviation 9) |
| ReViWo | 100 001 (reference) | reference hyperparameters unchanged |

The DrM agent and the RL protocol do not depend on the pretraining length; the export records the step it came from.

## Evaluation process and seeds

`train_rl.py` saves a policy snapshot (encoder adapter + actor) every 10,000 agent steps; the companion job
`eval_rl.py` evaluates each snapshot with the protocol above, stepping a pool of MuJoCo worker processes in lockstep
with one batched policy. Episode reset seeds depend only on (run seed, evaluation index), so for a given seed every
encoder sees identical initial states and camera paths.

| Stage | Use | Seeds |
|---|---|---|
| 0 | DrM pipeline sanity, CNN, hammer and shelf-place | 2000 (a final-comparison seed: the runs count as Stage 4 CNN seed 2000 only if code and protocol are unchanged afterwards, otherwise they are rerun in Stage 4) |
| 1 | our encoder, development tasks (hammer, pick-place), including improvement iterations | 1000, 1001, 1002 (decision proxies: 1000-1005 on hammer) |
| 2 | RL ablations (hammer, pick-place) | 1000, 1001, 1002 |
| 3, 4 | final comparison, all eight tasks, ours and baselines | 2000, 2001, 2002 |

Stage 3/4 seeds are fresh with respect to development; results on the six non-development tasks are never used for
method decisions. The Stage 0 CNN runs validate the RL pipeline only; they never inform method decisions.

## Development RL proxies (user directives of 2026-10-09)

Every method decision that uses RL compares encoders with 6 hammer seeds (1000-1005) at 400k agent steps under the
full evaluation protocol (S2). Decision quantity: the mean over seeds of the last-5 training-camera success; margin
max(0.10, 2 x standard error of the difference of means). When the compared objects are loss variants trained with two
pretraining seeds, RL seeds 1000-1002 use pretraining seed 0 and 1003-1005 pretraining seed 1. Two-seed proxies and
pick-place proxies are context only. The pick-place DrM + CNN run with seed 2000 to 1M agent steps (S4) tells whether
pick-place is solvable under this protocol; it counts as a Stage 4 seed only if code and protocol stay unchanged.

## Reported metrics

`runs/<id>/eval.jsonl` (or a remote companion's `runs/<id>-eval/eval.jsonl`) records success and return for each
training camera, each held-out camera and both trajectories at every evaluation point; `train.jsonl` records dormant
ratios, noise, lambda and losses. **Report both success and return as last-5, final and AUC** (user, 2026-10-15):
last-5 is the arithmetic mean of the final five evaluations, final is the 1M evaluation, and AUC is the arithmetic
mean of the 100 equally spaced evaluations from 10k through 1M. Report the six training cameras, four held-out
cameras and two moving-camera trajectories separately; condition-level return is the mean of its camera/path
returns. Per task: per-seed values, mean and 95% bootstrap CI. Aggregate success: interquartile mean (IQM) and
probability of improvement over tasks and seeds (rliable-style), with 10,000 stratified seed-within-task bootstrap
samples (RNG seed 0). Returns have task-dependent scales: report per task, not a raw-return cross-task IQM.
`runs/setup/tools/rl_reporting.py` produces the complete per-run/per-task report.

## Measurements

Replay and throughput (GPU 4/5, RTX PRO 6000; details in `docs/COMPUTE_PLAN.md`):

| Item | CNN (disk memmap frames) | Frozen splatter4d (RAM fp16 latents) |
|---|---|---|
| Replay size at 1M transitions | 46 GB on disk (one 3x128x128 uint8 frame per state, sparse until written) | 0.54 GB in RAM (256-d fp16 latent + proprio/action/reward) |
| Process RAM | 3.2 GB (plus page cache of the memmap) | 2.2 GB |
| Batch of 256 stacks | 15.5 ms gather into reused pinned buffers in a prefetch thread, hidden by the update | < 1 ms (numpy gather of latents) |
| Update | 30 ms (GPU-bound conv backward) | ~10-20 ms (latency-bound small kernels) |
| Agent steps per second | 52 alone; 80 in total per GPU (the GPU saturates) | 39 alone; ~190 in total per GPU with 8 runs |

Design choices backed by these numbers:
- **Disk replay sampling uses a prefetch thread, not DataLoader workers.** With DataLoader workers the batch cost was
  161 ms (2 workers), 66 ms (6 workers) and 105 ms (12 workers): every batch allocates and page-faults fresh 75 MB
  shared/pinned buffers, and the pin-memory copy is serial. Gathering with multi-threaded `index_select` straight
  from the memmap into reused pinned buffers costs 15.5 ms and overlaps the GPU update, so sampling no longer
  appears in the update time (31.8 ms with sampling vs 30.1 ms without).
- **Frozen encoders sample in-process from RAM.** Latents are tiny, so no worker processes are needed, and the
  latent replay of a 1M-step run fits in 0.54 GB.
- **Updates synchronise only what DrM needs** (the actor dormant ratio, used by the exploration rule, and
  perturbation events); diagnostics such as losses and the critic dormant ratio are converted only at logging steps.
  This cut the CNN update from 48 ms to 30 ms without changing any computation.
- **Evaluation pools share one renderer per worker.** Each MuJoCo renderer holds its own copy of the scene textures
  (~0.44 GB); sharing it across a worker's environments is pixel-identical up to the renderer's own +/-1 intensity
  noise on all eight tasks. Pools (6 workers x 8 envs) are opened only while snapshots are pending.

## Vision-only variant (2026-10-23 campaign)
`env.proprio_indices=[]`: the environment returns an empty proprio vector, so the actor, both critics, the target
critic, the value network and the dormant-ratio passes receive only the visual representation (their trunks take
`repr_dim` inputs instead of `repr_dim + 4`). The simulator state is still used internally for rewards, termination and
success. Everything else is the standard protocol: the same RGB history (3 frames, 2 simulator steps apart), rewards,
episode length, training-camera sampling, evaluation schedule and camera groups; frozen encoders stay frozen and the CNN
keeps its random-shift augmentation. The replay stores the zero-width proprio field as an empty in-memory array (a
zero-size field cannot be memory-mapped). Runs carry the W&B tags `vision_only`, `no_proprio` (`wandb.tags`).
Tests: `tests/test_vision_only.py` (no proprio inputs; changing the simulator state with fixed images cannot change
actions, Q or V values, while the with-proprio control does; updates, dormant ratios and replay resume; resolved configs
differ only in `env.proprio_indices` and the tags).

## Camera suite v2 (evaluation only, user directive 2026-10-29)

The original held-out cameras eval0..3 often put the robot's proximal links between the camera and the workspace or
lose task-relevant parts of the scene (measured with the tool below: eval0's arm hides on average 57% of the object
footprint and the gripper is fully hidden in up to 50% of frames; eval3 never sees stick-push's goal and loses
coffee-push's mug in 48% of frames; eval2 loses window-open's goal in 79%; eval1 loses stick-push's goal in 67%;
`runs/setup/camera-suite-v2-definition-20261029.md`). They, the six training cameras, the
training data, pretraining and RL training are unchanged; eval0..3 stay in `eval.jsonl` and are reported separately
as an **occlusion-heavy stress test**. Camera suite v2 is a separate, versioned evaluation:

- **Held-out cameras `heldout_v2` = vis0..vis3** (`s4d/rl/camera_suites.py`), on the training orbit (look-at
  (0, 0.6, 0), radius 1 m, fovy 45 deg), one common rig for all twelve tasks:

  | camera | tilt | azimuth | nearest training camera (= multi-view slot it replaces) |
  |---|---|---|---|
  | vis0 | 55 | 140 | train5, 26.0 deg |
  | vis1 | 25 | 200 | train1, 25.7 deg |
  | vis2 | 40 | 300 | train2, 27.3 deg |
  | vis3 | 25 | 340 | train0, 25.7 deg |

  The cameras are 36-117 deg apart; from the old held-out cameras vis0 is 20 deg (eval1) and vis2 15 deg (eval2)
  away, the others at least 42 deg. Three registered task-specific exceptions (the common camera falls outside the
  training rig's worst case on that task; numbers in the definition file), applied identically to every method:
  door-open uses (25, 120) for vis1 (gripper hidden in 27% of frames) and (50, 80) for vis2 (object 24% / gripper 22%
  hidden); window-open uses (40, 260) for vis3 (goal location hidden in 49% of frames).
- **Selection, before any policy evaluation and without any return, reconstruction or representation score.** The
  rule `runs/setup/camera-suite-v2-rule-20261029.json` was registered (commit 74372da) before any candidate was
  measured. `scripts/camera_visibility.py` renders representative trajectories (per task 10 scripted-expert episodes
  at the collection noise levels and 5 uniform-random episodes, fixed seeds, RL timing) and measures per frame, from
  MuJoCo segmentation and depth, the visible share of the manipulated object(s), the gripper and the goal location,
  each also in a diagnostic render without the robot's proximal links (arm occlusion), and the table workspace's
  in-image and unoccluded fractions (`s4d/rl/visibility.py`). Diagnostic renders are analysis-only: policies always
  receive the normal scene. Candidates (tilt 25-65 deg x azimuth 0-350 deg, >= 25 deg from every training camera)
  were first required to match the least observable training camera on every criterion of every task (registered
  rule; 0/138 eligible), then its per-task score (amendment A1; 0/138). Amendment A2 (registered before its outcome
  was computed) chooses the 4 admissible cameras, pairwise >= 30 deg apart, that maximise the smallest worst-task
  visibility score, with task exceptions only outside the training rig's global worst case; confirmed on every frame.
  As evaluated, every (vis camera, task) pair is within that envelope (worst-task scores 0.66-0.91 vs 0.62-0.92 for
  train0..5 and -0.38-0.53 for eval0..3). Frozen definition, rule history, numbers and hashes:
  `runs/setup/camera-suite-v2-definition-20261029.{json,md}`.
- **Stronger moving cameras** `traj_{lateral,circular}_{2x,3x}`: the reference paths (base train1, 72-pose period,
  starting pose 0 at reset) with lateral +/-0.24 / +/-0.36 m and circular +/-20/12 / +/-30/18 deg (azimuth/tilt),
  validated with the same measurement on every frame: camera >= 0.165 m outside all geometry, no clipping, >= 96% of
  the workspace in view at every frame and never wholly occluded, hidden-frame shares within 0.04 of train1. The
  requested levels are frozen (amendment A3: the per-target hidden-run limit added to the rule is reported, not a
  gate, because it also fails the unchanged 1x circular path and is not monotonic in amplitude; the literal 10%
  reductions are recorded as a non-adopted alternative). The period is fixed, so a larger amplitude is also a faster
  camera: these are combined range-and-speed stress tests (3x: up to 0.81 m/s and 66 deg/s lateral, 1.63 m/s and
  105 deg/s circular; one render per 25 ms agent step). Ranges, speeds per agent and simulator step, period, phase
  and timing are in the definition file.
- **Evaluation.** `scripts/eval_rl.py --suites v2` appends to a separate `eval_cameras_v2.jsonl` (never `eval.jsonl`)
  with groups `vis0..3`, `traj_*_2x/3x` and, by default, matched references re-run in the same pass on the same
  seeds: `ref_train0..5` and `ref_traj_{lateral,circular}` (`--v2-reference none|train|train+traj`). 20 episodes per
  group (the historical per-camera count). Seeds are paired: one list per (run seed, evaluation index) from
  `default_rng([run_seed, eval_index, 7, 20261029])`, used by every v2 group, so every method with the same run seed
  and every camera condition share initial states; the v1 seed stream is untouched. Multi-view methods keep the
  registered replacement rule (vis* replace their nearest training camera's slot, trajectories replace train1's) and
  receive per-frame K and c2w of the pose actually rendered. `--snapshots last5` re-evaluates a finished run's
  960k..1M snapshots (final = 1M; last-5 = mean of the five) and exits; `--snapshots final` only 1M.
- **Cost.** With the default references a v2 point is 320 episodes (7 waves of the default 48-environment pool, vs 5
  for v1): about 1.4x a v1 evaluation, i.e. ~60 s per snapshot for a frozen-encoder DrM policy (csw4: v1 median 42 s
  locally) and ~45 s for DrM + CNN (v1 median 31 s on the remote hosts); `--v2-reference none` (160 episodes, 4
  waves) costs ~0.8x. Last-5 re-evaluation of one run: 5 snapshots, about 5-6 minutes including pool start-up.
- **Analysis-only filtering.** The visibility tool's near-plane check takes the minimum depth after a 3x3 median
  (MuJoCo's depth buffer contains isolated single-pixel spikes, identical in `mujoco.Renderer`); this never touches
  policy observations or evaluation renders.
- **Reporting.** Per condition: `heldout_v2_*` (mean over vis0..3), `traj_2x_*`, `traj_3x_*` (lateral and circular
  means), with `ref_train_cameras_*` and `ref_trajectories_*` as matched references; success and return, final and
  last-5. Never average or pool v1 and v2 numbers.
