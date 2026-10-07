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
which is not a campaign task. All eight campaign tasks use the defaults (`configs/rl/tasks.yaml`,
`drm_overrides: {}`). Units: every step count is in agent steps (one `env.step` = action repeat 2 simulator steps),
the counter the official agent receives.

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

## Evaluation process and seeds

`train_rl.py` saves a policy snapshot (encoder adapter + actor) every 10,000 agent steps; the companion job
`eval_rl.py` evaluates each snapshot with the protocol above, stepping a pool of MuJoCo worker processes in lockstep
with one batched policy. Episode reset seeds depend only on (run seed, evaluation index), so for a given seed every
encoder sees identical initial states and camera paths.

| Stage | Use | Seeds |
|---|---|---|
| 0 | DrM pipeline sanity, CNN, hammer and shelf-place | 2000 (a final-comparison seed: the runs count as Stage 4 CNN seed 2000 only if code and protocol are unchanged afterwards, otherwise they are rerun in Stage 4) |
| 1 | our encoder, development tasks (hammer, pick-place), including improvement iterations | 1000, 1001, 1002 |
| 2 | RL ablations (hammer, pick-place) | 1000, 1001, 1002 |
| 3, 4 | final comparison, all eight tasks, ours and baselines | 2000, 2001, 2002 |

Stage 3/4 seeds are fresh with respect to development; results on the six non-development tasks are never used for
method decisions. The Stage 0 CNN runs validate the RL pipeline only; they never inform method decisions.

## Reported metrics

`runs/<id>/eval.jsonl` records success and return for each training camera, each held-out camera and both
trajectories at every evaluation point; `train.jsonl` records dormant ratios, noise, lambda and losses. Final success
is the last evaluation (1M agent steps); success AUC is the mean success over all evaluation points. Per task: mean
and 95% bootstrap CI over seeds. Aggregate: interquartile mean (IQM) and probability of improvement over tasks and
seeds (rliable-style), with training cameras, held-out cameras and trajectories reported separately.

## Measurements

Filled in from Phase C timing runs.
