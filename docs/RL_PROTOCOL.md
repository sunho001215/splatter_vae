# DrQ-v2 Meta-World RL protocol

Source of truth: the reference repository at commit `c0abf56` (`origin/Dynamic3D`):
`agents/drqv2/train_drqv2_metaworld.py`, `agents/drqv2/drqv2_metaworld.py`, `agents/drqv2/replay_buffer.py`,
`agents/common/encoders.py`, `agents/common/head.py`, `agents/drqv2/evaluate_camera_trajectory.py` and the
per-task configs `agents/drqv2/config/{cnn,splattervae,sincro,reviwo}/<task>.yaml`. The port lives in `s4d/rl/`,
`scripts/train_rl.py`, `scripts/eval_rl.py` and `configs/rl/`.

## Kept from the reference

| Item | Value |
|---|---|
| Environment | `gym.make("Meta-World/MT1", env_name="<task>-v3", seed=seed)`, `_freeze_rand_vec=False`, training reset seed `seed + reset_count` |
| Observation | 128x128 RGB from one free camera, `frame_stack=3`, plus proprio = state obs `[0:4]` (hand xyz, gripper) |
| Frame spacing | consecutive agent steps, i.e. 2 simulator steps apart (action repeat 2), for **every** method |
| Training cameras | the six training cameras of the pretraining rig (orbit radius 1.0, look-at (0, 0.6, 0), fovy 45); each training episode draws one uniformly |
| Action repeat / episode | 2 / 250 simulator steps (125 agent steps) |
| Training length | 1,000,000 agent steps; 4,000 uniform-random seed steps |
| Updates | every 2 agent steps, 1 gradient step, batch 256, n-step 3, discount 0.99, continuation flag 0 at the time limit |
| Replay capacity | 1,000,000 transitions |
| Optimiser / nets | Adam lr 1e-4; actor and twin critic trunks Linear-LayerNorm-Tanh, MLP hidden 1024, target tau 0.01, `stddev_clip=0.3` |
| CNN encoder | DrQ-v2 ConvNet trained by the critic loss, random-shift augmentation (pad 4), `feature_dim=50` |
| Frozen encoders | **no image augmentation** (reference `augment_pixels` defaults to `backbone_trainable`, i.e. false for frozen encoders), features computed once per environment step, trainable projection head, `feature_dim=256` |
| Projection heads | fused-state encoders (splatter4d, SinCro): `SmallPostEncoderMLPHead`; per-frame encoders (ReViWo): `FrameMLPStackHead` |
| Training-camera evaluation | every 10,000 agent steps, 120 deterministic episodes, 20 on each of the six training cameras |
| Trajectory evaluation | `evaluate_camera_trajectory.py` defaults: base `cam1`, 72 looping poses, lateral amplitude 0.12 m, circular +/-10 deg azimuth and +/-6 deg elevation, 20 episodes per trajectory |

## Exploration schedule per task (deviation from the reference, user decision)

The schedule unit is the reference unit, agent steps: the training-loop counter of `train_drqv2_metaworld.py`
(lines 504-524) is passed to `schedule(stddev_schedule, step)`, and one iteration is one `env.step`, i.e. two
simulator steps. Difficulty categories are from Seo et al., *Masked World Models for Visual Control*
(CoRL 2022, arXiv:2206.14244), Appendix F, verified from the paper text. The reference used
`linear(1.0,0.1,150000)` for every task. Every method uses the same per-task schedule.

| Task | MWM category | stddev schedule (agent steps) | Reference value |
|---|---|---|---|
| door-open | easy | `linear(1.0,0.1,100000)` | `linear(1.0,0.1,150000)` |
| peg-unplug-side | easy | `linear(1.0,0.1,100000)` | `linear(1.0,0.1,150000)` |
| hammer | medium | `linear(1.0,0.1,250000)` | `linear(1.0,0.1,150000)` |
| peg-insert-side | medium | `linear(1.0,0.1,250000)` | not configured |
| bin-picking | medium | `linear(1.0,0.1,250000)` | not configured |
| pick-place | hard | `linear(1.0,0.1,500000)` | not configured |
| stick-push | very hard | `linear(1.0,0.1,500000)` | `linear(1.0,0.1,150000)` |
| shelf-place | very hard | `linear(1.0,0.1,500000)` | not configured |

Source of the table: `configs/rl/tasks.yaml`; resolution: `s4d.rl.protocol.resolve_config`; test:
`tests/test_rl.py::test_every_task_resolves_to_its_difficulty_schedule_for_every_method`.

## Additions and deviations

1. **Held-out cameras.** The four held-out cameras of the pretraining rig (`eval0..eval3`) are evaluated at every
   evaluation point, 20 episodes each. They are never used for training or for method selection.
2. **Trajectory evaluation at every evaluation point.** The reference ran it once on final checkpoints.
3. **Evaluation episodes per point:** 120 on training cameras (reference) + 80 on held-out cameras + 40 on
   trajectories = 240. The training-camera part keeps the reference count; the additions use the reference
   trajectory default of 20 episodes per group.
4. **Separate evaluation process.** `train_rl.py` saves a policy snapshot (encoder adapter + actor) every 10,000
   agent steps; the companion job `eval_rl.py` evaluates every snapshot with the identical protocol, so evaluation
   never slows training. It steps a pool of MuJoCo worker processes in lockstep and batches the policy.
5. **Evaluation seeds.** Episode reset seeds depend only on (run seed, evaluation index), so for a given seed every
   encoder is evaluated on identical initial states and camera paths. The reference used one evaluation
   environment with seed `seed + 1` whose reset counter continued across evaluation points.
6. **Pretraining frame spacing (user decision).** Encoders are pretrained with strides {2, 4, 6} simulator steps,
   uniform per sample, so the RL spacing of 2 is inside the pretraining distribution. `train_rl.py` refuses a
   splatter4d export whose recorded strides exclude 2. The T=1 ablation encoder receives only the newest frame.
7. **Crash resume.** `train_rl.py` checkpoints every 50,000 agent steps and resumes from `latest.pt`, returning the
   replay buffer to its checkpointed contents (see below).
8. **Tasks without a reference config.** peg-insert-side, bin-picking, pick-place and shelf-place use the same
   protocol as the four reference tasks (door-open, hammer, peg-unplug-side, stick-push).

## Replay design per method type

| Method type | Stored per environment step | Backing | Stacks |
|---|---|---|---|
| Frozen encoders (splatter4d, SinCro, ReViWo) | fp16 latent computed once, plus proprio, action, reward, continuation | in-process RAM arrays; snapshotted to `runs/<id>/replay/ram_snapshot.npz` at checkpoints | splatter4d and SinCro store one stack-level latent per state; per-frame encoders store one per-frame latent and rebuild the stack by index |
| Pixel encoder (CNN) | one uint8 128x128 RGB frame, plus proprio, action, reward, continuation | preallocated memmap under `runs/<id>/replay/`, read by DataLoader workers | rebuilt by index; no stacked duplicates |

Sampling follows the reference rules (frames before the episode start repeat the first frame; no sample crosses an
episode boundary or an overwritten ring slot) and is vectorised per batch. Equivalence of latent replay with
encoding the stored frames: `tests/test_rl.py::test_frozen_latent_replay_update_equals_encoding_stored_frames`.
Measured RAM, disk and throughput: see "Measurements" (Phase C).

## Seeds

| Stage | Use | Seeds |
|---|---|---|
| 0 | CNN pipeline sanity (hammer) | 1000 |
| 1 | our encoder, development tasks (hammer, pick-place), including improvement iterations | 1000, 1001, 1002 |
| 2 | RL ablations (hammer, pick-place) | 1000, 1001, 1002 |
| 3, 4 | final comparison, all eight tasks, ours and baselines | 2000, 2001, 2002 |

Stage 3/4 seeds are fresh with respect to development. Results on the six non-development tasks are never used for
method decisions.

## Reported metrics

`runs/<id>/eval.jsonl` records success and return for each training camera, each held-out camera and both
trajectories at every evaluation point. Final success is the last evaluation (1M agent steps); success AUC is the
mean success over all evaluation points. Per task: mean and 95% bootstrap CI over seeds. Aggregate: interquartile
mean (IQM) and probability of improvement over tasks and seeds (rliable-style), with training cameras, held-out
cameras and trajectories reported separately.

## Measurements

Filled in from Phase C timing runs.
