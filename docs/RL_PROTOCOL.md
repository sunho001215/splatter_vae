# DrQ-v2 Meta-World RL protocol

Source of truth: the reference repository at commit `c0abf56` (`origin/Dynamic3D`):
`agents/drqv2/train_drqv2_metaworld.py`, `agents/drqv2/drqv2_metaworld.py`, `agents/drqv2/replay_buffer.py`,
`agents/common/encoders.py`, `agents/common/head.py`, `agents/drqv2/evaluate_camera_trajectory.py` and the
per-task configs `agents/drqv2/config/{cnn,splattervae,sincro,reviwo}/<task>.yaml`. The port lives in `s4d/rl/`,
`scripts/train_rl.py` and `configs/rl/`.

## Kept from the reference

| Item | Value |
|---|---|
| Environment | `gym.make("Meta-World/MT1", env_name="<task>-v3", seed=seed)`, `_freeze_rand_vec=False`, reset seed `seed + reset_count` |
| Observation | 128x128 RGB from one free camera, `frame_stack=3`, plus proprio = state obs `[0:4]` (hand xyz, gripper) |
| Cameras | the six training cameras of the pretraining rig (orbit radius 1.0, look-at (0, 0.6, 0), fovy 45); each training episode draws one uniformly |
| Action repeat / episode | 2 / 250 simulator steps (125 agent steps) |
| Training length | 1,000,000 agent steps; 4,000 uniform-random seed steps |
| Updates | every 2 agent steps, 1 gradient step, batch 256, n-step 3, discount 0.99 |
| Replay | memmap ring of 1,000,000 transitions; pixels (CNN) or fp16 cached features (frozen encoders) |
| Optimiser / nets | Adam lr 1e-4; actor and twin critic trunks Linear-LayerNorm-Tanh, MLP hidden 1024 |
| Exploration | `stddev_schedule=linear(1.0,0.1,150000)`, `stddev_clip=0.3`, target tau 0.01 |
| CNN encoder | DrQ-v2 ConvNet trained by the critic loss, random-shift augmentation (pad 4), `feature_dim=50` |
| Frozen encoders | no augmentation, features cached in replay, trainable projection head, `feature_dim=256` |
| Projection heads | fused-state encoders (splatter4d, SinCro): `SmallPostEncoderMLPHead`; per-frame encoders (ReViWo): `FrameMLPStackHead` |
| Training-camera evaluation | every 10,000 agent steps, 120 deterministic episodes split evenly over the six cameras (20 each) |
| Trajectory evaluation | `evaluate_camera_trajectory.py` defaults: base `cam1`, 72 looping poses, lateral amplitude 0.12 m, circular +/-10 deg azimuth and +/-6 deg elevation, 20 episodes |

## Additions and deviations

1. **Held-out cameras.** The four held-out cameras of the pretraining rig (`eval0..eval3`) are evaluated at every
   evaluation point, 20 episodes each. They are never used for training or method selection on the six final tasks.
2. **Trajectory evaluation at every evaluation point.** The reference ran it once on final checkpoints; it now runs
   with the other evaluations, 20 episodes per trajectory.
3. **Batched evaluation.** A pool of 10 environments steps in lockstep (seeds `seed + 1 + 7919 i`) so the frozen
   encoder runs once per step for the whole pool. The reference used one evaluation environment with seed `seed + 1`.
   Initial states differ from the reference but are identical across encoders for a given seed.
4. **Frame spacing for the splatter4d encoder.** Pretraining windows use simulator-step strides {3, 6, 9}. With
   action repeat 2, consecutive stacked frames are 2 simulator steps apart, which no pretraining window used. The RL
   side therefore stacks agent steps `t-6, t-3, t` (frame gap 3 = 6 simulator steps, the only representable
   pretraining stride and the median one). The rule is `s4d.rl.env.matching_frame_gap`; the stride list is stored in
   the encoder export. The T=1 ablation receives only the newest frame. Other encoders keep the reference gap of 1.
   Open item: SinCro and ReViWo are re-pretrained on our data in Phase E; their temporal spacing is recorded there.
5. **Crash resume.** `scripts/train_rl.py` checkpoints every 50k agent steps and resumes from `latest.pt`, reopening
   the replay buffer. Transitions written after the checkpoint stay in replay, so a resumed run is not bit-identical.
6. **Tasks without a reference config.** pick-place, peg-insert-side, shelf-place and bin-picking use the same
   protocol as the four reference tasks (door-open, hammer, peg-unplug-side, stick-push).

## Seeds

| Phase | Use | Seeds |
|---|---|---|
| D | RL proxy for method selection (hammer, pick-place only) | 1000, 1001 |
| F | main comparison, all eight tasks | 2000, 2001, 2002 |
| G | RL ablations (hammer, pick-place), paired with the Phase F full-method runs | 2000, 2001, 2002 |

Phase F seeds are fresh with respect to Phase D. Results on the six non-development tasks are never used for
method decisions.

## Reported metrics

Per run, `runs/<id>/eval.jsonl` records success and return for each training camera, each held-out camera and both
trajectories at every evaluation point. Final success is the last evaluation (1M agent steps); success AUC is the
mean success over all evaluation points. Per task: mean and 95% bootstrap CI over seeds. Aggregate: interquartile
mean (IQM) and probability of improvement over tasks and seeds (rliable-style), with training cameras and held-out
cameras reported separately.
