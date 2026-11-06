# Real-world offline FQL on frozen baseline representations and end-to-end

Branches `realworld-fql-20261106` (agent FQL, design and implementation) and `rwfql-decisions-20261106` (agent FQL2:
the user's decisions of 2026-11-06 ~11:20). Design and implementation only: no training was run, nothing was
registered or launched, and nothing here talks to the robot. The frozen encoders come from the real-world baseline
pretrainings (`docs/realworld_baselines.md`); one condition trains its CNN end-to-end with FQL.

## User decisions 2026-11-06 ~11:20 KST (implemented; not open any more)

| Item | Decision | Where |
|---|---|---|
| Outcomes | every uploaded episode is a successful demonstration; success-state proxy = the final observation; endings with a reset / reversal of the completed task are FLAGGED, flags never change the label | `s4d/rwlearn/outcomes.py`, `outcomes.json`, `docs/realworld_ending_flags.json` |
| `action/flange_pose` | the pose command actually SENT; keep the command-based base-frame delta-pose contract (no measured displacement) | `s4d/rwlearn/actions.py`, `tests/test_rwlearn_actions.py::test_user_decision_contract_*` |
| Gripper | continuous closure command, 2c - 1 (no binarisation) | same |
| FQL alpha | 10 with the reference configuration for every condition; NO alpha sweep. Later tuning follows a predefined real-robot evaluation protocol, not offline action error (the offline action errors are logged as diagnostics only, never used for selection) | `configs/realworld/fql/base.yaml`, plan rows `fql_budget` |
| Encoders | task-specific pretraining only; the joint-task rows and configs are removed | `s4d/rwlearn/configs.py`, `launch.py` |
| SNeRL | 3 input cameras; validation on separate episodes; input-view and novel-view reconstruction reported separately; native multi-view FQL condition kept, its extra inputs labelled | `s4d/rwlearn/snerl_eval.py`, `fql_data.CONDITIONS` |
| New baseline | ONE end-to-end random-init CNN + FQL (trained jointly, not frozen), official FQL pixel handling, matched to the other conditions | `s4d/rwlearn/pixel.py`, condition `e2e_cnn` |
| Single view | random camera per sample, fixed through the history and the paired next observation | `OfflineTransitions` / `OfflinePixelTransitions`, tests |
| Calibration hold | only geometry-dependent consumers wait for `calibration_validated`; the 100-episode gate is unchanged | `s4d/rwlearn/gate.py` |

## Implementation choice: faithful PyTorch port + official-code equivalence

The reference is the official FQL, github.com/seohongpark/fql at commit `e8cd16eb490332924dfa2492097219f181765933`
(MIT). It is ported to `s4d/rwlearn/fql.py` rather than vendored as a JAX submodule, for four reasons:

1. Every frozen encoder is a PyTorch module (`s4d/rl/encoders.py`), and so are the feature extraction and offline
   inference. One framework end to end avoids a JAX<->torch boundary at inference time.
2. JAX, flax, optax and distrax would be a new locked dependency set and a new remote image. The baseline expansion
   policy forbids locked shared dependency changes. JAX also preallocates most of a shared GPU by default.
3. The agent is small (MLPs, flow matching, a 2-member critic), so a port can be checked exactly. That gives stronger
   evidence than running vendored code in an untested environment.
4. The official checkout stays pinned by URL and commit. The fixture tool verifies the commit and that the tree is
   clean before importing it.

**What is reproduced.**

- **Networks.** MLP with Xavier-uniform kernels (variance_scaling(1, fan_avg, uniform)), zero biases, tanh-GELU,
  LayerNorm eps 1e-6 after hidden activations, and a 2-member `Value` ensemble.
- **Critic.** MSE to `r + gamma * mask * mean(target Q(s', clip(pi(s'))))`.
- **Actor.** BC flow matching plus `alpha` times the distillation of the 10-step Euler flow into the one-step policy,
  minus Q.
- **Stop-gradients.** These sit exactly where the official code places them: the target critic, the flow target,
  the next actions, and the critic inside the actor's Q term (`functional_call` with detached parameters).
- **Optimiser.** One Adam step on the summed loss.
- **Target update.** Polyak from the pre-update critic, as the official `target_update` reads the old network.
- **Logging.** The official `grad/{max,min,norm}` statistics are reproduced.

**Evidence.**

- *Exact equivalence* (`tests/test_rwlearn_fql.py`, fixture `tests/fixtures/rwlearn/fql_official_small.npz`
  produced by the official code). The official `get_config()` is used with 4x32 hidden layers and everything else
  official, on one batch of 256. Over five `agent.update` calls the port starts from the official initial parameters
  and consumes the official random draws. It reproduces every logged value and every online and target parameter to
  float32 round-off, plus `sample_actions` and `compute_flow_actions`.
- *Full official configuration on a toy task* (`docs/realworld_fql_toy_equivalence.json`, tools
  `runs/setup/tools/realworld_fql_20261106/fql_reference_jax.py` + `fql_toy_compare.py`). The configuration is
  4x512 MLPs, batch 256, lr 3e-4, alpha 10, flow_steps 10, discount 0.99 and tau 0.005, on a one-step bimodal bandit
  for 3000 updates with the same initial parameters, batches and draws. The largest relative difference of any logged
  value over all 3000 updates is 2.4e-4 (`q_max`); the losses differ by at most 1e-4. The final parameters differ by
  a median relative L2 of 4.7e-6 (max 1.5e-4). The final one-step and flow actions differ by at most 6.3e-6. Both
  implementations give identical behaviour statistics.
- A behavioural toy test: the Q-steered one-step policy picks the better behaviour mode while the BC flow keeps both.

## Policy input (all baselines; `s4d/rwlearn/fql_data.py`)

The input is the frozen encoder output for a 3-step observation history plus proprioception:

- **History.** Three consecutive policy steps of one viewpoint, padded with the run's first frame at run starts.
- **Adapter.** The campaign's frozen-encoder adapter, trainable and 256-d, exactly as in the sim DrM protocol:
  `FrameMLPStackHead` over the 3 per-frame features, or `SmallPostEncoderMLPHead` over SinCro's fused history state.
  Each FQL network gets its own adapter instance, mirroring the official per-network `encoder` modules, and the
  target critic gets a copy.
- **Proprioception (10-d).** Measured flange position, the first two columns of R (the 6D representation, continuous
  unlike Euler angles or quaternions), and the measured gripper closure. The campaign's sim protocol gives every
  method proprio (`docs/BASELINES.md`: identical proprio across baselines), so all real-world baselines get it too.

| Condition | Encoder input per step | Viewpoint per sample |
|---|---|---|
| `dynamo`, `reviwo` | one RGB frame | random camera (1 of 3), fixed for history and next observation |
| `sincro` | 3-frame history -> one state | same |
| `mvp3d_single` (B) | one RGB-D frame + K + c2w | same |
| `snerl` | 3 RGB views + poses (privileged) | all 3 cameras (the rig has exactly 3; SNeRL mean-pools them) |
| `mvp3d_multiview` (A) | 3 RGB-D views fused (privileged) | all 3 cameras (fused cloud; order irrelevant) |
| `e2e_cnn` | one RGB frame, 3 frames channel-stacked into a jointly trained IMPALA CNN | random camera (1 of 3), fixed for history and next observation |

The extra observation inputs of the multi-view / depth conditions relative to the single-view reference (1 RGB view +
proprio) are recorded per condition (`fql_data.CONDITIONS[c].extra_inputs`, dataset attributes `inputs` /
`extra_inputs`, policy metadata) and labelled in every plan row and report.

**Camera randomisation** (`OfflineTransitions.sample`). Batches are uniform over (camera timeline, step)
transitions, so the viewpoint is drawn per training sample. Every index of the observation history and of its paired
next observation lies in that one camera's timeline, which the test asserts. Single-view timelines are each camera's
own native frames (observation, proprio and action share that camera's timestamps). The camera is therefore drawn
with probability proportional to its valid transitions; measured on all 200 recordings (2026-11-06): 1/3 each to four
decimals (drawer-open 14,743 / 14,745 / 14,743, mug-hanging 22,427 / 22,427 / 22,426 transitions per camera).
Single-view conditions (and `e2e_cnn`) depend on no cross-camera sync, only on each camera's robot <-> camera sync
(nearest robot sample <= 6.2 ms on every frame, PREP2). Multi-view conditions use the synchronised steps of criterion
M5 (`docs/realworld_baselines.md`, "sync_index": mutual nearest frames without a time cap, robot-surface motion
<= 5 mm). Each step is timestamped at its LATEST constituent frame, so no camera image is newer than the
proprioception and the start of the action interval. Without this, a camera 6.6-22.9 ms after the reference would see
commands already inside the labelled action (Codex review).

**Features.** Frozen features are computed once per observation time with the export the policy will run, and stored
in the FQL dataset file next to the transitions. These are regenerable per-frame vectors keyed to the export's sha256,
not window copies. The pipeline test checks that live encoding of store frames equals the cached features.

## Action contract (Franka Panda + Robotiq 2F-85; `s4d/rwlearn/actions.py`)

Evidence: `docs/realworld_fql_action_report.json`, computed over every local recording at 04:00 (76 drawer-open,
29 mug-hanging; 33,402 + 20,787 valid steps), all three camera timelines each.

| Item | Decision | Evidence / reason |
|---|---|---|
| Source | recorded `action/flange_pose` = the command actually SENT (`sent`=`engaged`=1 on every used row) | **Confirmed by the user (2026-11-06).** `action/flange_pose` stays within `max_position_lead` = 0.040 m of the measured `robot/flange_pose` on every row. Rows flagged `lead_limited` have the raw operator target `action/flange_target` about 4 cm further. `limit_lead` reproduces the sent command from target + measured pose (median < 5 mm, tested). |
| Point / frame | flange (`panda_link8`) origin; increments in the base frame `panda_link0` | `T_flange_tcp` is the identity in every recording, so TCP == flange (checked per episode; otherwise refused). `conventions_json`: all recorded poses are flange poses; base +x forward, +y left, +z up. Base-frame deltas are the standard OSC-delta convention and are view independent. |
| Delta | `a = [p_goal - p_k, rotvec(R_goal R_k^T)]`, goal = the step's command, reference = the MEASURED pose at the latest robot sample <= tau_k | The pose is Markov in the policy's observation (unlike command-to-command increments, which need the unobserved previous command). Execution: `goal = exp(r) R_k, p_k + dp`. |
| Rotation | rotation vector (axis-angle, rad) | Increments are small (median 6 deg, max 19.5 deg), so a minimal, continuous representation suffices. The 6D representation is used only for the absolute orientation input. |
| Control interval | one policy step = one camera frame (nominal 1/15 s; every camera's measured median period is checked within 1 %; measured 66.715 ms). The ~6.7 commands at the 100 Hz loop inside a step are aggregated by zero-order hold at the step end: the last command strictly before tau_{k+1}. | The controller tracks the latest command. The next observation is taken at tau_{k+1}. A step is invalid if any of its commands was not sent or engaged, if it spans a recording pause, if it is longer than 1.5 periods (dropped frame), or if either endpoint lacks a fresh state. A fresh state means a robot sample <= 15 ms old and a gripper register sample <= 100 ms old, both in the same segment. |
| Units / normalisation | m, rad. Policy action = `[dp / 0.0700 m, r / 0.499 rad, 2c - 1]` | The scales come from the recorder's own teleop limits: lead 0.04 m + 0.3 m/s x 1.5 periods; 20 deg + 1.5 rad/s x 1.5 periods. Every recorded command maps inside [-1, 1]: the pose dimensions reach at most 0.68. A policy output can never ask for more than the teleop envelope. |
| Gripper | command closure c in [0,1] (0 open, 1 closed), Robotiq position register = 3 + 227c (`conventions_json`) at the same row; policy g = 2c - 1, **continuous (user decision; never binarised)** | The command ramps over ~0.2 s, so it is mostly binary; the intermediate values of the ramp are kept. Mug-hanging is closed 48 % of steps; drawer-open never closes (c <= 0.5). The measured register stalls at the object, e.g. 210 on the mug handle, which is why proprio uses the measured closure. `to_robot_command(binary_gripper=False)` is the default and the trained contract. |
| Command vs measured | Not derived from measured poses. Comparison: the commanded offset has median 4.5 cm (drawer) / 3.3 cm (mug), versus measured motion of 5.7 / 4.2 mm per step. The cosine between commanded offset and measured motion has median 0.986 / 0.979. | The commanded goal leads the robot by about the lead limit; the robot moves ~13 % of the offset per step. Replaying measured deltas as goals would drive the impedance controller far more slowly than the operator did. |

**Round trip.** For every valid step of every local recording,
`to_robot_command(measured_pose, normalise(recorded))` equals the recorded sent command to 6e-16 (report and test on
the fixture). `to_robot_command` returns:

- the flange goal `[x y z qx qy qz qw]` in the base frame, the `action/flange_pose` convention;
- the gripper closure and its Robotiq register;
- the physical delta;
- `execution_flange_pose`: the goal passed through the recorder's RADIAL lead envelope (0.04 m, 20 deg around the
  measured pose). The per-dimension normalisation box is wider; its corner has norm sqrt(3) x 0.07 m.

Speed limits act over time and belong to the robot-side executor. That executor must keep the recorder's safety
layer.

## Rewards, terminals, transitions (`s4d/rwlearn/transitions.py`)

**What the recordings contain.** Events are only `ARMED`, `switch_controller`, `motion ENABLED/DISABLED` and
`RECORDING/recording paused` per segment. `status` = complete. There is no task-outcome or success marker. Recordings
have 1-4 segments: drawer-open 44/48 single-segment; mug-hanging mostly 2. Pauses of about 1.5 s fall between
segments with motion disabled and no command rows.

**Definition (implemented; `reward="ogbench_final_success"`; outcomes decided by the user).**

- A transition is a valid policy step (see the contract above). Transitions never cross a pause or a dropped frame.
  At each segment start the frame history restarts.
- Outcomes are explicit inputs: the `rw-outcomes` v1 document `outcomes.json` (written by `scripts/realworld_learning.py
  inputs` next to the gate report and split; every episode `outcome: success` per the user decision, with its ending
  decision and flag codes for traceability). The builder refuses without it (or the explicit
  `--assume-final-success`, kept for smoke builds) and stores the document in the dataset attributes. The plan rows
  pass `--outcomes <inputs>/outcomes.json`.
- For a successful episode, the success state is the recording's FINAL observation: the last recorded frame of
  every constituent camera. The builder verifies that the store's frames are the recording's frames. The label
  applies only if that frame is reached by a valid transition. An unusable tail, or a sync step dropped at the end,
  never moves the label earlier; the timeline is reported as `success_frame_unreachable`.
- A failed episode has no success state: its last transition is a truncation (`mask` 1).
- `r_k = success(s_{k+1}) - 1` and `mask_k = 1 - success(s_{k+1})`. This is FQL's own OGBench single-task convention
  (`ogbench/relabel_utils.py`: rewards = successes - 1, masks = 1 - successes), so the official `alpha` scale applies.
- A segment end before a pause is a frame-stack boundary, not a terminal (`mask` 1).
- The alternative `"sparse_final"` (r = 1 at the success transition, 0 otherwise, same masks) is implemented.

## Pipeline and tooling

- `scripts/realworld_fql_dataset.py build --store ... --condition ... --export runs/pretrain/<encoder>/encoder.pt`
  writes the FQL dataset file: features, proprio, normalised actions, rewards, masks, runs and measured poses per
  episode and timeline. It refuses without a passed gate unless `--smoke` (<= 3 episodes).
- `scripts/train_realworld_fql.py` follows the official offline loop: uniform batches and one update per step,
  1M steps, `log_interval` 5000. Every `eval_interval` it logs the official validation losses plus offline action
  errors of the one-step policy and the BC flow against the recorded commands (mm, deg, gripper) on validation
  episodes. Checkpoints are lossless: agent, optimiser, sampler and noise generators. It writes `policy.pt`.
- `s4d/rwlearn/inference.py::OfflinePolicy` loads `policy.pt` and checks the encoder export's sha256. `act(history,
  measured pose, gripper register)` returns the robot command; `replay` compares against a recorded episode. There is
  no robot I/O.
- `scripts/realworld_learning.py plan` emits rows for every consumer whose gate level passed ("Training plan"
  below):
  - `rw-<b>-<task>-pre` (task-specific only);
  - `rw-fqldata-<cond>-<task>`, which depends on the encoder row (`e2e_cnn`: on the task's DynaMo + ReViWo rows);
  - `rw-fql-<cond>-<task>-s{0,1,2}`, which depend on the dataset row.

  All are remote rows with `role: realworld_fql` or `role: realworld_pretrain`. Nothing is registered.

## Outcomes and ending flags (`s4d/rwlearn/outcomes.py`)

Every uploaded episode is a successful demonstration (user decision): `outcome: success`, success-state proxy = the
recording's final observation. The ending check FLAGS recordings whose ending may include a reset or reversal of the
completed task; a flag never changes the label. It reads robot data only (h5py, read-only; 200 recordings in 0.7 s,
45 MB RSS): measured flange pose, the sent gripper command, the Robotiq object status gOBJ, the external joint-torque
estimate `robot/tau_ext` and the camera frame times (every flag carries robot rows, seconds and per-camera frame
indices). Image evidence is reserved for episodes the robot data cannot decide (`needs_image_review`;
`review_sheet` tiles first / event / final colour frames of every camera from the original HDF5).

- **drawer-open.** The pull runs from the minimum of the flange along the drawer axis (principal horizontal
  direction, within 18 deg of base +y on every recording) to the end of the FIRST completed pull: forward progress
  (>= 1 mm) advances it only within 3 cm of the pull height (median height of the progress so far), so no plateau
  or lift (also a diagonal one) does; it ends at a 5 cm step back or a 3 cm rise; contact height = the lower of the
  pull-end and pull heights. Push-back = more than 5 cm of travel BEHIND the drawer front inside the handle corridor
  (below contact height + 3 cm, within 5 cm laterally of the handle) after having been at the front (within 3 cm or
  beyond it), in ANY corridor interval after the pull (also after lifting off and coming back down):
  `reversal_suspected`, unless the external torque during that travel shows free motion (geometry and torque
  disagree: `needs_image_review`); corridor travel away from the front is reviewed unless the torque shows contact.
  Also suspect: an external torque while moving back that reaches half of the pull's contact level above free motion.
  Every drawer-open operator returns about 60 % of the pull toward the start pose above the open drawer; that is
  information (`return_toward_start`).
- **mug-hanging.** Holds are the runs of gOBJ 2, merged across a flicker shorter than 1 s in which the fingers never
  opened (register within 20 of the holding level); a release opens the fingers, so a hang and a later re-grasp are
  separate holds even when the command never drops below 0.5. The hang release ends the last hold. Flagged: ending
  commanded closed or still holding, or a new grasp within 5 cm of where the previous object was released (mug taken
  off the hook). Review: several holds elsewhere, or the flange coming back within 3 cm of the release position
  (possible knock-off). Information: a closure after the release without object contact, the return toward the
  start pose.

**Result on all 200 recordings** (local raw HDF5, 2026-11-06 12:01; full report
`/home/ws/ws/realworld-ops/fql2/ending_report.json`, compact `docs/realworld_ending_flags.json`): drawer-open
100 no_reversal, 0 suspected, 0 review; mug-hanging 100 no_reversal, 0 suspected, 0 review. Closest margins:
the largest backward travel behind the drawer front in the handle corridor is 1.9 cm (drawer-open/episode_000010;
threshold 5 cm), away from the front 0.5 cm (episode_000121), the largest backward contact fraction is 0.41
(threshold 0.5; torque decisive on 96 of 100). Information
flags: `return_toward_start` 100 drawer-open / 33 mug-hanging; mug-hanging/episode_000018 closes the open gripper for
0.09 s after the hang release without object contact (robot rows 1240-1249, frames 184-186) and reopens. Every
mug-hanging recording has exactly one hold (episode_000056's two gOBJ runs are a 0.32 s flicker with the fingers
closed at 204-210) and ends open (gOBJ 3). A visual audit of the final frame of all
200 recordings (one camera each; `/home/ws/ws/realworld-ops/fql2/{drawer,mug}_final.png`) agrees: every drawer is open,
every mug hangs on the stand.

## End-to-end CNN + FQL (`e2e_cnn`; `s4d/rwlearn/pixel.py`)

A randomly initialised CNN trained JOINTLY with FQL from the offline data (never frozen). Reference: the official FQL
pixel handling at `seohongpark/fql@e8cd16e` (README :44-45, :63-64, :175-184: `--agent.encoder=impala_small
--p_aug=0.5 --frame_stack=3`).

| Part | Official source (e8cd16e) | Here |
|---|---|---|
| Encoder | `utils/encoders.py` :103-108 `impala_small = partial(ImpalaEncoder, num_blocks=1)`; :60-100 `ImpalaEncoder` (stacks (16, 32, 32), width 1, no dropout, no layer norm, MLP (512,) `activate_final=True`); :10-57 `ResnetStack` (3x3 conv SAME Xavier-uniform, 3x3/2 max-pool SAME, residual relu-conv-relu-conv) | `ImpalaEncoder(**IMPALA_SMALL)`: identical layers, init, SAME padding (`max_pool_same`, -inf padding at the high end), flatten in H, W, C order; at 128 px the flat feature is 16 x 16 x 32 = 8192 -> 512 |
| Normalisation | `encoders.py` :84 `x.astype(float32) / 255.0`, nothing else | same (uint8 frames from the compact store) |
| Encoders per network | `agents/fql.py` :196-202 one per critic / BC flow / one-step flow; :226 + :241-242 target critic = copy incl. encoder; :162-163 + :230-232 BC-flow encoder shared by the BC loss and the Euler flow target | `FQLAgent.create(encoder=...)` (one instance per network, target copy, Polyak on all critic parameters); `_OFFICIAL_ENCODERS` maps the official parameter tree |
| Gradients | `agents/fql.py` :22-80: critic loss through the critic encoder; BC loss through the BC-flow encoder; distillation + Q through the one-step encoder; critic in the Q term and the flow target stop-gradient | the port's losses (unchanged) |
| Frame stack | `utils/datasets.py` :71-85: `[ob[t-2], ob[t-1], ob[t]]` concatenated on the channel axis, padded with the episode's first frame; next = `[ob[t-1], ob[t], next_ob[t]]` | the shared 3-step history (`FRAME_STACK` 3), padded with the first frame of the recording run; `stack_frames` = channel concatenation, oldest first |
| Augmentation | `datasets.py` :86-89, :100-110, :15-31: one coin per batch (`p_aug` 0.5); random crop, edge padding 3, per-sample offsets in [0, 6]^2, the same offsets for observation and next observation, 4-D arrays only; `main.py` :85-88 also on the validation dataset | `augment` / `random_crop` (index clamping = edge padding, exact on uint8); validation batches augmented as official |
| Dataset actions | `envs/env_utils.py` :138-146 clip to [-1 + 1e-5, 1 - 1e-5] | `data.action_clip_eps: 1e-5`, applied in every condition's loader |

**Matched with the other FQL conditions:** task data and 90/10 episode split (`split_seed0.json`), the action
contract and rewards (`outcomes.json`), 10-d proprioception (concatenated to the 512-d CNN output, as the frozen
conditions concatenate it to their 256-d adapter output), the 3-step history, one random camera per sample fixed
through the history and the paired next observation, the compact store's 128 x 128 centre-crop RGB, FQL agent
configuration (alpha 10, 4 x 512 MLPs, batch 256, lr 3e-4), seeds 0/1/2 and 1M offline steps.
**Differences from the official visual OGBench runs (documented):** 128 px instead of 64 px images (crop padding stays
3 px), proprioception input, alpha 10 (official visual tasks 100-300), 1M instead of 500k steps.

**Evidence.** `tests/test_rwlearn_pixel.py` against `tests/fixtures/rwlearn/fql_official_pixel.npz` produced by the
official code (`fql_reference_jax.py pixel`): the `impala_small` parameter layout at 128 x 128 x 9 equals the
official tree; its output on uint8 inputs equals the official to 2e-5; flax max-pool SAME (odd and even sizes) and
`batched_random_crop` exactly; the frame stacking of `Dataset.sample(frame_stack=3)` exactly; three `agent.update`
calls of an official FQL agent with an `ImpalaEncoder` (4 x 4 stacks, (32,) MLP) reproduce every logged value and every
parameter (encoders and target encoders included) to float32 round-off, plus `sample_actions` / `compute_flow_actions`.
`tests/test_rwlearn_fql_pipeline.py::test_e2e_cnn_*` covers the dataset, camera-per-sample invariants, joint training
(every network's encoder moves), `OfflinePolicy.act`, `replay` from the store, and the pixel pin: the dataset records
each timeline's `rgb_sha256` at build time and the loader and replay refuse a store whose frames differ, so the
dataset file's sha256 (the trainer's resume identity) also identifies the training pixels. `tests/test_rwlearn_trainers_gpu.py` (marker `gpu`, release
chain) runs the dataset builder and the trainer as subprocesses with a lossless 2 + 2 = 4-step resume.

Compute (measured 2026-11-06 on a local MIG 1g.24gb slice, official configuration, batch 256, 128 px): 281 ms per
update with the default kernels, peak 6.8 GiB (1M steps = 78 h on such a slice; a full GPU is several times faster);
frames of a task (~45k / ~80k per task across 3 cameras, 2.2 / 3.9 GB uint8) live on the GPU. Row resources: 12 GB
GPU, 24 GB RAM. Determinism: with the default (fast) cuDNN kernels two uninterrupted GPU runs already differ (126 of
132 tensors after 4 steps, up to 2 % relative: convolution / max-pool backward are nondeterministic), so a resumed run
can only match "as well as two uninterrupted runs". `train.deterministic: true` makes runs bitwise reproducible
(2 + 2 steps == 4 steps in every parameter, Adam moment and generator; `tests/test_rwlearn_trainers_gpu.py` proves
resume losslessness this way) at +12 % time per update (314 ms) and is off for real runs.

## Calibration hold per consumer (`s4d/rwlearn/gate.py`)

User decision 2026-11-06 item 2: hold only geometry-dependent outputs and their training eligibility until the
calibration is validated. The 100-episode-per-task gate is unchanged; geometry-dependent consumers additionally need
the per-episode record `calibration_validated`.

| Consumer | Class | Why |
|---|---|---|
| ours (deferred) | geometry | base-frame dynamic 3D tracks, depth fused across cameras, camera poses |
| SNeRL (pretrain + FQL `snerl`) | geometry | encoder input poses and every rendered ray come from the extrinsics |
| 3D-MVP (pretrain + FQL `mvp3d_multiview`, `mvp3d_single`) | geometry | RGB-D back-projected into the base frame, base-frame bounds |
| SinCro (pretrain + FQL `sincro`) | geometry | rendering targets along rays from K / c2w; the FQL condition uses that encoder |
| ReViWo (pretrain + FQL) | image | synchronised multi-view RGB, timestamps only |
| DynaMo (pretrain + FQL) | image | single-camera RGB |
| e2e CNN + FQL | image | RGB + proprioception (robot kinematics) + sent commands |

An unknown consumer is held like a geometric one, and the consumer is the CALLING TRAINER's fixed identity
(`realworld_classes(..., consumer=<b>)` in `scripts/train_<b>.py`; FQL: the dataset's condition): a config that
declares another `data_interface.consumer` is refused. Short (<= 10k-step plumbing) runs of geometry-dependent
consumers are held too, unless the explicit `smoke_without_calibration` opt-in is set (synthetic test stores only;
never in generated configs or plan rows). The aggregate "all recorded stages done" check skips the hold stage (ING's
`set_stage` folds every stage into `preprocessing.status`, so a pending `calibration_validated` would otherwise hold
everyone); the recorded aggregate must equal `set_stage`'s aggregation of the recorded stages exactly (a missing,
unknown or stale aggregate fails). Two bindings make the
validation stick to the data actually trained on: the gate requires `metrics.calibrate_build_id` = the CURRENT
`calibrate` record's `metrics.build_id`, and every geometry-dependent trainer requires the training store's
`upstream.calibrate` (compact store attribute) to equal that id and every store camera to be listed as validated
(`check_store_calibration`; FQL datasets record the store's ids at build time and `check_dataset_gate` compares).

**Record contract for agent CAL** (`ingest_worker.py set-stage KEY calibration_validated JSON`, one per episode):

| Field | Required | Value |
|---|---|---|
| `status` | yes | `done` once validated and accepted; `failed` (+ `failure` {message, ...}) when rejected or ambiguous; `pending` / `running` while open (all of these hold geometric consumers only) |
| `code_version` | yes | e.g. `calibration_validation@1` |
| `output_format` | yes when done | `rw-calibration-validation/<n>`, e.g. `rw-calibration-validation/1` |
| `input_sha256` | yes | the episode's raw sha256 (set_stage refuses a mismatch) |
| `metrics.decision` | yes | `accepted` (anything else holds) |
| `metrics.calibrate_build_id` | yes | the `build_id` of the `calibrate` stage record that was validated (= the manifest's `stages.calibrate.metrics.build_id`); after a recalibration, the NEW build id |
| `metrics.cameras` | yes | list of the validated camera serials (store cameras outside it are refused) |
| `metrics.*` evidence | recommended | scope (episode / session / shared), multistart spread, mask / frame sensitivity, independent robot-frame, depth-agreement, reprojection and cross-view metrics, report path |
| `code_commit`, `config_sha256`, `outputs`, `host`, `started_at`, `finished_at`, `attempt`, `notes` | optional | as for other stages |

Every field must be one of set_stage's `STAGE_FIELDS` (top-level extras are refused; everything else goes under
`metrics`). Tested: `tests/test_rwlearn_gate.py::test_ingest_worker_set_stage_accepts_the_calibration_record` writes
the record through the committed `set_stage` on a temporary manifest.

## Training plan (rows not registered; `scripts/realworld_learning.py plan`, preview `docs/realworld_training_plan.yaml`)

Host: remote 192.168.10.22 (`host: remote`, roles `realworld_pretrain` / `realworld_fql`); commit: pinned by the
coordinator at registration (`plan --commit <release>`). Order: pretraining rows priority 1, FQL rows priority 2;
every frozen-encoder FQL dataset row depends on its encoder's pretraining row; the `e2e_cnn` dataset row depends on
the task's DynaMo and ReViWo pretraining rows (baselines first, without waiting on calibration-held rows). Our own
real-world pretraining stays deferred until the architecture is final. 66 rows: 10 pretraining + 14 dataset builds +
42 FQL runs. Under the calibration hold the plan emits only the image-level rows (4 pretraining + 6 dataset builds +
18 FQL runs) and lists the rest as refused with the gate's reasons.

| Task | FQL condition | Encoder pretraining (task-specific, documented protocol, 1 run) | Observation (extra vs single view) | FQL seeds x budget | GPU / RAM GB | Calibration-dependent |
|---|---|---|---|---|---|---|
| each | `dynamo` | `rw-dynamo-<task>-pre`: 40 epochs, seed 42 (12 / 32) | 1 RGB view (none) | 0,1,2 x 1M steps, batch 256, alpha 10 | 6 / 12 | no |
| each | `reviwo` | `rw-reviwo-<task>-pre`: 100,001 steps, seed 42 (4 / 16) | 1 RGB view (none) | same | 6 / 12 | no |
| each | `sincro` | `rw-sincro-<task>-pre`: 300k steps (50 epochs cap), seed 42 (12 / 24) | 1 RGB view (none) | same | 6 / 12 | yes |
| each | `snerl` | `rw-snerl-<task>-pre`: 2.4M iterations (official), 3 input cameras, seed 0 (16 / 20) | +2 RGB views, +3 camera poses (native multi-view) | same | 6 / 12 | yes |
| each | `mvp3d_multiview` | `rw-mvp3d-<task>-pre`: 15 epochs, seed 0 (12 / 24) | +2 RGB views, +depth, +K / extrinsics | same | 6 / 12 | yes |
| each | `mvp3d_single` | same encoder as above | +depth, +K / extrinsics | same | 6 / 12 | yes |
| each | `e2e_cnn` | none: IMPALA-small random init, trained jointly with FQL | 1 RGB view (none) | same | 12 / 24 | no |

Each task is drawer-open and mug-hanging (gated separately). Frozen-condition dataset builds: 8 GB GPU / 16 GB RAM;
the e2e build is CPU work (1 / 8). Pretraining resources are the campaign's Meta-World rows of the same trainers.

## Readiness checklist

| | done | pending |
|---|---|---|
| FQL port | exact + full-config equivalence vs official; pixel path exact vs official | - |
| action contract | user-confirmed; round trip exact on 105 recordings; decision pinned by tests | - |
| outcomes | user decision implemented; ending check on all 200 recordings: 0 flagged | - |
| rewards / terminals | implemented, outcomes from `outcomes.json` | - |
| datasets / camera randomisation | frozen + e2e, tests incl. camera-per-sample invariants | - |
| training / inference | trainer (frozen + e2e), lossless resume (CPU dry run), `OfflinePolicy` (frozen + e2e) | GPU tests in the release chain |
| gate | 100-episode gate + per-consumer calibration hold, set_stage contract tested | CAL's `calibration_validated` records; gate passes |
| launch | 66 planned rows, role-classified, not registered | coordinator registration with the pinned release |

## Cross-camera sync criterion M5 (coordinator decision 2026-11-06; PREP2's review)

Evidence: `/home/ws/ws/realworld-ops/pre/reviews/sync-criterion-20261106.{md,json,analyze.py}` (PREP2). The cameras are
free-running; per-recording offsets are constant and reach half a period; the former raw time caps were chosen on
single recordings without motion evidence and left 12 mug-hanging and 16 drawer-open recordings without any complete
3-camera set. Robot-surface motion within a set's time spread: p95 typically 1.8-3.9 mm, worst 7.0 mm (about 1 px at
128 px; below the 7.8 mm held-out geometric residual).

**Implemented (multi-view consumers):** `train_store.mutual_sync` pairs mutual nearest frames with no time cap;
`set_surface_motion` stores, per set, the largest displacement of the robot's URDF surface (the stage's
`RobotGeometry`, 300 points per link) between any two members, robot state interpolated to each member's timestamp
(`inf` when a member has no valid robot state); `sync/surface_disp_m` + attributes (criterion, pairing, surface, default
threshold, per-episode counts: complete / accepted / rejected for motion / rejected without robot state) and the stage
summary `sync` (copied to the manifest by `manifest_fields`). Loaders filter with a configurable threshold, default
5 mm: `RealworldWindowDataset(max_set_motion_m=...)` (our method) and `rwlearn.store.sync_index(max_set_motion_m=...)`
(baselines, multi-view FQL). Stores without the motion (schema < 1.1.0) are refused unless `sync_criterion="time"` is
asked for explicitly. `train_data` writer version 1 -> 2, schema 1.0.0 -> 1.1.0 (`train_data@2`,
`s4d.realworld.train/1.1.0`): a new stage fingerprint, so the dispatcher re-exports `train_data` for every episode
(depth, masks and tracks are reused; one re-export together with the calibration revision if timing allows). The
`train_data` key `max_sync_ns` (30 ms mug-hanging config, 20 ms drawer-open template) is now ONLY the per-camera
robot <-> camera tolerance (unchanged values; nearest robot sample <= 6.2 ms everywhere).

**Single-view consumers** (FQL single view, e2e CNN + FQL, DynaMo) use each camera's own frames and robot sync; tested
to be identical whether every set is accepted or rejected.

**Accepted sets / windows (M5, 5 mm; this implementation on the raw recordings, every frame in one TrackEverything
window as in PREP2's count, `/home/ws/ws/realworld-ops/fql2/m5_counts_raw.json`):** mug-hanging 22,429 of 22,563
complete sets, 43,562 windows (PREP2: 22,431 / 43,566; largest per-episode difference 1 set / 4 windows, the stage's
denser surface sampling finds slightly larger maxima); drawer-open 14,850 of 14,850 sets, 29,036 windows (identical).
No recording is left without windows (former caps: 12 + 16). Agreement test on mug-hanging 45, 50, 55, 1 and
drawer-open 11, 0 (`tests/test_realworld_sync_m5.py`, local raw data only).

**Contract checks and QA per consumer family (PREP2, branch rw-contract-20261106).** `contract_checks.py` reports
`report["singleview"]` (per camera: frames, `robot/ok` fraction, `rwlearn.transitions.timelines(multiview=False)`
transitions under the action contract; CPU, no cross-camera set) and `report["multiview"]`
(`RealworldWindowDataset` with the default M5 threshold, `validate_batch`, forward check, SNeRL scenes from the
store's M5 sets via `depth_stages.episode_frame_sets`). An empty multi-view split fails only the multi-view family; the
process exits 0 once both families are recorded. The viewer uses the same M5 sets; `configured_sets_policy` never uses
`train_data.max_sync_ns` as a cross-camera cap. The QA (`rwprep_local.py qa --families`, schema rwprep-qa/3) turns
the gates into three families - `singleview` (recording, store frames, single-view contract family: every consumer),
`multiview` (M5 sets, multi-view contract family), `geometry` (per-episode calibration, metric depth, tracks) - and
passes the episode (the dispatcher's `done`, the `qa` record's status) iff the base single-view family passes. The
dispatcher copies the families into the `qa` stage record (`metrics.families[<family>].passed`); `gate.py` requires
each consumer's families (`CONSUMER_FAMILIES`: ours / SNeRL / 3D-MVP / SinCro all three, ReViWo singleview +
multiview, DynaMo / e2e CNN singleview) on all 100 episodes. Single-view FQL conditions have their own gate identities:
ReViWo needs singleview; SinCro and 3D-MVP single-view need singleview + geometry. They retain their encoder's
calibration requirements without requiring cross-camera sets for their observations. These checks apply in addition
to the base gate and, for geometric consumers, the calibration hold. A `qa` record without families counts as the
pre-family QA (rwprep-qa/2 required every family) unless it declares rwprep-qa/3 (fail closed). Unknown QA formats and
malformed family verdicts fail closed.

## Review of the decisions branch (Codex gpt-6.1-sol xhigh, read-only, on e4c137f) and what changed

No mismatch was found in the IMPALA layer order, initialisation, pooling, flattening, normalisation, augmentation or
action clipping; gradient routing and target-encoder Polyak updates match the official agent; plan rows and configs
match the documentation; no frozen-encoder or Meta-World SNeRL regression. Findings, all fixed:

- **High: short runs bypassed the calibration hold** (<= 10k-step smoke exception). Geometric short runs now need the
  hold too, except with the explicit `smoke_without_calibration` opt-in.
- **High: the pretraining hook trusted the configured consumer.** Each trainer passes its fixed consumer; a conflicting
  `data_interface.consumer` is refused.
- **Medium: drawer pull end = global maximum** (a later free motion beyond the pull end hid a push-back) and **the
  first lift ended the handle-height check** (re-lowering and pushing was missed). Now the first completed pull and
  every handle-height interval after it.
- **Medium: mug holds from the command threshold** could merge a hang and a re-grasp. Now gOBJ holds with an
  explicit flicker merge rule.
- **Medium: the e2e resume identity did not pin the pixels.** Per-timeline `rgb_sha256` at build, verified on load.
- **Medium: `replay` failed for `e2e_cnn`.** Pixel replay from the store (pinned frames).
- **Low: an open hold excused a missing / unknown aggregate status.** The aggregate must equal set_stage's.

The real-data result is unchanged after the detector changes (200 / 200 no_reversal).

A second, targeted Codex pass on 04cd272 confirmed fixes 2, 4, 6, 7 and 8 and found the rest incomplete or new
regressions, all fixed afterwards: the smoke opt-in accepts only a literal `True` (the string "false" was truthy); the
pull end advances only on >= 1 mm forward progress and its contact height is the height at that progress (a plateau
lift no longer raises it; a pull descending while pulling and then a lifted return is no longer flagged); push-back
needs the handle corridor and the drawer front (a lowered motion elsewhere is reviewed, not suspected); the mug merge
reference is the run right before each gap; the release row maps the first gripper sample after the hold; the merge
test uses a sub-second gap and checks both outcomes.

A third Codex pass (on 14fd240: M5 + these fixes) found, and this branch fixed: a non-finite interpolated robot state
could yield a 0 mm set (now `inf`, rejected); `surface_disp_m` stored as float32 made counts and loaders disagree at
the threshold (now float64, one comparison everywhere); a diagonal lift could still advance the pull end (progress
now counts only within 3 cm of the pull height; contact height = the lower of pull-end and pull heights); a motion
lowered beyond the open front counted as push-back (now only travel behind the front counts). Not fixed here (other
owners at that review): `depth_stages.configured_sets_policy` passed `train_data.max_sync_ns` to `frame_sets` as a
cross-camera cap for the SNeRL contract check and the viewer; the consumer-family change above now uses the store's
M5 sets. Real data after all fixes:
200 / 200 no_reversal; M5 counts unchanged.

## Review (Codex gpt-6.1-sol xhigh, read-only, on eeaafb2) and what changed

The review found no discrepancy in the FQL mathematics (initialisation, GELU, LayerNorm eps, ensemble reductions,
stop-gradients, Adam, clipping, pre-update Polyak). It found no camera-index mixing and no Meta-World path
regression. Findings and how each was resolved:

- **Fixed: per-episode action scales.** The scales depended on each episode's measured period, so multi-episode builds
  failed on clock jitter. The contract now uses the nominal 1/15 s and checks each camera within 1 %.
- **Fixed: multi-view causality.** Steps are timestamped at their latest frame.
- **Fixed: transitions into an invalid next state.** Both endpoints now need a state.
- **Fixed: gate scope.**
  - The gate only checked inventory. Trainers now also require the split to be a disjoint train/validation partition
    of exactly the gated episodes, the store's tasks to equal the target's, and no truncating data cap.
  - `inputs --target` is enforced.
  - The FQL trainer checks the split against the dataset.
- **Fixed: gate predicate.**
  - It now mirrors ING's: `exclude` / `accept` overrides, `source_missing` always blocks, and unknown flags block.
  - It now requires a sha256 and an equal `remote_sha256`; unknown validation states fail.
- **Fixed: DataLoader forks.** Locator-opened stores are released after indexing and reopened per process.
- **Fixed: stage names and depth quantisation.** Stage names are now the preprocessing dispatcher's. Depth is
  quantised exactly as the compact store does it (verified bitwise).
- **Fixed: smaller issues.**
  - Inference noise moves to the policy device.
  - Validation draws from its own generator, so it never shifts training randomness.
  - Resume refuses checkpoints whose encoder, contract, dataset, split, smoke status or agent config differ.
  - The gripper register freshness limit is 100 ms.
- **Adopted as design: outcomes and the execution envelope.** Outcomes are explicit inputs; the success label sits
  only at the final frame; the execution pose applies the radial lead envelope.
- **Second, targeted Codex pass on 7bc391b.** It confirmed the fixes above and found five remaining problems, all
  fixed:
  - success could still move earlier when synchronisation dropped tail frames; it is now anchored to the recording
    end;
  - non-boolean outcome labels are refused;
  - recordings ING does not count (blocked, excluded, source not present) are excluded and listed rather than failing
    the gate;
  - duplicated validation episodes are refused;
  - data caps are checked only for full runs and only when they truncate.
- **SNeRL inputs (decided 2026-11-06):** 3 of 3 input cameras; input-view and leave-one-camera-out novel-view
  reconstruction reported separately on the validation episodes (`docs/realworld_baselines.md`).
