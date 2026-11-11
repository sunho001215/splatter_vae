# Real-world offline FQL on frozen baseline representations and end-to-end

Unified pool (directive 2026-11-11 ~20:15 KST): LOCAL (8 MIG slices, 150 GB RAM budget) and REMOTE are
interchangeable training hosts for Meta-World, real-world and DROID (`configs/hosts/roles.yaml`, version 2).
Data locality still applies: preprocessing and typed release verification stay remote; remote-resident training
runs locally only with a verified resolver in `ops/jobs/realworld_local.py`. Resolvers cover FQL training,
real-world workspace stats, our real-world pretraining, and exports from locally produced run directories.
Remote dependency files require verified backup checksums; a remote run directory cannot be consumed locally.
`needs`, `fetch`, `verify`, `plan` and `place` inventory, mirror, check and explicitly place never-started rows.
Started runs use `ops/jobs/migration.py`; begin permits either admitting destination, while switch-to-remote requires
the separate remote release procedure. Meta-World keeps its pool/campaign release; other families keep their pins.
Remote reboot/admission holds never hold local placement or admission. A queue's adjacent `HOLD` holds that queue;
`hosts.remote.hold` / `enabled: false` hold only remote admission. The RTX 5090 inference host never schedules.
Local baseline admission guarantees each active/ready family an equal share of RAM, GPU slots and GPU memory,
borrowing only unclaimed guarantees and retaining trainer head-of-line reservation per family (`ops/jobs/fair_share.py`).


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
| FQL alpha | 10 with the reference configuration for every condition; NO alpha sweep. Later tuning follows a predefined real-robot evaluation protocol, not offline action error (the offline action errors are logged as diagnostics only, never used for selection). **Superseded for frozen encoders by the user decision 2026-11-11 ~23:10 (section "FQL setting for pretrained encoders")** | `configs/realworld/fql/base.yaml`, plan rows `fql_budget` |
| Encoders | task-specific pretraining only; the joint-task rows and configs are removed | `s4d/rwlearn/configs.py`, `launch.py` |
| SNeRL | 3 input cameras; validation on separate episodes; input-view and novel-view reconstruction reported separately; native multi-view FQL condition kept, its extra inputs labelled | `s4d/rwlearn/snerl_eval.py`, `fql_data.CONDITIONS` |
| New baseline | ONE end-to-end random-init CNN + FQL (trained jointly, not frozen), official FQL pixel handling, matched to the other conditions | `s4d/rwlearn/pixel.py`, condition `e2e_cnn` |
| Single view | random camera per sample, fixed through the history and the paired next observation | `OfflineTransitions` / `OfflinePixelTransitions`, tests |
| Calibration hold | only geometry-dependent consumers wait for `calibration_validated`; the 100-episode gate is unchanged | `s4d/rwlearn/gate.py` |
| covering 98 episodes | user decision 2026-11-10: task-specific gate amendment on its 98 valid episodes. The same episodes and splits apply to every method; there is no duplication and no timestamp repair | `configs/realworld/gate_amendments/covering-98-20261110.json`, `docs/realworld_baselines.md` |
| pick-and-place 97 episodes | user decision 2026-11-10: task-specific gate amendment on its 97 valid episodes, with the exact list and checksums pinned. The same split applies to every method; the 3 exclusions are documented; there is no duplication and no repair | `configs/realworld/gate_amendments/pick-and-place-97-20261110.json`, `docs/realworld_baselines.md` |

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
not window copies. The pipeline test checks that live encoding of store frames equals the cached features. For the two
3D-MVP conditions only, the adapter reads them standardised (next section).

## Superseded 3D-MVP rows (2026-11-10; documentation only, queue entries unchanged)

The 12 FQL rows `rwcal-20261109-fql-mvp3d_{single,multiview}-{drawer-open,mug-hanging}-s{0,1,2}` (and their policies)
are SUPERSEDED: vision-blind (un-normalised 3D-MVP read-out) and 8/12 critics diverged (MVP3D-FQL-DIAG.md). They stay
quarantined and are never reported as 3D-MVP results. Replacements (user decisions 2026-11-10): native-RealSense-depth
chain `rwcal-20261110-nativedepth-mvp3d-<task>` -> `rwcal-20261110-mvp3d-<task>-pre-native` ->
`rwcal-20261110-fqldata-mvp3d_<view>-<task>-native` -> `rwcal-20261110-fql-mvp3d_<view>-<task>-s{0,1,2}`
(`encoder.feature_norm=train_standardize`), FQL held until each dataset passes the standardised feature check. The
2026-11-09 encoders (`rwcal-20261109-mvp3d-<task>-pre`, FS depth) and fqldata datasets are kept unchanged.

## Frozen-feature standardisation (3D-MVP only; 2026-11-10)

**Why.** 3D-MVP's frozen read-out is the per-view max-pool of the last residual stream of a pre-norm transformer with
no final norm (`docs/baselines_mvp3d.md` §6). On the real-world stores it is a large constant vector plus a small
signal: offset/signal ||mu|| / sqrt(sum var) = 84-136 on the four 3D-MVP FQL datasets (ReViWo 0.03, DynaMo 2.5,
SinCro 0.57-0.74; Meta-World 3D-MVP 3.5-4.7). The adapter's first operation is Linear -> LayerNorm; with ~99 % of its
input the same vector for every sample, the LayerNorm output is constant to ~1e-3 (random-projection unit std
0.007-0.010; ReViWo 0.99) and, LayerNorm being scale-invariant, no gradient removes the offset. All 12 trained
real-world 3D-MVP FQL policies (`rwcal-20261109-fql-mvp3d_*`) are therefore vision-blind (replacing the features
changes the action by <= 2e-7) and 8 of 12 critics ran away to Q = +4 000..+20 000 (feasible range [-100, 0]).
Diagnosis: `/home/ws/ws/realworld-ops/serving/MVP3D-FQL-DIAG.md`; this fix and its validation:
`/home/ws/ws/realworld-ops/serving/MVP-STATUS.md`.

**What** (`encoder.feature_norm: train_standardize`; default `none`). Per dimension d, the adapter input is
`(f_d - mu_d) / sigma_d`:

- *Fit.* `fql_data.fit_feature_standardization(train.features)`: mean and population standard deviation over every
  stored row (observation time) of the TRAINING split's timelines (`OfflineTransitions(split["train"])`; validation
  episodes never enter), two passes with float64 accumulation on CPU, sigma floored at 1e-3 x max sigma (no 3D-MVP
  dimension is near the floor), kept as float32. The fit runs once at the start of a run; its descriptor (kind, rows,
  dim, floor, sha256 of the float32 mean||std) joins the resume identity, so a run never resumes under a different
  standardisation.
- *Where.* `fql_data.ObsEncoder` holds mean / std as non-trainable buffers `feat_mean` / `feat_std` of every network's
  encoder (critic, target critic, BC flow, one-step policy; constant under Adam and the Polyak update) and applies them
  to the float32 view of the stored float16 features, in front of the unchanged campaign adapter (`FrameMLPStackHead`;
  `s4d/rl/encoders.py` is not touched). Registering the buffers consumes no randomness: the adapter's initial weights
  are those of the unstandardised run with the same seed.
- *Persistence.* The buffers are part of the agent state in `policy.pt` / `latest.pt`; the checkpoint meta carries the
  descriptor (`meta["feature_norm"]`). `OfflinePolicy` (and so `DeployPolicy`) rebuilds the encoders with the buffers,
  loads them strictly (buffers without the meta record, or the reverse, do not load) and checks every network's buffers
  against the recorded sha256. `check_realworld_deploy.py contracts` writes the descriptor into the run's
  `<run>.deploy_contract.json`, and `deploy.contracts.read_training_contract` refuses a sidecar whose record differs
  from the checkpoint's (a stale sidecar without one included).
- *Opt-in.* Only the 3D-MVP FQL rows set it (`ops/jobs/realworld_plan.py::FQL_CONDITION_SETS`, preview
  `docs/realworld_training_plan.yaml`); the trainer refuses it for `e2e_cnn`. With `none`, ObsEncoder state, forward,
  FQL updates, checkpoint meta, resume identity and deployment are bitwise those of d9b759c (tests + MVP-STATUS V4).
  Not changed: the encoder exports, the FQL datasets (reused as built), the FQL agent and its hyperparameters (alpha 10,
  `normalize_q_loss` false), every other condition, and the Meta-World DrM path (its 3D-MVP rows read the same
  read-out un-normalised; a user decision, MVP3D-FQL-DIAG.md §7).

**Validation before the reruns** (2026-11-10, criteria fixed in advance; numbers and tools in
`/home/ws/ws/realworld-ops/serving/MVP-STATUS.md`, evidence `serving/evidence/mvpfix/`):

- Feature check on the standardised inputs (4 datasets): training split offset/signal <= 6e-6, LN unit std
  0.988-0.995; held-out validation split 0.07-0.12 / 0.985-0.988 (raw: 84-136 / 0.007-0.011).
- 60k-step local-CPU pilots (exact trainer loop; mvp3d_single mug s0 and mvp3d_multiview drawer s1, the originals'
  worst seeds, + a raw control): Q stays feasible (q_mean -31..-15, q_max <= 0.30; finite everywhere). Replacing the
  image/depth features by another held-out sample's changes the action by 0.18-0.32 (DynaMo 0.12-0.13, ReViWo
  0.19-0.25 at 1M; raw 3D-MVP 3e-8 already at 20k-60k), the critic by 4-13; adapter LayerNorm unit std 0.98, no
  dead units. Multi-view drawer's critic overfits the 90 training episodes (validation critic loss 1.4 -> 3.7 from
  20k to 60k vs training 0.03; Q stays in range): watch it in the reruns.
- Deployment: with vision in use the train (FoundationStereo) vs deploy (native RealSense) depth difference now
  matters - see `docs/realworld_deployment.md` "Findings".

Precedent: RVT-1's `rot_ver=1` path normalises the same pooled global feature with `BatchNorm1d` before its MLP
(`third_party/rvt/rvt/mvt/mvt_single.py`, `feat_fc_init_bn`); a frozen read-out has no trainable bias that could absorb
the offset, so it is removed with statistics of the training data.

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
  every constituent camera. The builder maps store frames to raw colour indices by exact timestamp equality
  (`cameras/cam_<serial>/t`, rounded with `rint(t * 1e9)`, as in the panda adapter; never `t_recv` or `stereo/t`).
  The retained indices must equal the adapter's validated `stereo/color_index`: only colour frames without a
  stereo record may be absent. Missing matches, duplicates, reordered frames and segment mismatches fail closed.
  Legacy array fixtures without pairing metadata retain the previous raw-prefix restriction. Dataset `frames`
  remain compact-store indices, `raw_frames` record the corresponding recorder indices, and the `frame_mapping`
  provenance attribute records per-camera counts, the rule and every omitted raw index. The label
  applies only if that frame is reached by a valid transition. An unusable tail, or a sync step dropped at the end,
  never moves the label earlier; the timeline is reported as `success_frame_unreachable`.
- A failed episode has no success state: its last transition is a truncation (`mask` 1).
- `r_k = success(s_{k+1}) - 1` and `mask_k = 1 - success(s_{k+1})`. This is FQL's own OGBench single-task convention
  (`ogbench/relabel_utils.py`: rewards = successes - 1, masks = 1 - successes), so the official `alpha` scale applies.
- A segment end before a pause is a frame-stack boundary, not a terminal (`mask` 1).
- The alternative `"sparse_final"` (r = 1 at the success transition, 0 otherwise, same masks) is implemented.

The builder and the other `actions.load_raw` consumer (`contract_checks.singleview_family`) share this mapping.
Actions and proprioception still use the same robot-time queries at the exactly matched colour timestamps and
segments; multi-view timelines map each camera independently and keep the latest constituent time. Outcomes and
ending-review evidence already use raw recording frames. Frozen encoders, pixel loading/replay and
`train_realworld_fql` use the dataset's compact-store `frames`, so image reads keep their original index space.

## Pipeline and tooling

- `scripts/realworld_fql_dataset.py build --store ... --condition ... --export runs/pretrain/<encoder>/encoder.pt`
  writes the FQL dataset file: features, proprio, normalised actions, rewards, masks, runs and measured poses per
  episode and timeline. It refuses without a passed gate unless `--smoke` (<= 3 episodes).
- `scripts/train_realworld_fql.py` follows the official offline loop: uniform batches and one update per step,
  1M steps, `log_interval` 5000. Every `eval_interval` it logs the official validation losses plus offline action
  errors of the one-step policy and the BC flow against the recorded commands (mm, deg, gripper) on validation
  episodes. Checkpoints are lossless: agent, optimiser, sampler and noise generators. It writes `policy.pt`.
- `s4d/rwlearn/inference.py::OfflinePolicy` loads `policy.pt` and checks the encoder export's sha256 (and, for the
  3D-MVP rows, the frozen-feature standardisation's; previous section). `act(history,
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
| ours (pretraining `s4d/rwlearn/ours.py`; FQL `ours`, consumer `fql_ours` = families singleview + geometry) | geometry | base-frame dynamic 3D tracks, depth fused across cameras, camera poses |
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
real-world pretraining stays deferred until the architecture is final. 33 rows per task (66 for drawer-open +
mug-hanging; the preview lists all four targets, 132): 5 pretraining + 7 dataset builds + 21 FQL runs. Under the
calibration hold the plan emits only the image-level rows (per task 2 pretraining + 3 dataset builds + 9 FQL runs) and
lists the rest as refused with the gate's reasons; SNeRL / 3D-MVP are also refused while a task's scene geometry is
unmeasured (pick-and-place; covering was measured 2026-11-12; below).

| Task | FQL condition | Encoder pretraining (task-specific, documented protocol, 1 run) | Observation (extra vs single view) | FQL seeds x budget | GPU / RAM GB | Calibration-dependent |
|---|---|---|---|---|---|---|
| each | `dynamo` | `rw-dynamo-<task>-pre`: 40 epochs, seed 42 (12 / 32) | 1 RGB view (none) | 0,1,2 x 1M steps, batch 256, alpha 10 | 6 / 12 | no |
| each | `reviwo` | `rw-reviwo-<task>-pre`: 100,001 steps, seed 42 (4 / 16) | 1 RGB view (none) | same | 6 / 12 | no |
| each | `sincro` | `rw-sincro-<task>-pre`: 300k steps (50 epochs cap), seed 42 (12 / 24) | 1 RGB view (none) | same | 6 / 12 | yes |
| each | `snerl` | `rw-snerl-<task>-pre`: 2.4M iterations (official), 3 input cameras, seed 0 (16 / 20) | +2 RGB views, +3 camera poses (native multi-view) | same | 6 / 12 | yes |
| each | `mvp3d_multiview` | `rw-mvp3d-<task>-pre`: 15 epochs, seed 0 (12 / 24) | +2 RGB views, +depth, +K / extrinsics | same + `encoder.feature_norm=train_standardize` | 6 / 12 | yes |
| each | `mvp3d_single` | same encoder as above | +depth, +K / extrinsics | same + `encoder.feature_norm=train_standardize` | 6 / 12 | yes |
| each | `e2e_cnn` | none: IMPALA-small random init, trained jointly with FQL | 1 RGB view (none) | same | 12 / 24 | no |

Each task is drawer-open, mug-hanging, covering or pick-and-place (gated separately; new tasks: last section).
Frozen-condition dataset builds: 8 GB GPU / 16 GB RAM; the e2e build is CPU work (1 / 8). Pretraining resources are
the campaign's Meta-World rows of the same trainers.

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
store's M5 sets via `stages.depth.episode_frame_sets`). An empty multi-view split fails only the multi-view family; the
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
to the base gate and, for geometric consumers, the calibration hold. Every episode must have
`train_data.output_format` `s4d.realworld.train/1.1.0` or later and `qa.output_format` `rwprep-qa/3` with explicit
per-family results. Missing, older or unknown QA formats and malformed family verdicts fail closed; stage-name
overrides cannot waive these requirements. Reports without family results cannot release a consumer.

**Calibration-optional stores (2026-11-06 release fix).** When a camera's `calibrate` result is rejected, held or
missing (including all-camera rejection by the hard 0.05 m motion-coverage gate), `train_data` retains every RGB
camera's frames, timestamps, recording segments and robot sync/state. Per-camera `calibration` metadata records the
status and reasons; affected stores also expose `calibration_status` and `geometry_available`. Cameras without
usable calibration omit extrinsics (`w2c`), world-frame tracks and depth products. Robot FK points and M5 robot-surface
motion use only robot kinematics and remain available; cross-camera gripper visibility is omitted. Multi-view and
geometry readers refuse the episode, and contract checks record a multi-view family failure without a server call.
The independent single-view family can pass all its own recording, frame and robot-sync requirements.

Affected files, stage configs and summaries carry `layout: calibration_optional/1`; `manifest_fields` also carries
the marker and per-camera calibration statuses. Keep `train_data@2` and schema `1.1.0`: fully accepted episodes retain
the existing layout, fingerprint and bytes, proven against the frozen pre-change writer on identical synthetic inputs
with creation time and code provenance pinned. Existing accepted outputs are reused without re-export. Only affected
episodes acquire the new config marker and are rebuilt if an older export had silently dropped rejected cameras.

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
owners at that review): `stages.depth.configured_sets_policy` passed `train_data.max_sync_ns` to `frame_sets` as a
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

## New real-world tasks: covering and pick-and-place (2026-11-09)

The four targets in `s4d/rwlearn/tasks.py` share one learning contract, unchanged: task-specific encoders; the
100-episode gate per task; FQL alpha 10; continuous gripper; action = sent-command base-frame flange delta pose
(rotation vector) + gripper; single view = a random camera per sample, fixed through the history and the paired next
observation; SNeRL with three input cameras; one end-to-end CNN + FQL condition; FQL seeds 0/1/2; the seeded 90/10
episode split and the offline validation of the existing tasks. Reward `success(s') - 1` with success at the final
frame; every uploaded demo stays labelled successful, flagged or not. FQL needs no per-task config
(`configs/realworld/fql/base.yaml`). The test recordings are evidence, never gate or training episodes:
`test/covering.h5` is byte-identical to covering episode 0; `test/pick_and_place.h5` (recorder episode index 2) is
not a dataset episode.

**Ending check** (`outcomes.check_covering`, `check_pick_and_place`; robot data first, read-only). The FIRST hold
(gOBJ 2 runs, merged across flickers as for mug-hanging) and its release are the task event; a later grasp never
redefines it. Completed covering: the towel is grasped at a corner, laid over the can, released, and the open gripper
retreats. Completed pick-and-place: the object (banana) is grasped, put on the plate, released, open retreat.
`reversal_suspected`: ending commanded closed (> 0.5) or still holding, or a later grasp within 5 cm of the release
(towel pulled off the can / object taken back). `needs_image_review`: no hold, horizontal grasp-to-release transport
below the floor, a release across a recording pause or without a confirmed physical opening (register <= 25), several
holds elsewhere, a closure after the release, a return within 3 cm of the release after leaving it by 5 cm, an unclear
final opening (command <= 0.1, register <= 25, gOBJ 3). Events in the recordings are teleop enable / pause /
lead-limit messages, not task labels; robot data cannot show coverage or which plate was used. covering transport
floor 10 cm (observed 16.1-22.2 cm). **pick-and-place thresholds are PROVISIONAL** (`thresholds_provisional: true` in
every result): the only evidence is `test/pick_and_place.h5` (one hold 4.06-11.50 s, 22.3 cm transport, final command
0 / register 3 / gOBJ 3; frames: banana from the table onto the pink plate, a blue plate also present, open retreat).
Its 5 cm transport floor and the inherited 5 / 3 cm revisit distances must be re-validated on the arriving episodes
(including any real reset ending) before they are trusted; no destination, axis or object size is assumed.

Per covering episode, read-only snapshot 2026-11-09 22:27 KST (36 arrivals, numbering gap at 8;
`docs/realworld_newtasks_evidence.json`): every episode has exactly one hold, ends open (command 0-0.0111, register
3-5, gOBJ 3) and carries the informational `return_toward_start`; all stay in the dataset with outcome `success`.

| covering episodes | ending check |
|---|---|
| 000000-000007, 000009-000021, 000023-000036 (35) | `no_reversal` |
| 000022 | `needs_image_review`: `revisit_release_location` (robot rows 727-861, 7.27-8.61 s, frames 109-129 of every camera); the final frames show the towel still over the can, the flag is kept for the coordinator |

Frames inspected: final frames of 0-7 and 9-22 (Codex n01) and first / release / final of 23, 25, 27, 29, 31 (Claude):
the towel conceals the can, fingers open. The two test recordings: `no_reversal`. No `reversal_suspected` so far.

**Plan and registration.** The planner refuses SNeRL / 3D-MVP and their FQL conditions for a task whose scene
geometry is unmeasured (`docs/realworld_baselines.md`), so before that measurement a gate-passed new task plans 19
rows (ReViWo, DynaMo, SinCro; their datasets / FQL runs and e2e_cnn), after it the full 33, exactly the rows each
existing task got:

| Family | Pretraining | Dataset builds | FQL runs (seeds 0/1/2) | Rows |
|---|---|---|---|---|
| image (100-episode gate) | ReViWo, DynaMo | ReViWo, DynaMo, e2e_cnn | 9 | 14 |
| geometry (+ calibration hold) | SNeRL, 3D-MVP, SinCro | SinCro, SNeRL, 3D-MVP multiview + single | 12 | 19 |

Registration (proposal, not installed: `/home/ws/ws/refactor-20261109/codex/n01_out/register_realworld_newtasks_20261110.py`
+ its pure test; proposed home `runs/setup/tools/` after the release gate): one task per run, `--family image` and
`--family geometry` independently but never a partial family, ids `rwnt-20261110-<plan id without rw->`, the plan's
priorities / GPU / RAM / restarts / role, release pinned from `runs/setup/role-release.json` as read at registration,
learning inputs (`scripts/realworld_learning.py inputs --task T --ending-report R`: store.yaml, gate_report.json,
split_seed0.json, outcomes.json) staged remotely and verified against the live gate as the two existing registrars do.
Dry run by default; `--publish` / `--recover` follow the existing locked, journalled queue append.

## Native-depth 3D-MVP (2026-11-10)

The ordinary compact store contains FoundationStereo depth. Opt-in 3D-MVP runs can instead train on the exact
native RealSense depth used by the deployment client. Other conditions, ordinary store descriptors and simulation
keep their existing data path. No FoundationStereo or S2M2 runs in this stage.

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 nice -n 15 .venv/bin/python -I scripts/realworld_native_depth.py build \
  --store <store.yaml> --rig <rig-loma-drawer-open.json> \
  --out-dir <native-overlays> --out-store <store-native.yaml> --workers 4
```

Use the task's published rig: `rig-loma-drawer-open.json` for drawer-open, `rig-loma-mug-hanging-fx.json` for
mug-hanging. Run separately per task. The stage hashes each raw file and checks it against the compact episode,
checks every camera's colour-stream timestamps, then writes every frame's
`RecordedEpisode.frame(depth=True, correction=load_rig_session(...)["native_depth_correction"][cam],
alignment="rs_align").depth` unchanged as float32 metres (0 invalid). This reuses deployment alignment, colour-z,
native sensor scale, centre crop and nearest 128 resampling. It writes one atomic overlay per compact episode,
with raw/rig SHA-256, corrections, alignment, counts and timestamps; inputs are never modified. Episode processes
are bounded by `--workers`, with one BLAS/OpenMP thread each. A valid existing overlay is skipped; an existing
invalid overlay refuses and must be removed explicitly to rebuild. `--episodes task/episode_id ...` stages a subset;
the output descriptor is published only once every episode's overlay exists and verifies.

The output descriptor preserves the input keys and adds `depth_source: native_realsense`, `depth_overlays`
(one path per input `paths` entry, in the same order), and `rig_document_sha256`. `open_store` returns a compact
store whose `depth()` reads the overlay through a process-local LRU of at most `max_open` handles (default 16).
Raw identity, counts and timestamps are checked when each handle opens; replaced files are reopened and verified.
RGB, masks, intrinsics, poses, synchronization and raw action provenance still come from the original store.

Create separately named rows/artifacts with this chain:

1. Native-depth stage above.
2. `train_mvp3d.py` pretraining with `dataset.hdf5_path=<store-native.yaml>` and the existing task split/gate.
   The saved run config's `dataset.hdf5_path` is the pretraining depth provenance; no trainer change is needed.
3. `realworld_fql_dataset.py build --store <store-native.yaml> --condition mvp3d_single|mvp3d_multiview`
   with the new pretraining encoder export and the existing gate/outcomes arguments.
4. FQL training on that derived dataset with `encoder.feature_norm=train_standardize`.
5. Deployment with the matching native training contract and the same task rig's native corrections.

The FQL builder refuses a native store for any other condition; real-world pretraining gates refuse it for other
baselines. Native FQL datasets record `training_depth: native_realsense` and `native_depth_provenance` (overlay
paths, format/version, raw/rig hashes, per-camera counts and corrections). The checkpoint copies `training_depth`
only when present in the dataset; the deploy sidecar includes that source and provenance and must match a
checkpoint's recorded source. Native contracts describe the shared client depth processing. Ordinary datasets
gain no attributes and retain their existing FoundationStereo contract text. Planned rows are unchanged: this
chain is opt-in and requires new row names, output paths and native store arguments.

## FQL setting for pretrained encoders (user decision 2026-11-11 ~23:10 KST; FQLNQ)

DIAG's matched-budget diagnostic (realworld-ops/diag/DIAG-REPORT.md section 11, Codex-verified) showed that with
alpha 10 the one-step actor of a frozen-encoder policy drifts to about twice the BC error (26-30 vs 11-13 mm on
drawer-open), while `agent.alpha=10` + `agent.normalize_q_loss=true` (actor Q term scaled by
lam = 1 / mean|Q(s, a_onestep)|, `s4d/rwlearn/fql.py`) met the pre-stated rule for both tested encoders, is invariant to
each encoder's Q scale and had the best critic calibration. The user adopted it:

| Rows | Setting | Ids | Priority |
|---|---|---|---|
| every FQL run on a PRETRAINED (frozen) encoder: ours, ReViWo, SinCro, DynaMo, SNeRL, 3D-MVP single + multiview; all four tasks; seeds 0-2 | `--set ... agent.alpha=10.0 agent.normalize_q_loss=true` (`a10nq`) | `rwnq-20261111-fql-<cond>-<task>-s<k>` | the source row's (2) |
| comparison baseline | `agent.alpha=100.0` (in place) + `agent.normalize_q_loss=false` (explicit) (`a100`) | `rwa100-20261111-fql-<cond>-<task>-s<k>` | 3 (drawer-open / mug-hanging), 4 = lowest (covering / pick-and-place) |
| End-to-end CNN (`e2e_cnn`, `e2e_cnn_noaug`) | UNCHANGED (alpha 10, no normalisation) | existing ids | unchanged |

* The setting rides on `--set` overrides only. `configs/realworld/fql/base.yaml` is unchanged (`normalize_q_loss:
  false`), so every E2E row and every existing row keeps exactly its configuration. Existing runs, checkpoints and
  datasets are preserved as recorded results; the new settings have new ids (`derived_from` names the alpha-10 source
  row; ours rows name their `protocol_template`).
* Registrars: `runs/setup/tools/register_fql_nq_20261111.py` (reviwo, sincro, dynamo, mvp3d single / multiview on
  drawer-open + mug-hanging), `register_fql_nq_followup_20261112.py` (`--group snerl`: from the paused alpha-10 SNeRL
  rows, behind the SNeRL feature-check hold; `--group ours`: datasets + FQL rows behind the ours feature-check hold, on
  a release that defines the condition), `register_realworld_newtasks_20261110.py` (covering / pick-and-place: each
  frozen-encoder plan row becomes its a10nq row + a100 comparison; e2e rows unchanged). All are dry runs by default and
  publish only under the coordinator's admission pauses; the remote driver ignores `priority` for role rows (queue
  order), so the a100 pauses are what keep the comparison behind the a10nq rows there.
* Policy staging (`ops/policy_serving/staging.py`): rwnq rows are served under distinct ids
  `<cond>-<task>-s<k>-a10nq` (flag `fql_setting_a10nq`; the legacy `<cond>-<task>` aliases keep naming the alpha-10
  ids), a setting row whose own `config.yaml` does not carry its setting fails criterion b, and rwa100 rows are staged
  (`stage` / `hook`) only with `--allow-comparison`. Robot-PC selection by condition / task / seed then matches two
  entries and asks for an explicit `--policy` (fail closed).
* Monitoring: `realworld-ops/fqlnq/tools/fql_monitor.py` (Q scale, lam = -actor/q_loss / actor/q, critic stability,
  validation one-step vs BC-flow error; flags fixed in its docstring).

## FQL condition `ours` (2026-11-12; FQLNQ)

`ours` reads the real-world-pretrained Original Splatter4D export (`rwcal-20261111-ours-<task>-pre` ->
`-export/encoder.pt`) exactly as the simulation downstream protocol does (`s4d.rl.encoders.Splatter4DStateEncoder`):
ONE camera's 3-frame RGB history (frame spacing 1 camera frame, inside the pretraining strides 1/2/3; DEVIATIONS D7) ->
one joint state (`policy_state`, 1 slot x 256), stored as a sequence feature like SinCro's and read by the same
`SmallPostEncoderMLPHead`. Single-view protocol (random camera per sample) as the other single-RGB conditions; consumer
`fql_ours` (the geometry class of the ours encoder; each camera's own timeline, so no multi-view family).
Datasets `rwcal-20261111-fqldata-ours-<task>` take the SinCro dataset row of the task as protocol template (same store,
gate report, outcomes, footprint; remote only). FQL rows are a10nq + a100 on the SinCro FQL rows' protocol (split,
seed, 1M steps, batch 256, intervals). Before launch, each ours dataset passes the same feature check as SinCro / SNeRL
(offset / signal and LayerNorm spread, `ops/policy_serving/criteria.py feature_conditioning`); whether a failing
dataset is standardised (`encoder.feature_norm=train_standardize`) is a decision for the user. Trained ours
checkpoints get the condition's deploy contract (`InputSpec.from_meta`); the labelled serving fixture is unchanged.
