# Real-world offline FQL on frozen baseline representations

Branch `realworld-fql-20261106` (agent FQL, 2026-11-06). This branch covers design and implementation only. No
training was run, and nothing here talks to the robot. The encoders come from the real-world baseline pretrainings
(`docs/realworld_baselines.md`).

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

**Camera randomisation** (`OfflineTransitions.sample`). Batches are uniform over (camera timeline, step)
transitions, so the viewpoint is drawn per training sample. Every index of the observation history and of its paired
next observation lies in that one camera's timeline, which the test asserts. Single-view timelines are each camera's
own native frames (observation, proprio and action share that camera's timestamps). Multi-view conditions use the
synchronised steps. Each step is timestamped at its LATEST constituent frame, so no camera image is newer than the
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
| Source | recorded `action/flange_pose` = the command actually SENT (`sent`=`engaged`=1 on every used row) | `action/flange_pose` stays within `max_position_lead` = 0.040 m of the measured `robot/flange_pose` on every row. Rows flagged `lead_limited` have the raw operator target `action/flange_target` about 4 cm further. `limit_lead` reproduces the sent command from target + measured pose (median < 5 mm, tested). **Inferred; the file does not document `action/flange_pose`. Please confirm.** |
| Point / frame | flange (`panda_link8`) origin; increments in the base frame `panda_link0` | `T_flange_tcp` is the identity in every recording, so TCP == flange (checked per episode; otherwise refused). `conventions_json`: all recorded poses are flange poses; base +x forward, +y left, +z up. Base-frame deltas are the standard OSC-delta convention and are view independent. |
| Delta | `a = [p_goal - p_k, rotvec(R_goal R_k^T)]`, goal = the step's command, reference = the MEASURED pose at the latest robot sample <= tau_k | The pose is Markov in the policy's observation (unlike command-to-command increments, which need the unobserved previous command). Execution: `goal = exp(r) R_k, p_k + dp`. |
| Rotation | rotation vector (axis-angle, rad) | Increments are small (median 6 deg, max 19.5 deg), so a minimal, continuous representation suffices. The 6D representation is used only for the absolute orientation input. |
| Control interval | one policy step = one camera frame (nominal 1/15 s; every camera's measured median period is checked within 1 %; measured 66.715 ms). The ~6.7 commands at the 100 Hz loop inside a step are aggregated by zero-order hold at the step end: the last command strictly before tau_{k+1}. | The controller tracks the latest command. The next observation is taken at tau_{k+1}. A step is invalid if any of its commands was not sent or engaged, if it spans a recording pause, if it is longer than 1.5 periods (dropped frame), or if either endpoint lacks a fresh state. A fresh state means a robot sample <= 15 ms old and a gripper register sample <= 100 ms old, both in the same segment. |
| Units / normalisation | m, rad. Policy action = `[dp / 0.0700 m, r / 0.499 rad, 2c - 1]` | The scales come from the recorder's own teleop limits: lead 0.04 m + 0.3 m/s x 1.5 periods; 20 deg + 1.5 rad/s x 1.5 periods. Every recorded command maps inside [-1, 1]: the pose dimensions reach at most 0.68. A policy output can never ask for more than the teleop envelope. |
| Gripper | command closure c in [0,1] (0 open, 1 closed), Robotiq position register = 3 + 227c (`conventions_json`) at the same row; policy g = 2c - 1, continuous | The command ramps over ~0.2 s, so it is mostly binary. Mug-hanging is closed 48 % of steps; drawer-open never closes (c <= 0.5). The measured register stalls at the object, e.g. 210 on the mug handle, which is why proprio uses the measured closure. Binarise at 0.5 for execution only if wanted (open decision). |
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

**Proposal (implemented; `reward="ogbench_final_success"`).**

- A transition is a valid policy step (see the contract above). Transitions never cross a pause or a dropped frame.
  At each segment start the frame history restarts.
- Outcomes are explicit inputs. The builder refuses without per-episode labels (`--outcomes`) or the explicit
  assumption `--assume-final-success`; the planned rows carry the assumption until labels exist.
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
- `scripts/realworld_learning.py plan` emits rows after the gate passes:
  - `rw-<b>-<target>-pre`;
  - `rw-fqldata-<cond>-<task>-enc<target>`, which depends on the encoder row;
  - `rw-fql-<cond>-<task>-enc<target>-s{0,1,2}`, which depend on the dataset row.

  All are remote rows with `role: realworld_fql` or `role: realworld_pretrain`. Nothing is registered.

## Open decisions for the user

1. **Success labels.** Are all 100 recordings per task successful demonstrations, with success at the final frame?
   If not, provide per-episode outcomes (or success times), and say whether failed attempts should be included as
   negatives.
2. **Semantics of `action/flange_pose`.** This doc infers it is the lead-limited sent command. Please confirm with
   the recorder code.
3. **`alpha`.** The official default is 10. The paper tunes it per environment. Proposal: a small sweep {3, 10, 30, 100} per task on one encoder. There is no simulator, so selection
   needs offline metrics plus a real-robot evaluation protocol, which is outside this branch.
4. **Gripper at execution.** Continuous (default; exact round trip) or binarised at 0.5.
5. **FQL encoder for each task.** Use the per-task encoder (default) or the jointly pretrained one
   (`plan --fql-encoder joint`).
6. **Random-init encoder arms** (the campaign's pretrained-vs-random check). Not planned by default; they would use each
   trainer's `encoder_init.pt` where it writes one.
7. **Action chunking.** None, matching official FQL: one action per 66.7 ms step. QC-FQL-style chunking would be a
   deviation.

## Readiness checklist

| | done | pending |
|---|---|---|
| FQL port | exact + full-config equivalence vs official | - |
| action contract | implemented; round trip exact on 105 recordings; fixture tests | user confirmation of `action/flange_pose` semantics |
| rewards / terminals | implemented (two variants) | success-label decision |
| datasets / camera randomisation | implemented; pipeline tests on 3 encoder kinds | compact store merged (format branch) |
| training / inference | trainer, offline validation metrics, `OfflinePolicy`, replay | GPU suite on the merged release |
| launch | gate + plan rows, role-classified | gate passes (0/100 preprocessed per task at 03:10); encoders pretrained |

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
- **Kept as an open decision: SNeRL inputs.** SNeRL's 3-of-3 input views versus 2 inputs with the third held out
  (`docs/realworld_baselines.md`).
