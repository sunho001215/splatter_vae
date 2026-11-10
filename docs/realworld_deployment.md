# Real-world FQL deployment: scientific inference contract

Scope: the science half of network policy inference (user directive 2026-11-09 ~22:15). A CPU robot-side client
preprocesses camera frames exactly as the training store did; a GPU server runs the frozen encoder + FQL policy and
returns the action bound to the measured pose of its observation. This document covers `s4d/rwlearn/deploy/`
(no sockets, threads, serialization or robot I/O); transport, sessions on the wire and the server process are in
`docs/realworld_policy_serving.md`. Nothing here commands a robot.

| module | side | role |
|---|---|---|
| `contracts.py` | both (torch-free) | `InputSpec` from checkpoint meta + training contract; `bind` per session; JSON round trip |
| `preprocessing.py` | client (numpy + Pillow) | `preprocess_frame`: the training-store crop/resize of RGB, depth, validity, K |
| `realsense.py` | client | live `RealSenseRGBD` (`rs.align`, optional `pyrealsense2`), recorded alignment, rig session loader |
| `replay.py` | client (dry run) | `RecordedEpisode`: recorded H5 through the client path + the training state binding |
| `history.py` | server (torch-free) | `Frame`, `HistoryWindow` (per-session cache, run rule), `ObservationWindow` |
| `errors.py` | both (torch-free) | typed refusals (`DeployRefusal` subclasses of `ValueError`) |
| `policy.py` | server | `DeployPolicy` over `OfflinePolicy` (one interface, all conditions), timings |
| `fixtures.py` | tests | tiny labelled FIXTURE policies of every condition |

Zero-behaviour refactors for sharing: `rwlearn/observation.py` (Condition table, `FRAME_STACK`; `fql_data` and `pixel`
re-export them), `realworld/frame_preprocessing.py` (`nearest_resample`, `training_depth`, `IMAGE_SIZE`; `train_store`
imports them), `actions.measured_proprio` (the conversion `OfflinePolicy.act` already used).

## Input contract per condition

Derived, never guessed: trained camera ids = the FQL dataset's per-episode store cameras (sorted serials; the
multi-view order of `store.sync_index`); history = `transitions.history_indices` / `store.history`; period and action
normalisation = the checkpoint's action contract; state binding = `actions.step_table` defaults. Old checkpoints do not
carry the camera ids, so the server reads them from the FQL dataset once or from a sidecar
(`check_realworld_deploy.py contracts` -> `<run>.deploy_contract.json`, checked against the checkpoint's condition,
export sha256, action contract and image size).

| condition | views (session choice) | modalities | calibration | feature | status |
|---|---|---|---|---|---|
| dynamo | 1 of the 3 trained cameras | rgb | - | per-frame, 3-frame stack | trained (drawer, mug) |
| reviwo | 1 of 3 | rgb | - | per-frame, 3-frame stack | trained (drawer, mug) |
| e2e_cnn | 1 of 3 | rgb (9-channel stack) | - | joint CNN | trained (drawer) |
| mvp3d_single | 1 of 3 | rgb + native depth + validity | K, c2w of the view | per-step, 3-step stack | trained (drawer, mug) |
| mvp3d_multiview | all 3, order 070988, 073201, 076054 | rgb + native depth | K, c2w of 3 views | fused per step | trained (drawer, mug) |
| sincro | 1 of 3 | rgb | - | one state per 3-frame history | FIXTURE only (no FQL policy) |
| snerl | all 3 | rgb | K, c2w of 3 views | per step | FIXTURE only (no FQL policy) |
| ours | fixture protocol (1, rgb) | rgb | - | sequence | FIXTURE only (deferred) |

Common to all: 128 x 128 training grid; history 3 consecutive camera frames (multi-view: synchronised steps) at the
trained period 1/15 s (each interval within 1 % = `actions.PERIOD_TOLERANCE`); a gap (seq jump or > 1.5 periods,
`store.runs_from`) starts a new run; proprio = 10-D [flange xyz, R[:,0], R[:,1], (register - 3) / 227] from the
measured flange pose (panda_link8, base frame, xyzw quaternion) and Robotiq register; action = normalised
[dp / pos_scale, rotvec / rot_scale, 2c - 1] (base-frame left increment, continuous gripper), decoded by
`actions.to_robot_command` (goal pose, closure, register, recorder lead envelope). Multi-view sets: mutual nearest frames
with robot-surface motion <= 5 mm (M5, `store.MAX_SET_MOTION_M`); the caller supplies that evidence per step.
`InputSpec.train_deploy_differences` lists every known train/deploy input difference of the condition.

## History and binding

`HistoryWindow.assemble(end_ids)` returns exactly 3 frames per camera ending at the given camera frames or raises
(missing frames never compress the spacing). `pad_run_start=True` applies the training rule `store.history`
(frame-stack reset at a run start: [f0, f0, f0], [f0, f0, f1]; after a gap the new run is padded the same way;
multi-view runs start once every camera is inside its run); the result reports `padded`. `reset()` starts a new run
explicitly; `DeployPolicy.reset_session` requires a new session id and a fresh history.

The measured pose / register passed to `infer` must be the samples bound to `window.observation_time_ns` (multi-view:
latest constituent frame): latest robot sample at or before it, at most 15 ms old; latest gripper sample, at most
100 ms old (`actions.step_table`). `obs_ids = window.binding(ref, pose_time_ns=, gripper_time_ns=)` carries those
times and the frame ids; `infer` refuses any mismatch. The returned goal pose is relative to that measured pose and
must never be re-applied to a newer pose.

## Client preprocessing (bit-exact with the training store)

`preprocess_frame(rgb, depth=, valid=, K_native=, native_correction=)`: centred square crop (min(H, W), integer
offsets) -> 128 x 128 with PIL BILINEAR (antialiased) for RGB; depth x native correction, invalid -> 0
(`training_depth`), nearest native pixel per output-pixel centre (`nearest_resample`); validity = depth > 0;
K = crop affine applied to the continuous-centre colour intrinsics. Imports numpy, Pillow and the crop helpers only
(no torch / scipy / h5py; tested).

Evidence (`check_realworld_deploy.py preprocess`, released calrel-6c3b9ec training-store episode mug-hanging-000001,
raw sha256 def246e2..., all 623 frames of 3 cameras; `/home/ws/ws/realworld-ops/serving/evidence/`):
RGB client - store max |diff| 0, K 0.0, c2w (rig document) - store 0.0, and the training FoundationStereo depth pushed
through the CLIENT resampling (x the store's per-camera FS scale) - store depth 0.0 m. The recorded colour is the
recorder's JPEG decoded by PIL, so recorded replay is bit-exact; live frames are uncompressed colour (see differences).

## RealSense depth (3D-MVP single and multi-view)

Deploy contract (agent DEP, `/home/ws/ws/realworld-ops/serving/DEPTH-COMPAT.md`, verified against real librealsense
2.59.1 on recorded z16 and bitwise against the release `sensor_depth` stage and the compact store):
`rs.align(color)` -> depth-camera z16 on the colour grid -> 0.1-6.0 m gate (the stage's range, on the device z as the
stage gates before its warp) -> **exact conversion to colour-camera z** (`rs.align` writes the depth-camera z of the
source pixel; per colour pixel centre `z_c = (z_d - t_dc,z) / (e3 . R_dc . ray_c)`, `T_dc = inv(T_color_depth)`;
`depth_z_to_color_z`) -> x the validated native correction -> the training centre crop + nearest resampling,
K128 = crop affine x continuous K (librealsense ppx/ppy + 0.5).

Live (`RealSenseRGBD(profile, correction=...)`): `rs.align` created once; `T_color_depth` from the device extrinsics
(`depth.get_extrinsics_to(color)`, column-major rotation) unless given; the per-pixel colour-z denominator is computed
once per intrinsics; bgr8/rgb8 -> RGB; distorted colour streams refused. Both SDK timestamps, domains and frame numbers
are returned; the client maps them to the common host monotonic clock (as the recorder did) before building `Frame`s.
Recorded replay (`RecordedEpisode.frame(..., alignment="rs_align")`, default) runs DEP's validated numpy emulation of
`rs.align` (`emulate_rs_align`; interior bitwise equal to real librealsense, differences only on border rows/columns
-> output row 127; the port equals DEP's tool bitwise on 18 recorded frames, `evidence/rs_align_port_vs_dep.json`)
followed by the same colour-z conversion; `alignment="sensor_depth"` uses the release stage kernel instead
(`align_recorded_depth`, already colour-camera z; the registration the native scale was fitted on).

Depth-scale correction: the calibration release (6c3b9ec) estimates a per-camera, per-depth-source scale jointly with
the rig pose (`pipeline.DEPTH_SCALE_RULE`); the training store multiplied FoundationStereo depth by
`depth_scale['foundation_stereo_depth']`. Deployment multiplies native depth by the SAME document's
`depth_scale['sensor_depth']` (`load_rig_session`; drawer 0.9989 / 1.0003 / 1.0172, mug 0.9974 / 0.9983 / 1.0172 for
070988 / 073201 / 076054; without it 076054 reads ~1.9 % short); an FS factor is refused
(`validate_native_correction`), identity is explicit and labelled. Cameras are accepted with the release's effective
status (`calib.rig.rig_camera_status`, incl. the mug fixedness exception). Training data, weights and normalisation are
unchanged; the InputSpec lists the difference in `train_deploy_differences`.

Residual train (FS) vs deploy (native) difference. DEP (16 release episodes, inside the 3D-MVP bounds): median ratio
0.9938-1.0020 with the correction, median |dz| 4.3-7.2 mm, 12-20 mm in the depth-edge band (20-23 % of pixels,
23-43 % of them > 25 mm), native holes 2.0-4.7 % where FS has depth, native 4-7x noisier over time. Cross-check here
(mug-hanging-000001, whole 128-px crop incl. far background, all frames, rs.align path): |dz| median 9.9 / 8.1 / 6.8 mm,
p95 121 / 121 / 89 mm, signed median -0.4 / -4.7 / -2.7 mm; native valid 0.926 / 0.927 / 0.912, FS-only 6.5-7.7 %
(sensor_depth registration: 9.7 / 7.7 / 6.5 mm). Effect on the trained 3D-MVP FQL policies: the same windows with the
store's FS depth vs native depth change the action by <= 3e-7 (normalised) - see "Findings".

## DeployPolicy

`DeployPolicy(checkpoint, device=, export_path= | export_root=, dataset_path= | training=)` loads once
(`export_root` / the repo-relative meta `export_path`, e.g. a staged checkpoint root mirroring `runs/`);
`reset_session(...)` binds the session; `warm_up()`; `infer(window, measured_pose7, gripper_register, obs_ids,
noise=None)` runs `DeployPolicy.act` under `torch.inference_mode()` and adds `measured_pose7`, `obs_ids`,
`frame_timestamps_ns`, `metadata` (condition, fixture, depth provenance, feature precision, padded) and `timings`
(validation / encoder / FQL / policy / total ms; encoder and actor timed with forward hooks and CUDA synchronisation on
GPU; FQL includes the trainable projection head). `act` reuses `OfflinePolicy.features` / `act_from_features` /
`act_from_pixels` and, for frozen encoders, rounds the features to **float16** before the agent, exactly as training
saw them (`fql_data.encode_timeline` stores `.to(torch.float16)`, the FQL encoder reads `.float()`;
`InputSpec.feature_precision`). `OfflinePolicy.act` itself is unchanged and feeds unrounded float32 features.
Checkpoints trained with the frozen-feature standardisation (3D-MVP reruns from 2026-11-10; `docs/realworld_fql.md`)
carry its mean / std as encoder buffers inside `policy.pt`: the same `OfflinePolicy` load path rebuilds and
sha256-checks them and the encoder applies `(f16(f) - mean) / std` after the float16 rounding, exactly as in training
(test: `infer` == `act` == the adapter on the explicitly standardised features, bitwise). The client contract
(`InputSpec`) is unchanged; the deploy-contract sidecar carries the checkpoint's `feature_norm` record and a sidecar
whose record differs (or is missing) is refused, so regenerate sidecars (`check_realworld_deploy.py contracts`) for
the rerun checkpoints.
`noise` (1, 7) float32 fixes the FQL noise (direct vs network equality); None uses the agent's own generator.
`reset_session(concurrent=True)` keeps other live sessions on the same loaded model (`end_session`, `live_sessions`);
by default a reset replaces them. Every refusal is a typed `DeployRefusal(ValueError)` (`errors.py`:
`IncompleteHistory`, `IrregularHistory`, `Unsynchronized`, `BindingRefused`, `SessionMismatch`, `ContractViolation`)
for the serving layer's error codes.

Equality evidence (`check_realworld_deploy.py equality`, CPU, recorded windows through the client path, identical FQL
noise): `infer` == `DeployPolicy.act` (training precision) bitwise on every window for dynamo, reviwo, e2e_cnn,
mvp3d_single, mvp3d_multiview (drawer-open-000000, 6 windows each) and mvp3d_single, mvp3d_multiview, reviwo, dynamo
(mug-hanging-000001, 8 windows, M5 sets from the store); repeated calls deterministic; proprio == the FQL dataset's
proprio exactly at every window. Versus the unrounded `OfflinePolicy.act`: max |d action_norm| 1.1e-4 / 1.6e-4 (dynamo
/ reviwo drawer), 5.7e-5 / 1.2e-4 (dynamo / reviwo mug), <= 2.9e-7 (3D-MVP), 0 (e2e_cnn). Versus the FQL dataset's
own float16 features + proprio: ReViWo mug features bitwise equal and action identical; DynaMo 68-74 % of features
bitwise (CPU vs the dataset's GPU), action <= 1.4e-3. SinCro, SNeRL and ours: labelled fixtures only
(`tests/test_rwlearn_deploy.py`).

## Findings and open questions

* JPEG: training RGB went through the recorder's JPEG (q90); live frames do not. Not emulated (encoder unknown);
  emulating it would be an explicit client option.
* Feature precision: FQL trained on float16-stored frozen features; `DeployPolicy` reproduces that rounding,
  `OfflinePolicy.act` does not (action difference <= 1.6e-4 normalised; reported per window in the evidence).
* 3D-MVP: frozen features have a large constant offset (|f| mean ~70, max ~440, per-dim temporal std ~0.45) and the
  trained FQL policies ignore them (zeroed or time-shuffled features change the action by <= 6e-6, vs 0.23-0.52 for
  DynaMo / ReViWo). The depth-source difference is therefore numerically irrelevant for these checkpoints, but the
  3D-MVP baseline's use of vision needs a separate review (out of scope here).
* Live multi-view M5 needs the robot-surface motion across each set (URDF FK of the joint states at the frame times);
  replay takes it from the store. The client/server helper for live runs is open.
* drawer-open has no local released store: its multi-view evidence windows use a labelled flange-motion PROXY for M5.

## Commands

```bash
cd <repo>; export PYTHONPATH=. CUDA_VISIBLE_DEVICES=
python scripts/check_realworld_deploy.py --out contracts.json contracts --runs-root runs --out-dir <dir> <run>...
python scripts/check_realworld_deploy.py --out pre.json preprocess --raw <episode.h5> \
  --store <train_data/<ep>/episode.h5> --rig <rig document> --root <processed root> --episode <ep>
python scripts/check_realworld_deploy.py --out eq.json equality --runs-root runs --raw <episode.h5> \
  --rig <rig document> [--store <train_data episode.h5>] --contracts <dir> <run>...
```
