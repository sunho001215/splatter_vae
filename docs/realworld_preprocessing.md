# Real-world preprocessing (Franka Panda + Robotiq + three RealSense rigs)

Branch `realworld-preproc-20261031`, package `s4d/realworld/`. This is a separate, self-contained pipeline. It
is not the DROID loader, and no existing module imports it. A test enforces that rule
(`test_existing_code_does_not_import_realworld`).

**Status in one line:** the numerical core is validated on synthetic fixtures only. Those parts are the contract,
stereo conventions, rectification, calibration estimator and gates, 2D-to-3D lifting, and export to the training
contract. No real recording, real robot asset or real upstream model has been run. See
[realworld_preprocessing_STATUS.md](realworld_preprocessing_STATUS.md) and the
"Validated vs. awaiting real data" table at the end.

```
adapter (format-specific, read-only)  ──>  ingest ──> stereo_depth ──> calibrate ──> track2d ──> lift3d ──> export
                                            │          (FoundationStereo  (visual init +   (AllTracker   (same-frame    (DROID-track
                                            │           worker)            robot-grounded   worker)       depth + calib) cache layout)
                                            │                              refinement)
                                            └──> optional, baseline-only: sam3_masks (SAM 3 worker), dino_snerl (SNeRL DINO teacher)
```

## 1. On-disk layout and restartability

```
<root>/episodes/<episode_id>/normalized/episode.json          contract document (schema s4d.realworld.episode/1.0.0)
<root>/episodes/<episode_id>/normalized/arrays.npz            robot trajectory + per-stream timestamps
<root>/episodes/<episode_id>/normalized/frames/<stream>.h5    frames exactly as recorded (dtype/channels untouched)
<root>/episodes/<episode_id>/derived/<stage>/...              stage outputs
<root>/episodes/<episode_id>/derived/<stage>/_COMPLETE.json   completion record (fingerprint, config, inputs, model identity)
<export_dir>/{scratch,dinov2}.h5 + manifest.json              training cache (DROID-track layout)
```

The original recording is never written. Adapters only read it, and the provenance stores each source path with
its sha256.

`s4d.realworld.io.run_stage` makes every stage idempotent and restartable:
- The fingerprint is a hash of the stage version, its config and the fingerprints of its inputs.
- If the record exists with the same fingerprint, the stage is skipped.
- Otherwise the stage is built in a fresh `.<stage>.building.*` directory, which is swapped in atomically. The
  record is written last.
- An interrupted build leaves the previous complete result untouched (tested).
- Work runs per episode, and frames are processed in bounded chunks: `chunk`, tracker `window`/`step`.

## 2. Normalized episode contract (`s4d/realworld/contract.py`)

`SCHEMA_VERSION = "s4d.realworld.episode/1.0.0"`. The dataclasses are:

| Dataclass | Fields |
|---|---|
| `Intrinsics` | width, height, fx, fy, cx, cy, skew, `pixel_convention` (`integer_centers` as librealsense/OpenCV, or `continuous`), `distortion_model` + coefficients (OpenCV plumb_bob/rational, RealSense model names kept verbatim, Kannala-Brandt), `source`. `K()` converts to the repository's continuous convention. |
| `CameraStream` | `stream_id`, `rig_id` (one RealSense), `eye` (left/right/mono), `modality` (`rgb` or `infrared`), `channels`, `channel_order` (RGB/BGR/GRAY), dtype (uint8/uint16), `rectified` (True/False/None = unknown), serial, sensor name, timestamp source, IR emitter state. |
| `StereoGeometry` | per rig: `status` (`metric`/`missing`/`unverified`), `T_right_left` (left-camera points into the right camera, metres), `baseline_m`, source. |
| `RobotDescription` | arm (`franka_panda`), exact `gripper_model`, composed URDF path, mesh root, base/flange/gripper frames, `T_flange_gripper` mount, gripper-state semantics, driven gripper joint, asset hashes. |
| `RobotTrajectory` | int64 ns timestamps, joints (T,7) rad, gripper state (T,), optional recorded EE pose plus its frame (cross-check only), extra state with units. |
| `Provenance`, `EpisodeMeta` | clock, `assumptions` (every adapter guess awaiting confirmation), conventions. |

Conventions (also written into every episode):
- Units are metres, radians, and int64 nanoseconds on one declared clock.
- Cameras use OpenCV axes.
- `T_a_b` maps points from frame b into frame a.
- The world frame is the robot base `panda_link0`, so `T_cam_base` is the same thing as `w2c` in `s4d/data/contract.py`.
- Pixel coordinates are continuous: the centre of the top-left pixel is (0.5, 0.5).
- Depth is metric camera-z with 0 meaning invalid.
- Disparity is `x_left - x_right`.

`validate_episode` returns errors and warnings.

Errors make the episode unusable. They include:
- infrared stored with 3 channels ("replicated grayscale is not rgb");
- RGB without 3 channels;
- non-monotonic timestamps;
- unknown distortion or pixel convention;
- a stereo baseline that disagrees with the transform;
- a stereo pair with mixed modalities or sizes;
- non-rigid mounts;
- wrong joint shapes;
- an EE pose without a frame.

Warnings disable stages explicitly. Examples are missing stereo geometry ("never invented"), unknown
rectification, and an unknown gripper or missing URDF (calibration refuses to run).

## 3. Adapter guide (`s4d/realworld/adapters/`)

`EpisodeAdapter` is the only code that knows a format. It has three methods:
- `list_episodes()`
- `read_meta(id) -> EpisodeMeta`
- `iter_frames(id, stream, start, stop)`, which yields `(index, timestamp_ns, array)` exactly as recorded.

There are three implementations:

- **`template_realsense`** (`template.py`, config `configs/realworld/template_adapter.yaml`). No layout is
  invented: each method raises `NotImplementedError` naming the open question. The module docstring is the
  checklist for the real adapter:
  - which RealSense streams were recorded: IR left/right in Y8 or Y16 (rectified or not), and/or the colour sensor;
  - the IR emitter state;
  - `rs2_intrinsics` and their pixel convention;
  - the IR1-to-IR2 `rs2_extrinsics`;
  - the colour-to-depth extrinsic if RGB is used;
  - the timestamp domain and how it relates to the robot clock;
  - joint units and clock, gripper-state semantics and the exact Robotiq model;
  - which frame the EE pose refers to;
  - the URDF assets and the measured mount.
- **`synthetic`** (`synthetic_adapter.py`). Ray-cast textured boxes (floor, table, wall and a moving object)
  seen by three rigs:
  - two rigs have metric infrared stereo;
  - one rig has `status="missing"`, to exercise explicit reporting;
  - optionally a rotated right camera, so that rectification has to happen;
  - a stand-in Panda kinematic chain (official joint origins) with primitive link geometry and a two-finger
    mimic gripper. This is NOT the real Panda or Robotiq meshes.

**Writing the real adapter:**
1. Subclass `EpisodeAdapter` and decorate it with `@register_adapter`.
2. Fill `read_meta` from the recorded metadata.
3. Put every guess into `assumptions`.
4. Hash source files in `source_files`.
5. Run `python -m s4d.realworld.cli validate --config <cfg>` until there are no errors.

Nothing downstream should need to change, except possibly adding a distortion model. For example, RealSense
`inverse_brown_conrady` is declared but not implemented for undistortion, and the stage refuses it explicitly.

## 4. Stages

### 4.1 `ingest` (`pipeline.ingest`)
- Validates the contract and writes the normalized copy: JSON, NPZ, and per-stream HDF5 with lzf compression and
  one chunk per frame.
- Checks every frame's shape, dtype and timestamp against the metadata.
- The record carries the validation warnings and the adapter assumptions.

### 4.2 `stereo_depth` (`pipeline.stereo_depth`, `stereo.py`, `geometry.py`)
- **Rectification.**
  - Streams flagged rectified must carry no distortion, and K is used as is.
  - Unrectified streams are rectified with Bouguet's method (`geometry.stereo_rectify`: half-rotations, baseline
    aligned with +x, shared `K_rect`). Then `rectify_map`/`remap` undistort and rotate.
  - Rectified images are stored next to the depth, and all later stages work on the rectified-left grid.
  - Unknown rectification is an error.
  - Rigs whose stereo geometry is not `metric` are skipped with the reason recorded. Metric scale is never invented.
- **Depth.** `z = fx * B / (d - doffs)`, where `doffs = cx_left - cx_right` (0 after Bouguet with a shared K).
- **Validity.** The value must be finite, satisfy `d - doffs > min_disparity`, and the matched pixel must lie
  inside the other image:
  - left reference: `x - d >= 0`, the same rule as FoundationStereo's `remove_invisible`;
  - right reference: `x + d <= W`.
  - Depth is 0 wherever the result is invalid, and the `valid` mask is stored.
- **Reference images.**
  - FoundationStereo returns disparity on the left image.
  - The right-image depth uses flip-and-swap (`stereo.flip_and_swap_right_disparity`): feed `flip(right)` as left
    and `flip(left)` as right, then flip the result back. With x' = W - x, the magnitude `x_l - x_r` is preserved,
    and so is doffs.
  - A naive input swap without flipping asks for negative disparities. The tests show it fails, while
    flip-and-swap matches ground truth.
- **Resize, crop and flip** are `PixelAffine` maps in continuous coordinates.
  - `apply_K` gives the intrinsics after any chain of them.
  - Depth is resampled nearest-neighbour (`warp_affine(order=0)`), and images bilinearly.
  - `rescale_disparity` converts disparity measured on a resized copy back to the original grid: x scale only;
    crops shift both eyes equally.
- **Infrared input.** FoundationStereo expects 3-channel images, so IR pairs are channel-replicated as model input
  only, and the stage record says so. The stored streams stay 1-channel infrared.

### 4.3 `calibrate`: per-episode, robot-grounded extrinsics (`calibration.py`)

The method is adapted from DepthWorld ([paper](https://arxiv.org/abs/2610.08780),
[code](https://github.com/Jai2500/depthworld), [project page](https://jaibardhan.com/depthworld/)). Each camera
has one constant `T_cam_base` per episode; poses may differ between episodes. The robot model is the full
articulated URDF: arm plus the actual gripper driven by the recorded gripper state. The EE pose alone is never
used.

1. **Visual evidence.** 2D observations of robot landmarks (`landmark_links`, URDF link origins) per frame, in
   continuous coordinates of the rectified-left image. The 3D positions come from forward kinematics. They are
   supplied by a *landmark source*:
   - an annotation file (`cli.AnnotationLandmarkSource`), for example manual clicks or a marker detector;
   - in the future, a render-and-match worker, the counterpart of DepthWorld's LoMa-G URDF-render-to-image
     matching. **Not implemented.**
2. **Visual initialisation.**
   - RANSAC PnP: normalised DLT, then a Cauchy-robust reprojection refinement.
   - When metric stereo depth exists, also a RANSAC Kabsch fit of depth-lifted landmarks (bootstrap depth, no scale
     estimation).
   - The candidate with more inliers and lower reprojection error wins.
   - A previous episode's pose may be used only as a fallback prior (`T_init`).
3. **Robot-grounded refinement** (`scipy.optimize.least_squares`, Cauchy loss over the 6-DoF extrinsic):
   - landmark reprojection residuals, scaled by `cauchy_px` (5 px) and weighted by confidence;
   - robot-surface depth residuals: URDF surface points (meshes or primitives) projected and compared with bilinear
     stereo depth, scaled by `cauchy_depth_m` (10 mm, DepthWorld's c3d). Two gates apply:
     - self-occlusion uses the robot's own z-buffer (back and side surfaces would otherwise bias the result; this
       was measured);
     - a one-sided occlusion test keeps in-front residuals and drops points hidden behind something closer.
   - The gates are recomputed over `outer_iters` outer iterations.
   - An optional robot mask (for example from SAM 3) further gates the depth points and gives a mask hit-rate
     diagnostic.
4. **Diagnostics and gates.** Nothing is silently accepted: `status` is `accepted` only if every gate passes, and
   the reasons are listed otherwise. Each item below is reported, and the ones with thresholds act as gates:
   - per-frame robot-in-view fraction and landmark count, giving "insufficient robot visibility";
   - motion coverage: principal extents of the flange path and the number of distinct poses ("poor motion coverage");
   - optimiser status ("optimisation failed");
   - Jacobian condition number;
   - covariance standard deviation of rotation and translation ("uncertain extrinsic");
   - reprojection and robot-depth medians on the fit frames AND on held-out frames (every 5th visible frame);
   - landmark inlier fraction;
   - init-to-final change;
   - robot-state sync error;
   - URDF FK versus the recorded EE pose (`fk_vs_recorded_ee`).
   - Overlay PNGs show the projected robot surface (green), predicted landmarks (blue) and observed landmarks (red).
   - The covariance is a lower bound when there are systematic model errors.

**Simplifications versus DepthWorld.** This is not the paper's method, and we do not claim the full multi-episode
optimisation.

| DepthWorld | Here |
|---|---|
| One factor graph per physical robot over ~1k–9k episodes. Per-episode ext1/ext2 corrections, plus a shared wrist hand-eye (6 DoF) and joint offsets δq (7 DoF). GPU Levenberg-Marquardt with a Schur complement. | One camera of one episode at a time. 6 DoF only. No hand-eye, no joint offsets, no wrist camera. |
| Initial calibration from factory extrinsics, S2M2 stereo depth (FoundationStereo was tested and rejected there), RoMa-v2 ext-ext correspondences with MAGSAC, and two-pass bundle adjustment. | Initialisation from robot landmarks (PnP / depth Kabsch). Depth from FoundationStereo (the requirement here). No ext-ext correspondences. |
| Residuals: EE reprojection, wrist-ext 2D/3D, URDF-landmark reprojection (LoMa-G render matches) and featuremetric (dropped in deployment). Cauchy with c3d = 10 mm, c_l = 50 px, c_ee = 5 px. | URDF-landmark reprojection plus a robot-surface depth term. The depth term is PointWorld-style; DepthWorld uses robot depth and mask IoU only for evaluation. Cauchy 5 px / 10 mm. |
| Mask IoU and robot-depth error as evaluation metrics. | Robot-depth residuals on held-out frames; mask hit-rate if a mask is given. |

### 4.4 `track2d` (AllTracker worker) and 4.5 `lift3d` (`tracking.py`)
- **AllTracker conventions** (from upstream `demo.py`):
  - trajectories are `flow + grid`, where `grid = linspace(0, W-1)`, i.e. integer pixel centres;
  - `visconf` channel 0 is visibility and channel 1 is confidence, P(error ≤ 12 px);
  - frames are 1,T,3,H,W with values 0..255, and H and W must be multiples of 8.
  - `alltracker_to_tracks2d` adds 0.5 to reach the continuous convention.
  - Queries are a regular grid at the first frame of each `window`-frame chunk, with chunks starting every `step`
    frames.
- **Lifting.**
  - Depth is sampled from the same frame on the tracked rectified-left grid, by nearest pixel, never interpolated
    across edges.
  - Points are unprojected with that grid's K and mapped to the base frame with the accepted `T_cam_base`.
  - Each (time, point) gets an explicit status: `valid`, `occluded` (vis < 0.5), `low_confidence` (conf < 0.5),
    `out_of_image`, `no_depth`, `depth_edge` (3x3 neighbourhood spread > 5 %), or `not_tracked`.
  - Positions are NaN unless the status is valid.
  - Point ids are local to the camera and the query; `global_id` = `camera/q<frame>/<id>`.
  - **No identity is assumed across cameras.** Cameras without an accepted calibration or without metric depth
    are skipped, and the record states why.

### 4.6 `export` (`export.py`, loader `dataset.RealworldCacheDataset`)
- **Layout.** Exactly the accepted DROID-track cache layout (`s4d/data/droid/convert.py`): `windows/{i:05d}/` with
  - `images` (T,V,3,H,W) uint8, `depth` (T,V,1,H,W) metres, `K` (continuous), `w2c`/`c2w`;
  - `motion_score`, `track_src`/`track_disp` (P,V,N,3), `track_weight`, `track_count`;
  - `gripper_points` (T,G,3), `gripper_weight` (P,V,G), `probe_state` (T,7).
- The format string is `splatter4d_realworld_v1_tracks`, and the default resolutions are `scratch` 144x256 and
  `dinov2` 140x252.
- **Target helpers.** `track_targets`, `splat_score` and `depth_test` are imported unchanged from
  `s4d.data.droid.pointworld`, so the targets, methods and losses are unchanged.
- **Gripper points.** 32 surface samples of the URDF gripper links via FK at the recorded gripper state; the
  DROID cache uses EE-pose boxes instead.
- **Probe state.** Flange xyz, flange Euler xyz and the gripper state, in the same 7-vector layout as DROID's
  cartesian pose plus gripper.
- **Timeline.** Windows (t, t+s, t+2s) follow the reference camera. Other cameras are matched by nearest
  timestamp and robot state is interpolated. Frames beyond `max_sync_ns` are dropped, and the maximum errors are
  recorded.
- **Window validity.** A window must fall inside one tracking chunk for every camera.
- **Infrared.** The training model expects RGB, so infrared cameras are exported only with
  `allow_infrared_input=True`. The images are then channel-replicated, and the manifest says
  "infrared … replicated … (NOT rgb)".
- **Mock outputs.** Data derived from mock models is refused unless `allow_mock=True`. In that case the manifest
  carries `MOCK_MODEL_OUTPUTS: true` and the loader warns.
- **Loader.** The loader validates every item with `s4d.data.contract.validate_batch`.

### 4.7 Optional, baseline-only stages (`baseline_stages.py`)
Found by reading `s4d/baselines/*`:

- **SNeRL** needs 4-class semantics (`background, robot, object, fixture`; in Meta-World these come from MuJoCo
  body ids via `class_table`) and DINO teacher features.
  - The teacher is `ViTExtractor('dino_vits8', stride=4)` from the pinned `third_party/dino`, with the official
    `dino_deitsmall8_pretrain.pth` (sha256-checked).
  - Features are layer 11, facet `key`, no bin or CLS.
  - Normalisation is the scalar `(x-0.45)/0.225` (not ImageNet per-channel).
  - The 31x31x384 grid at 128x128 is bilinearly upsampled.
  - SNeRL computes these online (`s4d.baselines.snerl.model.dino_target`).
- **`dino_snerl`** is an optional cache.
  - It resizes the exported images to 128x128 (bilinear, /255).
  - The `dino_snerl_worker` runs in the consumer's own venv. It returns the token grid computed with the first
    lines of `dino_target` verbatim, and checks on the first image that the consumer's upsampling of the tokens
    equals `dino_target` exactly (`verified_against_consumer`).
  - Storage is float16 tokens.
  - DINO on infrared requires `allow_infrared` because it is a domain shift.
- **`sam3_masks`.** Prompts from configurable text are mapped to SNeRL classes, using the SAM 3 image API
  (`build_sam3_image_model` / `Sam3Processor.set_image` / `set_text_prompt`) or the video API
  (`build_sam3_video_predictor`, `start_session` / `add_prompt` / `propagate_in_video`). The stage stores:
  - raw instances: masks, scores, boxes, prompt index, and instance ids that are valid only within a video session;
  - the prompts, the model identity, and per-frame per-prompt `count`, where 0 means an absent detection, listed
    in `<cam>_absent.json`;
  - a derived semantic map (highest score wins) with per-frame class presence.
- **3D-MVP** needs RGB-D, K, c2w and scene bounds in the base frame. The bounds must be chosen for the real
  workspace; this is not a model stage.
- **SinCro, ReViWo and DynaMo** use RGB only.

## 5. Robot assets: what must be supplied

`RobotGeometry` refuses to run without these:
1. **Panda URDF.** From `frankarobotics/franka_description` (pinned in `configs/realworld/upstream_pins.yaml`),
   xacro-expanded. Joint origins are checked against Franka's modified-DH parameters (`robot.panda_dh_fk`, with
   joint8/flange at +0.107 m), and a test compares FK against them.
2. **The exact Robotiq model.** The plan says "Robotiq adaptive gripper", which does not say whether it is the
   2F-85 or the 2F-140.
   - Supply its URDF and meshes, for example `PickNikRobotics/ros2_robotiq_gripper` `robotiq_description`
     (2F-85 or 2F-140 macro) or `ros-industrial-attic/robotiq`.
   - Meshes must be STL or OBJ (the parser is numpy-only). Collada `.dae` visual meshes must be converted, or the
     collision STLs used.
   - Mimic joints are supported. The closed linkage is approximated by the URDF mimic joints, as upstream does.
3. **The measured mount transform `T_flange_gripper`** (`panda_link8` to the gripper base link), including the
   coupling thickness and the yaw.
   - Franka's `external_gripper_example` uses `xyz 0 0 0, rpy 0 0 +pi/2`, and the DROID code comments the same
     +pi/2 yaw.
   - Robotiq states the Panda flange is ISO 9409-1-50-4-M6 with coupling GRP-CPL-062 and no adapter plate. This
     was not cross-checked against a Franka datasheet.
   - The coupling offset must be measured.
   - Combine the two URDFs with `robot.compose_urdf`.
4. **The gripper-state mapping.** `robot.gripper_joint_value` assumes a linear map to `finger_joint`
   (2F-85: 0..0.8 rad, 2F-140: 0..0.7 rad). **Verify it on the real gripper.**

## 6. Isolated workers and environments

`s4d/realworld/workers.py` defines a file-based subprocess protocol, `s4d.realworld.worker/1`:
- The caller writes `request.json` and `inputs.npz` and runs `<env python> -I s4d/realworld/workers/<name>_worker.py`.
- The worker writes `outputs.npz` and `response.json`.
- The response carries the model identity: repo, `git rev-parse HEAD`, checkpoint and its sha256, and `is_mock`.
- A mismatch with the pinned commit, or an unexpected mock, is an error.
- Upstream code is imported only inside the worker scripts.

| Worker | Environment (set up by the operator; nothing was installed here) |
|---|---|
| `foundation_stereo_worker.py` | Upstream `environment.yml` (conda `foundation_stereo`: py3.11, torch 2.4.1, torchvision 0.19.1, xformers 0.0.28.post1, flash-attn separately). Checkpoint `23-51-11/model_best_bp2.pth` with `cfg.yaml` (ViT-L), or `11-33-40` (ViT-S). |
| `alltracker_worker.py` | `conda create -n alltracker python=3.12.8`, then `pip install -r requirements.txt` at the pinned commit. Upstream leaves torch unpinned: record the installed versions. Checkpoint `alltracker.pth` (Hugging Face `aharley/alltracker`). |
| `sam3_worker.py` | Python ≥3.12, torch 2.10 (cu128), `pip install -e .` at the pinned commit. Gated checkpoint `facebook/sam3`. |
| `dino_snerl_worker.py` | This repository's `.venv`. It is the consumer environment, with `third_party/dino` at 7c446df. |
| `mock_worker.py` | Any Python. **Mock**: CPU block matching and fake tokens/masks, always flagged `is_mock`. |

**Upstream pins** (GitHub API, 2026-10-31; full list and notes in `configs/realworld/upstream_pins.yaml`):

| Repo | Commit |
|---|---|
| FoundationStereo | `6e8806816b533e4d13ddbb95ffa907b797060a62` |
| AllTracker | `e7553135e7b361590dbccd10e2b274b024f41cd6` |
| SAM 3 | `f1647e8d0c1d280991f4d1b7d974c2704537fe66` |
| DINO | `7c446df5b9f45747937fb0d72314eb9f7b66930a` |
| DepthWorld (reference only) | `6823c13ddcd96484de04604da7d52c9e12f5c86c` |

## 7. Run commands

```bash
# tests (small, CPU numerics; the GPU guard needs a MIG UUID)
cd <worktree> && CUDA_VISIBLE_DEVICES=MIG-... nice -n 10 /home/ws/ws/splatter4d/.venv/bin/python -m pytest -q \
  -p no:cacheprovider tests/test_realworld_stereo.py tests/test_realworld_calibration.py tests/test_realworld_pipeline.py \
  tests/test_realworld_contract_workers.py tests/test_realworld_baselines.py
# contract validation of an adapter's episodes
python -m s4d.realworld.cli validate --config configs/realworld/pipeline_example.yaml
# full run (after the real adapter, assets, worker environments and a landmark source exist)
python -m s4d.realworld.cli run --config <cfg.yaml> --episode <id> --stages ingest,stereo_depth,calibrate,track2d,lift3d,export
python -m s4d.realworld.cli status --root <output_root> --episode <id>
```

## 8. Validated vs. awaiting real data

| Item | State |
|---|---|
| Contract round trip; rejection of 3-channel IR, bad baselines, non-monotonic stamps, schema mismatch | validated (unit tests) |
| Disparity-to-depth scale, doffs, validity, missing baseline refused | validated (synthetic) |
| Right-reference flip-and-swap versus ground truth; naive swap fails | validated (synthetic, CPU block matcher) |
| Bouguet rectification (rows aligned, depth exact, remap inverse); in-pipeline rectification of a rotated rig (~0.9 % vs ~20 % without) | validated (synthetic) |
| K through resize/crop/flip; depth/image alignment after resize+crop | validated (synthetic) |
| URDF FK equals Franka DH; mimic joints; mount composition; safe URDF numbers | validated (stand-in URDF) |
| Calibration recovery (0.5 px noise, 5 % outliers, 3 mm depth noise): < 0.3°, < 5 mm; rejection for low visibility, static robot, no visual evidence | validated (synthetic robot, stand-in geometry) |
| 2D-to-3D lifting versus ground truth; occlusion flagged; camera-local ids; NaN for invalid | validated (oracle tracker) |
| Export passes `validate_batch`; refuses mock and infrared without opt-in; idempotent stages | validated (synthetic, MOCK models) |
| FoundationStereo, AllTracker and SAM 3 worker scripts | **UNVERIFIED**: written against upstream READMEs/demos at the pinned commits, never executed (no envs or checkpoints here) |
| DINO worker equality with `dino_target` | implemented; only runs with the official weights (not run here) |
| Real RealSense layout, modality, rectification state, clocks, intrinsics convention | **awaiting real data** (template adapter only) |
| Exact Robotiq model, URDF and meshes, mount transform, gripper-state mapping | **awaiting assets and measurements** |
| Landmark source on real images (annotation, marker or render-and-match) | interface only; render-and-match not implemented |
| Depth on an RGB camera (colour sensor ≠ IR left) | `geometry.reproject_depth` (z-buffer forward warp) validated on synthetic data; not yet wired into stages (export uses the IR-left grid) |
| Robot mask from the accepted calibration (`geometry.point_mask`, depth-gated) | validated (synthetic); optional alternative/check for the SAM 3 robot class |
