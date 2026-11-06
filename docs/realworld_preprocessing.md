# Real-world preprocessing (Franka Panda + Robotiq + three RealSense rigs)

**Standard pipeline (consolidated 2026-11-05/06).** Robot: Franka **Panda** (not FR3) with the DROID-style Robotiq
2F-85 installation and no wrist camera; three RealSense D435I RGB-D rigs recording colour, native depth and the
device's infrared pair. Configuration: `configs/realworld/mug_hanging.yaml` (the standard own-rig config; other tasks
change only the recording-specific settings: episodes, SAM 3 object/fixture prompts, frame-pairing tolerance).

```text
adapter (panda_teleop_h5, include_ir) -> ingest
  -> sensor_depth (native depth -> colour grid; calibration evidence only)
  -> foundation_stereo_depth (B1: FoundationStereo on each device IR pair -> metric left-IR depth -> colour grid)
  -> sam3_masks (robot masks = calibration evidence; full semantics for the SNeRL baseline)
  -> calibrate (robot-grounded, evidence: rgbd on sensor_depth + SAM 3 robot masks + URDF at the recorded joints)
  -> track_everything (single-view TrackEverything per colour camera with B1 depth, its K and the accepted pose;
                       writes the track2d + lift3d records)
  -> track_motion (noise-aware dynamic-track filter)
  -> train_data (compact training store: frames once per camera at 128x128, moving 3D tracks; section 4.6)
  [-> vis_geometry (viewer: native depth, compact; optional)]   [-> export (former window cache; configured only)]
```

RGB is used for appearance and tracking; the IR pair feeds only the stereo model; native depth is calibration
evidence only. A missing IR pair, a failed model call or an unconfigured tracker is an error: no other depth model or
tracker is substituted. Retired routes and interfaces (the raw-DROID estimation route with MapAnything, AllTracker,
stereo rectification and provided calibration; the depth/tracker comparison machinery) are listed in section 10.
DROID pretraining does not use this package: it reads DROID RGB, PointWorld's released depth and tracks and the
DROID-3D calibration (`s4d/data/droid/largescale/`, [droid_largescale.md](droid_largescale.md)).

Hosts (2026-11-06): bulk real-world preprocessing runs on the remote host; the RTX 5090 inference server
(`ops/inference_server`, `configs/realworld/inference_server.yaml`) is for lightweight inference and visualization
(single episodes). Package: `s4d/realworld/`; existing training and data modules do not import it
(`test_existing_code_does_not_import_realworld`). See [status](realworld_preprocessing_STATUS.md) and
[TrackEverything](realworld_trackeverything.md).

## 1. On-disk layout and restartability

```
<root>/episodes/<episode_id>/normalized/episode.json          contract document (schema s4d.realworld.episode/1.1.0)
<root>/episodes/<episode_id>/normalized/arrays.npz            robot trajectory + per-stream timestamps
<root>/episodes/<episode_id>/normalized/frames/<stream>.h5    frames exactly as recorded (dtype/channels untouched)
<root>/episodes/<episode_id>/derived/<stage>/...              stage outputs
<root>/episodes/<episode_id>/derived/<stage>/_COMPLETE.json   completion record (fingerprint, config, inputs, model identity)
<root>/train_data/<episode_id>/episode.h5 + _COMPLETE.json      training store (schema s4d.realworld.train/1.0.0)
<root>/vis_geometry/<episode_id>/geometry.h5 + _COMPLETE.json  viewer depth store (optional)
<export_dir>/{scratch,dinov2}.h5 + manifest.json              former window cache (configured only; superseded)
```

The original recording is never written. Adapters only read it, and the provenance stores each source path with
its sha256.

`s4d.realworld.io.run_stage` makes every stage idempotent and restartable:
- The fingerprint is a hash of the stage version, its config and its inputs. The config includes the full
  worker spec: commit, checkpoint, params and the mock flag. Inputs are the upstream *build ids*, which change
  on every rebuild, including a forced rebuild with an identical config, so downstream stages are invalidated.
  Model and kernel stages save every chunk durably (`io.ChunkStore`) and acknowledge a runner result only after
  saving it, so an interrupted stage resumes at its first unsaved chunk.
- If the record exists with the same fingerprint, the stage is skipped.
- Otherwise the stage is built in a fresh `.<stage>.building.*` directory, which is swapped in atomically. The
  record is written last.
- An interrupted build leaves the previous complete result untouched (tested).
- Work runs per episode, and frames are processed in bounded chunks: `chunk`, tracker `window`/`step`.
- Each run saves its resolved config as `<root>/pipeline.yaml` (the frame-set policy of viewers and contract checks).
- A per-stage lock file holds the builder's pid. A second builder is refused, and a lock left by a dead pid is
  taken over.

## 2. Normalized episode contract (`s4d/realworld/contract.py`)

`SCHEMA_VERSION = "s4d.realworld.episode/1.1.0"` (1.0.0 documents, without RGB-D rigs, still load). The dataclasses are:

| Dataclass | Fields |
|---|---|
| `Intrinsics` | width, height, fx, fy, cx, cy, skew, `pixel_convention` (`integer_centers` as librealsense/OpenCV, or `continuous`), `distortion_model` + coefficients (OpenCV plumb_bob/rational, RealSense model names kept verbatim, Kannala-Brandt), `source`. `K()` converts to the repository's continuous convention. |
| `CameraStream` | `stream_id`, `rig_id` (one RealSense), `eye` (left/right/mono), `modality` (`rgb`, `infrared`, or `depth`), `channels`, `channel_order` (RGB/BGR/GRAY), dtype (uint8/uint16), depth units/scale/kind, `rectified` (True/False/None = unknown), serial, sensor name, timestamp source, IR emitter state. |
| `StereoGeometry` | per rig: `status` (`metric`/`missing`/`unverified`), `T_right_left` (left-camera points into the right camera, metres), `baseline_m`, source. Part of the schema; stereo camera views are not processed by the stages (`pipeline.camera_views` refuses them). |
| `IrStereoGeometry` | the device IR pair of an RGB-D rig (never a camera view): IR-left K, equal IR-right K, recorded IR1->IR2 extrinsic and baseline, colour<-IR-left extrinsic. |
| `RgbdGeometry` | colour stream, depth stream, metric status, recorded `T_color_depth`, pairing tolerance. |
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

The supported adapters are:

- **`panda_teleop_h5`** (`panda_teleop.py`, the own rig): `panda_teleop_episode` format v2, JPEG colour + PNG uint16
  depth per RealSense, recorded intrinsics, depth scale and depth-to-colour extrinsics. Device stamps mapped onto
  the host monotonic clock are used; arrival times are not substituted. Colour/depth pairing, invalid flags and
  missing calibration are checked. Robotiq register 0..255 has its own clock and is interpolated as recorded.
  Colour and stereo (depth/IR) records are paired through `stereo/color_index`. A colour frame whose stereo record the
  recorder dropped (e.g. drawer-open episodes 9 and 33: one camera, one record) is left out of that camera's
  normalized colour stream, because the contract pairs colour frame k with depth/IR frame k (`RgbdGeometry`,
  `IrStereoGeometry`, equal counts checked by `validate_episode`) and no depth/IR frame or timestamp is invented; its
  recorded index and time are listed in the colour stream notes, the rig's `pairing` text and the provenance notes
  (`colour_frames_without_stereo_record`), and the camera's timeline shows the gap (downstream treats it like a
  dropped frame: timestamp matching, gap-aware windows and motion segments). An index map that does not map strictly
  increasing into the colour frames is refused ("not paired 1:1"). The task is the dataset folder; a recorder
  `task_name` that disagrees (drawer-open episodes 7-10 say `pick-and-place`; every `test/` file) is reported as a
  warning and in the provenance notes (`task_identity`), never used; that note is part of the ingest fingerprint
  (`adapter.ingest_identity` -> ingest config `source_identity`), so a normalized episode cached without it is rebuilt,
  while every other recording keeps its earlier fingerprint.
  `configs/realworld/mug_hanging.yaml` supplies the real robot assets and `evidence: rgbd` calibration.
  `include_ir: true` also exposes the device-rectified infrared pair of each RealSense as `<serial>/ir_left`,
  `<serial>/ir_right` with `EpisodeMeta.ir_stereo` per rig (`IrStereoGeometry`). The adapter refuses a pair unless the
  recorder's `pair_ok` flag is set, both IR frames carry the SAME hardware timestamp, and the depth stream's frame IS
  the IR-left frame. Nothing is fabricated: without recorded IR frames and extrinsics there is no `ir_stereo` and the
  FoundationStereo stage refuses the rig.
- **`synthetic`** (`synthetic_adapter.py`, tests only). Ray-cast textured boxes (floor, table, wall and a moving
  object) seen by three infrared stereo rigs (one with `status="missing"`), a stand-in Panda kinematic chain (official
  joint origins) with primitive link geometry and a two-finger mimic gripper. NOT the real Panda or Robotiq meshes.
  The tests expose the same scene as RGB-D rigs (`tests/test_realworld_trackeverything.ColourEpisode`) for the
  end-to-end stage tests.

**Writing another recording adapter:**
1. Subclass `EpisodeAdapter` and decorate it with `@register_adapter`.
2. Fill `read_meta` from the recorded metadata.
3. Put every guess into `assumptions`.
4. Hash source files in `source_files`.
5. Run `python -m s4d.realworld.cli validate --config <cfg>` until there are no errors.

Nothing downstream should need to change, except possibly adding a distortion model. RealSense `brown_conrady` is OpenCV plumb_bob. RealSense `modified_brown_conrady` and
`inverse_brown_conrady` with non-zero coefficients are refused explicitly (`NotImplementedError`). All-zero
coefficients mean no distortion, whatever the model name.

## 4. Stages

### 4.1 `ingest` (`pipeline.ingest`)
- Validates the contract and writes the normalized copy: JSON, NPZ, and per-stream HDF5 with lzf compression and
  one chunk per frame.
- Checks every frame's shape, dtype and timestamp against the metadata.
- The record carries the validation warnings and the adapter assumptions.

### 4.2 Depth (`depth_stages.py`, `ir_stereo_depth.py`, `geometry.py`)

Every stage works on the camera views of the RGB-D rigs (`pipeline.camera_views`): the recorded, zero-distortion
colour grid of each rig. Depth, tracks, masks and the export all live on that grid.

- **`sensor_depth`**: metric device depth is filtered by configured range and z-buffer registered to the colour grid
  with recorded intrinsics and `T_color_depth`. Zeroes and disocclusions stay invalid. The calibration's depth
  evidence (`stages.calibrate.depth_stage: sensor_depth`); never used for lifting or export by the standard config.
- **`foundation_stereo_depth`** (B1; RGB-D rigs with a recorded IR pair; `ir_stereo_depth.py`, worker
  `foundation_stereo_worker.py` task `ir_pair_disparity`): FoundationStereo (pinned commit and ViT-L checkpoint
  `23-51-11`) on the device-rectified IR pair at native 640x480 (InputPadder 32, unpadded), grayscale replicated to 3
  channels as model input only. Left-referenced disparity `d = x_left - x_right`, right-referenced disparity by
  flip-and-swap (left-right error and IR validity summarised per stream in the stage summary); depth
  `z = fx_ir * B / d` with the recorded baseline; invalid where `d <= min_disparity`, the match leaves the right
  image, or `z` is outside [`min_depth_m`, `max_depth_m`]. The IR-left frame is the native-depth frame, so the IR
  depth is registered to the colour grid with exactly the `sensor_depth` registration (z-buffer, same splat). A failed
  model output is an error; native depth is never substituted. Output layout 2 (2026-11-06): only `depth` and `valid`
  are persisted; layout 1 also stored per-pixel diagnostics (`source_index` on the colour grid; `disparity_left`,
  `lr_error`, `ir_valid` on the IR grid; ~9 MB per three-camera frame) that no retained consumer reads. The
  computation version (fingerprint) is unchanged, so layout-1 outputs remain valid and are compacted in place by the
  operational repack (`output_layout` and the repack provenance in the completion record).
  The IR contract requires `T_right_left[0,3] < 0` for the positive left-minus-right disparity convention.

All depth stages use `<stage>/<safe stream>.h5`: `depth` (N,H,W) float32 camera-z metres, `valid` bool, and attrs
`K`, `stream_id`, `rig_id`, `grid`. `default_depth_stage(meta, "calibrate")` is `sensor_depth` and
`default_depth_stage(meta, "lift")` is `foundation_stereo_depth`; the standard config also names both explicitly
(`calibrate.depth_stage`, `track_everything.depth_stage`, `export.depth_stage`).

Synchronized frame sets (`depth_stages.frame_sets`; viewer and contract checks) pair every frame exactly once within
the tolerance, or, with mutual pairing (`allow_unmatched`), list unmatched frames. The policy of a root comes from its
`pipeline.yaml`: the export's `max_sync_ns` with mutual pairing, otherwise strict (`configured_sets_policy`).
Image resize/crop uses explicit continuous-coordinate pixel affines; depth uses nearest-neighbour sampling.
Server worker settings explicitly override interpreter, repo and checkpoint paths for remote runs; adapter and output
paths remain on the host.

### 4.3 `calibrate`: per-episode, robot-grounded extrinsics (stage `calibration_stage.py`, solver `calibration.py`)

The method is adapted from DepthWorld ([paper](https://arxiv.org/abs/2610.08780),
[code](https://github.com/Jai2500/depthworld), [project page](https://jaibardhan.com/depthworld/)). Each camera
has one constant `T_cam_base` per episode; poses may differ between episodes. The robot model is the full
articulated URDF: arm plus the actual gripper driven by the recorded gripper state. The EE pose alone is never
used.

1. **Visual evidence.** 2D observations of robot landmarks (`landmark_links`, URDF link origins) per frame, in
   continuous coordinates of the camera view's grid. The 3D positions come from forward kinematics. They are
   supplied by a *landmark source*:
   - an annotation file (`cli.AnnotationLandmarkSource`, config `stages.calibrate.landmark_source`), for example
     manual clicks or a marker detector;
   - in the future, a render-and-match worker, the counterpart of DepthWorld's LoMa-G URDF-render-to-image
     matching. **Not implemented.**
   The annotation file format is
   `{camera_id: {frame_index: {"uv": [[u, v] | null, ...], "conf": [...]}}}`. Coordinates are continuous, on
   the view grid. Observations on a grid with distortion are undistorted to the ideal pinhole image automatically. Landmarks are listed in `landmark_links` order. Any source flagged `"is_mock": true` in
   its `identity` blocks export unless mock outputs are explicitly allowed.
2. **Visual initialisation.**
   - RANSAC PnP: normalised DLT, then a Cauchy-robust reprojection refinement.
   - When metric depth exists, also a RANSAC Kabsch fit of depth-lifted landmarks (bootstrap depth, no scale
     estimation).
   - The candidate with more inliers and lower reprojection error wins.
   - A previous episode's pose may be used only as a fallback prior (`T_init`).
3. **Robot-grounded refinement** (`scipy.optimize.least_squares`, Cauchy loss over the 6-DoF extrinsic):
   - landmark reprojection residuals, scaled by `cauchy_px` (5 px) and weighted by confidence;
   - robot-surface depth residuals: URDF surface points (meshes or primitives) projected and compared with bilinear
     depth, scaled by `cauchy_depth_m` (10 mm, DepthWorld's c3d). Two gates apply:
     - self-occlusion uses the robot's own z-buffer (back and side surfaces would otherwise bias the result; this
       was measured);
     - a one-sided occlusion test keeps in-front residuals and drops points hidden behind something closer.
   - The gates are recomputed over `outer_iters` outer iterations.
   - An optional robot mask (for example from SAM 3) further gates the depth points and gives a mask hit-rate
     diagnostic.
4. **Diagnostics and gates.** Nothing is silently accepted: `status` is `accepted` only if every gate passes, and
   the reasons are listed otherwise. Non-finite values never pass a gate: an observed landmark predicted behind
   the camera counts as infinitely wrong. The covariance uses only active residuals. Each item below is reported, and the ones with thresholds act as gates:
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

**RGB-D evidence (`evidence: rgbd`, the mug-hanging episodes).** No 2D landmarks: the initialisation registers the
articulated Panda + Robotiq surface (recorded joints and gripper state) to the SAM 3 "robot arm"/"end effector"-masked
NATIVE depth of `init_frames` frames (global search over orientations x rolls with centroid-aligned translations,
trimmed-mean score, then trimmed ICP; `init_t_indices` records those frames). The refinement is the same depth term.
Gates: robot-depth median on fit and held-out frames, held-out explained fraction (moving measured depth within
`explained_tol_m` of the posed robot), covariance and conditioning. Frames used by the initialisation are NOT
independent: `calibration.evidence_frames` returns fit + initialisation frames, and independent-frame checks use only
the remaining held-out frames. The calibration depends on the native depth (it is the evidence); the 2026-11-05 depth
comparison reused ONE frozen calibration (section 9).

**Multi-start refinement** (`refine_starts`, 2026-11-05): the robust objective has local optima near the
initialisation (one camera stalled at 13 mm from the single start while starts 3 deg / 3 cm away reached 7.6 mm). The
configured 13 starts are the initialisation plus +-3 deg about and +-3 cm along each camera axis; the solution with the
lowest median |depth residual| over the FIT frames is kept (held-out frames never enter the choice). Every start's
nfev, seconds and fit score are in `diagnostics.multistart`; per-phase kernel time in `diagnostics.timing_s`; the
stage's local preparation (robot state, depth/mask loading, FK + surface sampling, overlays), the kernel call with
transfer, and the durable save are in the record's `stage_timing_s`.

**Simplifications versus DepthWorld.** This is not the paper's method, and we do not claim the full multi-episode
optimisation.

| DepthWorld | Here |
|---|---|
| One factor graph per physical robot over ~1k–9k episodes. Per-episode ext1/ext2 corrections, plus a shared wrist hand-eye (6 DoF) and joint offsets δq (7 DoF). GPU Levenberg-Marquardt with a Schur complement. | One camera of one episode at a time. 6 DoF only. No hand-eye, no joint offsets, no wrist camera. |
| Initial calibration from factory extrinsics, S2M2 stereo depth, RoMa-v2 ext-ext correspondences with MAGSAC, and two-pass bundle adjustment. | Initialisation by registering the articulated robot surface to the masked native depth (RGB-D evidence), or from robot landmarks (PnP / depth Kabsch). No ext-ext correspondences. |
| Residuals: EE reprojection, wrist-ext 2D/3D, URDF-landmark reprojection (LoMa-G render matches) and featuremetric (dropped in deployment). Cauchy with c3d = 10 mm, c_l = 50 px, c_ee = 5 px. | URDF-landmark reprojection plus a robot-surface depth term. The depth term is PointWorld-style; DepthWorld uses robot depth and mask IoU only for evaluation. Cauchy 5 px / 10 mm. |
| Mask IoU and robot-depth error as evaluation metrics. | Robot-depth residuals on held-out frames; mask hit-rate if a mask is given. |

### 4.4 `track_everything`: tracks and 3D tracks (`trackeverything_stage.py`, worker `trackeverything_worker.py`)

TrackEverything (pinned `ayushjain1144/trackeverything` dfb29b5, official checkpoint `katefgroup/TrackEverything`
`model.pth`, CC BY-NC 4.0) runs SINGLE-VIEW per colour camera, never its multiview path. Our metric depth
(`depth_stage`, B1), its K and the accepted `T_cam_base` are passed, so its geometry estimation is bypassed
(`geometry_backend="pi3"` keeps the cam0 alignment and median-distance normalisation of training; Pi3 itself is never
loaded). Frames are resized to 512x384 (`K[:2] *= 0.8`, continuous convention, RGB Lanczos, depth nearest) with
`use_input_resolution=True`. Queries: a regular grid (`query_grid`, `grid_stride` 16) at the first frame of each
`window`-frame chunk (16), chunks every `step` (8) frames (`pipeline.chunk_queries`); each clip is one query window of
at most 16 frames, including frames across recording pauses (pause segmentation is downstream in `track_motion`).
Births with missing depth or a 3x3 depth spread above 5% are excluded from inference and counted; they remain
NO_DEPTH/EDGE at birth and NOT_TRACKED elsewhere (floor query-depth lookup, no hole filling). TE's own dense
boundary-depth replacement is retained and disclosed in metadata. Worker clips are saved durably before their ack and
reused on a retry until both records are swapped in.

Outputs (`tracking.py` contract): the `track2d` record (2D positions on the colour grid) and the `lift3d` record
(base-frame positions `X_base = E0^-1 [norm * X; 1]`, sigmoid visibility, confidence 1). Each (time, point) has an
explicit status: `valid`, `occluded` (visibility < 0.5), `out_of_image` (also behind the camera), `no_depth` /
`depth_edge` (birth depth support), or `not_tracked`; positions are NaN unless valid. Point ids are local to the camera
and the query window (`global_id` = `camera/q<frame>/<id>`); **no identity is assumed across cameras.** Cameras without
an accepted calibration or with unverified rig geometry are skipped and the record states why.

### 4.5b `track_motion`: noise-aware dynamic-track filter (`track_motion.py`, intentional change 2026-11-05)

Before this stage nothing filtered tracks by motion: every valid grid track (mostly static background) reached the
export. `track_motion` classifies every lifted track of one camera and one query window as `invalid`, `static`,
`moving` or `uncertain`; ids are never matched across cameras or windows. Filtering stored outputs does NOT reduce
tracker inference cost (the whole query grid is still tracked).

1. **Segments** never connected: after an identity loss (`occluded`, `low_confidence`, `out_of_image`,
   `not_tracked`), after more than `max_depth_hole_frames` (2) consecutive depth holes, at a recording pause
   (> `max_gap_factor` 1.5 median periods) and at a physically implausible jump (> `max_speed_m_s` 1.5 m/s).
2. **Statistics** on a median-of-3 smoothed copy (outputs are never smoothed): image extent `extent_px`, its metric
   size `lateral_m = extent_px * z / f`, and the depth extent along the ray `extent_z_m`.
3. **Thresholds from observed noise**, per camera: quantiles (`0.995` move / `0.9` static) over reference tracks that
   stay at least 0.15 m from the calibrated articulated robot surface; depth thresholds per depth bin, and a bin with
   fewer than `min_bin_samples` reference segments is uncalibrated (no depth-only move, no static claim; thresholds
   are never borrowed from a nearer bin). Floors: 1.5 px and 5 mm.
4. **Temporal support**: at least `min_support` (2) smoothed samples beyond the threshold.
5. **Class**: moving if any segment of >= 5 valid samples moves; a moving track keeps its FULL valid trajectory,
   stationary intervals included, with segment ids so trails and supervision never cross a break.

Outputs per chunk: `class`, `segment` (T,N), per-segment statistics; the record holds thresholds, reference sizes, the
per-sample noise and the counts per camera. Overlays: `analysis/track_motion_overlays.py`. The export
(`track_source: track_motion`) keeps only moving tracks and computes the motion score within segments; the loader
contract is unchanged.

### 4.6 `train_data`: compact training store (`train_store.py`, loaders `train_windows.py`, `train_store.CompactStore`)

Data-format policy of 2026-11-06: training frames are stored ONCE per camera and timestamp; temporal windows,
viewpoints and every motion target are assembled in the loader. One HDF5 file per episode
(`<output_root>/train_data/<episode>/episode.h5`, `io.run_stage`: fingerprinted record, atomic swap-in; inputs are the
build ids of ingest, the depth stage, calibrate, track2d, lift3d, track_motion and, when SNeRL is declared, sam3_masks).

- **Image geometry.** The centred square of the 640x480 colour grid (480x480 at x0 = 80, integer offsets) resized to
  128x128. Pixel affine `u' = a (u - x0)`, `v' = a (v - y0)`, `a = 128/480` (continuous convention); the file stores
  `K_native`, `K = A K_native`, the affine, the crop box and the accepted `w2c`. RGB: PIL BILINEAR (antialiased; the
  DROID square-crop `crop_rgb` and the SNeRL contract check). Depth, validity, semantics: the native pixel containing
  each output pixel centre (`crop_depth` rule; no mixing across edges). World positions are never rescaled.
- **Per camera** (`cameras/<serial>/`): `timestamps_ns`, `segment` (recording segment per frame), `rgb` (F,128,128,3)
  uint8, `depth` (F,128,128) float32 metres, bitwise the depth stage's sampled values (0 = invalid: the validity mask),
  `semantic` (F,128,128) uint8 + `class_present` (SNeRL classes; only when SNeRL is declared), `robot/` (state at
  this camera's frame times: ok/sync error, joints, gripper, probe_state, 32 gripper FK points), `tracks/` (moving
  `track_motion` tracks only: per TrackEverything query window the frames, and per track the camera-local id, base-frame
  xyz float32 bitwise from lift3d, status, motion segment, visibility, confidence). Frames: one per gzip chunk; every
  array is lossless (gzip, shuffle for multi-byte types).
- **Sync** (`sync/`): over the reference (first) camera, the mutual nearest frame of every camera within `max_sync_ns`
  (-1 otherwise; the `frame_sets(allow_unmatched=True)` rule), the offsets, and per set the reference-time gripper points'
  2 cm depth test against each camera's NATIVE depth (the export's gripper visibility; 32 bits per camera and set).
  `robot/` keeps the recorded trajectory. No dense XYZ, no window copies, no DINO features.
- **Our method** (`RealworldWindowDataset`): the former export's batch contract (validate_batch per item) at 128x128.
  Windows `(t, t+s, t+2s)` on the reference timeline under the export's rules (robot state valid, every camera
  synchronised, all three frames of a camera in one query window, the latest) plus one recording segment per camera and
  window, and EVERY reference step inside the window span at most `max_gap_factor` periods (no window over a pause or a
  dropped reference frame; the export checked only the sampled frames); strides and `validation_tail_frames` are loader
  settings. Cameras may have different frame counts (dropped colour frames without a stereo record): the sync table
  pairs frames by timestamp, so a gap only removes the sets that lack the frame. Track targets as the export (segment breaks unsupervised) plus the square
  crop: a pair is supervised only if the track projects into the stored image at both endpoints (DROID square-crop
  rule); the same for the gripper points. Optional `num_views` draws a viewpoint subset per (index, epoch) among the
  cameras eligible for that window (a window then needs only `num_views` eligible cameras).
- **RGB resampling is a deliberate change** from the former export (PIL's antialiased bilinear instead of
  `warp_affine` point-sampled bilinear, which aliases at the 3.75x reduction); the SNeRL contract check and the DROID
  square-crop profile already used PIL bilinear.
- **Baselines / FQL** (`CompactStore`): the `FrameStore` protocol of the learning adapters (`episodes`, `task`,
  `cameras` = serials, `timestamps`, `segments`, `rgb`, `depth` (metres), `semantic`, `intrinsics` (128 grid), `w2c`,
  `raw_recording`, `image_hw`, `has_semantic`); `EpisodeFile` adds tracks, robot state, sync sets and native K/affine.
- **SNeRL DINO targets** are computed online by the trainer from the stored RGB exactly as for Meta-World
  (`uint8 / 255 -> dino_target`); `train_store.verify_snerl_teacher` checks the pinned teacher, the weights' sha256 and
  that the online targets equal the cached-teacher path (`dino_snerl_worker`).
- **Measured** (2026-11-06, four processed episodes, 179-228 frames x 3 cameras): 39-51 MB per episode (depth 54 %,
  RGB 41 %, tracks 2-3 %, robot/semantic/sync the rest; 73-93 MB uncompressed) vs 1.52-1.97 GB of the former export
  (x38-39); writer 9-11 s per episode on CPU (peak RSS 0.25 GB); window loader 79 windows/s in one process, 367 with 8
  workers (main RSS 0.66 GB, worker <= 0.56 GB). Against the former export on the same episodes: identical windows,
  query windows, frames, gripper/probe/pose targets and track counts; source/displacement targets identical wherever
  supervised; the square crop drops 0-1.75 % of supervised track pairs (gripper weights differ in 64/450 windows of one
  episode, where the gripper leaves the square). Stored RGB/depth/semantics equal an independent recomputation bitwise.
- **Ingestion**: `python -m s4d.realworld.cli run --config CFG --episode ID --stages train_data` (config
  `stages.train_data: {task, depth_stage, max_sync_ns}`), or `train_store.write_episode(root, episode, out, task=...)`;
  `train_store.manifest_fields(record)` gives the manifest fields (format, schema and writer version, code commit,
  fingerprint, build id, file sha256/bytes).

### 4.6a `vis_geometry` (optional; `vis_store.py`)

The viewer reads high-resolution RGB/IR from the recording; the only expensive derived geometry, the native-resolution
depth of the depth stage, is kept once per episode as uint16 0.1 mm (gzip + shuffle, ~0.19 MB per 640x480 frame, ~0.1 GB
per episode vs ~1.9 GB of the `foundation_stereo_depth` stage directory with its diagnostics) with native K, pose and
timestamps (`vis_store.native_depth`). Point clouds are rebuilt from depth + K + pose; viewer caches stay bounded and
regenerable.

### 4.6c `export` (former window cache; `export.py`, loader `dataset.RealworldCacheDataset`; runs only when configured)
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
- **Probe state.** Flange xyz, flange Euler xyz, and the gripper closure in [0,1] (0 = open, from the recorded
  state through the URDF joint range). It has the same 7-vector layout as DROID's cartesian pose plus gripper;
  DROID uses its EE pose rather than the flange.
- **Timeline.** Windows (t, t+s, t+2s) follow the reference camera. Other cameras are matched by nearest
  timestamp and robot state is interpolated.
  - Frames beyond `max_sync_ns` are dropped. The default is a third of the reference frame period.
  - Windows where two times match the same frame of another camera are dropped.
  - The maximum sync errors are recorded.
- **Split.** `validation_tail_frames` puts windows lying entirely in the last N reference frames into the
  validation split. Windows that straddle the boundary are dropped, so no frame is shared between splits.
- **Window validity.** A window must fall inside one tracking chunk for every camera.
- **Depth and tracks.** `depth_stage` (B1 in the standard config) supplies the exported depth; `track_source:
  track_motion` exports only moving tracks. The manifest's `motion_target` names the tracker from the `track2d` record
  (TrackEverything; a record of another tracker is refused).
- **Infrared.** RGB-D camera views are colour streams; `allow_infrared_input` (export of channel-replicated infrared
  views, manifest "infrared … replicated … (NOT rgb)") is kept in the interface but has no infrared view to apply to.
- **Mock outputs.** Data derived from mock models is refused unless `allow_mock=True`. In that case the manifest
  carries `MOCK_MODEL_OUTPUTS: true` and the loader warns.
- **Loader.** The loader validates every item with `s4d.data.contract.validate_batch`.

### 4.6b Viewer (`viewer_data.py`, `scripts/realworld_viewer*.py`)

The viewer uses the root's `pipeline.yaml` frame-set policy (the export's `max_sync_ns` with mutual pairing), and
otherwise requires strict pairing. Multi-camera playback contains only synchronized sets; unmatched frame indices are
reported by `viewer_data.describe()`. Point clouds come from a named depth source (`--source NAME=ROOT:STAGE:LABEL`,
e.g. `foundation_stereo_depth`); trails show `track_motion`-moving tracks only.

### 4.7 Optional, baseline-only stages (`baseline_stages.py`)
**Scope (2026-11-05): only our method is pretrained on DROID** (from PointWorld's released data, never through this
package, never with SAM 3/DINO). For real-world episodes `required_baseline_stages(cfg)` derives exactly the auxiliary
stages the configured `baselines` need (a config without baselines runs none), and the CLI refuses an explicit stage
outside that scope. SAM 3 also supplies the robot masks of the RGB-D calibration (prompts
"robot arm" and "end effector"; "robotic gripper" detected nothing on this rig). Found by reading `s4d/baselines/*`:

- **SNeRL** needs 4-class semantics (`background, robot, object, fixture`; in Meta-World these come from MuJoCo
  body ids via `class_table`) and DINO teacher features.
  - The teacher is `ViTExtractor('dino_vits8', stride=4)` from the pinned `third_party/dino`, with the official
    `dino_deitsmall8_pretrain.pth` (sha256-checked).
  - Features are layer 11, facet `key`, no bin or CLS.
  - Normalisation is the scalar `(x-0.45)/0.225` (not ImageNet per-channel).
  - The 31x31x384 grid at 128x128 is bilinearly upsampled.
  - SNeRL computes these online (`s4d.baselines.snerl.model.dino_target`).
- **DINO targets are online** (2026-11-06): the trainer computes `dino_target` from the stored 128x128 RGB; the former
  `dino_snerl` cache stage is retired. The `dino_snerl_worker` stays as the cached-teacher reference of the contract
  check (`snerl_check`) and of `train_store.verify_snerl_teacher`: it returns the token grid computed with the first
  lines of `dino_target` verbatim and checks that the consumer's upsampling of the tokens equals `dino_target` exactly.
- **`sam3_masks`.** Prompts from configurable text are mapped to SNeRL classes, using the SAM 3 image API
  (`build_sam3_image_model` / `Sam3Processor.set_image` / `set_text_prompt`) or the video API
  (`build_sam3_video_predictor`, `start_session` / `add_prompt` / `propagate_in_video`). The stage stores:
  - raw instances: masks, scores, boxes, prompt index, and instance ids that are valid only within a video session;
  - the prompts, the model identity, and per-frame per-prompt `count`, where 0 means an absent detection, listed
    in `<cam>_absent.json`;
  - a derived semantic map (highest score wins) with per-frame class presence.
  - Segmentation runs on the recorded colour grid, the grid of depth, tracks and export.
  - The video API gets a fresh frame directory per call.
- **3D-MVP** needs RGB-D, K, c2w and scene bounds in the base frame. The bounds must be chosen for the real
  workspace; this is not a model stage.
- **SinCro, ReViWo and DynaMo** use RGB only.

## 5. Robot assets: what must be supplied

`RobotGeometry` refuses to run without these:
1. **Panda URDF.** The robot is a Franka **Panda** (`panda_link0..8`, `panda_joint1..7`; the earlier provisional FR3
   description is replaced). From `frankarobotics/franka_description` (pinned in `configs/realworld/upstream_pins.yaml`),
   xacro-expanded. Joint origins are checked against Franka's modified-DH parameters (`robot.panda_dh_fk`, with
   joint8/flange at +0.107 m), and a test compares FK against them. The installation is the DROID-style Robotiq
   2F-85 on the ISO-50 coupling WITHOUT the wrist camera or its bracket; the coupling stays in the model
   (`MountConfig.coupling_geometry`: PickNik `ur_to_robotiq_adapter` mesh, NOMINAL geometry, not measured).
   Consistency check on episode 0 (`analysis/mount_gripper_check.py`, independent frames only; evidence, not a
   measurement): yaw +-90 deg about `panda_link8` fits (1-cm depth inliers 0.61; yaw 0: 0.11; 180 deg is not
   identifiable), the flange->base offset plateaus at 8-14 mm with the nominal 11.0 mm at the peak, and the DROID
   register mapping (3..230) fits closed fingers marginally better than the linear 0..255 map (0.605 vs 0.591).
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
     +pi/2 yaw. Its xyz = 0 omits the coupling.
   - Robotiq states the Panda/FR3 flange is ISO 9409-1-50-4-M6 ("DIN ISO 9409-1-A50" in Franka's datasheet), with
     coupling GRP-CPL-062 and no adapter plate.
   - Robotiq's drawings give a provisional 11.0 mm from the flange face to the 2F-85 base frame
     (`robotiq_85_base_link`): a 16.9 mm coupling, minus its 3 mm robot-side recess, minus a 2.9 mm gripper-side
     recess taken from the blank coupling.
   - Mount configurations live in `s4d/realworld/mounts.py`, with a status (measured / provisional / assumption)
     and the missing measurement. Compose with
     `scripts/realworld_compose_robot_urdf.py --mount-config NAME` (or with explicit `--mount-xyz/--mount-rpy/
     --mount-source`). The manifest records the stack and its sources.
   - The coupling offset on our own rig must still be measured; the drawings are not a measurement.
   - Combine the two URDFs with `robot.compose_urdf`.
4. **The gripper-state mapping.** `robot.gripper_joint_value` assumes a linear map to `finger_joint`
   (2F-85: 0..0.8 rad, 2F-140: 0..0.7 rad). **Verify it on the real gripper.**

## 6. Isolated workers and environments

`s4d/realworld/workers/__init__.py` (package `s4d.realworld.workers`) defines a file-based subprocess protocol, `s4d.realworld.worker/1`:
- The caller writes `request.json` and `inputs.npz` and runs `<env python> -I s4d/realworld/workers/<name>_worker.py`.
- The worker writes `outputs.npz` and `response.json`.
- The response carries the model identity: repo, `git rev-parse HEAD`, checkpoint and its sha256, and `is_mock`.
- A mismatch with the pinned commit, or an unexpected mock, is an error.
- Upstream code is imported only inside the worker scripts.

| Worker | Environment (`ops/inference_server/setup_envs.sh`) |
|---|---|
| `foundation_stereo_worker.py` | Own-rig depth (B1). Env `foundation_stereo` (Python 3.11, `locks/foundation_stereo.txt`), pinned `NVlabs/FoundationStereo` checkout imported from `repo_dir`; ViT-L checkpoint `23-51-11/model_best_bp2.pth` (sha256 `60e79bde...`) + its `cfg.yaml`, loaded strictly. Task `ir_pair_disparity` only. The EdgeNeXt stem is constructed without its ImageNet download (every weight is overwritten by the strict load); upstream's visualisation-only `open3d` import is replaced by an empty placeholder module. |
| `trackeverything_worker.py` | Own-rig tracker. Env `trackeverything` (Python 3.10, torch 2.8.0+cu129, `torch_scatter` 2.1.2 pt28cu129; `locks/trackeverything.txt`), pinned checkout as package `trackeverything`; `model.pth` (sha256 `4553975d...`, CC BY-NC 4.0). Single-view only. |
| `sam3_worker.py` | Python ≥3.12, torch 2.10 (cu128), `pip install -e .` at the pinned commit. Gated checkpoint `facebook/sam3`. Calibration robot masks + SNeRL semantics. |
| `dino_snerl_worker.py` | This repository's environment (the consumer's), with `third_party/dino` at 7c446df. SNeRL baseline only. |
| `s4d_kernels_worker.py` | This repository's environment: the pipeline's own numerical kernels (`kernels.py`) at the reviewed commit. |
| `mock_worker.py` | Any Python, tests only. **Mock**: a transport echo, fake tokens/masks and the TrackEverything oracle adapter; outputs are always flagged `is_mock`. |

**Upstream pins** (full list and notes in `configs/realworld/upstream_pins.yaml`):

| Repo | Commit |
|---|---|
| FoundationStereo | `6e8806816b533e4d13ddbb95ffa907b797060a62` |
| TrackEverything | `dfb29b505cf2ea954f8697fb612c2429668dd3a0` |
| SAM 3 | `f1647e8d0c1d280991f4d1b7d974c2704537fe66` |
| DINO | `7c446df5b9f45747937fb0d72314eb9f7b66930a` |
| DepthWorld (reference only) | `6823c13ddcd96484de04604da7d52c9e12f5c86c` |

## 7. Run commands

```bash
# CPU-only tests (the conftest guard needs an allowed MIG UUID)
CUDA_VISIBLE_DEVICES=MIG-8da55219-bd6b-5260-8be1-3b2a76ac733c nice -n 10 /home/ws/ws/splatter4d/.venv/bin/python -m pytest -q -p no:cacheprovider tests/test_realworld_*.py
/home/ws/ws/splatter4d/.venv/bin/ruff check s4d/realworld ops/inference_server scripts/realworld_*.py tests/test_realworld_*.py
python -m s4d.realworld.cli validate --config configs/realworld/mug_hanging.yaml
# Standard pipeline (every configured stage; local LocalRunner file protocol)
python -m s4d.realworld.cli run --config configs/realworld/mug_hanging.yaml --episode <id>
python -m s4d.realworld.cli status --root <output_root> --episode <id>
# Same stages with every compute call in the inference container (single episode; re-running resumes)
python -I ops/inference_server/run_episode.py --pipeline configs/realworld/mug_hanging.yaml --server configs/realworld/inference_server.yaml --episode <id> [--output-root <new root>] [--stages a,b] [--force-stage s]
# Data/model contract checks on the export (inference only)
python -I ops/inference_server/contract_checks.py --pipeline configs/realworld/mug_hanging.yaml --server configs/realworld/inference_server.yaml --episode <id> [--output-root <root>]
# Viewer (viewer env with viser and PyYAML; CPU only, no MuJoCo): recordings + static client + index, browser check, video
python scripts/realworld_viewer.py record --source B1=<root>:foundation_stereo_depth:"B1 IR depth" ... --episode <id> --out <dir>
python scripts/realworld_viewer_check.py <dir> --recording <name> --chromium <chrome>
python scripts/realworld_viewer_video.py <dir> --recording <name> --video <out.mp4> --chromium <chrome>
```

**MuJoCo.** Neither preprocessing nor the viewer needs MuJoCo (robot meshes are posed by URDF forward kinematics).
Only the simulation-side tests and environments use it. In `shkim_rw` (`/opt/envs/s4d`, mujoco 3.10.0) EGL
rendering failed with `Failed to load libEGL.so.1` (or PyOpenGL's `'NoneType' object has no attribute
'eglQueryString'`) because the container has no GLVND loader; inside our environment only, setting
`LD_LIBRARY_PATH=/opt/envs/tools/lib` (conda-forge libglvnd in the tools env) makes it work (51/53 MuJoCo tests; the 2
remaining guard tests assumed two approved GPU UUIDs and now skip on a single-GPU allocation).

## 8. Validated vs. awaiting real data

| Item | State |
|---|---|
| Episode contract, registered RGB-D sensor depth, timestamps, pixel affines | CPU contract and geometry tests |
| RGB-D calibration (Panda + Robotiq, SAM 3 masks, 13 starts) | real episodes (mug-hanging 0/1, drawer-open, pick-and-place, covering): all cameras accepted |
| FoundationStereo on RealSense IR pairs (B1) | real episodes; conventions reviewed by Codex; regression tests for LR taps |
| TrackEverything single-view on B1 | real episodes (section 9 compares it with AllTracker on mug-hanging episode 1) |
| Dynamic-track filter, export with moving tracks | real episodes |
| Viewer (viser recordings, browser check, videos) | real episodes, headless Chromium checks pass |
| Exact own-rig Robotiq mount | nominal DROID stack, consistency-checked; not measured |
| Real-image landmarks or render-and-match source | annotation interface; render-and-match remains unimplemented |

## 9. Episode 1: five depth configurations (2026-11-05; historical evidence for the selection)

This section is the historical evidence for the standard configuration; the comparison tooling (and the MapAnything,
AllTracker and filtered-B1 stages it used) is retired (section 10). Its outputs are kept unchanged.
`configs/realworld/mug_hanging_ep1.yaml` (now `mug_hanging.yaml`), base root `processed/ep1/base` (shared stages incl. the frozen calibration),
variant roots `processed/ep1/{B0,B1,E1,E2,E3}`; report `processed/ep1/base/reports/<episode>/depth_variants/`.
No ground-truth depth exists: every number is consistency or diagnostic evidence; the calibration was fitted with
native depth, which favours B0 in robot and cross-view metrics.

| question | evidence (joint common support unless stated; median) | reading |
|---|---|---|
| B0 vs B1 | coverage 93.2 vs 98.7 % (boundaries 85.9 vs 95.4 %); cross-view mug 6.7 vs 5.0 mm, fixture 12.0 vs 4.7 mm, robot 5.3 vs 4.5 mm, scene 9.3 vs 9.4 mm; robot-model agreement 10.4 vs 10.4 mm (independent frames); static temporal std 3.7-5.5 vs 0.5-0.9 mm; B1 pixels where native depth is missing agree with other cameras' native depth at 31 mm (native vs native 10 mm) | B1 more complete and more cross-view consistent on objects, fixture and robot despite the B0-fitted calibration; its fills are less reliable than its measured-overlap pixels |
| B0 vs E1 | coverage 93.0 %; cross-view mug 10.5, robot 7.8 mm; robot-model 13.6 mm | MapAnything with native depth does not improve on its input |
| B1 vs E2 | E2 worse on every consistency metric (e.g. robot 8.0 vs 4.5 mm, gripper 9.1 vs 3.7 mm) | MapAnything degrades the FoundationStereo input |
| E1 vs E2 | E2 better on scene, mug, fixture, boundaries; worse on gripper (9.1 vs 6.5 mm); fills 24.8 vs 38.5 mm | FoundationStereo is the better MapAnything input except at the gripper |
| E2 vs E3 | small differences (fixture 6.2 vs 5.8 mm, boundary 9.2 vs 8.8 mm), similar coverage | the a-priori 1 px LR filter changes little |

Selected for lifting and export (recommendation for this rig, not a global optimum): **B1**, FoundationStereo on the
device IR pairs registered to the colour grid; native depth remains the calibration evidence. Timings (server): B1
667 s for 3 x 208 IR pairs (156 calls), LR filter 14 s, MapAnything 202-208 s per variant, lift 33-44 s, motion filter
14-17 s, export 200-232 s. Calibration profiling: `analysis/out/calib_profiling_ep1.md` (realworld workspace).

**Trackers on the selected depth (B1), per camera** (`analysis/tracker_compare.py`; identical query frames, pixels and
windows; tracks inferred independently by each tracker; proxy metrics, no ground-truth tracks):

| camera | valid samples AT / TE | static-scene 3D spread AT / TE | robot-FK error median (p90) AT / TE | moving tracks AT / TE |
|---|---|---|---|---|
| 135122070988 | 0.945 / 0.938 | 1.54 / 0.13 mm | 2.7 (7.4) / 2.2 (5.3) mm | 2729 / 2645 |
| 135122073201 | 0.942 / 0.931 | 1.14 / 0.13 mm | 3.0 (8.3) / 2.5 (5.7) mm | 1972 / 2069 |
| 135122076054 | 0.939 / 0.925 | 1.03 / 0.11 mm | 2.4 (6.8) / 2.3 (5.3) mm | 2531 / 2325 |

AT = AllTracker + per-frame lifting with B1; TE = TrackEverything single-view with B1, K and the frozen pose. The trackers
agree to 1.5-1.9 mm (3D median; 0.24-0.27 px in 2D). TE refuses about 5 % of queries whose birth depth is missing or
on a depth edge (no invented depth) and predicts 3D directly, so per-frame depth holes do not invalidate its samples;
its static points are about ten times steadier and its tracks follow the forward-kinematics prediction of robot
points slightly more closely. Runtime: TE 288 s (78 single-view clips) vs AllTracker 293 s + lifting 44 s; peak GPU
allocation 21.3 GiB vs 5.3 GiB. The robot-FK proxy uses the shared calibration and joints (independent of both
trackers); static steadiness can also reflect temporal smoothing. Recommended combination for this rig (not a global
optimum): **B1 depth + TrackEverything** for 3D motion targets, with AllTracker + lifting as the cross-check; TE's
checkpoint is CC BY-NC 4.0 (research use).

## 10. Retired interfaces (2026-11-05/06)

Retired when B1 + TrackEverything became the standard (directive 2026-11-05) and when DROID pretraining was confirmed
to use PointWorld's released depth/tracks + DROID-3D calibration only (2026-11-05 correction; nothing in this package
ran on large-scale DROID). Historical outputs, reports and raw recordings are kept; no compatibility layer remains.

| Retired | Replacement / note |
|---|---|
| Five-way depth comparison: `ops/inference_server/run_depth_variants.py`, `depth_compare.py`, `depth_compare_variants.py`, `depth_report.py`; `s4d/realworld/diagnostics.py`; kernels `depth_set_diagnostics`, `depth_compare_v2`, `depth_compare_multi`; config keys `depth_variant`, `tracker_comparison` | the selection is the standard config (section 9 keeps the evidence); variant roots under `processed/ep1/` are historical |
| `foundation_stereo_filtered` stage (`ir_stereo_depth.filter_stereo_depth`, E3 input) | B1 unfiltered (per-pixel `lr_error` no longer persisted since output layout 2; median in the stage summary) |
| FoundationStereo RGB `stereo_disparity` worker task | `ir_pair_disparity` (IR pair only) |
| `template_realsense` adapter (`adapters/template.py`, `configs/realworld/template_adapter.yaml`) | `panda_teleop_h5` (section 3) |
| Episode-0 configs `mug_hanging_rgbd.yaml`, `mug_hanging_rgbd_rawdepth.yaml`; `mug_hanging_ep1.yaml` (renamed) | `configs/realworld/mug_hanging.yaml` (standard; the episode-0 recording has no IR pair and is not run through a substitute depth) |
| `configs/realworld/pipeline_example.yaml` (synthetic stereo/MapAnything/AllTracker example) | the standard config |
| Raw-DROID route: `droid_raw_svo` adapter (`adapters/droid_raw.py`, `adapters/droid_svo.py`), `workers/svo_decode_worker.py`, `configs/realworld/droid_raw_validation.yaml`, `scripts/realworld_droid_diagnostics.py` (incl. its `sam3` subcommand), `cli.adapter_config`'s `calibration_source` coupling | DROID pretraining data: `s4d/data/droid/largescale/` (RLDS RGB + released PointWorld depth/tracks + DROID-3D calibration); [realworld_droid_validation.md](realworld_droid_validation.md) is historical |
| MapAnything: stages `mapanything_depth`, `mapanything_rig_depth`; `depth_stages.MapAnythingBackend`, `set_robot_motion`, `rectified_rig_pose`, `model_view_frames`, `sets_policy`, `DEPTH_STAGES`, prediction-plane helpers; `workers/mapanything_worker.py`; its weights, `ram_copies` (dinov2 hub code) and the `ram_copies` staging in `session.Server.stage_weights`; env recipe and pins | B1 (`foundation_stereo_depth`); `configured_sets_policy` reads only the root config (export tolerance) |
| AllTracker: stage `track2d` as a dispatched stage (`pipeline.track2d`, `TrackerBackend`, `WorkerTrackerBackend`, `backend_config`, `_model_input_rgb`), `tracking.alltracker_to_tracks2d`; `workers/alltracker_worker.py`; `locks/alltracker.txt`, weights, env recipe and pins | `track_everything` (writes the `track2d` + `lift3d` records) |
| Ordinary lifting: `lift3d` as a dispatched stage (`pipeline.lift3d`), kernel `lift_tracks`, `tracking.lift_tracks`, `tracking.status_counts` | written by `track_everything` (same chunk contract, `tracking.py`) |
| Stereo camera views: stage `stereo_rectify` (`pipeline.stereo_rectify`, `rig_rectification`, `grid_frames`, `calibration_image`), kernel `rectify_stereo`, stereo branches of calibrate/export/sam3_masks/track_everything, `depth_stages.view_K`'s `root`/`episode_id` (stereo grid) arguments | RGB-D views only; `pipeline.camera_views` refuses stereo views (the contract still represents stereo rigs; `geometry.stereo_rectify` remains a tested geometry function) |
| Provided calibration: `pipeline.calibrate_provided`, `stages.calibrate.source: adapter_provided`, `pipeline.applied_joint_correction` | robot-grounded `calibrate` (a `source` key is now an error) |
| Mock worker task `infer_views` (MapAnything/SAD oracle) | - |
| `dino_snerl` cache stage (`baseline_stages.dino_snerl`, `stages.dino_snerl`), the window `export` in the default stage graph (2026-11-06) | SNeRL DINO targets online from the stored RGB (`train_store.verify_snerl_teacher`); `train_data` (section 4.6); `export` still runs when configured |
| Defaults naming retired stages: `depth_stages.load_view_depth(stage="mapanything_depth")` (stage now required), `RGBD_LIFT_DEPTH = "mapanything_depth"` | `foundation_stereo_depth` |
