# Real-world preprocessing (Franka Panda + Robotiq + three RealSense rigs)

2026-11-05: FoundationStereo, removed in the morning (379bca5), was restored the same day as an OPTIONAL depth
backend on RealSense infrared pairs (`foundation_stereo_depth`) for a controlled five-way depth comparison
(section 9). Robot model: Franka **Panda** (not FR3) with the DROID-style Robotiq 2F-85 installation and no wrist
camera. Lifted tracks pass the noise-aware motion filter `track_motion` before export.

Maintained package: `s4d/realworld/`, branch `realworld-geometry-20261105`. Existing training and data modules do not
import it; `test_existing_code_does_not_import_realworld` enforces this boundary.

```text
Stereo: adapter -> ingest -> stereo_rectify -> mapanything_rig_depth -> calibrate
                                                                       |
                  export <- track_motion <- lift3d <- track2d <- mapanything_depth
RGB-D:  adapter -> ingest -> sensor_depth -> sam3_masks -> calibrate (evidence: rgbd) -> [depth for lifting] -> track2d
        -> lift3d -> track_motion -> export
        [depth for lifting] = sensor_depth (B0) | foundation_stereo_depth (B1, IR pairs) |
                              mapanything_depth with input depth B0 / B1 / foundation_stereo_filtered (E1 / E2 / E3)
DROID with provided poses: ingest -> stereo_rectify -> calibrate (source: adapter_provided) -> mapanything_depth -> ...
Optional comparison tracker: track_everything (TrackEverything single-view; replaces track2d + lift3d in its own root)
Optional baselines (real-world only; DROID pretrains only our method): sam3_masks, dino_snerl
```

Rig depth is needed by landmark calibration's depth term. Adapter-provided calibration can skip it. Stereo images,
tracks, masks, calibration and all depths share the rectified grid; raw and rectified grids are never mixed.
The RGB-D path registers recorded sensor depth for calibration and uses MapAnything depth for lifting/export.
New stereo MapAnything integration is checked with a flagged CPU mock; real-model stereo quality awaits validation.
See [status](realworld_preprocessing_STATUS.md) and [DROID validation](realworld_droid_validation.md).

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
- The fingerprint is a hash of the stage version, its config and its inputs. The config includes the full
  worker spec: commit, checkpoint, params and the mock flag. Inputs are the upstream *build ids*, which change
  on every rebuild, including a forced rebuild with an identical config, so downstream stages are invalidated.
  Stereo `track2d`, `calibrate`, and `sam3_masks` depend on `stereo_rectify` because they share that grid.
  MapAnything stages use durable chunks and acknowledge a runner result only after saving it.
- If the record exists with the same fingerprint, the stage is skipped.
- Otherwise the stage is built in a fresh `.<stage>.building.*` directory, which is swapped in atomically. The
  record is written last.
- An interrupted build leaves the previous complete result untouched (tested).
- Work runs per episode, and frames are processed in bounded chunks: `chunk`, tracker `window`/`step`.
- A per-stage lock file holds the builder's pid. A second builder is refused, and a lock left by a dead pid is
  taken over.

## 2. Normalized episode contract (`s4d/realworld/contract.py`)

`SCHEMA_VERSION = "s4d.realworld.episode/1.0.0"`. The dataclasses are:

| Dataclass | Fields |
|---|---|
| `Intrinsics` | width, height, fx, fy, cx, cy, skew, `pixel_convention` (`integer_centers` as librealsense/OpenCV, or `continuous`), `distortion_model` + coefficients (OpenCV plumb_bob/rational, RealSense model names kept verbatim, Kannala-Brandt), `source`. `K()` converts to the repository's continuous convention. |
| `CameraStream` | `stream_id`, `rig_id` (one RealSense), `eye` (left/right/mono), `modality` (`rgb`, `infrared`, or `depth`), `channels`, `channel_order` (RGB/BGR/GRAY), dtype (uint8/uint16), depth units/scale/kind, `rectified` (True/False/None = unknown), serial, sensor name, timestamp source, IR emitter state. |
| `StereoGeometry` | per rig: `status` (`metric`/`missing`/`unverified`), `T_right_left` (left-camera points into the right camera, metres), `baseline_m`, source. |
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

- **`panda_teleop_h5`** (`panda_teleop.py`): `panda_teleop_episode` format v2, JPEG colour + PNG uint16
  depth per RealSense, recorded intrinsics, depth scale and depth-to-colour extrinsics. Device stamps mapped onto
  the host monotonic clock are used; arrival times are not substituted. Colour/depth pairing, invalid flags and
  missing calibration are checked. Robotiq register 0..255 has its own clock and is interpolated as recorded.
  `configs/realworld/mug_hanging_rgbd.yaml` supplies the real robot assets and `evidence: rgbd` calibration.
  `include_ir: true` (episode 1, `configs/realworld/mug_hanging_ep1.yaml`) also exposes the device-rectified infrared
  pair of each RealSense as `<serial>/ir_left`, `<serial>/ir_right` with `EpisodeMeta.ir_stereo` per rig
  (`IrStereoGeometry`: IR-left K, equal IR-right K, recorded IR1->IR2 extrinsic with R = I and its baseline, the
  recorded colour<-IR-left extrinsic). The adapter refuses a pair unless the recorder's `pair_ok` flag is set, both IR
  frames carry the SAME hardware timestamp, and the depth stream's frame IS the IR-left frame. Nothing is fabricated:
  without recorded IR frames and extrinsics there is no `ir_stereo` and the FoundationStereo stage refuses the rig.
- **`droid_raw_svo`** (`droid_raw.py`, `droid_svo.py`): genuine raw ZED left/right SVO pairs, factory calibration,
  isolated decoder and optional DROID-3D provided poses. Joint corrections remain tied to that calibration source.

**Writing the real adapter:**
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

### 4.2 Shared grids and depth (`pipeline.py`, `depth_stages.py`, `geometry.py`)

- **`sensor_depth`** (RGB-D only): metric device depth is filtered by configured range and z-buffer registered to
  the colour grid with recorded intrinsics and `T_color_depth`. Zeroes and disocclusions stay invalid. Calibration
  uses this stage; it is also the default RGB-D metric depth input to MapAnything.
- **`stereo_rectify`** (stereo only): each rig has `<rig>.h5`, with `K_left`, `K_right`, `T_rr_rl` (metres) and
  rectification metadata. Already-rectified streams are recorded without copying images, preserving their K and
  recorded `T_right_left`. Otherwise Bouguet rotations, `rectify_map` and bilinear `remap(order=1)` produce both
  `left/rectified_image` and `right/rectified_image`; both K equal `K_rect` and
  `T_rr_rl = [R2 R_right_left R1^T | R2 t_right_left]`. Images retain recorded channel order; model inputs convert
  BGR to RGB and replicate infrared to three channels. Unknown rectification is refused; nonmetric rigs are skipped.
  `grid_frames`, `calibration_image` and SAM 3 use this grid and refuse an absent required rectification.
- **`mapanything_rig_depth`**: one `infer_views` call per metric rig and time, with two rectified views. The left
  camera pose is identity and the right pose is `T_rr_rl` in the rig frame (inverted for worker `cam2world`). No
  world calibration is read. Known metric poses are mandatory (`use_pose: false` is refused for both stereo depth stages); no sensor depth is
  supplied. Both eyes are saved, and landmark calibration reads the left stream's depth.
- **`mapanything_depth`**: synchronized frame sets use accepted calibrated poses. Each metric stereo rig contributes
  the rectified left eye; `include_right_eyes: true` (default) also adds its right eye with
  `T_right_base = T_rr_rl @ T_left_base`. The accepted left pose already refers to the shared grid: no additional
  left rectification rotation is applied. RGB-D views and their depth input remain unchanged. Predicted poses are
  diagnostics only. Primary-view depth drives lifting/export; right-eye outputs are optional and never consumed by
  training. Nonmetric stereo rigs are omitted, and missing/rejected metric-view calibration is an error.
  Recordings whose cameras are not hardware-synchronised (episode 1: one camera runs a constant 22.9 ms behind, one
  lacks the last frame) set `allow_unmatched: true`: a set is a reference frame whose MUTUAL nearest frame in every
  other view lies within `max_sync_ns` (must be below half a period); every other frame runs as a single view and is
  listed in the record, never merged into a set. `input_depth_stage` selects the MapAnything input depth
  (`sensor_depth`, `foundation_stereo_depth` or `foundation_stereo_filtered`); `output_stage` names the result.
  MapAnything's missing-depth representation is 0 (never NaN) with its validity mask; invalid input pixels are passed
  as 0.
- **`foundation_stereo_depth`** (optional, RGB-D rigs with a recorded IR pair; `ir_stereo_depth.py`, worker
  `foundation_stereo_worker.py`): FoundationStereo (pinned commit and ViT-L checkpoint `23-51-11`) on the
  device-rectified IR pair at native 640x480 (InputPadder 32, unpadded), grayscale replicated to 3 channels as model
  input only. Left-referenced disparity `d = x_left - x_right`, right-referenced disparity by flip-and-swap; depth
  `z = fx_ir * B / d` with the recorded baseline; invalid where `d <= min_disparity`, the match leaves the right
  image, or `z` is outside [`min_depth_m`, `max_depth_m`]. The IR-left frame is the native-depth frame, so the IR
  depth is registered to the colour grid with exactly the `sensor_depth` registration (z-buffer, same splat) and the
  winning IR pixel is stored (`source_index`). A failed model output is an error; native depth is never substituted.
- **`foundation_stereo_filtered`** (E3 input): keeps a `foundation_stereo_depth` colour pixel only if its source IR
  pixel passes the left-right consistency check `|d_L(x) - d_R(x - d_L(x))| <= max_lr_error_px` (1.0 px). It only
  invalidates: retained values are asserted bitwise identical; no erosion/dilation (thin structures keep their
  pixels unless they fail the check). Both right-disparity interpolation taps must be finite and exceed
  `min_disparity`. LR errors are stored and thresholded as float32. The IR contract requires
  `T_right_left[0,3] < 0` for the positive left-minus-right disparity convention.
  Thresholds and retained fractions are in the record.

All depth stages use `<stage>/<safe stream>.h5`: `depth` (N,H,W) float32 camera-z metres, `valid` bool, and attrs
`K`, `stream_id`, `rig_id`, `grid`. MapAnything also writes confidence, non-ambiguity and pose/scale diagnostics.
`default_depth_stage(meta, "calibrate")` selects `sensor_depth` for RGB-D and `mapanything_rig_depth` for stereo;
`"lift"` selects `mapanything_depth` for both. Mixed RGB-D/stereo episodes require explicit depth selection.
The automatic MapAnything depth input is `sensor_depth` for pure RGB-D and none for stereo; explicit
`input_depth_stage` overrides it. It is not a fallback to another model.

Synchronized sets must pair every frame exactly once within the tolerance. Sets exceeding the robot-motion
threshold are processed as individual views. Views currently need a common H,W; differing grids are refused.
Image resize/crop uses explicit continuous-coordinate pixel affines; depth uses nearest-neighbour sampling.
Server worker settings explicitly override interpreter, repo, checkpoint and `hf_dir` paths for remote runs; adapter
and output paths remain on the host. Existing RGB-D worker specs resolve identically.

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
   The annotation file format is
   `{camera_id: {frame_index: {"uv": [[u, v] | null, ...], "conf": [...]}}}`. Coordinates are continuous, on
   the rectified-left grid. Observations on a raw grid with distortion are undistorted to the ideal pinhole
   image automatically. Landmarks are listed in `landmark_links` order. Any source flagged `"is_mock": true` in
   its `identity` blocks export unless mock outputs are explicitly allowed.
2. **Visual initialisation.**
   - RANSAC PnP: normalised DLT, then a Cauchy-robust reprojection refinement.
   - When metric rig depth exists, also a RANSAC Kabsch fit of depth-lifted landmarks (bootstrap depth, no scale
     estimation).
   - The candidate with more inliers and lower reprojection error wins.
   - A previous episode's pose may be used only as a fallback prior (`T_init`).
3. **Robot-grounded refinement** (`scipy.optimize.least_squares`, Cauchy loss over the 6-DoF extrinsic):
   - landmark reprojection residuals, scaled by `cauchy_px` (5 px) and weighted by confidence;
   - robot-surface depth residuals: URDF surface points (meshes or primitives) projected and compared with bilinear
     rig depth, scaled by `cauchy_depth_m` (10 mm, DepthWorld's c3d). Two gates apply:
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
the remaining held-out frames. The calibration depends on the native depth (it is the evidence); depth comparisons
reuse ONE frozen calibration (section 9).

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
| Initial calibration from factory extrinsics, S2M2 stereo depth, RoMa-v2 ext-ext correspondences with MAGSAC, and two-pass bundle adjustment. | Initialisation from robot landmarks (PnP / depth Kabsch). Depth from MapAnything posed in each stereo rig frame. No ext-ext correspondences. |
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
  - `tracker_hw` optionally runs the tracker on bilinearly resized frames. The query grid then lives on the
    tracker grid, and positions are mapped back to the full rectified grid with the inverse `PixelAffine` before
    lifting (tested).
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
  - `lift3d.depth_stage` selects the lifting depth (default `mapanything_depth`; the depth variants of section 9 set
    it explicitly).

### 4.5b `track_motion`: noise-aware dynamic-track filter (`track_motion.py`, intentional change 2026-11-05)

Before this stage nothing filtered tracks by motion: every valid grid track (mostly static background) reached the
export. `track_motion` classifies every lifted track of one camera and one query window as `invalid`, `static`,
`moving` or `uncertain`; ids are never matched across cameras or windows. Filtering stored outputs does NOT reduce
AllTracker inference cost (the dense grid is still tracked).

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
- **Infrared.** The training model expects RGB, so infrared cameras are exported only with
  `allow_infrared_input=True`. The images are then channel-replicated, and the manifest says
  "infrared … replicated … (NOT rgb)".
- **Mock outputs.** Data derived from mock models is refused unless `allow_mock=True`. In that case the manifest
  carries `MOCK_MODEL_OUTPUTS: true` and the loader warns.
- **Loader.** The loader validates every item with `s4d.data.contract.validate_batch`.

### 4.6b `track_everything` (optional comparison tracker; `trackeverything_stage.py`, worker `trackeverything_worker.py`)

TrackEverything (pinned `ayushjain1144/trackeverything` dfb29b5, official checkpoint `katefgroup/TrackEverything`
`model.pth`, CC BY-NC 4.0) runs SINGLE-VIEW per colour camera, never its multiview path. Our metric depth, K and the
accepted `T_cam_base` are passed, so its geometry estimation is bypassed (`geometry_backend="pi3"` keeps the
cam0 alignment and median-distance normalisation of training; Pi3 itself is never loaded). Frames are resized to the
training resolution 512x384 (`K[:2] *= 0.8`, continuous convention, RGB Lanczos, depth nearest) with
`use_input_resolution=True`. Queries are EXACTLY AllTracker's (`chunk_queries` frames, `query_grid` pixels mapped by
the same affine); each clip is one query window of at most 16 frames, including frames across recording pauses.
Pause segmentation remains downstream in `track_motion` for both trackers. Births with missing depth or a 3x3
depth spread above 5% are excluded from inference and counted; they remain NO_DEPTH/EDGE at birth and NOT_TRACKED
elsewhere. Our worker uses floor query-depth lookup without filling holes. TE's own dense boundary-depth
replacement is retained and disclosed in metadata. Outputs go to the `track2d` and
`lift3d` stage directories of a separate root (`<variant>-TE`), with base-frame positions
`X_base = E0^-1 [norm * X; 1]`, sigmoid visibility and the lift3d status codes, so `track_motion`, `export` and the
viewer run unchanged. Settings live under `tracker_comparison.track_everything` so a default base run never replaces
AllTracker tracks.

Variant roots reuse shared stages through symlinks. The CLI excludes these stages from default dispatch and
refuses explicit shared-stage requests; `run_stage` also refuses a symlinked stage directory, including with
`force`. Calibration settings stay in the variant config for downstream readers. `depth_compare_variants` checks
saved depth and lift3d calibration build IDs before comparing, allowing sensor/IR depth to be calibration-independent.
Its edge-to-RGB diagnostics use pixels valid in every compared source. Thin/small-structure retention is depth
coverage (not correctness) within 3 px of SAM 3 object/fixture mask boundaries. Pairwise and joint cross-view
aggregates are reported for all sets and separately for independent sets: none of an independent set's camera-frames
may be calibration evidence for that camera.

The viewer uses the root's saved MapAnything synchronization policy, or `pipeline.yaml` if the stage is absent,
and otherwise requires strict pairing. Multi-camera playback contains only synchronized sets; unmatched frame
indices are reported by `viewer_data.describe()`.

### 4.7 Optional, baseline-only stages (`baseline_stages.py`)
**Scope (2026-11-05): only our method is pretrained on DROID.** DROID configs run no SAM 3/DINO baseline
preprocessing (`configs/realworld/droid_raw_validation.yaml`: `baselines: []`); for real-world episodes
`required_baseline_stages(cfg)` derives exactly the auxiliary stages the configured `baselines` need, and the CLI
refuses an explicit stage outside that scope. SAM 3 also supplies the robot masks of the RGB-D calibration (prompts
"robot arm" and "end effector"; "robotic gripper" detected nothing on this rig). Found by reading `s4d/baselines/*`:

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
  - Segmentation runs on the same grid as depth, tracks and export: rectified images when `stereo_rectify`
    rectified the rig.
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

| Worker | Environment (set up by the operator; nothing was installed here) |
|---|---|
| `mapanything_worker.py` | Isolated MapAnything environment and pinned `facebookresearch/map-anything` checkout. Local `facebook/map-anything` weights, revision `a1d87e9086706fb9974f3be5a3e3a0ca5401c5aa`; `model.safetensors` (CC-BY-NC-4.0). Uses `infer_views`, known K/metric poses and optional sensor depth. |
| `alltracker_worker.py` | `conda create -n alltracker python=3.12.8`, then `pip install -r requirements.txt` at the pinned commit. Upstream leaves torch unpinned: record the installed versions. Checkpoint `alltracker.pth` (Hugging Face `aharley/alltracker`). |
| `sam3_worker.py` | Python ≥3.12, torch 2.10 (cu128), `pip install -e .` at the pinned commit. Gated checkpoint `facebook/sam3`. |
| `dino_snerl_worker.py` | This repository's `.venv`. It is the consumer environment, with `third_party/dino` at 7c446df. |
| `foundation_stereo_worker.py` | Optional (restored 2026-11-05). Env `foundation_stereo` (Python 3.11, `locks/foundation_stereo.txt`), pinned `NVlabs/FoundationStereo` checkout imported from `repo_dir`; ViT-L checkpoint `23-51-11/model_best_bp2.pth` (sha256 `60e79bde...`) + its `cfg.yaml`, loaded strictly. The EdgeNeXt stem is constructed without its ImageNet download (every weight is overwritten by the strict load); upstream's visualisation-only `open3d` import is replaced by an empty placeholder module. |
| `trackeverything_worker.py` | Optional comparison tracker. Env `trackeverything` (Python 3.10, torch 2.8.0+cu129, `torch_scatter` 2.1.2 pt28cu129; `locks/trackeverything.txt`), pinned checkout as package `trackeverything`; `model.pth` (sha256 `4553975d...`, CC BY-NC 4.0). Single-view only. |
| `mock_worker.py` | Any Python. **Mock**: `infer_views` echoes supplied depth; without depth only an exactly two-view rectified rig is accepted by the CPU SAD oracle. Fake tokens/masks and all depth outputs are always flagged `is_mock`. |

**Upstream pins** (maintained as of 2026-11-05; full list and notes in `configs/realworld/upstream_pins.yaml`):

| Repo | Commit |
|---|---|
| MapAnything | `3d10cf7a3016fc0f9bb13a071ee66c47b10be0d9` (v1.1.3+1, 2026-08-07) |
| AllTracker | `e7553135e7b361590dbccd10e2b274b024f41cd6` |
| SAM 3 | `f1647e8d0c1d280991f4d1b7d974c2704537fe66` |
| DINO | `7c446df5b9f45747937fb0d72314eb9f7b66930a` |
| FoundationStereo (optional) | `6e8806816b533e4d13ddbb95ffa907b797060a62` |
| TrackEverything (comparison) | `dfb29b505cf2ea954f8697fb612c2429668dd3a0` |
| DepthWorld (reference only) | `6823c13ddcd96484de04604da7d52c9e12f5c86c` |

## 7. Run commands

```bash
# CPU-only tests (the conftest guard needs an allowed MIG UUID)
CUDA_VISIBLE_DEVICES=MIG-8da55219-bd6b-5260-8be1-3b2a76ac733c nice -n 10 /home/ws/ws/splatter4d/.venv/bin/python -m pytest -q -p no:cacheprovider tests/test_realworld_*.py
/home/ws/ws/splatter4d/.venv/bin/ruff check s4d/realworld ops/inference_server scripts/realworld_*.py tests/test_realworld_*.py
python -m s4d.realworld.cli validate --config configs/realworld/pipeline_example.yaml
# Stereo with landmark calibration
python -m s4d.realworld.cli run --config <stereo.yaml> --episode <id> --stages ingest,stereo_rectify,mapanything_rig_depth,calibrate,mapanything_depth,track2d,lift3d,export
# RGB-D (recorded sensor depth calibration, then MapAnything lifting depth)
python -m s4d.realworld.cli run --config configs/realworld/mug_hanging_rgbd.yaml --episode <id> --stages ingest,sensor_depth,calibrate,mapanything_depth,track2d,lift3d,export
# DROID-3D provided calibration does not require rig depth
python -m s4d.realworld.cli run --config configs/realworld/droid_raw_validation.yaml --episode <id> --stages ingest,stereo_rectify,calibrate,mapanything_depth,track2d,lift3d,export
python -m s4d.realworld.cli status --root <output_root> --episode <id>
# Same stages remotely; local CLI uses LocalRunner file protocol
python -I ops/inference_server/run_episode.py --pipeline <pipeline.yaml> --server configs/realworld/inference_server.yaml --episode <id> --stages <comma-separated-stages>
# Episode 1 (IR pairs): shared base stages once, then the five depth variants on the frozen calibration (section 9)
python -I ops/inference_server/run_episode.py --pipeline configs/realworld/mug_hanging_ep1.yaml --server configs/realworld/inference_server.yaml --episode mug-hanging-test-000001 --stages ingest,sensor_depth,foundation_stereo_depth,foundation_stereo_filtered,sam3_masks,calibrate,track2d
python -I ops/inference_server/run_depth_variants.py --pipeline configs/realworld/mug_hanging_ep1.yaml --server configs/realworld/inference_server.yaml --episode mug-hanging-test-000001 [--variants B0,B1,E1,E2,E3]
python -I ops/inference_server/depth_compare_variants.py --pipeline configs/realworld/mug_hanging_ep1.yaml --server configs/realworld/inference_server.yaml --episode mug-hanging-test-000001
# TrackEverything single-view on a chosen variant's depth (roots <variant>-TE)
python -I ops/inference_server/run_depth_variants.py ... --variants <V> --tracker trackeverything
# Viewer (viewer env with viser and PyYAML; CPU only, no MuJoCo): recordings + static client + index, browser check, video
python scripts/realworld_viewer.py record --source B0=<root>:sensor_depth:"B0 native" ... --episode <id> --out <dir>
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
| Stereo record-only/Bouguet grids, right-from-left rectified transform, refusal to mix grids | CPU geometry and pipeline tests |
| RGB-D calibration (Panda + Robotiq, SAM 3 masks, 13 starts) | real episodes 0 and 1: all cameras accepted (episode 1 held-out 7.8 / 7.1 / 7.2 mm) |
| FoundationStereo on RealSense IR pairs, LR filter | real episode 1 (B1, B1f); conventions reviewed by Codex; regression tests for LR taps |
| MapAnything RGB-D sets with unmatched frames | real episode 1 (E1-E3) |
| Dynamic-track filter, export with moving tracks | real episodes 0 and 1 |
| TrackEverything single-view (matched AllTracker windows, no invented query depth) | real episode 1 on B1 depth; see section 9 |
| Viewer (viser recordings, browser check, videos) | episodes 0 and 1, headless Chromium checks pass |
| Exact own-rig Robotiq mount | nominal DROID stack, consistency-checked; not measured |
| Real-image landmarks or render-and-match source | annotation interface; render-and-match remains unimplemented |

## 9. Episode 1: five depth configurations (2026-11-05)

`configs/realworld/mug_hanging_ep1.yaml`, base root `processed/ep1/base` (shared stages incl. the frozen calibration),
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

