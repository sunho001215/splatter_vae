> **2026-11-05:** FoundationStereo was removed from the maintained pipeline and replaced by MapAnything
> (`docs/realworld_preprocessing.md`). This report is the historical 2026-10-31 evidence for the implementation it
> describes; its FoundationStereo measurements do not validate the maintained depth stage.

# Real-data validation of the real-world pipeline on a raw DROID episode (2026-10-31)

This validates items 5 (FoundationStereo + AllTracker, 3D lifting) and 6 (SNeRL DINO teacher, SAM 3 masks). It runs
the actual upstream models on genuine stereo pairs through this repository's own entry points:
`python -m s4d.realworld.cli run` with the `droid_raw_svo` adapter and `scripts/realworld_droid_diagnostics.py`.

**What it can and cannot establish.** DROID uses ZED 2 RGB stereo (external) and ZED Mini (wrist) cameras, on a
Franka with a Robotiq 2F-85.
- It validates the model integrations, the stereo/rectification geometry, units and conventions, timestamps, tracking
  and lifting on real images.
- It cannot establish the future RealSense stream configuration: infrared pairs, emitter, stream modes and the
  RealSense SDK intrinsics.
- It cannot establish the Panda–Robotiq mounting geometry of our own rig.
- One episode shows functional integration, not robustness.

## Data (read-only inputs; every output under `/home/ws/ws/validation-20261031/item5`)

| Input | Source | Size / hash |
|---|---|---|
| Raw episode `AUTOLab+0d4edc83+2023-10-21-19h-07m-04s` (the historical sample of `docs/droid_validation/`) | `https://storage.googleapis.com/gresearch/robotics/droid_raw/1.0.1/AUTOLab/success/2023-10-21/Sat_Oct_21_19:07:04_2023/` | 34.8 MB; `SHA256SUMS` beside the files |
| ZED factory calibration `SN22008760.conf`, `SN24400334.conf`, `SN18026681.conf` | `https://calib.stereolabs.com/?SN=<serial>` (the file the ZED SDK downloads) | 1.2–1.6 kB each; `SHA256SUMS` |
| DROID-3D extrinsics and intrinsics | HF dataset `jaibrdhn/droid_3d_extrinsics`, revision 69db9ccf9a51 | `extrinsics.jsonl` 100 MB, `camera_intrinsics.jsonl` 203 MB |
| PointWorld-DROID depth of the same episode (reference depth only) | `/home/ws/data/pointworld_droid_sample` (revision dd9aaeec) | sha256 c1c7e10c… |

The SVO files contain each ZED's raw side-by-side left|right HEVC stream, which is a genuine stereo pair from one
rig. The MP4 files are the SDK-rectified left views and serve as a reference. Decoding uses PyAV in an isolated
environment, so no ZED SDK is needed.

## Findings

### Format and timing
- SVO v1 layout: a header (`-11, width, height, serial, fps`), then `[uint64 ts_ns, uint32 type, uint32 size]` frame
  records of HEVC, interleaved with sensor blocks. 127 records per camera (key frames 0 and 120), each 2560×720.
- Each record timestamp equals `<serial>_frame_received` (ms) of `trajectory.h5`. All 127 frames map to trajectory
  steps 0–126; step 127 has no frame, which matches the 127-step RLDS episode.

### Factory calibration reading
The convention was decided against the SDK's own output. Eight hypotheses were tested (`rect_hypotheses_*.json`):

| Reading | ext1 L/R row error (median \|dy\|) | ext1 vs SDK MP4 (median \|d\|) |
|---|---|---|
| **R = Rodrigues(RX, CV, RZ) left→right, right centre (B, −TY, −TZ), `[*_DISTO]` rational model** | **0.41 px** | **0.32 px** |
| same, 5-coefficient `[*_CAM_HD]` model | 0.44 px | 1.44 px |
| same, centre (B, +TY, +TZ) | 0.36 px | 1.56 px |
| transposed rotation | 2.8 px | 1.5 px |

- Baselines equal DROID-3D's SDK baselines exactly: 120.058 mm, 120.155 mm and 63.061 mm.
- The wrist ZED Mini uses a 4-coefficient Kannala–Brandt model. Rows align at 0.7 px, but there is a 2.2–2.9 px
  horizontal offset to the SDK. This is not resolved; the wrist is not part of this fixed-rig run.

### The pipeline's own rectification (`diagnostics/rectify.json`)
- `geometry.stereo_rectify` + `rectify_map` + `remap` on the ingested real frames.
- R1/R2 equal OpenCV's `stereoRectify` to 2×10⁻⁶ °.
- With the SDK's rectified K (converted to the continuous convention, +0.5) and our R1, the rectified left view
  matches the SDK MP4:
  - ext1: dx 0.003 px, dy −0.055 px (median over 6501 SIFT matches);
  - ext2: dx +0.40 px, dy −0.05 px.
- With the pipeline's default K_rect, the left/right row residual is median |dy| 0.40 px (ext1) and 0.38 px (ext2).

<!-- results of the model stages are appended below as they are produced -->

## Models, environments and provenance

Every model runs in its own isolated uv environment under `/home/ws/ws/validation-20261031/envs` and is called
through the worker protocol. The environments use torch 2.10.0+cu129 because the GPUs are Blackwell (sm_120). The
upstream FoundationStereo pin, torch 2.4.1, has no sm_120 kernels; that is the main environment deviation.

| Model | Upstream commit | Weights (sha256) | Environment notes |
|---|---|---|---|
| FoundationStereo ViT-L `23-51-11` | NVlabs/FoundationStereo 6e88068 | official Google Drive file, 3,298,527,334 B (60e79bde…) | py3.11, torch 2.10. No xformers, so the vendored DINOv2 layers fall back to standard attention; flash-attn is unused because upstream calls SDPA. The `torch.hub` DINOv2 code is pre-populated at facebookresearch/dinov2 7764ea0f, since upstream would fetch the unpinned default branch. timm downloads `edgenext_small` pretrained weights at construction; the strict checkpoint load overwrites them. open3d needs libusb, supplied from `libusb-package` via `LD_LIBRARY_PATH`. |
| AllTracker | aharley/alltracker e755313 | HF `aharley/alltracker` rev c8ba3122 (ffd9ebcf…) | py3.12, torch 2.10, opencv-python-headless 4.10.0.84 (the headless build of the pinned version). torchvision `convnext_tiny` ImageNet weights are downloaded for initialisation only; the strict load overwrites them. |
| SAM 3 | facebookresearch/sam3 f1647e8 | gated HF `facebook/sam3` rev 3c879f39 `sam3.pt` (9999e234…) | py3.12, torch 2.10, numpy 1.26, setuptools < 81 (upstream imports `pkg_resources`). Run with `HF_HUB_OFFLINE=1`. |
| DINO ViT-S/8 (SNeRL teacher) | facebookresearch/dino 7c446df (`third_party/dino`) | `dino_deitsmall8_pretrain.pth` (55c8b267…, enforced) | The repository venv: the SNeRL consumer's own environment. |
| SVO decoder | PyAV 18.1.0 (libavcodec 62.28.102) | — | py3.11 media environment |

The SNeRL contract was checked in the pinned upstream (`third_party/snerl` d01533b, `nerf_pretrain/run_nerf.py`):
- `ViTExtractor('dino_vits8', 4)` at line 662;
- `(target − 0.45)/0.225`, then `extract_descriptors(…, 11, 'key', False).view(1, 31, 31, 384)`, then bilinear
  `F.interpolate` to (128, 128), at lines 712–715.

`s4d.baselines.snerl.model.dino_target` and the worker reproduce this exactly. The worker checks equality on the
first image of every call.

## Item 5: FoundationStereo through `stereo_depth` (`diagnostics/stereo.json`, `depth_*.png`, `cross_*.png`)

- **Run.** Both rigs, 127 frames each, left and right references, in 33.6 min on one MIG 1g.24gb slice. That is 16
  worker calls per rig, one per chunk and reference. Every record carries the model identity: commit 6e88068,
  checkpoint sha256 60e79bde…, `is_mock: false`.
- **Rectification and intrinsics.** The rectification ran inside the stage. The depth grid's K is the pipeline's
  K_rect: fx 533.17 / 533.93 px, the left raw principal point, continuous convention. Depth is metric camera-z in
  metres, with 0 marking invalid pixels.

| Check (every 10th frame) | ext1 | ext2 |
|---|---|---|
| valid fraction, left / right reference | 0.913 / 0.940 | 0.945 / 0.883 |
| median depth (m) | 0.868 | 0.721 |
| FoundationStereo − SIFT disparity at independent L/R matches: median, median \|·\| (px) | −0.024, 0.41 | −0.036, 0.49 |
| right-reference flip-and-swap vs left reference, \|d_R(x − d_L) − d_L\|: median px, fraction < 1 px | 0.32, 0.75 | 0.22, 0.86 |
| PointWorld-DROID full depth (320×180), signed median, relative median, fraction < 2 % | −1.4 mm, 0.61 %, 0.80 | −2.6 mm, 0.50 %, 0.91 |

**Cross-camera metric consistency (independent rigs, joined only by the DROID-3D extrinsics).** Each rig's left depth
is lifted with its pose, z-buffered into the other rig, and compared with that rig's own stereo depth.
- ext1→ext2: median signed −1.0 mm, median relative 0.75 %; 74.8 % of pixels below 2 %.
- ext2→ext1: median signed +3.6 mm, median relative 0.82 %.
- The tails come from occlusion edges and non-overlapping regions, which are included in the statistic.

This checks units, the baseline, rectified K and the w2c convention end to end.
- If DROID-3D's view matrix had been read as c2w, the lifted points would not land on the other camera's image at all.
- A baseline or fx error of x % would bias every comparison by x %.

The PointWorld depth is probably itself a stereo-network output, so it is a consistency reference, not ground truth.

## Recorded baseline run (revision 81b202c, 2026-11-02, `item5/runs/baseline-81b202c`)

The run is the clean checkout of 81b202c, with 0 dirty files, driven by `item5/run_all.sh`. It did every stage from
scratch except ingest, which was reused after its fingerprint matched. The workers were remapped to the temporary
validation reservation, MIG-8da55219 and MIG-c4ec2a95; the effective config is in `config.yaml` of the run.

Stage completion IDs (fingerprint prefixes):
- normalized c1337020
- stereo_depth 3c83da37
- calibrate 107acc2f
- track2d eb29bd35
- lift3d e2b08a85
- export: 148 windows
- sam3_masks
- dino_snerl

| Check | ext1 | ext2 |
|---|---|---|
| L/R row residual, median \|dy\| (px) | 0.41 | 0.38 |
| vs SDK-rectified MP4, median dx / dy (px) | 0.002 / −0.055 | 0.392 / −0.050 |
| FoundationStereo − SIFT disparity, median (px) | −0.024 | −0.036 |
| right-reference flip-and-swap consistency, median (px) | 0.32 | 0.22 |
| cross-rig depth via DROID-3D poses, median relative | 0.75 % (ext1→ext2) | 0.82 % (ext2→ext1) |
| PointWorld depth, median relative | 0.61 % | 0.50 % |
| lifted static tracks, median max deviation | 7.2 mm | 4.2 mm |
| lifted points vs the other rig's depth, median relative | 0.62 % | 0.53 % |
| AllTracker temporal indexing | ok | ok |
| SAM 3 box vs mask IoU (median) | 0.985 | 0.989 |
| SAM 3 detections (frames of 127): robot arm / gripper / blue brick / drawer | 113 / 0 / 95 / 127 | 127 / 0 / 0 / 127 |

**Export (`diagnostics/export.json`).** 148 windows load through `RealworldCacheDataset`, which calls
`s4d.data.contract.validate_batch`, and every one passes.
- Every float field is finite.
- images (3,2,3,144,256) uint8; depth max 4.15 m.
- 858 tracks per camera.
- gripper_points (3,32,3); probe_state (3,7).
- The validation split is empty by configuration (`validation_tail_frames: 0`).

**`dino_snerl` pipeline stage (not only the worker).** It ran on the exported windows:
- token grids (3,2,31,31,384), stored as float16, for all 148 windows;
- model identity: commit 7c446df, sha256 55c8b267…;
- `verified_against_consumer: true`, meaning the token path equals `s4d.baselines.snerl.model.dino_target`.

**SAM 3 consumer contract.**
- The class order equals SNeRL's `SEMANTIC_CLASSES`.
- Nearest resize to 128×128 gives int64 class ids ≤ 3.
- SNeRL's `CrossEntropyLoss` over the 5-way head is finite.
- Absent detections are recorded explicitly, never filled: "gripper" was not detected in either camera, and the brick
  was not detected in ext2.

## DROID robot/gripper assets and mount (DROID-specific; NOT our Panda–Robotiq rig)

**Assets.** The arm is FR3 from franka_description 7aeeddc; the gripper is the Robotiq 2F-85 from PickNik
ros2_robotiq_gripper a74d007, using its collision STLs.

**URDF vs robot metadata.** URDF flange FK (`fr3_link8`) equals DROID `cartesian_position` to 3e-8 m / 9e-7° over the
episode, and equals the Panda DH chain to 7e-16 m.

**Mount, checked against the images.** Gripper surface points (200 per link) are projected into both cameras every 2–4
frames, and the score is the fraction landing on SAM 3's robot mask (`item5/mount_check/*.json`).

| Mount yaw | Score |
|---|---|
| ±90° | 0.80 |
| 0° | 0.70 |
| 45° | 0.66–0.72 |

- The assumed yaw of +90° (Franka external_gripper_example) is confirmed; 180°-symmetric alternatives are
  indistinguishable for the 2F-85.
- Along the approach axis the score peaks between +1 and +2 cm (0.83 at +2 cm, 0.79 at 0; 76 frames). This 1-D scan was
  read at the time as a coupling of about 1.5 cm that the assumed xyz = 0 omits. It is superseded by the analysis
  below.

**Provisional mount (2026-11-02, `s4d/realworld/mounts.py` `droid_robotiq_2f85_iso50_coupling`).**

`T_fr3_link8→robotiq_85_base_link` = xyz (0, 0, **0.011** m), rpy (0, 0, +π/2).

Status: **PROVISIONAL**. It is applied as the default for guide-built DROID robots until a measurement exists. It is
derived from Robotiq drawings, the coupling's identity is inferred from photos, and it is not measured. Only magnitude
cross-checks corroborate it.

- **Parts (partly inferred).** The DROID assembly guide (`droid-dataset/droid` `33ae6a6`
  `docs/hardware-setup/assembly.md`, "Mounting Robotiq Gripper on Robot") names only a "gripper mount" taken from the
  gripper box.
  - Its photos show a Robotiq pigtail coupling (AGC-CPL-062-002 family) screwed directly to the Franka flange, with no
    spacer. The flange is ISO 9409-1-A50 on both Panda and FR3.
  - The coupling's exact part number and revision are not established.
  - The ZED Mini bracket (the guide's only printed parts, `hand_camera_part_{1,2}.stl`) screws to the link-7 ring. It
    is not in the stack.
- **Stack.** From the Robotiq 2F-85/2F-140 manual §6.1.1:
  - GRP-CPL-062 drawing: 16.9 mm overall, minus its 3 mm robot-side Ø63 F8 recess (which takes the flange).
  - Minus 2.9 mm for the gripper-side Ø71 recess (which takes the gripper's Ø71 h8 boss). This number comes from the
    dimensioned *blank* coupling AGC-CPL-BLANK-002 and is inferred to be the same on DROID's coupling.
  - Total: 11.0 mm from the flange face to the boss face. This assumes the flange spigot fully seats in the coupling.
  - The boss face is PickNik's `robotiq_85_base_link`: the URDF fingertip tops sit at 149.3 / 162.8 mm above it, which
    are Robotiq's open / closed heights.
- **Corroboration (magnitude only; neither identifies DROID's part).**
  - Robotiq's TCP table puts the TCP at 174.0 mm, coupling included: 174.0 − 162.8 = 11.2 mm.
  - PickNik's `ur_to_robotiq_adapter` (the same ISO-50 coupling, on UR) places the gripper at z = 0.011.
- **Scope.** Provisional default for DROID robots built per the guide, Panda and FR3 alike. Robots with verified
  different hardware need their own `MountConfig`.
  - Both arm generations share the same flange and flange frame. DROID's polymetis `ee_link` is `panda_link8` for
    both.
  - The `panda-` / `fr3-` serial prefixes are typed in by each lab and are inconsistent: `290102-1324143` carries both.
- **Assets.** The old asset (xyz = 0) omitted the coupling: neither input URDF contains one. The new asset is
  `/home/ws/ws/validation-20261031/droid_mount/robot_assets/fr3_robotiq_2f85_droid_iso50.urdf`.
  - It is composed with `--mount-config droid_robotiq_2f85_iso50_coupling`.
  - Its manifest records `status: provisional`, the stack, sources, scope, caveats and `missing_measurement`.
  - `configs/realworld/droid_raw_validation.yaml` now uses it. Results produced before this change carry the old
    asset's sha256 (`9d25470e…`) in their `episode.json`.

**What the images do and do not show** (`/home/ws/ws/validation-20261031/droid_mount/`, 172 frames, both cameras).
- An axial-only scan still peaks at +17–18 mm, both with and without DROID-3D `dq`, and in both cameras.
  - Re-run with the adapter's corrected geometry FK(q + dq) (below), the scores are identical to the manual `dq` run.
- A lateral control finds the best fit at dx = −12, dy = +4 mm in `fr3_link8`. A coupling cannot cause that offset.
  - The offset points towards the unmodelled ZED Mini.
  - Mask score 0.902 against 0.836 at no lateral shift.
  - Only the axial positions 5, 11, 17 and 23 mm were sampled.
- With the lateral shift free, the two scores disagree: the mask fraction peaks at the sampled +11 mm, the depth-inlier
  fraction at the sampled +17 mm.
- These are competing sampled optima under an unresolved registration and model bias: whole-robot mask, no camera or
  bracket geometry, calibration error. They give no confidence interval and no bound.
- The single-episode fit is not used, and it must not be applied to other robots.

**Not affected.**
- `data.gripper_offset_m` (−0.019) is a fitted shift of a 32-point surrogate in DROID's flange-frame EE pose. It never
  reads the URDF.
- DROID-3D `T_gw_static` / `dT_gw` is the wrist camera relative to the flange, and the coupling is not in that chain.
- Both stay unchanged (see `docs/droid_largescale.md` §2.5).

**Joint offsets with DROID-3D calibration (adapter fix, 2026-11-02).** The `droid_raw_svo` adapter now supplies
FK(q + dq) joints in exactly one case: the episode is calibrated with its own DROID-3D record, i.e. the CLI passes
`calibration_source: adapter_provided`.
- The record must match the episode and the robot serial, the serial must be established, and `dq` must be 7 finite
  values.
- Raw `q` is kept in `extra.joint_positions_raw`. The decision is recorded in `provenance.notes.joint_correction`.
- Otherwise the adapter keeps raw `q`, and it never uses another robot's `dq`.
- `calibrate_provided` refuses poses from any other record, and robot-grounded calibration refuses corrected joints.
- On the validation episode the FK cross-check stays at 2.9e-8 m, and the mount scores equal the manual `dq` run.

**Missing measurement (what would make the mount non-provisional).** On a guide-built DROID robot:
- Along the flange normal, the distance from the Franka flange face (`link8`) to the Robotiq 2F-85 Ø71 boss face
  (`robotiq_85_base_link`). Equivalently, flange face to closed fingertip ends, minus 162.8 mm.
- The coupling's part number and revision, or its dimensioned drawing or CAD.
- The gripper yaw relative to the flange dowel-pin hole.

- None of this establishes the mount of our future Panda–Robotiq rig, which must be measured.

## Status matrix (items 5 and 6, real DROID data)

| Component | Status | Evidence |
|---|---|---|
| Raw stereo ingest (SVO decode, frame/step matching) | VERIFIED-REAL | 127/127 frames per stream matched to trajectory steps |
| Rectification + intrinsics (factory calibration reading) | VERIFIED-REAL | SDK MP4 reproduced to ≤0.06 px dy; OpenCV equality |
| FoundationStereo left reference, metric depth | VERIFIED-REAL | SIFT, cross-rig and PointWorld agreement above |
| Right-reference procedure (flip-and-swap) | VERIFIED-REAL | L/R consistency 0.22–0.32 px |
| AllTracker tracks (identity, visibility, confidence, temporal indexing) | VERIFIED-REAL | tracks.json, overlays |
| Lifting to 3D with matching depth + poses | VERIFIED-REAL | static drift 4–7 mm; cross-rig 0.5–0.6 % |
| Calibration from provided poses (DROID-3D) | VERIFIED-REAL | positions within ~2 cm of DROID's own; cross-rig depth |
| Robot-grounded calibration (landmarks) | NOT-RUN on real data | no landmark annotations or detector for DROID |
| Export + batch contract | VERIFIED-REAL | 148/148 windows through validate_batch |
| DROID gripper mount | PROVISIONAL (not measured) | 11.0 mm Robotiq ISO-50 stack + yaw +90° from drawings, coupling identity inferred from guide photos (`mounts.py`); single-episode image optima disagree (mask 11 mm, depth 17 mm, with an unexplained 1.2 cm lateral preference) |
| DINO (SNeRL contract) worker + `dino_snerl` stage | VERIFIED-REAL | consumer equality, 148 windows |
| SAM 3 text-prompted temporal masks + `sam3_masks` stage | VERIFIED-REAL (with model misses recorded) | sam3.json |
| Wrist ZED Mini in the fixed-rig pipeline | NOT SUPPORTED / not run | KB4 model: unexplained 2.2–2.9 px offset; item 8 handles the moving wrist |

DROID ZED validation cannot establish our future RealSense stream configuration: infrared pairs, emitter, modes and
RealSense intrinsics. It also cannot establish our Panda–Robotiq mounting geometry. One episode shows functional
integration, not robustness.
