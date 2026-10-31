# Real-world preprocessing: status (2026-10-31)

Branch `realworld-preproc-20261031`, worktree `.worktrees/realworld-preproc`. It forks from `2605ba2`. It is not
merged or pushed, and no existing module imports `s4d.realworld`. Design and usage are in
[realworld_preprocessing.md](realworld_preprocessing.md).

## Done, validated on synthetic fixtures (CPU numpy/scipy; no real data, assets or models)
- **Versioned episode contract** `s4d.realworld.episode/1.0.0`:
  - it represents modality honestly (infrared is 1-channel and never relabelled RGB);
  - it records intrinsics with their pixel convention and distortion;
  - it records an explicit stereo status (`metric`/`missing`/`unverified`);
  - it also covers robot description and trajectory, provenance, conventions and adapter assumptions;
  - validation and a JSON/NPZ round trip are in place.
- **Adapter interface.** A template adapter for the planned RealSense rigs that invents no layout, and a
  synthetic adapter.
- **Stereo.**
  - Disparity to metric depth, including doffs and validity masks.
  - Right-reference depth by flip-and-swap, checked against ground truth; a naive swap fails.
  - Bouguet rectification and remap, including inside the pipeline (rotated right camera: 0.9 % median relative
    error, versus 20 % without rectification).
  - K tracking through resize, crop and flip.
  - Nearest-neighbour depth resampling.
  - Depth transfer to a second camera.
- **Robot.** URDF FK with mimic joints and safe number parsing, matching Franka DH. Mount composition,
  gripper-state mapping, surface sampling from meshes or primitives, and a depth-gated robot mask.
- **Calibration**, per episode and per camera.
  - Visual initialisation: RANSAC PnP, or depth-lifted Kabsch.
  - Cauchy-robust robot-grounded refinement: landmark reprojection plus robot-surface depth, with self-occlusion
    and occlusion gating.
  - Recovery under noise and 5 % outliers: < 0.3° and < 5 mm.
  - Explicit rejection of insufficient visibility, a static robot, missing visual evidence and uncertain or
    ill-conditioned solutions.
  - Held-out residuals, overlays, and a check of FK against the recorded EE pose.
- **Tracking.**
  - AllTracker integer-centre to continuous conversion.
  - Optional reduced-resolution tracking (`tracker_hw`), with positions mapped back to the depth grid by the
    inverse pixel affine (tested).
  - Same-frame depth lifting with per-point status codes (occluded, low confidence, out of image, no depth,
    depth edge, not tracked).
  - NaN for invalid points, camera-local ids, and no cross-camera identity.
- **Export to the accepted DROID-track training contract.**
  - Uses the same `s4d.data.droid.pointworld` helpers, and the loader passes `validate_batch`.
  - Refuses mock outputs, and infrared presented as model input, unless explicitly opted in; the manifest
    records it when that happens.
- **Restartable stages.**
  - Fingerprinted completion records, atomic swap-in, and the old result kept on interruption.
  - Bounded chunks and per-episode processing.
- **Worker protocol** (subprocess, `-I`, pinned commit check, mock refusal). Tested with the flagged mock worker.
- **Optional baseline stages.**
  - SNeRL DINO teacher spec pinned to the consumer code, with a token cache plus an in-worker equality check
    against `dino_target`.
  - SAM 3 masks with prompts mapped to SNeRL classes, instances, scores and explicit absent detections.

**Tests:** 42 passed across the five files below. Run with `nice -n 10` and the MIG UUID:
- `tests/test_realworld_stereo.py`: 10
- `tests/test_realworld_calibration.py`: 9
- `tests/test_realworld_pipeline.py`: 9
- `tests/test_realworld_contract_workers.py`: 11
- `tests/test_realworld_baselines.py`: 3

## Unverified / assumptions (must not be described as real integration)
- The FoundationStereo, AllTracker and SAM 3 worker scripts were written against the upstream READMEs and demos
  at the pinned commits and have **never been executed**: there are no environments or checkpoints on this host,
  and installing them was out of scope.
- The DINO worker only runs with the official weights; not executed here.
- Every end-to-end result uses MOCK models: a CPU block matcher, an oracle tracker and oracle landmarks.
- The synthetic robot is a stand-in: the official Panda joint origins with primitive links and a two-finger
  gripper. It is **not** the Panda or Robotiq meshes.
- The gripper-state to `finger_joint` mapping is assumed linear (2F-85: 0.8 rad, 2F-140: 0.7 rad).
- The Robotiq mount yaw of +π/2 follows Franka's example and DROID comments. The coupling offset is unknown.
- For RealSense intrinsics, the pixel convention (integer centres), the IR rectification in Y8 mode and the
  timestamp domains are assumptions until the first real recording.

## Remaining steps (in order)
1. Obtain one real episode, write the real adapter from `adapters/template.py`, and run `cli validate`. Then
   record the confirmed assumptions in the adapter.
2. Supply the robot assets: Panda URDF, the exact Robotiq model (2F-85 or 2F-140) URDF with STL/OBJ meshes, and
   the measured `T_flange_gripper`. Verify the gripper-state mapping and check `fk_vs_recorded_ee`.
3. Create the pinned worker environments: `foundation_stereo`, `alltracker` and `sam3` (gated). Download the
   checkpoints, record their sha256 in `configs/realworld/upstream_pins.yaml`, and run each worker once on real
   frames. Compare the FoundationStereo depth against RealSense on-board depth as a sanity check only.
4. Choose and implement the calibration landmark source for real images. Options: manual annotations of about
   10 frames per camera per episode, markers on the gripper, or a render-and-match worker (the DepthWorld LoMa-G
   analogue). Then tune the gates on real residuals.
5. If RGB appearance from the colour sensor is wanted, wire `geometry.reproject_depth` into a stage. That needs
   the depth-to-colour extrinsic from the recording.
6. Optional: a multi-episode joint refinement per physical setup, which is closer to DepthWorld. Also define
   3D-MVP scene bounds in the base frame for the real workspace.
