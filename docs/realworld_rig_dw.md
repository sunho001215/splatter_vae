# Board-free per-task rig calibration (`rig_dw`)

Status: study + implementation (branch `rw-calib-dw-20261107`). Results, verdicts and the board cross-validation are
in `/home/ws/ws/realworld-ops/calib-dw/REPORT.md`. Directive: user 2026-11-07 (board-free, DepthWorld-style, per task
independent, calibration.h5 only for a final post-freeze cross-validation).

## Why
The per-episode robot-grounded solve (`calibration.calibrate_camera`, calibrate@1) is ambiguous: each camera sees the
arm only through along-ray depth residuals, near-best solutions differ by 0.5-2 deg / up to 24 mm and frame subsets by
up to 3.4 deg / 37 mm (study CAL, `realworld-ops/calib/README.md`). Root causes found here:
1. **No cross-camera constraint**: each camera was solved alone; its weak direction (an orbit about the arm) is
   unconstrained. The current per-episode poses violate the static-scene epipolar geometry by ~3 px (Sampson median;
   the rig_dw solution ~0.2 px).
2. **Model error**: the composed URDF uses the Franka/Robotiq COLLISION meshes (86-976 triangles per link); they
   differ from the visual meshes by median 2-4 mm, p90 8-23 mm per Panda link - the size of the per-link systematic
   residuals the per-episode poses absorbed differently in different workspaces.
3. **Depth-scale coupling**: a camera fitted to robot depth absorbs the depth source's scale error as a translation
   along its optical axis (1 % at 1.4 m = 14 mm). The depth sources disagree by up to ~3 % between cameras.

## Method (per task, one rig for all 100 episodes, three cameras jointly)
* **Dense static correspondences** (`workers/romav2_match_worker.py`): RoMa v2 (github.com/Parskatt/romav2 @ 95c9968,
  weights romav2.0.1.pt sha256 1557dec0..., DINOv3 hub code adc25445) in the `precise` setting (bidirectional), 6 time
  points per episode x 3 camera pairs, nearest-time frames (|dt| <= 20 ms), 5000 balanced samples per pair; per match:
  cycle error, SAM 3 class/distance to robot+object pixels at both ends, FS and sensor depth, 5x5 FS depth range,
  MAGSAC++ fundamental inlier (static matches only).
* **Two-view RGB-D factor** (`rig_dw_extext.py`): latent base-frame point per match, residual
  `[pi_a - u_a, pi_b - u_b, log z_a - log d_a - alpha_a, log z_b - log d_b - alpha_b]` (whitened 0.5 px / 1 %), point
  eliminated by a per-match Schur complement; Cauchy on the factor. Filters and stratified subsampling in
  `rig_dw_matches.py`.
* **Robot factors** (`rig_dw_robot.py`, `rig_dw_frames.py`): URDF FK at the camera timestamp (+ effective time offset
  tau_c, surface velocity from FK at t +- 10 ms), VISUAL meshes (`robot_visual.py`, COLLADA reader, 1.5 mm samples),
  point z-buffer render on the robot crop; (i) model->observed silhouette: extreme sample of each rendered boundary
  pixel -> SAM signed distance (zero level at pixel edges); (ii) observed->model: SAM boundary point on that zero level
  -> point-to-line distance to the nearest model contour sample; (iii) point-to-plane depth with projective
  association. Exclusions independent of the fit where possible (image/crop border, SAM object/fixture adjacency,
  occluding edges from observed depth). Visible-surface speed gate 0.10 m/s.
* **Joint solve** (`rig_dw_solve.py`): per camera pose (left twist), alpha (log depth scale), tau; stages: rig shape
  from ext-ext (gauge camera 0) -> common Sim(3) of the rig (+ tau) from the robot -> everything; support re-selected
  per outer iteration, LM at fixed support; data-only information in tolerance units (0.35 deg, 8 mm, 1 %, 10 ms) with
  unidentified directions reported as unbounded.
* **Validation** (`rig_dw_run.py`): 80 fit / 20 held-out episodes (every 5th); held-out static correspondences
  (symmetric epipolar line distance, depth-lifted transfer both ways, 3D consistency), held-out robot evidence
  (along-ray depth, explained fraction, fixed-denominator contour distance, IoU), common-transform challenge, the
  current calibration on the same evidence; variants for stability and sensitivity; fixedness (`rig_dw_fixedness.py`).
* **Publication** (`rig_dw_publish.py`): a `calibration_rig` document (format s4d.realworld.rig_calibration/1) with
  per-camera status, T_cam_base, depth_scale (per depth stage), tau, empirical spread, held-out validation and the
  reference arrays for the fixedness test; `calibration_stage` rig mode (calibrate@2) consumes it and multiplies the
  episode's depth by the camera's `depth_scale` before its per-episode robot-depth checks.

## Not reproduced from DepthWorld
No wrist camera (their main grounding), no LoMa-G render-to-image matches or featuremetric term (silhouettes and
point-to-plane depth instead), no joint-encoder offsets (joint 1 offset is exactly a common rig yaw when the base is
hidden), no hand-eye correction; the mount stays nominal. Intrinsics are factory values (a per-camera focal multiplier
exists as a diagnostic variant only).
