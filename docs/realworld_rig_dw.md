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

## Results (frozen commit 65ab184; held-out = 20 episodes per task; details in `/home/ws/ws/realworld-ops/calib-dw/`)

Evidence: RoMa v2 6.84 M matches per task (3.6 M static MAGSAC inliers; 274 k used per fit after filters and
stratification), 1440 robot frames per fit (6 per camera per episode). Fit time ~16 min (8 CPUs).

| held-out (rig_dw / current per-episode calibrate@1) | drawer-open | mug-hanging |
|---|---|---|
| static epipolar line distance (px) | 0.28 / 4.24 | 0.25 / 4.36 |
| depth-lifted transfer a->b, b->a (px) | 2.13, 1.48 / 7.28 | 2.44, 1.73 / 8.94 |
| silhouette contour median (px) 070988 / 073201 / 076054 | 1.56 1.46 1.75 / 3.29 1.76 3.15 | 1.77 1.94 1.98 / 2.54 2.50 3.35 |
| robot depth median, FS x exp(alpha) (mm) | 9.1 11.3 11.0 | 9.0 10.9 12.1 |
| robot depth median, raw FS (mm) / current (raw) | 9.8 19.6 28.4 / 10.6 10.8 15.2 | 9.8 16.7 29.8 / 8.7 10.7 16.8 |
| explained fraction (alpha) / raw FS | .94 .88 .93 / .95 .79 .56 | .95 .90 .93 / .96 .82 .66 |
| fixed-denominator observed->model contour (px) | 2.00 2.00 2.07 | 2.33 3.00 2.68 |
| alpha (log depth scale) | +0.54 +2.18 +3.37 % | +0.73 +1.89 +3.52 % |
| tau (ms) | 6.8 12.6 13.3 | 7.9 2.8 9.9 |

The current per-episode poses differ from the rig_dw poses by median 0.7-2.3 deg / 13-36 mm.

Stability (G1, max pairwise rotation / centre): 8 multistarts (common rig perturbation 4 deg / 40 mm) all converge
to the same solution (<= 0.0001 deg); 5 random episode halves <= 0.04 deg / 0.7 mm; 2 workspace groups <= 0.11 deg /
3.7 mm; 2 temporal blocks <= 0.07 deg / 1.5 mm (mug-hanging 073201: 0.24 deg / 5.3 mm); F-fit vs all-100 refit <= 0.01 deg / 0.13 mm. **Leave-one-camera-out**
(robot factors withheld for one camera, its pose from the static RGB-D factors relative to the robot-grounded other
two), vs the full solution: drawer-open 0.55 deg / 10.1 mm (070988), 0.58 / 10.4 (073201), 0.15 / 2.2 (076054);
mug-hanging 0.84 / 13.6, 0.67 / 13.1, 0.51 / 7.8. The withheld camera's held-out epipolar distance gets worse
(drawer 0.29 -> 0.33 px; mug 0.25 -> 0.31 px) while its depth-lifted transfer gets better (drawer 3.84 -> 2.68 px,
mug 2.44 -> 1.78 px) and its silhouettes worse (1.56 -> 1.83 px, 1.77 -> 2.32 px): the static factor's FS-depth rows pull against the 2D + robot evidence (non-scale FS depth errors
that one alpha per camera cannot absorb). G6: all 12 common perturbations (0.35 deg / 8 mm) rejected by the held-out
robot evidence (frames and 20/20 episodes).

Sensitivities (diagnostics, not gated): SAM mask +-2 px = a pure common similarity of the rig about the workspace
(scale -1.4 % / +2.0 %, residual < 1 mm; ~7 mm of camera-to-workspace distance per px of boundary offset; relative
alphas unchanged); alpha fixed at 0 (FS metric trusted): scale -1.7 %, centres 6-12 mm, all held-out terms worse;
per-camera focal multiplier: -0.9..-2.7 % (inconsistent between tasks), scale -1.3..-1.7 %; sensor depth instead of FS:
<= 0.07 deg / 1.1 mm; no tau <= 0.05 deg / 0.9 mm; 1.0 mm mesh sampling <= 0.03 deg / 0.6 mm; match-subsampling seed
<= 0.001 deg (all re-run with the frozen code, both tasks); robot-depth-only grounding drifts 3 deg / 60 mm (not
usable); silhouettes-only == full (pre-freeze diagnostics).
Held-out signed silhouette residuals are positive in both directions (+0.9..+2.0 px): the rendered visual-mesh
silhouette is larger than the SAM masks at the solution (SAM under-segmentation of ~1-2 px is consistent with the
board's +1 % scale, below).

Cross-task diagnostic (independent per-task solutions, same physical cameras): 0.27 deg / 4.7 mm (070988),
0.11 / 1.5 (073201), 0.08 / 1.1 (076054); relative rig 0.35 / 0.28 deg. (Per-task consensus of the current
per-episode solve differed 2.0 deg / 29 mm, 0.65 / 8, 1.1 / 15.)

Post-freeze board cross-validation (calibration.h5, ZVDA-CB01 30 mm, 5 stationary placements, 15 views, board BA
0.28-0.37 px, leave-one-placement-out <= 0.18 deg / 3.3 mm; never fed back): relative camera poses rig vs board
drawer-open 0.13 / 0.13 deg, 9.3 / 5.7 mm; mug-hanging 0.36 / 0.17 deg, 5.1 / 1.9 mm. Board corners triangulated with
the rigs reproject to 0.16-0.29 px (median), p90 <= 0.67 px. Metric scale: rig baselines 0.4-1.2 % longer than the
board's, triangulated pitch 1.010 / 1.012 x nominal -> the silhouette-set common scale is ~1 % too large (~10-15 mm of
camera-to-workspace distance). Depth: board-implied FS corrections 0.982 / 1.015 / 1.019 vs rig exp(alpha)
1.005-1.007 / 1.019-1.022 / 1.034-1.036 (the camera-to-camera alpha pattern is confirmed, the common part is ~1.5 %
too high). Absolute base-frame pose and tau are not testable with an unattached board. Board-recording fixedness:
fixed vs the mug-hanging reference; vs the drawer-open reference camera 076054 shows 0.54 px / 0.07 deg (13/40
consistent patches).

## Verdict (pre-registered gates, DESIGN v1.1; `rig_dw_publish.verdict`)

**drawer-open: HELD** (all three cameras). 070988, 073201: G1 leave-one-camera-out 0.63 / 0.70 deg, 11.5 / 13.0 mm
(max pairwise; vs full 0.55 / 0.58 deg, 10.1 / 10.4 mm). 076054: G1 0.367 deg max pairwise (0.23 deg vs full) and G3
fixed-denominator contour 2.07 px > 2.0. Passed: G2 (9.1 / 11.3 / 11.0 mm, explained .94/.88/.93), G4, G5 (no motion
detected; G5a mostly "not verifiable", G5b no flags), G6.
**mug-hanging: HELD** (all three). G1 loco 0.95 / 0.83 / 0.73 deg, 15.0 / 16.2 / 10.8 mm; G3 2.33 / 3.00 / 2.68 px;
G2 076054 12.11 mm > 12; G5a flags 070988 (1 episode) and 073201 (13 episodes) by the frozen registration rule
(rotations up to 0.033 deg vs 0.03, shifts <= 0.19 px; G5b finds no inconsistent episode). Passed: G4, G6.
Not gated, but material: raw-FS depth disagrees with the robot by 17-30 mm for 073201 / 076054 (|alpha| > 1 %); the
training store lifts raw FS depth (`train_store.py`), so even a validated rig would need the published `depth_scale`
consumed downstream.

## Ambiguity statement
Well determined (internally and by the board): the relative rig and all rotations to ~0.1-0.4 deg (board 0.13-0.36
deg, triangulated board corners 0.16-0.29 px), cross-task agreement 0.08-0.27 deg / 1-5 mm, the camera-to-camera
depth-scale pattern (FS of 073201 / 076054 short by ~1.5-3 % relative to 070988). Not determined to the 0.35 deg /
8 mm tolerance: (1) each camera's pose without its own robot evidence (loco 0.5-0.8 deg / 8-16 mm), driven by
non-scale FS depth errors in the static RGB-D factor vs the 2D + robot evidence; (2) the common metric scale of the
rig about the workspace (camera-to-workspace distance): set by the SAM silhouettes, ~7 mm per px of mask-boundary
offset; the board says the rigs are ~1 % (~10-15 mm) too large, consistent with SAM under-segmentation of ~1-2 px
(positive signed silhouette residuals); alpha=0 or a focal multiplier would move it by 1.3-1.7 %; (3) the absolute
base-frame pose (common rig rigid motion) is supported only by robot silhouettes (G6 rejects 0.35 deg / 8 mm
perturbations) and cannot be checked with an unattached board; joint-1 offset / rig yaw and robot dimensional scale are
exact confounds; (4) tau 3-13 ms is an effective alignment offset, not a measured latency. Missing constraints: an
independent metric/orientation anchor per camera that does not depend on FS depth or SAM boundaries (a multi-placement
target seen by all cameras, ideally attached to the robot flange, or calibrated SAM boundary bias / depth model).

## Recommendation
Do not release `calibration_validated` for either task (both held by the pre-registered gates; no episode record
should get status done). The rig_dw poses are nevertheless far more consistent than the current per-episode poses
(static epipolar 0.25-0.28 px vs 4.2-4.4 px; silhouettes 1.5-2.0 px vs 1.8-3.4 px; cross-task 0.08-0.27 deg vs
0.65-2.0 deg). If the user accepts the residual ambiguity explicitly, the least-risk interim option is to use the
rig_dw rig (rig documents `realworld-ops/calib-dw/rig/rig-dw-<task>.json`, status held, with per-camera FS and sensor
`depth_scale`; per-episode record templates `rig/calibration_validated-<task>.template.json`, all status held) for geometry consumers only together with
consuming its per-camera FS depth scale, documenting the ~1 % common-scale and 0.5-0.8 deg single-camera
uncertainties; that is a user decision, not a validated release.

## Codex
r01 design review, r02 implementation of the ext-ext factor + tests, r03 code review (10 findings fixed), r04
depth-scale analysis, r05 pre-freeze review (7 findings fixed, G5 amendment rule); r06-r08 (frozen-result and board
analysis) failed with HTTP 429 (provider rate limit, 7 attempts 14:04-16:33 on 2026-11-08) - those analyses were done
without Codex. Prompts/outputs in `realworld-ops/calib-dw/codex/`.
