# DWCAL design v0 (2026-11-07) - board-free DepthWorld-style per-task rig calibration

Directive: /root/.claude/projects/-home-ws/memory/depthworld-calibration-directive.md. Prior study (agent CAL):
/home/ws/ws/realworld-ops/calib/README.md. DepthWorld notes: refs/depthworld_notes.md.
Board data (calibration.h5, capture-20261106 results) are NOT used for anything until the algorithm and all
parameters are frozen and committed; then only for an independent cross-validation that cannot change solutions.

## 0. Setting
* 3 fixed RealSense D435I: c0 = 135122070988 (left), c1 = 135122073201 (right), c2 = 135122076054 (front-centre).
  (left / right here = the ROBOT's perspective, base +y; the robot-PC user-facing names follow the OPERATOR's view
  since 2026-11-11: left = 135122073201, right = 135122070988, see configs/robot_pc/camera_names.yaml.)
  RGB 640x480 @ 15 Hz, pinhole (zero distortion reported), fx ~609. Inter-camera timestamp offsets constant
  (+6.5 ms, -12.5 ms); robot state ~100 Hz on the same host clock. FS metric depth from the IR pair registered to the
  RGB grid (factory colour<-IR extrinsic); sensor depth likewise. SAM 3 masks per frame: robot (arm + end effector),
  object (task objects), fixture (white box), background.
* Scene: wooden counter (strong texture), walls, shelves, sink, white box; robot behind the counter at 1.2-1.7 m,
  lower links hidden by the counter (base not visible).
* Per task: 100 episodes, cameras verified fixed (image registration <= 0.1 px). Unknowns: T_c = T_cam_c_base,
  c = 0..2 (18 DoF) + nuisances.

## 1. Root causes found so far (to be tested)
R1 Per-camera, robot-only evidence has a 2-3-D near-null space (rotation about an axis through a point 0.2-0.9 m in
   front of the camera; robot depth constrains only along-ray, silhouettes were not used) [CAL].
R2 Systematic model error: the URDF uses the Franka/Robotiq COLLISION meshes (200-400 triangles per link). Visual
   meshes (COLLADA, 12-29k triangles) differ from them by median 2-4 mm, p90 8-22 mm (link1-5 p90 20-23 mm; link5
   collision->visual p90 40 mm) - the same magnitude as CAL's "per-link systematic residuals" (5-20 mm). The pose
   absorbs these differently for different workspaces -> configuration-dependent bias between tasks.
R3 Depth-scale disagreement FS vs sensor (-1.5 % .. +0.2 % per camera) couples into translation along the optical axis.
R4 Possible camera-robot time offset (exposure vs stamp latency) biases poses in the direction of typical motion.
R5 No cross-camera constraint: each camera solved independently (no ext-ext term).

## 2. Method (per task, independent; one fixed rig for all its episodes)
Split: episodes in recording order; held-out H = every 5th (k % 5 == 2, 20 episodes); fit F = other 80. Replicates
on random halves of F (seeds), workspace groups (2-means of mean flange position), frame subsets. Final refit on all
100 (allowed) only after the held-out evaluation; the held-out numbers reported are from the F-fit.

### Stage A - dense static correspondences (RoMa v2, precise setting, bidirectional)
* Per fit episode, K_e = 4 time points; per pair (0,1), (0,2), (1,2) the nearest-time frames (|dt| <= 20 ms).
* model.sample(preds, 5000) (balanced). Filters: overlap >= 0.3; both ends outside the dilated (9 px) robot U object
  SAM masks of their own frame; FS depth valid at both ends and locally smooth (5x5 range <= 2 % of depth);
  bidirectional cycle consistency <= 1.5 px; MAGSAC++ fundamental matrix inlier (threshold 1.0 px,
  confidence 0.99999, 10k iters) per frame pair. Store uv_a, uv_b, d_a, d_b (FS and sensor), RoMa precision (2x2),
  overlap, episode, time.
* Pairwise init: MAGSAC essential on all pooled inliers of the pair -> R_ab, t_ab direction; metric scale from
  depth-lifted 3D-3D (median ratio); loop check over the 3 pairs.

### Stage B - rig-internal joint refinement (ext-ext only; gauge: c0 fixed, global scale free -> fixed by s_0 = 1)
For each match (a,b): X_a = s_a d_a K_a^-1 [u_a,1]; r_ab = pi(K_b, T_b T_a^-1 X_a) - u_b, symmetric r_ba.
Whitening: Sigma = sigma_m^2 I + sigma_d(d)^2 g g^T, g = d pi / d d_a (2-vector, along the epipolar line);
sigma_m = 0.5 px, sigma_d(d) = 0.01 d (FS relative depth noise; checked on data). Cauchy on the whitened norm (c = 3).
Unknowns: T_1 T_0^-1, T_2 T_0^-1 (12) + s_1, s_2 (log-scale, prior N(0, 0.03^2)).

### Stage C - robot-grounded registration of the whole rig (all 3 cameras jointly)
Unknowns: T_c (18) + s_c (3), initialised from Stage B composed with a global SE(3); ext-ext term kept.
Robot model: URDF FK at the camera timestamp (interpolated joints), Franka/Robotiq VISUAL meshes (COLLADA), surface
samples with normals. Frames: per fit episode up to 12 per camera, robot in view >= 25 %, flange speed <= 0.10 m/s
(limits time-offset bias to <= 2 mm at 20 ms).
* Depth term (along-ray): s_c d_obs(pi(X)) - z(X) at visible model points inside the SAM robot mask (z-buffer
  self-occlusion, scene-occlusion gate 3 cm, re-gated per outer iteration); Cauchy c = 10 mm.
* Silhouette term (2D, depth-free): model contour points (visible surface points on the boundary of the splatted
  model silhouette) -> signed distance (px) to the SAM robot-mask boundary; dropped where the observed depth just
  inside the contour is > 3 cm in front of the model (scene occluder), within 6 px of an object mask or the image
  border. Cauchy c = 3 px.
* Weights: per-residual whitening by its scale; each term normalised by its number of frames so that no term
  dominates by count (lambda_ee = lambda_depth = lambda_sil = 1 after normalisation).
* Global init: (i) rig aligned to the median of the current per-episode calibrate@1 poses (fit episodes only);
  (ii) independent robot-only init (existing rgbd_initialisation on camera c0) as a check. Multistart: 13 + 12 random
  perturbations (+-3 deg, +-3 cm) of the global SE(3).

### Stage D - robot-model corrections (only if identifiable)
Candidates: gripper-mount SE(3) (6), joint-encoder offsets (7), per-link translation offsets (CAL found some
stable). Identifiability: Fisher information eigen-analysis of the correction block given the rig (relative std <
prior), stability across replicates, held-out improvement on H. Otherwise kept nominal and reported.

## 3. Validation (task-internal; nothing here is used to select parameters unless stated)
V1 held-out ext-ext: perpendicular epipolar distance (px) + depth-lifted transfer error (px) on H matches.
V2 held-out 3D consistency of static points lifted from both cameras (mm).
V3 held-out robot depth median (mm) per camera, explained fraction (2 cm tol).
V4 held-out silhouette: IoU rendered(visual mesh) vs SAM mask; contour distance (px).
V5 stability: near-best multistart spread, F-half replicates (5 seeds), workspace groups, frame subsets, mask
   erode/dilate 2 px, depth source FS vs sensor, RoMa sampling seed.
V6 fixedness: every episode of the task registered to the task reference (calibration_rig.episode_motion).
V7 uncertainty: robust-weighted GN covariance, episode-cluster sandwich; per-block conditioning; empirical spread.
V8 comparison to the current calibration (calibrate@1 per episode) on the same held-out metrics.

## 4. Acceptance (per task, per camera) - fixed BEFORE results
CAL criteria unchanged: empirical spread (max pairwise over near-best multistart, F-half replicates, workspace groups)
<= 0.35 deg and <= 8 mm; held-out robot depth median <= 12 mm; explained >= 0.70; fixedness shift <= 0.25 px,
rot <= 0.03 deg, scale <= 1000 ppm. Added: held-out ext-ext perpendicular epipolar median <= 1.0 px and depth-lifted
transfer median <= 3.75 px (= 1 training pixel). Covariance reported for information only. A task's rig is
`validated` only if all three cameras pass; otherwise `held` with the failing criterion named.

## 5. Board cross-validation (after freeze commit only)
Fixedness of calibration.h5 frames vs each task reference first (cameras may have moved since 11-05). Then relative
camera poses from the board (ZVDA-CB01 detector, multi-camera BA) vs each task rig's relative poses; board corner
reprojection/triangulation consistency with the rig; absolute base-frame comparison only if the capture contains a
robot-attached reference. Results reported, never fed back.

---------------------------------------------------------------------------------------------------------------------
# DESIGN v1 (2026-11-07 ~13:20 KST) - changes after Codex rounds 1-3 and the first drawer-open run (r1)

Implemented (branch rw-calib-dw-20261107): s4d/realworld/{robot_visual, rig_dw_geom, rig_dw_extext, rig_dw_robot,
rig_dw_frames, rig_dw_matches, rig_dw_solve, rig_dw_run, rig_dw_data, rig_dw_fixedness}.py, workers/romav2_match_worker.py.
* Ext-ext: two-view RGB-D factor per match with an eliminated latent point (Codex r02), Cauchy c = 3 on the factor.
  sigma_px 0.5, sigma_logd 0.01. Matches: RoMa v2 precise, 6 times x 3 pairs x 5000 per episode; static (>9 px from
  robot/object), MAGSAC inlier, cycle <= 1.5 px, overlap >= 0.3; depth end used only if FS valid, 5x5 range <= 2 %
  and |FS - sensor| <= 5 %; <= 200 per (episode, time, pair), 8x6-grid stratified.
* Robot: visual meshes (COLLADA) sampled at 1.5 mm; point z-buffer render on the robot crop; m2o = extreme sample of
  every rendered boundary pixel -> observed SAM SDF (zero level at pixel edges); o2m = observed boundary point on the
  SDF zero level -> point-to-line to the nearest model contour sample; point-to-plane depth (FS) with projective
  association, frozen whitening; sigma_sil 1 px, sigma_depth 1 % + 2 mm; Cauchy c = 3 per residual; frames <= 6 per
  camera per fit episode (flange <= 0.10 m/s), visible-surface speed gate 0.10 m/s; support re-selected per outer
  iteration (<= 6), inner LM <= 8.
* Parameters per camera: pose, alpha (log depth scale, prior N(0, 0.05^2)), tau (s, prior N(0, 0.05^2)). Stages:
  shape (ext-ext only, gauge camera 0) -> common (exact Sim(3) of the rig + tau) -> full.
* Init: robust median of the current per-episode calibrate@1 poses of the FIT episodes (initialisation only).

## Acceptance (per task, per camera) - PRE-REGISTERED here before the stability batch
Gated (all must hold for `validated`):
G1 empirical spread <= 0.35 deg and <= 8 mm (max pairwise camera-centre distance / rotation angle) over: the
   near-best multistart solutions (starts 0..8, common-pose perturbation 4 deg / 40 mm about the workspace pivot;
   near-best = final fit cost within 0.1 % of the best), 5 random F-halves, 2 workspace groups, 2 temporal blocks,
   3 leave-one-camera-out robot groundings (relative to the full solution).
G2 held-out robot (20 episodes, 12 frames per camera, FS depth with the estimated depth scale, geometry at t + tau):
   along-ray depth median <= 12 mm, explained fraction >= 0.70.
G3 held-out silhouettes: support contour median <= 2.0 px and fixed-denominator observed->model median <= 2.0 px.
G4 held-out static correspondences: symmetric epipolar line distance median <= 1.0 px; depth-lifted transfer
   (a->b and b->a) median <= 3.75 px (one training pixel).
G5 fixedness: no episode/camera with detected motion (calibration_rig.episode_motion thresholds); segments that are not
   registrable are reported as "not verifiable".
G6 common-transform challenge: on held-out robot evidence (support fixed at the solution), each of the 12 common rig
   perturbations (+-0.35 deg about the 3 base axes through the workspace pivot, +-8 mm along the 3 axes) must raise
   the robust cost on > 50 % of held-out frames with a one-sided sign test p < 0.01 (otherwise the evidence cannot
   certify that tolerance).
Reported, not gated: raw-FS (alpha = 0) held-out depth/explained (flag if |alpha| > 0.01: downstream lifting must
apply the depth-scale correction exp(alpha) or be documented as biased), mask erode/dilate 2 px, sensor depth,
sample spacing 1.0 mm, no-tau, match-subsampling seed, local/sandwich covariance, cross-task diagnostic.

---------------------------------------------------------------------------------------------------------------------
# DESIGN v1.1 = FROZEN CONFIGURATION (2026-11-07 ~13:55 KST; amendments after Codex r04/r05, recorded BEFORE the
# stability batch and BEFORE any board data are opened)

Estimator unchanged from v1 (alpha free with prior 0.05, tau free with prior 50 ms, factory intrinsics; focal multiplier
only as a diagnostic variant; 6 fit frames / camera / episode, 12 held-out; 1.5 mm visual-mesh samples; RoMa matches
v1; protected evaluation sampling (FS depth, match seed 1)). Corrections (no estimator re-tuning):
* observed contour points now lie exactly on the zero level of the bilinear observed SDF (Newton along the normal);
  the old centre + (-sdf) n step missed it by up to 0.134 px on diagonals (applies to the o2m factor and the
  fixed-denominator G3 evaluator).
* G6: a perturbation counts as rejected only if BOTH the frame sign test and the episode-cluster sign test give
  p < 0.01 (20 held-out episodes: >= 16 worse).
* G1 multistart near-best: final states re-scored on the COMMON support of the full solution (cost_on_ref_support),
  within 0.1 % of the best. G1 needs the complete set (starts 1-8, halves 0-4, groups 0-1, blocks 0-1, loco x3);
  incomplete evidence never passes.
* G5 AMENDED (the pre-registered segment test fails its own sanity check: the reference episode of camera 076054 is
  flagged against itself; consensus fractions 0.11-0.35): G5a = whole-episode-median registration (1 segment, same
  thresholds; registrations with < 6 or < 60 % consistent patches are "not verifiable"; the reference
  self-registration must pass), G5b = per-episode static-match consistency under the shared rig: for every episode
  and camera pair, median symmetric epipolar line distance <= 1.0 px and (median - pair baseline over all episodes)
  + 2.58 SE <= 0.25 px; a camera is "moved" in an episode if both of its pairs violate (single-pair violations are
  reported). Rules fixed here before inspecting G5b values.
* depth:sensor variant evaluated on the protected FS evidence with alpha_FS = alpha_sensor - log(FS/sensor).
* G2 is evaluated with the estimated depth scale (as in v1); this choice was made after r1 (Codex r04): it is
  reported as "amended after r1, frozen before the stability batch", raw-FS results are reported next to it.
Diagnostics (not gated): mask erode/dilate 2 px, sensor depth, spacing 1.0 mm, no tau, match seed 1, focal, alpha0.
Pre-freeze diagnostics already observed (code before the r05 fixes): silhouettes-only == full (0.02 deg / 0.4 mm);
robot-depth-only grounding drifts 3 deg / 60 mm (not a usable grounding); mask +-2 px moves centres 13-19 mm and the
common alpha by -1.4 / +1.8 % (relative alphas stable) -> the common metric scale / along-axis distances are set by
the SAM silhouettes and are the main unresolved quantity (to be cross-checked, not tuned, by the board).
