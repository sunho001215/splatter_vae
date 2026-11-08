# Render-to-image robot 2D-3D rig calibration (`rig_loma`)

Status: development study (branch `rw-calib-loma-20261108`, LOMA stream of the 2026-11-08 calibration directive);
frozen a935aef, held-out batch done; release path (gate change, rig documents, records, depth scale, gripper map,
visual meshes): [realworld_calibration_release.md](realworld_calibration_release.md).
Notes / results / freeze proposal: `/home/ws/ws/realworld-ops/calib-loma/loma/` (NOTES.md, FREEZE_PROPOSAL.md,
results/). Predecessor: `rig_dw` (docs/realworld_rig_dw.md; both tasks HELD).

## Why
`rig_dw` grounded the rig in the robot base frame with SAM-3 robot silhouettes and masked FoundationStereo robot depth.
Its pose stayed tied to SAM boundary bias (mask +-2 px = a 1.4-2 % common scale change) and failed leave-one-camera-out
(LOCO 0.37-0.95 deg / 5.5-16 mm). The directive replaces SAM-based absolute-pose constraints by DepthWorld-style
(arXiv 2610.08780, App. B) render-to-image robot correspondences and forbids SAM masks in initialisation, optimisation,
correspondence filtering and solution selection (validation only).

## Method (per task, one fixed 3-camera rig, strictly per task)
1. **SAM-free frames** (`rig_loma_data.extract_episode`): per camera 16 frames per episode, robot state at the camera
   timestamp (<= 20 ms), flange speed <= 0.10 m/s, FK of every link at t and t +- 10 ms. Robot visibility is decided
   by rendering (>= 3000 rendered pixels), never by masks.
2. **SAM-free coarse initialisation** (`rig_loma_data.mask_free_init`): the existing RGB-D robot registration
   (`calibration.calibrate_camera`, evidence `rgbd`) without robot masks (moving-depth evidence), median over 8 fit
   episodes. Only a starting point.
3. **Renderer** (`rig_loma_render`): Franka/Robotiq VISUAL meshes (COLLADA, material colours, 175,604 triangles),
   exact torch triangle rasterizer (2x supersampling), non-PBR Lambert shading (camera key light + fill + ambient),
   tight crop = rendered-arm bounding box + 16 px. Render pixel -> exact surface point (ray / triangle-plane
   intersection) in its link frame -> base frame by FK at the camera timestamp, + FK surface velocity (time offset).
4. **Matcher** (`workers/loma_render_match_worker.py`): LoMa-G (github.com/davnords/LoMa @ 415caed, weights loma_G.pth)
   between render crop (template) and real crop, the official image-path preprocessing (DaD 1024 aspect-preserving,
   DeDoDe-G descriptors 784x784, mutual NN, filter threshold 0.1), 2048 keypoints.
5. **Geometric filters only**: render triangle hit, no render depth discontinuity > 2 cm within +-1 px (silhouette /
   occlusion boundaries), per-(episode, camera) pooled MAGSAC++ PnP inliers (4 px; the camera is fixed so all frames
   share one pose), per-frame cap 100 (round robin over links); Robotiq finger links dropped when the gripper state
   > 50 (URDF gripper mapping mismatch, see below).
6. **Iterated render-and-match**: correspondences are re-extracted at the updated pose (round r0 at the SAM-free init
   on 20 fit episodes -> solve -> round r2 on all episodes at that state), removing the template-attraction bias
   (synthetic: -0.07..-0.14 px at 1-8 deg render-pose error, ~0 at 0).
7. **Joint solve** (`rig_loma_solve.LomaProblem`, subclass of `rig_dw_solve.RigProblem`): robot 2D-3D reprojection
   term `(pi(K_c, T_c (X + tau_c V)) - u) / 1 px` (Cauchy c = 3 on the whitened norm) as the primary absolute-pose
   source + the unchanged rig_dw static two-view RGB-D factors (RoMa v2) with SAM-free static flags
   (`workers/static_flags_worker.py`: temporal change of grey value / sensor depth against the episode median,
   MAGSAC++ F on those static matches) + per-camera log depth scale alpha + time offset tau. Optional secondary
   match-depth term `(log z_c(X) - log d(u) - alpha_c) / 1 %` (off by default: validation). Stages as rig_dw.
8. **Depth source by name** (`--depth-stage`): FS / sensor columns of the match files, any other stage (S2M2) through
   `rig_loma_depth` sample files (template `{root}/{ep}/episodes/{ep}/derived/{stage}/{key}.h5`).
9. **Gripper-mount correction**: Fisher information of a 6-DoF mount twist (panda_link7 frame) given the rig
   (`mount_information`) and a diagnostic alternating estimate (`--variant mount`); not part of the default solve.

## Deviations from DepthWorld
* No wrist camera: DepthWorld's main grounding (wrist-external correspondences + hand-eye) is unavailable; the robot
  reprojection term is our only robot grounding (their "l" term, c_l = 50 px; ours 1 px whitening, Cauchy 3 px,
  because our matches are MAGSAC-filtered against a fixed camera and the term must carry the absolute pose).
* pyrender is not installable in the matcher container (no network, no GL stack); the renderer is an exact torch
  rasterizer with the same protocol (URDF visual meshes, current joints and extrinsics, non-PBR well-lit shading,
  tight arm crop, render pixel -> FK surface point). Lighting parameters are ours (paper gives none).
* One fixed rig per task (no per-episode extrinsic corrections), joint-encoder offsets not estimated (joint 1 offset ==
  common rig yaw with the base hidden), featuremetric term omitted (as in DepthWorld's deployed configuration).
* FoundationStereo depth by default (DepthWorld: S2M2); S2M2 selectable by name from the S2M2 stream.
* Iterative re-rendering (DepthWorld renders once with Stage-1 extrinsics).

## Robot model / conventions (loma/notes/robot_model_check.md)
* The remote asset copy contains only collision meshes: every remote robot-grounded run so far, including the frozen
  rig_dw solutions, used COLLISION meshes for all 18 links (surface.fallback_links), although rig_dw's docs say visual.
  rig_loma uses its own asset copy with the visual meshes (sha256-verified).
* Gripper mapping: code maps reg/255*0.8 rad, a real 2F-85 closes at reg ~227-230: at the mug-hanging grasp
  (reg 207-211) each pad renders 4.0-4.4 mm too far out; drawer-open <= 1.5 mm. Fingers excluded when reg > 50.
* Kinematics, units, frames, camera conventions (continuous pixel centres, depth K == colour K + 0.5), time sync:
  consistent; camera stamps are start-of-readout (rolling shutter) -> per-camera tau kept.

## Correspondence accuracy (no SAM)
* Synthetic (robot rendered with a different appearance over the real frame, known pose; 36 frames per task and
  perturbation): median error 0.74 px at zero render-pose error, 1.02-1.09 px at 1-8 deg / 10-80 mm, p90 2.2-2.6 px;
  after pooled MAGSAC + silhouette filter no match > 10 px, 3.5-5.7 % > 3 px; template-attraction bias ~0 at zero
  error, -0.07..-0.14 px (v) at 1-8 deg -> iterate render-and-match.
* Real images (fit episodes, RoMa time points, flange <= 0.1 m/s): LoMa matches of two cameras on the same surface
  point (<= 3 mm) lie 0.98 / 1.09 px (median, drawer / mug) from the epipolar lines of the ext-ext rig; triangulated vs
  FK 3.9 / 3.6 mm; RoMa ext-ext matches on robot pixels land on the same surface point within 3.3 / 3.0 mm (control
  33 / 31 mm); per-frame mean residual 0.45 / 0.38 px, lag-1 correlation 0.49 / 0.34; 450-750 matches per frame
  (DepthWorld: 277).

## Leave-one-camera-out term isolation
rig_dw code, global split (withheld camera vs the same-combination full solution, deg): ext 2D-only + outline 0.29 /
0.26 / 0.10 (drawer) but 22-25 mm (translation along the view direction unconstrained); + ext depth 0.56 / 0.58 /
0.14 (== frozen rig_dw with robot depth 0.55 / 0.58 / 0.15); 2D + outline + robot depth == 2D + outline; visual meshes
instead of collision meshes: contour 1.5-1.75 -> 1.0-1.1 px, LOCO unchanged (0.61 / 0.61 / 0.19). Mug likewise
(0.53 / 0.29 / 0.37 2D-only vs 0.85 / 0.69 / 0.52 with depth). **The ext-ext FS-depth rows drive the LOCO rotation
offset; robot depth and SAM outlines are not the cause.** rig_loma (dev): default sigma_logd 1 %: 0.24 / 0.32 / 0.02
(drawer), 0.34 / 0.49 / 0.12 (mug); 3 %: 0.07 / 0.16 / 0.04, 0.14 / 0.20 / 0.07; 5 %: 0.03 / 0.09 / 0.04,
0.06 / 0.06 / 0.02; ext 2D-only: 0.07-0.13 deg but 9-32 mm; the match-depth term changes nothing; S2M2 / native
depth behave like FS.

## Dev-split results and freeze proposal
See `/home/ws/ws/realworld-ops/calib-loma/loma/FREEZE_PROPOSAL.md` (sigma_logd 3 % + per-task gripper-mount
estimate). Dev pre-check with rig_dw's gates: G1 (incl. LOCO) passes for all six cameras, G2 / G3 / G4 pass (SAM
validation, mug fixed-o2m at 1.90-2.00 px), G6 with SAM evidence fails for drawer-open (roty+; the SAM-free 2D-3D
challenge rejects all 12 perturbations), mug-hanging G5a fails independently of the solution (as for rig_dw).
