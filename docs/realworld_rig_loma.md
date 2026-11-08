# Render-to-image robot 2D-3D rig calibration (`rig_loma`)

Status: development study (branch `rw-calib-loma-20261108`, LOMA stream of the 2026-11-08 calibration directive).
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
