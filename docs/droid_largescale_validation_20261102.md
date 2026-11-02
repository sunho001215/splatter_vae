# DROID large-scale data path: wrist QA, csw4 profile, multi-robot and scale checks (item 8, round 2, 2026-11-02)

Continues `docs/droid_largescale_validation_20261031.md`. All outputs, tools and logs are under
`/home/ws/ws/validation-20261031/item8/r2/` (full report: `item8/REPORT_20261102.md`). Small evidence copies are in
`docs/droid_largescale_validation/r2_*` and `figures/r2_*`.

**Scope.** Real data from 22 robots in 12 labs, but wrist depth for one robot only. DROID ZED recordings cannot
establish the stream configuration of our future RealSense rig or the mounting geometry of our Panda–Robotiq rig.

## 1. Data obtained (read-only inputs, derived outputs separate)

- **Downloads: 17.55 GB of the 25 GB budget.** Five PointWorld flow shards, listed and size-recorded before download
  (`r2_download_plan.json`), each verified against its LFS sha256:
  - `shard-000360`, `-000220` and `-000521`: other labs;
  - `shard-000002` and `-000003`: AUTOLab, whose depth sits in the already downloaded depth part 0.
- **Extraction.** Only the 305 selected members were stream-extracted through `scripts/droid_ls_extract_pointworld.py`:
  flows, cameras, and depth for the 232 AUTOLab episodes.
- **Why no wrist depth for other robots.** The depth package is a single zstd frame (no other frame magic in 600 MB),
  split into 252 parts and sorted by episode name. Part 0 holds 458 AUTOLab members, all robot `FR3-295341-1326595`.
  Any other robot's depth needs every preceding part (tens of parts at 4.9 GB each), which is far over budget.
- **RLDS.** 286 more shards were indexed in place through `scripts/droid_ls_build_rlds_index.py`: 345 shards in total.
  Their sha256 and stat were recorded before and after (§6).

## 2. Wrist QA: occlusion vs calibration (task 1)

Tool: `r2/tools/wrist_diag.py`.

- **Inputs.** Per episode, 12 canonical frames spread over the episode. External tracks seen by their own camera,
  exactly rebased into the DROID-3D world.
- **Scores.** Each pose hypothesis and negative control is scored by:
  - **photometric:** z-buffered colour MAE against the RLDS wrist JPEG, independent of PointWorld's wrist depth;
  - **depth, where available:**
    - the signed residual `r = z - d` on z-buffered pairs, per region (data-derived gripper mask vs scene) and with
      soft visibility;
    - the free-space-violation fraction `r < -3 cm`;
    - nearest-neighbour distance from wrist-depth points to the external cloud;
    - a per-episode camera-frame ICP refinement.

Hypotheses (rows below):

- `card`: `FLIP · T_gw · dT_gw · inv(FK(q+dq))`, then Rz(180);
- the same chain with only `dq`, with only `dT`, or with neither;
- the PointWorld recipe on `cartesian_position`;
- DROID's metadata hand-eye;
- time offsets of ±1 and ±2 raw steps, and the image's estimated capture time (−0.58 step);
- controls:
  - no Rz(180);
  - 2 cm shift;
  - 3° rotation;
  - another robot's hand-eye.

### 2.1 Results

**Robot `FR3-295341-1326595`, session 2023-10-21 (the 62 episodes of round 1):**

- **Occlusion dominates the earlier statistic.**
  - Earlier QA statistic, card: 0.112 m. With the per-episode ICP-refined pose (the best fit to the data): still
    0.070 m, with 11/62 under 3 cm.
  - The gripper mask covers 24 % of the view; 86 % of gripper-region points look occluded.
  - Sparse tracks leak through nearer surfaces: 53 % of scene points look occluded with a hard z-buffer, 24 % with soft
    visibility.
- **Calibration error of this session.** A consistent offset of the card pose: ICP 0.98 cm and 3.3°, per-axis std
  0.2-0.3 cm. The `dq`-only pose is closer: 0.73 cm and 2.0°.
- **Photometric, scene region:**

  | Pose | Colour MAE |
  |---|---|
  | card | 24.3 |
  | `dq` only | 19.3 |
  | ICP-refined | 17.5 |
  | no Rz(180) | 52.0 |

**Same robot, sessions 2023-10-27, 11-03 and 12-02 (232 new episodes):**

- The card pose is within 0.42 cm and 0.52° of the ICP optimum.
- Card is the best hypothesis: 135/232 by colour and 138/232 by nearest neighbour.
- Soft-visibility inliers: median 0.92.
- The earlier statistic passes 230/232.

So the round-1 preference for `dq`-only was specific to one session. **No chain or convention bug was found.**

**21 robots in 11 labs, 73 episodes, photometric only:**

- **Overall colour MAE, median over episodes:**

  | Pose | Colour MAE |
  |---|---|
  | card | 29.3 |
  | `dq` only | 27.0 |
  | no corrections | 27.0 |
  | DROID metadata hand-eye | 27.3 |
  | no Rz(180) | 59.3 |
  | another robot's hand-eye | 30.0 |
  | 2 cm shift | 31.3 |
  | 3° rotation | 32.3 |

- **Rz(180)** is required on every robot, including the five 672x376 cameras.
- **Timing.** The state at raw step `2i` is best: ±1 raw step gives 30.3 and 29.7; ±2 gives 31.3 and 31.3; the
  capture-time pose gives 30.0. This holds although the raw episode's wrist `estimated_capture` is 41 ms before the
  robot-state timestamp (`frame_received` +0.1 ms).
- **Pooled placeholder `unknown`.** Its `dT_gw` of 39° breaks the pose: IRIS card 79.2 vs `dq` only 42.5, RAIL 49 vs
  30 (`figures/r2_wrist_iris_unknown_pooled_dTgw.jpg`). This confirms that per-robot calibration requires an
  identified robot (commit `87b7846`).
- **Robots where card is clearly worse than `dq` only** (1-4 episodes each): RAIL `fr3-290102-1324143` (36.0 vs 26.5),
  RAIL `290102-1324143`, REAL `FR3-295341-1326783`. On no robot is card clearly better.
  - The AUTOLab result shows that this can be session-specific.
  - Without wrist depth for these robots, the evidence does **not** support a per-robot or per-variant rule, so none
    was implemented. Recorded as a user decision.
- **672x376 wrist intrinsics.** Independent x/y rescale, uniform-by-width and uniform-by-height are photometrically
  indistinguishable (< 0.5 grey levels; `r2_wrist672_K_check.json`).
- **Hand-eye frame.** `cartesian_position` = flange FK(q) on all 21 robots, Panda and FR3 (max 4e-8 m).

### 2.2 Fix: QA statistic v2 (commit `e2dc72a`)

The earlier statistic measured occlusion, not calibration. Statistic v2 (`join.WRIST_QA`; design in
`docs/droid_largescale.md` §4.4):

- frames spread over the episode;
- a gripper mask from the wrist depth;
- soft visibility;
- inlier fraction ≥ 0.30 and free-space-violation fraction ≤ 0.30.

Measured on 294 episodes of the one robot with wrist depth (`r2_wrist_qa_rule_eval_autolab294.json`):

| Pose | v2 passes | Earlier statistic passes |
|---|---|---|
| card | 294/294 | 230/294 |
| no Rz(180) | 9/294 | 0/294 |
| controller-EE recipe | 8/294 | 0/294 |
| 2 cm shift | 293/294 | 229/294 |
| 3° rotation | 292/294 | 227/294 |

- Neither statistic resolves 2 cm / 3° errors per episode.
- The thresholds come from one robot; re-validate when other robots' wrist depth is available.
- Manifest rebuilds through the CLI accept the wrist on 62/62 and 294/294, with depth stored in the RGB orientation.
- Wrist fields are now emitted (§4). They are supervision-only; csw4 does not read them.

## 3. Off-by-one PointWorld state frames (commit `3ba3d1c`)

- **AUTOLab.** The frame-mapping check rejected 13 of 294 episodes. Each had exactly one canonical frame whose
  PointWorld state equals RLDS raw step `2i − 1`, with error 0.000 m (7-23 mm against `2i`). The frame's depth and
  images are `2i` by timestamp.
- **Other labs.** In the 73-episode sample, 10 episodes had the same pattern (1-3 distinct frames, always `−1`).
- **Fix.** Such frames are invalidated for every camera, at most max(1, 5 %) of the timeline. Anything else still
  rejects the episode.
- **Result.** 294/294 accepted.

## 4. csw4 on DROID (task 2)

- **Synthetic views (commit `e807898`).** They used the Meta-World camera rig on DROID. Coverage on real batches:
  - Meta-World rig: 24.6 % covered, 8.5 % of patches ≥ 90 % covered, every row short of the 108 kept tokens;
  - the episode's own cameras: 77.2 % and 47.8 %.
  - New key `aug.synth_camera_rig`; non-Meta-World regimes must use `training_cameras`.
- **Anchor statistics (commit `175c0fe`).** A null `model.anchor_stats` silently kept the Meta-World workspace
  anchors; it is now refused outside Meta-World.
- **Profile (commit `7d77eda`).** `configs/droid/csw4_largescale.yaml`:
  - method keys equal the resolved Meta-World csw4 (machine-checked by `tests/test_droid_csw4_profile.py`);
  - all 24 differing keys are dataset-specific, with justifications;
  - the effective configuration is in `docs/droid_largescale_validation/csw4_effective_config.md`.
  - The 2026-10-31 smoke is identified as the legacy DROID-base profile; its evidence files are renamed
    `legacy_droid_base_smoke_20261031_*`.
- **Loss-function check on real windows.** This is not the trainer smoke. 20 steps, batch 2:
  - all losses and gradients finite;
  - total loss 2.732 → 2.653;
  - maximum pre-clip gradient norm 9.54;
  - 0.072 s per step, 0.51 GB GPU.
- **Trainer smoke: NOT-RUN.** `require_passed_tests()` needs a full-suite pass on these sources. The parent runs
  `r2/smoke_csw4.sh` after that.

## 5. Scale checks: 294 AUTOLab episodes, 4 sessions (task 3)

Tool: `r2/tools/check_loader_scale_r2.py`; output: `r2/checks/loader_scale_autolab294.json` (copy:
`r2_loader_scale_autolab294.json`). The data is 9,742 windows: 9,458 training and 284 validation.

- **Manifest.** 294/294 accepted. Wrist emitted for all 294 episodes. 13 frames invalidated as off-by-one.
- **Episode boundaries.**
  - No window crosses a clip or an episode.
  - 42 windows at 293 episode boundaries equal a direct RLDS read.
  - A shuffled batch of 16 spans 16 episodes.
- **Batch contract.** `validate_batch` passed on 120 random windows from 96 episodes; every float field is finite. Camera
  fields per window:
  - external: images (3,2,3,144,256), K, w2c and c2w, depth, motion score, 4,096 tracks per camera;
  - wrist: images, depth, K, w2c and c2w, valid, track weights.
  - Wrist values: valid on 100 % of frames, wrist depth valid 83.9 %, wrist-visible tracks 7.9 %.
- **Memory and handles.** 3,000 random windows over all 294 episodes, `max_open_files` 8:
  - RSS flat at 800 MB (growth 0 after the first 100);
  - open fds flat at 29 (8 flow + 8 depth + 8 RLDS + base), back to 5 after `close()`.
- **Workers.** 0 / 4 / 8 workers over 96 windows give identical sha256 digests of every tensor of every batch.
  - Per-kind pools stay ≤ 8.
  - The parent's fds are unchanged by worker loaders.
  - Worker RSS ≤ 524 MB.
- **Resume replay.** The trainer's own `make_train_loader` + `infinite` (4 workers, batch 4) was resumed at step 7. It
  reproduces batches 7-11 of an uninterrupted stream bit for bit. This checks the loader side only; the
  checkpoint/optimizer side is the trainer smoke, which is NOT-RUN.
- **Throughput.** Train split, batch 8, wrist decoding active on every episode, 384-core host under load:

  | Workers | Windows/s |
  |---|---|
  | 0 | 9.7 |
  | 4 | 39.2 |
  | 8 | 79.3 |

  Round 1, with no wrist fields emitted, measured 11.9 / 48.5 / 98.7.
- **Anchor statistics** (training split, 400 samples; `r2/anchor_stats/workspace_stats_autolab294.json`):
  - scene mean (0.765, 0.025, 0.131) m, std (0.54, 0.61, 0.29) m;
  - depth percentiles 1/50/99: 0.23 / 0.71 / 3.21 m.
  - One lab only; these are not corpus statistics.

This is one lab's sample of 294 episodes, 0.7 % of the PointWorld release. It does not establish full-dataset
robustness: corpus-scale handle churn, the depth-decode rate at 42k episodes, and other labs' depth are untested.

## 6. Input verification

Nothing was written under `/home/ws/data`. Caches, temporary files and logs went to `item8/r2/cache`.

| Check | Result |
|---|---|
| Round-1 manifest of 25 files (sha256) and 9 directories (stat), re-recorded after round 2 | **identical** |
| Full RLDS root, stat of all 2,051 entries, against the 2026-10-31 before-stat | **identical** |
| sha256 + size + mtime + inode + mode of the 286 newly read shards | **identical** |
| PointWorld cameras package part read from `/home/ws/data/pointworld_droid_sample` | sha256 equals the LFS sha256 |

The cameras package had no before-record of its own.

Files: `r2/checks/input_manifest_comparison_r2.json`, `input_rlds_stat_comparison_r2.json`,
`rlds_shards_new.sha256.{before,after}.txt` and `camera_package_after.txt`.

## 7. Status changes

| Row | Before | Now |
|---|---|---|
| Wrist pose chain and QA | PARTIAL (QA rejected 62/62) | **VERIFIED-REAL for one robot** (294 episodes, 4 sessions, depth-based). **PARTIAL across robots**: photometric only, 21 robots / 11 labs; per-robot `dT_gw` quality varies; no depth |
| 672x376 wrist intrinsics | NOT-RUN | **PARTIAL**: photometric, 5 episodes, 2 labs, `unknown` robots (wrist disabled for them anyway) |
| Multi-robot / multi-lab join | NOT-RUN | **PARTIAL**: identity, serials, state mapping and FK on 73 episodes / 21 robots / 11 labs. Full manifest (depth) only for AUTOLab |
| csw4 profile on DROID | config inherited DROID base | **profile + test done**; trainer smoke **NOT-RUN** (gate) |
