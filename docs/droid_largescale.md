# Large-scale DROID training data: audit, released schemas, conventions, join design

Branch `droid-largescale-20261031`, 2026-10-31. Code: `s4d/data/droid/largescale/`; tests:
`tests/test_droid_largescale.py` (fixtures in `tests/droid_largescale_fixtures.py`); CLIs:
`scripts/droid_ls_build_rlds_index.py` and `scripts/droid_ls_build_manifest.py`; config overlay:
`configs/droid/largescale.yaml`. Status and TODOs: `docs/droid_largescale_STATUS.md`.

The design below was first written without data access (synthetic fixtures only). On 2026-10-31 it was validated on
real bytes (read-only DROID RLDS, 62 PointWorld episodes, the DROID-3D files, one raw DROID episode); the findings,
fixes and numbers are in `docs/droid_largescale_validation_20261031.md` and are tagged **VERIFIED-REAL** below.
Each claim below carries a tag:

- **VERIFIED-REAL**: measured on real data on 2026-10-31 (scope stated with the claim).

- **VERIFIED-bytes**: read from released bytes.
- **VERIFIED-code**: read in the publisher's own reader or writer code at a pinned commit.
- **VERIFIED-sample**: measured earlier on the one real PointWorld sample episode (`docs/POINTWORLD_SCHEMA.md`).
- **ASSUMED**: not verified. Each assumption has a runtime guard or a TODO.

## 1. Audit: what existed and what was missing

### 1.1 History (present only in git history, not in the current tree)

| Commit (date) | What it was | Large-scale relevant | Wrist |
|---|---|---|---|
| `ac7c590` (2026-04-12) `dataset/droid_dataloader.py` | `DROIDPreprocessor` decoded **every** RLDS episode into per-episode frame directories, joined to `KarlP/droid` `cam2cam_extrinsics.json` by matching the RLDS `file_path` | Full decode and materialization, a local duplicate of the frames | no |
| `755a22f` / `b19d93e` / `4d44e7e` (2026-08) `dataset/droid/{calibration,rlds,sampling,transforms,cache}.py`, `scripts/train_droid.py` | Official post-hoc calibration from `KarlP/droid` (`cam2base_extrinsics.json`, `cam2base_extrinsic_superset.json`, `intrinsics.json` at 1280x720), path matcher, motion crops, teacher precompute (See3D, WAFT, X-Lens) | Teacher caches, exterior views | no |
| `2393db9` (2026-10-06) `dataset/droid/{shards,records,integrity,preprocessed_manifest,dataset}.py` | Offline "Stage-0" caches in indexed TAR shards with byte-offset sidecars, `EpisodeGroupedDistributedSampler`, integrity checks (see the quality incident doc in that commit) | Random access into *derived* shards (re-encoded data plus teacher outputs) | no |
| `c4633d8`, `3cdad57`, `28c8f6c` (2026-10-07/12) current `s4d/data/droid/` | **One** PointWorld-DROID sample episode, reread from RLDS and converted into an h5 cache. Introduced the 3D track-array batch contract | Single episode, everything materialized | wrist **excluded on purpose** (`rlds.py`, `convert.py`) |

### 1.2 Current tree before this branch

- `DroidCacheDataset` reads a converted single-episode cache.
- `rlds.load_episode` reads one episode with TensorFlow and TFDS by ordinal slice. That is not random access, and the
  current `.venv` has no TensorFlow at all.

Missing pieces, all added here:

- A large-scale lazy path.
- Any DROID-3D support.
- Any wrist-camera support.
- A corpus-level join with identity checks.
- Reconciling the PointWorld and DROID-3D calibrations.
- A way to plug in the existing trainer's sampler, resume and DDP without converting data first.

## 2. Released schemas observed (with sources)

### 2.1 DROID RLDS 1.0.1 (VERIFIED-bytes)

Sources: `features.json` and `dataset_info.json` (47 KB total) fetched from
`https://storage.googleapis.com/gresearch/robotics/droid/1.0.1/`.

- **Files:** `fileFormat: tfrecord`, 2048 train shards, template `{DATASET}-{SPLIT}.{FILEFORMAT}-{SHARD_X_OF_Y}`
  (that is, `droid_101-train.tfrecord-00000-of-02048`). 95,658 episodes, 1,865,993,126,270 bytes.
- **Episode metadata:** only `episode_metadata/file_path` and `episode_metadata/recording_folderpath`, both strings.
- **Per-step features:**
  - Images: `steps/observation/{exterior_image_1_left, exterior_image_2_left, wrist_image_left}`, Image (180,320,3),
    **jpeg**.
  - Robot state: `steps/observation/cartesian_position` (6), `joint_position` (7), `gripper_position` (1), all
    float64 with `encoding: none`.
  - Actions: `steps/action` (7) and `steps/action_dict/*`.
  - Other: `is_first`, `is_last`, `is_terminal`, `language_instruction{,_2,_3}`, `reward`, `discount`.
- **Not in the RLDS release (VERIFIED):** per-step timestamps, intrinsics, extrinsics.
- **LeRobot discrepancy:** LeRobot's DROID port mentions extra metadata such as `extrinsics_*`. Those fields are
  **absent** from this `features.json`, so nothing here relies on them.
- **Wire layout (VERIFIED-REAL, shards 463/464 = 104 records, then 3,034 records of 63 shards):** each record is one
  episode serialized by TFDS as a flat `tf.train.Example` with keys `steps/<path>` flattened over steps, numbers as
  packed `float_list` (float64 stored as float32: exactly `4 * width * T` bytes), one key set (the 23 keys of
  `features.json`, no more, no less), `is_first`/`is_last` true exactly at the first/last step, JPEG 180x320.
  - Record counts equal `dataset_info.json` `splits[].shardLengths` (shard 463: 56, 464: 48), which gives global
    ordinals: shard 463 row 27 = ordinal 21512 = the PointWorld sample episode (127 steps).
  - RLDS step `k` = raw `trajectory.h5` step `k` for k < 127 (the release drops the last of 128 raw steps); state
    agrees to 1.2e-7.
  - Guard (kept): the indexer refuses any record whose numeric payload is not a packed float list of exactly
    `4 * width * T` bytes, or whose per-camera frame counts differ.

### 2.2 PointWorld-DROID (`nvidia/PointWorld-DROID` @ `dd9aaeec`)

Sources: `docs/POINTWORLD_SCHEMA.md`, the HF tree API, `_shards_manifest.json`, and NVlabs/PointWorld data branch
`3872ec6`.

**Flow HDF5** `<uuid>_flows.h5` (VERIFIED-sample and VERIFIED-code):

- Root attributes:
  - `uuid`, `scene_path` (`gs://…/droid_raw/1.0.1/<lab>/<success|failure>/<date>/<folder>/`), `hdf5_path`
    (`success/<date>/<folder>/trajectory.h5`, no lab component).
  - `canonical_timestamps`: a JSON string holding a list of int ms. This is the ext1 camera timeline `[::2]`.
  - `ext1_cam_serial`, `ext2_cam_serial`.
  - Every other DROID metadata key, including `wrist_cam_serial`, `robot_serial` and
    `{ext1,ext2,wrist}_cam_extrinsics` (6-vectors, cam→base). `convert_2d_flows_to_3d.py:492-494` copies them all;
    present in all 62 inspected files (VERIFIED-REAL).
- Clip groups `"s:e"`:
  - Half-open on the canonical timeline, 11 frames, a new clip every 5 canonical steps.
  - HDF5 iterates them lexicographically, not chronologically.
- Per external camera only, group `camera_<serial>_ext`:
  - `intrinsic`: 3x3 for 320x180, written as ZED K × 0.25 **without a half-pixel shift**.
  - `extrinsic`: 4x4 **world→camera**, world = robot base.
  - `scene_flows`: (T,N,3) float16 **world positions** (not displacements).
  - Also `scene_visibility`, `scene_depth_valid_mask`, `scene_colors`, `scene_normals`, `initial_rgb` (JPEG),
    `initial_depth` (uint16 mm).
- Per clip: `gripper_pose` (T,7) is xyz plus quaternion **qx,qy,qz,qw**, the same data as DROID
  `cartesian_position`. `gripper_positions` is the finger angle, equal to raw × 0.725. `joint_positions` is also
  stored.
- **No wrist tracks, extrinsics or intrinsics** (VERIFIED-code: `compute_2d_flows.py:100-105`,
  `convert_2d_flows_to_3d.py:660-661`, `extrinsics_io.py:117-119`; the HF README says the same).

**Depth HDF5** `<uuid>_depth.h5`:

- Groups `<serial>+ext` and `<serial>+wrist` (VERIFIED-sample), with attrs `camera_serial`, `camera_type`,
  `is_wrist_camera`, plus an attribute-only group **`metadata`** (`uuid`, `camera_count`, `frame_count`,
  `write_complete`) in every inspected file (VERIFIED-REAL, 62/62; it was missing from `docs/POINTWORLD_SCHEMA.md`).
- Each group holds `depth` (F,180,320) uint16 mm, 0 = invalid, chunks (16,180,320) gzip, and `timestamps` (F,) int64
  ms, one per raw frame. The timestamps are exactly the raw `trajectory.h5` `<serial>_frame_received` values (127/127
  frames, all three cameras); `canonical_timestamps` = ext1 depth timestamps `[::2]` exactly (VERIFIED-REAL).
- uint16 saturation **65535 mm** occurs in external depth of 2 of 62 episodes; it is treated as invalid
  (`depth_mm_to_m`).
- No intrinsics are stored.
- The wrist groups come from an **unreleased** legacy pipeline (VERIFIED-code: the released `compute_depth.py`
  writes only `+ext` at 640x360). The source and storage orientation of wrist depth are therefore unknown and are
  decided from data (§4.4).

**Camera JSON** `<uuid>_cameras.json`:

- Keys `uuid`, `scene_path`, `optimization_success`, `optimization_summary`, and per external serial
  `optimized_extrinsics` (4x4 w2c, equal to the flow `extrinsic`).
- The extrinsics start from a VGGT initialization. They are then refined against FoundationStereo depth with URDF
  render-and-compare (VERIFIED-code, `extrinsics_pipeline.py`).

**Packaging** (VERIFIED-bytes, HF tree):

| Package | Layout | Random access |
|---|---|---|
| Flows | 771 shards `flows-fs-optimized/shard-NNNNNN/package.tar.zst.part-0000` (2.67 TB). `_shards_manifest.json` (`schema: pointworld_droid_flow_shards_v1`, `shards[].members[]`) maps each of 42,935 episodes to its shard | Per shard |
| Depth | 252 parts × 4.9 GB = 1.23 TB, one `tar.zst` stream | **No index:** sequential only |
| Cameras | 19.5 MB | — |

**License:** NVIDIA License, non-commercial (see `docs/POINTWORLD_SCHEMA.md`).

### 2.3 DROID-3D (`jaibrdhn/droid_3d_extrinsics` @ `69db9ccf`)

Sources: the dataset card, plus the authors' reader code in DepthWorld @ `6823c13d`
(`depth_extras/extrinsics.py`, `franka_fk.py`, `disparity_to_depth.py`, `assets/gripper2wrist_transforms.json`).
The data files were read on 2026-10-31 (VERIFIED-REAL: 25 records spot-checked field by field, all 73,217 /
71,006 lines parsed for identity checks); corrections to the card-based description are marked below.

**`extrinsics.jsonl`** (100 MB, 71,006 episodes, CC BY 4.0), one JSON record per line:

- `episode_uuid` is the canonical DROID uuid `<LAB>+<user_id>+YYYY-MM-DD-HHh-MMm-SSs` and is the join key. User ids
  are not always hex (`t3d58310`, ...) and TRI writes lower-case `tri+...` uuids under `TRI/` (VERIFIED-REAL).
- `traj_id` is DepthWorld's own ordinal. The card warns against joining on it.
- `lab`, `robot_serial`, `scene_path` (a local mount path `/mnt/datasets/droid_raw/droid_raw/1.0.1/<lab>/...`; its
  suffix parses). `robot_serial` includes the pooled placeholders `fr3-` (TRI) and `unknown` (IRIS + RAIL; 39-degree
  `dT_gw`), which do not identify one arm (VERIFIED-REAL).
- `T0_ext1_in_world` and `T1_ext2_in_world` are 4x4 **world→camera view matrices of the LEFT camera**, despite
  their names. World = `panda_link0`.
- `dT_gw_gripper_in_world_correction`: 4x4, per robot.
- `dq_joint_correction`: 7 values, rad, per robot.
- `_meta`: `n_iters`, `final_cost`, `stop_reason`.

**`camera_intrinsics.jsonl`** (203 MB):

- `cameras.{ext1,ext2,wrist}.{left,right}.{fx,fy,cx,cy,disto[12],width,height,fov_h,fov_v}`: ZED SDK 5.1.2
  **rectified** values (VERIFIED-REAL: `disto` all zero and left == right in every spot-checked record; not raw
  factory values). 1280x720 for every external camera; the wrist is 672x376 in 1,523 records (IRIS, RAIL, REAL,
  IPRL), where the independent x/y rescale to 320x180 is ASSUMED (no such episode was checked against images).
- Per camera: `serial`, `role`, `stereo_baseline_m` (0.120 ext, 0.063 wrist), `T_right_from_left` = identity
  rotation with translation `+baseline` along x, i.e. the pose of the right camera in the left frame despite its name
  (unused here), `fps`, `zed_sdk_version`, `episode_uuid_from_config`, `camera_role_from_config`.
- Also `traj_id`, `split`, `raw_dir`. **100 uuids appear twice** (73,217 lines, 73,117 uuids): DROID raw has the same
  uuid under `success/` and `failure/`; the record is selected by `raw_dir` suffix (VERIFIED-REAL).

**Wrist pose** (card and code):

- `view(t) = FLIP · T_gw_static · dT_gw · inv(FK(q(t)+dq))`.
- `FLIP = diag(-1,-1,1,1)`.
- `FK` is modified-DH Franka to `panda_link8`. This module matches the DH table and the matrix form exactly.
- `T_gw_static` = `mean_mat`, keyed per robot serial (exact case first, then case-insensitive) in the DepthWorld asset
  (29 keys). PointWorld's `real/gripper2wrist_transforms.json` has the same matrices (28 keys, max difference 0) plus
  each entry's `wrist_cam_serial`, used as an identity check (VERIFIED-REAL).
- Stored DROID wrist images are rotated 180° relative to that frustum, so `c2w = inv(view) · Rz(180)` (VERIFIED-REAL on
  real wrist images, photometric test: see the validation note §3.4).

**Also released:** `extrinsics_missing.json` (2,111 episodes) and `pseudolabels/` (VGGT-Ω poses, CC BY-NC). They are
not used.

### 2.4 Original DROID conventions (VERIFIED-code)

Source: `droid-dataset/droid` @ `33ae6a6`.

- **Pose 6-vectors** are `[x,y,z, euler]` with scipy `"xyz"` (extrinsic), in metres and radians.
- **Hand camera:** `get_camera_extrinsics` sets `T_base_cam(t) = change_pose_frame(gripper_offset, cartesian_position(t))`,
  which equals `T_base_ee(t) · T_ee_cam`.
- **Metadata `wrist_cam_extrinsics`** is the frame-0 composed pose, so
  `T_ee_cam = inv(T_base_ee(0)) · T_base_cam(0)` (`calibration.droid_wrist_hand_eye`).
- **ZED image flip** is `FLIP_MODE.OFF` everywhere.
- **Control rate** is 15 Hz.

## 3. Conventions used here (`geometry.py`)

- **Transforms:** `T_a_b` maps frame-b points into frame a. `w2c` = `T_cam_world`, `c2w` = its exact rigid inverse.
  Camera axes are OpenCV. World = robot base.
- **Direction of released matrices:**
  - PointWorld `extrinsic` is w2c (VERIFIED-sample: depth residual 0.63 mm for w2c, 512 mm for the inverse).
  - DROID-3D `T*_in_world` is w2c (VERIFIED-code).
  - DROID 6-vectors are c2w.
  - `w2c_direction_residual` re-decides the direction from depth for any matrix (tested).
- **Intrinsics:** the contract uses continuous pixel centres.
  - PointWorld K is treated as integer-centred at 320x180 (+0.5). This keeps ext tracks, depth and K mutually
    consistent the way PointWorld lifted them.
  - The writer actually scaled ZED K by 0.25 without a half-pixel shift. The physically exact continuous centre would
    be `0.25·cx + 0.125`, a ≤0.375 px difference that stays inside PointWorld's own lifting.
  - VERIFIED-REAL: the RLDS 320x180 frames are centre-aligned antialiased downsamples of the 1280x720 SDK-rectified
    MP4 frames (best sub-pixel shift of a 4x4 box downsample: 0.00 px on 7 frames x 3 cameras; decimation, the
    integer-centre `K x 0.25` mapping, fits much worse). So `(c + 0.5)·0.25` is exact for the image, and PointWorld's
    K is 0.375 px off in x and y (`K_droid3d_minus_pw_max_px` = 0.375-0.376 on all 62 episodes).
  - DROID-3D (SDK-rectified) K is converted as `(c + 0.5)·0.25`.
  - Each manifest row records `K_droid3d_minus_pw_max_px`.
- **Time:** canonical index `i` ↔ raw/RLDS step `2i` (ext1 timeline `[::2]`). Depth frames are matched by nearest
  timestamp.
- **Wrist (stored image orientation):**
  - DROID-3D model: `w2c(t) = inv(inv(view(t)) · Rz180)`. FLIP equals Rz(180), so this reduces exactly to
    `inv(FK(q+dq) · inv(dT_gw) · inv(T_gw_static))` (tested).
  - PointWorld recipe: `w2c(t) = mean_mat · inv(T_base_ee(cartesian_position(t)))`.
  - VERIFIED-REAL: DROID `cartesian_position` equals the Franka flange FK of `joint_position` (max 3.1e-8 m,
    6.6e-6 degrees on the sample; median 0 on all 62 episodes), so the flange-vs-controller-EE frame conflict of
    `mean_mat` does not arise for these robots: both recipes are the same chain.

## 4. Join design (`join.py`, manifest `s4d_droid_pointworld_droid3d_manifest_v1`)

### 4.1 Episode identity

Numeric ordinals such as `traj_id` and RLDS shard/row are never used as keys (tested).

- **PointWorld file name = flow `uuid` = DROID-3D `episode_uuid`:** exact string match.
- **PointWorld `scene_path` suffix:** `<lab>/<outcome>/<date>/<folder>` must equal the suffixes of both RLDS
  `file_path` and `recording_folderpath`, and must be **unique** among RLDS episodes. Duplicates are rejected.
- **`hdf5_path` tail** must equal the scene suffix.
- **DROID-3D records** are selected by uuid *and* path suffix (`scene_path` for extrinsics, `raw_dir` for
  intrinsics); a record whose suffix differs is an identity mismatch.
- **`uuid_matches_folder`:** records whether lab (case-insensitive) and uuid date-time equal the folder's asctime
  stamp (`Fri_Oct__6_...`, `_` or `:` separators). The `<date>` directory is not compared (it is the collection day).
  True for all 73,217 / 71,006 DROID-3D records and all 62 episodes (VERIFIED-REAL).

### 4.2 Camera identity, by serial

- **External:** flow attrs `ext{1,2}_cam_serial`, groups `camera_<serial>_ext`, depth `<serial>+ext`, camera-JSON
  keys and DROID-3D `cameras.ext{1,2}.serial` must all match.
- **Wrist:** depth `<serial>+wrist`, flow `wrist_cam_serial` (when present) and DROID-3D `cameras.wrist.serial` must
  match.
- RLDS image roles follow the same ext1/ext2/wrist assignment.
- **Static calibration:** must be constant across clips, and the camera JSON must equal the flow extrinsic.

### 4.3 Frames (validated mapping, `validate_frames`)

1. **Robot state:** every clip's PointWorld `gripper_pose` must equal RLDS `cartesian_position[2i]` within 2 mm and
   5 mrad. Otherwise the episode is rejected.
2. **Depth timing:** per camera, the depth frame nearest the canonical timestamp must *be* raw step `2i` within 34 ms.
   That frame is then `valid`.
   - Windows need valid ext1/ext2 frames.
   - Wrist validity is a per-frame mask (`wrist_valid`).

### 4.4 Wrist QA (`wrist_qa`, per episode)

1. Take external tracks that their own camera sees, in the training world frame.
2. Project them into the reconstructed wrist camera.
3. Compare with the released wrist depth, read as stored and as rotated 180°.
4. Record the orientation whose median residual is ≤ half the other. Apply it.
5. If the best median exceeds 3 cm, disable the wrist for that episode.

Results on fixtures:

- The true model gives 0 (as stored) against centimetres (rotated).
- Rotated storage is detected and undone.
- A hand-eye with an 8 cm error is rejected.
- A flange-relative hand-eye used with the PointWorld controller-EE recipe is rejected.

This is the data check for the wrist pose model, the hand-eye convention and the image/depth orientation on real
episodes.

Real data (VERIFIED-REAL, 62 episodes of one robot `FR3-295341-1326595`, one session): the orientation decision is
clear (wrist depth is stored in the RGB orientation; as-stored residual below rotated in 53/62), but **all 62 episodes
fail the 3 cm threshold** (as-stored median 0.18 m, range 0.045-0.58 m), so no wrist fields are emitted. Two reasons,
both measured: (1) the statistic is not robust to the gripper fingers and arm occluding the wrist view (even DROID's
own per-step wrist pose gives a 3.5 cm median; the signed 75th percentile is +0.24 m); (2) the published chain with
`dT_gw` fits this robot worse than DROID's raw wrist extrinsics and than a `dq`-only variant (photometric and depth
tests, §3.4 of the validation note). Neither the statistic nor the chain was changed (one robot is not enough
evidence); this is an open decision.

### 4.5 Calibration modes

**`droid3d`:**

- External w2c comes from DROID-3D.
- PointWorld tracks are **rebased exactly**: `X' = inv(w2c_d3) · w2c_pw · X`. Provenance permits this: each external
  camera's tracks were lifted from that camera's own depth, K and *static* w2c (VERIFIED-code). The camera-frame
  points, and so pixels and depths, are unchanged (tested to 1e-12).
- Swapping only the matrices is shown to break the depth test (tested).
- K stays PointWorld's.
- Gripper points use `FK(q+dq) · inv(FK(q)) · T_base_ee(cartesian)`. This is exact if `cartesian_position` is the
  controller FK of `q`; the manifest records the flange→cartesian offset for checking (VERIFIED-REAL: 0 within 2e-8 m).
- VERIFIED-REAL (sample episode): DROID-3D matrices are the LEFT camera's w2c (robot gripper points vs PointWorld
  depth: one-sided median 1.3-1.5 cm; the right-camera, `-baseline` and inverse readings give 19-95 cm). PointWorld's
  own calibration scores 0.9-1.0 cm on this test, but that depth is PointWorld's own optimisation target, so the test
  is not independent. The exact rebase moves pixels by at most 7e-5 px and depth by 2e-7 m; the naive matrix swap
  raises the track depth residual from 0.7 mm to 12.7 mm.
- Wrist uses the DROID-3D model.

**`pointworld`:**

- Released matrices and tracks as is.
- Wrist via the PointWorld recipe, which needs PointWorld's own `real/gripper2wrist_transforms.json`.

**Provenance:** every manifest row records its sources, the `w2c_source`, the `tracks` conversion, `K_source`,
`pose_source`, the check results, and `wrist_unavailable` reasons. Source files are never modified.

## 5. Large-scale data path (`rlds_index.py`, `pointworld_source.py`, `dataset.py`)

- **RLDS in place:**
  - The TensorFlow-free indexer walks each `tf.train.Example` and stores the byte span of every step's JPEG for each
    camera, plus each packed numeric payload. It reads one record at a time and is resumable and `--range`-parallel.
  - The encoder is byte-identical to protobuf's serializer (tested with a dynamic protobuf descriptor).
  - CRC32C framing is checked; a size change is refused.
  - A window preads 9 JPEGs plus a few KB of state. Nothing is copied or decoded beyond that.
- **PointWorld:**
  - The released packages are `tar.zst` streams, and depth has no index, so per-episode members are extracted once
    with `scripts/droid_ls_extract_pointworld.py` (`zstd -dc … | …`). It accepts only regular files that match the
    released layout, refuses links and path tricks, writes atomically, and is resumable.
  - Files are opened lazily through per-kind LRU handle pools (`max_open_files` per worker and kind) with 1 MB chunk
    caches. Only the 3 needed rows and frames are read.
- **Manifest:** streamed once at dataset construction.
  - Kept in memory: byte offsets per episode plus a packed int32 window table (16 bytes per window).
  - Rows are re-read through a 32-row LRU. Memory does not scale with per-episode payloads (tested).
- **Memory:** bounded. Tested: zero fd growth and less than 20 MB RSS growth over repeated passes.
- **Determinism:**
  - Map-style dataset over a deterministic window list, sorted by episode, clip and start.
  - Windows stay inside one clip (track identity is per clip).
  - Default stride 3 canonical = 6 raw steps.
  - Track subsampling above `max_tracks` is a pure function of (seed, episode, clip, serial).
- **Sharding and resume:** handled by the trainer's own `ResumableDistributedSampler` and `infinite()`. Tested:
  disjoint rank shards, `skip`, and a resumed stream that replays the same batches across an epoch boundary. Pickling
  drops handles, so each worker or rank opens its own (tested with 2 fork workers).
- **Split:** episode-disjoint hash split (`validation_percent`).
- **Batch:**
  - The shared contract fields (`validate_batch`, 3D tracks) plus `probe_state`.
  - Supervision-only extras: `robot_joint_position`, `wrist_images`, `wrist_depth`, `wrist_K`, `wrist_w2c`/`wrist_c2w`
    (T,4,4, per frame), `wrist_valid`, `wrist_track_weight` (P,V,N).
  - The trainer forward/backward runs on a fixture batch, and stripping all `wrist_*`/`robot_*` fields gives a
    bit-identical loss. Privileged state never reaches the encoder.

## 6. Verified vs assumed vs blocked

Updated 2026-10-31 with the real-data validation (`docs/droid_largescale_validation_20261031.md`).

| Item | Status |
|---|---|
| RLDS keys, JPEG 180x320, 2048 shards, no timestamps or calibration | VERIFIED-bytes, VERIFIED-REAL |
| RLDS Example wire layout (flattened `steps/…`, float64 stored as packed float32) | VERIFIED-REAL (63 shards, 3,034 records indexed; record counts = `shardLengths`) |
| RLDS episode identity (`file_path` = `recording_folderpath` suffix, unique) | VERIFIED-REAL on all 95,658 records (metadata-only scan) |
| PointWorld flow/depth/camera schema, w2c direction, positions not displacements, i↔2i mapping | VERIFIED-REAL on 62 episodes (plus the depth `metadata` group and the 65535 code, now handled) |
| PointWorld flow attrs `wrist_cam_serial`, `robot_serial` present | VERIFIED-REAL (62/62) |
| PointWorld timestamps | VERIFIED-REAL: raw `frame_received`; canonical = ext1 `[::2]`; nearest depth index = 2i in 62/62 (ext2 max 33 ms) |
| PointWorld wrist depth source and intrinsics | BLOCKED (unreleased pipeline); orientation VERIFIED-REAL (stored like the RGB) |
| DROID-3D field names and conventions | VERIFIED-REAL (corrections: rectified K, duplicate uuids, `T_right_from_left` semantics, placeholder robots, 672x376 wrists) |
| External extrinsics: DROID-3D = left-camera w2c | VERIFIED-REAL (robot-depth test on the sample) |
| RLDS 320x180 vs 1280x720 pixel grid | VERIFIED-REAL: centre-aligned area downsample; PointWorld K 0.375 px off |
| Wrist `mean_mat` reference frame | RESOLVED for these robots: `cartesian_position` = flange FK (VERIFIED-REAL) |
| Wrist pose chain and QA | PARTIAL: orientation VERIFIED-REAL; chain/QA open (all 62 rejected), see §4.4 |
| Franka FK (Panda/FR3 same DH) | VERIFIED against DepthWorld's table, published ready-pose flange numbers and real `cartesian_position` |
| Corpus-wide join rate | Identity only: 41,599 of 42,935 PointWorld episodes join DROID-3D and RLDS; frame/serial checks measured on 62/62 only |
| Real throughput | VERIFIED-REAL: 11.9 / 48.5 / 98.7 windows/s with 0 / 4 / 8 workers (CPU, 144x256, 62 episodes) |
