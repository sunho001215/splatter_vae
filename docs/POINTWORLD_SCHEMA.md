# PointWorld-DROID: inspected real sample

Inspected on 2026-10-07. This is a real public-release sample, not a synthetic fixture. No foundation model or upstream training/generation code was run.

## Provenance, license, and minimal download

- Dataset: [nvidia/PointWorld-DROID](https://huggingface.co/datasets/nvidia/PointWorld-DROID).
- Pinned release commit: `dd9aaeec94bb14e27ab6b16b6e4aa0dbcf3ef56f`.
- Public, ungated repository, confirmed through the Hugging Face dataset API.
- License: **NVIDIA License**, not Apache-2.0. The dataset card says research and development only. LICENSE.pdf section 3.3 restricts the Work and derivative works to **non-commercial use**, with an exception for NVIDIA Corporation and affiliates. Section 3.1 requires retaining notices and including the complete license when redistributing the Work.
- License artifact: `/home/ws/data/pointworld_droid_sample/metadata_download/LICENSE.pdf`, SHA-256 `4ff203c3f7997c7fed287a463d733f794934a79cfabb2936008fca0bcc8ad3d6`.
- The upstream PointWorld **code** carries Apache-2.0 notices; that does not change the dataset license.
- Free disk before downloading: approximately 3.6 TB, from `df -h /home/ws/data`.

Selected episode: `AUTOLab+0d4edc83+2023-10-21-19h-07m-04s`.

| Saved artifact, relative to `/home/ws/data/pointworld_droid_sample` | Bytes |
|---|---:|
| `flow_episode/AUTOLab+0d4edc83+2023-10-21-19h-07m-04s_flows.h5` | 52,381,334 |
| `depth_episode/AUTOLab+0d4edc83+2023-10-21-19h-07m-04s_depth.h5` | 12,770,346 |
| `camera_episode/AUTOLab+0d4edc83+2023-10-21-19h-07m-04s_cameras.json` | 2,127 |
| `camera_archive_download/package.tar.zst.part-0000` | 19,472,217 |
| `metadata_download/flows_shards_manifest.json` | 3,929,206 |

Only one episode was saved. The flow and depth packages were streamed through `zstd -dc` and Python `tarfile` in sequential-read mode. Only the exact selected regular-file basename was copied into a fresh destination; no archive extraction API or downloaded shell script was executed. Streams were closed as soon as the selected member was complete. This avoids downloading a whole flow shard of 3,530,788,523 bytes or a depth part of 4,900,000,000 bytes. The depth scan passed approximately 188.7 MB of uncompressed members before the selected member; skipped episodes were not saved. The compact camera archive was downloaded whole and its SHA-256 matches the Hugging Face LFS object.

- Flow file SHA-256: `a07623861b8443841509af8189dd30b291669236c721fae44dfc4c02f3a385be`.
- Depth file SHA-256: `c1c7e10ce6c546f3d5672ef423ffccb4837a910457824ecd1a0522a799f62aca`.
- Exact URLs, selected tar member names, sizes, and hashes are in each episode directory's `download_record.json`.
- Numeric inspection results are in `/home/ws/data/pointworld_droid_sample/inspection_report.json`.
- All Python that read downloaded files ran with `-I`. The initial schema research used no GPU imports. Adapter entrypoints subsequently enforce the allowed-UUID guard before GPU-aware imports; TensorFlow data loading remains CPU-only and no training was run by the adapter worker.

## Actual file schema

The repository exposes packaged archives, not independently downloadable episode files. The flow package now has independent `shard-000000` directories and a `_shards_manifest.json`; the flat unsharded flow hint is not the actual public download layout. The canonical paths inside the tar members remain `droid/flows-fs-optimized/<episode>_flows.h5`, `droid/depth_320x180/<episode>_depth.h5`, and `droid/cameras/<episode>_cameras.json`.

### Flow HDF5

Root attributes include `uuid`, `scene_path`, `trajectory_length=128`, `canonical_timestamps` as a JSON-list string of 64 integer millisecond timestamps, `domain="droid"`, `lab="AUTOLab"`, `user_id="0d4edc83"`, `date`, `timestamp`, `success`, original camera serials/extrinsics, recording paths, and motion-filter thresholds. Three additional root datasets are `ee_pos_motion_magnitudes`, `ee_rot_motion_magnitudes`, and `has_gripper_movement`.

Actual clip keys, sorted numerically: `0:11`, `5:16`, `15:26`, `20:31`, `25:36`, `30:41`, `50:61`. Each clip uses a half-open interval on the canonical timeline and has `demo_length=11`. HDF5 lexicographic iteration is not chronological; parse the integers explicitly.

Each clip has only these external-camera groups:

- `camera_22008760_ext`, corresponding to metadata `ext1_cam_serial`.
- `camera_24400334_ext`, corresponding to metadata `ext2_cam_serial`.

For `0:11`, their track counts are 28,010 and 36,863 respectively. Counts vary by clip and camera; track identities are meaningful within one clip, not across clips.

| Camera dataset | Actual shape | dtype and semantics |
|---|---|---|
| `initial_rgb` | `(1,)` | HDF5 variable-length object array of JPEG bytes; decoded RGB is `(180,320,3)` uint8 |
| `initial_depth` | `(180,320)` | uint16, millimeters, zero invalid; attrs `format="uint16_mm"`, `scale="millimeters"` |
| `intrinsic` | `(3,3)` | float32, pixel units for 320 by 180 |
| `extrinsic` | `(4,4)` | float32, **world-to-camera**, OpenCV projection convention |
| `scene_flows` | `(11,N,3)` | float16, **world-space tracked point positions**, not displacement vectors |
| `scene_colors` | `(11,N,3)` | uint8 RGB |
| `scene_normals` | `(11,N,3)` | int8, decode by dividing by 127 |
| `scene_visibility` | `(11,N)` | bool |
| `scene_depth_valid_mask` | `(11,N)` | bool |

Displacement is `scene_flows[target] - scene_flows[source]`. Do not subtract tracks from different clips or assume all world points are visible from another camera.

Actual intrinsics:

```text
22008760: fx=fy=131.086624, cx=159.944641, cy=92.569595
24400334: fx=fy=132.954758, cx=159.037979, cy=86.002190
```

Per-clip robot datasets:

| Dataset | Shape | dtype / convention |
|---|---|---|
| `gripper_open` | `(11,1)` | bool |
| `gripper_pose` | `(11,7)` | float32, xyz followed by quaternion **qx,qy,qz,qw** |
| `gripper_positions` | `(11,)` | float32, **finger joint angle**, not raw normalized RLDS gripper position |
| `joint_positions` | `(11,7)` | float32 |
| `joint_velocities` | `(11,7)` | float32 |
| `joint_torques` | `(11,7)` | float32 |

The upstream annotation reader multiplies raw normalized gripper position by **0.725** before writing `gripper_positions`. Raw RLDS sign is 0=open, 1=closed. Do not treat the stored PointWorld finger angle as normalized position. Upstream Euler conversion uses `scipy.spatial.transform.Rotation.from_euler("xyz", angles)`, with radians and an xyzw quaternion result. These conventions were read from generation code, not guessed from values.

### Depth HDF5

No root attributes. Camera group names use a different syntax from flow groups:

- `22008760+ext`
- `24400334+ext`
- `18026681+wrist`

Each group contains `depth` with shape `(127,180,320)` uint16 millimeters and `timestamps` with shape `(127,)` int64 milliseconds. Dataset attributes include `units` and `write_complete=true`. Exclude `18026681+wrist` from the static-camera training adapter.

The full depth frame and the corresponding clip `initial_depth` are not byte-identical. Their common-valid median depth difference is zero, but hundreds of pixels differ because the annotation generator sanitizes initial depth. Use the full depth package for all three times, treating zero as invalid.

### Camera JSON

Top-level keys: `uuid`, `scene_path`, `optimization_success=true`, `optimization_summary`, `22008760`, and `24400334`. Serial entries contain `optimized_extrinsics`, identical to the flow HDF5's `extrinsic`. The JSON does **not** contain intrinsics; read them from the flow camera group.

`scene_path` is:

```text
gs://gresearch/robotics/droid_raw/1.0.1/AUTOLab/success/2023-10-21/Sat_Oct_21_19:07:04_2023/
```

Root flow attributes also include:

```text
hdf5_path=success/2023-10-21/Sat_Oct_21_19:07:04_2023/trajectory.h5
ext1_mp4_path=success/2023-10-21/Sat_Oct_21_19:07:04_2023/recordings/MP4/22008760.mp4
ext2_mp4_path=success/2023-10-21/Sat_Oct_21_19:07:04_2023/recordings/MP4/24400334.mp4
```

These provide exact/suffix keys for RLDS `file_path` and `recording_folderpath` matching. The selected episode was subsequently independently read from the original RLDS release at canonical train ordinal `21512`. Both returned paths match the PointWorld scene suffix. The old Stage-0 index contributed only the ordinal/path hint, not any old RGB, depth, flow, calibration, or teacher outputs. Matching succeeded for the one selected episode; this is not a corpus-wide match rate.

## Timeline, frame evidence, and measured checks

The official generation code takes the first external camera's timestamps and applies `[::time_skip_ratio]`, whose default is 2. It then nearest-timestamp aligns all other cameras and raw robot proprioception to this canonical timeline. `skip_every=5` controls clip starts, not raw-frame subsampling.

For this real sample:

| Timeline measurement | Value |
|---|---:|
| Raw trajectory length attribute | 128 |
| Stored camera depth frames | 127 |
| Canonical timeline length | 64 |
| Canonical timestep deltas | 133, 134, 150, or 151 ms |
| ext1 nearest-depth indices | exactly 0,2,4,...,126 |
| ext1 timestamp residual, median / max | 0 / 0 ms |
| ext2 nearest-depth indices | exactly 0,2,4,...,126 |
| ext2 timestamp residual, median / max | 1 / 17 ms |

Within one clip, canonical indices `i,i+3,i+6` correspond to independently verified raw RLDS indices `2i,2i+6,2i+12`. All 77 stored clip robot-state observations agree with those raw indices: xyz maximum error is zero, extrinsic-XYZ Euler to xyzw quaternion geodesic maximum error is `1.11337e-7` radians, and normalized raw gripper versus PointWorld joint angle divided by `0.725` differs by at most `1.49012e-8`. Every clip-start JPEG also matches the expected even raw index best among nearby raw frames for all 14 camera/clip checks. The source timeline is not exactly periodic; per-camera depth still uses nearest measured timestamps.

Stored extrinsics are unambiguously **world-to-camera**. Projecting world tracks with the stored matrix gives the following initial-frame residuals; using its inverse is wrong:

| Camera | Stored matrix median depth residual | Inverse matrix median depth residual |
|---|---:|---:|
| 22008760 | 0.631 mm | 511.600 mm |
| 24400334 | 0.823 mm | 175.317 mm |

Across all sample clips, times, and both cameras, visible depth-valid tracks have median absolute depth residual **0.757 mm** against nearest-timestamp full depth. The sample's bidirectional cross-camera median relative depth error is **1.535%**, over 2,225,729 projected track observations. Cross-camera selection required positive target depth, in-image projection, source visibility/depth validity, and projected z no greater than target depth plus 2 cm. This target-depth occlusion selection is not an independent calibration test and must be stated when reporting the metric.

Generation code explicitly calls external-camera matrices world-to-camera and inverts DROID metadata camera-to-world poses. It generates robot geometry from raw DROID joint/cartesian state in that same frame. This supports robot-base coordinates. The independently matched raw Cartesian pose and normalized gripper now generate 32 approximate palm/finger points. The upstream fixed mount yaw maps native Robotiq opening x to local end-effector y; local +z is the approach axis. The opening stroke is 85 mm, with zero raw gripper position open and one closed. No PointWorld gripper pose is substituted for raw state.

### Completed adapter and sample validation

The active corrected cache is `/home/ws/data/droid_pointworld_cache_sample`. It contains `scratch.h5`, `dinov2.h5`, the shared manifest, independently read raw external RGB/state, the complete dataset `LICENSE.pdf`, alignment reports, loader measurements, and viewed overlays. Wrist RGB/depth is excluded from the converted observations and loader. An earlier research NPZ contains a wrist array, but the converter only selects the two external cameras and required raw state fields.

The seven clips produce 35 within-clip windows. Training uses 30 windows from the earlier clips. Validation uses five windows in `50:61`, temporally disjoint from training but from the same episode. This split is only sample-pipeline validation, not generalization evidence.

| Measured check | Result | Threshold / scope |
|---|---:|---|
| Selected episode path/state match | 1 / 1 | Sample only, not corpus rate |
| Clip-start RGB best nearby raw-frame match | 14 / 14 | Both cameras across all clips |
| Visible valid scene-track depth median | 0.757 mm | At most 10 mm; no residual-threshold selection |
| Track-based cross-camera median relative depth error | 1.535% | At most 3%; target-depth occlusion selection disclosed |
| Track-based cross-camera median without occlusion selection | 2.643% | Positive target depth and in-image projection |
| Constant local +z gripper offset | -19 mm | One fitted parameter, frozen for conversion |
| Training-fit unoccluded gripper residual median | 8.699 mm | 703 point/view observations |
| Disjoint validation-clip gripper residual median | 12.207 mm | At most 15 mm; 149 point/view observations |
| All-timeline unoccluded gripper residual median | 9.494 mm | 1,831 point/view observations |
| CPU focused tests | 16 passed | No failed tests |
| Contract validation | Both sizes, all 35 windows | Zero-worker and two-worker loaders |
| Scratch train throughput, two workers | 42.37 samples/s | Batch size two, CPU, one PyTorch thread |
| DINO train throughput, two workers | 43.66 samples/s | Batch size two, CPU, one PyTorch thread |
| Scratch train throughput, zero workers | 25.21 samples/s | Includes loader and contract validation |
| DINO train throughput, zero workers | 26.40 samples/s | Includes loader and contract validation |

Dense source-depth reprojection was subsequently implemented and executed separately from the track-based check. It lifts every valid native integer-centred depth pixel, transforms through robot-base world coordinates, and nearest-z splats into the opposite camera at all 64 times in both directions. Selected median relative error is **1.5513%** over **1,299,013** target pixels. The median without occlusion selection is **2.2385%** over **1,781,940** overlapping target pixels. Both pass the unchanged 3% threshold. The one-sided visibility test excludes points behind target depth plus 20 mm, not large in-front residuals. Teacher depth is not an independent visibility oracle.

Repository-local `docs/droid_dense_validation/` holds dense denominators, input hashes, actual depth-warp panels at canonical times 0 and 50, fused cached GT cloud and motion-score maps. The offline W&B run uses project `splatter4d-droid`; no hosted run URL or cloud sync exists. A separate `--no-wandb` execution reproduced the same numeric checks. Native sample and active cache hashes were unchanged before/after. Original earlier `docs/droid_validation` measurements are track-based and do not independently establish dense reprojection.

The gripper fit uses only training canonical indices. Candidate offsets span -30 to +60 mm in 1 mm increments. Every candidate's residual and coverage is preserved in the manifest. The fit uses positive measured depth, in-image projection, and projected point z no greater than measured depth plus 20 mm. In-front residuals are not clipped or discarded by the 15 mm acceptance threshold. Approximate geometry and visibility-filtered coverage remain limitations, especially when the fingers are occluded.

Native PointWorld intrinsics use integer pixel centers. Shared model geometry uses continuous centers at column/row plus `0.5`. Conversion adds `0.5` to the native principal point before independently scaling the x and y rows for 256x144 and 252x140. RGB uses bilinear resizing. Depth uses nearest-neighbor resizing and preserves zero invalid values. Sparse targets use source visibility plus depth-valid masks, a native and resized 20 mm source-surface test, nearest continuous pixel center, deterministic nearest-camera-z collision selection, and unit weights. Point displacement pairs are ordered 01, 12, 02 on the corresponding source grid. Each time's motion score uses the maximum norm over all three tracked-point displacement pairs divided by 30 mm and clipped to [0,1].

**Diagnosed correction:** the first conversion retained native pixel centers because an initial bare `python` edit failed while the shell continued. Focused tests detected the half-pixel mismatch and a middle-time score that omitted pair 02. Both were fixed with the explicit trusted environment interpreter. All real windows were regenerated, hashes were checked, both loader sizes passed contract validation, and both corrected cache-coordinate overlays were rendered and viewed. The previous arrays are preserved under `coordinate_v1_do_not_train/`; the active files are at the cache root. `coordinate_correction.json` records this discrepancy. No adapter-worker training used the incorrect cache.

Four saved overlays were viewed: native train and validation timestamp/RGB/depth grids, plus scratch and DINO grids using actual cached RGB, depth, continuous K, sparse source-grid displacement, and matched gripper points. Gripper marks lie on the observed robot, source-grid motion follows tracked surfaces, and text/legends are readable. Visibility-filtered sparse markers do not assert complete robot coverage.

Reproducible entrypoints, all executed from the trusted new repository with the explicit `.venv/bin/python -I` interpreter:

- `scripts/download_droid_sample.py --root <new-empty-approved-sample-directory>` pins the release, checks free disk and hashes, streams only the selected flow/depth member, and refuses non-regular selected members. The archive-copy safety test was executed. The full download script was not rerun because the same real selected files were already safely downloaded and verified.
- `scripts/convert_droid_sample.py --sample-root /home/ws/data/pointworld_droid_sample --output <existing-empty-approved-cache-directory> --rlds-root /home/ws/data/droid/1.0.1` independently rereads the raw episode, verifies both metadata paths and every stored clip state, fits the one offset, and writes both resolutions. The original release remains read-only.
- `scripts/inspect_droid_sample.py --sample-root /home/ws/data/pointworld_droid_sample --cache-root /home/ws/data/droid_pointworld_cache_sample` recomputes native alignment, compares real initial JPEGs to raw RGB, and saves native plus cached-coordinate overlays.
- `scripts/validate_droid_loader.py --root /home/ws/data/droid_pointworld_cache_sample --workers 0` or `--workers 2` validates every batch and measures throughput at both resolutions.

GPU-aware entrypoints require `CUDA_VISIBLE_DEVICES=GPU-76871c16-ff1d-f7b1-cdf5-fabbaf9df8ce` or the other allowed UUID and invoke the guard before GPU-aware imports. The adapter tests use CPU arrays/tensors. Source reads run under isolated Python. No downloaded scripts, upstream teacher models, source builds, installs, git operations, full-corpus downloads, or DROID training were performed by this adapter worker.

Machine-readable evidence: `manifest.json`, `alignment_report.json`, `loader_validation_workers0.json`, `loader_validation_workers2.json`, `adapter_verification.json`, and `coordinate_correction.json` in the active cache root. Overfit, encoder-weight loading, renderer validation, and single/two-GPU training gates are coordinator-owned and are not claimed here.

### Sources

- [Pinned dataset card](https://huggingface.co/datasets/nvidia/PointWorld-DROID/blob/dd9aaeec94bb14e27ab6b16b6e4aa0dbcf3ef56f/README.md)
- [Pinned dataset license](https://huggingface.co/datasets/nvidia/PointWorld-DROID/blob/dd9aaeec94bb14e27ab6b16b6e4aa0dbcf3ef56f/LICENSE.pdf)
- [Flow shard manifest](https://huggingface.co/datasets/nvidia/PointWorld-DROID/blob/dd9aaeec94bb14e27ab6b16b6e4aa0dbcf3ef56f/droid/flows-fs-optimized/_shards_manifest.json)
- [Generation branch, pinned commit](https://github.com/NVlabs/PointWorld/tree/3872ec6ee73146aa671192ef79b5dfbedc0246e3)
- [Canonical timeline generation](https://github.com/NVlabs/PointWorld/blob/3872ec6ee73146aa671192ef79b5dfbedc0246e3/real/compute_2d_flows.py)
- [Raw DROID state and camera conventions](https://github.com/NVlabs/PointWorld/blob/3872ec6ee73146aa671192ef79b5dfbedc0246e3/real/droid_utils.py)
- [3D annotation conversion and depth alignment](https://github.com/NVlabs/PointWorld/blob/3872ec6ee73146aa671192ef79b5dfbedc0246e3/real/convert_2d_flows_to_3d.py)
- [Euler and quaternion conventions](https://github.com/NVlabs/PointWorld/blob/3872ec6ee73146aa671192ef79b5dfbedc0246e3/transform_utils.py)
