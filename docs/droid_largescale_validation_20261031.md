# DROID large-scale data path: real-data validation (item 8, 2026-10-31)

Branch `droid-largescale-20261031`. This note records what was run on real data, what it found, what was fixed and
what is still open. The design is in `docs/droid_largescale.md`; the short status is in
`docs/droid_largescale_STATUS.md`. Full commands, logs and every intermediate file are outside the repository in
`/home/ws/ws/validation-20261031/item8/` (`REPORT.md` there). Small evidence files are copied to
`docs/droid_largescale_validation/`.

**Scope.** 62 PointWorld-DROID episodes, all from one AUTOLab session (2023-10-21, robot `FR3-295341-1326595`, the
same three cameras), plus one raw DROID episode for that session. This shows functional integration on real bytes.
It is not a corpus measurement, it covers one robot, one lab and one camera rig, and DROID cannot establish the
future RealSense stream configuration or the Panda–Robotiq mounting geometry of our own rig.

## 1. Inputs (read-only)

| Source | What was read | Check |
|---|---|---|
| DROID RLDS 1.0.1 `/home/ws/data/droid/1.0.1` | `dataset_info.json`, `features.json`; 63 of 2048 shards indexed (62 holding the episodes, plus 464); a metadata-only scan of all 2048 shards (1.9 M small preads, 20 s) | sha256 of the 2 metadata files and the 63 shards, stat of all 2,051 entries and of both directories: identical before and after |
| PointWorld-DROID @ `dd9aaeec` | Listing (1,030 files, 3.9 TB); downloaded flows `shard-000000` (3.53 GB) and depth `part-0000` (4.90 GB), both sha256-verified; the cameras package was already local | Stream-extracted the 62 episodes of `shard-000000`; the sample episode's flow/depth/camera files are byte-identical to the earlier sample |
| DROID-3D @ `69db9ccf` | `extrinsics.jsonl`, `camera_intrinsics.jsonl`, README | All lines parsed; 25 records spot-checked field by field |
| Hand-eye assets | DepthWorld `6823c13d` and PointWorld `3872ec6` `gripper2wrist_transforms.json` | Identical matrices; PointWorld's also names each entry's wrist camera |
| Raw DROID episode `AUTOLab+0d4edc83+2023-10-21-19h-07m-04s` | metadata, `trajectory.h5`, the three MP4s | Used for timestamps, state, resolution and wrist checks |

## 2. Integration fixes (each its own commit, each motivated by a real-data finding)

| Commit subject (abridged) | Real-data finding |
|---|---|
| Partial RLDS index with global shard ids, checked against `shardLengths` | Indexing 2 of 2048 shards could not be finalized; `dataset_info.json` record counts match (56/48) and give ordinal 21512 |
| uuid/folder diagnostic accepts the real folder formats | False for 24,652 of 73,217 DROID-3D records (asctime day padding, date directory past midnight, non-hex users, colon-free IRIS folders, `tri+` uuids); now true for all |
| Select DROID-3D records by uuid and path suffix | 100 uuids twice in `camera_intrinsics.jsonl` (success/ and failure/); intrinsics `raw_dir` was never checked |
| PointWorld depth files carry a `metadata` group | All 62 episodes were rejected (`unexpected PointWorld depth group 'metadata'`) |
| Extractor keeps members with non-hex user ids | 1,771 of 42,935 released members would have been silently ignored |
| Extractor reports a truncated tar stream | Extracting from `part-0000` alone crashed with no report after writing every wanted member |
| uint16 depth saturation (65535 mm) is invalid | 65.535 m depth targets in 2 of 62 episodes |
| Workspace statistics for the `droid_largescale` regime | `model.anchor_stats` is null and no tool could compute statistics from manifest episodes |
| Per-robot wrist calibration needs an identified robot | DROID-3D placeholder serials `fr3-` (6,203 PointWorld episodes) and `unknown` (475; 39-degree corrections, two labs) selected one pooled hand-eye entry |

Every fix has a regression test in `tests/test_droid_largescale.py` (40 → 61 tests). No model, loss or method
configuration changed.

## 3. Findings

### 3.1 RLDS

- Wire layout as assumed: one flat `tf.train.Example` per episode, the 23 `features.json` keys and no others, packed
  float32 payloads of exactly `4·width·T` bytes, `is_first`/`is_last` only at the episode ends (104/104 records),
  JPEG 180x320 RGB. 3,034 records of 63 shards indexed without a single refusal.
- Shard 463 row 27 = ordinal 21512 = `AUTOLab/success/2023-10-21/Sat_Oct_21_19:07:04_2023`, 127 steps.
- RLDS step `k` = raw `trajectory.h5` step `k` (128 raw steps, the last is dropped; state equal to 1.2e-7).
- pread + PIL decoding gives the same frames as the earlier TFDS decoding (best match at the same index for all
  127 frames x 3 cameras; mean absolute difference 1.35-1.61 grey levels from the JPEG decoders).
- All 95,658 records: `file_path` and `recording_folderpath` suffixes agree and are unique.

### 3.2 PointWorld

- Schema as in the design, plus the depth `metadata` group, per-group camera attrs, (16,180,320) gzip depth chunks,
  and the saturation code 65535 (handled).
- Timestamps: depth timestamps = raw `<serial>_frame_received` (127/127, three cameras); canonical timeline = ext1
  `[::2]`; nearest depth index = `2i` in 62/62 episodes (ext1 0 ms, ext2 at most 33 ms, wrist at most 29 ms).
- `scene_flows` are world positions (median norm 0.80 m; median clip displacement 2 mm, 6.7 % of points move more
  than 1 cm).
- Robot state: PointWorld `gripper_pose` = RLDS `cartesian_position[2i]` exactly in all but 2 of about 8,000 clip
  frames (62 episodes); both exceptions are one frame where PointWorld used raw step `2i-1` (1.75 mm, under the 2 mm
  gate). A faster episode would be rejected entirely for one such frame (open point).

### 3.3 DROID-3D (real bytes, against the parser)

- Field names, types and units as parsed (metres, radians, 4x4 lists). The parser needed no change; the join did
  (duplicates, placeholders).
- Intrinsics are ZED SDK 5.1.2 *rectified* values (distortion all zero, left = right), 1280x720, except 1,523 wrist
  cameras at 672x376 (IRIS, RAIL, REAL, IPRL; their rescale is unverified). `T_right_from_left` has translation
  `+baseline` (the right camera's pose in the left frame, despite the name; unused).
- `T0_ext1_in_world`/`T1_ext2_in_world` are the left camera's world-to-camera matrices: robot gripper points vs
  PointWorld depth give a one-sided median of 1.3-1.5 cm, against 19-95 cm when read as the right camera, shifted
  by `-baseline`, or inverted.
- DROID-3D vs PointWorld calibration on the sample: 2.1 cm / 3.8 degrees (ext1), 2.1 cm / 2.9 degrees (ext2);
  PointWorld vs the original DROID calibration: 0.5-0.7 cm / 0.9-1.0 degrees.

### 3.4 Geometry and resolution

- RLDS 320x180 frames are centre-aligned area downsamples of the 1280x720 SDK-rectified MP4 frames (box downsample
  best shift 0.00 px; decimation fits much worse). DROID-3D K converted as `(c+0.5)/4` is exact for the image;
  PointWorld's K is 0.375 px off (0.375-0.376 px on all 62 episodes) and is kept for consistency with its tracks.
- Exact static rebase: pixels move by at most 7e-5 px, depth by 2e-7 m; the naive matrix swap raises the track depth
  residual from 0.7 mm to 12.7 mm.
- DROID `cartesian_position` = Franka flange FK of `joint_position` (3e-8 m on the sample, 0 on all 62): the
  flange-vs-controller-EE ambiguity of the hand-eye does not arise for these robots.
- Wrist orientation: projecting coloured external tracks into the wrist image, the DROID-3D chain with the 180-degree
  rotation matches (colour error 33.0, null 66.7, cross-camera floor 17.0) and without it does not (54.3). The released
  wrist depth is stored in the RGB orientation.
- Wrist pose chain (62 episodes, one robot): in the DROID-3D world the `dq`-only variant fits best (best colour score
  in 44/62, best depth inliers in 41/62); the published chain with `dT_gw` is best in 1/62, and its camera sits 1.0 cm
  / 4.0 degrees from DROID's own per-step wrist pose. Not changed: one robot is not enough evidence.
- Wrist QA rejects 62/62 episodes at 3 cm (as-stored median 0.18 m). The median is inflated by the gripper fingers
  and arm occluding the wrist view (DROID's own wrist pose gives 3.5 cm). Wrist fields are therefore absent from the
  production manifest. A diagnostic manifest with the threshold at 10 cm (18 episodes) exercised the wrist path end
  to end (shapes, per-frame poses, depth, track visibility).

### 3.5 Join, batch contract, loader

- Manifest: 62/62 accepted in both calibration modes (identity, serials, state, timestamps); corpus identity estimate
  41,599 of 42,935 PointWorld episodes have a DROID-3D record and a unique RLDS episode (frame checks not included).
- Batch contract: 85 real windows validated field by field (shapes, dtypes, finite, ranges) plus a collated batch;
  4,096 tracks per camera, 85 % supervised, depth valid 92.5 % (max 44.9 m after the 65535 fix), motion score positive
  on 8.4 % of pixels.
- Episode boundaries: 3,129 windows over 62 episodes, none crosses a clip or episode; 42 boundary windows equal a
  direct RLDS read.
- Bounded resources: 1,500 random windows with `max_open_files` 8: open files flat at 29 (8 per kind), RSS 580-620 MB
  without growth, file count back to baseline after `close()`.
- Workers: 0, 4 and 8 DataLoader workers give bit-identical batches (sha256 of every tensor); no descriptor leaks.
- Throughput (CPU, 144x256, wrist decoding on): 11.9 / 48.5 / 98.7 windows/s with 0 / 4 / 8 workers. Depth reading
  (gzip chunks of 16 frames) dominates a window (46 ms of 81 ms); the HDF5 chunk-cache size makes no difference.

## 4. Training smoke (csw4, real data)

Gate: full native suite on the final sources (commit `87b7846`), local, MIG `2ebd9e11`, **1,283 passed, 0 failed,
0 errors, 0 skipped** in 486 s (`docs/tests.json`, `docs/droid_largescale_validation/full_suite_summary.json`). A
first run (1,277 passed, 1 failed) lacked the official DINO ViT-S/8 weights in the worktree cache. That was an
environment gap, not a code failure: the weights were copied from the main checkout's cache (sha256 = the pinned
value). Run 1 is superseded because the sources then changed.

Composition. The Meta-World csw4 recipe is `base + task + chamfer_lean + chamfer_lean_randmask + enc_cs + enc_w4`. The
four variant files are the method keys: `aug.synth_views`/`synth_render`, `loss.*`, and `model.encoder` mask, stem
and body. They apply to the DROID configuration unchanged. All 16 keys resolve to the same values as in the
Meta-World csw4 configuration:

    scripts/train.py --config configs/droid/pretrain.yaml configs/droid/largescale.yaml \
      configs/metaworld/variants/chamfer_lean.yaml configs/metaworld/variants/chamfer_lean_randmask.yaml \
      configs/metaworld/variants/enc_cs.yaml configs/metaworld/variants/enc_w4.yaml \
      --set data.manifest=… data.pointworld_root=… data.rlds_index_dir=… data.rlds_data_root=/home/ws/data/droid/1.0.1 \
            model.anchor_stats=<smoke-only statistics> train.steps=200 train.save_every=100 train.eval_every=100 \
            train.log_every=10 train.num_workers=4 eval.max_batches=4 --no-wandb

The base keys that differ between the Meta-World csw4 configuration and the DROID base keep their DROID values (see
`smoke_csw4_analysis.json` → `resolved_differences_metaworld_csw4_vs_droid_run`; open decision §6):

- decoder: dim 256 vs 128, heads 8 vs 4, scene/dynamic parents 1536/512 vs 768/256, scene offset_scale 0.25 vs 0.15;
- encoder: `num_slots` 16 vs 1, `slot_dim` 384 vs 256;
- loss: `consistency` 0.1 vs 0.5, `depth_align` scale vs none;
- `render.far` 4.0 vs 3.0;
- image size 144x256 vs 128x128;
- the schedule.

Anchor statistics came from the 60 training-split episodes only and are smoke-only. The resolved configuration is in
`smoke_csw4_resolved_config.yaml`.

Results (batch 2, 3,029 training and 100 validation windows; 0.76 GB GPU; about 0.09 s per step):

- **Reference run, 200 steps uninterrupted:**
  - total loss 2.93 → 1.49;
  - validation PSNR 9.99 → 14.45;
  - all losses and the 84 logged encoder/decoder gradient norms finite (the loop also raises on a non-finite loss or
    gradient).
  - Two occlusion diagnostics (`metric/hidden_fraction`, `metric/floater_fraction`) are NaN by design at one log step
    with no opaque Gaussian. They are metrics, not losses, and appear at the same step in both runs.
- **Interrupted run, then resumed:**
  - stopped at step 100 (`train.stop_step`), then resumed with `--resume auto` ("resumed from … latest.pt at step 100");
  - checkpoints at 100 and 200 hold the step, Adam per-parameter step counts 100/200 and scheduler `last_epoch`
    100/200, so the optimizer state was carried over, not reset.
- **Continuity:** over steps 110-200 the resumed losses differ from the reference by 0.007 on average, 0.025 at most.
  The first 100 steps of the two identically configured runs already differ by up to 0.049 (GPU nondeterminism), so
  the resume adds no break. Step-200 loss 1.4862 vs 1.4858.

## 5. Status matrix

| Item 8 component | State | Evidence |
|---|---|---|
| RLDS step index on real shards (wire layout, counts, boundaries, identity, decode, ordinal 21512) | VERIFIED-REAL | §3.1 |
| Partial index / global ordinals (`shardLengths`) | VERIFIED-REAL | fix 1 |
| DROID-3D parsing against real bytes | VERIFIED-REAL | §3.3 |
| Episode join (uuid, suffix, serials, duplicates, placeholders) | VERIFIED-REAL (62/62; corpus identity estimate only) | §3.5 |
| Frame/timestamp mapping (RLDS ↔ raw ↔ PointWorld ↔ depth) | VERIFIED-REAL | §3.2 |
| External extrinsics (left w2c), rebase, positions | VERIFIED-REAL | §3.3-3.4 |
| Resolution-aware intrinsics (0.375 px) | VERIFIED-REAL | §3.4 |
| Wrist orientation (image and depth) | VERIFIED-REAL | §3.4 |
| Wrist pose chain and wrist QA | PARTIAL (published chain fits worse than simpler variants; QA rejects 62/62) | §3.4 |
| Wrist intrinsics at 672x376 | NOT-RUN (no such episode extracted) | §3.3 |
| PointWorld extraction (strict, partial streams, non-hex users) | VERIFIED-REAL | fixes 4-6 |
| Batch contract through the dataset | VERIFIED-REAL (external), PARTIAL (wrist: diagnostic manifest only) | §3.5 |
| Multi-episode windows, memory, file handles, workers, throughput | VERIFIED-REAL (62 episodes, one session) | §3.5 |
| Training smoke via `scripts/train.py` (gate, csw4, finite, checkpoint, resume) | VERIFIED-REAL (200 steps) | §4 |
| Anchor statistics | PARTIAL (smoke-only from 60 episodes; corpus statistics not computed) | §4 |
| Corpus-scale join rate, wrist QA pass rate, multi-robot behaviour | NOT-RUN (one session only) | — |

## 6. Open decisions

1. **Wrist pose chain and QA statistic.** On this robot the published chain with `dT_gw` is worse than `dq`-only and
   than DROID's raw wrist extrinsics. The 3 cm median QA rejects every real episode because of occlusion. Check
   several robots before choosing a chain and an occlusion-robust statistic. Until then no wrist supervision is
   emitted.
2. **csw4 base keys on DROID.** Decide whether the DROID run should also take the Meta-World csw4 base values
   (`slot_dim` 256, `num_slots` 1, decoder 128/4 heads, Gaussian budget, `consistency` 0.5, `depth_align` none,
   `far` 3.0) or keep the DROID base. This is a method decision; nothing was changed.
3. **Calibration default.** DROID-3D external extrinsics fit the wrist photometric test better. PointWorld's fit
   PointWorld's own depth better (not independent). The overlay default stays `droid3d`.
4. **Track targets.** 2.6 % of source-visible tracks move more than 0.3 m within a window, and 88 % of those are not
   visible at the target time. The contract supervises by source visibility only.
5. **Proprio off-by-one.** PointWorld occasionally uses the neighbouring raw step for one proprio frame (2 of about
   8,000 frames). The 2 mm gate would reject a whole fast episode for it.
