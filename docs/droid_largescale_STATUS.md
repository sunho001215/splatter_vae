# DROID large-scale data path: status (2026-10-31, after the real-data validation)

Branch `droid-largescale-20261031` (worktree `.worktrees/droid-largescale`), based on `2605ba2`. Not merged or pushed.

- Design, schemas and conventions: `docs/droid_largescale.md`.
- Real-data validation (item 8), with numbers, fixes and open decisions: `docs/droid_largescale_validation_20261031.md`.
- Evidence files: `docs/droid_largescale_validation/`.
- Full commands and logs: `/home/ws/ws/validation-20261031/item8/REPORT.md`.

## Done

| Piece | Files | Real data |
|---|---|---|
| TF-free TFRecord/Example walker, in-place RLDS step index (resumable, `--range`, `--partial` with global shard ids, `shardLengths` check) | `s4d/data/droid/largescale/{tfrecord,rlds_index}.py`, `scripts/droid_ls_build_rlds_index.py` | VERIFIED-REAL: 63 shards, 3,034 episodes |
| PointWorld schema checks (depth `metadata` group, 65535 saturation), lazy bounded HDF5 access, strict tar-stream extractor (non-hex users, truncated streams) | `pointworld_source.py`, `extract.py`, `scripts/droid_ls_extract_pointworld.py` | VERIFIED-REAL: 62 episodes |
| DROID-3D parsing, Franka FK, wrist poses, exact static-camera track rebase | `calibration.py`, `geometry.py` | Parsing, external extrinsics and rebase VERIFIED-REAL; wrist chain PARTIAL |
| Join: identity by uuid and path suffix (DROID-3D duplicates resolved), serial-, frame- and timestamp-validated; per-robot calibration only for an identified robot; wrist QA; provenance manifest | `join.py`, `scripts/droid_ls_build_manifest.py` | 62/62 accepted; wrist QA rejects 62/62 |
| Map-style lazy dataset and `data.regime: droid_largescale` | `dataset.py`, `config.py`, `configs/droid/largescale.yaml`, one `elif` in `scripts/train.py` | VERIFIED-REAL: contract, workers, memory, handles, throughput |
| Workspace statistics for the regime | `scripts/compute_workspace_stats.py --regime droid_largescale` | Smoke-only statistics computed |

Existing regimes, configurations, model and loss code are unchanged.

## Tests

| Run | Result |
|---|---|
| `tests/test_droid_largescale.py` | 61 passed (40 before; 21 real-data regression tests added) |
| Full native suite, `scripts/run_tests.py` on `87b7846`, local, MIG `2ebd9e11` | **1,283 passed, 0 failed, 0 errors, 0 skipped** (`docs/tests.json`) |
| Training smoke, `scripts/train.py`, csw4 overlays on the DROID configuration, real data, 200 steps + stop-at-100/resume | Passed: finite losses and gradients, checkpoints, resume continuity within run-to-run noise |

## Real-data status (item 8)

| Component | State |
|---|---|
| RLDS index, wire layout, ordinal 21512, decode | VERIFIED-REAL |
| DROID-3D files against the parser | VERIFIED-REAL (rectified K, `T_right_from_left` semantics, duplicates and placeholder robots documented and handled) |
| Join, frames, timestamps, external extrinsics, rebase, intrinsics convention | VERIFIED-REAL (62 episodes, one session) |
| Wrist orientation | VERIFIED-REAL |
| Wrist pose chain and QA | PARTIAL: no wrist fields emitted (QA rejects all 62); see open decision 1 |
| Batch contract, multi-episode loading, workers, memory, file handles | VERIFIED-REAL |
| Training smoke and resume | VERIFIED-REAL (200 steps) |
| Anchor statistics | PARTIAL: smoke-only (60 episodes, one session); corpus statistics not computed |
| Corpus join rate, multi-robot and multi-lab behaviour, 672x376 wrist intrinsics | NOT-RUN |

A single session shows functional integration only. DROID cannot establish the RealSense stream configuration or the
Panda–Robotiq mounting geometry of our own rig.

## Open decisions

1. **Wrist pose chain and an occlusion-robust QA statistic.** On robot `FR3-295341-1326595` the published DROID-3D
   chain with `dT_gw` fits worse than `dq`-only and than DROID's raw wrist pose. Decide after checking several robots.
2. **csw4 base keys on DROID.** The DROID base keeps `slot_dim` 384, `num_slots` 16, decoder 256/8, its Gaussian
   budget, `consistency` 0.1, `depth_align` scale and `far` 4.0. The Meta-World csw4 base differs in all of these.
3. **Calibration default** (`droid3d` kept).
4. **Track targets that are invisible at the target time** (2.6 % of tracks move more than 0.3 m per window).
5. **Whole-episode rejection for a single off-by-one proprio frame.**

## Remaining steps (in order)

1. Index the RLDS release in place. Either all shards, or the shards that hold joinable episodes: a metadata-only
   scan of all 2048 shards takes 20 s, see the validation note.
2. Extract the PointWorld flows (771 shards) and the depth stream (252 parts, sequential), with about 3.9 TB of disk.
3. Build the manifest with `--calibration droid3d`, then review the rejection reasons and the wrist QA across robots.
4. Resolve open decisions 1-2. Compute corpus `model.anchor_stats` and refit or validate `gripper_offset_m` on
   several robots.
5. Rerun the full suite on the merged sources, then a short single-GPU overfit and the throughput check with 8
   workers.
