# DROID large-scale data path: status (2026-11-02, after round 2 of the real-data validation)

Branch `droid-largescale-20261031` (worktree `.worktrees/droid-largescale`), based on `2605ba2`. Pushed; not merged.

- Design, schemas and conventions: `docs/droid_largescale.md`.
- Real-data validation, round 1 (62 episodes, one session): `docs/droid_largescale_validation_20261031.md`.
- Round 2 (wrist QA, csw4 profile, 22 robots / 12 labs, 294-episode scale checks):
  `docs/droid_largescale_validation_20261102.md`.
- Effective csw4 configuration for DROID: `docs/droid_largescale_validation/csw4_effective_config.md`.
- ViT-S-sized profile with square crops (2026-11-02, branch `droid-vits-crop-20261102`):
  `docs/droid_largescale_validation/vits_effective_config.md`; preprocessing, crop, masking and the inference contract in
  `docs/droid_largescale.md` §7.
- Full commands and logs: `/home/ws/ws/validation-20261031/item8/REPORT.md` and `REPORT_20261102.md`.

## Done

| Piece | Files | Real data |
|---|---|---|
| TF-free TFRecord walker; in-place RLDS step index (resumable, `--range`, `--partial` with global shard ids, `shardLengths` check) | `s4d/data/droid/largescale/{tfrecord,rlds_index}.py`, `scripts/droid_ls_build_rlds_index.py` | VERIFIED-REAL: 345 shards |
| PointWorld schema checks, lazy bounded HDF5 access, strict tar-stream extractor | `pointworld_source.py`, `extract.py`, `scripts/droid_ls_extract_pointworld.py` | VERIFIED-REAL: 367 episodes |
| DROID-3D parsing, Franka FK, wrist poses, exact static-camera track rebase | `calibration.py`, `geometry.py` | VERIFIED-REAL; wrist chain as below |
| Join (see below) | `join.py`, `scripts/droid_ls_build_manifest.py` | 294/294 AUTOLab accepted; wrist QA v2 294/294 |
| Lazy dataset and `data.regime: droid_largescale` | `dataset.py`, `config.py`, `scripts/train.py` | VERIFIED-REAL on 294 episodes: contract, workers, memory, handles, resume replay |
| csw4 profile for DROID | `configs/droid/csw4_largescale.yaml`, `tests/test_droid_csw4_profile.py` | method keys = Meta-World csw4 (tested); loss-function check on real windows |
| Synthetic views from the episode's own cameras; anchor-statistics guard | `s4d/train/{augment,loop,perframe}.py` | coverage 24.6 % → 77.2 % on real batches |
| Workspace statistics for the regime | `scripts/compute_workspace_stats.py --regime droid_largescale` | smoke-only |
| ViT-S-sized profile: 384/12/6 on the patch-local CNN stem, 224 × 224, slot_dim 256 (decoder unchanged); every difference from csw4 registered | `configs/droid/vits_largescale.yaml`, `tests/test_droid_vits_profile.py` | resources measured on real windows (22.6 M parameters; batch 16: 6.5 GB peak allocated, 0.65 s/step on a MIG 1g.24gb slice) |
| Motion-informed stochastic square crop (one per camera and window), K / depth / tracks / motion score / wrist transformed, padding validity in the batch and the losses; centre crop for validation and inference | `s4d/data/droid/largescale/crop.py`, `dataset.py`, `config.py`, `s4d/data/contract.py`, `s4d/losses/rgb.py`, `s4d/train/loop.py` | 11 inspected real windows (`validation-20261031/droid_vits/samples/`); transformed-K error ≤ 1.4e-5 px; 68 windows/s with 8 workers |
| Single-stage masking: padded tokens forced masked, 147 of 588 kept (441 masked) for every crop | `s4d/model/encoder.py` (`sample_eligible_tokens`), `loop.crop_visible_tokens` | real batches: effective masked fraction in observed content 0.50–0.75 |
| Inference contract: central square crop + resize 224 with K transform; recorded in encoder exports of crop profiles | `crop.inference_center_crop`, `scripts/export_encoder.py` | unit-tested only; no deployment study |

The join covers:

- identity by uuid and path suffix;
- serial-, frame- and timestamp-validated frames;
- off-by-one state frames invalidated rather than rejecting the episode;
- per-robot calibration only for an identified robot;
- wrist QA v2;
- a provenance manifest.

The legacy DROID-base profile is `configs/droid/pretrain.yaml` + `largescale.yaml`. It is kept and identified, but it is
not csw4.

## Tests

| Run | Result |
|---|---|
| `tests/test_droid_largescale.py` | 64 passed (round 2 adds the QA-occlusion, consistency and off-by-one tests) |
| `tests/test_droid_csw4_profile.py`, `tests/test_non_metaworld_geometry.py` | 13 passed |
| Targeted suites affected by the rig change (16 files + DROID/contract/CLI/export) | 450 passed |
| Full native suite | **pending on the merged sources**. The parent runs it; the last full pass, `docs/tests.json`, is for `87b7846` |
| Trainer smoke with the csw4 profile (`scripts/train.py`, 200 steps, stop at 100 and resume) | **NOT-RUN**, blocked by the test gate; command in `item8/r2/smoke_csw4.sh` |
| ViT-S-sized profile (branch `droid-vits-crop-20261102`): `tests/test_droid_crop.py`, `tests/test_droid_vits_profile.py` and the touched suites (DROID, contract, encoder, export, decoder, losses, golden / equivalence files) | passed (counts in the branch handback); full native suite **pending** (parent) |
| Trainer smoke with the ViT-S-sized profile | **NOT-RUN**, pending the full-suite gate (same procedure as `smoke_csw4.sh` with `configs/droid/vits_largescale.yaml`) |

## Real-data status (item 8)

| Component | State |
|---|---|
| RLDS index, wire layout, decode | VERIFIED-REAL |
| DROID-3D files against the parser | VERIFIED-REAL |
| Join, frames, timestamps, external extrinsics, rebase, intrinsics convention | VERIFIED-REAL (AUTOLab, 294 episodes, 4 sessions). Depth-free identity, serial, state and FK checks: VERIFIED-REAL on 73 episodes, 21 robots, 11 labs |
| Wrist orientation (Rz 180) and pose chain composition | VERIFIED-REAL: photometric on 21 robots; depth-based on 1 robot |
| Wrist QA | VERIFIED-REAL for one robot (294/294; gross controls 97 % rejected). PARTIAL across robots: no wrist depth within budget |
| Per-robot `dT_gw` quality | PARTIAL: card worse than `dq`-only on 3 robots (photometric) and in one AUTOLab session; no rule implemented |
| 672x376 wrist intrinsics | PARTIAL: photometric, 5 episodes |
| Batch contract, loading, workers, memory, file handles, resume replay | VERIFIED-REAL on 294 episodes, single lab |
| Training smoke with the csw4 profile | NOT-RUN (gate); legacy-profile smoke of 2026-10-31 VERIFIED-REAL |
| ViT-S-sized profile: crop geometry, padding exclusion, forward/backward on real windows | VERIFIED-REAL on the 294-episode AUTOLab manifest (resource measurement, not the trainer smoke); trainer smoke NOT-RUN (gate) |
| Crop thresholds (2 cm / 0.4 m / 32 tracks) | PARTIAL: set from 80 camera-windows of one lab; re-check on the corpus |
| Anchor statistics, `render.far` justification | PARTIAL: one AUTOLab session; corpus statistics not computed |
| Corpus join rate, other labs through the full manifest (depth) | NOT-RUN (the depth stream is sequential: 252 parts of 4.9 GB) |

DROID cannot establish the RealSense stream configuration or the Panda–Robotiq mounting geometry of our own rig.

## Open decisions

1. **Per-robot or per-session wrist calibration.** The published chain is right on average and is the best fit for
   three of four AUTOLab sessions. One session shows a consistent 1.0 cm / 3.3° offset, and three robots look worse
   with `dT_gw` (photometric, 1-4 episodes each). Options:
   - keep the chain everywhere (current);
   - add a per-session refinement (ICP against the external cloud);
   - drop `dT_gw` for named robots.

   The last two need wrist depth from more robots.
2. **QA thresholds** were set from one robot, between the published chain and gross-error controls. Re-validate them
   with depth from other robots.
3. **Calibration default:** `droid3d` is kept.
4. **Track targets that are invisible at the target time** (2.6 % of tracks move more than 0.3 m per window).
5. **Image size 144x256** (dataset-specific by the user's rule) gives 144 tokens per frame instead of 64. The
   ViT-S-sized profile uses 224 × 224 square crops instead (196 tokens per frame); csw4 keeps 144x256.
6. **Crop per window, not per epoch.** The crop is a pure function of the window, following the dataset's seeding
   convention, so every epoch reuses it. This matters only when a small manifest is repeated for many epochs; at corpus
   scale each window is seen about once.
7. **Wrist crops are almost always broad** (193 / 200 `weak_motion`): few world-moving external tracks are visible to
   the wrist camera. The wrist fields are supervision-only and unused by the loss.

## Remaining steps (in order)

1. Parent: run the full suite on the merged sources, then `item8/r2/smoke_csw4.sh`: the csw4 trainer smoke with
   stop/resume, and the same smoke with `configs/droid/vits_largescale.yaml` (ViT-S-sized profile).
2. Index the RLDS release in place.
3. Extract flows (771 shards) and depth (252 sequential parts, about 3.9 TB of disk).
4. Build the manifest; review the rejection reasons and the wrist QA across robots.
5. Compute corpus `model.anchor_stats` and re-check `render.far`.
6. Resolve open decisions 1-2 with wrist depth from several robots.
