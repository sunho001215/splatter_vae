# DROID csw4 profile: effective configuration (2026-11-02, before any full training launch)

User decision: "For DROID training, preserve the accepted csw4 architecture and method objective. Make dataset-specific
geometry, timing, units, image handling, and loader settings explicit. Do not silently inherit different decoder sizes,
slot counts, or loss weights by selecting a different base YAML. Keep any legacy DROID-base smoke profile separately
identified, and document the effective configuration before a full training launch."

## Profiles

| Profile | Composition | Status |
|---|---|---|
| **DROID csw4** | `configs/droid/csw4_largescale.yaml` alone (self-contained) | the profile for DROID training |
| Meta-World csw4 (reference) | `configs/metaworld/base.yaml` + `tasks/<task>.yaml` + `variants/chamfer_lean.yaml` + `chamfer_lean_randmask.yaml` + `enc_cs.yaml` + `enc_w4.yaml` (`runs/setup/encx-definition-20261026.json`) | accepted csw4 |
| LEGACY DROID-base smoke | `configs/droid/pretrain.yaml` + `configs/droid/largescale.yaml` (+ the four csw4 variant files in the 2026-10-31 smoke) | kept, documented, not csw4 |

The legacy composition inherits the DROID base's architecture and objective: decoder dim 256 / 8 heads, Gaussian parents
1536 / 512, scene `offset_scale` 0.25, 16 slots of 384, `loss.consistency` 0.1, `loss.depth_align` scale with a
5000-step warm-up, `render.far` 4.0. With the csw4 variant files stacked on it, it now stops at the synthetic-camera guard
below. `tests/test_droid_csw4_profile.py::test_legacy_droid_base_composition_is_not_csw4` records both facts.

## Method keys: equal to Meta-World csw4 (machine-checked)

`tests/test_droid_csw4_profile.py` resolves both configurations with `s4d.config.load_config` and asserts:

- every `model.*` key except `model.encoder.image_height`, `model.encoder.image_width` and `model.anchor_stats` is equal,
  with the same key set;
- every `loss.*` key is equal, with the same key set;
- every `aug.*` key except `aug.synth_camera_rig` is equal, with the same key set;
- the optimizer keys `train.{batch_size, lr, min_lr, weight_decay, warmup_steps, grad_clip, bf16, seed}` are equal;
- every key that differs at all is in the test's documented `DATASET_SPECIFIC` list. This is checked against the
  hammer, pick-place and bin-picking task files.

The architecture is therefore csw4: patch-local CNN stem, encoder width 256 / depth 4 / 4 heads, 1 slot of 256, random
75 % temporal-token masking, decoder 128 / depth 2 / 4 heads, Gaussian parents 768 / 256 with scene `offset_scale` 0.15.
The objective is chamfer_lean: RGB + SSIM, depth L1 with `depth_valid_far`, Gaussian-space motion, coverage, InfoNCE,
consistency 0.5, Chamfer 5.0. Disabled terms: depth_grad, depth_hard, visibility and occlusion. No depth alignment. Two
synthetic views that also serve as render targets.

## Every key that differs from Meta-World csw4 (24 keys, all dataset-specific)

| Key | Meta-World csw4 | DROID csw4 | Justification |
|---|---|---|---|
| `data.regime` | `metaworld` | `droid_largescale` | the lazy RLDS × PointWorld-DROID × DROID-3D data path |
| `data.root`, `data.task`, `data.train_ratio`, `data.episodes` | Meta-World HDF5 / task / 0.96 / null | null / absent | Meta-World sources; this regime does not use them |
| `data.manifest`, `data.pointworld_root`, `data.rlds_index_dir`, `data.rlds_data_root` | absent | null | the four sources; null fails fast until each is given explicitly |
| `data.validation_percent` | absent | 2.0 | episode-disjoint hash split (validation episodes) |
| `data.strides` / `data.val_strides` → `data.stride` | [2, 4, 6] / [2, 6] simulator steps | 3 | timing: 3 canonical PointWorld steps = 6 raw steps at 15 Hz = 0.4 s between window frames (fixed DROID canonical timeline) |
| `data.max_tracks` | absent | 4096 | per-camera track budget of the batch contract |
| `data.with_wrist` | absent | true | supervision-only wrist fields, emitted only for QA-accepted episodes; csw4 does not read them |
| `data.gripper_offset_m` | absent | -0.019 | geometry (m): approach-axis gripper point offset fitted on the inspected DROID sample |
| `data.max_open_files` | absent | 8 | loader: open handles per worker and file kind; measured flat (item 8 §3.6) |
| `model.encoder.image_height` × `image_width` | 128 × 128 | 144 × 256 | image handling: 0.8 × the native 180 × 320 grid, aspect kept, K scaled exactly. 128 × 128 would scale 16:9 images anisotropically. Consequence: 9 × 16 = 144 patch tokens per frame instead of 64; the masking ratio is unchanged (0.75: 108 of 432 temporal tokens kept instead of 48 of 192) |
| `model.anchor_stats` | Meta-World task statistics | null | geometry: must be the DROID workspace statistics (`scripts/compute_workspace_stats.py --regime droid_largescale`). Null is refused at training start for every non-Meta-World regime, because the default anchors are the Meta-World workspace |
| `aug.synth_camera_rig` | absent (= `metaworld`) | `training_cameras` | geometry: synthetic views jittered around the episode's own cameras. The Meta-World rig is a pose set in the Meta-World world frame. Same offsets, same `synth_range_scale` |
| `train.num_workers` | 6 | 8 | loader: 98.7 windows/s with 8 workers vs 48.5 with 4 on DROID (item 8 §3.6, depth gzip decode bound) |
| `eval.orbit_center` | [0.0, 0.6, 0.0] | [0.6, 0.0, 0.15] | geometry: diagnostic orbit renders in the DROID robot-base frame (dynamic anchor mean of the sample: 0.63, -0.01, 0.15 m) |
| `wandb.project` | `splatter4d-metaworld` | `splatter4d-droid` | bookkeeping |

Dataset-specific keys that are allowed to differ but are kept at the csw4 value:

- `render.near` 0.05 and `render.far` 3.0.
  - Ext valid depth on the 62-episode AUTOLab sample (one session): 99th percentile 1.79 m; 0.21 % lies beyond 3.0 m
    and 0.015 % beyond 4.0 m. Wrist depth: 1st percentile 0.068 m.
  - On 294 AUTOLab episodes (four sessions): 99th percentile 3.05 m; 1.04 % beyond 3.0 m, 0.43 % beyond 4.0 m. The far
    plane therefore excludes about 1 % of valid depth (distant background) from the depth and Chamfer targets
    (`loss.depth_valid_far`). 3.0 is kept: it is the csw4 value, and the excluded fraction is small.
  - Sources: `item8/r2/checks/depth_range_62.json`, `item8/r2/checks/depth_range_autolab294.json`.
  - One lab only, so this must be rechecked on corpus statistics.
- `train.steps` 300000: fixed at launch. csw4 pretraining used 200000.
- `train.{prefetch_factor, log_every, eval_every, save_every}`.
- `eval.{orbit_distance, orbit_elevation, probe_every, probe_batches}`. The DROID regime has no probe loaders.

## Two Meta-World geometry inheritances found while building the profile (integration fixes)

1. **Synthetic views.** `aug.synth_views: 2` built the synthetic cameras from the Meta-World rig on DROID
   (`augment.jitter_cameras`: LOOKAT (0, 0.6, 0), radius 1, the six Meta-World training cameras). This included the
   2026-10-31 smoke. The fix adds `aug.synth_camera_rig`:
   - Default `metaworld` is bit-identical to the reference.
   - `training_cameras` (`augment.jitter_training_cameras`) works per synthetic view:
     - pick a random training camera of the sample;
     - set the look-at point on its optical axis at the median valid depth of its first frame;
     - apply the Meta-World offsets (azimuth ±10°, elevation ±6°, radius ±5 %, lateral ±0.12 m);
     - re-aim at the look-at point with the smallest rotation, so roll is kept.
   - A non-Meta-World regime with synthetic views or `loss.self_render` and the Meta-World rig (or a Meta-World viewpoint
     domain) is refused.
   - Real DROID windows: 24 batches × 2 samples × 2 views, 144 × 256. Source:
     `/home/ws/ws/validation-20261031/item8/r2/checks/csw4/synth_coverage_real.json`; image
     `r2/diagnostics/csw4/synth_views_real.png`.

     | Rig | Covered pixels | Patches ≥ 90 % covered | Rows short of the 108 kept tokens |
     |---|---|---|---|
     | Meta-World rig | 24.6 % | 8.5 % | 100 % (median 32 eligible tokens) |
     | Training cameras | 77.2 % | 47.8 % | 1.0 % (median 203.5 eligible tokens) |

   - The real views' valid-depth fraction is 92.4 %.
2. **Anchors.** `model.anchor_stats: null` silently kept the decoder's default anchors, which are the Meta-World
   workspace (mean (0, 0.6, 0.1) m). `loop.anchor_statistics_path` now refuses this for any non-Meta-World regime.

## Validation state

- **Loss-function check on real data.** This is not the trainer smoke.
  - Setup: 20 optimizer steps of `loop.forward_losses` with the profile on real windows; batch 2; MIG
    `c4ec2a95`; smoke anchor statistics.
  - Losses and gradients are finite throughout.
  - Total loss 2.73 → 2.65. The csw4 10k-step warm-up keeps the learning rate tiny.
  - Pre-clip gradient norm max 9.5; 0.07 s per step; 0.51 GB GPU; 4.41 M parameters.
  - Source: `/home/ws/ws/validation-20261031/item8/r2/checks/csw4/loss_check_real.json`.
- **Trainer smoke.** `scripts/train.py`: 200 steps, stop at 100, resume. NOT RUN: `require_passed_tests()` needs a
  full-suite pass on these sources. Commands: `/home/ws/ws/validation-20261031/item8/r2/smoke_csw4.sh`.

## Before a full DROID launch

- Set `model.anchor_stats` to corpus workspace statistics.
- Recheck `render.far` on corpus depth percentiles.
- Fix `train.steps`.
- Give the four data sources.
- Run the full suite on the merged sources, then the trainer smoke above.

Do not launch full-scale training from this note.
