# DROID ViT-S-sized profile: effective configuration (2026-11-02, before any training launch)

User decision (2026-11-02, supersedes the earlier rule that DROID encoder dimensions must equal csw4): a new explicit
DROID large-scale profile with 224 x 224 inputs, patch 16, transformer width 384 / depth 12 / 6 heads / MLP ratio 4 and our
existing patch-local convolutional stem; motion-informed stochastic square crops of the native 180 x 320 frames; one 75 %
masking budget over the full token grid with padded tokens forced masked. Temporal processing, the Gaussian
representation and the method objective are kept.

Naming: **ViT-S-sized backbone (384/12/6) with our patch-local convolutional stem**. Only the transformer body has ViT-S
dimensions; the stem is ours (`PatchCNNStem`), and there are no pretrained weights. The csw4 encoder (256/4/4) is not
ViT-S.

## Profiles

| Profile | File | Status |
|---|---|---|
| **DROID ViT-S-sized** | `configs/droid/vits_largescale.yaml` alone (self-contained) | new; resolved: `vits_largescale_resolved.yaml` |
| DROID csw4 | `configs/droid/csw4_largescale.yaml` | unchanged, all its tests unchanged |
| LEGACY DROID-base smoke | `configs/droid/pretrain.yaml` + `largescale.yaml` | unchanged |

## Every key that differs from the DROID csw4 profile (machine-checked)

`tests/test_droid_vits_profile.py::test_every_difference_from_droid_csw4_is_registered` resolves both profiles and fails
on any differing key that is not registered there with a justification. The registered keys:

| Key | DROID csw4 | DROID ViT-S-sized | Justification |
|---|---|---|---|
| `model.encoder.image_height` × `image_width` | 144 × 256 | 224 × 224 | image (user): square crops of the native grid resized to 224. 14 × 14 = 196 tokens per frame, 588 per 3-frame window |
| `model.encoder.width` | 256 | 384 | architecture (user): ViT-S-sized width |
| `model.encoder.depth` | 4 | 12 | architecture (user): ViT-S-sized depth |
| `model.encoder.heads` | 4 | 6 | architecture (user): ViT-S-sized heads; head dimension 64 as in csw4 |
| `model.encoder.mlp_ratio` | absent (default 4.0) | 4.0 | architecture (user): stated explicitly; same value as the default |
| `data.crop.*` (13 keys) | absent | see below | image (user): the motion-informed square crop and its evaluation policy |

`tests/test_droid_vits_profile.py::test_kept_method_keys_and_the_csw4_relation` also checks that every remaining method
key (`model.*`, `loss.*`, `aug.*`, the optimizer keys of `train.*`) equals the Meta-World csw4 composition, as the csw4
profile test does.

### Kept on purpose

- `model.encoder.stem: patch_cnn`, `patch_size: 16` (the stem needs 16 = four stride-2 convolutions).
- `num_frames: 3`, temporal embeddings, `num_slots: 1` state token, `mask_mode: random`, `mask_ratio: 0.75`,
  `drop_path: 0.1` (linear over 12 blocks instead of 4).
- **`slot_dim: 256`.** The state token is 384-wide inside the transformer. `slot_proj` (`Linear(384, 256)`) is the only
  connecting projection that changes. Why 256:
  - the decoder (`film`, `cond`, `slot_proj`, the Gaussian budget 768 / 256 parents) is parameter-for-parameter the csw4
    decoder (tested: identical names and shapes);
  - every downstream consumer of the encoder state keeps its dimension: `Encoder.state_dim` = 256, so
    `policy_state` stays (B, 256) per camera; `s4d/rl/encoders.py` and the export format are unchanged;
  - it is the csw4 method value. A 384 slot would change the decoder input layers and every policy input layer
    without being asked for.
- Every `loss.*` weight, `aug.*` (two synthetic views from the episode's own cameras), `render.*`, `eval.*`,
  `train.*` (batch 16, lr 5e-4, 10k warm-up) and `wandb.project`.

### `data.crop` (s4d/data/droid/largescale/crop.py)

| Key | Value | Meaning |
|---|---|---|
| `out_size` | 224 | output side; must equal the encoder image size (checked) |
| `side_min`, `side_max` | 180, 320 | crop side range in native pixels (full height … full width) |
| `motion_min_m` | 0.02 | a reliable track moves if its largest visible-to-visible world displacement exceeds 2 cm |
| `motion_max_m` | 0.4 | larger jumps are track / depth-edge failures |
| `min_moving_tracks` | 32 | support needed for a motion-informed crop |
| `extent_quantiles` | [0.05, 0.95] | robust motion-weighted extent of the moving tracks over the three history frames |
| `margin_frac`, `margin_min_px` | 0.2, 24 | context margin `max(24, 0.2 × extent)` on every side |
| `zoom_span_px` | 80 | side uniform in `[s_lo, s_lo + 80]` (≤ 320) |
| `broad_side_min` | 256 | broad fallback side uniform in `[256, 320]` |
| `jitter_frac` | 0.1 | vertical padding split: centre ± 0.1 s |
| `eval_policy` | center | validation windows: the padding-free central 180 × 180 square (= inference contract) |

The 2 cm threshold comes from the measured distribution of per-track displacements on 40 random real windows (80
camera-windows): median 1.8 mm, 75th percentile 7 mm. Even nearly static camera-windows have 4–7 % of tracks above 1 cm
(depth noise), so 1 cm was too permissive. The pooled 95th / 99th percentiles are 0.53 m / 0.90 m: jumps of that size
within 0.8 s are depth-edge or tracking failures (cf. open decision 4 of the status page), hence the 0.4 m cap. Tool:
`/home/ws/ws/validation-20261031/droid_vits/tools/explore_disp.py`. These values were set on one lab; re-check on the
corpus.

## Measured resources (new profile, real windows; not the csw4 numbers)

Real windows from the 294-episode AUTOLab manifest (one lab, four sessions); MIG 1g.24gb slice `c4ec2a95` (may be
shared with small jobs). Step = `forward_losses` under bf16 autocast + backward + clip + AdamW step, training mode
(crop masking, two synthetic views); median of steps 3–8. Source: `vits_resources_autolab294.json`
(`/home/ws/ws/validation-20261031/droid_vits/resources.json`, tool `droid_vits/tools/resources.py`).

| Parameters | Count |
|---|---|
| Encoder | 21,760,416 (stem 290,720; 12 blocks 21,293,568) |
| Decoder | 868,392 (= csw4 decoder) |
| Total | 22,628,808 (csw4 DROID profile: 4.41 M) |

| Batch | Peak allocated (GB) | Peak reserved (GB) | Step time (s) |
|---|---|---|---|
| 1 | 0.78 | 0.80 | 0.134 |
| 2 | 1.18 | 1.60 | 0.161 |
| 4 | 1.95 | 2.70 | 0.225 |
| 8 | 3.46 | 4.94 | 0.359 |
| 16 (profile) | 6.51 | 9.38 | 0.651 |

| Loader workers (batch 16, crop on) | Windows/s |
|---|---|
| 0 | 7.9 |
| 4 | 28.5 |
| 8 | 68.0 |

At batch 16 one step consumes 16 windows in 0.65 s (24.6 windows/s), so 8 workers keep up on this host; 4 workers are
marginal. The csw4 profile's 98.7 windows/s (8 workers) was measured without the crop on a 62-episode manifest.

## Validation state

- Focused tests: see `docs/droid_largescale_STATUS.md`.
- Inspection of transformed real samples: `/home/ws/ws/validation-20261031/droid_vits/samples/` (PNG panels and
  `samples.json`); summary in `docs/droid_largescale.md` §7.6.
- **Trainer smoke: NOT RUN.** `scripts/train.py` requires a full-suite pass on these sources (`docs/tests.json`). The
  parent runs the full suite, then the smoke with this profile (as `item8/r2/smoke_csw4.sh` with
  `CONFIG=configs/droid/vits_largescale.yaml`).

## Before a full launch

- Corpus `model.anchor_stats` (the AUTOLab-294 statistics are provisional, single-lab, one session group).
- Re-check `render.far` and the crop thresholds on corpus statistics.
- Full suite, then the trainer smoke with stop/resume.
