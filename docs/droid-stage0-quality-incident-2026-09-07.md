# Stage-0 quality hold — 2026-09-07

The requested dataset is **not complete or approved for training**. The full
preprocessing continuation is stopped. No further pose-threshold relaxation,
eligibility change, or depth regeneration is authorized by this report.
User direction is required after the repeated pose-safety failures.

## Preserved state

Branch: `DROID`. Starting and current HEAD: `4d44e7e`; the refactor remains
uncommitted. Dynamic3D was not modified. All existing staged/unstaged work and
completed shards remain in place.

| Stage | Completion markers | Quality status |
| --- | ---: | --- |
| Real RGB | 791 / 791 | Preserved |
| DA3 | 791 / 791 | Metric geometry under investigation |
| MegaFlow | 791 / 791 | Preserved |
| LagerNVS | 15 / 791 | Existing strict schema-v2 shards preserved |
| Final composition | 0 / 791 | Not started |

Completion markers indicate stored stage outputs, not independent final dataset
quality approval. Existing Lager shards are
`0,1,2,3,4,5,6,7,8,9,10,11,13,14,17`.

The canonical manifest specifies 33,195 episodes, 9,502,985 raw timesteps,
19,005,970 raw exterior frames, 3,178,786 retained timestamps, 3,046,082 windows,
6,357,572 real JPEGs, 6,357,572 depths, 6,224,806 flow fields, and 12,715,144 Lager
JPEGs. None of these eligibility/count decisions changed during diagnosis.

The three revision-5 pose-audit PIDs (`2171846`, `2171851`, `2171856`) are absent.
Workers 1 and 2 failed; the remaining worker 0 was stopped with SIGINT.
The bounded diagnostic completed successfully and released GPU 4.

## Repeated pose failures

At the scheduled inspection on 2026-09-07 14:27:46 UTC, these records failed:

| Shard | Global key | Episode | Raw timestep |
| --- | --- | --- | ---: |
| 248 | `0000999926` | `PennPAL+acda9df3+2023-06-18-19h-59m-31s` | 279 |
| 388 | `0001564077` | `IPRL+7790ec0a+2023-05-05-17h-35m-13s` | 3 |

Both reported `No safe symmetric LagerNVS alpha pair exists in band [0.25, 0.35]`.
Current pose-contract signature:
`fea255f28d1d7fe61f1763c71cd06f05a09ff8c22380f969262ab4ddea569694`.
Only 366 of 776 unfinished Lager shards passed that audit, accounting for
1,473,012 timestamps. Historical failures led to progressively broader
exceptional clearance policies; revision 5 still fails. The original strict-pose
pilot does **not** establish image quality for the newer exceptional policy.
Further automated retry/relaxation would violate the requested stop policy.

Evidence under `/home/ws/data/droid_stage0_preprocessed`:

- `reports/lagernvs_pose_safety/shard-00248.failure.json`
- `reports/lagernvs_pose_safety/shard-00388.failure.json`
- `logs/preprocessing/lagernvs-pose-audit-v5-worker-{00,01,02}.log`

## DA3 scale diagnostic

The existing adapter supplies synchronized calibrated views to the pinned nested
metric any-view model. After its nested metric branch aligns depth, the official
API divides depth by `predicted_baseline / supplied_baseline` and replaces output
poses with supplied poses. Consequently, testing that the final baseline and
poses match input is not independent evidence for metric-depth correctness.
The nested metric scale factor cancels algebraically from that final baseline
rescaling; an `is_metric` flag alone does not verify the resulting scale.

Production shards stored depth only, so their individual pose-alignment scale
factors cannot be recovered from saved metadata. A bounded re-inference used the
same pinned DA3 checkpoint/configuration on cached Q95 JPEGs for two failures
and one original pilot control. This is not a bit-exact raw-RLDS replay.

| Record | Calibrated baseline (m) | Divisor | Median before alignment A / B (m) | Median after alignment A / B (m) | Cached median A / B (m) |
| --- | ---: | ---: | --- | --- | --- |
| PennPAL failure | 0.5990 | 3.0056 | 0.4180 / 0.5628 | 0.1391 / 0.1873 | 0.141 / 0.182 |
| IPRL failure | 0.1182 | 14.2077 | 1.3689 / 0.8097 | 0.0963 / 0.0570 | 0.102 / 0.056 |
| ILIAD control | 0.9078 | 1.0579 | 0.5336 / 0.8658 | 0.5044 / 0.8184 | 0.501 / 0.821 |

The source images show substantial robot and room geometry. The IPRL camera-B
median of roughly 5.7 cm is visually suspect. These measurements identify where
the shrinkage occurs, **not a validated correction**. Simply disabling pose
alignment or multiplying cached depth by a heuristic factor is not approved.

Both failed episodes have only one direct `cam2base` pose; the other is derived
from released `cam2cam` data. Neither has an independent direct-pose comparison.
The release reports `metric_type=num_matches`, `quality_metric=1` for PennPAL
and `quality_metric=170` for IPRL. The relative calibration files are the existing
canonical inputs, not newly changed calibration. This supports investigating
calibration quality as well as predicted poses; it does not prove which is wrong.

An independent SIFT mutual-ratio-match triangulation diagnostic found zero
usable calibrated correspondences in all three low-resolution pairs. It is
therefore inconclusive and cannot certify either scale. Synthetic metric-depth,
reprojection-rejection, and negative-depth tests for triangulation all pass.

Machine-readable measurements, input payload hashes, and pre-alignment predicted
camera matrices are saved in
`reports/da3-scale-incident-20260907.json` under the derived dataset root.
The inspected source/depth panel is also preserved beside that report as
`da3-scale-incident-20260907-source-depth.jpg`.

## Safety and verification

An active `reports/preprocessing_quality_hold.json` now prevents both the
top-level launcher (including `--allow-missing-pilot-gate` and dry-run) and direct
stage workers from starting. The historical passing pilot report is preserved,
not overwritten. Read-only state inspection and diagnostics remain available.
After an explicitly reviewed, validated resolution, archive the hold marker
with its evidence; do not simply bypass it.

Relevant regression tests: **27 passed** (safety, codecs, offline adapters, and
the new scale diagnostic). Ruff and `git diff --check` passed. The preceding
validation run had 93 passing tests and a complete 1,201-timestamp pilot scan,
including 442 independently checked geometry timestamps and 64 loader windows.
Those checks do not resolve this broader geometry/eligibility issue.

Source metadata inventory still matches its saved baseline: 2,051 files,
1,866,281,754,039 bytes, fingerprint
`6f90d6f97e73d36e0243d89fe6e26621c05670c299294ee3759b0ed370d99ad3`.
This is an unchanged metadata inventory, not a full source-content hash.
No original DROID files or cached shards were changed by this diagnostic.

Available capacity at handoff: 6,798,420,860,928 bytes. The earlier pilot projected
1,029,772,464,075 final bytes and 2,462,755,287,867 safe required bytes.
Storage is not the blocker; these remain projections, not final dataset totals.

## Reproduction and decision needed

From `/home/ws/ws/droid_training`, inspect state without launching work:

```bash
.venv/bin/python -m scripts.preprocess_droid_stage0 status \
  --root /home/ws/data/droid_stage0_preprocessed
```

Reproduce the bounded diagnostic using only GPU 4 and local checkpoints:

```bash
CUDA_VISIBLE_DEVICES=GPU-76871c16-ff1d-f7b1-cdf5-fabbaf9df8ce \
HF_HUB_OFFLINE=1 OMP_NUM_THREADS=4 \
.preprocessing-envs/da3/bin/python -m scripts.diagnose_droid_da3_scale \
  --sample 248:999926 --sample 388:1564077 --sample 329:1327791 \
  --report /home/ws/data/droid_stage0_preprocessed/reports/da3-scale-incident-recheck.json
```

Decide how to handle original Stage-0-valid episodes that cannot satisfy
independent metric-geometry and safe-target requirements. Any stricter eligibility
filter or calibration replacement needs an explicit decision, and any revised
depth path needs a representative quality pilot before regeneration. Preserve
all existing artifacts for comparison. Do not substitute another teacher, relax
target safety again, silently drop episodes, or launch another multi-day retry.

Outstanding: resolve metric geometry and pose safety, validate the revised pilot,
finish Lager/composition, final whole-dataset integrity, completed-dataset loader
and training profile, single-GPU save/resume/W&B validation, and 3-GPU DDP pilot.
The requested successful-completion report cannot yet be issued.
