# DROID Training workspace summary

Prepared **2026-10-06**, with GitHub publication verification added below. This
is the consolidated instruction, implementation, evidence, and cleanup handoff.
Paths below are relative to this workspace unless absolute. Historical
measurements are labelled; they are not completion claims.

## State at cleanup, before GitHub publication

- Workspace: `/home/ws/ws/droid_training`; repository
  `sunho001215/splatter_vae`; branch **DROID**.
- Starting and current HEAD: `4d44e7e` (`update`). The extensive offline refactor
  is still **uncommitted**, with both staged and unstaged/untracked changes.
  Submodule replacements are staged. No reset, branch switch, or commit was made.
- The full dataset is **incomplete and not approved for normal training**.
  The original preprocessing goal is blocked; the summary/cleanup request does
  not authorize restarting processing or changing quality/eligibility rules.
- On this inspection there are no running DROID preprocessing, pose-audit,
  scale-diagnostic, or training processes. No worker was launched during cleanup.
- The active derived-root marker
  `reports/preprocessing_quality_hold.json` prevents the preprocessing launcher
  and direct stage workers from starting, including pilot-bypass/dry-run launch.
  Read-only state inspection and diagnostics remain available.
- Dynamic3D was not modified by the refactor or cleanup. Original DROID remains
  read-only. Existing source/configuration changes, outputs, and failures are
  preserved for continuation and comparison.

## User instructions and acceptance contract

The original 52-part request supersedes earlier DROID preprocessing/training
instructions where they conflict. The following register covers all its parts,
including requirements still outstanding; repeated final-summary constraints
remain binding.

| Original parts | Instructions to preserve |
| --- | --- |
| 1–3 | Continue the existing repository on DROID; inspect pwd/status/branch/last 15 commits/submodules/diff before edits. Inspect DROID config/dataset/preprocessing, online pipeline/training/loss/validation/model/visualization/profiling/tests/git metadata; classify keep/modify/remove/obsolete/untested/broken and avoid duplicate active implementations. Preserve unrelated user changes and Dynamic3D. Original `/home/ws/data/droid` must never be modified, moved, renamed, deleted, overwritten, or used for caches. Default derived root is `/home/ws/data/droid_stage0_preprocessed`; stop if capacity is insufficient, never silently choose another mount. Use only the three GPU UUIDs below and local CUDA indices. |
| 4–6 | Normal training is cached loading → crop/transforms → ViT-S → Gaussian decoder/gsplat → losses/backward, with no frozen foundation-model forward. Remove active X-Lens, MEMFOF, OnlineTeacherPipeline, obsolete submodules/config/imports/scripts/tests/docs; conservatively preserve external files and uncertain checkpoints. Offline models are DA3/MegaFlow/LagerNVS; isolate incompatible dependencies and record exact repo/checkpoint/environment/package provenance. Training must not need teacher imports. |
| 7–9 | One loader and one dataset, only existing canonical Stage-0-valid episodes; no second representation-only loader. Count exact episodes/raw steps/exterior frames/retained steps/windows from manifest, not the old 6.56M estimate. Retain raw `0,3,6,…`; histories are fixed `[t,t+6,t+12]` = retained `[i,i+2,i+4]` for training and validation. Remove randomized `[1,3,6]` stride probabilities. Store both native 320×180 real RGBs, self-contained from RLDS. JPEG Q95 initially; test Q97 if pilot warrants; real and Lager JPEG settings must match and be fixed after the pilot. |
| 10–11 | Official DA3 refreshed metric any-view `depth-anything/DA3NESTED-GIANT-LARGE-1.1`; synchronize Cam A/B at each retained timestamp, never mix timestamps into a geometry group. Validate K, c2w/w2c, resolution, metric scale/baseline/workspace/cross-view consistency, comparing old statistics where available. Full run requires sound two-view scale validation. Depth is full 180×320 uint16 millimetres: 0 invalid, 1…65535 = 0.001…65.535 m; derive validity, no float32/confidence/separate validity storage. Lossless bitshuffle+Zstd near level 3. |
| 12–16 | Official `cvg/megaflow`, `MegaFlow.from_pretrained("megaflow-flow")`, BF16 where supported. Offline native 180×320 forward `t→t+6` for both cameras, once per eligible retained timestamp; two interleaved phases are allowed. No t→t+3/stride variants/backward cache/downsampling. Int16 at 1/64 px, reserved invalid sentinel, derived validity, no confidence/validity image. Lossless numeric compression; test exact error/range. Real quality audit includes identical/static frames, arm/gripper/object/thin boundaries/occlusion/large motion, warps and temporal consistency; reject bad static-baseline or nonfinite behavior and save panels. |
| 17–22 | Offline verified posed `facebook/lagernvs_dl3dv_2-6_v_256`, 256×256, two raw/full calibrated sources unaffected by later motion crop; preserve camera/canonical validation. Exactly four targets per retained timestamp: `a1∈[.15,.25]`, `a2∈[.25,.35]`, `a3=1−a2`, `a4=1−a1`; deterministic episode/timestep RNG and excluded middle. Interpolation/scene-centered arc where valid + SLERP + small bounded perturbation + safe resampling; default ≤.03 baseline/3°, no >.05/5° without explicit validation. Reject away-facing, too-close, unsupported, invalid/extrapolated poses. Encode sources once and render four targets; verify official numerical equivalence. Same JPEG as real RGB, compact poses/K/alphas/perturbation/coverage metadata, optional cheap geometric bitpacked support. |
| 23–26 | Store each timeline record once, with identifiers/RGB/depth/gap6 flow/Lager×4/camera geometry/support/minimal metadata; windows reference records. Indexed uncompressed TAR/WIDS-style random access, individually compressed payloads, roughly 1–2 GiB shards initially, contiguous episodes where practical; alternate containers require throughput evidence. Deterministic partitioning/poses, checksums/progress/failure logs, `.partial` and atomic validated completion; skip verified completed shards on restart. Three independent GPU workers with disjoint deterministic episode/shard assignments, resident models, stage environments if needed, no unnecessary collectives or competing heavy processes. |
| 27–29 | Exact theoretical size plus representative empirical projection; measure RGB/numeric/support/metadata/index sizes, ratios and temporary/recovery overhead, require comfortable free-space margin. Pilot Q95 vs Q97 PSNR/SSIM/boundaries, CPU/optimized JPEG decode, loader batches/s/GPU idle, numeric decompression GB/s. Choose storage and throughput together. Cached loader returns six real RGBs, six depths, four flows, twelve Lager images plus geometry/support/metadata per history; no RLDS RGB decoder in normal training; persistent workers/pin/prefetch/batch JPEG/handle caches where beneficial, measure throughput. |
| 30–31 | Pad real 320×180 by 70 px top/bottom to 320²; crop size uniform [180,320], resize to 224². Forward-splat `|F01|` from t0 onto t1, combine with t1 `|F12|`, fill/smooth, argmax only feasible centers, image-center fallback for degenerate motion. One shared temporal crop per physical camera; cameras may differ. Transform RGB/depth/flow/geometric validity/K together, scale resized flow vectors. Lager supervision stays uncropped at canonical 256². |
| 32–38 | Keep geometric image validity, SSL tube masking, and batch token padding distinct. 16² patches: zero real-pixel fraction invalid, partial patches valid. Mask .60 of N_valid; keep `max(1, round(.4*N_valid))`, not fixed 78 of 196. Same patch IDs across three times; 50% motion/50% random visible mixture among valid patches, random fallback when static. Variable visible sequences use padding plus explicit SDPA attention validity for encoder/CLS and Gaussian cross-attention. Test dummy-value independence. Inference disables SSL masking, encodes valid patches, returns CLS [B,384], canonical patch_tokens [B,196,384] with documented zero invalid entries, and patch_validity [B,196]. |
| 39–42 | Preserve real RGB L1/SSIM, cross-view InfoNCE, metric and scale-invariant depth, dynamics/flow, visibility, Gaussian regularization and Lager RGB. Remove X-Lens/MEMFOF confidence weighting; depth/flow use numeric and geometric validity and usable correspondences. Average novel loss over batch/time/four targets to avoid 4× weight. Recompute DA3 workspace/decoder/near-far stats, compare old values before adoption. Clean pretrain YAML to explicit cached sources/encodings/strides/target counts/masks; no stale online config aliases. |
| 43–46 | Cached-only validation/W&B: temporal RGB, both flow pairs, aligned motion/crop/padding, teacher/rendered depth/error, RGB reconstruction/error, compact four-view Lager/support/error panels. Full clouds only t0, keep temporal tracking; camera/world frames if existing backend reliable. Representative pilot must pass data/JPEG/DA3/flow/Lager/storage/loader/model/render/backward gates before full run. Tests cover indexing/resume/codecs/poses/transforms/masking/attention/downstream shapes/2048 Gaussians/finite losses/import isolation/checkpoint/single GPU/3-GPU DDP. Profile cached pipeline components, mean/median/p95, samples/images/s/utilization/CPU/disk/VRAM; optimize decode/I/O/worker/prefetch/handles before invasive model changes. |
| 47–50 | One reproducible top-level workflow with dry-run/pilot/resume and all stages. Full run only after gates under persistent long-running goal supervision: durable logs/PID/start/command/GPU/schema/exit, no tail-f or constant polling, healthy inspections at most six-hourly. On failure smallest safe fix, retain shards/resume; same root cause three times or ambiguous/destructive fix means stop/report. Completion requires all episodes/timestamps/shards accounted for, terminal successful workers, no fatal log/unexpected partials, manifests/checksums/random decoding/windows/final checker. Independent final integrity checks all modalities/poses/alphas/keys/index, several hundred distributed samples, actual loader. Then short completed-cache training with W&B/save-resume and 3-GPU DDP, no teachers. |
| 51–52 | Final A–K report must include starting/final SHA/files/submodules/Dynamic3D, removed online stack, exact model provenance, exact counts/final bytes/ratios/margin/codecs, full preprocessing rates/utilization/time, quality visualizations, masking distributions, training profile/bottleneck, and exact manifest/pilot/integrity/full/resume/status/benchmark/single/DDP/resume commands. Do not declare success after code edits alone; preserve all repeated non-negotiable decisions. |

Authorized physical GPUs (no others):

```text
4  GPU-76871c16-ff1d-f7b1-cdf5-fabbaf9df8ce
5  GPU-d09f0338-71b9-d915-3c7f-e99754a3b639
6  GPU-6b33eef2-80c5-547e-6a61-7ab81c14d63b
```

Later user steering and clarifications:

- Repeated `/fast off` requests expressed a preference for thorough work; no
  repository setting was changed on that basis. The supplied continuation
  summary was used to build on prior work rather than restart it.
- The user questioned why Stage-0-valid episodes could fail geometry/pose
  checks. We inspected the actual acceptance code and canonical statistics:
  Stage-0 checks calibration mapping/numerical validity/plausibility, and
  direct-relative consistency only when independently checkable. It does not
  certify DA3 output scale or novel target clearance/support at every timestamp.
  A new preprocessing failure does **not** prove a source episode invalid.
  The earlier suggestion to decide exclusions was premature; preserve eligibility
  and investigate integration/safety assumptions first.
- The user asked what “14.2×” meant. We explained the actual division of depth
  by predicted/calibrated camera separation, its assumptions, and the unresolved
  correctness of both input calibration and predicted scale; details below.
- Current request, repeated after an intentional interruption: create this
  **single** root summary covering all instructions/work/evidence/status/issues/
  paths; inspect before cleanup; remove only proven disposable items; preserve
  all source/data/config/logs/intermediates/checkpoints/analysis/reproduction
  artifacts and uncertain items; verify and record removals/preservations/final
  state. No full preprocessing/training restart was requested by this cleanup.

## Work completed and repository changes

Initial work inspected the existing implementation and retained correct camera
conventions, canonical Lager cameras, temporal ViT/Gaussian structure, renderer,
useful losses, validation and visualization. The refactor replaces online
teacher execution with one cached Stage-0 path.

| Area | Implemented/changed work and important files |
| --- | --- |
| Cached dataset | `dataset/droid/{dataset,sampling,transforms,rlds,__init__}.py` updated; new `preprocessed_manifest.py` computes deterministic Stage-0 timeline/windows/shards and signed contracts; `codecs.py` encodes JPEG/depth/flow/support; `records.py` packs metadata/RGB/Lager; `shards.py` implements indexed TAR/checksums/atomic completion/recoverable interrupted outputs; `integrity.py` validates payloads, counts, poses and loader. `safety.py` guards source/output paths, source fingerprint and quality hold. RLDS decoding remains for offline extraction only. |
| Offline teachers | New `preprocessing/da3/`, `preprocessing/megaflow/`, `preprocessing/stage0/`; retained/updated `preprocessing/lagernvs/{__init__,camera,coverage,official,pose}.py`. Pinned models, independent environments, native modalities, four-target source reuse, deterministic safe poses, staged composition and resumability. `preprocessing/workspace.py` and workspace-stat command updated to cached DA3. |
| Model/masks | `models/splattervae/{backbones,model,decoder}.py` implement geometric validity, variable valid keep counts/tubes, attention padding isolation and downstream scattering; `models/gaussian/motion.py` drops obsolete paths. |
| Training/validation | `models/training/{config,loop,reconstruction,validation,visualization}.py` and `losses/{__init__,depth,flow}.py` updated for cached supervision, derived masks, averaged four-view loss, compact cached panels, t0 clouds/tracking. Training script supports single/DDP/checkpoint resume. |
| Config/docs/dependencies | `config/splattervae/droid/pretrain.yaml`, README, `.gitignore`, `.gitmodules`, `pyproject.toml` updated; isolated-environment constraints/docs added. `.gitmodules` stages DA3/MegaFlow additions and X-Lens/MEMFOF removals; Lager retained. |
| Commands | New `scripts/preprocess_droid_stage0.py`, teacher/pilot/pose audits, codec/loader benchmark, pilot-gate builder, final integrity checker, cached profile, environment creator and `diagnose_droid_da3_scale.py`. Updated train/smoke/loader/workspace scripts. |
| Tests | Added `test_stage0_codecs.py`, `test_offline_teachers.py`, `test_da3_scale_diagnostic.py`; updated dataset/geometry/loss/model/safety/training/transforms/workspace tests. Independent NumPy depth unprojection checks clearance from actual cached depth; corruption tests verify metadata cannot conceal forged poses/clearance. Signed audit checker now verifies shard ID as well as signature/counts. |

Earlier implementation removals (already present in the worktree before this
cleanup; retained Git history permits recovery):

- `models/training/online_preprocessing.py` and OnlineTeacherPipeline.
- `preprocessing/xlens/{__init__,official}.py`,
  `preprocessing/memfof/{__init__,official}.py`; active `third_party/XLens` and
  `third_party/MEMFOF` gitlinks/config/imports, replaced by DA3/MegaFlow.
- Obsolete `scripts/analyze_droid_checkpoints.py`,
  `profile_droid_end_to_end.py`, `smoke_test_real_droid_gpu.py`,
  `validate_lagernvs_droid.py`, `validate_lagernvs_target_poses.py`.
- Obsolete `tests/test_online_preprocessing.py` and
  `tests/test_teacher_preprocessing.py`.

Historical X-Lens/MEMFOF **outputs/checkpoints/profiling evidence** and old
submodule Git metadata remain intentionally preserved. They are not active
training integrations. No uncertain external checkpoint was deleted.

## Models, environments, storage and exact manifest

| Offline model | Repository commit | Checkpoint revision |
| --- | --- | --- |
| ByteDance-Seed/Depth-Anything-3; `depth-anything/DA3NESTED-GIANT-LARGE-1.1` | `3d835ec1a5802d64a8b8b15f817a1ab54809bfe4` | `b2359bdf726fb44ef62acca04d629dcf158053e7` |
| cvg/megaflow; `megaflow-flow` from Kristen-Z/MegaFlow | `ee5b61813db0a76ac0db9034899aade72a0d230c` | `b4c5c33800b8fa88e047d2eb70ae74b0feca606d` |
| facebookresearch/lagernvs; `facebook/lagernvs_dl3dv_2-6_v_256` | `665f727aba8298a04ff4c040fd6279a32ef23017` | `4026552953a72c5fb037501564dc673dd73c574e` |

Training `.venv` uses Python 3.10. Isolated `.preprocessing-envs/{da3,megaflow,
lagernvs}` use Python 3.12, PyTorch 2.8.0+cu128/torchvision 0.23.0+cu128;
recorded DA3 runtime includes NumPy 1.26.4, Pillow 12.0.0, numcodecs 0.13.1,
HF Hub 0.36.0, TensorFlow 2.20.0/TFDS 4.9.9. Stage/inventory metadata is the
authoritative resolved package record; checked-in constraints and adapter SHAs
define reconstruction. TFDS is isolated from CUDA during offline source decode.

| Manifest quantity | Exact full requirement | Representative pilot |
| --- | ---: | ---: |
| Eligible episodes | 33,195 (32,905 train / 290 validation) | 12 |
| Raw timesteps | 9,502,985 | 3,587 |
| Raw exterior frames | 19,005,970 | 7,174 |
| Retained timesteps | 3,178,786 | 1,201 |
| Training windows | 3,046,082 | 1,153 |
| Real JPEGs / DA3 depths, each | 6,357,572 | 2,402 |
| Forward gap6 flow fields | 6,224,806 | 2,354 |
| Lager JPEGs | 12,715,144 | 4,804 |
| Planned final shards | 791 | 6 |

Canonical eligibility:
`/ws/data/ws/droid_splattervae/manifests/canonical-full/calibration.jsonl.gz`,
SHA256 `212e409b561da4c897465b84181de05b87371269f942a9c20453a17378add2f3`.
Derived schema v2 signature:
`760f3465d61a7ef5aefaa3fd60313f7e3b78dcaf429dd9b0c8a30c261d1cf10a`.
Pipeline signature:
`dbe271b09a64d8613bd534e71e72ffb6f3b3b8f4503f29df175b52d94b13b80d`.

Stored real/Lager RGB: JPEG **Q95, 4:4:4**, Pillow, nonprogressive,
`optimize=false`. Depth uint16 mm and native flow int16 at 1/64 px use
Blosc bitshuffle+Zstd level 3. Flow sentinel `−32768`, usable numeric range
`[−511.984375,511.984375] px`; nearest rounding gives ≤1/128 px error for
representable finite values. No teacher confidence/extra validity images.
Support is geometric/bitpacked. TAR is uncompressed with byte-offset sidecars;
4096 retained timestamps per full shard, episodes kept contiguous. Lager
metadata evolved through v2/v3/v4; old verified strict v2 output remains readable,
while newer records store exceptional-tier thresholds/flags/source clearance.

## Evidence and results so far

- Real pilot generation, codec/model/loader audits and cached training profile
  ran before the original full launch. `pilot/reports/pilot_gate.json` records
  all eight historical gates true and visual review. Later scale/safety evidence
  supersedes its launch approval; the file is kept as history.
- Historical 93-test run passed. After independent geometry and quality-hold
  work, 27 relevant regression tests passed (overlapping subsets, not additive).
  Ruff, syntax compilation and `git diff --check` passed at those revisions.
- Pilot full integrity: 1,201 timestamps/5,981 indexed entries/six shards,
  all JPEG/numeric payloads decoded; 442 timestamps received independent depth
  geometry verification; 64 actual loader windows (32 train + 32 validation).
  Data/flow validity fractions were 1.0; four targets everywhere, legacy strict
  poses only; no unexpected partials in that scan.
- Pilot MegaFlow audit covered 96 fields, finite outputs, up to 54.58 px motion,
  static/zero-baseline warp comparisons and saved robot/object/flow panels.
  Lager source reconstructor ran once for four targets, with **zero measured
  max/mean difference** against four official calls in the audited case.
  Original DA3 pilot had plausible sampled depths/cross-view statistics, but its
  posed-baseline checks are now known to provide incomplete scale evidence.
- Small teacher-audit rates: DA3 3.79 images/s, MegaFlow 4.64 fields/s, Lager
  16.63 targets/s. These are pilot measurements, not sustained full-job rates.
- Q95 vs Q97 measured on 192 images; retained Q95. Codec-reference means:
  real Q95 ≈26,881 B, PSNR 50.28 dB/SSIM .99836; Q97 ≈33,267 B.
  Lager Q95 ≈19,946 B, 45.37 dB/.98964; Q97 ≈26,923 B, 46.64 dB/.99137.
  Full pilot payload averages differ (≈27,454 real and 20,105 Lager B/image).
  Original pixels/codec panels and exact rows are preserved.
- JPEG decode microbenchmark: Pillow 1,979, torchvision CPU 3,072,
  CUDA/nvJPEG 5,691 images/s. Numeric depth records: 230,400 raw / 88,909
  compressed B, 2.59×, 0.973 uncompressed GB/s; flow: 460,800 / 62,351 B,
  7.39×, 1.348 GB/s. Loader benchmark (100 batches, batch4/workers4):
  7.83 batches/s, 31.31 samples/s, 563.64 decoded images/s.
- Cached training profile (20 measured batch1 iterations, four workers):
  10.57 samples/s/190.31 images/s; all losses/gradients finite, loaded teacher
  modules `[]`; iteration mean/median/p95 92.63/78.55/176.65 ms, model-forward
  mean 20.30 ms, backward 32.66 ms, optimizer 6.34 ms. Observed loader wait
  .54 ms, transfer 1.41 ms; worker JPEG 12.53 ms, motion crop/transforms
  99.91 ms/sample. Worker timings overlap iteration timings and cannot be summed.
  Allocated/reserved peak .832/.865 GiB; sampled GPU utilization mean 17.3%,
  process disk reads 7.96 MB/s. Small-pilot data favors CPU transforms as a
  bottleneck; it does not settle full-dataset I/O or larger-batch throughput.
- Mask profile: N_valid min/mean/max 126/139.65/168; N_keep 50/55.825/67;
  actual masking mean .60035 (range .59740–.60317), mean 19.25 partial patches
  retained per view. Tests isolate batch padding from CLS/patch/decoder outputs.
- Cached-pilot single-GPU/DDP/W&B artifacts and checkpoints exist in
  `derived/pilot_train_{single,ddp,wandb}/`. They provide historical pilot evidence;
  the required post-full-dataset training validation is still outstanding.

Storage from exact counts and pilot:

| Component | Projected final bytes (not actual completed totals) |
| --- | ---: |
| Real JPEG | 174,543,690,502 |
| DA3 depth | 282,622,962,386 |
| MegaFlow | 194,059,948,038 |
| Lager JPEG | 255,632,808,894 |
| Lager metadata/support/framing | 107,795,812,046 |
| Timeline metadata / other framing | 209,249,345 / 57,218,148 |
| TAR/index/sidecar overhead | 14,850,774,716 |
| Total final / peak staging | 1,029,772,464,075 / 1,014,712,440,014 |
| Safe requirement incl. recovery + 20% | 2,462,755,287,867 |
| Theoretical uncompressed | 5,872,592,500,432 |

Available filesystem space at this inspection: **3,941,858,328,576 bytes**;
arithmetic margin over the historical safe requirement ≈1.479 TB. The earlier
September inspection had ≈6.798 TB free. Current capacity must be remeasured,
especially if correction needs parallel retained artifacts; do not reuse old
free-space numbers or silently change mount. No final full-run duration/rates/
storage total exists because required stages are incomplete.

## Processing failure, Stage-0 clarification, and assumptions

Current completion markers: **RGB 791/791; DA3 791/791; MegaFlow 791/791;
LagerNVS 15/791; final composition 0/791**. A completion marker certifies a
stored shard, not correct metric teacher output. Completed Lager IDs:
`0,1,2,3,4,5,6,7,8,9,10,11,13,14,17` (v2 strict poses; previously checksum
verified). Later continuation stopped after repeated pose-safety failures.

Stage-0 implementation (`dataset/droid/calibration.py`) accepts correct mapping,
finite K/poses, rigid rotations, ≤5 m translations, baseline [.05,3] m, and
direct-relative agreement ≤20°/.35 m when both direct poses exist. Convention
direction was compared on 23,055 dual-direct episodes (median forward error
2.92°/.047 m). Released `quality_metric` is not an eligibility filter.
Both failing episodes have one direct and one derived pose: independent
direct-relative error fields are `null`, not measured zero. Stage-0 does not
test teacher depth or generated target clearance/support.

Revision-5 pose audit failed at:

| Shard/key | Episode | Raw t | Evidence |
| --- | --- | ---: | --- |
| 248 / `0000999926` | `PennPAL+acda9df3+2023-06-18-19h-59m-31s` | 279 | No safe symmetric alpha pair in [.25,.35]; release num_matches quality 1 |
| 388 / `0001564077` | `IPRL+7790ec0a+2023-05-05-17h-35m-13s` | 3 | Same failure; release num_matches quality 170 |

Only 366/776 unfinished Lager shards passed the current audit (1,473,012
timestamps). Audit contract signature:
`fea255f28d1d7fe61f1763c71cd06f05a09ff8c22380f969262ab4ddea569694`.
Workers 1/2 failed; remaining worker 0 was stopped with SIGINT. Repeated earlier
attempts broadened exceptional clearance policies; they have **not** resolved
quality. Ordinary tier remains .03 baseline/3°, .60 coverage/.08 m clearance;
exceptional tier allows .05 baseline, .40 coverage and a robust source-relative
clearance floor `clip(.8*((1−alpha)*clearance_A+alpha*clearance_B),.02,.05)` m.
That new exceptional policy lacks representative image-quality certification;
the old strict pilot is insufficient. Do not relax further or restart retries.

DA3 diagnostic traced the posed path: nested metric branch first scales depth
and translations; API then divides depth by predicted/calibrated baseline and
replaces output poses with supplied poses. Final output-pose/baseline equality
is consequently not independent depth-scale proof. Nested metric scale cancels
in the last baseline ratio. Per-timestamp scale diagnostics were **not saved**
with production depth, so a full retrospective correction is not established.

Bounded GPU-4 replay on cached Q95 JPEGs (not bit-exact raw RLDS inference):

| Pair | Calibrated baseline | Depth divisor | Cam-B median before → after | Cached Cam-B median |
| --- | ---: | ---: | ---: | ---: |
| PennPAL failure | .5990 m | 3.0056 | .5628 → .1873 m | .182 m |
| IPRL failure | .1182 m | 14.2077 | .8097 → .0570 m | .056 m |
| ILIAD pilot control | .9078 m | 1.0579 | .8658 → .8184 m | .821 m |

For IPRL, predicted baseline ≈1.680 m divided by .1182 m gives 14.2, and every
depth in both synchronized views is divided by that factor. This is geometry
rescaling, not unit conversion. It is justified only if baseline calibration is
correct and predicted baseline/depth share a scale error. Neither condition has
been independently certified for this pair. Room/robot appearance makes 5.7 cm
median suspect; it could distort downstream pose checks. SIFT triangulation
found zero usable calibrated correspondences in all three pairs, so it is
inconclusive. Before-alignment depths are not proven ground truth either.

The initial “how should invalid episodes be handled?” question over-attributed
the failure to episodes. Correct continuation is to preserve Stage-0 eligibility
and investigate DA3 alignment/calibration/safety-test assumptions before deciding
on exclusions or replacement calibration. No eligibility or scale correction
was applied. The historical incident document is preserved verbatim for evidence;
this summary incorporates the subsequent clarification.

## Important locations and continuation commands

| Location | Purpose; preservation policy |
| --- | --- |
| `config/splattervae/droid/pretrain.yaml` | Active cached training contract/model/loss/loader settings. Workspace values adopted from the original DA3 pilot (center [.60235,.106555,.071765], spread .391264, parent displacement .257452, child radius .045054, near/far .0775/3.9768); revisit after scale resolution and full statistics. The config's pilot-validated label does not certify the full cache. |
| `dataset/droid/`, `models/`, `preprocessing/`, `scripts/`, `tests/` | Source implementation, offline wrappers/workflow, commands, regression tests; all preserved. `dataset/droid/calibration.py` and canonical calibration were not changed by the offline refactor. |
| `docs/droid-stage0-quality-incident-2026-09-07.md` | Detailed original failure/scale handoff, evidence and diagnostic command; preserved. |
| `.venv/`, `.preprocessing-envs/`, `.uv-cache/` | Training/isolated teacher runtimes and package cache; preserve binaries/symlinks/links/locks and reproducibility, no cleanup inside. Constraints/docs in `preprocessing/environments/`; creator in `scripts/create_droid_preprocessing_envs.sh`. |
| `third_party/{Depth-Anything-3,MegaFlow,LagerNVS}/`, `.git/` | Pinned source and all Git history/index/module repositories, including old XLens/MEMFOF metadata; preserved. |
| `checkpoints/` (~4.2 GiB), `derived/` (~2.5 GiB), `outputs/` (~3.9 GiB) | Model downloads/locks, cached-pilot checkpoints/W&B/panels, earlier online teacher audits/profiling/checkpoint analysis/training/visualization; preserve all despite legacy names. Sizes are inspection snapshots. |
| `/home/ws/data/droid` | Original read-only RLDS; absolutely no writes/deletions. Metadata inventory remains 2,051 files/1,866,281,754,039 bytes, SHA256 `6f90d6f97e73d36e0243d89fe6e26621c05670c299294ee3759b0ed370d99ad3`; metadata proof, not a full content hash. |
| `/ws/data/ws/droid_splattervae/calibration/official_posthoc/` and `manifests/canonical-full/` | Released calibration/provenance and original eligibility/statistics/splits; outside cleanup scope, untouched. |
| `/home/ws/data/droid_stage0_preprocessed/manifest.json`, `manifests/` | Signed schema/counts/episode/window/shard plans; authority for deterministic resume. |
| Derived-root `staging/{rgb,da3,megaflow,lagernvs}/shards/`, `shards/` | Intermediate TAR/index/completion markers and eventual final composition; preserve all complete and unfinished/recovery data. |
| Derived-root `pilot/` | Six-shard complete pilot, benchmark/quality/teacher/workspace/profile/integrity/gate reports, codec/teacher/quality panels and original codec references; preserve. |
| Derived-root `metadata/`, `logs/`, `reports/`, `recovery/` | Model cache/exact stage environment and signatures, persistent run/progress/error/telemetry evidence, current hold/scale report/pose reports, recoverable historical/partial outputs. None cleaned. |

Run commands from `/home/ws/ws/droid_training`. The full workflow and all
training/benchmark/rebuild/resume examples remain in README; launches are gated.
Do not rerun mutating commands merely to inspect state or overwrite old reports.

```bash
# Read-only current state; -B avoids rebuilding disposable bytecode caches.
.venv/bin/python -B -m scripts.preprocess_droid_stage0 status \
  --root /home/ws/data/droid_stage0_preprocessed

# Bounded diagnostic only, if further investigation is authorized/appropriate;
# give each new report a unique path to preserve earlier evidence.
CUDA_VISIBLE_DEVICES=GPU-76871c16-ff1d-f7b1-cdf5-fabbaf9df8ce \
HF_HUB_OFFLINE=1 OMP_NUM_THREADS=4 \
.preprocessing-envs/da3/bin/python -B -m scripts.diagnose_droid_da3_scale \
  --sample 248:999926 --sample 388:1564077 --sample 329:1327791 \
  --report /home/ws/data/droid_stage0_preprocessed/reports/da3-scale-recheck-UNIQUE.json

# Future full/resume command, currently rejected by quality hold/incomplete audit.
# Only after independent metric/safety/pilot/capacity gates pass, use a persistent
# logged launch under the requested long-running goal; omit --dry-run to run.
CUDA_VISIBLE_DEVICES=GPU-76871c16-ff1d-f7b1-cdf5-fabbaf9df8ce,GPU-d09f0338-71b9-d915-3c7f-e99754a3b639,GPU-6b33eef2-80c5-547e-6a61-7ab81c14d63b \
.venv/bin/python -B -m scripts.preprocess_droid_stage0 run \
  --root /home/ws/data/droid_stage0_preprocessed --droid-root /home/ws/data/droid \
  --workers 3 --stages lagernvs compose \
  --pilot-gate /home/ws/data/droid_stage0_preprocessed/pilot/reports/pilot_gate.json \
  --require-lagernvs-pose-audit --integrity full --dry-run
```

Recommended sequence to finish the original goal:

1. Diagnose two-view metric alignment with independent scene/robot geometry and
   calibration evidence; audit predicted relative poses/K and support/clearance
   logic. Distinguish integration error from uncertain calibration; include the
   failures and varied baselines/motion in the new representative pilot.
2. Make only evidence-supported corrections, retain old data, record per-item
   transient alignment diagnostics/provenance, invalidate affected audit/gate
   signatures, and revalidate metric depth/target appearance. A narrower episode
   set or different calibration/teacher would be an explicit scope decision.
3. Validate current exceptional pose policy or replace it with a justified safe
   implementation; stop repeated same-cause retries. Recompute storage overhead,
   pilot gate and full remaining-shard audit before archiving the quality hold.
4. Finish Lager/composition with persistent three-GPU supervision, six-hour
   healthy inspections and verified resume. Preserve all valid outputs.
5. Final whole-dataset integrity/accounting/checksums/≥several hundred distributed
   decodes and actual windows; recompute/adopt full DA3 workspace stats; loader/
   model profile/optimization; completed-cache single-GPU save/resume/W&B and
   three-GPU DDP smoke; then the complete A–K report with actual totals/timing.

## Cleanup record and verification

This cleanup is scoped to the repository root above; external dataset, calibration,
model and analysis locations are preserved. No source/config/log/checkpoint/
analysis artifact is a cleanup target. Inspection found no disposable temporary
`.tmp/.bak/.partial/.orig/*~/swap` files outside protected dependency directories.
Unfinished dataset artifacts are outside this cleanup scope and remain intact.

Removed after inspection (all paths relative to this workspace):

- `.pytest_cache/`: five regenerable bookkeeping files, 12,280 bytes; stale
  `lastfailed` referenced removed
  `test_fixed_pipeline_requires_native_two_iteration_memfof` (no longer in tests).
  Test reports/logs are preserved separately.
- `.ruff_cache/`: 125 untracked regenerable cache/tag files, 80,562 bytes.
- Fourteen project `__pycache__/` directories, containing 129 untracked .pyc
  files only (valid CPython cache headers, no links or source). Exact paths:
  `dataset/{__pycache__,droid/__pycache__}`;
  `models/{__pycache__,gaussian/__pycache__,splattervae/__pycache__,training/__pycache__,training/losses/__pycache__}`;
  `preprocessing/{__pycache__,da3/__pycache__,lagernvs/__pycache__,megaflow/__pycache__,stage0/__pycache__}`;
  `scripts/__pycache__`; `tests/__pycache__`.
- Empty top-level `utils/` and `visualize/`: inspected as empty, nonsymlink,
  untracked directories with no active project references; removed with rmdir.

Intentionally preserved: all listed source/config/tests/docs/Git/submodules;
all datasets/shards/partials/recovery/manifests/metadata/model weights/locks;
all checkpoints, logs/W&B/history/panels/profiles/codec references/analysis;
all installed environments/package caches; tracked zero-byte
`dataset/__init__.py` and `models/__init__.py` (package sources), and zero-byte
model-cache lock (uncertain cache state). Empty baseline/agent/metaworld/flow/
other preprocessing directory scaffolding is also preserved because its future
or unrelated role is uncertain. No deletion outside the repository.

Pre-cleanup verification baseline (excluding disposable caches and this summary):
456 regular source/artifact files, 6,814,119,752 bytes, combined relative-path/
content SHA256 `2a0dda6b42351c23003945fcda912e05c62d0fc5487ea77cc3be49d1638d07d2`.
Dependency/checkpoint download metadata fingerprint:
`728324f6874eb0617d2cf24b784c96ef2e6e7e0a3a64cdfa29a240d3bb3555d6`.
Existing staged/unstaged binary-diff SHA256:
`f548923c75749ca065a2bba5e534fd3c6f897e91c6e5a5458dc722393fbc7606` /
`28d0dfa47632a0586badb7aa51c4f24f3425d39b1f96b2ed3cee197375c679f8`.

Final cleanup result: **259 disposable files / 1,409,603 bytes of file content**
removed across 16 cache trees, plus the two empty top-level directories listed
above. Cache directories (including their now-empty internal directories) were
removed with a depth-first, exact-path file/empty-directory deletion; no broad
workspace deletion or Git cleanup command was used. Cache files regenerate on
normal Python/pytest/Ruff use; empty directories can be recreated with mkdir.
No source or result payload was deleted. This summary is the **only new file**
created by this cleanup request.

Verification completed:

- All 18 explicitly selected top-level cleanup targets are absent, and no
  project Python/pytest/Ruff cache directory remains outside protected dependency
  areas. Verification used `python -B` to avoid recreating bytecode.
- All **456 protected regular source/artifact files (6,814,119,752 bytes)** match
  the pre-cleanup content fingerprint. Dependency/checkpoint metadata and the
  existing staged/unstaged binary-diff fingerprints above also match exactly.
- All 80 project Python files passed syntax compilation in memory; YAML loaded
  through the actual training configuration validator and constructed TrainConfig.
  Importing that training entry point loaded no foundation-model modules.
- The full manifest's schema/pipeline signatures and 791-shard plan validate;
  the source metadata fingerprint is unchanged; quality hold still rejects launch.
  Installed environments, protected artifact directories and zero-byte package
  markers remain present. `git diff --check` passed. No full dataset scan or new
  training/GPU test was launched for this documentation/cache-only change.
- Branch/HEAD remain **DROID / 4d44e7e**. Existing changes retain their original
  staged/unstaged state; `WORKSPACE_SUMMARY.md` is new and untracked. No commit
  was made. Preprocessing remains stopped, full dataset/training acceptance
  remains incomplete, and the documented quality issue remains unresolved.

## GitHub publication preparation — 2026-10-06

The user subsequently instructed: “push the recent necessary code to the github.”
That authorizes committing/pushing the offline refactor and its reproducibility/
handoff documents to `sunho001215/splatter_vae`, branch `DROID`. The cleanup
snapshot and fingerprints above describe the state before this publication.
Use `git log -1` and `git status` for the current committed state.

- Reviewed all pending paths; source/config/tests/scripts/environment constraints,
  README, this summary, incident documentation and submodule references belong
  in the commit. Datasets, stage outputs, checkpoints, model downloads, logs,
  analyses, caches and installed environments remain local/ignored.
- Fetched `origin/DROID`: it matched the starting `4d44e7e`, with no divergent
  commits. Publication uses a normal fast-forward push, without force.
- Full current CPU regression suite: **97 passed in 8.64 s**; no GPU was exposed.
  Ruff (`F,B,I,RUF022`, excluding third-party sources), staged/unstaged whitespace
  checks, and clean pinned submodule worktree checks passed. Tests used disabled
  bytecode/pytest-cache output so cleanup targets stayed absent.
- Staging newly added source exposed two trailing-whitespace lines in
  `scripts/profile_droid_cached.py`; the expression's whitespace was cleaned
  without changing its behavior, and staged whitespace checks then passed.
- Publication preserves the unresolved DA3/Lager geometry issue and the quality
  hold. Passing unit tests does not approve the full dataset or complete the
  original preprocessing/training goal. No preprocessing or training was restarted.
