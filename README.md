# SplatterVAE DROID Stage-0 pretraining

Full preprocessing is currently **on quality hold**. RGB, DA3, and MegaFlow
have 791 completion markers each, but DA3 metric quality is under investigation;
LagerNVS has 15 completed shards and final composition has not started.
See [the 2026-09-07 incident report](docs/droid-stage0-quality-incident-2026-09-07.md).
The earlier pilot approval does not resolve this incident. The launcher and
direct stage workers fail closed while
`reports/preprocessing_quality_hold.json` exists under the dataset root.
Only archive that marker after an explicitly reviewed and validated resolution;
do not bypass it or relax the pose thresholds to resume.

The `DROID` branch trains the temporal ViT-S and Dynamic3D Gaussian decoder
only from an offline, self-contained Stage-0 cache. Normal training performs
indexed shard reads, motion-aware transforms, model forward/rendering, losses,
and backward; it does not import or execute a foundation model.

The original DROID release is strictly read-only:

```text
source RLDS:       /home/ws/data/droid
derived Stage-0:   /home/ws/data/droid_stage0_preprocessed
code:              /home/ws/ws/droid_training
```

Every output entry point rejects a destination inside the source tree.
Dynamic3D remains the rendering/visualization base and is not modified by the
offline data workflow.

## Cached pipeline contract

Only episodes marked valid by the canonical Stage-0 calibration manifest are
eligible. The exact full manifest currently contains:

```text
eligible episodes       33,195
raw timesteps         9,502,985
raw exterior frames  19,005,970
retained timesteps    3,178,786
training windows      3,046,082
deterministic shards        791
```

One record is stored for each raw timestamp `0,3,6,...`; temporal windows are
references into that timeline, never duplicated payloads. A training history
uses retained indices `[i,i+2,i+4]`, corresponding to raw timestamps
`[t,t+6,t+12]`.

Each retained record contains:

- two native `320x180` real-camera JPEGs;
- two `uint16` millimeter DA3 depth maps (`0` is invalid);
- two native `180x320` forward MegaFlow fields for `t -> t+6`, when present;
- four canonical `256x256` LagerNVS JPEGs and target-camera metadata;
- real-camera intrinsics and `c2w`/`w2c`, compact support masks, and record
  metadata.

Depth and flow arrays use bitshuffle + Zstd level 3 in individually indexed
payloads. Flow is signed int16 fixed point at `1/64` pixel and reserves
`-32768` for invalid values. RGB modalities both use deterministic JPEG Q95,
4:4:4 chroma, non-progressive encoding. TAR streams are uncompressed and have
sidecar random-access indexes and checksums.

New Lager records also store the selected pose-safety tier, its coverage and
clearance thresholds, and whether the bounded translation limit was escalated.
The small set of already-complete schema-2 shards predates exceptional tiers;
the reader deterministically interprets those records as ordinary strict-tier
poses.

## Offline teachers

The three teachers run only in preprocessing and have isolated Python 3.12,
PyTorch 2.8/CUDA 12.8 environments under `.preprocessing-envs`:

```text
DA3 repository       ByteDance-Seed/Depth-Anything-3
DA3 commit           3d835ec1a5802d64a8b8b15f817a1ab54809bfe4
DA3 model            depth-anything/DA3NESTED-GIANT-LARGE-1.1
DA3 revision         b2359bdf726fb44ef62acca04d629dcf158053e7

MegaFlow repository  cvg/megaflow
MegaFlow commit      ee5b61813db0a76ac0db9034899aade72a0d230c
MegaFlow model       megaflow-flow (Kristen-Z/MegaFlow)
MegaFlow revision    b4c5c33800b8fa88e047d2eb70ae74b0feca606d

LagerNVS repository  facebookresearch/lagernvs
LagerNVS commit      665f727aba8298a04ff4c040fd6279a32ef23017
LagerNVS model       facebook/lagernvs_dl3dv_2-6_v_256
LagerNVS revision    4026552953a72c5fb037501564dc673dd73c574e
```

DA3 receives the synchronized exterior-camera pair at one timestamp and uses
the calibrated posed two-view metric path. MegaFlow stores only forward
`t -> t+6` flow and reconstructs all fields through the two retained phases.
LagerNVS encodes the two source images once and renders four targets from the
shared reconstruction. Target alphas are symmetric: two lie in
`[0.15,0.35]`, two in `[0.65,0.85]`, and the center band is excluded. Poses
use SLERP/scene-centered interpolation, ordinary perturbations bounded by
`0.03` baseline and `3` degrees, DA3 geometry clearance, coverage checks, and
deterministic resampling. If all ordinary candidates are exhausted, an
explicitly flagged bounded tier permits at most `0.05` baseline translation;
the final flagged safety tier permits no less than `0.40` source coverage and
uses a source-calibrated robust-clearance floor. That floor is 80% of the
alpha-interpolated clearance of the two known-physical source cameras, clipped
to `[0.02,0.05] m`; this avoids demanding more clearance than the real camera
rig itself has while retaining a hard 2 cm collision floor. Rotation remains
bounded by `3` degrees in every tier. The exact tier contract and signature are stored in
`metadata/lagernvs-pose-safety-contract.json`.

Worker provenance files record the complete package inventory, source commit,
model/checkpoint revision, device, and encoding contract. See
`preprocessing/environments/README.md` for environment details.

## Training transforms and masks

For each physical camera, cached `F01` magnitude is forward-splatted onto the
middle-frame grid and combined with `F12`. The crop center is the maximum of
the smoothed aggregate inside the feasible region; zero motion uses the fixed
image center. A square size is sampled uniformly from `[180,320]` after
adding 70 pixels of vertical padding on both sides. The same crop is applied to
all three real RGB/depth/flow grids and intrinsics, then resized to `224x224`;
flow vectors are scaled with the resize. Lager targets remain in their
canonical `256x256` camera domain.

Vertical padding is not content. Patch validity is derived from real-pixel
coverage: zero-coverage patches are invalid, while partially covered boundary
patches remain valid. Sixty percent tube masking is computed relative to each
sample's valid patch count. Half of the visible patches are motion-prioritized
and half random, all from valid positions. Variable visible-token counts are
padded only at batch collation, and explicit attention masks prevent dummy
tokens from affecting CLS, valid patch tokens, or Gaussian cross-attention.

Inference disables SSL masking, returns `cls_token [B,384]`, scatters current
tokens into `patch_tokens [B,196,384]`, and returns
`patch_validity [B,196]`. The decoder produces `256 x 8 = 2,048` Gaussians.

## Setup

```bash
cd /home/ws/ws/droid_training
git submodule update --init --recursive
uv venv --python 3.10 .venv
uv sync --all-extras
export PATH="$PWD/.venv/bin:$PATH"
export PYTHONPATH="$PWD"

# Isolated offline-teacher environments.
./scripts/create_droid_preprocessing_envs.sh
```

All CUDA commands must expose only these devices:

```bash
GPU4=GPU-76871c16-ff1d-f7b1-cdf5-fabbaf9df8ce
GPU5=GPU-d09f0338-71b9-d915-3c7f-e99754a3b639
GPU6=GPU-6b33eef2-80c5-547e-6a61-7ab81c14d63b
```

## Manifest, pilot, and quality gates

Rebuild the exact full manifest without reading image payloads:

```bash
.venv/bin/python -m scripts.preprocess_droid_stage0 manifest \
  --root /home/ws/data/droid_stage0_preprocessed \
  --droid-root /home/ws/data/droid --mode full \
  --target-retained-per-shard 4096 \
  --jpeg-quality 95 --jpeg-subsampling 4:4:4 --zstd-level 3
```

Create and run a representative pilot (safe to rerun):

```bash
.venv/bin/python -m scripts.preprocess_droid_stage0 manifest \
  --root /home/ws/data/droid_stage0_preprocessed/pilot \
  --droid-root /home/ws/data/droid --mode pilot --pilot-episodes 12 \
  --target-retained-per-shard 256 \
  --jpeg-quality 95 --jpeg-subsampling 4:4:4 --zstd-level 3

CUDA_VISIBLE_DEVICES="$GPU4,$GPU5,$GPU6" \
  .venv/bin/python -m scripts.preprocess_droid_stage0 run \
  --root /home/ws/data/droid_stage0_preprocessed/pilot \
  --droid-root /home/ws/data/droid --workers 3 \
  --allow-missing-pilot-gate --integrity full
```

Audit quality, codecs, capacity, loader throughput, workspace scale, and the
real cached training step:

```bash
.venv/bin/python -m scripts.audit_droid_stage0_pilot \
  --root /home/ws/data/droid_stage0_preprocessed/pilot

CUDA_VISIBLE_DEVICES="$GPU4" .venv/bin/python \
  -m scripts.benchmark_droid_stage0 \
  --pilot-root /home/ws/data/droid_stage0_preprocessed/pilot \
  --full-root /home/ws/data/droid_stage0_preprocessed

.venv/bin/python -m scripts.compute_droid_workspace_stats \
  --config config/splattervae/droid/pretrain.yaml \
  --preprocessed-root /home/ws/data/droid_stage0_preprocessed/pilot \
  --maximum-episodes 12 --frames-per-episode 8 \
  --output /home/ws/data/droid_stage0_preprocessed/pilot/reports/workspace_stats_da3.json

CUDA_VISIBLE_DEVICES="$GPU4" .venv/bin/python \
  -m scripts.profile_droid_cached \
  --config config/splattervae/droid/pretrain.yaml \
  --preprocessed-root /home/ws/data/droid_stage0_preprocessed/pilot \
  --iterations 20 --warmup 3 --batch-size 1 --workers 4

# Run only after manually inspecting every codec/DA3/flow/Lager panel.
.venv/bin/python -m scripts.build_droid_pilot_gate \
  --pilot-root /home/ws/data/droid_stage0_preprocessed/pilot \
  --jpeg-quality 95 --visual-quality-reviewed
```

The current representative pilot has 12 episodes, 1,201 retained timestamps,
1,153 windows, and six shards. Full payload integrity and all automatic teacher
quality gates pass. Its empirical projection is approximately 1.030 TB final,
1.015 TB peak staging, and 2.463 TB required after recovery and 20% safety
overhead. Capacity is always recalculated from current free space before a full
launch.

## Full resumable preprocessing

After any pose-sampler recovery change, audit every remaining shard from
cached DA3 geometry before launching LagerNVS. Run one process per authorized
GPU with worker IDs 0, 1, and 2 (shown here for worker 0; substitute the other
UUID/ID pairs):

```bash
CUDA_VISIBLE_DEVICES="$GPU4" PYTHONPATH=. .venv/bin/python \
  scripts/audit_lagernvs_pose_safety.py \
  --root /home/ws/data/droid_stage0_preprocessed \
  --worker-id 0 --worker-count 3 --device cuda:0 \
  --skip-completed-lager
```

Every passing shard report is bound to both the dataset signature and the full
pose-sampler contract signature, so stale reports are recomputed on resume.

Preview the exact worker/stage commands:

```bash
CUDA_VISIBLE_DEVICES="$GPU4,$GPU5,$GPU6" \
  .venv/bin/python -m scripts.preprocess_droid_stage0 run \
  --root /home/ws/data/droid_stage0_preprocessed \
  --droid-root /home/ws/data/droid --workers 3 \
  --stages lagernvs compose \
  --pilot-gate /home/ws/data/droid_stage0_preprocessed/pilot/reports/pilot_gate.json \
  --require-lagernvs-pose-audit \
  --integrity full --dry-run
```

Launch, or resume, with the same command after removing `--dry-run`.
Shard assignment is deterministic (`shard_id mod 3`), completed shards are
checksum-verified and skipped, incomplete output uses `*.partial`, and each
worker writes a disjoint shard set. Inspect state without modifying it:

```bash
.venv/bin/python -m scripts.preprocess_droid_stage0 status \
  --root /home/ws/data/droid_stage0_preprocessed \
  --droid-root /home/ws/data/droid
```

The multi-day command must be launched persistently with stdout/stderr, PID,
start time, GPU assignment, schema signature, command, and final exit status
recorded. The orchestrator also writes one-minute, per-stage utilization and
memory telemetry for exactly the three authorized GPU UUIDs. A healthy process
is inspected at most once every six hours.

## Integrity and loader checks

The integrity scan checks recorded translation/rotation against the actual pose
matrices. On its distributed random/evenly spaced sample, it independently
reconstructs source and target clearances from cached DA3 depth and calibrated
cameras using NumPy. The report records how many timestamps received this
geometry check; metadata consistency alone is not treated as geometry evidence.

```bash
.venv/bin/python -m scripts.check_droid_stage0_integrity \
  --root /home/ws/data/droid_stage0_preprocessed \
  --random-samples 512 --loader-windows 512 \
  --full-payload-scan --decode-all-jpegs --finalize

.venv/bin/python -m scripts.validate_droid_loader \
  --config config/splattervae/droid/pretrain.yaml \
  --preprocessed-root /home/ws/data/droid_stage0_preprocessed \
  --split validation --workers 4 --samples 512
```

## Cached training, DDP, and resume

Single GPU:

```bash
CUDA_VISIBLE_DEVICES="$GPU4" .venv/bin/python -m scripts.train_droid \
  --config config/splattervae/droid/pretrain.yaml
```

Three-GPU DDP:

```bash
CUDA_VISIBLE_DEVICES="$GPU4,$GPU5,$GPU6" \
  .venv/bin/torchrun --standalone --nnodes=1 --nproc-per-node=3 \
  -m scripts.train_droid --config config/splattervae/droid/pretrain.yaml
```

Resume either topology from a saved checkpoint:

```bash
CUDA_VISIBLE_DEVICES="$GPU4" .venv/bin/python -m scripts.train_droid \
  --config config/splattervae/droid/pretrain.yaml \
  --resume /absolute/path/to/step-XXXXXXXX.pt
```

For an offline W&B visualization smoke, add `WANDB_MODE=offline`,
`--wandb-enabled`, validation/visualization intervals of one, and explicit
checkpoint/visualization directories. Validation reads cached teacher outputs;
it never reruns a teacher. Panels include real timelines, both cached flow
pairs and the aligned middle-frame aggregate, crop/padding validity, cached DA3
versus rendered depth, RGB reconstructions, compact four-view Lager panels,
t0-only Gaussian clouds, and t0-to-t1-to-t2 tracking.

## Tests

```bash
.venv/bin/ruff check --select F,B,I,RUF022 --exclude third_party \
  dataset models preprocessing scripts tests
.venv/bin/pytest -q
.venv/bin/python -m compileall -q dataset models preprocessing scripts tests
git diff --check
```

Tests cover Stage-0 selection/indexing, codecs and sentinel behavior, atomic
restart-safe shards, target alpha/pose safety, validity-aware tube masking,
variable-token attention isolation, downstream shapes, finite reconstruction
and loss gradients, cached-only imports, checkpoint resume, and distributed
infrastructure.

Nothing in these workflows creates or modifies a file under
`/home/ws/data/droid`.
