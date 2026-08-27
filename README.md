# SplatterVAE DROID pretraining

This `DROID` branch implements online-teacher pretraining for a reusable
three-frame ViT-S/16 representation. It preserves the Dynamic3D Gaussian
decoder and visualization foundations while replacing task-specific/offline
teacher paths with:

- online X-Lens metric depth;
- online MEMFOF `MEMFOF-Tartan-T-TSKH` flow at native DROID resolution with
  exactly two refinement iterations;
- optional online LagerNVS posed novel-view supervision, run for every logical
  sample whenever the Boolean `novel_view.enabled` flag is true.

The DROID source is always read-only. Neural teacher predictions are never
persistently cached.

## Pipeline

```text
raw DROID RGB [B,2 cameras,3 times,3,180,320]
  |
  +-- MEMFOF, native 180x320, iters=2
  |     -> middle-to-previous and middle-to-next flow
  |     -> max magnitude, smoothing, feasible argmax
  |     -> one motion-centered Uniform{180,...,320} crop per camera
  |     -> same crop across t0/t1/t2 -> 224x224 ViT input
  |
  +-- X-Lens, online frozen teacher
  |     -> metric depth, confidence, validity
  |     -> same 224x224 geometric transform for real-view losses
  |     -> raw-grid geometry for workspace and target coverage
  |
  +-- LagerNVS branch (when enabled), before the random ViT crop
        -> raw current exterior A/B views
        -> deterministic canonical 256x256 cameras
        -> arc interpolation + bounded perturbation + X-Lens coverage check
        -> LagerNVS RGB and GS render compared directly at 256x256
```

The flow-centered 224x224 crop is never used as LagerNVS conditioning. The two
camera pipelines intentionally branch at raw 320x180 RGB.

## Data and write-safety contract

Run development only from:

```text
/home/ws/ws/droid_training
```

Storage domains are separate:

```text
code:          /home/ws/ws/droid_training
DROID source:  /home/ws/data/droid/               strictly read-only
derived data:  /ws/data/ws/droid_splattervae/     configurable
project output:/home/ws/ws/droid_training/outputs/
```

`/ws/data/ws/droid/` was the previously supplied, incorrect dataset path. It is
not used by config, validation, preprocessing, training, or smoke commands.

Every output entrypoint rejects paths inside or containing the DROID source.
Real-data validators fingerprint the source before and after execution. The
audited metadata fingerprint is:

```text
files:  2,051
bytes:  1,866,281,754,039
SHA256: 6f90d6f97e73d36e0243d89fe6e26621c05670c299294ee3759b0ed370d99ad3
```

The source exposes TFDS/RLDS builder `1.0.1`, one `train` split, 95,658
episodes, and 2,048 TFRecord shards. Each step has synchronized
`exterior_image_1_left`, `exterior_image_2_left`, and `wrist_image_left` uint8
RGB at `H x W x C = 180 x 320 x 3`. Initial geometry training uses only the two
calibrated exterior cameras.

## Calibration audit

The full canonical Stage-0 manifest is configured at:

```text
/ws/data/ws/droid_splattervae/manifests/canonical-full/calibration.jsonl.gz
```

Full-release results:

```text
episodes inspected                 95,658
official-path matches              73,361
Stage-0 valid                      33,195
rejected                           62,463
train / validation                 32,905 / 290
insufficient exterior geometry     36,010
official-path match failed         22,034
invalid intrinsics                  1,518
missing intrinsics                  1,156
direct/relative disagreement        1,111
```

Official 1280x720 intrinsics are transformed exactly to the RLDS 320x180 grid.
`cam2base` is interpreted as `T_base<-camera`; robot base is the world frame.

## Motion-centered preprocessing

For every logical sample, one integer crop size is sampled uniformly from the
closed interval `[180,320]`. There is no global/local mixture and no random
spatial center.

For each physical camera independently:

1. Run MEMFOF on `[t0,t1,t2]` at 180x320.
2. Compute backward and forward flow magnitudes on the middle-frame grid.
3. Aggregate with `max`, pad the motion map by 70 pixels above and below, and
   smooth with a 15x15 average kernel.
4. Restrict candidate centers to those that keep the sampled square inside the
   320x320 canvas.
5. Select the exact feasible argmax. Only an effectively zero map uses the
   deterministic `(160,160)` fallback.
6. Apply this one crop to all three frames and every aligned modality.
7. Resize to 224x224. RGB uses bicubic interpolation; depth, confidence,
   validity, and flow use modality-appropriate transforms.

The original camera intrinsic `(fx,fy,cx,cy)` becomes, for crop origin
`(x0,y0)` and `scale=224/crop_size`:

```text
fx' = fx * scale
fy' = fy * scale
cx' = (cx - x0) * scale
cy' = (cy + 70 - y0) * scale
```

Spatially resized optical-flow vectors are also multiplied by `scale`. Padded
pixels are invalid for RGB, metric-depth, SI-depth, and flow losses. No semantic
segmentation input or mask exists.

The deterministic 100,000-draw audit observed the complete integer interval,
a mean of 249.982 versus the expected 250.0, and no endpoint mixture.

## Teachers and checkpoints

### X-Lens

The manually supplied checkpoint was moved from `/home/ws/model.safetensors`
to:

```text
/home/ws/ws/droid_training/checkpoints/xlens/model.safetensors
```

It is 148 MiB and has SHA256:

```text
266a0340b53e5cb996cc613a1b0c5966b5bcaeee1ec7c4431e4fc6e7d1e58a0c
```

The safetensors state dictionary is strictly compatible with the pinned
official ViT-S X-Lens configuration. The teacher is frozen, `eval()`, and runs
under inference mode with BF16 autocast.

### MEMFOF

Pinned sources and weights:

```text
repository commit: a51de9fc59c6fe20ba08e079372c7b583d58a712
model ID:          egorchistov/optical-flow-MEMFOF-Tartan-T-TSKH
model revision:    6c6c9aa3ad64f93aee8efbc2f7a6e4535814ee96
input:             [B,3,3,180,320] per camera
directions:        middle -> previous, middle -> next
iterations:        exactly 2
precision:         FP32
```

The pinned implementation needs one narrowly scoped correlation-pyramid fix at
native 180x320: it avoids an unused zero-size post-final downsample while
leaving every constructed correlation level unchanged.

Important quality finding: tensor shape, direction order, iteration count, and
crop argmax behavior pass, but the real-DROID teacher quality audit does not.
Mean flow magnitude was 33.51 px and p95 was 69.51 px on near-static
sequences; zero-flow photometric L1 was 0.0062-0.0064 while the official MEMFOF
warp error was 0.181-0.231. Identical frames produced 31.04 px mean and 67.14 px
p95 flow. The machine-readable smoke result therefore marks
`failed_real_photometric_sanity`. Do not interpret the mandated checkpoint and
two-iteration setting as validated supervision quality.

### LagerNVS

Pinned configuration:

```text
repository commit: 665f727aba8298a04ff4c040fd6279a32ef23017
checkpoint ID:      facebook/lagernvs_dl3dv_2-6_v_256
checkpoint revision:4026552953a72c5fb037501564dc673dd73c574e
posed image size:   256x256
```

The checkpoint is Hugging Face gated. An accepted license and `HF_TOKEN` are
required. Without that token, inference validation and LagerNVS-ON profiling
produce an explicit blocked summary rather than silently skipping samples.

The target sampler uses the two exterior cameras, quaternion SLERP, and
scene-centered camera-center arc interpolation when stable. Defaults are:

```text
alpha                         Uniform(0.15,0.85)
translation perturbation     <= 0.03 source baseline
rotation perturbation        <= 3 degrees, no roll axis
minimum source coverage      0.60
resample attempts            4
fallback                     conservative unperturbed midpoint
```

On 152 real X-Lens-backed targets across eight episodes, scene-centered arc was
used for every target, coverage was 0.8918 minimum and 0.9522 mean, no candidate
was rejected, and no fallback was needed. Translation and rotation maxima were
0.02977 baseline and 2.9948 degrees. These findings support the conservative
defaults; they do not substitute for actual LagerNVS image-quality validation.

## Representation and losses

- ViT-S/16 at 224x224: 384 dimensions, 12 blocks, 6 heads, RMSNorm, SwiGLU,
  approximately ViT-S parameter scale.
- Three-frame joint temporal attention with 60% tube masking.
- Output CLS `[B,384]` and current-frame patches `[B,196,384]`.
- Cross-view InfoNCE between synchronized exterior-camera histories.
- 256 parent/group queries, eight children per group, 2,048 dynamic Gaussians.
- Real RGB L1 + SSIM, X-Lens metric L1 + SI depth, MEMFOF flow/dynamics,
  visibility, and segmentation-independent Gaussian regularization.
- When enabled, visibility-aware LagerNVS RGB supervision at canonical 256x256
  with supported/unsupported pixel weights 1.0/0.0.

There is no segmentation, representation-consistency loss, SSL/EMA teacher,
action-conditioned future-latent prediction, or PointWorld path.

Encoder-only inference is independent of pretraining heads:

```python
features = model.inference_features(history)  # [B,3,3,224,224]
cls_token = features["cls_token"]             # [B,384]
patch_tokens = features["patch_tokens"]       # [B,196,384]
```

## Environment setup

```bash
cd /home/ws/ws/droid_training
git submodule update --init --recursive
uv venv --python 3.10 .venv
uv sync --all-extras
export PATH=/home/ws/ws/droid_training/.venv/bin:$PATH
export PYTHONPATH=/home/ws/ws/droid_training
```

For a fresh checkout containing the manually placed X-Lens checkpoint:

```bash
mkdir -p /home/ws/ws/droid_training/checkpoints/xlens
mv /home/ws/model.safetensors \
  /home/ws/ws/droid_training/checkpoints/xlens/model.safetensors
sha256sum /home/ws/ws/droid_training/checkpoints/xlens/model.safetensors
```

Checkpoint/model-cache directories are gitignored; do not commit model weights.

Every GPU command must expose only the authorized GPU:

```bash
export CUDA_VISIBLE_DEVICES=GPU-76871c16-ff1d-f7b1-cdf5-fabbaf9df8ce
.venv/bin/python -c 'import torch; print(torch.cuda.device_count(), torch.cuda.current_device(), torch.cuda.get_device_name(0))'
```

Inside applications the device is always `cuda:0`.

## Calibration, loader, and workspace commands

```bash
CFG=config/splattervae/droid/pretrain.yaml

# Build the full official post-hoc calibration manifest read-only against RLDS.
.venv/bin/python scripts/prepare_droid_calibration.py --config "$CFG"

# Real loader/schema check, including spawned workers.
.venv/bin/python scripts/validate_droid_loader.py \
  --config "$CFG" --split validation --workers 2 --samples 4

# Reproduce the audited 76-episode, three-frame online X-Lens workspace sample.
CUDA_VISIBLE_DEVICES=GPU-76871c16-ff1d-f7b1-cdf5-fabbaf9df8ce \
  .venv/bin/python scripts/compute_droid_workspace_stats.py \
  --config "$CFG" \
  --maximum-episodes 76 \
  --frames-per-episode 3 \
  --teacher-frame-batch 3 \
  --output outputs/teacher_validation/xlens/workspace_stats_76x3.json
```

The selected real-X-Lens workspace parameters are:

```text
global_center             [0.626891, 0.037195, 0.157134] m
anchor spread              0.463211 m
parent displacement scale  0.304793 m
child radius               0.053339 m
znear / zfar               0.292888 / 7.307050 m
```

They are checked into YAML with status
`validated_from_droid_xlens_stats_and_pilot` and were stable in a ten-step real
optimizer pilot.

## Teacher and geometry validation commands

```bash
CFG=config/splattervae/droid/pretrain.yaml
WS=outputs/teacher_validation/xlens/workspace_stats_76x3.json
GPU=GPU-76871c16-ff1d-f7b1-cdf5-fabbaf9df8ce

# Real X-Lens + MEMFOF + crop + K/flow/depth + forward/backward/render smoke.
CUDA_VISIBLE_DEVICES="$GPU" .venv/bin/python \
  scripts/smoke_test_real_droid_gpu.py \
  --config "$CFG" --workspace-stats "$WS" --samples 2 \
  --output-root outputs/teacher_validation/online_smoke_final

# Real target-pose distribution and X-Lens coverage, no Lager checkpoint needed.
CUDA_VISIBLE_DEVICES="$GPU" .venv/bin/python \
  scripts/validate_lagernvs_target_poses.py \
  --config "$CFG" --workspace-stats "$WS" \
  --samples 8 --draws-per-sample 16 \
  --output-root outputs/teacher_validation/lagernvs/target_pose_distribution

# Actual official LagerNVS inference; writes a blocked summary if access is gated.
CUDA_VISIBLE_DEVICES="$GPU" .venv/bin/python \
  scripts/validate_lagernvs_droid.py \
  --config "$CFG" --workspace-stats "$WS" --samples 4 \
  --output-root outputs/teacher_validation/lagernvs/inference
```

After accepting the LagerNVS model terms, expose `HF_TOKEN` in the shell before
the last command. Do not place Hugging Face caches in the DROID source.

## Training, W&B validation, and resume

LagerNVS OFF:

```bash
CUDA_VISIBLE_DEVICES=GPU-76871c16-ff1d-f7b1-cdf5-fabbaf9df8ce \
  torchrun --standalone --nproc_per_node=1 scripts/train_droid.py \
  --config config/splattervae/droid/pretrain.yaml \
  --novel-view-disabled
```

LagerNVS ON every iteration and every logical sample:

```bash
CUDA_VISIBLE_DEVICES=GPU-76871c16-ff1d-f7b1-cdf5-fabbaf9df8ce \
  torchrun --standalone --nproc_per_node=1 scripts/train_droid.py \
  --config config/splattervae/droid/pretrain.yaml \
  --novel-view-enabled
```

One-step deterministic offline-W&B validation:

```bash
WANDB_MODE=offline \
CUDA_VISIBLE_DEVICES=GPU-76871c16-ff1d-f7b1-cdf5-fabbaf9df8ce \
  torchrun --standalone --nproc_per_node=1 scripts/train_droid.py \
  --config config/splattervae/droid/pretrain.yaml \
  --novel-view-disabled --wandb-enabled --run-name droid-validation \
  --max-steps 1 --per-gpu-batch 1 --gradient-accumulation 1 --workers 0 \
  --validation-every-steps 1 --visualization-every-steps 1 \
  --checkpoint-every-steps 1 --validation-batches 1 \
  --num-visualization-samples 1 \
  --checkpoint-dir outputs/wandb_validation/checkpoints \
  --visualization-dir outputs/wandb_validation/visualization
```

Short real-DROID pilot:

```bash
CUDA_VISIBLE_DEVICES=GPU-76871c16-ff1d-f7b1-cdf5-fabbaf9df8ce \
  torchrun --standalone --nproc_per_node=1 scripts/train_droid.py \
  --config config/splattervae/droid/pretrain.yaml \
  --novel-view-disabled --max-steps 10 --per-gpu-batch 1 \
  --gradient-accumulation 1 --workers 2 \
  --validation-every-steps 5 --visualization-every-steps 5 \
  --checkpoint-every-steps 5 --validation-batches 1 \
  --checkpoint-dir outputs/pilot_corrected/checkpoints \
  --visualization-dir outputs/pilot_corrected/visualization
```

Resume from the corrected step-10 checkpoint:

```bash
CUDA_VISIBLE_DEVICES=GPU-76871c16-ff1d-f7b1-cdf5-fabbaf9df8ce \
  torchrun --standalone --nproc_per_node=1 scripts/train_droid.py \
  --config config/splattervae/droid/pretrain.yaml \
  --novel-view-disabled --resume \
  outputs/pilot_corrected/checkpoints/step-00000010.pt
```

## Checkpoint analysis

Evaluate every meaningful existing checkpoint on the same deterministic real
DROID batch:

```bash
CUDA_VISIBLE_DEVICES=GPU-76871c16-ff1d-f7b1-cdf5-fabbaf9df8ce \
  .venv/bin/python scripts/analyze_droid_checkpoints.py \
  --config config/splattervae/droid/pretrain.yaml \
  --checkpoint-dir outputs/pilot/checkpoints \
  --checkpoint-dir outputs/pilot_corrected/checkpoints \
  --checkpoint-dir outputs/resume_validation/checkpoints \
  --validation-samples 2 \
  --output-root outputs/checkpoint_analysis
```

Outputs are `metrics.csv`, `summary.json`, `plots/`, and `qualitative/`.
Current selections are step 11 for RGB and dynamics, step 5 for depth and
overall, and step 1 for representation only because every short checkpoint has
the same collapse warning: positive and negative cosine similarities are both
1.0. Novel-view checkpoint metrics remain unavailable until LagerNVS inference
is authorized.

## End-to-end profiling

The profiler uses CUDA events and synchronization, separates cold load from
steady state, clears allocator caches between batch points, records allocated
and reserved memory, and profiles target interpolation, perturbation, coverage,
and resampling control separately.

```bash
CUDA_VISIBLE_DEVICES=GPU-76871c16-ff1d-f7b1-cdf5-fabbaf9df8ce \
  .venv/bin/python scripts/profile_droid_end_to_end.py \
  --config config/splattervae/droid/pretrain.yaml \
  --mode both \
  --batch-sizes 1,2,4,8,16,32,64,128,160,192,256 \
  --offload-batch-size 64 \
  --warmup-iterations 2 --iterations 5 \
  --quiet \
  --output-root outputs/profiling/final
```

`--mode both` always profiles trainable-only and MEMFOF+X-Lens all-resident.
It additionally profiles LagerNVS ON-every-iteration when the gated checkpoint
loads; otherwise `profile.json` records the exact blocker. The sequential
offload experiment moves frozen models between CPU and GPU without reloading
weights from disk.

## Visualization

The existing Dynamic3D visualization remains the base. Rank 0 writes/logs a
small fixed validation subset only at configured intervals, reusing teacher
outputs from the validation forward pass. Namespaces include:

```text
val/input/          original t0/t1/t2 at 320x180
val/memfof/         backward/forward flow, magnitude, aggregate/smoothed motion
val/crop/           padded canvas, rectangle, center, crop size, 224 crop
val/xlens/          metric depth, confidence/validity, GS depth and error
val/reconstruction/ real RGB/depth/flow renders and errors
val/lagernvs/       source views, target metadata, Lager/GS/error/support
val/geometry/       t0 full cloud and camera-frame NPZ
val/tracking/       t0->t1->t2 Gaussian trajectories
```

Square 224/256 images are mapped to a natural 320x180 display camera with
intrinsic-aware resampling and a validity mask; they are never naively
stretched. Full static point-cloud output is t0 only, while temporal tracking
remains t0->t1->t2. The inherited backend reliably writes camera matrices and
point clouds but does not render camera frustums itself.

## Tests

```bash
.venv/bin/ruff check --select F,B,I,RUF022 \
  --exclude third_party \
  dataset models preprocessing scripts tests
.venv/bin/pytest -q
.venv/bin/python -m compileall -q \
  dataset models preprocessing scripts tests
```

The tests cover dataset safety, RLDS schema, calibration, temporal sampling,
padding/cropping, uniform crop sizes, flow argmax/fallback, temporal and
cross-camera crop behavior, exact K/depth/flow/validity transforms, flow-vector
scaling, Lager camera conventions, quaternion SLERP, coverage, bounded target
sampling, model contracts, losses, DDP, checkpointing, and visualization
infrastructure.

## Output organization

```text
outputs/checkpoint_analysis/
outputs/profiling/
outputs/teacher_validation/xlens/
outputs/teacher_validation/memfof/
outputs/teacher_validation/lagernvs/
outputs/visualization/{input,crop,depth,reconstruction,lagernvs,geometry,tracking}/
```

Nothing in these workflows writes under `/home/ws/data/droid/`.
