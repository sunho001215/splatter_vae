# SplatterVAE DROID pretraining

This branch is a DROID-only implementation for pretraining a reusable temporal
ViT-S visual encoder with cross-view invariance, calibrated 3D reconstruction,
and dynamics supervision. It uses the two synchronized exterior cameras from
DROID, official post-hoc calibration, offline X-Lens metric depth, and offline
WAFT flow. Optional See3D views are geometry-warped first and completed second;
they remain disabled until real-target validation is explicitly approved.

The reusable downstream component is the RGB encoder. The Gaussian decoder and
contrastive projector are pretraining-only heads.

## Architecture

- Three-frame causal history `[t-2s, t-s, t]`, with configurable strides
  `1/3/6` and learned temporal embeddings. No future frame is used.
- 224 x 224 RGB, 16 x 16 patches, 384 dimensions, 12 blocks, 6 heads,
  RMSNorm, SwiGLU hidden size 1024, LayerScale `1e-5`, and drop path `0.1`.
- Flow-guided temporal tube masking at 60%, with an even motion-prioritized and
  uniform visible-patch mixture.
- Raw CLS features feed a separate `384 -> 1024 -> 256` contrastive projector.
- 256 learned Gaussian slots cross-attend directly to all surviving temporal
  encoder tokens. Each slot predicts one parent and eight local children:
  256 x 8 = 2048 dynamic 3D Gaussians.
- Reconstruction uses RGB L1/DSSIM, expected-depth metric L1 and SI-log depth,
  rendered Gaussian flow, visibility, and segmentation-free regularization.

Encoder-only downstream use is intentionally independent of all pretraining
heads:

```python
features = model.inference_features(history)  # history: [B, 3, 3, 224, 224]
cls_token = features["cls_token"]             # [B, 384]
patch_tokens = features["patch_tokens"]       # [B, 196, 384]
```

Masking is disabled for this API. Policies may use CLS, all patch tokens, or
their own pooling without changing the encoder.

## Data and safety contract

Run all commands from this repository:

```bash
cd /home/ws/ws/droid_training
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
export PATH="$PWD/.venv/bin:$PATH"
```

The three storage domains are deliberately separate:

```text
code:          /home/ws/ws/droid_training
DROID source:  /ws/data/ws/droid                 (strictly read-only)
derived data:  /ws/data/ws/droid_splattervae     (configurable)
```

Every output entrypoint rejects a derived path inside, or containing, the DROID
source tree. Calibration manifests and HDF5 shard indices carry content-based
calibration identities; cache readers reject mismatched teacher, checkpoint,
resolution, preprocessing, or calibration provenance. Framework download
caches are redirected to `<derived_root>/metadata/model_cache/`.

The expected derived layout is:

```text
calibration/  manifests/  xlens/  waft/  see3d/
workspace_stats/  metadata/  logs/
```

Do not place checkpoints, manifests, temporary files, or model caches under the
DROID source directory.

## Environment and official models

Initialize the pinned official implementations:

```bash
git submodule update --init --recursive
uv venv --python 3.10 .venv
uv sync --all-extras
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
export PATH="$PWD/.venv/bin:$PATH"
```

The submodules are the official repositories:

- X-Lens: `third_party/XLens`
- WAFT: `dataset/third_party/WAFT`
- See3D: `third_party/See3D`

Obtain their official released weights using each upstream repository's
instructions. Set `depth.checkpoint_path`, `flow.checkpoint_path`,
`flow.depth_checkpoint_path`, and `novel_view.see3d.checkpoint_path` in
`config/splattervae/droid/pretrain.yaml`. Training never invokes these teachers
inside the optimizer loop.

All GPU commands below expose only the authorized device. It must be reported
inside PyTorch as `cuda:0`:

```bash
export CUDA_VISIBLE_DEVICES=GPU-76871c16-ff1d-f7b1-cdf5-fabbaf9df8ce
.venv/bin/python -c 'import os, torch; print(os.environ["CUDA_VISIBLE_DEVICES"]); print(torch.cuda.current_device(), torch.cuda.get_device_name(0))'
```

## Offline preparation

Use this order. All paths and sampling choices are controlled by
`config/splattervae/droid/pretrain.yaml`.

```bash
CFG=config/splattervae/droid/pretrain.yaml

# Download official post-hoc calibration, scan RLDS metadata read-only,
# infer/verify cam2cam direction, and write the Stage-0 manifest and split.
.venv/bin/python scripts/prepare_droid_calibration.py --config "$CFG"

# Cache synchronized exterior-camera metric depth at the 320 x 180 RLDS grid.
CUDA_VISIBLE_DEVICES=GPU-76871c16-ff1d-f7b1-cdf5-fabbaf9df8ce \
  .venv/bin/python scripts/precompute_xlens_droid.py --config "$CFG"

# Cache WAFT forward flow/confidence/uncertainty for gaps 1,2,3,6,12.
CUDA_VISIBLE_DEVICES=GPU-76871c16-ff1d-f7b1-cdf5-fabbaf9df8ce \
  .venv/bin/python scripts/precompute_waft_droid.py --config "$CFG"

# Back-project robust X-Lens geometry into the robot-base frame.
.venv/bin/python scripts/compute_droid_workspace_stats.py --config "$CFG"
```

Calibration preparation creates a persistent manifest with episode/path
matching, physical serials, original and RLDS-grid intrinsics, `c2w`/`w2c`,
direct/derived pose flags, split, validity, rejection reason, and summary
statistics. Splits are episode-level and grouped by recording session where the
metadata permits it.

Official calibration intrinsics are interpreted on their released 1280 x 720
grid and exactly transformed to 320 x 180 before global/local augmentation.
`cam2base` means `T_base<-camera`; robot base is the world frame.

### Workspace pilot

The checked-in spatial values are explicitly provisional, and normal training
refuses them. Use the generated statistics for a short geometry pilot:

```bash
CUDA_VISIBLE_DEVICES=GPU-76871c16-ff1d-f7b1-cdf5-fabbaf9df8ce \
  torchrun --standalone --nproc_per_node=1 scripts/train_droid.py \
  --config "$CFG" \
  --workspace-stats /ws/data/ws/droid_splattervae/workspace_stats/workspace_stats.json \
  --allow-unvalidated-workspace --max-steps 100
```

Inspect the logged parent/child position, scale, opacity, visibility,
out-of-frustum, RGB, depth, and flow diagnostics. Copy the selected values into
the YAML and set `decoder.workspace_parameter_status` to
`validated_from_droid_xlens_stats_and_pilot` before a full run.

## See3D validation and cache

See3D is not treated as a pose-controlled renderer. Known DROID RGB-D and
calibration first establish the target view through forward reprojection;
official See3D fills holes and disocclusions. Approval is deliberately a
separate human decision.

```bash
# A -> B and B -> A validation; writes metrics and qualitative grids but no approval.
CUDA_VISIBLE_DEVICES=GPU-76871c16-ff1d-f7b1-cdf5-fabbaf9df8ce \
  .venv/bin/python scripts/validate_see3d_droid.py --config "$CFG"

# After inspecting summary.json, summary.csv, and grids, approve cache creation.
.venv/bin/python scripts/validate_see3d_droid.py --config "$CFG" \
  --approve-existing-summary --approve-for-precompute

# Sample between real exterior poses, fuse two RGB-D warps, complete, score,
# reject low-confidence views, and write sharded See3D+X-Lens cache data.
CUDA_VISIBLE_DEVICES=GPU-76871c16-ff1d-f7b1-cdf5-fabbaf9df8ce \
  .venv/bin/python scripts/precompute_see3d_droid.py --config "$CFG"

# Separately approve the inspected cache/validation for training.
.venv/bin/python scripts/validate_see3d_droid.py --config "$CFG" \
  --approve-existing-summary --approve-for-training
```

Only then set `novel_view.enabled: true`. It defaults to false and additionally
requires an approved summary and compatible cache at startup.

## Training and resume

Single-GPU torchrun exercises the same DDP path as a larger launch:

```bash
CUDA_VISIBLE_DEVICES=GPU-76871c16-ff1d-f7b1-cdf5-fabbaf9df8ce \
  torchrun --standalone --nproc_per_node=1 scripts/train_droid.py \
  --config config/splattervae/droid/pretrain.yaml
```

Resume from an unwrapped checkpoint with matching world size:

```bash
CUDA_VISIBLE_DEVICES=GPU-76871c16-ff1d-f7b1-cdf5-fabbaf9df8ce \
  torchrun --standalone --nproc_per_node=1 scripts/train_droid.py \
  --config config/splattervae/droid/pretrain.yaml \
  --resume /ws/data/ws/droid_splattervae/logs/checkpoints/step-00005000.pt
```

The effective logical batch is
`per_gpu_logical_batch * gradient_accumulation_steps * WORLD_SIZE`. Encoder LR
is scaled from `1.5e-4` at batch 256 and the decoder uses the configured 2x
multiplier. Startup logs the batch calculation, both LRs, module parameter
counts, GPU mapping, and initialization diagnostics.

## Validation and development tests

CPU tests cover calibration matching/direction/poses, 1280 x 720 to RLDS
intrinsics, global/local RGB/depth/confidence/flow transforms, cache safety and
provenance, temporal sampling, model contracts, losses, distributed sampling,
and rank-aware checkpoint RNG state:

```bash
.venv/bin/ruff check --select F,B,I,RUF022 \
  --exclude third_party --exclude dataset/third_party \
  dataset models preprocessing scripts tests
.venv/bin/pytest -q
```

Synthetic forward/backward, expected-depth rasterization, Gaussian-flow loss,
contrastive gradients, and checkpoint resume use the authorized GPU:

```bash
CUDA_VISIBLE_DEVICES=GPU-76871c16-ff1d-f7b1-cdf5-fabbaf9df8ce \
  .venv/bin/python scripts/smoke_test_droid_gpu.py

CUDA_VISIBLE_DEVICES=GPU-76871c16-ff1d-f7b1-cdf5-fabbaf9df8ce \
  torchrun --standalone --nproc_per_node=1 \
  scripts/smoke_test_droid_gpu.py --ddp
```

Multi-GPU DDP is implemented, but it should only be runtime-tested when the
additional physical GPUs are explicitly authorized.
