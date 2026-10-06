# splatter4d

RGB-history state encoder with a world-frame dynamic Gaussian pretext, shared by Meta-World and PointWorld-DROID.

**Status: implementation and data validation, not a completed research result.** Native gsplat is incompatible with the installed PyTorch binary. Full collection, rendered training, research evaluation, and trained exports have not run. The obsolete Stage-0 cache was not deleted. Reference preservation has an unresolved discrepancy. See [results](docs/RESULTS.md), [assumptions](docs/ASSUMPTIONS.md), and [real PointWorld schema](docs/POINTWORLD_SCHEMA.md).

## Model and data contract

The RGB-only ViT encodes three chronological frames from one camera into K learned state tokens. Training uses motion-aware tube masking and per-time embeddings. Inference keeps every patch and flattens all slots. Neither camera parameters nor depth enter the encoder.

A slots-only set decoder produces a robot-base Gaussian scene. Separate scene and dynamic parents have distinct anchors, scales and local child radii. Scene translations are exactly zero. Dynamic translations define G1 from G0, then G2 from G1. Attributes other than positions are shared across time. A random source camera supplies each decoded state, which renders into all training cameras at all three times.

Training combines dynamically weighted RGB L1 and D-SSIM, depth alignment and gradients, center-only hard depth, detached-geometry world-displacement splats, coverage, visibility, multi-positive cross-view InfoNCE, and per-slot consistency. All network/rasterizer interactions are covered by explicitly synthetic tests. Actual Gaussian CUDA acceptance remains blocked.

| Batch field | Shape | Meaning |
|---|---|---|
| images | B,3,V,3,H,W | uint8 RGB |
| K | B,V,3,3 | float32 pixel intrinsics |
| w2c / c2w | B,V,4,4 | OpenCV cameras, robot-base world |
| depth | B,3,V,1,H,W | float32 meters; zero invalid |
| motion3d | B,3,V,3,H,W | world displacement pairs 01,12,02 |
| motion_weight | B,3,V,1,H,W | target support/confidence in [0,1] |
| motion_score | B,3,V,1,H,W | per-time dynamics score in [0,1] |

Optional probe state and a complete held-out evaluation group are validated when present. DROID has no fabricated held-out camera fields. Every collated batch is checked.

Meta-World uses 128x128, patch 16, one 256-dimensional slot, and 8,192 Gaussians. DROID uses sixteen 384-dimensional slots and 16,384 Gaussians. Scratch DROID uses 256x144 and patch 16. DINOv2 ViT-S/14 uses 252x140, patch 14, retained CLS, pretrained weights and interpolated spatial positions.

## Environment and safe execution

Use Python 3.10 and uv. Dependencies and Git revisions are pinned in `uv.lock`. Lock resolution succeeded without native builds:

```bash
uv lock --no-build-package gsplat --no-build-package fused-ssim
```

A standalone installation has **not** been validated. This checkout's temporary venv reads existing reference packages without editing them. Source builds were denied, and no alternate build route is used. An installation that requires compiling native packages must stop until appropriately authorized. The renderer never automatically invokes gsplat JIT. Do not interpret a lockfile as a functioning renderer.

Every GPU-aware command must first source the UUID definitions. MuJoCo processes must expose exactly one UUID. Two UUIDs are reserved for distributed jobs without MuJoCo.

```bash
source scripts/gpu_env.sh
CUDA_VISIBLE_DEVICES="$GPU4" .venv/bin/python -I scripts/run_tests.py
.venv/bin/ruff check s4d scripts tests
.venv/bin/ruff format --check s4d scripts tests
```

The test runner writes JSON, JUnit and console evidence under `docs`. It never skips native tests. Long collection and training require a complete passing suite for the exact current sources. Current native errors keep that gate closed. Check GPU occupancy before any future long job.

Outputs may only enter the new repository or the operation's explicitly approved new data root. Original DROID and both reference repos are read-only. Path guards do not override permission denials. `scripts/delete_stage0_cache.py` is disabled and always aborts. Do not use it as a deletion recipe.

Downloaded data is untrusted. Keep it separate from trusted scripts, use fresh empty destinations, and run Python readers with `-I`. No upstream teacher-generation code is required.

### Completed data validation

All eight Meta-World task gates and five-episode pilots are present. Their D2 motion and D3 fusion medians pass unchanged thresholds. They are not substitutes for the requested full datasets and saved splits.

Real DROID sample validation can be repeated with read-only inputs and repository-local outputs:

```bash
source scripts/gpu_env.sh
CUDA_VISIBLE_DEVICES="$GPU4" WANDB_MODE=offline .venv/bin/python -I scripts/droid_validate_alignment.py \
  --sample-root /home/ws/data/pointworld_droid_sample \
  --cache-root /home/ws/data/droid_pointworld_cache_sample \
  --output-root docs/droid_alignment_repeat
```

Add `--no-wandb` for local-only evidence. Existing evidence is in `docs/droid_dense_validation`. It includes true dense source-depth warps, world clouds, score maps, scene/gripper overlays, scalar thresholds and input hashes. Validation is one matched episode, not corpus coverage or cross-episode generalization.

The NVIDIA dataset license is included at `docs/pointworld_license/LICENSE.pdf`. Dataset-derived images/clouds are subject to its non-commercial terms. The upstream code's Apache license does not change those terms.

### Training commands for after the blocked gates are resolved

The following are protocol examples, **not completed runs**. Do not rerun denied shared-data writes or native builds. Full Meta-World data, manifests and per-task statistics are still missing.

```bash
source scripts/gpu_env.sh
CUDA_VISIBLE_DEVICES="$GPU4" WANDB_MODE=offline .venv/bin/python -I scripts/train.py \
  --config configs/metaworld/base.yaml configs/metaworld/tasks/hammer.yaml \
  --name metaworld-hammer-full-0

# Exact one-batch DROID overfit. Add dinov2.yaml before overfit.yaml for pretrained ViT-S/14.
CUDA_VISIBLE_DEVICES="$GPU5" WANDB_MODE=offline .venv/bin/python -I scripts/train.py \
  --config configs/droid/pretrain.yaml configs/droid/overfit.yaml \
  --name droid-sample-overfit-scratch-0
```

DROID pretrain.yaml deliberately defaults to the requested short smoke run, not full pretraining. Overfit repeats training-window indices [0,1] for training and evaluation. It makes no validation generalization claim. The original temporal ramp and depth-alignment warmup are not shortened.

Use layered ablation YAML files under `configs/metaworld/ablations`. `--set key.path=value` overrides a resolved setting. Existing runs require explicit `--resume auto`. Checkpoints save optimizer and per-rank random state.

For isolated-Python two-GPU smoke execution after acceptance gates are open:

```bash
CUDA_VISIBLE_DEVICES="$GPU4,$GPU5" WANDB_MODE=offline .venv/bin/python -I -m torch.distributed.run \
  --nproc-per-node=2 --no-python .venv/bin/python -I scripts/train.py \
  --config configs/droid/pretrain.yaml --set train.steps=50 train.eval_every=50 train.save_every=50 \
  --name droid-sample-ddp-smoke-0
```

`scripts/droid_compare_step0.py` separately compares a single global batch against two equal rank partitions. Both invocations must use identical config, indices and model initialization. It fixes source views, disables stochastic masks/dropout/jitter, uses FP32, and records fingerprints. This checks deterministic rendered-loss equality to absolute tolerance 1e-4. It is not ordinary bf16 training and has not executed natively.

Run logs use `splatter4d-metaworld` and `splatter4d-droid`. Offline W&B and `--no-wandb` are supported. Local mirrors contain JSONL scalars, evaluation JSON, PNG panels, GIF videos, PLY clouds and NumPy motion vectors. Rank-zero evaluation is synchronized across ranks. CPU/Gloo tests are not NCCL training evidence.

```bash
CUDA_VISIBLE_DEVICES="$GPU4" .venv/bin/python -I scripts/evaluate.py --run outputs/metaworld-hammer-full-0
CUDA_VISIBLE_DEVICES="$GPU4" .venv/bin/python -I scripts/summarize_run.py outputs/metaworld-hammer-full-0
CUDA_VISIBLE_DEVICES="$GPU4" .venv/bin/python -I scripts/export_encoder.py --run outputs/metaworld-hammer-full-0
```

These checkpoint commands require actual training artifacts, which are not currently available. Empty moving/static regions and missing distinct-episode retrieval protocols are reported unavailable, not passing.

## Encoder-only policy hook

The old DrQ-v2 wrapper in the reference `agents/common/encoders.py` consumes channel-stacked frames. A frozen adapter would restore the chronological time axis, then call the exported encoder. No agent port or RL training is included.

```python
# Apply the same UUID guard before importing torch in the policy entrypoint.
from s4d.gpu_guard import enforce_allowed_gpus

enforce_allowed_gpus()
from s4d.model.encoder import load_encoder

encoder = load_encoder("outputs/metaworld-hammer-full-0/encoder.pt").to("cuda")
# stacked_rgb: B,9,H,W uint8, oldest frame first, from one camera
history = stacked_rgb.reshape(stacked_rgb.shape[0], 3, 3, *stacked_rgb.shape[-2:])
state = encoder.policy_state(history.to("cuda"))
# Feed state to the existing actor/critic instead of its old vision representation.
```

The policy state has shape B,K*D_s. Camera matrices, decoder, depth, masks, gsplat and a pretrained-weight download are unnecessary at inference. Float RGB must be in [0,1]; uint8 is normalized internally. Match the exported resolution and temporal spacing. `load_encoder` returns a frozen evaluation module. Export/reload identity passed on untrained scratch and pretrained DINOv2 backbones, not on a trained research checkpoint.
