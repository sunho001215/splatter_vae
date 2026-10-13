# DynaMo baseline (Cui et al., NeurIPS 2024)

DynaMo pretrains a visual encoder in-domain with a self-supervised dynamics objective: a causal transformer infers a
latent "action" `z_{t+1}` from the frame embeddings (inverse dynamics), a second transformer predicts `s_{t+1}` from
`(s_t, z_{t+1})` (forward dynamics), and the loss is `1 - cos(prediction, stop-grad(s_{t+1}))` plus a covariance
regulariser on the embeddings. No actions, states, augmentations or camera parameters are used. Here it is
pretrained per task on our Meta-World data and then used as a frozen per-frame encoder in DrM, like the other
baselines.

## Source

| | |
|---|---|
| Repository | https://github.com/jeffacce/dynamo_ssl (submodule `third_party/dynamo`) |
| Pinned commit | `62874039e5d3fc663b0fccd2083b2dd2c8d9935e` (2024-10-27) |
| License | MIT, Copyright (c) 2024 Zichen Jeff Cui (`third_party/dynamo/LICENSE`) |
| Official environment | `conda_env.yml`: python 3.8, torch 2.0.1, torchvision 0.15.2, accelerate 0.22.0; launched as `python3 train.py --config-name=train_*` (one process, no accelerate config, so fp32) |
| Weights | none: the paper's main results use a randomly initialised ResNet-18 (`configs/encoder/resnet18_random.yaml`), and so do we. Nothing was downloaded. |

## Imported, patched, wrapped

**Imported unchanged** from the patched source tree (`s4d.third_party.source_tree("dynamo")`, a cached copy of the
pinned commit with the patch below applied), loaded by `s4d/baselines/dynamo/official.py` as the package
`s4d_third_party_dynamo.models` (nothing goes on `sys.path`; the upstream names `models`, `utils` and `datasets` never
enter `sys.modules`):

- `models/encoder/resnet.py::resnet18`: the encoder;
- `models/projector/inverse_dynamics.py::InverseDynamicsProjector`: the inverse-dynamics model;
- `models/ssl/dynamo.py::DynaMoSSL`: the forward-dynamics model, its optimiser, the loss and `step()`;
- through them, `models/transformer_encoder.py` (the GPT-style causal transformer and its `configure_optimizers`),
  `models/ema.py` and `models/ssl/base.py`.

**Patched:** `third_party/patches/dynamo/0001-dynamo-ssl-without-accelerate.patch` changes `models/ssl/dynamo.py`
only. It deletes three statements: `from accelerate import Accelerator`, the module-level `accelerator =
Accelerator()`, and `accelerator.prepare(forward_dynamics, forward_dynamics_optimizer)`. accelerate is not in the
locked environment. For the official single-process fp32 launch, `prepare` only moves the model to the device and
wraps the optimiser without changing its step. Our builder moves `DynaMoSSL` to the device instead. A test checks
that the cached tree differs from the checkout in this file only, and only by these removed lines.

**Wrapped / replicated** (our code). Each cites the upstream lines it follows:

| File | What |
|---|---|
| `s4d/baselines/dynamo/official.py` | loader of the official package under a unique name |
| `s4d/baselines/dynamo/data.py` | `DynaMoWindows`: our HDF5 -> the windows of the official `TrajectorySlicerDataset` (`datasets/core.py:128-141, 208-211`); `to_official_obs`: exact `obs / 255.0` |
| `s4d/baselines/dynamo/model.py` | `build_modules`: the official construction order and hydra interpolations (`train.py:56-61, 92-135`); encoder export / reload |
| `s4d/baselines/dynamo/training.py` | `set_seed_everywhere` (`utils/__init__.py:60-67`), `epoch_lr`/`start_epoch` (`train.py:170-173, 371-384`), `train_step` (`train.py:180-209`), `evaluate` (`train.py:211-273`), `EpochLog` (`train.py:339-369`), `EpochSampler` (replaces `DataLoader(shuffle=True)`, `train.py:163`) |
| `s4d/baselines/dynamo/rl.py` | `DynaMoFrameEncoder`, the frozen RL backbone (registered as `"dynamo"` in `s4d/rl/encoders.py::BACKBONES`) |
| `scripts/train_dynamo.py` | guarded trainer: data, outputs, W&B, checkpoints, `--resume auto`, exports |
| `scripts/dynamo_panels.py` | sanity panels |
| `configs/baselines/dynamo/<task>.yaml` | 8 tasks: hammer, pick-place, bin-picking, coffee-push, door-open, peg-insert-side, peg-unplug-side, stick-push. They are identical except for the data paths. |
| `configs/rl/encoders/dynamo.yaml` | RL encoder config (`feature_dim` 256) |

Why not run `train.py`?

- It imports hydra, accelerate and wandb at module level, and builds env workspaces.
- Its resume is broken: `load_snapshot` reads `self._keys_to_save` (`train.py:335`), which only `save_snapshot`
  sets. Even when it works, it resumes only at the 10-epoch snapshots.

`training.py` therefore replicates its loop line by line. `tests/test_baselines_dynamo.py` runs the **verbatim**
`Trainer.train`, `Trainer.eval`, `Trainer.adjust_lr` and `Trainer.log_append` source, extracted from the pinned
`train.py` with `ast`. It is bound to a stand-in for the four accelerate calls they make: `autocast` as a null
context, `backward`, `clip_grad_norm_` and `gather_for_metrics`, with their single-process fp32 meaning. The test
compares it against our functions for three epochs (lr 0, warmup, cosine). Parameters, all three optimiser states
and the epoch logs are bitwise equal.

## Pretraining recipe

The paper's SimSiam configuration (Kitchen / Push-T / LIBERO share `configs/ssl/dynamo_*.yaml` exactly). File:line
references are to the pinned commit.

| Setting | Value | Source |
|---|---|---|
| Encoder | ResNet-18 trunk without `fc` (global average pool, 512-d), ImageNet `Normalize` inside, input RGB in [0, 1], random init, `unit_norm` false | `configs/encoder/resnet18_random.yaml:1-4`, `models/encoder/resnet.py:6-40` |
| SyncBatchNorm | on (`sync_bn: True`); one process, so it computes plain batch norm over the batch's N·T·V = 320 frames | `train_sim_kitchen.yaml:30`, `train.py:95-98` |
| Inverse dynamics (projector) | causal transformer, 6 layers, 6 heads, width 120, dropout 0, per view | `configs/projector/inverse_dynamics_sim_kitchen.yaml:4-8`, `models/projector/inverse_dynamics.py` |
| Latent dim Z (`projector.output_dim`) | **39**. Unspecified for Meta-World; chosen: the Meta-World observation size, following the README rule "approximately the underlying state dimension" (`README.md:132`). Kitchen uses 64, Push-T 8, LIBERO 32. | chosen |
| Forward dynamics | causal transformer on `cat(s_t, z_{t+1})`, 6 layers, 6 heads, width 120, dropout 0, output 512 | `configs/ssl/dynamo_sim_kitchen.yaml:5-9`, `models/ssl/dynamo.py:53-62` |
| Target | SimSiam: the online encoder, stop-gradient (`ema_beta: null`). `beta_scheduling` and `projector_use_ema` are true in the config but inert without EMA. | `dynamo_sim_kitchen.yaml:13-15`, `dynamo.py:93-101` |
| Loss | `1 - cos(f(s_t, z_{t+1}), sg(s_{t+1}))` × `dynamics_loss_coef` 1.0 / V, plus `covariance_reg_coef` 0.04 × mean of the d(d−1) squared off-diagonal entries of cov(s) | `dynamo.py:13-21, 103-172`, `dynamo_sim_kitchen.yaml:10-11` |
| Views | V = 1 per sequence (`separate_single_views: true`) | `dynamo_sim_kitchen.yaml:20`; see data below |
| Window | 5 frames | `train_pusht.yaml:14`, `train_libero_goal.yaml:14` (Kitchen: 2) |
| Batch | 64 windows | `train_*.yaml:12` |
| Optimisers | three AdamW: encoder (`train.py:99-104`), projector (`transformer_encoder.py:223-250`, decay on ≥2-D weights only), forward model (`dynamo.py:63-67`, same); lr 1e-4, betas (0.9, 0.999), weight decay 0, not fused | `train_sim_kitchen.yaml:21-25` (Push-T: wd 1e-6) |
| Gradient clipping | 0.1, separately for encoder, projector and forward model | `train_*.yaml:26`, `train.py:193-202` |
| Step order | backward (`retain_graph=True`), clip ×3, encoder step, projector step, `ssl.step()` (forward step + its zero_grad), zero_grad encoder/projector | `train.py:183-209`, `dynamo.py:185-191` |
| LR schedule | set once per epoch for the encoder and projector only: `1e-4 · epoch / 5` for epochs 0-4 (so **epoch 0 trains with lr 0**: only the forward model and the BatchNorm statistics move), then half-cycle cosine to 0 over epochs 5-39. The forward model keeps 1e-4 throughout. | `train_sim_kitchen.yaml:31-33`, `train.py:170-173, 371-384` |
| Length | 40 epochs | `train_*.yaml:33` |
| Seed / determinism | 42; `torch.use_deterministic_algorithms(True)`, `CUBLAS_WORKSPACE_CONFIG=:4096:8` | `train_*.yaml:27`, `train.py:36`, `utils/__init__.py:60-67` |
| Precision | fp32 (accelerate's default `mixed_precision="no"` for `python3 train.py`) | `train.py:32-35, 183` |
| Augmentation | none | `train.py:180-189` |
| Validation | every epoch, the full validation set, encoder and projector in eval mode; the epoch means of the loss terms, plus `obs_enc`/`obs_proj` std and norm | `train.py:222-273` |
| Data loader | `num_workers` 15, `pin_memory`, shuffle, no `drop_last` (the last partial batch is used) | `train_*.yaml:13`, `train.py:149-167` |
| Input resolution | native frames (no resize, no crop); ours are 128×128 | `models/encoder/resnet.py`, datasets |
| Frame spacing | **2 simulator steps**. Unspecified for Meta-World; chosen: the RL observation spacing (`action_repeat` 2), through the official slicer's `frame_step` (upstream configs use 1). | chosen |
| Sequences | **every training camera's video of every training episode is its own single-camera trajectory (V = 1)**. Unspecified for Meta-World; chosen (user decision). Upstream datasets have V = 1 (Kitchen, Push-T) or V = 2 views per trajectory (LIBERO, Block Push). | chosen |
| Split | train / validation episodes of `splits/<task>_seed0.json` (240 / 10). Unspecified; chosen: the shared manifest, so all cameras of an episode sit in the same split. Upstream uses a random 0.95 trajectory split. | chosen |

**Resulting length.** Upstream slicing uses one window per start frame with `span = (5−1)·2+1 = 9` frames and
`start in range(T − 9)`. The `range` drops the last complete window, an upstream off-by-one that we keep. Forty
epochs on our data is **more than ~150k steps on every task**, and is kept at 40 epochs as decided:

| Task | Training windows | Steps / epoch | Steps (40 epochs) | Validation windows |
|---|---|---|---|---|
| hammer | 311 880 | 4 874 | **194 960** | 12 048 |
| pick-place | 290 040 | 4 532 | 181 280 | 11 676 |
| bin-picking | 348 360 | 5 444 | 217 760 | 11 598 |
| coffee-push | 309 378 | 4 835 | 193 400 | 16 068 |
| door-open | 268 200 | 4 191 | 167 640 | 10 212 |
| peg-insert-side | 370 122 | 5 784 | 231 360 | 10 746 |
| peg-unplug-side | 387 894 | 6 061 | 242 440 | 16 800 |
| stick-push | 331 110 | 5 174 | 206 960 | 14 700 |

Windows start at every frame, so both frame parities are used. Subsampling each video to every second frame first
and then taking consecutive windows would halve these counts. We did not do that, because the official slicer
expresses frame spacing through `frame_step`.

## Deviations from the paper / official code, and why

1. **Data and split** (the three "chosen" rows above). Our Meta-World data has six training cameras per episode,
   and upstream has no Meta-World setting.
2. **accelerate removed** (patch 0001). For one process in fp32 it does nothing numerically: `prepare` is the
   device move, `autocast` a null context, `backward`/`clip_grad_norm_` the torch calls, and gathering the identity.
3. **Trainer re-implemented** (see above for why `train.py` itself cannot run). The step, the schedule (including
   upstream's lr 0 in epoch 0 and the unscheduled forward-model optimiser), the clipping, the validation and the
   epoch means are replicated and tested bitwise against the verbatim upstream methods.
4. **Shuffling.** Upstream draws a new permutation each epoch from the global torch RNG
   (`DataLoader(shuffle=True)`). Ours seeds each epoch's permutation by (seed, epoch). It is equally uniform, and a
   resumed run continues the exact order mid-epoch.
5. **Checkpoints, resume and exports.**
   - Upstream pickles whole modules every 10 epochs (`snapshot.pt`, `encoder_{epoch}.pt`). We write
     `checkpoints/latest.pt` every 2 000 steps and at every epoch end. It holds the state dicts of the three
     modules and three optimisers, the epoch position, the epoch log accumulators and all RNG states, so
     `--resume auto` is lossless (tested bitwise, including a resume mid-epoch).
   - `encoder.pt` is the latest encoder (state dict + constructor arguments + `frame_spacing` + `image_size`).
   - `exports/encoder_epochNN.pt` mirrors upstream's 10-epoch encoder files.
   - `encoder_init.pt` is the same encoder at the seed's random initialisation, for the pretrained-vs-random check.
     `--init-export-only` writes it without training.
6. **Logging.**
   - W&B project `splatter4d-baselines` through `init_wandb` (offline-capable via `WANDB_MODE`).
   - `metrics.jsonl` has 50-step means in addition to upstream's per-epoch means, which keep upstream's key names
     (`ssl_train/*`, `ssl_eval/*`, `metrics/*`).
   - `wandb.watch(encoder)` gradient histograms (`train.py:110`) are not recorded.
7. **No env-specific offline evaluation.** `workspace.run_offline_eval` (`train.py:212-220`, task-specific probes
   and rollouts) has no Meta-World counterpart. The RL protocol is the downstream evaluation.
8. **Equivalent-by-construction details.**
   - The input conversion is the CPU `obs / 255.0` value for value, via a lookup table on the GPU.
   - SyncBatchNorm is kept and computes plain batch norm in one process. The RL export loads into plain BatchNorm,
     and the outputs are identical (tested).
   - `retain_graph=True` is kept; it has no numerical effect.
9. **Model size.** The official encoder is kept unchanged (user policy). It is larger than SinCro's RL-time encoder;
   see below.
10. **Upstream quirks that do not apply here.** With an EMA target and `projector_use_ema: true` (Block Push and the
    `your_dataset` template), the online projector receives no gradient and the inverse model never trains. Our
    SimSiam setting does not use EMA.

## Privileged inputs

**None beyond RGB.**

- Pretraining sees only RGB frames of one camera at a time: no actions, states, proprioception, depth, camera
  parameters or camera identity.
- The six training cameras enter only as separate videos; there is no cross-view pairing.
- At RL time the encoder sees one RGB camera (the standard env, no observation dict), and the policy gets proprio
  like every method.

## RL side

- `vision.encoder_type: dynamo` (`configs/rl/encoders/dynamo.yaml`).
- `DynaMoFrameEncoder` loads the export into the official `resnet18`, frozen, in eval mode (BatchNorm uses the
  running statistics from pretraining, as upstream's downstream code does under `eval_mode`). It maps each RGB
  frame in [0, 1] to its 512-d feature.
- The shared `FrameMLPStackHead` (feature_dim 256) stacks the 3-frame history, as for ReViWo. Features are cached
  in replay as fp16, as for every frozen encoder.
- Random-init arm: `vision.export_path=<run>/encoder_init.pt`.
- The export records `frame_spacing` 2. The per-frame encoder does not depend on it at RL time, so the shared
  protocol's spacing check is not needed.

## Size (RL-time encoder)

Measured on the CPU with `torch.utils.flop_counter.FlopCounterMode` (multiply-add = 2 FLOPs) on one 128×128 frame.
`tests/test_baselines_dynamo.py::test_rl_time_encoder_size` asserts these numbers.

| | Parameters reached by the RL forward | FLOPs per frame | Raw output per frame |
|---|---|---|---|
| **DynaMo** | **11 176 512** (11.18 M; plus 9 620 BatchNorm buffer entries) | **1.184 G** (3.553 G per 3-frame observation) | **512** (1 536 per observation) |
| ours | 1.73 M | 0.215 G | |
| SinCro | 9.06 M | 1.79 G | 256 (stack level) |
| ReViWo | 1.68 M | 0.417 G | 1 024 |

The FLOPs are all convolutions. Pretraining-only modules, not used at RL time: inverse dynamics 1 113 279 and
forward dynamics 1 175 192 parameters. DynaMo's encoder has 1.23× SinCro's parameters, at 0.66× its FLOPs per frame.

## Cost

- **Smoke throughput:** 0.090 s/step on hammer with the full data (1 000 steps), on GPU-76871c16 (RTX PRO 6000
  Blackwell Max-Q). The GPU was otherwise idle.
- **Data and memory:** data wait 0.011 s/step with 15 workers. Peak 3.2 GB GPU memory allocated. Host memory is
  2.3 GB RSS for the main process and 3.5 GB PSS in total with the 15 training and 15 validation workers.
- **One validation pass:** 18 s (12 048 windows), i.e. about 12 min over 40 epochs.
- **Estimate: about 42 GPU-hours for the 8 tasks on a dedicated GPU of this type.**
  - Per task: hammer 5.1 h, pick-place 4.7, bin-picking 5.6, coffee-push 5.1, door-open 4.4, peg-insert-side 6.0,
    peg-unplug-side 6.3, stick-push 5.4.
  - A shared GPU will be slower; the step is GPU-bound (0.078 s of the 0.090 s is compute).
- **Disk per run:** `latest.pt` 162 MB, each encoder export 45 MB (`encoder.pt`, `encoder_init.pt`, four epoch
  exports).

## Smoke results (hammer)

- **A: full data, 1 000 steps** (inside the official epoch 0, so encoder lr 0). The training dynamics loss fell from
  0.251 (first 50 steps) to 0.0245 (last 50) as the forward model learnt. This run also measured the throughput.
- **B: first 8 training episodes, 1 000 steps.**
  - `dataset.max_episodes=8`, 142 steps per epoch, so the official schedule ran epochs 0-6 including the warmup,
    with validation each epoch on all 12 048 validation windows.
  - The **validation loss fell every epoch**: 0.0353, 0.0272, 0.0221, 0.0198, 0.0180, 0.0175, 0.0140.
  - There was no collapse: per-dimension feature std 0.159 → 0.133, mean norm about 20, covariance term 3e-5.
  - W&B ran offline.

Figures in `docs/figures/baselines/dynamo/` (from run B; numbers in `smoke.json`, `retrieval.json`, `panels.json`):

- `loss_curves.png`: training and validation dynamics loss, covariance term, and the two learning rates (the
  encoder's per-epoch warmup steps; the forward model constant).
- `nn_retrieval.png`: validation queries and their top-3 nearest training frames (same camera, cosine), trained vs
  random init.
  - Over 2 118 queries, the median top-1 state error is: hand 1.4 cm / hammer 1.8 cm (trained), 1.4 / 1.9 (random
    init), 15.2 / 16.1 (a random frame from the same camera).
  - So both encoders retrieve state-matched frames within a camera. After 1 000 steps the trained encoder is not
    yet distinguishable from its initialisation on this test. Within-camera pixel similarity is easy even for
    random ResNet features.
- `pca_episode.png`: the first two principal components along validation episodes. The trained features trace a
  smoother path that is more concentrated on PC1: 75 % / 68 % / 48 % of variance vs 55 % / 49 % / 28 % at random
  init.

**RL end to end:** `scripts/train_rl.py --task hammer --encoder dynamo --seed 2000` with run B's `encoder.pt`, 600
DrM steps (no evaluation).

- 77 fps; 1.15 GB GPU; 2.1 GB RSS.
- The 1M-step replay of 512-d fp16 features is preallocated at 1.02 GB.

## Tests

`tests/test_baselines_dynamo.py` (12 tests) covers:

- the patched tree (pinned and clean checkout; only the accelerate lines removed; no name leaks);
- the windows against the verbatim official `TrajectorySlicerDataset`, sample by sample, including the exact
  `obs / 255.0`;
- window spacing, cameras and split separation;
- the resumable sampler;
- the encoder against torchvision's ResNet-18 trunk;
- the objective against a hand-written formula (causality of the inverse model, stop-gradient target, gradients);
- the training step and schedule against the verbatim `Trainer` methods (bitwise);
- the configs against the official YAML files;
- the RL-time size;
- export/reload identity and the random-init export;
- DrM end to end with the frozen encoder (shapes, fp16 cache, eval-mode BatchNorm, frozen state);
- the entry point: exports, refusal without `--resume auto`, and bitwise lossless resume mid-epoch across epoch
  boundaries.

## Usage

```bash
CUDA_VISIBLE_DEVICES=<uuid> .venv/bin/python -I scripts/train_dynamo.py --config configs/baselines/dynamo/hammer.yaml \
    --name dynamo-hammer [--resume auto]
CUDA_VISIBLE_DEVICES=<uuid> .venv/bin/python -I scripts/dynamo_panels.py --run runs/pretrain/dynamo-hammer
scripts/train_rl.py --encoder dynamo --set vision.export_path=runs/pretrain/dynamo-hammer/encoder.pt        # pretrained
scripts/train_rl.py --encoder dynamo --set vision.export_path=runs/pretrain/dynamo-hammer/encoder_init.pt   # random init
```

- Runs that are not capped at 1 000 steps or fewer (`train.max_global_steps`) require a passing `docs/tests.json`
  for the current sources.
- A run directory with a checkpoint is only continued with `--resume auto`.
