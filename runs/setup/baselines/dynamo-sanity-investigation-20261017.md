# DynaMo hammer sanity failure: read-only investigation (2026-10-17)

## Findings

1. **No plumbing bug found.**
   - `encoder.pt` holds the final weights. Its tensors are bitwise equal to `exports/encoder_epoch39.pt` (max abs diff 0, step 194960). It differs from `encoder_init.pt`.
   - All three pretrained RL runs and their evals record export sha256 `a8f2e047…`, which is the current `encoder.pt`. They started 13:27 UTC, after pretraining finished at about 13:22 UTC.
   - All three random-init runs record `f73a4ec4…`. That hash is identical for `runs/pretrain/s4-dynamo-hammer-init/encoder_init.pt` and `runs/s4-dynamo-hammer-rem-b/encoder_init.pt` (max abs diff 0).
   - Preprocessing is the same in pretraining and RL:
     - 128x128 RGB from the same MuJoCo renderer and camera rig, with no flip in either path.
     - [0,1] scaling, with ImageNet normalisation inside the official module.
     - Per-frame encoder on one camera.
     - Frozen, in eval mode.
2. **The recipe is faithful.** Every deviation from the official SimSiam configs is already documented in `docs/baselines_dynamo.md`, and none is undocumented.
3. **The pretrained representation is partially collapsed and over-trained, relative both to random init and to its own epoch-9 state.**
   - Within one camera trajectory (what the policy sees), compared with random init:
     - effective rank 3.65 vs 10.66;
     - 90% of the variance in 3.7 vs 16.4 dimensions;
     - relative spread (std/norm) 0.00164 vs 0.00287;
     - frame-to-frame change 0.019 vs 0.039.
   - A linear probe of hand and hammer xyz (per camera) gives mean R² 0.673 for final, 0.794 for random init, and 0.835 at epoch 9.
   - So pretraining helped at first (epoch 9 beats random init), then lost state information. Training loss kept falling (0.00076 → 0.00041) while held-out loss rose (minimum about 0.00046 at epochs 15–24, 0.0012–0.0013 from epoch 31 on).
4. **The RL head does not explain it through feature scale.**
   - `FrameMLPStackHead` (Linear → LayerNorm → ReLU ×2, stack of 3, Linear → LN → Tanh) removes the absolute scale.
   - The random-init features are actually *more* constant across samples: mean pairwise cosine 0.990 vs 0.902. After the first Linear+LN, across-sample variance is 1.0% of the activation energy for random init and 10% for final.
   - fp16 caching error (3e-5) is about 200x below the within-trajectory per-dimension std (0.008).
   - The deficit is information and dimensionality, not scale.
5. **Verdict:** a faithful-but-weak (over-trained, low-rank) DynaMo result, not an implementation bug.
   - The effect is consistent across seeds: random beats pretrained on every seed, by 0.12 / 0.115 / 0.265. A paired t-test gives t ≈ 3.4 (df 2, p ≈ 0.08), so it is borderline significant.

**Recommended next step (no retuning):**
- Accept the result as faithful, and keep the official final-epoch encoder for the eight tasks, as pre-registered.
- Record the evidence above in the experiment log.
- Optional: a clearly labelled diagnostic-only hammer arm on `exports/encoder_epoch09.pt` (3 seeds) to confirm the over-training mechanism. It must not replace the official final encoder, because choosing a checkpoint is a recipe choice.

## Evidence

### 1. Export and load plumbing
- Export writer: `scripts/train_dynamo.py`.
  - `save()` writes `encoder.pt` after `latest.pt` every 2000 steps and at the end (lines ~150-170, and the final `save()`).
  - `encoder_init.pt` is written before any update.
  - Format is in `s4d/baselines/dynamo/model.py::save_export` and `load_export`. `load_export` rebuilds the official `resnet18`, uses a strict `load_state_dict` and returns `eval().requires_grad_(False)`.
- `completion.json`: 194960 steps, 40 epochs, 4874 steps/epoch. `checkpoints/latest.pt` was not pulled back from remote, so the comparison was made against `exports/encoder_epoch39.pt` (step 194960): bitwise equal.
- Final vs init:
  - median relative change of the conv weights is 0.78;
  - conv weight norm ratios are 1.03-1.64;
  - the last BN `running_var` mean is 66.9 (init 1.0; the init has untrained BN statistics, mean 0 and variance 1).
- RL configs (`runs/bxs-dynamo-hammer-*/config.yaml:52-53`): `encoder_type: dynamo`.
  - Pretrained arm: `export_path: runs/s4-dynamo-hammer-rem-b/encoder.pt`.
  - Random arm: `runs/pretrain/s4-dynamo-hammer-init/encoder_init.pt`.
  - `image_size 128`, `frame_stack 3`, `feature_dim 256`.
  - `provenance.json` gives the hashes listed in Finding 1.
- `experiments/queue.yaml` (around lines 18681-18800) names the export `runs/pretrain/s4-dynamo-hammer/encoder.pt`, but the remote `-rem-b` jobs resolved it to the correct `-rem-b` export, as the hashes confirm.
- `bxs-dynamo-hammer-rand-s1000` ran locally at commit 852c71e; the other runs used e4206f1. `git diff 852c71e e4206f1 -- s4d/rl s4d/baselines/dynamo scripts/train_rl.py` is empty.
- RL path:
  - `s4d/rl/encoders.py::VisionEncoderAdapter`: frozen backbone, `eval().requires_grad_(False)`, `extract_cacheable_feature` with `obs/255` and per-frame reshape to `(B*3, 3, H, W)`, fp16 cache.
  - `s4d/baselines/dynamo/rl.py::DynaMoFrameEncoder` calls the official module, which applies the ImageNet `Normalize` (`third_party/dynamo/models/encoder/resnet.py:17-29`).
  - `set_update_mode` keeps the backbone in eval mode.
  - `s4d/rl/agent.py:208-209`: frozen features get no augmentation, the same for both arms.
- Pretraining input: `s4d/baselines/dynamo/data.py` reads HDF5 `rgb (T,10,128,128,3)` and permutes to CHW; `to_official_obs` = `/255`.
- RL render: `s4d/rl/env.py:175-181` uses `mujoco.Renderer.render()` with an HWC→CHW transpose. The collector uses the same renderer (`s4d/data/metaworld/collect.py:187-206`).
  - Camera conventions match: `orbit_camera` gives azimuth=phi and elevation=-theta, the same as the HDF5 camera table (train azimuth 0/180/-30/210/30/150, elevation -45/-45/-60/-60/-30/-30).
  - `tests/test_rl.py:349 test_env_frames_match_collector_rig…` covers this.

### 2. Recipe fidelity (`configs/baselines/dynamo/hammer.yaml` vs `third_party/dynamo` @ 6287403)
These match the official configs:
- ResNet-18, random init, 512-d, `unit_norm` false (`configs/encoder/resnet18_random.yaml`).
- Projector 6 layers / 6 heads / width 120 / dropout 0.
- Forward model 6/6/120, dropout 0.
- Covariance coefficient 0.04, dynamics coefficient 1.0.
- `ema_beta: null` (SimSiam; kitchen, push-t and libero ssl configs).
- `separate_single_views` true.
- lr 1e-4, betas (0.9, 0.999), weight decay 0 (kitchen/libero), clip 0.1.
- batch 64, window 5 (push-t/libero), warmup 5, 40 epochs, cosine schedule (kitchen/push-t), sync_bn, seed 42.
- No augmentation.

The training step and schedule are tested bitwise against the verbatim `Trainer` (`tests/test_baselines_dynamo.py:315`).

Deviations, all documented in `docs/baselines_dynamo.md` ("Pretraining recipe" table and "Deviations" list):
- `projector.output_dim` 39 (chosen).
- `frame_step` 2 (chosen; upstream 1).
- Each training camera's video is its own V=1 trajectory (chosen).
- Split from the shared manifest, 240/10 episodes (upstream: random 0.95).
- 128x128 native resolution.
- accelerate removed (patch 0001, numerically inert).
- Trainer re-implementation.
- Per-(seed, epoch) shuffling.
- Checkpoint and export cadence.
- No env-specific offline eval.
- The `your_dataset` template's `ema_beta` 0.99 is not used; this choice is justified (doc item 10, and following the paper's SimSiam configs).
- `docs/BASELINES.md:17` summarises the same choices.

### 3. Feature health (CPU, official module, eval mode; HDF5 read-only)
Data:
- Validation: all 10 validation episodes × 6 training cameras, every 2nd frame, 6312 frames in 60 trajectories.
- Probe training set: 40 training episodes × 6 cameras, every 4th frame, 15750 frames.

Within-trajectory statistics are means over the 60 validation trajectories. The probe is per-camera ridge regression (standardised features, best λ of {1, 10, 100, 1000}) on hand xyz and hammer xyz from `obs[0:3]` and `obs[4:7]`, fit on training episodes and tested on validation episodes.

| encoder | per-dim std | norm | std/norm | eff. rank | top-1 share | k90 | frame-to-frame Δ/‖f‖ | probe R² (mean) |
|---|---|---|---|---|---|---|---|---|
| init (random arm) | 0.0390 | 13.56 | 0.00287 | 10.66 | 0.451 | 16.4 | 0.0390 | 0.794 |
| epoch 9 | 0.0487 | 11.29 | 0.00432 | 3.07 | 0.656 | 3.0 | 0.0381 | **0.835** |
| epoch 29 | 0.0111 | 5.54 | 0.00200 | 3.36 | 0.639 | 3.2 | 0.0227 | 0.703 |
| final (pretrained arm) | 0.0079 | 4.82 | 0.00164 | 3.65 | 0.625 | 3.7 | 0.0190 | 0.673 |

- Per-coordinate probe R², final vs init: hand [0.46, 0.91, 0.26] vs [0.71, 0.97, 0.42]; hammer [0.67, 0.94, 0.81] vs [0.84, 0.98, 0.85].
- Probe pooled over cameras (best λ): final 0.48, init 0.54, epoch 9 0.66.
- The per-dim std of 0.0079 matches the logged `metrics/obs_enc_mean_std` of 0.0104. Validation batches are consecutive windows of one trajectory, so the log measures within-trajectory spread, and so does this table.
- Across mixed cameras and episodes (192 validation frames), the final encoder is less constant than init: mean pairwise cosine 0.902 vs 0.990, effective rank 9.3 vs 19.8, top-1 singular-value share 0.27 vs 0.35. No dimension is dead (std < 1e-6) in either.

Interpretation: the collapse is partial.
- Most of the spread that remains separates cameras and scenes. Within an episode, only about 3-4 directions vary.
- The hand and hammer positions that the policy needs are decoded worse than from random features.
- Epoch-wise metrics (`metrics.jsonl`):
  - training total loss falls monotonically (0.00076 at epoch 5 → 0.00041 at epoch 39);
  - held-out dynamics loss has its minimum around 0.00046 (epochs 15, 19, 24), rises to 0.00095-0.00136 from epoch 30 on, and ends at 0.00119;
  - covariance loss falls to 5e-8;
  - the embedding norm falls 19.1 → 4.8.
- The covariance term (off-diagonal covariance only, no variance hinge) shrinks as the scale shrinks, and the cosine dynamics loss is scale-invariant. So the objective as written does not oppose low-rank, shrinking embeddings. This is upstream behaviour, not our code.

### 4. RL-side scale handling
- `FrameMLPStackHead` (`s4d/rl/encoders.py:83-120`): per-frame Linear(512→256) → LayerNorm → ReLU → Linear(256→256) → LN → ReLU. The 3 frames are concatenated, then Linear(768→256) → LN → ReLU → Linear → LN → Tanh. Orthogonal initialisation. The same head is used for both arms.
- LayerNorm makes the absolute scale (4.8 vs 13.6) irrelevant. Adam also adapts to input scale.
- With a random orthogonal first layer, across-sample variance as a share of post-LN activation energy is 0.010 for init and 0.101 for final (192 mixed validation frames). The pretrained features are therefore *not* disadvantaged by being near-constant relative to their mean; random features are more so, and still did better.
- fp16 caching error is 3.4e-5, against a within-trajectory per-dim std of 0.008.
- So low variance is not plausibly the cause through scaling or quantisation. What plausibly hurts is the reduced within-episode dimensionality and state information (probe R² 0.67 vs 0.79).

### RL outcome detail
Last-5 training-camera success per camera, mean over 3 seeds:

| arm | train0 | train1 | train2 | train3 | train4 | train5 |
|---|---|---|---|---|---|---|
| pretrained | 0.22 | 0.52 | 0.19 | 0.43 | 0.33 | 0.41 |
| random | 0.32 | 0.77 | 0.41 | 0.74 | 0.37 | 0.48 |

- Held-out cameras: pretrained 0.004, random 0.063.
- Per-camera results are strongly bimodal (0 vs 0.9) within seeds.
- Seed-paired differences (random − pretrained): +0.12, +0.115, +0.265.
