# Stage 4 baselines: SinCro and ReViWo

Both baselines are pretrained per task on our Meta-World data and then used as frozen encoders in DrM. The rule is
**upstream code, original hyperparameters, changes only at the data interface**. This file lists every place where
that rule bends, and why.

## Sources

| What | Where | Pinned commit |
|---|---|---|
| SinCro model code | github.com/sunho001215/sincro (MIT, © 2025 Seungyeon Yoo) | `5c7c06d373233e7bc7d7c240fdc27b212e2535f4` |
| ReViWo model code | github.com/lafmdp/ReViWo (**no license stated**) | `ac0c24958c83366dbdceebfb1db9f9111fce56a0` |
| Reference trainers, configs, RL wrappers | sunho001215/splatter_vae, `baselines/{SinCro,ReViWo}/`, `agents/common/encoders.py` | `c0abf56` |

Vendored files, all verbatim except for their imports; each file's header names its source file:

- `s4d/baselines/sincro/`: `vit_block.py` (from `ViT_block.py`), `mae_encoder.py` (from `MV_mae_encoder.py`),
  `nerf_helpers.py` (from `run_nerf_helpers.py`), `nerf.py` (from `MV_run_nerf.py`).
- `s4d/baselines/reviwo/`:
  - `stransformer.py` (from `common/models/modules/stransformer.py`);
  - `multiview_vae.py` (from `common/models/multiview_vae.py`);
  - `utils.py` (from `common/utils.py`);
  - `configs.py` (`CodebookConfig` and `STTransConfig` from the reference `baselines/ReViWo/transformer.py`).
- `s4d/baselines/{sincro,reviwo}/training.py`: verbatim excerpts of the reference `baselines/*/train.py`:
  - SinCro: config dataclasses, `SimpleArgs`, `update_learning_rate`, `encode_sincro`, `forward_sincro_batch`,
    `render_full_image`;
  - ReViWo: `ReViWoTrainConfig`, `compute_reviwo_loss`.

Only definitions that the trainers and RL wrappers call are kept. The following upstream imports are dropped because
the kept code never uses them:

- SinCro: `load_blender`, tensorboard, `imageio`, `tqdm`.
- ReViWo: `tqdm`.

Star imports became explicit imports. Two names had to be re-added after trimming: `os` (imported at the top
of upstream `MV_run_nerf.py`), and `ndc_rays`, which `render` reached through `from .run_nerf_helpers import *`.

The vendored files are excluded from ruff (see `pyproject.toml`), so their code stays byte-identical to upstream.
Only the headers and imports differ.

New dependencies are pure-Python wheels, pinned in `pyproject.toml` and `uv.lock`:

- `einops==0.8.1`, used by SinCro;
- `scikit-learn==1.6.1`, needed for ReViWo's k-means codebook initialisation (it brings in `joblib` and
  `threadpoolctl`).

## The reference's lost SinCro edits

The reference trainer imports `create_nerf, render, img2mse, mse2psnr, get_rays, get_rays_np` from
`baselines.SinCro.sincro.MV_run_nerf`, and `MaskedViTEncoder` from `baselines.SinCro.sincro.MV_mae_encoder`. That
directory is empty in the reference commit; the uncommitted copy is lost.

Checked against the pinned SinCro commit, the edits needed only the following:

1. **Make the files importable as a package.** Upstream already uses relative imports
   (`from .run_nerf_helpers import *`, `from .ViT_block import ...`).
2. **Keep the module-global `device`**, which the trainer overwrites (`sincro_nerf.device = device`). Upstream
   defines it at module level.
3. **Keep the upstream signatures.** `create_nerf(args, basedir, expname)` accepts the trainer's `SimpleArgs`.
   `MaskedViTEncoder.SinCro_image_encoder(x, mask_ratio, T, is_ref)` and `SinCro_state_encoder(latent, ref_latent,
   mask, ids_restore)` match what `encode_sincro` and the reference RL wrapper call. `render(..., latent=, args=)`
   matches `forward_sincro_batch`.

No functional change was required, so none was made. The tests run `create_nerf`, `forward_sincro_batch` and the
full validation render on the vendored code.

## Deviations from the reference (and why)

1. **SinCro scene bounds are applied: `near=0.1`, `far=2.5` from the reference config.**
   - Upstream `MV_run_nerf.train()` adds `near`/`far` to the render kwargs. The reference trainer never does, so it
     rendered with `render()`'s defaults `near=0, far=1` and the config's bounds were dead.
   - Measured on hammer (6 training cameras, 5 episodes): z-depth (which equals NeRF's `t`, since `rays_d` has
     camera z = −1) has median 0.93 m and p99 3.67 m. With far=1, **39.5 % of pixels lie behind the far plane**; with
     far=2.5, 2.7 % do (distant background).
   - Upstream's own Meta-World settings were `near=0.02258, far=3.0`. We keep the reference config's 0.1/2.5.
   - **Decision for the parent:** keep 0.1/2.5 (current), or use upstream's far=3.0 (2.0 % beyond).
2. **SinCro frames are 2 simulator steps apart.** The reference used consecutive frames. This matches the RL
   observation (3 frames, `action_repeat` 2), as decided by the user. `frame_spacing` is part of the config's
   `data_interface`, is written into the export, and is checked against `env.action_repeat` when an RL config is
   resolved (`s4d/rl/protocol.py`).
3. **Train/validation split.**
   - The saved episode-level manifest (`splits/<task>_seed0.json`: 240 train, 10 validation episodes) replaces:
     - SinCro: the reference's `random_split` of *windows* (0.96/0.04), which leaks validation windows from training
       episodes;
     - ReViWo: its seeded episode shuffle (0.96).
   - `train_ratio` therefore has no effect. ReViWo's `dataset.seed` still seeds the run, as in the reference, but
     no longer shuffles episodes.
   - Only the 6 training cameras are used (camera table rows with `is_train`; the adapter asserts they come first).
4. **Outputs, logging and resume.**
   - Runs go to `runs/pretrain/<name>/`, with `config.yaml`, `metrics.jsonl`, `eval/*.png` and `checkpoints/latest.pt`.
   - Checkpoints store model, optimizer, step, epoch and all RNG states. A rerun with the same `--name` always
     resumes from `latest.pt`.
   - The reference's `train.ckpt_dir`, `train.resume_from_last` and `experiment` keys are dropped from the configs.
   - W&B project `splatter4d-baselines`.
   - Logging cadence (`i_print`, `log_every`), validation cadence (`eval_every`) and checkpoint cadence
     (`save_every`) are the reference values.
   - SinCro validation is the reference `run_validation`: one random validation window, unmasked, all views rendered,
     PSNR, and a ground-truth/render grid PNG.
   - ReViWo validation computes the validation losses on one batch and saves the reference `visualize` grid.
5. **Seeding.** SinCro seeds Python `random` in addition to the reference's torch and numpy seeds, so a resumed run
   restores every RNG it uses.
6. **Exports for RL** (not in the reference). `encoder.pt` is written at every save:
   - SinCro: the `MaskedViTEncoder` weights, `model_cfg` and `frame_spacing`;
   - ReViWo: the whole model plus its `reviwo` config and `img_size`.

   Loading rebuilds exactly the trainer's modules (`strict=True`). The reference RL wrapper built the SinCro encoder
   through `create_nerf` and discarded the NeRF; we build the same `MaskedViTEncoder` with the same arguments
   directly.
7. **RL wrappers** (`s4d/rl/encoders.py`) follow the reference `SinCroSceneEncoder` / `ReViWoInvariantEncoder`.
   - **SinCro:** the primary view is repeated as the reference views. The output is the fused state of the newest
     step, `(B, 256)`, fed to `SmallPostEncoderMLPHead`.
   - **ReViWo:** input `x*2-1`, output `z_l.flatten(1)` per frame, fed to `FrameMLPStackHead` over the 3 frames.
   - Two engineering-only changes:
     - SinCro's positional encodings are plain tensors upstream (with a `device` attribute). They are re-registered as
       non-persistent buffers and the attribute is kept in sync, so `.to(device)` works. The same tensors are
       re-registered, so values are unchanged.
     - ReViWo's k-means codebook initialisation is disabled after loading, as in the reference wrapper.
   - Both encoders are frozen with no augmentation. DrM's perturbation never touches them; the end-to-end test checks
     the state dict is bit-identical.
8. **Reference config keys with no effect** (unchanged in the configs):
   - ReViWo `min_time_gap` is unused by the reference trainer too (it says so).
   - `max_episodes`/`num_episodes` (null) and `max_frames_per_demo` (3000 / null) are implemented, but never bind on
     Meta-World's ≤ 500-step episodes.
   - `i_img`, `val_vis_nrow` and `val_random_sample` are SinCro display options.

## Data interface (`s4d/baselines/data.py`)

| | Reference sample | Our HDF5 → adapter |
|---|---|---|
| SinCro `images` | `(T, H, V, W, 3)` float in [0, 1] | `episodes/epXXX/rgb` `(T, 10, H, W, 3)` uint8, training cameras `[:6]`, frames `start + k·2` |
| SinCro `K` | principal point `(W-1)/2` (integer pixel centres, as `get_rays` assumes) | ours is continuous (`W/2`): `cx, cy −= 0.5` |
| SinCro `c2w` | MuJoCo/OpenGL camera frame (−z forward, +y up) | ours is OpenCV: `c2w_gl = c2w_cv @ diag(1, −1, −1, 1)` |
| SinCro windows | starts every `temporal_stride=3` frames | same, within one episode |
| ReViWo `images` | `(V, 3, H, W)` in [−1, 1], one timestep | same, every timestep of every split episode |

The camera conversion is verified in `tests/test_baselines.py`. A world point on SinCro's own `get_rays` ray for
pixel (i, j) projects, with our OpenCV `K`/`w2c`, to (i+0.5, j+0.5) for every training camera. The rig's look-at
point projects to the image centre in both conventions, in front of the camera in both.

## Hyperparameters (unchanged from the reference configs)

There are 8 tasks: door-open, hammer, peg-unplug-side, stick-push, pick-place, peg-insert-side, shelf-place and
bin-picking.

- The reference has configs for the first four. All its task configs are identical except for names and paths, so
  the other four use hammer's.
- `configs/baselines/{sincro,reviwo}/<task>.yaml` change only the dataset path, the W&B project and the
  `data_interface` block, and drop the output-location keys.

**SinCro**

| Group | Values |
|---|---|
| Steps and batch | 500 001 steps, batch 8 windows × 3 steps × 6 views, `N_rand` 2048 rays/step |
| Optimizer | Adam, lr 5e-4 decayed ×0.1 per 500k steps |
| NeRF | 8×256 coarse + fine, 64 + 128 samples, `multires` 10/4, `use_viewdirs` |
| Encoder | ViT on 128² with patch 16, embed 256, depth 4, 4 heads, MLP 1024 |
| State decoder | depth 2, 2 heads, output 256 |
| Masking and views | `mask_ratio` 0.75, 2 reference views |
| Contrastive term | margin 0.2, weight 0.0004 |
| Ray sampling | precrop 2000 steps at 0.5 |
| Bounds | near 0.1, far 2.5 |

**ReViWo**

| Group | Values |
|---|---|
| Steps and batch | 100 001 steps, batch 8 states × 6 views, 128² images, patch 16 |
| Optimizer | Adam, lr 3e-4 |
| Transformers | view encoder, latent encoder and decoder each have 8 layers, 8 heads, `n_embed` 128, dropout 0.1 |
| Codebooks | latent 512 × 16, view 64 × 16, β 0.25, k-means initialisation |
| Loss | `Weighted_MSE` |
| Loss weights | VQ 0.25; shuffled v/l/vl 2.0; latent consistency 0.5 and contrastive 0.5; view consistency 0.1 and contrastive 0.1; temperature 0.25; lower bound 0.9 |

## Measured cost

Setup: 50-step smoke runs on hammer with the reference configs (`--set train.max_global_steps=50`), on
GPU-d09f0338 (RTX PRO 6000 Blackwell Max-Q). The GPU was **shared**: other jobs kept it at about 95 % utilisation, so
dedicated-GPU times should be lower. Step time is the mean over steps 10–40; step 0 (about 5 s) includes start-up.

| | s/step | Peak GPU memory (allocated) | Full run (reference steps) | 8 tasks |
|---|---|---|---|---|
| SinCro | 0.53 | 7.8 GB | 500 001 steps ≈ 74 h, plus validation 500 × 6.2 s ≈ 0.9 h | ≈ 25 GPU-days |
| ReViWo | 0.58 | 1.7 GB | 100 001 steps ≈ 16 h | ≈ 5.4 GPU-days |

- **ReViWo is compute-bound.** Data loading takes 0.014 s/batch. The reference loss launches many small kernels;
  `normalize_tensor` loops over the batch in Python. Its small memory footprint lets it share a GPU.
- **Run length is set by steps.** The reference stops at `max_global_steps` whenever it is set; `num_epochs` applies
  only when it is null. On hammer's train split:
  - SinCro has 17 806 windows, so 2 225 steps per epoch and about 225 epochs;
  - ReViWo has 54 140 states, so 6 767 steps per epoch and about 15 epochs.
- **Seeded runs repeat.** A second seeded run reproduced step 0's loss exactly.

## Usage

```bash
CUDA_VISIBLE_DEVICES=<uuid> .venv/bin/python -I scripts/train_sincro.py --config configs/baselines/sincro/hammer.yaml --name sincro-hammer
CUDA_VISIBLE_DEVICES=<uuid> .venv/bin/python -I scripts/train_reviwo.py --config configs/baselines/reviwo/hammer.yaml --name reviwo-hammer
```

Runs longer than 1000 steps require a passing `docs/tests.json` for the current sources, and long runs go through
`scripts/jobs.py`.

For RL, set the encoder config and its export path:

- `scripts/train_rl.py --encoder sincro --set vision.export_path=runs/pretrain/sincro-hammer/encoder.pt`;
- `scripts/train_rl.py --encoder reviwo --set vision.export_path=runs/pretrain/reviwo-hammer/encoder.pt`.

The encoder configs are `configs/rl/encoders/{sincro,reviwo}.yaml`, with `feature_dim` 256.
