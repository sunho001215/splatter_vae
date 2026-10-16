# Baseline: 3D-MVP (reimplemented on RVT-1)

3D-MVP (Qi, Mo, Blukis, Fouhey, Fox, Goyal: "3D-MVP: 3D Multiview Pretraining for Manipulation", CVPR 2025,
arXiv 2406.18158) pretrains RVT's multi-view transformer as a masked autoencoder on point clouds rendered into RVT's
five virtual views. **No official code exists** (the project page still says "coming soon"), so the 3D-MVP parts are
reimplemented from the paper, on top of the official RVT code.

**Labels:**

- **Input:** "single RGB-D + extrinsics (privileged)". At RL time the encoder uses the current camera's metric depth
  and its K / c2w.
- **Implementation:** "reimplemented".

## Sources and license

| What | Where | Pinned commit | License |
|---|---|---|---|
| RVT (RVT-1 MVT, renderer conventions) | github.com/NVlabs/RVT, submodule `third_party/rvt` | `367995a1a2169b6352bf4e8b0ed405890462a3a0` | NVIDIA Source Code License (non-commercial) |
| 3D-MVP method | arXiv 2406.18158 v1 + v2 (CVPR version) | - | paper only |

RVT's nested submodules (PyRep, RLBench, YARR, PerAct) are not initialised and not needed.

### Imported, patched or wrapped

| File | Status | Contents |
|---|---|---|
| `rvt/mvt/attn.py` | **imported unmodified** | `PreNorm`, `Attention`, `FeedForward`/`GEGLU`, `Conv2DBlock`, `DenseBlock`, `cache_fn`: every layer of E and of the MAE decoder |
| `rvt/mvt/utils.py` | **imported, patched** | `place_pc_in_cube`. Patch `third_party/patches/rvt/0001-mvt-utils-drop-unused-matplotlib-import.patch` only removes the unused top-level `import matplotlib.pyplot` (matplotlib is not in our stack). |
| `s4d/baselines/mvp3d/rvt.py` | ours | loads both files by path from `s4d.third_party.source_tree("rvt")` under unique module names (no RVT package on `sys.path`) |
| `s4d/baselines/mvp3d/rvt_derived.py` | **modified RVT code, NVIDIA header kept** | see below |
| `renderer.py`, `model.py`, `data.py`, `rl.py`, `size.py`, `viz.py` in `s4d/baselines/mvp3d/` | ours | renderer, 3D-MVP parts, data adapter, RL backbone, size measurement, panels |
| `scripts/train_mvp3d.py`, `scripts/mvp3d_figures.py`, `scripts/mvp3d_size.py` | ours | trainer, figure panels, size table |
| `configs/baselines/mvp3d/<task>.yaml` (8 tasks), `configs/rl/encoders/mvp3d_{multiview,single}.yaml` | ours | configs |
| `s4d/rl/encoders.py` | one line | `"mvp3d"` entry in `BACKBONES` (lazy import of `s4d.baselines.mvp3d.rl`) |

`rvt_derived.py` holds the modified RVT code:

- `MVTEncoder` is `MVTSingle` (`rvt/mvt/mvt_single.py`) up to its attention layers. It keeps MVTSingle's attribute
  names: `input_preprocess`, `patchify`, `pos_encoding`, `fc_bef_attn`, `layers`.
- The within-view / cross-view layer loop (`self_cross_ver=1`) and the per-view max-pool, both from
  `MVTSingle.forward`.
- `pixel_loc`, from `MVTSingle.__init__`.
- The depth normalisation, from `PointsRendererWithDepth.forward`.
- The bounds crop, from `rvt_utils.move_pc_in_bound`. It returns a mask instead of removing points, so batches keep
  a fixed size.
- The channel order of `MVT.render`.

Not used:

- RVT's PyTorch3D renderer and RVT-2's CUDA point renderer (neither can be built here; see "Renderer");
- RVT's action head (`fc_aft_attn`, up-sampling, heatmaps, `feat_fc`), language and proprio inputs, and the agent.

## What is implemented (paper section)

- **R, the renderer (§3.2 Eq. 2; RVT-1 §3).** Our posed RGB-D images are lifted to a point cloud with our K and OpenCV
  c2w (`s4d.geometry.lift_depth`), cropped to the scene bounds and placed in the cube [-1, 1]^3 by those bounds. RVT's
  config uses `place_with_mean: False`, so `place_pc_in_cube` is imported and called with
  `with_mean_or_bounds=False`. The cloud is rendered into RVT-1's 5 orthographic views. Each view has 10 channels:
  - xyz in cube coordinates (3);
  - RGB in [0, 1] (3);
  - depth (1): the view-space z of the front point, mean-centred per view, with -1 at empty pixels;
  - RVT's constant pixel-location code (3).

  Empty pixels have xyz = RGB = 0.
- **E, the encoder (§3.2, Eq. 3/4).** RVT-1's MVT with image tokens only:
  - a 1x1 conv from 10 to 64 channels, then the patchify conv (kernel = stride = patch) with GroupNorm;
  - a learned positional embedding, then `fc_bef_attn` to the width;
  - 8 PreNorm attention + GEGLU x4 MLP layers: the first 4 within each view, the last 4 across all views.

  Its output z has 5N tokens x width.
- **Masking (§3.2; Table 2 / §4.4 "RGB").**
  - In each view of each sample, an independent random 75 % of the patches (tokens) is masked:
    L - int(0.25 L) of L, as in MAE's `random_masking`.
  - In masked patches only the RGB channels are replaced, by 0. xyz, depth and pixel location are kept.
- **D_MAE, the decoder (§3.2, Eq. 5).** A lightweight multiview transformer of RVT layers: 2 layers (1 within-view +
  1 cross-view), 8 heads, the encoder's width. Then a final LayerNorm and a linear head predicting each patch's
  RGB-D pixels.
- **Loss (Eq. 6).** L = 1/(5WH) sum_i sum_p ||I_i(p) - I~_i(p)||^2 over all pixels of all 5 views. The squared norm
  runs over the 4 reconstructed channels (RGB + depth); the loss is averaged over the batch.

## Pretraining recipe

"Paper" means arXiv 2406.18158 §3.2 "Implementation details" (v1 adds that the decoder has the encoder's hidden
size). RVT file:line refers to the pinned commit.

| Setting | Value | Source |
|---|---|---|
| Data | our in-domain Meta-World data, one task at a time, training-split time steps (`splits/<task>_seed0.json`) | **user decision** = the paper's in-domain ablation "pretrain on RLBench point clouds" (§4.4, Table 2: 67.5 vs 67.6 with Objaverse). Objaverse is not used. |
| Point cloud of a sample | **all 6 training cameras** of one time step (RGB-D + extrinsics), fused into one cloud; no random camera subsets, no single-camera clouds | **user decision (2026-10-14)**: the original in-domain fused multi-camera setting (RVT/3D-MVP fuse a fixed camera rig; RLBench uses 4). One pretraining per task; the same encoder serves both RL conditions. |
| Lifting | our K (continuous pixel coordinates), OpenCV c2w, metric depth (`depth > 0`) | our data contract (`s4d/geometry.py`) |
| Scene bounds | x [-0.6, 0.6], y [0.15, 1.15], z [-0.05, 0.75] m (RVT order `[x_min, y_min, z_min, x_max, y_max, z_max]`), all 8 tasks | **unspecified for our data**; derived from the data (see below). RVT uses its RLBench `SCENE_BOUNDS` (`utils/peract_utils.py:16`). |
| Cube placement | by the bounds; scale 2 / 1.2 m | RVT `configs/rvt.yaml:15` `place_with_mean: False`; `mvt/utils.py:16` |
| Views | top, front, back, left, right; orthographic, distance 1 | RVT `mvt/renderer.py:40` `get_cube_R_T`; paper §3.1 |
| Channels | xyz (cube), RGB [0, 1], depth (mean-centred, -1 empty), pixel location | RVT `mvt/mvt.py:219,245,262`, `renderer.py:263-270`, `mvt_single.py:205-216`. The paper's §3.1 lists "3D point coordinate in camera sensor frame" for the last 3 channels. We follow RVT-1's code, where these channels are the pixel-location code. |
| Render radius | 0.012 NDC | RVT `renderer.py:319` (BoxRenderer), RVT-2 `RVTBoxRenderer` |
| Resolution / patch | **48 px / 12** (paper 220 / 10) | size-matched by user decision (see "Size") |
| Width, heads | **256 = 8 heads x 32** (paper hidden 1024, 8 heads) | size-matched by user decision. The paper's 1024 is read as attn_dim = heads x dim_head = 1024 (RVT's convention: 512 = 8 x 64). |
| MLP | GEGLU, x4 (hidden 1024) | RVT `attn.py:77` `FeedForward(dim, mult=4)` |
| Layers | 8 = 4 within-view + 4 cross-view | paper (8 layers); RVT `self_cross_ver=1` (`mvt/config.py:28`, `mvt_single.py:496`) |
| Conv stem | 64 channels, lrelu, GroupNorm(4) in the patchify conv | RVT `mvt/config.py:18,22`, `mvt_single.py:221,244` |
| Language / proprio tokens | none | 3D-MVP pretrains E on scenes without language or robot state (§3.2: E maps the virtual images to z). RVT-1's config adds both for action training. |
| Positional embedding | learned, `torch.randn` init, image tokens only | RVT `pe_fix=True`, `mvt_single.py:191` |
| Attention dropout | 0.1 | **unspecified**; RVT-1 default `mvt/config.py:24` |
| Mask ratio | 0.75 per view | paper |
| Masked channels | RGB only | paper Table 2 / §4.4 (best strategy "RGB") |
| Mask fill value | 0 | **unspecified**, chosen 0: the RGB of empty pixels. MAE drops masked tokens, but with geometry visible every token stays in E. |
| Reconstruction target | RGB + depth of all 5 views (the clean render) | paper §3.2 ("reconstruct the multiview RGB-D images") |
| Loss | Eq. 6, all pixels | paper Eq. 6 |
| Decoder | 2 layers, 8 heads, width = encoder width (256), within/cross split 1 + 1 | paper (layers, heads), v1 (width), scaled with the encoder. The split is **unspecified**; it is kept proportional to E's. |
| Decoder embedding / head | linear embedding of z, learned positional embedding, LayerNorm, linear per-patch pixel head, no mask tokens | **unspecified**; MAE's (He et al. 2022) decoder parts. No mask tokens because RGB-only masking keeps every token. |
| Optimiser | AdamW, lr 1e-4, weight decay 0.01 | paper |
| AdamW betas / eps, decay groups | (0.9, 0.999), 1e-8, decay on all parameters | **unspecified**; PyTorch defaults |
| LR schedule | constant | **unspecified** for pretraining (warmup + cosine is given only for fine-tuning) |
| Batch | 3 | paper (= RVT `configs/rvt.yaml:3`) |
| Length | 15 epochs; 1 epoch = one pass over the task's training time steps (hammer: 54 140 samples, 18 046 steps per epoch, 270 690 steps) | paper (15 epochs); the epoch definition is a user decision |
| Augmentation | none | **unspecified**. RVT's SE(3) and `img_aug` noise belong to its action training; MAE's crops and flips do not apply to fixed virtual views. |
| Precision, clipping | fp32, no gradient clipping | **unspecified**; RVT default `amp: False` |
| Validation | Eq. 6 on 30 fixed validation samples (fused six-camera clouds, fixed masks), plus a reconstruction panel | the paper's validation is qualitative only |
| Seed | 0 | ours |

**Scene bounds.** Lifted points from all 6 training cameras were measured at 60 random training time steps per task,
for all 8 tasks. MuJoCo body ids give the categories:

| Category | x (m) | y (m) | z (m) |
|---|---|---|---|
| Task objects | [-0.47, 0.56] | [0.24, 1.06] | [0.00, 0.48] |
| Gripper | [-0.48, 0.57] | [0.32, 1.06] | [0.00, 0.54] |
| Robot arm | [-0.53, 0.60] | from -0.26 (base) | up to 0.75 |
| Table top | [-0.66, 0.66] | [0.20, 0.98] | ~0 |

The bounds cover every object, gripper and arm point above the table and cut the robot base, as RLBench's bounds
do. They also cut the floor and walls. The 1.2 m x extent sets the scale, so a 48 px view pixel spans 2.5 cm.

## Deviations (each with its reason)

1. **Pretraining data.** In-domain Meta-World instead of Objaverse (user decision; the paper's own §4.4 ablation).
   - The paper's RLBench clouds fuse 4 cameras; ours fuse all 6 training cameras of our rig (the original fused
     multi-camera setting, user decision 2026-10-14).
   - RL condition B (single camera) is our added condition: its single-view clouds are sparser than any
     pretraining cloud. Condition A (three fused cameras) is closer to, but still sparser than, pretraining.
2. **Size-matched encoder** (user decision): see "Size".
3. **Renderer: a pure-PyTorch replacement for RVT-1's renderer.** RVT-1 uses PyTorch3D's point rasteriser and
   `NormWeightedCompositor` (`renderer.py:296-330`); RVT-2 uses a CUDA point renderer. Neither PyTorch3D nor the
   extension can be built here.
   - **Same conventions.** The cameras are hard-coded. The test runs RVT's own `get_cube_R_T` through a
     transcription of PyTorch3D's `look_at_view_transform` and checks them. Pixel grid, channels and depth
     normalisation are RVT's.
   - **Different rasterisation.**
     - Each point is binned to the pixel that contains it.
     - A z-buffer keeps the nearest point: `scatter_reduce` amin on a packed `(quantised depth << 32 | point index)`
       key, as RVT-2 packs (depth, index) for `atomicMin`.
     - RVT-2's depth-tested screen-space splat follows, with the same 0.012 NDC radius: every pixel takes the
       nearest neighbour whose splat covers it.
   - **What RVT-1 does instead.** Every point is a disc of radius 0.012 NDC, and the up to 5 nearest discs covering
     a pixel centre are blended with weights 1 - d^2/r^2.
   - **Consequences.**
     - Features are those of a single front point, not a blend.
     - At 48 px the splat radius is 0.29 px, so a pixel is covered when any point falls inside it. RVT-1 at 48 px
       would cover only pixels with a point within 0.29 px of the centre.
     - Depth is the exact float z of the front point; RVT-2 quantises depth to 1/1000.
     - Splat ties keep the pixel's own point; RVT-2 takes the last tied neighbour.
4. **RVT-1's 10 channels follow its code, not the paper's prose.** The paper's §3.1 describes the last three
   channels as "3D point coordinate in camera sensor frame". In RVT-1's code they are the pixel-location code.
5. **E has no language or proprio tokens.** 3D-MVP's E is pretrained without them; RVT-1 adds them for its action
   decoder. The per-token stem width is therefore 64 (RVT-1 with proprio: 128).
6. **Frozen features instead of fine-tuning.** The paper fine-tunes E with RVT's action decoder; our RL protocol uses
   frozen encoders.
   - The read-out (user decision) is RVT's global image feature: a spatial max-pool per view, concatenated over the
     views (`mvt_single.py:526`).
   - It is applied to z, E's last layer, giving 5 x 256 = 1280 per frame. RVT pools after `fc_aft_attn`, which
     belongs to its action head (3D-MVP's D).
   - The RVT read-out also has a heatmap-weighted part, which needs the action head and is not used.
   - `FrameMLPStackHead` stacks the feature over the 3 frames (feature_dim 256).
7. **Determinism.** The trainer sets `cudnn.deterministic = True`. The default cuDNN convolution backward differs
   between identical runs (by about 1e-8). With deterministic kernels a resumed run is bitwise equal to an
   uninterrupted one (tested).

## Size (RL-time encoder; user decision: at most SinCro's)

Method: the real modules, CPU `torch.utils.flop_counter.FlopCounterMode`, multiply-add = 2 FLOPs, one RGB-D
observation rendered into the 5 views and encoded. The original configurations are built on the meta device (same
operators, no compute). `scripts/mvp3d_size.py` prints the table and writes `docs/baselines_mvp3d_size.json`.

| Configuration | Views x tokens | Width (heads x dim) | MLP hidden | Layers | E params | E FLOPs / frame | MAE decoder params |
|---|---|---|---|---|---|---|---|
| RVT-1 MVT, encoder part (220 px, patch 11), computed | 5 x 20x20 | 512 (8x64) | 2048 | 8 (4+4) | 34 269 568 | 175 962 624 000 | (9 938 916 at 2 x 512) |
| **3D-MVP paper (220 px, patch 10), computed** | 5 x 22x22 | 1024 (8x128) | 4096 | 8 (4+4) | 134 964 352 | 767 364 730 880 | 37 522 832 |
| **3D-MVP size-matched (48 px, patch 12), measured** | 5 x 4x4 | 256 (8x32) | 1024 | 8 (4+4) | **9 029 760** | **1 485 373 440** | 2 339 136 |
| SinCro (budget) | | | | | 9 064 704 | 1 786 078 891 | |

The size-matched encoder uses 99.6 % of the parameter budget and 83.2 % of the FLOP budget. At RL time:

- every encoder parameter is reached by the forward;
- rendering costs 1 769 472 FLOPs per frame (FlopCounterMode counts only the lift and view-transform matrix
  products, not the scatter);
- the raw encoder output z is 80 x 256 = 20 480 values, and the feature fed to the head is 1280.

Knobs changed, in the user's order, all with the reason "scaled to at most SinCro's encoder size by user decision":

1. **Transformer and MLP width: 1024 -> 256.** 8 heads are kept, at 32 instead of 128 per head; the GEGLU hidden
   goes from 4096 to 1024.
   - Width is reduced first, until the parameter budget is met. 256 is the largest 8-head width that fits:
     8 x 33 = 264 gives 9 563 656 parameters.
   - No width alone can meet the FLOP budget at 220 px. RVT's convolutional stem (1x1 conv 10->64 plus the 10x10
     patchify conv 64->64) costs 5 x 220² x (640 + 4096) x 2 = 2.29 GFLOPs per frame by itself. At width 256 the
     whole 220 px / patch-10 encoder costs 71.8 GFLOPs (8 999 296 parameters).
2. **Virtual-view resolution and patch: 220 px / 10 -> 48 px / 12** (22x22 -> 4x4 tokens per view).
   - At width 256 the FLOP budget allows at most 4x4 tokens per view: 5x5 tokens (60 px) cost 2.35 GFLOPs.
   - Among 4x4 grids, patch 12 is the largest patch within the parameter budget: patch 13 adds 102 400 patchify
     weights, giving 9 132 160 parameters.
   - It is also the closest to both budgets. Patch 10 at 40 px would give 8 849 536 parameters and 1.452 GFLOPs.
3. **Layers: unchanged** (8, split 4 + 4).

The pretraining-only MAE decoder is scaled in proportion: width 1024 -> 256, keeping 2 layers and 8 heads.

The alternative of narrower widths with more tokens was not taken. It reduces width beyond what the parameter budget
needs and ends far below it:

- 8x24 = 192 at 60 px / patch 12 (5x5 tokens): 5 351 296 parameters, 1.411 GFLOPs;
- 8x16 = 128 at 70 px / patch 10 (7x7 tokens): 2 545 984 parameters, 1.411 GFLOPs.

## Tests

`tests/test_mvp3d.py` has 21 tests.

```
CUDA_VISIBLE_DEVICES=GPU-76871c16-ff1d-f7b1-cdf5-fabbaf9df8ce PYTHONDONTWRITEBYTECODE=1 \
  /home/ws/ws/splatter4d/.venv/bin/python -I -m pytest -q -p no:cacheprovider tests/test_mvp3d.py
```

Fidelity against official code (verbatim RVT code, run on fixed inputs):

- **Patched tree.** `source_tree("rvt")` differs from the checkout only by the removed matplotlib import. attn.py,
  mvt_single.py, renderer.py and rvt_utils.py are byte-identical. The submodule stays clean.
- **Layer classes.** Every encoder and decoder block is RVT's own class (identity check, source file in the pinned
  tree).
- **MVTSingle trunk and read-out (subprocess worker `tests/_mvp3d_rvt_worker.py`).**
  - The worker builds RVT's own `MVTSingle` and loads our encoder's state dict into it: no unexpected keys, missing
    keys only in the action head.
  - The attention-trunk output equals our z bitwise, for the size-matched config.
  - RVT's per-view max-pool equals our global feature, and `pixel_loc` is identical.
- **Cube cameras.** RVT's `get_cube_R_T`, run from the pinned source through a PyTorch3D transcription, equals our
  hard-coded R / T, and the view names and order match.
- **Depth normalisation.** RVT's own `if with_depth:` block from `PointsRendererWithDepth.forward` gives bitwise our
  `normalize_depth`, including an empty view.
- **Crop and cube placement.** RVT's `move_pc_in_bound` (from source) and `place_pc_in_cube` agree with our batched
  mask and placement on fixed clouds with NaNs.

Renderer conventions:

- a known point lands on the expected pixel of every view, with the expected depth and xyz/RGB channels;
- empty pixels have depth -1 and xyz = RGB = 0;
- the z-buffer keeps the nearest point, and ties go to the lower index;
- the splat covers the 4-neighbourhood at 220 px and only the pixel itself at 48 px, and it is depth-tested;
- the table plane renders flat.

MAE parts:

- masking: exactly 75 % per view, independent across views and samples, uniform over positions; only RGB is
  zeroed;
- the loss equals Eq. 6 written as explicit loops;
- a training step is finite and updates both E and the decoder.

Data adapter:

- shapes and value ranges;
- every sample fuses all six training cameras, independent of the sample key, and other camera counts are rejected;
- camera convention: lifted pixels land on the scene planes and re-project to the pixel centres;
- the seeded epoch batches resume mid-epoch.

Export and RL:

- export/reload identity;
- for both RL conditions (3 fused cameras, 1 camera), the RL backbone through `VisionEncoderAdapter` with the dict
  observation (rgb/depth/K/c2w) equals the manual pipeline, is frozen, carries its condition label, and rejects a
  config without depth or with another camera count;
- the two RL condition configs share the encoder and agent settings and differ only in `env.views`;
- DrM end to end with the frozen encoder (the encoder stays bit-identical).

Trainer and size:

- 6 uninterrupted steps against 3 + resume 3 (mid-epoch, then across the epoch boundary): identical per-step losses
  and a bitwise-identical final encoder;
- `encoder_init.pt` is written and identical across runs, and `--resume never` refuses an existing run;
- the 8 task configs share the recipe;
- the size-matched encoder fits SinCro's budget, and the next width or token step does not.

Results:

- `tests/test_mvp3d.py`: **21 passed** (29 s).
- Regression check of the touched shared code: `tests/test_rl.py tests/test_third_party.py tests/test_baselines.py`:
  **36 passed**.

## Smoke run and cost

Setup: hammer, size-matched config, 1000 steps (the limit for runs without full-suite evidence), batch 3, 8 loader
workers, commit `0bf5217`.

GPU: GPU-d09f0338 (RTX PRO 6000 Blackwell Max-Q), otherwise idle. Resources used:

- 0.32 GB peak GPU memory;
- about 5.9 GB peak host RSS, summed over the main process and the 8 workers;
- 36.6 ms per step on average, including loading and validation.

| Step | Train loss (mean of last 50 steps) | RGB part | Depth part | Validation loss (30 fixed samples) |
|---|---|---|---|---|
| 0 | 2.247 (single step) | 1.114 | 1.133 | |
| 100 | 0.222 | 0.073 | 0.149 | |
| 250 | 0.097 | 0.029 | 0.067 | 0.0939 |
| 500 | 0.067 | 0.023 | 0.044 | 0.0621 |
| 750 | 0.045 | 0.016 | 0.029 | 0.0425 |
| 950-999 | 0.038 | 0.014 | 0.025 | 0.0340 (step 1000) |

**Pretraining cost estimate**, at 36.6 ms per step on this GPU:

- training-set sizes (time steps): hammer 54 140, pick-place 50 500, bin-picking 60 220, coffee-push 53 723,
  door-open 46 860, peg-insert-side 63 847, peg-unplug-side 66 809, stick-push 57 345;
- 15 epochs = 234 300-334 035 steps per task, about **2.4-3.4 GPU-hours per task**;
- **about 23 GPU-hours for all 8 tasks** (2 267 175 steps).

The tiny model leaves most of the GPU idle, so several tasks can share one GPU.

## Figures (`docs/figures/baselines/mvp3d/`)

All panels show hammer validation episode ep010, t = 40 (`scripts/mvp3d_figures.py`).

- `input_cameras.png`: the six training-camera images.
- `views_6cam.png`, `views_3cam.png` and `views_1cam.png`: the five virtual views (columns top, front, back, left,
  right; rows RGB, depth, xyz) at 48 px of the fused 6-camera cloud (pretraining), a 3-camera cloud (training cameras
  1, 2, 4; RL condition A) and the single-camera cloud from training camera 1 (RL condition B). The 3-camera panel is
  new and is regenerated with the next smoke.
- `views_6cam_220px.png`: the same 6-camera cloud at the paper's 220 px, which shows the conventions more clearly.
- `reconstruction.png`: the smoke checkpoint on the 6-, 3- and 1-camera clouds. Rows: RGB, masked RGB input,
  reconstruction, depth, reconstructed depth. The existing panel is from the superseded random-subset smoke (step
  1000, 6- and 1-camera clouds) and is replaced by the next smoke.

At 1000 steps the reconstructions are blurry but place the table, box, hammer and arm correctly. The right view's
predicted-depth speckle comes from RVT's encoding: its empty value -1 lies close to that view's nearest true depths.

## Usage

```bash
CUDA_VISIBLE_DEVICES=<uuid> .venv/bin/python -I scripts/train_mvp3d.py --config configs/baselines/mvp3d/hammer.yaml --name mvp3d-hammer
scripts/train_rl.py --encoder mvp3d_multiview --set vision.export_path=runs/pretrain/mvp3d-hammer/encoder.pt  # A
scripts/train_rl.py --encoder mvp3d_single --set vision.export_path=runs/pretrain/mvp3d-hammer/encoder.pt     # B
```

The run writes these outputs:

- `encoder.pt`: weights, `model_cfg`, `render_cfg` (bounds, radius) and `frame_spacing: null`, because 3D-MVP
  pretrains on single time steps;
- `encoder_init.pt`: the same encoder at the run's random initialisation, for the pretrained-vs-random check;
- `checkpoints/latest.pt`, used by `--resume auto` (the default).

## RL input conditions (user decision, 2026-10-14)

One pretrained encoder per task feeds two RL conditions, both on all 8 tasks x seeds 2000-2002 with the same adapter,
projection head and state size, 3-frame handling, proprio, DrM and full evaluation protocol as every other baseline.
Results are reported as two separate rows with these labels:

| Condition | Encoder config | Training camera protocol | Viewpoint-shift evaluation |
|---|---|---|---|
| A: **3D-MVP, multi-view RGB-D + extrinsics (privileged)** | `configs/rl/encoders/mvp3d_multiview.yaml` (`env.views: 3`) | per episode 3 of the 6 training cameras, drawn at random and fixed within the episode, fused into one cloud (SNeRL's protocol) | keep 2 of the 3 input cameras and replace the third by its perturbed counterpart (near/trajectory held-out sets) or the moving trajectory (`s4d/rl/evaluate.py` `multiview_groups`, shared with SNeRL) |
| B: **3D-MVP, single RGB-D + extrinsics (privileged)** | `configs/rl/encoders/mvp3d_single.yaml` (`env.views: 1`) | one random training camera per episode, as for the single-camera methods | the single-camera methods' protocol |

B is our added single-camera condition: it sees sparser single-view clouds than the fused six-camera clouds of
pretraining. Both labels are also "reimplemented" (no official 3D-MVP code). The encoder's FLOPs per frame do not
depend on the condition (always five virtual views per frame); only the point lifting and rendering scale with the
number of fused cameras.

Hammer sanity: pretrained-frozen versus random-frozen (the saved `encoder_init.pt`) for condition A, three RL seeds,
400k agent steps; condition B with the pretrained encoder for one seed to confirm the pipeline. Same priority and
placement as the other new baselines.

Runs over 1000 steps require a passing `docs/tests.json` for the current sources.
