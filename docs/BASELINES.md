# Stage 4 baselines

The final comparison comprises ours, SinCro, SNeRL, ReViWo, DynaMo, 3D-MVP and the end-to-end CNN. MV-MWM and
GNFactor are excluded. Representation baselines pretrain per task on the same episode split and use frozen encoders
in DrM. SinCro and ReViWo are already in the accepted campaign; the three new ports are not yet accepted.

## Expansion policy and adaptation plan (2026-10-13)

Use the exact-pin official submodules directly, with thin data, checkpoint and RL adapters. Models and losses must
not be rewritten. Minimal import/runtime patches live under `third_party/patches/<name>/` and apply to cached copies,
never the submodule checkout. Test and release identities include `.gitmodules`, gitlink pins and patch bytes; remote
releases transfer the pinned submodule contents separately. No locked shared dependency changes are permitted.

| Method | Pretraining input and objective | Frozen RL input / raw readout | Adaptation and outstanding gate |
|---|---|---|---|
| SNeRL | Three training-camera RGB images and known poses; official RGB, semantic and DINO-feature NeRF objectives | Three RGB views and poses, official mean-pooled 63-d feature; **multi-view RGB input (privileged)** | Direct official shared convolution encoder and NeRF; preserve rig K and official integer ray samples without a half-pixel offset. Final port, license scope, resume and full native acceptance remain pending. |
| DynaMo | Each camera video is a separate RGB trajectory; official stopped-target latent forward/inverse dynamics and covariance loss | Single RGB camera, official pooled ResNet-18 512-d feature | Original encoder unchanged. Keep 40 epochs, epoch-0 encoder/projector LR zero and the unscheduled forward optimizer. Meta-World choices: latent 39, window 5, frame gap 2, shared split. Import-only accelerate patch and exact-step/resume tests require bundle verification. |
| 3D-MVP | Original in-domain fused setting: each sample's point cloud fused from **all six training cameras** (RGB-D + extrinsics), five RVT virtual views, RGB-only 75% patch masking, RGB-D MAE reconstruction; no camera subsets or single-camera clouds; one pretraining per task | Two RL conditions sharing that encoder, per-view max pooling then concatenate: **A** "3D-MVP, multi-view RGB-D + extrinsics (privileged)" (3 of 6 training cameras per episode, fused; SNeRL protocol) and **B** "3D-MVP, single RGB-D + extrinsics (privileged)" (one camera per episode); both **reimplemented** | No official 3D-MVP code: use the pinned RVT trunk directly; only masking, MAE decoder and pretraining loop are ours. The returned duplicated trunk is not accepted and must be replaced by a thin official-RVT wrapper. Pooling is unspecified in the paper. |

**3D-MVP input conditions (user decision, 2026-10-14; supersedes the earlier random-subset pretraining and
single-camera-only RL input).** Pretraining follows the original fused multi-camera setting: every sample fuses all
six training cameras, with no random subsets and no single-camera clouds. The same pretrained encoder per task feeds
two RL conditions, both run on all eight tasks x seeds 2000-2002 and reported as **separate rows with their input
labels in every table**:

- **A, "3D-MVP, multi-view RGB-D + extrinsics (privileged)"**: SNeRL's camera protocol. Per episode, 3 of the 6
  training cameras are drawn at random and fixed within the episode; their RGB-D is fused into one cloud, then
  rendered to the virtual views, then fed to the frozen encoder. Viewpoint-shift evaluation keeps 2 of the 3 input
  cameras fixed and replaces the third by its perturbed counterpart (near/trajectory sets) or the moving trajectory.
- **B, "3D-MVP, single RGB-D + extrinsics (privileged)"**: one random training camera per episode and the same
  viewpoint-shift evaluation as the single-camera methods. **B is our added single-camera condition**: it sees
  sparser single-view clouds than in pretraining.

Everything else is identical to the other baselines in both conditions: adapter pattern, projection head and state
size, three-frame handling, proprio, DrM and the full evaluation protocol. The size cap is unchanged; encoder
FLOPs/frame do not depend on the condition (five virtual views per frame).

SNeRL's feature objective follows the official MSE rather than the paper's L1; unavailable auxiliary terms must be
reported, not invented. Its downstream encoder is fully frozen by the comparison protocol, unlike the partially
trainable official RL setup. Full-scene data differs from the official alpha-masked scenes. DynaMo adds no actions,
geometry or camera identity. For 3D-MVP, the permitted z-buffer point-splat fallback differs from RVT-1's PyTorch3D
disc blending; matching cameras/channels is not proof of native-renderer equivalence.

Only 3D-MVP is resized: at most **9,064,704 parameters** and **1,786,078,891 FLOPs/frame**, as measured for SinCro.
One 3D-MVP frame includes all five virtual views rendered from one RGB-D observation. Reduce transformer/MLP width,
then virtual-view resolution/patch size, then layers with a proportional within-/cross-view split; scale the MAE-only
decoder proportionally. All other encoder sizes stay unchanged, even when larger than SinCro. No original-size
training side runs are allowed.

### RL-time encoder size and pretraining cost

Multiply-add counts as two FLOPs. Count parameters reached by the frozen encoder readout, not pretraining-only
rendering/decoder or dynamics modules, and exclude the trainable RL projection head. For temporal encoders divide
three-frame observation FLOPs by three. New-port figures below are preliminary handback measurements, not full-suite
acceptance; 3D-MVP must be remeasured after the direct-RVT correction. GPU-hour estimates cover eight tasks, exclude
RL and are not actual campaign resource accounting.

| Method | Encoder parameters | GFLOPs/frame | Raw encoder output | State after head | Pretraining GPU-h, 8 tasks |
|---|---:|---:|---|---:|---|
| Ours (unchanged) | 1,726,592 | 0.214980 | 256 / three-frame history | 256 | Pending final-length accounting |
| SinCro (unchanged) | 9,064,704 | 1.786079 | 256 / three-frame history | 256 | ~356, shared-GPU smoke extrapolation |
| SNeRL (official size unchanged; preliminary) | 6,587,885 | 0.637811 / three-view frame | 63 / frame | 256 | ~456, latest bounded-smoke extrapolation; original 2.4M recipe unchanged |
| ReViWo (unchanged) | 1,683,712 | 0.416940 | 1,024 / frame | 256 | ~129, shared-GPU smoke extrapolation |
| DynaMo (official size unchanged; preliminary) | **11,176,512** | 1.184 | 512 / frame | 256 | ~42, dedicated-GPU smoke extrapolation |
| 3D-MVP (chosen; preliminary; same encoder for conditions A and B) | 9,029,760 | 1.485373 | 80 x 256 tokens; pooled 1,280 / frame | 256 | ~23, dedicated-GPU smoke extrapolation (superseded random-subset recipe; to be remeasured with fused six-camera samples) |
| CNN end-to-end (registered protocol) | 30,368 | 0.071069 | 103,968 / three-frame observation | 50 | 0; no representation pretraining |
| Original RVT-1 image-token encoder (**computed only**, not a comparison run) | 34,269,568 | 175.962624 | 2,000 x 512 tokens; pooled 2,560 / frame | Not trained | 0; no side run |

DynaMo is 1.23x SinCro in parameters and remains unchanged. CNN retains its already-registered DrM 50-d trunk;
the frozen methods all use the shared 256-d adapter output and three-frame history. This existing CNN difference is
recorded rather than silently changed after its completed runs.

Original RVT-1: 220-pixel views, patch 11, width 512, eight layers split 4+4. Chosen 3D-MVP: 48-pixel views, patch 12,
width 256 (8 heads x 32), the same eight layers split 4+4; MAE decoder two layers at width 256, 2,339,136 parameters.
Reason: **scaled to at most SinCro's encoder size by user decision**. The preliminary chosen trunk uses 99.6% of the
parameter budget and 83.2% of the FLOP budget; width 264 exceeds parameters, and 5x5 tokens/view at width 256 exceeds
FLOPs. These bounds must survive the official-trunk correction. The renderer's reported matrix-product cost is
0.001769 GFLOPs/frame separately; full renderer accounting remains pending.

### Acceptance and scientific order

1. Finish the thin ports, convention/official-function/resume tests and complete native bundle suite locally.
2. Integrate and deploy the immutable accepted bundle without stopping current campaign jobs; run the full suite
   in the remote image. Repeat GPU isolation only if guard or GPU-selection code changes; label reused evidence
   as carry-over, not a fresh test.
3. Pretrain hammer at the unchanged recipes. Save the exact same initialized encoder for the random-frozen arm.
   Run pretrained-frozen versus random-frozen DrM with three RL seeds and 400k agent steps; inspect panels and
   investigate lack of benefit before expanding to all tasks. For 3D-MVP this check uses condition A; condition B
   runs once (pretrained encoder, one seed) to confirm its pipeline.
4. Full-split retrieval and probes for every new baseline, rendering additionally for SNeRL. For SNeRL camera
   shifts replace one input view and hold the other two fixed; never silently alter the existing campaign's
   evaluation protocol.
   Tool: `scripts/eval_baseline_representation.py` (`s4d/baselines/representation.py`). It loads the export through
   the RL adapter of each RL encoder config (the 3D-MVP conditions are `mvp3d_multiview` and `mvp3d_single`) and runs
   our `Evaluator`'s retrieval, probes and state statistics, with our splits, windows, camera groups and targets.
   Only the encoder call differs. A window's state is the fp16 feature the agent caches: the per-frame features of
   the three frames, concatenated (`--feature stack`, the RL head's input). Multi-view columns follow
   `s4d/rl/evaluate.py`: training column i uses cameras (i, i+1, i+3) mod 6, and a held-out column replaces the
   first camera of its nearest training camera's column. The pretrained and `encoder_init.pt` arms are separate
   outputs. SNeRL rendering is not implemented yet.
5. Report adaptation, final sizes/costs, deviations, hammer sanity and panels once, then proceed to all eight tasks
   (3D-MVP: one pretraining per task, then conditions A and B, each 8 tasks x 3 seeds). Same priority and placement
   as the other new baselines.
   Never use the six non-development tasks' RL outcomes for method decisions.

Upstream execution/testing for RVT, DynaMo and SNeRL is currently permission-blocked. Reported bounded diagnostics
are not a release or the required full-pretraining sanity. SNeRL's handback reports 790 passes and one full-suite
virtual-gitlink fixture failure, so it is not accepted. Its historical 10k old-K/redraw smoke is context only; a
corrected-K 1,000-step diagnostic used explicit smoke-only seed 1, while all registered recipe configs retain seed 0.
Seed 0's dead coarse density remains a documented risk, not grounds for automatic reseeding. The reported corrected
smoke held-out panel has PSNR 14.10 dB and depth MAE 0.467 m, with only 55.6% valid depth inside its near/far bounds;
novel-view quality is weak. The existing accepted campaign continues independently.

### Hammer acceptance report: DynaMo and 3D-MVP (steps 3-5, 2026-10-17)

The upstream-permission paragraph above is historical. Release e4206f1 was accepted remotely: native suite and GPU
isolation in the image. All hammer runs used it. SNeRL is reported separately once its 2.4M-step pretraining
completes.

| | DynaMo | 3D-MVP |
|---|---|---|
| RL-time encoder | 11,176,512 params, 1.184 GFLOPs/frame, 512-d/frame (official size) | 9,029,760 params, 1.485 GFLOPs/frame (+0.0018 render), 1,280-d/frame; scaled to <= SinCro |
| Hammer pretraining | 194,960 steps (40 epochs), 4.99 h on one remote GPU | 270,690 steps, 3.04 h on one remote GPU |
| Eight-task estimate | ~40 GPU-h | ~24 GPU-h |
| Sanity: last-5 train success, pretrained vs random (seeds 1000-1002) | 0.349 vs 0.516: **failed** | A: 0.328 vs 0.268, passed (narrow); B: 0.328 (one seed) |
| Step 4 retrieval, train cams, newest frame (ours 0.94) | 0.014 (random 0.030) | A 0.029 (random 0.787); B 0.019 (random 0.297) |
| Step 4 hand-position R², train cams (ours 0.97) | 0.516 (random 0.466) | A 0.826 (0.838); B 0.527 (0.574) |
| Effective rank, pretrained vs random | 49 vs 249 | A 41 vs 475 |

- **Deviations:** none beyond those already documented (`docs/baselines_dynamo.md`, the 3D-MVP size and input sections
  above).
- **DynaMo investigation:** there was no plumbing bug. The run is faithful but weak: the representation partially
  collapses (effective rank 3.7 vs 10.7 within trajectories) and over-trains (probe R² 0.835 at epoch 9, 0.673 at
  the final epoch). Evidence: `runs/setup/baselines/dynamo-sanity-investigation-20261017.md`. The final-epoch encoder
  is kept; choosing a checkpoint would be retuning.
- **DynaMo epoch-9 diagnostic (user-requested, 2026-10-18, descriptive):** hammer DrM with the epoch-9 export,
  last-5 training success 0.251 [0.185, 0.298] vs final 0.349 and random init 0.516 (AUC 0.110 / 0.145 / 0.213).
  The earlier checkpoint is not better in RL despite its better probes; the final encoder stays.
- **Panels:** `docs/figures/baselines/dynamo/final/` and `docs/figures/baselines/mvp3d/final/`.
  - DynaMo same-camera nearest-neighbour state error is 2.1 cm trained vs 0.8 cm at random init (chance 20 cm).
  - The 3D-MVP MAE reconstruction is faithful for 6- and 3-camera clouds and noisier for one camera.
  - The 3D-MVP size measurement is in `docs/baselines_mvp3d_size-final.json`.
- **Caveats:**
  - The RL-input-stack probes (1536/3840-d) are not interpretable: they are underdetermined, which was pre-registered.
  - Multi-view columns share cameras, and 3D-MVP renders a fused world-frame cloud. Random-init multi-view retrieval
    is therefore inflated.
  - The sanity differences are small relative to the spread between seeds.

### Eight-task results: DynaMo and 3D-MVP (step 5, 2026-10-18)

All 72 new DrM runs are complete: 8 tasks × seeds 2000–2002 × 1M agent steps, 240 episodes every 10k steps. They are
reported with the final-comparison standard and are descriptive only; no new decision rule follows. Tool:
`runs/setup/tools/baselines_eight_task_report.py`. Report:
`runs/setup/baselines-eight-task-report-20261018-221713.json` and its `.md` digest, which has the per-task tables.
Curves: `docs/figures/baselines/eight_task/success_curves.png`. The ours / SinCro / ReViWo / CNN rows reproduce the
earlier final-comparison IQMs exactly.

IQM success over 8 tasks × 3 seeds, last-5 (AUC) [stratified bootstrap 95% CI on last-5]:

| method (input) | training cameras | held-out cameras | trajectories |
|---|---|---|---|
| Splatter4D (ours), single RGB | **0.756** (0.510) [0.670, 0.811] | 0.126 (0.105) [0.111, 0.140] | 0.721 (0.509) [0.650, 0.771] |
| DrM CNN, single RGB | 0.718 (0.409) [0.666, 0.768] | 0.008 (0.012) [0.004, 0.014] | **0.771** (0.438) [0.735, 0.809] |
| SinCro | 0.440 (0.250) [0.343, 0.528] | 0.003 (0.004) [0.002, 0.004] | 0.101 (0.093) [0.081, 0.123] |
| ReViWo | 0.386 (0.260) [0.274, 0.481] | 0.003 (0.009) [0.002, 0.006] | 0.229 (0.170) [0.196, 0.279] |
| DynaMo, single RGB | 0.199 (0.120) [0.158, 0.243] | 0.020 (0.027) [0.005, 0.039] | 0.071 (0.060) [0.033, 0.131] |
| 3D-MVP A, multi-view RGB-D + extrinsics (privileged) | 0.186 (0.095) [0.140, 0.237] | 0.099 (0.063) [0.064, 0.154] | 0.205 (0.102) [0.132, 0.284] |
| 3D-MVP B, single RGB-D + extrinsics (privileged) | 0.325 (0.179) [0.205, 0.445] | **0.144** (0.107) [0.094, 0.197] | 0.279 (0.136) [0.156, 0.405] |

- **DynaMo** has the lowest training-camera IQM among the RGB-only methods (0.199). It is behind ours on every task,
  ranging from 0.00 on peg-insert-side to 0.69 on hammer (ours 0.89) and 0.77 on stick-push (ours 1.00). This matches
  its failed hammer sanity check: the encoder is faithful but partially collapsed.
- **3D-MVP** learns poorly on training cameras. Condition A is at 0.00 on bin-picking, pick-place and stick-push;
  condition B is at 0.00 on bin-picking and pick-place. Where it does learn, its held-out-camera success is high:
  - door-open: A 0.82 and B 0.49, vs ours 0.35;
  - peg-unplug-side: A 0.45 and B 0.35, vs ours 0.19;
  - stick-push: B 0.44 vs ours 0.16;
  - hammer: A 0.29 vs ours 0.15.

  The held-out IQM of B (0.144) is above ours (0.126), but the CIs overlap. Both 3D-MVP conditions receive depth and
  camera extrinsics, a privileged input that the single-RGB methods do not have. Read these held-out numbers with
  that input label, not as a like-for-like comparison.
- **Fairness:**
  - No RL config differs from ours outside the encoder and input fields (`encoder_name`, `vision.*`, and for 3D-MVP
    `env.cameras/depth/views`).
  - Every point has 240 episodes.
  - Hosts: DynaMo and B ran on local MIG. A ran 14 runs local and 10 remote (host-neutral overlay). The local launch
    commits differ only in tooling and in the behaviour-preserving `agent.random_shift` gate. One older CNN run has
    no provenance file; its host (local) comes from the registry.
- SNeRL's eight-task runs follow its hammer acceptance. Its pretraining is at step 1.67M of 2.4M.

The remainder documents the already-accepted SinCro/ReViWo adaptations. Their rule is **upstream code, original
hyperparameters, changes only at the data interface**; exceptions are listed explicitly.

## Sources

| What | Where | Pinned commit |
|---|---|---|
| SinCro model code | github.com/sunho001215/sincro (MIT, © 2025 Seungyeon Yoo) | `5c7c06d373233e7bc7d7c240fdc27b212e2535f4` |
| ReViWo model code | github.com/lafmdp/ReViWo (**no license stated**) | `ac0c24958c83366dbdceebfb1db9f9111fce56a0` |
| SNeRL official code | github.com/jayLEE0301/snerl_official (`third_party/snerl`; MIT file under `snerl/`, root/pretrainer coverage unresolved) | `d01533b673ebcea1705b15d5e4549c67c757d9a9` |
| DynaMo official code | github.com/jeffacce/dynamo_ssl (`third_party/dynamo`; MIT) | `62874039e5d3fc663b0fccd2083b2dd2c8d9935e` |
| RVT official trunk | github.com/NVlabs/RVT (`third_party/rvt`; NVIDIA non-commercial research/evaluation license) | `367995a1a2169b6352bf4e8b0ed405890462a3a0` |
| DINO teacher for SNeRL | github.com/facebookresearch/dino (`third_party/dino`; Apache 2.0) | `7c446df5b9f45747937fb0d72314eb9f7b66930a` |

The ReViWo repository states no license: the vendored code is used for this internal research comparison only and must not be redistributed without the authors' permission.
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
   - **Decision (2026-10-07):** keep the reference config's 0.1/2.5 m. These are the original reference
     hyperparameters; the reference trainer only failed to pass them to `render()`.
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
9. **SinCro pretrains for exactly 300 000 steps on every task (user decision, 2026-10-08).**
   - The reference config sets `max_global_steps: 500001`; the configs now set `300000`. The loop runs while
     `global_step < max_global_steps`, so this is exactly 300 000 optimizer updates, and the last checkpoint and
     export are written at step 300 000 (`save_every` 30 000 divides it). The vendored `TrainConfig` default is
     300 001, so the new length matches the reference code's own default to within one step.
   - Everything else is unchanged, including `lrate_decay: 500`: the learning rate still follows
     5e-4 × 0.1^(step / 500 000), so it ends at 5e-4 × 0.1^0.6 ≈ 1.26e-4 rather than 5e-5.
   - ReViWo keeps its reference length (`max_global_steps: 100001`) and all other hyperparameters.

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

There are 8 tasks: door-open, hammer, peg-unplug-side, stick-push, pick-place, peg-insert-side, coffee-push and
bin-picking.

- The reference has configs for the first four. All its task configs are identical except for names and paths, so
  the other four use hammer's.
- `configs/baselines/{sincro,reviwo}/<task>.yaml` change only the dataset path, the W&B project and the
  `data_interface` block, and drop the output-location keys; SinCro's also set `max_global_steps: 300000`
  (deviation 9).

**SinCro**

| Group | Values |
|---|---|
| Steps and batch | **300 000 steps** (reference 500 001; deviation 9), batch 8 windows × 3 steps × 6 views, `N_rand` 2048 rays/step |
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
| SinCro | 0.53 | 7.8 GB | 500 001 steps ≈ 74 h, plus validation 500 × 6.2 s ≈ 0.9 h; **as run: 300 000 steps ≈ 44 h + 0.5 h** | ≈ 25 GPU-days at 500k; **≈ 15 GPU-days at 300k** |
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
