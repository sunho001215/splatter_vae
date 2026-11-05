# Real-world representation baselines: data adapters and pretraining protocol

Branch `realworld-fql-20261106` (agent FQL, 2026-11-06). Scope: SNeRL, 3D-MVP (one encoder; RL conditions A and B),
SinCro, ReViWo and DynaMo pretrained on the real-world drawer-open and mug-hanging episodes, per task and jointly. Our
own method's real-world pretraining is deferred (user directive 2026-11-06, item 5). Nothing here has been trained; no
GPU was used. Offline FQL on top of these encoders: `docs/realworld_fql.md`.

## Principle

Each baseline keeps its documented Meta-World protocol (`docs/BASELINES.md`, `docs/baselines_{dynamo,mvp3d}.md`,
`configs/baselines/<b>/drawer-open.yaml`). Only the data interface changes. Each real-world dataset class yields
exactly the sample of that baseline's Meta-World adapter: same keys, shapes, value ranges and camera conventions. The
models, losses, optimisers and schedules are untouched. The real-world configs are generated from the accepted
Meta-World configs by an explicit override list (`s4d/rwlearn/configs.py`). Each generated file states every changed
key and the reason. A test asserts that the committed files equal the generator output and that no other key differs.

## Data: the abstract `FrameStore` (data-format policy, user 2026-11-06 03:00)

Training frames are stored once per camera and timestamp: a centred square crop resized to 128 x 128 RGB uint8, plus
depth, validity and semantic masks at the same resolution, stored losslessly. Calibration, timestamps and the crop
affine are stored with them. Windows, synchronisation and viewpoints are assembled in the loader. The compact store
(`s4d.realworld.train_store.CompactStore`, format branch `realworld-format-20261106`) implements
`s4d/rwlearn/store.py::FrameStore` as-is. It is opened through `open_store({"kind": "compact", "paths": [...]})`.

| `FrameStore` method | contract |
|---|---|
| `episodes()`, `task(ep)` | keys `<task>/<episode stem>` (manifest key without `.h5`) |
| `cameras(ep)` | rig serials, sorted (the first is the sync reference) |
| `timestamps(ep, cam)`, `segments(ep, cam)` | (F,) ns on the recorder's host CLOCK_MONOTONIC; recording segment per frame |
| `rgb`, `depth`, `semantic(ep, cam, idx)` | (n,128,128,3) uint8; (n,128,128) float32 m (0 invalid, 0.1 mm quantised); (n,128,128) uint8 SNeRL classes |
| `intrinsics(ep, cam)`, `w2c(ep, cam)` | K of the stored grid (continuous pixel centres); OpenCV world-to-camera, world = robot base, the episode's ACCEPTED calibration |
| `raw_recording(ep)` | path of the read-only `panda_teleop_episode` file (FQL actions) |

The loader-side indexes are pure functions of the timestamps and segments; nothing derived from them is written:

- `sync_index` is used by the multi-view baselines. The reference camera's frames are kept where every other camera
  has a frame within 30 ms (offsets measured: +6.6 ms and -12.5 ms on drawer-open, -22.9 ms on mug-hanging). Steps
  that would share one native frame are dropped. Steps are split into runs at recording pauses and dropped frames.
- `camera_runs` gives each camera's own native frames. DynaMo uses them.
- `history` builds 3-step histories padded at run starts.

**Verified against the compact store.** The format branch's `CompactStore` was written from drawer-open-000000
(28 MB for 179 x 3 frames). It is bitwise equal to `ProcessedEpisodeStore` in RGB, depth, semantics, K, w2c,
timestamps and segments. Every adapter here runs on it, and a real FQL dataset slice was built from it: 533
transitions over 3 camera timelines.

`ProcessedEpisodeStore` reads the preprocessing outputs (`s4d.realworld` roots) directly. It applies exactly the
compact store's transforms: `crop_rgb(center_box)` (PIL bilinear, antialiased) for RGB, nearest-native-pixel
`crop_depth` for depth and semantics, 0.1 mm depth quantisation and `crop_intrinsics`. It exists for tests on real
slices and as an executable specification. `MemoryStore` is the synthetic test store.

**Episode split.** `split_episodes` gives a seeded 90/10 split per task (seed 0). The joint split is the union of the
two per-task splits. It is written as `split_seed0.json` (keys `train` and `validation`, the Meta-World manifest
format).

## Per-baseline adapters (`s4d/rwlearn/baseline_data.py`)

| Baseline | Sample (unchanged contract) | Real-world specifics |
|---|---|---|
| SNeRL | official draws: uniform step; `num_inputs` distinct input cameras; target uniform over all cameras; per-sample `default_rng([seed, i])` | **3 inputs of the rig's 3 cameras**, so the target is always an input view (Meta-World: 3 of 6). Validation keeps the drawn target when no held-out view exists (trainer fix, behaviour-preserving on Meta-World). Semantics are the SAM 3 classes from the store. **DINO targets stay online** in the trainer: the pinned frozen ViT-S/8 teacher with the Meta-World preprocessing `dino_target(extractor, rgb128/255)`; no DINO feature dataset. Poses come from the episode's cameras through `normalise_c2w(lookat, pose_scale)`. |
| 3D-MVP | all cameras fused per synchronised step (original in-domain fused setting); RGB-D, continuous K, OpenCV c2w | 3 cameras are fused (Meta-World: 6). Scene bounds are real-world, in the base frame. RL condition A uses the 3 cameras; B uses one random camera per sequence. |
| SinCro | 3-step windows every `temporal_stride` steps; `images` (T,H,V,W,3) in [0,1]; K with integer pixel centres; OpenGL c2w | Windows stay inside one run. Frame spacing is 1 camera frame, the real-world policy step. **Per-sample cameras**: see below. The encoder is built with `num_view` 6 (see below). |
| ReViWo | every synchronised step, all cameras, [-1,1] | 3 cameras (`camera_num` 3) |
| DynaMo | official `TrajectorySlicerDataset` windows (window 5) over single-camera videos | Trajectories are (episode, camera, run) over each camera's native frames, so no cross-camera matching is needed. Frame step is 1. Latent dim is 14 (chosen: 7 joints + gripper + 6-DoF object). |

**SinCro rays.** The reference `forward_sincro_batch` builds every sample's rays from the first sample's cameras. That
is only valid for Meta-World's single fixed rig. Each real episode has its own accepted calibration (episodes differ
by up to 1.7 deg and 20 mm). `sincro_forward_per_sample_cameras` is the reference source with exactly one block
replaced: the rays are built per sample from that sample's cameras. It is derived at import time from
`inspect.getsource` and fails if the reference block changes. The test checks that it is bit-identical to the
reference when the cameras agree, and that it differs when they do not.

**SinCro `num_view`.** Upstream `MaskedViTEncoder` creates `num_view // 3 + 1` decoder image tokens and asserts that
this equals 1 primary + 2 reference views = 3. With `num_view` 3 the upstream code asserts. The real-world trainer
therefore builds the encoder with `data_interface.sincro_encoder_num_view: 6`, which is the identical Meta-World
architecture. The three views enter only through the data. The export records it, so RL-time loading rebuilds the
same module.

## Scene parameters (measured; `scene_statistics`, `scripts/realworld_learning.py scene-stats`)

These were measured on the two processed episodes (drawer-open-000000 and mug-hanging-test-000001), centre-cropped
128 px views.

| Parameter | Value | Evidence |
|---|---|---|
| SNeRL `lookat` (base frame) | [0.334, -0.031, 0.353] m | least-squares intersection of the 3 optical axes (Meta-World: the rig look-at point) |
| SNeRL `pose_scale` | 3.79 | median camera distance 1.055 m (0.97-1.11) -> 4 official units, as official / Meta-World |
| SNeRL near/far | official 2 / 6 units = 0.53 / 1.58 m camera depth | 68.5 % of valid depth lies inside; the rest is room background at 1.6-2.6 m (Meta-World had a closed scene) |
| SinCro near/far | 0.1 / 2.5 m (unchanged) | 98.4 % of valid depth inside |
| 3D-MVP bounds | x [-0.15, 1.05], y [-0.70, 0.60], z [0.15, 1.00] m | contains 99.7 % robot, 99.0 % object, 99.8 % fixture points (SAM 3 classes); drops 43 % of background |

These values must be re-measured on the full stores before launch: `scene-stats` over all episodes, then a check that
they hold.

## Gate and launch

`s4d/rwlearn/gate.py` evaluates the user's full-data gate on the ingestion manifest (`rw-ingest-manifest` v2,
`rw-preprocess` v1). A task passes only if all of the following hold:

- ingestion is complete;
- exactly 100 valid recordings with unique indices exist, with no blocking flags, verified on the remote;
- no episode is still pending, incomplete or retrying;
- every required preprocessing stage is `done` for that file's sha256. The default stages are the preprocessing
  dispatcher's own record names: `normalized`, `calibrate`, `foundation_stereo_depth`, `sam3_masks`, `qa`, plus the
  compact store's `train_store`. The list is configurable; a name that is never recorded makes the gate fail closed.
  DINO is not required.

Joint targets need both tasks. On the live manifest at 03:10 the gate correctly fails: 0/100 preprocessed per task.

The gate is enforced twice:

1. `scripts/realworld_learning.py plan` refuses to emit rows for a failing target. Its rows declare
   `role: realworld_pretrain` or `role: realworld_fql` and `host: remote`. Under REL-B's `ops/jobs/host_roles.py`
   rules they classify to those roles both with and without the explicit role.
2. Every trainer's real-world branch checks `data_interface.gate_report`. A run longer than 10k steps refuses unless
   all of the following hold:
   - every store task passed;
   - the report's per-task `store_episodes` equal the store episodes exactly;
   - the store's tasks equal the config's `target` tasks;
   - the split manifest is a disjoint train/validation partition of exactly those episodes;
   - no data cap (`max_episodes`, `num_episodes`, `max_frames_per_demo`) truncates anything.

   The gate predicate mirrors the ingestion worker's counting:
   - records it does not count are excluded and listed: `exclude` overrides, blocking flags not cleared by an
     `accept` override (`source_missing` is never cleared), and files no longer present;
   - unknown flags and unknown validation states fail;
   - `remote_sha256` must equal the sha256;
   - exactly 100 counted, preprocessed episodes are required.

Nothing is registered or launched by this branch.

`scripts/realworld_learning.py inputs` writes the per-target `store.yaml`, `gate_report.json` and `split_seed0.json`
that the configs reference, under `/home/compu/kaist/sunho/realworld/learning_inputs/<target>/`. It refuses unless the
gate matches the store.

## Readiness checklist (per baseline)

| | SNeRL | 3D-MVP | SinCro | ReViWo | DynaMo |
|---|---|---|---|---|---|
| store adapter + tests (synthetic + real-slice smoke) | done | done | done | done | done |
| configs drawer-open / mug-hanging / joint (generated, tested) | done | done | done | done | done |
| encoder export for FQL (unchanged `encoder.pt`) | existing | existing | existing | existing | existing |
| needs from the store | RGB, semantics, cameras | RGB, depth, cameras | RGB, cameras | RGB | RGB |
| open decision | 3-of-3 inputs (no novel-view target) vs 2 inputs | bounds re-measure | none | none | latent dim 14 |
| blocked on | full-data gate; compact store merged; full GPU suite on the merged release | same | same | same | same |

Trainer edits are limited to a `data_interface.source == "realworld"` branch in `scripts/train_{snerl,sincro,reviwo,
dynamo,mvp3d}.py` and a behaviour-preserving `SNeRLScenes.camera()` accessor. Meta-World runs never import
`s4d.rwlearn`. The existing trainer entry-point tests need a GPU and were not run on this CPU-only branch: 15
GPU-only failures, all `CUDA unavailable` or the GPU guard. They must pass in the release suite.
