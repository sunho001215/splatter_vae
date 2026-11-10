# Real-world representation baselines: data adapters and pretraining protocol

Branches `realworld-fql-20261106` (agent FQL) and `rwfql-decisions-20261106` (agent FQL2, user decisions of
2026-11-06 ~11:20). Scope: SNeRL, 3D-MVP (one encoder; RL conditions A and B), SinCro, ReViWo and DynaMo pretrained on
the real-world drawer-open, mug-hanging, covering and pick-and-place episodes, PER TASK only (task-specific encoders;
the joint-task configs and rows are removed; covering / pick-and-place: last section). Our own method's real-world
pretraining is deferred (user directive 2026-11-06, item 5). Nothing here has been trained; no GPU was used. Offline
FQL on top of these encoders, and the end-to-end CNN + FQL baseline: `docs/realworld_fql.md`.

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

The loader-side indexes are built from the timestamps, segments and the store's per-set robot motion; nothing
window-level is written:

- `sync_index` is used by the multi-view consumers (SNeRL, 3D-MVP, SinCro, ReViWo, the multi-view FQL conditions).
  Criterion **M5** (coordinator decision 2026-11-06, from PREP2's review
  `/home/ws/ws/realworld-ops/pre/reviews/sync-criterion-20261106.md`): the three cameras are free-running with constant
  per-recording phase offsets up to half a period (33.4 ms), so frames are paired as MUTUAL nearest neighbours over
  the reference camera without a time cap, and a set is accepted iff the robot's URDF surface moved at most 5 mm
  across its member timestamps (`sync/surface_disp_m`, computed once in the `train_data` stage; the threshold is a
  loader argument). The former raw caps (30 ms mug-hanging, 20 ms drawer-open template) left mug-hanging 45-56 and
  drawer-open 11, 111-114, 116-124, 126, 128 without any complete set. Steps are split into runs at recording pauses
  and dropped frames. A compact store without the per-set motion (schema < 1.1.0) is refused; only synthetic test
  stores fall back to a time bound.
- `camera_runs` gives each camera's own native frames. DynaMo uses them; like every single-view consumer (FQL
  single view, e2e CNN + FQL) it depends on no cross-camera sync, only on each camera's robot <-> camera sync.
- `history` builds 3-step histories padded at run starts.

**Verified against the compact store.** The format branch's `CompactStore` was written from drawer-open-000000
(28 MB for 179 x 3 frames). It is bitwise equal to `ProcessedEpisodeStore` in RGB, depth, semantics, K, w2c,
timestamps and segments. Every adapter here runs on it, and a real FQL dataset slice was built from it: 533
transitions over 3 camera timelines.

`ProcessedEpisodeStore` reads the preprocessing outputs (`s4d.realworld` roots) directly. It applies exactly the
compact store's transforms: `crop_rgb(center_box)` (PIL bilinear, antialiased) for RGB, nearest-native-pixel
`crop_depth` for depth and semantics, 0.1 mm depth quantisation and `crop_intrinsics`. It exists for tests on real
slices and as an executable specification. `MemoryStore` is the synthetic test store.

**Episode split.** `split_episodes` gives a seeded 90/10 EPISODE-level split per task (seed 0), fixed and recorded as
`split_seed0.json` (keys `train` and `validation`, the Meta-World manifest format) in the task's learning-inputs
directory. Validation episodes are never trained on; every baseline's validation and every FQL condition's offline
validation use them.

## Per-baseline adapters (`s4d/rwlearn/baseline_data.py`)

| Baseline | Sample (unchanged contract) | Real-world specifics |
|---|---|---|
| SNeRL | official draws: uniform step; `num_inputs` distinct input cameras; target uniform over all cameras; per-sample `default_rng([seed, i])` | **3 input cameras (user decision)**: the rig has 3, so every training target is an input view (Meta-World: 3 of 6). Validation on the separate validation episodes reports `val/input_view/*` (all 3 cameras encoded, one of them rendered) and `val/novel_view_loco/*` (leave-one-camera-out at evaluation only: the rendered camera is removed from the inputs, 2 cameras encoded; the encoder mean-pools views) SEPARATELY (`s4d/rwlearn/snerl_eval.py`); the LOCO metric measures an unseen viewpoint with one fewer input view and is not comparable to Meta-World's 3-of-6 held-out views. Semantics are the SAM 3 classes from the store. **DINO targets stay online** in the trainer: the pinned frozen ViT-S/8 teacher with the Meta-World preprocessing `dino_target(extractor, rgb128/255)`; no DINO feature dataset. Poses come from the episode's cameras through `normalise_c2w(lookat, pose_scale)`. |
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
  compact store's stage `train_data`. The list is configurable; a name that is never recorded makes the gate fail closed.
  DINO is not required.

**Task-specific amendments** (user decisions 2026-11-10 for covering and pick-and-place). A versioned record
`configs/realworld/gate_amendments/<name>.json` (`rw-gate-amendment/1`) can fix one task's episode set. It lists the
valid episodes (manifest key -> sha256) and the excluded invalid ones. Its `target` replaces 100 for that task only.
There are two records. Excluded recordings are never repaired or duplicated.

- covering, `covering-98-20261110.json`: 98 episodes. 000057 and 000061 are excluded (one 32-bit device timestamp
  wrap each; `realworld-ops/rwx/AMEND-covering-98.md`).
- pick-and-place, `pick-and-place-97-20261110.json` (user decision 2026-11-10 ~14:55): 97 episodes. 000014 and
  000101 are excluded (one timestamp wrap each), and 000080 (a missing stereo record; `realworld-ops/rwx/
  AMEND-pick-and-place-97.md`).

For an amended task the ingest worker's `counts.complete` 100-rule does not apply. Instead the gate requires:

- the manifest's task episodes are exactly the listed and excluded keys, so a new recording fails the gate;
- the excluded episodes are still invalid;
- every listed episode is valid with the recorded sha256;
- every other check passes unchanged, on exactly `target` episodes.

The report records the per-task `target` and `amendment` (path, sha256). Splits are unchanged and therefore identical
across methods: every consumer binds the gate's episodes and derives `split_episodes` (seed 0, 10 %) from them.

Each task is gated alone (no joint target). GEOMETRY-DEPENDENT consumers (SNeRL, 3D-MVP, SinCro, our method)
additionally need the per-episode `calibration_validated` record bound to the current calibration, and a store built
from it (`docs/realworld_fql.md`, "Calibration hold per consumer"); ReViWo and DynaMo do not. Every generated config
names its consumer (`data_interface.consumer`), which the trainers pass to the gate check.

The gate is enforced twice:

1. `scripts/realworld_learning.py plan` refuses to emit rows for a failing target. Its rows declare
   `role: realworld_pretrain` or `role: realworld_fql` and `host: remote`. Under REL-B's `ops/jobs/host_roles.py`
   rules they classify to those roles both with and without the explicit role.
2. Every trainer's real-world branch checks `data_interface.gate_report`. A run longer than 10k steps refuses unless
   all of the following hold:
   - every store task passed;
   - manifest records match store episodes uniquely by task and recorded raw filename, with equal SHA256 when
     the store records it; `inputs` saves both manifest and store keys in `episode_identities`, and its bound report's
     `store_episodes` equal the actual store keys (for example `drawer-open/drawer-open-000000`);
   - the store's tasks equal the config's `target` tasks;
   - the split manifest is a disjoint train/validation partition of exactly those episodes;
   - no data cap (`max_episodes`, `num_episodes`, `max_frames_per_demo`) truncates anything.

   The gate predicate mirrors the ingestion worker's counting:
   - records it does not count are excluded and listed: `exclude` overrides, blocking flags not cleared by an
     `accept` override (`source_missing` is never cleared), and files no longer present;
   - unknown flags and unknown validation states fail;
   - `remote_sha256` must equal the sha256;
   - exactly 100 counted, preprocessed episodes are required (an amended task: exactly its record's `target`).

Nothing is registered or launched by this branch.

`scripts/realworld_learning.py inputs` writes the per-target `store.yaml`, `gate_report.json` and `split_seed0.json`
that the configs reference, under `/home/compu/kaist/sunho/realworld/learning_inputs/<target>/`. It refuses unless the
gate matches the store.

## Readiness checklist (per baseline)

| | SNeRL | 3D-MVP | SinCro | ReViWo | DynaMo |
|---|---|---|---|---|---|
| store adapter + tests (synthetic + real-slice smoke) | done | done | done | done | done |
| configs drawer-open / mug-hanging (generated, tested; task-specific only) | done | done | done | done | done |
| encoder export for FQL (unchanged `encoder.pt`) | existing | existing | existing | existing | existing |
| needs from the store | RGB, semantics, cameras | RGB, depth, cameras | RGB, cameras | RGB | RGB |
| decided / open | 3 inputs; input-view + LOCO novel-view validation | bounds re-measure | none | none | latent dim 14 |
| calibration-dependent (gate level) | yes | yes | yes | no | no |
| blocked on | full-data gate; compact store merged; full GPU suite on the merged release | same | same | same | same |

Trainer edits are limited to a `data_interface.source == "realworld"` branch in `scripts/train_{snerl,sincro,reviwo,
dynamo,mvp3d}.py` and a behaviour-preserving `SNeRLScenes.camera()` accessor. Meta-World runs never import
`s4d.rwlearn`. The existing trainer entry-point tests need a GPU and were not run on this CPU-only branch: 15
GPU-only failures, all `CUDA unavailable` or the GPU guard. They must pass in the release suite.

## Covering and pick-and-place (2026-11-09 directive)

`s4d/rwlearn/tasks.py` is the one registry of learning targets (drawer-open, mug-hanging, covering, pick-and-place);
`configs`, the planner (`ops/jobs/realworld_plan.py` = `launch`), the gate and outcome defaults, `baseline_data`'s
target check and `ops.jobs.host_roles.REALWORLD_TASKS` all read it. Every target keeps its own encoder, seeded split and
100-episode gate; `scripts/realworld_learning.py gate|plan|outcomes` default to all four (`--tasks covering` etc. for
one), so the default `gate` exit code is 0 only once every target passed. These are real-world tasks; nothing here
touches the cancelled Meta-World expansion. The generator `python -I scripts/realworld_learning.py configs` writes the
five baseline configs per new task; FQL keeps its shared config. Unchanged for drawer-open / mug-hanging (checked
against HEAD e52bfca): every generated config byte, gate result, plan row, the 200 ending-check results, and the
replayed plans of the two existing registrars; `docs/realworld_training_plan.yaml` gained the new tasks' preview rows
after its unchanged two-task rows.

**Scene geometry: pending, not inferred.** SNeRL's `lookat` / `pose_scale` and 3D-MVP's `bounds` were measured on
drawer-open / mug-hanging processed episodes. `configs.SCENE_PARAMETERS` maps those two tasks to exactly those values
and the new tasks to `None` (generated YAMLs hold `null`, with a TODO reason). Raw evidence
(`realworld_newtasks_evidence.json`, read-only): the same three camera serials; base-frame flange ranges

| Task (raw recordings) | x m | y m | z m | flange rows in MVP3D_BOUNDS |
|---|---|---|---|---|
| covering (36) | 0.294-0.664 | -0.357 - -0.004 | 0.464-0.645 | 100 % |
| pick-and-place (`test/pick_and_place.h5` only) | 0.368-0.568 | -0.047-0.100 | 0.470-0.583 | 100 % |
| drawer-open / mug-hanging (100 each, reference) | 0.307-0.628 | -0.208-0.248 | 0.487-0.766 | 100 % |

Covering reaches 15 cm further to the robot's right than the measured tasks. Flange containment says nothing about the
towel / can / plate extent or depth coverage, and the raw files carry no camera calibration (per-task rig calibration
is a separate check), so the old numbers are not copied. `configs.pending_scene_parameters(baseline, task)` is the
hook: while it is non-empty the planner refuses that baseline and its FQL conditions (SNeRL; 3D-MVP multiview and
single) with the reason "scene parameters not measured"; ReViWo, DynaMo, SinCro and e2e_cnn plan normally. SinCro reads
no scene constant, but its metric near/far (0.1 / 2.5 m) should be confirmed on the same statistics. The trainers are
unchanged (a null value fails at start-up).

Measurement recipe, per new task, on its calibration-validated processed store (all cameras, the whole task):

```bash
python -I scripts/realworld_learning.py scene-stats --store <task store.yaml>  # scene_statistics, MVP3D_BOUNDS
```

Take the optical-axis least-squares `lookat`, `pose_scale = 4 / median camera distance`, check SNeRL's 2/6-unit depth
window and >= 99 % robot / object / fixture containment per class for the bounds (foreground apart from room
background), then set the task's `SCENE_PARAMETERS` entry, regenerate the configs (the equality test pins them) and
release. Keep the old numbers only if the statistics support them.
