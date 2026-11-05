# Developer guide

## Directory map

| Path | Contents |
| --- | --- |
| `s4d/` | Research code: `data/` (DROID, Meta-World, held-out sets), `realworld/` (real-robot preprocessing), `model/` (encoder, decoder, Gaussians, CUDA rendering), `losses/`, `train/` (loop, DDP, checkpoints, augmentation), `rl/` (agents, environments, evaluation), `baselines/`, `diag/` (W&B logging, diagnostics), `config.py`, `geometry.py`, `third_party.py` (pinned upstream source trees for the baselines). Module map below. |
| `ops/` | Operational tooling. It is not imported by the research code except at the two points listed below. |
| `ops/remote/` | `access.py`: SSH/rsync transport, tunnel check, download budget. `artifact_check.py`: read-only on-host inspection (standard library only; its source is sent over SSH and executed by the remote host's python3). `sync_media.py`: local backup of remote runs. `ledger.py`: local ledger of the remote jobs (checkpoints, containers, host health). `outage_guard.py`: the 5-minute outage guard (admission hold, recovery check). `sync_wandb.py`: pull and upload of offline remote W&B runs. |
| `ops/jobs/` | `scheduler.py`: file-based scheduler (queue, registry, admission, launches). `remote_backend.py`: the scheduler's remote-host backend (containers over SSH, in-place verification, acceptance evidence). `run_identity.py`: run provenance and clean-release enforcement. `lifecycle.py`, `replay_host.py`, `cleanup_replay.py`: replay cleanup of completed RL runs (`replay_host.py` is also sent to the remote host). `admission_pause.py`: intentional admission pauses (`runs/setup/admission-pauses.json`) honoured by the drivers. `pool_placement.py`: host-neutral placement of never-launched rows. |
| `ops/resources/` | `gpu_guard.py`: fail-closed GPU UUID allowlist, runtime-path and EGL checks. `nvidia_query.py`: bounded `nvidia-smi` queries. `heartbeat.py`, `watch_events.py`: stall and event monitoring. `host_equivalence.py`: validation of the cross-host Meta-World diagnostic. `memory_budget.py`: the local 150 GB budget accounting. `memory_watch.py`: memory watchdog (`line`, `fits BYTES`, `alert`). |
| `ops/containers/` | `launch.py`: the remote `docker run` launcher (run by `scripts/remote/docker_run.sh`). `verification.py`: typed release-verification jobs (native suite, GPU isolation). |
| `ops/campaign/` | Campaign command modules, run through `runs/setup/tools/<same name>.py`: the admission drivers `stage2_local_driver.py`, `baseline_local_driver.py`, `remote_campaign_driver.py` (one flocked pass each); the lossless checkpoint pauses `pause_pretrain_at_checkpoint.py`, `pause_rl_at_checkpoint.py`, `pause_remote_pretrain_at_checkpoint.py`; `reconcile_unlaunched_remote.py` (refused remote launches); the release verification chain `release_chain.py`, `prepare_remote_verification.py`, `assemble_remote_acceptance.py`, `adopt_remote_suite_evidence.py`. Most of these are scripts whose module-level code is the command: importing one runs it. |
| `scripts/` | Command-line entry points (training, evaluation, data, figures) and `_bootstrap.py` (entry-point preamble: repo on `sys.path`, GPU guard, caches, test-evidence gate). |
| `scripts/remote/` | Image and container plumbing that is used at fixed paths: `Dockerfile`, `package_runtime.py` (both copied into the image bundle), `entry.py` (the in-container entry), and shell helpers. |
| `runs/setup/tools/` | Campaign records and fixed entry points: the entry points of the implementations moved to `ops/` (same file names, same command lines), dated one-off tools (`*_YYYYMMDD.py`), `.before-*` backups, stage-specific registrations and reports, research diagnostics, and the remote runner sources. Not fingerprinted and not imported by `s4d`, `ops` or `scripts`. |
| `analysis/` | `eval_config.py` (evaluation run configs; run by the drivers at this path), `queue_variants.py`, `review_rules.py`. |
| `configs/`, `tests/`, `experiments/`, `runs/`, `docs/` | Configurations, tests, scheduler queue/registry, run outputs and campaign records, documentation. |

## Research code map (`s4d/`)

Read in this order to follow one pretraining step: data contract -> dataset -> model -> losses -> training loop.

| Area | Modules | Responsibility |
| --- | --- | --- |
| Batch contract | `s4d/data/contract.py` | The batch dictionary every loader produces (`validate_batch`, `collate`, `PAIRS`, motion targets). |
| Meta-World data | `s4d/data/metaworld/{collect,cameras,dataset,heldout,heldout_render}.py`, `s4d/data/hardneg.py` | Simulator collection, the fixed camera rig, the window dataset, the held-out camera sets, hard negatives of pilot `hn`. |
| DROID sample path (first real data) | `s4d/data/droid/{convert,dataset,rlds,inspect,gripper,pointworld}.py` | The single inspected PointWorld-DROID episode: conversion to a cache, its loader, RLDS verification. These are also shared libraries: the large-scale DROID path uses `pointworld.py` (track targets, depth test, splatted motion score) and `gripper.py` (gripper surface points), and `s4d/realworld/export.py` uses `pointworld.py`. |
| DROID large-scale path | `s4d/data/droid/largescale/` | `tfrecord.py` + `rlds_index.py` (in-place RLDS access), `pointworld_source.py` + `extract.py` (released PointWorld-DROID files), `calibration.py` (DROID-3D, Franka FK, wrist poses), `geometry.py` (camera/pose conventions), `join.py` (identity join, frame validation, wrist QA, manifest), `crop.py` (square crop of the ViT-S profile), `dataset.py` (window dataset), `config.py` (datasets from a training config), `handles.py` (bounded file handles). See docs/droid_largescale.md. |
| Real-world preprocessing (own rig) | `s4d/realworld/` | `adapters/` (recording layouts; `panda_teleop.py`) -> `pipeline.py` (ingest, camera views, robot geometry, calibrate), `depth_stages.py` (sensor_depth, frame sets), `ir_stereo_depth.py` (foundation_stereo_depth: FoundationStereo on the device IR pair), `trackeverything_stage.py` (track_everything: the track2d + lift3d records), `track_motion.py` (dynamic-track filter) -> `export.py` (training windows) -> `dataset.py` (loader); `baseline_stages.py` (SAM 3 masks, SNeRL DINO cache). `contract.py` (normalized episode), `io.py` (atomic writes, completion records), `geometry.py`, `stereo.py` (disparity conventions), `tracking.py` (track chunk contract), `calibration.py`, `robot.py` (URDF kinematics), `mounts.py` (gripper mounts), `kernels.py` + `runners.py` (numerical kernels, local/remote runner), `viewer_data.py`, `workers/` (upstream models in their own environments, file/stream protocol), `cli.py` (`python -m s4d.realworld.cli`). Remote execution: `ops/inference_server/`. See docs/realworld_preprocessing.md. |
| Model | `s4d/model/{encoder,decoder,gaussians,render}.py` | ViT state encoder with masking, grouped Gaussian decoder, the Gaussian set container, gsplat rendering (FP32). |
| Losses | `s4d/losses/` | `rgb.py`, `depth.py`, `chamfer.py`, `motion.py`, `motion3d.py`, `invariance.py`, `occlusion.py`, `regularizers.py`. |
| Training | `s4d/train/` | `loop.py` (model bundle, `forward_losses`, the training loop), `augment.py` (camera jitter, synthetic and self-render views), `perframe.py` + `perframe_eval.py` (per-frame pilots; they reuse `loop.py`'s helpers), `evaluate.py`, `checkpoint.py`, `ddp.py`, `workers.py`. |
| Downstream RL | `s4d/rl/` | `protocol.py` (run configuration), `env.py` + `vecenv.py` (Meta-World environments), `encoders.py` (frozen/CNN policy encoders), the algorithms (`agent.py` DrM, `snerl_sac.py`, `sb3_ppo.py`, `dreamer_v1.py`, `flashsac.py`; `algorithms.py` maps a run's `algorithm` to its evaluation policy), `replay.py`, `evaluate.py`, `camera_suites.py` + `visibility.py` (evaluation camera suites), `run_io.py` (checkpoint/metrics files of the trainers). |
| Baselines | `s4d/baselines/{sincro,reviwo,snerl,dynamo,mvp3d}/`, `data.py`, `representation.py` | Ports or adapters around the pinned upstream code (`s4d/third_party.py`); see docs/BASELINES.md. |
| Diagnostics | `s4d/diag/` | W&B and local logging, panels, probes, point clouds, tracks, held-out diagnostics. |
| Shared | `s4d/config.py`, `s4d/geometry.py` | Layered YAML configuration; torch pinhole geometry. |

## Command-line entry points (`scripts/`) by purpose

The file names are fixed: queue rows, registrations and docs name them by path.

| Purpose | Scripts |
| --- | --- |
| Pretraining | `train.py` (ours: Meta-World, DROID cache, DROID large-scale), `train_sincro.py`, `train_reviwo.py`, `train_snerl.py`, `train_dynamo.py`, `train_mvp3d.py` |
| Downstream RL | `train_rl.py` (DrM), `train_sac.py`, `train_ppo.py`, `train_dreamer.py`, `train_flashsac.py`; `eval_rl.py` (companion evaluator) |
| Evaluation and export | `evaluate.py`, `export_encoder.py`, `summarize_run.py`, `eval_baseline_representation.py` |
| Meta-World data | `collect_metaworld.py`, `split_metaworld.py`, `check_metaworld.py`, `verify_metaworld_tasks.py`, `screen_tasks.py`, `render_heldout_sets.py`, `compute_workspace_stats.py`, `bench_loader.py`, `check_metaworld_equivalence.py`, `camera_visibility.py`, `selfrender_domain.py` |
| DROID sample (first real data) | `download_droid_sample.py`, `convert_droid_sample.py`, `inspect_droid_sample.py`, `validate_droid_loader.py`, `droid_validate_alignment.py`, `droid_track_smoke.py`, `droid_compare_step0.py` |
| DROID large-scale | `droid_ls_build_rlds_index.py`, `droid_ls_extract_pointworld.py`, `droid_ls_build_manifest.py` |
| Real world | `realworld_compose_robot_urdf.py`, `realworld_viewer.py`, `realworld_viewer_check.py`, `realworld_viewer_video.py` (the pipeline itself: `python -m s4d.realworld.cli`, remotely `ops/inference_server/run_episode.py`) |
| Diagnostics and figures | `diag_capacity.py`, `hardneg_report.py`, `selfrender_diagnostics.py`, `mvp3d_figures.py`, `mvp3d_size.py`, `dynamo_panels.py`, `render_snerl.py` |
| Verification | `run_tests.py`, `check_gpu_isolation.py`, `check_native_render.py` |
| Preamble and environment | `_bootstrap.py`, `gpu_env.sh`; `delete_stage0_cache.py` is a disabled stub |
| Compatibility aliases of `ops` | `jobs.py`, `_nvidia_query.py`, `heartbeat.py`, `watch_events.py`, `sync_media.py`, `cleanup_replay.py` |

Each scheduled GPU entry point keeps its own preamble (`sys.path` insert of `scripts/`, `from _bootstrap import ...`, an unindented `guard_gpus()` or `enforce_allowed_gpus()` call before torch is imported): the scheduler checks for that call in the script file (`GUARD_CALL` in `ops/jobs/scheduler.py`). Exception: `scripts/run_tests.py` (not a scheduled job) calls `guard_gpus()` inside `main`.

## Dependency boundaries

- `ops` may import `s4d`. It currently imports `s4d.third_party` (source identity) and, lazily, `s4d.model.encoder` (encoder export).
- `s4d` does not import `ops`, with two documented exceptions. Both are lazy imports of `ops.resources.gpu_guard` inside functions: `s4d/data/__init__.py` uses `validate_runtime_path` to authorize output paths, and `s4d/diag/wandb_log.py` uses `HOST_NAME` for the W&B host tag. `tests/test_ops_layout.py` enforces this.
- There are no import cycles. `ops.resources.gpu_guard` imports neither `s4d` nor any other `ops` module.
- The ML-specific CUDA kernels and device code stay in `s4d/model` and `s4d/train`.
- `ops/remote/artifact_check.py` and `ops/jobs/replay_host.py` must stay standard-library only, because they are executed on the remote host outside the containers.
- `ops` never imports `runs/setup/tools`; the moved implementations import each other through their `ops` paths. Nothing in `s4d`, `ops`, `scripts` or `tests` imports an `ops.campaign` module, because importing one runs its command (`tests/test_ops_layout.py`).
- The implementations moved from `runs/setup/tools` keep their hard-coded campaign root `/home/ws/ws/splatter4d` and the `sys.path` inserts of it, so they act on the main checkout's live state (registry, queue, locks, remote host) from wherever they are imported, a worktree included. Tests import only the import-safe ones, and only in subprocesses. Run their entry points from the main checkout: an entry point in another checkout (a worktree) runs that checkout's `ops`, so its scheduler module reads that checkout's `experiments/registry.jsonl` and `queue.yaml`, while every other hard-coded path still names the main checkout. Before refactor pass 2 a worktree's copy of these tools used the main checkout's scheduler. In the main checkout both are the same.

## Compatibility aliases

The old module paths remain importable. Each one is a short shim that replaces its `sys.modules` entry with the implementation, so the old and new names refer to the same module object. Monkeypatching, `isinstance`, and pickles that use the old path all behave as before. When a shim is run as a script (including `python -I`), it executes the implementation as `__main__` and leaves `sys.argv` unchanged, so the usage text stays the same. New code should import from `ops`.

| Old path | Implementation |
| --- | --- |
| `s4d.remote_access`, `s4d.remote_artifact_check` | `ops.remote.access`, `ops.remote.artifact_check` |
| `s4d.remote_jobs`, `s4d.lifecycle_cleanup`, `s4d.replay_cleanup_host`, `s4d.run_identity` | `ops.jobs.remote_backend`, `ops.jobs.lifecycle`, `ops.jobs.replay_host`, `ops.jobs.run_identity` |
| `s4d.gpu_guard`, `s4d.cross_host_equivalence` | `ops.resources.gpu_guard`, `ops.resources.host_equivalence` |
| `s4d.remote_launch`, `s4d.remote_verification` | `ops.containers.launch`, `ops.containers.verification` |
| `scripts/jobs.py`, `scripts/_nvidia_query.py` | `ops.jobs.scheduler`, `ops.resources.nvidia_query` |
| `scripts/heartbeat.py`, `scripts/watch_events.py` | `ops.resources.heartbeat`, `ops.resources.watch_events` |
| `scripts/sync_media.py`, `scripts/cleanup_replay.py` | `ops.remote.sync_media`, `ops.jobs.cleanup_replay` |
| `runs/setup/tools/memory_budget.py`, `memory_watch.py` | `ops.resources.memory_budget`, `ops.resources.memory_watch` |
| `runs/setup/tools/admission_pause.py`, `pool_placement.py` | `ops.jobs.admission_pause`, `ops.jobs.pool_placement` |
| `runs/setup/tools/remote_ledger.py`, `remote_outage_guard.py`, `sync_wandb_remote.py` | `ops.remote.ledger`, `ops.remote.outage_guard`, `ops.remote.sync_wandb` |
| `runs/setup/tools/{stage2_local_driver,baseline_local_driver,remote_campaign_driver}.py` | `ops.campaign.<same name>` |
| `runs/setup/tools/{pause_pretrain_at_checkpoint,pause_rl_at_checkpoint,pause_remote_pretrain_at_checkpoint,reconcile_unlaunched_remote}.py` | `ops.campaign.<same name>` |
| `runs/setup/tools/{release_chain,prepare_remote_verification,assemble_remote_acceptance,adopt_remote_suite_evidence}.py` | `ops.campaign.<same name>` |

`from scripts import jobs`, `import jobs` (with `scripts/` on `sys.path`) and `.venv/bin/python -I scripts/jobs.py ...` keep working. Likewise every documented launch command `.venv/bin/python -I runs/setup/tools/<tool>.py ...` of a moved tool keeps its arguments, output, exit codes and help text, and the dated one-off tools keep importing `pool_placement`, `admission_pause` and `memory_budget` by their old names. The entry points compute the repository root from their own location (`parents[3]`); `memory_watch` still inserts `runs/setup/tools` into `sys.path` (recomputed from its new location, same value). To get an independent copy of a module in a test, load the implementation file (for example `ops/jobs/scheduler.py`) with `importlib.util.spec_from_file_location`, not the shim.

Source fingerprints cover `mujoco_mig_setup.py`, `s4d/`, `ops/`, `scripts/`, `tests/` and `configs/`. The four scopes are `scripts/_bootstrap.source_fingerprints`, `ops.jobs.scheduler.source_identity`, `ops.jobs.remote_backend.commit_source_sha256` and the run-identity native diagnostic. Test evidence therefore covers the operational code, since refactor pass 2 including the campaign drivers and tools moved from `runs/setup/tools` (which is not fingerprinted): an edit of a driver now needs a fresh full suite before long work, like any other source edit. For commits that predate `ops/`, the fingerprints are unchanged.

## Common tasks

Every GPU command needs exactly one allowed UUID (`source scripts/gpu_env.sh`). Entry points call the guard before importing torch or MuJoCo.

```bash
# Train (pretraining) and evaluate
CUDA_VISIBLE_DEVICES="$GPU4" .venv/bin/python -I scripts/train.py --config configs/metaworld/base.yaml configs/metaworld/tasks/hammer.yaml --name <run>
CUDA_VISIBLE_DEVICES="$GPU4" .venv/bin/python -I scripts/evaluate.py --run outputs/<run>
# RL: scripts/train_rl.py, train_sac.py, train_ppo.py, train_dreamer.py, train_flashsac.py; evaluation: scripts/eval_rl.py

# Schedule: edit experiments/queue.yaml, then
.venv/bin/python -I scripts/jobs.py status | tick | daemon | reset <id>
.venv/bin/python -I scripts/watch_events.py --once      # events; scripts/heartbeat.py checks for stalls

# Remote release (clean pushed commit + pinned image, see docs/REMOTE.md)
scripts/remote/sync_code.sh <commit>                     # ops/remote/access.py sync-code
# The scheduler then starts remote jobs through scripts/remote/docker_run.sh -> ops.containers.launch -> scripts/remote/entry.py;
# the typed verification jobs (native_suite = scripts/run_tests.py, gpu_isolation = scripts/check_gpu_isolation.py) gate releases.

# Campaign drivers and monitors: entry points in runs/setup/tools, implementations in ops/ (see the alias table)
.venv/bin/python -I runs/setup/tools/stage2_local_driver.py --dry-run   # ops/campaign/stage2_local_driver.py
.venv/bin/python -I runs/setup/tools/memory_watch.py fits 40000000000   # memory admission before a full suite

# Tests (run targeted files; the full suite writes docs/tests.json evidence)
CUDA_VISIBLE_DEVICES="$GPU4" .venv/bin/python -m pytest -q -p no:cacheprovider tests/test_ops_layout.py tests/test_jobs.py
CUDA_VISIBLE_DEVICES="$GPU4" .venv/bin/python -I scripts/run_tests.py
.venv/bin/ruff check s4d ops scripts tests
```

## Deliberately not moved

- `scripts/run_tests.py` and `scripts/check_gpu_isolation.py` are typed verification job scripts. They run by path inside the containers, and the scheduler requires a literal guard call in the script file (`GUARD_CALL`), which a wrapper would not contain.
- `scripts/remote/entry.py`, `package_runtime.py`, `Dockerfile` and the shell helpers stay at fixed in-container and image-bundle paths. Moving them would change the image bundle.
- `s4d/third_party.py` serves the research baselines (`source_tree`). `ops` only reads its identity helper.
- `analysis/eval_config.py` prepares scientific evaluation configs and is run by the Stage 2 driver at that path.
- In `runs/setup/tools`, refactor pass 2 moved only the reusable operational drivers and libraries (alias table). Everything else stays byte-identical:
  - dated one-off tools (`*_YYYYMMDD.py`) and `.before-*` backups: campaign records that the logs cite by path;
  - the remote runner sources `remote_runner.py` (a copy runs on the remote host), `gen_remote_autonomy_plan_20261031.py`, `update_remote_plan_20261031.py` and `reconcile_remote_runner_20261031.py`;
  - stage-specific registrations (`register_*.py`) and reports (`rl_reporting.py`, `stage2_report.py`, `followup_report.py`, `followup_tables.py`, `baselines_eight_task_report.py`, `final_comparison.py`, `crop_reeval_report.py` and their `report_figures.py`): analysis records tied to fixed registrations, cited by docs, not operations;
  - research diagnostics (`actfut_*`, `afk2_*`, `chamfer_*`, `enc_*`, `encp_*`, `grad_conflict_diag.py`, `variant_repr_probes.py`, `paper_ablation_smoke_local.py`, `selfrender_*`);
  - retired or one-time operational tools: `evict_partial_cache.py` (retired 2026-10-21), `enc_gate.py` (retired 2026-10-25), `remote_container_cleanup.py` (the 2026-10-21 cleanup; the scheduler removes containers since), `queue_mig_limits.py`, `apply_stage2_placement.py`, `record_cuda_init_outage.py`, the provenance snapshots `job_provenance.py` and `local_device_provenance.py`, and `check_final_stage3_dispatch.py` (a Stage 3 fixture check that parses the remote driver's source at its old path; it already failed before the move, because the driver functions it executes need globals its namespace lacks).

## Code structure before and after refactor pass 2

Refactor pass 2 (branch `refactor2-20261031`, from `a8628a7`) moved the reusable operational drivers and libraries of `runs/setup/tools` into `ops`. The top level is unchanged: `s4d/` (research), `ops/` (operations), `scripts/` (thin entry points), `analysis/`, `configs/`, `tests/`, `experiments/`, `runs/` (outputs and campaign records), `docs/`, `third_party/`.

```text
before (a8628a7)                               after
ops/                                           ops/
  remote/   access, artifact_check,              remote/   access, artifact_check, sync_media,
            sync_media                                     + ledger, outage_guard, sync_wandb
  jobs/     scheduler, remote_backend,           jobs/     scheduler, remote_backend, run_identity,
            run_identity, lifecycle,                       lifecycle, replay_host, cleanup_replay,
            replay_host, cleanup_replay                    + admission_pause, pool_placement
  resources/ gpu_guard, nvidia_query,            resources/ gpu_guard, nvidia_query, heartbeat,
            heartbeat, watch_events,                       watch_events, host_equivalence,
            host_equivalence                               + memory_budget, memory_watch
  containers/ launch, verification               containers/ launch, verification
                                                 campaign/ (new) stage2_local_driver, baseline_local_driver,
                                                           remote_campaign_driver, pause_pretrain_at_checkpoint,
                                                           pause_rl_at_checkpoint, pause_remote_pretrain_at_checkpoint,
                                                           reconcile_unlaunched_remote, release_chain,
                                                           prepare_remote_verification, assemble_remote_acceptance,
                                                           adopt_remote_suite_evidence
runs/setup/tools/ (175 tracked files)          runs/setup/tools/ (the same 175 files)
  18 reusable drivers and libraries (full        the same 18 paths: 17-line entry points
  implementations)                               (implementation in ops/, same module objects)
  dated one-offs, .before-* backups,             unchanged, byte for byte
  registrations, reports, diagnostics,
  remote runner sources
```

## Refactor pass 3

Refactor pass 3 (branch `refactor3-20261102`, from `2230bc6`) changed no file locations and no behaviour. It removed
duplicated helpers inside the research code and its entry points:

| Before | After |
| --- | --- |
| `scripts/train_{rl,sac,ppo,dreamer,flashsac}.py`: five copies of `rss_gb`, `atomic_save`, `append_jsonl` | `s4d/rl/run_io.py` (new), imported by the five trainers |
| `s4d/train/perframe.py`: copies of `MOVING_SCORE`, `STATIC_SCORE`, `ACTIVE_OPACITY`, `_rows`, `_per_t`, `_nanmean`, `temporal_ramp` (as `_ramp`), `_float_set` | imported from `s4d/train/loop.py`, whose inline float32 cast of the decoded set is now the same `_float_set` |
| `s4d/data/droid/largescale/calibration.py`: hand-written `+0.5` integer-to-continuous pixel centres | `geometry.to_continuous_centres(K, integer_centres=True)` |
| `s4d/realworld/contract.py`: unused private `_array` | removed |

It also added the research code map and the entry-point table above. Behaviour evidence (public API signatures, `--help` of
every entry point except `check_native_render.py` and `_bootstrap.py`, resolved configurations, CPU numerical fingerprints of the Meta-World, DROID, export and real-world
paths) is in `/home/ws/ws/validation-20261031/refactor3/`.
