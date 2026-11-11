# Developer guide

Unified pool (directive 2026-11-11 ~20:15 KST): LOCAL (8 MIG slices, 150 GB RAM budget) and REMOTE are
interchangeable training hosts for Meta-World, real-world and DROID (`configs/hosts/roles.yaml`, version 2).
Data locality still applies: preprocessing and typed release verification stay remote; remote-resident training
runs locally only with a verified resolver in `ops/jobs/realworld_local.py`. Resolvers cover FQL training,
real-world workspace stats, our real-world pretraining, and exports from locally produced run directories.
Remote dependency files require verified backup checksums; a remote run directory cannot be consumed locally.
`needs`, `fetch`, `verify`, `plan` and `place` inventory, mirror, check and explicitly place never-started rows.
Started runs use `ops/jobs/migration.py`; begin permits either admitting destination, while switch-to-remote requires
the separate remote release procedure. Meta-World keeps its pool/campaign release; other families keep their pins.
Remote reboot/admission holds never hold local placement or admission. A queue's adjacent `HOLD` holds that queue;
`hosts.remote.hold` / `enabled: false` hold only remote admission. The RTX 5090 inference host never schedules.
Local baseline admission guarantees each active/ready family an equal share of RAM, GPU slots and GPU memory,
borrowing only unclaimed guarantees and retaining trainer head-of-line reservation per family (`ops/jobs/fair_share.py`).


## Directory map

| Path | Contents |
| --- | --- |
| `s4d/` | Research code: `data/` (DROID, Meta-World, held-out sets), `realworld/` (real-robot preprocessing), `model/` (encoder, decoder, Gaussians, CUDA rendering), `losses/`, `train/` (loop, DDP, checkpoints, augmentation), `rl/` (agents, environments, evaluation), `baselines/`, `diag/` (W&B logging, diagnostics), `config.py`, `geometry.py`, `rwlearn/` (real-world learning); `third_party.py` and `rwlearn/launch.py` are ops compatibility aliases. Module map below. |
| `s4d/realworld/calib/` | `core.py` (per-camera calibration), `rig.py` (fixed rigs), `board.py` (board geometry and solve), `release.py` (release checks and application); `dw/` and `loma/` hold the rig studies and shared machinery. |
| `s4d/realworld/stages/` | `calibrate.py`, `depth.py`, `ir_depth.py`, `semantics.py`, `trackeverything.py`, `motion.py`: the existing preprocessing stages and versions. |
| `s4d/realworld/robotics/` | `model.py` (URDF kinematics), `visual.py` (visual surfaces), `mounts.py`; `collada.py` shares only position-source parsing and fan triangulation. |
| `ops/` | Operational tooling; the research facades and explicit boundary exceptions are listed below. |
| `ops/remote/` | `access.py`: SSH/rsync transport, tunnel check, download budget. `artifact_check.py`: read-only on-host inspection (standard library only; its source is sent over SSH and executed by the remote host's python3). `sync_media.py`: local backup of remote runs. `ledger.py`: local ledger of the remote jobs (checkpoints, containers, host health). `outage_guard.py`: the 5-minute outage guard (admission hold, recovery check). `sync_wandb.py`: pull and upload of offline remote W&B runs. |
| `ops/jobs/` | `scheduler.py`: file-based scheduler (queue, registry, admission, launches). `remote_backend.py`: the scheduler's remote-host backend (containers over SSH, in-place verification, acceptance evidence). `run_identity.py`: run provenance and clean-release enforcement. `lifecycle.py`, `replay_host.py`, `cleanup_replay.py`: replay cleanup of completed RL runs (`replay_host.py` is also sent to the remote host). `admission_pause.py`: intentional admission pauses (`runs/setup/admission-pauses.json`) honoured by the drivers. `pool_placement.py`: host-neutral placement of never-launched rows. `arguments.py`: shared option inspection. `checkpoint_pause.py`: import-safe process identity, checkpoint metadata, session scan and registry writer. `realworld_plan.py`: queue plans for baseline pretraining and FQL (it reads the deployment root and input directories from `s4d.rwlearn.configs`). |
| `ops/resources/` | `gpu_guard.py`: fail-closed GPU UUID allowlist, runtime-path and EGL checks. `nvidia_query.py`: bounded `nvidia-smi` queries. `heartbeat.py`, `watch_events.py`: stall and event monitoring. `host_equivalence.py`: validation of the cross-host Meta-World diagnostic. `memory_budget.py`: the local 150 GB budget accounting. `memory_watch.py`: memory watchdog (`line`, `fits BYTES`, `alert`). `processes.py`: liveness (including zombie rejection). `reservations.py`: transient validation RAM reservation reader. |
| `ops/provenance/` | `hashing.py`: canonical JSON and 8-MiB streaming SHA-256. `upstreams.py`: pinned upstream identity, source caches and patch application (standard library only). |
| `ops/datasets/` | Resumable internet downloads of DROID RLDS / PointWorld-DROID / DROID-3D onto the remote data disk (docs/dataset_downloads.md): `catalog.py` (pinned sources, needed files, official checksums, plan), `transfer.py` (range-resumed HTTP, verification, rate limiter), `state.py` (atomic state, single-owner lock), `storage.py` (margin, final/peak projection, admission), `health.py` (host/network signals, pool/rate controller), `archives.py` (tar.zst staging, verification, publication), `runner.py` (owner process), `cli.py` (remote commands). Standard library only: runs on the remote host's python3 from a deployed copy. `control.py` is the local side (deploy, latency probe). |
| `ops/containers/` | `launch.py`: the remote `docker run` launcher (run by `scripts/remote/docker_run.sh`). `verification.py`: typed release-verification jobs (native suite, GPU isolation). |
| `ops/inference_server/` | Remote preprocessing commands, persistent stream client/session and contract checks. Worker executables and file/stream protocol implementations remain in `s4d/realworld/workers/`. |
| `ops/policy_serving/` | Real-world network policy serving (docs/realworld_policy_serving.md): `protocol.py` (wire format), `client.py` (CPU robot-side client), `bench.py`, `server.py` (persistent GPU server: catalog registry, lazy load, GPU-memory budget + LRU, SIGHUP reload), `config.py` (config + catalog reader), `staging.py` (inventory, staging of deployable policies to the inference host, monitoring hook), `criteria.py` (deployability checks of realworld-ops/serving/POL-CRITERIA.md). Science stays in `s4d/rwlearn/deploy/`. |
| `ops/robot_pc/` | Robot-PC client (Directive I, docs/robot_pc.md): `safety.py` (execution-mode guard, controller binding UNBOUND), `sinks.py` (Mock / DryRun / refusing Live sink), `validation.py` + `poses.py` (action validation), `executor.py` (local high-rate loop), `robot_state.py` (read-only state sources), `switch_controller.py` (the robot PC's controller interface: frames, RobotAction format, approval-gated RobotState reader), `tunnel.py` (direct ssh -L to the 5090), `cameras.py` + `devices.py` (acquisition, busy guard), `camera_warmup.py` (numpy colour stability gate before the first live policy step), `selection.py`, `loop.py`, `replay.py`, `dryrun.py`, `config.py`, `cli.py`. CPU only, no torch; never commands a robot. |
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
| Real-world preprocessing (own rig) | `s4d/realworld/` | `adapters/` (recording layouts; `panda_teleop.py`) -> `pipeline.py` (ingest, normalized-episode access, camera views, robot geometry), `stages/calibrate.py` (calibrate), `stages/depth.py` (sensor_depth, frame sets), `stages/ir_depth.py` (foundation_stereo_depth: FoundationStereo on the device IR pair), `stages/trackeverything.py` (track_everything: the track2d + lift3d records), `stages/motion.py` (dynamic-track filter) -> `export.py` (training windows) -> `dataset.py` (loader); `stages/semantics.py` (SAM 3 masks, SNeRL DINO cache). `contract.py` (normalized episode), `io.py` (atomic writes, completion records), `geometry.py`, `stereo.py` (disparity conventions), `tracking.py` (track chunk contract), `calib/core.py`, `robotics/model.py` (URDF kinematics), `robotics/mounts.py` (gripper mounts), `kernels.py` + `runners.py` (numerical kernels, local/remote runner), `viewer_data.py`, `workers/` (upstream models in their own environments, file/stream protocol), `cli.py` (`python -m s4d.realworld.cli`). Remote execution: `ops/inference_server/`. See docs/realworld_preprocessing.md. |
| Real-world calibration | `s4d/realworld/calib/{core,rig,board,release}.py`, `calib/dw/`, `calib/loma/` | Per-camera and fixed-rig calibration, DW and LoMa evidence/solve/publication. `calib/dw/common.py` owns the shared camera constants, split, match conversion, held-out static validation and common-pose perturbation. LoMa imports these without depending on the DW command module. |
| Real-world robot geometry | `s4d/realworld/robotics/{model,visual,mounts,collada}.py` | URDF/FK, visual mesh surfaces, gripper mounts and private shared COLLADA parsing. Stage implementations live in `stages/`; worker scripts stay in `workers/` for isolated execution. |
| Model | `s4d/model/{encoder,decoder,gaussians,render}.py` | ViT state encoder with masking, grouped Gaussian decoder, the Gaussian set container, gsplat rendering (FP32). |
| Losses | `s4d/losses/` | `rgb.py`, `depth.py`, `chamfer.py`, `motion.py`, `motion3d.py`, `invariance.py`, `occlusion.py`, `regularizers.py`. |
| Training | `s4d/train/` | `loop.py` (model bundle, `forward_losses`, the training loop), `augment.py` (camera jitter, synthetic and self-render views), `perframe.py` + `perframe_eval.py` (per-frame pilots; they reuse `loop.py`'s helpers), `evaluate.py`, `checkpoint.py`, `ddp.py`, `workers.py`. |
| Downstream RL | `s4d/rl/` | `protocol.py` (run configuration), `env.py` + `vecenv.py` (Meta-World environments), `encoders.py` (frozen/CNN policy encoders), `agent.py` (DrM), `snerl_sac.py` (standalone SAC), `algorithms.py` (SAC evaluation dispatch), `replay.py`, `evaluate.py`, `camera_suites.py` + `visibility.py` (evaluation camera suites), `run_io.py` (checkpoint/metrics files of the trainers). |
| Baselines | `s4d/baselines/{sincro,reviwo,snerl,dynamo,mvp3d}/`, `data.py`, `representation.py` | Ports or adapters around pinned upstream code (`s4d/third_party.py` aliases `ops/provenance/upstreams.py`); see docs/BASELINES.md. |
| Diagnostics | `s4d/diag/` | W&B and local logging, panels, probes, point clouds, tracks, held-out diagnostics. |
| Real-world learning | `s4d/rwlearn/` | Stores, actions, transitions, gates, baseline configs and FQL; `deploy/` (deployment contract, docs/realworld_deployment.md; `deploy/m5_live.py`: live multi-view M5 evidence for the robot PC, docs/robot_pc.md). `gate.py` reads the versioned task-specific gate amendments in `configs/realworld/gate_amendments/`. `tasks.py` is the registry of learning targets (`TARGETS`, `TARGET_TASKS`; import-free, re-exported by `configs.py`); `configs.py` owns the remote deployment root and `run_inputs_dir` (read by `ops.jobs.realworld_plan` and `ops.containers.mounts`; science never imports ops for them). The old `launch.py` aliases `ops.jobs.realworld_plan`. |
| Native-depth 3D-MVP | `s4d/rwlearn/native_depth.py` | CPU-only, atomic/resumable per-episode native RealSense depth overlays using deployment replay and published rig corrections; `NativeDepthStore` changes only compact-store depth. CLI: `scripts/realworld_native_depth.py build`. Opt-in descriptor and row chain: docs/realworld_fql.md. |
| Shared | `s4d/config.py`, `s4d/geometry.py` | Layered YAML configuration; torch pinhole geometry. |

## Operational code map (`ops/`)

| Responsibility | Authoritative implementation | Consumers and retained APIs |
| --- | --- | --- |
| Queue control and placement | `jobs/scheduler.py`, `host_roles.py`, `pool_placement.py`, `remote_backend.py` | Scheduler/driver entry points; evaluator partners use `host_roles._partner`. |
| Shared process and resource checks | `resources/processes.py`, `reservations.py` | `scheduler.pid_alive`, `heartbeat.alive` and both local drivers' `_validation_reservation` remain exported; reservation calls stay in the drivers. |
| Argument inspection and pause primitives | `jobs/arguments.py`, `checkpoint_pause.py` | `scheduler._option`, `lifecycle.option` re-export the parser. Pause commands forward identity/checkpoint reads, inject session identity and bind their registry paths. |
| Artifact and source provenance | `provenance/hashing.py`, `upstreams.py` | Run identity, host equivalence and remote access share canonical JSON/8-MiB hashing. Ops/bootstrap use upstreams directly; science source loaders retain the module-object alias. |
| Real-world plans | `jobs/realworld_plan.py` | Planner alias for scripts/registrars; deployment paths stay in `s4d.rwlearn.configs` (ops -> science edge only). |
| Transport and isolated execution | `remote/`, `containers/`, `inference_server/` | SSH/rsync, launch/verification and inference coordination. `remote/artifact_check.py` and `jobs/replay_host.py` remain standalone. |
| Dataset downloads | `datasets/` | Remote-side owner and CLI (standard library only, deployed to `/data_8T/kaist/sunho/tools/<sha>/`); `datasets/control.py` deploys and probes through `remote/access.py`. Reuses `s4d.data.droid.largescale.extract` (standard library only) for member extraction. |

## Command-line entry points (`scripts/`) by purpose

The file names are fixed: queue rows, registrations and docs name them by path.

| Purpose | Scripts |
| --- | --- |
| Pretraining | `train.py` (ours: Meta-World, DROID cache, DROID large-scale), `train_sincro.py`, `train_reviwo.py`, `train_snerl.py`, `train_dynamo.py`, `train_mvp3d.py` |
| Downstream RL | `train_rl.py` (DrM), `train_sac.py` (standalone SAC); `eval_rl.py` (companion evaluator) |
| Evaluation and export | `evaluate.py`, `export_encoder.py`, `summarize_run.py`, `eval_baseline_representation.py` |
| Meta-World data | `collect_metaworld.py`, `split_metaworld.py`, `check_metaworld.py`, `verify_metaworld_tasks.py`, `screen_tasks.py`, `render_heldout_sets.py`, `compute_workspace_stats.py`, `bench_loader.py`, `check_metaworld_equivalence.py`, `camera_visibility.py`, `selfrender_domain.py` |
| DROID sample (first real data) | `download_droid_sample.py`, `convert_droid_sample.py`, `inspect_droid_sample.py`, `validate_droid_loader.py`, `droid_validate_alignment.py`, `droid_track_smoke.py`, `droid_compare_step0.py` |
| DROID large-scale | `droid_ls_build_rlds_index.py`, `droid_ls_extract_pointworld.py`, `droid_ls_build_manifest.py`; downloads: `datasets/droid_datasets.py` (remote side), `datasets/droid_datasets_remote.py` (local control) |
| Robot PC (Directive I) | `robot_pc/`: `setup.sh`, `selfcheck.py`, `policies.py`, `replay.py`, `dry_run.py`, `bench.py`, `execute.py` (refuses), `tunnel.py` run ON the robot PC from the bundle; `make_bundle.py`, `route_bench.py` (one-command robot-PC benchmark over ssh) run on the development host (docs/robot_pc.md) |
| Real world | `realworld_compose_robot_urdf.py`, `realworld_m5_asset.py` (live M5 surface asset export/validation/negative cases for the robot PC, docs/robot_pc.md; development host only), `realworld_viewer.py`, `realworld_viewer_check.py`, `realworld_viewer_video.py` (the pipeline itself: `python -m s4d.realworld.cli`, remotely `ops/inference_server/run_episode.py`) |
| Diagnostics and figures | `diag_capacity.py`, `hardneg_report.py`, `selfrender_diagnostics.py`, `mvp3d_figures.py`, `mvp3d_size.py`, `dynamo_panels.py`, `render_snerl.py` |
| Verification | `run_tests.py`, `check_gpu_isolation.py`, `check_native_render.py` |
| Preamble and environment | `_bootstrap.py`, `gpu_env.sh`; `delete_stage0_cache.py` is a disabled stub |
| Compatibility aliases of `ops` | `jobs.py`, `_nvidia_query.py`, `heartbeat.py`, `watch_events.py`, `sync_media.py`, `cleanup_replay.py` |

Each scheduled GPU entry point keeps its own preamble (`sys.path` insert of `scripts/`, `from _bootstrap import ...`, an unindented `guard_gpus()` or `enforce_allowed_gpus()` call before torch is imported): the scheduler checks for that call in the script file (`GUARD_CALL` in `ops/jobs/scheduler.py`). Exception: `scripts/run_tests.py` (not a scheduled job) calls `guard_gpus()` inside `main`.

**Retired 2026-11-09: PPO, Dreamer, FlashSAC.** Their exclusive policy implementations/adapters, trainer scripts,
algorithm YAML files, dedicated tests and upstream submodules were removed. DrM, standalone SAC and shared encoders,
replay, environments and evaluation used by real-world FQL and supported workflows remain. Historical results, logs,
experiment records and trainer/provenance metadata are retained; new evaluation of retired checkpoints is unsupported.
`ops/jobs/retired.py` is the permanent refusal policy: scripts, historical run prefixes and evaluator references
(`--train-job`, `--run-dir`, dependencies and `reevaluates`). Scheduler eligibility applies it to default and scoped
queues, including crash retries and reset rows; host-role refusal precedes retry/drain exceptions; container
`validate_spec` also refuses retired launches. Drivers use these gates and filter retired mixed-registration rows.
The separate standalone SAC campaign pause remains in force; keeping SAC software does not resume that campaign.

## Dependency boundaries

- `ops` may import science implementations. `ops.jobs.realworld_plan` eagerly imports `s4d.rwlearn.configs` and
  `s4d.rwlearn.gate`; it keeps the configs path facade so science-side root overrides remain visible.
  `ops.jobs.host_roles` eagerly imports one science module, `s4d.rwlearn.tasks` (the import-free target registry, for
  `REALWORLD_TASKS`). Scheduler encoder export lazily imports `s4d.model.encoder`. Upstream identity imports now stay inside `ops.provenance`; container mounts read the real-world deployment root
  from `s4d.rwlearn.configs` (an ops -> science edge).
- Inference still crosses the boundary: `stream_client` eagerly imports `s4d.realworld.workers` and its framing/protocol;
  `command_support` lazily imports `WorkerSpec`. `run_episode` imports the real-world CLI partly eagerly and IO/adapters
  lazily; `contract_checks` lazily imports actions/transitions, data contracts and real-world dataset/pipeline/depth/store/
  windows. The deferred worker runtime has not moved.
- Outside compatibility aliases, direct research imports of `ops` are limited to two lazy GPU-guard exceptions: output-path authorization in `s4d/data/__init__.py` and the W&B host
  tag in `s4d/diag/wandb_log.py`. `tests/test_ops_layout.py` enforces that exact direct-import set. Scientific baseline
  loaders and `realworld/train_store.py` retain `s4d.third_party`, which resolves to the ops service. `baseline_data`
  imports `TARGET_TASKS` lazily from scientific configs, with no planner dependency. `scripts/_bootstrap.py` imports
  upstream identity directly from ops; its EGL bootstrap remains in place. `scripts/realworld_learning.py` and frozen
  registrars retain the planner alias. The indirect `s4d.rl.vecenv` -> `_bootstrap` -> GPU guard edge is outside the
  direct-import boundary test.
- Real-world stage and calibration code imports the implementation packages directly. `calib.loma.run` shares
  constants/helpers through `calib.dw.common`; COLLADA parsing is private to `robotics.collada`. Package initializers
  do not eagerly import implementations.
- The explicit-import graph has no eager import cycles. Including lazy imports, existing cycles remain in scheduler/
  host roles/backend/containers, training loop/per-frame, robot model/visual, pipeline/calibration/storage and adapter
  registration. The planner's eager science-config import (`ops.jobs.realworld_plan` -> `s4d.rwlearn.configs`) is a
  one-way ops -> science edge; science reaches ops only through the aliases and the two lazy exceptions above,
  so there is no eager package-level cycle (the lazy cycles listed here are not a claim of an acyclic graph).
  `ops.resources.gpu_guard` imports neither `s4d` nor other ops modules.
- The ML-specific CUDA kernels and device code stay in `s4d/model` and `s4d/train`.
- `ops/remote/artifact_check.py` and `ops/jobs/replay_host.py` must stay standard-library only, because they are executed on the remote host outside the containers.
- `ops/datasets/` (except `control.py`) and `s4d/data/droid/largescale/extract.py` must stay standard-library only for the same reason (`tests/test_dataset_downloads.py` runs the remote entry point isolated).
- `ops` never imports `runs/setup/tools`; the moved implementations import each other through their `ops` paths. Nothing in `s4d`, `ops`, `scripts` or `tests` imports an `ops.campaign` module, because importing one runs its command (`tests/test_ops_layout.py`).
- The implementations moved from `runs/setup/tools` keep their hard-coded campaign root `/home/ws/ws/splatter4d` and the `sys.path` inserts of it, so they act on the main checkout's live state (registry, queue, locks, remote host) from wherever they are imported, a worktree included. Tests import only the import-safe ones, and only in subprocesses. Run their entry points from the main checkout: an entry point in another checkout (a worktree) runs that checkout's `ops`, so its scheduler module reads that checkout's `experiments/registry.jsonl` and `queue.yaml`, while every other hard-coded path still names the main checkout. Before refactor pass 2 a worktree's copy of these tools used the main checkout's scheduler. In the main checkout both are the same.

## Compatibility aliases

The old module paths remain importable. Each one is a short shim that replaces its `sys.modules` entry with the implementation, so the old and new names refer to the same module object. Monkeypatching, `isinstance`, and pickles that use the old path all behave as before. When a shim is run as a script (including `python -I`), it executes the implementation as `__main__` and leaves `sys.argv` unchanged, so the usage text stays the same. New code should import from the implementation paths in the table (`ops` for operations; `s4d.realworld.calib`, `stages` and `robotics` for real-world research).

| Old path | Implementation |
| --- | --- |
| `s4d.realworld.calibration`, `calibration_rig`, `calibration_board`, `calibration_release` | `s4d.realworld.calib.core`, `rig`, `board`, `release` |
| `s4d.realworld.rig_dw_{boardcv,data,extext,fixedness,frames,geom,matches,publish,robot,run,solve}` | `s4d.realworld.calib.dw.<same suffix>` |
| `s4d.realworld.rig_loma_{corr,data,depth,publish,render,run,solve}` | `s4d.realworld.calib.loma.<same suffix>` |
| `s4d.realworld.calibration_stage`, `depth_stages`, `ir_stereo_depth` | `s4d.realworld.stages.calibrate`, `depth`, `ir_depth` |
| `s4d.realworld.baseline_stages`, `trackeverything_stage`, `track_motion` | `s4d.realworld.stages.semantics`, `trackeverything`, `motion` |
| `s4d.realworld.robot`, `robot_visual`, `mounts` | `s4d.realworld.robotics.model`, `visual`, `mounts` |
| `s4d.rwlearn.launch` | `ops.jobs.realworld_plan` |
| `s4d.third_party` | `ops.provenance.upstreams` |
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

The 31 real-world aliases are 16-line module-object shims (`tests/test_realworld_layout.py`). Existing `python -m s4d.realworld.rig_loma_run ...` commands and isolated script paths still work with the original argv and usage. Definitions report their implementation module (`__module__`, e.g. `s4d.realworld.calib.core`), so new pickles name the new path; pickles that name an old path still load through the aliases. `calib/release.py` computes both the ingest-worker path and the git working directory from its new location, resolving to the same paths. Package initializers contain only docstrings; the worker inventory, stage names/versions, fingerprint inputs and runtime file formats stay unchanged. The physical source paths and bytes do change, so source fingerprints require fresh evidence.

Source fingerprints cover `mujoco_mig_setup.py`, `s4d/`, `ops/`, `scripts/`, `tests/` and `configs/`. The four scopes are `scripts/_bootstrap.source_fingerprints`, `ops.jobs.scheduler.source_identity`, `ops.jobs.remote_backend.commit_source_sha256` and the run-identity native diagnostic. Test evidence therefore covers the operational code, since refactor pass 2 including the campaign drivers and tools moved from `runs/setup/tools` (which is not fingerprinted): an edit of a driver now needs a fresh full suite before long work, like any other source edit. For commits that predate `ops/`, the fingerprints are unchanged.

## Common tasks

Every GPU command needs exactly one allowed UUID (`source scripts/gpu_env.sh`). Entry points call the guard before importing torch or MuJoCo.

```bash
# Train (pretraining) and evaluate
CUDA_VISIBLE_DEVICES="$GPU4" .venv/bin/python -I scripts/train.py --config configs/metaworld/base.yaml configs/metaworld/tasks/hammer.yaml --name <run>
CUDA_VISIBLE_DEVICES="$GPU4" .venv/bin/python -I scripts/evaluate.py --run outputs/<run>
# RL: scripts/train_rl.py, train_sac.py; evaluation: scripts/eval_rl.py

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

**Single branch: `main` (since 2026-11-09; formerly `splatter4d`).** There is no other long-lived branch. Prepare the
candidate in a clean detached worktree, then publish that exact commit as a fast-forward (never force):

```bash
# From the clean detached candidate worktree; use the full candidate SHA.
git push origin <candidate-sha>:refs/heads/main
```

From the main checkout, verify that candidate using the clean detached worktree as the sync source:

```bash
.venv/bin/python -I runs/setup/tools/release_chain.py <full-sha> --image <digest> --tag <tag> --native-gpu <reserved-gate-GPU-UUID> --sync-repo <clean-detached-worktree>
```

After the release chain passes, run a dated `runs/setup/tools/merge_adopt_<sha>_<YYYYMMDD>.py <verify-id>` tool in the
main checkout. Prepare it with the full candidate and accepted-base SHAs. Under the admission hold
(`runs/setup/remote-hold-admissions`) and both driver locks (`stage2-driver.lock` and `remote-driver.lock` under
`runs/setup/`), it must fetch only `origin main`, assert `refs/remotes/origin/main == candidate`, assert `HEAD` equals
the full accepted base and the current branch is `main`, and require successful `merge-base --is-ancestor BASE
CANDIDATE`. Then `merge --ff-only` the candidate and run `adopt_remote_suite_evidence.py <verify-id>` to adopt the
checksum-verified suite. Before releasing the locks, assert `HEAD == candidate`, branch `main`, and upstream
`origin/main`. Preserve any existing admission hold unchanged; remove only a temporary hold created by this tool.
Use the main-to-main flow, without the branch-conversion steps in the historical rename adoption tool.

`sync_code.sh` always verifies pushed ancestry against `origin/main`. It selects the proof's branch/ref pair from
the target commit's own verifier: a literal `RELEASE_ORIGIN_REFS` mapping accepting `main` uses `main` /
`refs/remotes/origin/main`; otherwise it uses `splatter4d` / `refs/remotes/origin/splatter4d`. Both forms record
`verified_against: refs/heads/main` and install the verified tip at the release checkout's `origin/main`; the legacy
form also installs that tip at its `origin/splatter4d`. This compatibility ref does not require a server-side legacy
branch. Existing releases stay launchable, and re-syncing re-proves each in the form its own verifier accepts.

## Working with Codex (user policy v2, 2026-11-09, in force until the user changes it)

- Use the shared account first, through plain `codex` (`CODEX_HOME=/root/.codex-public`, provider `cliproxyapi`; the
  credential is sourced from `~/.config/cliproxy/claude.env` inside the same shell call and never printed).
- Pin the model per invocation with `-m gpt-6.1-sol`. Never use Astra and never substitute another model silently.
  Use `xhigh` reasoning for difficult implementation and review work, and cheap effort (low or medium) for trivial checks.
- Fall back to the user's personal account, `codex -c 'model_provider="openai"'`, only when the shared account has
  exhausted its allowance (`usage_limit_reached`, `model_cooldown`, HTTP 429) **or is in a known cooldown**. Check the
  effective provider and model in the Codex header.
- Never change global credentials, log out, or overwrite account configuration to switch providers. Respect each
  account's permissions and usage controls.
- Delegate bounded implementation, dependency audits, test analysis and independent review to Codex. Claude coordinates,
  judges the findings and integrates the work. Give each run only the relevant context. Avoid unnecessary agents,
  repeated full-repository reviews and redundant review cycles, and reuse earlier findings.
- **If neither account is available, Claude does the work immediately**, using Claude subagents where useful. Codex
  availability never pauses executable work: do not wait or poll for credits. Label Codex and Claude reviews accurately
  in records.
- Wrapper: `/home/ws/ws/validation-20261031/codex/run_codex.sh <workdir> <read-only|workspace-write> <prompt> <log>
  [effort=xhigh] [auto|shared|personal]`. `auto` tries the shared account and falls back on exhaustion or a known cooldown
  recorded in `/home/ws/ws/validation-20261031/codex/.shared-cooldown-until`. The wrapper prints the account, provider
  and model, fails unless the model is `gpt-6.1-sol`, and writes the final answer to `<log>.last.md`. If neither account
  is available, it exits **10** with **`CODEX UNAVAILABLE`**; Claude immediately continues the work itself.

## Deliberately not moved

- `scripts/run_tests.py` and `scripts/check_gpu_isolation.py` are typed verification job scripts. They run by path inside the containers, and the scheduler requires a literal guard call in the script file (`GUARD_CALL`), which a wrapper would not contain.
- `scripts/remote/entry.py`, `package_runtime.py`, `Dockerfile` and the shell helpers stay at fixed in-container and image-bundle paths. Moving them would change the image bundle.
- `analysis/eval_config.py` prepares scientific evaluation configs and is run by the Stage 2 driver at that path.
- In `runs/setup/tools`, refactor pass 2 moved only the reusable operational drivers and libraries (alias table). Everything else stays byte-identical:
  - dated one-off tools (`*_YYYYMMDD.py`) and `.before-*` backups: campaign records that the logs cite by path;
  - the remote runner sources `remote_runner.py` (a copy runs on the remote host), `gen_remote_autonomy_plan_20261031.py`, `update_remote_plan_20261031.py` and `reconcile_remote_runner_20261031.py`;
  - stage-specific registrations (`register_*.py`) and reports (`rl_reporting.py`, `stage2_report.py`, `followup_report.py`, `followup_tables.py`, `baselines_eight_task_report.py`, `final_comparison.py`, `crop_reeval_report.py` and their `report_figures.py`): analysis records tied to fixed registrations, cited by docs, not operations;
  - research diagnostics (`actfut_*`, `afk2_*`, `chamfer_*`, `enc_*`, `encp_*`, `grad_conflict_diag.py`, `variant_repr_probes.py`, `paper_ablation_smoke_local.py`, `selfrender_*`);
  - retired or one-time operational tools: `evict_partial_cache.py` (retired 2026-10-21), `enc_gate.py` (retired 2026-10-25), `remote_container_cleanup.py` (the 2026-10-21 cleanup; the scheduler removes containers since), `queue_mig_limits.py`, `apply_stage2_placement.py`, `record_cuda_init_outage.py`, the provenance snapshots `job_provenance.py` and `local_device_provenance.py`, and `check_final_stage3_dispatch.py` (a Stage 3 fixture check that parses the remote driver's source at its old path; it already failed before the move, because the driver functions it executes need globals its namespace lacks).

## Deferred infrastructure moves (2026-11-09)

- `s4d/realworld/workers/{__init__,_framing,_protocol,run_many}.py`: defer as one controlled runtime release;
  package/submodule identity, sibling imports, spawn/pickle lookup and external worker paths need joint validation.
  Executable worker files and `WORKER_DIR` remain unchanged.
- `s4d/realworld/runners.py` `LocalRunner`: defer with worker transport; keep lazy kernel dispatch and ACK semantics.
- `s4d/realworld/calib/release.py` application/CLI: defer because external release tools substitute the source file
  into pinned releases; injected writers/planners must remain compatible before introducing dependencies.
- `s4d/realworld/calib/core.py` `usable_cpus`: defer until calibration release validation; sizing runs at import time
  and feeds solver process pools.
- `scripts/_bootstrap.py` EGL bootstrap: defer while evaluator parents can spawn children; UUID checks, MuJoCo setup
  and EGL verification must retain their order. Only its upstream-service import changes in this consolidation.
- TrackEverything runtime telemetry: defer with the pinned worker release; preserve cache ownership, torch injection,
  hashes and flags, and the executable worker path.

The remote checkpoint watcher's embedded reader also stays byte-identical: it checks the ZIP archive and asserts a
step key, unlike the local helper's integer-opcode check and RuntimeError. Whole-file and 1-MiB hash variants, atomic
writers and source-fingerprint policies have different contracts and remain separate. `replay_host.py` and
`artifact_check.py` remain standalone, standard-library-only files with no imports of the shared helpers.

## Operational consolidation (2026-11-09)

Shared helpers now have one import-safe implementation. Liveness and option parsing re-export the same objects;
`pool_placement.partner` re-exports `host_roles._partner`. SHA-256 retains 8-MiB reads. Canonical JSON keeps the
`run_identity._json_bytes(payload: dict | int)` signature through a forwarder to `hashing.json_bytes(value)`.
RAM reservation calls remain at their original driver initialization points. Pause commands keep their function
names and signatures: registry writes pass each command's current `REGISTRY`, and session scans pass its current
`identity` callback. Script-only root bootstraps preserve isolated execution after introducing shared imports; the
normal import path stays inert in the new support modules. Boot-id reads, command timing, guard literals,
fingerprint scopes and fixed paths stay in place.
The whole upstream service moves together, preserving `_git` monkeypatching through module identity; only the
`__file__` parent depth changes to resolve the same repository. Existing test behavior assertions are unchanged;
`tests/test_ops_layout.py` extends alias/location and boundary expectations, and `tests/test_ops_helpers.py` covers
the extracted primitives without importing campaign commands. Source bytes/locations change and require fresh
coordinator test evidence and the normal release gate.

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
| `scripts/train_{rl,sac,ppo,dreamer,flashsac}.py`: five copies of `rss_gb`, `atomic_save`, `append_jsonl` | `s4d/rl/run_io.py` (new), then imported by the five trainers; after the 2026-11-09 retirement only DrM/SAC remain |
| `s4d/train/perframe.py`: copies of `MOVING_SCORE`, `STATIC_SCORE`, `ACTIVE_OPACITY`, `_rows`, `_per_t`, `_nanmean`, `temporal_ramp` (as `_ramp`), `_float_set` | imported from `s4d/train/loop.py`, whose inline float32 cast of the decoded set is now the same `_float_set` |
| `s4d/data/droid/largescale/calibration.py`: hand-written `+0.5` integer-to-continuous pixel centres | `geometry.to_continuous_centres(K, integer_centres=True)` |
| `s4d/realworld/contract.py`: unused private `_array` | removed |

It also added the research code map and the entry-point table above. Behaviour evidence (public API signatures, `--help` of
every entry point except `check_native_render.py` and `_bootstrap.py`, resolved configurations, CPU numerical fingerprints of the Meta-World, DROID, export and real-world
paths) is in `/home/ws/ws/validation-20261031/refactor3/`.
