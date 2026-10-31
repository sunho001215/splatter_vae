# Developer guide

## Directory map

| Path | Contents |
| --- | --- |
| `s4d/` | Research code: `data/` (DROID, Meta-World, held-out sets), `model/` (encoder, decoder, Gaussians, CUDA rendering), `losses/`, `train/` (loop, DDP, checkpoints, augmentation), `rl/` (agents, environments, evaluation), `baselines/`, `diag/` (W&B logging, diagnostics), `config.py`, `geometry.py`, `third_party.py` (pinned upstream source trees for the baselines). |
| `ops/` | Operational tooling. It is not imported by the research code except at the two points listed below. |
| `ops/remote/` | `access.py`: SSH/rsync transport, tunnel check, download budget. `artifact_check.py`: read-only on-host inspection (standard library only; its source is sent over SSH and executed by the remote host's python3). `sync_media.py`: local backup of remote runs. |
| `ops/jobs/` | `scheduler.py`: file-based scheduler (queue, registry, admission, launches). `remote_backend.py`: the scheduler's remote-host backend (containers over SSH, in-place verification, acceptance evidence). `run_identity.py`: run provenance and clean-release enforcement. `lifecycle.py`, `replay_host.py`, `cleanup_replay.py`: replay cleanup of completed RL runs (`replay_host.py` is also sent to the remote host). |
| `ops/resources/` | `gpu_guard.py`: fail-closed GPU UUID allowlist, runtime-path and EGL checks. `nvidia_query.py`: bounded `nvidia-smi` queries. `heartbeat.py`, `watch_events.py`: stall and event monitoring. `host_equivalence.py`: validation of the cross-host Meta-World diagnostic. |
| `ops/containers/` | `launch.py`: the remote `docker run` launcher (run by `scripts/remote/docker_run.sh`). `verification.py`: typed release-verification jobs (native suite, GPU isolation). |
| `scripts/` | Command-line entry points (training, evaluation, data, figures) and `_bootstrap.py` (entry-point preamble: repo on `sys.path`, GPU guard, caches, test-evidence gate). |
| `scripts/remote/` | Image and container plumbing that is used at fixed paths: `Dockerfile`, `package_runtime.py` (both copied into the image bundle), `entry.py` (the in-container entry), and shell helpers. |
| `configs/`, `tests/`, `experiments/`, `runs/`, `docs/` | Configurations, tests, scheduler queue/registry, run outputs and campaign records, documentation. |

## Dependency boundaries

- `ops` may import `s4d`. It currently imports `s4d.third_party` (source identity) and, lazily, `s4d.model.encoder` (encoder export).
- `s4d` does not import `ops`, with two documented exceptions. Both are lazy imports of `ops.resources.gpu_guard` inside functions: `s4d/data/__init__.py` uses `validate_runtime_path` to authorize output paths, and `s4d/diag/wandb_log.py` uses `HOST_NAME` for the W&B host tag. `tests/test_ops_layout.py` enforces this.
- There are no import cycles. `ops.resources.gpu_guard` imports neither `s4d` nor any other `ops` module.
- The ML-specific CUDA kernels and device code stay in `s4d/model` and `s4d/train`.
- `ops/remote/artifact_check.py` and `ops/jobs/replay_host.py` must stay standard-library only, because they are executed on the remote host outside the containers.

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

`from scripts import jobs`, `import jobs` (with `scripts/` on `sys.path`) and `.venv/bin/python -I scripts/jobs.py ...` keep working. To get an independent copy of a module in a test, load the implementation file (for example `ops/jobs/scheduler.py`) with `importlib.util.spec_from_file_location`, not the shim.

Source fingerprints cover `mujoco_mig_setup.py`, `s4d/`, `ops/`, `scripts/`, `tests/` and `configs/`. The four scopes are `scripts/_bootstrap.source_fingerprints`, `ops.jobs.scheduler.source_identity`, `ops.jobs.remote_backend.commit_source_sha256` and the run-identity native diagnostic. Test evidence therefore covers the operational code. For commits that predate `ops/`, the fingerprints are unchanged.

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

# Tests (run targeted files; the full suite writes docs/tests.json evidence)
CUDA_VISIBLE_DEVICES="$GPU4" .venv/bin/python -m pytest -q -p no:cacheprovider tests/test_ops_layout.py tests/test_jobs.py
CUDA_VISIBLE_DEVICES="$GPU4" .venv/bin/python -I scripts/run_tests.py
.venv/bin/ruff check s4d ops scripts tests
```

## Deliberately not moved

- `scripts/run_tests.py` and `scripts/check_gpu_isolation.py` are typed verification job scripts. They run by path inside the containers, and the scheduler requires a literal guard call in the script file (`GUARD_CALL`), which a wrapper would not contain.
- `scripts/remote/entry.py`, `package_runtime.py`, `Dockerfile` and the shell helpers stay at fixed in-container and image-bundle paths. Moving them would change the image bundle.
- `s4d/third_party.py` serves the research baselines (`source_tree`). `ops` only reads its identity helper.
- `runs/setup/tools/*` holds campaign records that the logs reference by path. Most of these tools use `from scripts import jobs` and `s4d.*` operational imports, which the aliases keep working. They are phase-2 candidates.
