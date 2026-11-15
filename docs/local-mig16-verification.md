Temporary local MIG16 expansion — 2026-11-15

User authorization: 16 x MIG 1g.24gb on GPUs 4-7, RAM ceiling 225,000,000,000 bytes.
The branch was created from ee85ad9 as requested, then fast-forwarded inside the isolated worktree to
736d821 (four already-adopted main commits) before committing the expansion. Main itself was not changed.

Compatibility: ops/jobs/local_devices.py reads the planned commit's literal guard/config allowlist using git,
without executing old code, and intersects it with the actual entry-point checkout and current authorization.
Stage 2 packing, baseline fair-share, real-world placement, priority reservations, scheduler choice and the final
launch check share this predicate. A refused device leaves the row pending without creating a process or attempt.
Pinned scripts receive their own host config so an old guard never loads main's expanded 16-device config.
Read-only audit of all 12 distinct current local queue pins found eight accepted devices for every pin.
The committed-new-release test accepts all sixteen. Remote container launch continues to refuse local devices;
its local exclusion set derives from APPROVED_HOST_GPUS and all four LOCAL_MIG_PARENTS.

Memory accounting keeps clean-file-cache exclusion, 10 GB transient overhead, 2 GB staging reserve and the
60 GiB host reserve. Training/preprocessing admission follows the shared 213 GB training ceiling; explicit
preprocessing queue caps remain binding. Device and fair-share capacities derive from the configured limits.

Queue tools: legacy top-level limits and nested hosts.local.limits are both supported; dry-run by default; --apply adds only eight local limits (max_jobs: 3, max_mem_gb: 22).
--rollback refuses and lists live GPU 6/7 jobs/processes, then removes those limits. Both mutation modes use
stage2-driver.lock followed by experiments/registry.lock, timestamped backups, fsync and atomic replacement.
Job rows are preserved byte-for-byte. No reserved queue key is introduced.
The shell rollback prepares a git revert in a separate candidate worktree for normal review/adoption.
The candidate-aware adoption helper retains complete-suite, pushed-origin/main, source-fingerprint and driver-lock
checks, accepts evidence from the new MIGs, and leaves the queue and remote holds untouched.
See runs/setup/tools/ROLLBACK-local-mig16-20261115.md for exact adoption, apply and rollback commands.

Verification:
- Full native suite: 4,040 collected tests + 27 subtests = 4,067 passes; zero failures, errors or skips.
  docs/tests.json; committed snapshot: docs/local-mig16-tests.json. Sources unchanged during the suite;
  final source fingerprints match the report exactly. Duration: 668.7 seconds.
- Command: CUDA_VISIBLE_DEVICES=MIG-0b9a2ef1-2f30-5730-ab39-c31e8e7242b6
  /home/ws/ws/splatter4d/.venv/bin/python scripts/run_tests.py --partition all --shards-cpu 4 --shards-gpu 1
  Environment: S4D_TEST_CPU_THREADS=32; the report records serial threads=32, parallel threads per shard=6.
- Operational tools: 12 passed, tests live beside the tools. Both queue schemas are covered.
  In-memory live-queue preview: 8 -> 16 local limits, job rows byte-identical; no write or lock creation
  (docs/local-mig16-queue-dry-run.json).
- Focused final fixes: 42 passed (pinned guards, malformed configurations, preprocessing and scheduler regression).
- CUDA: gpu_guard followed by a 4 MiB CUDA tensor allocation/matmul on GPU 6 MIG-f30be823-f40d-5292-9474-427746dd3703
  and GPU 7 MIG-0b9a2ef1-2f30-5730-ab39-c31e8e7242b6; both passed (docs/local-mig16-smoke.json).
- Lint: 25 changed Python files, zero introduced findings; 37 pre-existing style findings remain in the two prep files
  (docs/local-mig16-lint.json). Shell syntax and git diff whitespace checks passed.
- Recursive submodules initialized; DINO checkpoint copied into the candidate cache.
- No live queue, registry, process, topology, Docker, other worktree, push or adoption operation was performed.
  Test processes and outputs were confined to this candidate; only worktree/submodule Git metadata was created.

Implementation files (relative to 736d821):

- configs/hosts/local.yaml
- docs/gpu_isolation.json
- docs/local-mig16-smoke.json
- ops/campaign/baseline_local_driver.py
- ops/campaign/pause_pretrain_at_checkpoint.py
- ops/campaign/pause_rl_at_checkpoint.py
- ops/campaign/stage2_local_driver.py
- ops/jobs/fair_share.py
- ops/jobs/local_devices.py
- ops/jobs/migration.py
- ops/jobs/priority.py
- ops/jobs/realworld_local.py
- ops/jobs/rwprep_mig.py
- ops/jobs/scheduler.py
- ops/resources/gpu_guard.py
- ops/resources/memory_budget.py
- ops/resources/memory_watch.py
- runs/setup/tools/ROLLBACK-local-mig16-20261115.md
- runs/setup/tools/apply_local_mig16_20261115.py
- runs/setup/tools/merge_adopt_local_mig16_20261115.py
- runs/setup/tools/rollback_local_mig16_20261115.sh
- runs/setup/tools/test_apply_local_mig16_20261115.py
- runs/setup/tools/test_merge_adopt_local_mig16_20261115.py
- scripts/gpu_env.sh
- scripts/realworld_prep_register.py
- scripts/test_shards.py
- tests/test_gpu_guard.py
- tests/test_local_devices.py
- tests/test_memory_admission.py
- tests/test_realworld_local.py
- tests/test_rwprep_mig.py
