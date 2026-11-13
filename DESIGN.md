# Ours continuation review fixes (2026-11-13)

This supplements [the continuation design](docs/ours_ext_20261113.md). All edits are confined to this worktree. The external design note, main checkout, queues, run directories and the original 200k checkpoint are unchanged.

## Entry-point acceptance

Both new scripts use the established `_bootstrap` preamble and an unindented `guard_gpus()` call before importing training code. `tests/test_ours_priority.py` compiles the scheduler's actual `GUARD_CALL` and `check_guarded` definitions from its AST and runs that acceptance function against both script files. It never imports a campaign entry point or executes either GPU script. Like existing launched scripts, these scripts now guard even before `--help`.

## Priority and starvation

Every candidate order uses numeric queue `priority` first (default 5, smaller first), followed by its previous key. Registration age, campaign rank, companion status, seed, encoder-pilot rank and family fairness cannot put an older priority-2 FQL row ahead of priority -1 Ours pretraining or priority 0 Ours FQL. Equal-priority behavior retains the previous ordering and capacity policy.

Paths audited and covered:

- Scheduler eligibility already orders `(priority, registration order)` and accepts negative priorities in `load_queue`; both local and remote ticks now also protect blocked priority tiers for GPU slots/memory, host RAM, CPU threads and disk.
- Remote `dispatch_key` prefixes its old key with priority. Remote candidates and movable pool units are planned in shared ascending priority tiers, keeping remote-first ordering within ties. Pool units keep their old internal order and rollback behavior.
- Remote release batches group by `(priority, release, image)` and dispatch in priority order. A higher-priority batch still pending after its scheduler tick blocks lower-priority batches in the pass. Different release pins cannot bypass the ordering. The release-chain driver itself has no campaign-row rank: `drive()` delegates its verification rows to the same scheduler. `pick_native_gpu()` ranks devices, not jobs; its existing device-choice key is unchanged. Existing release verification reservations and acceptance gates remain prerequisites for campaign work.
- Baseline local candidate sorting prefixes the complete previous order with priority, and passes priority into fair-share selection. Fair-share runs the unchanged family policy one priority tier at a time, charging earlier admissions as active reservations. Family guarantees and short-evaluator exceptions apply within a tier. A waiting ready row prevents lower tiers from borrowing capacity.
- The baseline driver's Stage 2 yield now preserves higher-priority candidates. The Stage 2 driver prefixes its old rank with priority, defers to higher-priority local ready rows served by another driver, and protects blocked tiers in its memory admission loop.
- The local placement planner already sorted by priority. It now protects blocked tiers and considers existing pending local reservations in priority order, excluding lower-priority unstarted rows while placing a higher tier. Running reservations are always charged. The pool mutation helper only changes host fields; its sorted list of moved IDs is not a dispatch order. Compatibility scripts in `runs/setup/tools` delegate to these implementations and were not edited.

**Backfill choice:** no newly admitted lower-priority work may consume capacity on a host while a higher-priority ready row waits for capacity. We hold the entire host against strictly lower priorities, recomputing from live reservations each pass. This is deliberately conservative: without reliable remaining runtimes, a smaller job on another GPU can still delay the head through shared RAM, CPU or disk. Same-priority backfill, including the existing hybrid-head reservation and family evaluator rules, remains unchanged. Running jobs are never stopped, moved, signalled or preempted by this policy. As running work drains, the higher-priority row receives the first capacity it can use. An impossible request also holds lower tiers until corrected; failing closed is preferable to silently violating strict priority.

All priority consumers in `ops/jobs` and `ops/campaign` were searched. No new nonnegative check was introduced. Tests load -1 through real queue validation and exercise competing ready priorities -1, 0, 2 and default 5. The coordinator's reprioritization of other rows is assumed as directed. The 0..9 operator `set_priority` tool/range is unchanged; the registrar writes explicit negative queue priorities directly.

## Complete AdamW resume

`validate_resume_state` now receives an optimizer built from the actual CPU model. Before staging or accepting the source/latest checkpoint it requires every parameter ID in every saved group to have exactly one state entry, the saved/live group counts and lengths to agree, both moment tensors to match the corresponding live parameter shape, and every scalar step counter to equal the checkpoint step exactly. Missing, extra or duplicate states, missing groups, fractional counters and incompatible shapes fail closed. The existing scheduler/science/schedule checks remain.

Continuation training also requests the same strict check inside `load_checkpoint`, before any state is loaded, so direct continuation-loop resumes cannot bypass it. Legacy loads retain their previous default behavior. The wrapper's temporary validation model/optimizer remain on CPU and are released before training starts.

Tests remove a state from a later group, remove a whole group, corrupt either moment shape and both shapes together, and change counters; both wrapper validation and strict checkpoint loading reject them. The actual tiny CPU train loop compares uninterrupted continuation with two resumed jobs at `num_workers=0` and `num_workers=2`, crossing an epoch boundary. LR, model weights, optimizer state, RNG restoration and sampled indices remain identical. This validates the deterministic multiworker stream without running the full real-data/GPU model.

## CPU verification

The repository conftest enforces a native GPU before collection. CPU runs therefore use `--noconftest`, GPUs hidden, `nice -n 10`, and four OMP/MKL/OpenBLAS threads. A temporary `/tmp/ours_cpu_policy.py` plugin reuses only the existing reservation-isolation fixtures (AST-extracted from conftest) and `tests.remote_regime._s4d_remote_regime`, needed by legacy remote fakes. It does not stub the native guard or expose CUDA. Scheduler tests use synthetic queues/registries under `/tmp`; campaign policy is tested through actual AST fragments, with no driver entry points imported or executed.

The legacy `test_detached_launch_success_restart_once_then_fail` requires a real GPU in its worker and is excluded from CPU verification. No GPU integration or operational launch is authorized here.

Final results:

- Affected files plus scheduler/driver coverage: **1,179 passed, 2 deselected**, 105.57 s. Files: `test_run_tests_sharding`, `test_ours_priority`, `test_ours_continuation`, `test_ours_ext_registration`, `test_ours_length_eval`, `test_fair_share`, `test_realworld_local`, `test_jobs`, `test_remote_jobs`, `test_remote_mig`, `test_release_reservation`, `test_release_repin_review`, `test_scheduler_daemon_reexec`, `test_host_roles`, `test_remote_launch`, `test_pool_pin` (all under `tests/`). The second deselection is `test_the_metaworld_marker_is_registered_and_used`, whose nested pytest imports the GPU-enforcing conftest.
- After the final local-packing refinement, `test_realworld_local.py` and `test_ours_priority.py`: **198 passed**, 16.83 s. The planner now recomputes each tier with earlier selections included as pending local rows, preserving their higher priority ahead of existing lower-priority pending locals. A failed higher-priority pool claim also holds lower-priority remote dispatch and removes its abandoned admission charges.
- Targeted Ruff checks pass for both new entry scripts, continuation/checkpoint validation, local planner, fair-share/barrier code and new priority/resume tests. `git diff --check` passes. CPU training emits only the expected pin-memory warning when CUDA is hidden.

The final broad command used the environment described above, `PYTHONPATH=/tmp:.`, `python -m pytest --noconftest -p no:cacheprovider -p ours_cpu_policy -q`, the 16 files listed above, and `-k 'not test_detached_launch_success_restart_once_then_fail and not test_the_metaworld_marker_is_registered_and_used' --basetemp=/tmp/ours-review-tests-final`. Logs are `/tmp/ours-review-tests-final.log` and `/tmp/ours-review-planner-final.log`. Collecting the sharding file before legacy pool helpers keeps its namespace-package imports in this worktree; those helpers prepend the canonical main path at import. An earlier combined run exposed this harness import-order issue and was corrected without editing the main checkout.

An initial broad attempt also entered the legacy detached-worker test with CUDA hidden; it was interrupted during its wait and excluded from the final CPU run. No native-GPU worker passed its guard. Existing uncommitted `docs/tests.*` and collection/shard artifacts were present before this task and are intentionally excluded from this commit. No requested code fix remains open; native-GPU integration and operational rollout remain outside this task.
