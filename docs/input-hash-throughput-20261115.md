Admission input hashing, 2026-11-15
=================================

The baseline driver's readiness loop calls `realworld_local.localize` while
holding `stage2-driver.lock`. `_inputs` calls phase-A `scheduler_inputs`, which
calls `verified_job`, `verify_envelope`, and the hybrid `require_gate`. Failures
become `LocalInputError` and remove that candidate's priority reservation. These
checks run before capacity selection, even when the row cannot fit immediately.
The registrar also calls `scheduler_inputs` for each row under its admission and
registry locks, after already verifying the shared gate.

Every row reread 3,849 shared files (17,770,579,900 bytes): the frozen bundle,
environment identities, qualification evidence, and measurement records. The
largest files were the 3,450,062,241-byte SAM3 checkpoint and 3,298,527,334-byte
FoundationStereo checkpoint. Qualification evidence includes six approximately
0.8 GB FoundationStereo HDF5 files for covering and pick-and-place. The 98 rows'
distinct raw recordings add 20,108,438,486 bytes that still require SHA checks.

`scheduler.input_verification_pass` reuses only file digests within one pass,
keyed by `(st_dev, st_ino, st_size, st_mtime_ns, st_ctime_ns)`. Every lookup stats
the path. A miss checks the opened descriptor before and after reading, and the
path after reading; a concurrent change fails closed without caching its result.
Inventory, permissions, pinned git HEADs, expected SHA comparisons, and placement
scope checks still execute for every row. Nested calls share the outer pass;
exit, including exceptions, discards the cache and restores the digest functions.
No persistent digest database is trusted or written.

The adapter lives in the scheduler's existing operational scope and temporarily
adapts the two existing digest functions. This preserves byte-identical
`local_phase_a.py` and `realworld_local.py`, whose sources are pinned in existing
placement records. Comparing phase-A execution scope with adopted `cbf8833`
returns `identical=True`, no differences, and only the scheduler as an existing
operational difference. The baseline loop, scheduler tick, phase-A registrar,
and phase-A placement entry point establish a pass. Verification stays inside
the established lock transaction; there is no new check-to-admission gap.

Read-only reproduction (from the candidate worktree):

```sh
PYTHONDONTWRITEBYTECODE=1 /home/ws/ws/splatter4d/.venv/bin/python \
  runs/setup/tools/benchmark_input_hashes.py --output /tmp/hcache-replay
```

The helper copies the real queue and registry once and snapshots ownership,
roles, and source identities. It calls only `localize`, never a driver or tick;
it takes no locks, launches nothing, and writes only the new `/tmp` directory.
It compares complete returned args/input records and refusal reasons, and checks
the candidate's source hashes against every row's existing placement scope.
Byte counts measure bytes passed through SHA, rather than physical disk reads.
`--snapshot DIR --cached-only` can replay the identical snapshot against its
saved `before.json` without repeating the uncached terabyte-scale reads.

Validation covers cache reuse, hardlink aliases, nested passes and expiry,
mtime/size/inode/ctime invalidation, concurrent mutation/replacement, exception
cleanup, resolver restoration, preserved test doubles, and identical phase-A
admission/refusal results after changes to weights, environment identities,
qualification evidence, raw recordings, job/gate manifests, and inventories.
The focused cache/input/phase-A regression run passed all 256 tests.

Measured replay (98 pending hybrid rows, local queue snapshot)
-----------------------------------------------------------

| Metric | Original SHA reads | Pass cache |
| --- | ---: | ---: |
| Elapsed seconds | 1,337.39 | 54.06 |
| Bytes hashed | 1,761,837,208,386 | 37,881,174,420 |
| Digest requests | 377,888 | 377,888 |
| Cache hits | 0 | 373,839 |
| Frozen bytes hashed | 700,000,638,092 | 7,142,863,654 |
| Evidence bytes hashed | 1,041,728,131,808 | 10,629,872,280 |
| Raw bytes hashed | 20,108,438,486 | 20,108,438,486 |

The final adapter is 24.7 times faster in this replay and hashes 46.5 times
fewer bytes. Every admission decision is identical. All 98 rows pass the input
checks and then refuse with `local code differs from the release the row was
placed with`, in both versions. Their placement records already differ from
adopted `cbf8833` in `ops/jobs/realworld_local.py`, `ops/jobs/rwprep_mig.py`, and
`ops/resources/memory_budget.py`. The candidate has exactly the same scoped
source identities as the baseline snapshot. This patch preserves those
refusals; the coordinator must reconcile the existing placement drift separately.
No queue, registry, ownership record, lock, or launch was written by the replay.
Synthetic regression cases also exercise identical successful admission records.

The original snapshot and complete decisions remain under
`/tmp/hcache-queue-benchmark`; the final adapter replay is
`/tmp/hcache-queue-benchmark-verified` (exit 0). Its compact summary is committed
beside this report as `input-hash-throughput-20261115.json`. The initial cached
pilot measured 54.99 seconds; the final repeat above validates the same snapshot
with explicit baseline/candidate source-drift diagnostics.

Complete guarded local native suite: **4,128 passed, zero failed/errors/skips**
in 706.37 seconds, partition `all`, one approved
`MIG-0b9a2ef1-2f30-5730-ab39-c31e8e7242b6`. Source fingerprints remained
unchanged throughout. `docs/tests.json`, `docs/tests.log`, and
`docs/tests.junit.xml` retain the complete adoption evidence in this worktree.
The existing isolated baseline policy test's scheduler fake now supplies the
real pass context; its original admission/reservation assertions still pass.
