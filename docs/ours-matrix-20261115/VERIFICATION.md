# Verified candidate

Tested source: `5a4cb5381f93476b23c835d825c8acb032e5ac25`, based on main
`ee85ad967ad382e8cb3162ee1cc0ca16ad4d2965`. The final branch adds only documentation
and unedited evidence above this source. Branch: `rwv3-matrix-20261115`;
implementation worktree: `/home/ws/ws/splatter4d/.worktrees/rwv3`.

Evidence worktree: **`/home/ws/ws/rwm`**, detached at the exact tested source.
Its `docs/tests.json`, `docs/tests.junit.xml` and `docs/tests.log` are the original
runner outputs; byte-for-byte copies are archived under `evidence/` here.

Full local suite: **4049 passed, 0 failed, 0 errors, 0 skipped**, exit 0,
`full_suite_passed=true`, `source_changed_during_suite=false`. The runner collected
4022 tests plus 27 subtests; all **440 GPU tests** passed. Duration 851.18 seconds.
All six shards succeeded, with exact complete test coverage. Independent fingerprint
checks confirm both candidate and evidence checkout match the unedited report.
`expected_commit`/`partition` in the runner's report are null; the actual command
explicitly selected all tests. These null values are supported by the existing
local adoption helper; no report fields were changed.

```bash
cd /home/ws/ws/rwm
S4D_TEST_CPU_THREADS=32 \
CUDA_VISIBLE_DEVICES=MIG-c3511b41-5f78-52f1-9b93-9c6794cdbbdd \
/home/ws/ws/splatter4d/.venv/bin/python scripts/run_tests.py \
 --partition all --shards-cpu 4 --shards-gpu 1
```

All four top-level and five nested submodules were provisioned locally at their
pins; the existing DINO weights were copied into the evidence cache. Exact pins,
weight SHA and environment are in `evidence/environment.json`. The test-layout
check caught an initially missing classification for the new test file; that was
fixed before the passing complete run. Local-only registrar imports were also
made lazy so remote marker deselection can collect without the runs/tools bind.
Earlier incomplete/failed runs are not acceptance evidence.

Additional checks: **108** focused matrix/dsreset/layout tests, **86** existing
400k/continuation/task-eval registrar compatibility tests, and **55** final matrix
and shard-layout checks passed. The existing dsreset tests verify identical S
reward tensors, S success tensors inside SF, failure truncation/bootstrap masks,
label-inclusive identity and ordinary FQL learning from failures. No reward or
FQL implementation was changed.

Read-only live planning produced **85 additions**: 84 rows for the twelve scratch
chains plus the missing PNP continuation evaluator, zero old modifications/removals,
`registered=false`. Four S pretrains have their actual local mirror and statistics
closures verified; eight wait for the seven standard dsreset publication/statistics
IDs listed in `evidence/verification.json`. Both existing continuation chains have
export/fqldata/FQL s0–2, with covering's task-aware evaluation reused. There are
42 deployment hooks (36 matrix policies plus six continuation policies).

All 85 proposed rows passed both host role classifiers. Each of the ten distinct
entrypoint import/config scopes was checked against the pinned release. The
415-row standard dsreset publication preview, using
`ours_input_barriers_only=true`, contains no new legacy Ours learning chains;
its actual input validation remains mandatory at publication time. These are
non-executable previews; no unpublished data hashes or completion records were
invented.

The live ownership audit compares every recorded scope to the original main
baseline: **zero candidate scope changes**, with pre-existing drift/missing-scope
records preserved. `absolute_scope_identical=false` is retained honestly in the
report; the candidate does not repair or repin older drift. Shared training,
optimizer, scheduler, checkpoint, Chamfer/FQL and task-evaluation implementations
are byte-identical to base. The only shared resolver additions admit new matrix
evaluation configs and the new isolated paired evaluation alias.

No push, adoption, live hold publication, registration, placement, preprocessing,
experiment launch, Robot PC operation or remote operation was performed. Other
workers' running controllers, trials and jobs were left alone. The exact
coordinator commands are in RUNBOOK.md; VPN reconciliation is in
REMOTE-RECOVERY.md. The adoption helper's normal pushed-candidate prerequisite
still applies when the coordinator later performs the authorized release.
