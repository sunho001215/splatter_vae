# Combined S/SF release evidence

Candidate branch: `dsreset-20261113-release`, source worktree
`/home/ws/ws/splatter4d/.worktrees/dsreset-release`. Tested source commit: `95b3c4cbce6f2faa2549b253abf392bc245c51a4`.
It combines ce8b68e with dsreset-prep-route 01746f4 (route source fb3baee),
then fixes the preprocessing/calibration/followup compiler defaults to priority -4.
The final candidate adds only documentation and these unchanged generated reports.

## Scope invariant

For every persisted live/placed row, candidate-vs-row mismatch filenames equal
main 9c901bf7b0acabe399420c9b5e9a1c2b6bc42479-vs-row mismatch filenames. Every
candidate digest in those scopes also equals main, rejecting a third digest even
when the mismatch filenames would stay the same. Covering-S matches its persisted
scope absolutely. All eight running rows have persisted scopes and zero changes
from main. The 78 inherited placement mismatches remain exactly unchanged.

`tests/fixtures/dsreset-live-scopes.yaml` captures 89 ownership records (85 with
scopes, four historical records without them), 19 unique scope maps and all eight
running registry rows, including the new SNeRL pick-and-place pretraining row.
It records original ownership SHA256s and the registry snapshot SHA256. JSON syntax
with a YAML extension includes the fixture in the established source fingerprints.
The fixture replaces mutable host lookups in both regression tests. Synthetic
regressions also reject digest changes hidden behind an equal mismatch set.

`execution-scope-audit.json` is the deterministic snapshot audit. A fresh live
ownership audit after the suite also passes with 89 records and
zero candidate changes; external evidence is pipeline/combined-live-scope-audit.json.
No operational ownership, placement, pins, queue or registry was rewritten.

## Full local suite

**4009 passed, 0 failed, 0 errors, 0 skipped**, exit 0,
`full_suite_passed=true`; 3982 collected tests plus 27 subtests.
All 440 GPU tests passed. Duration: 794.10 seconds.
All six shards succeeded; every test file is covered exactly once. Before/after
fingerprints are identical, including the committed scope fixture. Candidate and
evidence-worktree fingerprints were independently checked against the report.

Evidence worktree: `/home/ws/ws/splatter4d/.worktrees/dss` at `95b3c4cbce6f2faa2549b253abf392bc245c51a4`.
Reports: `docs/tests.json`, `docs/tests.junit.xml`, `docs/tests.log`; unedited copies
are archived in this branch. All four top-level submodules and all five nested RVT
submodules were initialized recursively at their pins. DINO weights are available
at `.cache/torch/hub/checkpoints/dino_deitsmall8_pretrain.pth`.

```bash
cd /home/ws/ws/splatter4d/.worktrees/dss
S4D_TEST_CPU_THREADS=32 \
CUDA_VISIBLE_DEVICES=MIG-c3511b41-5f78-52f1-9b93-9c6794cdbbdd \
/home/ws/ws/splatter4d/.venv/bin/python scripts/run_tests.py \
  --partition all --shards-cpu 4 --shards-gpu 1
```

`expected_commit` is null in the actual runner report. The normal adoption helper
accepts null or the tested evidence HEAD, and explicitly permits documentation-only
commits on top when all tested source fingerprints match. The final candidate is
such a descendant; the evidence worktree stays at the exact tested commit.
No report fields were edited after execution.

## Preprocessing commands

`PREP-ROUTE.md` uses the combined release source worktree and short evidence path.
The external pipeline/PREP-ROUTE.md pins the final candidate SHA explicitly.
An offline recompile of the actual canonical plans produced exactly 248 unique
jobs, all priority -4: drawer success 98, drawer failure 50, covering failure 50,
and pick-and-place failure 50. See `combined-compiler-evidence.json`. It launched
nothing. All four established external helper SHA256s still match the route audit.

No push, adoption, deployment or real preprocessing/training launch was performed.
The old dispatcher STOP and all live execution scopes remain untouched.
