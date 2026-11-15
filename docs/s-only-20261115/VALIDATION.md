# Validation

The complete native suite passed on the local host: **4092 passed, 0 failed,
0 errors, 0 skipped**;4065 collected tests plus27 reported subtests. Duration:
710.77seconds. All10 shards exited0. The sole visible GPU was authorized GPU7 MIG
`MIG-0b9a2ef1-2f30-5730-ab39-c31e8e7242b6`.

Command (from this worktree):

```bash
CUDA_VISIBLE_DEVICES=MIG-0b9a2ef1-2f30-5730-ab39-c31e8e7242b6 \
OMP_NUM_THREADS=24 MKL_NUM_THREADS=24 OPENBLAS_NUM_THREADS=24 S4D_TEST_CPU_THREADS=24 \
/home/ws/ws/splatter4d/.venv/bin/python scripts/run_tests.py --shards-cpu 8 --shards-gpu 1
```

`docs/tests.json` has `full_suite_passed=true`, `source_changed_during_suite=false`,
empty problems and exact full collection coverage. Its complete source/upstream
fingerprint map was independently compared with `_bootstrap.source_fingerprints()`
afterward and matches. `docs/tests.log` and `docs/tests.junit.xml` are retained.
Intermediate temporary test directories/cache reports were removed after completion.

Focused admission regression set: **23 passed**. The offline DINO environment
repair check also passed. The first two incomplete attempts were environmental:
uninitialized submodule checkouts, then missing pre-provisioned DINO weights.
Local submodule clones use exactly the repository gitlink pins; DINO weights were
copied locally only after matching official SHA256
`55c8b267f479b614ec20858947ce361ba804846c193e20be74263a096bb6d75e`.
No test was waived, skipped, patched or mocked to mask those missing assets.

The actual never-launched PNP Chamfer0.1 row's owned execution-source proof passes:
`admission-proof.json` records8 old-release MIGs and16 eligible current MIGs, with
no row/ownership/registry/result writes. The S-only registrar preview also passes:
98 qualified S drawer rows, zero registrations, no staging. Audit invariants and
shell syntax were checked separately. The coordinator-only registrar-stop script
was run in preview mode only; no live process was signalled.
