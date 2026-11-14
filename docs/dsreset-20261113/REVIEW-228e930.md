# Independent review fixes for 228e930

The release branch already includes main 9c901bf. No preprocessing dispatcher,
pre/ route, operational queue, registry, placement, ownership or pin was edited.

* Snapshot joins use (task, directory label, filename), including exclusions
  whose ingest SHA is null. Original dictionary/record keys survive; conflicting
  hashes or duplicate identities are refused. The regression covers all five
  covering/pick-and-place exclusions and reevaluates the resulting manifests.
* tasks.py, configs.py, gate.py, identity.py and outcomes.py under s4d/rwlearn
  are byte-identical to main 9c901bf. New rows use scripts/dsreset_* entrypoints.
  These install process-local private gate/config/outcome modules and a separate
  label-aware identity module before importing the established trainers. Importing
  the new helpers alone never changes the legacy modules. Static execution scope
  traversal includes both each entrypoint and its original trainer.
* New readiness/measurement/input entrypoints also select that process-local
  runtime. The old generic training-plan document again lists its original tasks;
  drawer-open-v2 and S/SF rows belong to the paired registrar.
* The registrar checks the approved amendment itself, exact success filenames
  and hashes (covering 98, pick-and-place 97, drawer-open-v2 98 with 77/79 excluded),
  exactly 50 SF failures, identical S/SF success rows/splits, and identical
  amendment/exclusion evidence. Self-sized partial gates cannot authorize rows.
* New prep/calib/qa rows, including native-depth/workspace-stat preparation, have
  priority -4; pre/pre400k have Ours -3 / others 0; export/fqldata/FQL/eval have
  Ours -1 / others 1. Reused existing rows retain their priorities and scripts.

## Persisted-scope history (superseded by the combined release)

`execution-scope-audit.json` compares every ownership record read from adopted
main against the candidate, including third-party pins and a second comparison
against main 9c901bf. At capture there are 88 placed records, 84 with persisted
execution scopes. All have zero candidate-introduced file changes. The running
rwv2-20261113-pre400k-ours-covering-S scope matches every persisted digest.

78 older records still differ from the candidate AND adopted main. For example,
the six running pick-and-place E2E FQL rows pin older pyproject.toml and
ops/containers/suite_partition.py digests. Older historical scopes also disagree
with one another. Four legacy placement records have no persisted scope.
Restoring the five shared modules cannot reconcile these pre-existing mismatches.
No placement records or existing row pins were rewritten to conceal them.

The suite includes both the zero-release-changes regression and the requested
strict comparison of every persisted scope with the candidate. The strict check
must fail while these incompatible old records remain. `docs/tests.json` is the
actual full-suite result, not a passing acceptance assertion. Coordinator action
on obsolete placements / eligible retries is required to satisfy the absolute
zero-difference adoption gate; this candidate must not be advertised as accepted
while that gate fails.

## Validation setup

All four pinned submodules, including Dynamo, were initialized using local
submodule sources. DINO weights were provisioned at
.cache/torch/hub/checkpoints/dino_deitsmall8_pretrain.pth. Full-suite invocation:

```sh
S4D_TEST_CPU_THREADS=32 \
CUDA_VISIBLE_DEVICES=MIG-c3511b41-5f78-52f1-9b93-9c6794cdbbdd \
/home/ws/ws/splatter4d/.venv/bin/python scripts/run_tests.py \
  --partition all --shards-cpu 4 --shards-gpu 1
```

The new subprocess regression constructs a drawer-open-v2 store whose success
and failure recordings share a basename, checks the bound pretraining gate,
builds an FQL dataset, checks the FQL training gate, and verifies failure reward
-1 / bootstrap mask 1 versus successful terminal reward 0 / mask 0. It also
checks that importing/using the paired runtime leaves the original module
objects unchanged, and verifies the committed new-task baseline configs.

Full-suite result: 3,993 passed, 1 failed, zero errors/skips (3,967 collected
items plus 27 subtests), 785.0 seconds. All 440 GPU tests passed. Sources
were unchanged throughout. The sole failure is
`test_every_live_or_placed_persisted_scope_is_identical_to_candidate`.

The incompatible digests required by two running rows are recorded in
[incompatible-live-pins.json](incompatible-live-pins.json). No single file can
match both values; this is independent of S/SF source changes.

## Combined S/SF and container preprocessing release

The release now merges dsreset-prep-route (01746f4, source fb3baee) into the fixed
ce8b68e pipeline. Only generated suite-report conflicts needed resolution; the
reports will be replaced by a fresh combined full-suite run.

The acceptance invariant is relative to main 9c901bf7b0acabe399420c9b5e9a1c2b6bc42479:
for every persisted placement, candidate-vs-row and main-vs-row mismatch sets must
be equal, AND every scoped candidate digest must equal main. The latter also
rejects a third digest that leaves the mismatch filenames unchanged. Covering-S
still matches its persisted scope absolutely; all eight running rows are exactly
unchanged from main. Four old placements have no scope and are reported explicitly.

The deterministic ownership fixture is tests/fixtures/dsreset-live-scopes.yaml
(JSON syntax, YAML extension so the existing suite source fingerprint includes it).
It captures 89 placements, 85 persisted scopes, 78 inherited mismatches, all eight
running registry rows, original record SHA256s and registry SHA256. Nineteen unique
scope maps avoid duplicate fixture content. No operational records were edited.
The refreshed execution-scope-audit.json records the combined-tree comparison.
The earlier absolute-compatibility blocker and failed-suite figures above are
historical; the combined release uses this corrected invariant and fresh evidence.

The preprocessing, calibration and followup compilers now also emit priority -4
directly; the route prepare/attach overrides retain the same priority. These new
entrypoints are absent from every live persisted execution scope.
