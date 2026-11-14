# Final S/SF handoff — 2026-11-14

Tested source: **0cb605b9fd24a8633907479082fd0ada599e48ca**.
Evidence worktree: `/home/ws/ws/splatter4d/.worktrees/dsreset` (keep HEAD here).
Final candidate: branch `dsreset-20261113-release`, docs-only descendant in
`/home/ws/ws/splatter4d/.worktrees/dsreset-release`. Resolve its full SHA with
`git rev-parse dsreset-20261113-release`. Fingerprinted sources are identical.
Current main **9c901bf7b0acabe399420c9b5e9a1c2b6bc42479** is merged; re-merge
returned already up to date after main moved during verification.

Full `scripts/run_tests.py --shards-cpu 6 --shards-gpu 1` PASSED:
**3989 passed, 0 failed/errors/skips** (3962 collected tests +27 subtests), exit0,
793.31 seconds. One authorized local MIG c3511b41-5f78-52f1-9b93-9c6794cdbbdd;
source_changed_during_suite=false. Expected commit0cb605b. Report/log/JUnit are in
`docs/tests.{json,log,junit.xml}`. Earlier failed/interrupted runs are superseded.
Focused calibration/criteria/dispatcher/gate selection172 passed.

The complete ordered coordinator runbook is
`/home/ws/ws/realworld-ops/dsreset-20261113/pipeline/RUNBOOK.md`.
Its permanent host mount/service helpers and staging commands are archived beside
this document. Syntax/embedded Python/bash checks passed; Docker-host/network
operations were not executed. Shared per-tick registry lock plus nonblocking
stage2 lock serialize BOTH local scheduler/driver launch routes. The controller
uses campaign PIDs/cgroup and the pinned prep image Python: the workspace itself
has no Docker/socket/systemd and its venv is not a host interpreter.

Immediate action: stage matching LOCAL phase-A weights, inspect/install the host
prep image, install exact verified host aliases, adopt source through the standard
helper, compile248 A jobs and enable dsreset-prep@local-phaseA. All SAM3/FS pins
match local files: no weight download is needed before A. Download two later
teacher pins and stage LoMa assets while A runs. Do not revive the stopped old
whole-GPU remote dispatcher or weaken its load admission. Remote is optional,
requires disjoint jobs, shared remote launch lock and a fail-closed local hold.

Drawer amendment: configs/realworld/gate_amendments/drawer-open-v2-98-20261114.json,
SHA256041042f1ff98d220b20dfd5ee159ed113eb79ad29a357134f246e8e5514ccbe7.
Exclude77/79 identically from S/SF; raw untouched. Standard validator independently
reconfirmed both failed. 79 lacks a recorded modulus/accepted repair contract.
Authoritative canonical manifests: drawer S98/SF148, covering S98/SF148, PNP S97/SF147.
248 new prep episodes,195 reused successes,443 total distinct valid recordings.

Task-local LoMa-G jobs, bounded phase B/QA, ordinary release records, NEW ingest
snapshot, native SF-TRAIN bounds and drawer scene measurements are runnable on
the same route. Passes publish/consume automatically under established criteria;
no new manual acceptance is introduced. Runtime measured drawer SNeRL settings
avoid another source release. Historical40GB LoMa/TE routes do not fit these MIGs.

Final real queue GRAPH preview: pipeline/registration-preview-final.json,461 NEW
rows, registered=false, final tested source pin.477 template minus16 equivalent
existing Ours-S producers. Dry-run/execution still require real accepted artifacts
and the named placement pause. Existing healthy jobs/queue/registry/pauses were
not changed by the implementer. No network, training/prep operation, deployment,
robot actuation, inference restart or active-policy reload was executed.

Release adoption remains a coordinator operation (reviewed push + normal adoption
helper; even its dry-run fetches network). Use frozen evidence worktree dsreset
and final docs-only candidate branch. Then follow RUNBOOK in artifact-gate order
through461 registered rows and persistent ordinary placement/scheduling.
