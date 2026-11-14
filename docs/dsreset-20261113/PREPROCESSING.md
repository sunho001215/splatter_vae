# Immediate local MIG preprocessing

Exact commands: RUNBOOK.md, operational copy
`/home/ws/ws/realworld-ops/dsreset-20261113/pipeline/RUNBOOK.md`.

The reusable dispatcher is ops/jobs/rwprep_mig.py; ops/campaign/rwprep_mig.py is
its thin entrypoint. Standard local placement rejects upstream prep and lacks its
worker environment. The dedicated queue reuses established rwprep tools/image and
ordinary stage fingerprints without weakening the campaign guard regex.

Compile248 A rows (drawer148, covering failure50, PNP failure50), reusing195 old
success outputs. All three A weight pins match local files; later teacher downloads
need not delay A. Coordinator inspects/installs the Docker-host image.

The workspace has no Docker access/systemd. A host service runs the pinned-image
utility controller with LIVE main locks, campaign PIDs/cgroup and canonical shared
bind aliases. Workers have no network/socket, read-only inputs/source, one accepted
1g.24gb MIG, CPU/RAM/swap/PID/tmpfs caps. Fstab and enabled host services survive
reboot; no guessed host venv or transient systemd-run is needed.

Admission: fresh MIG UUID/profile/parent/process evidence, one idle MIG/job,
138GB charge+growth ceiling inside150GB,60GiB host reserve, CPU quota/load/idle/PSI
and disk floors. Every dispatcher owns an independent durable priority hold;
ready/running/unknown prep blocks NEW training while healthy trainers continue.
Missing dependencies do not reserve. STOP drains/reconciles; three attempts,
durable launch intents, exact-name recovery and stage idempotence prevent duplicate
writers. Unknown vanished workers need explicit diagnosis, not deleted state.

A declares16GiB GPU/48GiB RAM/4CPUs (measured<=13.1GiB GPU). B declares7GiB GPU/
16GiB RAM with TE cap5.5GiB (measured<=5.61GiB GPU/6.2GB RAM). Historical40GB TE
and four-process LoMa cannot fit. LoMa is split into single-process matching jobs,
CPU rasterization,20GiB allocation cap/21GiB declaration: a conservative bound,
not a new measured peak. OOM remains visible.

Task-local LoMa graphs wait for A and automatically publish only on established
acceptance. B requires hashed accepted rigs; followup jobs emit ordinary calibration
release plans and native SF-TRAIN scene/bounds reports. No manual pass signoff or
extra source release is required. Scientific failures remain failures.

Remote uses --host remote with host-specific config, accepted16-MIG inventory,
disjoint IDs and coordinated local priority-hold mirroring. Current load~9000
correctly refuses admission. Keep the old dispatcher's STOP and ordinary remote
same-commit acceptance. Neither is a dependency of immediate LOCAL progress.
