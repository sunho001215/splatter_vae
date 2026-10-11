# Second-server campaign extension

Status (2026-10-11 17:10 +09:00): release `5a7177e` deployed; remote full native suite 728/0/0/0 and all four one-UUID isolation checks passed; cross-host v2 accepted as equivalent by the user (ratio 1.02, registered criterion not met); Stage 4 CNN DrM running remotely since 16:47 with offline W&B. Earlier history below; the 200e5e6 failure and later fixes are in RESULTS/EXPERIMENT_LOG.

## Connectivity and access

- Remote: `compu@192.168.10.22`; permitted host filesystem root is `/home/compu/kaist/sunho` (`~/kaist/sunho`). Its existence and ownership were checked at initial inventory; contents were not listed then.
- Initial check from the main Docker container at 2026-10-10 18:20 +09:00: TCP port 22 answered and returned an OpenSSH banner. The connection source was `172.17.0.5`. The container uses an `eth0` bridge-network interface, a connected `172.17.0.0/16` route, and the default gateway `172.17.0.1`. Its PID 1 and agent share the same network namespace. This is container-side evidence; the host route/VPN facts are user-provided, not independently inspected.
- The FortiGate VPN runs on the main HOST, not this container. No VPN start, stop, restart, reconfiguration, or host-network change is permitted or attempted.
- OpenSSH client was already installed. `sshpass` and `rsync` were installed only inside the main container. Credentials are read from the user-designated local credentials file, never copied or printed. Password authentication uses `sshpass -e`; passwords are never command-line arguments.
- SSH multiplexing: `ControlMaster=auto`, `ControlPersist=600`, control sockets and known-host records under local `runs/remote/access`. No remote SSH key installation or writes to `~/.ssh`.
- First observed SSH host key: ED25519 `SHA256:AbgsuCXv6V1LvlnFsHF+5qQPxXbZoBcPeP6OBBPUFWI`. Retain that key; never disable host-key verification on subsequent connections.
- Access helpers wait mechanically with bounded backoff (approximately 10 minutes) when requested. Scheduler reachability checks must mark an unreachable host unknown, not fail/relaunch its jobs, and must not hold up healthy local reconciliation indefinitely.

## Initial remote inventory (2026-10-10 18:25 +09:00)

| Resource | Observed |
|---|---|
| OS | Ubuntu 24.04.3 LTS |
| CPU | 384 logical CPUs |
| RAM | 1.5 TiB total, approximately 1.4 TiB available |
| SSD filesystem | `/dev/nvme0n1p3`, 7,528,923,684,864 bytes total, 4,118,894,247,936 bytes available |
| Docker | 28.5.1, account has Docker access, NVIDIA runtime present |
| GPU model | RTX PRO 6000 Blackwell Max-Q, 97,887 MiB each |
| Driver | 580.95.05; differs from the main host's 580.178.04 |
| Authorized GPUs | all four below idle at inventory (1 MiB each, no compute processes) |

| Remote index | Only permitted UUID |
|---|---|
| 0 | `GPU-4391fcee-537f-d408-d138-10d8a3866eea` |
| 1 | `GPU-6122c7a9-eaa6-b539-6005-f20e339da4ea` |
| 2 | `GPU-9e9f1e97-b2ca-04e0-eb2a-8035398daa79` |
| 3 | `GPU-ca65e1b7-c0ac-6757-7d3f-816c3119a4c1` |

Foreign work exists on remote GPUs 4-7. It is neither eligible capacity nor ours to stop. Recheck authorized-device processes before each admission; any foreign process blocks that device. Main-server GPU permission remains only its original GPU-4/5 UUIDs, regardless of remote capacity.

## Mechanics and decisions registered before remote results

1. **Data transfer, not regeneration.** Transfer only data needed by scheduled work: initially development-task HDF5 files, their fixed split manifests, workspace statistics, and held-out sets. Preserve container dataset paths so numerical configs need not change. Verify complete SHA-256 manifests before use. No renderer-noise allowance is invoked for transferred data. No DROID data or experiments are transferred.
2. **Local authority, detached execution.** The main queue, append-only registry, PROGRESS, EXPERIMENT_LOG, RESULTS, and decision evidence remain the sole record. The remote is an execution backend, not a second independent scheduler. Jobs carry `host=remote`, an immutable code commit and image digest, data/config identity, deterministic owned container name, and attempt number.
3. **Unknown is not failed.** Reserve a launch intent locally before contacting Docker. A dropped SSH response never justifies a second container. Reconcile the deterministic owned container after reconnection. A remote success satisfies dependencies only after required results have returned and checksums verify.
4. **Filesystem/Docker boundaries.** All remote host files, release checkouts, data, manifests, artifacts, and build contexts are beneath the allowed root, with symlink/realpath checks before writes. Only owned `s4d-` images/containers/networks; bind sources beneath that root; no named volumes, foreign-resource removal, or system-wide prune. Campaign code runs in detached Docker, not an SSH-bound process.
5. **Code identity.** Campaign jobs must use a clean immutable checkout of a commit already pushed to `origin/splatter4d`; never edit remote code. A shared decision uses the same commit, numerical configs, checksummed data, and protocol. Existing item-1 runs predate the extension and cannot retroactively acquire identical whole-commit provenance. Keep all 20 on the main host for now rather than claim migration is automatically equivalent.
6. **Runtime identity.** Build from the same `uv.lock`, matching `docs/runtime_versions.json` and exact prebuilt native artifacts for sm_120. No gsplat/fused-ssim source/JIT-build workaround. Store the immutable image digest and runtime/native hashes. Driver difference is explicit and must be tested, not dismissed.
7. **Validation gates.** Full suite inside the remote image and one-UUID CUDA/EGL isolation on each authorized GPU before long runs. Native tests are neither mocked nor skipped. Cross-host step-0 loss and short-run equivalence must be demonstrated before any decision mixes hosts. Main CUDA/native acceptance recovered; the clean pushed release exists, but fresh same-input comparisons have not been demonstrated and the bootstrap guard exception remains unauthorized.
8. **Resource admission.** Remote SSD free floor is at least 15% of total: 1,129,338,552,730 bytes (approximately 1.13 TB), recomputed from measured capacity if it changes. Also reserve declared future per-run growth, including 46 GB per 1M-step CNN replay, before launch. Initial remote host-RAM reserve: 120 GiB (about 8% of total), with the same launch-ramp accounting; use measured job PSS. Per-GPU GPU-memory cap 90 GiB and at most 20 job slots are ceilings, not permission to compete with foreign processes. Local disk floor remains 300 GB; deployed RAM reserve is now 128 GiB tenant headroom, with new heavy admissions held for long-window footprint evidence.
9. **Placement.** Item ordering is unchanged. Keep both decision arms/all seeds on one host when possible. After verification, prioritize item-1 full evaluations if needed, item-2 validations and both six-seed hammer RL arms on the remote, then leave-one-out/length work and planned Stage 3/4 capacity. Do not rerun favorable seeds, tune rules, or use non-development RL to decide the method. Baseline pretraining may fill spare capacity only under the existing campaign plan.
10. **Result return.** Sync metrics, evaluation JSONL, summaries, exports, logs and panels to the main server with resumable `rsync --partial --append-verify` plus checksums. Checkpoints return only on explicit need. Do not transfer/delete replay by broad patterns. Delete only a completed run's literal replay path after its final evaluation is written; preserve agent snapshots, metrics, and logs.

## Runtime build and integration evidence (2026-10-10 19:16-19:26)

- Immutable amd64 userspace base: `ubuntu:22.04@sha256:08ea48a03a3e78ebc7cd526e6a275053223aadd88bfc09cc49b06d5281525fde`.
- Owned image: `s4d-runtime-03cca5218096dc88`; immutable Docker image ID `sha256:c3ffea7f7c114dd12755f9d537f760a6d5b31c96c923233185b775e0324799b7`. It has no registry RepoDigest because it was built locally, not published; jobs must select the immutable ID, never the mutable tag.
- Bundle manifest SHA-256: `03cca5218096dc8837e63e10ef2ddc847af8a12787dd57e3eceb0be6381b2981`; unchanged uv.lock SHA-256 `66aa2a250c6081d33f4fa9861207099bbc82b3a19a5a8ef8869d6aedb1ee062a`. Python 3.10.19 and uv 0.10.4 are pinned; the uv archive checksum is checked before execution.
- The image build used only the authorized root for its context/logs and only an owned `s4d-` image. Exact installed native payloads were copied without native compilation. Nonroot UID/GID 1000; immutable interpreter/venv under `/opt/s4d`; code workdir `/workspace/splatter4d`.
- Detached CPU-only container `s4d-runtime-integrity-03cca5218096dc88` verified every recorded native package input (1,948 files) byte-for-byte and every runtime package version. It had no GPU device request, no network, and a read-only root filesystem; its sole writable bind was beneath the authorized root. This is byte/version evidence, **not CUDA/EGL or full-suite acceptance**.
- Local evidence: `runs/remote/runtime/build-03cca5218096dc8837e63e10ef2ddc847af8a12787dd57e3eceb0be6381b2981.{json,log}`, `integrity.json`, and `integrity-container.json` in the same directory. Native payload metadata preserves build-generator setuptools 84.0.0 separately from runtime setuptools 82.0.1.
- Integration fixes remain review-only: identical root-EGL-helper fingerprints across suite/scheduler/release checks; read-only reuse of a matching deterministic split manifest, refusal to rewrite mismatches; explicit Docker `--shm-size 8g` (capacity, not a batch/loader change); literal baseline `encoder.pt` export returned without requesting all checkpoints. Numerical result paths are mandatory before remote success releases dependencies.
- The resumable transfer completed at 19:33 +09:00: eight hammer/pick-place files totaling 40,611,169,250 bytes, with fixed split seed 0. Every remote file checksum was verified before final publication. Verified manifest: `runs/remote/data/56bd8ce9c59e81077061f577888f708541c2c14d605248f0cbdf81521e54bae1.json`; semantic identity `56bd8ce9c59e81077061f577888f708541c2c14d605248f0cbdf81521e54bae1`. No DROID files were transferred. Native dataset-loader acceptance and cross-host comparisons remain separate gates.
- Main-host follow-up: NVIDIA queries hung in kernel D state; at 19:22 a fresh process passed the unchanged original GPU-5 UUID guard and a real CUDA sum (16.0). The complete review suite started at 19:25 and failed after 1,675.4 seconds: 492 passed, one failed, no errors or skips, sources unchanged. Its real detached workers took approximately 904 seconds to exit with correct codes, exceeding the unchanged 180-second check. Exact blocked operation is not known; native diagnostics will expose it on the next run. Recovery still needs complete acceptance before source merge/push. No unpushed candidate is being launched remotely.
- A bounded verification-only bootstrap for creating the cross-host evidence was considered, but permission denied deferring the `cross_host` admission guard without explicit user approval. No exception is authorized or deployed. That guard remains unchanged; no fabricated equivalence report or alternate launcher/test route may be used.

## User infrastructure and shared-resource update (20:15)

- The user identified the main incident as host `systemctl daemon-reload` under the systemd cgroup driver removing
  NVIDIA runtime-hook device access for new container processes. They applied runtime DeviceAllow rules on the host;
  no agent host repair was attempted. Fresh bounded NVML query passed, followed by actual unchanged-main CUDA/EGL
  isolation on both approved local UUIDs at 20:02. Evidence is in `docs/gpu_isolation.json` and
  `runs/setup/gpu-isolation-host-fix-20261010-2003.log`. This does not accept unsettled review source or remote devices.
- Prefer CDI for owned remote containers: `--device nvidia.com/gpu=<UUID>`, never automatic legacy-hook fallback.
  Review implementation verifies Docker 28's actual CDI `DeviceRequests` encoding and retains exact CVD, graphics,
  immutable image, ownership, root and all acceptance gates. Actual remote CDI spec/device availability remains
  unverified. If remote NVML fails, hold remote admission and report; never repair remote host infrastructure.
- Shared-host memory declarations must cover measured long-window whole-tree footprint, loader shmem, pinned memory,
  hot data/replay page cache and retained allocator growth, with margin and meaningful headroom for other tenants.
  Historical 13 GB/PSS-only declarations are provisional, not complete-footprint acceptance. A read-only four-hour
  owned-session/cgroup/cache sampling window is being collected; no running comparison's loader/config changes.
  After all remote gates, favor RAM-heavy pretraining, CNN replay and wide RL sweeps there without splitting decision
  arms across hosts absent actual equivalence. See COMPUTE_PLAN for measured evidence and admission policy updates.
- The cross-host bootstrap exception remains unauthorized. The stopped observer assignment made no edits; no new
  execution path or gate deferral was implemented.

## Reconciliation isolation follow-up (20:34)

Review lost-response reconciliation now verifies nonprivileged/read-only-root configuration, host PID namespace,
exactly 8 GiB shm and all eight exact bind source/destination/RW pairs before recognizing ownership or syncing
results. Previously its GPU/image/label checks alone did not reject a wrongly configured container; launcher reuse
checks could be bypassed by reconciliation. The launcher now also checks privilege/rootfs on reuse. Targeted
CPU boundary checks passed, including running/exited corrupted containers remaining unknown without redispatch.
Complete local native acceptance subsequently passed as `bomqe4f3o`: 625 passed, zero failed/errors/skips,
669.64 seconds, all 210 source fingerprints unchanged and independently checked. Evidence is under
`.worktrees/review/runs/acceptance/local-20261010-2034/`; source integration remains under HOLD. No remote native
acceptance is inferred. Settled-source local CUDA/EGL checks also passed on both approved UUIDs with actual frames
and matching source hashes; their supplied host-PID flag is an assertion, not independently inspected main Docker
PID-mode proof. Remote checks must inspect actual `PidMode=host` on the owned container before PID/device acceptance.
Review source commit `2850b30` integrated under main merge `b91b7c3` at 21:03. All 210 production fingerprints
match the accepted suite; its report/log/JUnit are published unchanged in `docs/tests.*` with original review
provenance retained. Detached local scheduler PID 2819571 and accepted tracked watcher PID 2819468 are live.
Production reserve parses as 128 GiB; remote remains disabled/CDI. Memory-heavy admission remains held while
long-window evidence is collected. Remote clean pushed-release deployment/native gates are still pending.

## Clean release and CDI registration (21:12)

- Accepted integration/evidence pushed as `200e5e63f6238180312c0007b4ecc1f33b669dd3` to `origin/splatter4d`.
  Secure checksum-verified bundle deployment completed at 21:10 beneath
  `/home/compu/kaist/sunho/releases/200e5e63f6238180312c0007b4ecc1f33b669dd3/code`; clean exact checkout and
  pushed ancestry enforced by the access helper. Bundle SHA-256
  `c4a4ef4e9fe71d80d9d95fac3db64ccc0b28ad1881901cfcbfad7b3a9b1705cd`. Main proof:
  `runs/remote/releases/200e5e63f6238180312c0007b4ecc1f33b669dd3/provenance.json`. Independent read-only follow-up
  checked exact HEAD/empty status and matched EGL helper, scheduler, durable-query helper, uv.lock, original local
  test report and release-provenance hashes against main files; `runs/remote/access/release-identity-200e5e6.log`.
- Fresh remote inventory at 21:11: Docker 28.5.1; existing immutable image ID unchanged; NVIDIA queries returned
  success. All four permitted UUIDs still idle (1 MiB each, no compute/graphics processes); foreign GPUs 4-7
  remain excluded. Free SSD 4,054,387,412,992 bytes, still above the 15% floor. MemAvailable 1,506,675,684 KiB.
- `nvidia-ctk cdi list` registers UUID-specific device names for all four authorized GPUs. This is registration
  evidence only, not successful Docker injection, CUDA/EGL or host-PID validation; none of those gates is waived.
  Evidence `runs/remote/access/inventory-cdi-20261010-2110.log`. No connectivity outage or remote host action.
- A narrow ignore rule excludes only the generated `experiments/local_gpu_query.json`, alongside the existing
  watcher cursor, so runtime query bookkeeping does not make immutable source releases appear dirty. No source,
  numerical configuration or accepted fingerprint changed. Operational registry/runs and credentials were not staged.

## Hard budgets and acceptance blockers (2026-10-10 21:47)

- Required campaign limits are local <=150 decimal GB (steady target 130 GB, plus existing 128 GiB host reserve)
  and remote <=1 decimal TB (provisional steady target 900 GB, plus host reserve and SSD floor). Full charged cache
  counts; no Docker working-set subtraction. Latest local full cgroup charge at 22:38 is 215.09 GB, PSS 184.43 GB;
  this violates the required ceiling. Evidence `runs/setup/memory-ceiling-20261010-2240.json` (observed 22:38); earlier snapshots
  remain unchanged. New heavy launches remain held.
- All 20 active item-1 configs already have six workers/prefetch 2. The user explicitly authorized standing lossless
  checkpoint pauses/resumes and the complete hammer group pause at 22:23, superseding the earlier approval blocker.
  Controller `bv3z5vw1s` is armed for all ten fresh next boundaries; no completed pause/migration is inferred from
  arming. Every required experiment and scientific setting remains. Fresh permission-layer blocks are reported with
  exact command/reason rather than bypassed or silently awaited.
- After gates, remote is the default for item 2, wide RL sweeps, Stage 4 baseline work and future pretraining batches.
  Migration requires resumed-checkpoint same-next-step loss within numerical noise in addition to step-0/short-run
  equivalence; every arm/both seeds of a task moves together with explicit registry step-host ranges/checkpoint hashes.
- Read-only audit found remote suite fixture defects: `tests/test_remote_access.py` wrote temp fixtures at the immutable
  repository root; DROID protocol path validation rejected authorized external pytest temp. Narrow fixes/boundary tests
  now exist only in review. They preserve runtime-root/symlink/traversal and all native acceptance boundaries.
- Initial acceptance remains circular: scheduler `RemoteBackend.ready`, `acceptance_manifest`, launcher
  `evidence_artifacts` and remote entry require already-passing suite/isolation/cross-host evidence. No existing
  compliant scheduler-only bootstrap was found. The ordinary campaign's read-only acceptance mount and job identity
  are also unsuitable for initial suite publication. **The user explicitly authorized bounded bootstrap at 22:23**;
  typed, fixed-command verification purposes are now being implemented only in review. This authorization permits
  producing real suite/isolation/equivalence evidence; it does not establish passing evidence or admit scientific
  work before all gates. Local native acceptance within budget and clean pushed deployment must precede bootstrap.
- New budget code is review-only, no full native suite is launched above local budget, and no source is merged/pushed
  as accepted without that suite. The previously pushed clean release remains the accepted integration, not a new
  budget-enforcement or remote-native acceptance claim.
- Completed remote-accounting diffs were independently reviewed; 136 pure boundary checks passed without real
  Docker/SSH, GPU/native fixtures or campaign-worker launches. Snapshot and fresh launcher admission measure owned
  cgroup `memory.current` without working-set/cache subtraction, and conservatively reserve all outside-container
  nonfree host RAM using MemTotal-MemFree, including foreign usage. Incomplete accounting holds admission while
  preserving observed container state. This may over-restrict placement and is not an all-times kernel cap.
- Both campaign acceptance validators now require true `checkpoint_resume_within_noise`, a valid checkpoint SHA-256
  and positive integer resume step, in addition to the existing suite/isolation/data/step-0/short-run requirements.
  Only synthetic boundary fixtures were changed/tested; no accepted evidence was edited and no real cross-host
  checkpoint loss was measured. Bootstrap authorization is now explicit; gate execution and acceptance remain pending.

## Verification ledger

| Gate | Status | Evidence |
|---|---|---|
| Container TCP route and SSH access | PASS | Initial connection and inventory above |
| Authorized root / Docker access / idle GPUs | PASS at 21:11 inventory only | All four permitted UUIDs idle; recheck at admission |
| Clean pushed immutable release | PASS | `5a7177e` clean exact checkout; `runs/remote/releases/5a7177eb41541feb3f193fb62e7459c5b80274e5/provenance.json` |
| UUID-specific CDI registration and execution | PASS for four isolation containers | Inspected exact one-UUID CDI injection; original reports and inspections under `runs/verify-isolation-<0..3>-200e5e6-20261010/` and `runs/remote/verification/` |
| Immutable image / exact native runtime bytes | PASS for build/version/byte identity only | Image ID and 1,948-file audit above; native execution still pending |
| Remote full native suite | PASS (5a7177e) | 728/0/0/0, `runs/verify-native-5a7177e-20261011/evidence/`; earlier 200e5e6 failure 589/36 kept in RESULTS |
| Per-GPU CUDA/EGL isolation | PASS on all four UUIDs (5a7177e) | `runs/verify-isolation-<0..3>-5a7177e-20261011/evidence/gpu_isolation.json` |
| Scheduled data identity | PASS for initial transferred scope | Eight files verified at 19:33; semantic identity `56bd8ce9c59e81077061f577888f708541c2c14d605248f0cbdf81521e54bae1` |
| Cross-host v2 fresh step-0 and 200-update equivalence | ACCEPTED by user decision | Step 0 bit-identical; RMSE 4.687e-5 vs same-host 4.588e-5 (criterion not met); `runs/remote/acceptance/5a7177e/cross_host.json` |
| Checkpoint continuation for actual migrations only | PENDING / no migration | Original local checkpoint, next 50 updates, same measured-noise criterion; complete hammer group moves together |
| Remote long-job admission | OPEN | Manifest `runs/remote/acceptance/5a7177e/manifest.json`; job-by-job under 1 TB; W&B offline |

## Outages

No remote connectivity outage observed during initial setup. Record each detected outage's start, end, duration and affected unknown-state jobs here; never infer a remote job failure from loss of connectivity.
