# splatter4d Meta-World campaign — progress

Last updated: 2026-10-11 15:10 +09:00

## Current phase
- **Done:** Phase A (E0), Phase B (8 tasks, D2/D3 pass), Phase C (timing, `docs/COMPUTE_PLAN.md`; M2 overfit gate failed
  after three diagnosed iterations, recorded), Stage 0 (DrM + CNN hammer 0.70 / shelf-place 0), Stage 4 prep (SinCro,
  ReViWo ported), mid-campaign review items 1-5 (EXPERIMENT_LOG; coffee-push replaces shelf-place; synthetic near
  views adopted into the method configuration C).
- **Stage 1 (in progress), user directives of 2026-10-09** (all rules pre-registered in EXPERIMENT_LOG before results):
  - **S1 (done, counted iteration 4):** neither V1 (synthetic invariance only) nor V2 (C + self-render) qualifies;
    C remains synthetic views as invariance positives and render targets. Two seeds per configuration/task,
    full-split evaluation at 100k of the 200k schedule; fixed margins and output in
    `docs/decisions/s1_rule_2026-10-10.txt`. Seed-sensitive motion collapse recorded in `docs/RESULTS.md`.
  - **Item 1 (depth redesign):** code merged and tested (occlusion/free-space loss, far-plane depth validity, usage
    diagnostics, visible-only CD-centers, GPU time). All 20 D0-D4 x 2 seeds x 2 tasks runs remain required on C;
    all ten hammer runs are independently proven checkpoint-paused (nine at 40k, D2 seed 0 at 50k);
    all ten pick-place trainers completed normally at100k; final D1seed1 exit08:27:32, original checkpoint/config/
    whole-session teardown verified, no cancellation or schedule change. All ten pick-place standalone full-split
    results independently verified; final D1seed1 exit08:34:40, original full coverage/metrics/source/config/hashes/
    whole-session teardown proved08:36:52. Immutable ten-result inventory
    `runs/setup/item1-pick-place-fullsplit-inventory-20261011-083652.json`; previous nine summary hashes unchanged.
    Ten of20 results complete, none running, ten hammer evaluations await held original trainers; globalHOLD stays.
    Pick-place D0-only margins frozen08:16:43 in
    `runs/setup/item1-d0-pick-place-margins-20261011-081643.json` and EXPERIMENT_LOG before any variant comparison;
    hammer-specific margins await its reference pair. No item1 decision until all20 results.
  - **Item 2 (M3D):** code merged and tested (Gaussian-space motion loss, binned relative EPE, EPE in mm).
    Waits for the item-1 winner (2 seeds x 2 tasks) + 6-seed hammer RL proxies per variant.
  - **Item 3 (prefetch, done):** default 2 x 6 workers; measured peak PSS 9.61 GB versus 11.29 GB for 4 x 8.
    Historical pretraining declarations are 13 GiB; existing 4 x 8 jobs were not restarted (15% saving < 40% rule).
    Four-hour sampled per-run PSS peaks are 9.15–10.77 GB, not lifetime or independently isolated evaluation peaks.
  - **S2-S6:** 6-seed hammer RL proxies for RL-based decisions; base300k/synth300k stopped (context only; length study
    redone for the final configuration); S4 CNN pick-place seed 2000 completed at 1M (final train-camera success
    0.675; OOM-resume/mixed-curve caveat recorded); S5 workspace-statistics anchors kept; S6 gate screens retired.
  - `method-frozen-v1` only after items 1-2 (then item 1e leave-one-out and the A200/A300 length study).
- **Pretraining lengths:** ours default 300k, final L chosen on hammer/pick-place before the freeze (same L for all
  8 tasks and ablations); SinCro exactly 300k; ReViWo 100 001 (reference).

## Hard RAM budget update (2026-10-11 15:10)
- **Latest full local charge: 23.74 decimal GB container memory.current**, observed at15:09:35; host
  MemAvailable387.71GB, above the60GiB reserve. Charge-only all-active-job operational-health evidence
  `runs/setup/campaign-health-20261011-150935.json`; no completePSS/new admission inferred.
  All cache/kernel/background charge remains included. The whole held hammer group's unchanged13GiB/run
  reservations total139,586,437,120bytes; plus charge23,741,833,216bytes equals163,328,270,336bytes>150GB,
  so full-group local resume remains held. Host headroom does not remove the local cap; charge alone blocks
  admission. No partial-group resume/cache exclusion.
  Earlier14:09:41 charge23.70GB/hostavailable373.80GB remains unchanged in
  `runs/setup/campaign-health-20261011-140941.json`; charge-only, not admission.
  Earlier13:08:45 charge23.70GB/hostavailable301.21GB remains unchanged in
  `runs/setup/campaign-health-20261011-130845.json`; charge-only, not admission.
  Earlier12:08:48 charge23.73GB/hostavailable297.61GB remains unchanged in
  `runs/setup/campaign-health-20261011-120848.json`; charge-only, not admission.
  Earlier11:08:50 charge23.74GB/hostavailable283.78GB remains unchanged in
  `runs/setup/campaign-health-20261011-110850.json`; charge-only, not admission.
  Earlier10:09:43 charge23.73GB/hostavailable315.37GB remains unchanged in
  `runs/setup/campaign-health-20261011-100943.json`; charge-only, not admission.
  Earlier09:09:04 charge23.72GB/hostavailable296.58GB remains unchanged in
  `runs/setup/campaign-health-20261011-090904.json`; charge-only, not admission.
  Earlier08:36:52 postcompletion charge23.72GB/hostavailable292.73GB remains unchanged in
  `runs/setup/campaign-health-20261011-083652.json`; charge-only, not admission.
  Earlier08:31:08 poststart charge30.44GB/hostavailable282.73GB remains unchanged in
  `runs/setup/item1-fulleval-d1-seed1-start-20261011-083108.json`; not second D1 admission.
  Earlier08:16:39 charge32.33GB/hostavailable344.85GB remains unchanged in
  `runs/setup/campaign-health-20261011-081639.json`; charge-only, not admission.
  Earlier08:10:56 poststart charge46.35GB/hostavailable327.56GB remains unchanged in
  `runs/setup/item1-fulleval-d3-both-seeds-start-20261011-081056.json`; that was not a second D3 admission.
  Earlier07:30:45 charge81.88GB/hostavailable168.95GB remains unchanged in
  `runs/setup/campaign-health-20261011-073045.json`;07:26:52 charge82.98GB/hostavailable174.54GB remains in
  `runs/setup/item1-fulleval-d1-seed0-start-20261011-072652.json`.
- **Earlier08:30 D1seed1 admission (result now complete):** complete51 end-live PID/start-tick/PSS coverage, zero missing identities/
  substantive errors; three explicit transient disappearances. Accounted23.74GB plus12GiB reservation12.88GB
  equals36.62GB<150GB; hostavailable291.63GB exceeds reservation plus60GiB/ramp0. Real GPU ownership/limits/disk/
  unchanged accepted200e5e6 complete210-source/original625-pass report gates passed. Original existing job only,
  unmodified scoped scheduler/sole main registry/no resource overrides/globalHOLD retained; non-atomic snapshot,
  no lifetime/all-times guarantee. `runs/setup/item1-fulleval-d1-seed1-admission-20261011-083012.json`.
- **Earlier08:09 D3 pair admission:** complete end-live PID/PSS coverage, zero missing live identities/errors;
  accounted32.70GB plus both12GiB reservations25.77GB equals58.47GB<150GB. Hostavailable301.77GB exceeds
  both reservations plus60GiB; actual GPU ownership/limits/disk/accepted210-source625-pass gates passed.
  `runs/setup/item1-fulleval-d3-both-seeds-admission-20261011-080956.json`; non-atomic snapshot, no lifetime/
  all-times guarantee. Only the two existing original evaluations dispatched; no resource override/globalHOLD retained.
- **Earlier07:45 D4seed1 admission:** complete end-live PID/PSS coverage, zero missing identities/errors and two
  explicit transient disappearances; accounted73.52GB plus12GiB equals86.40GB<150GB, hostavailable185.98GB.
  `runs/setup/item1-fulleval-d4-seed1-admission-20261011-074501.json`; result later completed/verified.
  Actual07:45:45 poststart charge80.38GB/hostavailable183.72GB, not second admission, remains in
  `runs/setup/item1-fulleval-d4-seed1-start-20261011-074545.json`.
- **Earlier07:26 D1seed0 admission:** complete end-live PID/PSS coverage, zero missing live identities/errors;
  two short-lived PID disappearances explicitly recorded. Accounted75.88GB plus12GiB equals88.76GB<150GB;
  hostavailable184.46GB exceeds reservation plus60GiB. Actual GPU ownership/limits, disk and accepted210-source
  gates passed. `runs/setup/item1-fulleval-d1-seed0-admission-20261011-072611.json`; non-atomic snapshot,
  not lifetime peak/all-times guarantee. GlobalHOLD stays, only the existing original evaluation dispatched.
- **Earlier07:22 combined admission:** complete end-live PID/PSS coverage, zero missing live identities/errors;
  three short-lived PID disappearances explicitly retained. Accounted full charge68.64GB plus all three unchanged
  12GiB reservations (38.65GB combined) equals107.29GB<150GB. Hostavailable225.10GB exceeds all reservations
  plus60GiB; actual GPU ownership/limits and disk floor passed. Evidence
  `runs/setup/item1-fulleval-seed0-batch-admission-20261011-072226.json`, non-atomic snapshot, not an all-times
  or lifetime guarantee. Accepted200e5e6/210-source map still equals original625/625 suite; only the three existing
  jobs were dispatched. GlobalHOLD remains against legacy paused replay, with no source changes/remote admission.
- **Earlier07:09 D0seed1 admission:** complete end-live coverage, accounted97.67GB plus12GiB equals110.56GB;
  original `runs/setup/item1-fulleval-d0-seed1-admission-20261011-070937.json`. That evaluation is now completed.
- **Earlier06:57 D2seed1 admission:** all141 end-live identities covered; PSS89.26GB/fullcharge105.42GB
  plus12GiB equals118.30GB. Original `runs/setup/item1-fulleval-admission-20261011-065706.json` and06:58
  poststart charge112.11GB in `runs/setup/item1-fulleval-start-20261011-0657.json` stay unchanged; result completed.
  Earlier06:17:33charge115.32GB/hostavailable213.49GB remains in `runs/setup/campaign-health-followup-20261011-061733.json`.
  Earlier06:08:03charge114.80GB/hostavailable203.74GB remains in `runs/setup/campaign-health-20261011-060803.json`.
  Earlier05:07:58charge114.62GB/hostavailable161.78GB remains in `runs/setup/campaign-health-20261011-050758.json`.
  Earlier04:08charge109.00GB/hostavailable29.39GB prevented new local admission at that time;
  original `runs/setup/campaign-health-20261011-040801.json` remains unchanged.
  Earlier03:08charge124.18GB/hostavailable149.88GB remains in `runs/setup/campaign-health-20261011-030805.json`.
  Earlier02:12charge130.29GB/hostavailable181.70GB remains in `runs/setup/campaign-health-20261011-0214.json`,
  whose actual sample time is recorded inside. Earlier01:21charge128.22GB/hostavailable177.20GB remains
  in `runs/setup/campaign-health-20261011-0111.json`. At 00:52:36 charge was 131.41 GB with the native suite running.
  Fresh prelaunch maximum charge 128.15 GB plus its 15 GB reservation was 143.15 GB <150 GB;
  complete live PID coverage/conservative PSS sum 95.70 GB, host available 176.72 GB. Original admission evidence
  `runs/setup/portability-native-memory-20261011-0049.json`; no all-times or lifetime-peak claim.
  At 00:30 charge was 150.77 GB (`runs/setup/campaign-health-20261011-0030.json`), following a 148.88 GB observation.
  Previous charge-only evidence `runs/setup/memory-charge-20261011-0002.json` remains unchanged at 178.15 GB.
  Last PSS subtotal at 23:49 was 153.31 GB (owned 149.52 GB), explicitly incomplete because two transient processes
  exited during the 209-process scan; `runs/setup/memory-ceiling-20261010-post-pause4.json`. No new PSS is claimed.
  No cache is excluded and mapped shmem/file PSS is not added again. Earlier complete snapshot at 22:38 remains
  215.09 GB charge / 184.43 GB PSS in `runs/setup/memory-ceiling-20261010-2240.json`, unchanged.
- Required local hard cap **150 GB**, steady target **130 GB**; the22:44 user directive returns the additional host-available reserve to **60 GiB**.
  Required remote cap **1 decimal TB**, provisional steady target 900 GB, with existing SSD floor and host reserve.
  All future memory-heavy/many-process work defaults to remote after acceptance. Full policy in COMPUTE_PLAN.
- All **20 required item-1 configs already use num_workers=6 / prefetch_factor=2**; no redundant 4x8-to-2x6 restart
  can produce further saving. Every required 100k experiment remains on its original 200k schedule.
- **Standing checkpoint-pause/resume authorization received at 22:23.** All ten hammer runs are independently
  proven lossless: nine at 40k and D2 seed 0 at 50k. Exact checkpoint hashes,
  CRC, all 718 normalized latest/numbered payload members, last logged step and complete session teardown were checked.
  Original controller `bv3z5vw1s` ended after its first pause exceeded the old worker-teardown wait; no extra signal.
  Rearmed controller `bjhmh9q29` completed exit 0 after the final fresh 40k pause; its log `runs/setup/checkpoint-pause-hammer-20261010-2325.log`. Original 100k stop/200k schedules stay.
  Nothing is cancelled. HOLD prevents legacy replay relaunch until proven checkpoints, actual budget and validated
  full-group resume/migration are ready. Earlier permission denials are historical, superseded by explicit approval.
- Admission/heartbeat/intentional-pause accounting and two immutable-remote fixture fixes are being developed only
  in `.worktrees/review`. Six pure-CPU accounting checks and thirteen admission/pause-replay checks passed, including
  missing live PSS/shared PSS, PID reuse, late exit, conflicting pause proof and double-exemption rejection. These are
  not native acceptance. Completed remote-accounting/resume-gate diffs were independently reviewed; 136 additional
  pure remote boundary checks passed without real Docker/SSH, native fixtures or campaign launches. Outside-container
  accounting intentionally reserves all nonfree remote host RAM, including foreign usage. No full native suite is
  launched above budget; no review source is merged or deployed.
- Remote migration requires real step-0/short-run equivalence **plus resumed-checkpoint same-next-step loss within
  numerical noise**, complete comparison-group movement, and explicit registry host step ranges/checkpoint hashes.
  Remote scientific admission remains disabled. **Bounded bootstrap explicitly authorized at 22:23**: real remote
  suite, four one-UUID CUDA/EGL checks, genuine cross-host and checkpoint-resume checks, then admitted remote jobs
  only after gates pass. **Superseding user decision22:44:** start the remote suite/isolation from deployed accepted
  `200e5e6` without waiting for local RAM or new review code; unfinished typed bootstrap is frozen off the critical path.
  Remote-only work requires the real suite/isolation, equal first-batch/seed and step0forward1e-4relative, followed by
  C/D0hammerseed0for200steps with measured remote repeat noise (unsmoothed total-loss RMSE). Checkpoint comparison
  is only for actually migrated runs: next50steps under the same noise criterion. The local paired check is bounded
  <=15GBand may run with measured headroom below150GB; normal review integration still requires its full suite.
  After those nonmigration gates, independentStage4CNNDrM8tasks x seeds2000-2002and transferred-task SinCro/ReViWo
  pretraining start remotely. The first verification actually started at 23:00:32; the previous scientific ETA
  is withdrawn pending genuine full-suite acceptance and numerical gates.
  Fresh permission-layer blocks must be reported with the exact command and reason.
- **Actual remote native suite start23:00:32+09:00**: registered `verify-native-200e5e6-20261010`, container
  `s4d-verify-native-200e5e6-20261010-a1`, imagec3ff..., UUIDGPU-4391... throughCDI. Accepted `scripts/jobs.py` owns
  dispatch/reconciliation using a fixed-purpose operational verification adapter, not the unfinishedreviewroute.
  Real Docker inspection independently verifiedhostPID/read-onlysource/exactimage/mounts/8GiBshm/64GiBRAMbound.
  The suite finished at **23:05:54: 589 passed / 36 failed / 0 errors / 0 skipped**, 317.77 seconds, unchanged source,
  exit 1, no OOM. Original report/log/JUnit were checksum-returned unchanged to that run's `evidence/`; hashes and
  exact failure categories are in RESULTS. The wrapper's host/container run-path mismatch prevented its initial
  evidence copy/exit-file write; fixed for subsequent verification without altering the failed attempt or reports.
  Native worker CUDA passed; its inherited OOM score 500 prevented nonroot lowering to requested 300. Suite-only
  initial score 0 corrects that environment without extra capabilities or changes to production policy/tests.
  Minimal fixture/export/evaluator/runtime-path repairs remain review-only, not accepted or deployed. A frozen seven-file
  portability candidate is at `.worktrees/review/runs/acceptance/portability-200e5e6-20261010/code`; all210sourcehashes
  differ from200e5e6onlyat the registered seven files, preserving all unrelated unfinished review work. Its full native
  suite actually launched at 00:50:01 as `verify-native-portability-200e5e6-local-20261010`, PID3031654, GPU4UUID.
  Local launch preparation adds independent Git metadata and a separately hashed operational top-level GPU-guarded
  entrypoint; accepted guard check passes unchanged. One-job native queue reserves exactly15GB and60GiB host headroom,
  retains global HOLD, and has no auto-retry. Mechanical waiter `bwxuxx2fm` completed; actual GPU ownership, complete
  live-PID PSS coverage, unchanged candidate and accepted scheduler admission passed before launch. No native acceptance
  is inferred from dispatch; original full report, unfiltered counts and both source maps must verify on completion.
  Independent four-UUID CUDA/EGL dispatch completed as `bdm2zc5ry`; it does not need a full-suite pass to run, but scientific
  admission still requires both gates and v2 numerics. Original 00:44–01:44 scientific ETA is no longer reliable;
  a passing full suite and local paired-check headroom must first be established.
- **Prospective user seed policy23:00:** item1anditem2keep both pretrainingseeds; item2reports per-seedcollapse
  and dynamicshareat10kand100k. Unregistered downstreamleave-one-out/length/screens default toonepretrainingseed0
  with margins fixedfrom existingreferencepairspread+floors. Ifadoptedconfigurationstillcollapsesafteritem2, report
  numbersbeforelaunchand letuserdecide seedcount again. AllsixRLproxyseedsandfinalseeds2000-2002remainunchanged.
- **Historical 23:04–00:05 checks:** localreserve60GiBis appliedto operational mainqueueandreviewqueue underHOLD; hardcapadmissioncoderemainsreview-only.
  At23:04localcharge213.91GBandzeroverifiedpauses; no newheavy/localnativework. Localwatcher queryfailedfive times
  withQueryUnavailable; durablequeryownershipretained, no foreignsignals/driver/cgroup/VPNaction. Watcher task
  `ba83fxlvw` completed on the remote crash event; its successor `b56bdd1i8` was rearmed only after completion.
  Accepted scheduler remains alive. **First proven pause:** `i1-d0-seed1-hammer` at fresh40k, original100kstop/200kschedule
  unchanged. SIGTERM publication latency26.6ms; original controller stopped on60secondworker-teardownwait. The entire
  session later exited without another signal; all718 normalized checkpoint archive members equal latest, CRCvalid,
  exactSHA256/proof in `runs/i1-d0-seed1-hammer/checkpoint_pause.json` and registry. Charge atproof208.75GB, stillabovecap.
  Remainingnine are rearmed as `bjhmh9q29`, PID2961453, with a900second teardownwait, preserving all identity/boundary
  checks and no extra signals; log `runs/setup/checkpoint-pause-hammer-20261010-2325.log`.
  Four remote isolation checks now passed actualCUDA+EGL; original reports synced/verified by accepted daemon. No science
  admission or v2numerics yet. Five hammer pauses are now independently proven: D0 seeds0/1, D1 seed1, D2 seed1,
  D4 seed1, all40k. At 00:05 all fifteen remaining trainers have advanced, are alive in their owned sessions,
  and have log activity under29seconds with no inspected traceback. Watcher `b1r8jmeq0`, PID2991284, is alive
  with a9.9second beacon; scheduler/controller/observer are live, disk4.8TiBfree. No duplicate monitor or sampler.
  This watcher replaced completed `bmt455pyv` once after the fifth planned exit. Legacy crash labels are backed
  by checkpoint-pause proofs; no fabricated restart exemption. Local query unavailability remains reported, not repaired
  by host/driver/VPN actions.

### Latest requested campaign health check (15:09–15:10)
- `runs/setup/campaign-health-20261011-150935.json` covers every active registry job: only original stalled native
  verification; no scientific job active locally/remotely or eligible original job. Ten pick-place trainers/results
  remain complete (10/20 required); ten original held hammer sessions independently gone/checkpoint-pause proofs present.
- Exactly one existing tracked watcher `bw36yh48g`/PID3455689/start419667412, exact absolute main command and
  13.54second beacon verified15:09:35; daemon2819571 retains original identity. No rearm/duplicate monitor/new waiter.
- Full charge23,741,833,216bytes/hostavailable387,705,036,800bytes/diskfree5,191,325,392,896bytes. Whole-group
  unchanged reservations plus charge163,328,270,336bytes>150GB; charge alone blocks admission despite host headroom.
  Charge-only/non-atomic health, not completePSS/new admission/lifetime proof; no cache exclusion/partial resume.
- Native3031654/3031655 retain original identities, log839.04minutes old, finalJSON/JUnit/exit absent; original
  confirmed AF_UNIX stall stderr SHA unchanged. Exact-parent `signal.pidfd_send_signal(fd, signal.SIGINT)` remains
  denied absent explicit interruption approval, not generic campaign requests/automated events. CPU ticking is not
  progress/acceptance; no signal/alternate route/retry/duplicate suite/replacement native waiter.
- Real GPU query: native parent onGPU4/812MiB, GPU5 empty. Main200e5e6 complete210-source maps match both original
  625-pass maps/report hash, not new suite/candidate acceptance. HOLD/checkpoints/inventory/frozen margins/seeds/
  schedules/thresholds/all20-result requirement retained; no comparison/decision/repeated evaluation/item2/proxy
  launch/source integration/commit/push/deployment/remote science/v2numerics/hammer resume/migration/cancellation/
  RL finish/replay deletion. No newly admissible work; existing monitoring remains armed.

### Historical requested campaign health check (14:09–14:10)
- `runs/setup/campaign-health-20261011-140941.json` covers every active registry job: only original stalled native
  verification; no scientific job active locally/remotely or eligible original job. Ten pick-place trainers/results
  remain complete (10/20 required); ten original held hammer sessions independently gone/checkpoint-pause proofs present.
- Exactly one existing tracked watcher `bw36yh48g`/PID3455689/start419667412, exact absolute main command and
  7.09second beacon verified14:09:41; daemon2819571 retains original identity. No rearm/duplicate monitor/new waiter.
- Full charge23,696,949,248bytes/hostavailable373,796,024,320bytes/diskfree5,193,824,825,344bytes. Whole-group
  unchanged reservations plus charge163,283,386,368bytes>150GB: improved host availability does not remove the
  local cap, charge alone blocks admission. Charge-only/non-atomic health, not completePSS/new admission/lifetime
  guarantee; no cache exclusion/partial-group resume.
- Native3031654/3031655 retain original identities, log779.13minutes old, finalJSON/JUnit/exit absent; original
  confirmed AF_UNIX stall stderr SHA unchanged. Denied exact-parent `signal.pidfd_send_signal(fd, signal.SIGINT)`
  requires explicit interruption approval, not generic campaign requests/automated events. CPU ticking is not
  progress/acceptance; no signal/alternate route/retry/duplicate suite/replacement native waiter.
- Real GPU query: native parent onGPU4/812MiB, GPU5 empty. Main200e5e6 complete210-source maps match both original
  625-pass maps/report hash, not new suite/candidate acceptance. HOLD/checkpoints/inventory/frozen margins/seeds/
  schedules/thresholds/all20-result requirement retained; no comparison/decision/repeated evaluation/item2/proxy
  launch/source integration/commit/push/deployment/remote science/v2numerics/hammer resume/migration/cancellation/
  RL finish/replay deletion. No newly admissible work; existing monitoring remains armed.

### Historical requested campaign health check (13:08–13:09)
- `runs/setup/campaign-health-20261011-130845.json` covers every active registry job: only original stalled native
  verification; no scientific job active locally/remotely or eligible original job. Ten pick-place trainers/results
  remain complete (10/20 required); ten original held hammer sessions independently gone/checkpoint-pause proofs present.
- Exactly one existing tracked watcher `bw36yh48g`/PID3455689/start419667412, exact absolute main command and
  18.11second beacon verified13:08:45; daemon2819571 retains original identity. No rearm/duplicate monitor/new waiter.
- Full charge23,704,461,312bytes/hostavailable301,206,192,128bytes/diskfree5,194,716,532,736bytes. Whole-group
  unchanged reservations plus charge163,290,898,432bytes>150GB; charge alone blocks admission, no partial resume/
  cache exclusion. Charge-only/non-atomic health, not completePSS/new admission/lifetime guarantee.
- Native3031654/3031655 retain original identities, log718.20minutes old, finalJSON/JUnit/exit absent; original
  confirmed AF_UNIX stall stderr SHA unchanged. Exact-parent `signal.pidfd_send_signal(fd, signal.SIGINT)` remains
  denied absent explicit interruption approval, not generic campaign requests/automated events. CPU ticking is not
  progress/acceptance; no signal/alternate route/retry/duplicate suite/replacement native waiter.
- Genuine GPU query: native parent onGPU4/812MiB, GPU5 empty. Main200e5e6 complete210-source maps match both
  original625-pass maps/report hash, not new suite/candidate acceptance. HOLD/checkpoints/inventory/frozen margins/
  seeds/schedules/thresholds/all20-result requirement retained; no comparison/decision/repeated evaluation/item2/
  proxy launch/partial-group resume/source integration/commit/push/deployment/remote science/v2numerics/hammer resume/
  migration/cancellation/RL finish/replay deletion. No newly admissible work; existing monitoring remains armed.

### Historical requested campaign health check (12:08–12:09)
- `runs/setup/campaign-health-20261011-120848.json` covers every active registry job: only original stalled native
  verification; no scientific job active locally/remotely or eligible original job. Ten pick-place trainers/results
  remain complete (10/20 required); ten original held hammer sessions independently gone/checkpoint-pause proofs present.
- Exactly one existing tracked watcher `bw36yh48g`/PID3455689/start419667412, exact absolute main command and
  7.21second beacon verified12:08:48; daemon2819571 retains original identity. No rearm/duplicate monitor/new waiter.
- Full charge23,725,817,856bytes/hostavailable297,610,649,600bytes/diskfree5,204,740,108,288bytes. Whole-group
  unchanged reservations plus charge163,312,254,976bytes>150GB; charge alone blocks admission, no partial resume/
  cache exclusion. Charge-only/non-atomic observation, not completePSS/new admission/lifetime guarantee.
- Native3031654/3031655 retain original identities, log658.25minutes old, finalJSON/JUnit/exit absent; original
  confirmed AF_UNIX stall stderr SHA unchanged. Denied exact-parent `signal.pidfd_send_signal(fd, signal.SIGINT)`
  still requires explicit interruption approval, not generic campaign requests/automated events. CPU ticking is not
  suite progress/acceptance; no signal/alternate route/retry/duplicate suite/replacement native waiter.
- Real GPU query: native parent onGPU4/812MiB, GPU5 empty. Main200e5e6 complete210-source maps equal both original
  625-pass maps/report hash, not new suite/candidate acceptance. HOLD/checkpoints/inventory/frozen margins/seeds/
  schedules/thresholds/all20-result requirement retained; no variant comparison/decision/repeated evaluation/item2/
  proxy launch/source integration/commit/push/deployment/remote science/v2numerics/hammer resume/migration/cancellation/
  RL finish/replay deletion. No newly admissible work; existing monitoring remains armed.

### Historical requested campaign health check (11:08–11:09)
- `runs/setup/campaign-health-20261011-110850.json` covers every active registry job: only original stalled native
  verification; no scientific job active locally/remotely or eligible original job. Ten pick-place trainers/full-split
  results complete (10/20 required); ten original held hammer sessions independently gone/checkpoint-pause proofs present.
- Exactly one existing tracked watcher `bw36yh48g`/PID3455689/start419667412, exact absolute main command and
  14.61second beacon verified11:08:50; scheduler2819571 retains its original identity. No rearm/duplicate monitor.
- Full charge23,738,851,328bytes/hostavailable283,784,126,464bytes/diskfree5,199,193,767,936bytes. Whole-group
  unchanged reservations plus charge163,325,288,448bytes>150GB; charge alone blocks admission, no partial resume/
  cache exclusion. Charge-only/non-atomic observation, not completePSS/new admission/lifetime guarantee.
- Native3031654/3031655 retain original identities, log598.28minutes old, finalJSON/JUnit/exit absent; original
  confirmed AF_UNIX stall stderr SHA unchanged. Exact-parent `signal.pidfd_send_signal(fd, signal.SIGINT)` remains
  denied absent explicit interruption approval, not generic campaign requests/automated events. CPU ticking is not
  progress/acceptance; no signal/alternate route/retry/duplicate suite/replacement native waiter.
- Genuine GPU query: native parent onGPU4/812MiB, GPU5 empty. Main200e5e6 complete210-source maps match both
  original625-pass maps/report hash, not new suite/candidate acceptance. HOLD/original checkpoints/inventory/frozen
  margins/seeds/schedules/thresholds/all20-result requirement retained; no variant comparison/decision/repeated
  evaluation/item2/proxy launch/source integration/commit/push/deployment/remote science/v2numerics/hammer resume/
  migration/cancellation/RL finish/replay deletion. Existing monitoring remains armed, no newly admissible work.

### Historical requested campaign health check (10:08–10:10)
- `runs/setup/campaign-health-20261011-100943.json` covers all active registry jobs: only original stalled native
  verification; no scientific job active locally/remotely or eligible original job. Ten pick-place trainers/results
  remain complete (10/20 required); all ten original held hammer sessions independently gone, pause proofs present.
- Exactly one existing tracked watcher `bw36yh48g`/PID3455689/start419667412, exact absolute main command and
  11.11second beacon verified10:09:43; daemon2819571/session2819571/start415518078 alive. No rearm/duplicate monitor.
- Full cgroup charge23,733,637,120bytes/hostavailable315,371,159,552bytes/diskfree5,197,498,392,576bytes. Whole
  unchanged hammer reservations plus charge163,320,074,240bytes>150GB: charge alone blocks full-group admission,
  no partial resume/cache exclusion. Charge-only/non-atomic health, not completePSS/new admission/lifetime guarantee.
- Native3031654/3031655 retain original identities, log539.16minutes old, finalJSON/JUnit/exit absent; original
  confirmed AF_UNIX stall stderr SHA unchanged. Denied exact-parent `signal.pidfd_send_signal(fd, signal.SIGINT)`
  still requires explicit interruption approval, not generic campaign requests/automated events. No signal/alternate
  route/retry/duplicate suite/native waiter; CPU ticking is not suite progress/acceptance.
- Real GPU query: original native parent onGPU4/812MiB, GPU5 empty. Main200e5e6 complete210-source maps equal both
  original625-pass maps/report hash; not new acceptance. GlobalHOLD/original checkpoints/frozen margins/seeds/
  schedules/thresholds/all20-result requirement retained; no variant decision/repeated evaluation/item2/proxy launch/
  source integration/commit/push/deployment/remote science/v2numerics/hammer resume/migration/cancellation/RL finish/
  replay deletion. No newly admissible work; monitoring remains armed.

### Historical requested campaign health check (09:08–09:17)
- `runs/setup/campaign-health-20261011-090904.json` covers every active registry job: only original stalled native
  verification; no scientific job active locally/remotely, no newly eligible original job. Ten pick-place trainers/
  full-split results remain complete (10/20 required); all ten held hammer original sessions independently gone.
- Exactly one live watcher `bw36yh48g`/PID3455689/start419667412, exact absolute main Python command and17.78second
  beacon verified09:09:04; daemon2819571 retains original session/start415518078. Preliminary substring matches
  included wrappers/diagnostic text, not duplicate Python watchers; preserved check uses exact command equality.
  Existing tracked watcher stays armed; no rearm/termination/duplicate monitor or new waiter.
- Charge23,724,085,248bytes/hostavailable296,578,805,760bytes/diskfree5,205,032,054,784bytes; full unchanged hammer
  reservations plus charge163,310,522,368bytes>150GB. Charge-only/non-atomic operational health, not completePSS/
  new admission/lifetime guarantee; full charge alone prevents whole-group admission, no partial resume/cache exclusion.
- Native3031654/3031655 retain exact original identities, log478.52minutes old, original finalJSON/JUnit/exit absent.
  Confirmed AF_UNIX stall and exact-parent `signal.pidfd_send_signal(fd, signal.SIGINT)` denial remain binding;
  generic campaign-check requests/automated events are not explicit interruption approval. CPU ticking is not suite
  progress/acceptance; no signal/alternate route/retry/duplicate suite/replacement native waiter.
- Genuine GPU query sees original native parent onGPU4/GPU5 empty; accepted200e5e6 complete210-source maps and
  original625-pass report unchanged, not newly executed acceptance. GlobalHOLD, original inventory/frozen margins/
  seeds/schedules/thresholds/all20-result requirement retained. No source change/integration/commit/push/deployment/
  remote science/v2numerics/hammer resume/migration/cancellation/RL finish/replay deletion or variant decision.

### Historical full-split completion, ten-result inventory and blocked-group health (08:34–08:38)
- Final original D1seed1 pick-place evaluation completed08:34:40.882420/exit0/attempt1. Independently verified
  step100000/full windows1996/1916/batches125/120, all original console-file metrics equal, genuine GPU guard/
  full-window start, original accepted complete launch-source map/625-pass evidence/config/hashes and entire
  session3450316 gone. `runs/setup/item1-fulleval-d1-seed1-completion-20261011-083652.json`; original summary SHA
  `925743dd8802ee6fa8f8e079ee9f86514a4fdf9c5078ed2b2a068a752e9d588c`. Raw observations in EXPERIMENT_LOG.
- Reverified all ten pick-place originals with full coverage/metrics/provenance/config/hashes/session teardown;
  prior nine summary hashes unchanged. Inventory `runs/setup/item1-pick-place-fullsplit-inventory-20261011-083652.json`.
  All ten pick-place trainers and ten standalone results complete;10/20 required results verified/none running,
  ten hammer evaluations await held original trainers. Fixed reference-only pick-place D0 margins/input hashes
  unchanged; hammer-specific margins unavailable, no substitution/variant means/deltas/gates/winner/item2/proxies.
- Fresh `runs/setup/campaign-health-20261011-083652.json` covers every active registry job: only original blocked
  native stall; no scientific job active locally/remotely. All ten original held hammer sessions remain gone.
  Completed watcher `bu332l3jl` output read and genuine evaluation completion handled; rearmed once as `bw36yh48g`/
  PID3455689/start419667412, sole exact-command watcher/beacon18.14seconds verified08:36:52. Daemon2819571 unchanged.
- Charge23.72GB/hostavailable292.73GB/disk5.210TBfree is charge-only, not completePSS/new admission. Full hammer
  reservations plus charge163.31GB>150GB, already blocked by charge alone: no partial-group or full-group resume.
  Genuine GPU query shows original native parent onGPU4/GPU5 empty. Native3031654/3031655 exact identities,
  log446.31minutes old, original finalJSON/JUnit/exit absent; confirmed AF_UNIX stall and exact-parent SIGINT denial
  remain binding. No signal/alternate route/retry/duplicate native suite/replacement native waiter/source change/
  integration/commit/push/deployment/remote science/v2numerics/hammer resume/migration/cancellation/shortened schedule/
  reseeding/RL completion/replay deletion. GlobalHOLD and all20-result decision requirement retained.

### Historical trainer completion, admitted evaluation and full operational health (08:27–08:32)
- Final pick-place trainer D1seed1 completed08:27:32.631833/exit0 at100000 on original200000 schedule/seed1/
  sixworkers/prefetch2/BF16/batch16/config. Entire original session2423619 gone, both checkpoint CRCs valid,
  all718 normalized members equal and exact100000 step verified nonexecutingly. Numbered checkpoint hash
  `3dae0436e09dd886b653e033f1d095ccab95ecbec5b062b342b01c8fcad1fec7`; exact latest/config/console hashes and
  original legacy launch snapshot absence preserved in admission record. All ten pick-place trainers now complete.
- Existing D1seed1 full-split job admitted above, unchanged original definition/dependency/arguments in
  `runs/setup/item1-fulleval-d1-seed1-pick-place-20261011-083012.yaml`; actual accepted scheduler launch
  08:30:14.227601/GPU5UUID/session3450316/attempt1. Genuine guard/full-window start logged08:30:17 and directly
  verified08:31:08: original checkpoint100000/windows1996/1916, owned one-UUID session/no startup traceback/exit.
  Original dispatch/start `runs/setup/item1-fulleval-d1-seed1-{dispatch-20261011-083012,start-20261011-083108}.json`.
  Nine of20 original results verified, one evaluation running, ten hammer evaluations await held original trainers;
  no item1 decision/variant comparison. Reference-only pick-place D0 margins remain unchanged, hammer margins pending.
- Startup record covers every active registry job: original D1seed1 evaluator and known blocked native stall;
  no active remote job. All ten hammer original sessions remain gone. Completed watcher `b8yrtq7ru` output read,
  real D1 trainer completion handled, exact background watcher rearmed once as `bu332l3jl`/PID3449271/start419626277.
  Exactly one watcher/exact absolute main command/5.65second beacon verified08:31:08; daemon2819571 unchanged.
- Charge30.44GB/hostavailable282.73GB/disk5.209TBfree is charge-only poststart, not second admission. Charge plus
  full hammer139.59GB reservations=170.02GB>150GB: no local group resume. Actual GPU query sees native parent
  onGPU4 and original evaluator parent3450317 onGPU5. Native3031654/3031655 retain exact identities, log440.59min
  old, finalJSON/JUnit/exit absent; AF_UNIX stall and interruption denial remain binding. No signal/alternate route/
  native retry/duplicate suite/replacement native waiter/source integration/commit/push/deployment/remote science/
  v2numerics/hammer resume/migration/cancellation/shortened schedule/reseeding/RL completion/replay deletion.
  GlobalHOLD retained; all20 results/both task-specific margin sets/original item1 rule still required.

### Historical completed results, post-event health and reference-only margins (08:14–08:17)
- Both D3 original standalone evaluations completed08:14:49.105455/106129, exit0/attempt1. Independently verified
  step100000/full windows1996/1916/batches125/120, every original console/file metric equal, genuine UUID guard,
  original accepted complete launch-source map/625-pass evidence/hashes and entire sessions3431484/3431485 gone.
  Evidence `runs/setup/item1-fulleval-d3-seed{0,1}-completion-20261011-081639.json`; original outputs unchanged.
- Fresh `runs/setup/campaign-health-20261011-081639.json` covers every active registry job. D1seed1 advances97,500,
  original parent2423622/start412705289/session2423619, log9.12seconds old/no inspected traceback; only other active
  job is the known blocked native stall. Nine trainers complete; nine standalone results verified, none running,
  eleven await dependencies/admission. No active remote job. All ten held original hammer sessions remain gone.
- Watcher `bsly0028t` completed on the two actual D3 events and was rearmed once as `b8yrtq7ru`/PID3439778/
  start419548279. Exactly one exact-command watcher and16.83second beacon verified08:16:39; daemon2819571 retains
  its identity. Earlier saved-health attempt crossed the genuine watcher completion boundary, failed its single-live
  watcher assertion and wrote no health artifact; it is not claimed as a successful observation/admission.
- D0 pick-place reference-only eleven decision margins frozen08:16:43 with unchanged registered formulas/readers/
  floors, full precision/input hashes in `runs/setup/item1-d0-pick-place-margins-20261011-081643.json` and numerical
  table in EXPERIMENT_LOG. Three additional margins explicitly diagnostic-only, no added gates. No variant comparison
  or winner; hammer margins await its own pair, all20 results required before deciding/item2/proxies.
- Charge32.33GB/hostavailable344.85GB/disk5.205TBfree is charge-only, not completePSS/new admission. Full hammer
  reservations plus charge171.92GB>150GB: no group resume. Genuine GPU query sees native parent onGPU4 and trainer
  parent onGPU5 only. Native3031654/3031655 exact identities/log426.10minutes old/no finalJSON/JUnit/exit;
  explicit interruption denial remains binding. No signal/alternate route/retry/duplicate suite/source integration/
  push/deployment/remote science/v2numerics/hammer resume or migration/cancellation/schedule change/reseeding.
  GlobalHOLD stays; no RL completion or replay deletion.

### Historical completions, admissions and full operational health (07:42–08:11)
- D4seed1 trainer completed07:42:40; original100k checkpoint/config/schedule/seed and entire session2476889
  teardown proved. Its existing evaluation launched07:45:02.126187 onGPU4UUID/session3405564; genuine startup
  verified07:45:45 and original full-split result completed07:53:56. Five further full-split results are independently
  verified with original hashes/full batches125/120/console-file metric equality and entire evaluator teardown:
  D0seed0 at07:49:19, D4seed0 at07:50:51, D2/D1seed0 at07:51:22, D4seed1 at07:53:56. Original proof files
  `runs/setup/item1-fulleval-{d0,d1,d2,d4}-seed0-completion-20261011-075453.json` and
  `runs/setup/item1-fulleval-d4-seed1-completion-20261011-075453.json`; raw stride6/pair02 metrics in EXPERIMENT_LOG.
- D3seed1 completed07:52:24, D3seed0 at08:06:40; both100k checkpoints CRC-valid/all718 normalized members equal,
  original200k schedules/seeds/sixworkers/prefetch2/configs and entire sessions2423625/2423624 teardown proved.
  Legacy original trainer launch events lack commit/source snapshots; that absence is preserved explicitly.
  Fresh combined admission above; existing D3 evaluations launched08:09:56.599436/601989, attempt1,
  GPU5UUID/sessions3431484/3431485. Direct08:10:56 verifies both actual one-UUID environments, genuine guards,
  checkpoint100000/full windows1996/1916, no inspected traceback/exit. Original dispatch/start records
  `runs/setup/item1-fulleval-d3-both-seeds-{dispatch-20261011-080956,start-20261011-081056}.json`.
- Startup evidence covers every active registry job: D1seed1 advances to96,150 with2.61second log age/no traceback,
  both newly started evaluators are live, and the known blocked native stall remains. No active remote job. Nine
  pick-place trainers complete, one continues; seven of20 standalone results verified, two running, eleven await
  dependencies/admission. No item1 decision; record numerical D0 margins before comparing variants.
- Exactly one current watcher `bsly0028t`/PID3430038, exact absolute main command/beacon18.09seconds;
  daemon2819571 identity unchanged. Its predecessors each completed on genuine events and were rearmed once,
  never duplicated. All ten held original hammer sessions remain gone. Disk5.207TBfree (`df -h`:4.8TiB), no RL
  finish/replay deletion. Charge46.35GB plus139.59GB full hammer reservations equals185.93GB>150GB: no local resume.
- Native launcher3031654/parent3031655 retain original start ticks/session, log420.38minutes old, finalJSON/JUnit/
  exit absent. Existing AF_UNIX stall and exact-parent interruption denial remain; no signal/alternate route/retry/
  duplicate suite/replacement native waiter, source change/integration/push/deployment/remote science/v2numerics,
  hammer resume or migration. GlobalHOLD retained; no cancellation/shortened schedule/reseeding.

### Historical full operational health (07:30–07:31)
- `runs/setup/campaign-health-20261011-073045.json` covers every active registry job: four advancing pick-place
  trainers, four active standalone evaluators, and the previously diagnosed stalled native verification. No active
  remote job. Trainer steps89,650/92,150/97,600/99,950 advance since07:26:52, with0.1–14.8second log ages and
  no inspected traceback. Evaluators retain owned sessions/assignedUUIDs, genuine CUDA processes and positive
  CPU deltas over the07:29:57→07:30:45 interval; quiet per-stride logs are not treated as a stall or completion.
  Two standalone results remain verified, four evaluations unfinished, fourteen wait for dependencies/admission.
- Exactly one watcher `b20v8j9i3`/PID3384819 is live with the exact absolute main command and17.54second beacon;
  scheduler2819571 unchanged/alive. No duplicate/rearm while the tracked watcher runs. All ten original held
  hammer sessions independently remain gone. Disk5.200TBfree (`df -h`:4.8TiB), no RL finish/replay deletion.
- Native launcher3031654/parent3031655 retain original start ticks/session; original log is380.20minutes old,
  finalJSON/JUnit/exit absent. Existing AF_UNIX stall remains confirmed, not repaired or accepted. Exact-parent
  `signal.pidfd_send_signal(fd, signal.SIGINT)` remains permission-denied; no signal/alternate route/retry/duplicate
  suite or replacement native waiter. GlobalHOLD retained. No new admitted dependency at the snapshot, source
  change/integration/push/deployment/remote science/v2numerics/hammer resume or migration.

### Historical D1seed0 completion/admission and health (07:24–07:27)
- `i1-d1-seed0-pick-place` completed100k/exit0 at07:24:15, original200k schedule/seed0/sixworkers/prefetch2.
  Checkpoint CRC, all718 normalized latest/numbered members, step100000, original train/eval config and entire
  session2413367 teardown verified. Exact hashes and registry exit in
  `runs/setup/item1-fulleval-d1-seed0-admission-20261011-072611.json`; fresh budget/host/GPU/disk/source gates above.
- Existing `fulleval-i1-d1-seed0-pick-place-100k` launched07:26:11.353785, attempt1, GPU5UUID/session3385436;
  original definition/dependency/args unchanged in its one-job operational queue, unmodified accepted scoped
  scheduler/sole registry/no resource overrides/globalHOLD retained. Direct07:26:52 verifies guard, exact one-UUID
  owned session, checkpoint100000/full windows1996/1916 and no inspected traceback/exit. Original dispatch/start
  `runs/setup/item1-fulleval-d1-seed0-{dispatch-20261011-072611,start-20261011-072652}.json`.
- Six pick-place trainers complete, four remain live/progressing at89,050–99,300steps with7.0–15.4second log ages
  and no inspected traceback. Two standalone results verified, four evaluations running, fourteen await original
  dependencies/admission. All ten hammer checkpoint-paused comparisons and all20 final results remain required.
  No item1 decision, thresholds/seeds/schedules unchanged.
- Tracked watcher `b20v8j9i3`/PID3384819 replaces completed `bcwjom6h3` once after its genuine D1 completion;
  exact absolute main command, one watcher and6.27second beacon verified07:26:52. Scheduler2819571 remains live.
  Disk5.203TBfree (`df -h`:4.8TiB), no RL finish/replay deletion. Native launcher3031654/parent3031655 retain exact
  identities/no exit; interruption remains denied, with no signal/retry/duplicate suite. No integration/push/
  deployment/remote science/v2numerics/hammer resume or migration.

### Historical verified standalone results and further admissions (07:06–07:23)
- D2seed1 and D0seed1 pick-place standalone evaluations completed exit0 at07:06:53 and07:16:36, respectively.
  Original summaries/logs/config/checkpoints are hashed, step100000/full windows1996/1916 and batches125/120
  verified, all console metrics equal file metrics, and entire evaluator sessions3348664/3360837 independently
  gone. Evidence `runs/setup/item1-fulleval-{d2,d0}-seed1-completion-20261011-072046.json`. Raw stride6/pair02
  metrics are recorded in EXPERIMENT_LOG, without a comparison winner or new margin/threshold.
- D0seed1 trainer completed07:06:22; its unchanged existing evaluation was admitted and launched07:09:37,
  real guard/full-window startup logged07:09:41. Original07:09 admission/dispatch and completed-result records
  preserve that provenance; no shortened evaluation, config change or new seed.
- D0seed0/D2seed0/D4seed0 trainers completed100k at07:15:34/07:18:07/07:20:40, exit0. Original200k schedules,
  seed0, sixworkers/prefetch2 and paired comparison requirements retained. Exact checkpoint hashes, CRC-valid
  archives, all718 normalized member equality, step100000/config equality and entire trainer-session teardown
  verified in the07:22 admission record; original registry exits/final in-training summary hashes in startup record.
- Three original seed0 full-split jobs actually launched07:22:26–27, attempt1, GPU5UUID, sessions3380163/3380164/
  3380173. Original job definitions/dependencies/args unchanged in operational queue
  `runs/setup/item1-fulleval-pick-place-seed0-d0-d2-d4-20261011-072222.yaml`. Accepted scoped `jobs.tick`, sole
  main registry, no resource overrides, globalHOLD retained. Direct07:22:58 startup verifies real guards,
  exact one-UUID environments and full windows{2:1996,6:1916} at100k; no inspected traceback/exit.
  Evidence `runs/setup/item1-fulleval-seed0-batch-{admission,dispatch}-20261011-072226.json` and
  `runs/setup/item1-fulleval-seed0-batch-start-20261011-072258.json`. Two of20 results complete, three running,
  fifteen await dependencies/admission; five pick-place trainers complete, five continue, ten hammer held.
- Latest tracked watcher `bcwjom6h3`/PID3379467 has exact main command, one watcher process and6.38second
  beacon at07:22:58. Each predecessor rearmed exactly once after genuine completion, no duplicate monitor.
  Earlier07:20:46 health snapshot honestly caught its predecessor's completed-event boundary before replacement.
  Daemon2819571 stays alive. `runs/setup/campaign-health-followup-20261011-072046.json` verifies remaining
  trainers advancing88,100–99,550steps with3.6–28.8second log ages/no inspected traceback, all ten held original
  hammer sessions gone, disk5.207TBfree. No RL finish/replay deletion.
- Native launcher/parent retain exact identities but remain stalled, with no finalJSON/JUnit/exit at07:20:46;
  explicit interruption approval still needed. No signal/retry/duplicate suite, source integration/commit/push,
  deployment/v2numerics/remote science, hammer resume/migration, cancellation, schedule change or reseeding.

### Historical first item1 completion and full-split start (06:52–06:58)
- `i1-d2-seed1-pick-place` completed at06:52:31, exit0, exactly100k on its original200k schedule. Config, seed1,
  six workers/prefetch2 and final in-training strides2/6 remain unchanged; entire session2423623 is gone. Original
  numbered100k checkpoint SHA-256 `126ecbc82394b9523695952ec4ef07cec5f4eb00b318891e1f3a006c90b8986e`;
  both latest/numbered CRC-valid and all718 normalized members equal, bothstep100000. Original completion evidence
  `runs/setup/item1-first-completion-20261011-0652.json`. In-training metrics do not decide item1.
- Actual accepted scheduler launch06:57:07, `fulleval-i1-d2-seed1-pick-place-100k`, attempt1, PID/session3348664,
  GPU5UUID. Original already-registered job/dependency/args/config copied unchanged to the one-job operational
  `runs/setup/item1-fulleval-d2-seed1-pick-place-20261011-0655.yaml`; no duplicate job or protocol change.
  Fresh full-footprint admission above; sole registry, unmodified `jobs.tick`, no resource override and globalHOLD
  retained. Original dispatch/start artifacts are `runs/setup/item1-fulleval-{dispatch-20261011-065706,start-20261011-0657}.json`.
- At06:58 actual GPU guard/owned-sessionUUID and full validation start are verified: checkpoint100000,
  windows{2:1996,6:1916}, no inspected traceback/exit. At that historical observation evaluation was unfinished;
  nine other pick-place trainers, ten held hammer runs and all20 full-split results remained required. The later
  completed standalone result is verified above; this original startup record is not rewritten. No item1 decision.
- Completed watcher `bh91jxnaq` emitted the real completion event and was rearmed once as `b3z9idtrr`, PID3346557.
  Exact absolute main command verified, beacon17.45seconds at06:58. Scheduler unchanged, globalHOLD retained;
  disk4.8TiBfree, no RL finish/replay deletion. Denied native interruption remains untouched, no signal/retry/duplicate
  suite. No integration, commit/push, remote deployment/scientific launch, v2 acceptance, hammer resume or migration.

### Earlier read-only follow-up (06:17)
- `runs/setup/campaign-health-followup-20261011-061733.json`: all ten pick-place trainers advance to79,100–94,750
  steps, with registered session/start-tick identities and expected run IDs in commands. Logs2.9–20.1seconds old,
  no inspected traceback, no exit file or100k completion. SchedulerPID2819571 and watcherPID3026196 match their
  exact commands; watcher.alive was updated in06:17. All ten held hammer original PIDs remain gone.
- Charge115.32GB/hostavailable213.49GB; no completePSS or new admission. Disk remains4.8TiBfree. Native launcher
  identity remains unchanged/alive with no exit file; original finalJSON/JUnit were last checked absent at06:08.
  Denied interruption remains untouched, no retry or duplicate launch; HOLD and existing monitor remain in place.

### Full operational health (06:08)
- Sample `runs/setup/campaign-health-20261011-060803.json`: all ten pick-place trainers are alive in their owned
  sessions, match expected commands, and advance to77,800–93,300logged steps. Logs2.3–17.9seconds old; no inspected
  traceback or stalled campaign trainer. None has reached100k or completed yet. All ten original hammer PIDs remain
  gone with their pauses held unchanged. All20 required `fulleval-i1-*-100k` jobs are already registered with their
  original trainer dependencies and numbered100k checkpoints; no duplicate evaluation was added or launched.
- Watcher `bh91jxnaq`/PID3026196 and schedulerPID2819571 remain live with exact expected commands; watcher beacon
  age10.35seconds. Disk4.8TiBfree, no RL completion or replay deletion. GlobalHOLD remains. Hostavailable203.74GB is
  above60GiB reserve; fullcharge114.80GB is below150GB. This charge-only check does not admit new work.
- The native verification remains stalled (297.5minutes without log output), with owned launcher/session still
  alive but no finalJSON/JUnit/exit. Original diagnostics and source freeze remain; the denied exact-parent interrupt
  has not been retried. Explicit workload-interruption approval is still needed to recover it. Expired90minute
  waiter is not replaced; no native duplicate, source integration, deployment, resume/migration or scientific launch.
  Earlier05:08 and04:08 original health samples remain unchanged in `runs/setup/`.

### Earlier operational health and blocked recovery (02:12)
- At02:12:15 all ten pick-place trainers are healthy and advance, with owned commands/sessions, logs3.7–19.6seconds
  old and no inspected traceback; steps53,250–64,250. Watcher `bh91jxnaq`/PID3026196 and schedulerPID2819571
  remain live; beacon17.0seconds. No campaign trainer stalled. MainHOLD retained; disk4.8TiBfree.
- Native candidate job **is stalled**, with no output since01:10:33 and no final JSON/JUnit/exit. Captured parent
  stderr proves real multiprocessing tensor transport errors `OSError: AF_UNIX path too long`; operational candidate
  cache.pymp directories are115bytes before the listener name. Partial630dot/oneF progress is not acceptance.
  The likely final worker-regression wait and separate untriaged earlyF are recorded honestly in RESULTS.
- Original captured stderr and unchanged210source hashes are preserved in the registered native run's
  `stall-captured-stderr.log` / `stall-diagnostic.json`. **SIGINT to exact owned parent3031655 was denied by the
  permission layer; no signal sent, no alternate route or duplicate job.** Task33 recovery needs explicit approval
  to interrupt that verification workload, then original failure evidence/teardown and a fresh admitted full suite
  at a shorter authorized physical candidate/cache path. Sources/assertions/thresholds remain frozen and unchanged.
- Watcher is not duplicated. Native-result waiter `b23nd555i` subsequently **expired at its 90-minute bound**,
  exit1/TimeoutError, with no suite exit observed, signal, relaunch or replacement waiter. This is a waiter timeout,
  not a completed native-suite result or interruption approval. No acceptance, integration, push, deployment,
  resume/migration or scientific launch. Health artifact `runs/setup/campaign-health-20261011-0214.json` records
  actual02:12:15; no new live health, PSS or admission is inferred from the notification.

### Earlier operational health (01:21)
- Original sample: `runs/setup/campaign-health-20261011-0111.json`, actual observation **01:21:00 +09:00**.
  All ten pick-place trainers advanced since 00:30, match their registered commands/sessions, and have logs
  0.1–21.3 seconds old with no inspected traceback. Logged steps range 46,500–57,150. No stalled trainer found.
- All ten original hammer PIDs remain gone and their lossless pauses remain held. Scheduler PID2819571 and
  watcher `bh91jxnaq`/PID3026196 match the expected commands; watcher beacon age16.7seconds. No duplicate monitor.
- The native suite was advancing beyond91% at the direct check in this turn; its final original JSON/JUnit were not
  yet present then. That partial progress is not acceptance. Existing result waiter `b23nd555i` remains armed;
  no restart, new waiter, source integration, remote deployment or scientific launch. Global HOLD remains.

### Completed memory window and earlier monitors (00:53)
- Four-hour observer `bfu6v5ovg` completed exit 0 at 00:12:59, not running: **102 samples** from 20:12:40,
  full-charge peak **227.57 GB**, owned-PSS peak **188.32 GB** on a different sample, per-run sampled peaks
  **9.15–10.77 GB**. Median actual cadence 142.59 seconds; one transient process-exit error. Original JSONL/summary
  remain under `runs/setup/memory-window-20261010-2015*`; complete results and limitations are in COMPUTE_PLAN.
  No cache subtraction, double-counted shmem, lifetime-peak claim or new workload admission is inferred.
- At 00:30 all twelve then-active trainers advanced, matched owned commands/sessions and had fresh logs
  (2.3–66.4 seconds), with no inspected traceback. D4 seed 0 subsequently checkpoint-paused and was independently
  verified, followed by the final D1 seed 0 pause. Ten pick-place trainers continue; accepted scheduler remains,
  pause controller and four-hour observer completed without duplication.
- Watcher `bh91jxnaq`, PID3026196, replaces completed `bczjqakzb` once after the final D1 seed 0 exit; beacon age
  5.9 seconds at 00:52. Headroom waiter completed; result waiter `b23nd555i` awaits the detached native suite without
  retry or signalling. No duplicate watcher, controller or observer is launched. Main HOLD stays;
  intentional pause crash labels retain their original registry events and verified proofs, with no restart exemption.
- Remote ledger is unchanged: four genuine CUDA/EGL isolation checks PASS, full native suite FAILED, fresh v2
  numerics not run. The frozen seven-file candidate remains unaccepted; no merge, deployment, resume, migration
  or scientific launch has occurred. The candidate suite is running, with genuine UUID-guard/test startup confirmed;
  next action is independent original-report/JUnit/source-map verification before any integration or deployment.

## Historical GPU incident and scheduler recovery (2026-10-10)
- At 18:07 NVIDIA management queries returned `Failed to initialize NVML: Unknown Error`; scheduler pid 957531
  exited. The first fresh guarded suite attempt saw zero CUDA devices. At 19:22 a fresh unchanged-main GPU-5 guard
  and real CUDA operation succeeded, but management queries still intermittently hang in kernel D state.
- The current watcher is tracked task `bb5esaarp` (pid 2697085), re-armed after the earlier timeout event. At 19:41
  its last beacon was 0.5 minutes old; its NVIDIA query child was in D state. HOLD remains. Full review acceptance
  finished at approximately 19:53: 492 passed, one failed, no errors/skips, sources unchanged; no acceptance claim.
  The native scheduler worker took approximately 904 seconds, beyond its unchanged 180-second check. Review source
  freeze lifted for durable query fixes and flushed/timed diagnostics; full acceptance must be rerun.
- Read-only audit confirmed two review query-cleanup defects: process-local watcher query ownership does not survive
  `--once` rearming, and scheduler `subprocess.run(timeout=60)` can block on unbounded reap of a D-state child. Fix with
  durable child identity and nonblocking cleanup after the current suite ends, then rerun the complete native suite.
- **Host repair verified at 20:02.** The user identified host `systemctl daemon-reload` with the systemd cgroup
  driver removing hook-injected NVIDIA devices from new processes, and applied runtime DeviceAllow rules on the
  host. Fresh bounded `nvidia-smi` returned exit 0. Unchanged-main `scripts/check_gpu_isolation.py` then passed real
  CUDA matmul and MetaWorld/MuJoCo EGL on GPU 4 and GPU 5, with each C+G child visible only on its assigned UUID
  (PIDs 2746214/2746662, EGL devices 16/17). Numeric/unset CVD rejected. Evidence: `docs/gpu_isolation.json` and
  `runs/setup/gpu-isolation-host-fix-20261010-2003.log`; previous isolation evidence was preserved under `runs/setup/`.
- **Source acceptance and scheduler recovery completed at 21:04.** Review commit `2850b30` integrated under HOLD
  as `b91b7c3`; all 210 production fingerprints exactly match the 625/625 native suite. Its original report/log/JUnit
  were copied byte-for-byte to `docs/tests.*`, retaining review provenance; prior evidence is preserved under
  `runs/setup/tests-before-extension-20261010-2108/`. Detached scheduler PID 2819571 is live in its own session.
  Accepted watcher is tracked task `b6jz1d99z`, PID 2819468; old watcher stopped before replacement, cursor retained.
- GPU/source holds are resolved. Global HOLD now remains for conservative shared-host memory admission while the
  four-hour observer finishes, not for GPU repair or stale test evidence. Current MemAvailable is only 47.6 GiB,
  below the deployed 128 GiB shared-tenant reserve. No healthy trainer restarted or comparison setting changed.
  Outage-related CUDA/NVML initialization failures are infrastructure failures, not chargeable restart failures;
  no qualifying new campaign exit has been observed, so no exemption event was fabricated.

## Historical second-server extension setup (2026-10-10)
- TCP/SSH access and remote Docker inventory verified from this container; four authorized remote GPUs were idle.
  Owned Docker resources and all host writes stay under `/home/compu/kaist/sunho`; no host VPN action is permitted.
- `docs/REMOTE.md` contains inventory, identity/transfer mechanics and pending gates. Decisions are registered in
  EXPERIMENT_LOG and both-host packing/disk reserves in COMPUTE_PLAN. The local registry remains authoritative.
- Exact installed native binaries were packaged and verified inside the built image, not replaced by mismatched
  cached wheels. Host guard, detached backend, result verification, provenance and watcher changes passed the full
  native suite and integrated under HOLD; no remote acceptance or campaign admission is inferred.
- No remote long job is admitted yet. Remote full suite, four one-UUID CUDA/EGL checks and cross-host initial-loss/
  short-run equivalence remain pending. Main post-repair NVML and both actual CUDA/EGL isolation checks passed at 20:02;
  remote CDI availability, native acceptance and cross-host bootstrap permission remain separate pending gates.
- The existing 20 item-1 runs stay local and healthy; item ordering/decision rules are unchanged. At 18:48 they
  reached 17.4k-24.35k, watcher pid 2612702 was live, disk free was 5230 GB, and the only heartbeat problem was the
  already documented dead scheduler. No new exit/failure or trainer restart was observed.

### Extension setup update, 19:28
- Owned remote image built: `sha256:c3ffea7f7c114dd12755f9d537f760a6d5b31c96c923233185b775e0324799b7`.
  Detached CPU-only Docker audit verified all runtime versions and 1,948 native package files byte-for-byte; this
  does not establish native CUDA/EGL acceptance. Build, image inspection and integrity evidence are under
  `runs/remote/runtime/`; exact details are in REMOTE.
- The eight explicit development-data files (40.61 GB) finished resumable SHA-256-verified transfer at 19:33.
  Semantic identity `56bd8ce9c59e81077061f577888f708541c2c14d605248f0cbdf81521e54bae1`; verified manifest under
  `runs/remote/data/`. Native loader acceptance, equivalence and remote long-job admission remain pending.
- Main NVML queries subsequently hung in kernel D state, including the watcher's child; no reset, driver change,
  permission change or foreign-process action was used. The tracked watcher eventually returned its timeout event
  and was re-armed as `bb5esaarp`. At 19:22 a fresh process passed the unchanged main GPU-5 UUID guard and a native
  CUDA sum=16.0. Full review acceptance is now running, log `runs/setup/remote-extension-local-suite-20261010-1926.log`;
  HOLD remains, scheduler remains down, production sources remain unchanged.
- At 19:13 all 20 item-1 PIDs were alive and free disk was 5238 GB. Several runs were at their scheduled 20k
  evaluations (last training-log writes 19:00-19:02); this is not evidence of a crash and none was restarted.
- Review fixes cover immutable host boundaries/provenance, verified numerical result return, read-only split reuse,
  8 GiB Docker shared memory, baseline encoder exports, and nonblocking watcher query-timeout cleanup. CPU boundary
  checks are not native acceptance. Remote suite, four real isolation checks and cross-host equivalence are pending.

### Shared-host RAM update, 20:15
- Read-only four-hour/two-minute sampling of all 20 owned item-1 sessions is running as tracked task `bfu6v5ovg`.
  Evidence `runs/setup/memory-window-20261010-2015.jsonl`; first sample: 185.6 GB whole-tree PSS, 198.1 GB container
  memory.current, 8.81-9.84 GB PSS per run. These are initial values, not a lifetime peak. Cache, shmem, pinned-memory
  uncertainty, growth and evaluation transients are being measured/accounted before final declarations.
- Review queue provisionally raises local tenant headroom to 128 GiB (was 60) with the ten-minute ramp unchanged.
  No memory-heavy local admissions during measurement, and no active comparison settings changed. Prefer eligible
  remote heavy/many-process jobs only after its native and comparability gates; detailed policy in COMPUTE_PLAN.

### Settled-source acceptance update, 20:34
- Fixed duplicate top-level RAM-reserve keys in the review queue; verified the actual parsed local reserve is
  128 GiB, with one authoritative key and the ten-minute ramp. This is not yet deployed production admission.
- Read-only audit found incomplete runtime/mount validation in lost-response remote reconciliation. Review now
  rejects privileged/writable-root containers, non-host PID namespace, anything other than 8 GiB shm, and any
  differing source/destination/readonly mapping among the eight expected binds. Launcher reuse also checks rootfs
  and privilege policy. All 66 targeted CPU boundary cases passed; native acceptance is separate.
- Complete guarded native suite started at 20:31, tracked task `bomqe4f3o`, with the original thresholds unchanged.
  Log `runs/setup/remote-extension-local-suite-20261010-2034.log`; evidence
  `.worktrees/review/runs/acceptance/local-20261010-2034/`. At 20:43 the suite passed 625/625, with zero failures/
  errors/skips and all 210 source fingerprints unchanged and independently verified. Source integration and
  scheduler recovery follow under HOLD; shared-resource admission remains independently gated.
- Settled-source native CUDA/EGL checks also returned real frames on both local UUIDs (PIDs 2805280/2805958,
  EGL devices 16/17), with each NVML-visible PID confined to its assigned UUID and numeric/unset/remote CVD rejected.
  Evidence under the suite directory's `isolation-gpu4/` and `isolation-gpu5/`. Their host-PID flag is an assertion,
  not independently inspected main Docker PID-mode proof; do not reuse these local reports as remote host-PID acceptance.
- At 20:33 all 20 item-1 launcher PIDs were present, with recent output under one minute old and steps 22.5k-33.65k.
  Watcher PID 2697085 is live. Disk: `df -h` reported 4.8T available. Four-hour memory observation remains running;
  no active run settings changed. Remote remains disabled pending native/CDI/isolation/data/equivalence gates.

### Clean remote release update, 21:12
- Accepted integration/evidence pushed as `200e5e6`; exact clean release checksum-verified beneath the authorized
  remote root at 21:10. Main provenance under `runs/remote/releases/200e5e63f6238180312c0007b4ecc1f33b669dd3/`.
- Fresh remote inventory: all four authorized UUIDs idle, NVIDIA queries successful, UUID-specific CDI names listed,
  immutable image unchanged, SSD free 4.054 TB. Registration alone is not actual CDI/CUDA/EGL acceptance. Remote
  remains disabled, with all native/isolation/equivalence gates intact and no approved cross-host bootstrap exception.
- Scheduler/watcher recovery and native evidence are committed/pushed; only runtime artifacts remain untracked.
  The four-hour memory observer and all 20 unchanged comparisons continue; no new memory-heavy admission.

## Event handling
- Event watcher: `python3 -I /home/ws/ws/splatter4d/scripts/watch_events.py --once` as a tracked background task, re-armed after each event.
- Hourly check: user `/loop`, session cron job `af519ff6` at :47 (expires after 7 days).
- Scheduler daemon: `scripts/jobs.py daemon --interval 30` (pid `experiments/daemon.pid`, log `experiments/daemon.log`).
  `touch experiments/HOLD` while editing or testing sources (every file in `s4d/ scripts/ tests/ configs/` is
  fingerprinted by the test gate); release only when source acceptance and independent resource-admission holds resolve.

## Heartbeats
- 2026-10-10 22:38: all 20 item-1 sessions alive with newest activity under one minute; watcher beacon 0.13 minutes
  old and scheduler PID 2819571 live. Ten real checkpoint-pause requests armed, zero completed pauses at sampling;
  no stall or native/heavy launch. Full charge **215.09 GB**, whole-cgroup PSS **184.43 GB** (owned sessions 180.45 GB),
  Pss_Shmem 77.72 GB already included, all cache charged. Cap remains violated. Disk free 5219 decimal GB. Both
  checkpoint-pause/resume and bounded bootstrap now explicitly authorized; HOLD remains. Evidence
  `runs/setup/memory-ceiling-20261010-2240.json`, whose observed_at is the authoritative 22:38 timestamp.
- 2026-10-10 22:10: all 20 item-1 launchers alive, watcher PID 2819468 live with a fresh beacon, scheduler PID
  2819571 live; no new registry exit/failure or stall detected. Three older hammer logs are at the scheduled 30k
  checkpoint/evaluation boundary; two show video encoding. Maximum activity age 6.7 minutes. Local full charge
  **226.32 GB**, whole-cgroup PSS **192.59 GB**, including **79.85 GB** Pss_Shmem, with no cache exclusion; mandatory
  150 GB cap remains violated. Disk free 5221 decimal GB. No restart, pause, migration or new heavy/native launch;
  HOLD and prior permission blocks remain. Evidence `runs/setup/memory-ceiling-20261010-2210.json`.
- 2026-10-10 22:06: conservative local charge **215.38 GB**, whole-cgroup PSS **190.72 GB**, including **79.36 GB**
  Pss_Shmem; all charged file cache retained. The mandatory **150 GB** cap is still violated. Latest liveness check:
  all 20 item-1 launchers alive, output age at most 2.7 minutes, scheduler PID 2819571 alive, watcher beacon age
  0.3 minutes; no stall detected. Free disk 5220 decimal GB. No trainer pause/restart/migration or new heavy/native
  launch. Six pure accounting and thirteen admission/replay checks passed; all changes remain review-only and no
  native acceptance or production enforcement is claimed. Temporary unavailable live PSS failed closed rather than
  being silently omitted. Evidence `runs/setup/memory-ceiling-20261010-2206.json`; HOLD and permission blockers remain.
- 2026-10-10 21:47: all 20 item-1 owned launch sessions are alive/non-zombie with matching session IDs and no exit
  files. Conservative local charged footprint **227.43 GB** (container memory.current), whole-cgroup PSS **190.08 GB**;
  PSS shared-memory component **79.13 GB**, already included. No file-cache exclusion. Mandatory 150 GB ceiling
  remains violated because checkpoint-bound pause execution was permission-denied; no experiment was cancelled,
  shortened, reseeded or migrated. Scheduler/watch tasks remain present under HOLD; new heavy/native launches held.
  Evidence `runs/setup/memory-ceiling-20261010.json`; budget code remains review-only, remote acceptance still pending.
- 2026-10-10 21:04: all 20 item-1 sessions remain alive and unchanged, with steps 24.15k-35.55k at the pre-merge
  check; scheduled 30k evaluations explain two older console writes. Accepted detached scheduler PID 2819571 and
  tracked watcher PID 2819468 are live, beacon age 9 seconds. No new launch/exit/failure or trainer restart.
  Production test gate accepts all 210 source fingerprints; 625 passed, zero failed/errors/skips. Global HOLD
  remains only for shared-host memory admission; MemAvailable 47.6 GiB, reserve 128 GiB. Remote disabled pending
  actual native/isolation/equivalence evidence. Four-hour memory observer still running. Disk remains 4.8T free.

- 2026-10-10 20:35: all 20 registered running jobs have live, non-zombie launch PIDs with matching owned sessions,
  no exit files, and output under one minute old. Steps advanced to 22.7k-33.85k; no trainer stall found. Watcher
  PID 2697085 is live, beacon age 19 seconds. Complete review suite `bomqe4f3o` is executing after the GPU-5 guard,
  with visible test progress and no acceptance claim. Memory observer is live, latest sample covers all 20 jobs
  with zero read errors (only 22 minutes into the four-hour window). Disk free 4,871.6 GiB. Scheduler remains down
  pending genuine source acceptance; HOLD retained for acceptance/shared-resource admission. No healthy job restarted.

- 2026-10-10 19:41: all 20 registered item-1 jobs have live PIDs; 13 are writing training steps (22.1k-29.7k), seven
  are in scheduled 20k evaluation with stride-2 panels already written and last output 9-20 minutes old. No new
  registry exits/failures or verified permanent stall. Watcher beacon age 0.5 min; scheduler remains dead. Disk free
  5235 GB. Trainers were not restarted. Local review suite still running, with one visible failure; HOLD retained.

- 20:13: HEARTBEAT_OK — 8 running jobs active (last activity < 3 min), watcher and scheduler alive, 5375 GB free.
  hammer pretraining at 5k/200k (0.175 s/step), pick-place at 2.9k (0.34 s/step), gate arms at 2.0k (3a) / 1.3k (3b).

- 21:20: HEARTBEAT_OK — 7 running jobs. Two Stage 0 evaluations took ~20 min (20:40-21:03) while another test run
  shared GPU 5; back to ~80-100 s and catching up. Heartbeat now counts eval companions' writes in the training run
  (they looked idle 25+ min). Progress: hammer pretraining 24k/200k (PSNR 23.7, retrieval 0.73 at 20k), pick-place
  12k/200k, gate 3b at 7k (PSNR 25.4, rel EPE 0.38 s2), Stage 0 at 230k (hammer train-camera success 0.23).

- 22:08: HEARTBEAT_OK — 9 running jobs active, watcher and scheduler alive, 5352 GB free. Screens at ~1.4k/6k
  (no difference from the reference yet); hammer base 39k/200k (0.26 s/step while sharing GPU 4 with the screens).

- 22:15 GPU 5 process audit: all 29 GPU processes belong to active registered jobs (Stage 0 train 864132/864134,
  their eval companions 1018246/1018247 with 12 spawned render workers each, pick-place pretraining 1255925); no
  orphans, no collectors, no foreign processes.

- 23:10: HEARTBEAT_PROBLEM 1 — the watcher had been dead for 25 min because I did not re-arm it after the
  `screen-s1-motion20` completion event; restarted (the registry cursor replays missed events). 10 running jobs OK.
  GPU 4 at 98% with 5 pretraining jobs (~0.4 s/step each); available RAM 90 GB (loader shared memory) — no further
  memory-heavy launches until some finish. Stage 0 at ~430k/1M (~6 h left).

- 00:08: HEARTBEAT_OK, but Stage 0 had slowed to 5 steps/s (page-cache thrashing from pretraining loaders). Loader
  workers reduced to 12/8; each pretraining run restarts after its next checkpoint (hammer base and it1-hammer done at
  00:13; pick-place base at 50k, it1-pick-place and it2-hammer at 10k, it2-pick-place at 20k pending). See log entry.

- 00:57: all six pretraining runs restarted from their checkpoints with 12/8 loader workers (no steps lost);
  available RAM 76 -> ~160 GB; Stage 0 back to ~25 steps/s.

- 01:10: HEARTBEAT_OK — 10 running jobs, 184 GB RAM available, Stage 0 at ~29 steps/s (hammer 573k, shelf-place
  553k). A second simultaneous ~20 min Stage 0 evaluation spike (both ended 01:04; first one 21:03) — evaluators
  caught up within one snapshot each time; watching for a third occurrence before investigating further.
  Pretraining: hammer base 67k, pick-place base 55k, iterations 16-20k (all ~0.4 s/step).

- 02:10: HEARTBEAT_OK, but available RAM 25 GB from other tenants' growth (~290 GB anon not ours); Stage 0 hammer at
  1.8 steps/s. Pausing the 4 iteration runs at their 30k checkpoints (held behind `hold-memory` in the queue).

- 02:27: host OOM killer killed both base pretraining runs (other tenants' memory). Orphaned loader workers
  terminated; scheduler now kills leftover session processes. Iterations pausing at 30k; base runs resume from
  70k / 60k with 8 workers once all four are paused (scheduler HOLD until then).

- 02:55: all four iteration runs paused at their 30k checkpoints (held behind `hold-memory`); base runs resumed at
  02:53 from 70k (hammer) and 60k (pick-place) with 8 loader workers. Available RAM 76 GB; resume the iterations when
  it stays above ~120 GB.

- 03:18: resume fast-forward fixed (index-level skip); pick-place base restarted onto it. Available RAM ~300 GB
  (other tenants released memory) -> iterations released and resumed from 30k. 10 jobs running.

- 04:10: HEARTBEAT_OK — 10 jobs, 189 GB RAM available. Eval spikes investigated: 9 per evaluator since 20:40, mostly
  simultaneous on both (e.g. 03:44-04:05), i.e. a shared external cause: every job inherits nice 5 from the agent
  shell, so CPU-bound eval workers yield to other tenants under host load (load 110-230). Not reniced (that would
  compete with other users); evaluators catch up, results unaffected. Stage 0 at 782k / 752k (~2.5 h left);
  base pretraining 81k / 70k (100k export in ~2-3 h); iterations at 37.5k.

- 05:10: HEARTBEAT_OK — 10 jobs. Stage 0 at 882k / 851k; hammer evaluator 6 snapshots behind (catches up after
  training). Queued the base-encoder DrM proxy: `export-s1-base-{task}-100k` waits for the 100k checkpoint, then
  `s1-proxy-base-{task}-s{1000,1001}` (200k agent steps) and their evaluators start on GPU 5.

- 06:08: HEARTBEAT_OK — 12 jobs (incl. 2 export jobs waiting for 100k). Stage 0 hammer 983k (train-camera success
  0.62-0.64), shelf-place 949k (0.0); hammer base 98.4k, pick-place base 92.1k, iterations ~54k.

- 07:08: HEARTBEAT_OK — 14 jobs, 155 GB RAM available, 5.4 TB free. Base pretraining 107k / 105k; iterations ~63k.
  Base-encoder proxy (100k encoder) on hammer at 100k agent steps: train-camera success 0.42 / 0.34, held-out 0.07 /
  0.04 (seeds 1000 / 1001); for reference the Stage 0 CNN had 0.02 / 0.00 at 100k (different seed; informal).
  Pick-place proxies at ~50k: 0.0 so far.

- 08:08: HEARTBEAT_OK — 8 jobs + 4 export jobs. Base hammer proxies done (final train-camera 0.13 / 0.29, peak 0.44
  near 100k then decline); base pick-place s1001 done (0.0), s1000 finishing. Queued iteration proxies: exports wait
  for the iterations' 100k checkpoints (~3 h), then 8 DrM proxy runs + evaluators on GPU 5. RAM available 93 GB.

- 08:45: OOM cascade from other tenants (5 pretraining runs killed). Scheduler now admits jobs only when host RAM
  allows (`ram_gb` + 40 GB reserve). Hammer base relaunched; pick-place base, it1 x2, it2-pick-place waiting for RAM.

- 09:20: more OOM kills (it1-pick-place, it2-hammer). OOM priorities added (iterations most killable). Running: both
  base runs, it1-hammer; waiting for RAM: it1-pick-place, it2-hammer, it2-pick-place (79 GB available, need 80).

- 09:21: HEARTBEAT_OK — running: base hammer 116k, base pick-place 115k, it1-hammer 75k (0.22-0.25 s/step) + 4 export
  waiters; waiting for host RAM (78 GB available, 80 needed): it1-pick-place (70k ckpt), it2-hammer (70k), it2-pick-place (70k).

- 09:23: host RAM freed up (~104 GB available); the scheduler readmitted it1-pick-place (attempt 6, resumes from
  70k) and it2-hammer (attempt 4, from 70k) on GPU 4. it2-pick-place still waits (64 GB available after the launches).

- 09:28: it2-pick-place readmitted on GPU 4 (attempt 4, from 70k). GPU 4 then ran five pretraining jobs at 98%
  utilisation (0.32-0.35 s/step) while GPU 5 ran one (0.30 s/step), so its queue entry was pinned to GPU 5 and the job
  was stopped (SIGTERM to its own session) one minute after resuming; relaunched on GPU 5 at 09:30 (attempt 5, from
  the same 70k checkpoint).

- 10:15: queued full-split `scripts/evaluate.py` runs (job dirs `runs/fulleval-*`, outputs kept out of the training
  runs) for base/it1/it2 at 100k (iteration comparison; in-training validation is too noisy, see EXPERIMENT_LOG) and
  base at 200k, plus the base-200k exports and DrM proxies (seeds 1000/1001). Base 100k evaluations started on GPU 5.

- 10:17: HEARTBEAT_OK — 14 running jobs (6 pretraining, 6 export waiters, 2 full-split evaluations), watcher and
  scheduler alive, 5400 GB free, 217 GB RAM available. Base 126k/200k (0.27 s/step, 200k at ~15:45); it1-hammer 85k,
  it1-pick-place 79k, it2-hammer 80k, it2-pick-place 80k (100k at ~11:45-12:15). Nothing stuck.

- 10:20: full-split evaluations of the base encoders at 100k done (4.5 min each; ~2000 windows per stride). hammer /
  pick-place at s2: moving PSNR 25.1 / 16.2 dB, held-out PSNR 14.8 / 14.6, rel. EPE 0->2 0.69 / 0.98, train-camera
  retrieval 0.83 / 0.95, hand-velocity probe R² 0.64 / 0.26. The in-training validation (first 8 batches) is biased,
  not just noisy: it gave rel. EPE 0.47 for hammer at 100k vs 0.69 on the full split. Iteration and length
  comparisons use the full split only.

- 11:07: HEARTBEAT_OK — 12 running jobs (6 pretraining, 6 export waiters), watcher and scheduler alive, 5378 GB
  free, 166 GB RAM available. Base hammer 135k / pick-place 140k (200k at ~17:00 / ~16:00); it2-pick-place 95k and
  it1-hammer 94k (100k at ~11:30-11:40), it2-hammer 89k and it1-pick-place 88k (~12:10). Nothing stuck.

- 12:19-12:43: review code merged twice (d3f7364, ac02dd0; suites 222/222 and 223/223). The second merge vectorises
  the oracle and synthetic-view splatting (a per-window loop made a full evaluation ~45 min; now ~13 min). Launched:
  3b DrM + CNN on plate-slide and assembly (light evaluation), full-split evaluations with the new metrics (base,
  it1, it2 at 100k; S1 at 6k), gate screens g-ref and 4a-4c. Declared GPU memory set to measured values (pretraining
  ~4.5 GB actual, declared 7; evaluations 5; CNN 5; frozen RL 2; evaluators 3) and slots raised to 20 per GPU, because
  the inflated declarations (12 per pretraining run) blocked GPU 4 at 1 GB of real use per 3 declared. Pending proxies
  (item 5) moved to priority 3, after the screens, as the review orders. Both GPUs run at ~95 %, so everything is
  slower (CNN ~14 agent steps/s).

- 12:43: HEARTBEAT_OK — 36 running jobs active, watcher and scheduler alive, 5350 GB free, 96 GB RAM available.
  Contention: with 6 evaluations, 4 gate screens and 2 CNN runs added, pretraining slowed from ~0.3 to 0.6-0.8 s/step
  (base hammer 149k, pick-place 159k; it1/it2 102-115k past their 100k comparison point); evaluations end within
  ~30 min. Two old-named evaluations (fulleval-s1-it2-hammer, fulleval-s1-it1-pick-place) launched with the merged
  code and duplicate their fulleval2 twins; left to finish (~20 min of GPU).

- 13:07: HEARTBEAT_OK — 36 running jobs active, watcher and scheduler alive, 5350 GB free, 200 GB RAM available.
  No exits since 12:41: the six full-split evaluations (base, it1, it2 at 100k) have finished stride 2 and are on
  stride 6 (GPU contention); gate screens 4d/4e wait for GPU 4's declared-memory budget (81 of 90 GB).

- 13:54: iteration-1 runs ended at 110k / 100k (rejected configuration; `train.stop_step` + SIGTERM, both exited 0).
  Gate reference done (6k); S1 not promoted. Item 2 screens crop x2 and synth-hammer running (0.8 s/step under load).

- 14:07: HEARTBEAT_OK — 33 running jobs active, watcher and scheduler alive, 5344 GB free, 261 GB RAM available.
  Gate 4a diverged (PSNR 21 -> 7.4 dB near step 1000, never recovered; not promoted); 4b 5.4k, 4c 5.8k, 4d 3.4k,
  4e 3.1k of 6k, all stable (PSNR 24.2-24.9). Item 2 screens: crop 3.4k (0.7-0.8 s/step), synth 0.7-0.8k
  (1.0-1.1 s/step), self-render just started; ~20-29 h to 100k at the current contention. Base hammer 154k,
  pick-place 165k; it2 hammer 111k, pick-place 123k.

- 15:07: HEARTBEAT_OK — 32 running jobs active, watcher and scheduler alive, 5342 GB free, 263 GB RAM available.
  Item 2 screens: crop 7.7k / 8.6k, synth 4.4k / 4.3k, self-render 4.2k / 3.6k (0.67-0.95 s/step). Item 3b at 300k
  target: plate-slide 115k (train-camera success 0.65 at 100k), assembly 131k (0.00 at 125k). Hammer 400k proxies:
  base 295k, it1 254-277k, it2 59-91k. Base pretraining 159k / 170k; it2 116k / 127k. Second-seed gate reference 1.3k.

- 16:07: HEARTBEAT_OK — 32 running jobs active, watcher and scheduler alive, 5334 GB free, 255 GB RAM available.
  No exits since 14:50 (long runs). Item 2 screens 7.7-13.9k; hammer 400k proxies: base 378k, it1 320-343k, it2
  132-157k; 3b plate-slide 0.89 at 150k (too easy), assembly 0.00 at 150k -> coffee-push and drawer-open CNN runs
  queued now (ranking order). Base pretraining 164k / 174k.

- 17:07: HEARTBEAT_OK — 26 running jobs active, watcher and scheduler alive, 5318 GB free, 272 GB RAM available.
  Item 2 screens: crop 16.8k / 18.5k, synth 11.4k / 11.2k, self-render 12.7k / 12.1k (0.67-1.08 s/step). 3b at
  300k target: plate-slide 1.00 at 200k (too easy), assembly 0.00 at 200k, coffee-push 0.11 and drawer-open 0.08 at
  25k. Hammer 400k proxies: base done, it1 s1000 done / s1001 391k, it2 209-228k. Base pretraining 169k / 179k;
  it2 126k / 137k.

- 18:07: HEARTBEAT_OK — 24 running jobs active, watcher and scheduler alive, 5317 GB free, 159 GB RAM available.
  Item 2 screens: crop 22.4k / 23.2k, synth 14.8k / 14.7k, self-render 17.9k / 17.5k. 3b: plate-slide 0.94 at 250k,
  assembly 0.00 at 250k, coffee-push 0.29 and drawer-open 0.44 at 75k. it2 hammer proxies 302k / 314k.
  Base pretraining 174k / 185k; it2 131k / 142k.

- 19:07: HEARTBEAT_OK — 17 running jobs active, watcher and scheduler alive, 5345 GB free, 158 GB RAM available.
  Fewer jobs, faster steps (0.38-0.88 s/step). Item 2 screens: crop 29.1k / 28.0k, synth 18.2k / 18.1k, self-render
  23.9k / 23.6k. 3b: plate-slide (0.89) and assembly (0.00) fail at 300k; coffee-push 0.17 and drawer-open 0.38 at
  125k. it2 hammer proxies trained to 400k (s1001 evaluation finishing). Base pretraining 179k / 192k; it2 136k / 149k.

- 19:49: base pick-place pretraining done (200k; A200). A300 runs for base queued (`s1-pretrain-*-base300k`).

- 20:07: HEARTBEAT_OK — 18 running jobs active, watcher and scheduler alive, 5339 GB free, 176 GB RAM available.
  Item 2 screens: crop 39.9k / 34.4k, synth 22.9k / 22.7k, self-render 33.1k / 33.0k. 3b: coffee-push 0.17 and
  drawer-open 0.92 at 200k. Base hammer 185k/200k; base300k runs at 1.8k / 2.0k (0.41 s/step). Base pick-place 200k
  evaluated (PSNR +0.4 dB over 100k; retrieval and EPE flat); its 200k proxies running.

- 21:07: HEARTBEAT_OK — 18 running jobs active, watcher and scheduler alive, 5328 GB free, 127 GB RAM available.
  Item 2 screens: crop 48.0k / 40.4k, synth 27.3k / 27.1k, self-render 40.4k / 40.2k. 3b: coffee-push 0.42 at 250k,
  drawer-open 0.94 at 275k (300k within ~30 min). Base hammer 192k/200k; base300k runs 10.0k / 10.3k.

- 21:38: item 3c decided — coffee-push replaces shelf-place (CNN 0.18 at 300k; plate-slide 0.89 and drawer-open 1.00
  too easy, assembly 0.00). Verification gate passed (20/20); pilot -> collection -> split / stats / D2-D3 checks ->
  held-out sets queued.

- 22:07: HEARTBEAT_OK — 11 running jobs active, watcher and scheduler alive, 5350 GB free, 123 GB RAM available.
  Base hammer pretraining done (200k; A200 for both tasks); its full evaluation and export running, 400k hammer
  proxies follow. Item 2 screens: crop 56.2k / 49.4k, synth 33.8k / 33.5k, self-render 47.9k / 47.8k (0.27-0.47
  s/step). base300k runs 18.5k / 18.8k. coffee-push collection 24/250 episodes (~2.5 h total).

- 23:07: HEARTBEAT_OK — 13 running jobs active, watcher and scheduler alive, 5343 GB free, 132 GB RAM available.
  Item 2 screens: crop 64.3k / 58.8k, synth 42.1k / 41.8k, self-render 55.3k / 55.4k. base300k 26.1k / 26.3k.
  base-200k hammer proxies 102k / 103k of 400k. coffee-push collection 77/250 (~4 h total, done ~02:00).

- 2026-10-09 00:07: HEARTBEAT_OK — 13 running jobs active, watcher and scheduler alive, 5343 GB free, 149 GB RAM
  available. Item 2 screens: crop 73.0k / 70.6k, synth 51.0k / 50.6k, self-render 63.3k / 63.3k (0.31-0.45 s/step;
  crop reaches 100k ~03:00, synth ~06:30). base300k 35.1k / 35.2k. base-200k hammer proxies 230k / 232k (0.49 / 0.36
  at 220k). coffee-push collection 191/250 (done ~01:15).

- 01:09: HEARTBEAT_OK — 12 running jobs active, watcher and scheduler alive, 5332 GB free, 144 GB RAM available.
  coffee-push data done (250 episodes, D2/D3 pass, stats done); its held-out render failed the replay check (target
  marker sites not in qpos) — fixed in the worktree (exact replay on coffee-push, other 8 tasks unchanged), suite
  running there, then merge and re-render. The watcher was down 01:00-01:07 (not re-armed after one event; replayed
  the missed events on restart). Item 2 screens: crop 82.0k / 82.9k, synth 60.1k / 60.0k, self-render 71.4k / 71.4k.
  base300k 44.4k / 44.7k; base-200k hammer proxies 360k / 365k.

- 02:07: HEARTBEAT_OK — 8 running jobs (both GPUs at 97-98 %), watcher and scheduler alive, 5330 GB free, 141 GB RAM
  available. Item 2 screens: crop 90.6k / 95.8k, synth 71.0k / 70.5k, self-render 79.3k / 79.2k. base300k 53.1k /
  53.5k. coffee-push fully ready (held-out sets rendered). Queued for the length study: exports at 100k/200k/300k of
  the base300k runs (waiting for the checkpoints), their full-split evaluations, hammer proxies (400k agent steps,
  seeds 1000/1001) at each point, pick-place proxies at 300k (reported, not counted).

- 03:07: HEARTBEAT_OK — 13 running jobs (7 training, 6 export waiters), watcher and scheduler alive, 5336 GB free,
  155 GB RAM available. Item 2: crop pick-place done and evaluated (fails 2d on pick-place: trajectory retrieval
  +0.00, CD-render +15 %); crop hammer 99.8k; synth 86.0k / 84.8k; self-render 87.1k / 87.1k (100k ~04:00).
  base300k 62.3k / 62.7k.

- 04:07: HEARTBEAT_OK — 11 running jobs, watcher and scheduler alive, 5326 GB free, 185 GB RAM available. Item 2:
  crop rejected (2d); synth hammer evaluated (trajectory retrieval 0.33 -> 0.74, hand-position R² 0.49 -> 0.89,
  CD-render p90 halved); synth pick-place done at 100k, evaluation running; self-render ~95k.

- 04:45: item 2d decided — synthetic near views adopted (passes on both tasks); crop and self-render rejected.
  Synthetic views abandon the dynamic group on pick-place (relative EPE 1.00, dynamic share 0.06): raised with the
  user. Queued: counted iteration 3 = synth (exports at 100k + hammer 400k / pick-place 200k proxies); length study
  for C = base + synth (screen runs continue to 200k = A200; synth300k runs = A300; exports, full evaluations and
  proxies follow). Base A300 runs continue as context / fallback.

- 05:07: HEARTBEAT_OK — 21 running jobs, watcher and scheduler alive, 5320 GB free, 61 GB RAM available (other
  tenants high). Iteration 3 proxies running (hammer 400k, pick-place 200k, seeds 1000/1001). Synth A200 continuations
  at 102.6k (hammer) and just resumed (pick-place); synth300k hammer started, pick-place waits for host RAM. base300k
  86.4k / 86.3k. Awaiting the user's call on the synth motion collapse (following the rule meanwhile).

- 06:07: HEARTBEAT_OK — 22 running jobs, watcher and scheduler alive, 5323 GB free, 201 GB RAM available. Synth A200
  continuations 112.8k / 110.3k; synth300k 8.0k / 6.7k; base300k 96.4k / 96.3k (base300k 100k exports next).
  Iteration-3 proxies: hammer 188-199k (0.07-0.23), pick-place 149-181k (~0.0).

- 07:07: HEARTBEAT_OK — 20 running jobs, watcher and scheduler alive, 5338 GB free, 193 GB RAM available. Synth A200
  continuations 124.2k / 121.6k (200k ~14:00); synth300k 15.1k / 13.6k (0.50-0.53 s/step; 300k in ~40 h with base300k
  sharing the GPUs); base300k 105.9k / 106.2k (context / fallback). Iteration-3 hammer proxies 338-356k (0.19-0.59);
  base300k@100k hammer proxies 52-55k.

- 08:07: HEARTBEAT_OK — 16 running jobs, watcher and scheduler alive, 5332 GB free, 95 GB RAM available. Synth A200
  continuations 137.3k / 133.9k; synth300k 22.0k / 20.5k; base300k 115.1k / 115.6k. Iteration 3 proxies done (no
  hammer RL gain within two seeds). Awaiting the user's call on the synth motion collapse.

- 11:18: HEARTBEAT_PROBLEM — host memory exhausted (6 GB available, swap full; pretraining >200 s/step); OOM kills of
  base300k@100k hammer proxies + evaluators, base300k hammer and synth300k pick-place pretraining. Paused both synth300k
  runs (held behind `hold-memory`); available back to ~36 GB, runs at ~0.5 s/step. Killed jobs restart from
  checkpoints as RAM admits. See EXPERIMENT_LOG "Fourth host-memory incident".

- 11:37-11:51: second OOM round (all pretraining killed); pretraining jobs now declare 30 GB RAM; base300k pick-place
  held behind `hold-memory` with the synth300k runs (concurrency 3: synth to-200k x2, base300k hammer). Scheduler fix
  merged (dfc5270, suite 224/224): jobs launched in the last 10 min count their declared RAM against available host
  memory; daemon restarted on it (pid in `experiments/daemon.pid`).

- 12:07: HEARTBEAT_OK — 13 running jobs, watcher and scheduler alive, 5344 GB free, 86 GB RAM available. Three
  pretraining runs at 0.23-0.26 s/step: synth to-200k 147.4k / 135.8k (200k ~16:00 / ~17:00), base300k hammer 115.8k.
  Held: base300k pick-place, synth300k x2 (memory; user decision on synth pending).

- 13:01: host memory recovered (245 GB available). Released synth300k hammer (the rule's length-study critical path;
  resumes from 20k) -> four pretraining runs. synth300k pick-place follows when a to-200k run finishes; base300k
  pick-place stays held (fallback) until the user decides on synthetic views.

- 13:07: HEARTBEAT_OK — 10 running jobs, watcher and scheduler alive, 5346 GB free, 226 GB RAM available. synth
  to-200k 161.8k / 151.3k; base300k hammer 133.1k; synth300k hammer 21.2k. Memory ample, so synth300k pick-place is
  released now (critical path of the length study) instead of waiting for a to-200k run to finish. base300k pick-place
  stays held (fallback).

- 14:07: HEARTBEAT_OK — 11 running jobs (5 pretraining), watcher and scheduler alive, 5345 GB free, 208 GB RAM
  available. synth to-200k 172.9k / 166.5k (200k ~16:30 / ~16:10); base300k hammer 148.1k; synth300k 32.8k / 34.5k
  (0.24-0.30 s/step; 300k in ~20-22 h).

- 15:07: HEARTBEAT_OK — 11 running jobs, watcher and scheduler alive, 5344 GB free, 220 GB RAM available. synth
  to-200k 184.4k / 181.5k (200k ~16:20 / ~16:15); base300k hammer 162.9k; synth300k 44.3k / 48.9k.

- 16:07: HEARTBEAT_OK — 11 running jobs, watcher and scheduler alive, 5331 GB free, 119 GB RAM available. synth
  to-200k 195.2k / 195.8k (200k ~16:30); base300k hammer 175.9k; synth300k 55.0k / 62.7k.

- 17:30: HEARTBEAT_OK — watcher and scheduler alive, 4.9 TB free, ~190 GB RAM available. All 10 S1 runs running.
  Items 1-3 code merged (ddd43dc, suite 239/239). GPU 4 had received 6 of the 10 S1 runs (my suite and smoke tests on
  GPU 5 blocked launches there) and ran them at ~0.9 s/step vs ~0.44 on GPU 5; the two youngest (synthinv-seed1 and
  synthsr-seed0 pick-place, 200 steps, no checkpoint) were restarted from step 0 pinned to GPU 5 (same seeds, so the
  same training). Loader benchmark relaunched from the main checkout (GPU 4).

- 18:07: HEARTBEAT_OK — watcher and scheduler alive, 4.9 TB free, 104 GB RAM available (host at 311 GB used). Split
  defect found and fixed (EXPERIMENT_LOG): the six seed-1 S1 runs were stopped and rerun on the fixed split
  (`s1v-*-seed1-split0-*`, merge 81692be, suite 240/240); 2 started, 4 + the two reference re-evaluations wait for host
  RAM. Seed-0 S1 runs at 4.4k-8.2k (0.30-0.49 s/step); synth200k proxies hammer 163k/166k (of 400k), pick-place s1000
  192k; S4 CNN pick-place 80k; loader benchmark at 6 of 12 settings.

- 19:07: HEARTBEAT_OK — watcher and scheduler alive, 4.9 TB free, 165 GB RAM available. Item 3 decided and merged
  (prefetch 2 x 6 workers; `ram_gb` 13 / 15 from measured PSS; reported). All 10 S1 runs running: seed 0 at
  10.9k-16.2k, seed 1 (fixed split) at 1.6k-7.2k; GPU 4 runs at 0.73-0.81 s/step next to the CNN run and two proxies,
  GPU 5 runs at 0.45-0.52. Slowest S1 run reaches 100k in ~20 h (sooner once the hammer proxies finish, 265k/270k of
  400k). S4 CNN pick-place 150k.

- 20:07: HEARTBEAT_OK — watcher and scheduler alive, 4.9 TB free, 117 GB RAM available. S1 seed 0 at 15.9k-24.1k,
  seed 1 (fixed split) at 5.7k-14.3k (GPU 4 runs 0.75-0.89 s/step, GPU 5 0.44-0.51); synth200k hammer proxies
  350k/356k of 400k; S4 CNN pick-place 212k (success 0 so far).

- 21:07: HEARTBEAT_OK — watcher and scheduler alive, 4.9 TB free, 135 GB RAM available. Hammer proxies done (context
  logged; replays deleted). S1 seed 0 at 21.3k-32.0k, seed 1 at 10.2k-21.6k (0.45-0.72 s/step; GPU 4 faster since the
  proxies ended); slowest reaches 100k in ~18 h (~15:00 tomorrow). S4 CNN pick-place 279k.

- 22:07: HEARTBEAT_OK — watcher and scheduler alive, 4.8 TB free, 142 GB RAM available, 12 jobs running. S1 seed 0
  at 27.1k-40.0k, seed 1 at 15.3k-29.3k (0.45-0.76 s/step). S4 CNN pick-place 351k, first non-zero success (0.07 on
  the training cameras at 350k).

- 23:07: HEARTBEAT_OK — watcher and scheduler alive, 4.8 TB free, 158 GB RAM available, 12 jobs running. S1 seed 0
  at 32.9k-48.0k, seed 1 at 20.4k-36.8k (0.43-0.74 s/step). S4 CNN pick-place 424k, training-camera success 0.16.

- 2026-10-10 00:07: HEARTBEAT_OK — watcher and scheduler alive, 4.8 TB free, 154 GB RAM available, 12 jobs running.
  S1 seed 0 at 38.8k-55.9k, seed 1 at 25.6k-44.3k (0.39-0.70 s/step; slowest at 100k in ~14 h). S4 CNN pick-place
  496k, training-camera success 0.18.

- 01:07: HEARTBEAT_OK — watcher and scheduler alive, 4.8 TB free, 250 GB RAM available, 12 jobs running. S1 seed 0
  at 44.8k-64.3k, seed 1 at 30.7k-51.9k (0.42-0.75 s/step). S4 CNN pick-place 572k, training-camera success 0.24.
  Spare host memory is not used for baseline pretraining now: both GPUs carry 5 S1 runs each and extra GPU load would
  delay the S1 -> item 1 -> item 2 critical path.

- 02:07: HEARTBEAT_OK — watcher and scheduler alive, 4.8 TB free, 189 GB RAM available, 12 jobs running. S1 seed 0
  at 50.7k-72.5k, seed 1 at 35.9k-59.5k (0.42-0.68 s/step; slowest at 100k ~13:30). S4 CNN pick-place 648k, success
  0.22.

- 03:07: HEARTBEAT_OK with one slowdown — watcher and scheduler alive, 4.8 TB free, but host RAM available fell to
  44-47 GB (host 353 GB used; other tenants grew, page cache shrank from ~255 to ~143 GB). S1 runs unaffected (seed 0
  at 57.8k-80.5k, seed 1 at 41.1k-66.8k, 0.44-0.61 s/step). The S4 CNN pick-place run (693k, success 0.42 at 690k) is
  not stalled but slowed from ~15 to 2.5 agent steps/s: its 33 GB frame memmap no longer stays in page cache and
  every random frame fault reads ~1.5 MB through the 2 MB device readahead (805 MB/s of reads, 538 major faults/s).
  The run is left as is (still a Stage 4 candidate seed). Fix for future CNN runs: MADV_RANDOM on the frame memmap
  (I/O hint only, sampled bytes unchanged; worktree commit, suite running).

- 03:30: host-memory incident — S4 CNN pick-place trainer OOM-killed at 693k (exit 137; available RAM 2-6 GB from
  other tenants' growth). Held until the S1 runs reach 100k, then resumes from its 650k checkpoint. Replay readahead fix merged
  (25793e0, suite 241/241). All 10 S1 runs unaffected; RAM available back to 40-69 GB. Details in EXPERIMENT_LOG.

- 04:07: HEARTBEAT_OK — watcher and scheduler alive, 4.8 TB free, 191 GB RAM available. S1 seed 0 at 66.3k-88.7k,
  seed 1 at 48.1k-74.1k (0.43-0.51 s/step); first S1 run reaches 100k ~05:30, the slowest ~11:30; their evaluations
  start automatically (configs in place). CNN pick-place trainer held (incident 03:16).

- 05:18: host-memory incident 6 — three S1 seed-1 runs and the CNN eval companion OOM-killed (host 400 GB used, swap
  full, pressure 82 %); I stopped the two slowest seed-1 runs to end the thrash (seed-0 runs back to 0.32-0.53
  s/step). The five resume from checkpoints (80k, 4 x 50k) when 75 GB is available. Watcher re-armed (it had exited on
  the crash events). Reference evaluations R seed 0 done for both tasks.

- 06:58: host memory still exhausted by other tenants (415 GB used, pressure 91-95 %). Two more S1 runs OOM-killed
  (synthsr-seed0-pick-place 80.3k, synth-seed1-split0-hammer 82.2k; resume from 80k). 3 S1 runs running, 7 waiting
  for 75 GB available. S1 completion now depends on when host memory frees.

- 07:07: watcher and scheduler alive; memory pressure gone (0 % over 60 s) but only 49 GB available (host 392 GB
  used), below the 75 GB the scheduler needs to resume an S1 run. First S1 run done (`s1v-synthinv-seed0-hammer`
  100k); its evaluation waits for RAM. Two S1 runs training fast on the emptied GPUs (0.18-0.20 s/step:
  synthsr-seed0-hammer 89.7k, synthinv-seed0-pick-place 78.2k). The 60 GB reserve is kept for now: the last two
  incidents came from 100+ GB jumps by other tenants within an hour.

- 07:49: host memory recovered (104-123 GB available); the scheduler resumed all seven waiting S1 runs from their
  checkpoints between 07:19 and 07:49. 9 S1 runs training, 1 done (synthinv-seed0-hammer, evaluated). CNN pick-place
  trainer stays held until S1 ends.

- 08:07: HEARTBEAT_OK — watcher and scheduler alive, 4.8 TB free, 107 GB RAM available, pressure ~1 %. S1: 2 done
  and evaluated (synthinv-seed0-hammer, synthsr-seed0-hammer) + both R seed-0 evaluations; 8 training (seed 0
  pick-place 89.3k / 91.4k, seed-1 hammer 53.1k-87.0k, seed-1 pick-place 52.2k-54.4k; 0.29-0.52 s/step). Slowest
  reaches 100k ~15:00, S1 decision ~15:30. Item-1 runs wait for S1 (it may change C).

- 09:07: HEARTBEAT_OK — watcher and scheduler alive, 4.8 TB free, 172 GB RAM available. S1: seed-0 pick-place runs
  at 99.1k / 99.8k (evaluations next), seed-1 hammer 60.3k-95.8k, seed-1 pick-place 59.1k-64.8k (0.30-0.50 s/step);
  4 of 12 evaluations done. Slowest at 100k ~14:50.

- 10:07: HEARTBEAT_OK — watcher and scheduler alive, 4.8 TB free, 188 GB RAM available, no pressure. S1: 8 of 12
  evaluations done; 4 seed-1 runs left (68.1k-77.1k, 0.22-0.34 s/step; slowest at 100k ~13:10). S1 margins for
  hammer recorded (R seed 1 shows the motion collapse on hammer). CNN pick-place resumed at 10:08 from 650k on
  GPU 5 (memory recovered; GPU 5 had one S1 run).

- 11:07: HEARTBEAT_OK — watcher and scheduler alive, 4.8 TB free, 185 GB RAM available. S1 seed-1 runs at
  79.0k-90.0k (0.24-0.36 s/step; slowest at 100k ~12:50). CNN pick-place 766k at ~46 agent steps/s (page cache warm
  again; 1M in ~1.5 h).

- 12:07: HEARTBEAT_OK — watcher and scheduler alive, 4.8 TB free, 201 GB RAM available. S1: 9 of 12 evaluations
  done; last three runs at 89.9k-94.8k (100k by ~13:00). CNN pick-place 883k, training-camera success 0.61 at 880k.

- 13:10: **S1 decided** — neither V1 nor V2 qualifies (both fail (c); V1 also misses (a) with pick-place dynamic share
  0.37 < 0.5); C unchanged, collapse reported as a limitation (it is a per-seed event in every variant). **Item 1
  queued** on C: D0-D4 x 2 seeds x 2 tasks (`i1-*`, 2 x 6 loader, `ram_gb` 13) + 20 full-split evaluations (configs
  in place for the started runs); 13 of 20 runs started at 13:05, the rest wait for host RAM. S4 CNN pick-place
  finished: 0.675 training cameras at 1M (replay deleted).

- 14:07: watcher and scheduler alive, 4.8 TB free, 108 GB RAM available. 18 item-1 runs at 1.6k-3.3k (0.78-0.89
  s/step, GPU-bound at 96 % on both GPUs). The last two (d4 pick-place) were blocked by the per-GPU declared-memory cap
  (`mem_gb` 10 x 9 = 90), not by RAM; measured peak is 3.7 GB allocated / ~4.7 GB per process, so item-1 jobs now
  declare `mem_gb` 6 and both started (all 20 running; finishing together instead of a slow 2-run tail). New
  diagnostics are logged as expected (e.g. at ~2k: d0 hammer utilisation 0.50 / hidden 0.90; d2 0.92 / 0.21).
  All 20 runs reach 100k in ~25 h (~15:00 tomorrow).

- 15:07: watcher and scheduler alive, 4.8 TB free. All 20 item-1 runs at 3.2k-6.9k (0.88-1.04 s/step). Host memory
  tight again: 76 GB available, memory pressure ~20 % (60 s and 5 min); this campaign holds 164 GB PSS (20 runs x
  ~8.2 GB), other tenants ~300 GB. No action yet (no kills); runs checkpoint every 10k (~2.7 h at this speed).

- 16:07: watcher and scheduler alive, 4.8 TB free, 71 GB RAM available, memory pressure 16-28 %. All 20 item-1 runs
  at 6.8k-10.1k; five are in their 10k in-training evaluation (retrieval + probes; 3-12 min so far, writing panels),
  not stalled. No kills.

- 17:15-17:16: HEARTBEAT_OK — the existing tracked watcher (pid 2413157) and scheduler (957531) are alive; no new
  item-1 exits or crashes. All 20 runs advanced to 11.0k-17.8k, with logs updated within 42 s; the previously quiet
  10k evaluations finished, including D2 seed 0 hammer (now 15.5k). Both GPUs at 96% utilisation; all trainer
  environments contain exactly their assigned allowed UUID. 4.8 TB disk free, ~108 GiB RAM available, memory pressure
  ~8-11%. Process-tree PSS totals 175 GiB (7.98-9.38 GiB/run); this is a post-evaluation snapshot, not a measurement
  of the transient evaluation peak. No healthy job restarted or protocol changed.

- 18:07-18:13: HEARTBEAT_PROBLEM — NVIDIA/NVML queries fail, and scheduler pid 957531 exited on query status 255.
  All 20 training jobs remain alive and progressing (14.8k-21.5k at the initial check), with no new registry failures;
  D0 seed-1 pick-place finished its 20k evaluation by 18:13. 4.8 TB disk free, 107 GiB RAM available, pressure 13-16%.
  Watcher re-armed (`bo345vngk`, pid 2612702); new launches held. Scheduler recovery prepared in the review worktree,
  but full-suite attempt failed at the GPU guard (fresh process sees zero CUDA devices), so no production merge or
  restart using unverified code. Host/container GPU visibility must recover first; existing jobs are untouched.

## Job table (running or pending)
| id | GPU | status | log |
|---|---|---|---|
| i1-{d0..d4}-seed{0,1}-{hammer,pick-place} (item 1, 100k of the 200k schedule, 2 x 6 loader) | 4/5 | All ten pick-place trainers completed100k; ten hammer checkpoint-paused/held | runs/pretrain/i1-*/log.txt |
| fulleval-i1-*-100k (20 registered full-split evaluations) | any | All ten pick-place completed/verified; none running; ten hammer evaluations await held original trainers; global HOLD retained | runs/fulleval-i1-*/console.log |
| base300k / synth300k pretraining and their export/proxy waiters | - | stopped and held (S3; checkpoints kept) | - |

Completed: data collection and checks, timing runs, Stage 0, review screens and proxies, S1 (10 runs + 12
evaluations), item-3 benchmark, S4 CNN pick-place (see Disk table and EXPERIMENT_LOG).

## Disk
| time | free on /home/ws |
|---|---|
| 2026-10-07 17:00 | 5578 GB |
| 2026-10-07 18:45 | 5.0 TB |
| 2026-10-07 19:02 | 5.0 TB (after timing-cnn-c1 finished; deleted runs/timing-cnn-c1/replay/, 0.98 GB) |
| 2026-10-07 19:55 | 4.9 TB (timing runs finished; their replays are kept: no final evaluation) |
| 2026-10-08 06:30 | 4.9 TB (stage0-drm-cnn-hammer-s2000 completed with final eval at 1M; deleted its replay/, 46 GB) |
| 2026-10-08 06:50 | 4.9 TB (stage0-drm-cnn-shelf-place-s2000 completed with final eval at 1M; deleted its replay/, 46 GB) |
| 2026-10-08 07:35 | 5.0 TB (s1-proxy-base-hammer-s1001 completed with final eval at 200k; deleted its replay/, 0.56 GB) |
| 2026-10-08 07:45 | 5.0 TB (s1-proxy-base-hammer-s1000 completed with final eval at 200k; deleted its replay/, 0.56 GB) |
| 2026-10-08 08:08 | 5.0 TB (s1-proxy-base-pick-place-s1001 completed with final eval at 200k; deleted its replay/, 0.56 GB) |
| 2026-10-08 08:47 | 5.0 TB (s1-proxy-base-pick-place-s1000 completed with final eval at 200k; deleted its replay/, 0.56 GB) |
| 2026-10-08 13:46 | 4.9 TB (s1-proxy-it2-lambdadyn4-pick-place-s1000 completed with final eval at 200k; deleted its replay/, 0.58 GB) |
| 2026-10-08 13:48 | 4.9 TB (s1-proxy-it2-lambdadyn4-pick-place-s1001 completed with final eval at 200k; deleted its replay/, 0.58 GB) |
| 2026-10-08 16:14 | 4.9 TB (s1-proxy-it1-decdim256-pick-place-s1000 completed with final eval at 200k; deleted its replay/, 0.58 GB) |
| 2026-10-08 16:30 | 4.9 TB (s1-proxy400-base-hammer-s1001 completed with final eval at 400k; deleted its replay/, 0.58 GB) |
| 2026-10-08 16:31 | 4.9 TB (s1-proxy400-base-hammer-s1000 completed with final eval at 400k; deleted its replay/, 0.58 GB) |
| 2026-10-08 16:46 | 4.9 TB (s1-proxy-it1-decdim256-pick-place-s1001 completed with final eval at 200k; deleted its replay/, 0.58 GB) |
| 2026-10-08 17:01 | 4.9 TB (s1-proxy-it1-decdim256-hammer-s1000 completed with final eval at 400k; deleted its replay/, 0.58 GB) |
| 2026-10-08 17:18 | 4.9 TB (s1-proxy-it1-decdim256-hammer-s1001 completed with final eval at 400k; deleted its replay/, 0.58 GB) |
| 2026-10-08 18:41 | 4.9 TB (s3b-cnn-assembly-s2000 completed with final eval at 300k; deleted its replay/, 14 GB used of a 49 GB sparse memmap) |
| 2026-10-08 19:05 | 4.9 TB (s3b-cnn-plate-slide-s2000 completed with final eval at 300k; deleted its replay/, ~14 GB) |
| 2026-10-08 19:06 | 4.9 TB (s1-proxy-it2-lambdadyn4-hammer-s1000 completed with final eval at 400k; deleted its replay/, 0.58 GB) |
| 2026-10-08 19:10 | 4.9 TB (s1-proxy-it2-lambdadyn4-hammer-s1001 completed with final eval at 400k; deleted its replay/, 0.58 GB) |
| 2026-10-08 21:30 | 4.9 TB (s3b-cnn-drawer-open-s2000 completed with final eval at 300k; deleted its replay/, ~14 GB) |
| 2026-10-08 21:38 | 4.9 TB (s3b-cnn-coffee-push-s2000 completed with final eval at 300k; deleted its replay/, ~14 GB) |
| 2026-10-08 21:51 | 4.9 TB (s1-proxy-base200k-pick-place-s1000 completed with final eval at 200k; deleted its replay/, 0.58 GB) |
| 2026-10-08 22:07 | 4.9 TB (s1-proxy-base200k-pick-place-s1001 completed with final eval at 200k; deleted its replay/, 0.58 GB) |
| 2026-10-09 01:27 | 4.9 TB (s1-proxy-base200k-hammer-s1001 completed with final eval at 400k; deleted its replay/, 0.58 GB) |
| 2026-10-09 01:31 | 4.9 TB (s1-proxy-base200k-hammer-s1000 completed with final eval at 400k; deleted its replay/, 0.58 GB) |
| 2026-10-09 06:19 | 4.9 TB (s1-proxy-it3-synth-pick-place-s1000 completed with final eval at 200k; deleted its replay/, 0.58 GB) |
| 2026-10-09 06:37 | 4.9 TB (s1-proxy-it3-synth-pick-place-s1001 completed with final eval at 200k; deleted its replay/, 0.58 GB) |
| 2026-10-09 07:26 | 4.9 TB (s1-proxy-it3-synth-hammer-s1000 completed with final eval at 400k; deleted its replay/, 0.58 GB) |
| 2026-10-09 07:37 | 4.9 TB (s1-proxy-it3-synth-hammer-s1001 completed with final eval at 400k; deleted its replay/, 0.58 GB) |
| 2026-10-09 13:03 | 4.9 TB (s1-proxy-base300k-100k-hammer-s1001 completed with final eval at 400k; deleted its replay/, 0.58 GB) |
| 2026-10-09 13:05 | 4.9 TB (s1-proxy-base300k-100k-hammer-s1000 completed with final eval at 400k; deleted its replay/, 0.58 GB) |
| 2026-10-09 18:01 | 4.9 TB (s1-proxy-synth200k-pick-place-s1001 completed with final eval at 200k; deleted its replay/, 0.56 GB) |
| 2026-10-09 18:15 | 4.9 TB (s1-proxy-synth200k-pick-place-s1000 completed with final eval at 200k; deleted its replay/, 0.56 GB) |
| 2026-10-09 20:39 | 4.9 TB (s1-proxy-synth200k-hammer-s1001 completed with final eval at 400k; deleted its replay/, 0.56 GB) |
| 2026-10-09 20:43 | 4.9 TB (s1-proxy-synth200k-hammer-s1000 completed with final eval at 400k; deleted its replay/, 0.56 GB) |
| 2026-10-10 12:56 | 4.8 TB (s4-drm-cnn-pick-place-s2000 completed with final eval at 1M; deleted its replay/, 46 GB apparent; the earlier ~33 GB used measurement was at 693k, not the final replay) |

## Next actions
All ten hammer checkpoint pauses are complete and held; whole-group local resume or migration still requires
budget/gate admission. Native verification recovery is blocked by the fresh permission-layer denial for SIGINT to
parent3031655. Preserve the original failed/incomplete evidence, frozen sources and HOLD; do not retry that outcome
without explicit interruption approval. All ten pick-place100k trainers and all ten original full-split results are
independently verified; no scientific job remains active. Whole hammer-group local resume still fails the150GB cap
(163.33GB including unchanged reservations at15:09:35); remote migration still requires actual repaired suite/numerical/
checkpoint-continuation gates. Nothing is cancelled, no partial-group resume or seed/schedule change.
Keep exactly one tracked watcher (`bw36yh48g`/PID3455689 verified15:09:35) and handle genuine events. Pick-place
D0-only numerical margins remain frozen in EXPERIMENT_LOG and immutable evidence; hammer margins await its own
reference pair.
Make no variant comparison/item1 decision until all20 results and both reference-margin sets. Remote native
and genuine cross-host numerical gates remain required; no blocked action is retried or gate waived.

1. Item 1: runs reach 100k (~1-1.5 days with 13-20 concurrent runs) -> 20 evaluations -> margins from D0's two seeds
   (written before comparing) -> item-1 rule -> report 1d to the user.
2. Item 2: M3D on the item-1 winner (2 seeds x 2 tasks; `analysis/queue_variants.py item2`) + 6-seed hammer RL
   proxies for M2D and M3D -> item-2 rule -> report 2b.
3. Item 1e leave-one-out, length study (A200/A300 of the final configuration, 6-seed RL), `method-frozen-v1`, Stage 1
   final RL (seeds 1000-1002, 1M).
