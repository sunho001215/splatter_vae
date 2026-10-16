# splatter4d Meta-World campaign — progress

Last updated: 2026-10-16 13:40 +09:00 (restart after the hardware update).
**Restarted.** Local devices are now 12 MIG 1g.24gb instances (GPUs 4/5/6); guard/EGL/scheduler adapted on main
(`756dc9b`, `d95e863`), isolation passed on all 12 (CUDA+EGL), native suite 778/0/0/0. Old daemon PID2819571 is gone
(container restart); one scheduler daemon, one watcher (venv python), Stage 2 and remote driver loops, memory alert
(147 GB) and cache-eviction loop run as tracked tasks. `experiments/HOLD` stays (scoped drivers admit); the Phase 3
`runs/setup/HOLD` and the expired shutdown whitelist were cleared after reconciliation (renamed `*.cleared/expired-20261016`).
All 14 remote jobs that survived the downtime exited 0 and were final-synced with verified checksums (8 pretrain-only
variants, 3 crop-hammer DrM pairs). No-consistency pick-place resumed losslessly from its 180k checkpoint on MIG.
New registration: Stage 2 RL arm for no-synth (rule pre-registered in EXPERIMENT_LOG 10-16 12:35). New baselines
SNeRL/DynaMo/3D-MVP integrate cleanly; pinned upstream code now executes (863/0/0/0 on the integration branch).
Chronology: `EXPERIMENT_LOG.md`; failures: `RESULTS.md`; restart state: `HANDOFF.md` (shutdown ledger is historical).

## Codex jobs (until integrated or discarded)
Codex cannot run shell commands in this container (bwrap namespace denied); reviews run as read-only tasks with the
diff inline.
| id | purpose | worktree | outcome |
|---|---|---|---|
| review / adversarial-review (2 runs) | MIG guard/scheduler | `.worktrees/mig` | blocked by sandbox, no findings possible |
| task-mvafnh17-iday3i | MIG guard/scheduler review (inline diff) | `.worktrees/mig` | 1 finding (unavailable process listing read as idle), fixed `d95e863`, integrated |
| task-mvafyrt7-0bpdom | baseline bundle review (inline code) | `.worktrees/bx-int` | 1 P0 claim (multi-view shift keeps 2 fixed views) = registered user protocol, not changed; caveat in reports |
| task-mvahby60-5isaes | patch proposal: port the remote verification bootstrap (native suite + GPU isolation in the image) from WIP `8f8f952` | `.worktrees/verify` | applied with manual fixes (7 rejected hunks, removed its campaign-path reuse check, reserved `verify-` ids); suite pending |
| task-mvajgv75-oizyou | adversarial review of the adapted verification bundle | `.worktrees/verify` | running |

## Current phase
- **Done:** Phase A/B/C, Stage 0, review items 1-5 and Stage 1. Eight tasks: bin-picking, coffee-push,
  door-open, hammer, peg-insert-side, peg-unplug-side, pick-place, stick-push.
- **Frozen method: C + D0 + M3D, 200k.** C retains synthetic near views as invariance positives/render targets;
  D0 retained by the registered item-1 rule; M3D adopted after the paired representation and six-seed hammer checks.
  No M3D seed collapsed at 100k (dynamic share 0.93-0.96); downstream unregistered pretraining defaults to seed 0.
- **Length study complete:** all eight full evaluations and sixteen RL proxy closeouts verified. A300 wins 5/40,
  loses 1/40; hammer RL +0.074722 < registered margin 0.201791. **L = 200,000** for every final task/ours ablation.
  Evidence: `runs/setup/length-verification-20261013.json`; scores and unchanged rule in RESULTS.
- **`method-frozen-v1` tagged and pushed:** accepted source226d8ae, full eight-task config embedded in the tag
  annotation and `runs/setup/method-frozen-v1.json`. No unaccepted baseline code is frozen into the method.
- **Stage 3 registered:** six fresh 200k pretraining jobs, six guarded exports, 24 DrM trainers + companions
  (eight tasks, seeds2000-2002, 1M agent steps). Reuse verified A200 hammer/pick-place encoders. Speculative300k
  attempts remain context only; no further speculative300k admissions. Registration evidence under `runs/setup/`.
- **Stage 2 attribution registered, now split across hosts** (item 1e skipped; host-neutral decision below). Margins were recorded
  before registration (`runs/setup/stage2-reporting-margins-20261013.json`). 108 jobs: 10 variants x 2 dev tasks at
  200k seed 0, 20 full-split evaluations, 8 exports, and 30 DrM 1M pairs (seeds 1000-1002; full method + T=1/no-motion/
  pure-no-InfoNCE/single-group). Attribution only; the frozen method does not change. 2c smoke-only/M3D-only bundle
  integrated (d81cbca); no DROID campaign training.
- **Final comparison complete:** ours24/24, SinCro/ReViWo48/48, CNN24/24. All16 accepted baseline pretrains complete;
  lengths unchanged: SinCro300k, ReViWo100,001. Success **and return**, last-5/final/AUC, per-task seed means/95% CIs:
  `runs/setup/rl-report-20261015-203425.json`; no frozen-method change follows from final results.

## Rules in force
- Local <=150GB full cgroup charge; measured job-by-job admission (trainer10.77GB PSS plus dataset page cache),
  no group/all-or-nothing requirement. Host reserve60GiB, local disk floor300GB. Remote <=1TB and >=15% free disk.
- **Host-neutral placement (user, 2026-10-15 20:09):** either host may run any job or checkpoint continuation;
  compared arms may span hosts and existing references stay valid. No host-equivalence or migration-comparison
  gate. Record every run's host; GPU/security/resource restrictions remain. Earlier placement rules are superseded.
- Frozen method/hyperparameters do not change; six non-development RL tasks never inform method decisions.
- Health notes only on change. Completed replays are currently retained: automatic runtime-computed deletion
  was removed after a permission gate. Disk reservations/floors remain enforced; W&B never blocks numerical work.

## Source and remote status
- Main sourcec5a7571 is integrated/pushed: intentional-stop accounting/refused-launch reconciliation plus bounded,
  redacted launcher diagnostics. Full native **764/0/0/0** (310.65s), all214 fingerprints verified against the commit;
  `runs/setup/native-tests-launch-diagnostics-20261015.json`. The preceding accounting-only suite passed763.
  Remote training remains accepted226d8ae / immutable image c3ffea7… (760/0/0/0 + four GPU isolation checks).
  No model/config/GPU-guard changes, so no healthy trainer was stopped for this scheduler-only integration.
- Historical cross-host characterization is not a placement gate after the20:09 user decision. Runtime source/image,
  dataset/checksum and GPU isolation safeguards remain: `runs/remote/acceptance/226d8ae/manifest.json`.
- Permission layer denied stopping/restarting old scheduler PID2819571; it remains under HOLD with its old imported
  reducer. Scoped drivers use the tested fix. Renewed permission is required to stop that daemon for final shutdown.
- Offline W&B stays in each remote run folder, pulled/uploaded by `runs/setup/tools/sync_wandb_remote.py`;
  status `runs/remote/wandb/status.json`. No remote login/key and no visualization gate on decisions.
- `experiments/HOLD` and Phase3 `runs/setup/HOLD` remain. Both driver loops and all known tracked campaign
  monitoring tasks are stopped; hourly cron cancelled, no schedules remain. `remote_campaign_driver.py` is the
  restart controller, not currently running. The denied daemon still reconciles under global HOLD and holds the
  registry lock; controller-stop operational event is preserved in the verification JSON for locked append later.
  Scientific180k pause already recorded/zero failures. Configuration/dispatch fixtures100passed; no new native suite.
- Review-only SNeRL/DynaMo/3D-MVP bundle remains unaccepted and upstream execution permission-blocked.
  Optional ports do not delay the freeze or existing campaign.

## Campaign jobs (placement/progress verified 2026-10-15 20:52–20:54)
| work | host | state | evidence/log |
|---|---|---|---|
| A200/A300 length study | local representation / remote RL | complete:8 full evaluations,16 proxies | runs/setup/length-verification-20261013.json |
| SinCro pretraining | remote four allowed GPUs | **8/8 complete** (stick-push at 23:42) | runs/setup/remote-driver-20261012.log |
| ReViWo pretraining | remote |8/8 complete | same |
| SinCro/ReViWo RL | remote | **Complete: 48/48** (8 tasks x seeds 2000-2002 x 2 methods). Two genuine DrM dormant-ratio failures, checked and recorded as is: ReViWo stick-push s2001 (0.000) and SinCro stick-push s2002 (0.017); see EXPERIMENT_LOG 10-14 21:23 and 10-15 08:20. Seed 2002 (last-5): ReViWo hammer 0.722, bin-picking 0.300, pick-place 0.000, coffee-push 0.397, door-open 0.925, peg-insert-side 0.000, peg-unplug-side 0.653, stick-push 0.888; SinCro hammer 0.025, bin-picking 0.795, pick-place 0.257, coffee-push 0.402, door-open 0.883, peg-insert-side 0.030, peg-unplug-side 0.707, stick-push 0.017 | runs/remote/finished/s4-drm-*.json |
| final Stage3 200k | remote | **Complete: 24/24** (8 tasks x seeds 2000-2002, 1M). Final comparison (pre-registered protocol) vs SinCro/ReViWo/CNN reported 08:41: IQM training-camera final success ours 0.76, SinCro 0.44, ReViWo 0.31, CNN 0.72; held-out cameras ours 0.13 vs <= 0.01; trajectories ours 0.72, CNN 0.76. See RESULTS "Final in-domain comparison" | runs/setup/final-comparison-20261015-084143.json |
| Stage2 attribution (10-16 08:00) | local GPUs4/5 + remote GPUs0–3 | **RL complete:30/30**.11/20 original ablation pretrains and10/20 full evaluations closed out (including both no-synth reports); no-consistency hammer completed200k, pick-place intentionally paused180k. Eight remote pretrain-only containers verified running/finite/advanced07:59; late-sync mirrors still unverified | runs/setup/stage2-report-20261016-070832.json; HANDOFF ledger |
| Crop follow-up (10-16 06:24) | pretraining/evaluation local GPUs4/5; DrM remote | Historical four re-evaluations reported. **4/4 pretrains, full evaluations and exports complete/driver-verified**. Representation endpoint report done, single-seed caveats in RESULTS. Three hammer crop DrM trainer/evaluator pairs run remotely, seeds1000–1002; nine pairs pending, no completed crop RL comparison | runs/setup/stage2-report-20261016-062420.json; HANDOFF tables |

### Placement and completion estimates (all10-16 KST, not guarantees)
- All crop/cropnosynth pretraining/evaluation/export work completed locally by06:24, superseding earlier crop ETAs.
  Both no-synth pretrains/full evaluations completed/closed out by07:08. No local scientific jobs/processes remain
  at07:49, but controllers/monitoring still run; this is not final all-process shutdown.
- No-consistency hammer completed200k normally; registered config, checkpoint ZIP/hash and final loss verified,
  driver closeout07:43:48. Pick-place **intentionally paused180k**; checkpoint hash/payloads, owned-session exit and
  no-steps-lost proof verified, zero charged failures reverified07:30. Reducer status `crashed` plus matching
  `intentional_stops` denotes a restartable intentional pause, not ordinary failure. Hash/resume ledger in HANDOFF;
  both no-consistency full evaluations pending under the expired07:15 local whitelist. Earlier ETA artifact
  `runs/setup/shutdown-eta-review-20261016-0640.json` is historical, superseded by actual hammer completion.
  No deadline extension, new local admission, remote stop or retry of denied scheduler-daemon stop.
- Late numerical sync completed07:48:39: **0/14 verified live snapshots**, nine checksum/change verification errors,
  five RemoteUnreachable errors. Report `runs/setup/late-remote-result-sync-20261016-0730.json`; all `final=False`,
  partial mirrors unverified, no restart authorization. Brief reconnect events do not prove trainer failure. W&B
  retry/pull task `ba1a6vvlo` failed with2400s TimeoutExpired (observed08:11); atomic status stayed unchanged,243 old
  persisted uploads, peg-unplug-side outcome unconfirmed. Local sync session exited naturally, no duplicate retry.
  Evidence `runs/setup/late-remote-wandb-sync-20261016-0730.json`; previous accepted completed results remain valid.
- Remote GPU0: no-lambda-dyn hammer + misscaled-none hammer; GPU1: their pick-place runs.
  GPU2: no-coverage hammer + misscaled-scale hammer; GPU3: their pick-place runs.
  Early measured step times imply **08:12–09:33 training-only** endpoints; planning allowance of25% for validation/
  checkpointing/sharing gives roughly**11:03–12:43**. No first10k validation had yet completed at observation, so
  allowance is not a measured confidence bound; later RL sharing may delay further. Full evaluations follow only
  while the local dispatcher is alive, or after reconnect. Already-running remote containers survive local downtime.
- Remote observations:~60.4GiB summed container RAM, all four GPUs9.24–9.33GiB used (nvidia-smi MiB converted),
  SSD3.397TB free of7.529TB (~45.1%), host available1.496TB. Local full cgroup118.8GB, below150GB.
- Eight remote pretrains use2.0/GPU of2.8 load ceiling. **At most four crop RL trainer/evaluator pairs fit alongside**
  (one/GPU), and only after verified encoders/resources. Some of12 registered pairs may stay pending during downtime.

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
| 2026-10-12 01:33 | 4.8TB (i2-proxy-m2d-hammer-s1003 completed with final eval at 400k; deleted its replay/, 0.58 GB) |
| 2026-10-12 02:11 | 4.8TB (i2-proxy-m2d-hammer-s1000 completed with final eval at 400k; deleted its replay/, 0.58 GB) |
| 2026-10-12 02:11 | 4.8TB (i2-proxy-m2d-hammer-s1002 completed with final eval at 400k; deleted its replay/, 0.58 GB) |
| 2026-10-12 02:11 | 4.8TB (i2-proxy-m2d-hammer-s1005 completed with final eval at 400k; deleted its replay/, 0.58 GB) |
| 2026-10-12 02:13 | 4.8TB (i2-proxy-m2d-hammer-s1004 completed with final eval at 400k; deleted its replay/, 0.58 GB) |
| 2026-10-12 02:14 | 4.8TB (i2-proxy-m2d-hammer-s1001 completed with final eval at 400k; deleted its replay/, 0.58 GB) |
| 2026-10-12 03:09 | remote 3.44 TB (s4-drm-cnn-peg-insert-side-s2000 completed with final eval at 1M; deleted its remote replay/, 49.2 GB) |
| 2026-10-12 03:41 | remote 3.46 TB (s4-drm-cnn-peg-unplug-side-s2001 completed with final eval at 1M; deleted its remote replay/, 49.2 GB) |
| 2026-10-12 03:50 | remote 3.50 TB (s4-drm-cnn-coffee-push-s2001 completed with final eval at 1M; deleted its remote replay/, 49.2 GB) |
| 2026-10-12 06:21 | remote 3.40 TB (s4-drm-cnn-coffee-push-s2000 completed with final eval at 1M; deleted its remote replay/, 49.2 GB) |
| 2026-10-12 06:33 | remote 3.44 TB (s4-drm-cnn-peg-unplug-side-s2000 completed with final eval at 1M; deleted its remote replay/, 49.2 GB) |
| 2026-10-12 06:46 | remote 3.47 TB (s4-drm-cnn-door-open-s2000 completed with final eval at 1M; deleted its remote replay/, 49.2 GB) |
| 2026-10-12 06:46 | remote 3.52 TB (s4-drm-cnn-door-open-s2001 completed with final eval at 1M; deleted its remote replay/, 49.2 GB) |
| 2026-10-12 06:55 | remote 3.56 TB (s4-drm-cnn-pick-place-s2001 completed with final eval at 1M; deleted its remote replay/, 49.2 GB) |
| 2026-10-12 07:11 | remote 3.59 TB (s4-drm-cnn-stick-push-s2000 completed with final eval at 1M; deleted its remote replay/, 49.2 GB) |
| 2026-10-12 07:26 | remote 3.63 TB (s4-drm-cnn-hammer-s2001 completed with final eval at 1M; deleted its remote replay/, 49.2 GB) |
| 2026-10-12 07:37 | remote 3.67 TB (s4-drm-cnn-stick-push-s2001 completed with final eval at 1M; deleted its remote replay/, 49.2 GB) |
| 2026-10-12 08:09 | 4.7TB (i2-proxy-m3d-hammer-s1003 completed with final eval at 400k; deleted its replay/, 0.58 GB) |
| 2026-10-12 08:09 | 4.7TB (i2-proxy-m3d-hammer-s1004 completed with final eval at 400k; deleted its replay/, 0.58 GB) |
| 2026-10-12 08:10 | 4.7TB (i2-proxy-m3d-hammer-s1005 completed with final eval at 400k; deleted its replay/, 0.58 GB) |
| 2026-10-12 09:13 | 4.7TB (i2-proxy-m3d-hammer-s1000 completed with final eval at 400k; deleted its replay/, 0.58 GB) |
| 2026-10-12 09:13 | 4.7TB (i2-proxy-m3d-hammer-s1002 completed with final eval at 400k; deleted its replay/, 0.58 GB) |
| 2026-10-12 09:15 | 4.7TB (i2-proxy-m3d-hammer-s1001 completed with final eval at 400k; deleted its replay/, 0.58 GB) |
| 2026-10-12 09:38 | remote 3.60 TB (s4-drm-cnn-bin-picking-s2000-r4 completed with final eval at 1M; deleted its remote replay/, 49.2 GB) |
| 2026-10-12 09:59 | remote 3.63 TB (s4-drm-cnn-hammer-s2000 completed with final eval at 1M; deleted its remote replay/, 49.2 GB) |
| 2026-10-12 10:09 | remote 3.66 TB (s4-drm-cnn-bin-picking-s2001 completed with final eval at 1M; deleted its remote replay/, 49.2 GB) |
| 2026-10-12 10:15 | remote 3.71 TB (s4-drm-cnn-peg-insert-side-s2001 completed with final eval at 1M; deleted its remote replay/, 49.2 GB) |
| 2026-10-12 10:15 | remote 3.76 TB (s4-drm-cnn-bin-picking-s2002 completed with final eval at 1M; deleted its remote replay/, 49.2 GB) |
| 2026-10-12 12:32 | remote 3.71 TB (s4-drm-cnn-coffee-push-s2002 completed with final eval at 1M; deleted its remote replay/, 49.2 GB) |
| 2026-10-12 12:48 | remote 3.75 TB (s4-drm-cnn-hammer-s2002 completed with final eval at 1M; deleted its remote replay/, 49.2 GB) |
| 2026-10-12 12:53 | remote 3.79 TB (s4-drm-cnn-door-open-s2002 completed with final eval at 1M; deleted its remote replay/, 49.2 GB) |
| 2026-10-12 13:44 | remote 3.81 TB (s4-drm-cnn-pick-place-s2002 completed with final eval at 1M; deleted its remote replay/, 49.2 GB) |
| 2026-10-12 13:50 | remote 3.85 TB (s4-drm-cnn-peg-insert-side-s2002 completed with final eval at 1M; deleted its remote replay/, 49.2 GB) |
| 2026-10-12 13:56 | remote 3.90 TB (s4-drm-cnn-peg-unplug-side-s2002 completed with final eval at 1M; deleted its remote replay/, 49.2 GB) |
| 2026-10-12 14:52 | remote 3.92 TB (s4-drm-cnn-stick-push-s2002 completed with final eval at 1M; deleted its remote replay/, 49.2 GB) |
| 2026-10-12 23:12 | remote 3.77 TB (ls-proxy-a200-pick-place-s1000 completed with final eval at 200k; deleted its remote replay/, 0.6 GB) |
| 2026-10-12 23:12 | remote 3.77 TB (ls-proxy-a200-pick-place-s1001 completed with final eval at 200k; deleted its remote replay/, 0.6 GB) |
| 2026-10-13 00:33 | remote 3.77 TB (ls-proxy-a200-hammer-s1000 completed with final eval at 400k; deleted its remote replay/, 0.6 GB) |
| 2026-10-13 00:44 | remote 3.77 TB (ls-proxy-a200-hammer-s1002 completed with final eval at 400k; deleted its remote replay/, 0.6 GB) |
| 2026-10-13 00:54 | remote 3.77 TB (ls-proxy-a200-hammer-s1001 completed with final eval at 400k; deleted its remote replay/, 0.6 GB) |
| 2026-10-13 00:54 | remote 3.77 TB (ls-proxy-a200-hammer-s1004 completed with final eval at 400k; deleted its remote replay/, 0.6 GB) |
| 2026-10-13 01:06 | remote 3.77 TB (ls-proxy-a200-hammer-s1003 completed with final eval at 400k; deleted its remote replay/, 0.6 GB) |
| 2026-10-13 01:25 | remote 3.77 TB (ls-proxy-a200-hammer-s1005 completed with final eval at 400k; deleted its remote replay/, 0.6 GB) |
| 2026-10-13 08:18 | remote 3.77 TB (s4-drm-reviwo-hammer-s2000 completed with final eval at 1000k; deleted its remote replay/, 2.1 GB) |
| 2026-10-13 08:40 | remote 3.77 TB (s4-drm-reviwo-pick-place-s2000 completed with final eval at 1000k; deleted its remote replay/, 2.1 GB) |
| 2026-10-13 12:44 | remote 3.75 TB (ls-proxy-a300-pick-place-s1000 completed with final eval at 200k; deleted its remote replay/, 0.6 GB) |
| 2026-10-13 12:50 | remote 3.75 TB (ls-proxy-a300-pick-place-s1001 completed with final eval at 200k; deleted its remote replay/, 0.6 GB) |
| 2026-10-13 13:25 | remote 3.75 TB (ls-proxy-a300-hammer-s1003 completed with final eval at 400k; deleted its remote replay/, 0.6 GB) |
| 2026-10-13 13:30 | remote 3.75 TB (ls-proxy-a300-hammer-s1000 completed with final eval at 400k; deleted its remote replay/, 0.6 GB) |
| 2026-10-13 13:30 | remote 3.75 TB (ls-proxy-a300-hammer-s1002 completed with final eval at 400k; deleted its remote replay/, 0.6 GB) |
| 2026-10-13 13:38 | remote 3.75 TB (ls-proxy-a300-hammer-s1001 completed with final eval at 400k; deleted its remote replay/, 0.6 GB) |
| 2026-10-13 13:38 | remote 3.75 TB (ls-proxy-a300-hammer-s1004 completed with final eval at 400k; deleted its remote replay/, 0.6 GB) |
| 2026-10-13 13:47 | remote 3.75 TB (ls-proxy-a300-hammer-s1005 completed with final eval at 400k; deleted its remote replay/, 0.6 GB) |
| 2026-10-13 15:57 | remote 3.74 TB (s4-drm-reviwo-coffee-push-s2000 completed with final eval at 1000k; remote replay retained) |
| 2026-10-13 16:05 | remote 3.74 TB (s4-drm-reviwo-door-open-s2000 completed with final eval at 1000k; remote replay retained) |
| 2026-10-13 20:28 | remote 3.72 TB (s4-drm-reviwo-bin-picking-s2000 completed with final eval at 1000k; remote replay retained) |
| 2026-10-13 20:37 | remote 3.72 TB (s4-drm-reviwo-peg-insert-side-s2000 completed with final eval at 1000k; remote replay retained) |
| 2026-10-13 22:17 | remote 3.71 TB (s4-drm-sincro-coffee-push-s2000 completed with final eval at 1000k; remote replay retained) |
| 2026-10-13 22:45 | remote 3.71 TB (s4-drm-reviwo-peg-unplug-side-s2000 completed with final eval at 1000k; remote replay retained) |
| 2026-10-13 23:19 | remote 3.70 TB (s4-drm-sincro-hammer-s2000 completed with final eval at 1000k; remote replay retained) |
| 2026-10-13 23:53 | remote 3.70 TB (s4-drm-sincro-door-open-s2000 completed with final eval at 1000k; remote replay retained) |
| 2026-10-14 04:47 | remote 3.67 TB (s4-drm-sincro-pick-place-s2000 completed with final eval at 1000k; remote replay retained) |
| 2026-10-14 04:47 | remote 3.67 TB (s4-drm-sincro-bin-picking-s2000 completed with final eval at 1000k; remote replay retained) |
| 2026-10-14 04:47 | remote 3.67 TB (s4-drm-sincro-peg-insert-side-s2000 completed with final eval at 1000k; remote replay retained) |
| 2026-10-14 05:21 | remote 3.67 TB (s4-drm-reviwo-stick-push-s2000 completed with final eval at 1000k; remote replay retained) |
| 2026-10-14 07:25 | remote 3.65 TB (s4-drm-sincro-peg-unplug-side-s2000 completed with final eval at 1000k; remote replay retained) |
| 2026-10-14 08:04 | remote 3.65 TB (s4-drm-sincro-stick-push-s2000 completed with final eval at 1000k; remote replay retained) |
| 2026-10-14 09:22 | remote 3.64 TB (s3-drm-splatter4d-hammer-s2000 completed with final eval at 1000k; remote replay retained) |
| 2026-10-14 10:02 | remote 3.63 TB (s3-drm-splatter4d-pick-place-s2000 completed with final eval at 1000k; remote replay retained) |
| 2026-10-14 11:20 | remote 3.62 TB (s4-drm-reviwo-hammer-s2001 completed with final eval at 1000k; remote replay retained) |
| 2026-10-14 11:45 | remote 3.62 TB (s4-drm-sincro-hammer-s2001 completed with final eval at 1000k; remote replay retained) |
| 2026-10-14 12:10 | remote 3.61 TB (s3-drm-splatter4d-bin-picking-s2000 completed with final eval at 1000k; remote replay retained) |
| 2026-10-14 13:05 | remote 3.60 TB (s3-drm-splatter4d-coffee-push-s2000 completed with final eval at 1000k; remote replay retained) |
| 2026-10-14 13:47 | remote 3.60 TB (s4-drm-reviwo-pick-place-s2001 completed with final eval at 1000k; remote replay retained) |
| 2026-10-14 13:47 | remote 3.60 TB (s3-drm-splatter4d-door-open-s2000 completed with final eval at 1000k; remote replay retained) |
| 2026-10-14 14:20 | 4.7TB (s2rl-full-hammer-s1000 completed with final eval at 1000k; replay retained) |
| 2026-10-14 14:24 | remote 3.59 TB (s4-drm-sincro-pick-place-s2001 completed with final eval at 1000k; remote replay retained) |
| 2026-10-14 14:24 | remote 3.59 TB (s4-drm-sincro-bin-picking-s2001 completed with final eval at 1000k; remote replay retained) |
| 2026-10-14 15:19 | remote 3.58 TB (s4-drm-reviwo-bin-picking-s2001 completed with final eval at 1000k; remote replay retained) |
| 2026-10-14 16:06 | remote 3.57 TB (s3-drm-splatter4d-peg-insert-side-s2000 completed with final eval at 1000k; remote replay retained) |
| 2026-10-14 16:26 | remote 3.57 TB (s4-drm-reviwo-coffee-push-s2001 completed with final eval at 1000k; remote replay retained) |
| 2026-10-14 16:35 | remote 3.57 TB (s4-drm-sincro-coffee-push-s2001 completed with final eval at 1000k; remote replay retained) |
| 2026-10-14 16:52 | 4.6TB (s2rl-noinfonce-hammer-s1000 completed with final eval at 1000k; replay retained) |
| 2026-10-14 16:58 | 4.6TB (s2rl-t1-hammer-s1000 completed with final eval at 1000k; replay retained) |
| 2026-10-14 17:17 | 4.6TB (s2rl-nomotion-hammer-s1000 completed with final eval at 1000k; replay retained) |
| 2026-10-14 17:24 | 4.6TB (s2rl-singlegroup-hammer-s1000 completed with final eval at 1000k; replay retained) |
| 2026-10-14 17:28 | remote 3.56 TB (s4-drm-sincro-door-open-s2001 completed with final eval at 1000k; remote replay retained) |
| 2026-10-14 17:41 | remote 3.56 TB (s4-drm-reviwo-door-open-s2001 completed with final eval at 1000k; remote replay retained) |
| 2026-10-14 17:49 | remote 3.55 TB (s3-drm-splatter4d-peg-unplug-side-s2000 completed with final eval at 1000k; remote replay retained) |
| 2026-10-14 18:46 | 4.5TB (s2rl-full-pick-place-s1000 completed with final eval at 1000k; replay retained) |
| 2026-10-14 18:57 | 4.5TB (s2rl-t1-pick-place-s1000 completed with final eval at 1000k; replay retained) |
| 2026-10-14 19:12 | remote 3.54 TB (s3-drm-splatter4d-stick-push-s2000 completed with final eval at 1000k; remote replay retained) |
| 2026-10-14 19:30 | remote 3.54 TB (s4-drm-reviwo-peg-insert-side-s2001 completed with final eval at 1000k; remote replay retained) |
| 2026-10-14 19:42 | remote 3.54 TB (s4-drm-sincro-peg-insert-side-s2001 completed with final eval at 1000k; remote replay retained) |
| 2026-10-14 19:50 | remote 3.54 TB (s4-drm-reviwo-peg-unplug-side-s2001 completed with final eval at 1000k; remote replay retained) |
| 2026-10-14 20:36 | remote 3.53 TB (s3-drm-splatter4d-hammer-s2001 completed with final eval at 1000k; remote replay retained) |
| 2026-10-14 20:41 | remote 3.53 TB (s4-drm-sincro-peg-unplug-side-s2001 completed with final eval at 1000k; remote replay retained) |
| 2026-10-14 20:50 | remote 3.53 TB (s3-drm-splatter4d-pick-place-s2001 completed with final eval at 1000k; remote replay retained) |
| 2026-10-14 20:50 | remote 3.53 TB (s3-drm-splatter4d-bin-picking-s2001 completed with final eval at 1000k; remote replay retained) |
| 2026-10-14 21:14 | remote 3.52 TB (s4-drm-sincro-stick-push-s2001 completed with final eval at 1000k; remote replay retained) |
| 2026-10-14 21:22 | remote 3.52 TB (s4-drm-reviwo-stick-push-s2001 completed with final eval at 1000k; remote replay retained) |
| 2026-10-14 21:50 | remote 3.51 TB (s3-drm-splatter4d-coffee-push-s2001 completed with final eval at 1000k; remote replay retained) |
| 2026-10-14 22:19 | remote 3.51 TB (s3-drm-splatter4d-door-open-s2001 completed with final eval at 1000k; remote replay retained) |
| 2026-10-14 22:19 | remote 3.51 TB (s3-drm-splatter4d-peg-insert-side-s2001 completed with final eval at 1000k; remote replay retained) |
| 2026-10-14 23:11 | 4.5TB (s2rl-full-hammer-s1001 completed with final eval at 1000k; replay retained) |
| 2026-10-14 23:21 | 4.5TB (s2rl-noinfonce-pick-place-s1000 completed with final eval at 1000k; replay retained) |
| 2026-10-14 23:21 | 4.5TB (s2rl-noinfonce-hammer-s1001 completed with final eval at 1000k; replay retained) |
| 2026-10-14 23:27 | 4.5TB (s2rl-nomotion-pick-place-s1000 completed with final eval at 1000k; replay retained) |
| 2026-10-14 23:33 | 4.5TB (s2rl-nomotion-hammer-s1001 completed with final eval at 1000k; replay retained) |
| 2026-10-14 23:38 | remote 3.49 TB (s3-drm-splatter4d-peg-unplug-side-s2001 completed with final eval at 1000k; remote replay retained) |
| 2026-10-14 23:49 | 4.5TB (s2rl-t1-hammer-s1001 completed with final eval at 1000k; replay retained) |
| 2026-10-14 23:49 | 4.5TB (s2rl-singlegroup-hammer-s1001 completed with final eval at 1000k; replay retained) |
| 2026-10-14 23:49 | 4.5TB (s2rl-full-pick-place-s1001 completed with final eval at 1000k; replay retained) |
| 2026-10-14 23:59 | remote 3.49 TB (s3-drm-splatter4d-stick-push-s2001 completed with final eval at 1000k; remote replay retained) |
| 2026-10-15 00:57 | 4.5TB (s2rl-nomotion-pick-place-s1001 completed with final eval at 1000k; replay retained) |
| 2026-10-15 01:02 | 4.5TB (s2rl-t1-pick-place-s1001 completed with final eval at 1000k; replay retained) |
| 2026-10-15 01:21 | remote 3.48 TB (s4-drm-reviwo-hammer-s2002 completed with final eval at 1000k; remote replay retained) |
| 2026-10-15 02:14 | remote 3.47 TB (s4-drm-reviwo-bin-picking-s2002 completed with final eval at 1000k; remote replay retained) |
| 2026-10-15 02:22 | remote 3.46 TB (s4-drm-sincro-hammer-s2002 completed with final eval at 1000k; remote replay retained) |
| 2026-10-15 02:22 | remote 3.46 TB (s4-drm-reviwo-pick-place-s2002 completed with final eval at 1000k; remote replay retained) |
| 2026-10-15 02:51 | remote 3.45 TB (s4-drm-sincro-bin-picking-s2002 completed with final eval at 1000k; remote replay retained) |
| 2026-10-15 03:14 | remote 3.45 TB (s4-drm-reviwo-coffee-push-s2002 completed with final eval at 1000k; remote replay retained) |
| 2026-10-15 03:22 | remote 3.45 TB (s4-drm-sincro-pick-place-s2002 completed with final eval at 1000k; remote replay retained) |
| 2026-10-15 03:22 | remote 3.45 TB (s4-drm-reviwo-door-open-s2002 completed with final eval at 1000k; remote replay retained) |
| 2026-10-15 03:59 | 4.5TB (s2rl-full-hammer-s1002 completed with final eval at 1000k; replay retained) |
| 2026-10-15 04:02 | remote 3.44 TB (s4-drm-sincro-coffee-push-s2002 completed with final eval at 1000k; remote replay retained) |
| 2026-10-15 04:10 | remote 3.44 TB (s4-drm-sincro-door-open-s2002 completed with final eval at 1000k; remote replay retained) |
| 2026-10-15 04:27 | remote 3.44 TB (s4-drm-sincro-peg-insert-side-s2002 completed with final eval at 1000k; remote replay retained) |
| 2026-10-15 04:45 | 4.5TB (s2rl-noinfonce-pick-place-s1001 completed with final eval at 1000k; replay retained) |
| 2026-10-15 04:56 | 4.5TB (s2rl-nomotion-hammer-s1002 completed with final eval at 1000k; replay retained) |
| 2026-10-15 05:07 | 4.5TB (s2rl-noinfonce-hammer-s1002 completed with final eval at 1000k; replay retained) |
| 2026-10-15 05:14 | 4.5TB (s2rl-t1-hammer-s1002 completed with final eval at 1000k; replay retained) |
| 2026-10-15 05:15 | remote 3.43 TB (s4-drm-reviwo-peg-insert-side-s2002 completed with final eval at 1000k; remote replay retained) |
| 2026-10-15 05:31 | 4.5TB (s2rl-singlegroup-hammer-s1002 completed with final eval at 1000k; replay retained) |
| 2026-10-15 06:15 | remote 3.42 TB (s4-drm-sincro-peg-unplug-side-s2002 completed with final eval at 1000k; remote replay retained) |
| 2026-10-15 06:46 | remote 3.42 TB (s3-drm-splatter4d-hammer-s2002 completed with final eval at 1000k; remote replay retained) |
| 2026-10-15 06:51 | remote 3.42 TB (s4-drm-reviwo-peg-unplug-side-s2002 completed with final eval at 1000k; remote replay retained) |
| 2026-10-15 07:07 | remote 3.42 TB (s3-drm-splatter4d-bin-picking-s2002 completed with final eval at 1000k; remote replay retained) |
| 2026-10-15 07:17 | remote 3.42 TB (s3-drm-splatter4d-pick-place-s2002 completed with final eval at 1000k; remote replay retained) |
| 2026-10-15 07:48 | remote 3.42 TB (s4-drm-reviwo-stick-push-s2002 completed with final eval at 1000k; remote replay retained) |
| 2026-10-15 07:48 | remote 3.42 TB (s3-drm-splatter4d-door-open-s2002 completed with final eval at 1000k; remote replay retained) |
| 2026-10-15 07:54 | remote 3.42 TB (s3-drm-splatter4d-coffee-push-s2002 completed with final eval at 1000k; remote replay retained) |
| 2026-10-15 07:59 | remote 3.42 TB (s3-drm-splatter4d-peg-insert-side-s2002 completed with final eval at 1000k; remote replay retained) |
| 2026-10-15 08:01 | 4.5TB (s2rl-full-pick-place-s1002 completed with final eval at 1000k; replay retained) |
| 2026-10-15 08:06 | 4.5TB (s2rl-singlegroup-pick-place-s1000 completed with final eval at 1000k; replay retained) |
| 2026-10-15 08:19 | remote 3.42 TB (s4-drm-sincro-stick-push-s2002 completed with final eval at 1000k; remote replay retained) |
| 2026-10-15 08:25 | remote 3.42 TB (s3-drm-splatter4d-peg-unplug-side-s2002 completed with final eval at 1000k; remote replay retained) |
| 2026-10-15 08:40 | remote 3.42 TB (s3-drm-splatter4d-stick-push-s2002 completed with final eval at 1000k; remote replay retained) |
| 2026-10-15 09:22 | 4.5TB (s2rl-singlegroup-pick-place-s1001 completed with final eval at 1000k; replay retained) |
| 2026-10-15 09:27 | 4.5TB (s2rl-t1-pick-place-s1002 completed with final eval at 1000k; replay retained) |
| 2026-10-15 09:58 | 4.5TB (s2rl-nomotion-pick-place-s1002 completed with final eval at 1000k; replay retained) |
| 2026-10-15 09:58 | 4.5TB (s2rl-noinfonce-pick-place-s1002 completed with final eval at 1000k; replay retained) |
| 2026-10-15 10:23 | 4.5TB (s2rl-singlegroup-pick-place-s1002 completed with final eval at 1000k; replay retained) |
| 2026-10-16 13:13 | remote 3.36TB (s2rl-crop-hammer-s1000 completed with final eval at 1000k; replay retained) |
| 2026-10-16 13:13 | remote 3.36TB (s2rl-crop-hammer-s1001 completed with final eval at 1000k; replay retained) |
| 2026-10-16 13:13 | remote 3.36TB (s2rl-crop-hammer-s1002 completed with final eval at 1000k; replay retained) |

## Next actions
1. Remote controller: verified exports > baseline/final200k pretraining > RL fill; job-by-job admission as capacity
   frees, full numerical result sync, offline W&B attempt, completed replay retained. Old speculative300k stays out.
2. Stage3: six final200k encoders -> guarded exports -> eight-task RL seeds2000-2002; two dev encoders reused.
3. Stage2 local driver (`runs/setup/tools/stage2_local_driver.py`, every5min): verified closeouts -> eval configs/
   exports -> RL; job-by-job below the140GB admission ceiling. Report against the recorded margins once each
   comparison's runs are all complete; no post-freeze method/hyperparameter selection.
4. Continue SinCro/ReViWo Stage4. New SNeRL/DynaMo/3D-MVP execution remains blocked; no unaccepted release or long run.
