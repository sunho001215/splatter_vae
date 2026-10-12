# splatter4d Meta-World campaign — progress

Last updated: 2026-10-12 14:35 +09:00. Concise current state; chronology and evidence are in
`docs/EXPERIMENT_LOG.md`, failures in `docs/RESULTS.md`, the full earlier version of this file in commit `01e303b`.

## Current phase
- **Done:** Phase A, Phase B (8 tasks: bin-picking, coffee-push, door-open, hammer, peg-insert-side,
  peg-unplug-side, pick-place, stick-push), Phase C, Stage 0, Stage 4 prep (SinCro, ReViWo ported), review items
  1-5, S1 (C remains the base: synthetic near views as invariance positives and render targets).
- **Stage 1 — item 1 decided 23:10: D0 stays** (registered rule over all 20 verified results; D2 and D1 fail
    their utilisation and PSNR gates). Table in RESULTS; output `runs/setup/item1-decision-20261011.txt`.
- **Item 2 decided 2026-10-12: M3D adopted** (12/16 motion pairs better, no guard worse; RL 0.756 vs 0.666 over 6
  hammer seeds). No M3D seed collapsed (dynamic share 0.93–0.96 at 100k), so later comparisons default to one seed.
  Table in RESULTS; output `runs/setup/item2-decision-20261012.txt`.
- **User decisions 2026-10-12:** item 1e skipped (term contributions measured by the Stage 2 ablations); 2c smoke only.
- **2c done (smoke level) and bundled with the M3D-only training path on `review-items`** (3cdad57, 75e2f5d, 28c8f6c,
  6e976f8; not integrated). DROID batches now carry 3D track arrays and gripper FK points; contract checks pass on
  the local sample; M3D probe loss -46 % over 300 steps (-39 % on one fixed batch). Details in EXPERIMENT_LOG.
  Next: one full native suite on the bundle, then integration.
- **Length study running** (C + D0 + M3D, one seed): A200 = i2-m3d-seed0-<task> continued 100k -> 200k, A300 new
  300k runs; then 8 full evaluations, 16 RL proxies, the pre-registered rule, `method-frozen-v1`.
- Pretraining lengths: ours default 300k (final L chosen on hammer/pick-place before the freeze); SinCro 300k;
  ReViWo 100 001.

## Rules in force (user decisions 2026-10-11)
- Local <= 150 GB full cgroup charge; admit job by job with measured footprints (item-1 trainer 10.77 GB PSS);
  no group or all-or-nothing admission. Host reserve 60 GiB, disk floor 300 GB. Remote <= 1 TB.
- Comparability = same code, config, data and protocol on one host, or after a passed cross-host check; runs need
  not start together. Decisions wait until every run in the comparison has finished.
- Health entries only when something changes.

## Source and remote status
- Main `5a7177e` (pushed, deployed): portability repairs, cross-host v2 producer, remote W&B offline and RL
  evaluation `--output-dir`; local and remote native suites 728/0/0/0; four remote isolation checks passed.
- Cross-host v2: step 0 bit-identical; 200-update RMSE 4.687e-5 vs same-host 4.588e-5 (ratio 1.02, registered
  criterion not met) — **accepted as equivalent by the user**; acceptance manifest `runs/remote/acceptance/5a7177e/`.
- Remote W&B: offline in each run folder; pulled and uploaded by `runs/setup/tools/sync_wandb_remote.py`
  (status `runs/remote/wandb/status.json`). W&B never blocks decisions.
- Global `experiments/HOLD` stays: it blocks the scheduler from relaunching old held jobs (S3 base/synth300k,
  s1v seed-1 runs); current work is dispatched through scoped queues in `runs/setup/`.

## Running jobs
| id | host / GPU | status | log |
|---|---|---|---|
| ls-a200-m3d-seed0-{hammer,pick-place} | local GPU 4 | running from 100k (13:26), ~107k at 14:25 | runs/pretrain/i2-m3d-seed0-*/ |
| ls-a300-m3d-seed0-{hammer,pick-place} | local GPU 5 | running (13:26), ~10k at 14:25 | runs/pretrain/ls-a300-*/ |
| s4-sincro-{hammer,pick-place}, s4-reviwo-{hammer,pick-place,bin-picking} | local GPU 4/5 | running; 11 more admitted as memory frees | runs/pretrain/s4-*/ |
| s4-drm-cnn-stick-push-s2002 (+ `-eval`) | remote | 22 of 23 CNN pairs finished | runs/s4-drm-cnn-*/, runs/remote/stage4-finished/ |

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

## Next actions
1. Length study: the per-minute driver admits evaluations, exports and proxies as trainers finish; apply the rule.
2. Bundle (`review-items`): full native suite once 15 GB local headroom exists, then integrate under HOLD and push.
3. `method-frozen-v1`, Stage 2 ablations, Stage 3 final RL; Stage 4 SinCro/ReViWo DrM after their pretraining.
