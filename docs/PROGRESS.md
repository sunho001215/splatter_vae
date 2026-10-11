# splatter4d Meta-World campaign — progress

Last updated: 2026-10-11 17:10 +09:00. Concise current state; chronology and evidence are in
`docs/EXPERIMENT_LOG.md`, failures in `docs/RESULTS.md`, the full earlier version of this file in commit `01e303b`.

## Current phase
- **Done:** Phase A, Phase B (8 tasks: bin-picking, coffee-push, door-open, hammer, peg-insert-side,
  peg-unplug-side, pick-place, stick-push), Phase C, Stage 0, Stage 4 prep (SinCro, ReViWo ported), review items
  1-5, S1 (C remains the base: synthetic near views as invariance positives and render targets).
- **Stage 1 — item 1 (depth redesign D0–D4, 2 seeds x hammer/pick-place, 100k of the 200k schedule):**
  - pick-place: 10/10 trainers and 10/10 full-split results verified
    (`runs/setup/item1-pick-place-fullsplit-inventory-20261011-083652.json`); D0-only margins frozen
    (`runs/setup/item1-d0-pick-place-margins-20261011-081643.json`).
  - hammer: all 10 trainers resumed 15:45 from proven checkpoints (40k; D2 seed 0 at 50k), attempt 2, unchanged
    seeds/schedules; ~110 steps/min, 100k expected around 00:30–01:30 on 2026-10-12.
  - Next: 10 hammer full-split evaluations → freeze hammer D0 margins → item-1 rule over all 20 results.
- **Item 2 (M3D vs M2D)** waits for the item-1 decision; then 2 seeds x 2 tasks + 6-seed hammer RL proxies.
- `method-frozen-v1` after items 1-2, item 1e leave-one-out and the A200/A300 length study.
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
| i1-{d0..d4}-seed{0,1}-hammer | local GPU4/GPU5 (5 each) | training 40k→100k (attempt 2) | runs/pretrain/i1-*-hammer/log.txt |
| fulleval-i1-*-hammer-100k (10) | local | pending (deps on trainers) | runs/fulleval-i1-*/console.log |
| s4-drm-cnn-<task>-s2000 (+ `-eval`), bin-picking as `-r4`; s2001 for 5 tasks | remote GPU 0-3 | 12 pairs training to 1M (first 16:47, last 17:13) | runs/s4-drm-cnn-*/ (synced) |
| remaining s2001 (3 tasks) + s2002 (8 tasks) | remote | 11 pairs pending headroom (<= 1 TB) | — |

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
1. Stage 4 CNN: admit remaining pairs as measured footprints allow; sync W&B as runs finish; then SinCro/ReViWo
   pretraining on the remote (check media panels in their offline W&B runs).
2. Hammer trainers reach 100k → 10 full-split evaluations (admitted job by job) → hammer D0 margins →
   item-1 rule → report 1d.
3. Item 2 + RL proxies after item 1; then item 1e, length study, freeze, Stage 3 final RL.
