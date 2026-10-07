# splatter4d Meta-World campaign — progress

Last updated: 2026-10-08 07:10

## Current phase
- **Phase A** complete (E0). **Phase B** complete: 8 tasks, 162 GB, D2/D3 pass (EXPERIMENT_LOG "Phase B result").
- **Phase C** in progress: RL timing done (CNN GPU-bound ~80 steps/s/GPU; frozen ~190 steps/s/GPU with 8 runs);
  pretraining 0.22 s/step at 8 loader workers (loader-bound; 32 workers now); `docs/COMPUTE_PLAN.md` written.
  M2 overfit gate FAILED after three diagnosed iterations (G1-G3; recorded in RESULTS.md).
- **Stage 0** done: DrM + CNN hammer 0.70 train-camera / 0.02 held-out success at 1M; shelf-place 0 (no reward ever: sparse v3 reward). See EXPERIMENT_LOG.
- **Stage 1** started: baseline pretraining (200k steps) on hammer and pick-place.
- **Stage 4 prep** done: SinCro and ReViWo ported (merge 67f5984, `docs/BASELINES.md`); suite 199/199. Measured cost
  under load: SinCro 0.53 s/step (500k reference steps ~74 h/task), ReViWo 0.58 s/step (100k steps ~16 h/task).
  SinCro's budget needs the user's decision before Stage 4 (raised in the Stage 1 report).

## Event handling
- Event watcher: `python3 -I scripts/watch_events.py --once` as a background task, re-armed after each event.
- Hourly check: user `/loop`, session cron job `af519ff6` at :47 (expires after 7 days).
- Scheduler daemon: `scripts/jobs.py daemon --interval 30` (pid `experiments/daemon.pid`, log `experiments/daemon.log`).
  `touch experiments/HOLD` while editing or testing sources (every file in `s4d/ scripts/ tests/ configs/` is
  fingerprinted by the test gate); remove it after the suite passes.

## Heartbeats
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

## Job table (running or pending)
| id | GPU | status | log | W&B |
|---|---|---|---|---|
| stage0-drm-cnn-hammer-s2000 (+ -eval) | GPU 5 | running | runs/stage0-drm-cnn-hammer-s2000/console.log | splatter4d-rl / stage0-drm-cnn-hammer-s2000 |
| stage0-drm-cnn-shelf-place-s2000 (+ -eval) | GPU 5 | running | runs/stage0-drm-cnn-shelf-place-s2000/console.log | splatter4d-rl / stage0-drm-cnn-shelf-place-s2000 |
| s1-pretrain-hammer-base | GPU 4 | running (200k) | runs/s1-pretrain-hammer-base/console.log, runs/pretrain/s1-pretrain-hammer-base/ | splatter4d-metaworld / s1-pretrain-hammer-base |
| s1-pretrain-pick-place-base | GPU 5 | running (200k) | runs/s1-pretrain-pick-place-base/console.log, runs/pretrain/s1-pretrain-pick-place-base/ | splatter4d-metaworld / s1-pretrain-pick-place-base |
| export-s1-base-{hammer,pick-place}-100k -> s1-proxy-base-{task}-s{1000,1001} (+ -eval) | GPU 5 | waiting for 100k checkpoints | runs/<id>/console.log | splatter4d-rl |
| s1-it1-decdim256-{hammer,pick-place} (iteration 1, 200k schedule, compared at 100k) | GPU 4 | running (resumed 03:17 from 30k) | runs/pretrain/s1-it1-*/log.txt | splatter4d-metaworld |
| s1-it2-lambdadyn4-{hammer,pick-place} (iteration 2) | GPU 4 | running (resumed 03:17 from 30k) | runs/pretrain/s1-it2-*/log.txt | splatter4d-metaworld |

Completed: all `collect-/split-/stats-/check-<task>` (32), timing runs (`timing-*`, `timing2-*`), gate it1/it2.

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

## Next actions
1. M2 iteration 3 -> record M2 outcome (RESULTS.md if still failing after three iterations).
2. Stage 1 pretraining -> M3-M7 at checkpoints; export encoders; DrM RL with seeds 1000-1002 on hammer and pick-place.
3. Stage 0 -> compare hammer with the DrM paper; record in EXPERIMENT_LOG.
4. Report to the user at the end of Stage 1.
