# splatter4d Meta-World campaign — progress

Last updated: 2026-10-07 21:20

## Current phase
- **Phase A** complete (E0). **Phase B** complete: 8 tasks, 162 GB, D2/D3 pass (EXPERIMENT_LOG "Phase B result").
- **Phase C** in progress: RL timing done (CNN GPU-bound ~80 steps/s/GPU; frozen ~190 steps/s/GPU with 8 runs);
  pretraining 0.22 s/step at 8 loader workers (loader-bound; 32 workers now); `docs/COMPUTE_PLAN.md` written.
  M2 overfit gate failed iterations 1 and 2 (G1, G2); iteration 3 (two arms) running.
- **Stage 0** running: DrM + CNN, seed 2000, hammer and shelf-place.
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

## Job table (running or pending)
| id | GPU | status | log | W&B |
|---|---|---|---|---|
| stage0-drm-cnn-hammer-s2000 (+ -eval) | GPU 5 | running | runs/stage0-drm-cnn-hammer-s2000/console.log | splatter4d-rl / stage0-drm-cnn-hammer-s2000 |
| stage0-drm-cnn-shelf-place-s2000 (+ -eval) | GPU 5 | running | runs/stage0-drm-cnn-shelf-place-s2000/console.log | splatter4d-rl / stage0-drm-cnn-shelf-place-s2000 |
| s1-pretrain-hammer-base | GPU 4 | running (200k) | runs/s1-pretrain-hammer-base/console.log, runs/pretrain/s1-pretrain-hammer-base/ | splatter4d-metaworld / s1-pretrain-hammer-base |
| s1-pretrain-pick-place-base | GPU 5 | running (200k) | runs/s1-pretrain-pick-place-base/console.log, runs/pretrain/s1-pretrain-pick-place-base/ | splatter4d-metaworld / s1-pretrain-pick-place-base |
| gate-m2-hammer-it3b (12k steps) | GPU 5 | running | runs/pretrain/gate-m2-hammer-it3b/log.txt | splatter4d-metaworld |

Completed: all `collect-/split-/stats-/check-<task>` (32), timing runs (`timing-*`, `timing2-*`), gate it1/it2.

## Disk
| time | free on /home/ws |
|---|---|
| 2026-10-07 17:00 | 5578 GB |
| 2026-10-07 18:45 | 5.0 TB |
| 2026-10-07 19:02 | 5.0 TB (after timing-cnn-c1 finished; deleted runs/timing-cnn-c1/replay/, 0.98 GB) |
| 2026-10-07 19:55 | 4.9 TB (timing runs finished; their replays are kept: no final evaluation) |

## Next actions
1. M2 iteration 3 -> record M2 outcome (RESULTS.md if still failing after three iterations).
2. Stage 1 pretraining -> M3-M7 at checkpoints; export encoders; DrM RL with seeds 1000-1002 on hammer and pick-place.
3. Stage 0 -> compare hammer with the DrM paper; record in EXPERIMENT_LOG.
4. Report to the user at the end of Stage 1.
